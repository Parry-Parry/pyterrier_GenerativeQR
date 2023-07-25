from typing import Any
import pyterrier as pt
from collections import Counter
import pandas as pd
from pyterrier.model import push_queries
    
class GenerativeQR(pt.Transformer):
    default = 'Improve the search effectiveness by suggesting expansion terms for the query: {input_query}'
    def __init__(self, 
                 model : Any, 
                 prompt : str = None,
                 beta : float = 0.5,
                 return_counts : bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt if prompt else self.default
        self.beta = beta
        self.return_counts = return_counts
    
    def logic(self, query):
        first_row = query.iloc[0]
        input_query = first_row['query']
        
        prompt = self.prompt.format(input_query = input_query)
        output =  self.model.generate(prompt)

        tokens = output.split()
        count = Counter(output)

        weighted_query = ' '.join([f'{token}^{self.beta}' for token in tokens])
        new_query =  f'{input_query} {weighted_query}'

        new_frame = {'qid' : first_row['qid'], 'query' : new_query}
        if self.return_counts: new_frame['counts'] = len(count)

        return pd.DataFrame.from_records([new_frame])

    def transform(self, inputs):

        outputs = inputs.copy()
        queries = outputs[['qid', 'query']]
        x = queries.groupby('qid').apply(self.logic)
        print(x)
        queries = pd.concat(x, axis=0)

        queries = queries.set_index('qid')['query'].to_dict()
        push_queries(outputs, inplace = True)
        outputs['query'] = outputs.apply(lambda x: queries[x]['query'], axis = 1)
        if self.return_counts: outputs['counts'] = outputs.apply(lambda x: queries[x]['counts'], axis = 1)

        return outputs

class GenerativePRF(pt.Transformer):
    essential = ['docno', 'qid', 'query']
    default = 'Improve the search effectiveness by suggesting expansion terms for the query:{input_query}, based on the given context information: {context}'
    def __init__(self, 
                 model : Any, 
                 prompt : str = None,
                 beta : float = 0.5,
                 return_counts : bool = False,
                 text_attr :str = 'text',
                 type : str = 'topp',
                 **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt if prompt else self.default
        self.beta = beta
        self.return_counts = return_counts
        self.text_attr = text_attr

        self.essential.append(self.text_attr)

        if type =='topp':
            self.context_extract = self.get_context_topp
        elif type == 'firstp':
            self.context_extract = self.get_context_firstp
        elif type == 'maxp':
            self.context_extract = self.get_context_maxp
        else:
            raise ValueError("type must be either topp, firstp or maxp")

    def get_context_topp(self, inputs, k):
        inputs = inputs.sort_values('rank')
        return ' '.join(inputs.head(k)[self.text_attr].values)
    
    def get_context_firstp(self, inputs, k):
        inputs = inputs.groupby('docno').apply(lambda x: x.head(1))
        inputs = inputs.sort_values('rank')
        return ' '.join(inputs.head(k)[self.text_attr].values)
    
    def get_context_maxp(self, inputs, k):
        inputs = inputs.groupby('docno').apply(lambda x: x.sort_values('score').head(1))
        return ' '.join(inputs.head(k)[self.text_attr].values)
    
    def logic(self, query):

        query = query.sort_values('rank')
        first_row = query.iloc[0]
        input_query = first_row['query']
        k = min(self.k, len(query))

        context = self.context_extract(query, k)
        prompt = self.prompt.format(input_query = input_query, context = context)
        output =  self.model.generate(prompt)

        tokens = output.split()
        count = Counter(output)

        weighted_query = ' '.join([f'{token}^{self.beta}' for token in tokens])
        new_query =  f'{input_query} {weighted_query}'

        new_frame = {'qid' : first_row['qid'], 'query' : new_query}
        if self.return_counts: new_frame['counts'] = len(count)

        return pd.DataFrame.from_records([new_frame])

    def transform(self, inputs):

        for attr in self.essential:
            if attr not in inputs.columns:
                raise ValueError(f"Input must contain {attr} column")

        queries = inputs.copy().groupby('qid').apply(self.logic)
        queries = queries.set_index('qid')['query'].to_dict()

        outputs = inputs.copy()
        push_queries(outputs, inplace = True)
        outputs['query'] = outputs['qid'].apply(lambda x: queries[x]['query'], axis = 1)
        if self.return_counts: outputs['counts'] = outputs['qid'].apply(lambda x: queries[x]['counts'], axis = 1)

        return outputs
        
        