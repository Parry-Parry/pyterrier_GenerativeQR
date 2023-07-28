from typing import Any
import pyterrier as pt
from collections import Counter
import pandas as pd
from pyterrier.model import push_queries
import logging
    
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
        prompt = self.prompt.format(input_query = query)
        output =  self.model.generate(prompt)[0]
        #tokens = output.split(' ')[len(query.split(' ')):]
        tokens = output.split(' ')

        weighted_query = ' '.join([f'{token}^{self.beta}' for token in tokens])
        new_query =  f'{query} {weighted_query}'

        return new_query

    def transform(self, inputs):

        outputs = inputs.copy()
        queries = outputs[['qid', 'query']].drop_duplicates()
        queries['new'] = queries['query'].apply(lambda x: self.logic(x))
        
        queries = queries.set_index('qid')['new'].to_dict()
        push_queries(outputs, inplace = True)
        outputs['query'] = outputs['qid'].apply(lambda x: queries[x])

        return outputs

class GenerativePRF(pt.Transformer):
    essential = ['docno', 'qid', 'query']
    default = 'Improve the search effectiveness by suggesting expansion terms for the query:{input_query}, based on the given context information: {context}'
    def __init__(self, 
                 model : Any, 
                 prompt : str = None,
                 beta : float = 0.5,
                 k : int = 3,
                 return_counts : bool = False,
                 text_attr :str = 'text',
                 type : str = 'topp',
                 **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt if prompt else self.default
        self.beta = beta
        self.k = k
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
        logging.info('Generating new query')
        query = query.sort_values('rank')
        first_row = query.iloc[0]
        input_query = first_row['query']
        k = min(self.k, len(query))

        context = self.context_extract(query, k)
        prompt = self.prompt.format(input_query = input_query, context = context)
        output =  self.model.generate(prompt)[0]
        logging.info(f'Generating done')

        tokens = output.split()
        count = Counter(output)

        weighted_query = ' '.join([f'{token}^{self.beta}' for token in tokens])
        new_query =  f'{input_query} {weighted_query}'

        new_frame = {'qid' : first_row['qid'], 'query' : new_query}

        return pd.DataFrame.from_records([new_frame])

    def transform(self, inputs):

        for attr in self.essential:
            if attr not in inputs.columns:
                raise ValueError(f"Input must contain {attr} column")

        queries = inputs.copy().groupby('qid').apply(self.logic)
        queries = queries.set_index('qid')['query'].to_dict()

        outputs = inputs.copy()
        push_queries(outputs, inplace = True)
        outputs['query'] = outputs['qid'].apply(lambda x: queries[x])

        return outputs
        
        