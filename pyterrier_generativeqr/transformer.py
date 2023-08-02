from typing import Any
import pyterrier as pt
from pyterrier.model import push_queries, query_columns

class GenerativeExpansion(pt.Transformer):
    def __init__(self, 
                 model : Any, 
                 beta : float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.beta = beta
    
    def postprocess(self, input_query, output_query):
        tokens = output_query.split(' ')
        weighted_query = ' '.join([f'{token}^{self.beta}' for token in tokens])
        new_query =  f'{input_query} {weighted_query}'
        return new_query
    
class GenerativeQR(GenerativeExpansion):
    default = 'Improve the search effectiveness by suggesting expansion terms for the query: {input_query}'
    def __init__(self, 
                 model : Any, 
                 prompt : str = None,
                 beta : float = 0.5,
                 **kwargs):
        super().__init__(model=model, beta=beta, **kwargs)

        self.prompt = prompt if prompt else self.default

    def transform(self, inputs):
        
        queries = inputs[['qid', 'query']].copy().drop_duplicates()

        prompts = list(map(lambda x : self.prompt.format(input_query=x), queries['query'].tolist()))
        model_outputs = self.model.generate(prompts)

        queries['new'] = list(map(lambda x, y : self.postprocess(x, y), queries['query'].tolist(), model_outputs))

        return queries[['qid', 'query' 'new']].rename(columns = {'query' : 'query_0', 'new' : 'query'})

class GenerativePRF(GenerativeExpansion):
    essential = ['docno', 'qid', 'query']
    default = 'Improve the search effectiveness by suggesting expansion terms for the query:{input_query}, based on the given context information: {context}'
    def __init__(self, 
                 model : Any, 
                 prompt : str = None,
                 beta : float = 0.5,
                 k : int = 3,
                 text_attr :str = 'text',
                 type : str = 'topp',
                 **kwargs):
        super().__init__(model=model, beta=beta, **kwargs)

        self.prompt = prompt if prompt else self.default
        self.k = k
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
    
    def get_context(self, query):
        query = query.sort_values('rank')
        k = min(self.k, len(query))
        return self.context_extract(query, k)

    def transform(self, inputs):

        for attr in self.essential:
            if attr not in inputs.columns:
                raise ValueError(f"Input must contain {attr} column")

        context = inputs.copy().groupby('qid').apply(self.get_context)
        queries = inputs.copy().drop_duplicates('qid')

        prompts = list(map(lambda x, y : self.prompt.format(input_query=x, context=y), queries['query'].tolist(), context.tolist()))
        model_outputs = self.model.generate(prompts)

        queries['new'] = list(map(lambda x, y : self.postprocess(x, y), queries['query'].tolist(), model_outputs))

        queries = push_queries(queries, keep_original=True)
        queries = queries.rename(columns = {'new' : 'query'})
        
        return queries[query_columns(queries)]
        
        