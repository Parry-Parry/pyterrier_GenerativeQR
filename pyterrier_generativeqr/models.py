from typing import List, Union
import pandas as pd
from abc import abstractmethod
from numpy import array_split
import torch
import re
from pyterrier.model import split_df
from .configs import creative

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

class GenericModel:
    def __init__(self, 
                 generation_config : dict = None, 
                 num_return_sequences : int = None,
                 batch_size : int = 1, 
                 device = 'cpu') -> None:
        
        if not generation_config: generation_config = creative
        if num_return_sequences: generation_config['num_return_sequences'] = num_return_sequences
        self.generation_config = generation_config
        self.batch_size = batch_size
        self.device = torch.device(device)

    @abstractmethod
    def logic(self, input : Union[str, pd.Series]) -> Union[str, List[str]]:
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def generate(self, inputs : Union[str, pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
        print(inputs)
        if isinstance(inputs, str):
            return self.logic(inputs)
        else: 
            return pd.concat([pd.Series(self.logic(chunk.tolist())) for chunk in split_df(inputs, len(inputs) // self.batch_size)], axis=0)

class FLANT5(GenericModel):
    def __init__(self, 
                 model_name : str, 
                 generation_config: dict = None, 
                 num_return_sequences: int = 1, 
                 batch_size: int = 1, 
                 device = 'cpu',
                 **kwargs) -> None:
        super().__init__(generation_config, num_return_sequences, batch_size, device)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def logic(self, input : Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input, str): input = [input]

        inputs = self.tokenizer(input, padding = True, truncation = True, return_tensors = 'pt').to(self.device)
        outputs = self.model.generate(**inputs, **self.generation_config)
        outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
        return list(map(clean, outputs_text))