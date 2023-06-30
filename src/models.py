from typing import List, Any, Union
import pandas as pd
from abc import abstractmethod
from numpy import array_split
import torch
import re

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

class GenericModel:
    def __init__(self, 
                 model : Any, 
                 tokenizer : Any,
                 generation_config : dict = None, 
                 batch_size : int = 1) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.batch_size = batch_size

    @abstractmethod
    def logic(self, input : Union[str, pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def generate(self, input : Union[str, pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
        if input is isinstance(input, str):
            return self.logic(input)
        elif isinstance(input, pd.Series) or isinstance(input, pd.DataFrame):
            return pd.concat([self.logic(chunk) for chunk in array_split(input, len(input) // self.batch_size)])
        else: 
            raise TypeError("Input must be a string or a pandas Object")
        