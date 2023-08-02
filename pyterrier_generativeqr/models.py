from typing import List, Union
import pandas as pd
from abc import abstractmethod
import torch
from math import ceil
from .configs import creative
from .util import clean, batch, concatenate, cut_prompt

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
    
    def generate(self, inputs : Union[str, list]) -> Union[str, pd.Series]:
        if isinstance(inputs, str):
            return self.logic(inputs)
        else: 
            return concatenate([self.logic(chunk) for chunk in batch(inputs, ceil(len(inputs) / self.batch_size))])

class Seq2SeqLM(GenericModel):
    def __init__(self, 
                 model_name : str, 
                 generation_config: dict = None, 
                 num_return_sequences: int = 3, 
                 batch_size: int = 1, 
                 device = 'cuda',
                 **kwargs) -> None:
        super().__init__(generation_config, num_return_sequences, batch_size, device)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if 'device_map' not in kwargs: 
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(self.device)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def logic(self, input : Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input, str): input = [input]

        inputs = self.tokenizer(input, padding = True, truncation = True, return_tensors = 'pt').to(self.device)
        outputs = self.model.generate(**inputs, **self.generation_config)
        outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)

        return list(map(clean, outputs_text))
    
class CausalLM(GenericModel):
    def __init__(self, 
                 model_name : str, 
                 generation_config: dict = None, 
                 num_return_sequences: int = 3, 
                 batch_size: int = 1, 
                 device = 'cuda',
                 cutoff_prompt : bool = True,
                 **kwargs) -> None:
        super().__init__(generation_config, num_return_sequences, batch_size, device)

        self.cutoff = cutoff_prompt

        from transformers import AutoModelForCausalLM, AutoTokenizer
        if 'device_map' not in kwargs: 
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def logic(self, input : Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input, str): input = [input]

        inputs = self.tokenizer(input, padding = True, truncation = True, return_tensors = 'pt').to(self.device)
        outputs = self.model.generate(**inputs, **self.generation_config)
        outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)

        if self.cutoff: outputs_text = list(map(cut_prompt, outputs_text, input))
        return list(map(clean, outputs_text))