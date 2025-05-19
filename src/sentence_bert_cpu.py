import torch

from beir.retrieval.models import SentenceBERT
from typing import Union
from sentence_transformers import SentenceTransformer

class CPUSentenceBERT(SentenceBERT):
    def __init__(self, model_path: Union[str, tuple] = None, sep: str = " ", **kwargs):
        self.sep = sep
        self.device = "cpu"
        
        if isinstance(model_path, str):
            # in case of e5-small model, torch.float16 slightly changes the result so they are a little bit
            # different than the paper ones. In case of the e5 instruct model, this data type brings us closer
            # to the official metrics
            self.q_model = SentenceTransformer(model_path, device=self.device, model_kwargs={"torch_dtype":torch.float16})
            self.doc_model = self.q_model
        
        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformer(model_path[0], device=self.device, model_kwargs={"torch_dtype":torch.float16})
            self.doc_model = SentenceTransformer(model_path[1], device=self.device, model_kwargs={"torch_dtype":torch.float16})