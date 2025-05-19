import os
from typing import Union
from torch import Tensor
import numpy as np
from src.sentence_bert_cpu import CPUSentenceBERT
from sentence_transformers import SentenceTransformer
from src.constants import BATCH_SIZE, MAX_SEQ_LENGTH, PROMPTS


class ModelWrapper:
    """ModelWrapper to handle specific edge cases. Since we do now want to manipulate source code of BEIR, I had to create something on top.
    For example, E5 model has to apply query prefix on passages in case of Quora dataset.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def _set_prompt(self, dataset_name):
        prompt = PROMPTS["cqadupstack"]["query"] if "cqadupstack" in dataset_name else PROMPTS[dataset_name]["query"]
        self.retriever.model.prompt = prompt

    def set_dataset_specific_params(self, dataset_name):
        # Quora is symetric (both query and passage are questions), so E5 has to treat documents as queries
        if isinstance(self.retriever.model, DefaultModel):
            if self.retriever.model.use_prompt:
                self._set_prompt(dataset_name)
            if dataset_name == "quora":
                self.retriever.model.doc_as_query = True
            else:
                self.retriever.model.doc_as_query = False
        else:
            pass
        # if any other model will require special processing for any of the datasets, additional logic will
        # be added there

    def __getattr__(self, name):
        return getattr(self.retriever, name)

class DefaultModel(CPUSentenceBERT):
    """Default model class used for our experiments. It can handle both models with prompt templates, as well
    as models which do not require any input text formatting.
    """

    def __init__(self, model_path, query_prefix: str, passage_prefix: str, instruct_prefix: str = None, use_prompt: bool = False):
        super().__init__(model_path)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.instruct_prefix = instruct_prefix
        self.use_prompt = use_prompt 
        self.doc_as_query = False
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        self.device_ids = list(range(len(devices)))
        self.prompt = None

        if self.q_model.max_seq_length > MAX_SEQ_LENGTH:
            self.q_model.max_seq_length = MAX_SEQ_LENGTH
        if self.doc_model.max_seq_length > MAX_SEQ_LENGTH:
            self.doc_model.max_seq_length = MAX_SEQ_LENGTH
        
    def encode_queries(
        self, queries: list[str], batch_size: int = BATCH_SIZE, **kwargs
    ) -> Union[list[Tensor], np.ndarray, Tensor]:
        if self.use_prompt:
            # Fortunately all instruct models apply the same template which is 'Instruct: <prompt>\nQuery: <query>
            prompt = self.prompt if self.prompt is not None else ""
            instruct_prefix = self.instruct_prefix if self.instruct_prefix is not None else ""
            query_prefix = self.query_prefix if self.query_prefix is not None else ""
            queries = [instruct_prefix + prompt + query_prefix +query for query in queries]
        elif self.query_prefix is not None:
            queries = [self.query_prefix + query for query in queries]
            
        query_embeddings = self.encode(self.q_model, queries, batch_size, **kwargs)
        return query_embeddings

    def encode_corpus(
        self, corpus: Union[list[dict[str, str]], dict[str, list]], batch_size: int = BATCH_SIZE, **kwargs
    ) -> Union[list[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [
                (
                    (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                )
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]

        if self.doc_as_query:
            return self.encode_queries(sentences, batch_size)

        elif self.passage_prefix is not None:
            sentences = [self.passage_prefix + sentence for sentence in sentences]
            
        corpus_embeddings = self.encode(self.doc_model, sentences, batch_size, **kwargs)
        return corpus_embeddings
    
    def encode(self, model: SentenceTransformer, sequences: list[str], batch_size: int, **kwargs):
        if len(self.device_ids) > 1:
            embeddings = self.encode_multi_process(sequences, model, batch_size)
        else:
            embeddings = model.encode(sequences, batch_size=batch_size, **kwargs)
        
        # Even though the `normalize_embeddings` argument is set to True, it does not work for some models 
        # like https://huggingface.co/WhereIsAI/UAE-Large-V1, so I introduced a normalization for sanity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings
    
    def encode_multi_process(self, sequences: list[str], model: SentenceTransformer, batch_size: int):
        pool = model.start_multi_process_pool(target_devices=self.device_ids)
        embeddings = model.encode_multi_process(sequences, pool, batch_size=batch_size)
        model.stop_multi_process_pool(pool)
        return embeddings
