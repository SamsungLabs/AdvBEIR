from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os
import logging
import faiss
import pandas as pd
from faiss import write_index
from tqdm import tqdm
from src.constants import BATCH_SIZE, INDICES_DIR

RETRIEVAL_BATCH_SIZE = 256
class DRESV2(DRES):
    """A restructured class of DRES from BEIR. DRES has index building and query embeddings calculation all
    located in the search method.This is why it builds the same index two times when we try to evaluate
    benchmark with it (because query embeddings have to be calculated twice, once for original and perturbed
    query, so search method of DRES is invoked twice). With DRESV2, you can dynamically update the
    index/query embeddings. Thanks to that we can evaluate benchmarks without re-encoding the whole corpus
    for the second time."""

    def __init__(self, model, batch_size: int = BATCH_SIZE, corpus_chunk_size: int = 50000, use_cache: bool = True,  **kwargs):
        super().__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.index = None
        self.use_cache = use_cache
        self.original_query_results_cache = {}
                
    def create_index(self, corpus: dict[str, dict[str, str]], hash: str) -> None:
        """Creates a FAISS index with normalized embeddings (dot product == cosine similarity in case of
        normalized vectors)"""
        logging.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_ids = sorted(
            corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True
        )
        corpus_data = [corpus[cid] for cid in corpus_ids]

        corpus_embeddings = self.model.encode_corpus(
            corpus_data,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
        )
        dim = corpus_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_embeddings)
        self.index = index
        if self.use_cache == True:
            write_index(index, os.path.join(INDICES_DIR, hash + ".index"))

        texts = [corpus[cid]["text"] for cid in corpus_ids]
        titles = [corpus[cid]["title"] for cid in corpus_ids]
        self.index_df = pd.DataFrame({"id": corpus_ids, "text": texts, "title": titles})
    
    def encode_queries(self, queries: list[str]) -> None:
        """Prepares normalized embeddings of queries"""
        logging.info("Encoding Queries...")
        query_embeddings = self.model.encode_queries(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
        )
        self.query_embeddings = query_embeddings

    def search(self, queries: dict[str, str], top_k: int, **kwargs) -> dict[str, dict[str, float]]:
        """Searches for the top-k documents given a query and then assigns the scores between the query and
        the document in the results."""

        assert (
            self.index is not None
        ), "Index has not been created, you should call create_index first before using search method"
        query_ids = list(queries.keys())
        queries = list(queries.values())
        self.encode_queries(queries=queries)
        
        self.results = {qid: {} for qid in query_ids}
        for i in tqdm(range(0, len(queries), RETRIEVAL_BATCH_SIZE), desc="Searching the index"):
            batch_queries = self.query_embeddings[i : i + RETRIEVAL_BATCH_SIZE]
            batch_query_ids = query_ids[i : i + RETRIEVAL_BATCH_SIZE]
            scores, index_indices = self.index.search(batch_queries, k=top_k)
            corpus_indices = [self.index_df.iloc[indices]["id"].tolist() for indices in index_indices]
        
            for qid, score_list, indices in zip(batch_query_ids, scores, corpus_indices):
                for score, corpus_id in zip(score_list, indices):
                    self.results[qid][str(corpus_id)] = score.item()
                    
        return self.results
