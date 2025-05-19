from beir.retrieval.evaluation import EvaluateRetrieval

from src.retrievers import DRESV2


class EvaluateRetrievalV2(EvaluateRetrieval):
    def __init__(
        self,
        retriever: DRESV2,
        k_values: list[int] = [1, 3, 5, 10, 100, 1000],
        score_function: str = "cos_sim",
    ):
        super().__init__(retriever, k_values, score_function)

    def retrieve(self, queries: dict[str, str], **kwargs) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        return self.retriever.search(queries, self.top_k, **kwargs)
