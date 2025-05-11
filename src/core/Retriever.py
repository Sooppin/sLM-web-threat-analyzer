import json
from typing import Optional
from logging import Logger

from src.core.JsonIndexer import JsonIndexer, ModelConfig
from src.common.LoggerManager import LoggerManager
from src.utils.CUDALimit import CUDALimit


class Retriever:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.limit = CUDALimit()
        self.limit.set_memory_limit()

        self.logger: Logger = LoggerManager.get()
        self.config = config or ModelConfig
        self.indexer = JsonIndexer(config)
        self.vectorstore = None

        self.logger.info(f"Retriever initialized...")
        self.logger.info(f"Using model: {self.config.model_name}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Using Device: {self.config.device}")

    def search_with_score(self, _query: str):
        self.vectorstore = self.indexer.load_vectorstore()

        if self.vectorstore is None:
            self.logger.error("Vectorstore is not loaded")
        self.logger.info("Vectorstore is loaded")

        q_embedding = self.indexer.embeddings.embed_query(_query)
        self.logger.info(f"Query embedding Finished")

        results_with_score = self.vectorstore.similarity_search_with_score_by_vector(
            embedding=q_embedding,
            k=self.config.k
        )

        result_dicts = []
        for doc, l2_distance in results_with_score:
            cosine_similarity = 1 - (l2_distance ** 2) / 2

            if cosine_similarity < self.config.similarity_threshold:
                self.logger.info(f"doc: {doc.page_content}")
                self.logger.info(f"Cosine Similarity {cosine_similarity:.3f} below threshold, returning None.")
                return None

            result_dicts.append({
                "score": float(round(cosine_similarity, 3)),
                "attack_syntax": doc.page_content,
                "attack_type": doc.metadata.get("attack_type", "ERROR"),
                "dict_gpt_api_summary": doc.metadata.get("dict_gpt_api_summary", "ERROR"),
            })

        self.logger.info(f"Retrieval Finished")

        return json.dumps(result_dicts, ensure_ascii=False)
