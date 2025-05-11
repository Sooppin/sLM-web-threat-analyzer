import os
from logging import Logger
from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

import torch
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores.faiss import FAISS

from src.common.Constants import Constants
from src.common.LoggerManager import LoggerManager
from src.utils.DecodeURL import DecodeURL
from src.utils.CUDALimit import CUDALimit

@dataclass
class ModelConfig:
    model_name: str = Constants.EMB_MODEL_NAME
    model_path: str = Constants.EMB_MODEL_PATH
    device: str = Constants.DEVICE
    batch_size: int = Constants.BATCH_SIZE
    cache_dir: str = Constants.DIR_EMBEDDINGS_CACHE
    similarity_threshold: float = Constants.THRESHOLD
    k: int = Constants.K


class JsonIndexer:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.limit = CUDALimit()
        self.limit.set_memory_limit()

        self.logger: Logger = LoggerManager.get()
        self.config = config or ModelConfig
        self.decoder = DecodeURL()

        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 모델 이름을 기반으로 FAISS 인덱스 경로 설정
        self.cache_dir = os.path.join(base_path, self.config.cache_dir)
        self.model_name_sanitized = self.config.model_name.replace("/", "_")
        self.embeddings_dir = os.path.join(self.cache_dir, f"{self.model_name_sanitized}_embeddings")

        self.faiss_path = os.path.join(self.cache_dir,
                                       f"index_{self.model_name_sanitized}_{self.config.batch_size}")

        self.embeddings = self._initialize_embeddings()
        self.vectorstore: Optional[FAISS] = None

        self.logger.info(f"JsonIndexer initialized...")

    @staticmethod
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["payload"] = record.get("payload")
        metadata["attack_type"] = record.get("attack_type")
        metadata["idx"] = record.get("idx")
        metadata["dict_gpt_api_summary"] = record.get("dict_gpt_api_summary")
        return metadata

    def _initialize_embeddings(self) -> CacheBackedEmbeddings:
        model_kwargs = {
            "device": self.config.device,
        }
        encode_kwargs = {
            "normalize_embeddings": True,
        }

        base_embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        cache_store = InMemoryByteStore()

        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=base_embeddings,
            document_embedding_cache=cache_store,
            namespace=self.model_name_sanitized,
            batch_size=self.config.batch_size
        )

    def load_and_embed(self, file_path: str) -> None:
        decoded_path = self.decoder.decode_file(file_path)
        self.logger.info(f"Decoded and Filtered file path: {decoded_path}")
        try:
            loader = JSONLoader(
                file_path=decoded_path,
                jq_schema=".",
                content_key="attack_syntax",
                text_content=False,
                json_lines=True,
                metadata_func=self.metadata_func
            )
            docs = loader.load()
            total_docs = len(docs)
            self.logger.info(f"Loaded {total_docs} documents")

            self.process_documents(docs)

        except Exception as e:
            self.logger.error(f"Failed to load and embed documents from {file_path}")
            raise

    def process_documents(self, docs: list) -> None:
        with torch.no_grad():
            total_docs = len(docs)
        self.logger.info(f"Starting embedding process for {total_docs} documents...")
        self.logger.info(f"Embedding model: {self.config.model_name}")
        self.logger.info(f"Batch size: {self.config.batch_size}")

        batches = [docs[i:i + self.config.batch_size] for i in range(0, total_docs, self.config.batch_size)]

        for batch in tqdm(batches, desc="Processing documents", total=len(batches)):
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(
                    documents=batch,
                    embedding=self.embeddings
                )
            else:
                self.vectorstore.add_documents(batch)

        self._save_index()

        self.logger.info(f"Document processing and saving completed. Path: {self.faiss_path}")

    def _save_index(self) -> None:
        if self.vectorstore is not None:
            os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
            self.vectorstore.save_local(self.faiss_path)
            self.logger.info(f"Saved FAISS index to {self.faiss_path}")
        else:
            self.logger.info("No vectorstore to save")

    def load_vectorstore(self):
        try:
            vectorstore = FAISS.load_local(
                self.faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info(f"Loaded FAISS index from {self.faiss_path}")
            return vectorstore
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index from {self.faiss_path} \n{e}")
