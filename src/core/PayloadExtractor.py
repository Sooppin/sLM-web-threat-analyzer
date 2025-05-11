import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from logging import Logger
from src.common.Constants import Constants
from src.common.LoggerManager import LoggerManager
from src.utils.CUDALimit import CUDALimit


class PayloadExtractor:
    def __init__(self):
        self.limit = CUDALimit()
        self.limit.set_memory_limit()

        self.logger: Logger = LoggerManager.get()
        self.model_path = os.path.abspath(Constants.EXT_MODEL_PATH)

        if not os.path.exists((self.model_path)):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.logger.info(f"Loading model from: {self.model_path}")

        # model & tokenizer load
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        self.logger.info("Model loaded successfully.")

        self.device = Constants.DEVICE
        self.logger.info(f"Using device: {self.device}")

        self.model.to(self.device)

        self.max_length = Constants.MAX_LENGTH
        self.prefix = Constants.PREFIX
        self.logger.info(f"prefix text : {self.prefix}")

    def extract_syntax(self, payload: str) -> str:
        try:
            if not isinstance(payload, str) or not payload.strip():
                self.logger.warning("Received empty or invalid payload.")
                return ""

            self.logger.info(f"Processing payload: {payload}")

            # Pre-processing
            input_text = f"{self.prefix}{payload}" if self.prefix else payload
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                return_tensors='pt',
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Model Inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=1,
                    do_sample=False
                )

            # Post-processing
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = re.sub(r"<\s", "<", generated_text)

            self.logger.info(f"Extracted syntax: {generated_text}")

            return generated_text

        except Exception as e:
            self.logger.error(f"Error syntax extracting: {str(e)}")
            return ""
