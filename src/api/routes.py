import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.common.LoggerManager import LoggerManager
from src.core.PayloadExtractor import PayloadExtractor
from src.core.Retriever import Retriever


router = APIRouter()

extractor = PayloadExtractor()
retriever = Retriever()
Logger = LoggerManager.get()

class PayloadRequest(BaseModel):
    payload: str

@router.post("/analyze")
async def process_payload(item: PayloadRequest):
    """
    웹 공격 페이로드를 받아 공격 구문 추출 후 유사 공격 정보를 검색합니다.
    """
    try:
        extracted_text = extractor.extract_syntax(item.payload)

        if not extracted_text or extracted_text.strip() == "" or extracted_text not in item.payload:
            Logger.info(f"No valid attack syntax extracted.")
            return {
                "code": "3",
                "payload": item.payload,
                "extract_syntax": "None",
                "retrieved_results": "None"
            }

        context = retriever.search_with_score(extracted_text.strip())

        if context:
            code = "1"
            retrieved_results = json.loads(context)[0]
        else:
            code = "2"
            retrieved_results = "None"

        return {
            "code": code,
            "payload": item.payload,
            "extract_syntax": extracted_text,
            "retrieved_results": retrieved_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
