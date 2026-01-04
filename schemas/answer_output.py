from pydantic import BaseModel
from typing import List


class AnswerOutput(BaseModel):
    answer: str
    citations: List[str]
    confidence: str  # "high" | "medium" | "low"
