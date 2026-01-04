from pydantic import BaseModel
from typing import List


class PlannerDecision(BaseModel):
    tools: List[str]  # ["tour_guide"], ["law"], or both
    reason: str
