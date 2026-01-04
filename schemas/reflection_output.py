from pydantic import BaseModel


class ReflectionDecision(BaseModel):
    grounded: bool
    needs_more_context: bool
    reason: str
