import json
from langchain_ollama import ChatOllama
from schemas.planner_output import PlannerDecision


class PlannerAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3:8b",  # or deepseek-r1
            temperature=0,
        )

        with open("prompts/planner.txt", "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def decide(self, question: str) -> PlannerDecision:
        prompt = self.prompt_template.format(question=question)

        response = self.llm.invoke(prompt).content

        try:
            data = json.loads(response)
            return PlannerDecision(**data)
        except Exception as e:
            # Safe fallback (IMPORTANT)
            return PlannerDecision(
                tools=["law"],
                reason="Fallback due to parsing error",
            )
