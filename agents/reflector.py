import json
from langchain_ollama import ChatOllama
from schemas.reflection_output import ReflectionDecision


class ReflectionAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3:8b",
            temperature=0,
        )

        with open("prompts/reflection.txt", "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def reflect(self, question, answer, documents) -> ReflectionDecision:
        context_text = "\n\n".join(
            [doc.page_content for doc in documents]
        )

        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            context=context_text,
        )

        response = self.llm.invoke(prompt).content

        try:
            return ReflectionDecision(**json.loads(response))
        except Exception:
            # Fail-safe: assume grounded to avoid loops
            return ReflectionDecision(
                grounded=True,
                needs_more_context=False,
                reason="Fallback",
            )
