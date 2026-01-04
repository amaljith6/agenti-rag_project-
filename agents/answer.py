import json
from langchain_ollama import ChatOllama
from schemas.answer_output import AnswerOutput


class GroundedAnswerAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3:8b",  # stable, CPU-friendly
            temperature=0,
        )

        with open("prompts/answer.txt", "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def answer(self, question: str, documents) -> AnswerOutput:
        context_blocks = []
        sources = set()

        for doc in documents:
            context_blocks.append(doc.page_content)
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])

        context_text = "\n\n".join(context_blocks)

        prompt = self.prompt_template.format(
            question=question,
            context=context_text,
        )

        response = self.llm.invoke(prompt).content

        try:
            data = json.loads(response)
            return AnswerOutput(
                answer=data.get("answer", ""),
                citations=data.get("citations", list(sources)),
                confidence=data.get("confidence", "low"),
            )
        except Exception:
            # Safe fallback (CRITICAL)
            return AnswerOutput(
                answer="I do not have enough information to answer this question.",
                citations=list(sources),
                confidence="low",
            )
