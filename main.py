from agents.planner import PlannerAgent
from agents.tour_tool import TourGuideRetrieverTool
from agents.law_tool import LawRetrieverTool
from agents.answer import GroundedAnswerAgent
from agents.reflector import ReflectionAgent


planner = PlannerAgent()
tour_tool = TourGuideRetrieverTool()
law_tool = LawRetrieverTool()
answer_agent = GroundedAnswerAgent()
reflector = ReflectionAgent()


def run_agentic_rag(question: str):
    decision = planner.decide(question)

    documents = []
    if "tour_guide" in decision.tools:
        documents.extend(tour_tool.run(question, k=5))
    if "law" in decision.tools:
        documents.extend(law_tool.run(question, k=5))

    # First answer
    answer = answer_agent.answer(question, documents)

    # Reflection
    reflection = reflector.reflect(
        question,
        answer.answer,
        documents,
    )

    if reflection.needs_more_context:
        # Corrective retrieval (broaden search)
        documents = []
        if "tour_guide" in decision.tools:
            documents.extend(tour_tool.run(question, k=8))
        if "law" in decision.tools:
            documents.extend(law_tool.run(question, k=8))

        answer = answer_agent.answer(question, documents)

    return answer


if __name__ == "__main__":
    question = input("Ask a question: ")

    result = run_agentic_rag(question)

    print("\nANSWER:")
    print(result.answer)
    print("CONFIDENCE:", result.confidence)
