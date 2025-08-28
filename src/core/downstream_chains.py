from typing import Dict, Any

from langchain_core.runnables import RunnableLambda, RunnableBranch


# 각각의 체인은 main.py에서 초기화 후 주입 가능
def build_dispatcher_chain(
    router_chain,
    rag_chain,
    summarizer_chain,
    recommender_chain,
    chat_llm_chain,
    confidence_threshold: float = 0.7,
):
    """
    사용자의 질문 의도에 따라 적절한 체인으로 분기하는 디스패처 체인을 빌드합니다.

    1. router_chain: 사용자의 질문을 분석하여 의도(intent)와 신뢰도(confidence)를 파악합니다.
    2. RunnableBranch: 신뢰도가 임계값 이상일 경우에만 의도에 따른 체인을 실행합니다.
    3. RunnableRouter: 파악된 의도에 따라 RAG, 요약, 추천, 일반 대화 체인으로 작업을 분기합니다.
    """

    def route_to_qa(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """RAG QA 체인으로 라우팅합니다. `question` 키를 사용합니다."""
        return {"answer": rag_chain.invoke({"question": input_dict["question"]})}

    def route_to_summarize(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """요약 체인으로 라우팅합니다. `question` 키를 사용합니다."""
        # main.py에서 생성된 summarizer_chain은 질문(str)을 입력받아 검색과 요약을 모두 수행합니다.
        summary_result = summarizer_chain.invoke(input_dict["question"])
        # 요약 체인의 출력(e.g., {'output_text': '...'})에서 실제 답변을 추출합니다.
        return {"answer": summary_result.get("output_text", "요약 결과를 생성하지 못했습니다.")}

    def route_to_recommend(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """추천 체인으로 라우팅합니다. `question` 키를 사용합니다."""
        return {"answer": recommender_chain.invoke(input_dict["question"])}

    def route_to_chat(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """일반 대화 체인으로 라우팅합니다. `question` 키를 사용합니다."""
        return {"answer": chat_llm_chain.invoke(input_dict["question"])}

    # RunnableRouter 대신 RunnableBranch를 사용하여 의도에 따라 분기
    intent_branch = RunnableBranch(
        (lambda x: x.get("intent") == "qa", route_to_qa),
        (lambda x: x.get("intent") == "summarize", route_to_summarize),
        (lambda x: x.get("intent") == "recommend", route_to_recommend),
        (lambda x: x.get("intent") == "chat", route_to_chat),
        # 위에 해당하지 않는 모든 경우 (fallback 등)
        RunnableLambda(lambda x: {"answer": "죄송하지만 해당 기능은 지원하지 않습니다."})
    )

    # 신뢰도(confidence)에 따라 분기 처리
    confidence_branch = RunnableBranch(
        # 신뢰도가 임계값보다 높으면, 위에서 정의한 intent_branch를 실행
        (lambda x: x.get("confidence", 0.0) > confidence_threshold, intent_branch),
        # 신뢰도가 낮거나 confidence 키가 없을 경우의 기본 응답
        RunnableLambda(lambda x: {"answer": "질문 의도를 명확하게 파악하지 못했습니다. 더 자세히 질문해주시겠어요?"}),
    )

    return router_chain | confidence_branch
