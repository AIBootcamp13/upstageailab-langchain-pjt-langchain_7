from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

ROUTER_PROMPT_TEMPLATE = """
당신은 사용자의 질문을 QA Engine에서 처리할 Intent를 분류하는 분류기입니다.

사용자의 질문을 보고 아래 Intent 중 하나로만 분류하세요:
- qa: 구체적 사실 기반 질문, 리포트 내용 기반 답변 필요
- summarize: 문서나 리포트를 요약해달라는 요청
- recommend: 문서/리포트/자료 추천 요청
- chat: 일반적인 대화나 정보성 질문
- fallback: 위 어느 것도 아니거나 알 수 없음

질문: "{question}"

출력은 아래 JSON 형식으로만:
{{
"intent": "qa|summarize|recommend|chat|fallback",
"confidence": 0.0~1.0
}}"""

class RouterChain:
    INTENT_DESCRIPTION = {
        "qa": "리포트 기반 질의응답",
        "summarize": "문서 요약 요청",
        "recommend": "리포트/자료 추천 요청",
        "chat": "일반 대화",
        "fallback": "분류 불가 또는 미지원 질문"
    }
    
    def __init__(self, retriever, llm, vectorstore):
        self.retriever = retriever
        self.llm = llm
        self.vectorstore = vectorstore

        intent_schema = [
            ResponseSchema(name="intent", description="질문의 목적"),
            ResponseSchema(name="confidence", description="0~1 사이의 확신도")
        ]
        parser = StructuredOutputParser.from_response_schemas(intent_schema)
        prompt = PromptTemplate.from_template(
            template=ROUTER_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        self.router_chain = prompt | self.llm | parser
    
    def classify_intent(self, question):
        try:
            # 클래스 초기화 시 생성된 라우터 체인 사용
            result = self.router_chain.invoke({"question": question})
            intent = result.get("intent", "fallback")
            confidence = float(result.get("confidence", 0.0))

            # confidence 검증
            if confidence < 0 or confidence > 1:
                confidence = 0.0
                intent = "fallback"

            return {
                "intent": intent,
                "confidence": confidence, 
                "description": self.INTENT_DESCRIPTION.get(intent, "알 수 없는 의도"),
            }

        except Exception as e:
            # LLM 출력 포맷 에러 시 fallback
            return {
                "intent": "fallback",
                "confidence": 0.0,
                "description": f"Router Error: {str(e)}",
            }

    # QA 체인 (RAG 기반)
    @staticmethod
    def build_qa_chain(llm, retriever, memory):
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            chain_type="stuff",
        )


    # Summarizer 체인
    @staticmethod
    def build_summarizer_chain(llm):
        return load_summarize_chain(
            llm=llm,
            chain_type="map_reduce"
        )

    # Recommend 체인 
    @staticmethod
    def build_recommend_chain(llm, vector_store):
        def recommend(query):
            docs = vector_store.search(query, k=5)
            return [d.metadata["title"] for d in docs]
        return recommend

    # Chat 체인
    @staticmethod
    def build_chat_chain(llm):
        def chat(query):
            return llm.predict(query)
        return chat

# ======================
# 6. 예시 실행 (단독 실행 시 테스트)
# ======================
if __name__ == "__main__":
    import json

    from langchain.llms.fake import FakeListLLM

    # --- 테스트를 위한 Mock 객체 생성 ---
    # 예상 질문에 대한 LLM의 가짜 응답 목록
    responses = [
        json.dumps({"intent": "summarize", "confidence": 0.9}),
        json.dumps({"intent": "summarize", "confidence": 0.8}),
        json.dumps({"intent": "recommend", "confidence": 0.95}),
        json.dumps({"intent": "chat", "confidence": 0.7}),
        json.dumps({"intent": "fallback", "confidence": 0.5}),
    ]
    
    # FakeLLM: 미리 정해진 응답을 순서대로 반환하는 테스트용 LLM
    fake_llm = FakeListLLM(responses=responses)
    
    # RouterChain 인스턴스 생성 (retriever, vectorstore는 이 테스트에선 불필요)
    router = RouterChain(retriever=None, llm=fake_llm, vectorstore=None)

    # --- 테스트 질문 목록 ---
    examples = [
        "삼성전자 투자 포인트 요약해줘",
        "이 PDF 리포트 요약해줘",
        "자동차 섹터 리포트 추천해줘",
        "오늘 주식시장 분위기 어때?",
        "내일 주가 예측해줘" # fallback 케이스
    ]

    for q in examples:
        result = router.classify_intent(q)
        print(f"Q: {q}")
        print(f"→ Intent: {result['intent']} ({result['confidence']:.2f}) / {result['description']}")
        print("-" * 60)