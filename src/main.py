# %%
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda

from config import load_environment, configure_langsmith
from core import LLMManager, VectorStoreManager, QAChain, RouterChain, build_dispatcher_chain
from utils import PDFProcessor


# %%
# 1. 각종 변수 세팅
load_environment()
configure_langsmith()

# TODO: 나중에는 yaml로 옮기든 파라미터 넣어주는 식으로 바꾸든지 해야함.
llm_provider = "upstage"  # openai / claude / upstage / ollama
embedding_provider = "upstage" # openai / huggingface / upstage
vector_store_path = Path("../upstage_vectorstore")

# %%
# 2. LLM/Embedding 설정
llm_manager = LLMManager(
    llm_provider=llm_provider,
    embedding_provider=embedding_provider,
    model_name="solar-pro2-250710",
    embedding_name="solar-embedding-1-large",
    temperature=0
)

# 3. 폴더 안에 있는 문서 처리
pdf_processor = PDFProcessor(
    folder_path=Path("../data"),       # PDF 폴더 경로
    embeddings=llm_manager.get_embeddings(),
    store_path=vector_store_path,
    chunk_size=1000,
    chunk_overlap=100
)
retriever = pdf_processor.process()
# 4. 벡터스토어 불러오기
vector_manager = VectorStoreManager(llm_manager.get_embeddings(), store_path=vector_store_path)
retriever = vector_manager.load_store()


# 5. QA 체인 생성
# qa_chain = QAChain(retriever, llm_manager.get_llm(), vector_manager)

# %%
# 6. 질문 실행
# print(qa_chain.run("어떤 기업의 리포트가 있어?"))

# %%
# print(qa_chain.run("이 기업의 시가총액은 얼마야?"))

# %%
# print(qa_chain.run("카카오의 리포트 요약해줘."))
# %%
# print(qa_chain.run("삼성전자 기업의 리포트 요약해줘"))
# %%

# Router 체인 생성
router_chain_instance = RouterChain(retriever, llm_manager.get_llm(), vector_manager)

# ---------------------
# 7. Downstream 체인들 생성
# ---------------------

# 대화 기록을 저장하기 위한 메모리
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 각 의도에 맞는 체인들을 RouterChain의 정적 메서드를 사용하여 생성
llm = llm_manager.get_llm()
rag_chain = router_chain_instance.build_qa_chain(llm, retriever, memory)
base_summarizer_chain = router_chain_instance.build_summarizer_chain(llm)
# vector_manager가 vectorstore 역할을 하도록 전달
recommender_chain = router_chain_instance.build_recommend_chain(llm, vector_manager)
chat_llm_chain = router_chain_instance.build_chat_chain(llm)

# 요약 체인은 '검색 -> 요약'의 두 단계로 구성되어야 합니다.
# 1. retriever로 관련 문서를 검색합니다.
# 2. 검색된 문서를 base_summarizer_chain의 'input_documents'로 전달합니다.
summarizer_chain = (
    retriever | RunnableLambda(lambda docs: {"input_documents": docs}) | base_summarizer_chain
)

# router_chain_instance는 Runnable이 아니므로 LCEL 체인에 직접 연결할 수 없습니다.
# classify_intent 메서드를 호출하고 그 결과를 원본 입력과 병합하는 RunnableLambda를 생성합니다.
def classify_and_passthrough(input_dict: dict) -> dict:
    """의도를 분류하고 원본 질문을 다음 체인으로 전달합니다."""
    question = input_dict["question"]
    classification_result = router_chain_instance.classify_intent(question)
    # 원본 입력과 분류 결과를 합쳐서 다음 단계로 전달
    return {**input_dict, **classification_result}

intent_classifier_runnable = RunnableLambda(classify_and_passthrough)

# ---------------------
# 8. Dispatcher 체인 생성
# ---------------------
# 의도에 따라 적절한 체인으로 분기하는 최종 체인
dispatcher = build_dispatcher_chain(
    router_chain=intent_classifier_runnable,  # Runnable로 변환된 분류기
    rag_chain=rag_chain,
    summarizer_chain=summarizer_chain,
    recommender_chain=recommender_chain,
    chat_llm_chain=chat_llm_chain,
    confidence_threshold=0.7  # 신뢰도 임계값 설정
)

# ---------------------
# 9. 체인 실행 예시
# ---------------------
def qa_endpoint(question: str):
    """최종적으로 구성된 dispatcher 체인을 실행하는 엔드포인트 함수"""
    result = dispatcher.invoke({"question": question})
    return result

# 예시 질문 실행
print("--- 요약 요청 ---")
print(qa_endpoint("카카오의 리포트 요약해줘."))
print("\n--- RAG(QA) 요청 ---")
print(qa_endpoint("삼성전자의 최근 실적은 어때?"))