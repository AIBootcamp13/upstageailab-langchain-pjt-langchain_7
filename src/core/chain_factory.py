from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda

from .llm_manager import LLMManager
from .vector_store import VectorStoreManager
from .router_chain import RouterChain
from .downstream_chains import build_dispatcher_chain
from utils import PDFProcessor

class ChainFactory:
    """
    애플리케이션에 필요한 모든 체인을 생성하고 관리하는 팩토리 클래스.
    설정을 기반으로 Dispatcher 체인을 생성합니다.
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): 애플리케이션 전체 설정 딕셔너리
        """
        self.config = config
        self.llm_manager = self._initialize_llm_manager()
        self.vector_manager = VectorStoreManager(
            self.llm_manager.get_embeddings(), 
            store_path=self.config["vector_store_path"]
        )
        self.retriever = self._prepare_retriever()

    def _initialize_llm_manager(self) -> LLMManager:
        """LLM 및 임베딩 매니저를 초기화합니다."""
        return LLMManager(
            llm_provider=self.config["llm_provider"],
            embedding_provider=self.config["embedding_provider"],
            model_name=self.config["model_name"],
            embedding_name=self.config["embedding_name"],
            temperature=self.config.get("temperature", 0)
        )

    def _prepare_retriever(self):
        """벡터 스토어를 준비하고 Retriever를 반환합니다."""
        vector_store_path = self.config["vector_store_path"]
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            print(f"'{vector_store_path}'가 존재하지 않거나 비어있어 새로 생성합니다...")
            pdf_processor = PDFProcessor(
                folder_path=self.config["data_folder_path"],
                embeddings=self.llm_manager.get_embeddings(),
                store_path=vector_store_path,
                chunk_size=self.config.get("chunk_size", 1000),
                chunk_overlap=self.config.get("chunk_overlap", 100),
            )
            pdf_processor.process()
        else:
            print(f"기존 벡터스토어 '{vector_store_path}'를 로드합니다.")
        
        # if/else 블록이 끝난 후, 디스크에 생성되었거나 이미 존재하던
        # 벡터 스토어를 로드하여 retriever 객체를 반환합니다.
        # 이로써 항상 유효한 retriever 객체가 반환됨을 보장합니다.
        return self.vector_manager.load_store()

    def create_dispatcher(self):
        """모든 하위 체인을 구성하고 최종 Dispatcher 체인을 생성합니다."""
        router_chain_instance = RouterChain(self.retriever, self.llm_manager.get_llm(), self.vector_manager)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        
        llm = self.llm_manager.get_llm()
        rag_chain = router_chain_instance.build_qa_chain(llm, self.retriever, memory)
        base_summarizer_chain = router_chain_instance.build_summarizer_chain(llm)
        recommender_chain = router_chain_instance.build_recommend_chain(llm, self.vector_manager)
        chat_llm_chain = router_chain_instance.build_chat_chain(llm)

        summarizer_chain = (
            self.retriever | RunnableLambda(lambda docs: {"input_documents": docs}) | base_summarizer_chain
        )

        def classify_and_passthrough(input_dict: dict) -> dict:
            question = input_dict["question"]
            classification_result = router_chain_instance.classify_intent(question)
            return {**input_dict, **classification_result}

        intent_classifier_runnable = RunnableLambda(classify_and_passthrough)

        dispatcher = build_dispatcher_chain(
            router_chain=intent_classifier_runnable,
            rag_chain=rag_chain,
            summarizer_chain=summarizer_chain,
            recommender_chain=recommender_chain,
            chat_llm_chain=chat_llm_chain,
            confidence_threshold=self.config.get("confidence_threshold", 0.7)
        )
        
        return dispatcher