from typing import Dict, Any, List
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores.base import VectorStoreRetriever

from utils import format_docs_with_citations, load_prompt, is_broad_question
from core import VectorStoreManager


class QAChain:
    def __init__(self, retriever: VectorStoreRetriever, llm, vectorstore: VectorStoreManager, prompt_file: Path = Path("qa_prompt.jinja")):
        self.retriever = retriever
        self.llm = llm
        self.vectorstore = vectorstore
        num_companies = self.vectorstore.get_num_companies()
        self.prompt = load_prompt(prompt_file)

    def run(self, question: str) -> RunnableSequence:
        context_chain = self.retriever | RunnableLambda(format_docs_with_citations)
        chain: RunnableSequence = (
            {"context": context_chain, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)


class QAMemoryChain:
    def __init__(self, retriever: VectorStoreRetriever, llm, vectorstore: VectorStoreManager, prompt_file: Path = Path("qa_memory_prompt.jinja")):
        """답변을 기억하는 체인입니다."""
        self.retriever = retriever
        self.llm = llm
        self.vectorstore = vectorstore
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
        
        loaded_prompt = load_prompt(prompt_file, Path("src/prompts"))

        # 1. 소스 문서를 포함하여, 프롬프트에 필요한 모든 입력을 준비하는 단계
        setup_and_retrieval = RunnableParallel(
            # context 키에 retriever가 반환하는 '문서 리스트'를 할당합니다.
            context=self.retriever, 
            question=RunnablePassthrough(),
            num_companies=RunnableLambda(lambda q: self.vectorstore.get_num_companies()),
            broad_question=RunnableLambda(is_broad_question),
            chat_history=RunnableLambda(
                lambda q: self.memory.load_memory_variables({"question": q}).get("chat_history")
            )
        )
        
        # 2. 답변을 생성하는 체인
        rag_chain_from_docs: RunnableSequence = (
            {
                "context": lambda x: format_docs_with_citations(x["context"]), # 문서 리스트를 문자열로 변환
                "question": lambda x: x["question"],
                "num_companies": lambda x: x["num_companies"],
                "broad_question": lambda x: x["broad_question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | loaded_prompt
            | self.llm
            | StrOutputParser()
        )

        # 3. 최종 체인: 문서 검색 후, 해당 문서와 질문을 사용하여 답변 생성
        self.qa_chain = setup_and_retrieval.assign(answer=rag_chain_from_docs)


    def run(self, question: str) -> Dict[str, Any]:
        """
        질문을 처리하고 답변과 소스 문서를 반환합니다.
        """
        # 체인을 실행하면 'context' (문서 리스트)와 'answer' (문자열)가 포함된 딕셔너리를 반환합니다.
        response = self.qa_chain.invoke(question)
        
        # 대화 내용을 메모리에 저장합니다.
        self.memory.save_context({"question": question}, {"answer": response["answer"]})
        
        # 소스 문서(response['context'])에서 인용 정보 추출
        citations = []
        source_documents = response.get('context', [])
        for doc in source_documents:
            src = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            citations.append(f"(출처: {src}, 페이지 {page})")
        
        final_answer = response['answer']
        if citations:
            final_answer += "\n\n참고 문서:\n" + "\n".join(citations)

        return {"answer": final_answer, "source_documents": source_documents}
    
    def clear_memory(self) -> None:
        """대화 히스토리를 초기화합니다."""
        self.memory.clear()
        
    def get_chat_history(self) -> List[Dict[str, str]]:
        """현재까지의 대화 히스토리를 반환합니다."""
        return self.memory.chat_memory.messages


    