from typing import List
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStoreManager:
    """벡터 스토어를 관리하고 문서 및 기업 데이터를 처리하는 클래스"""

    def __init__(self, embeddings, store_path: Path = Path("vectorstore")):
        """임베딩과 저장 경로를 사용하여 VectorStoreManager 초기화"""
        self.embeddings = embeddings
        self.vectorstore: FAISS = None
        self.store_path: Path = store_path
        self.num_companies: int = 0  # 고유 기업 수를 저장하는 변수
        self._company_log: Path = store_path / "company_log.json"  # 기업 로그 파일 경로

    def build_store(self, documents: List[Document]) -> None:
        """새로운 벡터 스토어를 문서로 생성"""
        self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        self._update_company_log(documents)

    def save_store(self) -> None:
        """벡터 스토어를 로컬에 저장"""
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        self.vectorstore.save_local(str(self.store_path))

    def as_retriever(self):
        """벡터 스토어를 검색기로 반환"""
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        return self.vectorstore.as_retriever(search_kwargs={'k': 5})  # 질문과 가장 유사한 문서 5개를 검색

    def _update_company_log(self, documents: List[Document]) -> None:
        """문서에서 고유 기업 수를 계산하고 기업 로그를 업데이트"""
        unique_sources = set(doc.metadata.get('source', '') for doc in documents)
        self.num_companies = len(unique_sources)
        with open(self._company_log, 'w', encoding='utf-8') as f:
            json.dump(list(unique_sources), f, ensure_ascii=False, indent=2)

    def load_store(self):
        """기존 벡터 스토어를 불러오고 기업 수를 로드"""
        if self.store_path.exists():
            self.vectorstore = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            # 기존 기업 로그 로드
            if self._company_log.exists():
                with open(self._company_log, 'r', encoding='utf-8') as f:
                    self.num_companies = len(json.load(f))
            return self.as_retriever()
        else:
            raise FileNotFoundError("저장된 벡터 스토어가 없습니다.")

    def add_documents(self, documents: List[Document]) -> None:
        """기존 스토어에 문서를 추가하고 기업 수를 업데이트"""
        if self.vectorstore is None:
            self.load_store()
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local(str(self.store_path))
        
        # 기업 수 업데이트
        unique_sources = set(doc.metadata.get('source', '') for doc in documents)
        current_sources = set()
        if self._company_log.exists():
            with open(self._company_log, 'r', encoding='utf-8') as f:
                current_sources = set(json.load(f))
        new_sources = unique_sources - current_sources
        self.num_companies += len(new_sources)
        
        # 업데이트된 기업 로그 저장
        with open(self._company_log, 'w', encoding='utf-8') as f:
            json.dump(list(current_sources.union(unique_sources)), f, ensure_ascii=False, indent=2)

    def get_num_companies(self) -> int:
        """현재 저장된 고유 기업 수를 반환"""
        return self.num_companies