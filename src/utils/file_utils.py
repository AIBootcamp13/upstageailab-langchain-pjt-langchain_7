import time
import json
from pathlib import Path

from tqdm import tqdm
from openai import RateLimitError

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager


class PDFProcessor:
    """폴더 내 PDF 파일을 처리하고 벡터 스토어에 통합하는 클래스"""

    def __init__(
        self,
        folder_path: Path,
        embeddings,
        store_path: Path = Path("vectorstore"),
        chunk_size: int = 1000,
        chunk_overlap: int = 50
    ):
        """폴더 경로, 임베딩, 처리 매개변수로 PDFProcessor 초기화"""
        self.folder_path = folder_path
        self.embeddings = embeddings
        self.store_path = store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self) -> set:
        """processed_files.json에서 이전에 처리된 PDF 파일 집합을 로드"""
        processed_log = self.store_path / "processed_files.json"
        if processed_log.exists():
            with open(processed_log, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()

    def _get_new_pdfs(self) -> list[Path]:
        """아직 처리되지 않은 폴더 내 새 PDF 파일을 식별"""
        pdf_files = list(self.folder_path.glob("*.pdf"))
        return [pdf for pdf in pdf_files if pdf.name not in self.processed_files]

    def _process_pdf_files(self, new_pdfs: list[Path]) -> list:
        """새 PDF 파일을 문서 청크로 처리"""
        all_chunks = []
        for pdf in tqdm(new_pdfs, desc="Processing PDFs into chunks"):
            processor = DocumentProcessor(pdf, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            docs = processor.load()
            chunks = processor.split(docs)
            all_chunks.extend(chunks)
        return all_chunks

    def _update_vector_store(self, chunks: list) -> None:
        manager = VectorStoreManager(self.embeddings, store_path=self.store_path)
        batch_size = 10
        max_retries = 5
        for i in tqdm(range(0, len(chunks), batch_size), desc="Updating vector store"):
            batch = chunks[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    if self.store_path.exists() and any(self.store_path.iterdir()):
                        manager.load_store()
                        manager.add_documents(batch)
                    else:
                        manager.build_store(batch)
                        manager.save_store()
                    break
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = 30 * (2 ** attempt)  # 지수 백오프: 30초, 60초, 120초, 240초, 480초
                    print(f"Rate limit hit. Retrying in {wait_time // 60} minutes {wait_time % 60} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
            time.sleep(15)  # 배치 간 대기

    def _save_processed_files(self) -> None:
        """처리된 PDF 파일 집합을 processed_files.json에 저장"""
        processed_log = self.store_path / "processed_files.json"
        with open(processed_log, "w", encoding="utf-8") as f:
            json.dump(list(self.processed_files), f, ensure_ascii=False, indent=2)

    def process(self) -> None:
        """폴더 내 새 PDF를 처리하고 벡터 스토어에 추가"""
        # store_path 디렉토리가 존재하는지 확인
        self.store_path.mkdir(parents=True, exist_ok=True)

        # 새 PDF 식별
        new_pdfs = self._get_new_pdfs()

        if not new_pdfs:
            print("새로운 PDF가 없습니다. 업데이트 불필요.")
            return

        print(f"새로 감지된 PDF: {[pdf.name for pdf in new_pdfs]}")

        # 새 PDF를 청크로 처리
        all_chunks = self._process_pdf_files(new_pdfs)

        # 새 청크로 벡터 스토어 업데이트 또는 생성
        self._update_vector_store(all_chunks)

        # 처리된 파일 로그 업데이트
        self.processed_files.update([pdf.name for pdf in new_pdfs])
        self._save_processed_files()

        print("벡터스토어 업데이트 완료.")