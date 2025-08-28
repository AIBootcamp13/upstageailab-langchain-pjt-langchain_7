from typing import List
from langchain_core.documents import Document


def format_docs_with_citations(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        pg = d.metadata.get("page", "?")
        parts.append(f"[{i}] {d.page_content}\n(출처: {src}, 페이지 {pg})")
    return "\n\n".join(parts)


def is_broad_question(question: str) -> bool:
    broad_keywords = ["알려줘", "무슨 정보 있어?", "어떤 기업 있어?"]
    return len(question.split()) < 3 or any(keyword in question for keyword in broad_keywords)