# %%
from pathlib import Path
from config import load_environment, configure_langsmith
from core.chain_factory import ChainFactory


def main():
    """애플리케이션의 메인 실행 함수"""
    # 1. 환경 설정 로드
    load_environment()
    configure_langsmith()

    # 2. 설정 정의
    # 이 파일(main.py)의 부모 디렉터리(src)의 부모 디렉터리(프로젝트 루트)를 기준으로 경로 설정
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # TODO: 나중에는 yaml 파일에서 이 설정을 로드하도록 변경
    config = {
        "llm_provider": "upstage",
        "embedding_provider": "upstage",
        "model_name": "solar-pro2-250710",
        "embedding_name": "solar-embedding-1-large",
        "temperature": 0,
        "vector_store_path": PROJECT_ROOT / "upstage_vectorstore",
        "data_folder_path": PROJECT_ROOT / "data",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "confidence_threshold": 0.7,
    }

    # 3. 팩토리를 사용하여 Dispatcher 체인 생성
    print("Dispatcher 체인을 초기화합니다...")
    factory = ChainFactory(config)
    dispatcher = factory.create_dispatcher()
    print("초기화 완료.")
    
    # 4. 예시 질문 실행
    print("\n--- 요약 요청 ---")
    print(dispatcher.invoke({"question": "카카오의 리포트 요약해줘."}))
    
    print("\n--- RAG(QA) 요청 ---")
    print(dispatcher.invoke({"question": "삼성전자의 최근 실적은 어때?"}))

if __name__ == "__main__":
    main()