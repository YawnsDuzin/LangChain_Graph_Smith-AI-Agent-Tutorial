# app/vectorstore.py
"""
벡터 스토어 관리 모듈

이 모듈은 RAG 파이프라인에서 벡터 데이터베이스를 관리합니다.
문서 임베딩을 저장하고, 유사도 기반 검색을 수행합니다.

RAG 파이프라인에서의 역할:
1. 문서 로딩 (Document Loading)
2. 청크 분할 (Text Splitting)
3. 임베딩 생성 (Embedding) <- 이 모듈
4. 벡터 저장 (Vector Store) <- 이 모듈
5. 검색 (Retrieval) <- 이 모듈
6. 생성 (Generation)

벡터 스토어 개념:
-----------------
벡터 스토어는 텍스트의 임베딩(숫자 벡터)을 저장하고 검색하는 데이터베이스입니다.

작동 원리:
1. 텍스트 → 임베딩 모델 → 숫자 벡터 (예: 1536차원)
2. 벡터를 벡터 스토어에 저장
3. 검색 시: 질문 → 임베딩 → 가장 유사한 벡터 찾기 → 원본 텍스트 반환

유사도 측정:
- 코사인 유사도 (Cosine Similarity): 방향의 유사성 측정
- 유클리디안 거리 (Euclidean Distance): 직선 거리 측정
- 내적 (Dot Product): 크기와 방향 모두 고려
"""

from typing import List, Optional

# =============================================================================
# LangChain 임포트 설명
# =============================================================================

# langchain_core.documents.Document
# ---------------------------------
# 텍스트와 메타데이터를 담는 표준 데이터 구조
from langchain_core.documents import Document

# langchain_openai.OpenAIEmbeddings
# ---------------------------------
# OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터로 변환합니다.
#
# 임베딩이란?
# - 텍스트를 고정 길이의 숫자 벡터로 변환
# - 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치
# - 예: "강아지"와 "개"의 벡터는 "자동차"보다 서로 가까움
#
# 지원 모델:
# - text-embedding-3-small: 1536차원, 빠르고 저렴 (권장)
# - text-embedding-3-large: 3072차원, 더 정확하지만 비용 높음
# - text-embedding-ada-002: 1536차원, 이전 세대
#
# 사용 예시:
#   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#   vector = embeddings.embed_query("안녕하세요")
#   # vector: [0.123, -0.456, 0.789, ...] (1536개의 숫자)
from langchain_openai import OpenAIEmbeddings

# langchain_community.vectorstores.Chroma
# ----------------------------------------
# ChromaDB를 LangChain에서 사용하기 위한 래퍼 클래스입니다.
#
# ChromaDB 특징:
# - 오픈소스 벡터 데이터베이스
# - 로컬 파일 기반 (SQLite) 또는 서버 모드 지원
# - 메타데이터 필터링 지원
# - 간단한 설치와 사용 (pip install chromadb)
#
# 다른 벡터 스토어 옵션:
# - Pinecone: 클라우드 기반, 대규모 프로덕션용
# - Weaviate: 그래프 기능 포함, 클라우드/로컬
# - FAISS: Facebook AI, 빠른 로컬 검색
# - Milvus: 대규모 분산 시스템용
# - Qdrant: Rust 기반, 고성능
from langchain_community.vectorstores import Chroma

# langchain_core.retrievers.BaseRetriever
# ----------------------------------------
# 모든 리트리버의 기본 클래스입니다.
# 타입 힌트에 사용됩니다.
#
# 리트리버란?
# - 질문을 받아 관련 문서를 검색하는 컴포넌트
# - 벡터 스토어를 감싸서 통일된 인터페이스 제공
# - 체인에서 검색 단계를 담당
from langchain_core.retrievers import BaseRetriever

from app.config import Config


class VectorStoreManager:
    """
    벡터 스토어 관리 클래스

    ChromaDB 기반 벡터 스토어의 생성, 로드, 검색을 담당합니다.

    주요 기능:
    - 벡터 스토어 생성 및 로드
    - 유사도 검색 (점수 포함/미포함)
    - 문서 추가/삭제
    - 리트리버 생성

    사용 예시:
        manager = VectorStoreManager()

        # 새 벡터 스토어 생성
        manager.create_vectorstore(documents)

        # 검색
        results = manager.similarity_search("LangChain이란?")
    """

    def __init__(self):
        """
        VectorStoreManager 초기화

        OpenAI 임베딩 모델과 저장 경로를 설정합니다.
        """
        # =====================================================================
        # OpenAIEmbeddings 상세 설명
        # =====================================================================
        #
        # 생성자 파라미터:
        #
        # model (str): 사용할 임베딩 모델
        #   - "text-embedding-3-small": 1536차원, $0.02/1M tokens
        #   - "text-embedding-3-large": 3072차원, $0.13/1M tokens
        #   - "text-embedding-ada-002": 1536차원 (레거시)
        #
        # openai_api_key (str, optional): API 키
        #   - 생략하면 OPENAI_API_KEY 환경변수 자동 사용
        #
        # dimensions (int, optional): 출력 벡터 차원 수
        #   - text-embedding-3 모델에서만 지원
        #   - 예: dimensions=256 으로 차원 축소 가능 (검색 속도 향상)
        #
        # 주요 메서드:
        # - embed_query(text): 단일 텍스트 임베딩 (검색용)
        # - embed_documents(texts): 여러 텍스트 임베딩 (저장용)
        #
        # 사용 예시:
        #   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        #
        #   # 단일 쿼리 임베딩
        #   query_vector = embeddings.embed_query("LangChain이란?")
        #
        #   # 여러 문서 임베딩
        #   doc_vectors = embeddings.embed_documents(["문서1", "문서2"])
        # =====================================================================
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL  # "text-embedding-3-small"
        )

        # 벡터 스토어 인스턴스 (나중에 초기화)
        self.vectorstore: Optional[Chroma] = None

        # ChromaDB 데이터 저장 경로
        self.persist_directory = str(Config.VECTORDB_DIR)

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        새 벡터 스토어 생성

        문서 리스트를 임베딩하여 ChromaDB에 저장합니다.

        Args:
            documents (List[Document]): 저장할 문서 청크 리스트

        Returns:
            Chroma: 생성된 벡터 스토어 인스턴스

        Chroma.from_documents() 상세 설명:
        -----------------------------------
        이 클래스 메서드는 문서를 임베딩하고 벡터 스토어에 저장합니다.

        파라미터:
        - documents: List[Document] - 저장할 문서들
        - embedding: Embeddings - 사용할 임베딩 모델
        - persist_directory: str - 데이터 저장 경로 (영구 저장)
        - collection_name: str - 컬렉션 이름 (데이터 그룹)

        내부 동작:
        1. 각 문서의 page_content를 임베딩 모델로 벡터화
        2. 벡터와 원본 텍스트, 메타데이터를 ChromaDB에 저장
        3. persist_directory에 SQLite 파일로 영구 저장

        저장 구조:
        - vectordb/
          ├── chroma.sqlite3  # 메인 데이터베이스
          ├── [collection_id]/
          │   ├── data_level0.bin  # 벡터 데이터
          │   ├── header.bin
          │   ├── index_metadata.pickle
          │   └── length.bin
        """
        print("🔄 벡터 스토어 생성 중...")

        # Chroma.from_documents: 문서를 임베딩하고 저장
        # 이 한 줄에서 모든 임베딩과 저장이 수행됨
        self.vectorstore = Chroma.from_documents(
            documents=documents,              # 저장할 문서 청크들
            embedding=self.embeddings,        # OpenAI 임베딩 모델
            persist_directory=self.persist_directory,  # 저장 경로
            collection_name="rag_collection"  # 컬렉션 이름
        )

        print(f"✅ 벡터 스토어 생성 완료: {len(documents)}개 문서")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """
        기존 벡터 스토어 로드

        이전에 저장된 ChromaDB를 불러옵니다.

        Returns:
            Chroma: 로드된 벡터 스토어 인스턴스

        Chroma 생성자 상세 설명:
        -------------------------
        기존 데이터를 로드할 때는 생성자를 직접 호출합니다.

        파라미터:
        - persist_directory: str - 데이터가 저장된 경로
        - embedding_function: Embeddings - 검색 시 사용할 임베딩 모델
        - collection_name: str - 로드할 컬렉션 이름

        주의사항:
        - 저장할 때와 같은 임베딩 모델을 사용해야 함
        - 다른 모델 사용 시 검색 결과가 부정확해짐
        - 컬렉션 이름이 일치해야 함
        """
        print("🔄 벡터 스토어 로드 중...")

        # Chroma 생성자로 기존 데이터 로드
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,  # 주의: 여기선 embedding_function
            collection_name="rag_collection"
        )

        print("✅ 벡터 스토어 로드 완료")
        return self.vectorstore

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = None
    ) -> BaseRetriever:
        """
        리트리버 생성

        벡터 스토어를 리트리버로 변환하여 체인에서 사용할 수 있게 합니다.

        Args:
            search_type (str): 검색 유형
                - "similarity": 단순 유사도 검색 (기본값)
                - "mmr": Maximal Marginal Relevance (다양성 고려)
                - "similarity_score_threshold": 점수 임계값 적용

            k (int, optional): 반환할 문서 수. None이면 Config.RETRIEVAL_K 사용

        Returns:
            BaseRetriever: 생성된 리트리버 인스턴스

        as_retriever() 상세 설명:
        --------------------------
        벡터 스토어를 리트리버 인터페이스로 변환합니다.

        search_type 옵션:
        1. "similarity" (기본값):
           - 가장 유사한 k개 문서 반환
           - 빠르고 단순함

        2. "mmr" (Maximal Marginal Relevance):
           - 유사하면서도 다양한 문서 반환
           - 중복 내용 방지
           - search_kwargs: {"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
             - fetch_k: 초기 후보 수
             - lambda_mult: 다양성 vs 관련성 균형 (0~1)

        3. "similarity_score_threshold":
           - 임계값 이상 점수의 문서만 반환
           - search_kwargs: {"score_threshold": 0.5}

        search_kwargs 파라미터:
        - k: 반환할 문서 수
        - score_threshold: 최소 유사도 점수
        - filter: 메타데이터 필터링
          예: {"source": "manual.pdf"}

        리트리버 사용 예시:
        -------------------
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20}
        )

        # invoke 메서드로 검색
        docs = retriever.invoke("LangChain이란?")
        """
        # 벡터 스토어가 없으면 로드
        if self.vectorstore is None:
            self.load_vectorstore()

        # k값 설정 (기본값: Config.RETRIEVAL_K = 4)
        k = k or Config.RETRIEVAL_K

        # =====================================================================
        # as_retriever() 메서드
        # =====================================================================
        # 벡터 스토어를 리트리버 인터페이스로 변환
        #
        # 리트리버는 검색을 위한 표준 인터페이스를 제공:
        # - invoke(query): 동기 검색
        # - ainvoke(query): 비동기 검색
        # - batch(queries): 배치 검색
        #
        # 체인에서 사용 예시:
        #   chain = retriever | prompt | llm | parser
        #   result = chain.invoke("질문")
        # =====================================================================
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,  # 검색 유형
            search_kwargs={"k": k}    # 검색 파라미터
        )

        return retriever

    def similarity_search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        유사도 검색

        질문과 가장 유사한 문서 청크를 검색합니다.

        Args:
            query (str): 검색 질문
            k (int, optional): 반환할 문서 수

        Returns:
            List[Document]: 유사도 순으로 정렬된 문서 리스트

        similarity_search() 상세 설명:
        --------------------------------
        가장 기본적인 벡터 검색 메서드입니다.

        내부 동작:
        1. query를 임베딩 모델로 벡터화
        2. 벡터 스토어에서 유사한 벡터 검색
        3. 원본 Document 객체 반환

        유사도 계산:
        - ChromaDB는 기본적으로 L2 거리 (유클리디안) 사용
        - 거리가 작을수록 더 유사함
        - 코사인 유사도도 설정 가능

        사용 예시:
            results = vectorstore.similarity_search(
                query="LangChain 설치 방법",
                k=3
            )
            for doc in results:
                print(doc.page_content[:100])
                print(doc.metadata)
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K

        # similarity_search: 단순 유사도 검색
        # 반환값: List[Document] (k개)
        results = self.vectorstore.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """
        점수와 함께 유사도 검색

        유사도 점수를 포함하여 검색 결과를 반환합니다.

        Args:
            query (str): 검색 질문
            k (int, optional): 반환할 문서 수

        Returns:
            List[tuple]: (Document, score) 튜플 리스트

        similarity_search_with_score() 상세 설명:
        -------------------------------------------
        유사도 점수를 함께 반환하여 검색 품질을 평가할 수 있습니다.

        반환 형식:
        [(Document, score), (Document, score), ...]

        점수 해석:
        - ChromaDB: 거리 값 (낮을수록 유사)
          - 0.0: 완전히 동일
          - 0.0~0.5: 매우 유사
          - 0.5~1.0: 어느 정도 유사
          - 1.0+: 유사하지 않음

        주의: 점수 체계는 벡터 스토어마다 다름
        - Pinecone: 0~1 (높을수록 유사)
        - FAISS: 거리 (낮을수록 유사)

        사용 예시:
            results = vectorstore.similarity_search_with_score("질문")
            for doc, score in results:
                print(f"점수: {score:.3f}")
                print(f"내용: {doc.page_content[:100]}")
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K

        # similarity_search_with_score: 점수 포함 검색
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def add_documents(self, documents: List[Document]) -> None:
        """
        문서 추가

        기존 벡터 스토어에 새 문서를 추가합니다.

        Args:
            documents (List[Document]): 추가할 문서 리스트

        add_documents() 상세 설명:
        ---------------------------
        기존 컬렉션에 새 문서를 추가합니다.

        내부 동작:
        1. 새 문서들을 임베딩
        2. 기존 컬렉션에 벡터 추가
        3. 영구 저장 (persist_directory 설정 시)

        중복 처리:
        - ChromaDB는 기본적으로 ID 기반 중복 방지
        - 같은 ID로 추가하면 덮어쓰기
        - ID 없이 추가하면 새로 생성

        주의사항:
        - 대량 추가 시 배치 처리 권장
        - 너무 많은 문서는 메모리 이슈 발생 가능
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        # add_documents: 새 문서 추가
        self.vectorstore.add_documents(documents)
        print(f"✅ {len(documents)}개 문서 추가됨")

    def delete_collection(self) -> None:
        """
        컬렉션 삭제

        현재 벡터 스토어의 컬렉션을 삭제합니다.

        delete_collection() 상세 설명:
        --------------------------------
        컬렉션과 모든 데이터를 삭제합니다.

        주의사항:
        - 되돌릴 수 없음!
        - 데이터 백업 권장
        - 삭제 후 새 벡터 스토어 생성 필요
        """
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print("🗑️ 컬렉션 삭제됨")


# =============================================================================
# 사용 예시 및 테스트
# =============================================================================
if __name__ == "__main__":
    """
    벡터 스토어 테스트

    실행 방법:
        python -m app.vectorstore
    """
    from app.document_loader import DocumentProcessor

    # 1. 문서 처리
    processor = DocumentProcessor()
    chunks = processor.process_documents()

    if chunks:
        # 2. 벡터 스토어 생성
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks)

        # 3. 검색 테스트
        print("\n--- 검색 테스트 ---")
        results = vs_manager.similarity_search("LangChain이란?")
        for i, doc in enumerate(results):
            print(f"\n[결과 {i+1}]")
            print(f"내용: {doc.page_content[:150]}...")
            print(f"출처: {doc.metadata.get('source', 'Unknown')}")

        # 4. 점수 포함 검색 테스트
        print("\n--- 점수 포함 검색 ---")
        results_with_score = vs_manager.similarity_search_with_score("LangChain이란?")
        for doc, score in results_with_score:
            print(f"점수: {score:.4f} | {doc.page_content[:80]}...")
    else:
        print("처리할 문서가 없습니다.")
