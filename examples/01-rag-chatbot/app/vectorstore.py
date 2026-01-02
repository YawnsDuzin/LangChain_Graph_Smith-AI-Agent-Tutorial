# app/vectorstore.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from app.config import Config


class VectorStoreManager:
    """벡터 스토어 관리 클래스"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL
        )
        self.vectorstore: Optional[Chroma] = None
        self.persist_directory = str(Config.VECTORDB_DIR)

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """새 벡터 스토어 생성"""
        print("🔄 벡터 스토어 생성 중...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="rag_collection"
        )

        print(f"✅ 벡터 스토어 생성 완료: {len(documents)}개 문서")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """기존 벡터 스토어 로드"""
        print("🔄 벡터 스토어 로드 중...")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        print("✅ 벡터 스토어 로드 완료")
        return self.vectorstore

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = None
    ) -> BaseRetriever:
        """리트리버 생성"""
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K

        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        return retriever

    def similarity_search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """유사도 검색"""
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K
        results = self.vectorstore.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """점수와 함께 유사도 검색"""
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def add_documents(self, documents: List[Document]) -> None:
        """문서 추가"""
        if self.vectorstore is None:
            self.load_vectorstore()

        self.vectorstore.add_documents(documents)
        print(f"✅ {len(documents)}개 문서 추가됨")

    def delete_collection(self) -> None:
        """컬렉션 삭제"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print("🗑️ 컬렉션 삭제됨")


# 사용 예시
if __name__ == "__main__":
    from app.document_loader import DocumentProcessor

    # 문서 처리
    processor = DocumentProcessor()
    chunks = processor.process_documents()

    # 벡터 스토어 생성
    vs_manager = VectorStoreManager()
    vs_manager.create_vectorstore(chunks)

    # 검색 테스트
    results = vs_manager.similarity_search("LangChain이란?")
    for doc in results:
        print(f"- {doc.page_content[:100]}...")
