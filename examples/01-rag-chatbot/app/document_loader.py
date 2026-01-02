# app/document_loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config


class DocumentProcessor:
    """문서 로딩 및 처리 클래스"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """PDF 파일 로딩"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self._add_metadata(documents, file_path)

    def load_text(self, file_path: str) -> List[Document]:
        """텍스트 파일 로딩"""
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        return self._add_metadata(documents, file_path)

    def load_markdown(self, file_path: str) -> List[Document]:
        """마크다운 파일 로딩"""
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        return self._add_metadata(documents, file_path)

    def load_directory(self, directory_path: str = None) -> List[Document]:
        """디렉토리의 모든 문서 로딩"""
        if directory_path is None:
            directory_path = str(Config.DATA_DIR)

        all_documents = []
        path = Path(directory_path)

        # PDF 파일
        for pdf_file in path.glob("**/*.pdf"):
            docs = self.load_pdf(str(pdf_file))
            all_documents.extend(docs)

        # 텍스트 파일
        for txt_file in path.glob("**/*.txt"):
            docs = self.load_text(str(txt_file))
            all_documents.extend(docs)

        # 마크다운 파일
        for md_file in path.glob("**/*.md"):
            docs = self.load_markdown(str(md_file))
            all_documents.extend(docs)

        print(f"📄 총 {len(all_documents)}개 문서 로드됨")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        splits = self.text_splitter.split_documents(documents)
        print(f"📝 총 {len(splits)}개 청크로 분할됨")
        return splits

    def _add_metadata(self, documents: List[Document], file_path: str) -> List[Document]:
        """문서에 메타데이터 추가"""
        file_name = Path(file_path).name
        for doc in documents:
            doc.metadata["source"] = file_name
            doc.metadata["file_path"] = file_path
        return documents

    def process_documents(self, directory_path: str = None) -> List[Document]:
        """전체 문서 처리 파이프라인"""
        documents = self.load_directory(directory_path)
        splits = self.split_documents(documents)
        return splits


# 사용 예시
if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    print(f"처리된 청크 수: {len(chunks)}")
