# app/document_loader.py
"""
문서 로딩 및 처리 모듈

이 모듈은 RAG 파이프라인의 첫 번째 단계인 문서 처리를 담당합니다.
PDF, TXT, Markdown 파일을 로드하고, 청크로 분할합니다.

RAG 파이프라인 개요:
1. 문서 로딩 (Document Loading) <- 이 모듈
2. 청크 분할 (Text Splitting) <- 이 모듈
3. 임베딩 생성 (Embedding)
4. 벡터 저장 (Vector Store)
5. 검색 (Retrieval)
6. 생성 (Generation)

LangChain 문서 로더 개념:
- 다양한 소스(파일, URL, DB 등)에서 데이터를 읽어 Document 객체로 변환
- Document 객체는 page_content(텍스트)와 metadata(메타정보)를 포함
- 일관된 인터페이스로 다양한 형식 처리 가능
"""

from pathlib import Path
from typing import List

# =============================================================================
# LangChain 임포트 설명
# =============================================================================

# langchain_core.documents.Document
# ---------------------------------
# LangChain의 핵심 데이터 구조입니다.
# 모든 문서 처리 파이프라인에서 사용되는 표준 형식입니다.
#
# 구조:
#   Document(
#       page_content: str,  # 실제 텍스트 내용
#       metadata: dict      # 부가 정보 (출처, 페이지 번호, 날짜 등)
#   )
#
# 예시:
#   doc = Document(
#       page_content="LangChain은 LLM 프레임워크입니다.",
#       metadata={"source": "guide.pdf", "page": 1}
#   )
from langchain_core.documents import Document

# langchain_community.document_loaders
# ------------------------------------
# 다양한 형식의 문서를 로드하는 클래스들입니다.
# langchain_community 패키지에는 커뮤니티가 개발한 통합(integrations)이 포함됩니다.
#
# 사용 가능한 주요 로더:
# - PyPDFLoader: PDF 파일 (pypdf 라이브러리 사용)
# - TextLoader: 일반 텍스트 파일
# - UnstructuredMarkdownLoader: 마크다운 파일
# - CSVLoader: CSV 파일
# - JSONLoader: JSON 파일
# - WebBaseLoader: 웹 페이지
# - YoutubeLoader: 유튜브 자막
# - DirectoryLoader: 디렉토리 전체 로드
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

# langchain_text_splitters.RecursiveCharacterTextSplitter
# --------------------------------------------------------
# 텍스트를 의미 있는 단위로 분할하는 클래스입니다.
# "Recursive"인 이유: 여러 구분자를 순차적으로 시도하여 가장 적절한 분할점을 찾음
#
# 분할 전략 (순서대로 시도):
# 1. 단락 분리 ("\n\n")
# 2. 줄바꿈 ("\n")
# 3. 문장 종결 (".", "!", "?")
# 4. 쉼표 (",")
# 5. 공백 (" ")
# 6. 문자 단위 ("")
#
# 왜 사용하나?
# - 문장 중간에서 자르지 않음
# - 의미 단위를 최대한 유지
# - 청크 크기를 균일하게 유지
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Config


class DocumentProcessor:
    """
    문서 로딩 및 처리 클래스

    이 클래스는 다양한 형식의 문서를 로드하고 RAG에 적합한 청크로 분할합니다.

    주요 기능:
    - PDF, TXT, MD 파일 로드
    - 문서를 적절한 크기의 청크로 분할
    - 메타데이터 자동 추가

    사용 예시:
        processor = DocumentProcessor()

        # 단일 파일 로드
        docs = processor.load_pdf("/path/to/file.pdf")

        # 디렉토리 전체 처리
        chunks = processor.process_documents("/path/to/docs")
    """

    def __init__(self):
        """
        DocumentProcessor 초기화

        RecursiveCharacterTextSplitter 인스턴스를 생성합니다.
        설정값은 Config 클래스에서 가져옵니다.
        """
        # =====================================================================
        # RecursiveCharacterTextSplitter 상세 설명
        # =====================================================================
        #
        # 파라미터 설명:
        #
        # chunk_size (int): 각 청크의 최대 문자 수
        #   - 너무 작으면: 문맥 손실, 검색 정확도 저하
        #   - 너무 크면: 관련 없는 내용 포함, LLM 토큰 낭비
        #   - 권장: 500~2000 (문서 특성에 따라 조절)
        #
        # chunk_overlap (int): 인접 청크 간 겹치는 문자 수
        #   - 목적: 문맥 연속성 유지, 경계에서 정보 손실 방지
        #   - 권장: chunk_size의 10~20%
        #   - 예: chunk_size=1000이면 overlap=100~200
        #
        # length_function (callable): 텍스트 길이 계산 함수
        #   - len: 문자 수 기준 (기본값, 빠름)
        #   - tiktoken: 토큰 수 기준 (더 정확하지만 느림)
        #
        # separators (list): 분할 시 사용할 구분자 우선순위
        #   - 첫 번째 구분자로 시도, 청크가 너무 크면 다음 구분자 시도
        #   - ["\n\n", "\n", ".", " ", ""] 형태로 구성
        #   - 빈 문자열 ""은 최후의 수단 (문자 단위 분할)
        #
        # 작동 방식:
        # 1. 첫 번째 구분자("\n\n")로 텍스트 분할
        # 2. 각 조각이 chunk_size보다 크면 다음 구분자로 재분할
        # 3. chunk_size 이하가 될 때까지 반복
        # 4. 인접 청크에 overlap만큼 중복 포함
        # =====================================================================
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,           # 최대 1000자
            chunk_overlap=Config.CHUNK_OVERLAP,     # 200자 겹침
            length_function=len,                     # 문자 수로 길이 계산
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            # 분할 우선순위:
            # 1. 단락("\n\n") - 가장 자연스러운 분할
            # 2. 줄바꿈("\n") - 문단 내 구분
            # 3. 문장 종결부호(".", "!", "?") - 문장 단위
            # 4. 쉼표(",") - 절 단위
            # 5. 공백(" ") - 단어 단위
            # 6. 빈문자("") - 문자 단위 (최후의 수단)
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        PDF 파일 로딩

        PyPDFLoader를 사용하여 PDF 파일의 각 페이지를 Document로 변환합니다.

        Args:
            file_path (str): PDF 파일 경로

        Returns:
            List[Document]: 페이지별 Document 리스트

        PyPDFLoader 상세 설명:
        -----------------------
        - pypdf 라이브러리를 내부적으로 사용
        - 각 페이지가 별도의 Document 객체가 됨
        - metadata에 페이지 번호(page) 자동 추가
        - 텍스트 추출만 지원 (이미지, 표는 추출 안됨)

        예시:
            loader = PyPDFLoader("manual.pdf")
            docs = loader.load()
            # docs[0].page_content: 1페이지 텍스트
            # docs[0].metadata: {'source': 'manual.pdf', 'page': 0}

        대안 로더:
        - PyMuPDFLoader: 더 빠른 PDF 처리 (fitz 라이브러리)
        - PDFMinerLoader: 레이아웃 분석 지원
        - PDFPlumberLoader: 표 추출에 강점
        - UnstructuredPDFLoader: 복잡한 PDF에 적합
        """
        # PyPDFLoader 인스턴스 생성
        # 파일 경로만 전달하면 됨
        loader = PyPDFLoader(file_path)

        # load() 메서드로 PDF 로드
        # 반환값: List[Document] (페이지당 하나의 Document)
        documents = loader.load()

        # 커스텀 메타데이터 추가
        return self._add_metadata(documents, file_path)

    def load_text(self, file_path: str) -> List[Document]:
        """
        텍스트 파일 로딩

        TextLoader를 사용하여 일반 텍스트 파일을 Document로 변환합니다.

        Args:
            file_path (str): 텍스트 파일 경로

        Returns:
            List[Document]: Document 리스트 (보통 1개)

        TextLoader 상세 설명:
        ----------------------
        - 가장 단순한 로더
        - 파일 전체를 하나의 Document로 로드
        - encoding 파라미터로 문자 인코딩 지정 가능

        파라미터:
        - file_path: 파일 경로
        - encoding: 문자 인코딩 (기본값: 시스템 기본값)
        - autodetect_encoding: True면 자동 인코딩 감지

        주의사항:
        - 한글 파일은 encoding="utf-8" 필수
        - 대용량 파일은 메모리 이슈 가능 (청크 분할 권장)
        """
        # TextLoader 인스턴스 생성
        # encoding="utf-8"로 한글 파일 지원
        loader = TextLoader(file_path, encoding="utf-8")

        # load() 메서드로 파일 로드
        # 텍스트 파일은 보통 1개의 Document 반환
        documents = loader.load()

        return self._add_metadata(documents, file_path)

    def load_markdown(self, file_path: str) -> List[Document]:
        """
        마크다운 파일 로딩

        UnstructuredMarkdownLoader를 사용하여 Markdown 파일을 Document로 변환합니다.

        Args:
            file_path (str): 마크다운 파일 경로

        Returns:
            List[Document]: Document 리스트

        UnstructuredMarkdownLoader 상세 설명:
        -------------------------------------
        - unstructured 라이브러리 기반
        - 마크다운 구조를 인식하여 파싱
        - 헤더, 리스트, 코드 블록 등을 적절히 처리

        mode 파라미터:
        - "single": 전체를 하나의 Document로 (기본값)
        - "elements": 각 요소(헤더, 단락 등)를 별도 Document로
        - "paged": 페이지 단위로 분할

        대안:
        - MarkdownHeaderTextSplitter: 헤더 기준 분할에 특화
        - MarkdownLoader: 기본 마크다운 로더

        주의: unstructured 패키지 설치 필요
        pip install unstructured
        """
        # UnstructuredMarkdownLoader 인스턴스 생성
        loader = UnstructuredMarkdownLoader(file_path)

        # load() 메서드로 마크다운 파일 로드
        documents = loader.load()

        return self._add_metadata(documents, file_path)

    def load_directory(self, directory_path: str = None) -> List[Document]:
        """
        디렉토리의 모든 문서 로딩

        지정된 디렉토리에서 PDF, TXT, MD 파일을 모두 찾아 로드합니다.

        Args:
            directory_path (str, optional): 디렉토리 경로.
                                           None이면 Config.DATA_DIR 사용

        Returns:
            List[Document]: 모든 문서의 Document 리스트

        작동 방식:
        1. 디렉토리 내 모든 .pdf 파일 검색 및 로드
        2. 디렉토리 내 모든 .txt 파일 검색 및 로드
        3. 디렉토리 내 모든 .md 파일 검색 및 로드
        4. 모든 Document를 하나의 리스트로 병합

        참고: glob("**/*.pdf")는 하위 디렉토리도 재귀적으로 검색

        대안 - DirectoryLoader 사용:
        ----------------------------
        from langchain_community.document_loaders import DirectoryLoader

        loader = DirectoryLoader(
            path="./docs",
            glob="**/*.pdf",  # 파일 패턴
            loader_cls=PyPDFLoader  # 사용할 로더
        )
        docs = loader.load()
        """
        # 경로가 없으면 기본 데이터 디렉토리 사용
        if directory_path is None:
            directory_path = str(Config.DATA_DIR)

        all_documents = []
        path = Path(directory_path)

        # PDF 파일 로드
        # glob("**/*.pdf"): 현재 디렉토리와 모든 하위 디렉토리에서 .pdf 파일 검색
        for pdf_file in path.glob("**/*.pdf"):
            docs = self.load_pdf(str(pdf_file))
            all_documents.extend(docs)

        # 텍스트 파일 로드
        for txt_file in path.glob("**/*.txt"):
            docs = self.load_text(str(txt_file))
            all_documents.extend(docs)

        # 마크다운 파일 로드
        for md_file in path.glob("**/*.md"):
            docs = self.load_markdown(str(md_file))
            all_documents.extend(docs)

        print(f"📄 총 {len(all_documents)}개 문서 로드됨")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서를 청크로 분할

        로드된 문서들을 RecursiveCharacterTextSplitter로 청크로 분할합니다.

        Args:
            documents (List[Document]): 분할할 Document 리스트

        Returns:
            List[Document]: 분할된 청크 Document 리스트

        split_documents() 메서드 상세 설명:
        -----------------------------------
        - 입력: List[Document]
        - 출력: List[Document] (더 작은 청크들)

        작동 방식:
        1. 각 Document의 page_content를 청크로 분할
        2. 원본 metadata를 각 청크에 복사
        3. 새로운 Document 객체들 생성

        메타데이터 처리:
        - 원본 metadata가 모든 청크에 유지됨
        - 필요시 청크 인덱스 등 추가 정보 포함 가능

        예시:
            # 원본: 5000자 문서 1개
            docs = [Document(page_content="..."*5000, metadata={"source": "a.pdf"})]

            # 분할 후: 약 1000자씩 6개 청크
            chunks = splitter.split_documents(docs)
            # len(chunks) ≈ 6
            # chunks[0].metadata == {"source": "a.pdf"}  # 메타데이터 유지
        """
        # split_documents: Document 리스트를 입력받아 청크로 분할
        splits = self.text_splitter.split_documents(documents)
        print(f"📝 총 {len(splits)}개 청크로 분할됨")
        return splits

    def _add_metadata(self, documents: List[Document], file_path: str) -> List[Document]:
        """
        문서에 메타데이터 추가 (private 메서드)

        각 Document에 파일 정보를 메타데이터로 추가합니다.

        Args:
            documents (List[Document]): 메타데이터를 추가할 Document 리스트
            file_path (str): 원본 파일 경로

        Returns:
            List[Document]: 메타데이터가 추가된 Document 리스트

        메타데이터의 중요성:
        -------------------
        1. 출처 추적: 답변 생성 시 출처 표시 가능
        2. 필터링: 특정 문서만 검색 가능
        3. 디버깅: 문제 발생 시 원인 파악 용이

        추가되는 메타데이터:
        - source: 파일명 (예: "manual.pdf")
        - file_path: 전체 파일 경로
        """
        file_name = Path(file_path).name
        for doc in documents:
            # metadata는 dict이므로 키-값 추가 가능
            doc.metadata["source"] = file_name
            doc.metadata["file_path"] = file_path
        return documents

    def process_documents(self, directory_path: str = None) -> List[Document]:
        """
        전체 문서 처리 파이프라인

        문서 로드와 청크 분할을 한 번에 수행합니다.

        Args:
            directory_path (str, optional): 문서 디렉토리 경로

        Returns:
            List[Document]: 처리된 청크 Document 리스트

        파이프라인 흐름:
        1. load_directory(): 모든 문서 로드
        2. split_documents(): 청크로 분할
        3. 결과 반환

        이 메서드가 RAG 전처리의 핵심입니다.
        반환된 청크들은 벡터 스토어에 저장됩니다.
        """
        # 1단계: 문서 로드
        documents = self.load_directory(directory_path)

        # 2단계: 청크 분할
        splits = self.split_documents(documents)

        return splits


# =============================================================================
# 사용 예시 및 테스트
# =============================================================================
if __name__ == "__main__":
    """
    이 파일을 직접 실행할 때의 테스트 코드

    실행 방법:
        python -m app.document_loader

    이 코드는 모듈로 import될 때는 실행되지 않습니다.
    """
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    print(f"처리된 청크 수: {len(chunks)}")

    # 첫 번째 청크 내용 미리보기
    if chunks:
        print("\n--- 첫 번째 청크 미리보기 ---")
        print(f"내용: {chunks[0].page_content[:200]}...")
        print(f"메타데이터: {chunks[0].metadata}")
