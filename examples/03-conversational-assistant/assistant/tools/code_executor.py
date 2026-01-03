# assistant/tools/code_executor.py
"""
============================================================================
코드 실행 도구 모듈
============================================================================

이 모듈은 Python 코드 실행, 분석, 생성 도구를 제공합니다.
보안상의 이유로 execute_python 도구는 사용자 승인이 필요합니다.

핵심 개념:
    - 코드 샌드박싱: 위험한 명령어 필터링
    - subprocess: 격리된 환경에서 코드 실행
    - 임시 파일: tempfile을 사용한 안전한 코드 저장

보안 고려사항:
    - 위험한 import 문 차단 (os, subprocess, sys 등)
    - eval(), exec(), open() 등 위험 함수 차단
    - 타임아웃 설정으로 무한 루프 방지
    - TOOLS_REQUIRING_APPROVAL에 등록하여 사용자 승인 필요

사용 예시:
    from assistant.tools.code_executor import execute_python

    # 코드 실행 (사용자 승인 필요)
    result = execute_python.invoke({
        "code": "print('Hello, World!')"
    })

============================================================================
"""
from langchain_core.tools import tool
import subprocess
import tempfile
import os
from typing import Optional


@tool
def execute_python(code: str, timeout: int = 30) -> str:
    """
    ========================================================================
    Python 코드를 실행합니다.
    ========================================================================

    주의: 이 도구는 안전한 환경에서만 사용하세요.
    실제 시스템에서 코드를 실행하므로 사용자 승인이 필요합니다.

    Args:
        code: 실행할 Python 코드
            여러 줄 코드 지원
        timeout: 실행 제한 시간 (초, 기본값: 30)
            무한 루프 방지용

    Returns:
        실행 결과 또는 에러 메시지

    보안 제한:
        다음 패턴이 포함된 코드는 실행 거부됨:
        - import os, import subprocess, import sys
        - __import__, eval(), exec()
        - open(), shutil, rmdir, remove, unlink

    사용 예시:
        # 간단한 계산
        execute_python.invoke({
            "code": "result = sum(range(100))\\nprint(result)"
        })

        # 리스트 처리
        execute_python.invoke({
            "code": '''
            numbers = [1, 2, 3, 4, 5]
            squared = [x**2 for x in numbers]
            print(squared)
            '''
        })

    주의사항:
        - 이 도구는 TOOLS_REQUIRING_APPROVAL에 등록되어 있음
        - LLM이 이 도구를 호출하려 하면 사용자 승인 프로세스 진행
        - 승인 없이는 실행되지 않음
    ========================================================================
    """
    # ========================================================================
    # 위험한 패턴 필터링
    # ========================================================================
    # 코드 실행 전에 위험한 명령어가 있는지 검사합니다.
    #
    # 필터링 전략:
    #   1. 시스템 접근 모듈 차단 (os, subprocess, sys)
    #   2. 동적 실행 함수 차단 (__import__, eval, exec)
    #   3. 파일 시스템 접근 차단 (open, shutil, rmdir 등)
    #
    # 한계점:
    #   - 문자열 조작으로 우회 가능 (예: getattr)
    #   - 프로덕션에서는 Docker 컨테이너 등 격리 환경 필요
    # ========================================================================
    dangerous_patterns = [
        "import os", "import subprocess", "import sys",
        "__import__", "eval(", "exec(", "open(",
        "shutil", "rmdir", "remove", "unlink"
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return f"보안 오류: '{pattern}' 사용이 제한됩니다."

    try:
        # ====================================================================
        # 임시 파일에 코드 저장
        # ====================================================================
        # tempfile.NamedTemporaryFile 상세 설명:
        #
        # 임시 파일을 생성하는 Python 표준 라이브러리 함수입니다.
        # 프로세스 간 코드 전달에 안전하게 사용됩니다.
        #
        # 주요 매개변수:
        #   - mode (str): 파일 모드 ('w' = 쓰기)
        #   - suffix (str): 파일 확장자
        #   - delete (bool): 닫을 때 삭제 여부
        #       False로 설정하여 수동 삭제 (subprocess가 읽을 수 있도록)
        #
        # 사용 패턴:
        #   with tempfile.NamedTemporaryFile(...) as f:
        #       f.write(content)
        #       f.name  # 파일 경로
        # ====================================================================
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False  # 수동으로 삭제
        ) as f:
            f.write(code)
            temp_file = f.name

        # ====================================================================
        # subprocess로 코드 실행
        # ====================================================================
        # subprocess.run() 상세 설명:
        #
        # 새로운 프로세스를 생성하여 명령을 실행합니다.
        # 메인 프로세스와 분리되어 실행되므로 격리 효과가 있습니다.
        #
        # 주요 매개변수:
        #   - args (list): 실행할 명령과 인자
        #   - capture_output (bool): stdout/stderr 캡처
        #   - text (bool): 출력을 문자열로 디코딩
        #   - timeout (int): 타임아웃 (초)
        #       초과 시 subprocess.TimeoutExpired 예외 발생
        #
        # 반환값 (CompletedProcess):
        #   - returncode: 종료 코드 (0 = 성공)
        #   - stdout: 표준 출력
        #   - stderr: 표준 에러
        #
        # 보안 고려:
        #   - shell=True 사용 금지 (쉘 인젝션 위험)
        #   - 입력 검증 필수
        # ====================================================================
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # 임시 파일 삭제 (보안)
        os.unlink(temp_file)

        # 결과 조합
        output = result.stdout
        if result.stderr:
            output += f"\n에러:\n{result.stderr}"

        return output if output.strip() else "코드가 성공적으로 실행되었습니다. (출력 없음)"

    except subprocess.TimeoutExpired:
        # 타임아웃 발생 시 임시 파일 정리 시도
        try:
            os.unlink(temp_file)
        except:
            pass
        return f"실행 시간 초과 ({timeout}초)"
    except Exception as e:
        return f"실행 오류: {str(e)}"


@tool
def explain_code(code: str, language: str = "python") -> str:
    """
    ========================================================================
    코드를 분석하고 설명합니다.
    ========================================================================

    코드의 구조를 분석하여 기본적인 통계를 제공합니다.
    LLM이 추가적인 설명을 제공할 수 있는 기반 정보입니다.

    Args:
        code: 분석할 코드
        language: 프로그래밍 언어 (기본값: "python")
            현재는 Python만 지원

    Returns:
        코드 분석 결과
        - 총 라인 수
        - 함수 정의 수
        - 클래스 정의 수
        - 주석 수
        - import 문 수

    사용 예시:
        explain_code.invoke({
            "code": '''
            import math

            def calculate_area(radius):
                # 원의 면적 계산
                return math.pi * radius ** 2

            class Circle:
                def __init__(self, radius):
                    self.radius = radius
            '''
        })

        # 결과:
        # 코드 분석 결과 (python):
        #   - 총 라인 수: 10
        #   - 함수 정의: 1
        #   - 클래스 정의: 1
        #   - 주석: 1
        #   - import 문: 1
    ========================================================================
    """
    # ========================================================================
    # 간단한 구조 분석
    # ========================================================================
    # 라인 단위로 코드를 분석합니다.
    # startswith()를 사용한 패턴 매칭으로 구조 파악
    #
    # 분석 항목:
    #   - 'def ': 함수 정의
    #   - 'class ': 클래스 정의
    #   - '#': 주석 (라인 시작)
    #   - 'import ', 'from ': import 문
    #
    # 한계점:
    #   - 멀티라인 문자열 내의 패턴도 카운트됨
    #   - 들여쓰기 레벨 무시
    #   - docstring을 주석으로 인식하지 않음
    # ========================================================================
    lines = code.strip().split('\n')
    analysis = {
        "총 라인 수": len(lines),
        "함수 정의": sum(1 for l in lines if l.strip().startswith('def ')),
        "클래스 정의": sum(1 for l in lines if l.strip().startswith('class ')),
        "주석": sum(1 for l in lines if l.strip().startswith('#')),
        "import 문": sum(1 for l in lines if l.strip().startswith('import ') or l.strip().startswith('from '))
    }

    result = f"코드 분석 결과 ({language}):\n"
    for key, value in analysis.items():
        result += f"  - {key}: {value}\n"

    return result


@tool
def generate_code(description: str, language: str = "python") -> str:
    """
    ========================================================================
    설명을 바탕으로 코드 템플릿을 생성합니다.
    ========================================================================

    코드 작성의 시작점이 될 수 있는 기본 템플릿을 생성합니다.
    함수 또는 클래스 템플릿을 제공합니다.

    Args:
        description: 생성할 코드에 대한 설명
            "class" 또는 "클래스"가 포함되면 클래스 템플릿 생성
            그 외에는 함수 템플릿 생성
        language: 프로그래밍 언어 (기본값: "python")
            현재는 Python만 지원

    Returns:
        생성된 코드 템플릿

    사용 예시:
        # 함수 템플릿 생성
        generate_code.invoke({
            "description": "두 숫자의 최대공약수를 계산하는 함수"
        })

        # 클래스 템플릿 생성
        generate_code.invoke({
            "description": "사용자 정보를 관리하는 클래스"
        })

    참고:
        이 도구는 기본 템플릿만 제공합니다.
        실제 로직은 LLM이 추가로 작성해야 합니다.
    ========================================================================
    """
    # ========================================================================
    # 코드 템플릿 정의
    # ========================================================================
    # 문자열 템플릿과 .format()을 사용하여 설명을 삽입합니다.
    #
    # {description} 플레이스홀더:
    #   템플릿 내에서 설명이 삽입될 위치
    #   .format(description=...) 호출 시 치환됨
    # ========================================================================
    templates = {
        "함수": '''
def example_function(param1, param2):
    """
    {description}

    Args:
        param1: 첫 번째 매개변수
        param2: 두 번째 매개변수

    Returns:
        결과 값
    """
    # TODO: 구현
    pass
''',
        "클래스": '''
class ExampleClass:
    """
    {description}
    """

    def __init__(self):
        """초기화"""
        pass

    def method(self):
        """메서드"""
        pass
'''
    }

    # 설명에 "class" 또는 "클래스"가 포함되면 클래스 템플릿 사용
    if "class" in description.lower() or "클래스" in description:
        return templates["클래스"].format(description=description)
    else:
        return templates["함수"].format(description=description)
