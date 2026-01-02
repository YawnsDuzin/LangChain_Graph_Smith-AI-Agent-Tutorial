# assistant/tools/code_executor.py
from langchain_core.tools import tool
import subprocess
import tempfile
import os
from typing import Optional


@tool
def execute_python(code: str, timeout: int = 30) -> str:
    """
    Python 코드를 실행합니다. 주의: 이 도구는 안전한 환경에서만 사용하세요.

    Args:
        code: 실행할 Python 코드
        timeout: 실행 제한 시간 (초)

    Returns:
        실행 결과 또는 에러 메시지
    """
    # 위험한 명령어 필터링
    dangerous_patterns = [
        "import os", "import subprocess", "import sys",
        "__import__", "eval(", "exec(", "open(",
        "shutil", "rmdir", "remove", "unlink"
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return f"보안 오류: '{pattern}' 사용이 제한됩니다."

    try:
        # 임시 파일에 코드 저장
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        # 코드 실행
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # 임시 파일 삭제
        os.unlink(temp_file)

        output = result.stdout
        if result.stderr:
            output += f"\n에러:\n{result.stderr}"

        return output if output.strip() else "코드가 성공적으로 실행되었습니다. (출력 없음)"

    except subprocess.TimeoutExpired:
        return f"실행 시간 초과 ({timeout}초)"
    except Exception as e:
        return f"실행 오류: {str(e)}"


@tool
def explain_code(code: str, language: str = "python") -> str:
    """
    코드를 분석하고 설명합니다.

    Args:
        code: 분석할 코드
        language: 프로그래밍 언어

    Returns:
        코드 설명
    """
    # 간단한 구조 분석
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
    설명을 바탕으로 코드 템플릿을 생성합니다.

    Args:
        description: 생성할 코드에 대한 설명
        language: 프로그래밍 언어

    Returns:
        생성된 코드 템플릿
    """
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

    if "class" in description.lower() or "클래스" in description:
        return templates["클래스"].format(description=description)
    else:
        return templates["함수"].format(description=description)
