# assistant/tools/utilities.py
"""
============================================================================
유틸리티 도구 모듈
============================================================================

이 모듈은 다양한 유틸리티 도구를 제공합니다.
시간/날짜 처리, JSON 포맷팅, 텍스트 요약 등의 기능을 포함합니다.

핵심 개념:
    - datetime: Python 표준 날짜/시간 라이브러리
    - pytz: 시간대 처리 라이브러리
    - dateutil: 유연한 날짜 파싱 라이브러리
    - json: JSON 데이터 처리

사용 예시:
    from assistant.tools.utilities import get_current_time, calculate_date

    # 현재 시간
    time = get_current_time.invoke({"timezone": "Asia/Seoul"})

    # 날짜 계산
    date = calculate_date.invoke({
        "operation": "add",
        "days": 30
    })

============================================================================
"""
from langchain_core.tools import tool
from datetime import datetime, timedelta
import json


@tool
def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """
    ========================================================================
    현재 시간을 반환합니다.
    ========================================================================

    지정된 시간대의 현재 날짜와 시간을 반환합니다.
    pytz 라이브러리를 사용하여 시간대를 처리합니다.

    Args:
        timezone: 시간대 문자열 (기본값: "Asia/Seoul")
            예시:
            - "Asia/Seoul" (한국)
            - "Asia/Tokyo" (일본)
            - "America/New_York" (미국 동부)
            - "Europe/London" (영국)
            - "UTC" (협정 세계시)

    Returns:
        포맷팅된 현재 날짜와 시간 문자열
        형식: "YYYY년 MM월 DD일 요일 HH:MM:SS (시간대)"

    사용 예시:
        # 한국 시간
        get_current_time.invoke({})  # 기본값 사용

        # 뉴욕 시간
        get_current_time.invoke({"timezone": "America/New_York"})

        # UTC
        get_current_time.invoke({"timezone": "UTC"})
    ========================================================================
    """
    try:
        # ====================================================================
        # pytz를 사용한 시간대 처리
        # ====================================================================
        # pytz 상세 설명:
        #
        # pytz는 Python에서 시간대를 처리하는 표준 라이브러리입니다.
        # IANA 시간대 데이터베이스를 기반으로 합니다.
        #
        # 주요 함수:
        #   - pytz.timezone(name): 시간대 객체 생성
        #   - datetime.now(tz): 특정 시간대의 현재 시간
        #   - dt.astimezone(tz): 시간대 변환
        #
        # 시간대 이름 형식:
        #   "대륙/도시" (예: "Asia/Seoul", "America/New_York")
        #   또는 "UTC", "GMT" 등
        #
        # 주의사항:
        #   - 설치 필요: pip install pytz
        #   - Python 3.9+ 에서는 zoneinfo 모듈 사용 가능
        # ====================================================================
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)

        # strftime() 포맷 문자열:
        # %Y: 4자리 연도
        # %m: 2자리 월
        # %d: 2자리 일
        # %A: 요일 전체 이름 (영어)
        # %H: 24시간 형식 시
        # %M: 분
        # %S: 초
        # %Z: 시간대 약어
        return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S (%Z)")

    except ImportError:
        # pytz가 설치되지 않은 경우 로컬 시간 반환
        now = datetime.now()
        return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S")
    except Exception as e:
        return f"시간 조회 오류: {str(e)}"


@tool
def calculate_date(
    operation: str,
    days: int = 0,
    weeks: int = 0,
    months: int = 0,
    base_date: str = None
) -> str:
    """
    ========================================================================
    날짜를 계산합니다.
    ========================================================================

    기준 날짜에서 일/주/월을 더하거나 빼서 새 날짜를 계산합니다.

    Args:
        operation: 연산 유형
            - 'add': 날짜 더하기
            - 'subtract': 날짜 빼기
        days: 더하거나 뺄 일 수 (기본값: 0)
        weeks: 더하거나 뺄 주 수 (기본값: 0)
        months: 더하거나 뺄 월 수 (기본값: 0)
            실제로는 30일 × months로 계산
        base_date: 기준 날짜 (없으면 오늘)
            다양한 형식 지원: "2024-01-15", "Jan 15, 2024" 등

    Returns:
        계산된 날짜 문자열

    사용 예시:
        # 오늘로부터 30일 후
        calculate_date.invoke({
            "operation": "add",
            "days": 30
        })

        # 2주 전
        calculate_date.invoke({
            "operation": "subtract",
            "weeks": 2
        })

        # 특정 날짜에서 3개월 후
        calculate_date.invoke({
            "operation": "add",
            "months": 3,
            "base_date": "2024-06-01"
        })
    ========================================================================
    """
    try:
        # ====================================================================
        # 기준 날짜 파싱
        # ====================================================================
        # dateutil.parser 상세 설명:
        #
        # dateutil은 유연한 날짜 파싱을 제공하는 라이브러리입니다.
        # 다양한 형식의 날짜 문자열을 자동으로 인식합니다.
        #
        # 지원 형식 예시:
        #   - "2024-01-15" (ISO 형식)
        #   - "Jan 15, 2024"
        #   - "15/01/2024"
        #   - "January 15th, 2024"
        #   - "2024.01.15"
        #
        # 사용:
        #   from dateutil import parser
        #   dt = parser.parse("Jan 15, 2024")  # datetime 객체 반환
        #
        # 설치:
        #   pip install python-dateutil
        # ====================================================================
        if base_date:
            from dateutil import parser as date_parser
            base = date_parser.parse(base_date)
        else:
            base = datetime.now()

        # ====================================================================
        # 날짜 계산
        # ====================================================================
        # timedelta 상세 설명:
        #
        # datetime.timedelta는 시간 간격을 나타내는 클래스입니다.
        # 날짜/시간 연산에 사용됩니다.
        #
        # 생성:
        #   timedelta(days=1)  # 1일
        #   timedelta(weeks=1)  # 1주 = 7일
        #   timedelta(hours=24)  # 24시간
        #
        # 연산:
        #   new_date = date + timedelta(days=10)  # 10일 후
        #   new_date = date - timedelta(weeks=2)  # 2주 전
        #
        # 주의:
        #   - timedelta는 months 매개변수가 없음
        #   - 월 계산은 30일로 근사
        #   - 정확한 월 계산이 필요하면 dateutil.relativedelta 사용
        # ====================================================================
        total_days = days + (weeks * 7) + (months * 30)

        if operation == "add":
            result = base + timedelta(days=total_days)
        elif operation == "subtract":
            result = base - timedelta(days=total_days)
        else:
            return "operation은 'add' 또는 'subtract'여야 합니다."

        return f"결과: {result.strftime('%Y년 %m월 %d일 %A')}"

    except Exception as e:
        return f"날짜 계산 오류: {str(e)}"


@tool
def days_between(date1: str, date2: str) -> str:
    """
    ========================================================================
    두 날짜 사이의 일수를 계산합니다.
    ========================================================================

    두 날짜 간의 차이를 일 단위로 계산합니다.
    순서와 관계없이 항상 양수 값을 반환합니다.

    Args:
        date1: 첫 번째 날짜
            다양한 형식 지원
        date2: 두 번째 날짜

    Returns:
        날짜 차이 문자열

    사용 예시:
        # 두 날짜 간 차이
        days_between.invoke({
            "date1": "2024-01-01",
            "date2": "2024-12-31"
        })  # "2024-01-01와 2024-12-31 사이: 365일"

        # 다양한 형식 지원
        days_between.invoke({
            "date1": "Jan 1, 2024",
            "date2": "Dec 31, 2024"
        })
    ========================================================================
    """
    try:
        from dateutil import parser as date_parser

        d1 = date_parser.parse(date1)
        d2 = date_parser.parse(date2)

        # abs()로 항상 양수 반환
        diff = abs((d2 - d1).days)

        return f"{date1}와 {date2} 사이: {diff}일"

    except Exception as e:
        return f"날짜 파싱 오류: {str(e)}"


@tool
def format_json(json_string: str, indent: int = 2) -> str:
    """
    ========================================================================
    JSON 문자열을 보기 좋게 포맷팅합니다.
    ========================================================================

    압축된 JSON을 들여쓰기가 있는 읽기 쉬운 형태로 변환합니다.

    Args:
        json_string: JSON 문자열
            예: '{"name":"John","age":30}'
        indent: 들여쓰기 크기 (기본값: 2)
            각 레벨당 공백 수

    Returns:
        포맷팅된 JSON 문자열

    사용 예시:
        format_json.invoke({
            "json_string": '{"name":"홍길동","age":30,"city":"서울"}'
        })

        # 결과:
        # {
        #   "name": "홍길동",
        #   "age": 30,
        #   "city": "서울"
        # }
    ========================================================================
    """
    try:
        # ====================================================================
        # JSON 파싱 및 포맷팅
        # ====================================================================
        # json 모듈 상세 설명:
        #
        # Python 표준 라이브러리의 JSON 처리 모듈입니다.
        #
        # 주요 함수:
        #   - json.loads(s): JSON 문자열 → Python 객체
        #   - json.dumps(obj): Python 객체 → JSON 문자열
        #   - json.load(f): 파일에서 JSON 읽기
        #   - json.dump(obj, f): JSON을 파일에 쓰기
        #
        # json.dumps() 주요 매개변수:
        #   - indent (int): 들여쓰기 크기 (None이면 압축)
        #   - ensure_ascii (bool): False면 유니코드 그대로 출력
        #   - sort_keys (bool): 키 정렬 여부
        #
        # 예시:
        #   data = {"name": "홍길동"}
        #   json.dumps(data, indent=2, ensure_ascii=False)
        #   # 결과:
        #   # {
        #   #   "name": "홍길동"
        #   # }
        # ====================================================================
        data = json.loads(json_string)
        # ensure_ascii=False: 한글 등 유니코드를 이스케이프하지 않음
        return json.dumps(data, indent=indent, ensure_ascii=False)

    except json.JSONDecodeError as e:
        return f"JSON 파싱 오류: {str(e)}"
    except Exception as e:
        return f"JSON 포맷팅 오류: {str(e)}"


@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    ========================================================================
    텍스트를 요약합니다.
    ========================================================================

    간단한 추출 요약을 수행합니다.
    첫 3문장을 추출하여 요약으로 제공합니다.

    Args:
        text: 요약할 텍스트
        max_length: 최대 요약 길이 (기본값: 100)
            초과 시 잘리고 "..."이 추가됨

    Returns:
        요약된 텍스트

    사용 예시:
        summarize_text.invoke({
            "text": "인공지능은 컴퓨터 과학의 한 분야입니다. 기계가 인간처럼 학습하고 추론하는 것을 목표로 합니다. 최근 딥러닝의 발전으로 큰 주목을 받고 있습니다. 다양한 산업에서 활용되고 있습니다."
        })

    참고:
        이 도구는 단순 추출 요약만 제공합니다.
        더 정교한 요약이 필요하면 LLM에게 직접 요청하세요.
    ========================================================================
    """
    # ========================================================================
    # 추출 요약 (Extractive Summarization)
    # ========================================================================
    # 추출 요약 vs 생성 요약:
    #
    # 추출 요약 (Extractive):
    #   - 원문에서 중요한 문장을 그대로 추출
    #   - 빠르고 단순
    #   - 새로운 표현 생성 안함
    #
    # 생성 요약 (Abstractive):
    #   - 새로운 문장으로 요약 생성
    #   - LLM 필요
    #   - 더 자연스러운 요약 가능
    #
    # 이 도구는 추출 요약을 사용합니다.
    # 첫 3문장을 추출하는 단순한 방식입니다.
    # ========================================================================

    # 문장 분리 (마침표 기준)
    sentences = text.split('.')

    # 3문장 이하면 그대로 반환
    if len(sentences) <= 3:
        return text

    # 첫 3문장 추출
    summary = '. '.join(sentences[:3]) + '.'

    # 최대 길이 초과 시 자르기
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return f"요약:\n{summary}"


# ============================================================================
# 유틸리티 도구 목록
# ============================================================================
# 이 리스트는 tools/__init__.py에서 ALL_TOOLS에 포함됩니다.
# 새 도구를 추가하면 이 리스트에도 추가해야 합니다.
# ============================================================================
utility_tools = [
    get_current_time,
    calculate_date,
    days_between,
    format_json,
    summarize_text
]
