# assistant/tools/utilities.py
from langchain_core.tools import tool
from datetime import datetime, timedelta
import json


@tool
def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """
    현재 시간을 반환합니다.

    Args:
        timezone: 시간대 (기본값: Asia/Seoul)

    Returns:
        현재 날짜와 시간
    """
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S (%Z)")
    except:
        now = datetime.now()
        return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S")


@tool
def calculate_date(
    operation: str,
    days: int = 0,
    weeks: int = 0,
    months: int = 0,
    base_date: str = None
) -> str:
    """
    날짜를 계산합니다.

    Args:
        operation: 'add' 또는 'subtract'
        days: 일 수
        weeks: 주 수
        months: 월 수
        base_date: 기준 날짜 (없으면 오늘)

    Returns:
        계산된 날짜
    """
    try:
        if base_date:
            from dateutil import parser as date_parser
            base = date_parser.parse(base_date)
        else:
            base = datetime.now()

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
    두 날짜 사이의 일수를 계산합니다.

    Args:
        date1: 첫 번째 날짜
        date2: 두 번째 날짜

    Returns:
        날짜 차이
    """
    try:
        from dateutil import parser as date_parser
        d1 = date_parser.parse(date1)
        d2 = date_parser.parse(date2)
        diff = abs((d2 - d1).days)
        return f"{date1}와 {date2} 사이: {diff}일"
    except Exception as e:
        return f"날짜 파싱 오류: {str(e)}"


@tool
def format_json(json_string: str, indent: int = 2) -> str:
    """
    JSON 문자열을 보기 좋게 포맷팅합니다.

    Args:
        json_string: JSON 문자열
        indent: 들여쓰기 크기

    Returns:
        포맷팅된 JSON
    """
    try:
        data = json.loads(json_string)
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except Exception as e:
        return f"JSON 파싱 오류: {str(e)}"


@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    텍스트를 요약합니다.

    Args:
        text: 요약할 텍스트
        max_length: 최대 요약 길이

    Returns:
        요약된 텍스트
    """
    # 간단한 추출 요약
    sentences = text.split('.')
    if len(sentences) <= 3:
        return text

    # 첫 3문장 추출
    summary = '. '.join(sentences[:3]) + '.'
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return f"요약:\n{summary}"


# 모든 유틸리티 도구
utility_tools = [
    get_current_time,
    calculate_date,
    days_between,
    format_json,
    summarize_text
]
