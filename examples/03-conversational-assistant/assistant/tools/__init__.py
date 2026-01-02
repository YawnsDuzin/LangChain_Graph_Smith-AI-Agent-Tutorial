# assistant/tools/__init__.py
from assistant.tools.search import web_search, news_search, wikipedia_search
from assistant.tools.calculator import calculate, unit_convert, percentage_calculate
from assistant.tools.code_executor import execute_python, explain_code, generate_code
from assistant.tools.utilities import (
    get_current_time,
    calculate_date,
    days_between,
    format_json,
    summarize_text
)

# 모든 도구 목록
ALL_TOOLS = [
    # 검색 도구
    web_search,
    news_search,
    wikipedia_search,

    # 계산 도구
    calculate,
    unit_convert,
    percentage_calculate,

    # 코드 도구
    execute_python,
    explain_code,
    generate_code,

    # 유틸리티 도구
    get_current_time,
    calculate_date,
    days_between,
    format_json,
    summarize_text,
]

# 카테고리별 도구
TOOL_CATEGORIES = {
    "검색": [web_search, news_search, wikipedia_search],
    "계산": [calculate, unit_convert, percentage_calculate],
    "코드": [execute_python, explain_code, generate_code],
    "유틸리티": [get_current_time, calculate_date, days_between, format_json],
}

# 승인이 필요한 도구 (Human-in-the-Loop)
TOOLS_REQUIRING_APPROVAL = [
    "execute_python",  # 코드 실행은 승인 필요
]
