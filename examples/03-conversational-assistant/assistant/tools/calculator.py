# assistant/tools/calculator.py
"""
============================================================================
계산 도구 모듈
============================================================================

이 모듈은 수학 계산, 단위 변환, 퍼센트 계산 도구를 제공합니다.
LLM이 수학적 연산이 필요할 때 사용할 수 있는 도구들입니다.

핵심 개념:
    - 안전한 eval 사용: 위험한 함수 제한
    - 단위 변환 매핑: 람다 함수를 활용한 변환
    - 퍼센트 연산: 다양한 비율 계산

보안 고려사항:
    calculate 도구는 eval()을 사용하므로 보안에 주의가 필요합니다.
    __builtins__를 빈 딕셔너리로 설정하여 내장 함수 접근을 차단합니다.

사용 예시:
    from assistant.tools.calculator import calculate, unit_convert

    # 수학 계산
    result = calculate.invoke({"expression": "sqrt(16) + 2 ** 3"})

    # 단위 변환
    result = unit_convert.invoke({
        "value": 100,
        "from_unit": "km",
        "to_unit": "mile"
    })

============================================================================
"""
from langchain_core.tools import tool
import math
from typing import Union


@tool
def calculate(expression: str) -> str:
    """
    ========================================================================
    수학 표현식을 계산합니다.
    ========================================================================

    문자열로 주어진 수학 표현식을 평가하고 결과를 반환합니다.
    기본 연산(+, -, *, /, **, %)과 수학 함수(sqrt, sin, cos 등)를 지원합니다.

    Args:
        expression: 계산할 수학 표현식
            예: "2 + 2", "sqrt(16)", "sin(3.14)", "2 ** 10"

    Returns:
        계산 결과 문자열

    지원하는 연산:
        - 기본 연산: +, -, *, /, //, %, **
        - 수학 함수: sqrt, sin, cos, tan, log, log10, exp
        - 상수: pi, e
        - 기타: abs, round, pow

    사용 예시:
        # 기본 연산
        calculate.invoke({"expression": "2 + 2 * 3"})  # 8

        # 제곱근
        calculate.invoke({"expression": "sqrt(16)"})  # 4.0

        # 삼각함수
        calculate.invoke({"expression": "sin(pi / 2)"})  # 1.0

        # 거듭제곱
        calculate.invoke({"expression": "2 ** 10"})  # 1024

    보안:
        이 도구는 제한된 환경에서 eval()을 사용합니다.
        허용된 함수만 사용 가능하며, 시스템 접근은 차단됩니다.
    ========================================================================
    """
    try:
        # ====================================================================
        # 안전한 eval 구현
        # ====================================================================
        # eval()은 문자열을 Python 코드로 실행하는 함수입니다.
        # 보안 위험이 있으므로 사용 가능한 함수를 제한합니다.
        #
        # eval(expression, globals, locals) 매개변수:
        #   - expression: 평가할 문자열
        #   - globals: 전역 네임스페이스 (dict)
        #   - locals: 지역 네임스페이스 (dict)
        #
        # 보안 전략:
        #   1. __builtins__를 빈 딕셔너리로 설정
        #      → 내장 함수(open, exec, import 등) 접근 차단
        #   2. 허용할 함수만 명시적으로 전달
        #      → 화이트리스트 방식
        #
        # 주의: 이 방법도 완벽하지 않습니다.
        # 프로덕션에서는 더 안전한 파싱 라이브러리 사용 권장
        # (예: numexpr, sympy)
        # ====================================================================
        safe_dict = {
            # math 모듈의 함수들
            "sqrt": math.sqrt,      # 제곱근
            "sin": math.sin,        # 사인
            "cos": math.cos,        # 코사인
            "tan": math.tan,        # 탄젠트
            "log": math.log,        # 자연로그
            "log10": math.log10,    # 상용로그
            "exp": math.exp,        # 지수 (e^x)

            # 상수
            "pi": math.pi,          # 원주율 (3.14159...)
            "e": math.e,            # 자연상수 (2.71828...)

            # 내장 함수 (안전한 것만)
            "abs": abs,             # 절대값
            "round": round,         # 반올림
            "pow": pow,             # 거듭제곱
        }

        # eval with restricted globals
        # __builtins__: {} 로 설정하여 내장 함수 차단
        result = eval(expression, {"__builtins__": {}}, safe_dict)

        return f"계산 결과: {expression} = {result}"

    except ZeroDivisionError:
        return "계산 오류: 0으로 나눌 수 없습니다."
    except ValueError as e:
        return f"계산 오류: 잘못된 값입니다. {str(e)}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """
    ========================================================================
    단위를 변환합니다.
    ========================================================================

    다양한 단위 간의 변환을 수행합니다.
    길이, 무게, 온도 단위를 지원합니다.

    Args:
        value: 변환할 값
        from_unit: 원래 단위
            길이: km, mile, m, ft, cm, inch
            무게: kg, lb, g, oz
            온도: celsius, fahrenheit, kelvin
        to_unit: 변환할 단위

    Returns:
        변환된 값 문자열

    지원하는 변환:
        길이:
            - km ↔ mile (킬로미터 ↔ 마일)
            - m ↔ ft (미터 ↔ 피트)
            - cm ↔ inch (센티미터 ↔ 인치)

        무게:
            - kg ↔ lb (킬로그램 ↔ 파운드)
            - g ↔ oz (그램 ↔ 온스)

        온도:
            - celsius ↔ fahrenheit (섭씨 ↔ 화씨)
            - celsius ↔ kelvin (섭씨 ↔ 켈빈)

    사용 예시:
        # 킬로미터를 마일로
        unit_convert.invoke({
            "value": 100,
            "from_unit": "km",
            "to_unit": "mile"
        })  # "100 km = 62.1371 mile"

        # 섭씨를 화씨로
        unit_convert.invoke({
            "value": 25,
            "from_unit": "celsius",
            "to_unit": "fahrenheit"
        })  # "25 celsius = 77.0000 fahrenheit"
    ========================================================================
    """
    # ========================================================================
    # 변환 매핑 딕셔너리
    # ========================================================================
    # 각 변환을 람다 함수로 정의합니다.
    # 키: (원래 단위, 변환 단위) 튜플
    # 값: 변환 람다 함수
    #
    # 람다 함수 설명:
    #   lambda x: 표현식
    #   - x: 입력값
    #   - 표현식: 변환 공식
    #
    # 예: lambda x: x * 0.621371
    #     킬로미터 → 마일 변환 (1 km = 0.621371 mile)
    # ========================================================================
    conversions = {
        # 길이 변환
        ("km", "mile"): lambda x: x * 0.621371,      # km → mile
        ("mile", "km"): lambda x: x * 1.60934,       # mile → km
        ("m", "ft"): lambda x: x * 3.28084,          # m → ft
        ("ft", "m"): lambda x: x * 0.3048,           # ft → m
        ("cm", "inch"): lambda x: x * 0.393701,      # cm → inch
        ("inch", "cm"): lambda x: x * 2.54,          # inch → cm

        # 무게 변환
        ("kg", "lb"): lambda x: x * 2.20462,         # kg → lb
        ("lb", "kg"): lambda x: x * 0.453592,        # lb → kg
        ("g", "oz"): lambda x: x * 0.035274,         # g → oz
        ("oz", "g"): lambda x: x * 28.3495,          # oz → g

        # 온도 변환
        # 섭씨 → 화씨: F = C × 9/5 + 32
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        # 화씨 → 섭씨: C = (F - 32) × 5/9
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        # 섭씨 → 켈빈: K = C + 273.15
        ("celsius", "kelvin"): lambda x: x + 273.15,
        # 켈빈 → 섭씨: C = K - 273.15
        ("kelvin", "celsius"): lambda x: x - 273.15,
    }

    # 단위를 소문자로 변환하여 매칭
    key = (from_unit.lower(), to_unit.lower())

    if key in conversions:
        result = conversions[key](value)
        # 소수점 4자리까지 표시
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"지원하지 않는 단위 변환입니다: {from_unit} → {to_unit}"


@tool
def percentage_calculate(operation: str, value1: float, value2: float = None) -> str:
    """
    ========================================================================
    퍼센트 관련 계산을 수행합니다.
    ========================================================================

    다양한 퍼센트 관련 계산을 지원합니다.

    Args:
        operation: 계산 유형
            - 'of': value1%의 value2 계산 (예: 20%의 500)
            - 'change': value1에서 value2로의 변화율
            - 'increase': value1을 value2% 증가
            - 'decrease': value1을 value2% 감소
        value1: 첫 번째 값
        value2: 두 번째 값 (필요한 경우)

    Returns:
        계산 결과 문자열

    사용 예시:
        # 20%의 500은?
        percentage_calculate.invoke({
            "operation": "of",
            "value1": 20,
            "value2": 500
        })  # "20%의 500 = 100.0"

        # 100에서 150으로의 변화율
        percentage_calculate.invoke({
            "operation": "change",
            "value1": 100,
            "value2": 150
        })  # "100에서 150로의 변화율 = 50.00%"

        # 1000을 15% 증가
        percentage_calculate.invoke({
            "operation": "increase",
            "value1": 1000,
            "value2": 15
        })  # "1000을 15% 증가 = 1150.0"

        # 1000을 20% 감소
        percentage_calculate.invoke({
            "operation": "decrease",
            "value1": 1000,
            "value2": 20
        })  # "1000을 20% 감소 = 800.0"
    ========================================================================
    """
    try:
        if operation == "of" and value2:
            # value1%의 value2
            # 공식: (퍼센트 / 100) × 값
            result = (value1 / 100) * value2
            return f"{value1}%의 {value2} = {result}"

        elif operation == "change" and value2:
            # value1에서 value2로의 변화율
            # 공식: ((새 값 - 원래 값) / 원래 값) × 100
            if value1 == 0:
                return "오류: 원래 값이 0이면 변화율을 계산할 수 없습니다."
            change = ((value2 - value1) / value1) * 100
            return f"{value1}에서 {value2}로의 변화율 = {change:.2f}%"

        elif operation == "increase" and value2:
            # value1을 value2% 증가
            # 공식: 원래 값 × (1 + 퍼센트/100)
            result = value1 * (1 + value2/100)
            return f"{value1}을 {value2}% 증가 = {result}"

        elif operation == "decrease" and value2:
            # value1을 value2% 감소
            # 공식: 원래 값 × (1 - 퍼센트/100)
            result = value1 * (1 - value2/100)
            return f"{value1}을 {value2}% 감소 = {result}"

        else:
            return "지원하지 않는 연산입니다. 'of', 'change', 'increase', 'decrease' 중 선택하세요."

    except Exception as e:
        return f"계산 오류: {str(e)}"
