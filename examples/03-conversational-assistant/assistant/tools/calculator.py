# assistant/tools/calculator.py
from langchain_core.tools import tool
import math
from typing import Union


@tool
def calculate(expression: str) -> str:
    """
    수학 표현식을 계산합니다.

    Args:
        expression: 계산할 수학 표현식 (예: "2 + 2", "sqrt(16)", "sin(3.14)")

    Returns:
        계산 결과
    """
    try:
        # 안전한 수학 함수 매핑
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
        }

        # eval with restricted globals
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"계산 결과: {expression} = {result}"

    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """
    단위를 변환합니다.

    Args:
        value: 변환할 값
        from_unit: 원래 단위 (예: km, mile, kg, lb, celsius, fahrenheit)
        to_unit: 변환할 단위

    Returns:
        변환된 값
    """
    conversions = {
        # 길이
        ("km", "mile"): lambda x: x * 0.621371,
        ("mile", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("cm", "inch"): lambda x: x * 0.393701,
        ("inch", "cm"): lambda x: x * 2.54,

        # 무게
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("g", "oz"): lambda x: x * 0.035274,
        ("oz", "g"): lambda x: x * 28.3495,

        # 온도
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("celsius", "kelvin"): lambda x: x + 273.15,
        ("kelvin", "celsius"): lambda x: x - 273.15,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"지원하지 않는 단위 변환입니다: {from_unit} → {to_unit}"


@tool
def percentage_calculate(operation: str, value1: float, value2: float = None) -> str:
    """
    퍼센트 관련 계산을 수행합니다.

    Args:
        operation: 계산 유형 ('of', 'change', 'increase', 'decrease')
        value1: 첫 번째 값
        value2: 두 번째 값 (필요한 경우)

    Returns:
        계산 결과
    """
    try:
        if operation == "of" and value2:
            # value1%의 value2
            result = (value1 / 100) * value2
            return f"{value1}%의 {value2} = {result}"

        elif operation == "change" and value2:
            # value1에서 value2로의 변화율
            change = ((value2 - value1) / value1) * 100
            return f"{value1}에서 {value2}로의 변화율 = {change:.2f}%"

        elif operation == "increase" and value2:
            # value1을 value2% 증가
            result = value1 * (1 + value2/100)
            return f"{value1}을 {value2}% 증가 = {result}"

        elif operation == "decrease" and value2:
            # value1을 value2% 감소
            result = value1 * (1 - value2/100)
            return f"{value1}을 {value2}% 감소 = {result}"

        else:
            return "지원하지 않는 연산입니다. 'of', 'change', 'increase', 'decrease' 중 선택하세요."

    except Exception as e:
        return f"계산 오류: {str(e)}"
