# LangSmith 완벽 가이드

## 목차
1. [LangSmith란?](#langsmith란)
2. [주요 기능](#주요-기능)
3. [설치 및 설정](#설치-및-설정)
4. [트레이싱](#트레이싱)
5. [평가(Evaluation)](#평가evaluation)
6. [프롬프트 관리](#프롬프트-관리)
7. [데이터셋 관리](#데이터셋-관리)
8. [모니터링 및 알림](#모니터링-및-알림)
9. [베스트 프랙티스](#베스트-프랙티스)

---

## LangSmith란?

LangSmith는 **LLM 애플리케이션의 개발, 테스트, 배포, 모니터링**을 위한 통합 플랫폼입니다. LangChain 팀에서 개발했으며, LangChain 및 LangGraph 애플리케이션과 원활하게 통합됩니다.

### LangSmith가 해결하는 문제

1. **디버깅의 어려움**: LLM 체인의 중간 단계를 확인하기 어려움
2. **성능 측정**: 응답 품질을 객관적으로 평가하기 어려움
3. **프롬프트 관리**: 버전 관리 및 A/B 테스트의 필요성
4. **운영 모니터링**: 프로덕션 환경에서의 성능 추적

### 핵심 기능 요약

| 기능 | 설명 |
|------|------|
| **트레이싱** | 모든 LLM 호출과 체인 실행을 추적 |
| **평가** | 자동화된 품질 평가 및 회귀 테스트 |
| **프롬프트 허브** | 프롬프트 버전 관리 및 공유 |
| **데이터셋** | 테스트 데이터 관리 |
| **모니터링** | 실시간 성능 및 비용 모니터링 |

---

## 주요 기능

### 1. 트레이싱 (Tracing)
모든 LLM 호출, 체인 실행, 에이전트 행동을 시각화합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      트레이스 시각화 예시                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔵 Chain: rag_chain                               [2.3s]       │
│  ├── 📝 Prompt: question_prompt                    [0.1s]       │
│  │   └── input: "LangChain이란?"                               │
│  │   └── output: "다음 질문에 답변해주세요..."                    │
│  │                                                              │
│  ├── 🔍 Retriever: vectorstore                     [0.5s]       │
│  │   └── query: "LangChain이란?"                               │
│  │   └── documents: [doc1, doc2, doc3]                         │
│  │                                                              │
│  ├── 🤖 LLM: gpt-4                                 [1.5s]       │
│  │   └── tokens: 1,234 (in: 500, out: 734)                     │
│  │   └── cost: $0.045                                          │
│  │                                                              │
│  └── 📤 Output Parser                              [0.2s]       │
│      └── result: "LangChain은 LLM 기반..."                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 평가 (Evaluation)
LLM 출력의 품질을 자동으로 평가합니다.

- **정확성(Correctness)**: 답변이 정답과 일치하는지
- **관련성(Relevance)**: 질문과 관련된 답변인지
- **근거성(Groundedness)**: 제공된 컨텍스트에 기반한 답변인지
- **유해성(Harmfulness)**: 유해한 내용이 포함되어 있는지

### 3. 프롬프트 관리
프롬프트를 중앙에서 관리하고 버전을 추적합니다.

```python
from langsmith import Client

client = Client()

# 프롬프트 저장
client.push_prompt(
    "my-org/rag-prompt",
    object={
        "template": "컨텍스트: {context}\n\n질문: {question}\n\n답변:",
        "input_variables": ["context", "question"]
    }
)

# 프롬프트 불러오기
prompt = client.pull_prompt("my-org/rag-prompt")
```

---

## 설치 및 설정

### 1. 패키지 설치

```bash
pip install langsmith
```

### 2. API 키 발급

1. [LangSmith](https://smith.langchain.com/) 웹사이트 접속
2. 계정 생성 및 로그인
3. Settings > API Keys에서 새 키 생성

### 3. 환경 변수 설정

```bash
# .env 파일
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-awesome-project
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 4. Python에서 설정

```python
import os
from dotenv import load_dotenv

load_dotenv()

# 환경 변수 확인
assert os.getenv("LANGCHAIN_API_KEY"), "LANGCHAIN_API_KEY가 설정되지 않았습니다."

# 트레이싱 활성화
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-project"
```

---

## 트레이싱

### 자동 트레이싱

LangChain/LangGraph를 사용하면 자동으로 트레이싱됩니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 환경 변수가 설정되어 있으면 자동으로 추적됨
chat = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("다음을 요약해주세요: {text}")

chain = prompt | chat

# 이 호출은 자동으로 LangSmith에 기록됨
result = chain.invoke({"text": "긴 텍스트..."})
```

### 수동 트레이싱

`@traceable` 데코레이터를 사용하여 커스텀 함수를 추적합니다.

```python
from langsmith import traceable

@traceable(name="custom_function")
def my_custom_function(input_text: str) -> str:
    """커스텀 처리 로직"""
    processed = input_text.upper()
    return processed

@traceable(name="complex_pipeline")
def complex_pipeline(query: str) -> dict:
    """복잡한 파이프라인"""

    # 단계 1: 전처리
    processed = my_custom_function(query)

    # 단계 2: LLM 호출
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model="gpt-4")
    response = chat.invoke(processed)

    return {
        "original": query,
        "processed": processed,
        "response": response.content
    }

# 실행 - 모든 단계가 추적됨
result = complex_pipeline("hello world")
```

### 런 이름 및 메타데이터 추가

```python
from langsmith import traceable
from langchain_core.runnables import RunnableConfig

@traceable(
    name="qa_chain",
    metadata={"version": "1.0", "model": "gpt-4"}
)
def qa_with_metadata(question: str) -> str:
    # 로직
    pass

# 또는 런타임에 메타데이터 추가
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4")
result = chat.invoke(
    "질문",
    config=RunnableConfig(
        run_name="custom_run_name",
        metadata={"user_id": "user-123", "session": "abc"}
    )
)
```

### 트레이스에서 피드백 수집

```python
from langsmith import Client

client = Client()

# 특정 런에 피드백 추가
client.create_feedback(
    run_id="run-uuid-here",
    key="user_rating",
    score=0.9,
    comment="매우 유용한 답변이었습니다"
)
```

---

## 평가(Evaluation)

### 데이터셋 생성

```python
from langsmith import Client

client = Client()

# 데이터셋 생성
dataset = client.create_dataset(
    dataset_name="qa-evaluation-set",
    description="Q&A 시스템 평가를 위한 데이터셋"
)

# 예제 추가
examples = [
    {
        "inputs": {"question": "파이썬이란 무엇인가요?"},
        "outputs": {"answer": "파이썬은 고급 프로그래밍 언어입니다."}
    },
    {
        "inputs": {"question": "머신러닝의 정의는?"},
        "outputs": {"answer": "머신러닝은 데이터로부터 학습하는 AI의 한 분야입니다."}
    }
]

for example in examples:
    client.create_example(
        inputs=example["inputs"],
        outputs=example["outputs"],
        dataset_id=dataset.id
    )
```

### 평가 실행

```python
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

client = Client()

# 평가할 체인 정의
prompt = ChatPromptTemplate.from_template(
    "다음 질문에 간결하게 답변해주세요: {question}"
)
model = ChatOpenAI(model="gpt-4")
chain = prompt | model

# 타겟 함수 정의
def predict(inputs: dict) -> dict:
    response = chain.invoke(inputs)
    return {"answer": response.content}

# 평가자 정의
def correctness_evaluator(run, example) -> dict:
    """정확성 평가"""
    predicted = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")

    # 간단한 유사도 체크 (실제로는 더 정교한 로직 사용)
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, predicted.lower(), expected.lower()).ratio()

    return {
        "key": "correctness",
        "score": similarity,
        "comment": f"유사도: {similarity:.2f}"
    }

def relevance_evaluator(run, example) -> dict:
    """관련성 평가 (LLM 기반)"""
    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(model="gpt-4", temperature=0)

    question = example.inputs.get("question", "")
    answer = run.outputs.get("answer", "")

    evaluation_prompt = f"""
    질문: {question}
    답변: {answer}

    이 답변이 질문과 관련이 있는지 0~1 사이의 점수로 평가하세요.
    숫자만 응답하세요.
    """

    response = judge.invoke(evaluation_prompt)
    try:
        score = float(response.content.strip())
    except:
        score = 0.5

    return {
        "key": "relevance",
        "score": score
    }

# 평가 실행
results = evaluate(
    predict,
    data="qa-evaluation-set",
    evaluators=[correctness_evaluator, relevance_evaluator],
    experiment_prefix="gpt4-qa-test"
)

print(f"평가 완료: {results}")
```

### 내장 평가자 사용

```python
from langsmith.evaluation import LangChainStringEvaluator, evaluate

# 내장 평가자 사용
evaluators = [
    # 정확성 평가
    LangChainStringEvaluator("qa"),

    # 관련성 평가
    LangChainStringEvaluator(
        "criteria",
        config={"criteria": "relevance"}
    ),

    # 유해성 평가
    LangChainStringEvaluator(
        "criteria",
        config={"criteria": "harmfulness"}
    )
]

results = evaluate(
    predict_function,
    data="my-dataset",
    evaluators=evaluators
)
```

---

## 프롬프트 관리

### 프롬프트 허브 사용

```python
from langsmith import Client
from langchain import hub

client = Client()

# 프롬프트 푸시 (저장)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다."),
    ("human", "{question}")
])

client.push_prompt(
    "my-org/expert-qa",
    object=prompt,
    description="전문가 Q&A 프롬프트"
)

# 프롬프트 풀 (불러오기)
loaded_prompt = hub.pull("my-org/expert-qa")

# 특정 버전 불러오기
versioned_prompt = hub.pull("my-org/expert-qa:v2")
```

### 프롬프트 버전 관리

```python
from langsmith import Client

client = Client()

# 모든 버전 조회
prompts = client.list_prompts(prompt_name="my-org/expert-qa")

# 버전 간 비교
v1 = client.pull_prompt("my-org/expert-qa:v1")
v2 = client.pull_prompt("my-org/expert-qa:v2")

# 새 버전으로 업데이트
updated_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 분야의 최고 전문가입니다. 정확하고 친절하게 답변해주세요."),
    ("human", "{question}")
])

client.push_prompt(
    "my-org/expert-qa",
    object=updated_prompt,
    description="개선된 전문가 Q&A 프롬프트"
)
```

---

## 데이터셋 관리

### 데이터셋 CRUD

```python
from langsmith import Client

client = Client()

# 생성
dataset = client.create_dataset(
    dataset_name="customer-support-qa",
    description="고객 지원 Q&A 데이터셋"
)

# 조회
datasets = list(client.list_datasets())
specific_dataset = client.read_dataset(dataset_name="customer-support-qa")

# 예제 추가
client.create_example(
    inputs={"query": "배송은 얼마나 걸리나요?"},
    outputs={"response": "일반적으로 2-3일 소요됩니다."},
    dataset_id=dataset.id
)

# 대량 업로드
examples = [
    {"inputs": {"query": "반품 방법은?"}, "outputs": {"response": "마이페이지에서 신청하세요."}},
    {"inputs": {"query": "결제 수단은?"}, "outputs": {"response": "카드, 계좌이체를 지원합니다."}}
]

client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id
)

# 삭제
client.delete_dataset(dataset_id=dataset.id)
```

### CSV에서 데이터셋 가져오기

```python
import pandas as pd
from langsmith import Client

client = Client()

# CSV 로드
df = pd.read_csv("qa_data.csv")

# 데이터셋 생성
dataset = client.create_dataset("imported-qa-dataset")

# 데이터 추가
for _, row in df.iterrows():
    client.create_example(
        inputs={"question": row["question"]},
        outputs={"answer": row["answer"]},
        dataset_id=dataset.id
    )
```

---

## 모니터링 및 알림

### 대시보드 활용

LangSmith 대시보드에서 확인할 수 있는 메트릭:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangSmith 대시보드                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 프로젝트 개요: my-awesome-project                           │
│                                                                 │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐      │
│  │ 총 런 수       │ │ 평균 지연시간    │ │ 총 비용        │      │
│  │   15,234      │ │    1.2초       │ │   $45.67      │      │
│  │ ▲ 12% (7일)   │ │ ▼ 8% (7일)    │ │ ▲ 5% (7일)    │      │
│  └────────────────┘ └────────────────┘ └────────────────┘      │
│                                                                 │
│  📈 시간별 런 수                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │     ╭──╮                                            │       │
│  │    ╭╯  ╰╮    ╭──╮                                   │       │
│  │ ───╯    ╰────╯  ╰──────────────────────────────     │       │
│  │ 00   04   08   12   16   20   24 (시간)             │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  🔴 에러율: 0.5%                                                │
│  ⚠️ 경고: 3건 (지연시간 > 5초)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 프로그래밍 방식 모니터링

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# 최근 런 조회
runs = client.list_runs(
    project_name="my-project",
    start_time=datetime.now() - timedelta(hours=24),
    is_root=True,  # 최상위 런만
    error=False    # 성공한 런만
)

# 통계 계산
latencies = []
costs = []

for run in runs:
    if run.end_time and run.start_time:
        latency = (run.end_time - run.start_time).total_seconds()
        latencies.append(latency)

    if run.total_cost:
        costs.append(run.total_cost)

avg_latency = sum(latencies) / len(latencies) if latencies else 0
total_cost = sum(costs)

print(f"평균 지연시간: {avg_latency:.2f}초")
print(f"총 비용: ${total_cost:.4f}")
```

### 에러 모니터링

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# 에러가 발생한 런 조회
error_runs = client.list_runs(
    project_name="my-project",
    start_time=datetime.now() - timedelta(hours=1),
    error=True
)

for run in error_runs:
    print(f"런 ID: {run.id}")
    print(f"에러: {run.error}")
    print(f"시간: {run.start_time}")
    print("---")
```

### 알림 설정 (웹훅 활용)

```python
import requests
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

def check_and_alert():
    """에러율 체크 및 알림"""

    # 최근 1시간 런 조회
    all_runs = list(client.list_runs(
        project_name="my-project",
        start_time=datetime.now() - timedelta(hours=1),
        is_root=True
    ))

    error_runs = [r for r in all_runs if r.error]

    error_rate = len(error_runs) / len(all_runs) if all_runs else 0

    # 에러율이 5% 초과시 알림
    if error_rate > 0.05:
        webhook_url = "https://hooks.slack.com/services/xxx"
        requests.post(webhook_url, json={
            "text": f"⚠️ LangSmith 알림: 에러율 {error_rate:.1%} (최근 1시간)"
        })

# 주기적 실행 (예: cron 또는 스케줄러 사용)
```

---

## 베스트 프랙티스

### 1. 프로젝트 구조화

```python
import os

# 환경별 프로젝트 분리
env = os.getenv("ENVIRONMENT", "development")
os.environ["LANGCHAIN_PROJECT"] = f"my-app-{env}"

# 예: my-app-development, my-app-staging, my-app-production
```

### 2. 의미있는 런 이름 사용

```python
from langsmith import traceable

@traceable(name="process_customer_query")
def process_query(query: str, customer_id: str) -> str:
    """고객 쿼리 처리"""
    # 로직
    pass

# 런타임에 동적 이름
from langchain_core.runnables import RunnableConfig

result = chain.invoke(
    {"query": query},
    config=RunnableConfig(
        run_name=f"query-{customer_id}-{query[:20]}"
    )
)
```

### 3. 메타데이터 활용

```python
from langsmith import traceable

@traceable(
    metadata={
        "version": "2.1",
        "model": "gpt-4",
        "feature": "customer-support"
    },
    tags=["production", "customer-support"]
)
def customer_support_chain(query: str) -> str:
    pass
```

### 4. 피드백 루프 구축

```python
from langsmith import Client
import uuid

client = Client()

def process_with_feedback(query: str, session_id: str) -> dict:
    """피드백 수집이 가능한 처리"""

    # 런 ID 생성
    run_id = str(uuid.uuid4())

    # 처리 실행
    result = chain.invoke(
        {"query": query},
        config={"run_id": run_id}
    )

    return {
        "answer": result,
        "run_id": run_id  # 클라이언트에 반환하여 피드백 수집
    }

def submit_feedback(run_id: str, rating: float, comment: str = None):
    """사용자 피드백 제출"""
    client.create_feedback(
        run_id=run_id,
        key="user_rating",
        score=rating,
        comment=comment
    )
```

### 5. 평가 자동화

```python
# CI/CD 파이프라인에서 실행
from langsmith import evaluate

def run_evaluation():
    """자동화된 평가 실행"""

    results = evaluate(
        predict_function,
        data="regression-test-dataset",
        evaluators=[accuracy_evaluator, latency_evaluator],
        experiment_prefix=f"ci-{os.getenv('BUILD_NUMBER', 'local')}"
    )

    # 품질 게이트
    avg_score = sum(r.score for r in results) / len(results)
    if avg_score < 0.8:
        raise Exception(f"품질 기준 미달: {avg_score:.2f} < 0.8")

    return results
```

---

## 통합 예시: 전체 워크플로우

```python
import os
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 환경 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-qa-system"

client = Client()

# 프롬프트 허브에서 불러오기
from langchain import hub
prompt = hub.pull("my-org/qa-prompt")

# 모델 설정
model = ChatOpenAI(model="gpt-4", temperature=0)

# 체인 구성
chain = prompt | model

@traceable(
    name="qa_pipeline",
    metadata={"version": "1.0"}
)
def qa_pipeline(question: str, user_id: str) -> dict:
    """전체 Q&A 파이프라인"""

    result = chain.invoke(
        {"question": question},
        config={
            "metadata": {"user_id": user_id}
        }
    )

    return {
        "answer": result.content,
        "model": "gpt-4"
    }

# 사용
response = qa_pipeline(
    question="LangSmith의 주요 기능은?",
    user_id="user-123"
)

print(response["answer"])
# LangSmith 대시보드에서 이 런을 확인할 수 있음
```

---

## 다음 단계

- [튜토리얼: RAG 챗봇 구축](../tutorials/01-rag-chatbot.md)
- [튜토리얼: 멀티 에이전트 시스템](../tutorials/02-multi-agent-system.md)
- [튜토리얼: 대화형 AI 어시스턴트](../tutorials/03-conversational-assistant.md)
