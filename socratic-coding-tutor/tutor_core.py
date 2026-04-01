import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

Route = Literal["teach", "review", "quiz"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_CANDIDATES = (
    Path(__file__).resolve().parent / ".env",
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "ai-agent-playground" / ".env",
)

for env_path in ENV_CANDIDATES:
    if env_path.exists():
        load_dotenv(env_path, override=False)


class TutorState(MessagesState):
    route: Route | None
    research_notes: str
    active_agent: str | None


@dataclass
class TutorTurnResult:
    answer: str
    history: list[BaseMessage]
    route: Route
    active_agent: str
    research_notes: str


LEARNING_REFERENCE = [
    {
        "topic": "재귀",
        "aliases": ["recursion", "recursive", "base case", "팩토리얼", "dfs"],
        "concept": "재귀는 함수가 자기 자신을 다시 호출하며 큰 문제를 작은 문제로 쪼개는 방식입니다. 반드시 멈추는 조건(base case)이 함께 있어야 합니다.",
        "example": "factorial(n)은 n * factorial(n-1)로 줄일 수 있고, n == 1일 때 멈춥니다.",
        "common_mistakes": [
            "base case를 빼먹어 무한 호출이 발생한다",
            "문제가 실제로 더 작아지지 않아 종료되지 않는다",
        ],
        "quiz_tip": "재귀 문제는 '언제 멈추는가'와 '한 단계에서 무엇을 줄이는가'를 같이 묻는 문제가 자주 나옵니다.",
    },
    {
        "topic": "리스트",
        "aliases": ["list", "append", "slice", "index", "mutable"],
        "concept": "리스트는 순서가 있고 수정 가능한 컬렉션입니다. 같은 리스트 객체를 여러 변수가 참조할 수 있어 변경이 공유될 수 있습니다.",
        "example": "items = [1, 2]; items.append(3)은 기존 리스트를 직접 수정합니다.",
        "common_mistakes": [
            "items = items.append(3)처럼 append 반환값을 변수에 다시 담는다",
            "슬라이싱이 새 리스트를 만든다는 점을 놓친다",
        ],
        "quiz_tip": "append와 + 연산, 얕은 복사와 깊은 복사의 차이를 묻는 문제가 자주 나옵니다.",
    },
    {
        "topic": "딕셔너리",
        "aliases": ["dictionary", "dict", "key", "value", "hash map"],
        "concept": "딕셔너리는 키-값 쌍으로 데이터를 저장합니다. 키로 빠르게 값을 찾을 수 있고, 키는 보통 변경 불가능한 값이어야 합니다.",
        "example": "user = {'name': 'Ada', 'level': 3}; user['name']은 'Ada'를 반환합니다.",
        "common_mistakes": [
            "존재하지 않는 키에 바로 접근해 KeyError가 난다",
            "리스트 같은 변경 가능한 값을 키로 사용하려고 한다",
        ],
        "quiz_tip": "리스트와 딕셔너리의 접근 방식 차이, get 메서드 용도를 비교하는 문제가 자주 나옵니다.",
    },
    {
        "topic": "for 루프",
        "aliases": ["for loop", "iteration", "range", "enumerate"],
        "concept": "for 루프는 반복 가능한 객체를 순회하며 각 원소를 차례대로 처리합니다. 인덱스가 필요하면 enumerate를 같이 쓰는 편이 명확합니다.",
        "example": "for index, value in enumerate(items): print(index, value)",
        "common_mistakes": [
            "리스트를 순회하면서 동시에 구조를 크게 변경한다",
            "range(len(items))를 무조건 사용해 가독성을 떨어뜨린다",
        ],
        "quiz_tip": "enumerate와 range(len(...))의 차이, 반복 중 변경 위험성을 묻는 문제가 자주 나옵니다.",
    },
    {
        "topic": "함수",
        "aliases": ["function", "parameter", "argument", "return", "type hint"],
        "concept": "함수는 입력을 받아 결과를 반환하는 재사용 가능한 코드 블록입니다. 이름, 매개변수, 반환값이 분명할수록 읽기 쉬워집니다.",
        "example": "def add(a: int, b: int) -> int: return a + b",
        "common_mistakes": [
            "print와 return의 역할을 혼동한다",
            "함수 이름이 동작을 충분히 설명하지 못한다",
        ],
        "quiz_tip": "매개변수와 인자, return과 print 차이를 구분하는 문제가 자주 나옵니다.",
    },
    {
        "topic": "클래스와 객체",
        "aliases": ["class", "object", "instance", "__init__", "method"],
        "concept": "클래스는 설계도이고 객체는 그 설계도로 만든 실제 값입니다. 객체마다 속성 값은 다를 수 있지만 같은 메서드 구조를 공유합니다.",
        "example": "class Dog: ...; d = Dog('Coco')에서 Dog는 클래스, d는 객체입니다.",
        "common_mistakes": [
            "클래스 변수와 인스턴스 변수를 구분하지 못한다",
            "self가 가리키는 대상을 혼동한다",
        ],
        "quiz_tip": "클래스/객체 구분, self 역할, 생성자 호출 시점을 묻는 문제가 자주 나옵니다.",
    },
    {
        "topic": "디버깅",
        "aliases": ["debug", "traceback", "error", "bug", "exception"],
        "concept": "디버깅은 에러 메시지와 실행 흐름을 좁혀가며 원인을 찾는 과정입니다. 재현 조건, 입력값, 마지막으로 성공했던 지점을 확인하는 순서가 효과적입니다.",
        "example": "traceback의 마지막 줄부터 보고, 문제 함수에 print나 breakpoint를 추가해 실제 값을 확인합니다.",
        "common_mistakes": [
            "에러 메시지를 끝까지 읽지 않고 감으로 수정한다",
            "한 번에 여러 군데를 바꿔 원인 추적이 어려워진다",
        ],
        "quiz_tip": "traceback 읽는 순서와 최소 재현 단계를 묻는 문제가 자주 나옵니다.",
    },
]


AGENT_LABELS: dict[Route, str] = {
    "teach": "Tutor Agent",
    "review": "Review Agent",
    "quiz": "Quiz Agent",
}


def _create_supervisor_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _create_researcher_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)


def _create_specialist_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^0-9a-zA-Z가-힣_+#]+", _normalize(text)) if token]


@tool
def search_learning_reference(query: str) -> str:
    """로컬 학습 레퍼런스에서 개념, 예제, 흔한 실수, 퀴즈 포인트를 찾습니다."""
    normalized_query = _normalize(query)
    tokens = _tokenize(query)
    ranked: list[tuple[int, dict[str, object]]] = []

    for entry in LEARNING_REFERENCE:
        searchable_parts = [
            entry["topic"],
            *entry["aliases"],
            entry["concept"],
            entry["example"],
            *entry["common_mistakes"],
            entry["quiz_tip"],
        ]
        searchable_text = " ".join(str(part) for part in searchable_parts)
        searchable_normalized = _normalize(searchable_text)

        score = 0
        if _normalize(str(entry["topic"])) in normalized_query:
            score += 3
        for alias in entry["aliases"]:
            if _normalize(alias) in normalized_query:
                score += 2
        for token in tokens:
            if token in searchable_normalized:
                score += 1
        if score:
            ranked.append((score, entry))

    if not ranked:
        return (
            "관련 레퍼런스를 찾지 못했습니다.\n"
            "질문을 더 구체적으로 다시 검색하세요. 예: '재귀 base case', 'dict get', 'for loop enumerate'"
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    sections = []
    for _, entry in ranked[:3]:
        common_mistakes = "; ".join(str(item) for item in entry["common_mistakes"])
        sections.append(
            "\n".join(
                [
                    f"주제: {entry['topic']}",
                    f"개념: {entry['concept']}",
                    f"예제: {entry['example']}",
                    f"흔한 실수: {common_mistakes}",
                    f"퀴즈 포인트: {entry['quiz_tip']}",
                ]
            )
        )

    return "\n\n".join(sections)


TOOLS = [search_learning_reference]

SUPERVISOR_SYSTEM_PROMPT = """당신은 Education Agent의 Supervisor입니다.
사용자 메시지를 분석해 아래 세 가지 중 하나만 고르세요.

- teach: 개념 설명, 문법 차이, 학습 가이드 요청
- review: 코드 리뷰, 에러 원인 분석, 디버깅 요청
- quiz: 퀴즈 출제, 정답 확인, 시험 대비 요청

반드시 teach, review, quiz 중 하나만 답하세요."""

RESEARCHER_SYSTEM_PROMPT = """당신은 Education Agent의 Researcher입니다.
사용자의 최근 질문을 보고 필요한 학습 배경을 짧은 연구 노트로 정리하세요.

규칙:
- 먼저 search_learning_reference 도구 사용 여부를 검토하세요.
- 관련 정보가 있으면 아래 형식으로 4~6줄로 요약하세요.
  1. 핵심 개념
  2. 흔한 실수
  3. 튜터가 바로 활용할 질문 포인트
- 관련 정보가 부족하면 '관련 레퍼런스 부족'을 먼저 적고, 질문에서 보이는 핵심 키워드만 정리하세요.
- 한국어로 답하세요."""

TEACH_SYSTEM_PROMPT = """당신은 소크라테스식 프로그래밍 튜터입니다.

규칙:
- Researcher가 정리한 노트를 먼저 반영하세요.
- 정답을 일방적으로 주입하지 말고 학생이 스스로 생각할 질문을 던지세요.
- 한국어로 답하세요.

응답 형식:
1. 질문을 인정하는 짧은 한 줄
2. 핵심 개념 설명 또는 비유 2~3줄
3. 학생이 생각할 유도 질문 2~3개
4. 바로 해볼 작은 연습 1개"""

REVIEW_SYSTEM_PROMPT = """당신은 소크라테스식 코드 리뷰어입니다.

규칙:
- Researcher가 정리한 노트를 먼저 반영하세요.
- 직접 정답을 고쳐주기보다 학생이 문제를 찾게 만드는 질문 중심으로 답하세요.
- 한국어로 답하세요.

응답 형식:
1. 잘한 점 1~2개
2. 생각해볼 질문 2~3개
3. 디버깅 순서 또는 다음 수정 포인트 1개"""

QUIZ_SYSTEM_PROMPT = """당신은 프로그래밍 퀴즈 마스터입니다.

규칙:
- Researcher가 정리한 노트를 먼저 반영하세요.
- 학생이 답을 맞히면 왜 맞는지 짧게 설명하고, 틀리면 힌트만 주세요.
- 한국어로 답하세요.

응답 형식:
1. 퀴즈 제목 또는 채점 결과
2. 문제 2~3개 또는 답변 평가
3. 마지막에 다음 학습 포인트 1개"""


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content)


def _latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def route_question(state: TutorState) -> Route:
    messages = state["messages"]
    recent_messages = messages[-6:]
    transcript = "\n".join(
        f"{getattr(message, 'type', 'message')}: {_message_text(message)}"
        for message in recent_messages
    )
    supervisor_llm = _create_supervisor_llm()
    response = supervisor_llm.invoke(
        [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=transcript),
        ]
    )

    route = _message_text(response).strip().lower()
    if route in ("teach", "review", "quiz"):
        return route
    return "teach"


def run_researcher(question: str) -> str:
    researcher_llm = _create_researcher_llm()
    messages: list[BaseMessage] = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for _ in range(3):
        response = researcher_llm.invoke(messages)
        messages.append(response)

        if not isinstance(response, AIMessage) or not response.tool_calls:
            return _message_text(response)

        for tool_call in response.tool_calls:
            tool_name = str(tool_call.get("name", ""))
            if tool_name != "search_learning_reference":
                continue
            tool_args = tool_call.get("args", {}) or {}
            tool_result = search_learning_reference.invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=str(tool_call.get("id", "")),
                )
            )

    return "관련 레퍼런스를 충분히 정리하지 못했습니다. 질문의 핵심 키워드만 바탕으로 답변하세요."


def _compose_specialist_prompt(system_prompt: str, research_notes: str) -> str:
    notes = research_notes.strip() or "관련 레퍼런스가 비어 있습니다. 사용자 질문만 바탕으로 답하세요."
    return f"{system_prompt}\n\nResearcher 노트:\n{notes}"


def _invoke_specialist_response(state: TutorState, system_prompt: str) -> AIMessage:
    specialist_llm = _create_specialist_llm()
    return specialist_llm.invoke(
        [
            SystemMessage(content=_compose_specialist_prompt(system_prompt, state.get("research_notes", ""))),
            *state["messages"],
        ]
    )


def supervisor_node(state: TutorState) -> dict[str, Route]:
    return {"route": route_question(state)}


def researcher_node(state: TutorState) -> dict[str, str]:
    question = _latest_user_message(state["messages"])
    return {"research_notes": run_researcher(question)}


def handoff_node(state: TutorState) -> dict[str, str]:
    route = state.get("route")
    if route not in AGENT_LABELS:
        route = "teach"
    return {"active_agent": AGENT_LABELS[route]}


def teach_node(state: TutorState) -> dict[str, list[AIMessage]]:
    return {"messages": [_invoke_specialist_response(state, TEACH_SYSTEM_PROMPT)]}


def review_node(state: TutorState) -> dict[str, list[AIMessage]]:
    return {"messages": [_invoke_specialist_response(state, REVIEW_SYSTEM_PROMPT)]}


def quiz_node(state: TutorState) -> dict[str, list[AIMessage]]:
    return {"messages": [_invoke_specialist_response(state, QUIZ_SYSTEM_PROMPT)]}


def route_after_handoff(state: TutorState) -> Route:
    route = state.get("route")
    if route in ("teach", "review", "quiz"):
        return route
    return "teach"


def build_graph():
    builder = StateGraph(TutorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("handoff", handoff_node)
    builder.add_node("teach", teach_node)
    builder.add_node("review", review_node)
    builder.add_node("quiz", quiz_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge(START, "researcher")
    builder.add_edge(["supervisor", "researcher"], "handoff")
    builder.add_conditional_edges(
        "handoff",
        route_after_handoff,
        {"teach": "teach", "review": "review", "quiz": "quiz"},
    )

    for node_name in ("teach", "review", "quiz"):
        builder.add_edge(node_name, END)

    return builder.compile()


def _latest_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _message_text(message)
    return ""


def _ensure_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "OPENAI_API_KEY가 필요합니다. .env 또는 환경 변수에 키를 설정한 뒤 다시 실행하세요."
    )


def run_turn_detailed(
    graph,
    user_message: str,
    history: list[BaseMessage] | None = None,
) -> TutorTurnResult:
    _ensure_api_key()
    prior_messages = history or []
    result = graph.invoke(
        {
            "messages": [*prior_messages, HumanMessage(content=user_message)],
            "route": None,
            "research_notes": "",
            "active_agent": None,
        }
    )

    answer = _latest_ai_text(result["messages"])
    route = result.get("route")
    if route not in ("teach", "review", "quiz"):
        route = "teach"
    active_agent = result.get("active_agent") or AGENT_LABELS[route]
    research_notes = result.get("research_notes", "")
    updated_history = [*prior_messages, HumanMessage(content=user_message), AIMessage(content=answer)]

    return TutorTurnResult(
        answer=answer,
        history=updated_history,
        route=route,
        active_agent=active_agent,
        research_notes=research_notes,
    )


def run_turn(
    graph,
    user_message: str,
    history: list[BaseMessage] | None = None,
) -> tuple[str, list[BaseMessage]]:
    result = run_turn_detailed(graph, user_message, history)
    return result.answer, result.history
