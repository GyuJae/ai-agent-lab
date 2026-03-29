import argparse
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

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
    route: Literal["teach", "review", "quiz"] | None


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


def _create_router_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _create_specialist_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(TOOLS)


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

ROUTER_SYSTEM_PROMPT = """당신은 프로그래밍 학습 튜터의 라우터입니다.
사용자 메시지를 분석해 아래 세 가지 중 하나만 고르세요.

- teach: 개념 설명, 문법 차이, 학습 가이드 요청
- review: 코드 리뷰, 에러 원인 분석, 디버깅 요청
- quiz: 퀴즈 출제, 정답 확인, 시험 대비 요청

반드시 teach, review, quiz 중 하나만 답하세요."""

TEACH_SYSTEM_PROMPT = """당신은 소크라테스식 프로그래밍 튜터입니다.

규칙:
- 개념, 문법, 예제, 차이점을 설명할 때는 먼저 search_learning_reference 도구 사용 여부를 검토하세요.
- 이미 대화 안에 도구 결과가 있으면 같은 주제로 반복 호출하지 말고 최종 답변을 만드세요.
- 정답을 일방적으로 주입하지 말고 학생이 스스로 생각할 질문을 던지세요.
- 한국어로 답하세요.

응답 형식:
1. 질문을 인정하는 짧은 한 줄
2. 핵심 개념 설명 또는 비유 2~3줄
3. 학생이 생각할 유도 질문 2~3개
4. 바로 해볼 작은 연습 1개"""

REVIEW_SYSTEM_PROMPT = """당신은 소크라테스식 코드 리뷰어입니다.

규칙:
- 코드 패턴, 에러 메시지, 자료구조 선택과 관련된 맥락이 필요하면 search_learning_reference 도구를 사용하세요.
- 이미 도구 결과가 있다면 다시 호출하지 말고 그 근거를 바탕으로 피드백을 정리하세요.
- 직접 정답을 고쳐주기보다 학생이 문제를 찾게 만드는 질문 중심으로 답하세요.
- 한국어로 답하세요.

응답 형식:
1. 잘한 점 1~2개
2. 생각해볼 질문 2~3개
3. 디버깅 순서 또는 다음 수정 포인트 1개"""

QUIZ_SYSTEM_PROMPT = """당신은 프로그래밍 퀴즈 마스터입니다.

규칙:
- 퀴즈 주제, 난이도, 핵심 포인트를 잡기 위해 search_learning_reference 도구를 사용할 수 있습니다.
- 이미 도구 결과가 있으면 반복 호출하지 말고 퀴즈 또는 채점 결과를 작성하세요.
- 학생이 답을 맞히면 왜 맞는지 짧게 설명하고, 틀리면 힌트만 주세요.
- 한국어로 답하세요.

응답 형식:
1. 퀴즈 제목 또는 채점 결과
2. 문제 2~3개 또는 답변 평가
3. 마지막에 다음 학습 포인트 1개"""


def route_question(state: TutorState) -> Route:
    messages = state["messages"]
    recent_messages = messages[-4:]
    transcript = "\n".join(
        f"{getattr(message, 'type', 'message')}: {message.content}"
        for message in recent_messages
    )
    router_llm = _create_router_llm()
    response = router_llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=transcript),
        ]
    )

    route = str(response.content).strip().lower()
    if route in ("teach", "review", "quiz"):
        return route
    return "teach"


def router_node(state: TutorState) -> dict[str, Route]:
    return {"route": route_question(state)}


def _run_specialist(state: TutorState, system_prompt: str) -> dict[str, list[AIMessage]]:
    specialist_llm = _create_specialist_llm()
    response = specialist_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def teach_node(state: TutorState) -> dict[str, list[AIMessage]]:
    return _run_specialist(state, TEACH_SYSTEM_PROMPT)


def review_node(state: TutorState) -> dict[str, list[AIMessage]]:
    return _run_specialist(state, REVIEW_SYSTEM_PROMPT)


def quiz_node(state: TutorState) -> dict[str, list[AIMessage]]:
    return _run_specialist(state, QUIZ_SYSTEM_PROMPT)


def route_after_router(state: TutorState) -> Route:
    route = state.get("route")
    if route in ("teach", "review", "quiz"):
        return route
    return "teach"


def route_after_specialist(state: TutorState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "done"


def route_back_from_tools(state: TutorState) -> Route:
    return route_after_router(state)


def build_graph():
    builder = StateGraph(TutorState)
    builder.add_node("router", router_node)
    builder.add_node("teach", teach_node)
    builder.add_node("review", review_node)
    builder.add_node("quiz", quiz_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"teach": "teach", "review": "review", "quiz": "quiz"},
    )

    for node_name in ("teach", "review", "quiz"):
        builder.add_conditional_edges(
            node_name,
            route_after_specialist,
            {"tools": "tools", "done": END},
        )

    builder.add_conditional_edges(
        "tools",
        route_back_from_tools,
        {"teach": "teach", "review": "review", "quiz": "quiz"},
    )

    return builder.compile()


def _latest_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip()
    return ""


def _conversation_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [message for message in messages if getattr(message, "type", "") != "tool"]


def _ensure_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "OPENAI_API_KEY가 필요합니다. .env 또는 환경 변수에 키를 설정한 뒤 다시 실행하세요."
    )


def run_turn(graph, user_message: str, history: list[BaseMessage] | None = None) -> tuple[str, list[BaseMessage]]:
    _ensure_api_key()
    prior_messages = history or []
    result = graph.invoke(
        {
            "messages": [*prior_messages, HumanMessage(content=user_message)],
            "route": None,
        }
    )
    updated_history = _conversation_history(result["messages"])
    return _latest_ai_text(result["messages"]), updated_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Socratic Coding Tutor")
    parser.add_argument("question", nargs="?", help="튜터에게 바로 물어볼 질문")
    args = parser.parse_args()

    graph = build_graph()

    if args.question:
        answer, _ = run_turn(graph, args.question)
        print(answer)
        return

    history: list[BaseMessage] = []
    print("Socratic Coding Tutor")
    print("종료하려면 'exit'를 입력하세요.\n")

    while True:
        try:
            question = input("학생> ").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer, history = run_turn(graph, question, history)
        print(f"\n튜터> {answer}\n")


if __name__ == "__main__":
    main()
