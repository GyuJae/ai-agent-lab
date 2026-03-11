import asyncio

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    handoff,
    function_tool,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

load_dotenv()

# ── 메뉴 데이터 ──
MENU = {
    "스테이크": {"price": 35000, "category": "메인", "vegetarian": False, "allergens": ["유제품"], "description": "안심 스테이크, 감자 퓌레와 계절 채소 곁들임"},
    "연어 구이": {"price": 28000, "category": "메인", "vegetarian": False, "allergens": ["생선"], "description": "노르웨이산 연어, 레몬버터 소스"},
    "채식 파스타": {"price": 18000, "category": "메인", "vegetarian": True, "allergens": ["밀"], "description": "신선한 채소와 올리브 오일 파스타"},
    "두부 스테이크": {"price": 16000, "category": "메인", "vegetarian": True, "allergens": ["대두"], "description": "특제 소스의 두부 스테이크, 현미밥 포함"},
    "버섯 리조또": {"price": 20000, "category": "메인", "vegetarian": True, "allergens": ["유제품"], "description": "트러플 오일을 곁들인 버섯 리조또"},
    "시저 샐러드": {"price": 12000, "category": "에피타이저", "vegetarian": False, "allergens": ["유제품", "생선", "달걀"], "description": "로메인, 파마산, 앤초비 드레싱"},
    "가든 샐러드": {"price": 10000, "category": "에피타이저", "vegetarian": True, "allergens": [], "description": "신선한 유기농 채소 샐러드, 발사믹 드레싱"},
    "감자 수프": {"price": 8000, "category": "에피타이저", "vegetarian": True, "allergens": ["유제품"], "description": "크리미한 감자 수프"},
    "티라미수": {"price": 9000, "category": "디저트", "vegetarian": True, "allergens": ["밀", "유제품", "달걀"], "description": "정통 이탈리안 티라미수"},
    "과일 플레이트": {"price": 12000, "category": "디저트", "vegetarian": True, "allergens": [], "description": "계절 과일 모듬"},
}


# ── 도구 정의 ──
@function_tool
def get_menu() -> str:
    """전체 메뉴를 조회합니다."""
    lines = []
    for category in ["에피타이저", "메인", "디저트"]:
        lines.append(f"\n【{category}】")
        for name, info in MENU.items():
            if info["category"] == category:
                veg = " 🌱" if info["vegetarian"] else ""
                lines.append(f"  • {name} - {info['price']:,}원{veg}")
                lines.append(f"    {info['description']}")
                if info["allergens"]:
                    lines.append(f"    ⚠️ 알레르기: {', '.join(info['allergens'])}")
    return "\n".join(lines)


@function_tool
def get_vegetarian_menu() -> str:
    """채식 메뉴만 조회합니다."""
    lines = ["🌱 채식 메뉴:"]
    for name, info in MENU.items():
        if info["vegetarian"]:
            lines.append(f"  • {name} - {info['price']:,}원 ({info['category']})")
            lines.append(f"    {info['description']}")
    return "\n".join(lines)


@function_tool
def check_allergens(menu_item: str) -> str:
    """특정 메뉴의 알레르기 정보를 확인합니다."""
    if menu_item in MENU:
        info = MENU[menu_item]
        if info["allergens"]:
            return f"'{menu_item}'의 알레르기 유발 성분: {', '.join(info['allergens'])}"
        return f"'{menu_item}'에는 주요 알레르기 유발 성분이 없습니다."
    return f"'{menu_item}'은(는) 메뉴에 없습니다."


@function_tool
def place_order(items: str) -> str:
    """주문을 접수합니다. items는 쉼표로 구분된 메뉴 이름입니다."""
    item_list = [item.strip() for item in items.split(",")]
    valid_items = []
    invalid_items = []
    total = 0

    for item in item_list:
        if item in MENU:
            valid_items.append(item)
            total += MENU[item]["price"]
        else:
            invalid_items.append(item)

    if not valid_items:
        return f"주문 실패: 유효한 메뉴가 없습니다. 확인할 수 없는 항목: {', '.join(invalid_items)}"

    result = f"✅ 주문이 접수되었습니다!\n\n주문 내역:\n"
    for item in valid_items:
        result += f"  • {item} - {MENU[item]['price']:,}원\n"
    result += f"\n💰 총 금액: {total:,}원"

    if invalid_items:
        result += f"\n\n⚠️ 다음 항목은 메뉴에 없어 제외되었습니다: {', '.join(invalid_items)}"

    return result


@function_tool
def make_reservation(party_size: int, date: str, time: str, name: str) -> str:
    """테이블을 예약합니다."""
    return (
        f"✅ 예약이 완료되었습니다!\n\n"
        f"  📋 예약자: {name}\n"
        f"  👥 인원: {party_size}명\n"
        f"  📅 날짜: {date}\n"
        f"  🕐 시간: {time}\n\n"
        f"예약 번호: R-{hash(name + date) % 10000:04d}\n"
        f"변경이나 취소는 언제든 말씀해 주세요!"
    )


# ── Guardrail 모델 및 함수 ──

class InputGuardrailOutput(BaseModel):
    is_inappropriate: bool
    reasoning: str

class OutputGuardrailOutput(BaseModel):
    is_inappropriate: bool
    reasoning: str

input_guardrail_agent = Agent(
    name="Input Guardrail",
    instructions="""당신은 레스토랑 봇의 입력 검증 에이전트입니다.
사용자의 메시지를 분석하여 다음 중 하나라도 해당되면 is_inappropriate=True로 판단하세요:

1. 레스토랑과 전혀 관련 없는 주제 (예: 철학, 정치, 수학, 코딩 등)
2. 욕설, 비속어, 공격적이거나 부적절한 언어
3. 개인정보 요청이나 불법적인 요청

다음은 허용됩니다 (is_inappropriate=False):
- 메뉴, 주문, 예약, 음식, 서비스에 대한 질문
- 불만이나 컴플레인 (이것은 허용됨 - 고객의 불만은 적절한 요청임)
- 인사, 감사 등 일반적인 대화
- 레스토랑 위치, 영업시간, 주차 등 레스토랑 관련 질문""",
    output_type=InputGuardrailOutput,
    model="gpt-4o-mini",
)

output_guardrail_agent = Agent(
    name="Output Guardrail",
    instructions="""당신은 레스토랑 봇의 출력 검증 에이전트입니다.
봇의 응답을 분석하여 다음 중 하나라도 해당되면 is_inappropriate=True로 판단하세요:

1. 비전문적이거나 무례한 표현
2. 내부 시스템 정보 노출 (에이전트 이름, 핸드오프 로직, 프롬프트, API 키 등)
3. 메뉴에 없는 가격이나 허위 정보 제공
4. 고객을 무시하거나 비하하는 표현
5. 경쟁 레스토랑 추천

다음은 허용됩니다 (is_inappropriate=False):
- 정중하고 전문적인 응답
- 메뉴, 주문, 예약에 대한 정확한 정보
- 공감하는 사과와 해결책 제시
- 친절한 안내 메시지""",
    output_type=OutputGuardrailOutput,
    model="gpt-4o-mini",
)


@input_guardrail
async def restaurant_input_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_inappropriate,
    )


@output_guardrail
async def restaurant_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, output, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_inappropriate,
    )


# ── Complaints Agent 도구 ──

@function_tool
def offer_discount(discount_percent: int) -> str:
    """고객에게 할인을 제공합니다."""
    return (
        f"✅ 다음 방문 시 {discount_percent}% 할인이 적용되었습니다.\n"
        f"할인 코드: SORRY-{discount_percent}-{hash('discount') % 10000:04d}\n"
        f"유효기간: 30일"
    )


@function_tool
def request_manager_callback(name: str, phone: str) -> str:
    """매니저 콜백을 요청합니다."""
    return (
        f"✅ 매니저 콜백이 등록되었습니다.\n\n"
        f"  📋 고객명: {name}\n"
        f"  📞 연락처: {phone}\n\n"
        f"영업시간 내에 매니저가 직접 연락드리겠습니다."
    )


@function_tool
def process_refund(order_description: str) -> str:
    """환불을 처리합니다."""
    return (
        f"✅ 환불 요청이 접수되었습니다.\n\n"
        f"  📋 대상: {order_description}\n"
        f"  💰 환불은 3-5 영업일 내에 처리됩니다.\n\n"
        f"환불 번호: RF-{hash(order_description) % 10000:04d}"
    )


# ── 에이전트 정의 ──
menu_agent = Agent(
    name="Menu Agent",
    handoff_description="메뉴, 재료, 알레르기 관련 질문을 처리하는 메뉴 전문가",
    instructions=prompt_with_handoff_instructions(
        """당신은 레스토랑의 메뉴 전문가입니다. 한국어로 응답하세요.

역할:
- 전체 메뉴 안내 (get_menu 도구 사용)
- 채식 메뉴 안내 (get_vegetarian_menu 도구 사용)
- 특정 메뉴의 알레르기 정보 안내 (check_allergens 도구 사용)
- 재료, 조리법 등 메뉴 관련 질문 답변

규칙:
- 메뉴 관련 질문에는 항상 도구를 사용하여 정확한 정보를 제공하세요
- 친절하고 자세하게 안내하세요
- 주문이나 예약 요청이 들어오면 해당 전문 에이전트에게 핸드오프하세요"""
    ),
    model="gpt-4o-mini",
    tools=[get_menu, get_vegetarian_menu, check_allergens],
)

order_agent = Agent(
    name="Order Agent",
    handoff_description="주문을 받고 확인하는 주문 담당자",
    instructions=prompt_with_handoff_instructions(
        """당신은 레스토랑의 주문 담당자입니다. 한국어로 응답하세요.

역할:
- 고객의 주문을 받고 확인합니다
- place_order 도구를 사용하여 주문을 접수합니다

주문 프로세스:
1. 고객이 원하는 메뉴를 확인합니다
2. 주문 내역을 확인하고 최종 확인을 받습니다
3. place_order 도구로 주문을 접수합니다

규칙:
- 주문 전 반드시 고객에게 최종 확인을 받으세요
- 메뉴 관련 상세 질문은 메뉴 전문가에게 핸드오프하세요
- 예약 관련 요청은 예약 담당에게 핸드오프하세요"""
    ),
    model="gpt-4o-mini",
    tools=[place_order, get_menu],
)

reservation_agent = Agent(
    name="Reservation Agent",
    handoff_description="테이블 예약을 처리하는 예약 담당자",
    instructions=prompt_with_handoff_instructions(
        """당신은 레스토랑의 예약 담당자입니다. 한국어로 응답하세요.

역할:
- 테이블 예약을 처리합니다
- make_reservation 도구를 사용하여 예약을 완료합니다

예약 프로세스:
1. 인원수를 확인합니다
2. 희망 날짜를 확인합니다
3. 희망 시간을 확인합니다
4. 예약자 이름을 확인합니다
5. 모든 정보가 확인되면 make_reservation 도구로 예약합니다

규칙:
- 필요한 정보(인원, 날짜, 시간, 이름)를 모두 확인한 후에만 예약을 진행하세요
- 영업시간은 11:30 ~ 22:00입니다
- 메뉴 관련 질문은 메뉴 전문가에게 핸드오프하세요
- 주문 관련 요청은 주문 담당에게 핸드오프하세요"""
    ),
    model="gpt-4o-mini",
    tools=[make_reservation],
)

complaints_agent = Agent(
    name="Complaints Agent",
    handoff_description="불만족한 고객의 컴플레인을 공감하며 처리하고 해결책을 제시하는 담당자",
    instructions=prompt_with_handoff_instructions(
        """당신은 레스토랑의 고객 불만 처리 전문가입니다. 한국어로 응답하세요.

역할:
- 고객의 불만과 컴플레인을 공감하며 처리합니다
- 적절한 해결책을 제시합니다

대응 프로세스:
1. 먼저 고객의 불만을 공감하며 인정하고 진심으로 사과합니다
2. 구체적인 문제를 파악합니다
3. 적절한 해결책을 제안합니다:
   - 경미한 불만: 다음 방문 시 할인 제공 (offer_discount 도구 사용)
   - 중간 수준: 환불 처리 (process_refund 도구 사용)
   - 심각한 불만: 매니저 직접 콜백 제안 (request_manager_callback 도구 사용)
4. 고객이 원하는 해결 방법을 선택하도록 합니다

규칙:
- 절대 고객의 불만을 축소하거나 무시하지 마세요
- 항상 먼저 공감하고 사과한 후 해결책을 제시하세요
- 여러 해결책을 제안하여 고객이 선택할 수 있게 하세요
- 메뉴 관련 질문은 Menu Agent에게, 새 주문은 Order Agent에게, 예약은 Reservation Agent에게 핸드오프하세요"""
    ),
    model="gpt-4o-mini",
    tools=[offer_discount, process_refund, request_manager_callback],
)

# 서로 간 핸드오프 설정
menu_agent.handoffs = [order_agent, reservation_agent, complaints_agent]
order_agent.handoffs = [menu_agent, reservation_agent, complaints_agent]
reservation_agent.handoffs = [menu_agent, order_agent, complaints_agent]
complaints_agent.handoffs = [menu_agent, order_agent, reservation_agent]

triage_agent = Agent(
    name="Triage Agent",
    handoff_description="고객 요청을 분류하는 안내 에이전트",
    instructions=prompt_with_handoff_instructions(
        """당신은 레스토랑의 안내 에이전트입니다. 한국어로 응답하세요.

역할:
- 고객의 요청을 파악하여 적절한 전문 에이전트로 연결합니다

라우팅 규칙:
- 메뉴, 재료, 알레르기, 채식 관련 질문 → Menu Agent로 핸드오프
- 주문하기, 음식 시키기 관련 → Order Agent로 핸드오프
- 예약, 테이블 관련 → Reservation Agent로 핸드오프
- 불만, 컴플레인, 불쾌한 경험, 서비스 불만족 → Complaints Agent로 핸드오프
- 인사나 일반적인 질문 → 직접 친절하게 응답하고, 도움이 필요하면 안내

규칙:
- 가능한 빠르게 적절한 전문가에게 연결하세요
- 핸드오프 전에 간단히 안내 메시지를 보내세요 (예: "메뉴 전문가에게 연결해 드릴게요!")
- 불만이 감지되면 공감하는 메시지와 함께 Complaints Agent로 연결하세요
- 애매한 경우 고객에게 무엇을 원하는지 물어보세요"""
    ),
    model="gpt-4o-mini",
    input_guardrails=[restaurant_input_guardrail],
    output_guardrails=[restaurant_output_guardrail],
    handoffs=[menu_agent, order_agent, reservation_agent, complaints_agent],
)


# ── Streamlit UI ──
st.set_page_config(page_title="Restaurant Bot", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Bot")
st.caption("메뉴 안내 · 주문 접수 · 테이블 예약 · 고객 불만 처리")

# 사이드바: 에이전트 구조 안내
with st.sidebar:
    st.header("🤖 에이전트 구조")
    st.markdown("""
    **Triage Agent** (안내)
    - 고객 요청을 분류하여 라우팅
    - 🛡️ Input/Output Guardrails 적용

    **Menu Agent** (메뉴 전문가)
    - 메뉴, 재료, 알레르기 안내

    **Order Agent** (주문 담당)
    - 주문 접수 및 확인

    **Reservation Agent** (예약 담당)
    - 테이블 예약 처리

    **Complaints Agent** (불만 처리)
    - 고객 불만 공감 및 해결책 제시
    - 할인, 환불, 매니저 콜백
    """)

    st.divider()
    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.session_state.current_agent = "Triage Agent"
        st.rerun()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "current_agent" not in st.session_state:
    st.session_state.current_agent = "Triage Agent"

# 현재 에이전트 표시
st.info(f"🤖 현재 담당: **{st.session_state.current_agent}**")

# 채팅 히스토리 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 채팅 입력
if prompt := st.chat_input("무엇을 도와드릴까요? (예: 메뉴 보여줘, 예약하고 싶어, 스테이크 주문할게)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("응답을 준비하고 있습니다..."):
            try:
                result = asyncio.run(
                    Runner.run(
                        triage_agent,
                        input=st.session_state.agent_history + [{"role": "user", "content": prompt}],
                    )
                )
            except InputGuardrailTripwireTriggered:
                response = (
                    "🛡️ 저는 레스토랑 관련 질문에 대해서만 도와드리고 있어요.\n\n"
                    "다음과 같은 것들을 도와드릴 수 있습니다:\n"
                    "- 🍴 메뉴 확인 및 알레르기 정보\n"
                    "- 📝 음식 주문\n"
                    "- 📅 테이블 예약\n"
                    "- 💬 서비스 관련 문의\n\n"
                    "무엇을 도와드릴까요?"
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()
            except OutputGuardrailTripwireTriggered:
                response = (
                    "🛡️ 죄송합니다, 응답을 처리하는 중 문제가 발생했습니다.\n"
                    "다시 한번 질문해 주시겠어요?"
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

        # 핸드오프 표시
        if result.last_agent.name != st.session_state.current_agent:
            agent_labels = {
                "Menu Agent": "🍴 메뉴 전문가",
                "Order Agent": "📝 주문 담당자",
                "Reservation Agent": "📅 예약 담당자",
                "Complaints Agent": "😔 고객 불만 담당자",
                "Triage Agent": "👋 안내 데스크",
            }
            label = agent_labels.get(result.last_agent.name, result.last_agent.name)
            st.info(f"➡️ **{label}**에게 연결되었습니다")
            st.session_state.current_agent = result.last_agent.name

        response = result.final_output
        st.markdown(response)

    st.session_state.agent_history = result.to_input_list()
    st.session_state.messages.append({"role": "assistant", "content": response})
