import streamlit as st

from tutor_core import build_graph, run_turn_detailed


@st.cache_resource
def get_graph():
    return build_graph()


st.set_page_config(page_title="Education Agent", page_icon="🎓", layout="wide")
st.title("🎓 Education Agent")
st.caption("Supervisor · Researcher · Tutor/Review/Quiz Specialist")

with st.sidebar:
    st.header("아키텍처")
    st.markdown(
        """
**Supervisor Agent**
- 사용자 요청을 `teach`, `review`, `quiz`로 라우팅합니다.

**Researcher Agent**
- 로컬 학습 레퍼런스를 찾아 이번 턴의 배경 노트를 만듭니다.

**Tutor Agent**
- 개념 설명과 학습 유도 질문을 제공합니다.

**Review Agent**
- 코드/에러 분석 질문을 리뷰 중심으로 안내합니다.

**Quiz Agent**
- 퀴즈 출제와 답변 평가를 담당합니다.
"""
    )
    st.divider()
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.last_agent = "Supervisor Agent"
        st.session_state.last_route = "-"
        st.session_state.last_research_notes = ""
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_agent" not in st.session_state:
    st.session_state.last_agent = "Supervisor Agent"
if "last_route" not in st.session_state:
    st.session_state.last_route = "-"
if "last_research_notes" not in st.session_state:
    st.session_state.last_research_notes = ""

st.info(f"현재 담당: **{st.session_state.last_agent}** | 라우트: **{st.session_state.last_route}**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("research_notes"):
            with st.expander("Researcher 노트"):
                st.markdown(message["research_notes"])

if prompt := st.chat_input("개념 설명, 코드 리뷰, 퀴즈 요청을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Education Agent가 답변을 준비하고 있습니다..."):
            result = run_turn_detailed(get_graph(), prompt, st.session_state.history)

        st.markdown(result.answer)
        with st.expander("이번 턴 처리 보기"):
            st.markdown(f"- Route: `{result.route}`")
            st.markdown(f"- Specialist: `{result.active_agent}`")
            st.markdown(result.research_notes or "Researcher 노트 없음")

    st.session_state.history = result.history
    st.session_state.last_agent = result.active_agent
    st.session_state.last_route = result.route
    st.session_state.last_research_notes = result.research_notes
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "research_notes": result.research_notes,
        }
    )
