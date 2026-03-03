import asyncio

import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool

load_dotenv()

agent = Agent(
    name="Life Coach",
    instructions="""당신은 따뜻하고 격려하는 라이프 코치입니다.
    - 사용자의 목표 달성, 습관 형성, 동기부여를 돕습니다
    - 웹 검색을 활용하여 과학적 근거가 있는 조언을 제공합니다
    - 항상 한국어로 응답합니다
    - 구체적이고 실행 가능한 조언을 제공합니다
    - 사용자를 격려하고 긍정적인 톤을 유지합니다""",
    model="gpt-4o-mini",
    tools=[WebSearchTool()],
)

st.title("🌱 Life Coach Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("무엇이든 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    user_input = {"role": "user", "content": prompt}

    with st.chat_message("assistant"):
        with st.spinner("생각하고 있어요..."):
            result = asyncio.run(
                Runner.run(agent, input=st.session_state.agent_history + [user_input])
            )
            response = result.final_output

        st.markdown(response)

    st.session_state.agent_history = result.to_input_list()
    st.session_state.messages.append({"role": "assistant", "content": response})
