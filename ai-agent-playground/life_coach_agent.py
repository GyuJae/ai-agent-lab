import asyncio
import json
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, WebSearchTool, FileSearchTool

load_dotenv()

client = OpenAI()

VECTOR_STORE_CONFIG = os.path.join(os.path.dirname(__file__), "vector_store_config.json")
DEFAULT_GOALS_FILE = os.path.join(os.path.dirname(__file__), "my_goals.txt")


def get_or_create_vector_store():
    """벡터 스토어를 가져오거나 새로 생성합니다."""
    if os.path.exists(VECTOR_STORE_CONFIG):
        with open(VECTOR_STORE_CONFIG, "r") as f:
            config = json.load(f)
        try:
            client.vector_stores.retrieve(config["vector_store_id"])
            return config["vector_store_id"]
        except Exception:
            pass

    vs = client.vector_stores.create(name="Life Coach - 개인 목표 & 일기")

    with open(VECTOR_STORE_CONFIG, "w") as f:
        json.dump({"vector_store_id": vs.id, "uploaded_files": []}, f)

    # 기본 목표 문서 자동 업로드
    if os.path.exists(DEFAULT_GOALS_FILE):
        upload_file_to_vector_store(vs.id, DEFAULT_GOALS_FILE)

    return vs.id


def upload_file_to_vector_store(vector_store_id, file_path):
    """로컬 파일을 벡터 스토어에 업로드합니다."""
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="assistants")
    client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store_id, file_id=file.id
    )
    _record_uploaded_file(os.path.basename(file_path), file.id)
    return file.id


def upload_bytes_to_vector_store(vector_store_id, content, filename):
    """바이트 데이터를 벡터 스토어에 업로드합니다."""
    temp_path = os.path.join(os.path.dirname(__file__), f"_temp_{filename}")
    with open(temp_path, "wb") as f:
        f.write(content)
    try:
        file_id = upload_file_to_vector_store(vector_store_id, temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return file_id


def save_diary_entry(vector_store_id, entry_text):
    """일기 항목을 벡터 스토어에 저장합니다."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    filename = f"diary_{date_str}_{time_str.replace(':', '')}.txt"
    content = f"[일기 - {date_str} {time_str}]\n\n{entry_text}".encode("utf-8")
    return upload_bytes_to_vector_store(vector_store_id, content, filename)


def _record_uploaded_file(filename, file_id):
    """업로드된 파일 기록을 저장합니다."""
    if os.path.exists(VECTOR_STORE_CONFIG):
        with open(VECTOR_STORE_CONFIG, "r") as f:
            config = json.load(f)
    else:
        config = {"uploaded_files": []}
    config.setdefault("uploaded_files", [])
    config["uploaded_files"].append(
        {"filename": filename, "file_id": file_id, "uploaded_at": datetime.now().isoformat()}
    )
    with open(VECTOR_STORE_CONFIG, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_uploaded_files():
    """업로드된 파일 목록을 반환합니다."""
    if os.path.exists(VECTOR_STORE_CONFIG):
        with open(VECTOR_STORE_CONFIG, "r") as f:
            config = json.load(f)
        return config.get("uploaded_files", [])
    return []


# ── 벡터 스토어 초기화 ──
vector_store_id = get_or_create_vector_store()

# ── 에이전트 설정 ──
agent = Agent(
    name="Life Coach",
    instructions="""당신은 따뜻하고 격려하는 라이프 코치입니다.

핵심 역할:
- 사용자의 목표 달성, 습관 형성, 동기부여를 돕습니다
- 파일 검색(file_search)을 통해 사용자가 업로드한 개인 목표와 일기를 참조합니다
- 웹 검색(web_search)을 활용하여 과학적 근거가 있는 조언을 제공합니다

작업 방식:
1. 사용자가 목표나 진행 상황에 대해 물으면, 먼저 file_search로 관련 목표와 기록을 찾습니다
2. 찾은 목표/기록을 바탕으로 개인화된 피드백을 제공합니다
3. 필요시 web_search로 최신 연구나 팁을 찾아 추천합니다
4. 시간에 따른 진행 상황을 추적하고 격려합니다

규칙:
- 항상 한국어로 응답합니다
- 구체적이고 실행 가능한 조언을 제공합니다
- 사용자를 격려하고 긍정적인 톤을 유지합니다
- 목표 문서에서 찾은 정보를 명시적으로 인용하며 언급합니다
- 웹 검색 결과와 개인 목표를 결합하여 맞춤형 조언을 제공합니다
- 일기 기록이 있다면 과거 기록을 참조하여 진행 상황을 추적합니다""",
    model="gpt-4o-mini",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            vector_store_ids=[vector_store_id],
            max_num_results=5,
            include_search_results=True,
        ),
    ],
)

# ── Streamlit UI ──
st.set_page_config(page_title="Life Coach Agent", page_icon="🌱", layout="wide")
st.title("🌱 Life Coach Agent")
st.caption("개인 목표 관리 · 일기 기록 · 진행 상황 추적 · 맞춤형 코칭")

# ── 사이드바: 문서 관리 ──
with st.sidebar:
    st.header("📁 문서 관리")

    # 파일 업로드
    st.subheader("목표 문서 업로드")
    uploaded_file = st.file_uploader(
        "개인 목표 문서를 업로드하세요 (PDF, TXT)",
        type=["pdf", "txt"],
        key="goal_upload",
    )
    if uploaded_file and st.button("📤 업로드", key="upload_btn"):
        with st.spinner("업로드 중..."):
            upload_bytes_to_vector_store(
                vector_store_id, uploaded_file.read(), uploaded_file.name
            )
        st.success(f"'{uploaded_file.name}' 업로드 완료!")

    st.divider()

    # 일기 작성
    st.subheader("📝 일기 작성")
    diary_text = st.text_area(
        "오늘의 일기를 작성하세요",
        placeholder="오늘 운동을 했다. 스쿼트 3세트, 런지 3세트를 완료했다...",
        key="diary_input",
    )
    if diary_text and st.button("💾 일기 저장", key="diary_btn"):
        with st.spinner("저장 중..."):
            save_diary_entry(vector_store_id, diary_text)
        st.success("일기가 저장되었습니다!")

    st.divider()

    # 업로드된 파일 목록
    st.subheader("📋 업로드된 파일")
    files = get_uploaded_files()
    if files:
        for f in files:
            uploaded_at = f.get("uploaded_at", "")[:10]
            st.text(f"• {f['filename']} ({uploaded_at})")
    else:
        st.text("아직 업로드된 파일이 없습니다.")

# ── 채팅 인터페이스 ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("무엇이든 물어보세요! (예: 내 운동 목표 달성은 잘 되어가고 있어?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    user_input = {"role": "user", "content": prompt}

    with st.chat_message("assistant"):
        with st.spinner("목표를 확인하고 조언을 준비하고 있어요..."):
            result = asyncio.run(
                Runner.run(agent, input=st.session_state.agent_history + [user_input])
            )
            response = result.final_output

        st.markdown(response)

    st.session_state.agent_history = result.to_input_list()
    st.session_state.messages.append({"role": "assistant", "content": response})
