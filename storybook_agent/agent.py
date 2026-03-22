import json

from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools import ToolContext
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Callbacks: 진행 상황 표시
# ---------------------------------------------------------------------------
def before_story_writer(callback_context):
    """스토리 작성 시작 전 진행 상황을 표시합니다."""
    print("\n📖 [진행 상황] 스토리 작성 중...")
    callback_context.state["progress"] = "스토리 작성 중..."
    return None


def after_story_writer(callback_context):
    """스토리 작성 완료 후 진행 상황을 표시합니다."""
    pages = callback_context.state.get("story_pages", [])
    print(f"✅ [진행 상황] 스토리 작성 완료! ({len(pages)}페이지)")
    callback_context.state["progress"] = f"스토리 작성 완료 ({len(pages)}페이지)"
    return None


def _make_before_illustrator_callback(page_num: int):
    """각 페이지별 삽화 생성 시작 콜백을 생성합니다."""
    def before_illustrator(callback_context):
        print(f"🎨 [진행 상황] 이미지 {page_num}/5 생성 중...")
        callback_context.state["progress"] = f"이미지 {page_num}/5 생성 중..."
        return None
    return before_illustrator


def _make_after_illustrator_callback(page_num: int):
    """각 페이지별 삽화 생성 완료 콜백을 생성합니다."""
    def after_illustrator(callback_context):
        print(f"✅ [진행 상황] 이미지 {page_num}/5 생성 완료!")
        callback_context.state["progress"] = f"이미지 {page_num}/5 생성 완료"
        return None
    return after_illustrator


def after_pipeline(callback_context):
    """전체 파이프라인 완료 후 콜백."""
    print("\n🎉 [진행 상황] 동화책 생성 완료!")
    callback_context.state["progress"] = "동화책 생성 완료!"
    return None


# ---------------------------------------------------------------------------
# Tool: Story Writer가 생성한 스토리를 State에 저장
# ---------------------------------------------------------------------------
async def save_story_to_state(
    story_json: str, tool_context: ToolContext
) -> dict:
    """Save the generated story data to session state.

    Args:
        story_json: A JSON string containing a list of page objects.
            Each object must have 'page', 'text', and 'visual' keys.
            Example:
            [
              {"page": 1, "text": "옛날 옛적에...", "visual": "숲 속의 작은 오두막"},
              {"page": 2, "text": "...", "visual": "..."}
            ]

    Returns:
        Confirmation that the story was saved.
    """
    try:
        pages = json.loads(story_json)
        tool_context.state["story_pages"] = pages
        return {
            "status": "success",
            "message": f"{len(pages)}페이지 스토리가 저장되었습니다.",
            "pages": len(pages),
        }
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"JSON 파싱 오류: {e}"}


# ---------------------------------------------------------------------------
# Tool: 특정 페이지의 이미지를 생성 (ParallelAgent용 개별 페이지 생성)
# ---------------------------------------------------------------------------
def _make_generate_single_page_image(page_num: int):
    """특정 페이지 번호의 이미지를 생성하는 도구 함수를 만듭니다."""

    async def generate_single_page_image(tool_context: ToolContext) -> dict:
        """Read a specific page from state and generate its illustration.

        Reads the story page from session state and generates an image
        using Imagen 3.0, then saves it as an artifact.

        Returns:
            A summary of the generated image for this page.
        """
        pages = tool_context.state.get("story_pages")
        if not pages:
            return {"status": "error", "message": "State에 스토리 데이터가 없습니다."}

        if page_num > len(pages):
            return {"status": "error", "message": f"페이지 {page_num}이 존재하지 않습니다."}

        page = pages[page_num - 1]
        visual_desc = page["visual"]
        prompt = (
            f"Children's storybook illustration, cute and colorful style, "
            f"soft pastel colors, friendly characters: {visual_desc}"
        )

        client = genai.Client()

        try:
            response = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/png",
                ),
            )

            image_bytes = response.generated_images[0].image.image_bytes
            artifact = types.Part.from_bytes(
                data=image_bytes, mime_type="image/png"
            )
            filename = f"page_{page_num}.png"
            version = await tool_context.save_artifact(
                filename=filename, artifact=artifact
            )

            return {
                "page": page_num,
                "filename": filename,
                "version": version,
                "status": "success",
            }
        except Exception as e:
            return {
                "page": page_num,
                "status": "error",
                "message": str(e),
            }

    # 함수 이름과 docstring을 페이지별로 고유하게 설정
    generate_single_page_image.__name__ = f"generate_page_{page_num}_image"
    generate_single_page_image.__doc__ = (
        f"Generate an illustration for page {page_num} of the storybook. "
        f"Reads page {page_num} data from state and creates an image using Imagen 3.0. "
        f"You MUST call this tool to generate the image for page {page_num}. "
        f"This tool takes no arguments."
    )

    return generate_single_page_image


# ---------------------------------------------------------------------------
# Agent 1: Story Writer Agent
# ---------------------------------------------------------------------------
story_writer_agent = Agent(
    name="story_writer",
    model="gemini-2.5-flash",
    description="어린이 동화를 작성하는 에이전트",
    instruction="""당신은 어린이 동화 작가입니다.
사용자가 제시한 테마를 바탕으로 5페이지 분량의 어린이 동화를 작성하세요.

반드시 다음 규칙을 따르세요:
1. 정확히 5페이지로 구성합니다.
2. 각 페이지에는 어린이(4-8세)가 이해할 수 있는 짧고 따뜻한 문장을 작성합니다.
3. 각 페이지마다 삽화를 위한 시각적 설명(visual)을 영어로 작성합니다.
4. 반드시 save_story_to_state 도구를 호출하여 스토리를 저장하세요.

JSON 형식 예시:
[
  {"page": 1, "text": "옛날 옛적에, 베니라는 작은 토끼가 살았습니다.", "visual": "A small white rabbit standing in front of a mushroom house in a green forest"},
  {"page": 2, "text": "베니는 탐험을 좋아했는데, 오늘은 하늘이 보라색이었어요!", "visual": "A curious rabbit looking up at a purple sky with wonder"}
]

스토리를 저장한 후, 각 페이지의 내용을 사용자에게 보여주세요:
- Page N:
  Text: "..."
  Visual: "..."
""",
    tools=[save_story_to_state],
    output_key="story_output",
    before_agent_callback=before_story_writer,
    after_agent_callback=after_story_writer,
)


# ---------------------------------------------------------------------------
# Agent 2: 5개의 Illustrator Agent (ParallelAgent로 동시 실행)
# ---------------------------------------------------------------------------
page_illustrator_agents = []
for i in range(1, 6):
    page_agent = Agent(
        name=f"illustrator_page_{i}",
        model="gemini-2.5-flash",
        description=f"동화 {i}페이지의 삽화를 생성하는 에이전트",
        instruction=f"""당신은 어린이 동화의 삽화를 그리는 일러스트레이터입니다.

당신의 임무는 동화책의 {i}페이지 삽화를 생성하는 것입니다.
반드시 generate_page_{i}_image 도구를 호출하여 이미지를 생성하세요.

이미지 생성이 완료되면 결과를 보고하세요:
- Page {i}: ✅ 이미지 생성 완료 (filename)
또는
- Page {i}: ❌ 이미지 생성 실패 (에러 메시지)
""",
        tools=[_make_generate_single_page_image(i)],
        before_agent_callback=_make_before_illustrator_callback(i),
        after_agent_callback=_make_after_illustrator_callback(i),
    )
    page_illustrator_agents.append(page_agent)

# ParallelAgent: 5개의 이미지를 동시에 생성
parallel_illustrator = ParallelAgent(
    name="parallel_illustrator",
    description="5개의 삽화를 동시에 생성하는 병렬 에이전트",
    sub_agents=page_illustrator_agents,
)


# ---------------------------------------------------------------------------
# Root Agent: Sequential Pipeline
# ---------------------------------------------------------------------------
root_agent = SequentialAgent(
    name="storybook_creator",
    description="어린이 동화책을 만드는 파이프라인: 스토리 작성 → 삽화 병렬 생성",
    sub_agents=[story_writer_agent, parallel_illustrator],
    after_agent_callback=after_pipeline,
)
