import json

from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext
from google import genai
from google.genai import types


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
# Tool: State에서 스토리를 읽어 각 페이지의 이미지를 생성
# ---------------------------------------------------------------------------
async def generate_page_images(tool_context: ToolContext) -> dict:
    """Read story pages from state and generate an illustration for each page.

    Reads the 'story_pages' key from session state, generates an image for
    each page's visual description using Imagen, and saves each image as
    an artifact.

    Returns:
        A summary of generated images per page.
    """
    pages = tool_context.state.get("story_pages")
    if not pages:
        return {"status": "error", "message": "State에 스토리 데이터가 없습니다."}

    client = genai.Client()
    results = []

    for page in pages:
        page_num = page["page"]
        visual_desc = page["visual"]
        prompt = (
            f"Children's storybook illustration, cute and colorful style, "
            f"soft pastel colors, friendly characters: {visual_desc}"
        )

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

            results.append(
                {
                    "page": page_num,
                    "filename": filename,
                    "version": version,
                    "status": "success",
                }
            )
        except Exception as e:
            results.append(
                {
                    "page": page_num,
                    "status": "error",
                    "message": str(e),
                }
            )

    return {"status": "success", "images": results}


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
)


# ---------------------------------------------------------------------------
# Agent 2: Illustrator Agent
# ---------------------------------------------------------------------------
illustrator_agent = Agent(
    name="illustrator",
    model="gemini-2.5-flash",
    description="동화의 각 페이지에 삽화를 생성하는 에이전트",
    instruction="""당신은 어린이 동화의 삽화를 그리는 일러스트레이터입니다.

이전 에이전트가 State에 저장한 스토리 데이터를 읽어 각 페이지의 이미지를 생성합니다.

반드시 generate_page_images 도구를 호출하여 모든 페이지의 이미지를 생성하세요.

이미지 생성이 완료되면 각 페이지의 결과를 다음 형식으로 보여주세요:
- Page N: ✅ 이미지 생성 완료 (filename)
또는
- Page N: ❌ 이미지 생성 실패 (에러 메시지)
""",
    tools=[generate_page_images],
)


# ---------------------------------------------------------------------------
# Root Agent: Sequential Pipeline
# ---------------------------------------------------------------------------
root_agent = SequentialAgent(
    name="storybook_creator",
    description="어린이 동화책을 만드는 파이프라인: 스토리 작성 → 삽화 생성",
    sub_agents=[story_writer_agent, illustrator_agent],
)
