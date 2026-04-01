# Socratic Coding Tutor

교육용 에이전트 예제입니다. `Supervisor -> Researcher -> Specialist` 구조를 사용해 질문을 분류하고, 로컬 학습 레퍼런스를 바탕으로 설명·리뷰·퀴즈를 제공합니다.

## 구조

- `Tutor Agent`: 개념 설명과 학습 유도 질문
- `Review Agent`: 코드/에러 분석 중심 피드백
- `Quiz Agent`: 퀴즈 출제와 답변 평가
- `Researcher Agent`: 로컬 학습 레퍼런스 검색 및 요약
- `Supervisor Agent`: 사용자 질문을 `teach`, `review`, `quiz`로 라우팅

고급 패턴
- 멀티 에이전트 아키텍처
- 병렬 워크플로우: `Supervisor`와 `Researcher`가 같은 턴에서 병렬로 준비 작업 수행

## 실행

먼저 `OPENAI_API_KEY`를 `.env` 또는 환경 변수로 설정합니다.

CLI:

```bash
cd socratic-coding-tutor
python main.py "재귀 base case를 쉽게 설명해줘"
```

대화형 CLI:

```bash
cd socratic-coding-tutor
python main.py
```

Streamlit UI:

```bash
cd socratic-coding-tutor
streamlit run streamlit_app.py
```

## 검증

```bash
cd socratic-coding-tutor
python -m unittest discover -s tests
```
