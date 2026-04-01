import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import tutor_core


class TutorCoreTests(unittest.TestCase):
    def test_search_learning_reference_returns_ranked_match(self):
        result = tutor_core.search_learning_reference.invoke({"query": "재귀 base case 설명해줘"})

        self.assertIn("주제: 재귀", result)
        self.assertIn("퀴즈 포인트", result)

    @patch("tutor_core._ensure_api_key")
    @patch("tutor_core._invoke_specialist_response")
    @patch("tutor_core.run_researcher")
    @patch("tutor_core.route_question")
    def test_run_turn_detailed_uses_selected_specialist(
        self,
        route_question_mock,
        run_researcher_mock,
        invoke_specialist_mock,
        ensure_api_key_mock,
    ):
        route_question_mock.return_value = "quiz"
        run_researcher_mock.return_value = "핵심 개념: 재귀\n흔한 실수: base case 누락"
        invoke_specialist_mock.return_value = AIMessage(content="퀴즈 1. base case는 무엇인가요?")
        ensure_api_key_mock.return_value = None

        graph = tutor_core.build_graph()
        result = tutor_core.run_turn_detailed(graph, "재귀 퀴즈 내줘")

        self.assertEqual(result.route, "quiz")
        self.assertEqual(result.active_agent, "Quiz Agent")
        self.assertEqual(result.answer, "퀴즈 1. base case는 무엇인가요?")
        self.assertEqual(result.history[-1].content, "퀴즈 1. base case는 무엇인가요?")

        specialist_state, _ = invoke_specialist_mock.call_args[0]
        self.assertEqual(specialist_state["research_notes"], "핵심 개념: 재귀\n흔한 실수: base case 누락")


if __name__ == "__main__":
    unittest.main()
