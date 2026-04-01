import argparse

from langchain_core.messages import BaseMessage

from tutor_core import build_graph, run_turn


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
