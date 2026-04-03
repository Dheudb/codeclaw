"""
AskUserQuestionTool — structured user interaction via multiple-choice questions.

Mirrors Claude Code's AskUserQuestionTool: presents 1–4 questions with options,
collects user answers, and returns them to the agent. Supports multi-select.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool


class QuestionOption(BaseModel):
    label: str = Field(..., description="Display text for this option.")
    description: str = Field("", description="Optional longer description.")


class Question(BaseModel):
    question: str = Field(..., description="The question text to present to the user.")
    options: List[QuestionOption] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="2–4 answer options. Users can always provide custom text via 'Other'.",
    )
    multiSelect: bool = Field(
        False,
        description="If true, allow the user to select multiple options.",
    )


class AskUserQuestionInput(BaseModel):
    questions: List[Question] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="1–4 questions to present to the user.",
    )


class AskUserQuestionTool(BaseAgenticTool):
    name = "ask_user_question_tool"
    description = (
        "Asks the user multiple choice questions to gather information, "
        "clarify ambiguity, understand preferences, make decisions or offer them choices."
    )
    input_schema = AskUserQuestionInput
    risk_level = "low"
    is_read_only = True

    def prompt(self) -> str:
        return """Use this tool when you need to ask the user questions during execution. This allows you to:
1. Gather user preferences or requirements
2. Clarify ambiguous instructions
3. Get decisions on implementation choices as you work
4. Offer choices to the user about what direction to take

Usage notes:
- Users will always be able to select "Other" to provide custom text input
- Use multiSelect: true to allow multiple answers to be selected for a question
- If you recommend a specific option, make that the first option in the list and add "(Recommended)" at the end of the label

Plan mode note: In plan mode, use this tool to clarify requirements or choose between approaches BEFORE finalizing your plan. Do NOT use this tool to ask "Is my plan ready?" or "Should I proceed?" — use exit_plan_mode for plan approval. IMPORTANT: Do not reference "the plan" in your questions (e.g., "Do you have feedback about the plan?", "Does the plan look good?") because the user cannot see the plan in the UI until you call exit_plan_mode. If you need plan approval, use exit_plan_mode instead."""

    def build_permission_summary(self, questions: List[dict] = None) -> str:
        questions = questions or []
        lines = [f"Asking {len(questions)} question(s):"]
        for i, q in enumerate(questions[:4]):
            text = q.get("question", "")[:100]
            opt_count = len(q.get("options", []))
            lines.append(f"  Q{i+1}: {text} ({opt_count} options)")
        return "\n".join(lines)

    async def execute(self, questions: List[dict] = None) -> str:
        """
        Present questions to the user and collect answers.

        In CLI mode this uses rich prompts; in SDK mode it returns the
        question payload for the host to render.
        """
        questions = questions or []
        ui_callback = self.context.get("ask_user_callback")

        if ui_callback is None:
            return self._fallback_text_prompt(questions)

        import inspect
        result = ui_callback(questions)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, dict):
            return self._format_answers(questions, result)
        return str(result)

    def _fallback_text_prompt(self, questions: list) -> str:
        """Synchronous fallback when no UI callback is registered."""
        answers = {}
        for i, q in enumerate(questions):
            q_text = q.get("question", f"Question {i+1}")
            options = q.get("options", [])
            multi = q.get("multiSelect", False)

            print(f"\n{'─'*60}")
            print(f"  {q_text}")
            for j, opt in enumerate(options):
                label = opt.get("label", f"Option {j+1}")
                desc = opt.get("description", "")
                desc_str = f" — {desc}" if desc else ""
                print(f"    [{j+1}] {label}{desc_str}")
            print(f"    [0] Other (type custom answer)")

            if multi:
                raw = input("  Select one or more (comma-separated numbers): ").strip()
            else:
                raw = input("  Select: ").strip()

            if not raw or raw == "0":
                custom = input("  Your answer: ").strip()
                answers[q_text] = custom
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    selected = []
                    for idx in indices:
                        if 0 <= idx < len(options):
                            selected.append(options[idx].get("label", ""))
                    answers[q_text] = selected if multi else (selected[0] if selected else raw)
                except (ValueError, IndexError):
                    answers[q_text] = raw

        return self._format_answers(questions, answers)

    def _format_answers(self, questions: list, answers: dict) -> str:
        lines = []
        for q in questions:
            q_text = q.get("question", "")
            answer = answers.get(q_text, "(no answer)")
            if isinstance(answer, list):
                answer_str = ", ".join(str(a) for a in answer)
            else:
                answer_str = str(answer)
            lines.append(f"Q: {q_text}\nA: {answer_str}")
        return "\n\n".join(lines)
