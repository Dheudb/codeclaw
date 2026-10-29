"""Question tool contract: explicit nested schema, no guessed input, actual answers."""
import asyncio
import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from xuloom.tools.ask_user_question_tool import AskUserQuestionTool, AskUserQuestionInput, QUESTION_EXAMPLE


class QuestionContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tool = AskUserQuestionTool()
        self.tool.context = {}

    def test_provider_schema_is_inline_and_examples_validate(self):
        definition = self.tool.get_tool_definition()
        schema = definition["input_schema"]
        self.assertNotIn('"$ref"', json.dumps(schema))
        question = schema["properties"]["questions"]["items"]
        self.assertEqual(set(question["required"]), {"question", "options"})
        self.assertEqual(question["additionalProperties"], False)
        option = question["properties"]["options"]["items"]
        self.assertEqual(option["required"], ["label"])
        self.assertEqual(question["properties"]["options"]["minItems"], 2)
        self.assertEqual(question["properties"]["options"]["maxItems"], 4)
        for example in schema["examples"]:
            AskUserQuestionInput(**example)
        self.assertIn("Do not place label/description directly in questions", definition["description"])
        self.assertFalse(self.tool.is_concurrency_safe_call())

    async def test_flat_options_empty_and_duplicate_inputs_do_not_open_picker(self):
        called = []
        self.tool.context = {"ask_user_callback": lambda q: called.append(q)}
        invalid = [
            {"questions": [{"label": "A"}, {"label": "B"}]},
            {"questions": [{"question": " ", "options": [{"label": "A"}, {"label": "B"}]}]},
            {"questions": [{"question": "Q", "options": [{"label": "A"}, {"label": " A "}]}]},
            {"questions": [QUESTION_EXAMPLE["questions"][0]] * 2},
            {"questions": [{"question": "Q", "options": [{"label": "A"}, {"label": "B"}], "typo": True}]},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                await self.tool(**payload)
        self.assertEqual(called, [])

    async def test_multiselect_and_custom_answers_are_returned(self):
        questions = [{"question": "Topics?", "options": [{"label": "A"}, {"label": "B"}], "multiSelect": True},
            {"question": "Other preference?", "options": [{"label": "X"}, {"label": "Y"}]}]

        async def answer(payload):
            self.assertTrue(payload[0]["multiSelect"])
            self.assertFalse(payload[1]["multiSelect"])
            self.assertEqual([item["question"] for item in payload], ["Topics?", "Other preference?"])
            return {"Topics?": ["A", "B"], "Other preference?": "My own answer"}

        self.tool.context = {"ask_user_callback": answer}
        result = await self.tool(questions=questions)
        self.assertIn("A: A, B", result)
        self.assertIn("A: My own answer", result)

    async def test_cli_select_and_custom_text(self):
        with patch("builtins.input", side_effect=["2"]):
            self.assertIn("A: Broader change", await self.tool(**QUESTION_EXAMPLE))
        with patch("builtins.input", side_effect=["0", "Custom choice"]):
            self.assertIn("A: Custom choice", await self.tool(**QUESTION_EXAMPLE))

    async def test_cancel_pending_question_does_not_invent_answer(self):
        opened = asyncio.Event()

        async def picker(questions):
            opened.set()
            await asyncio.Future()

        self.tool.context = {"ask_user_callback": picker}
        run = asyncio.create_task(self.tool(**QUESTION_EXAMPLE))
        await asyncio.wait_for(opened.wait(), 1)
        run.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run
