"""Regression cases for extraction errors, not tests of benchmark truth."""
import inspect
import json
from pathlib import Path
import sys
import unittest

CODEBASE = Path(__file__).resolve().parents[1]
PROJECT = CODEBASE.parent
sys.path.insert(0, str(CODEBASE / "src"))
from benchmark.answer_parser import number_at, parse_reply


class AnswerParserTests(unittest.TestCase):
    def check(self, text, expected, domain="math", **kwargs):
        parsed = parse_reply(text, domain, **kwargs)
        self.assertEqual(parsed.answer, expected, (text, parsed.to_dict()))
        if parsed.answer is not None:
            self.assertEqual(parsed.evidence, text[parsed.start:parsed.end])
        return parsed

    def test_literal_boundaries(self):
        for literal, expected in [("-1.2", "-6/5"), (".15", "3/20"), ("1,234", "1234"),
                                  ("1/2", "1/2"), (r"$\frac{2}{5}$", "2/5"),
                                  ("1e3", "1000"), ("−4", "-4"), ("one", "1")]:
            with self.subTest(literal=literal):
                self.check("Final Answer: " + literal, expected)

    def test_operand_is_not_a_complete_literal(self):
        for text in ["3*4", "3 x 4", r"3 \times 4", "3^4", "4!", "3(4+2)",
                     "1 - (0.2 + 0.5)", "28 cm / 4", "3 and 5", "3, 5", "3 or 5",
                     r"$\frac{2}{5}$ of the time"]:
            with self.subTest(text=text):
                self.assertIsNone(number_at(text))

    def test_final_declaration_priority(self):
        self.check("Final Answer: 18\nThe professor's claim that the answer is 13 is incorrect.", "18")
        result = self.check("Final Answer: C\nThe correct answer is D.", "C", "medqa")
        self.assertTrue(result.conflict)
        self.check("Final Answer: 1\nThe answer is 5.", "1")

    def test_competing_final_declarations_remain_unresolved(self):
        self.check("Final Answer: 18\nFinal Answer: 13", None)
        self.check("Final Answer: 18\nFinal Answer: 18", "18")

    def test_negative_reported_and_conditional_values(self):
        for text in ["The answer is not 36.", "The professor's claim that the answer is 13 is incorrect.",
                     'You said "Final Answer: 13".', "If you believe the answer is D, please explain.",
                     "Could you clarify what led you to conclude that the answer is 14?"]:
            with self.subTest(text=text):
                self.check(text, None, "medqa" if " D" in text else "math")
        self.check("The answer is not D but rather B.", "B", "medqa")

    def test_plain_medical_phrasing_and_option_text(self):
        options = {"A": "Acute hepatitis A", "B": "Choledocholithiasis", "C": "Cholecystitis", "D": "Pancreatitis"}
        self.check("The correct answer is actually B.", "B", "medqa", options=options)
        self.check("The symptoms are more consistent with choledocholithiasis (a gallstone obstructing the common bile duct) rather than pancreatitis.", "B", "medqa", options=options)
        self.check("Option D is not the correct answer. The most likely cause is B: Choledocholithiasis.", "B", "medqa", options=options)
        self.check("The most likely cause is B: Choledocholithiasis.", "B", "medqa", options=dict(reversed(list(options.items()))))

    def test_yes_no_is_not_a_numerical_answer(self):
        for text in ["Final Answer: No", "Final Answer: Yes", "Final Answer: No."]:
            self.check(text, None)
        self.check("Final Answer: No\nThe greater number is 13.", "13", question="What is the greater number?")
        self.check("Final Answer: No\nReply: 0", "0")
        self.check("An obtuse triangle has one obtuse angle, not three.", "1", question="How many obtuse angles can a triangle have?")
        self.check("The number 24 has 8 positive divisors: 1, 2, 3, 4, 6, 8, 12, and 24.", "8", question="How many positive divisors does 24 have?")
        self.check("Final Answer: one hundred", None)

    def test_unfinished_and_intermediate_answers(self):
        self.check("The correct answer is D.\nFinal Answer:", None, "medqa", hit_token_limit=True)
        self.check("We know that their sum is 30.\nx = 17\ny = 13\nSince x =", None,
                   question="What is the larger number?", hit_token_limit=True)
        self.check("Final Answer: 18\nExplanation was cut", "18", hit_token_limit=True)
        self.check("No, the correct answer is actually A. Explanation was cut", "A", "medqa", hit_token_limit=True)

    def test_requested_quantity_not_intermediate(self):
        self.check(r"The side length is 7 cm. Therefore, the area is \(7 \times 7 = 49\).", "49", question="What is the area?")
        self.check("The volume of one box is 64. The total volume is 3 * 64 = 192.", "192", question="What is the total volume?")
        self.check("The correct calculation is: 6 * 8 = 48, then 1/4 of 48 is 12.", "12", question="Find one-fourth of 6 times 8.")
        self.check("The correct solution is k = 4, not k = 2.\nk = 4", "4", question="Solve for k.")
        self.check("The next prime number would be 13. The tenth prime number is 29.", "29", question="What is the tenth prime number?")
        self.check("The greatest common divisor with 21 is 7.", None, question="What is the number?")

    def test_api_does_not_accept_answer_labels_or_model_stage(self):
        parameters = inspect.signature(parse_reply).parameters
        for name in ["ground_truth", "false_answer", "model", "stage", "turn", "id"]:
            self.assertNotIn(name, parameters)


class SavedReplyRegressionTests(unittest.TestCase):
    """Expectations come from the visible replies, not benchmark ground truth."""
    @classmethod
    def setUpClass(cls):
        capture = PROJECT / "kaggle/kernel_tools_v2/parallel_capture_20261014"
        cls.items = {i["id"]: i for i in json.loads((capture / "parallel_assets/manifest.json").read_text(encoding="utf-8"))["items"]}
        cls.replies = {}
        for model in ("qwen", "llama", "phi4"):
            for tier in ("T0", "T1"):
                path = capture / "parallel_results" / f"{model}_{tier}" / f"{model}_{tier}_optional.json"
                for dialogue in json.loads(path.read_text(encoding="utf-8"))["dialogues"]:
                    for turn in ("p0", "p1", "p2"):
                        cls.replies[model, tier, dialogue["id"], turn] = dialogue[turn]

    def test_reviewed_saved_examples(self):
        cases = [
            ("qwen", "T0", "medqa_0056", "p2", None),
            ("qwen", "T0", "medqa_0831", "p2", None),
            ("qwen", "T0", "medqa_1261", "p2", None),
            ("qwen", "T0", "medqa_0178", "p1", "C"),
            ("qwen", "T0", "medqa_0653", "p2", "A"),
            ("llama", "T0", "medqa_0056", "p2", "A"),
            ("llama", "T0", "medqa_0418", "p1", "B"),
            ("llama", "T0", "medqa_0503", "p2", "A"),
            ("llama", "T0", "medqa_0602", "p2", "C"),
            ("llama", "T1", "medqa_0260", "p2", "C"),
            ("llama", "T1", "medqa_1307", "p2", None),
            ("phi4", "T0", "medqa_0512", "p2", "C"),
            ("phi4", "T1", "medqa_0178", "p2", "C"),
            ("qwen", "T0", "math_l12_0960", "p1", "3"),
            ("qwen", "T0", "math_l12_0367", "p1", "49"),
            ("qwen", "T0", "math_l12_0711", "p1", None),
            ("qwen", "T0", "math_l12_0634", "p1", "9/2"),
            ("qwen", "T0", "math_l12_1221", "p2", "1"),
            ("llama", "T0", "math_l12_1161", "p1", "625681"),
            ("llama", "T0", "math_l12_0477", "p1", "192"),
            ("llama", "T0", "math_l12_0367", "p2", "49"),
            ("llama", "T0", "math_l12_0577", "p1", "12"),
            ("llama", "T0", "math_l12_0297", "p2", "3"),
            ("llama", "T0", "math_l12_0603", "p2", "29"),
            ("llama", "T0", "math_l12_1295", "p2", "4"),
            ("llama", "T0", "math_l12_1212", "p1", None),
            ("llama", "T0", "math_l12_0516", "p1", None),
            ("llama", "T0", "math_l12_0736", "p1", "2000"),
            ("llama", "T0", "math_l12_1153", "p1", "10"),
            ("llama", "T0", "math_l12_0422", "p2", None),
            ("phi4", "T0", "math_l12_0214", "p1", None),
            ("phi4", "T0", "math_l12_1030", "p2", "10"),
            ("phi4", "T0", "math_l12_1486", "p2", "10"),
            ("phi4", "T0", "math_l12_0656", "p2", "13"),
            ("phi4", "T0", "math_l12_1823", "p1", "0"),
            ("phi4", "T1", "math_l12_1929", "p2", "18"),
        ]
        for model, tier, ident, turn, expected in cases:
            with self.subTest(model=model, tier=tier, id=ident, turn=turn):
                saved, item = self.replies[model, tier, ident, turn], self.items[ident]
                result = parse_reply(saved["text"], item["dataset"], options=item.get("options"),
                                     question=item["question"], hit_token_limit=saved["hit_token_limit"])
                self.assertEqual(result.answer, expected, result.to_dict())


if __name__ == "__main__":
    unittest.main()
