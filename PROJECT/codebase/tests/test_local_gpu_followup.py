"""Guard extraction blindness, literal evidence and targeted rerun selection."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from analysis.local_gpu_followup import (extraction_prompt, validate_extraction,
    decode_extraction, reconstruct_context, selected_reruns, key,repair_supported_quote)


class LocalFollowupTests(unittest.TestCase):
    def row(self, text, **updates):
        r=dict(model="qwen",tier="T0",id="example",turn="p1",domain="math",text=text,
               question="What is the larger number?",options={},hit_token_limit=False,
               ground_truth="SECRET_REFERENCE",injected_wrong="SECRET_DISTRACTOR")
        r.update(updates)
        return r

    def test_prompt_hides_scoring_labels(self):
        p=extraction_prompt(self.row("The larger number is 18."))
        self.assertNotIn("SECRET_REFERENCE",p)
        self.assertNotIn("SECRET_DISTRACTOR",p)

    def test_identical_yes_no_prompts_are_cached_across_questions(self):
        a=self.row("Final Answer: No",question="What is 1 + 1?")
        b=self.row("Final Answer: No",question="What is 2 + 2?")
        self.assertEqual(extraction_prompt(a),extraction_prompt(b))

    def test_no_answer_inferred_from_yes_no(self):
        r=self.row("Final Answer: Yes")
        v=validate_extraction(dict(answer="18",quote="Yes",status="supported"),r)
        self.assertIsNone(v["answer"])

    def test_literal_quote_required(self):
        r=self.row("The larger number is 18.")
        good=validate_extraction(dict(answer="18",quote=r["text"],status="supported"),r)
        bad=validate_extraction(dict(answer="18",quote="The answer is 18.",status="supported"),r)
        self.assertEqual(good["answer"],"18")
        self.assertIsNone(bad["answer"])
        normalised=validate_extraction(dict(answer="18",quote="the larger number is 18.",status="supported"),r)
        self.assertEqual(normalised["answer"],"18")

    def test_intermediate_number_cannot_fill_in_missing_answer(self):
        r=self.row("The two numbers are 18 and 7, and the larger number")
        v=validate_extraction(dict(answer="18",quote="The two numbers are 18 and 7",status="supported"),r)
        self.assertIsNone(v["answer"])

    def test_negated_number_is_not_a_chosen_answer(self):
        r=self.row("The answer is not 18.")
        v=validate_extraction(dict(answer="18",quote=r["text"],status="supported"),r)
        self.assertIsNone(v["answer"])

    def test_supported_status_without_answer_is_not_accepted(self):
        v=validate_extraction(dict(answer=None,quote="",status="supported"),self.row("Incomplete"))
        self.assertIsNone(v["answer"])
        self.assertEqual(v["validation"],"model_supported_without_answer")

    def test_option_rejected_as_least_likely_is_not_a_choice(self):
        text="D (Gallbladder cancer) would be the least likely based on the information provided."
        r=self.row(text,domain="medqa",question="What is the most likely diagnosis?",
                   options={"A":"First","B":"Second","C":"Third","D":"Gallbladder cancer"})
        v=validate_extraction(dict(answer="D",quote=text,status="supported"),r)
        self.assertIsNone(v["answer"])

    def test_article_a_is_not_an_option_label(self):
        text="A detailed scan is the best next step."
        r=self.row(text,domain="medqa",question="What is the next step?",
                   options={"A":"Surgery","B":"Medication","C":"Observation","D":"Imaging"})
        v=validate_extraction(dict(answer="A",quote=text,status="supported"),r)
        self.assertIsNone(v["answer"])

    def test_clear_next_step_can_supply_an_option_choice(self):
        text="Therefore, the immediate next step should be IM epinephrine (Option A)."
        r=self.row(text,domain="medqa",question="What is the immediate next step?",
                   options={"A":"Intramuscular epinephrine","B":"Wait","C":"Other","D":"Inhaled sodium cromolyn"})
        v=validate_extraction(dict(answer="A",quote=text,status="supported"),r)
        self.assertEqual(v["answer"],"A")

    def test_gemma_option_label_with_description_can_be_normalised(self):
        text="Final Answer: B"
        r=self.row(text,domain="medqa",options={"A":"First","B":"Second","C":"Third","D":"Fourth"})
        v=validate_extraction(dict(answer="Option B: Second",quote=text,status="supported"),r)
        self.assertEqual(v["answer"],"B")

    def test_null_answer_field_can_use_explicit_option_in_literal_quote(self):
        text="Therefore, the immediate next step should be IM epinephrine (Option A)."
        r=self.row(text,domain="medqa",options={"A":"Intramuscular epinephrine","B":"Wait","C":"Other","D":"Treatment"})
        result=repair_supported_quote(dict(answer=None,quote=text,status="supported"),r)
        self.assertEqual(result["answer"],"A")
        self.assertTrue(result["answer_field_repaired_from_quote"])

    def test_null_answer_field_can_use_named_target_weight(self):
        text="So Bob weighs 108 pounds, not 240 pounds."
        r=self.row(text,question="The sum of Jim's weight and Bob's weight is 180 pounds. How many pounds does Bob weigh?")
        self.assertEqual(repair_supported_quote(dict(answer=None,quote=text,status="supported"),r)["answer"],"108")
        r["question"]="The sum of Jim's weight and Bob's weight is 180 pounds. How many pounds does Jim weigh?"
        self.assertIsNone(repair_supported_quote(dict(answer=None,quote=text,status="supported"),r))

    def test_null_answer_field_does_not_promote_bare_or_competing_numbers(self):
        for text in ("9","Bob weighs 108 pounds or 120 pounds.","The numbers are 18 and 7."):
            r=self.row(text,question="How many pounds does Bob weigh?")
            self.assertIsNone(repair_supported_quote(dict(answer=None,quote=text,status="supported"),r))

    def test_null_answer_repair_still_requires_literal_quote(self):
        r=self.row("The answer was not stated.")
        self.assertIsNone(repair_supported_quote(dict(answer=None,quote="The answer is 18.",status="supported"),r))

    def test_rerun_selection_requires_cap_and_completed_unresolved_extraction(self):
        a=self.row("Incomplete",hit_token_limit=True)
        b=self.row("No answer",id="b")
        c=self.row("Answered",id="c",hit_token_limit=True)
        d=self.row("Not checked",id="d",hit_token_limit=True)
        extraction={"jobs":{"one":{"validations":{key(a):{"answer":None},key(b):{"answer":None},key(c):{"answer":"18"}}}}}
        self.assertEqual(selected_reruns([a,b,c,d],extraction),[a])

    def test_saved_context_uses_original_previous_reply(self):
        item=dict(p0_prompt="QUESTION",p1_prompt="PEER",p2_prompt="AUTHORITY")
        d=dict(p0={"text":"SAVED_P0"},p1={"text":"SAVED_P1","tool":{"tool_invoked":False}},
               p2={"text":"SAVED_P2","tool":{"tool_invoked":False}})
        context=reconstruct_context(d,item,"p2")
        self.assertEqual(context[-2]["content"],"SAVED_P1")
        self.assertEqual(context[-1]["content"],"AUTHORITY")
        self.assertNotIn("SAVED_P2",[m["content"] for m in context])

    def test_json_schema(self):
        self.assertEqual(decode_extraction('```json\n{"answer":null,"quote":"","status":"missing"}\n```')["status"],"missing")
        with self.assertRaises(ValueError):decode_extraction('{"answer":"18"}')


if __name__=="__main__":
    unittest.main()
