"""Evidence-based extraction from saved benchmark replies (answer-parser-4.0).

This module reads a response, never the reference answer or injected distractor.
The same rules apply to all models and turns. Explicit final declarations take
priority. Missing/ambiguous answers remain unresolved rather than being guessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import re
import unicodedata

VERSION = "answer-parser-4.0"


@dataclass(frozen=True)
class Candidate:
    value: str
    start: int
    end: int
    evidence: str
    rule: str
    priority: int
    subject: str = ""


@dataclass
class ParseResult:
    answer: str | None
    status: str
    rule: str | None
    evidence: str | None
    start: int | None
    end: int | None
    conflict: bool
    flags: list[str]
    candidates: list[dict]
    version: str = VERSION

    def to_dict(self):
        return asdict(self)


NUMBER = r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
NUMBER_RE = re.compile(NUMBER)
WORD_NUMBERS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(), range(21)))
NUMBER_WORD = r"(?:" + "|".join(WORD_NUMBERS) + r")"
ADVERBS = r"(?:(?:indeed|actually|still|always|exactly|approximately|most\s+likely|usually)\s+)*"
DECOR = r"[\s*$]*(?:\\\(|\\\[)?[\s*$]*"
FINAL_RE = re.compile(r"\bfinal\s+answer\s*[:=]\s*", re.I)
ANSWER_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:correct\s+|final\s+|best\s+)?answer|(?:the\s+)?result)"
    r"(?:\s+(?:based\s+on|according\s+to|from|to)\b[^\n.!?;]{0,130}?)?"
    r"\s+(?:is\s+confirmed\s+to\s+be|is|remains|would\s+(?:still\s+)?be|should\s+(?:indeed\s+)?be)\s*[:*]?\s*" + ADVERBS,
    re.I)
QUANTITY = (
    r"answer|result|value|solution|area|perimeter|volume|radius|diameter|length|distance|"
    r"hypotenuse|height|width|weight|age|number|integer|sum|mean|average|probability|percentage|"
    r"digit|remainder|factor|divisor|multiple|gcd|gcf|lcm|sine|cosine|tangent|root|floor|ceiling|"
    r"cost|price|amount|time|requirement|representation|evaluation|rounding|supplement|calculation|"
    r"apartment|coefficient|fraction|binomial|"
    r"count|sides|ways|order|combinations|possibilities|term")
GOAL_EXTRA = r"greater|larger|smaller|positive|negative|total|next|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
MED_SUBJECT = (
    r"answer|option|choice|diagnosis|cause|explanation|treatment|response|step|approach|"
    r"test|pathogen|agent|disinfectant|intervention|assessment|recommendation")
QUALIFIED_MED = re.compile(
    r"\b(?:most\s+(?:likely|appropriate|fitting|sensitive|critical|probable)|best|correct|"
    r"primary|initial|appropriate|likely)\b[^\n.!?;]{0,170}?\b(?:" + MED_SUBJECT + r")\b"
    r"[^\n.!?;]{0,100}?\b(?:is|remains|would\s+(?:still\s+)?be|should\s+be)\s*:?\s*" + ADVERBS,
    re.I)


def canonical_number(value):
    value = unicodedata.normalize("NFKC", value).replace("−", "-").replace(",", "").replace(" ", "")
    try:
        number = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def number_at(text, offset=0, *, allow_words=False):
    """Read a complete literal, not an operand/list/partial decimal/exponent."""
    decoration = re.match(DECOR, text[offset:])
    start = offset + decoration.end()
    latex = re.match(r"\\(?:d?frac)\s*\{(" + NUMBER + r")\}\s*\{(" + NUMBER + r")\}", text[start:])
    if latex:
        end = start + latex.end()
        raw = latex[1] + "/" + latex[2]
    else:
        match = NUMBER_RE.match(text, start)
        if not match:
            word = re.match(NUMBER_WORD + r"\b", text[start:], re.I) if allow_words else None
            if not word:
                return None
            end = start + word.end()
            if re.match(r"\s+(?:hundred|thousand|million|billion|point|and|or|times|of)\b", text[end:], re.I):
                return None
            return str(WORD_NUMBERS[word[0].lower()]), start, end
        end = match.end()
        ratio = re.match(r"\s*/\s*(" + NUMBER + r")", text[end:])
        if ratio:
            end += ratio.end()
            raw = match[0] + "/" + ratio[1]
        else:
            raw = match[0]
    tail = text[end:]
    # A scalar must not be the first item of a list or an unfinished expression.
    if re.match(r"(?:[\d]|[.,]\d|[_^!]|\s*(?:[+*/×÷=]|\*\*|-(?=\s*[\d(])|\((?=\d)))", tail):
        return None
    if re.match(r"\s+(?:times\b|of\b|x\s+\d)", tail, re.I):
        return None
    if re.match(r"\s*\\(?:times|cdot|div)\b", tail):
        return None
    if re.match(r"\$?\s+of\b", tail, re.I):
        return None
    if re.match(r"\s*(?:cm|mm|km|ml|m|ft|kg|lb|inches|units)\b\s*[/+*×÷=]", tail, re.I):
        return None
    if re.match(r"\s*(?:,\s*(?:and\s+)?|and\s+|or\s+)" + NUMBER, tail, re.I):
        return None
    if tail and tail[0].isalpha() and not re.match(r"(?:cm|mm|km|ml|m|ft|kg|lb|inches|units)\b", tail, re.I):
        return None
    value = canonical_number(raw)
    return (value, start, end) if value is not None else None


def letter_at(text, offset=0):
    match = re.match(r"[\s*]*(?:(?:option|choice)\s*)?\(?([A-D])\)?(?=\s*[:,;.(\u2014)*]|\s*$|\s*\n)", text[offset:], re.I)
    if not match:
        return None
    start, end = offset + match.start(1), offset + match.end(1)
    return match[1].upper(), start, end


def statement_start(text, position):
    # Do not split decimal points; sentences are separated by punctuation+space.
    splits = list(re.finditer(r"[.!?]\s+|\n", text[:position]))
    return splits[-1].end() if splits else 0


def rejected_or_reported(text, start, end):
    before = text[statement_start(text, start):start]
    after = text[end:end + 100]
    if re.search(r"\b(?:not|never)\s*(?:indeed\s+|actually\s+)?$", before, re.I):
        return True
    if re.search(r"\b(?:claim|assertion|suggestion|guess|thought|statement)\b[^.!?\n]{0,100}$", before, re.I):
        return True
    if re.search(r"\b(?:you|the\s+(?:user|professor|specialist))\s+(?:said|claimed|suggested|think|thought)\b[^.!?\n]*$", before, re.I):
        return True
    if re.search(r"\b(?:if\s+you|if\s+we\s+assume|if\s+we\s+consider|if\s+.*?instead|hypothetically|if\s+.*?referring)\b", before, re.I):
        return True
    if re.search(r"\b(?:clarify|led\s+you\s+to\s+conclude|what\s+made\s+you)\b", before, re.I):
        return True
    if before.count('"') % 2:
        return True
    if re.match(r"\s*\??\s*(?:is|was|would\s+be)\s+(?:not\s+correct|incorrect|wrong|not\s+the\s+correct)", after, re.I):
        return True
    if re.match(r"\s*\?", after):
        return True
    return False


def option_matches(fragment, options):
    """Match supplied option text, never medical knowledge or reference labels."""
    matches = []
    for key, option in options.items():
        words = re.findall(r"\w+", option, re.UNICODE)
        if not words:
            continue
        pattern = r"(?<!\w)" + r"[\W_]+".join(re.escape(w) for w in words) + r"(?!\w)"
        for match in re.finditer(pattern, fragment, re.I):
            matches.append((key, match.start(), match.end()))
    return matches


def med_value_at(text, offset, options):
    if re.match(r"\s*(?:not|never)\b", text[offset:], re.I):
        return None
    direct = letter_at(text, offset)
    if direct:
        return direct
    tail = text[offset:]
    # Stay inside this claim; do not search the next sentence for a label.
    stop = re.search(r"[.!?](?:\s|$)|\n\s*\n|\b(?:but|however|rather\s+than)\b", tail, re.I)
    fragment = tail[:stop.start()] if stop else tail[:200]
    labels = list(re.finditer(r"\((?:option\s+)?([A-D])(?:\)|$)", fragment, re.I))
    if len({m[1].upper() for m in labels}) == 1 and labels:
        m = labels[0]
        return m[1].upper(), offset + m.start(1), offset + m.end(1)
    matches = option_matches(fragment, options)
    if len({m[0] for m in matches}) == 1 and matches:
        key, start, end = matches[0]
        return key, offset + start, offset + end
    return None


def target_words(question):
    question = question.replace(r"\lfloor", " floor ").replace(r"\lceil", " ceiling ")
    tails = list(re.finditer(r"\b(?:what|find|determine|compute|calculate|evaluate|how\s+many)\b", question, re.I))
    target = question[tails[-1].start():] if tails else question.split(".")[-1]
    return set(re.findall(r"\b(?:" + QUANTITY + "|" + GOAL_EXTRA + r")\b", target.lower()))


def parse_reply(text, domain, *, options=None, question="", hit_token_limit=False):
    """Extract the model's stated answer; do not solve the question.

    Explicit scalar Final Answer wins, with conflicts recorded separately.
    Without one, use clear positive answer/quantity statements. Distinct equally
    supported answers remain ambiguous. Weak arithmetic is not used at a cap.
    """
    if domain not in {"medqa", "math"}:
        raise ValueError(f"Unknown dataset: {domain}")
    options = options or {}
    candidates = []
    flags = []

    def add(parsed, rule, priority, anchor=None, subject=""):
        if not parsed:
            return
        value, start, end = parsed
        begin = start if anchor is None else anchor
        if rejected_or_reported(text, start, end):
            return
        candidate = Candidate(value, start, end, text[start:end], rule, priority, subject)
        if candidate not in candidates:
            candidates.append(candidate)

    value_at = (lambda offset: med_value_at(text, offset, options)) if domain == "medqa" else (lambda offset: number_at(text, offset, allow_words=True))
    finals = list(FINAL_RE.finditer(text))
    for final in finals:
        add(value_at(final.end()), "final_declaration", 100, final.start())
    # An empty terminal final declaration does not justify inventing its value.
    unfinished_final = bool(re.search(r"\bFinal\s*(?:Answer)?\s*:?\s*$", text, re.I))
    if unfinished_final:
        flags.append("unfinished_final_declaration")
    bare_yes_no = bool(re.fullmatch(r"\s*Final\s+Answer:\s*(?:yes|no)[.!]?\s*", text, re.I))
    if bare_yes_no:
        flags.append("yes_no_only")
    if not finals:
        parsed = letter_at(text) if domain == "medqa" else number_at(text)
        if parsed:
            # Bare values must finish the first line or precede explanation punctuation.
            tail = text[parsed[2]:].split("\n")[0].strip(" *$")
            if not tail or tail.startswith((".", ":", ")")):
                add(parsed, "leading_answer", 95)
    for match in ANSWER_RE.finditer(text):
        add(value_at(match.end()), "answer_statement", 85, match.start())
    if domain == "medqa":
        for match in QUALIFIED_MED.finditer(text):
            add(value_at(match.end()), "medical_answer_statement", 75, match.start())
        for match in re.finditer(r"\b(?:more\s+consistent\s+with|characteristic\s+of|suggestive\s+of|"
                                 r"most\s+consistent\s+with|strongly\s+suggests)\s+", text, re.I):
            add(med_value_at(text, match.end(), options), "named_diagnosis_statement", 70, match.start())
        for match in re.finditer(r"\bnot\b[^\n.!?;]{0,140}?\bbut\s+(?:rather|instead)\s+", text, re.I):
            add(med_value_at(text, match.end(), options), "explicit_replacement", 85, match.start())
        # "Option A is the most appropriate response"; negatives never qualify.
        positive = re.compile(r"\b(?:option|choice)\s+([A-D])\s+(?:is|remains)\s+" + ADVERBS +
                              r"(?:the\s+)?(?:correct|best|most\s+appropriate|more\s+appropriate)\b", re.I)
        for match in positive.finditer(text):
            add((match[1].upper(), match.start(1), match.end(1)), "option_is_answer", 75, match.start())
        # A named option/parenthetic label followed by an affirmative recommendation.
        subject_positive = re.compile(
            r"\s+(?:is|remains|would\s+be|should\s+be)\s+" + ADVERBS +
            r"(?:the\s+|a\s+)?(?:most\s+(?:likely|appropriate|probable|prudent)|best|correct|"
            r"more\s+(?:appropriate|likely)|better\s+choice|suitable\s+treatment)\b", re.I)
        for match in subject_positive.finditer(text):
            begin = statement_start(text, match.start())
            fragment = text[begin:match.start()]
            labels = list(re.finditer(r"\((?:option\s+)?([A-D])\)", fragment, re.I))
            matches = [(m[1].upper(), begin + m.start(1), begin + m.end(1)) for m in labels]
            matches += [(k, begin + s, begin + e) for k, s, e in option_matches(fragment, options)]
            if matches:
                add(max(matches, key=lambda x: x[2]), "named_option_is_answer", 70, begin)
        # A clear selection in a stand-alone labelled line after "... is:".
        for match in re.finditer(r"(?:answer|response|diagnosis|choice|step)[^\n.!?]{0,100}\bis\s*:\s*\n", text, re.I):
            add(value_at(match.end()), "answer_on_next_line", 75, match.start())
    else:
        goals = target_words(question)
        specific_goals = goals - {"answer", "result", "value", "solution", "number", "integer", "calculation", "amount"}
        for match in re.finditer(r"\b(?:is\s+equal\s+to|is|equals|would\s+be|must\s+be|should\s+be|yields|gives)\s+" + ADVERBS, text, re.I):
            begin = statement_start(text, match.start())
            subject = text[begin:match.start()]
            # Use the local clause, not an earlier quantity in the same sentence.
            separators = list(re.finditer(r",\s+(?:and\s+|so\s+|then\s+|which\s+|where\s+|(?=the\b|its\b|their\b|it\b|when\b))|\b(?:because|since)\s+", subject, re.I))
            if separators:
                subject = subject[separators[-1].end():]
            if not re.search(r"\b(?:" + QUANTITY + r")\b", subject, re.I):
                continue
            subject_words = set(re.findall(r"\b(?:" + QUANTITY + "|" + GOAL_EXTRA + r")\b", subject.lower()))
            if specific_goals and not (specific_goals & subject_words) and not (subject_words & {"answer", "result", "solution", "calculation"}):
                continue
            if goals and not specific_goals and not (goals & subject_words) and not (subject_words & {"answer", "result", "solution", "calculation"}):
                continue
            physical = {"area", "perimeter", "volume", "radius", "diameter", "length", "distance", "height", "width", "weight", "age"}
            if (goals & physical) and (subject_words & physical) and not (goals & physical & subject_words):
                continue
            ordinals = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"}
            if (goals & ordinals) and (subject_words & (ordinals | {"next"})) and not (goals & ordinals & subject_words):
                continue
            if re.search(r"\b(?:we\s+know|given|problem\s+states|initially|we\s+also\s+know)\b", subject, re.I):
                continue
            if text.rfind("(", begin, match.start()) > text.rfind(")", begin, match.start()) and re.search(r"\(\s*which\b", subject, re.I):
                continue
            parsed = number_at(text, match.end())
            # When an assertion gives a written calculation, use its explicit RHS,
            # never compute the question or grab a number from a later sentence.
            if parsed is None:
                rest = text[match.end():]
                limit = re.search(r"[.!?](?:\s|$)|\n\s*\n|,\s*not\b|\b(?:but|however)\b", rest, re.I)
                fragment = rest[:limit.start()] if limit else rest[:300]
                equals = list(re.finditer(r"=\s*", fragment))
                if equals:
                    last = equals[-1]
                    parsed = number_at(text, match.end() + last.end())
            if parsed:
                goal_overlap = len(goals & subject_words)
                conclusion = bool(re.match(r"\s*(?:so|therefore|thus|hence|in\s+conclusion)\b", subject, re.I))
                correct = bool(re.search(r"\bcorrect\b", subject, re.I))
                currency = bool(re.search(r"\bdollars?\b", question, re.I) and text[max(0, parsed[1]-1):parsed[1]] == "$")
                add(parsed, "quantity_statement", 60 + min(goal_overlap, 3) * 2 + (8 if conclusion else 0) + (6 if correct else 0) + (5 if currency else 0), match.start(), subject)
        for match in re.finditer(r"\b(?:there\s+(?:are|is)(?:\s+only)?|(?:can|must|needs?\s+to|would)\s+(?:make|form|buy|play|expect)|"
                                 r"(?:rounds?\s+(?:up|down)\s+to)|(?:round\s+(?:up|down)\s+to)|"
                                 r"(?:rounds?\s+(?:this\s+)?up\s+to)|(?:result\s+in)|(?:to\s+get)|(?:yields))\s+", text, re.I):
            if re.match(r"to\s+get", match[0], re.I):
                context = text[statement_start(text, match.start()):match.start()]
                if not re.search(r"\b(?:divide|multiply|subtract|add|round|simplify)\b", context, re.I):
                    continue
            parsed = number_at(text, match.end())
            if parsed is None:
                fragment = re.split(r"[.!?](?:\s|$)|\n\s*\n", text[match.end():], maxsplit=1)[0]
                equals = list(re.finditer(r"=\s*", fragment))
                if equals:
                    parsed = number_at(text, match.end() + equals[-1].end())
            add(parsed, "quantity_action", 70, match.start())
        # A written transformation can answer a compound expression without us
        # calculating it: the response itself must state the transformed value.
        if re.search(r"\\sqrt[^\n]*\^\{?2", question):
            for match in re.finditer(r"\bwhen\s+squared,?\s+it\s+(?:equals|is)\s+", text, re.I):
                add(number_at(text, match.end()), "stated_transformed_result", 75, match.start())
        for match in re.finditer(r"\bReply\s*:\s*", text, re.I):
            add(number_at(text, match.end()), "reply_declaration", 85, match.start())
        qtails = list(re.finditer(r"\b(?:what|find|determine|compute|calculate|evaluate|how\s+many)\b", question, re.I))
        qtarget = question[qtails[-1].start():] if qtails else question
        if re.search(r"\bhow\s+many\b|\bnumber\s+of\b", qtarget, re.I):
            qwords = {w.rstrip("s") for w in re.findall(r"[a-z]+", qtarget.lower())}
            for match in re.finditer(r"\b(?:has|have|contains)\s+(?:only\s+)?", text, re.I):
                parsed = number_at(text, match.end(), allow_words=True)
                if not parsed:
                    continue
                tail_words = re.findall(r"[a-z]+", text[parsed[2]:parsed[2] + 70].lower())[:3]
                overlap = ({w.rstrip("s") for w in tail_words} & qwords) - {"a", "the", "of", "and", "is", "to"}
                if overlap:
                    add(parsed, "requested_count_statement", 70, match.start())
        # Explicit arithmetic written as "then 1/4 of 48 is 12" is a result,
        # whereas 48 earlier in "6 * 8 = 48" is only an intermediate value.
        for match in re.finditer(r"\b(?:then|so|thus|therefore),?\s+(?:" + NUMBER + r"(?:\s*/\s*" + NUMBER + r")?)"
                                 r"\s+(?:of|times)\s+" + NUMBER + r"\s+is\s+", text, re.I):
            add(number_at(text, match.end()), "written_arithmetic_statement", 70, match.start())
        for match in re.finditer(r"\\boxed\s*\{([^{}]*)\}", text):
            parsed = number_at(text, match.start(1))
            if parsed and parsed[2] <= match.end(1):
                add(parsed, "boxed_answer", 90, match.start())
        # Equation-only results are a fallback, never an arbitrary last-number rule.
        if not hit_token_limit:
            equations = []
            for match in re.finditer(r"=\s*", text):
                parsed = number_at(text, match.end())
                if not parsed:
                    continue
                begin = statement_start(text, match.start())
                before = text[begin:match.start()]
                if not before.strip() and begin > 0:
                    before = text[max(0, begin - 200):match.start()]
                if re.search(r"\b(?:if|suppose|example|let\s+us\s+assume)\b", before, re.I):
                    continue
                equations.append((parsed, match, before))
            if equations:
                parsed, match, before = equations[-1]
                # Distinct solved variables are not interchangeable with a requested answer.
                variable_assignments = set(re.findall(r"\b([a-z])\s*=\s*[+\-]?\d", text))
                line = text[statement_start(text, match.start()):match.start()]
                ratios = re.findall(r"\b([a-z]\s*/\s*[a-z])\s*=", line + "=", re.I)
                q = re.sub(r"\\(?:d?frac)\{([a-z])\}\{([a-z])\}", r"\1/\2", question)
                requested_ratio = any(r.replace(" ", "") in q.replace(" ", "") for r in ratios)
                if len(variable_assignments) <= 1 or requested_ratio:
                    add(parsed, "written_equation_result", 40, match.start(), before)
    candidates.sort(key=lambda c: (c.start, c.end, c.rule))
    explicit = [c for c in candidates if c.rule == "final_declaration"]
    strong = [c for c in candidates if c.priority >= 75]
    conflict = len({c.value for c in explicit}) > 1
    if explicit:
        selected = explicit[-1]
        conflict |= any(c.value != selected.value for c in strong)
        if len({c.value for c in explicit}) > 1:
            selected, status = None, "conflicting_final_declarations"
        else:
            status = "parsed"
    elif unfinished_final:
        selected, status = None, "unfinished_final_answer"
    elif candidates:
        maximum = max(c.priority for c in candidates)
        best = [c for c in candidates if c.priority == maximum]
        values = {c.value for c in best}
        conflict |= len(values) > 1
        # Equal-priority competing claims are not resolved by reference correctness.
        if len(values) > 1:
            selected, status = None, "ambiguous_answer_statements"
        else:
            selected, status = best[-1], "parsed"
            conflict |= any(c.value != selected.value for c in strong)
    else:
        selected, status = None, "yes_no_only" if bare_yes_no else "no_supported_answer"
    if conflict:
        flags.append("conflicting_answer_statements")
    if hit_token_limit:
        flags.append("token_cap_reached")
    return ParseResult(selected.value if selected else None, status,
                       selected.rule if selected else None, selected.evidence if selected else None,
                       selected.start if selected else None, selected.end if selected else None,
                       conflict, flags, [asdict(c) for c in candidates])
