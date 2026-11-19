"""
Checks on the code-side guards in study/generate.py. No model needed.

Each case is a string the model actually produced while the practice
screens were being built, and each guard exists because a prompt
instruction did not hold. These are the decisions made in code, so they
are the ones that can be pinned exactly.

Run:  python -m tests.study_checks
"""

import sys

sys.path.insert(0, ".")

from study.generate import (
    _HEDGE, _INLINE_OPTIONS, _MENTIONS_SOURCE, _cite_material, _example_answer,
    _option, _standalone, _to_student, score_from_coverage,
)

# (input, expected after repair, should still be rejected as referring to the source)
STANDALONE = [
    ("According to the passage, what is the minimum speed?", "What is the minimum speed?", False),
    ("What are stereotypes according to the passage?", "What are stereotypes?", False),
    ("What is the role of a data mart as described in the passage?", "What is the role of a data mart?", False),
    ("In the text editor, which key saves a file?", "In the text editor, which key saves a file?", False),
    ("What is the first step mentioned in the passage for the presentation layer?",
     "What is the first step mentioned in the passage for the presentation layer?", True),
    ("What does OLTP stand for in the context of the passage?",
     "What does OLTP stand for in the context of the passage?", True),
    ("What is internal validity in the context of a study?",
     "What is internal validity in the context of a study?", False),
    ("What is required based on the given statements?", "What is required based on the given statements?", True),
]

HEDGED = [
    ("Most likely an abbreviation for 'Microphone System Browser' or a similar tool.", True),
    ("Probably refers to the exam browser.", True),
    ("Mettl System Browser, the software required to take the assessment.", False),
]

COACHING = [
    ("Hey there! Your answer is a good start, but let's dive deeper.", ""),
    ("Certainly! I would forecast demand from past sales.", "I would forecast demand from past sales."),
    ("Sure, business intelligence turns raw data into decisions.",
     "Business intelligence turns raw data into decisions."),
]


def main() -> int:
    failures = 0

    def check(ok: bool, label: str):
        nonlocal failures
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")

    print("question repair")
    for raw, want, still_refers in STANDALONE:
        got = _standalone(raw)
        check(got == want and bool(_MENTIONS_SOURCE.search(got)) == still_refers, raw)

    print("options")
    check(_option("C) Clicking F5 to refresh") == "Clicking F5 to refresh", "letter prefix stripped")
    check(_INLINE_OPTIONS.split("Which is prohibited?\n\nA) Using the mic\nB) Covering the webcam")[0].strip()
          == "Which is prohibited?", "inline option list cut from the question")

    print("hedged definitions")
    for text, hedged in HEDGED:
        check(bool(_HEDGE.search(text)) == hedged, text)

    print("wording for the student")
    check(_cite_material("The passage states, 'minimum 10 mbps'.") == "Your material states, 'minimum 10 mbps'.",
          "explanation names the student's material")
    check(_to_student("The student's answer lacks detail. It would help if the student was specific.")
          == "Your answer lacks detail. It would help if you were specific.", "feedback in the second person")
    for raw, want in COACHING:
        check(_example_answer(raw) == want, raw)

    print("interview score from per-point coverage")
    check(score_from_coverage(["missing"] * 4) == 1, "all missed -> 1")
    check(score_from_coverage(["partly"] * 4) == 3, "all partly -> 3")
    check(score_from_coverage(["covered", "covered", "covered", "missing"]) == 4, "3 of 4 covered -> 4")
    check(score_from_coverage(["covered"] * 4) == 5, "all covered -> 5")

    print(f"\n  {failures} failed" if failures else "\n  all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
