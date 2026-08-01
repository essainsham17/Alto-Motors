
import os
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"

DATASET_PATH = "dataset.jsonl"


class JudgeVerdict(BaseModel):
    agrees: bool = Field(description="True if the classification looks correct.")
    reason: str = Field(description="One short sentence explaining the verdict.")


def get_judge_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    base = ChatGroq(model=JUDGE_MODEL, temperature=0, api_key=api_key)
    return base.with_structured_output(JudgeVerdict)


JUDGE_PROMPT = """You are an independent quality reviewer for Alto Motors' customer
inquiry classifier. You did NOT produce this classification — review it with
fresh eyes.

Categories: Test Drive Booking, Financing & EMI Query, Trade-In Valuation, Other.
A message can have a primary AND secondary category if it mixes two requests.
Vague messages (no real signal) should have primary_category = "Other" and low confidence.

Customer message:
{inquiry}

The classifier decided:
- Primary category: {primary}
- Secondary category: {secondary}
- Confidence: {confidence}
- Detected language: {language}
- Its stated reasoning: {reasoning}

Independently judge: does this classification look correct? Consider whether the
primary/secondary split makes sense, whether the confidence is honest, and
whether the reasoning actually supports the category chosen."""


def judge_classification(state: dict) -> JudgeVerdict:
    """Runs the judge on one classification. Returns a JudgeVerdict."""
    judge_llm = get_judge_llm()
    prompt = JUDGE_PROMPT.format(
        inquiry=state["inquiry"],
        primary=state["primary_category"],
        secondary=state["secondary_category"] or "none",
        confidence=state["confidence"],
        language=state["detected_language"],
        reasoning=state["reasoning"],
    )
    return judge_llm.invoke([{"role": "user", "content": prompt}])


def log_agreement(state: dict, verdict: JudgeVerdict):
    """Appends a verified-correct classification to the growing dataset."""
    record = {
        "inquiry": state["inquiry"],
        "primary_category": state["primary_category"],
        "secondary_category": state["secondary_category"],
        "confidence": state["confidence"],
        "detected_language": state["detected_language"],
        "reasoning": state["reasoning"],
        "judge_reason": verdict.reason,
    }
    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    from app import Inquiry_Desk, TEST_MESSAGES

    agreed, flagged = 0, 0
    for msg in TEST_MESSAGES:
        state = Inquiry_Desk(msg)
        verdict = judge_classification(state)
        print("\n" + "=" * 70)
        print(f"MESSAGE    : {msg}")
        print(f"CLASSIFIED : {state['primary_category']} "
              f"(secondary: {state['secondary_category']}, conf: {state['confidence']})")
        if verdict.agrees:
            agreed += 1
            log_agreement(state, verdict)
            print(f"JUDGE      : ✅ AGREE — {verdict.reason}")
        else:
            flagged += 1
            print(f"JUDGE      : 🚩 DISAGREE — {verdict.reason}")
    print(f"\nSUMMARY: {agreed} agreed / {flagged} flagged")