import os
from typing import Optional, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from judge import judge_classification, log_agreement
from mailer import forward_to_finance, forward_to_general_review, send_test_drive_confirmation

load_dotenv()

CONFIDENCE_THRESHOLD = 0.7
GENERATIVE_CATEGORIES = ["Financing & EMI Query", "Trade-In Valuation"]


# ---------------------------------------------------------------------------
# STRUCTURED OUTPUT SCHEMA
# ---------------------------------------------------------------------------

class Classification(BaseModel):
    primary_category: Literal[
        "Test Drive Booking", "Financing & EMI Query", "Trade-In Valuation", "Other", "Greeting"
    ] = Field(description="The main intent of the customer's message.")
    secondary_category: Optional[
        Literal["Test Drive Booking", "Financing & EMI Query", "Trade-In Valuation", "Other", "Greeting"]
    ] = Field(default=None, description="A second intent, only if the message mixes two requests.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One short sentence explaining the primary_category choice.")
    detected_language: str = Field(description="'English', 'Arabic', or 'Arabic-English mix'.")


# ---------------------------------------------------------------------------
# GRAPH STATE
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    inquiry: str
    primary_category: Optional[str]
    secondary_category: Optional[str]
    confidence: Optional[float]
    reasoning: Optional[str]
    detected_language: Optional[str]
    draft_reply: Optional[str]
    human_review: Optional[bool]
    judge_agrees: Optional[bool]
    judge_reason: Optional[str]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Get a free key at console.groq.com")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


llm = get_llm()
structured_llm = llm.with_structured_output(Classification)


REPLY_PROMPT = """You are writing a reply on behalf of Alto Motors, a car dealership \
selling two brands: Karva (mass-market sedans and SUVs) and Renzo (premium sedans \
and performance cars).

Write a short, warm, professional reply to the customer message below.

CRITICAL RULES:
- Reply in the SAME language the customer used. If mixed language, reply in English.
  Never mix languages in one reply.
- NEVER state any specific number: no prices, no EMI amounts, no interest rates,
  no trade-in values. Acknowledge the request and say a specialist will follow up.
- Keep it to 2-3 sentences.

Context:
Primary intent: {primary}
Secondary intent: {secondary}

Customer message:{inquiry}
Inquiry language: {detected_language}"""


def get_available_slots():
    return ["Saturday 10:00 AM", "Saturday 4:00 PM", "Monday 11:00 AM"]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------

def classify_node(state: GraphState) -> GraphState:
    CLASSIFY_PROMPT = """You are a Inquiry Desk assistant for Alto Motors, a car dealership \
selling two brands: Karva (mass-market sedans and SUVs) and Renzo (premium sedans \
and performance cars).

Classify the customer's inquiry into one of:
- Test Drive Booking
- Financing & EMI Query
- Trade-In Valuation
- Other
- Greeting

Rules:
1. MIXED INTENT: if the message contains two separate requests, set primary_category
to the main one and secondary_category to the other.
2. VAGUE MESSAGES: messages like "is this still available?" give no real signal about
WHICH car or service, but ARE a real business inquiry. Use "Other" and score BELOW 0.5.
3. GREETINGS / CHITCHAT: messages like "hi", "thanks", "good morning" are not a business
inquiry at all — nothing to classify, nothing unclear about them. Use "Greeting" with
HIGH confidence (0.9+), not "Other". Do not confuse a friendly opener with a vague inquiry.
4. Never return null for primary_category - always pick a category.
5. LANGUAGE: messages may be in English, Arabic, or a mix. Classify them the same
way regardless, and record which in detected_language.
6. Be honest with confidence. A low score on a genuinely unclear inquiry is correct,
not a failure — but greetings should score high, since there's nothing ambiguous
about a hello."""

    result = structured_llm.invoke([
        {"role": "system", "content": CLASSIFY_PROMPT},
        {"role": "user", "content": state["inquiry"]},
    ])
    return {
        "primary_category": result.primary_category,
        "secondary_category": result.secondary_category,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "detected_language": result.detected_language,
    }


def greeting_node(state: GraphState) -> GraphState:
    """
    Handles greetings/chitchat ("hi", "thanks", "good morning"). No judge,
    no email, no dataset logging — there's no classification risk to review
    and no one who needs to be notified about a hello.
    """
    GREETING_PROMPT = """You are replying on behalf of Alto Motors, a car dealership.
The customer sent a greeting or casual message, not a business inquiry.
Reply warmly in ONE short sentence, in the SAME language they used, and invite
them to ask about test drives, financing, or trade-in valuation.

Customer message: {inquiry}"""
    reply = llm.invoke(GREETING_PROMPT.format(inquiry=state["inquiry"]))
    return {"draft_reply": reply.content, "human_review": False,
            "judge_agrees": None, "judge_reason": "Not reviewed — greeting/chitchat, no business content"}


def judge_node(state: GraphState) -> GraphState:
    """
    Runs on EVERY message during the 1-2 week trust-building window. Uses a
    different model (see judge.py) to independently review the classification
    BEFORE routing happens, so a disagreement can redirect the message to a
    human even if confidence looked high.

    This node is deleted from the graph entirely once the system has earned
    enough trust — it is not meant to run forever.
    """
    verdict = judge_classification(state)
    if verdict.agrees:
        log_agreement(state, verdict)
    return {"judge_agrees": verdict.agrees, "judge_reason": verdict.reason}


def human_review_node(state: GraphState) -> GraphState:
    VAGUE_REPLY_PROMPT = """You are writing a reply on behalf of Alto Motors, a car dealership \
selling two brands: Karva (mass-market sedans and SUVs) and Renzo (premium sedans \
and performance cars).

The customer's message was too vague to understand what they need, OR an internal
reviewer flagged it for a second look.

Write a short, polite reply asking them to clarify what vehicle or service they are
asking about. Reply in the SAME language they used. Do NOT guess what they meant and
do NOT mention any specific car model. Keep it to 2 sentences.

Customer message:{inquiry}
Inquiry language: {detected_language}"""

    reply = llm.invoke(VAGUE_REPLY_PROMPT.format(
        inquiry=state["inquiry"], detected_language=state["detected_language"]
    ))
    result = {"draft_reply": reply.content, "human_review": True}

    reason = "Low confidence" if state["confidence"] < CONFIDENCE_THRESHOLD else \
              f"Judge disagreed: {state['judge_reason']}"
    forward_to_general_review({**state, **result}, reason=reason)

    return result


def test_drive_node(state: GraphState) -> GraphState:
    slots = get_available_slots()
    prompt = (
        REPLY_PROMPT.format(
            primary=state["primary_category"],
            secondary=state["secondary_category"] or "none",
            inquiry=state["inquiry"],
            detected_language=state["detected_language"],
        )
        + f"\n\nThese test drive slots are currently available: {', '.join(slots)}. "
          "Offer these specific slots in your reply."
    )
    reply = llm.invoke(prompt)
    result = {"draft_reply": reply.content, "human_review": False}
    send_test_drive_confirmation({**state, **result})
    return result


def generative_category_node(state: GraphState) -> GraphState:
    prompt = REPLY_PROMPT.format(
        primary=state["primary_category"],
        secondary=state["secondary_category"] or "none",
        inquiry=state["inquiry"],
        detected_language=state["detected_language"],
    )
    reply = llm.invoke(prompt)
    result = {"draft_reply": reply.content, "human_review": False}
    forward_to_finance({**state, **result})
    return result


# ---------------------------------------------------------------------------
# ROUTING FUNCTION
# ---------------------------------------------------------------------------

def route_after_classify(state: GraphState) -> str:
    if state["primary_category"] == "Greeting":
        return "greeting_node"
    return "judge_node"


def route(state: GraphState) -> str:
    if not state["judge_agrees"]:
        return "human_review_node"
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "human_review_node"
    if state["primary_category"] == "Test Drive Booking":
        return "test_drive_node"
    if state["primary_category"] in GENERATIVE_CATEGORIES:
        return "generative_category_node"
    return "human_review_node"


# ---------------------------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("classify_node", classify_node)
workflow.add_node("greeting_node", greeting_node)
workflow.add_node("judge_node", judge_node)
workflow.add_node("human_review_node", human_review_node)
workflow.add_node("test_drive_node", test_drive_node)
workflow.add_node("generative_category_node", generative_category_node)

workflow.set_entry_point("classify_node")
workflow.add_conditional_edges("classify_node", route_after_classify)
workflow.add_conditional_edges("judge_node", route)
workflow.add_edge("greeting_node", END)
workflow.add_edge("human_review_node", END)
workflow.add_edge("test_drive_node", END)
workflow.add_edge("generative_category_node", END)

app_compiled = workflow.compile()


def Inquiry_Desk(inquiry: str) -> GraphState:
    return app_compiled.invoke({
        "inquiry": inquiry,
        "primary_category": None,
        "secondary_category": None,
        "confidence": None,
        "reasoning": None,
        "detected_language": None,
        "draft_reply": None,
        "human_review": None,
        "judge_agrees": None,
        "judge_reason": None,
    })


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

TEST_MESSAGES = [
    "Hi, I'd like to book a test drive for the Renzo GT this weekend.",
    "I want to trade in my old Karva sedan and also check financing for a new Renzo.",
    "is this still available?",
    "ابغى اسوق تجربة لسيارة رينزو الجديدة",
    "Salam, I need EMI options for Karva w kaman badi a3rif value of my old car.",
]

if __name__ == "__main__":
    for msg in TEST_MESSAGES:
        s = Inquiry_Desk(msg)
        print("\n" + "=" * 70)
        print(f"MESSAGE     : {msg}")
        print(f"PRIMARY     : {s['primary_category']}")
        print(f"SECONDARY   : {s['secondary_category']}")
        print(f"CONFIDENCE  : {s['confidence']}")
        print(f"JUDGE AGREES: {s['judge_agrees']} ({s['judge_reason']})")
        print(f"HUMAN?      : {s['human_review']}")
        print(f"REPLY       : {s['draft_reply']}")