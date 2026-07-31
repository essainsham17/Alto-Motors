import os
from typing import Optional, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

load_dotenv()

CONFIDENCE_THRESHOLD = 0.7   # below this -> human review, no auto-reply

GENERATIVE_CATEGORIES = ["Financing & EMI Query", "Trade-In Valuation"]


# ---------------------------------------------------------------------------
# STRUCTURED OUTPUT SCHEMA
# ---------------------------------------------------------------------------

class Classification(BaseModel):
    primary_category: Literal[
        "Test Drive Booking",
        "Financing & EMI Query",
        "Trade-In Valuation",
        "Other"
    ] = Field(description="The main intent of the customer's message.")

    secondary_category: Optional[
        Literal[
            "Test Drive Booking",
            "Financing & EMI Query",
            "Trade-In Valuation",
            "Other"
        ]
    ] = Field(
        default=None,
        description="A second intent, ONLY if the message clearly contains two "
                    "separate requests. Otherwise null.",
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in primary_category. Vague or incomplete "
                    "messages that give no real signal MUST score below 0.5.",
    )

    reasoning: str = Field(
        description="One short sentence explaining why you chose the primary_category category."
    )

    detected_language: str = Field(
        description="'English', 'Arabic', or 'Arabic-English mix'."
    )


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
- Reply in the SAME language the customer used. If they wrote in Arabic, reply in
  Arabic. If English, if mixed language reply in english, reply in English. Never mix languages in one reply.
- NEVER state any specific number: no prices, no EMI amounts, no interest rates,
  no trade-in values. You do not have access to that data. Acknowledge the request
  and say a specialist will follow up with exact figures.
- Keep it to 2-3 sentences. No greetings longer than one line.

Context:
Primary intent: {primary}
Secondary intent: {secondary}

Customer message:{inquiry}
Inquiry language: {detected_language}"""





# ---------------------------------------------------------------------------
# MOCK CALENDAR
# ---------------------------------------------------------------------------

def get_available_slots():
    return ["Saturday 10:00 AM", "Saturday 4:00 PM", "Monday 11:00 AM"]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------

def classify_node(state: GraphState) -> GraphState:
    CLASSIFY_PROMPT = """You are a triage assistant for Alto Motors, a car dealership \
selling two brands: Karva (mass-market sedans and SUVs) and Renzo (premium sedans \
and performance cars).

Classify the customer's inquiry into one of:
- Test Drive Booking
- Financing & EMI Query
- Trade-In Valuation
- Other

Rules:
1. MIXED INTENT: if the message contains two separate requests, set primary_category
to the main one and secondary_category to the other.
2. VAGUE MESSAGES: messages like "is this still available?" give no real signal.
Use "Other" as the primary_category and score BELOW 0.5 confidence. Never return
null for primary_category - always pick a category, and use "Other" when nothing fits.
3. LANGUAGE: messages may be in English, Arabic, or a mix. Classify them the same
way regardless, and record which in detected_language.
4. Be honest with confidence. A low score on an unclear message is the correct
output, not a failure."""

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


def human_review_node(state: GraphState) -> GraphState:
    VAGUE_REPLY_PROMPT = """You are writing a reply on behalf of Alto Motors, a car dealership \
selling two brands: Karva (mass-market sedans and SUVs) and Renzo (premium sedans \
and performance cars).

The customer's message was too vague to understand what they need.

Write a short, polite reply asking them to clarify what vehicle or service they are
asking about. Reply in the SAME language they used. Do NOT guess what they meant and
do NOT mention any specific car model. Keep it to 2 sentences.

Customer message:{inquiry}
Inquiry language: {detected_language}"""

    reply = llm.invoke(VAGUE_REPLY_PROMPT.format(inquiry=state["inquiry"], detected_language=state["detected_language"]))
    return {"draft_reply": reply.content, "human_review": True}


def test_drive_node(state: GraphState) -> GraphState:
    slots = get_available_slots()
    
    prompt = (
        REPLY_PROMPT.format(
            primary=state["primary_category"],
            secondary=state["secondary_category"] or "none",
            inquiry=state["inquiry"],
            detected_language=state["detected_language"]
        )
        + f"\n\nThese test drive slots are currently available: {', '.join(slots)}. "
          "Offer these specific slots in your reply."
    )
    reply = llm.invoke(prompt)
    return {"draft_reply": reply.content, "human_review": False}

#
def generative_category_node(state: GraphState) -> GraphState:
    prompt = REPLY_PROMPT.format(
        primary=state["primary_category"],
        secondary=state["secondary_category"] or "none",
        inquiry=state["inquiry"],
        detected_language=state["detected_language"]
    )
    reply = llm.invoke(prompt)
    return {"draft_reply": reply.content, "human_review": False}


# ---------------------------------------------------------------------------
# ROUTING FUNCTION
# ---------------------------------------------------------------------------

def route(state: GraphState) -> str:
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
workflow.add_node("human_review_node", human_review_node)
workflow.add_node("test_drive_node", test_drive_node)
workflow.add_node("generative_category_node", generative_category_node)

workflow.set_entry_point("classify_node")
workflow.add_conditional_edges("classify_node", route)
workflow.add_edge("human_review_node", END)
workflow.add_edge("test_drive_node", END)
workflow.add_edge("generative_category_node", END)

app_compiled = workflow.compile()


def triage(inquiry: str) -> GraphState:
    return app_compiled.invoke({
        "inquiry": inquiry,
        "primary_category": None,
        "secondary_category": None,
        "confidence": None,
        "reasoning": None,
        "detected_language": None,
        "draft_reply": None,
        "human_review": None,
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
        s = triage(msg)
        print("\n" + "=" * 70)
        print(f"MESSAGE   : {msg}")
        print(f"PRIMARY   : {s['primary_category']}")
        print(f"SECONDARY : {s['secondary_category']}")
        print(f"CONFIDENCE: {s['confidence']}")
        print(f"LANGUAGE  : {s['detected_language']}")
        print(f"REASONING : {s['reasoning']}")
        print(f"HUMAN?    : {s['human_review']}")
        print(f"REPLY     : {s['draft_reply']}")