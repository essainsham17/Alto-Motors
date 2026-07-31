# Alto Motors — Inquiry Triage Prototype

Round 2 task — Strategic Automation Associate, Legend Motors.

Alto Motors' showroom gets 80–100 customer inquiries a day (WhatsApp, web forms,
walk-ins), asking about test drives, financing/EMI, or trade-in valuation. A
coordinator manually reads and routes every one of them — roughly 5 hours a day —
and delays or misroutes sometimes cost a sale.

This prototype automates the classification and first-response step, while never
letting the AI invent information it doesn't actually have.

## What it does

1. **Classifies** an inquiry into Test Drive Booking / Financing & EMI Query /
   Trade-In Valuation / Other — handling **mixed intent** (two requests in one
   message) with a primary + secondary category.
2. **Scores its own confidence.** Vague or unclear messages (`"is this still
   available?"`) score low on purpose, rather than guessing.
3. **Routes on that confidence:**
   - Below threshold (0.7) → flagged for a human, no auto-reply sent.
   - Test Drive Booking → checked against a mock calendar, replies with real
     available slots (a lookup, not a guess).
   - Financing & EMI / Trade-In Valuation → drafts an acknowledgment only.
     **The AI never states a price, EMI rate, or trade-in value** — those are
     negotiable numbers it has no source of truth for, so it hands off to a
     salesperson instead of risking a hallucinated figure.
4. **Handles Arabic, English, and Arabic-English mixed messages**, and always
   replies in the same language the customer used (never a mixed-language reply).

## Architecture

A LangGraph state graph with 4 nodes:

```
Customer inquiry
      │
Classify + score confidence   (1 LLM call, structured Pydantic output)
      │
 conditional edge (confidence ≥ 0.7 ?)
      │
      ├── below threshold ──────────────► human_review_node
      │                                   (flag only, no auto-reply)
      │
      └── above threshold, by category
              ├── Test Drive Booking ───► test_drive_node
              │                           (checks mock calendar, offers slots)
              └── Financing / Trade-In ─► generative_category_node
                                          (acknowledges only, routes to a
                                           salesperson — never states numbers)
```

**Why LangGraph** over Flowise/n8n/CrewAI: this is one well-defined decision
(classify → escalate or handle), not an open-ended multi-agent negotiation.
A code-first tool gives explicit, auditable control over that one branch point,
which matters since a wrong auto-reply reaching a real customer is worse than a
slower one.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com/keys),
then create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

## Run it

**Terminal demo** (runs 5 built-in test messages covering clean, mixed-intent,
vague, Arabic, and Arabic-English mixed cases):
```bash
python app.py
```

**Web interface + API:**
```bash
python -m uvicorn server:app --reload --port 8000
```
Then open `http://localhost:8000` in your browser. The same endpoint
(`POST /api/triage`, JSON in → JSON out) is what a WhatsApp Business API
webhook would call in production — no redesign needed to connect it.

## Known limitations (by design, for a 1–2 day prototype)

- The calendar is a hardcoded mock — in production this would call Alto
  Motors' real booking system via an API or MCP connector.
- No real financing/trade-in data source is connected — intentionally, since
  the AI should never state a number it can't verify.
- No persistent logging yet. In production, high-confidence classifications
  would be logged to build a labeled dataset for a future, cheaper classifier,
  and an LLM-as-judge pass would run against live traffic for 1–2 weeks after
  launch before removing human oversight.