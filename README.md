# Alto Motors — Inquiry Desk

Round 2 task — Strategic Automation Associate, Legend Motors.

Alto Motors' showroom gets 80–100 customer inquiries a day (WhatsApp, web forms,
walk-ins), asking about test drives, financing/EMI, or trade-in valuation. A
coordinator manually reads and routes every one of them — roughly 5 hours a day —
and delays or misroutes sometimes cost a sale.

This prototype automates the classification and first-response step, while never
letting the AI invent information it doesn't actually have.

## What it does

1. **Classifies** an inquiry into Test Drive Booking / Financing & EMI Query /
   Trade-In Valuation / Other / Greeting — handling **mixed intent** (two
   requests in one message) with a primary + secondary category.
2. **Scores its own confidence.** Vague or unclear messages score low on
   purpose, rather than guessing.
3. **A second, independent model reviews every classification** before
   anything is routed — deliberately a different model from the one that
   classified it, so it isn't just checking its own work. Disagreement sends
   the message to a coordinator regardless of how confident the first model was.
4. **Routes based on both signals** (confidence + reviewer agreement):
   - **Greeting / chitchat** — answered directly in chat. No reviewer, no
     email, no ticket. Nothing to review.
   - **Test Drive Booking** — checked against a mock calendar, replies with
     real available slots (a lookup, not a guess), and a confirmation email
     is sent.
   - **Financing & EMI / Trade-In Valuation** — the AI never states a price,
     rate, or valuation. It acknowledges only, and the full ticket is emailed
     to the finance/sales mailbox.
   - **Low confidence, or the reviewer disagrees** — flagged for a
     coordinator, emailed to the general review mailbox, with a draft
     prepared but never auto-sent.
5. **Handles Arabic, English, and Arabic-English mixed messages**, and always
   replies in the same language the customer used — never a mixed-language
   reply.
6. **Every agreed classification is logged** to a growing dataset
   (`dataset.jsonl`). Once that dataset is large enough, it's the training
   data for a smaller, cheaper classifier — at which point the reviewing
   model comes out of the live path.

## Architecture

```
Customer inquiry
      │
Classify (1 LLM call, structured Pydantic output)
      │
  Greeting? ──────────────────────────► Reply directly. No review, no email.
      │ no
      ▼
Independent reviewer (a different model checks the classification)
      │
  Reviewer disagrees, OR confidence < threshold?
      │
      ├── yes ─────────────────────────► Coordinator mailbox, draft prepared,
      │                                   nothing auto-sent
      └── no, by category
              ├── Test Drive Booking ──► Mock calendar check, replies with
              │                          slots, confirmation emailed
              └── Financing / Trade-In ► Acknowledge only, salesperson
                                          mailbox emailed with full ticket
```

**Why LangGraph** over Flowise/n8n/CrewAI/AutoGen: this is one well-defined
decision (classify → review → escalate or handle), not an open-ended
multi-agent negotiation. A code-first tool gives explicit, auditable control
over that branch point, which matters since a wrong auto-reply reaching a
real customer is worse than a slower one.

**Why a second model as reviewer, not the same model checking itself:** a
model asked to verify its own answer tends to defend it rather than
genuinely re-examine it. An independent model is far more likely to catch a
confidently-wrong classification that a confidence score alone would miss.

## Observability

Every call — classify, review, and reply — is traced through **LangSmith**,
so any message's full path through the graph can be inspected: what was
classified, what the reviewer said, why, and what was sent. This also gives
real, measured token usage and cost per message rather than estimates.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=alto-motors-inquiry-desk
GMAIL_ADDRESS=your_sending_account@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
```

- Free Groq key: [console.groq.com](https://console.groq.com/keys)
- Free LangSmith key: [smith.langchain.com](https://smith.langchain.com)
- Gmail App Password requires 2-Step Verification enabled on the sending
  account (Google Account → Security → App Passwords)

Mailbox addresses (finance, general review) and the demo customer email are
set directly in `mailer.py` — they aren't secrets, so they don't need to be
in `.env`.

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
webhook would call in production — no redesign needed to connect it. The
same reasoning applies to a website form submitting to this endpoint, or a
coordinator manually entering a walk-in inquiry through this same page.

**Standalone judge batch check** (optional — outside the live graph, for a
one-off review pass over a set of messages):
```bash
python judge.py
```

## Known limitations (by design, for a 1–2 day prototype)

- The booking calendar is a hardcoded mock — in production this would call
  Alto Motors' real booking system via an API or MCP connector.
- No real financing/trade-in data source is connected — intentionally, since
  the AI should never state a number it can't verify.
- No conversation memory — each message is triaged independently. A
  production version tracking a customer across multiple messages (e.g. to
  collect their email for a booking confirmation) would need a `thread_id`
  per conversation and a checkpointer, which this deliberately doesn't include.
- Email delivery uses direct Gmail SMTP, which is fine for a demo but not
  how this would be built for production — a transactional email service
  (SendGrid, AWS SES) or routing tickets through Slack/a CRM would be the
  real approach, and would avoid spam-filtering inconsistencies personal
  Gmail accounts sometimes trigger.