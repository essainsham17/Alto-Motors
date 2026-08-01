
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

MAILBOX_FINANCE = "essainsham987@gmail.com"
MAILBOX_GENERAL = "pes2pp2@gmail.com"

DEMO_CUSTOMER_EMAIL = "essa.21ai223@iceas.ac.in"


def _send(recipient: str, subject: str, body: str):
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not all([sender, password, recipient]):
        raise RuntimeError("Missing GMAIL_ADDRESS, GMAIL_APP_PASSWORD, or mailbox address in .env")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
    print(f"[mailer] sent to {recipient}: {subject}")


def _ticket_body(state: dict, reason: str) -> str:
    return f"""A customer inquiry needs your attention.

REASON: {reason}

CUSTOMER MESSAGE:
{state['inquiry']}

CLASSIFICATION:
Primary category   : {state['primary_category']}
Secondary category : {state['secondary_category'] or 'none'}
Confidence         : {state['confidence']}
Language           : {state['detected_language']}
Reasoning          : {state['reasoning']}

DRAFTED REPLY (review before sending — not sent to the customer automatically):
{state['draft_reply']}
"""


def forward_to_finance(state: dict):
    subject = f"[Alto Motors] Financing/Trade-In — {state['primary_category']}"
    _send(MAILBOX_FINANCE, subject, _ticket_body(state, "Financing or trade-in — needs a salesperson"))


def forward_to_general_review(state: dict, reason: str):
    subject = f"[Alto Motors] Needs review — {reason}"
    _send(MAILBOX_GENERAL, subject, _ticket_body(state, reason))


def send_test_drive_confirmation(state: dict):
    """
    Sends a booking confirmation to the customer's email.

    DEMO NOTE: uses a fixed stand-in address (DEMO_CUSTOMER_EMAIL) since this
    prototype has no multi-turn memory to actually collect the customer's real
    email mid-conversation. In production this would be the address the
    customer provided, tracked against their conversation thread_id.
    """
    subject = "Your Alto Motors test drive is confirmed"
    body = f"""Hi,

Thanks for booking a test drive with Alto Motors! Here's a summary:

YOUR MESSAGE:
{state['inquiry']}

CONFIRMATION:
{state['draft_reply']}

We look forward to seeing you.
— Alto Motors
"""
    _send(DEMO_CUSTOMER_EMAIL, subject, body)