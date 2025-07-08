# sending sms alerts

import os
from twilio.rest import Client

def send_sms(message: str, to: str) -> bool:
    """Send SMS via Twilio. Returns True if success, False otherwise."""
    sid = os.getenv("TWILIO_SID")
    token = os.getenv("TWILIO_TOKEN")
    from_number = os.getenv("TWILIO_FROM")
    if not sid or not token or not from_number:
        raise EnvironmentError("Twilio credentials not set in environment")
    if not to.startswith('+'):
        raise ValueError("Phone number must start with '+' and country code")
    client = Client(sid, token)
    client.messages.create(body=message, from_=from_number, to=to)
    return True