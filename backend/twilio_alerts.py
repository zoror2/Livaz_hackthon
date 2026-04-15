"""
twilio_alerts.py
Automated voice call alerts using Twilio when flood risk is CRITICAL.
"""

import os
from twilio.rest import Client

# Twilio credentials (set via environment variables, fallback to defaults for dev)
ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID", "AC82a29f3d3bb8e4ee0ba0b06a1f9eb1c5")
AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "c3dab9cfa07d7bfd60f79aa0d64797bb")
TWILIO_FROM  = os.getenv("TWILIO_FROM_NUMBER", "+15077365262")

# Phone number to call (verified on trial account)
ALERT_TO     = "+919148896989"

client = Client(ACCOUNT_SID, AUTH_TOKEN)


def trigger_emergency_call(
    lat: float,
    lon: float,
    score: int,
    risk_level: str,
    rainfall: str,
    flood_pct: float,
    forecast_48h: float = 0.0,
):
    """
    Makes an automated voice call with TTS alert message.
    Called automatically when risk score >= 75 (CRITICAL).
    """
    message = (
        f"<speak>"
        f"<prosody rate='95%'>"
        f"This is an urgent early warning from the Advaya Climate Risk Engine. "
        f"Critical flood risk predicted at latitude {lat:.2f}, longitude {lon:.2f}. "
        f"Risk score: {score} out of 100. "
        f"Satellite analysis shows {flood_pct:.1f} percent flood coverage in this region. "
        f"Weather forecast predicts {forecast_48h:.0f} millimeters of rainfall in the next 48 hours. "
        f"<break time='500ms'/>"
        f"Please evacuate to higher ground immediately. Do not wait for the flood to arrive. "
        f"This is a predictive alert powered by NASA Prithvi EO 2.0 satellite intelligence."
        f"</prosody>"
        f"</speak>"
    )

    twiml = f'<Response><Say voice="alice" language="en-IN">{message}</Say></Response>'

    try:
        call = client.calls.create(
            twiml=twiml,
            to=ALERT_TO,
            from_=TWILIO_FROM,
        )
        print(f"[TWILIO] Emergency call initiated: SID={call.sid}")
        return {"status": "call_initiated", "call_sid": call.sid}
    except Exception as e:
        print(f"[TWILIO] Call failed: {e}")
        return {"status": "call_failed", "error": str(e)}


def send_shelter_sms(sms_body: str):
    """
    Sends shelter info via WhatsApp (Twilio Sandbox).
    Works on trial accounts for international delivery.
    """
    try:
        msg = client.messages.create(
            body=sms_body,
            to=f"whatsapp:{ALERT_TO}",
            from_="whatsapp:+14155238886",  # Twilio WhatsApp Sandbox
        )
        print(f"[TWILIO] WhatsApp alert sent: SID={msg.sid}")
        return {"status": "whatsapp_sent", "msg_sid": msg.sid}
    except Exception as e:
        print(f"[TWILIO] WhatsApp failed: {e}")
        return {"status": "whatsapp_failed", "error": str(e)}
