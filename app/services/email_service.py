from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from aiosmtplib import SMTP
from fastapi import HTTPException

from app.core.config import settings
from app.services.security_service import generate_otp_data


async def _send_email(to_emails: list[str], subject: str, body: str):
    msg = MIMEMultipart()
    msg["From"] = (
        f'"OptiTrade Support" <{settings.ALIAS_EMAIL or settings.GMAIL_ACCOUNT}>'
    )
    msg["To"] = ", ".join(to_emails)
    msg["Subject"] = subject
    msg["Reply-To"] = settings.ALIAS_EMAIL or settings.GMAIL_ACCOUNT
    msg.attach(MIMEText(body, "plain"))

    try:
        smtp = SMTP(
            hostname=settings.SMTP_SERVER, port=settings.SMTP_PORT, start_tls=True
        )
        await smtp.connect()
        await smtp.login(settings.GMAIL_ACCOUNT, settings.GMAIL_PASSWORD)
        await smtp.send_message(msg)
        await smtp.quit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


async def send_registration_otp(email: str, user_data: dict):
    otp = generate_otp_data(email, user_data)
    subject = "Your One-Time Password (OTP) for OptiTrade Registration"
    body = f"Dear User,\n\nYour One-Time Password (OTP) for OptiTrade registration is: {otp}. This OTP will expire in 5 minutes.\n\nThank you for choosing OptiTrade.\n\nBest regards,\nOptiTrade Support"
    await _send_email([email], subject, body)


async def send_password_reset_otp_email(email: str):
    otp = generate_otp_data(email)
    subject = "Your Password Reset OTP for OptiTrade"
    body = f"Dear User,\n\nYour password reset OTP for OptiTrade is: {otp}. This OTP will expire in 5 minutes.\n\nIf you did not request this, please ignore this email.\n\nBest regards,\nOptiTrade Support"
    await _send_email([email], subject, body)


async def send_feedback_confirmation(message: str):
    subject = "New Feedback Submitted"
    body = f"You received new feedback:\n\n{message}"
    to_emails = [
        settings.GMAIL_ACCOUNT,
        settings.FEEDBACK_CC_EMAIL_WEB,
        settings.FEEDBACK_CC_EMAIL_APP,
    ]
    await _send_email(to_emails, subject, body)
