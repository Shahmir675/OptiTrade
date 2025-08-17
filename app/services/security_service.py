from datetime import datetime, timedelta

import bcrypt
import jwt
import pyotp

from app.core.config import settings
from app.utils.timezones import PAKISTAN_TIMEZONE, UTC

otp_storage = {}


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")
    return encoded_jwt


def generate_otp_data(email: str, user_data: dict = None):
    totp = pyotp.TOTP(pyotp.random_base32())
    otp = totp.now()
    utc_time = datetime.now(UTC)
    pst_time = utc_time.astimezone(PAKISTAN_TIMEZONE)

    otp_entry = {"otp": otp, "expires_at": pst_time + timedelta(minutes=5)}
    if user_data:
        otp_entry["user_data"] = user_data

    otp_storage[email] = otp_entry
    return otp


def verify_otp(email: str, otp_code: str) -> tuple[bool, str | None, dict | None]:
    if email not in otp_storage or otp_storage[email]["otp"] != otp_code:
        return False, "Invalid OTP.", None

    utc_time = datetime.now(UTC)
    pst_time = utc_time.astimezone(PAKISTAN_TIMEZONE)

    if pst_time > otp_storage[email]["expires_at"]:
        del otp_storage[email]
        return False, "OTP expired.", None

    user_data_to_return = otp_storage[email].get("user_data")
    return True, "OTP verified.", user_data_to_return


def clear_otp(email: str):
    if email in otp_storage:
        del otp_storage[email]
