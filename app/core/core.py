import os
import secrets
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    GMAIL_ACCOUNT: str = os.getenv("GMAIL_ACCOUNT")
    GMAIL_PASSWORD: str = os.getenv("GMAIL_PASSWORD")
    ALIAS_EMAIL: str = os.getenv("ALIAS_EMAIL")
    SMTP_SERVER: str = os.getenv("SMTP_SERVER")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    
    STOCKS_JSON_PATH: str = os.getenv("STOCKS_JSON_PATH", "stocks.json")
    NEWS_JSON_PATH: str = os.getenv("NEWS_JSON_PATH", "news.json")
    STATUS_CODES_DIR: str = os.getenv("STATUS_CODES_DIR", "app/static/status_codes/")
    
    JWT_SECRET: str = os.getenv("JWT_SECRET_KEY", secrets.token_hex(16))

    YFINANCE_PROXY_HTTP: str | None = os.getenv("YFINANCE_PROXY_HTTP")
    YFINANCE_PROXY_HTTPS: str | None = os.getenv("YFINANCE_PROXY_HTTPS")

    CURL_CFFI_IMPERSONATE: str = 'chrome'

settings = Settings()