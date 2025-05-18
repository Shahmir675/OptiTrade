from pydantic import BaseModel, EmailStr
from typing import List, Optional
from decimal import Decimal
from datetime import datetime

class UserBalanceResponse(BaseModel):
    user_id: int
    cash_balance: float
    portfolio_value: float
    net_worth: float
    class Config:
        orm_mode = True

class BuyStockRequest(BaseModel):
    user_id: int
    symbol: str
    quantity: int
    order_type: str = "market"
    limit_price: Optional[Decimal] = None

class SellStockRequest(BaseModel):
    user_id: int
    symbol: str
    quantity: int
    order_type: str = "market"
    limit_price: Optional[Decimal] = None

class OrderResponse(BaseModel):
    order_id: int
    user_id: int
    symbol: str
    order_type: str
    price: float 
    quantity: int
    timestamp: datetime 
    order_status: bool
    filled_quantity: int
    remaining_quantity: int
    class Config:
        orm_mode = True
        json_encoders = {
            Decimal: lambda v: float(v)
        }


class LoginRequest(BaseModel):
    email: EmailStr
    u_pass: str

class SignupRequest(BaseModel):
    u_name: str
    email: EmailStr
    u_pass: str

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class WatchlistCreate(BaseModel):
    stock_symbol: str

class WatchlistResponse(BaseModel):
    id: int
    stock_symbol: str
    class Config:
        orm_mode = True

class PortfolioResponseItem(BaseModel):
    user_id: int
    symbol: str
    quantity: int
    average_price: float
    current_value: Optional[float]
    total_invested: Optional[float]
    class Config:
        orm_mode = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }


class PortfolioResponse(BaseModel):
    portfolio: List[PortfolioResponseItem]

class PortfolioHistoryItem(BaseModel):
    user_id: int
    symbol: str
    quantity: Optional[int] = None
    average_price: Optional[Decimal] = None
    current_value: Optional[Decimal] = None
    total_invested: Optional[Decimal] = None
    snapshot_date: datetime

    class Config:
        orm_mode = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None,
        }

class PortfolioHistoryResponse(BaseModel):
    history: List[PortfolioHistoryItem]

class TransactionResponse(BaseModel):
    id: int
    user_id: int
    symbol: str
    quantity: int
    order_type: str
    limit_price: Optional[float]
    transaction_type: str
    created_at: datetime
    price_per_share: float
    total_price: float

    class Config:
        orm_mode = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }

class UserResponse(BaseModel):
    id: int
    u_name: str
    email: EmailStr
    class Config:
        orm_mode = True