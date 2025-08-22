from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, EmailStr


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
        json_encoders = {Decimal: lambda v: float(v)}


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
        json_encoders = {Decimal: lambda v: float(v) if v is not None else None}


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
        json_encoders = {Decimal: lambda v: float(v) if v is not None else None}


class UserResponse(BaseModel):
    id: int
    u_name: str
    email: EmailStr
    image_url: Optional[str]

    class Config:
        orm_mode = True


class FeedbackCreate(BaseModel):
    feedback_message: str


class FeedbackOut(BaseModel):
    feedback_id: int
    feedback_message: str
    created_at: datetime
    feedback_status: str

    class Config:
        orm_mode = True


# Analytics Models

class VaRRequest(BaseModel):
    user_id: int
    confidence_level: float = 0.95
    time_horizon_days: int = 1
    historical_days: int = 252


class VaRResponse(BaseModel):
    var_percentage: float
    var_dollar: float
    confidence_level: float
    time_horizon_days: int
    current_portfolio_value: float
    mean_daily_return: float
    daily_volatility: float
    observations_used: int


class MaxDrawdownRequest(BaseModel):
    user_id: int
    historical_days: int = 252


class MaxDrawdownResponse(BaseModel):
    max_drawdown_percentage: float
    peak_date: str
    trough_date: str
    recovery_date: Optional[str]
    peak_value: float
    trough_value: float
    current_value: float
    drawdown_duration_days: int
    observations_used: int


class SharpeRatioRequest(BaseModel):
    user_id: int
    risk_free_rate: float = 0.02
    historical_days: int = 252


class SharpeRatioResponse(BaseModel):
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    risk_free_rate: float
    excess_return: float
    observations_used: int


class BetaRequest(BaseModel):
    user_id: int
    market_symbol: str = "SPY"
    historical_days: int = 252


class BetaResponse(BaseModel):
    beta: float
    alpha_annualized: float
    r_squared: float
    correlation: float
    portfolio_volatility: float
    market_volatility: float
    market_symbol: str
    p_value: float
    observations_used: int


class ConcentrationRequest(BaseModel):
    user_id: int


class HoldingInfo(BaseModel):
    symbol: str
    value: float
    weight: float
    quantity: int


class ConcentrationResponse(BaseModel):
    herfindahl_hirschman_index: float
    effective_number_of_holdings: float
    actual_number_of_holdings: int
    concentration_level: str
    total_portfolio_value: float
    top_5_holdings_weight: float
    top_holdings: List[HoldingInfo]
    all_holdings: List[HoldingInfo]


class ComprehensiveAnalyticsRequest(BaseModel):
    user_id: int
    confidence_level: float = 0.95
    risk_free_rate: float = 0.02
    market_symbol: str = "SPY"
    historical_days: int = 252


class AnalyticsParameters(BaseModel):
    confidence_level: float
    risk_free_rate: float
    market_symbol: str
    historical_days: int


class ComprehensiveAnalyticsResponse(BaseModel):
    user_id: int
    calculation_date: str
    parameters: AnalyticsParameters
    var: Optional[VaRResponse]
    maximum_drawdown: Optional[MaxDrawdownResponse]
    sharpe_ratio: Optional[SharpeRatioResponse]
    beta: Optional[BetaResponse]
    concentration: Optional[ConcentrationResponse]
