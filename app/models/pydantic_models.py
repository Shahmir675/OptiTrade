from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Union, Dict, Any
from enum import Enum

from pydantic import BaseModel, EmailStr, RootModel, ConfigDict


class UserBalanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    user_id: int
    cash_balance: float
    portfolio_value: float
    net_worth: float


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
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={Decimal: lambda v: float(v)}
    )
    
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
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    stock_symbol: str


class PortfolioResponseItem(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={Decimal: lambda v: float(v) if v is not None else None}
    )
    
    user_id: int
    symbol: str
    quantity: int
    average_price: float
    current_value: Optional[float]
    total_invested: Optional[float]


class PortfolioResponse(BaseModel):
    portfolio: List[PortfolioResponseItem]


class PortfolioHistoryItem(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={Decimal: lambda v: float(v) if v is not None else None}
    )
    
    user_id: int
    symbol: str
    quantity: Optional[int] = None
    average_price: Optional[Decimal] = None
    current_value: Optional[Decimal] = None
    total_invested: Optional[Decimal] = None
    snapshot_date: datetime


class PortfolioHistoryResponse(BaseModel):
    history: List[PortfolioHistoryItem]


class TransactionResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={Decimal: lambda v: float(v) if v is not None else None}
    )
    
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


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    u_name: str
    email: EmailStr
    image_url: Optional[str]


class FeedbackCreate(BaseModel):
    feedback_message: str


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    feedback_id: int
    feedback_message: str
    created_at: datetime
    feedback_status: str


# Analytics Models - Pydantic v2 Compatible

class AnalyticsErrorType(str, Enum):
    """Standardized error types for analytics calculations"""
    INSUFFICIENT_DATA = "insufficient_data"
    CALCULATION_ERROR = "calculation_error"
    MARKET_DATA_UNAVAILABLE = "market_data_unavailable"
    INVALID_PARAMETERS = "invalid_parameters"
    PORTFOLIO_EMPTY = "portfolio_empty"
    API_ERROR = "api_error"


class AnalyticsError(BaseModel):
    """Base error model for all analytics failures"""
    error_type: AnalyticsErrorType
    message: str
    details: Optional[str] = None


# Request Models
class VaRRequest(BaseModel):
    user_id: int
    confidence_level: float = 0.95
    time_horizon_days: int = 1
    historical_days: int = 252


class MaxDrawdownRequest(BaseModel):
    user_id: int
    historical_days: int = 252


class SharpeRatioRequest(BaseModel):
    user_id: int
    risk_free_rate: float = 0.02
    historical_days: int = 252


class BetaRequest(BaseModel):
    user_id: int
    market_symbol: str = "SPY"
    historical_days: int = 252


class ConcentrationRequest(BaseModel):
    user_id: int


class ComprehensiveAnalyticsRequest(BaseModel):
    user_id: int
    confidence_level: float = 0.95
    risk_free_rate: float = 0.02
    market_symbol: str = "SPY"
    historical_days: int = 252


# Success Response Models
class VaRSuccess(BaseModel):
    var_percentage: float
    var_dollar: float
    confidence_level: float
    time_horizon_days: int
    current_portfolio_value: float
    mean_daily_return: float
    daily_volatility: float
    observations_used: int


class MaxDrawdownSuccess(BaseModel):
    max_drawdown_percentage: float
    peak_date: str
    trough_date: str
    recovery_date: Optional[str]
    peak_value: float
    trough_value: float
    current_value: float
    drawdown_duration_days: int
    observations_used: int


class SharpeRatioSuccess(BaseModel):
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    risk_free_rate: float
    excess_return: float
    observations_used: int


class BetaSuccess(BaseModel):
    beta: float
    alpha_annualized: float
    r_squared: float
    correlation: float
    portfolio_volatility: float
    market_volatility: float
    market_symbol: str
    p_value: float
    observations_used: int


class HoldingInfo(BaseModel):
    symbol: str
    value: float
    weight: float
    quantity: int


class ConcentrationSuccess(BaseModel):
    herfindahl_hirschman_index: float
    effective_number_of_holdings: float
    actual_number_of_holdings: int
    concentration_level: str
    total_portfolio_value: float
    top_5_holdings_weight: float
    top_holdings: List[HoldingInfo]
    all_holdings: List[HoldingInfo]


# Pydantic v2 RootModel for Union Types
class VaRResponse(RootModel[Union[VaRSuccess, AnalyticsError]]):
    root: Union[VaRSuccess, AnalyticsError]


class MaxDrawdownResponse(RootModel[Union[MaxDrawdownSuccess, AnalyticsError]]):
    root: Union[MaxDrawdownSuccess, AnalyticsError]


class SharpeRatioResponse(RootModel[Union[SharpeRatioSuccess, AnalyticsError]]):
    root: Union[SharpeRatioSuccess, AnalyticsError]


class BetaResponse(RootModel[Union[BetaSuccess, AnalyticsError]]):
    root: Union[BetaSuccess, AnalyticsError]


class ConcentrationResponse(RootModel[Union[ConcentrationSuccess, AnalyticsError]]):
    root: Union[ConcentrationSuccess, AnalyticsError]


# Parameters and Main Response
class AnalyticsParameters(BaseModel):
    confidence_level: float
    risk_free_rate: float
    market_symbol: str
    historical_days: int


class ComprehensiveAnalyticsResponse(BaseModel):
    user_id: int
    calculation_date: str
    parameters: AnalyticsParameters
    var: Optional[VaRResponse] = None
    maximum_drawdown: Optional[MaxDrawdownResponse] = None
    sharpe_ratio: Optional[SharpeRatioResponse] = None
    beta: Optional[BetaResponse] = None
    concentration: Optional[ConcentrationResponse] = None


# Validation Response Models for the validation endpoint
class PortfolioSummary(BaseModel):
    has_holdings: bool
    total_holdings: int
    total_portfolio_value: float
    holdings: List[Dict[str, Any]]


class HistoricalDataInfo(BaseModel):
    has_history: bool
    history_records: int
    date_range: Dict[str, Optional[str]]


class ReturnsDataInfo(BaseModel):
    has_returns: bool
    return_observations: int
    sufficient_for_analytics: bool


class AnalyticsReadiness(BaseModel):
    can_calculate_var: bool
    can_calculate_sharpe: bool
    can_calculate_beta: bool
    can_calculate_drawdown: bool
    can_calculate_concentration: bool


class ValidationResponse(BaseModel):
    user_id: int
    validation_timestamp: str
    portfolio_summary: PortfolioSummary
    historical_data: HistoricalDataInfo
    returns_data: ReturnsDataInfo
    analytics_readiness: AnalyticsReadiness
    recommendations: List[str]


class ValidationErrorResponse(BaseModel):
    user_id: int
    validation_timestamp: str
    error: str
    status: str

class StopLossOrderRequest(BaseModel):
    user_id: int
    symbol: str
    quantity: int
    stop_price: Decimal
    order_type: str

class TakeProfitOrderRequest(BaseModel):
    user_id: int
    symbol: str
    quantity: int
    take_profit_price: Decimal
    order_type: str

class StopLossOrderResponse(BaseModel):
    order_id: int
    user_id: int
    symbol: str
    order_type: str
    stop_price: Decimal
    quantity: int
    timestamp: datetime
    order_status: bool
    filled_quantity: int
    remaining_quantity: int

class LimitOrderResponse(BaseModel):
    order_id: int
    user_id: int
    symbol: str
    order_type: str
    price: Decimal
    quantity: int
    timestamp: datetime
    order_status: bool
    filled_quantity: int
    remaining_quantity: int

class AllOrdersResponse(BaseModel):
    limit_orders: List[LimitOrderResponse]
    stop_loss_orders: List[StopLossOrderResponse] 