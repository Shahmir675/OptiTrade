"""
Portfolio Analytics Router

This module provides API endpoints for portfolio analytics calculations including:
- Value at Risk (VaR)
- Maximum Drawdown (MDD)
- Sharpe Ratio
- Beta
- Portfolio Concentration
- Comprehensive Analytics Summary

Updated with professional error handling using Union types for success/error states.
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.models import pydantic_models as pyd_models
from app.services.analytics_service import AnalyticsService

from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Portfolio Analytics"])


@router.post("/var", response_model=pyd_models.VaRResponse)
async def calculate_var_endpoint(
    request: pyd_models.VaRRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Value at Risk (VaR) for a user's portfolio

    VaR estimates the potential loss in portfolio value over a specific time period
    at a given confidence level using historical simulation method.

    - **user_id**: User ID for portfolio analysis
    - **confidence_level**: Confidence level (0.95 = 95%, 0.99 = 99%)
    - **time_horizon_days**: Time horizon in days (typically 1 day)
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful VaR calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_var(
        user_id=request.user_id,
        db=db,
        confidence_level=request.confidence_level,
        time_horizon=request.time_horizon_days,
        days=request.historical_days
    )
    return result


@router.get("/var/{user_id}", response_model=pyd_models.VaRResponse)
async def calculate_var_get_endpoint(
    user_id: int,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    historical_days: int = 252,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Value at Risk (VaR) for a user's portfolio (GET method)

    VaR estimates the potential loss in portfolio value over a specific time period
    at a given confidence level using historical simulation method.

    - **user_id**: User ID for portfolio analysis
    - **confidence_level**: Confidence level (0.95 = 95%, 0.99 = 99%)
    - **time_horizon_days**: Time horizon in days (typically 1 day)
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful VaR calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_var(
        user_id=user_id,
        db=db,
        confidence_level=confidence_level,
        time_horizon=time_horizon_days,
        days=historical_days
    )
    return result


@router.post("/max-drawdown", response_model=pyd_models.MaxDrawdownResponse)
async def calculate_max_drawdown_endpoint(
    request: pyd_models.MaxDrawdownRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Maximum Drawdown (MDD) for a user's portfolio

    MDD measures the largest peak-to-trough decline in portfolio value,
    indicating the worst-case scenario for portfolio losses.

    - **user_id**: User ID for portfolio analysis
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful MDD calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_maximum_drawdown(
        user_id=request.user_id,
        db=db,
        days=request.historical_days
    )
    return result


@router.get("/max-drawdown/{user_id}", response_model=pyd_models.MaxDrawdownResponse)
async def calculate_max_drawdown_get_endpoint(
    user_id: int,
    historical_days: int = 252,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Maximum Drawdown (MDD) for a user's portfolio (GET method)

    MDD measures the largest peak-to-trough decline in portfolio value,
    indicating the worst-case scenario for portfolio losses.

    - **user_id**: User ID for portfolio analysis
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful MDD calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_maximum_drawdown(
        user_id=user_id,
        db=db,
        days=historical_days
    )
    return result


@router.post("/sharpe-ratio", response_model=pyd_models.SharpeRatioResponse)
async def calculate_sharpe_ratio_endpoint(
    request: pyd_models.SharpeRatioRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Sharpe Ratio for a user's portfolio

    Sharpe Ratio measures risk-adjusted returns by comparing portfolio returns
    to the risk-free rate relative to portfolio volatility.

    - **user_id**: User ID for portfolio analysis
    - **risk_free_rate**: Annual risk-free rate (default 2%)
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful Sharpe ratio calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_sharpe_ratio(
        user_id=request.user_id,
        db=db,
        risk_free_rate=request.risk_free_rate,
        days=request.historical_days
    )
    return result


@router.get("/sharpe-ratio/{user_id}", response_model=pyd_models.SharpeRatioResponse)
async def calculate_sharpe_ratio_get_endpoint(
    user_id: int,
    risk_free_rate: float = 0.02,
    historical_days: int = 252,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Sharpe Ratio for a user's portfolio (GET method)

    Sharpe Ratio measures risk-adjusted returns by comparing portfolio returns
    to the risk-free rate relative to portfolio volatility.

    - **user_id**: User ID for portfolio analysis
    - **risk_free_rate**: Annual risk-free rate (default 2%)
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful Sharpe ratio calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_sharpe_ratio(
        user_id=user_id,
        db=db,
        risk_free_rate=risk_free_rate,
        days=historical_days
    )
    return result


@router.post("/beta", response_model=pyd_models.BetaResponse)
async def calculate_beta_endpoint(
    request: pyd_models.BetaRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Beta for a user's portfolio relative to a market index

    Beta measures portfolio sensitivity to market movements. A beta of 1.0
    indicates the portfolio moves with the market, >1.0 is more volatile,
    and <1.0 is less volatile than the market.

    - **user_id**: User ID for portfolio analysis
    - **market_symbol**: Market index symbol (default SPY)
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful beta calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_beta(
        user_id=request.user_id,
        db=db,
        market_symbol=request.market_symbol,
        days=request.historical_days
    )
    return result


@router.get("/beta/{user_id}", response_model=pyd_models.BetaResponse)
async def calculate_beta_get_endpoint(
    user_id: int,
    market_symbol: str = "SPY",
    historical_days: int = 252,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Beta for a user's portfolio relative to a market index (GET method)

    Beta measures portfolio sensitivity to market movements. A beta of 1.0
    indicates the portfolio moves with the market, >1.0 is more volatile,
    and <1.0 is less volatile than the market.

    - **user_id**: User ID for portfolio analysis
    - **market_symbol**: Market index symbol (default SPY)
    - **historical_days**: Number of historical days to use for calculation

    Returns either successful beta calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_beta(
        user_id=user_id,
        db=db,
        market_symbol=market_symbol,
        days=historical_days
    )
    return result


@router.post("/concentration", response_model=pyd_models.ConcentrationResponse)
async def calculate_concentration_endpoint(
    request: pyd_models.ConcentrationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Portfolio Concentration using Herfindahl-Hirschman Index (HHI)

    HHI measures portfolio diversification. Lower values indicate better
    diversification, while higher values suggest concentration risk.

    - **user_id**: User ID for portfolio analysis

    HHI Interpretation:
    - < 0.15: Low concentration (well diversified)
    - 0.15-0.25: Moderate concentration
    - > 0.25: High concentration (concentrated portfolio)

    Returns either successful concentration calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_portfolio_concentration(
        user_id=request.user_id,
        db=db
    )
    return result


@router.get("/concentration/{user_id}", response_model=pyd_models.ConcentrationResponse)
async def calculate_concentration_get_endpoint(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate Portfolio Concentration using Herfindahl-Hirschman Index (HHI) (GET method)

    HHI measures portfolio diversification. Lower values indicate better
    diversification, while higher values suggest concentration risk.

    - **user_id**: User ID for portfolio analysis

    HHI Interpretation:
    - < 0.15: Low concentration (well diversified)
    - 0.15-0.25: Moderate concentration
    - > 0.25: High concentration (concentrated portfolio)

    Returns either successful concentration calculation or detailed error information.
    """
    result = await AnalyticsService.calculate_portfolio_concentration(
        user_id=user_id,
        db=db
    )
    return result


@router.post("/comprehensive", response_model=pyd_models.ComprehensiveAnalyticsResponse)
async def get_comprehensive_analytics_endpoint(
    request: pyd_models.ComprehensiveAnalyticsRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive portfolio analytics including all metrics

    This endpoint calculates all available analytics metrics in a single request:
    - Value at Risk (VaR)
    - Maximum Drawdown (MDD)
    - Sharpe Ratio
    - Beta
    - Portfolio Concentration

    Each metric can succeed or fail independently, providing detailed error
    information for failed calculations while still returning successful ones.

    - **user_id**: User ID for portfolio analysis
    - **confidence_level**: VaR confidence level (default 95%)
    - **risk_free_rate**: Annual risk-free rate (default 2%)
    - **market_symbol**: Market index for beta calculation (default SPY)
    - **historical_days**: Number of historical days to use

    Returns comprehensive analytics with success/error state for each metric.
    """
    result = await AnalyticsService.get_comprehensive_analytics(
        user_id=request.user_id,
        db=db,
        confidence_level=request.confidence_level,
        risk_free_rate=request.risk_free_rate,
        market_symbol=request.market_symbol,
        days=request.historical_days
    )
    return result


@router.get("/comprehensive/{user_id}", response_model=pyd_models.ComprehensiveAnalyticsResponse)
async def get_comprehensive_analytics_get_endpoint(
    user_id: int,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.02,
    market_symbol: str = "SPY",
    historical_days: int = 252,
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive portfolio analytics including all metrics (GET method)

    This endpoint calculates all available analytics metrics in a single request:
    - Value at Risk (VaR)
    - Maximum Drawdown (MDD)
    - Sharpe Ratio
    - Beta
    - Portfolio Concentration

    Each metric can succeed or fail independently, providing detailed error
    information for failed calculations while still returning successful ones.

    - **user_id**: User ID for portfolio analysis
    - **confidence_level**: VaR confidence level (default 95%)
    - **risk_free_rate**: Annual risk-free rate (default 2%)
    - **market_symbol**: Market index for beta calculation (default SPY)
    - **historical_days**: Number of historical days to use

    Returns comprehensive analytics with success/error state for each metric.
    """
    result = await AnalyticsService.get_comprehensive_analytics(
        user_id=user_id,
        db=db,
        confidence_level=confidence_level,
        risk_free_rate=risk_free_rate,
        market_symbol=market_symbol,
        days=historical_days
    )
    return result


@router.get("/health")
async def analytics_health_check() -> Dict[str, str]:
    """
    Health check endpoint for analytics service
    """
    return {
        "status": "healthy",
        "service": "portfolio_analytics",
        "version": "2.0.0",  # Updated version to reflect professional error handling
        "features": [
            "professional_error_handling",
            "union_response_types",
            "detailed_error_messages",
            "graceful_failure_handling"
        ]
    }


# New endpoint for error validation and debugging
# Replace the validation endpoint in your analytics router with this fixed version:

from datetime import datetime
from typing import List, Dict, Any, Union

# Add these new imports to your router
from app.models.pydantic_models import (
    ValidationResponse, 
    ValidationErrorResponse,
    PortfolioSummary,
    HistoricalDataInfo, 
    ReturnsDataInfo,
    AnalyticsReadiness
)

# Fixed validation endpoint
@router.get("/validate/{user_id}", response_model=Union[ValidationResponse, ValidationErrorResponse])
async def validate_user_portfolio(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Validate user portfolio data and return diagnostic information
    
    This endpoint helps debug analytics calculation issues by providing
    detailed information about the user's portfolio state.
    
    - **user_id**: User ID to validate
    
    Returns diagnostic information about portfolio state and data availability.
    """
    try:
        # Get basic portfolio data
        portfolio_data = await AnalyticsService.get_portfolio_data(user_id, db)
        
        # Get portfolio history
        history_df = await AnalyticsService.get_portfolio_history(user_id, db, 252)
        
        # Get returns
        returns = await AnalyticsService.calculate_portfolio_returns(user_id, db, 252)
        
        total_value = sum(item["current_value"] for item in portfolio_data) if portfolio_data else 0
        
        return ValidationResponse(
            user_id=user_id,
            validation_timestamp=datetime.now().isoformat(),
            portfolio_summary=PortfolioSummary(
                has_holdings=len(portfolio_data) > 0,
                total_holdings=len(portfolio_data),
                total_portfolio_value=total_value,
                holdings=portfolio_data
            ),
            historical_data=HistoricalDataInfo(
                has_history=not history_df.empty,
                history_records=len(history_df) if not history_df.empty else 0,
                date_range={
                    "start": str(history_df['date'].min()) if not history_df.empty else None,
                    "end": str(history_df['date'].max()) if not history_df.empty else None
                }
            ),
            returns_data=ReturnsDataInfo(
                has_returns=not returns.empty,
                return_observations=len(returns) if not returns.empty else 0,
                sufficient_for_analytics=len(returns) >= 30 if not returns.empty else False
            ),
            analytics_readiness=AnalyticsReadiness(
                can_calculate_var=len(returns) >= 30 and total_value > 0,
                can_calculate_sharpe=len(returns) >= 30,
                can_calculate_beta=len(returns) >= 30,
                can_calculate_drawdown=len(history_df) >= 2 if not history_df.empty else False,
                can_calculate_concentration=len(portfolio_data) > 0 and total_value > 0
            ),
            recommendations=_generate_recommendations(portfolio_data, history_df, returns, total_value)
        )
        
    except Exception as e:
        logger.error(f"Error validating user portfolio {user_id}: {str(e)}")
        return ValidationErrorResponse(
            user_id=user_id,
            validation_timestamp=datetime.now().isoformat(),
            error=str(e),
            status="validation_failed"
        )


def _generate_recommendations(portfolio_data, history_df, returns, total_value) -> List[str]:
    """Generate recommendations based on portfolio validation"""
    recommendations = []
    
    if not portfolio_data:
        recommendations.append("Add holdings to your portfolio to enable analytics calculations")
    elif total_value <= 0:
        recommendations.append("Ensure portfolio holdings have positive values")
    
    if history_df.empty:
        recommendations.append("Portfolio history tracking needs to be enabled")
    elif len(history_df) < 30:
        recommendations.append("Wait for more historical data to accumulate (need 30+ days)")
    
    if returns.empty:
        recommendations.append("No return data available - check portfolio history")
    elif len(returns) < 30:
        recommendations.append(f"Need {30 - len(returns)} more days of return data for reliable analytics")
    
    if len(portfolio_data) == 1:
        recommendations.append("Consider diversifying portfolio with additional holdings")
    
    if not recommendations:
        recommendations.append("Portfolio is ready for all analytics calculations")
    
    return recommendations
