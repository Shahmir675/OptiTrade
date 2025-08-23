"""
Portfolio Analytics Service

This module provides comprehensive portfolio analytics calculations including:
- Value at Risk (VaR)
- Maximum Drawdown (MDD)
- Sharpe Ratio
- Beta
- Portfolio Concentration (Herfindahl-Hirschman Index)

Updated with professional error handling using Union types for success/error states.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sqlalchemy_models import Portfolio, PortfolioHistory
from app.services.data_service import get_yfinance_stock_data_async
from app.models.pydantic_models import (
    AnalyticsError, AnalyticsErrorType,
    VaRResponse, VaRSuccess,
    MaxDrawdownResponse, MaxDrawdownSuccess,
    SharpeRatioResponse, SharpeRatioSuccess,
    BetaResponse, BetaSuccess,
    ConcentrationResponse, ConcentrationSuccess,
    ComprehensiveAnalyticsResponse, AnalyticsParameters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service class for portfolio analytics calculations with professional error handling"""

    @staticmethod
    async def get_portfolio_data(user_id: int, db: AsyncSession) -> List[Dict]:
        """Get current portfolio holdings for a user"""
        query = select(Portfolio).where(Portfolio.user_id == user_id)
        result = await db.execute(query)
        portfolios = result.scalars().all()
        
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "average_price": float(p.average_price),
                "current_value": float(p.current_value),
                "total_invested": float(p.total_invested),
            }
            for p in portfolios
        ]

    @staticmethod
    async def get_portfolio_history(
        user_id: int, 
        db: AsyncSession, 
        days: int = 252
    ) -> pd.DataFrame:
        """Get portfolio history for the specified number of days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query = (
            select(PortfolioHistory)
            .where(
                PortfolioHistory.user_id == user_id,
                PortfolioHistory.snapshot_date >= start_date
            )
            .order_by(PortfolioHistory.snapshot_date)
        )
        
        result = await db.execute(query)
        history = result.scalars().all()
        
        if not history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for h in history:
            data.append({
                "date": h.snapshot_date,
                "symbol": h.symbol,
                "quantity": h.quantity,
                "average_price": float(h.average_price) if h.average_price else 0.0,
                "current_value": float(h.current_value) if h.current_value else 0.0,
                "total_invested": float(h.total_invested) if h.total_invested else 0.0
            })
        
        df = pd.DataFrame(data)
        return df

    @staticmethod
    async def calculate_portfolio_returns(
        user_id: int,
        db: AsyncSession,
        days: int = 252
    ) -> pd.Series:
        """Calculate daily portfolio returns"""
        history_df = await AnalyticsService.get_portfolio_history(user_id, db, days)

        if history_df.empty:
            logger.warning(f"No portfolio history found for user {user_id}")
            return pd.Series()

        # Group by date and sum current values to get total portfolio value per day
        daily_values = (
            history_df.groupby("date")["current_value"]
            .sum()
            .sort_index()
        )

        if len(daily_values) < 2:
            logger.warning(f"Insufficient portfolio history data for user {user_id}")
            return pd.Series()

        # Calculate daily returns
        returns = daily_values.pct_change().dropna()
        return returns

    @staticmethod
    async def calculate_var(
        user_id: int,
        db: AsyncSession,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        days: int = 252
    ) -> VaRResponse:
        """
        Calculate Value at Risk (VaR) using historical simulation method
        
        Returns:
            VaRResponse with either success data or error information
        """
        try:
            # Validate parameters
            if not (0.01 <= confidence_level <= 0.99):
                return VaRResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INVALID_PARAMETERS,
                    message="Confidence level must be between 0.01 and 0.99",
                    details=f"Provided: {confidence_level}"
                ))

            if time_horizon <= 0 or days <= 0:
                return VaRResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INVALID_PARAMETERS,
                    message="Time horizon and historical days must be positive",
                    details=f"Time horizon: {time_horizon}, Days: {days}"
                ))

            # Check if portfolio exists
            portfolio_data = await AnalyticsService.get_portfolio_data(user_id, db)
            if not portfolio_data:
                return VaRResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.PORTFOLIO_EMPTY,
                    message="No portfolio holdings found for user",
                    details=f"User ID: {user_id}"
                ))

            returns = await AnalyticsService.calculate_portfolio_returns(user_id, db, days)
            
            if returns.empty or len(returns) < 30:
                return VaRResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INSUFFICIENT_DATA,
                    message="Insufficient historical data for VaR calculation",
                    details=f"Need at least 30 data points, got {len(returns)}"
                ))
            
            # Scale returns for time horizon
            scaled_returns = returns * np.sqrt(time_horizon)
            
            # Calculate VaR using percentile method
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(scaled_returns, var_percentile)
            
            # Additional statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Get current portfolio value
            current_value = sum(item["current_value"] for item in portfolio_data)
            
            # VaR in dollar terms
            var_dollar = abs(var_value * current_value)
            
            return VaRResponse(root=VaRSuccess(
                var_percentage=float(var_value),
                var_dollar=float(var_dollar),
                confidence_level=confidence_level,
                time_horizon_days=time_horizon,
                current_portfolio_value=float(current_value),
                mean_daily_return=float(mean_return),
                daily_volatility=float(std_return),
                observations_used=len(returns)
            ))
            
        except Exception as e:
            logger.error(f"Error calculating VaR for user {user_id}: {str(e)}")
            return VaRResponse(root=AnalyticsError(
                error_type=AnalyticsErrorType.CALCULATION_ERROR,
                message="Failed to calculate VaR",
                details=str(e)
            ))

    @staticmethod
    async def calculate_maximum_drawdown(
        user_id: int,
        db: AsyncSession,
        days: int = 252
    ) -> MaxDrawdownResponse:
        """
        Calculate Maximum Drawdown (MDD)
        
        Returns:
            MaxDrawdownResponse with either success data or error information
        """
        try:
            if days <= 0:
                return MaxDrawdownResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INVALID_PARAMETERS,
                    message="Historical days must be positive",
                    details=f"Provided: {days}"
                ))

            history_df = await AnalyticsService.get_portfolio_history(user_id, db, days)
            
            if history_df.empty:
                return MaxDrawdownResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INSUFFICIENT_DATA,
                    message="No portfolio history data available",
                    details=f"User ID: {user_id}, Days requested: {days}"
                ))
            
            # Get daily portfolio values by summing current values
            daily_values = (
                history_df.groupby("date")["current_value"]
                .sum()
                .sort_index()
            )
            
            if len(daily_values) < 2:
                return MaxDrawdownResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INSUFFICIENT_DATA,
                    message="Insufficient data for drawdown calculation",
                    details=f"Need at least 2 data points, got {len(daily_values)}"
                ))
            
            # Calculate running maximum (peak)
            running_max = daily_values.expanding().max()
            
            # Calculate drawdown
            drawdown = (daily_values - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.idxmin()
            
            # Find peak date (start of drawdown period)
            peak_date = running_max.loc[:max_drawdown_date].idxmax()
            
            # Calculate recovery information
            recovery_date = None
            if max_drawdown_date < daily_values.index[-1]:
                peak_value = daily_values.loc[peak_date]
                post_drawdown = daily_values.loc[max_drawdown_date:]
                recovery_mask = post_drawdown >= peak_value
                if recovery_mask.any():
                    recovery_date = post_drawdown[recovery_mask].index[0]
            
            return MaxDrawdownResponse(root=MaxDrawdownSuccess(
                max_drawdown_percentage=float(max_drawdown),
                peak_date=peak_date.isoformat(),
                trough_date=max_drawdown_date.isoformat(),
                recovery_date=recovery_date.isoformat() if recovery_date else None,
                peak_value=float(daily_values.loc[peak_date]),
                trough_value=float(daily_values.loc[max_drawdown_date]),
                current_value=float(daily_values.iloc[-1]),
                drawdown_duration_days=(max_drawdown_date - peak_date).days,
                observations_used=len(daily_values)
            ))
            
        except Exception as e:
            logger.error(f"Error calculating MDD for user {user_id}: {str(e)}")
            return MaxDrawdownResponse(root=AnalyticsError(
                error_type=AnalyticsErrorType.CALCULATION_ERROR,
                message="Failed to calculate maximum drawdown",
                details=str(e)
            ))

    @staticmethod
    async def calculate_sharpe_ratio(
        user_id: int,
        db: AsyncSession,
        risk_free_rate: float = 0.02,
        days: int = 252
    ) -> SharpeRatioResponse:
        """
        Calculate Sharpe Ratio
        
        Returns:
            SharpeRatioResponse with either success data or error information
        """
        try:
            if risk_free_rate < 0 or risk_free_rate > 1:
                return SharpeRatioResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INVALID_PARAMETERS,
                    message="Risk-free rate should be between 0 and 1",
                    details=f"Provided: {risk_free_rate}"
                ))

            if days <= 0:
                return SharpeRatioResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INVALID_PARAMETERS,
                    message="Historical days must be positive",
                    details=f"Provided: {days}"
                ))

            returns = await AnalyticsService.calculate_portfolio_returns(user_id, db, days)
            
            if returns.empty or len(returns) < 30:
                return SharpeRatioResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INSUFFICIENT_DATA,
                    message="Insufficient data for Sharpe ratio calculation",
                    details=f"Need at least 30 data points, got {len(returns)}"
                ))
            
            # Convert annual risk-free rate to daily
            daily_risk_free_rate = risk_free_rate / 252
            
            # Calculate excess returns
            excess_returns = returns - daily_risk_free_rate
            
            # Calculate Sharpe ratio
            mean_excess_return = excess_returns.mean()
            std_excess_return = excess_returns.std()
            
            if std_excess_return == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = mean_excess_return / std_excess_return
            
            # Annualize the Sharpe ratio
            annualized_sharpe = sharpe_ratio * np.sqrt(252)
            
            # Additional metrics
            annualized_return = returns.mean() * 252
            annualized_volatility = returns.std() * np.sqrt(252)
            
            return SharpeRatioResponse(root=SharpeRatioSuccess(
                sharpe_ratio=float(annualized_sharpe),
                annualized_return=float(annualized_return),
                annualized_volatility=float(annualized_volatility),
                risk_free_rate=risk_free_rate,
                excess_return=float(annualized_return - risk_free_rate),
                observations_used=len(returns)
            ))
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio for user {user_id}: {str(e)}")
            return SharpeRatioResponse(root=AnalyticsError(
                error_type=AnalyticsErrorType.CALCULATION_ERROR,
                message="Failed to calculate Sharpe ratio",
                details=str(e)
            ))

    @staticmethod
    async def calculate_beta(
        user_id: int,
        db: AsyncSession,
        market_symbol: str = "SPY",
        days: int = 252
    ) -> BetaResponse:
        """
        Calculate portfolio beta relative to market index

        Returns:
            BetaResponse with either success data or error information
        """
        try:
            if days <= 0:
                return BetaResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INVALID_PARAMETERS,
                    message="Historical days must be positive",
                    details=f"Provided: {days}"
                ))

            if not market_symbol or not market_symbol.strip():
                return BetaResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INVALID_PARAMETERS,
                    message="Market symbol cannot be empty",
                    details="Please provide a valid market symbol (e.g., SPY, ^GSPC)"
                ))

            # Get portfolio returns
            portfolio_returns = await AnalyticsService.calculate_portfolio_returns(user_id, db, days)

            if portfolio_returns.empty or len(portfolio_returns) < 30:
                return BetaResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INSUFFICIENT_DATA,
                    message="Insufficient portfolio data for beta calculation",
                    details=f"Need at least 30 data points, got {len(portfolio_returns)}"
                ))

            # Get market data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

            try:
                market_data = await get_yfinance_stock_data_async(market_symbol, start_date, end_date)
            except Exception as e:
                return BetaResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.MARKET_DATA_UNAVAILABLE,
                    message=f"Failed to fetch market data for {market_symbol}",
                    details=str(e)
                ))

            if market_data is None or market_data.empty:
                return BetaResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.MARKET_DATA_UNAVAILABLE,
                    message=f"No market data available for {market_symbol}",
                    details="Please check the market symbol or try a different one"
                ))

            # Calculate market returns
            market_returns = market_data["Close"].pct_change().dropna()

            # Align dates
            common_dates = portfolio_returns.index.intersection(market_returns.index)

            if len(common_dates) < 30:
                return BetaResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.INSUFFICIENT_DATA,
                    message="Insufficient overlapping data for beta calculation",
                    details=f"Need at least 30 overlapping data points, got {len(common_dates)}"
                ))

            aligned_portfolio = portfolio_returns.loc[common_dates]
            aligned_market = market_returns.loc[common_dates]

            # Calculate beta using linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                aligned_market, aligned_portfolio
            )

            beta = slope
            alpha = intercept * 252  # Annualized alpha
            r_squared = r_value ** 2

            # Additional metrics
            correlation = aligned_portfolio.corr(aligned_market)
            portfolio_vol = aligned_portfolio.std() * np.sqrt(252)
            market_vol = aligned_market.std() * np.sqrt(252)

            return BetaResponse(root=BetaSuccess(
                beta=float(beta),
                alpha_annualized=float(alpha),
                r_squared=float(r_squared),
                correlation=float(correlation),
                portfolio_volatility=float(portfolio_vol),
                market_volatility=float(market_vol),
                market_symbol=market_symbol,
                p_value=float(p_value),
                observations_used=len(common_dates)
            ))

        except Exception as e:
            logger.error(f"Error calculating beta for user {user_id}: {str(e)}")
            return BetaResponse(root=AnalyticsError(
                error_type=AnalyticsErrorType.CALCULATION_ERROR,
                message="Failed to calculate beta",
                details=str(e)
            ))

    @staticmethod
    async def calculate_portfolio_concentration(
        user_id: int,
        db: AsyncSession
    ) -> ConcentrationResponse:
        """
        Calculate portfolio concentration using Herfindahl-Hirschman Index (HHI)

        Returns:
            ConcentrationResponse with either success data or error information
        """
        try:
            portfolio_data = await AnalyticsService.get_portfolio_data(user_id, db)

            if not portfolio_data:
                return ConcentrationResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.PORTFOLIO_EMPTY,
                    message="No portfolio holdings found for user",
                    details=f"User ID: {user_id}"
                ))

            # Calculate total portfolio value
            total_value = sum(item["current_value"] for item in portfolio_data)

            if total_value <= 0:
                return ConcentrationResponse(root=AnalyticsError(
                    error_type=AnalyticsErrorType.PORTFOLIO_EMPTY,
                    message="Portfolio has no positive value",
                    details=f"Total portfolio value: {total_value}"
                ))

            # Calculate weights
            weights = []
            holdings_info = []

            for item in portfolio_data:
                weight = item["current_value"] / total_value
                weights.append(weight)
                holdings_info.append({
                    "symbol": item["symbol"],
                    "value": item["current_value"],
                    "weight": weight,
                    "quantity": item["quantity"]
                })

            # Calculate HHI (sum of squared weights)
            hhi = sum(w ** 2 for w in weights)

            # Calculate effective number of holdings
            effective_holdings = 1 / hhi if hhi > 0 else 0

            # Concentration categories
            if hhi < 0.15:
                concentration_level = "Low"
            elif hhi < 0.25:
                concentration_level = "Moderate"
            else:
                concentration_level = "High"

            # Find largest holdings
            holdings_info.sort(key=lambda x: x["weight"], reverse=True)
            top_5_holdings = holdings_info[:5]
            top_5_weight = sum(h["weight"] for h in top_5_holdings)

            return ConcentrationResponse(root=ConcentrationSuccess(
                herfindahl_hirschman_index=float(hhi),
                effective_number_of_holdings=float(effective_holdings),
                actual_number_of_holdings=len(portfolio_data),
                concentration_level=concentration_level,
                total_portfolio_value=float(total_value),
                top_5_holdings_weight=float(top_5_weight),
                top_holdings=top_5_holdings,
                all_holdings=holdings_info
            ))

        except Exception as e:
            logger.error(f"Error calculating concentration for user {user_id}: {str(e)}")
            return ConcentrationResponse(root=AnalyticsError(
                error_type=AnalyticsErrorType.CALCULATION_ERROR,
                message="Failed to calculate portfolio concentration",
                details=str(e)
            ))

    @staticmethod
    async def get_comprehensive_analytics(
        user_id: int,
        db: AsyncSession,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.02,
        market_symbol: str = "SPY",
        days: int = 252
    ) -> ComprehensiveAnalyticsResponse:
        """
        Get comprehensive portfolio analytics with professional error handling

        Returns:
            ComprehensiveAnalyticsResponse with all analytics metrics
        """
        try:
            # Run all calculations concurrently
            var_task = AnalyticsService.calculate_var(user_id, db, confidence_level, 1, days)
            mdd_task = AnalyticsService.calculate_maximum_drawdown(user_id, db, days)
            sharpe_task = AnalyticsService.calculate_sharpe_ratio(user_id, db, risk_free_rate, days)
            beta_task = AnalyticsService.calculate_beta(user_id, db, market_symbol, days)
            concentration_task = AnalyticsService.calculate_portfolio_concentration(user_id, db)

            var_result, mdd_result, sharpe_result, beta_result, concentration_result = await asyncio.gather(
                var_task, mdd_task, sharpe_task, beta_task, concentration_task
            )

            return ComprehensiveAnalyticsResponse(
                user_id=user_id,
                calculation_date=datetime.now().isoformat(),
                parameters=AnalyticsParameters(
                    confidence_level=confidence_level,
                    risk_free_rate=risk_free_rate,
                    market_symbol=market_symbol,
                    historical_days=days
                ),
                var=var_result,
                maximum_drawdown=mdd_result,
                sharpe_ratio=sharpe_result,
                beta=beta_result,
                concentration=concentration_result
            )

        except Exception as e:
            logger.error(f"Error calculating comprehensive analytics for user {user_id}: {str(e)}")
            # Return a response with all metrics as errors
            error = AnalyticsError(
                error_type=AnalyticsErrorType.CALCULATION_ERROR,
                message="Failed to calculate comprehensive analytics",
                details=str(e)
            )
            
            return ComprehensiveAnalyticsResponse(
                user_id=user_id,
                calculation_date=datetime.now().isoformat(),
                parameters=AnalyticsParameters(
                    confidence_level=confidence_level,
                    risk_free_rate=risk_free_rate,
                    market_symbol=market_symbol,
                    historical_days=days
                ),
                var=VaRResponse(root=error),
                maximum_drawdown=MaxDrawdownResponse(root=error),
                sharpe_ratio=SharpeRatioResponse(root=error),
                beta=BetaResponse(root=error),
                concentration=ConcentrationResponse(root=error)
            )