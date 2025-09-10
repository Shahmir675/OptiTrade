"""
Sector Analysis and Investment Recommendations
Provides comprehensive sector-level insights for portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


class SectorAnalyzer:
    """
    Comprehensive sector analysis for investment decisions
    """
    
    def __init__(self, stocks_data: pd.DataFrame, historical_data: pd.DataFrame):
        self.stocks_data = stocks_data
        self.historical_data = historical_data
        
        # Sector mappings and classifications
        self.sector_classifications = self._initialize_sector_classifications()
        self.sector_metrics = {}
        self.sector_trends = {}
        self.sector_correlations = None
        
        # Calculate base metrics
        self._calculate_sector_metrics()
    
    def _initialize_sector_classifications(self) -> Dict[str, Dict]:
        """Initialize sector classifications and characteristics"""
        return {
            'Technology': {
                'risk_level': 'high',
                'growth_profile': 'high_growth',
                'cyclicality': 'moderate',
                'interest_rate_sensitivity': 'high',
                'sub_sectors': ['Software', 'Semiconductors', 'IT Services', 'Hardware']
            },
            'Healthcare': {
                'risk_level': 'moderate',
                'growth_profile': 'stable_growth',
                'cyclicality': 'defensive',
                'interest_rate_sensitivity': 'low',
                'sub_sectors': ['Pharmaceuticals', 'Biotechnology', 'Medical Devices', 'Healthcare Services']
            },
            'Financial Services': {
                'risk_level': 'high',
                'growth_profile': 'cyclical_growth',
                'cyclicality': 'high',
                'interest_rate_sensitivity': 'very_high',
                'sub_sectors': ['Banks', 'Insurance', 'Investment Services', 'Real Estate']
            },
            'Consumer Cyclical': {
                'risk_level': 'moderate',
                'growth_profile': 'cyclical_growth',
                'cyclicality': 'high',
                'interest_rate_sensitivity': 'moderate',
                'sub_sectors': ['Retail', 'Automotive', 'Hotels & Entertainment', 'Apparel']
            },
            'Consumer Defensive': {
                'risk_level': 'low',
                'growth_profile': 'stable_growth',
                'cyclicality': 'defensive',
                'interest_rate_sensitivity': 'low',
                'sub_sectors': ['Food & Beverages', 'Household Products', 'Tobacco', 'Discount Stores']
            },
            'Industrials': {
                'risk_level': 'moderate',
                'growth_profile': 'cyclical_growth',
                'cyclicality': 'high',
                'interest_rate_sensitivity': 'moderate',
                'sub_sectors': ['Aerospace & Defense', 'Industrial Equipment', 'Transportation', 'Construction']
            },
            'Energy': {
                'risk_level': 'very_high',
                'growth_profile': 'volatile_growth',
                'cyclicality': 'very_high',
                'interest_rate_sensitivity': 'moderate',
                'sub_sectors': ['Oil & Gas', 'Renewable Energy', 'Energy Equipment', 'Utilities']
            },
            'Basic Materials': {
                'risk_level': 'high',
                'growth_profile': 'cyclical_growth',
                'cyclicality': 'very_high',
                'interest_rate_sensitivity': 'moderate',
                'sub_sectors': ['Mining', 'Chemicals', 'Steel', 'Paper & Forest Products']
            },
            'Communication Services': {
                'risk_level': 'moderate',
                'growth_profile': 'moderate_growth',
                'cyclicality': 'moderate',
                'interest_rate_sensitivity': 'moderate',
                'sub_sectors': ['Telecommunications', 'Media', 'Entertainment', 'Interactive Media']
            },
            'Utilities': {
                'risk_level': 'low',
                'growth_profile': 'low_growth',
                'cyclicality': 'defensive',
                'interest_rate_sensitivity': 'high',
                'sub_sectors': ['Electric Utilities', 'Gas Utilities', 'Water Utilities', 'Renewable Utilities']
            },
            'Real Estate': {
                'risk_level': 'moderate',
                'growth_profile': 'moderate_growth',
                'cyclicality': 'high',
                'interest_rate_sensitivity': 'very_high',
                'sub_sectors': ['REITs', 'Real Estate Development', 'Real Estate Services']
            }
        }
    
    def _calculate_sector_metrics(self):
        """Calculate comprehensive metrics for each sector"""
        print("Calculating sector metrics...")
        
        for sector in self.sector_classifications.keys():
            sector_stocks = self.stocks_data[self.stocks_data['sector'] == sector]
            
            if len(sector_stocks) == 0:
                continue
            
            # Basic sector statistics
            total_market_cap = sector_stocks['marketCap'].sum()
            avg_market_cap = sector_stocks['marketCap'].mean()
            median_market_cap = sector_stocks['marketCap'].median()
            stock_count = len(sector_stocks)
            
            # Price performance metrics
            avg_price_change = sector_stocks['pctchange'].mean()
            volatility = sector_stocks['30d_realized_volatility_annualized'].mean()
            
            # Valuation metrics (if available)
            dividend_stocks = sector_stocks.dropna(subset=['dividend_yield'])
            avg_dividend_yield = dividend_stocks['dividend_yield'].mean() if len(dividend_stocks) > 0 else 0
            
            # Store metrics
            self.sector_metrics[sector] = {
                'stock_count': stock_count,
                'total_market_cap': total_market_cap,
                'avg_market_cap': avg_market_cap,
                'median_market_cap': median_market_cap,
                'avg_price_change': avg_price_change,
                'avg_volatility': volatility,
                'avg_dividend_yield': avg_dividend_yield,
                'market_cap_weight': total_market_cap / self.stocks_data['marketCap'].sum() if self.stocks_data['marketCap'].sum() > 0 else 0
            }
    
    def get_sector_performance_analysis(self, lookback_days: int = 252) -> Dict[str, Any]:
        """Get comprehensive sector performance analysis"""
        performance_analysis = {}
        
        if self.historical_data is None or len(self.historical_data) == 0:
            return self._get_static_sector_analysis()
        
        # Calculate sector returns
        sector_returns = self._calculate_sector_returns(lookback_days)
        
        for sector in self.sector_classifications.keys():
            if sector not in sector_returns:
                continue
            
            returns = sector_returns[sector]
            
            if len(returns) < 30:  # Need minimum data
                continue
            
            # Performance metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Risk metrics
            var_95 = np.percentile(returns, 5)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Trend analysis
            recent_trend = self._analyze_sector_trend(returns, window=30)
            momentum = self._calculate_momentum(returns)
            
            performance_analysis[sector] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'recent_trend': recent_trend,
                'momentum_score': momentum,
                'recommendation': self._generate_sector_recommendation(
                    sector, annualized_return, volatility, momentum
                ),
                'risk_adjusted_score': self._calculate_risk_adjusted_score(
                    annualized_return, volatility, sharpe_ratio
                )
            }
        
        return performance_analysis
    
    def _get_static_sector_analysis(self) -> Dict[str, Any]:
        """Fallback sector analysis based on static data"""
        performance_analysis = {}
        
        for sector, metrics in self.sector_metrics.items():
            # Use current market metrics to estimate performance
            estimated_return = metrics['avg_price_change'] / 100 if 'avg_price_change' in metrics else 0
            estimated_volatility = metrics.get('avg_volatility', 0.2)
            
            performance_analysis[sector] = {
                'total_return': estimated_return,
                'annualized_return': estimated_return,
                'volatility': estimated_volatility,
                'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0,
                'var_95': -0.05,  # Conservative estimate
                'max_drawdown': -0.20,  # Conservative estimate
                'recent_trend': 'neutral',
                'momentum_score': 0.0,
                'recommendation': self._generate_sector_recommendation(
                    sector, estimated_return, estimated_volatility, 0.0
                ),
                'risk_adjusted_score': estimated_return / estimated_volatility if estimated_volatility > 0 else 0
            }
        
        return performance_analysis
    
    def _calculate_sector_returns(self, lookback_days: int) -> Dict[str, pd.Series]:
        """Calculate daily returns for each sector"""
        sector_returns = {}
        
        # Get cutoff date
        max_date = pd.to_datetime(self.historical_data['Date'].max())
        cutoff_date = max_date - timedelta(days=lookback_days)
        
        # Filter data
        recent_data = self.historical_data[
            pd.to_datetime(self.historical_data['Date']) >= cutoff_date
        ].copy()
        
        for sector in self.sector_classifications.keys():
            # Get stocks in this sector
            sector_stocks = self.stocks_data[self.stocks_data['sector'] == sector]['symbol'].tolist()
            
            if not sector_stocks:
                continue
            
            # Filter historical data for sector stocks
            sector_data = recent_data[recent_data['Ticker'].isin(sector_stocks)]
            
            if len(sector_data) == 0:
                continue
            
            # Calculate sector index (market cap weighted if possible)
            sector_prices = []
            dates = sorted(sector_data['Date'].unique())
            
            for date in dates:
                date_data = sector_data[sector_data['Date'] == date]
                
                if len(date_data) > 0:
                    # Simple equal-weighted average (can be enhanced to market-cap weighted)
                    avg_price = date_data['Adj Close'].mean()
                    sector_prices.append(avg_price)
            
            if len(sector_prices) > 1:
                # Calculate returns
                returns = pd.Series(sector_prices).pct_change().dropna()
                sector_returns[sector] = returns
        
        return sector_returns
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _analyze_sector_trend(self, returns: pd.Series, window: int = 30) -> str:
        """Analyze recent sector trend"""
        if len(returns) < window:
            return 'neutral'
        
        recent_returns = returns.tail(window)
        cumulative_return = (1 + recent_returns).prod() - 1
        
        if cumulative_return > 0.05:
            return 'strong_uptrend'
        elif cumulative_return > 0.02:
            return 'uptrend'
        elif cumulative_return < -0.05:
            return 'strong_downtrend'
        elif cumulative_return < -0.02:
            return 'downtrend'
        else:
            return 'neutral'
    
    def _calculate_momentum(self, returns: pd.Series) -> float:
        """Calculate momentum score"""
        if len(returns) < 60:
            return 0.0
        
        # 3-month momentum
        three_month_return = (1 + returns.tail(60)).prod() - 1
        
        # 1-month momentum
        one_month_return = (1 + returns.tail(20)).prod() - 1
        
        # Combined momentum score
        momentum_score = 0.7 * three_month_return + 0.3 * one_month_return
        
        return momentum_score
    
    def _generate_sector_recommendation(self, 
                                       sector: str, 
                                       annual_return: float, 
                                       volatility: float, 
                                       momentum: float) -> str:
        """Generate investment recommendation for sector"""
        sector_info = self.sector_classifications.get(sector, {})
        risk_level = sector_info.get('risk_level', 'moderate')
        
        # Score components
        return_score = 1 if annual_return > 0.10 else 0.5 if annual_return > 0.05 else 0
        risk_score = 1 if volatility < 0.15 else 0.5 if volatility < 0.25 else 0
        momentum_score = 1 if momentum > 0.05 else 0.5 if momentum > 0 else 0
        
        # Weight by risk tolerance
        if risk_level == 'low':
            total_score = 0.2 * return_score + 0.5 * risk_score + 0.3 * momentum_score
        elif risk_level == 'high' or risk_level == 'very_high':
            total_score = 0.5 * return_score + 0.2 * risk_score + 0.3 * momentum_score
        else:  # moderate
            total_score = 0.4 * return_score + 0.3 * risk_score + 0.3 * momentum_score
        
        if total_score > 0.7:
            return 'strong_buy'
        elif total_score > 0.5:
            return 'buy'
        elif total_score > 0.3:
            return 'hold'
        else:
            return 'underweight'
    
    def _calculate_risk_adjusted_score(self, 
                                     annual_return: float, 
                                     volatility: float, 
                                     sharpe_ratio: float) -> float:
        """Calculate overall risk-adjusted score"""
        # Normalize components
        return_component = min(annual_return * 5, 1)  # Cap at 20% return
        volatility_component = max(1 - volatility * 2, 0)  # Penalize high vol
        sharpe_component = min(sharpe_ratio / 2, 1)  # Normalize Sharpe
        
        # Combined score
        score = 0.4 * return_component + 0.3 * volatility_component + 0.3 * sharpe_component
        
        return max(0, min(1, score))  # Ensure 0-1 range
    
    def get_sector_diversification_analysis(self, 
                                          current_portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Analyze sector diversification of current portfolio"""
        # Map stocks to sectors
        sector_allocation = {}
        total_allocation = sum(current_portfolio.values())
        
        for stock, weight in current_portfolio.items():
            stock_info = self.stocks_data[self.stocks_data['symbol'] == stock]
            
            if len(stock_info) > 0:
                sector = stock_info.iloc[0]['sector']
                if pd.notna(sector):
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
        
        # Calculate diversification metrics
        num_sectors = len(sector_allocation)
        max_sector_weight = max(sector_allocation.values()) if sector_allocation else 0
        
        # Herfindahl-Hirschman Index for concentration
        hhi = sum(weight**2 for weight in sector_allocation.values())
        
        # Diversification score (1 is perfectly diversified across 10 sectors)
        ideal_sectors = 10
        diversification_score = min(num_sectors / ideal_sectors, 1.0) * (1 - hhi)
        
        return {
            'sector_allocation': sector_allocation,
            'num_sectors': num_sectors,
            'max_sector_weight': max_sector_weight,
            'hhi_concentration': hhi,
            'diversification_score': diversification_score,
            'recommendations': self._get_diversification_recommendations(
                sector_allocation, diversification_score
            )
        }
    
    def _get_diversification_recommendations(self, 
                                           sector_allocation: Dict[str, float],
                                           diversification_score: float) -> List[str]:
        """Generate diversification recommendations"""
        recommendations = []
        
        if diversification_score < 0.3:
            recommendations.append("Portfolio is poorly diversified - consider adding exposure to underrepresented sectors")
        
        # Check for over-concentration
        for sector, weight in sector_allocation.items():
            if weight > 0.3:
                recommendations.append(f"Reduce concentration in {sector} (currently {weight:.1%})")
            elif weight > 0.25:
                recommendations.append(f"Monitor concentration in {sector} (currently {weight:.1%})")
        
        # Check for missing defensive sectors
        defensive_sectors = ['Consumer Defensive', 'Healthcare', 'Utilities']
        missing_defensive = [s for s in defensive_sectors if s not in sector_allocation]
        
        if len(missing_defensive) > 0:
            recommendations.append(f"Consider adding defensive exposure: {', '.join(missing_defensive)}")
        
        # Check for growth sectors
        growth_sectors = ['Technology', 'Healthcare']
        missing_growth = [s for s in growth_sectors if s not in sector_allocation]
        
        if len(missing_growth) > 0:
            recommendations.append(f"Consider growth sector exposure: {', '.join(missing_growth)}")
        
        return recommendations
    
    def get_sector_rotation_strategy(self, 
                                   market_regime: str = 'neutral',
                                   economic_cycle: str = 'expansion') -> Dict[str, Any]:
        """Generate sector rotation strategy based on market conditions"""
        
        rotation_strategies = {
            'early_expansion': {
                'overweight': ['Technology', 'Consumer Cyclical', 'Industrials'],
                'underweight': ['Utilities', 'Consumer Defensive'],
                'rationale': 'Economic growth accelerating, favor cyclical sectors'
            },
            'mid_expansion': {
                'overweight': ['Technology', 'Healthcare', 'Financial Services'],
                'underweight': ['Energy', 'Basic Materials'],
                'rationale': 'Sustained growth phase, quality growth preferred'
            },
            'late_expansion': {
                'overweight': ['Energy', 'Basic Materials', 'Financial Services'],
                'underweight': ['Technology', 'Consumer Cyclical'],
                'rationale': 'Inflation concerns, favor value and commodity sectors'
            },
            'contraction': {
                'overweight': ['Consumer Defensive', 'Healthcare', 'Utilities'],
                'underweight': ['Consumer Cyclical', 'Energy', 'Basic Materials'],
                'rationale': 'Economic slowdown, favor defensive sectors'
            }
        }
        
        strategy = rotation_strategies.get(economic_cycle, rotation_strategies['mid_expansion'])
        
        # Adjust based on market regime
        if market_regime == 'high_volatility':
            # Add more defensive positioning
            strategy['overweight'] = [s for s in strategy['overweight'] if s in ['Healthcare', 'Consumer Defensive', 'Utilities']]
            strategy['underweight'].extend(['Technology', 'Energy'])
        
        return {
            'strategy': strategy,
            'market_regime': market_regime,
            'economic_cycle': economic_cycle,
            'implementation_timeline': '3-6 months',
            'rebalancing_frequency': 'monthly'
        }
    
    def get_top_sectors_recommendation(self, 
                                     risk_tolerance: str = 'moderate',
                                     investment_horizon: str = 'medium',
                                     num_sectors: int = 5) -> Dict[str, Any]:
        """Get top sector recommendations based on user profile"""
        
        # Get sector performance analysis
        sector_performance = self.get_sector_performance_analysis()
        
        # Filter and score sectors based on risk tolerance
        sector_scores = {}
        
        for sector, metrics in sector_performance.items():
            sector_info = self.sector_classifications.get(sector, {})
            
            # Base score from performance
            base_score = metrics.get('risk_adjusted_score', 0.5)
            
            # Risk adjustment
            risk_level = sector_info.get('risk_level', 'moderate')
            
            if risk_tolerance == 'conservative':
                if risk_level in ['low', 'moderate']:
                    risk_adjustment = 1.2
                else:
                    risk_adjustment = 0.6
            elif risk_tolerance == 'aggressive':
                if risk_level in ['high', 'very_high']:
                    risk_adjustment = 1.3
                else:
                    risk_adjustment = 0.8
            else:  # moderate
                risk_adjustment = 1.0
            
            # Time horizon adjustment
            growth_profile = sector_info.get('growth_profile', 'stable_growth')
            
            if investment_horizon == 'long':
                if growth_profile in ['high_growth', 'stable_growth']:
                    horizon_adjustment = 1.2
                else:
                    horizon_adjustment = 0.9
            elif investment_horizon == 'short':
                if growth_profile in ['low_growth', 'stable_growth']:
                    horizon_adjustment = 1.1
                else:
                    horizon_adjustment = 0.8
            else:  # medium
                horizon_adjustment = 1.0
            
            # Final score
            final_score = base_score * risk_adjustment * horizon_adjustment
            sector_scores[sector] = final_score
        
        # Get top sectors
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        top_sectors = sorted_sectors[:num_sectors]
        
        # Generate allocation recommendations
        total_score = sum(score for _, score in top_sectors)
        allocations = {}
        
        for sector, score in top_sectors:
            weight = score / total_score if total_score > 0 else 1.0 / len(top_sectors)
            allocations[sector] = {
                'recommended_weight': weight,
                'score': score,
                'rationale': self._get_sector_rationale(sector, sector_performance.get(sector, {}))
            }
        
        return {
            'top_sectors': allocations,
            'diversification_target': len(top_sectors),
            'rebalancing_threshold': 0.05,  # 5%
            'review_frequency': 'quarterly'
        }
    
    def _get_sector_rationale(self, sector: str, metrics: Dict) -> str:
        """Generate rationale for sector recommendation"""
        sector_info = self.sector_classifications.get(sector, {})
        
        rationale_parts = []
        
        # Performance component
        if metrics.get('sharpe_ratio', 0) > 1.0:
            rationale_parts.append("strong risk-adjusted returns")
        elif metrics.get('annualized_return', 0) > 0.1:
            rationale_parts.append("solid return potential")
        
        # Risk component
        risk_level = sector_info.get('risk_level', 'moderate')
        if risk_level == 'low':
            rationale_parts.append("defensive characteristics")
        elif risk_level == 'high':
            rationale_parts.append("growth potential")
        
        # Trend component
        trend = metrics.get('recent_trend', 'neutral')
        if trend in ['uptrend', 'strong_uptrend']:
            rationale_parts.append("positive momentum")
        
        # Fundamental component
        growth_profile = sector_info.get('growth_profile', 'stable_growth')
        if growth_profile == 'high_growth':
            rationale_parts.append("secular growth trends")
        elif growth_profile == 'stable_growth':
            rationale_parts.append("stable business fundamentals")
        
        if rationale_parts:
            return f"Recommended due to {', '.join(rationale_parts)}"
        else:
            return "Balanced risk-return profile"
    
    def export_analysis(self, filepath: str):
        """Export comprehensive sector analysis to JSON"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'sector_classifications': self.sector_classifications,
            'sector_metrics': self.sector_metrics,
            'performance_analysis': self.get_sector_performance_analysis(),
            'rotation_strategy': self.get_sector_rotation_strategy(),
            'top_recommendations': {
                'conservative': self.get_top_sectors_recommendation('conservative'),
                'moderate': self.get_top_sectors_recommendation('moderate'),
                'aggressive': self.get_top_sectors_recommendation('aggressive')
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Sector analysis exported to {filepath}")


def create_sector_analyzer(stocks_json_path: str, 
                          historical_data_path: str = None) -> SectorAnalyzer:
    """Create and initialize a SectorAnalyzer instance"""
    
    # Load stocks data
    stocks_data = []
    with open(stocks_json_path, 'r') as f:
        for line in f:
            stocks_data.append(json.loads(line.strip()))
    
    stocks_df = pd.DataFrame(stocks_data)
    
    # Load historical data if available
    historical_df = None
    if historical_data_path:
        try:
            historical_df = pd.read_csv(historical_data_path)
            if 'Date' in historical_df.columns:
                historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            historical_df = pd.DataFrame()
    
    return SectorAnalyzer(stocks_df, historical_df or pd.DataFrame())