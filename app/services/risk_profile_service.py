from typing import List
from app.models import pydantic_models as pyd_models


def calculate_risk_score(assessment: pyd_models.RiskAssessmentRequest) -> float:
    base_score = assessment.risk_tolerance * 10
    age_factor = max(0, 100 - assessment.age)
    experience_multiplier = {"beginner": 0.8, "intermediate": 1.0, "advanced": 1.2}[assessment.investment_experience]
    timeline_factor = min(assessment.investment_timeline * 5, 30)
    
    return min(100, (base_score + age_factor * 0.3 + timeline_factor) * experience_multiplier)


def get_risk_category(score: float) -> str:
    if score >= 70:
        return "Aggressive"
    elif score >= 40:
        return "Moderate"
    else:
        return "Conservative"


def get_risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def generate_recommendations(score: float) -> List[dict]:
    if score >= 80:
        return [
            {
                "type": "increase_growth_allocation",
                "title": "Focus on high-growth stocks",
                "description": "Consider increasing allocation to growth and tech stocks for maximum returns",
                "priority": "high",
                "impact": "Potential 3-5% annual return increase with higher volatility"
            },
            {
                "type": "enable_momentum_trading",
                "title": "Enable momentum-based trading",
                "description": "AI can execute momentum trades and swing positions for enhanced returns",
                "priority": "medium",
                "impact": "Capture short-term market movements"
            }
        ]
    elif score >= 60:
        return [
            {
                "type": "diversify_sectors",
                "title": "Diversify across growth sectors",
                "description": "Spread investments across technology, healthcare, and emerging market stocks",
                "priority": "medium",
                "impact": "Reduce sector-specific risk while maintaining growth potential"
            },
            {
                "type": "enable_rebalancing",
                "title": "Enable automatic rebalancing",
                "description": "Allow AI to rebalance portfolio quarterly to maintain target allocations",
                "priority": "high",
                "impact": "Maintain optimal risk-return profile"
            }
        ]
    elif score >= 40:
        return [
            {
                "type": "balanced_allocation",
                "title": "Maintain balanced stock allocation",
                "description": "Mix of dividend stocks, blue-chip stocks, and moderate growth stocks",
                "priority": "medium",
                "impact": "Steady growth with moderate risk and regular income"
            },
            {
                "type": "enable_dividend_focus",
                "title": "Focus on dividend-paying stocks",
                "description": "AI can prioritize stocks with consistent dividend history",
                "priority": "low",
                "impact": "Generate regular passive income"
            }
        ]
    else:
        return [
            {
                "type": "conservative_stocks",
                "title": "Focus on blue-chip and dividend stocks",
                "description": "Prioritize established companies with strong fundamentals and dividend history",
                "priority": "high",
                "impact": "Minimize volatility while providing steady returns"
            },
            {
                "type": "enable_loss_protection",
                "title": "Enable automatic stop-loss orders",
                "description": "AI will set protective stop-losses to limit downside risk",
                "priority": "high",
                "impact": "Protect against significant losses during market downturns"
            }
        ]


def get_ai_trading_parameters(risk_score: float) -> dict:
    """Returns AI trading parameters based on risk score"""
    if risk_score >= 80:
        return {
            "max_position_size": 0.15,
            "stop_loss_threshold": 0.15,
            "take_profit_threshold": 0.25,
            "rebalance_frequency": "weekly",
            "sector_concentration_limit": 0.40,
            "enable_options_trading": True,
            "enable_margin_trading": True,
            "volatility_tolerance": "high"
        }
    elif risk_score >= 60:
        return {
            "max_position_size": 0.10,
            "stop_loss_threshold": 0.12,
            "take_profit_threshold": 0.20,
            "rebalance_frequency": "bi_weekly",
            "sector_concentration_limit": 0.30,
            "enable_options_trading": False,
            "enable_margin_trading": False,
            "volatility_tolerance": "medium_high"
        }
    elif risk_score >= 40:
        return {
            "max_position_size": 0.08,
            "stop_loss_threshold": 0.10,
            "take_profit_threshold": 0.15,
            "rebalance_frequency": "monthly",
            "sector_concentration_limit": 0.25,
            "enable_options_trading": False,
            "enable_margin_trading": False,
            "volatility_tolerance": "medium"
        }
    else:
        return {
            "max_position_size": 0.05,
            "stop_loss_threshold": 0.08,
            "take_profit_threshold": 0.12,
            "rebalance_frequency": "quarterly",
            "sector_concentration_limit": 0.20,
            "enable_options_trading": False,
            "enable_margin_trading": False,
            "volatility_tolerance": "low"
        }


def get_stock_allocation_strategy(risk_score: float) -> dict:
    """Returns stock allocation strategy based on risk score"""
    if risk_score >= 80:
        return {
            "growth_stocks": 0.60,
            "tech_stocks": 0.25,
            "dividend_stocks": 0.10,
            "international_stocks": 0.05,
            "preferred_sectors": ["technology", "biotech", "renewable_energy", "emerging_markets"],
            "avoid_sectors": ["utilities", "consumer_staples"]
        }
    elif risk_score >= 60:
        return {
            "growth_stocks": 0.45,
            "tech_stocks": 0.20,
            "dividend_stocks": 0.25,
            "international_stocks": 0.10,
            "preferred_sectors": ["technology", "healthcare", "financial", "consumer_discretionary"],
            "avoid_sectors": []
        }
    elif risk_score >= 40:
        return {
            "growth_stocks": 0.30,
            "tech_stocks": 0.15,
            "dividend_stocks": 0.40,
            "international_stocks": 0.15,
            "preferred_sectors": ["healthcare", "financial", "consumer_staples", "industrials"],
            "avoid_sectors": ["biotech", "cryptocurrency"]
        }
    else:
        return {
            "growth_stocks": 0.15,
            "tech_stocks": 0.10,
            "dividend_stocks": 0.60,
            "international_stocks": 0.15,
            "preferred_sectors": ["utilities", "consumer_staples", "healthcare", "financial"],
            "avoid_sectors": ["biotech", "cryptocurrency", "emerging_markets"]
        }