"""
Example Usage of the Portfolio Optimizer System
Demonstrates how to use the LSTM + PPO portfolio optimization system
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add the portfolio_optimizer directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio_optimizer import PortfolioOptimizer
from utils.sector_analyzer import create_sector_analyzer
from utils.data_loader import StockDataLoader


def main():
    """Main example demonstrating the portfolio optimization system"""
    
    print("=" * 60)
    print("Portfolio Optimizer with LSTM + PPO - Example Usage")
    print("=" * 60)
    
    # Configuration
    STOCKS_JSON_PATH = "../../app/static/stocks.json"
    HISTORICAL_DATA_PATH = "../10_Year_Historical_Scaled.csv"
    
    # User risk profile example
    risk_profile = {
        'risk_tolerance': 'moderate',  # conservative, moderate, aggressive
        'investment_horizon': 'long',  # short, medium, long
        'preferred_sectors': ['Technology', 'Healthcare'],
        'excluded_sectors': ['Energy'],  # Exclude for ESG reasons
        'target_volatility': 0.15,     # 15% annual volatility target
        'max_drawdown_tolerance': 0.20  # 20% maximum drawdown
    }
    
    print("\n1. User Risk Profile:")
    print(f"   Risk Tolerance: {risk_profile['risk_tolerance']}")
    print(f"   Investment Horizon: {risk_profile['investment_horizon']}")
    print(f"   Preferred Sectors: {', '.join(risk_profile['preferred_sectors'])}")
    
    # Initialize Portfolio Optimizer
    print("\n2. Initializing Portfolio Optimizer...")
    optimizer = PortfolioOptimizer(
        stocks_json_path=STOCKS_JSON_PATH,
        historical_data_path=HISTORICAL_DATA_PATH,
        risk_profile=risk_profile,
        initial_balance=100000.0,
        max_stocks=30  # Reduced for faster training in example
    )
    
    print(f"   Selected {len(optimizer.stock_symbols)} stocks for optimization")
    print(f"   Historical data: {len(optimizer.historical_data)} records")
    
    # Train the system (reduced parameters for example)
    print("\n3. Training the AI System...")
    print("   This may take several minutes...")
    
    training_success = optimizer.train_system(
        lstm_epochs=50,    # Reduced for example
        ppo_episodes=200,  # Reduced for example
        save_models=True
    )
    
    if training_success:
        print("   ✓ Training completed successfully!")
    else:
        print("   ✗ Training failed, continuing with available components...")
    
    # Get portfolio recommendations
    print("\n4. Getting Portfolio Recommendations...")
    
    try:
        recommendations = optimizer.get_portfolio_recommendations(
            portfolio_value=100000.0
        )
        
        print(f"   Generated recommendations for {recommendations['timestamp']}")
        print(f"   Portfolio Value: ${recommendations['portfolio_value']:,.2f}")
        
        # Display top allocations
        print("\n   Top Stock Allocations:")
        sorted_allocations = sorted(
            recommendations['recommended_allocations'].items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )[:10]  # Top 10
        
        for symbol, allocation in sorted_allocations:
            print(f"     {symbol:<6} {allocation['weight']:.2%} "
                  f"(${allocation['dollar_amount']:>8,.0f}) "
                  f"- {allocation['sector']}")
        
        # Display sector allocation
        print("\n   Sector Allocation:")
        for sector, weight in recommendations['sector_allocations'].items():
            print(f"     {sector:<20} {weight:.2%}")
        
        # Risk metrics
        risk_metrics = recommendations['risk_metrics']
        print(f"\n   Risk Metrics:")
        print(f"     Portfolio Volatility: {risk_metrics['portfolio_volatility']:.2%}")
        print(f"     Portfolio Beta: {risk_metrics['portfolio_beta']:.2f}")
        print(f"     Sharpe Ratio Est.: {risk_metrics['sharpe_estimate']:.2f}")
        
    except Exception as e:
        print(f"   Error getting recommendations: {e}")
        print("   This is expected if training wasn't fully successful")
    
    # Sector Analysis
    print("\n5. Sector Analysis...")
    
    try:
        sector_analyzer = create_sector_analyzer(STOCKS_JSON_PATH, HISTORICAL_DATA_PATH)
        
        # Get sector recommendations
        sector_recs = sector_analyzer.get_top_sectors_recommendation(
            risk_tolerance=risk_profile['risk_tolerance'],
            investment_horizon=risk_profile['investment_horizon']
        )
        
        print("   Top Sector Recommendations:")
        for sector, info in sector_recs['top_sectors'].items():
            print(f"     {sector:<20} {info['recommended_weight']:.2%} "
                  f"- {info['rationale']}")
        
    except Exception as e:
        print(f"   Error in sector analysis: {e}")
    
    # Investment Insights
    print("\n6. Investment Insights...")
    
    try:
        insights = optimizer.get_investment_insights()
        
        print("   Market Analysis:")
        market_analysis = insights['market_analysis']
        print(f"     Market Trend: {market_analysis['market_trend']}")
        print(f"     Volatility Regime: {market_analysis['volatility_regime']}")
        
        print("\n   Investment Recommendations:")
        for rec in insights['recommendations'][:3]:  # Top 3
            print(f"     • {rec}")
        
    except Exception as e:
        print(f"   Error getting insights: {e}")
    
    # Save the trained system
    print("\n7. Saving Trained System...")
    
    try:
        save_info = optimizer.save_system("portfolio_optimizer_trained")
        print(f"   System saved with timestamp: {save_info['timestamp']}")
        
        if save_info['lstm_model']:
            print(f"   LSTM Model: {save_info['lstm_model']}")
        if save_info['ppo_model']:
            print(f"   PPO Model: {save_info['ppo_model']}")
        print(f"   Configuration: {save_info['config']}")
        
    except Exception as e:
        print(f"   Error saving system: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed! Check the generated files for detailed results.")
    print("=" * 60)


def demonstrate_advanced_features():
    """Demonstrate advanced features of the system"""
    print("\n" + "=" * 60)
    print("Advanced Features Demonstration")
    print("=" * 60)
    
    # This would demonstrate features like:
    # - Dynamic rebalancing
    # - Risk management
    # - Performance attribution
    # - Backtesting capabilities
    
    print("\nAdvanced features available:")
    print("• Dynamic Portfolio Rebalancing")
    print("• Risk-Adjusted Performance Optimization") 
    print("• Sector Rotation Strategies")
    print("• Multi-Objective Optimization")
    print("• Real-time Market Adaptation")
    print("• ESG Integration Capabilities")
    print("• Performance Attribution Analysis")
    print("• Stress Testing and Scenario Analysis")


def create_custom_risk_profiles():
    """Create example custom risk profiles"""
    
    risk_profiles = {
        'conservative_retiree': {
            'risk_tolerance': 'conservative',
            'investment_horizon': 'medium',
            'preferred_sectors': ['Consumer Defensive', 'Healthcare', 'Utilities'],
            'excluded_sectors': ['Energy', 'Basic Materials'],
            'target_volatility': 0.10,
            'max_drawdown_tolerance': 0.10,
            'min_dividend_yield': 0.03,
            'max_single_position': 0.10
        },
        
        'aggressive_growth': {
            'risk_tolerance': 'aggressive',
            'investment_horizon': 'long',
            'preferred_sectors': ['Technology', 'Healthcare', 'Consumer Cyclical'],
            'excluded_sectors': ['Utilities'],
            'target_volatility': 0.25,
            'max_drawdown_tolerance': 0.35,
            'min_dividend_yield': 0.00,
            'max_single_position': 0.20
        },
        
        'balanced_growth': {
            'risk_tolerance': 'moderate',
            'investment_horizon': 'long',
            'preferred_sectors': [],
            'excluded_sectors': [],
            'target_volatility': 0.15,
            'max_drawdown_tolerance': 0.20,
            'min_dividend_yield': 0.01,
            'max_single_position': 0.15
        },
        
        'esg_focused': {
            'risk_tolerance': 'moderate',
            'investment_horizon': 'long',
            'preferred_sectors': ['Technology', 'Healthcare', 'Consumer Defensive'],
            'excluded_sectors': ['Energy', 'Basic Materials'],
            'target_volatility': 0.18,
            'max_drawdown_tolerance': 0.25,
            'esg_filter': True,
            'carbon_intensity_limit': 100,
            'max_single_position': 0.12
        }
    }
    
    return risk_profiles


if __name__ == "__main__":
    try:
        main()
        demonstrate_advanced_features()
        
        print("\n" + "=" * 60)
        print("Custom Risk Profiles Available:")
        profiles = create_custom_risk_profiles()
        for name, profile in profiles.items():
            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  Risk Tolerance: {profile['risk_tolerance']}")
            print(f"  Target Volatility: {profile['target_volatility']:.1%}")
            print(f"  Max Drawdown: {profile['max_drawdown_tolerance']:.1%}")
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print("This is normal if data files are not available.")
        print("The system is designed to work with real market data.")
    
    print("\nFor production use:")
    print("1. Ensure all data files are available and updated")
    print("2. Increase training epochs for better performance")
    print("3. Implement real-time data feeds")
    print("4. Add proper error handling and logging")
    print("5. Integrate with your portfolio management system")