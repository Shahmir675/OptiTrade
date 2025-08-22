#!/usr/bin/env python3
"""
Test script for Portfolio Analytics API endpoints
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = 1

def test_analytics_endpoints():
    """Test all analytics endpoints"""
    
    print("🧪 Testing Portfolio Analytics API Endpoints")
    print("=" * 50)
    
    # Test health check first
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/analytics/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test VaR endpoint (GET)
    print("\n2. Testing VaR Calculation (GET)...")
    try:
        response = requests.get(
            f"{BASE_URL}/analytics/var/{TEST_USER_ID}",
            params={
                "confidence_level": 0.95,
                "time_horizon_days": 1,
                "historical_days": 252
            }
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ VaR calculation successful")
            print(f"   VaR: {data.get('var_percentage', 'N/A'):.2%}")
            print(f"   Dollar VaR: ${data.get('var_dollar', 0):,.2f}")
        else:
            print(f"❌ VaR calculation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ VaR calculation error: {e}")
    
    # Test VaR endpoint (POST)
    print("\n3. Testing VaR Calculation (POST)...")
    try:
        payload = {
            "user_id": TEST_USER_ID,
            "confidence_level": 0.95,
            "time_horizon_days": 1,
            "historical_days": 252
        }
        response = requests.post(
            f"{BASE_URL}/analytics/var",
            json=payload
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ VaR calculation (POST) successful")
            print(f"   VaR: {data.get('var_percentage', 'N/A'):.2%}")
        else:
            print(f"❌ VaR calculation (POST) failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ VaR calculation (POST) error: {e}")
    
    # Test Maximum Drawdown
    print("\n4. Testing Maximum Drawdown...")
    try:
        response = requests.get(f"{BASE_URL}/analytics/max-drawdown/{TEST_USER_ID}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Maximum Drawdown calculation successful")
            print(f"   Max Drawdown: {data.get('max_drawdown_percentage', 'N/A'):.2%}")
            print(f"   Duration: {data.get('drawdown_duration_days', 'N/A')} days")
        else:
            print(f"❌ Maximum Drawdown failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Maximum Drawdown error: {e}")
    
    # Test Sharpe Ratio
    print("\n5. Testing Sharpe Ratio...")
    try:
        response = requests.get(
            f"{BASE_URL}/analytics/sharpe-ratio/{TEST_USER_ID}",
            params={"risk_free_rate": 0.02}
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Sharpe Ratio calculation successful")
            print(f"   Sharpe Ratio: {data.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"   Annual Return: {data.get('annualized_return', 'N/A'):.2%}")
        else:
            print(f"❌ Sharpe Ratio failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Sharpe Ratio error: {e}")
    
    # Test Beta
    print("\n6. Testing Beta Calculation...")
    try:
        response = requests.get(
            f"{BASE_URL}/analytics/beta/{TEST_USER_ID}",
            params={"market_symbol": "SPY"}
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Beta calculation successful")
            print(f"   Beta: {data.get('beta', 'N/A'):.2f}")
            print(f"   Correlation: {data.get('correlation', 'N/A'):.2%}")
        else:
            print(f"❌ Beta calculation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Beta calculation error: {e}")
    
    # Test Portfolio Concentration
    print("\n7. Testing Portfolio Concentration...")
    try:
        response = requests.get(f"{BASE_URL}/analytics/concentration/{TEST_USER_ID}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Portfolio Concentration calculation successful")
            print(f"   HHI Score: {data.get('herfindahl_hirschman_index', 'N/A'):.3f}")
            print(f"   Concentration Level: {data.get('concentration_level', 'N/A')}")
            print(f"   Effective Holdings: {data.get('effective_number_of_holdings', 'N/A'):.1f}")
        else:
            print(f"❌ Portfolio Concentration failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Portfolio Concentration error: {e}")
    
    # Test Comprehensive Analytics
    print("\n8. Testing Comprehensive Analytics...")
    try:
        response = requests.get(f"{BASE_URL}/analytics/comprehensive/{TEST_USER_ID}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Comprehensive Analytics successful")
            print(f"   User ID: {data.get('user_id', 'N/A')}")
            print(f"   Calculation Date: {data.get('calculation_date', 'N/A')}")
            
            # Check which metrics were calculated successfully
            metrics = ['var', 'maximum_drawdown', 'sharpe_ratio', 'beta', 'concentration']
            for metric in metrics:
                if metric in data and not data[metric].get('error'):
                    print(f"   ✅ {metric.replace('_', ' ').title()}: Available")
                else:
                    print(f"   ❌ {metric.replace('_', ' ').title()}: {data.get(metric, {}).get('error', 'Missing')}")
        else:
            print(f"❌ Comprehensive Analytics failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Comprehensive Analytics error: {e}")
    
    # Test PDF Report
    print("\n9. Testing PDF Report Generation...")
    try:
        response = requests.get(f"{BASE_URL}/analytics/pdf-report/{TEST_USER_ID}")
        if response.status_code == 200:
            print("✅ PDF Report generation successful")
            print(f"   Content Type: {response.headers.get('content-type', 'N/A')}")
            print(f"   Content Length: {len(response.content)} bytes")
        else:
            print(f"❌ PDF Report failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ PDF Report error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Analytics API Testing Complete")

def test_frontend_integration():
    """Test frontend integration patterns"""
    
    print("\n🎨 Testing Frontend Integration Patterns")
    print("=" * 50)
    
    # Simulate frontend service usage
    class AnalyticsService:
        def __init__(self, base_url):
            self.base_url = base_url
        
        def get_portfolio_analytics(self, user_id, options=None):
            if options is None:
                options = {}
            
            params = {
                'confidence_level': options.get('confidenceLevel', 0.95),
                'risk_free_rate': options.get('riskFreeRate', 0.02),
                'market_symbol': options.get('marketSymbol', 'SPY'),
                'historical_days': options.get('historicalDays', 252)
            }
            
            response = requests.get(f"{self.base_url}/analytics/comprehensive/{user_id}", params=params)
            return response.json() if response.status_code == 200 else None
    
    # Test the service
    analytics_service = AnalyticsService(BASE_URL)
    
    print("\n1. Testing Frontend Service Pattern...")
    try:
        result = analytics_service.get_portfolio_analytics(
            TEST_USER_ID, 
            {
                'confidenceLevel': 0.99,
                'riskFreeRate': 0.025,
                'marketSymbol': 'QQQ',
                'historicalDays': 126
            }
        )
        
        if result:
            print("✅ Frontend service pattern works")
            print(f"   Retrieved data for user {result.get('user_id', 'N/A')}")
        else:
            print("❌ Frontend service pattern failed")
    except Exception as e:
        print(f"❌ Frontend service error: {e}")
    
    print("\n" + "=" * 50)
    print("🎨 Frontend Integration Testing Complete")

if __name__ == "__main__":
    print(f"🚀 Starting Analytics API Tests at {datetime.now()}")
    print(f"📍 Base URL: {BASE_URL}")
    print(f"👤 Test User ID: {TEST_USER_ID}")
    
    # Run tests
    test_analytics_endpoints()
    test_frontend_integration()
    
    print(f"\n✨ All tests completed at {datetime.now()}")
    print("\n📝 Notes:")
    print("   - Make sure the FastAPI server is running on localhost:8000")
    print("   - Ensure test user has portfolio data for meaningful results")
    print("   - Some metrics may fail if insufficient historical data exists")
