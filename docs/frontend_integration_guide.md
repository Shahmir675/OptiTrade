# Frontend Integration Guide for Portfolio Analytics

## Overview
This guide shows how to integrate the Portfolio Analytics API with your frontend application, what to display to users, and how to generate PDF reports.

## Backend API Integration

### 1. Frontend Service Layer (JavaScript/TypeScript)

```javascript
// services/analyticsService.js
class AnalyticsService {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async getPortfolioAnalytics(userId, options = {}) {
        const {
            confidenceLevel = 0.95,
            riskFreeRate = 0.02,
            marketSymbol = 'SPY',
            historicalDays = 252
        } = options;

        try {
            // Using GET method for simplicity
            const params = new URLSearchParams({
                confidence_level: confidenceLevel,
                risk_free_rate: riskFreeRate,
                market_symbol: marketSymbol,
                historical_days: historicalDays
            });

            const response = await fetch(
                `${this.baseURL}/analytics/comprehensive/${userId}?${params}`,
                {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.getAuthToken()}` // If using auth
                    }
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching analytics:', error);
            throw error;
        }
    }

    async getIndividualMetric(userId, metric, options = {}) {
        const endpoints = {
            'var': 'var',
            'drawdown': 'max-drawdown',
            'sharpe': 'sharpe-ratio',
            'beta': 'beta',
            'concentration': 'concentration'
        };

        const endpoint = endpoints[metric];
        if (!endpoint) {
            throw new Error(`Unknown metric: ${metric}`);
        }

        try {
            const params = new URLSearchParams(options);
            const response = await fetch(
                `${this.baseURL}/analytics/${endpoint}/${userId}?${params}`
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${metric}:`, error);
            throw error;
        }
    }

    getAuthToken() {
        // Return your auth token from localStorage, cookies, etc.
        return localStorage.getItem('authToken');
    }
}

export default new AnalyticsService();
```

### 2. React Component Example

```jsx
// components/PortfolioAnalytics.jsx
import React, { useState, useEffect } from 'react';
import analyticsService from '../services/analyticsService';
import AnalyticsCard from './AnalyticsCard';
import AnalyticsChart from './AnalyticsChart';
import PDFGenerator from './PDFGenerator';

const PortfolioAnalytics = ({ userId }) => {
    const [analytics, setAnalytics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [options, setOptions] = useState({
        confidenceLevel: 0.95,
        riskFreeRate: 0.02,
        marketSymbol: 'SPY',
        historicalDays: 252
    });

    useEffect(() => {
        fetchAnalytics();
    }, [userId, options]);

    const fetchAnalytics = async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await analyticsService.getPortfolioAnalytics(userId, options);
            setAnalytics(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleOptionsChange = (newOptions) => {
        setOptions({ ...options, ...newOptions });
    };

    if (loading) return <div className="loading">Loading analytics...</div>;
    if (error) return <div className="error">Error: {error}</div>;
    if (!analytics) return <div>No data available</div>;

    return (
        <div className="portfolio-analytics">
            <div className="analytics-header">
                <h2>Portfolio Risk & Performance Analytics</h2>
                <PDFGenerator analytics={analytics} userId={userId} />
            </div>

            <div className="analytics-controls">
                <AnalyticsControls 
                    options={options} 
                    onChange={handleOptionsChange} 
                />
            </div>

            <div className="analytics-grid">
                <AnalyticsCard
                    title="Value at Risk (VaR)"
                    data={analytics.var}
                    type="var"
                />
                <AnalyticsCard
                    title="Maximum Drawdown"
                    data={analytics.maximum_drawdown}
                    type="drawdown"
                />
                <AnalyticsCard
                    title="Sharpe Ratio"
                    data={analytics.sharpe_ratio}
                    type="sharpe"
                />
                <AnalyticsCard
                    title="Beta"
                    data={analytics.beta}
                    type="beta"
                />
                <AnalyticsCard
                    title="Portfolio Concentration"
                    data={analytics.concentration}
                    type="concentration"
                />
            </div>

            <div className="analytics-charts">
                <AnalyticsChart analytics={analytics} />
            </div>
        </div>
    );
};

export default PortfolioAnalytics;
```

### 3. Analytics Display Cards

```jsx
// components/AnalyticsCard.jsx
import React from 'react';

const AnalyticsCard = ({ title, data, type }) => {
    if (data?.error) {
        return (
            <div className="analytics-card error">
                <h3>{title}</h3>
                <p className="error-message">{data.error}</p>
            </div>
        );
    }

    const renderContent = () => {
        switch (type) {
            case 'var':
                return (
                    <div className="var-content">
                        <div className="main-metric">
                            <span className="value">{(data.var_percentage * 100).toFixed(2)}%</span>
                            <span className="label">Daily VaR</span>
                        </div>
                        <div className="sub-metrics">
                            <div className="metric">
                                <span className="value">${data.var_dollar?.toLocaleString()}</span>
                                <span className="label">Dollar Amount</span>
                            </div>
                            <div className="metric">
                                <span className="value">{(data.confidence_level * 100)}%</span>
                                <span className="label">Confidence Level</span>
                            </div>
                        </div>
                    </div>
                );

            case 'drawdown':
                return (
                    <div className="drawdown-content">
                        <div className="main-metric">
                            <span className="value">{(data.max_drawdown_percentage * 100).toFixed(2)}%</span>
                            <span className="label">Maximum Drawdown</span>
                        </div>
                        <div className="sub-metrics">
                            <div className="metric">
                                <span className="value">{data.drawdown_duration_days}</span>
                                <span className="label">Duration (days)</span>
                            </div>
                            <div className="metric">
                                <span className="value">{new Date(data.peak_date).toLocaleDateString()}</span>
                                <span className="label">Peak Date</span>
                            </div>
                        </div>
                    </div>
                );

            case 'sharpe':
                return (
                    <div className="sharpe-content">
                        <div className="main-metric">
                            <span className="value">{data.sharpe_ratio?.toFixed(2)}</span>
                            <span className="label">Sharpe Ratio</span>
                        </div>
                        <div className="sub-metrics">
                            <div className="metric">
                                <span className="value">{(data.annualized_return * 100).toFixed(2)}%</span>
                                <span className="label">Annual Return</span>
                            </div>
                            <div className="metric">
                                <span className="value">{(data.annualized_volatility * 100).toFixed(2)}%</span>
                                <span className="label">Volatility</span>
                            </div>
                        </div>
                    </div>
                );

            case 'beta':
                return (
                    <div className="beta-content">
                        <div className="main-metric">
                            <span className="value">{data.beta?.toFixed(2)}</span>
                            <span className="label">Beta vs {data.market_symbol}</span>
                        </div>
                        <div className="sub-metrics">
                            <div className="metric">
                                <span className="value">{(data.correlation * 100).toFixed(1)}%</span>
                                <span className="label">Correlation</span>
                            </div>
                            <div className="metric">
                                <span className="value">{(data.r_squared * 100).toFixed(1)}%</span>
                                <span className="label">R-Squared</span>
                            </div>
                        </div>
                    </div>
                );

            case 'concentration':
                return (
                    <div className="concentration-content">
                        <div className="main-metric">
                            <span className="value">{data.herfindahl_hirschman_index?.toFixed(3)}</span>
                            <span className="label">HHI Score</span>
                        </div>
                        <div className="sub-metrics">
                            <div className="metric">
                                <span className="value">{data.concentration_level}</span>
                                <span className="label">Risk Level</span>
                            </div>
                            <div className="metric">
                                <span className="value">{data.effective_number_of_holdings?.toFixed(1)}</span>
                                <span className="label">Effective Holdings</span>
                            </div>
                        </div>
                    </div>
                );

            default:
                return <div>No data available</div>;
        }
    };

    return (
        <div className={`analytics-card ${type}`}>
            <h3>{title}</h3>
            {renderContent()}
        </div>
    );
};

export default AnalyticsCard;

### 4. Analytics Controls Component

```jsx
// components/AnalyticsControls.jsx
import React from 'react';

const AnalyticsControls = ({ options, onChange }) => {
    const handleChange = (field, value) => {
        onChange({ [field]: value });
    };

    return (
        <div className="analytics-controls">
            <div className="control-group">
                <label>Confidence Level</label>
                <select
                    value={options.confidenceLevel}
                    onChange={(e) => handleChange('confidenceLevel', parseFloat(e.target.value))}
                >
                    <option value={0.90}>90%</option>
                    <option value={0.95}>95%</option>
                    <option value={0.99}>99%</option>
                </select>
            </div>

            <div className="control-group">
                <label>Risk-Free Rate</label>
                <input
                    type="number"
                    step="0.001"
                    min="0"
                    max="0.1"
                    value={options.riskFreeRate}
                    onChange={(e) => handleChange('riskFreeRate', parseFloat(e.target.value))}
                />
            </div>

            <div className="control-group">
                <label>Market Index</label>
                <select
                    value={options.marketSymbol}
                    onChange={(e) => handleChange('marketSymbol', e.target.value)}
                >
                    <option value="SPY">S&P 500 (SPY)</option>
                    <option value="QQQ">NASDAQ (QQQ)</option>
                    <option value="IWM">Russell 2000 (IWM)</option>
                    <option value="VTI">Total Market (VTI)</option>
                </select>
            </div>

            <div className="control-group">
                <label>Historical Period</label>
                <select
                    value={options.historicalDays}
                    onChange={(e) => handleChange('historicalDays', parseInt(e.target.value))}
                >
                    <option value={63}>3 Months</option>
                    <option value={126}>6 Months</option>
                    <option value={252}>1 Year</option>
                    <option value={504}>2 Years</option>
                </select>
            </div>
        </div>
    );
};

export default AnalyticsControls;
```

## PDF Generation

### 1. Frontend PDF Generation (using jsPDF)

```jsx
// components/PDFGenerator.jsx
import React from 'react';
import jsPDF from 'jspdf';
import 'jspdf-autotable';

const PDFGenerator = ({ analytics, userId }) => {
    const generatePDF = () => {
        const doc = new jsPDF();
        const pageWidth = doc.internal.pageSize.width;
        const margin = 20;

        // Header
        doc.setFontSize(20);
        doc.text('Portfolio Analytics Report', margin, 30);

        doc.setFontSize(12);
        doc.text(`User ID: ${userId}`, margin, 45);
        doc.text(`Generated: ${new Date().toLocaleDateString()}`, margin, 55);

        let yPosition = 70;

        // Value at Risk
        if (analytics.var && !analytics.var.error) {
            doc.setFontSize(16);
            doc.text('Value at Risk (VaR)', margin, yPosition);
            yPosition += 10;

            const varData = [
                ['Metric', 'Value'],
                ['Daily VaR (%)', (analytics.var.var_percentage * 100).toFixed(2) + '%'],
                ['VaR Amount ($)', '$' + analytics.var.var_dollar?.toLocaleString()],
                ['Confidence Level', (analytics.var.confidence_level * 100) + '%'],
                ['Portfolio Value', '$' + analytics.var.current_portfolio_value?.toLocaleString()],
                ['Daily Volatility', (analytics.var.daily_volatility * 100).toFixed(2) + '%']
            ];

            doc.autoTable({
                startY: yPosition,
                head: [varData[0]],
                body: varData.slice(1),
                margin: { left: margin, right: margin },
                styles: { fontSize: 10 }
            });

            yPosition = doc.lastAutoTable.finalY + 15;
        }

        // Maximum Drawdown
        if (analytics.maximum_drawdown && !analytics.maximum_drawdown.error) {
            doc.setFontSize(16);
            doc.text('Maximum Drawdown', margin, yPosition);
            yPosition += 10;

            const mddData = [
                ['Metric', 'Value'],
                ['Max Drawdown (%)', (analytics.maximum_drawdown.max_drawdown_percentage * 100).toFixed(2) + '%'],
                ['Peak Date', new Date(analytics.maximum_drawdown.peak_date).toLocaleDateString()],
                ['Trough Date', new Date(analytics.maximum_drawdown.trough_date).toLocaleDateString()],
                ['Duration (days)', analytics.maximum_drawdown.drawdown_duration_days.toString()],
                ['Peak Value', '$' + analytics.maximum_drawdown.peak_value?.toLocaleString()],
                ['Trough Value', '$' + analytics.maximum_drawdown.trough_value?.toLocaleString()]
            ];

            doc.autoTable({
                startY: yPosition,
                head: [mddData[0]],
                body: mddData.slice(1),
                margin: { left: margin, right: margin },
                styles: { fontSize: 10 }
            });

            yPosition = doc.lastAutoTable.finalY + 15;
        }

        // Add new page if needed
        if (yPosition > 250) {
            doc.addPage();
            yPosition = 30;
        }

        // Sharpe Ratio
        if (analytics.sharpe_ratio && !analytics.sharpe_ratio.error) {
            doc.setFontSize(16);
            doc.text('Sharpe Ratio', margin, yPosition);
            yPosition += 10;

            const sharpeData = [
                ['Metric', 'Value'],
                ['Sharpe Ratio', analytics.sharpe_ratio.sharpe_ratio?.toFixed(2)],
                ['Annual Return (%)', (analytics.sharpe_ratio.annualized_return * 100).toFixed(2) + '%'],
                ['Annual Volatility (%)', (analytics.sharpe_ratio.annualized_volatility * 100).toFixed(2) + '%'],
                ['Risk-Free Rate (%)', (analytics.sharpe_ratio.risk_free_rate * 100).toFixed(2) + '%'],
                ['Excess Return (%)', (analytics.sharpe_ratio.excess_return * 100).toFixed(2) + '%']
            ];

            doc.autoTable({
                startY: yPosition,
                head: [sharpeData[0]],
                body: sharpeData.slice(1),
                margin: { left: margin, right: margin },
                styles: { fontSize: 10 }
            });

            yPosition = doc.lastAutoTable.finalY + 15;
        }

        // Beta
        if (analytics.beta && !analytics.beta.error) {
            doc.setFontSize(16);
            doc.text('Portfolio Beta', margin, yPosition);
            yPosition += 10;

            const betaData = [
                ['Metric', 'Value'],
                ['Beta', analytics.beta.beta?.toFixed(2)],
                ['Market Symbol', analytics.beta.market_symbol],
                ['Correlation', (analytics.beta.correlation * 100).toFixed(1) + '%'],
                ['R-Squared', (analytics.beta.r_squared * 100).toFixed(1) + '%'],
                ['Portfolio Volatility', (analytics.beta.portfolio_volatility * 100).toFixed(2) + '%'],
                ['Market Volatility', (analytics.beta.market_volatility * 100).toFixed(2) + '%']
            ];

            doc.autoTable({
                startY: yPosition,
                head: [betaData[0]],
                body: betaData.slice(1),
                margin: { left: margin, right: margin },
                styles: { fontSize: 10 }
            });

            yPosition = doc.lastAutoTable.finalY + 15;
        }

        // Portfolio Concentration
        if (analytics.concentration && !analytics.concentration.error) {
            if (yPosition > 200) {
                doc.addPage();
                yPosition = 30;
            }

            doc.setFontSize(16);
            doc.text('Portfolio Concentration', margin, yPosition);
            yPosition += 10;

            const concentrationData = [
                ['Metric', 'Value'],
                ['HHI Score', analytics.concentration.herfindahl_hirschman_index?.toFixed(3)],
                ['Concentration Level', analytics.concentration.concentration_level],
                ['Effective Holdings', analytics.concentration.effective_number_of_holdings?.toFixed(1)],
                ['Actual Holdings', analytics.concentration.actual_number_of_holdings?.toString()],
                ['Top 5 Weight', (analytics.concentration.top_5_holdings_weight * 100).toFixed(1) + '%']
            ];

            doc.autoTable({
                startY: yPosition,
                head: [concentrationData[0]],
                body: concentrationData.slice(1),
                margin: { left: margin, right: margin },
                styles: { fontSize: 10 }
            });

            yPosition = doc.lastAutoTable.finalY + 15;

            // Top Holdings
            if (analytics.concentration.top_holdings?.length > 0) {
                doc.setFontSize(14);
                doc.text('Top Holdings', margin, yPosition);
                yPosition += 10;

                const holdingsData = [
                    ['Symbol', 'Value ($)', 'Weight (%)', 'Quantity']
                ];

                analytics.concentration.top_holdings.forEach(holding => {
                    holdingsData.push([
                        holding.symbol,
                        '$' + holding.value.toLocaleString(),
                        (holding.weight * 100).toFixed(1) + '%',
                        holding.quantity.toString()
                    ]);
                });

                doc.autoTable({
                    startY: yPosition,
                    head: [holdingsData[0]],
                    body: holdingsData.slice(1),
                    margin: { left: margin, right: margin },
                    styles: { fontSize: 9 }
                });
            }
        }

        // Footer
        const pageCount = doc.internal.getNumberOfPages();
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(10);
            doc.text(
                `Page ${i} of ${pageCount}`,
                pageWidth - margin - 30,
                doc.internal.pageSize.height - 10
            );
        }

        // Save the PDF
        doc.save(`portfolio-analytics-${userId}-${new Date().toISOString().split('T')[0]}.pdf`);
    };

    return (
        <button
            className="pdf-button"
            onClick={generatePDF}
            disabled={!analytics}
        >
            📄 Download PDF Report
        </button>
    );
};

export default PDFGenerator;
```

### 2. Backend PDF Generation Endpoint

```python
# Add to app/routers/analytics.py
from fastapi.responses import Response
from app.services.pdf_service import PDFReportGenerator

@router.get("/pdf-report/{user_id}")
async def generate_pdf_report(
    user_id: int,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.02,
    market_symbol: str = "SPY",
    historical_days: int = 252,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate and download PDF report for portfolio analytics
    """
    try:
        # Get comprehensive analytics
        analytics_data = await AnalyticsService.get_comprehensive_analytics(
            user_id=user_id,
            db=db,
            confidence_level=confidence_level,
            risk_free_rate=risk_free_rate,
            market_symbol=market_symbol,
            days=historical_days
        )

        # Generate PDF
        pdf_generator = PDFReportGenerator()
        pdf_content = pdf_generator.generate_analytics_report(analytics_data, user_id)

        # Return PDF response
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio-analytics-{user_id}.pdf"
            }
        )

    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating PDF report"
        )
```

## CSS Styling

### Analytics Dashboard Styles

```css
/* styles/analytics.css */
.portfolio-analytics {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.analytics-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #e0e0e0;
}

.analytics-header h2 {
    color: #2c3e50;
    margin: 0;
    font-size: 2rem;
}

.pdf-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}

.pdf-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.pdf-button:disabled {
    background: #cccccc;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.analytics-controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 12px;
    border: 1px solid #e9ecef;
}

.control-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.control-group label {
    font-weight: 600;
    color: #495057;
    font-size: 14px;
}

.control-group select,
.control-group input {
    padding: 10px 12px;
    border: 1px solid #ced4da;
    border-radius: 6px;
    font-size: 14px;
    transition: border-color 0.3s ease;
}

.control-group select:focus,
.control-group input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.analytics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 24px;
    margin-bottom: 40px;
}

.analytics-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    border: 1px solid #e9ecef;
    transition: all 0.3s ease;
}

.analytics-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.analytics-card h3 {
    margin: 0 0 20px 0;
    color: #2c3e50;
    font-size: 1.2rem;
    font-weight: 600;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 10px;
}

.analytics-card.var {
    border-left: 4px solid #e74c3c;
}

.analytics-card.drawdown {
    border-left: 4px solid #f39c12;
}

.analytics-card.sharpe {
    border-left: 4px solid #27ae60;
}

.analytics-card.beta {
    border-left: 4px solid #3498db;
}

.analytics-card.concentration {
    border-left: 4px solid #9b59b6;
}

.main-metric {
    text-align: center;
    margin-bottom: 20px;
}

.main-metric .value {
    display: block;
    font-size: 2.5rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 8px;
}

.main-metric .label {
    display: block;
    font-size: 14px;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sub-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.metric {
    text-align: center;
    padding: 12px;
    background: #f8f9fa;
    border-radius: 8px;
}

.metric .value {
    display: block;
    font-size: 1.1rem;
    font-weight: 600;
    color: #495057;
    margin-bottom: 4px;
}

.metric .label {
    display: block;
    font-size: 12px;
    color: #6c757d;
}

.analytics-card.error {
    border-left: 4px solid #dc3545;
    background: #fff5f5;
}

.error-message {
    color: #dc3545;
    font-style: italic;
    text-align: center;
    margin: 20px 0;
}

.loading {
    text-align: center;
    padding: 40px;
    font-size: 1.2rem;
    color: #6c757d;
}

.error {
    text-align: center;
    padding: 40px;
    color: #dc3545;
    background: #fff5f5;
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    margin: 20px 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .analytics-header {
        flex-direction: column;
        gap: 20px;
        text-align: center;
    }

    .analytics-controls {
        grid-template-columns: 1fr;
    }

    .analytics-grid {
        grid-template-columns: 1fr;
    }

    .sub-metrics {
        grid-template-columns: 1fr;
    }

    .main-metric .value {
        font-size: 2rem;
    }
}

/* Animation for loading states */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 2s infinite;
}
```

## Data Flow Summary

### Frontend to Backend Communication

1. **User Interaction**: User selects parameters in the frontend controls
2. **API Call**: Frontend makes GET/POST request to analytics endpoints
3. **Data Processing**: Backend calculates analytics using historical data
4. **Response**: Backend returns JSON with calculated metrics
5. **Display**: Frontend renders the data in cards and charts
6. **PDF Generation**: User can generate PDF report (frontend or backend)

### Key Data Points to Display

#### Value at Risk (VaR)
- **Primary**: VaR percentage (e.g., -2.5%)
- **Secondary**: Dollar amount, confidence level, portfolio value
- **Interpretation**: "5% chance of losing more than this amount daily"

#### Maximum Drawdown
- **Primary**: Maximum drawdown percentage (e.g., -15%)
- **Secondary**: Peak/trough dates, duration, recovery status
- **Interpretation**: "Worst decline from peak to trough"

#### Sharpe Ratio
- **Primary**: Sharpe ratio value (e.g., 1.25)
- **Secondary**: Annual return, volatility, excess return
- **Interpretation**: ">1.0 = Good, >2.0 = Excellent"

#### Beta
- **Primary**: Beta coefficient (e.g., 1.15)
- **Secondary**: Correlation, R-squared, market symbol
- **Interpretation**: ">1.0 = More volatile than market"

#### Portfolio Concentration
- **Primary**: HHI score (e.g., 0.18)
- **Secondary**: Concentration level, effective holdings
- **Interpretation**: "<0.15 = Well diversified"

### Error Handling

1. **Insufficient Data**: Show user-friendly message with suggestions
2. **API Errors**: Display error state with retry option
3. **Loading States**: Show loading indicators during calculations
4. **Partial Failures**: Display available metrics, note missing ones

This integration provides a complete solution for displaying portfolio analytics with professional styling and comprehensive error handling.
```
```
