# OptiTrade - Trading app using Deep Learning and Reinforcement Learning for portfolio optimization

## 1. Introduction
In today’s global age, financial markets play a crucial role in the efficient allocation of capital among nations, corporations, and investors. They facilitate resource flow by connecting those who need funding with those who have capital to invest. Financial markets include stock markets, bond markets, commodity markets, cryptocurrency markets, and foreign exchange markets. These markets foster economic growth by providing liquidity, enabling risk management, and offering investment opportunities.

### Stock Market
The stock market is a key component of the financial system where shares of publicly traded companies are bought and sold. It allows companies to raise capital by issuing shares, representing ownership stakes. Investors purchase these shares with the expectation of earning returns through dividends and capital appreciation.

### Stock Trading
Stock trading involves buying and selling shares in the stock market. It facilitates capital movement and allows investors to profit from market fluctuations. Successful stock trading requires tools and strategies to manage risks and optimize returns, emphasizing the need for advanced trading applications.

### Portfolio Optimization
Portfolio optimization determines the best mix of assets to achieve the highest possible return for a given risk level. Using Modern Portfolio Theory (MPT), investors diversify portfolios to manage risk effectively. Mathematical models and algorithms further refine portfolio strategies to improve performance.

## 1.1 Overview
This project aims to develop **OptiTrade**, a comprehensive stock trading app integrating advanced features for effective investment management. The app will provide real-time market data for NASDAQ-listed stocks, advanced charting tools, order management, portfolio management, and optimization. It will also offer alerts on market movements and provide strategic suggestions for better investment decisions.

## 1.2 Background
Traditional investment models, such as Modern Portfolio Theory (MPT), emphasize diversification to balance risk and return. However, these methods often rely on historical data and static assumptions, limiting their adaptability to dynamic markets.

## 1.3 Motivation
This project addresses the limitations of traditional investment methods by offering a robust solution for managing investments. It aims to overcome challenges such as emotional decision-making, market volatility, and information overload by providing real-time strategic insights.

## 1.4 Problem Statement
Traditional portfolio optimization relies on historical data and static models, which may underperform in dynamic market conditions. Machine learning (ML) and deep learning (DL) show promise but face challenges in data requirements and interpretability.

## 1.5 Aims and Objectives
The primary objectives of OptiTrade include:
- Real-time market data for NASDAQ stocks.
- Advanced charting tools for technical analysis.
- Order management with features like market orders, limit orders, and stop-loss.
- Portfolio management and optimization tools.
- Alerts on market movements and portfolio updates.
- Strategic suggestions for improving investment decisions.
- A user-friendly interface for both novice and veteran traders.

## 1.6 Significance of the Study
This project addresses the shortcomings of traditional investment models and offers a practical solution for managing investments in dynamic markets. The app aims to improve decision-making, risk management, and investment performance through advanced features and optimization techniques.

## 2. Literature Review
Modern Portfolio Theory (MPT), introduced by Harry Markowitz, emphasizes portfolio diversification to manage risk. Strategies like Strategic Asset Allocation (SAA) and Tactical Asset Allocation (TAA) are used to balance long-term investment goals with short-term market opportunities. Methods like mean-variance optimization and the Capital Asset Pricing Model (CAPM) extend these ideas but face limitations in dynamic markets.

Machine learning (ML) and artificial intelligence (AI) methods have advanced portfolio management. Techniques such as neural networks, genetic algorithms, and reinforcement learning (RL) allow for dynamic adaptation to market conditions, improving decision-making and portfolio optimization.

## 3. Methodology

### Kanban Methodology for Project Management
OptiTrade will implement the Kanban methodology using Notion for project management. The board will include columns such as Backlog, To Do, In Progress, Testing, Review, and Done to track tasks, from frontend features like chart design to backend API connectivity and AI model development.

### 3.1 Proposed Solution
OptiTrade offers real-time market data, advanced charting tools, and dynamic portfolio management using a modified Deep Q-Network (DQN) and Long Short-Term Memory (LSTM) models. The app also provides alerts, notifications, and strategic insights to help users manage risks and optimize returns.

### 3.2 Tools and Techniques
- **Frontend Development**: React, Next.js, React Native
- **Backend Development**: Node.js, Express.js, PostgreSQL
- **API Development**: Django Rest Framework (DRF) or FastAPI
- **AI Components**: PyTorch, LSTM models, DQN for portfolio optimization

### 3.3 Work Plan
The project will be executed in six phases:
1. **Project Initialization**: Define scope, objectives, and team structure.
2. **Research and Design**: Conduct literature review, feasibility analysis, and design system architecture.
3. **Development**: Frontend, backend, and AI components development.
4. **Integration and Testing**: Integrate components and perform functional, performance, and user acceptance testing.
5. **Deployment and Evaluation**: Deploy the app, collect feedback, and optimize functionality.
6. **Documentation and Final Report**: Prepare user manuals, technical specifications, and final project report.

## 4. Limitations
- **Regulatory Approval**: Real-world stock trading requires regulatory approval.
- **Brokerage and Transaction Costs**: Real-world transactions incur costs, so the app will utilize paper trading for testing.
- **Synthetic Order Books**: Order books in the app will be synthetic, mirroring real market conditions but not fully replicating live trading complexities.

## 5. Conclusion
OptiTrade addresses the limitations of traditional investment models by incorporating real-time data, dynamic portfolio optimization, and advanced AI techniques like DQN and LSTM. By offering comprehensive tools for portfolio management, the app supports investors in making informed decisions, managing risks, and optimizing returns.
