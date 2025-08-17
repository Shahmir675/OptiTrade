import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from pytz import timezone
from server import Dividend, Portfolio, UserBalanceModel, get_stocks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

load_dotenv(dotenv_path="/home/shahmir/Backend/OptiTrade/.env.test")

PAKISTAN_TIME = timezone("Asia/Karachi")
DATABASE_URL = os.getenv("DATABASE_URL")


async def calculate_dividends():
    engine = create_async_engine(DATABASE_URL)
    async with AsyncSession(engine) as session:
        portfolio_query = select(Portfolio)
        result = await session.execute(portfolio_query)
        portfolios = result.scalars().all()

        for portfolio in portfolios:
            stock_data = get_stocks(portfolio.symbol)

            if not stock_data or not stock_data.get("last_dividend_amount"):
                continue

            ex_dividend_date = stock_data["last_dividend_date"]
            ex_dividend_date = datetime.strptime(ex_dividend_date, "%Y-%m-%d").date()
            frequency = stock_data["payment_frequency"]

            if not is_dividend_due(ex_dividend_date, frequency):
                continue

            dividend_check = await session.execute(
                select(Dividend).where(
                    Dividend.user_id == portfolio.user_id,
                    Dividend.stock_symbol == portfolio.symbol,
                    Dividend.ex_dividend_date == ex_dividend_date,
                )
            )

            if dividend_check.scalar():
                continue

            dividend_amount = portfolio.quantity * stock_data["last_dividend_amount"]

            user_balance = await session.execute(
                select(UserBalanceModel).where(
                    UserBalanceModel.user_id == portfolio.user_id
                )
            )
            balance = user_balance.scalar()
            print("dividend_amount", dividend_amount)
            balance.cash_balance += dividend_amount
            balance.net_worth += dividend_amount

            new_dividend = Dividend(
                user_id=portfolio.user_id,
                stock_symbol=portfolio.symbol,
                amount=dividend_amount,
                payment_date=datetime.now(PAKISTAN_TIME).strftime("%Y-%m-%d"),
                ex_dividend_date=ex_dividend_date,
            )
            session.add(new_dividend)

        print("Finished Successfully!")
        await session.commit()


def is_dividend_due(ex_dividend_date: datetime.date, frequency: str) -> bool:
    today = datetime.now(PAKISTAN_TIME).date()

    if frequency == "Quarterly":
        return (today - ex_dividend_date).days >= 90
    elif frequency == "Semi-Annually":
        return (today - ex_dividend_date).days >= 180
    elif frequency == "Annually":
        return (today - ex_dividend_date).days >= 365
    return False


if __name__ == "__main__":
    asyncio.run(calculate_dividends())
