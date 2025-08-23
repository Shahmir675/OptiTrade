import json
from datetime import datetime, time
from decimal import ROUND_HALF_UP, Decimal

import aiohttp
import pytz
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.sqlalchemy_models import (
    Order,
    Portfolio,
    StopLossOrder,
    Transaction,
    UserModel,
)

load_dotenv(dotenv_path="/home/shahmir/Backend/OptiTrade/.env.test")

stocks = []
with open(r"app/static/stocks.json", "r") as file:
    for line in file:
        stock = json.loads(line)
        stocks.append(stock)

MIN_TICK_SIZE = Decimal("0.50")
ABSOLUTE_MAX_PRICE = Decimal("10000")
PRICE_TOLERANCE = Decimal("0.10")


def is_trading_time():
    pkt = pytz.timezone("Asia/Karachi")
    et = pytz.timezone("US/Eastern")
    now_pkt = datetime.now(pkt)
    now_et = now_pkt.astimezone(et)

    market_open = time(9, 30)
    market_close = time(16, 0)

    if now_et.weekday() >= 5:
        return False

    if now_pkt.weekday() == 4:
        if now_pkt.time() >= time(19, 30) or now_pkt.time() < time(2, 0):
            return True

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=str(now_et.year) + "-01-01", end=str(now_et.year) + "-12-31"
    ).to_pydatetime()
    if now_et.date() in [h.date() for h in holidays]:
        return False

    return market_open <= now_et.time() <= market_close


def validate_limit_price(limit_price, current_price, order_type):
    limit_price = Decimal(limit_price)
    current_price = Decimal(current_price)

    lower_bound = (current_price * (1 - PRICE_TOLERANCE)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    upper_bound = (current_price * (1 + PRICE_TOLERANCE)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    if limit_price < MIN_TICK_SIZE:
        raise ValueError(
            f"Limit price {limit_price} is below the minimum tick size of {MIN_TICK_SIZE}."
        )
    if limit_price > ABSOLUTE_MAX_PRICE:
        raise ValueError(
            f"Limit price {limit_price} exceeds the maximum allowable price of {ABSOLUTE_MAX_PRICE}."
        )

    if order_type == "buy" and limit_price > current_price:
        raise ValueError(
            f"Buy limit price {limit_price} must be at or below the current price {current_price}."
        )
    if order_type == "sell" and limit_price < current_price:
        raise ValueError(
            f"Sell limit price {limit_price} must be at or above the current price {current_price}."
        )

    if not (lower_bound <= limit_price <= upper_bound) and order_type == "buy":
        raise ValueError(
            f"Limit price {limit_price} is outside the acceptable range of {lower_bound} to {current_price}."
        )

    if not (lower_bound <= limit_price <= upper_bound) and order_type == "sell":
        raise ValueError(
            f"Limit price {limit_price} is outside the acceptable range of {current_price} to {upper_bound}."
        )


async def get_user_by_id(user_id: int, db: AsyncSession):
    result = await db.execute(
        select(UserModel)
        .where(UserModel.id == user_id)
        .options(selectinload(UserModel.balance))
    )
    return result.scalar_one_or_none()


async def _get_current_price(symbol: str) -> Decimal:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://archlinux.tail9023a4.ts.net/stocks/prices"
        ) as response:
            response.raise_for_status()
            data = await response.json()
            price = next(
                (
                    stock_info["price"]
                    for stock_info in data
                    if stock_info["symbol"] == symbol
                ),
                None,
            )
            if price is None:
                raise ValueError(f"Symbol {symbol} not found in price feed.")
            return Decimal(str(price))


async def store_limit_order(
    user_id: int,
    symbol: str,
    quantity: int,
    limit_price: Decimal,
    order_type: str,
    db: AsyncSession,
):
    new_order = Order(
        user_id=user_id,
        symbol=symbol,
        order_type=order_type,
        price=limit_price,
        quantity=quantity,
        timestamp=datetime.now(),
        order_status=False,
        filled_quantity=0,
        remaining_quantity=quantity,
    )
    db.add(new_order)
    await db.commit()

    if order_type == "buy":
        return f"Limit order placed: Buy {quantity} shares of {symbol} at or below {limit_price} per share."
    elif order_type == "sell":
        return f"Limit order placed: Sell {quantity} shares of {symbol} at or above {limit_price} per share."


async def store_stop_loss_order(
    user_id: int,
    symbol: str,
    quantity: int,
    stop_price: Decimal,
    order_type: str,
    db: AsyncSession,
):
    try:
        current_price = await _get_current_price(symbol)
        stop_price = Decimal(stop_price)

        if order_type == "sell" and stop_price >= current_price:
            raise ValueError("Stop-loss sell price must be BELOW current market price.")
        if order_type == "buy" and stop_price <= current_price:
            raise ValueError("Stop-loss buy price must be ABOVE current market price.")

        new_order = StopLossOrder(
            user_id=user_id,
            symbol=symbol,
            order_type=order_type,
            stop_price=stop_price,
            quantity=quantity,
            timestamp=datetime.now(),
            order_status=False,
            filled_quantity=0,
        )
        db.add(new_order)
        await db.commit()

        return f"Stop-loss order created: {order_type} {quantity} {symbol} @ {stop_price}"

    except Exception as e:
        await db.rollback()
        raise Exception(f"Error storing stop-loss order: {str(e)}")


async def process_stop_loss_orders(db: AsyncSession):
    pending_orders_result = await db.execute(
        select(StopLossOrder).where(StopLossOrder.order_status == False)
    )
    pending_orders = pending_orders_result.scalars().all()

    for order in pending_orders:
        try:
            current_price = await _get_current_price(order.symbol)
            triggered = False
            if order.order_type == "sell" and current_price <= order.stop_price:
                await sell_stock(
                    order.user_id,
                    order.symbol,
                    order.remaining_quantity,
                    db,
                    order_type="market",
                )
                triggered = True
            elif order.order_type == "buy" and current_price >= order.stop_price:
                await buy_stock(
                    order.user_id,
                    order.symbol,
                    order.remaining_quantity,
                    db,
                    order_type="market",
                )
                triggered = True

            if triggered:
                order.order_status = True
                order.filled_quantity = order.remaining_quantity
                order.remaining_quantity = 0
                await db.commit()

        except Exception as e:
            await db.rollback()
            print(f"Failed to process stop loss order {order.order_id}: {str(e)}")
            continue

async def process_limit_orders(db: AsyncSession):
    pending_orders_result = await db.execute(
        select(Order).where(Order.order_status == False)
    )
    pending_orders = pending_orders_result.scalars().all()
    
    for order in pending_orders:
        try:
            current_price = await _get_current_price(order.symbol)
            triggered = False
            
            if order.order_type == "buy" and current_price <= order.price:
                await buy_stock(order.user_id, order.symbol, order.remaining_quantity, db, order_type="market")
                triggered = True
            elif order.order_type == "sell" and current_price >= order.price:
                await sell_stock(order.user_id, order.symbol, order.remaining_quantity, db, order_type="market")
                triggered = True

            if triggered:
                order.order_status = True
                order.filled_quantity = order.quantity
                await db.commit()
        except Exception:
            await db.rollback()
            continue

async def buy_stock(
    user_id: int,
    symbol: str,
    quantity: int,
    db: AsyncSession,
    order_type: str = "market",
    limit_price: Decimal | None = None,
):
    current_price = await _get_current_price(symbol)

    user = await get_user_by_id(user_id, db)
    if not user or not user.balance:
        raise Exception(f"User with ID {user_id} not found or has no balance entry.")

    user_balance = user.balance
    u_balance = Decimal(user_balance.cash_balance).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    u_portfolio = Decimal(user_balance.portfolio_value)

    portfolio_item = await db.get(Portfolio, (user_id, symbol))

    if order_type == "market":
        # if not is_trading_time():
        #     raise Exception("Only limit orders can be placed outside market hours.")

        total_cost = (Decimal(quantity) * current_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        if u_balance < total_cost:
            raise Exception("Insufficient funds")

        if portfolio_item:
            portfolio_quantity = portfolio_item.quantity
            portfolio_avg_price = Decimal(portfolio_item.average_price)
            portfolio_total_invested = Decimal(portfolio_item.total_invested)

            new_quantity = portfolio_quantity + quantity
            portfolio_item.average_price = (
                (portfolio_quantity * portfolio_avg_price + total_cost) / new_quantity
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            portfolio_item.total_invested = (
                portfolio_total_invested + total_cost
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            portfolio_item.quantity = new_quantity
            portfolio_item.current_value = new_quantity * current_price
        else:
            new_portfolio_item = Portfolio(
                user_id=user_id,
                symbol=symbol,
                quantity=quantity,
                average_price=current_price,
                current_value=quantity * current_price,
                total_invested=total_cost,
            )
            db.add(new_portfolio_item)

        new_cash_balance = (u_balance - total_cost).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        new_portfolio_value = (u_portfolio + quantity * current_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        user_balance.cash_balance = new_cash_balance
        user_balance.portfolio_value = new_portfolio_value
        user_balance.net_worth = (new_cash_balance + new_portfolio_value).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        transaction = Transaction(
            user_id=user_id,
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            transaction_type="buy",
            price_per_share=current_price,
            total_price=total_cost,
        )
        db.add(transaction)
        await db.commit()
        return f"Successfully bought {quantity} shares of {symbol} at {current_price} per share."

    elif order_type == "limit":
        if limit_price is None or limit_price <= 0:
            raise ValueError("Limit price must be specified and greater than zero.")
        if quantity <= 0:
            raise ValueError("Quantity must be a positive value.")

        limit_price = Decimal(limit_price)
        validate_limit_price(limit_price, current_price, "buy")

        if current_price <= limit_price:
            total_cost = (Decimal(quantity) * current_price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if u_balance < total_cost:
                raise Exception("Insufficient funds")

            if portfolio_item:
                portfolio_quantity = portfolio_item.quantity
                portfolio_avg_price = Decimal(portfolio_item.average_price)
                portfolio_total_invested = Decimal(portfolio_item.total_invested)
                new_quantity = portfolio_quantity + quantity
                portfolio_item.average_price = (
                    (portfolio_quantity * portfolio_avg_price + total_cost)
                    / new_quantity
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                portfolio_item.total_invested = (
                    portfolio_total_invested + total_cost
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                portfolio_item.quantity = new_quantity
                portfolio_item.current_value = new_quantity * current_price
            else:
                new_portfolio_item = Portfolio(
                    user_id=user_id,
                    symbol=symbol,
                    quantity=quantity,
                    average_price=current_price,
                    current_value=quantity * current_price,
                    total_invested=total_cost,
                )
                db.add(new_portfolio_item)

            new_cash_balance = (u_balance - total_cost).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            new_portfolio_value = (u_portfolio + quantity * current_price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            user_balance.cash_balance = new_cash_balance
            user_balance.portfolio_value = new_portfolio_value
            user_balance.net_worth = (new_cash_balance + new_portfolio_value).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            transaction = Transaction(
                user_id=user_id,
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                transaction_type="buy",
                price_per_share=current_price,
                total_price=total_cost,
            )
            db.add(transaction)
            await db.commit()
            return f"Limit order executed: {quantity} shares of {symbol} bought at {current_price} per share."
        else:
            return await store_limit_order(
                user_id, symbol, quantity, limit_price, "buy", db
            )


async def sell_stock(
    user_id: int,
    symbol: str,
    quantity: int,
    db: AsyncSession,
    order_type: str = "market",
    limit_price: Decimal | None = None,
):
    current_price = await _get_current_price(symbol)

    user = await get_user_by_id(user_id, db)
    if not user or not user.balance:
        raise Exception(f"User with ID {user_id} not found or has no balance entry.")

    user_balance = user.balance
    u_balance = Decimal(user_balance.cash_balance)
    u_portfolio = Decimal(user_balance.portfolio_value)

    portfolio_item = await db.get(Portfolio, (user_id, symbol))
    if not portfolio_item or portfolio_item.quantity < quantity:
        raise Exception("Not enough shares to sell")

    if order_type == "market":
        # if not is_trading_time():
        #     raise Exception("Only limit orders can be placed outside market hours.")

        total_sale = (Decimal(quantity) * current_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        new_quantity = portfolio_item.quantity - quantity

        if new_quantity > 0:
            portfolio_item.total_invested = (
                new_quantity * portfolio_item.average_price
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            portfolio_item.current_value = (new_quantity * current_price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            portfolio_item.quantity = new_quantity
        else:
            await db.delete(portfolio_item)

        new_cash_balance = (u_balance + total_sale).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        new_portfolio_value = (u_portfolio - quantity * current_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        user_balance.cash_balance = new_cash_balance
        user_balance.portfolio_value = new_portfolio_value
        user_balance.net_worth = (new_cash_balance + new_portfolio_value).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        transaction = Transaction(
            user_id=user_id,
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            limit_price=None,
            transaction_type="sell",
            price_per_share=current_price,
            total_price=total_sale,
        )
        db.add(transaction)
        await db.commit()
        return f"Successfully sold {quantity} shares of {symbol} at {current_price} per share."

    elif order_type == "limit":
        if limit_price is None or limit_price <= 0:
            raise ValueError("Limit price must be specified and greater than zero.")
        if quantity <= 0:
            raise ValueError("Quantity must be a positive value.")

        limit_price = Decimal(limit_price)
        validate_limit_price(limit_price, current_price, "sell")

        if current_price >= limit_price:
            total_sale = (Decimal(quantity) * current_price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            new_quantity = portfolio_item.quantity - quantity

            if new_quantity > 0:
                portfolio_item.total_invested = (
                    new_quantity * portfolio_item.average_price
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                portfolio_item.current_value = (new_quantity * current_price).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                portfolio_item.quantity = new_quantity
            else:
                await db.delete(portfolio_item)

            new_cash_balance = (u_balance + total_sale).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            new_portfolio_value = (u_portfolio - quantity * current_price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            user_balance.cash_balance = new_cash_balance
            user_balance.portfolio_value = new_portfolio_value
            user_balance.net_worth = (new_cash_balance + new_portfolio_value).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            transaction = Transaction(
                user_id=user_id,
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                transaction_type="sell",
                price_per_share=current_price,
                total_price=total_sale,
            )
            db.add(transaction)
            await db.commit()
            return f"Limit order executed: {quantity} shares of {symbol} sold at {current_price} per share."
        else:
            return await store_limit_order(
                user_id, symbol, quantity, limit_price, "sell", db
            )
