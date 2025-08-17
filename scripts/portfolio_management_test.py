import json
import os
from datetime import datetime, time
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import urlparse

import psycopg2
import pytz
import requests
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar

load_dotenv(dotenv_path="/home/shahmir/Backend/OptiTrade/.env.test")
db_url = os.getenv("DATABASE_URL")
parsed_url = urlparse(db_url)

db_config = {
    "dbname": parsed_url.path[1:],
    "user": parsed_url.username,
    "password": parsed_url.password,
    "host": parsed_url.hostname,
    "port": parsed_url.port,
}

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
    if now_et.date() in holidays:
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


def get_user_by_id(user_id, conn):
    with conn.cursor() as cur:
        query = f"""
        SELECT * FROM users WHERE id={user_id}"""
        cur.execute(query)
        result = cur.fetchall()
        return result


def store_limit_order(user_id, symbol, quantity, limit_price, order_type):
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COALESCE(MAX(order_id), 0) + 1 FROM orders")
            order_id = cursor.fetchone()[0]

            order_query = """
                INSERT INTO orders (order_id, user_id, symbol, order_type, price, quantity, timestamp, order_status, filled_quantity, remaining_quantity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            cursor.execute(
                order_query,
                (
                    order_id,
                    user_id,
                    symbol,
                    order_type,
                    limit_price,
                    quantity,
                    datetime.now(),
                    False,
                    0,
                    quantity,
                ),
            )

            conn.commit()

            if order_type == "buy":
                return f"Limit order placed: Buy {quantity} shares of {symbol} at or below {limit_price} per share."
            elif order_type == "sell":
                return f"Limit order placed: Sell {quantity} shares of {symbol} at or above {limit_price} per share."


def store_stop_loss_order(user_id, symbol, quantity, stop_price, order_type):
    response = requests.get("https://archlinux.tail9023a4.ts.net/stocks/prices")
    data = response.json()
    current_price = [
        stock_info["price"] for stock_info in data if stock_info["symbol"] == symbol
    ][0]
    current_price = Decimal(str(current_price))

    triggered = False
    message = ""
    try:
        if order_type == "sell":
            if current_price <= stop_price:
                sell_stock(user_id, symbol, quantity, order_type="market")
                triggered = True
                message = f"Stop loss order executed: Sold {quantity} shares of {symbol} at market price {current_price}."
        elif order_type == "buy":
            if current_price >= stop_price:
                buy_stock(user_id, symbol, quantity, order_type="market")
                triggered = True
                message = f"Stop loss order executed: Bought {quantity} shares of {symbol} at market price {current_price}."
        else:
            raise ValueError("Invalid order_type. Must be 'buy' or 'sell'.")
    except Exception as e:
        return f"Error processing stop loss order: {str(e)}"

    if not triggered:
        try:
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT COALESCE(MAX(order_id), 0) + 1 FROM stop_loss_orders"
                    )
                    order_id = cursor.fetchone()[0]

                    insert_query = """
                        INSERT INTO stop_loss_orders 
                            (order_id, user_id, symbol, order_type, stop_price, quantity, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(
                        insert_query,
                        (
                            order_id,
                            user_id,
                            symbol,
                            order_type,
                            stop_price,
                            quantity,
                            datetime.now(),
                        ),
                    )
                    conn.commit()
                    message = f"Stop loss order placed: {order_type} {quantity} shares of {symbol} when price reaches {stop_price}."
        except Exception as e:
            message = f"Error storing stop loss order: {str(e)}"

    return message


def process_stop_loss_orders():
    """
    Process all pending stop loss orders by checking if their stop conditions are met with current prices.
    Executes market orders and updates the database accordingly.
    """
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT order_id, user_id, symbol, order_type, stop_price, remaining_quantity 
                FROM stop_loss_orders 
                WHERE order_status = FALSE
            """
            )
            pending_orders = cursor.fetchall()

            for order in pending_orders:
                (
                    order_id,
                    user_id,
                    symbol,
                    order_type,
                    stop_price,
                    remaining_quantity,
                ) = order

                response = requests.get(
                    "https://archlinux.tail9023a4.ts.net/stocks/prices"
                )
                data = response.json()
                current_price = [
                    stock_info["price"]
                    for stock_info in data
                    if stock_info["symbol"] == symbol
                ][0]
                current_price = Decimal(str(current_price))

                try:
                    if order_type == "sell" and current_price <= stop_price:
                        sell_stock(
                            user_id, symbol, remaining_quantity, order_type="market"
                        )
                        cursor.execute(
                            """
                            UPDATE stop_loss_orders 
                            SET order_status = TRUE, 
                                filled_quantity = remaining_quantity,
                                remaining_quantity = 0 
                            WHERE order_id = %s
                        """,
                            (order_id,),
                        )
                        conn.commit()
                    elif order_type == "buy" and current_price >= stop_price:
                        buy_stock(
                            user_id, symbol, remaining_quantity, order_type="market"
                        )
                        cursor.execute(
                            """
                            UPDATE stop_loss_orders 
                            SET order_status = TRUE, 
                                filled_quantity = remaining_quantity,
                                remaining_quantity = 0 
                            WHERE order_id = %s
                        """,
                            (order_id,),
                        )
                        conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"Failed to process stop loss order {order_id}: {str(e)}")
                    continue


def store_take_profit_order(user_id, symbol, quantity, take_profit_price, order_type):
    response = requests.get("https://archlinux.tail9023a4.ts.net/stocks/prices")
    data = response.json()
    current_price = [
        stock_info["price"] for stock_info in data if stock_info["symbol"] == symbol
    ][0]
    current_price = Decimal(str(current_price))

    triggered = False
    message = ""
    try:
        if order_type == "sell":
            if current_price >= take_profit_price:
                sell_stock(user_id, symbol, quantity, order_type="market")
                triggered = True
                message = f"Take profit executed: Sold {quantity} shares of {symbol} at {current_price} (target: {take_profit_price})."
        elif order_type == "buy":
            if current_price <= take_profit_price:
                buy_stock(user_id, symbol, quantity, order_type="market")
                triggered = True
                message = f"Take profit executed: Bought {quantity} shares of {symbol} at {current_price} (target: {take_profit_price})."
        else:
            raise ValueError("Invalid order_type. Must be 'buy' or 'sell'.")
    except Exception as e:
        return f"Error processing take profit order: {str(e)}"

    if not triggered:
        try:
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT COALESCE(MAX(order_id), 0) + 1 FROM take_profit_orders"
                    )
                    order_id = cursor.fetchone()[0]

                    insert_query = """
                        INSERT INTO take_profit_orders 
                            (order_id, user_id, symbol, order_type, take_profit_price, quantity, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(
                        insert_query,
                        (
                            order_id,
                            user_id,
                            symbol,
                            order_type,
                            take_profit_price,
                            quantity,
                            datetime.now(),
                        ),
                    )
                    conn.commit()
                    message = f"Take profit order placed: {order_type} {quantity} shares of {symbol} when price reaches {take_profit_price}."
        except Exception as e:
            message = f"Error storing take profit order: {str(e)}"

    return message


def process_take_profit_orders():
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT order_id, user_id, symbol, order_type, take_profit_price, remaining_quantity 
                FROM take_profit_orders 
                WHERE order_status = FALSE
            """
            )
            pending_orders = cursor.fetchall()

            for order in pending_orders:
                (
                    order_id,
                    user_id,
                    symbol,
                    order_type,
                    take_profit_price,
                    remaining_quantity,
                ) = order

                response = requests.get(
                    "https://archlinux.tail9023a4.ts.net/stocks/prices"
                )
                data = response.json()
                current_price = [
                    stock_info["price"]
                    for stock_info in data
                    if stock_info["symbol"] == symbol
                ][0]
                current_price = Decimal(str(current_price))

                try:
                    if order_type == "sell" and current_price >= take_profit_price:
                        sell_stock(
                            user_id, symbol, remaining_quantity, order_type="market"
                        )
                        cursor.execute(
                            """
                            UPDATE take_profit_orders 
                            SET order_status = TRUE, 
                                filled_quantity = remaining_quantity,
                                remaining_quantity = 0 
                            WHERE order_id = %s
                        """,
                            (order_id,),
                        )
                        conn.commit()
                    elif order_type == "buy" and current_price <= take_profit_price:
                        buy_stock(
                            user_id, symbol, remaining_quantity, order_type="market"
                        )
                        cursor.execute(
                            """
                            UPDATE take_profit_orders 
                            SET order_status = TRUE, 
                                filled_quantity = remaining_quantity,
                                remaining_quantity = 0 
                            WHERE order_id = %s
                        """,
                            (order_id,),
                        )
                        conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"Failed to process take profit order {order_id}: {str(e)}")
                    continue


def buy_stock(user_id, symbol, quantity, order_type="market", limit_price=None):
    with psycopg2.connect(**db_config) as conn:
        response = requests.get("https://archlinux.tail9023a4.ts.net/stocks/prices")
        data = response.json()
        current_price = [
            stock_info["price"] for stock_info in data if stock_info["symbol"] == symbol
        ][0]
        current_price = Decimal(str(current_price))

        with conn.cursor() as cursor:
            user_query = """
                SELECT u.id, u.u_name, ub.cash_balance, ub.portfolio_value, ub.net_worth
                FROM users u
                JOIN user_balance ub ON u.id = ub.user_id
                WHERE u.id = %s;
            """
            cursor.execute(user_query, (user_id,))
            user_id, u_name, u_balance, u_portfolio, u_net_worth = cursor.fetchone()

            u_balance = Decimal(u_balance).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            portfolio_query = """
                SELECT * FROM portfolio WHERE user_id=%s AND symbol=%s;
            """
            cursor.execute(portfolio_query, (user_id, symbol))
            portfolio_row = cursor.fetchone()

            print(f"Order type: {order_type}")

            if order_type == "market":
                # if not is_trading_time():
                #     raise Exception("Only limit orders can be placed outside market hours.")

                total_cost = (Decimal(quantity) * current_price).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                if u_balance < total_cost:
                    raise Exception("Insufficient funds")

                if portfolio_row:
                    portfolio_quantity = portfolio_row[2]
                    portfolio_avg_price = Decimal(portfolio_row[3])
                    portfolio_total_invested = Decimal(portfolio_row[4])

                    new_quantity = portfolio_quantity + quantity
                    new_avg_price = (
                        (portfolio_quantity * portfolio_avg_price + total_cost)
                        / new_quantity
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    new_total_invested = (
                        portfolio_total_invested + total_cost
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                    update_portfolio_query = """
                        UPDATE portfolio
                        SET quantity = %s, average_price = %s, current_value = %s, total_invested = %s
                        WHERE user_id = %s AND symbol = %s;
                    """
                    cursor.execute(
                        update_portfolio_query,
                        (
                            new_quantity,
                            new_avg_price,
                            new_quantity * current_price,
                            new_total_invested,
                            user_id,
                            symbol,
                        ),
                    )
                else:
                    insert_portfolio_query = """
                        INSERT INTO portfolio (user_id, symbol, quantity, average_price, current_value, total_invested)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """
                    cursor.execute(
                        insert_portfolio_query,
                        (
                            user_id,
                            symbol,
                            quantity,
                            current_price,
                            quantity * current_price,
                            total_cost,
                        ),
                    )

                new_cash_balance = (u_balance - total_cost).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                update_balance_query = """
                    UPDATE user_balance
                    SET cash_balance = %s
                    WHERE user_id = %s;
                """
                cursor.execute(update_balance_query, (new_cash_balance, user_id))

                new_portfolio_value = (
                    Decimal(u_portfolio) + quantity * current_price
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                new_net_worth = (new_cash_balance + new_portfolio_value).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                update_balance_query = """
                    UPDATE user_balance
                    SET portfolio_value = %s, net_worth = %s
                    WHERE user_id = %s;
                """
                cursor.execute(
                    update_balance_query, (new_portfolio_value, new_net_worth, user_id)
                )
                cursor.execute(
                    """
                        INSERT INTO transactions (user_id, symbol, quantity, order_type, limit_price, transaction_type, created_at, price_per_share, total_price)
                        VALUES (%s, %s, %s, %s, %s, 'buy', NOW(), %s, %s);
                    """,
                    (
                        user_id,
                        symbol,
                        quantity,
                        order_type,
                        limit_price,
                        current_price,
                        total_cost,
                    ),
                )

                conn.commit()
                return f"Successfully bought {quantity} shares of {symbol} at {current_price} per share."

            elif order_type == "limit":
                if limit_price is None or limit_price <= 0:
                    raise ValueError(
                        "Limit price must be specified and greater than zero."
                    )

                if quantity <= 0:
                    raise ValueError("Quantity must be a positive value.")

                limit_price = Decimal(limit_price)
                current_price = Decimal(current_price)

                validate_limit_price(limit_price, current_price, "buy")

                if current_price <= limit_price:
                    total_cost = (Decimal(quantity) * current_price).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    if u_balance < total_cost:
                        raise Exception("Insufficient funds")

                    if portfolio_row:
                        portfolio_quantity = portfolio_row[2]
                        portfolio_avg_price = Decimal(portfolio_row[3])
                        portfolio_total_invested = Decimal(portfolio_row[4])
                        new_quantity = portfolio_quantity + quantity
                        new_avg_price = (
                            (portfolio_quantity * portfolio_avg_price + total_cost)
                            / new_quantity
                        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                        new_total_invested = (
                            portfolio_total_invested + total_cost
                        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                        update_portfolio_query = """
                            UPDATE portfolio
                            SET quantity = %s, average_price = %s, current_value = %s, total_invested = %s
                            WHERE user_id = %s AND symbol = %s;
                        """
                        cursor.execute(
                            update_portfolio_query,
                            (
                                new_quantity,
                                new_avg_price,
                                new_quantity * current_price,
                                new_total_invested,
                                user_id,
                                symbol,
                            ),
                        )
                    else:
                        insert_portfolio_query = """
                            INSERT INTO portfolio (user_id, symbol, quantity, average_price, current_value, total_invested)
                            VALUES (%s, %s, %s, %s, %s, %s);
                        """
                        cursor.execute(
                            insert_portfolio_query,
                            (
                                user_id,
                                symbol,
                                quantity,
                                current_price,
                                quantity * current_price,
                                total_cost,
                            ),
                        )

                    new_cash_balance = (u_balance - total_cost).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    update_balance_query = """
                        UPDATE user_balance
                        SET cash_balance = %s
                        WHERE user_id = %s;
                    """
                    cursor.execute(update_balance_query, (new_cash_balance, user_id))

                    new_portfolio_value = (
                        Decimal(u_portfolio) + quantity * current_price
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    new_net_worth = (new_cash_balance + new_portfolio_value).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    update_balance_query = """
                        UPDATE user_balance
                        SET portfolio_value = %s, net_worth = %s
                        WHERE user_id = %s;
                    """
                    cursor.execute(
                        update_balance_query,
                        (new_portfolio_value, new_net_worth, user_id),
                    )
                    cursor.execute(
                        """
                        INSERT INTO transactions (user_id, symbol, quantity, order_type, limit_price, transaction_type, created_at, price_per_share, total_price)
                        VALUES (%s, %s, %s, %s, %s, 'buy', NOW(), %s, %s);
                    """,
                        (
                            user_id,
                            symbol,
                            quantity,
                            order_type,
                            limit_price,
                            current_price,
                            total_cost,
                        ),
                    )

                    conn.commit()
                    return f"Limit order executed: {quantity} shares of {symbol} bought at {current_price} per share."
                else:
                    store_limit_order(
                        user_id, symbol, quantity, limit_price, order_type="buy"
                    )
                    return f"Limit order placed: Buy {quantity} shares of {symbol} at or below {limit_price} per share."


def sell_stock(user_id, symbol, quantity, order_type="market", limit_price=None):
    with psycopg2.connect(**db_config) as conn:
        response = requests.get("https://archlinux.tail9023a4.ts.net/stocks/prices")
        data = response.json()
        current_price = [
            stock_info["price"] for stock_info in data if stock_info["symbol"] == symbol
        ][0]
        current_price = Decimal(str(current_price))

        with conn.cursor() as cursor:
            user_query = """
                SELECT u.id, u.u_name, ub.cash_balance, ub.portfolio_value, ub.net_worth
                FROM users u
                JOIN user_balance ub ON u.id = ub.user_id
                WHERE u.id = %s;
            """
            cursor.execute(user_query, (user_id,))
            user_id, u_name, u_balance, u_portfolio, u_net_worth = cursor.fetchone()

            portfolio_query = """
                SELECT * FROM portfolio WHERE user_id=%s AND symbol=%s;
            """
            cursor.execute(portfolio_query, (user_id, symbol))
            portfolio_row = cursor.fetchone()

            if not portfolio_row or portfolio_row[2] < quantity:
                raise Exception("Not enough shares to sell")

            portfolio_quantity = portfolio_row[2]
            portfolio_avg_price = Decimal(portfolio_row[3])

            if order_type == "market":

                # if not is_trading_time():
                #     raise Exception("Only limit orders can be placed outside market hours.")

                total_sale = (Decimal(quantity) * current_price).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                new_quantity = portfolio_quantity - quantity
                new_total_invested = (new_quantity * portfolio_avg_price).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                new_current_value = (new_quantity * current_price).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                if new_quantity > 0:
                    update_portfolio_query = """
                        UPDATE portfolio
                        SET quantity = %s, current_value = %s, total_invested = %s
                        WHERE user_id = %s AND symbol = %s;
                    """
                    cursor.execute(
                        update_portfolio_query,
                        (
                            new_quantity,
                            new_current_value,
                            new_total_invested,
                            user_id,
                            symbol,
                        ),
                    )
                else:
                    delete_portfolio_query = """
                        DELETE FROM portfolio WHERE user_id = %s AND symbol = %s;
                    """
                    cursor.execute(delete_portfolio_query, (user_id, symbol))

                new_cash_balance = (Decimal(u_balance) + total_sale).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                update_balance_query = """
                    UPDATE user_balance
                    SET cash_balance = %s
                    WHERE user_id = %s;
                """
                cursor.execute(update_balance_query, (new_cash_balance, user_id))

                new_portfolio_value = (
                    Decimal(u_portfolio) - quantity * current_price
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                new_net_worth = (new_cash_balance + new_portfolio_value).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                update_balance_query = """
                    UPDATE user_balance
                    SET portfolio_value = %s, net_worth = %s
                    WHERE user_id = %s;
                """
                cursor.execute(
                    update_balance_query, (new_portfolio_value, new_net_worth, user_id)
                )
                cursor.execute(
                    """
                    INSERT INTO transactions (user_id, symbol, quantity, order_type, limit_price, transaction_type, created_at, total_price, price_per_share)
                    VALUES (%s, %s, %s, %s, %s, 'sell', NOW(), %s, %s);
                """,
                    (
                        user_id,
                        symbol,
                        quantity,
                        order_type,
                        None,
                        total_sale,
                        current_price,
                    ),
                )

                conn.commit()
                return f"Successfully sold {quantity} shares of {symbol} at {current_price} per share."

            elif order_type == "limit":
                if limit_price is None or limit_price <= 0:
                    raise ValueError(
                        "Limit price must be specified and greater than zero."
                    )

                if quantity <= 0:
                    raise ValueError("Quantity must be a positive value.")

                validate_limit_price(limit_price, current_price, "sell")

                if current_price >= limit_price:
                    total_sale = (Decimal(quantity) * current_price).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    new_quantity = portfolio_quantity - quantity
                    new_total_invested = (new_quantity * portfolio_avg_price).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    new_current_value = (new_quantity * current_price).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    if new_quantity > 0:
                        update_portfolio_query = """
                            UPDATE portfolio
                            SET quantity = %s, current_value = %s, total_invested = %s
                            WHERE user_id = %s AND symbol = %s;
                        """
                        cursor.execute(
                            update_portfolio_query,
                            (
                                new_quantity,
                                new_current_value,
                                new_total_invested,
                                user_id,
                                symbol,
                            ),
                        )
                    else:
                        delete_portfolio_query = """
                            DELETE FROM portfolio WHERE user_id = %s AND symbol = %s;
                        """
                        cursor.execute(delete_portfolio_query, (user_id, symbol))

                    new_cash_balance = (Decimal(u_balance) + total_sale).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    update_balance_query = """
                        UPDATE user_balance
                        SET cash_balance = %s
                        WHERE user_id = %s;
                    """
                    cursor.execute(update_balance_query, (new_cash_balance, user_id))

                    new_portfolio_value = (
                        Decimal(u_portfolio) - quantity * current_price
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    new_net_worth = (new_cash_balance + new_portfolio_value).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    update_balance_query = """
                        UPDATE user_balance
                        SET portfolio_value = %s, net_worth = %s
                        WHERE user_id = %s;
                    """
                    cursor.execute(
                        update_balance_query,
                        (new_portfolio_value, new_net_worth, user_id),
                    )

                    conn.commit()
                    cursor.execute(
                        """
                        INSERT INTO transactions (user_id, symbol, quantity, order_type, limit_price, transaction_type, total_price, price_per_share, created_at)
                        VALUES (%s, %s, %s, %s, %s, 'sell', %s, %s, NOW());
                    """,
                        (
                            user_id,
                            symbol,
                            quantity,
                            order_type,
                            limit_price,
                            total_sale,
                            current_price,
                        ),
                    )

                    return f"Limit order executed: {quantity} shares of {symbol} sold at {current_price} per share."
                else:
                    store_limit_order(
                        user_id, symbol, quantity, limit_price, order_type="sell"
                    )
                    return f"Limit order placed: Sell {quantity} shares of {symbol} at or above {limit_price} per share."
