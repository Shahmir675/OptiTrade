DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS dividends CASCADE;
DROP TABLE IF EXISTS portfolio CASCADE;
DROP TABLE IF EXISTS watchlist CASCADE;
DROP TABLE IF EXISTS user_balance CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    u_name VARCHAR NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    u_pass VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_users_id ON users (id);

CREATE TABLE user_balance (
    user_id INTEGER PRIMARY KEY,
    cash_balance DOUBLE PRECISION NOT NULL,
    portfolio_value DOUBLE PRECISION NOT NULL DEFAULT 0.00,
    net_worth DOUBLE PRECISION NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE watchlist (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    stock_symbol VARCHAR(7) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT unique_user_stock UNIQUE (user_id, stock_symbol)
);
CREATE INDEX IF NOT EXISTS ix_watchlist_id ON watchlist (id);

CREATE TABLE portfolio (
    user_id INTEGER NOT NULL,
    symbol VARCHAR(6) NOT NULL,
    quantity INTEGER NOT NULL,
    average_price DECIMAL(10, 2) NOT NULL,
    current_value DECIMAL(10, 2),
    total_invested DECIMAL(10, 2),
    PRIMARY KEY (user_id, symbol),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT ck_portfolio_quantity CHECK (quantity >= 0),
    CONSTRAINT ck_portfolio_average_price CHECK (average_price >= 0)
);

CREATE TABLE dividends (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    stock_symbol VARCHAR(7) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    payment_date DATE NOT NULL,
    ex_dividend_date DATE NOT NULL,
    paid_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT unique_dividend_payment UNIQUE (user_id, stock_symbol, ex_dividend_date)
);
CREATE INDEX IF NOT EXISTS ix_dividends_id ON dividends (id);

CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    symbol VARCHAR(7) NOT NULL,
    quantity INTEGER NOT NULL,
    order_type VARCHAR(10) NOT NULL,
    limit_price DECIMAL(10, 2),
    transaction_type VARCHAR(4) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    price_per_share DECIMAL(10, 2) NOT NULL,
    total_price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT ck_transactions_order_type CHECK (order_type IN ('market', 'limit')),
    CONSTRAINT ck_transactions_transaction_type CHECK (transaction_type IN ('buy', 'sell'))
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    symbol VARCHAR(7),
    order_type VARCHAR(10),
    price DECIMAL(10, 2),
    quantity INTEGER,
    "timestamp" TIMESTAMP WITHOUT TIME ZONE,
    order_status BOOLEAN,
    filled_quantity INTEGER,
    remaining_quantity INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE take_profit_orders (
    order_id SERIAL PRIMARY KEY,
    user_id INT,
    symbol VARCHAR(7),
    order_type VARCHAR(10) CHECK (order_type IN ('buy', 'sell')),
    take_profit_price DECIMAL(10, 2) NOT NULL,
    quantity INT CHECK (quantity > 0),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_status BOOLEAN DEFAULT FALSE,
    filled_quantity INT DEFAULT 0,
    remaining_quantity INT GENERATED ALWAYS AS (quantity - filled_quantity) STORED,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE stop_loss_orders (
    order_id SERIAL PRIMARY KEY,
    user_id INT,
    symbol VARCHAR(7),
    order_type VARCHAR(10) CHECK (order_type IN ('buy', 'sell')),
    stop_price DECIMAL(10, 2) NOT NULL,
    quantity INT CHECK (quantity > 0),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_status BOOLEAN DEFAULT FALSE,
    filled_quantity INT DEFAULT 0,
    remaining_quantity INT GENERATED ALWAYS AS (quantity - filled_quantity) STORED,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

SELECT * FROM users LIMIT 5;

SELECT * FROM user_balance LIMIT 5;

SELECT * FROM watchlist LIMIT 5;

SELECT * FROM portfolio LIMIT 5;

SELECT * FROM dividends LIMIT 5;

SELECT * FROM transactions LIMIT 5;

SELECT * FROM orders LIMIT 5;

SELECT * FROM take_profit_orders LIMIT 5;

SELECT * FROM stop_loss_orders LIMIT 5;