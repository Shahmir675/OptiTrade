import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.base import Base
from app.db.session import engine
from app.services.price_feed_service import start_price_fetching_task
from app.services.data_service import load_initial_data
from app.routers import (
    auth,
    users,
    stocks,
    watchlist,
    portfolio_actions,
    transactions,
    orders,
    general,
    indices
)

app = FastAPI(docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    load_initial_data()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # asyncio.create_task(start_price_fetching_task())

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(stocks.router)
app.include_router(watchlist.router)
app.include_router(portfolio_actions.router)
app.include_router(transactions.router)
app.include_router(orders.router)
app.include_router(indices.router)
app.include_router(general.router)