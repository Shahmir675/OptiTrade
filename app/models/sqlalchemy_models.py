from datetime import datetime

from sqlalchemy import (
   DECIMAL,
   TIMESTAMP,
   Boolean,
   CheckConstraint,
   Column,
   Float,
   ForeignKey,
   Integer,
   PrimaryKeyConstraint,
   String,
   Text,
   UniqueConstraint,
   Computed,
   JSON
)
from sqlalchemy.orm import relationship

from app.db.base import Base
from app.utils.timezones import PAKISTAN_TIMEZONE


class UserBalanceModel(Base):
   __tablename__ = "user_balance"

   user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
   cash_balance = Column(Float, nullable=False)
   portfolio_value = Column(Float, nullable=False, default=0.00)
   net_worth = Column(Float, nullable=False)
   user = relationship("UserModel", back_populates="balance")


class UserModel(Base):
   __tablename__ = "users"

   id = Column(Integer, primary_key=True, index=True)
   u_name = Column(String, nullable=False)
   email = Column(String, unique=True, nullable=False)
   u_pass = Column(String, nullable=False)
   balance = relationship("UserBalanceModel", back_populates="user", uselist=False)
   watchlist = relationship(
       "Watchlist", back_populates="user", cascade="all, delete-orphan"
   )
   image_url = Column(String, nullable=True)
   risk_profile = relationship("RiskProfile", back_populates="user", uselist=False)


class Watchlist(Base):
   __tablename__ = "watchlist"
   id = Column(Integer, primary_key=True, index=True)
   user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
   stock_symbol = Column(String(7), nullable=False)
   user = relationship("UserModel", back_populates="watchlist")
   __table_args__ = (
       UniqueConstraint("user_id", "stock_symbol", name="unique_user_stock"),
   )


class Portfolio(Base):
   __tablename__ = "portfolio"

   user_id = Column(
       Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
   )
   symbol = Column(String(6), nullable=False)
   quantity = Column(Integer, nullable=False)
   average_price = Column(DECIMAL(10, 2), nullable=False)
   current_value = Column(DECIMAL(10, 2))
   total_invested = Column(DECIMAL(10, 2))

   __table_args__ = (
       PrimaryKeyConstraint("user_id", "symbol"),
       CheckConstraint("quantity >= 0"),
       CheckConstraint("average_price >= 0"),
   )


class PortfolioHistory(Base):
   __tablename__ = "portfolio_history"

   user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
   symbol = Column(String(6))
   quantity = Column(Integer)
   average_price = Column(DECIMAL(10, 2))
   current_value = Column(DECIMAL(10, 2))
   total_invested = Column(DECIMAL(10, 2))
   snapshot_date = Column(TIMESTAMP, nullable=False)

   __table_args__ = (
       PrimaryKeyConstraint(
           "user_id", "symbol", "snapshot_date", name="portfolio_history_pkey"
       ),
   )


class Dividend(Base):
   __tablename__ = "dividends"

   id = Column(Integer, primary_key=True, index=True)
   user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
   stock_symbol = Column(String(7), nullable=False)
   amount = Column(Float, nullable=False)
   payment_date = Column(String(10), nullable=False)
   ex_dividend_date = Column(String(10), nullable=False)
   paid_at = Column(
       String(20), default=lambda: datetime.now(PAKISTAN_TIMEZONE).isoformat()
   )

   __table_args__ = (
       UniqueConstraint(
           "user_id",
           "stock_symbol",
           "ex_dividend_date",
           name="unique_dividend_payment",
       ),
   )


class Transaction(Base):
   __tablename__ = "transactions"

   id = Column(Integer, primary_key=True, autoincrement=True)
   user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
   symbol = Column(String(7), nullable=False)
   quantity = Column(Integer, nullable=False)
   order_type = Column(String(10), nullable=False)
   limit_price = Column(DECIMAL(10, 2))
   transaction_type = Column(String(4), nullable=False)
   created_at = Column(
       TIMESTAMP(timezone=True), default=lambda: datetime.now(PAKISTAN_TIMEZONE)
   )
   price_per_share = Column(DECIMAL(10, 2), nullable=False)
   total_price = Column(DECIMAL(10, 2), nullable=False)

   __table_args__ = (
       CheckConstraint("order_type IN ('market', 'limit')"),
       CheckConstraint("transaction_type IN ('buy', 'sell')"),
   )


class Order(Base):
   __tablename__ = "orders"

   order_id = Column(Integer, primary_key=True, autoincrement=True)
   user_id = Column(Integer, ForeignKey("users.id"))
   symbol = Column(String(7))
   order_type = Column(String(10))
   price = Column(DECIMAL(10, 2))
   quantity = Column(Integer)
   timestamp = Column(TIMESTAMP)
   order_status = Column(Boolean)
   filled_quantity = Column(Integer)
   remaining_quantity = Column(Integer)


class StopLossOrder(Base):
   __tablename__ = "stop_loss_orders"

   order_id = Column(Integer, primary_key=True, autoincrement=True)
   user_id = Column(Integer, ForeignKey("users.id"))
   symbol = Column(String(7))
   order_type = Column(String(10))
   stop_price = Column(DECIMAL(10, 2))
   quantity = Column(Integer)
   timestamp = Column(
       TIMESTAMP(timezone=True), default=lambda: datetime.now(PAKISTAN_TIMEZONE)
   )
   order_status = Column(Boolean, default=False)
   filled_quantity = Column(Integer, default=0)
   remaining_quantity = Column(Integer, Computed("quantity - filled_quantity"), nullable=False)


class Feedback(Base):
   __tablename__ = "feedback"

   feedback_id = Column(Integer, primary_key=True, index=True)
   feedback_message = Column(Text, nullable=False)
   created_at = Column(
       TIMESTAMP(timezone=True), default=lambda: datetime.now(PAKISTAN_TIMEZONE)
   )
   feedback_status = Column(String(20), default="todo")


class RiskProfile(Base):
   __tablename__ = "risk_profiles"

   id = Column(Integer, primary_key=True, index=True)
   user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
   risk_score = Column(Float, nullable=False)
   risk_category = Column(String(50), nullable=False)
   risk_level = Column(String(20), nullable=False)
   last_assessment_date = Column(TIMESTAMP(timezone=True), nullable=False)
   next_assessment_due = Column(String(20), nullable=False)
   has_completed_assessment = Column(Boolean, default=True, nullable=False)
   portfolio_alignment = Column(JSON, nullable=False)
   breakdown = Column(JSON, nullable=False)
   created_at = Column(
       TIMESTAMP(timezone=True), default=lambda: datetime.now(PAKISTAN_TIMEZONE)
   )
   updated_at = Column(
       TIMESTAMP(timezone=True), 
       default=lambda: datetime.now(PAKISTAN_TIMEZONE),
       onupdate=lambda: datetime.now(PAKISTAN_TIMEZONE)
   )

   user = relationship("UserModel", back_populates="risk_profile")
   recommendations = relationship("RiskRecommendation", back_populates="risk_profile", cascade="all, delete-orphan")


class RiskRecommendation(Base):
   __tablename__ = "risk_recommendations"

   id = Column(Integer, primary_key=True, index=True)
   risk_profile_id = Column(Integer, ForeignKey("risk_profiles.id", ondelete="CASCADE"), nullable=False)
   type = Column(String(100), nullable=False)
   title = Column(String(200), nullable=False)
   description = Column(Text, nullable=False)
   priority = Column(String(20), nullable=False)
   impact = Column(String(200), nullable=False)

   risk_profile = relationship("RiskProfile", back_populates="recommendations")

   __table_args__ = (
       CheckConstraint("priority IN ('low', 'medium', 'high')"),
   )


class RiskAssessment(Base):
   __tablename__ = "risk_assessments"

   id = Column(Integer, primary_key=True, index=True)
   user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
   age = Column(Integer, nullable=False)
   investment_experience = Column(String(50), nullable=False)
   risk_tolerance = Column(Integer, nullable=False)
   investment_timeline = Column(Integer, nullable=False)
   financial_goals = Column(JSON, nullable=False)
   income_stability = Column(String(50), nullable=False)
   created_at = Column(
       TIMESTAMP(timezone=True), default=lambda: datetime.now(PAKISTAN_TIMEZONE)
   )

   __table_args__ = (
       CheckConstraint("risk_tolerance >= 1 AND risk_tolerance <= 10"),
       CheckConstraint("investment_timeline > 0"),
       CheckConstraint("investment_experience IN ('beginner', 'intermediate', 'advanced')"),
   )