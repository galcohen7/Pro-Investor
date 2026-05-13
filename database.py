"""
Pro-Investor — Database Layer (SQLAlchemy ORM).

Supports:
  - SQLite  (default, zero-config, works on Streamlit Cloud)
  - PostgreSQL (set DATABASE_URL env var for production)

Tables:
  users               — investor profiles
  recommendation_logs — every AI recommendation with price + success tracking
"""

import logging
import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# ── Engine setup ──────────────────────────────────────────────────────────────
_raw_url = os.environ.get("DATABASE_URL", "sqlite:///pro_investor.db")

# Some cloud providers (Heroku/Render) still emit the legacy "postgres://" scheme
if _raw_url.startswith("postgres://"):
    _raw_url = _raw_url.replace("postgres://", "postgresql://", 1)

# SQLite needs check_same_thread=False for Streamlit's multi-thread model
_connect_args = {"check_same_thread": False} if _raw_url.startswith("sqlite") else {}

engine = create_engine(_raw_url, connect_args=_connect_args, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── ORM Models ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class User(Base):
    """Investor profile — stores preferences for personalised recommendations."""
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String(100), nullable=False, default="Guest")
    budget          = Column(Float,   nullable=False, default=10_000.0)
    risk_tolerance  = Column(String(20), nullable=False, default="medium")
    duration_months = Column(Integer, nullable=False, default=12)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RecommendationLog(Base):
    """
    Persists every AI recommendation for accuracy tracking over time.
    success_score is filled in later (e.g. via a cron job) comparing
    the recommendation verdict against the actual price change.
    """
    __tablename__ = "recommendation_logs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    user_id          = Column(Integer, nullable=True)
    ticker           = Column(String(20), nullable=False)
    verdict          = Column(String(10), nullable=False)   # BUY / WAIT / AVOID
    price_at_time    = Column(Float,  nullable=True)
    investment_score = Column(Float,  nullable=True)
    probability      = Column(Float,  nullable=True)
    expected_return  = Column(Float,  nullable=True)
    risk_factor      = Column(Float,  nullable=True)
    ai_response      = Column(Text,   nullable=True)        # truncated Hebrew text
    success_score    = Column(Float,  nullable=True)        # filled later for accuracy tracking
    created_at       = Column(DateTime, default=datetime.utcnow)


# ── DB lifecycle ──────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialised (tables created / verified).")
    except Exception as exc:
        logger.error(f"DB init failed: {exc}")


# ── CRUD helpers ──────────────────────────────────────────────────────────────
def get_or_create_user(
    name: str = "Guest",
    budget: float = 10_000.0,
    risk_tolerance: str = "medium",
    duration_months: int = 12,
) -> int:
    """Upsert a user profile and return the user ID."""
    with SessionLocal() as session:
        user = session.query(User).filter_by(name=name).first()
        if not user:
            user = User(
                name=name,
                budget=budget,
                risk_tolerance=risk_tolerance,
                duration_months=duration_months,
            )
            session.add(user)
        else:
            user.budget          = budget
            user.risk_tolerance  = risk_tolerance
            user.duration_months = duration_months
            user.updated_at      = datetime.utcnow()
        session.commit()
        session.refresh(user)
        logger.info(f"User upserted: id={user.id} name={name}")
        return user.id


def save_recommendation(
    ticker: str,
    verdict: str,
    price: float,
    score_data: dict,
    ai_response: str,
    user_id: int | None = None,
) -> int:
    """Persist one recommendation. Returns the new record ID."""
    with SessionLocal() as session:
        rec = RecommendationLog(
            user_id          = user_id,
            ticker           = ticker.upper(),
            verdict          = verdict,
            price_at_time    = price,
            investment_score = score_data.get("investment_score"),
            probability      = score_data.get("probability_profit"),
            expected_return  = score_data.get("expected_return"),
            risk_factor      = score_data.get("risk_factor"),
            ai_response      = (ai_response or "")[:4000],
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        logger.info(f"Recommendation saved: {ticker} → {verdict} (id={rec.id})")
        return rec.id


def get_recent_recommendations(limit: int = 50) -> list[dict]:
    """Fetch the N most recent recommendations for the history view."""
    with SessionLocal() as session:
        rows = (
            session.query(RecommendationLog)
            .order_by(RecommendationLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":       r.id,
                "ticker":   r.ticker,
                "verdict":  r.verdict,
                "price":    r.price_at_time,
                "score":    r.investment_score,
                "return":   r.expected_return,
                "success":  r.success_score,
                "date":     r.created_at.strftime("%d/%m %H:%M") if r.created_at else "",
            }
            for r in rows
        ]


def get_stats() -> dict:
    """Aggregate stats for the dashboard sidebar."""
    with SessionLocal() as session:
        total = session.query(RecommendationLog).count()
        buys  = session.query(RecommendationLog).filter_by(verdict="BUY").count()
        waits = session.query(RecommendationLog).filter_by(verdict="WAIT").count()
        avoids= session.query(RecommendationLog).filter_by(verdict="AVOID").count()
        return {"total": total, "buys": buys, "waits": waits, "avoids": avoids}
