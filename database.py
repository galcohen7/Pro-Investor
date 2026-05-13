"""
Pro-Investor — Database Layer (SQLAlchemy ORM).

Supports:
  - SQLite  (default, zero-config, works on Streamlit Cloud)
  - PostgreSQL (set DATABASE_URL env var for production)

Tables:
  users               — investor profiles
  recommendation_logs — every AI recommendation with price + success tracking
"""

import hashlib
import logging
import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, text,
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
    password_hash   = Column(String(64),  nullable=True)   # nullable so old rows aren't broken
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
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    try:
        Base.metadata.create_all(bind=engine)
        # Idempotent migration: add password_hash column if an older DB exists
        with engine.connect() as conn:
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(64)"))
                conn.commit()
            except Exception:
                pass  # Column already exists — that's fine
        logger.info("Database initialised (tables created / verified).")
    except Exception as exc:
        logger.error(f"DB init failed: {exc}")


# ── Auth helpers ──────────────────────────────────────────────────────────────
def create_user(
    name: str,
    password: str,
    budget: float = 10_000.0,
    risk_tolerance: str = "medium",
    duration_months: int = 12,
) -> int:
    """
    Creates a new user account. Raises ValueError if the name is already taken.
    Returns the new user ID.
    """
    name = name.strip()
    if not name:
        raise ValueError("שם משתמש לא יכול להיות ריק")
    with SessionLocal() as session:
        if session.query(User).filter_by(name=name).first():
            raise ValueError(f"שם המשתמש '{name}' כבר תפוס — נסה שם אחר")
        user = User(
            name=name,
            password_hash=_hash_password(password),
            budget=budget,
            risk_tolerance=risk_tolerance,
            duration_months=duration_months,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        logger.info(f"New user created: id={user.id} name={name}")
        return user.id


def authenticate_user(name: str, password: str) -> dict | None:
    """
    Verifies credentials. Returns a profile dict on success, None on failure.
    The dict is detached from the session and safe to store in st.session_state.
    """
    with SessionLocal() as session:
        user = session.query(User).filter_by(name=name.strip()).first()
        if user and user.password_hash == _hash_password(password):
            return {
                "id":               user.id,
                "name":             user.name,
                "budget":           user.budget,
                "risk_tolerance":   user.risk_tolerance,
                "duration_months":  user.duration_months,
            }
    return None


def update_user_profile(
    user_id: int,
    budget: float,
    risk_tolerance: str,
    duration_months: int,
) -> None:
    """Persists sidebar profile changes back to the DB."""
    with SessionLocal() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.budget          = budget
            user.risk_tolerance  = risk_tolerance
            user.duration_months = duration_months
            user.updated_at      = datetime.utcnow()
            session.commit()


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
