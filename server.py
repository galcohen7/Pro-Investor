"""
Pro-Investor — Async TCP Socket Backend Server.

Run independently from the Streamlit UI:
    python server.py

The Streamlit app (client) connects to this server to offload
all heavy computation (scoring, indicators, live prices).
If the server is unreachable, the app falls back to direct local calls.

Wire Protocol (newline-delimited JSON over TCP):
  Request  → {"action": "<action>", ...params}  \\n
  Response → {"success": true,  "data": {...}}   \\n
          or {"success": false, "error": "<msg>"} \\n

Supported actions:
  ping        → health check
  price       → live asset price (ticker)
  indicators  → technical indicators (ticker)
  info        → fundamental data (ticker)
  score       → full investment score + human-readable layer (ticker, risk, months)
"""

import asyncio
import json
import logging
import os
import sys

sys.path.append(os.path.dirname(__file__))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TCP-SERVER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pro_investor.server")

HOST            = os.environ.get("SERVER_HOST", "127.0.0.1")
PORT            = int(os.environ.get("SERVER_PORT", "8765"))
MAX_READ_BYTES  = 65_536   # 64 KB — generous for any reasonable request


# ── Action dispatcher ─────────────────────────────────────────────────────────
def _dispatch(action: str, params: dict) -> dict:
    """
    Route a parsed action to the correct local engine function.
    All imports are deferred so this module loads fast.
    """
    from data_engine import get_asset_info, get_live_price
    from scoring_engine import score_ticker, score_to_human_readable
    from technical_indicators import get_all_indicators

    if action == "ping":
        return {"status": "ok", "server": "Pro-Investor TCP Backend"}

    elif action == "price":
        sym   = str(params.get("ticker", "")).strip().upper()
        price = get_live_price(sym)
        return {"ticker": sym, "price": round(price, 2)}

    elif action == "indicators":
        sym = str(params.get("ticker", "")).strip().upper()
        return get_all_indicators(sym)

    elif action == "info":
        sym = str(params.get("ticker", "")).strip().upper()
        return get_asset_info(sym)

    elif action == "score":
        sym    = str(params.get("ticker", "")).strip().upper()
        risk   = str(params.get("risk", "medium")).lower().strip()
        months = int(float(str(params.get("months", 12))))
        if risk not in ("low", "medium", "high"):
            risk = "medium"
        raw         = score_ticker(sym, risk, months)
        raw["human"] = score_to_human_readable(raw)
        raw.pop("indicators", None)     # keep payload lean
        return raw

    else:
        raise ValueError(f"Unknown action: '{action}'")


# ── Connection handler ────────────────────────────────────────────────────────
async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Process one client request and close the connection."""
    peer = writer.get_extra_info("peername")
    logger.info(f"New connection from {peer}")
    try:
        raw_bytes = await asyncio.wait_for(reader.read(MAX_READ_BYTES), timeout=10.0)
        if not raw_bytes:
            logger.warning(f"{peer}: empty request, closing.")
            return

        request = json.loads(raw_bytes.decode("utf-8"))
        action  = str(request.get("action", "")).lower()
        params  = {k: v for k, v in request.items() if k != "action"}

        logger.info(f"{peer} → action={action} params={params}")

        data     = _dispatch(action, params)
        response = json.dumps({"success": True,  "data": data})
        logger.info(f"{peer} ← OK ({len(response)} bytes)")

    except asyncio.TimeoutError:
        logger.warning(f"{peer}: read timeout.")
        response = json.dumps({"success": False, "error": "Read timeout"})

    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error(f"{peer}: bad JSON — {exc}")
        response = json.dumps({"success": False, "error": f"Bad JSON: {exc}"})

    except Exception as exc:
        logger.error(f"{peer}: handler error — {exc}", exc_info=True)
        response = json.dumps({"success": False, "error": str(exc)})

    finally:
        try:
            writer.write((response + "\n").encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        logger.info(f"{peer}: connection closed.")


# ── Entry point ───────────────────────────────────────────────────────────────
async def _main() -> None:
    from database import init_db
    init_db()   # ensure tables exist on startup

    server = await asyncio.start_server(_handle_client, HOST, PORT)
    bound  = [s.getsockname() for s in server.sockets]
    logger.info("=" * 55)
    logger.info("  Pro-Investor Backend TCP Server")
    logger.info(f"  Listening on {bound}")
    logger.info("  Press Ctrl+C to stop.")
    logger.info("=" * 55)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
