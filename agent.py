"""
Autonomous AI Investment Agent — powered by Groq (free LLM API).

LLM   : LLaMA-3.3-70b-versatile via Groq (OpenAI-compatible tool-use API)
Embed : Sentence-Transformers all-MiniLM-L6-v2  (runs 100% locally)
VecDB : ChromaDB                                 (runs 100% locally)

The agent executes an agentic loop:
  1. Send user query + system context to Groq
  2. If the model requests tool calls, execute them locally
  3. Feed results back into the conversation
  4. Repeat until the model returns a final text answer
"""

import json
import os

from groq import Groq

from data_engine import get_live_price, get_asset_info
from technical_indicators import get_all_indicators
from scoring_engine import score_ticker
from rag_pipeline import RAGPipeline

GROQ_MODEL = "llama-3.3-70b-versatile"  # best free Groq model with tool-use support

SYSTEM_PROMPT = """You are Pro-Investor, an elite AI investment advisor specializing in quantitative financial analysis.

You have access to real-time market data, technical indicators, a proprietary scoring algorithm, and a curated financial knowledge base. Your mission is to synthesize all available signals into clear, actionable investment recommendations.

Your scoring formula:
  Score = (Probability_Profit x Expected_Return) / Risk_Factor

Structure every recommendation with these sections:
1. **Market Snapshot** - current price, trend, key metrics
2. **Technical Signals** - RSI, MACD, Bollinger Bands interpretation
3. **Risk Assessment** - volatility and risk-adjusted score
4. **RAG Insights** - relevant context from the knowledge base
5. **Recommendation** - clear buy/hold/avoid with confidence level

Always acknowledge market uncertainty. Past performance does not guarantee future results."""

# ── Tool schemas in OpenAI / Groq format ──────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_live_price",
            "description": "Fetches the current live market price of a financial asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. 'AAPL', 'BTC-USD', 'SPY'",
                    }
                },
                "required": ["ticker_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "Calculates RSI, MACD, Bollinger Bands, annualized volatility, and trend direction for a ticker. Runs locally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "The ticker to analyze.",
                    }
                },
                "required": ["ticker_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_investment_score",
            "description": "Computes Score = (P_profit x Expected_Return) / Risk_Factor and returns a full scoring breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "The ticker to score.",
                    },
                    "risk_tolerance": {
                        "type": "string",
                        "description": "User risk tolerance: 'low', 'medium', or 'high'.",
                    },
                    "duration_months": {
                        "type": "integer",
                        "description": "Investment horizon in months.",
                    },
                },
                "required": ["ticker_symbol", "risk_tolerance", "duration_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the local RAG knowledge base (ChromaDB + Sentence-Transformers) for relevant financial context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_asset_info",
            "description": "Fetches fundamental data: company name, sector, market cap, P/E ratio, 52-week high/low.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "The ticker to look up.",
                    }
                },
                "required": ["ticker_symbol"],
            },
        },
    },
]


class InvestmentAgent:
    """
    Agentic loop backed by Groq's OpenAI-compatible chat-completions endpoint.
    All heavy computation (embeddings, indicators, scoring) runs locally.
    Only the LLM inference call goes to Groq's free API.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.rag = RAGPipeline()

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        risk_tolerance: str,
        duration_months: int,
    ) -> str:
        """Dispatches a tool call to the appropriate local function and returns a JSON string."""
        try:
            if tool_name == "get_live_price":
                price = get_live_price(tool_input["ticker_symbol"])
                return json.dumps({"ticker": tool_input["ticker_symbol"], "price": round(price, 2)})

            elif tool_name == "get_technical_indicators":
                return json.dumps(get_all_indicators(tool_input["ticker_symbol"]))

            elif tool_name == "get_investment_score":
                result = score_ticker(
                    tool_input["ticker_symbol"],
                    tool_input.get("risk_tolerance", risk_tolerance),
                    tool_input.get("duration_months", duration_months),
                )
                result.pop("indicators", None)  # strip nested dict to keep payload concise
                return json.dumps(result)

            elif tool_name == "search_knowledge_base":
                return self.rag.get_context_string(tool_input["query"])

            elif tool_name == "get_asset_info":
                return json.dumps(get_asset_info(tool_input["ticker_symbol"]))

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def run(self, user_query: str, user_profile: dict) -> str:
        """
        Executes the full agentic loop for a single user query.

        user_profile expected keys:
          budget (float), risk_tolerance (str), duration_months (int)
        """
        risk_tolerance = user_profile.get("risk_tolerance", "medium")
        duration_months = user_profile.get("duration_months", 6)
        budget = user_profile.get("budget", 0)

        system_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Current User Profile:\n"
            f"  * Budget: ${budget:,.0f}\n"
            f"  * Risk Tolerance: {risk_tolerance}\n"
            f"  * Investment Horizon: {duration_months} months\n"
        )

        # Groq uses the standard OpenAI messages format
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]

        # Agentic loop — runs until finish_reason is 'stop'
        while True:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls

                # Append the assistant message (including its tool_calls) to history
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                # Execute every requested tool and append results as "tool" messages
                for tc in tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    result = self._execute_tool(
                        tc.function.name, tool_input, risk_tolerance, duration_months
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": result,
                        }
                    )

            else:
                # finish_reason == "stop" (or "length") — return the final text
                return choice.message.content or ""
