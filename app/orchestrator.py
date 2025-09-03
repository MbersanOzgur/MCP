# app/orchestrator.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from .config import settings

router = APIRouter()

# ---------- MCP helpers ----------

async def mcp_rpc(client: httpx.AsyncClient, method: str, params: Optional[dict] = None, _id: int = 1) -> dict:
    payload = {"jsonrpc": "2.0", "id": _id, "method": method}
    if params is not None:
        payload["params"] = params
    r = await client.post(settings.MCP_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "result" not in data:
        raise HTTPException(502, f"MCP error: {data.get('error')}")
    return data["result"]

def _mcp_to_gemini_function_declarations(mcp_tools: List[dict]) -> List[dict]:
    fns = []
    for t in mcp_tools:
        schema: Dict[str, Any] = t.get("inputSchema") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        parameters = {"type": "OBJECT", "properties": props}
        if required:
            parameters["required"] = required
        fns.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": parameters
        })
    return fns

# ---------- Gemini helpers ----------

def _gemini_headers() -> dict:
    return {"Content-Type": "application/json"}

def _gemini_url(model: str, path: str = "generateContent") -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{path}?key={settings.GEMINI_API_KEY}"

async def _gemini_generate_with_tools(
    client: httpx.AsyncClient,
    messages: List[dict],
    function_declarations: List[dict],
) -> dict:
    contents = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        g_role = "model" if role == "assistant" else "user"
        contents.append({"role": g_role, "parts": [{"text": text}]})

    body = {
        "contents": contents,
        "tools": [{"function_declarations": function_declarations}],
        "tool_config": {"function_calling_config": {"mode": "AUTO"}},
    }
    resp = await client.post(_gemini_url(settings.MODEL_NAME), headers=_gemini_headers(), json=body, timeout=90)
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, f"Gemini error (first call): {resp.text}")
    return resp.json()

def _extract_first_function_call(gemini_json: dict) -> Optional[dict]:
    cands = gemini_json.get("candidates") or []
    if not cands:
        return None
    parts = (cands[0].get("content") or {}).get("parts") or []
    for p in parts:
        fc = p.get("functionCall")
        if fc:
            return fc
    return None

# --- Simple text generation (no tools) ---

async def _gemini_generate_text(client: httpx.AsyncClient, messages: List[dict]) -> str:
    """Second-pass call: ask Gemini to summarize results into readable prose."""
    contents = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        g_role = "model" if role == "assistant" else "user"
        contents.append({"role": g_role, "parts": [{"text": text}]})
    body = {"contents": contents}
    resp = await client.post(_gemini_url(settings.MODEL_NAME), headers=_gemini_headers(), json=body, timeout=60)
    if resp.status_code >= 400:
        return ""
    data = resp.json()
    cands = data.get("candidates") or []
    if not cands:
        return ""
    parts = (cands[0].get("content") or {}).get("parts") or []
    for p in parts:
        if "text" in p:
            return p["text"].strip()
    return ""

# ---------- Result normalization ----------

def _normalize_by_marketplace(tool_name: str, mcp_json: dict) -> dict:
    if not isinstance(mcp_json, dict):
        return {}
    if "by_marketplace" in mcp_json and isinstance(mcp_json["by_marketplace"], dict):
        return mcp_json["by_marketplace"]
    if "items" in mcp_json and isinstance(mcp_json["items"], list):
        mk_map = {
            "list_amazon_products": "amazon",
            "list_trendyol_products": "trendyol",
            "list_hepsiburada_products": "hepsiburada",
        }
        mk = mk_map.get(tool_name, "marketplace")
        return {mk: mcp_json["items"]}
    return {}

def _flatten_items(by_marketplace: dict, max_items: int = 10) -> List[dict]:
    rows: List[dict] = []
    for mk, items in (by_marketplace or {}).items():
        for it in items or []:
            rows.append({
                "marketplace": mk,
                "item_name": it.get("item_name"),
                "item_type": it.get("item_type"),
                "item_price": it.get("item_price"),
                "item_rate": it.get("item_rate"),
            })
    # sort by price asc, then higher rating
    rows = sorted(rows, key=lambda x: (x.get("item_price", 10**12), -float(x.get("item_rate") or 0)))
    return rows[:max_items]

def _format_items_for_prompt(items: List[dict]) -> str:
    """Turn items into a compact, human-readable list for LLM summarization."""
    lines = []
    for i, it in enumerate(items, 1):
        name = it.get("item_name") or ""
        mk = it.get("marketplace") or ""
        typ = it.get("item_type") or ""
        price = it.get("item_price")
        rate = it.get("item_rate")
        price_str = "N/A" if price is None else f"{price}"
        rate_str = "N/A" if rate is None else f"{rate}"
        lines.append(f"{i}. {name} — {mk} — type: {typ} — price: {price_str} — rating: {rate_str}")
    return "\n".join(lines)

def _fallback_summary(items: List[dict], user_text: str) -> str:
    """Deterministic summary if LLM text generation fails."""
    if not items:
        return "Üzgünüm, uygun ürün bulamadım. Aramayı netleştirebilir misin?"
    first = items[0]
    cheapest = f"{first.get('item_name')} ({first.get('marketplace')}) — ₺{first.get('item_price')} · {first.get('item_rate')}★"
    others = []
    for it in items[1:4]:
        others.append(f"- {it.get('item_name')} ({it.get('marketplace')}) — ₺{it.get('item_price')} · {it.get('item_rate')}★")
    tail = "\n".join(others)
    return f"{user_text}\nEn uygun seçenek: {cheapest}\nDiğerleri:\n{tail}"

# ---------- Public API endpoint ----------

@router.post("/chat")
async def chat(payload: dict, mode: str = Query("text", enum=["text", "json"])):
    """
    Body: { "message": "user text here" }
    mode:
      - text (default): returns a plain-text natural-language answer
      - json: returns { items: [...], answer: "..." }
    """
    if not settings.GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY is not set in .env")
    user_text = (payload.get("message") or "").strip()
    if not user_text:
        raise HTTPException(400, "message is required")

    async with httpx.AsyncClient() as client:
        # tools → function declarations
        tools_list = await mcp_rpc(client, "tools/list")
        mcp_tools = tools_list.get("tools") or []
        fn_decls = _mcp_to_gemini_function_declarations(mcp_tools)

        # first Gemini call (may request a tool)
        msgs = [
            {"role": "system", "content": "Use functions to fetch products. If needed, pick the right marketplace tool."},
            {"role": "user", "content": user_text},
        ]
        g1 = await _gemini_generate_with_tools(client, msgs, fn_decls)

        # execute tool if requested
        fc = _extract_first_function_call(g1)
        if not fc:
            answer = await _gemini_generate_text(client, [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": f"The user asked: {user_text}\nNo matching tool was used. Reply briefly asking them to rephrase or be more specific."}
            ]) or "Sorry, I couldn’t find a matching tool. Please rephrase or be more specific."
            return PlainTextResponse(answer) if mode == "text" else JSONResponse({"items": [], "answer": answer})

        tool_name = fc["name"]
        args = fc.get("args") or {}
        mcp_res = await mcp_rpc(client, "tools/call", {"name": tool_name, "arguments": args}, _id=101)
        mcp_json = (mcp_res.get("content") or [{}])[0].get("json")

        by_mk = _normalize_by_marketplace(tool_name, mcp_json)
        items = _flatten_items(by_mk, max_items=10)

        # second Gemini call: create an easy-to-read answer from the items
        if items:
            items_blob = _format_items_for_prompt(items)
            answer = await _gemini_generate_text(client, [
                {
                    "role": "system",
                    "content": (
                        "Write a short, friendly answer summarizing a small product list.\n"
                        "- Start with the cheapest good option.\n"
                        "- Mention price with the TL symbol (₺) using the given numbers.\n"
                        "- Mention rating briefly with a star (e.g., 4.6★).\n"
                        "- If multiple marketplaces exist, contrast in 1–2 sentences.\n"
                        "- No markdown tables. Keep under 6 lines."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{user_text}\n\n"
                        f"Here are the structured items (sorted by price asc, then rating desc):\n{items_blob}"
                    ),
                },
            ]) or _fallback_summary(items, user_text)
        else:
            answer = await _gemini_generate_text(client, [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": f"The user asked: {user_text}\nNo items were found. Suggest 2–3 alternative queries in one short paragraph."}
            ]) or "No matching items found. Try specifying the product type, brand, or a price range."

        # Return in requested mode
        if mode == "text":
            return PlainTextResponse(answer)
        return JSONResponse({"items": items, "answer": answer})
