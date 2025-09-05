# app/orchestrator.py
from __future__ import annotations
import json, time
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
    """
    Normalize various MCP result shapes into:
        { <marketplace>: [ {item...}, ... ], ... }
    Supports:
      - {"product": {...}}                        (single product)
      - {"items": [...]}                          (flat list)
      - {"by_marketplace": {"amazon":[...], ...}} (already grouped)
    """
    if not isinstance(mcp_json, dict):
        return {}
    # Single-product tools
    if "product" in mcp_json and isinstance(mcp_json["product"], dict):
        p = mcp_json["product"]
        mk = (p.get("marketplace") or "marketplace")
        return {mk: [p]}
    # Already grouped
    if "by_marketplace" in mcp_json and isinstance(mcp_json["by_marketplace"], dict):
        return mcp_json["by_marketplace"]
    # Flat list (infer marketplace)
    if "items" in mcp_json and isinstance(mcp_json["items"], list):
        mk_map = {
            "list_amazon_products": "amazon",
            "list_trendyol_products": "trendyol",
            "list_hepsiburada_products": "hepsiburada",
        }
        inferred = (mcp_json["items"][0] or {}).get("marketplace") if mcp_json["items"] else None
        mk = mk_map.get(tool_name, inferred or "marketplace")
        return {mk: mcp_json["items"]}
    return {}

def _flatten_items(by_marketplace: dict, max_items: int = 10) -> List[dict]:
    rows: List[dict] = []
    for mk, items in (by_marketplace or {}).items():
        if isinstance(items, dict):  # tolerate single product dict
            items = [items]
        for it in items or []:
            rows.append({
                "marketplace": mk,
                "item_name": it.get("item_name"),
                "item_type": it.get("item_type"),
                "item_price": it.get("item_price"),
                "item_rate": it.get("item_rate"),
            })
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
    """Deterministic summary if LLM text generation fails (English-only)."""
    if not items:
        return "Sorry, I couldn’t find any suitable products. Could you clarify your search?"
    first = items[0]
    cheapest = f"{first.get('item_name')} ({first.get('marketplace')}) — ₺{first.get('item_price')} · {first.get('item_rate')}★"
    others = []
    for it in items[1:4]:
        others.append(f"- {it.get('item_name')} ({it.get('marketplace')}) — ₺{it.get('item_price')} · {it.get('item_rate')}★")
    tail = "\n".join(others)
    return f"{user_text}\nBest option: {cheapest}\nOther options:\n{tail}"

def _format_item_types_for_answer(types: List[str], user_text: str) -> str:
    """English-only response for list_item_types."""
    if not types:
        return "I couldn’t retrieve a category list right now. Please specify the product type (e.g., toothbrush, toothpaste)."
    preview = ", ".join(types[:10]) + ("…" if len(types) > 10 else "")
    return (
        f"{user_text}\n"
        f"Some available categories are: {preview}.\n"
        f"Please choose one or include it in your search (e.g., 'toothbrush' or 'mouthwash')."
    )

# ---------- Public API endpoint with tracing ----------

@router.post("/chat")
async def chat(
    payload: dict,
    mode: str = Query("text", enum=["text", "json"]),
    debug: bool = Query(False, description="If true, include decision trace in response"),
):
    """
    Body: { "message": "user text here" }
    mode:
      - text (default): returns a plain-text natural-language answer
      - json: returns { items: [...], answer: "..." }
    debug:
      - if true, returns a 'trace' list with all decisions
    """
    if not settings.GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY is not set in .env")
    user_text = (payload.get("message") or "").strip()
    if not user_text:
        raise HTTPException(400, "message is required")

    # ---- decision trace ----
    trace: List[dict] = []
    def step(label: str, data: dict | str):
        rec = {"t": round(time.time(), 3), "step": label, "data": data}
        trace.append(rec)
        print(json.dumps({"trace": rec}, ensure_ascii=False))  # NDJSON logs

    step("received_user_text", {"message": user_text, "mode": mode})

    async with httpx.AsyncClient() as client:
        # tools → function declarations
        tools_list = await mcp_rpc(client, "tools/list")
        mcp_tools = tools_list.get("tools") or []
        fn_decls = _mcp_to_gemini_function_declarations(mcp_tools)
        step("tools_listed", {"tools": [t.get("name") for t in mcp_tools]})

        # first Gemini call (may request a tool)
        msgs = [
            {"role": "system", "content": "Use functions to fetch products. If needed, pick the right marketplace tool."},
            {"role": "user", "content": user_text},
        ]
        step("gemini_tool_selection_request", {"messages": msgs})
        g1 = await _gemini_generate_with_tools(client, msgs, fn_decls)

        # Only log the useful bits (tool + args), not the full raw response
        fc = _extract_first_function_call(g1)
        step("gemini_tool_selection", {
            "tool": (fc or {}).get("name"),
            "args": (fc or {}).get("args", {}),
        })

        # execute tool if requested
        if not fc:
            step("no_function_call", {"reason": "model did not call a tool"})
            answer = await _gemini_generate_text(client, [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": f"The user asked: {user_text}\nNo matching tool was used. Reply briefly asking them to rephrase or be more specific."}
            ]) or "Sorry, I couldn’t find a matching tool. Please rephrase or be more specific."
            step("final_answer_without_tool", {"answer": answer})
            if mode == "text":
                return PlainTextResponse(answer)
            return JSONResponse({"items": [], "answer": answer, **({"trace": trace} if debug else {})})

        tool_name = fc["name"]
        args = fc.get("args") or {}
        step("function_call_decoded", {"tool": tool_name, "args": args})

        mcp_res = await mcp_rpc(client, "tools/call", {"name": tool_name, "arguments": args}, _id=101)
        step("mcp_call_result_raw", mcp_res)
        mcp_json = (mcp_res.get("content") or [{}])[0].get("json") or {}

        # SPECIAL CASE: list_item_types → produce a short suggestion answer and return
        if tool_name == "list_item_types" and isinstance(mcp_json, dict) and "item_types" in mcp_json:
            types = mcp_json.get("item_types") or []
            answer = _format_item_types_for_answer(types, user_text)
            step("final_answer_item_types", {"count": len(types)})
            if mode == "text":
                return PlainTextResponse(answer)
            return JSONResponse({"items": [], "answer": answer, **({"trace": trace} if debug else {})})

        # Normal product-shaped flows
        by_mk = _normalize_by_marketplace(tool_name, mcp_json)
        items = _flatten_items(by_mk, max_items=10)
        step("normalized_items", {
            "per_marketplace_counts": {k: len(v if isinstance(v, list) else [v]) for k, v in (by_mk or {}).items()},
            "flattened_count": len(items),
        })

        # second Gemini call: create an easy-to-read answer from the items
        if items:
            items_blob = _format_items_for_prompt(items)
            step("gemini_answer_request", {"items_blob_preview": items_blob[:2000]})
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
            step("final_answer_with_items", {"answer": answer})
        else:
            step("no_items_found", {})
            answer = await _gemini_generate_text(client, [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": f"The user asked: {user_text}\nNo items were found. Suggest 2–3 alternative queries in one short paragraph."}
            ]) or "No matching items found. Try specifying the product type, brand, or a price range."
            step("final_answer_no_items", {"answer": answer})

        if mode == "text":
            return PlainTextResponse(answer)
        return JSONResponse({"items": items, "answer": answer, **({"trace": trace} if debug else {})})
