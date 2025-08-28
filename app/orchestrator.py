# app/orchestrator.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
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

# ---------- Public API endpoint ----------

@router.post("/chat")
async def chat(payload: dict):
    """
    Body: { "message": "user text here" }
    Returns: { "items": [ {marketplace, item_name, item_type, item_price, item_rate}, ... ] }
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
            return {"items": []}

        tool_name = fc["name"]
        args = fc.get("args") or {}
        mcp_res = await mcp_rpc(client, "tools/call", {"name": tool_name, "arguments": args}, _id=101)
        mcp_json = (mcp_res.get("content") or [{}])[0].get("json")

        by_mk = _normalize_by_marketplace(tool_name, mcp_json)
        items = _flatten_items(by_mk, max_items=10)
        return {"items": items}
