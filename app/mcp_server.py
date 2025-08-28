# app/mcp_server.py
from __future__ import annotations

import os, csv, threading, difflib
from typing import Any, Dict, List, Optional, TypedDict, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .config import settings  # loads .env values (GEMINI key, CORS, etc.)

# ──────────────────────────────────────────────────────────────────────────────
# JSON-RPC 2.0 types (MCP uses JSON-RPC with methods like tools/list, tools/call)
# ──────────────────────────────────────────────────────────────────────────────

class RPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class RPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Any] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

# ──────────────────────────────────────────────────────────────────────────────
# Server setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="MCP Price Demo")

# CORS (so your frontend can call this server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=(["*"] if settings.CORS_ORIGINS == "*" else [o.strip() for o in settings.CORS_ORIGINS.split(",")]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "products.csv")

if not os.path.exists(CSV_PATH):
    print(f"[WARN] products.csv not found at: {CSV_PATH} — server will return empty results.")

class ProductRow(TypedDict):
    item_name: str
    item_type: str
    item_rate: float
    item_price: float
    marketplace: str

class ProductStore:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._rows: List[ProductRow] = []
        self._mtime: float = 0.0
        self._load_if_changed(force=True)

    def _read_csv(self) -> List[ProductRow]:
        rows: List[ProductRow] = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        rows.append(ProductRow(
                            item_name=r["item_name"].strip(),
                            item_type=r["item_type"].strip().lower(),
                            item_rate=float(str(r["item_rate"]).replace(",", ".")),
                            item_price=float(str(r["item_price"]).replace(",", ".")),
                            marketplace=r["marketplace"].strip().lower(),
                        ))
                    except Exception:
                        # Skip malformed row in demo
                        continue
        except FileNotFoundError:
            # Return empty list if file missing (already warned)
            return []
        return rows

    def _load_if_changed(self, force: bool = False):
        try:
            m = os.path.getmtime(self.csv_path)
        except FileNotFoundError:
            return
        if force or m != self._mtime:
            with self._lock:
                try:
                    m2 = os.path.getmtime(self.csv_path)
                except FileNotFoundError:
                    return
                if force or m2 != self._mtime:
                    self._rows = self._read_csv()
                    self._mtime = m2

    def all(self) -> List[ProductRow]:
        self._load_if_changed()
        return list(self._rows)

STORE = ProductStore(CSV_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# Fuzzy helpers
# ──────────────────────────────────────────────────────────────────────────────

def _closest(needle: str, candidates: List[str], cutoff: float = 0.72) -> Optional[str]:
    needle = (needle or "").strip().lower()
    if not needle or not candidates:
        return None
    matches = difflib.get_close_matches(needle, [c.lower() for c in candidates], n=1, cutoff=cutoff)
    return matches[0] if matches else None

def _name_like(name: str, needle: str) -> bool:
    """Accept substring OR fuzzy similarity on item_name."""
    if not needle:
        return True
    if needle in name:
        return True
    return difflib.SequenceMatcher(a=needle, b=name).ratio() >= 0.72

# ──────────────────────────────────────────────────────────────────────────────
# MCP "modern" tool registry definition
# - tools/list returns this schema (names + JSON schemas)
# - tools/call dispatches to functions below
# ──────────────────────────────────────────────────────────────────────────────

# JSON schema fragments for tool arguments
AMAZON_ARGS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Substring to match in item_name (case-insensitive)."},
        "item_type": {"type": "string", "description": "Filter by category, e.g., toothbrush, toothpaste, mouthwash."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}
    },
    "additionalProperties": False
}
TRENDYOL_ARGS = AMAZON_ARGS
HEPSI_ARGS = AMAZON_ARGS
DECIDER_ARGS = {
    "type": "object",
    "properties": {
        "marketplaces": {
            "type": "array",
            "items": {"type": "string", "enum": ["amazon", "trendyol", "hepsiburada"]},
            "description": "Which marketplaces to include."
        },
        "query": {"type": "string", "description": "Substring to match in item_name."},
        "item_type": {"type": "string", "description": "Filter by category (item_type)."},
        "limit_per_marketplace": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}
    },
    "required": ["marketplaces"],
    "additionalProperties": False
}

ToolResult = Dict[str, Any]

def _filter_rows(
    rows: List[ProductRow],
    marketplace: Optional[str] = None,
    query: Optional[str] = None,
    item_type: Optional[str] = None,
    limit: int = 50
) -> List[ProductRow]:
    """Exact filter first; if empty and filters were provided, retry with fuzzy matching."""
    out: List[ProductRow] = []
    q = (query or "").strip().lower()
    t = (item_type or "").strip().lower()

    # Exact pass
    for r in rows:
        if marketplace and r["marketplace"] != marketplace:
            continue
        if q and q not in r["item_name"].lower():
            continue
        if t and r["item_type"] != t:
            continue
        out.append(r)
        if len(out) >= limit:
            return out

    # Fuzzy pass (only if nothing found and at least one filter present)
    if not out and (q or t):
        # fuzzy type: snap to closest existing type (within marketplace if given)
        t2: Optional[str] = None
        if t:
            type_pool = sorted({r["item_type"] for r in rows if (not marketplace or r["marketplace"] == marketplace)})
            t2 = _closest(t, type_pool, cutoff=0.72)

        for r in rows:
            if marketplace and r["marketplace"] != marketplace:
                continue
            if q and not _name_like(r["item_name"].lower(), q):
                continue
            if t2 and r["item_type"] != t2:
                continue
            out.append(r)
            if len(out) >= limit:
                break

    return out

def list_amazon(params: Dict[str, Any]) -> ToolResult:
    limit = int(params.get("limit", 50))
    return {"items": _filter_rows(STORE.all(), "amazon", params.get("query"), params.get("item_type"), limit)}

def list_trendyol(params: Dict[str, Any]) -> ToolResult:
    limit = int(params.get("limit", 50))
    return {"items": _filter_rows(STORE.all(), "trendyol", params.get("query"), params.get("item_type"), limit)}

def list_hepsiburada(params: Dict[str, Any]) -> ToolResult:
    limit = int(params.get("limit", 50))
    return {"items": _filter_rows(STORE.all(), "hepsiburada", params.get("query"), params.get("item_type"), limit)}

def list_products_decider(params: Dict[str, Any]) -> ToolResult:
    mks: List[str] = params.get("marketplaces", [])
    query = params.get("query")
    item_type = params.get("item_type")
    limit = int(params.get("limit_per_marketplace", 50))

    result: Dict[str, Any] = {"by_marketplace": {}}
    for mk in mks:
        items = _filter_rows(STORE.all(), mk, query, item_type, limit)
        result["by_marketplace"][mk] = items
    return result

def list_item_types(_: Dict[str, Any]) -> ToolResult:
    """Expose distinct item_type values so the LLM/UI can suggest valid categories."""
    types = sorted({r["item_type"] for r in STORE.all()})
    return {"item_types": types}

ToolEntry = Dict[str, Any]

TOOLS: Dict[str, Dict[str, Any]] = {
    "list_amazon_products": {
        "name": "list_amazon_products",
        "description": "List products from Amazon with optional query and item_type filters.",
        "input_schema": AMAZON_ARGS,
        "handler": list_amazon,
    },
    "list_trendyol_products": {
        "name": "list_trendyol_products",
        "description": "List products from Trendyol with optional query and item_type filters.",
        "input_schema": TRENDYOL_ARGS,
        "handler": list_trendyol,
    },
    "list_hepsiburada_products": {
        "name": "list_hepsiburada_products",
        "description": "List products from Hepsiburada with optional query and item_type filters.",
        "input_schema": HEPSI_ARGS,
        "handler": list_hepsiburada,
    },
    "list_products_decider": {
        "name": "list_products_decider",
        "description": "Decide which marketplaces to query and return a merged, per-marketplace listing.",
        "input_schema": DECIDER_ARGS,
        "handler": list_products_decider,
    },
    "list_item_types": {
        "name": "list_item_types",
        "description": "List all distinct item_type values present in the catalog.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": list_item_types,
    },
}

def _mcp_tools_list_payload() -> Dict[str, Any]:
    """Return MCP-style tools metadata (names + JSON Schemas)."""
    tools_meta: List[ToolEntry] = []
    for key, t in TOOLS.items():
        tools_meta.append({
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["input_schema"],
        })
    return {"tools": tools_meta}

def _mcp_tools_call_result_payload(value: Any) -> Dict[str, Any]:
    """
    MCP convention: return a content array with a JSON object.
    (This mirrors modern MCP tool response shape.)
    """
    return {
        "content": [
            {"type": "json", "json": value}
        ]
    }

# ──────────────────────────────────────────────────────────────────────────────
# Single JSON-RPC endpoint (HTTP) for MCP
# methods supported: tools/list, tools/call
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/")
async def mcp_endpoint(req: Request):
    # Parse JSON safely → return JSON-RPC -32700 on invalid body
    try:
        data = await req.json()
    except Exception:
        return RPCResponse(id=None, error={"code": -32700, "message": "Parse error: invalid JSON"})

    rpc = RPCRequest(**data)

    try:
        if rpc.method == "tools/list":
            return RPCResponse(id=rpc.id, result=_mcp_tools_list_payload())

        if rpc.method == "tools/call":
            # Expected params: {"name": <tool_name>, "arguments": {...}}
            if not rpc.params or "name" not in rpc.params:
                return RPCResponse(
                    id=rpc.id,
                    error={"code": -32602, "message": "Missing 'name' in params for tools/call."}
                )
            name = rpc.params["name"]
            args = rpc.params.get("arguments", {}) or {}

            # Helpful validation for the decider
            if name == "list_products_decider" and not args.get("marketplaces"):
                return RPCResponse(
                    id=rpc.id,
                    error={"code": -32602, "message": "list_products_decider requires 'marketplaces': [\"amazon\"|\"trendyol\"|\"hepsiburada\"]"}
                )

            tool = TOOLS.get(name)
            if not tool:
                return RPCResponse(
                    id=rpc.id,
                    error={"code": -32601, "message": f"Tool '{name}' not found."}
                )

            # Run
            result = tool["handler"](args)
            return RPCResponse(id=rpc.id, result=_mcp_tools_call_result_payload(result))

        # (Optional) implement "initialize" if your client expects it.
        if rpc.method == "initialize":
            return RPCResponse(id=rpc.id, result={"capabilities": {"tools": True}})

        return RPCResponse(
            id=rpc.id,
            error={"code": -32601, "message": f"Method '{rpc.method}' not found."}
        )
    except Exception as e:
        return RPCResponse(
            id=rpc.id,
            error={"code": -32000, "message": f"Internal error: {e!r}"}
        )

from .orchestrator import router as orchestrator_router
app.include_router(orchestrator_router, prefix="/api")
