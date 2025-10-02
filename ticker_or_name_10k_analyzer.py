# ticker_or_name_10k_analyzer_safe.py
# Ticker or company name -> find CIK -> fetch latest 10-K/10-Q -> analyze via Upstage SOLAR (HTTP)
# Safe for frequent testing: polite rate limiting, retries, and ticker-map caching.

import os
import re
import json
import time
import difflib
from datetime import datetime, timedelta, timezone  # ← added timezone

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
load_dotenv()  # reads .env from the current working directory

# --------- CONFIG (SET THESE) ----------
# near the top, after imports
UPSTAGE_API_KEY    = os.getenv("UPSTAGE_API_KEY", "")
SEC_USER_AGENT     = os.getenv("SEC_USER_AGENT", "")
MARKET_AUX_API_KEY = os.getenv("MARKET_AUX_API_KEY", "")
# ---------------------------------------

MODEL = "solar-pro2"
UPSTAGE_CHAT_COMPLETIONS_URL = "https://api.upstage.ai/v1/chat/completions"

SYSTEM_PROMPT = """
You are a clear, no-fluff financial analyst and teacher.

TASKS
A) Inferred Sector/Industry (before any ratios)
- From the 10-K/10-Q text (esp. Business description, Risk Factors) and the company/ticker, infer:
  • Sector (broad), Industry (specific), Sub-sector (optional)
  • Confidence: 0.0–1.0
  • Evidence: short quotes/phrases you used (e.g., “membership warehouse retailer”, “smartphones”, “semiconductor foundry”).
- When SEC SIC is provided, PREFER the SIC hint to guide the choice. If SIC conflicts with the text, pick the closest match and explain briefly.
- Choose EXACTLY ONE industry label from this controlled list (output a single label only):
  {Tech Hardware, Semiconductors, Software, Internet Platforms/Services, Consumer Electronics Retail, Brick-and-Mortar Retail (General),
   Grocery/Consumer Staples Retail, Apparel/Discretionary Retail, Membership Warehouse Retail, Consumer Packaged Goods (Staples),
   Utilities, Energy E&P, Integrated Oil & Gas, Industrials (Transportation/Logistics), Industrials (Capital Goods),
   Financials (Banks), Financials (Insurance), Real Estate (REIT), Healthcare (Pharma/Biotech), Healthcare Providers, Telecom}

B) Key Numbers (verbatim if possible, must come first in output)
- If a FROZEN_XBRL_JSON block is provided, COPY those figures EXACTLY for overlapping items and do not re-extract from tables.
- Always list the following if present (using XBRL values when available):
  Revenue, Gross Profit, Operating Income, Net Income, EPS (basic/diluted if present),
  Cash & Equivalents, Total Debt, Total Assets, Total Liabilities, Equity,
  Operating CF, Investing CF, Financing CF.
- If a figure isn't visible anywhere, say: Not found (and mention what section/page you’d check).
- Show numbers before moving to qualitative analysis. If filing text conflicts with XBRL, keep the XBRL number and note the discrepancy/units/period scope.

C) Ratios & Multiples (show formulas & inputs)
- Profitability: Gross Margin, Operating Margin, Net Margin
- Liquidity: Current Ratio, Quick Ratio (if possible)
- Leverage: Debt/Equity, Interest Coverage
- Valuation (ONLY if price/share or market cap is in the text; otherwise mark N/A): P/E, P/S, P/B

D) Interpret with Benchmarks (industry-aware)
- Select benchmark ranges based on inferred industry (from A). If uncertain, use “generic” heuristics and state that explicitly.
- For each metric: Company Value → Benchmark Range → Verdict (e.g., “healthy”, “elevated”, “below peer norm”).
- Example heuristic ranges (adjust only if filing text justifies):
  • Tech Hardware: P/E 12–25, Net Margin 8–20%, D/E 0–1.0, Current 1.5–3.0
  • Semiconductors: P/E 15–30, Net Margin 15–35%, D/E 0–0.8, Current 2.0–4.0
  • Software: P/E 25–60, Net Margin 10–30%, D/E 0–0.6, Current 1.2–3.0
  • Membership Warehouse Retail: P/E 20–40, Net Margin 2–4%, D/E 0.3–1.0, Current 0.9–1.3
  • Grocery/Staples Retail: P/E 10–20, Net Margin 1–3%, D/E 0.5–1.5, Current 0.8–1.3
  • Brick-and-Mortar Retail: P/E 8–18, Net Margin 2–6%, D/E 0.5–2.0, Current 1.0–1.8
  • Banks: P/E 8–14, ROE 8–15% (note: CET1 adequacy > liquidity ratios)
  • REITs: P/FFO > P/E, Debt/EBITDA & coverage most relevant
- Always state if you’re using generic vs. industry-specific benchmarks.

E) Risks & Opportunities
- List 3–5 bullets, balanced, tied to either the filing text or logical inference.
- Explicitly note whether each is “from filing” or “inferred”.

F) Recent News Headlines (if provided by user input)
- Compare the headlines to the risks/opportunities.
- Explicitly state: “Reinforces risk X” or “Contradicts opportunity Y”.
- If no news headlines were supplied, skip this section.

G) Plain-English Summary (5–8 sentences)
- Summarize the financial position, valuation vs. benchmarks, strengths/weaknesses, and whether recent news shifts the risk picture.

OUTPUT FORMAT (use these headers, in order)
1) Key Numbers
2) Ratios (with formulas & benchmarks)
3) Inferred Sector/Industry
   - Sector:
   - Industry (single label):
   - Sub-sector (optional):
   - Confidence (0.0–1.0):
   - Evidence:
   - Benchmark Set Used: <industry name or 'generic'>
4) Risks/Opportunities
5) Recent News Alignment
6) Summary

RULES
- Numbers must always appear first in the output.
- Prefer SIC when provided; choose a single industry label from the list.
- When FROZEN_XBRL_JSON is present, use it as the source of truth for overlapping items.
- Be transparent about assumptions; never invent figures not in the text.
- Keep tone concise, professional, and structured.
"""

CACHE_PATH = ".sec_ticker_cache.json"
TICKER_CACHE_TTL_HOURS = 24

# ---- polite SEC session (rate limit + retries) ----
session = requests.Session()
SEC_HEADERS_BASE = {
    "User-Agent": SEC_USER_AGENT or "PLEASE_SET_YOUR_EMAIL",
    "Accept-Encoding": "gzip, deflate",
}
last_sec_request_ts = 0.0
MIN_SEC_INTERVAL = 0.6  # seconds between SEC calls (polite: ~1-2 rps max)

def sec_request(url, *, headers=None, timeout=30, method="GET", attempts=5):
    """Polite SEC request with spacing + retry/backoff for 429/403/5xx."""
    global last_sec_request_ts
    hdrs = SEC_HEADERS_BASE.copy()
    if headers:
        hdrs.update(headers)

    # polite spacing
    elapsed = time.time() - last_sec_request_ts
    if elapsed < MIN_SEC_INTERVAL:
        time.sleep(MIN_SEC_INTERVAL - elapsed)

    backoff = 1.0
    for i in range(attempts):
        try:
            resp = session.request(method, url, headers=hdrs, timeout=timeout)
            last_sec_request_ts = time.time()
            # throttle / server busy
            if resp.status_code in (429, 403, 500, 502, 503, 504):
                if i < attempts - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if i < attempts - 1:
                time.sleep(backoff)
                backoff = min(backoff * 2, 8)
                continue
            raise

# ---- helpers ----
def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def read_pdf_to_text_bytes(b: bytes) -> str:
    tmp = "temp_sec_doc.pdf"
    with open(tmp, "wb") as f:
        f.write(b)
    try:
        return extract_text(tmp) or ""
    finally:
        try:
            os.remove(tmp)
        except:
            pass

def read_html_to_text_bytes(b: bytes) -> str:
    soup = BeautifulSoup(b, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))

# ---- MarketAux news (optional) ----
def fetch_recent_news_marketaux(ticker: str, api_key: str, limit: int = 5) -> str:
    """
    Returns a markdown string with up to `limit` headlines for the ticker.
    If `api_key` is blank or the API errors, returns an empty string.
    Sorted for determinism.
    """
    if not api_key or not ticker:
        return ""
    try:
        url = (
            "https://api.marketaux.com/v1/news/all"
            f"?symbols={ticker}&filter_entities=true&language=en&api_token={api_key}"
        )
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return ""
        data = r.json()
        articles = data.get("data", [])[:limit]
        # sort deterministically: oldest→newest, then title
        articles = sorted(articles, key=lambda a: ((a.get("published_at") or ""), (a.get("title") or "")))
        rows = []
        for article in articles:
            title = article.get("title") or ""
            link = article.get("url") or ""
            published = article.get("published_at") or ""
            rows.append(f"- {title} ({published}) {link}")
        return "\n".join(rows)
    except Exception:
        return ""

# ---- ticker map caching ----
def load_ticker_map_cached():
    # if cache exists and fresh, use it
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                blob = json.load(f)
            ts = datetime.fromisoformat(blob["timestamp"])
            # handle old naive timestamps gracefully
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - ts < timedelta(hours=TICKER_CACHE_TTL_HOURS):
                return blob["data"]
    except Exception:
        pass

    # fetch fresh
    url = "https://www.sec.gov/files/company_tickers.json"
    r = sec_request(url, timeout=30)
    data = r.json()

    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "data": data}, f)
    except Exception:
        pass

    return data

def find_company(query: str):
    """
    Returns a tuple (cik10, ticker, title). Tries:
      1) Exact/upper ticker match
      2) Case-insensitive name contains
      3) Fuzzy name match (difflib)
    """
    data = load_ticker_map_cached()
    records = []
    for _, row in data.items():
        ticker = (row.get("ticker") or "").upper()
        title = row.get("title") or ""
        cik = str(row.get("cik_str") or "").zfill(10)
        records.append((cik, ticker, title))

    q = query.strip()
    q_upper = q.upper()

    # 1) exact ticker match
    for cik, tk, ttl in records:
        if tk == q_upper:
            return cik, tk, ttl

    # 2) simple name contains
    name_hits = [(cik, tk, ttl) for cik, tk, ttl in records if q.lower() in ttl.lower()]
    if len(name_hits) == 1:
        return name_hits[0]
    if len(name_hits) > 1:
        candidates = [ttl for _, _, ttl in name_hits]
        best = difflib.get_close_matches(q, candidates, n=1, cutoff=0.0)
        if best:
            best_name = best[0]
            for rec in name_hits:
                if rec[2] == best_name:
                    return rec

    # 3) fuzzy across all names
    all_names = [ttl for _, _, ttl in records]
    best = difflib.get_close_matches(q, all_names, n=1, cutoff=0.0)
    if best:
        best_name = best[0]
        for rec in records:
            if rec[2] == best_name:
                return rec

    return None

def fetch_latest_filing_text(cik10: str, form: str = "10-K"):
    subs_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = sec_request(subs_url, timeout=30)
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    fdates = recent.get("filingDate", [])
    rdates = recent.get("reportDate", [])

    candidates = []
    for i, f in enumerate(forms):
        if f and f.upper() == form.upper() and i < len(accs) and i < len(docs):
            candidates.append({
                "acc_nodash": accs[i].replace("-", ""),
                "doc": docs[i],
                "filing_date": fdates[i] if i < len(fdates) else "",
                "report_date": rdates[i] if i < len(rdates) else "",
            })

    if not candidates:
        return f"[SEC] No recent {form} found.", {}

    # pick most recent
    def parse_dt(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except:
            return datetime.min

    candidates.sort(key=lambda x: parse_dt(x["filing_date"]), reverse=True)
    chosen = candidates[0]
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{chosen['acc_nodash']}/{chosen['doc']}"

    r = sec_request(doc_url, timeout=60)
    content_type = (r.headers.get("content-type") or "").lower()

    if "html" in content_type or chosen["doc"].lower().endswith((".htm", ".html", ".txt")):
        text = read_html_to_text_bytes(r.content)
    elif chosen["doc"].lower().endswith(".pdf") or "pdf" in content_type:
        text = read_pdf_to_text_bytes(r.content)
    else:
        try:
            text = r.text
        except:
            text = "[SEC] Unsupported document format."

    meta = {
        "doc_url": doc_url,
        "filing_date": chosen.get("filing_date", ""),
        "report_date": chosen.get("report_date", ""),
        "doc_name": chosen.get("doc", ""),
    }
    return text, meta

# ---- SIC fetch + industry caching / normalization ----
def fetch_company_sic(cik10: str):
    try:
        subs_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        r = sec_request(subs_url, timeout=30)
        data = r.json()
        return data.get("sic"), data.get("sicDescription")
    except Exception:
        return None, None

INDUSTRY_CACHE_FILE = ".industry_cache.json"

def load_industry_cache():
    try:
        with open(INDUSTRY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_industry_cache(cache: dict):
    try:
        with open(INDUSTRY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

NORMALIZE_INDUSTRY = {
    "online services": "Internet Platforms/Services",
    "internet services": "Internet Platforms/Services",
    "internet platforms": "Internet Platforms/Services",
    "cloud software": "Software",
    "saas": "Software",
    "software-as-a-service": "Software",
    "consumer electronics": "Tech Hardware",
    "hardware": "Tech Hardware",
    "warehouse club": "Membership Warehouse Retail",
    "retail (warehouse club)": "Membership Warehouse Retail",
}

def normalize_industry(label: str) -> str:
    key = (label or "").strip().lower()
    return NORMALIZE_INDUSTRY.get(key, label)

# ---- XBRL (Company Facts) helpers: pull authoritative numbers ----
XBRL_TAGS = {
    "Assets": "us-gaap/Assets",
    "Liabilities": "us-gaap/Liabilities",
    "StockholdersEquity": "us-gaap/StockholdersEquity",
    "StockholdersEquityAlt": "us-gaap/StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "AssetsCurrent": "us-gaap/AssetsCurrent",
    "LiabilitiesCurrent": "us-gaap/LiabilitiesCurrent",
    "CashAndCashEquivalents": "us-gaap/CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebt": "us-gaap/LongTermDebtNoncurrent",
    "DebtCurrent": "us-gaap/DebtCurrent",
}

def _unit_multiplier(unit_name):
    u = (unit_name or "").lower()
    if u == "usd":
        return 1.0
    if "usd(m)" in u or "usdm" in u or "million" in u:
        return 1_000_000.0
    if "thousand" in u or "usd(th)" in u or "usd thousands" in u:
        return 1_000.0
    return 1.0

def _prefer_units(units_dict):
    if not isinstance(units_dict, dict):
        return []
    usd = []
    m = []
    th = []
    other = []
    for unit_name, facts in units_dict.items():
        ln = unit_name.lower()
        if ln == "usd":
            usd.append((unit_name, facts))
        elif "usd(m)" in ln or "usdm" in ln or "million" in ln:
            m.append((unit_name, facts))
        elif "thousand" in ln or "usd(th)" in ln or "usd thousands" in ln:
            th.append((unit_name, facts))
        else:
            other.append((unit_name, facts))
    return usd + m + th + other

def _parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None

def _best_fact_for_date(facts_list, target_date):
    if not facts_list:
        return None
    scored = []
    for f in facts_list:
        end = _parse_date(f.get("end") or "")
        if not end:
            continue
        form = (f.get("form") or "").upper()
        val = f.get("val")
        if val is None:
            continue
        dist = abs((end - target_date).days) if target_date else 9999
        fy_penalty = 0 if (target_date and f.get("fy") == target_date.year) else 30
        form_bonus = -5 if form in ("10-K", "10-Q") else 0
        scored.append((dist + fy_penalty + form_bonus, end, f))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=False)
    return scored[0][2]

def fetch_company_facts(cik10: str):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    try:
        r = sec_request(url, timeout=30)
        return r.json()
    except Exception:
        return None

def get_xbrl_values_for_filing(cik10: str, report_date_str: str) -> dict:
    out = {}
    facts = fetch_company_facts(cik10)
    if not facts:
        return out

    root = facts.get("facts", {})
    report_date = _parse_date(report_date_str) if report_date_str else None

    def pull(label: str, *alt_labels: str):
        path = XBRL_TAGS.get(label)
        alts = [XBRL_TAGS.get(a) for a in alt_labels if XBRL_TAGS.get(a)]
        for concept_path in [path] + alts:
            if not concept_path:
                continue
            ns, name = concept_path.split("/", 1)
            ns_dict = root.get(ns, {})
            concept = ns_dict.get(name, {})
            units = concept.get("units")
            if not units:
                continue
            for unit_name, facts_list in _prefer_units(units):
                best = _best_fact_for_date(facts_list, report_date)
                if best:
                    mult = _unit_multiplier(unit_name)
                    try:
                        val = float(best.get("val")) * mult
                    except Exception:
                        continue
                    out[label] = {
                        "value_usd": val,
                        "as_of": best.get("end"),
                        "raw_unit": unit_name,
                        "source_tag": concept_path,
                        "form": best.get("form"),
                    }
                    return

    pull("Assets")
    pull("Liabilities")
    pull("StockholdersEquity", "StockholdersEquityAlt")
    pull("AssetsCurrent")
    pull("LiabilitiesCurrent")
    pull("CashAndCashEquivalents")
    pull("LongTermDebt")
    pull("DebtCurrent")

    return out

def format_xbrl_md(x: dict) -> str:
    if not x:
        return ""
    def fmt_usd(v):
        try:
            return "${:,.0f}".format(v)
        except Exception:
            return str(v)
    lines = ["Authoritative XBRL facts (SEC; normalized to USD):"]
    order = [
        "Assets", "Liabilities", "StockholdersEquity",
        "AssetsCurrent", "LiabilitiesCurrent",
        "CashAndCashEquivalents", "LongTermDebt", "DebtCurrent",
    ]
    labels = {
        "Assets": "Total Assets",
        "Liabilities": "Total Liabilities",
        "StockholdersEquity": "Stockholders’ Equity",
        "AssetsCurrent": "Current Assets",
        "LiabilitiesCurrent": "Current Liabilities",
        "CashAndCashEquivalents": "Cash & Cash Equivalents",
        "LongTermDebt": "Long-Term Debt (Noncurrent)",
        "DebtCurrent": "Debt (Current)",
    }
    for k in order:
        if k in x:
            d = x[k]
            lines.append(f"- {labels[k]}: {fmt_usd(d['value_usd'])} (as of {d.get('as_of')}; {d.get('source_tag')} {d.get('form')})")
    lines.append("Use these figures as authoritative. If table-text differs, explain units/period/scope differences.")
    return "\n".join(lines)

def shrink_xbrl_for_prompt(x: dict) -> dict:
    """Keep only deterministic, minimal fields for the frozen JSON."""
    out = {}
    for k, v in (x or {}).items():
        out[k] = {
            "value_usd": v.get("value_usd"),
            "as_of": v.get("as_of"),
            "tag": v.get("source_tag"),
        }
    return out

# ---- Upstage SOLAR via HTTP ----
def call_solar_http(user_text: str, temperature: float = 0.0) -> str:  # deterministic
    if not UPSTAGE_API_KEY or not SEC_USER_AGENT:
        raise RuntimeError("Please set UPSTAGE_API_KEY and SEC_USER_AGENT at the top of this file.")
    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "temperature": temperature,
        "top_p": 0.0,  # ← fully deterministic sampling
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }
    resp = requests.post(UPSTAGE_CHAT_COMPLETIONS_URL, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:600]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, indent=2)

# ---- main ----
def main():
    if not UPSTAGE_API_KEY or not SEC_USER_AGENT:
        print("⚠️  Set UPSTAGE_API_KEY and SEC_USER_AGENT (with a real email) at the top of this file.")
        return

    print("=== Stock Filing Analyzer (Ticker OR Company Name) ===")
    query = input("Enter ticker or company name: ").strip()
    if not query:
        print("No input. Exiting.")
        return

    choice = input("Choose filing [1=10-K, 2=10-Q] (default 1): ").strip()
    form = "10-K" if choice in ("", "1") else "10-Q"

    print(f"\nLooking up: {query} …")
    try:
        found = find_company(query)
    except Exception as e:
        print("[SEC lookup error]", e)
        return

    if not found:
        print("Could not find a matching company on the SEC ticker list.")
        return

    cik10, ticker, title = found
    print(f"Found: {title} (Ticker: {ticker}) | CIK: {cik10}")

    # --- SIC metadata + sticky industry hint ---
    sic, sic_desc = fetch_company_sic(cik10)
    if sic_desc:
        print(f"SEC SIC: {sic} - {sic_desc}")

    cache = load_industry_cache()
    sticky_industry = cache.get(ticker)
    if sticky_industry:
        sticky_hint = f"Prior industry decision for {ticker}: {sticky_industry}. Use this unless the filing explicitly contradicts it."
    else:
        sticky_hint = "No prior industry decision cached."

    print(f"Fetching latest {form} …")
    try:
        filing_text, meta = fetch_latest_filing_text(cik10, form=form)
    except Exception as e:
        print("[SEC fetch error]", e)
        return

    if not filing_text or filing_text.startswith("[SEC]"):
        print(filing_text if filing_text else f"Could not fetch {form} text.")
        return

    print("Analyzing with Upstage SOLAR …")
    MAX_CHARS = 180_000

    # --- Authoritative XBRL values for the filing date ---
    xbrl_vals = get_xbrl_values_for_filing(cik10, meta.get("report_date",""))
    xbrl_md = format_xbrl_md(xbrl_vals)
    frozen_json = json.dumps(shrink_xbrl_for_prompt(xbrl_vals), sort_keys=True)

    sic_line = f"SEC SIC: {sic} - {sic_desc}" if sic_desc else "SEC SIC: Unknown"
    user_block = (
        sticky_hint + "\n\n" +
        f"Company: {title} (Ticker: {ticker}) | Filing: {form}\n"
        f"{sic_line}\n"
        f"Filing date: {meta.get('filing_date','')} | Report date: {meta.get('report_date','')}\n"
        f"Source: {meta.get('doc_url','')}\n\n"
        + (xbrl_md + "\n\n" if xbrl_md else "")
        + "FROZEN_XBRL_JSON (copy these figures verbatim where overlapping):\n"
        + frozen_json + "\n\n"
        + "FILING TEXT (may be truncated):\n\n"
        + f"{filing_text[:MAX_CHARS]}"
    )

    # ---- Append MarketAux news (optional) ----
    news_md = fetch_recent_news_marketaux(ticker, MARKET_AUX_API_KEY, limit=5)
    if news_md:
        user_block += f"\n\nRecent News Headlines (MarketAux):\n{news_md}\n\nPlease consider whether any headlines reinforce or contradict the filing's risks/opportunities."

    try:
        analysis = call_solar_http(user_block, temperature=0.0)  # deterministic
    except Exception as e:
        print("[LLM error]", e)
        return

    print("\n========== ANALYSIS ==========\n")
    print(analysis)

    # ---- parse decided industry and cache it (normalized) ----
    try:
        m = re.search(r"Industry(?:\s*\(single label\))?:\s*(.+)", analysis, re.IGNORECASE)
        if m:
            decided = m.group(1).strip()
            decided_norm = normalize_industry(decided)
            if decided_norm:
                cache[ticker] = decided_norm
                save_industry_cache(cache)
    except Exception:
        pass

    safe_ticker = (ticker or title).replace(" ", "_")
    out_name = f"{safe_ticker}_{form}_analysis.txt"
    try:
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(analysis)
        print(f"\n✅ Saved to: {out_name}")
    except Exception as e:
        print("[Save error]", e)

if __name__ == "__main__":
    main()
