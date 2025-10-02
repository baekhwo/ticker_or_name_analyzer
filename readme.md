# Stock Filing Analyzer (Ticker or Company Name)

This project analyzes company filings (10-K / 10-Q) from the SEC EDGAR system and generates a structured financial report using **Upstage SOLAR API**.

---

## Features
- Search company by ticker or name → auto-detect CIK
- Fetch most recent **10-K or 10-Q** from SEC
- Extract & normalize key financial data (via XBRL)
- Analyze filings with Upstage SOLAR (`solar-pro2`)
- Fetch and align recent news from MarketAux API
- Cache ticker/industry decisions for efficiency

---

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
