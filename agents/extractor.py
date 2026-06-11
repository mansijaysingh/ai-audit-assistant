from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# ── PROMPTS ───────────────────────────────────────────────────────────────────

BS_PL_PROMPT = PromptTemplate.from_template(
    """
You are a financial audit data extraction expert for Indian CA office documents.

CRITICAL: There are TWO completely different entities in this document. Never mix them.

ENTITY 1 — CLIENT (The business being audited):
- Their firm name is the MAIN HEADING at the very top of the document
- Example: "M/S SHREE RANI SATI OIL TRADERS"
- They sign on the LEFT side at the bottom
- They are labeled as "Prop." or "Proprietor" or "Director"
- Extract into: client_firm_name, proprietor_name, pan_number

ENTITY 2 — AUDITOR (The CA firm doing the audit):
- They appear on the RIGHT side under "AUDITOR'S REPORT"
- They are labeled as "Chartered Accountants" or "Partner"
- They have M. No. (Membership Number) and FRN (Firm Registration Number)
- Extract into: ca_firm_name, ca_partner_name

=== SECTION A: CLIENT INFORMATION (Left side / Top heading) ===
{client_context}

=== SECTION B: AUDITOR INFORMATION (Right side / Auditor's Report) ===
{auditor_context}

=== SECTION C: FINANCIAL FIGURES ===
{financial_context}

Return ONLY this JSON. Use null if a field is not found.
Use plain numbers without commas or currency symbols (e.g. 946579.16 not "9,46,579.16").

{{
  "client_firm_name": "firm name from TOP HEADING — the business being audited",
  "proprietor_name": "proprietor/owner name from LEFT signature side",
  "pan_number": "PAN number of the client/assessee",
  "assessment_year": "e.g. 2024-25",
  "ca_firm_name": "CA/audit firm name from RIGHT side under AUDITOR'S REPORT",
  "ca_partner_name": "partner name from RIGHT side signature",
  "sales": null,
  "turnover": null,
  "gross_profit": null,
  "net_profit": null,
  "opening_stock": null,
  "closing_stock": null,
  "purchases": null,
  "capital_closing": null,
  "fixed_assets_closing": null,
  "debtors": null,
  "creditors": null,
  "cash": null,
  "bank_balance": null,
  "total_assets": null,
   "assessment_year": "",
    "financial_year": "",

}}

Rules:
- Return ONLY valid JSON. No markdown. No explanation. No extra text.
- NEVER put CA firm name in client_firm_name.
- NEVER put client firm name in ca_firm_name.
"""
)

CB3_PROMPT = PromptTemplate.from_template(
    """
You are a financial audit data extraction expert for Indian CA office documents.

CRITICAL: There are TWO completely different entities in Form 3CB. Never mix them.

ENTITY 1 — CLIENT (Assessee being audited):
- Appears INSIDE the form body as "Name of the Assessee"
- Their PAN is under "Permanent Account Number"
- Their address is the business address inside the form
- Extract into: client_firm_name, client_address, pan_number, gst_number

ENTITY 2 — AUDITOR (CA firm who prepared this 3CB):
- Their firm name appears at the TOP as the LETTERHEAD
- They sign at the BOTTOM with M. No. and FRN
- Extract into: ca_firm_name, ca_partner_name, membership_number, frn

=== SECTION A: LETTERHEAD (CA firm — top of form) ===
{auditor_context}

=== SECTION B: FORM BODY (Client/Assessee details) ===
{client_context}

=== SECTION C: FINANCIAL FIGURES ===
{financial_context}

=== SECTION D: SIGNATURES AND UDIN ===
{signature_context}

Return ONLY this JSON. Use null if a field is not found.

{{
  "client_firm_name": "assessee name from INSIDE the form body",
  "client_address": "assessee business address from form body",
  "pan_number": "PAN of the assessee",
  "gst_number": "GST registration number if present",
  "ca_firm_name": "CA firm name from LETTERHEAD at top",
  "ca_partner_name": "partner name from bottom signature",
  "membership_number": "M. No. of the signing partner",
  "frn": "Firm Registration Number",
  "assessment_year": "assessment year",
  "turnover": null,
  "net_profit": null,
  "depreciation": null,
  "udin": "UDIN number if present, else null",
  "assessment_year": "",
    "financial_year": "",
}}

Rules:
- Return ONLY valid JSON. No markdown. No explanation.
- If not found, use null.
- Numbers must be plain floats without commas.
- NEVER put CA firm name in client_firm_name.
- NEVER put client firm name in ca_firm_name.
"""
)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _deduplicate(docs):
    """Remove duplicate chunks and return joined string."""
    seen = set()
    unique = []
    for doc in docs:
        content = doc.page_content.strip()
        if content and content not in seen:
            seen.add(content)
            unique.append(content)
    return "\n\n".join(unique)


def _retrieve(retriever, query, k=3):
    """Fetch top-k unique chunks for a query."""
    docs = retriever.invoke(query)
    return _deduplicate(docs[:k])


# ── CONTEXT BUILDERS ──────────────────────────────────────────────────────────

def _build_bs_pl_context(retriever):
    """
    Build 3-section context for Balance Sheet / P&L documents.
    Keeps client and auditor information in separate sections so
    GPT never confuses the two entities.
    Token-efficient: deduplicates chunks and limits per section.
    """
    client_context = _retrieve(
        retriever,
        "firm name heading top M/S proprietor owner assessee business name",
        k=3
    )
    auditor_context = _retrieve(
        retriever,
        "auditor chartered accountant right side signature partner CA firm report",
        k=3
    )
    financial_context = "\n\n".join(filter(None, [
    _retrieve(retriever, "sales turnover gross profit net profit", k=2),
    _retrieve(retriever, "opening stock closing stock purchases wages freight", k=2),
    _retrieve(retriever, "capital fixed assets debtors creditors cash bank balance", k=2),
]))

    return {
        "client_context":   client_context   or "Not found.",
        "auditor_context":  auditor_context  or "Not found.",
        "financial_context": financial_context or "Not found.",
    }


def _build_cb3_context(retriever):
    """
    Build 4-section context for 3CB / 3CD documents.
    Separates letterhead (CA) from form body (client) for clarity.
    """
    auditor_context = _retrieve(
        retriever,
        "chartered accountants letterhead top firm address phone email",
        k=3
    )
    client_context = _retrieve(
        retriever,
        "name of assessee address permanent account number PAN GST registration number",
        k=3
    )
    financial_context = "\n\n".join(filter(None, [
        _retrieve(retriever, "turnover sales net profit gross profit", k=2),
        _retrieve(retriever, "depreciation fixed assets", k=2),
    ]))
    signature_context = _retrieve(
        retriever,
        "UDIN membership number M No FRN partner signature date place",
        k=3
    )

    return {
        "auditor_context":   auditor_context   or "Not found.",
        "client_context":    client_context    or "Not found.",
        "financial_context": financial_context or "Not found.",
        "signature_context": signature_context or "Not found.",
    }


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def extract_financial_data(retriever, document_type: str) -> dict:
    """
    Extract structured financial data from a document using RAG.

    Args:
        retriever   : FAISS retriever built from the document
        document_type : "current" | "previous" | "3cb"

    Returns:
        dict with all extracted fields, or {} if extraction fails
    """
    parser = JsonOutputParser()

    try:
        if document_type in ("current", "previous"):
            context = _build_bs_pl_context(retriever)
            chain = BS_PL_PROMPT | llm | parser

        elif document_type == "3cb":
            context = _build_cb3_context(retriever)
            chain = CB3_PROMPT | llm | parser

        else:
            raise ValueError(f"Unknown document_type: '{document_type}'. Use 'current', 'previous', or '3cb'.")

        result = chain.invoke(context)
        return result

    except Exception as e:
        print(f"[ExtractorAgent] ERROR for '{document_type}': {e}")
        return {}


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ExtractorAgent ready.")
    print("Usage: extract_financial_data(retriever, document_type)")
    print("document_type options: 'current', 'previous', '3cb'")