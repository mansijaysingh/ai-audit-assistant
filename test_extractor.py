from utils.pdf_reader import extract_document
from utils.rag_builder import build_vectorstore
from agents.extractor import extract_financial_data

# ── PDF FILES — apne actual file names yahan likho ────────────────────────────
current_pdf  = "BSnew.pdf"
previous_pdf = "BSold.pdf"
cb3_pdf      = "3cb.pdf"

# ── STEP 1: EXTRACT TEXT ──────────────────────────────────────────────────────
print("Step 1: Extracting text from PDFs...")
current_text  = extract_document(current_pdf)
previous_text = extract_document(previous_pdf)
cb3_text      = extract_document(cb3_pdf)

# ── STEP 2: BUILD FAISS RETRIEVERS ────────────────────────────────────────────
print("\nStep 2: Building FAISS vector stores...")
current_retriever  = build_vectorstore({"current":  current_text})
previous_retriever = build_vectorstore({"previous": previous_text})
cb3_retriever      = build_vectorstore({"3cb":      cb3_text})

# ── STEP 3: EXTRACT DATA ──────────────────────────────────────────────────────
print("\nStep 3: Running Extractor Agent...")
current_data  = extract_financial_data(current_retriever,  "current")
previous_data = extract_financial_data(previous_retriever, "previous")
cb3_data      = extract_financial_data(cb3_retriever,      "3cb")

# ── STEP 4: PRINT RESULTS ─────────────────────────────────────────────────────
print("\n========== CURRENT YEAR DATA ==========")
for key, val in current_data.items():
    print(f"  {key}: {val}")

print("\n========== PREVIOUS YEAR DATA ==========")
for key, val in previous_data.items():
    print(f"  {key}: {val}")

print("\n========== 3CB DATA ==========")
for key, val in cb3_data.items():
    print(f"  {key}: {val}")

# ── STEP 5: SANITY CHECK ──────────────────────────────────────────────────────
print("\n========== SANITY CHECK ==========")
print(f"  Client firm (current)  : {current_data.get('client_firm_name')}")
print(f"  CA firm (current)      : {current_data.get('ca_firm_name')}")
print(f"  Client firm (3CB)      : {cb3_data.get('client_firm_name')}")
print(f"  CA firm (3CB)          : {cb3_data.get('ca_firm_name')}")
print(f"  Net Profit (current)   : {current_data.get('net_profit')}")
print(f"  Net Profit (3CB)       : {cb3_data.get('net_profit')}")
print(f"  UDIN (3CB)             : {cb3_data.get('udin')}")
print(f"  Membership No (3CB)    : {cb3_data.get('membership_number')}")