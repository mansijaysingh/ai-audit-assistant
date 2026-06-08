from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini"
)


BALANCE_SHEET_FIELDS = [
    "firm name",
    "PAN number",
    "assessment year",

    "sales",
    "turnover",

    "gross profit",
    "net profit",

    "opening stock",
    "closing stock",

    "purchases",

    "capital",

    "fixed assets",

    "sundry debtors",

    "sundry creditors",

    "cash",

    "bank balance"
]


CB3_FIELDS = [
    "firm name",
    "CA name",
    "PAN number",
    "assessment year",
    "turnover",
    "net profit",
    "depreciation",
    "UDIN",
    "membership number",
    "GST turnover"
]


prompt = PromptTemplate.from_template(
    """
You are a financial audit data extraction expert.

Document Type:
{document_type}

Extract structured financial data from the provided context.

If document_type is "current" or "previous", extract these fields:
firm_name
pan_number
assessment_year

sales
turnover

gross_profit
net_profit

opening_stock
closing_stock

purchases

capital_closing

fixed_assets_closing

debtors

creditors

cash

bank_balance

If document_type is "3cb", extract these fields:
firm_name
ca_name
pan_number
assessment_year
turnover
net_profit
depreciation
udin
membership_number
gst_turnover

Instructions:
- Return ONLY valid JSON.
- If a field is not found, return null.
- Do not explain anything.
- Do not return markdown.
- Use numbers without commas where possible.

Context:
{context}
"""
)


def get_context_for_extraction(retriever, document_type):

    if document_type == "3cb":
        fields = CB3_FIELDS
    else:
        fields = BALANCE_SHEET_FIELDS

    contexts = []

    for field in fields:

        docs = retriever.invoke(field)

        chunk = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        contexts.append(chunk)

    return "\n\n".join(contexts)


def extract_financial_data(context, document_type):

    parser = JsonOutputParser()

    chain = prompt | llm | parser

    result = chain.invoke(
        {
            "context": context,
            "document_type": document_type
        }
    )

    return result