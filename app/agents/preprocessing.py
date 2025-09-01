# reconciliation_agent.py

from pathlib import Path
import camelot
import pandas as pd
import json
from typing import Dict, Any

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from pydantic import BaseModel


# ---------------------------
# Tool Definitions
# ---------------------------

# --- ERP Loader ---
class ERPInput(BaseModel):
    file_path: str


def load_erp_data(file_path: str) -> str:
    """Load ERP data from an Excel file and return as JSON string."""
    df = pd.read_excel(file_path)
    return df.to_json(orient="records")


load_erp_data_tool = StructuredTool.from_function(
    load_erp_data, args_schema=ERPInput, name="load_erp_data"
)


# --- Bank PDF Loader ---
class BankInput(BaseModel):
    file_path: str


def load_bank_pdf(file_path: str) -> str:
    """Load bank statement data from a PDF and return as JSON string."""
    tables = camelot.read_pdf(file_path, pages="all")
    if not tables:
        return json.dumps([])
    df = tables[0].df
    df.columns = df.iloc[0]  # first row as headers
    df = df[1:]
    return df.to_json(orient="records")


load_bank_pdf_tool = StructuredTool.from_function(
    load_bank_pdf, args_schema=BankInput, name="load_bank_pdf"
)


# --- Normalizer ---
class NormalizeInput(BaseModel):
    json_data: str
    source: str


def normalize_dataframe(json_data: str, source: str) -> str:
    """Normalize ERP/Bank data into a standard format and return as JSON string."""
    df = pd.DataFrame(json.loads(json_data))

    if source == "ERP":
        mapping = {
            "Invoice No": "invoice_no",
            "Invoice Date": "invoice_date",
            "Amount": "amount"
        }
    elif source == "Bank":
        mapping = {
            "Txn ID": "invoice_no",
            "Date": "invoice_date",
            "Debit": "debit",
            "Credit": "credit",
            "Balance": "balance"
        }
    else:
        raise ValueError(f"Unknown source: {source}")

    df.rename(columns=mapping, inplace=True)

    # Keep only relevant columns
    if source == "ERP":
        df = df[["invoice_no", "invoice_date", "amount"]]
    else:  # Bank
        df = df[["invoice_no", "invoice_date", "debit", "credit", "balance"]]

    return df.to_json(orient="records")


normalize_dataframe_tool = StructuredTool.from_function(
    normalize_dataframe, args_schema=NormalizeInput, name="normalize_dataframe"
)


# --- Reconciliation ---
class ReconcileInput(BaseModel):
    erp_json: str
    bank_json: str


def reconcile_data(erp_json: str, bank_json: str) -> str:
    """Reconcile ERP invoices with Bank transactions by invoice_no and amount."""
    erp_df = pd.DataFrame(json.loads(erp_json))
    bank_df = pd.DataFrame(json.loads(bank_json))

    # Ensure numeric values
    erp_df["amount"] = pd.to_numeric(erp_df["amount"], errors="coerce").fillna(0)
    bank_df["debit"] = pd.to_numeric(bank_df.get("debit", 0), errors="coerce").fillna(0)
    bank_df["credit"] = pd.to_numeric(bank_df.get("credit", 0), errors="coerce").fillna(0)

    # Compute effective transaction amount
    bank_df["amount"] = bank_df["credit"] - bank_df["debit"]

    # Merge on invoice_no
    merged = pd.merge(
        erp_df,
        bank_df,
        on="invoice_no",
        how="outer",
        suffixes=("_erp", "_bank")
    )

    # Reconciliation status
    def check_match(row):
        if pd.isna(row["amount_erp"]) or pd.isna(row["amount_bank"]):
            return "Missing in one source"
        elif abs(row["amount_erp"] - row["amount_bank"]) < 1e-2:
            return "Matched"
        else:
            return "Mismatch"

    merged["status"] = merged.apply(check_match, axis=1)

    return merged.to_json(orient="records")


reconcile_data_tool = StructuredTool.from_function(
    reconcile_data, args_schema=ReconcileInput, name="reconcile_data"
)


# ---------------------------
# Agent Setup
# ---------------------------

llm = ChatOllama(model="phi4-mini:3.8b")

tools = [
    load_erp_data_tool,
    load_bank_pdf_tool,
    normalize_dataframe_tool,
    reconcile_data_tool,
]

agent = create_react_agent(llm, tools, debug=True)


# ---------------------------
# Run Example
# ---------------------------

if __name__ == "__main__":
    erp_path = "/home/sudamasharma/aai/app/data/erp_data.xlsx"
    bank_path = "/home/sudamasharma/aai/app/data/bank_data.pdf"

    query = f"""
    Load ERP data from {erp_path} and Bank data from {bank_path}.
    Normalize both datasets, then perform reconciliation of invoices and return the results.
    """

    result = agent.invoke({"messages": [("user", query)]})

    print("\n=== AGENT OUTPUT ===")
    print(result["messages"][-1].content)
