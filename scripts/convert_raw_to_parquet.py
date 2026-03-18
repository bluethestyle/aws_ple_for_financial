#!/usr/bin/env python3
"""
Convert all raw test datasets to typed Parquet.

Reads CSV/Excel from data/raw/*/ and writes typed Parquet to data/converted/.
Each dataset gets explicit column types — no object/mixed types allowed.

Usage:
    python scripts/convert_raw_to_parquet.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("convert")

RAW = Path("data/raw")
OUT = Path("data/converted")
OUT.mkdir(parents=True, exist_ok=True)


def _money_to_float(s):
    """'$29,278' → 29278.0"""
    if isinstance(s, str):
        return float(s.replace("$", "").replace(",", ""))
    return float(s)


def convert_ealtman_users():
    """#1a: ealtman2019 — sd254_users.csv (2000명 고객 프로필)"""
    name = "01_financial_users"
    p = RAW / "ealtman2019credit-card-transactions" / "sd254_users.csv"
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()

    # PII drop
    df.drop(columns=["Address", "Apartment"], inplace=True, errors="ignore")

    # Types
    df["Person"] = df["Person"].astype("string")
    df["Current Age"] = df["Current Age"].astype("int8")
    df["Retirement Age"] = df["Retirement Age"].astype("int8")
    df["Birth Year"] = df["Birth Year"].astype("int16")
    df["Birth Month"] = df["Birth Month"].astype("int8")
    df["Gender"] = df["Gender"].astype("category")
    df["City"] = df["City"].astype("string")
    df["State"] = df["State"].astype("category")
    df["Zipcode"] = df["Zipcode"].astype("string")
    df["Latitude"] = df["Latitude"].astype("float32")
    df["Longitude"] = df["Longitude"].astype("float32")
    for col in ["Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt"]:
        df[col] = df[col].apply(_money_to_float).astype("float32")
    df["FICO Score"] = df["FICO Score"].astype("int16")
    df["Num Credit Cards"] = df["Num Credit Cards"].astype("int8")

    _save(df, name)


def convert_ealtman_cards():
    """#1b: ealtman2019 — sd254_cards.csv (카드 정보)"""
    name = "01_financial_cards"
    p = RAW / "ealtman2019credit-card-transactions" / "sd254_cards.csv"
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()

    # PII drop
    df.drop(columns=["Card Number", "CVV"], inplace=True, errors="ignore")

    df["User"] = df["User"].astype("int32")
    df["CARD INDEX"] = df["CARD INDEX"].astype("int8")
    df["Card Brand"] = df["Card Brand"].astype("category")
    df["Card Type"] = df["Card Type"].astype("category")
    df["Expires"] = df["Expires"].astype("string")
    df["Has Chip"] = (df["Has Chip"] == "YES").astype("int8")
    df["Cards Issued"] = df["Cards Issued"].astype("int8")
    df["Credit Limit"] = df["Credit Limit"].apply(_money_to_float).astype("float32")
    df["Acct Open Date"] = pd.to_datetime(df["Acct Open Date"], format="%m/%Y", errors="coerce")
    df["Year PIN last Changed"] = pd.to_numeric(df["Year PIN last Changed"], errors="coerce").astype("Int16")
    df["Card on Dark Web"] = (df["Card on Dark Web"] == "Yes").astype("int8")

    _save(df, name)


def convert_ealtman_transactions():
    """#1c: ealtman2019 — credit_card_transactions-ibm_v2.csv (2400만 거래)"""
    name = "01_financial_transactions"
    p = RAW / "ealtman2019credit-card-transactions" / "credit_card_transactions-ibm_v2.csv"
    logger.info("Reading %s (large file, may take a minute)...", p.name)
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.str.strip()

    df["User"] = df["User"].astype("int32")
    df["Card"] = df["Card"].astype("int8")
    df["Year"] = df["Year"].astype("int16")
    df["Month"] = df["Month"].astype("int8")
    df["Day"] = df["Day"].astype("int8")
    df["Time"] = df["Time"].astype("string")
    df["Amount"] = df["Amount"].apply(_money_to_float).astype("float32")
    df["Use Chip"] = df["Use Chip"].astype("category")
    df["Merchant Name"] = df["Merchant Name"].astype("string")
    df["Merchant City"] = df["Merchant City"].astype("string")
    df["Merchant State"] = df["Merchant State"].astype("category")
    df["Zip"] = df["Zip"].astype("string")
    df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce").astype("Int16")
    df["Errors?"] = df["Errors?"].fillna("").astype("string")
    df["Is Fraud?"] = (df["Is Fraud?"] == "Yes").astype("int8")

    _save(df, name)


def convert_sakshi():
    """#2: sakshigoyal7 — BankChurners.csv (1만 명 고객, churn)"""
    name = "02_financial_bank_churners"
    p = RAW / "sakshigoyal7credit-card-customers" / "BankChurners.csv"
    df = pd.read_csv(p)

    # Drop Naive Bayes leakage columns
    drop_cols = [c for c in df.columns if "Naive_Bayes" in c]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    df["CLIENTNUM"] = df["CLIENTNUM"].astype("int64")
    df["Attrition_Flag"] = (df["Attrition_Flag"] == "Attrited Customer").astype("int8")
    df.rename(columns={"Attrition_Flag": "Churn"}, inplace=True)
    df["Customer_Age"] = df["Customer_Age"].astype("int8")
    df["Gender"] = df["Gender"].astype("category")
    df["Dependent_count"] = df["Dependent_count"].astype("int8")
    df["Education_Level"] = df["Education_Level"].astype("category")
    df["Marital_Status"] = df["Marital_Status"].astype("category")
    df["Income_Category"] = df["Income_Category"].astype("category")
    df["Card_Category"] = df["Card_Category"].astype("category")
    df["Months_on_book"] = df["Months_on_book"].astype("int16")
    df["Total_Relationship_Count"] = df["Total_Relationship_Count"].astype("int8")
    df["Months_Inactive_12_mon"] = df["Months_Inactive_12_mon"].astype("int8")
    df["Contacts_Count_12_mon"] = df["Contacts_Count_12_mon"].astype("int8")
    df["Credit_Limit"] = df["Credit_Limit"].astype("float32")
    df["Total_Revolving_Bal"] = df["Total_Revolving_Bal"].astype("int32")
    df["Avg_Open_To_Buy"] = df["Avg_Open_To_Buy"].astype("float32")
    df["Total_Amt_Chng_Q4_Q1"] = df["Total_Amt_Chng_Q4_Q1"].astype("float32")
    df["Total_Trans_Amt"] = df["Total_Trans_Amt"].astype("int32")
    df["Total_Trans_Ct"] = df["Total_Trans_Ct"].astype("int16")
    df["Total_Ct_Chng_Q4_Q1"] = df["Total_Ct_Chng_Q4_Q1"].astype("float32")
    df["Avg_Utilization_Ratio"] = df["Avg_Utilization_Ratio"].astype("float32")

    _save(df, name)


def convert_rajat():
    """#3: rajatsurana979 — credit_card_transaction_flow.csv"""
    name = "03_financial_comprehensive"
    p = RAW / "rajatsurana979comprehensive-credit-card-transactions-dataset" / "credit_card_transaction_flow.csv"
    df = pd.read_csv(p)

    # PII drop
    df.drop(columns=["Name", "Surname"], inplace=True, errors="ignore")

    df["Customer ID"] = df["Customer ID"].astype("int64")
    df["Gender"] = df["Gender"].astype("category")
    df["Birthdate"] = pd.to_datetime(df["Birthdate"], format="%d-%m-%Y", errors="coerce")
    df["Transaction Amount"] = df["Transaction Amount"].astype("float32")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df["Merchant Name"] = df["Merchant Name"].astype("string")
    df["Category"] = df["Category"].astype("category")

    _save(df, name)


def convert_rees46():
    """#4: REES46 — rees46_customer_model.csv (11만 명, 230D+)"""
    name = "04_ecommerce_rees46"
    p = RAW / "fridrichmrtne-commerce-churn-dataset-rees46" / "rees46_customer_model.csv"
    df = pd.read_csv(p)

    df["user_id"] = df["user_id"].astype("int64")

    # All numeric feature columns → float32 (except targets)
    target_cols = ["target_event", "target_revenue", "target_customer_value",
                   "target_customer_value_lag1", "target_actual_profit"]
    feature_cols = [c for c in df.columns
                    if c not in ["row_id", "user_id", "time_step"] + target_cols]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Targets
    df["target_event"] = pd.to_numeric(df["target_event"], errors="coerce").astype("int8")
    df["target_revenue"] = pd.to_numeric(df["target_revenue"], errors="coerce").astype("float32")
    df["target_customer_value"] = pd.to_numeric(df["target_customer_value"], errors="coerce").astype("float32")
    df["time_step"] = pd.to_numeric(df["time_step"], errors="coerce").astype("int8")

    df.drop(columns=["row_id"], inplace=True, errors="ignore")

    _save(df, name)


def convert_nabihazahid():
    """#5: nabihazahid — E Commerce Customer Insights and Churn Dataset.csv"""
    name = "05_ecommerce_churn_2025"
    p = RAW / "nabihazahide-commerce-customer-insights-and-churn-dataset" / "E Commerce Customer Insights and Churn Dataset.csv"
    df = pd.read_csv(p)

    df["order_id"] = df["order_id"].astype("string")
    df["customer_id"] = df["customer_id"].astype("string")
    df["age"] = df["age"].astype("int8")
    df["product_id"] = df["product_id"].astype("string")
    df["country"] = df["country"].astype("category")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")
    df["cancellations_count"] = df["cancellations_count"].astype("int8")
    df["subscription_status"] = df["subscription_status"].astype("category")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["unit_price"] = df["unit_price"].astype("float32")
    df["quantity"] = df["quantity"].astype("int16")
    df["purchase_frequency"] = pd.to_numeric(df["purchase_frequency"], errors="coerce").astype("int16")
    df["preferred_category"] = df["preferred_category"].astype("category")
    df["product_name"] = df["product_name"].astype("string")
    df["category"] = df["category"].astype("category")
    df["gender"] = df["gender"].astype("category")

    _save(df, name)


def convert_ankit():
    """#6: ankitverma2010 — E Commerce Dataset.xlsx"""
    name = "06_ecommerce_churn_analysis"
    p = RAW / "ankitverma2010ecommerce-customer-churn-analysis-and-prediction" / "E Commerce Dataset.xlsx"
    logger.info("Reading Excel: %s", p.name)
    df = pd.read_excel(p, sheet_name=0)

    # Standardize column names
    df.columns = df.columns.str.strip()

    int8_cols = ["Churn", "CityTier", "NumberOfDeviceRegistered", "SatisfactionScore",
                 "NumberOfAddress", "Complain"]
    for c in int8_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

    float32_cols = ["Tenure", "WarehouseToHome", "HourSpendOnApp",
                    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
                    "DaySinceLastOrder", "CashbackAmount"]
    for c in float32_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    cat_cols = ["PreferredLoginDevice", "PreferredPaymentMode", "Gender",
                "PreferedOrderCat", "MaritalStatus"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype("int64")

    _save(df, name)


def convert_ibm_telco():
    """#7: IBM Telco Customer Churn"""
    name = "07_telco_churn"
    p = RAW / "ibm-telco-churn" / "Telco-Customer-Churn.csv"
    df = pd.read_csv(p)

    df["customerID"] = df["customerID"].astype("string")
    df["gender"] = df["gender"].astype("category")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")
    df["Partner"] = (df["Partner"] == "Yes").astype("int8")
    df["Dependents"] = (df["Dependents"] == "Yes").astype("int8")
    df["tenure"] = df["tenure"].astype("int16")
    df["PhoneService"] = (df["PhoneService"] == "Yes").astype("int8")
    df["MonthlyCharges"] = df["MonthlyCharges"].astype("float32")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").astype("float32")
    df["Churn"] = (df["Churn"] == "Yes").astype("int8")

    cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaymentMethod"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    df["PaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype("int8")

    _save(df, name)


def convert_bank_churn():
    """#8: Bank Churn Modelling"""
    name = "08_bank_churn"
    p = RAW / "bank-churn-modelling" / "Churn_Modelling.csv"
    df = pd.read_csv(p)

    # Drop leakage/useless
    df.drop(columns=["RowNumber", "Surname"], inplace=True, errors="ignore")

    df["CustomerId"] = df["CustomerId"].astype("int64")
    df["CreditScore"] = df["CreditScore"].astype("int16")
    df["Geography"] = df["Geography"].astype("category")
    df["Gender"] = df["Gender"].astype("category")
    df["Age"] = df["Age"].astype("int8")
    df["Tenure"] = df["Tenure"].astype("int8")
    df["Balance"] = df["Balance"].astype("float32")
    df["NumOfProducts"] = df["NumOfProducts"].astype("int8")
    df["HasCrCard"] = df["HasCrCard"].astype("int8")
    df["IsActiveMember"] = df["IsActiveMember"].astype("int8")
    df["EstimatedSalary"] = df["EstimatedSalary"].astype("float32")
    df["Exited"] = df["Exited"].astype("int8")
    df.rename(columns={"Exited": "Churn"}, inplace=True)

    _save(df, name)


def convert_ecommerce_3files():
    """#9: eCommerce Transactions 3-file dataset"""
    name = "09_ecommerce_transactions"
    base = RAW / "ecommerce-transactions-3files"

    customers = pd.read_csv(base / "Customers.csv", encoding="utf-8-sig")
    products = pd.read_csv(base / "Products.csv", encoding="utf-8-sig")
    transactions = pd.read_csv(base / "Transactions.csv", encoding="utf-8-sig")

    # Clean column names
    for d in [customers, products, transactions]:
        d.columns = d.columns.str.strip()

    # Type customers
    customers["CustomerID"] = customers["CustomerID"].astype("string")
    customers["CustomerName"] = customers["CustomerName"].astype("string")
    customers["Region"] = customers["Region"].astype("category")
    customers["SignupDate"] = pd.to_datetime(customers["SignupDate"], errors="coerce")

    # Type products
    products["ProductID"] = products["ProductID"].astype("string")
    products["ProductName"] = products["ProductName"].astype("string")
    products["Category"] = products["Category"].astype("category")
    products["Price"] = products["Price"].astype("float32")

    # Type transactions
    transactions["TransactionID"] = transactions["TransactionID"].astype("string")
    transactions["CustomerID"] = transactions["CustomerID"].astype("string")
    transactions["ProductID"] = transactions["ProductID"].astype("string")
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"], errors="coerce")
    transactions["Quantity"] = transactions["Quantity"].astype("int16")
    transactions["TotalValue"] = transactions["TotalValue"].astype("float32")
    transactions["Price"] = transactions["Price"].astype("float32")

    # Join into single table
    merged = transactions.merge(customers, on="CustomerID", how="left")
    merged = merged.merge(products, on="ProductID", how="left", suffixes=("", "_product"))

    # Drop PII
    merged.drop(columns=["CustomerName"], inplace=True, errors="ignore")

    _save(merged, name)


def convert_uci_default():
    """#10: UCI Default of Credit Card Clients (Excel)"""
    name = "10_financial_default"
    p = RAW / "uci-default-credit" / "default of credit card clients.xls"
    logger.info("Reading Excel: %s", p.name)
    df = pd.read_excel(p, header=1)  # Row 0 is ID, row 1 is actual header

    df.rename(columns={"default payment next month": "Default", "ID": "CustomerID"}, inplace=True)

    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["Default"] = df["Default"].astype("int8")

    # SEX: 1=male, 2=female → category
    df["SEX"] = df["SEX"].map({1: "Male", 2: "Female"}).astype("category")
    # EDUCATION: 1=grad, 2=university, 3=high school, 4=others
    df["EDUCATION"] = df["EDUCATION"].astype("int8")
    # MARRIAGE: 1=married, 2=single, 3=others
    df["MARRIAGE"] = df["MARRIAGE"].astype("int8")
    df["AGE"] = df["AGE"].astype("int8")
    df["LIMIT_BAL"] = df["LIMIT_BAL"].astype("float32")

    # PAY_0 ~ PAY_6: repayment status (-1=on time, 1=1mo delay, ...)
    for c in [f"PAY_{i}" for i in range(7)]:
        if c in df.columns:
            df[c] = df[c].astype("int8")

    # BILL_AMT1 ~ BILL_AMT6: bill statement amount
    for c in [f"BILL_AMT{i}" for i in range(1, 7)]:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    # PAY_AMT1 ~ PAY_AMT6: previous payment amount
    for c in [f"PAY_AMT{i}" for i in range(1, 7)]:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    _save(df, name)


# ============================================================================
# Utility
# ============================================================================

def _save(df: pd.DataFrame, name: str):
    path = OUT / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(
        "✅ %s: %d rows × %d cols (%.1fMB) → %s",
        name, len(df), len(df.columns), size_mb, path,
    )
    # Print schema
    for col in df.columns:
        logger.info("    %-40s %s", col, df[col].dtype)
    logger.info("")


def main():
    logger.info("=" * 60)
    logger.info("Converting all raw datasets to typed Parquet")
    logger.info("=" * 60)

    converters = [
        ("01a users", convert_ealtman_users),
        ("01b cards", convert_ealtman_cards),
        ("01c transactions (large)", convert_ealtman_transactions),
        ("02 bank churners", convert_sakshi),
        ("03 comprehensive", convert_rajat),
        ("04 REES46", convert_rees46),
        ("05 ecommerce churn 2025", convert_nabihazahid),
        ("06 ecommerce churn analysis", convert_ankit),
        ("07 telco churn", convert_ibm_telco),
        ("08 bank churn", convert_bank_churn),
        ("09 ecommerce 3files", convert_ecommerce_3files),
        ("10 UCI default", convert_uci_default),
    ]

    results = []
    for label, func in converters:
        try:
            logger.info("--- %s ---", label)
            func()
            results.append((label, "✅"))
        except Exception as e:
            logger.error("❌ %s failed: %s", label, e)
            results.append((label, f"❌ {e}"))

    logger.info("=" * 60)
    logger.info("Summary:")
    for label, status in results:
        logger.info("  %s %s", status, label)


if __name__ == "__main__":
    main()
