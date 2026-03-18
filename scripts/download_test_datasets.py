#!/usr/bin/env python3
"""
Download and convert public test datasets to Parquet.

Run on SageMaker notebook or any environment with Kaggle CLI configured.

Prerequisites:
    pip install kaggle pandas pyarrow
    # Place Kaggle API key at ~/.kaggle/kaggle.json
    # Or set: export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx

Usage:
    python scripts/download_test_datasets.py --all
    python scripts/download_test_datasets.py --dataset financial_transactions
    python scripts/download_test_datasets.py --list

Output:
    data/raw/{dataset_name}/        ← original CSV
    data/converted/{dataset_name}.parquet  ← typed Parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("download_datasets")

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONVERTED_DIR = PROJECT_ROOT / "data" / "converted"


# ============================================================================
# Dataset registry
# ============================================================================

DATASETS: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # Financial datasets
    # -----------------------------------------------------------------------
    "financial_transactions": {
        "kaggle_id": "ealtman2019/credit-card-transactions",
        "description": "2000만 건 신용카드 거래 (2000명, 수십년 이력, MCC 포함)",
        "domain": "financial",
        "size": "~276MB ZIP",
        "columns": {
            "User": ("int64", "고객 ID"),
            "Card": ("int64", "카드 번호"),
            "Year": ("int16", "거래 연도"),
            "Month": ("int8", "거래 월"),
            "Day": ("int8", "거래 일"),
            "Time": ("string", "거래 시간"),
            "Amount": ("float64", "거래 금액 ($)"),
            "Use Chip": ("string", "결제 방식 (Chip/Swipe/Online)"),
            "Merchant Name": ("string", "가맹점명"),
            "Merchant City": ("string", "가맹점 도시"),
            "Merchant State": ("string", "가맹점 주"),
            "Zip": ("string", "우편번호"),
            "MCC": ("int16", "Merchant Category Code"),
            "Errors?": ("string", "오류 여부"),
            "Is Fraud?": ("string", "사기 여부"),
        },
        "strengths": "대규모 시계열 → TDA/HMM/Mamba 피처 생성 테스트 최적",
        "weaknesses": "인구통계 없음, churn 라벨 파생 필요",
    },
    "financial_customers": {
        "kaggle_id": "sakshigoyal7/credit-card-customers",
        "description": "1만 명 고객 프로필 + 이탈 라벨 (23개 속성)",
        "domain": "financial",
        "size": "~2MB",
        "columns": {
            "CLIENTNUM": ("int64", "고객 ID"),
            "Attrition_Flag": ("string", "이탈 여부 (Existing/Attrited)"),
            "Customer_Age": ("int8", "나이"),
            "Gender": ("string", "성별 (M/F)"),
            "Dependent_count": ("int8", "부양가족 수"),
            "Education_Level": ("string", "학력"),
            "Marital_Status": ("string", "결혼 상태"),
            "Income_Category": ("string", "소득 구간"),
            "Card_Category": ("string", "카드 등급 (Blue/Silver/Gold/Platinum)"),
            "Months_on_book": ("int16", "거래 기간 (월)"),
            "Total_Relationship_Count": ("int8", "보유 상품 수"),
            "Months_Inactive_12_mon": ("int8", "최근 12개월 비활동 월수"),
            "Contacts_Count_12_mon": ("int8", "최근 12개월 접촉 횟수"),
            "Credit_Limit": ("float64", "신용 한도"),
            "Total_Revolving_Bal": ("int32", "리볼빙 잔액"),
            "Avg_Open_To_Buy": ("float64", "평균 가용 한도"),
            "Total_Amt_Chng_Q4_Q1": ("float64", "Q4/Q1 금액 변화율"),
            "Total_Trans_Amt": ("int32", "총 거래 금액"),
            "Total_Trans_Ct": ("int16", "총 거래 건수"),
            "Total_Ct_Chng_Q4_Q1": ("float64", "Q4/Q1 건수 변화율"),
            "Avg_Utilization_Ratio": ("float64", "평균 한도 사용률"),
        },
        "strengths": "churn 라벨 + 인구통계 + 금융 지표 완비",
        "weaknesses": "개별 거래 시계열 없음 (집계값만)",
    },
    "financial_comprehensive": {
        "kaggle_id": "rajatsurana979/comprehensive-credit-card-transactions-dataset",
        "description": "거래 상세 + 고객 기본정보 (단일 데이터셋)",
        "domain": "financial",
        "size": "~1.3MB",
        "columns": {
            "Customer ID": ("int64", "고객 ID"),
            "Name": ("string", "이름 (PII — drop)"),
            "Surname": ("string", "성 (PII — drop)"),
            "Gender": ("string", "성별"),
            "Birthdate": ("string", "생년월일 → datetime"),
            "Transaction Amount": ("float64", "거래 금액"),
            "Date": ("string", "거래 일자 → datetime"),
            "Merchant Name": ("string", "가맹점명"),
            "Category": ("string", "거래 카테고리"),
        },
        "strengths": "거래 + 인구통계 한 파일, 소규모 빠른 테스트",
        "weaknesses": "규모 작음, churn 라벨 없음",
    },

    # -----------------------------------------------------------------------
    # E-commerce datasets
    # -----------------------------------------------------------------------
    "ecommerce_rees46": {
        "kaggle_id": "fridrichmrtn/e-commerce-churn-dataset-rees46",
        "description": "REES46 이커머스 거래 + churn + 행동 데이터",
        "domain": "ecommerce",
        "size": "대규모",
        "columns": {
            "user_id": ("int64", "고객 ID"),
            "event_type": ("string", "이벤트 유형 (view/cart/purchase)"),
            "product_id": ("int64", "상품 ID"),
            "category_id": ("int64", "카테고리 ID"),
            "category_code": ("string", "카테고리 코드"),
            "brand": ("string", "브랜드"),
            "price": ("float64", "가격"),
            "event_time": ("string", "이벤트 시각 → datetime"),
        },
        "strengths": "view/cart/purchase 퍼널 → ctr/cvr 직접 생성, 대규모",
        "weaknesses": "인구통계 없음",
    },
    "ecommerce_churn_2025": {
        "kaggle_id": "nabihazahid/e-commerce-customer-insights-and-churn-dataset",
        "description": "이커머스 고객 인사이트 + churn (2025)",
        "domain": "ecommerce",
        "size": "중규모",
        "columns": {
            "CustomerID": ("int64", "고객 ID"),
            "Age": ("int8", "나이"),
            "Gender": ("string", "성별"),
            "Income": ("float64", "소득"),
            "TotalSpend": ("float64", "총 지출"),
            "YearsAsCustomer": ("int8", "고객 연수"),
            "NumPurchases": ("int16", "구매 건수"),
            "AvgTransactionAmount": ("float64", "평균 거래 금액"),
            "NumSupportTickets": ("int8", "문의 건수"),
            "SatisfactionScore": ("float32", "만족도"),
            "LastPurchaseDaysAgo": ("int16", "최근 구매 경과일"),
            "Churn": ("int8", "이탈 여부 (0/1)"),
        },
        "strengths": "churn + 인구통계 + 행동 지표 완비, 깔끔한 구조",
        "weaknesses": "거래 시계열 없음 (집계값만)",
    },
    "ecommerce_churn_analysis": {
        "kaggle_id": "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction",
        "description": "이커머스 churn 분석용 (다양한 행동 피처)",
        "domain": "ecommerce",
        "size": "중규모",
        "columns": {
            "CustomerID": ("int64", "고객 ID"),
            "Churn": ("int8", "이탈 여부"),
            "Tenure": ("float32", "거래 기간 (월)"),
            "PreferredLoginDevice": ("string", "선호 로그인 기기"),
            "CityTier": ("int8", "도시 등급"),
            "WarehouseToHome": ("float32", "배송 거리"),
            "PreferredPaymentMode": ("string", "결제 방식"),
            "Gender": ("string", "성별"),
            "HourSpendOnApp": ("float32", "앱 사용 시간"),
            "NumberOfDeviceRegistered": ("int8", "등록 기기 수"),
            "PreferedOrderCat": ("string", "선호 카테고리"),
            "SatisfactionScore": ("int8", "만족도"),
            "MaritalStatus": ("string", "결혼 상태"),
            "NumberOfAddress": ("int8", "주소 수"),
            "Complain": ("int8", "불만 여부"),
            "OrderAmountHikeFromlastYear": ("float32", "전년 대비 주문 증가율"),
            "CouponUsed": ("float32", "쿠폰 사용 횟수"),
            "OrderCount": ("float32", "주문 건수"),
            "DaySinceLastOrder": ("float32", "마지막 주문 경과일"),
            "CashbackAmount": ("float32", "캐시백 금액"),
        },
        "strengths": "행동 피처 풍부 (기기, 앱시간, 쿠폰, 캐시백), churn 라벨",
        "weaknesses": "거래 시계열 없음",
    },
}


# ============================================================================
# Download + Convert
# ============================================================================

def download_dataset(name: str) -> Path:
    """Download a dataset from Kaggle."""
    info = DATASETS[name]
    kaggle_id = info["kaggle_id"]
    output_dir = RAW_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading '%s' from kaggle.com/datasets/%s", name, kaggle_id)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", kaggle_id,
        "-p", str(output_dir),
        "--unzip",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Downloaded to %s", output_dir)
    except subprocess.CalledProcessError as e:
        logger.error("Download failed: %s", e.stderr)
        raise
    except FileNotFoundError:
        logger.error(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then configure: export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx"
        )
        raise

    return output_dir


def convert_to_parquet(name: str) -> Path:
    """Convert downloaded CSV to typed Parquet."""
    import pandas as pd

    info = DATASETS[name]
    raw_dir = RAW_DIR / name
    output_path = CONVERTED_DIR / f"{name}.parquet"
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

    # Find CSV files
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")

    logger.info("Converting %d CSV file(s) for '%s'", len(csv_files), name)

    # Read (handle multiple CSVs by concatenating)
    dfs = []
    for csv_path in csv_files:
        logger.info("  Reading %s", csv_path.name)
        df = pd.read_csv(csv_path, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    logger.info("  Raw shape: %d rows × %d columns", len(df), len(df.columns))

    # Apply column types from registry
    col_specs = info.get("columns", {})
    for col_name, (dtype, description) in col_specs.items():
        if col_name not in df.columns:
            # Try case-insensitive match
            matches = [c for c in df.columns if c.lower() == col_name.lower()]
            if matches:
                df.rename(columns={matches[0]: col_name}, inplace=True)
            else:
                continue

        try:
            if dtype == "string":
                df[col_name] = df[col_name].astype(str).replace("nan", "")
            elif dtype.startswith("int"):
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0).astype(dtype)
            elif dtype.startswith("float"):
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(dtype)
            elif dtype == "datetime":
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
        except Exception as e:
            logger.warning("  Type cast failed for %s → %s: %s", col_name, dtype, e)

    # Drop PII columns if marked
    pii_cols = [c for c in ["Name", "Surname", "name", "surname"] if c in df.columns]
    if pii_cols:
        df.drop(columns=pii_cols, inplace=True)
        logger.info("  Dropped PII columns: %s", pii_cols)

    # Convert date strings to datetime where applicable
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

    # Save as Parquet
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("  Saved: %s (%.1fMB, %d rows × %d cols)", output_path, size_mb, len(df), len(df.columns))

    # Print schema summary
    logger.info("  Schema:")
    for col in df.columns:
        logger.info("    %-35s %s", col, df[col].dtype)

    return output_path


def upload_to_s3(parquet_path: Path, s3_base: str = "s3://aiops-ple-financial/data/test") -> str:
    """Upload Parquet to S3 (optional)."""
    import boto3

    s3_key = f"{s3_base.rstrip('/')}/{parquet_path.name}"
    bucket = s3_key.replace("s3://", "").split("/")[0]
    key = "/".join(s3_key.replace("s3://", "").split("/")[1:])

    s3 = boto3.client("s3")
    s3.upload_file(str(parquet_path), bucket, key)
    logger.info("Uploaded to %s", s3_key)
    return s3_key


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and convert test datasets")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--convert-only", action="store_true", help="Skip download, convert existing CSVs")
    parser.add_argument("--upload-s3", action="store_true", help="Upload converted Parquet to S3")
    parser.add_argument("--s3-base", type=str, default="s3://aiops-ple-financial/data/test")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable test datasets:\n")
        for name, info in DATASETS.items():
            print(f"  {name}")
            print(f"    Domain:  {info['domain']}")
            print(f"    Kaggle:  kaggle.com/datasets/{info['kaggle_id']}")
            print(f"    Size:    {info['size']}")
            print(f"    Desc:    {info['description']}")
            print(f"    Good at: {info['strengths']}")
            print(f"    Limits:  {info['weaknesses']}")
            print()
        return

    targets = list(DATASETS.keys()) if args.all else [args.dataset] if args.dataset else []

    if not targets:
        parser.print_help()
        return

    results = []
    for name in targets:
        if name not in DATASETS:
            logger.error("Unknown dataset: %s", name)
            continue

        try:
            if not args.convert_only:
                download_dataset(name)
            parquet_path = convert_to_parquet(name)

            if args.upload_s3:
                s3_uri = upload_to_s3(parquet_path, args.s3_base)
                results.append({"name": name, "parquet": str(parquet_path), "s3": s3_uri})
            else:
                results.append({"name": name, "parquet": str(parquet_path)})

        except Exception as e:
            logger.error("Failed for '%s': %s", name, e)
            results.append({"name": name, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("Download & Convert Summary")
    print("=" * 60)
    for r in results:
        status = "✅" if "error" not in r else "❌"
        print(f"  {status} {r['name']}")
        if "parquet" in r:
            print(f"     → {r['parquet']}")
        if "s3" in r:
            print(f"     → {r['s3']}")
        if "error" in r:
            print(f"     → ERROR: {r['error']}")


if __name__ == "__main__":
    main()
