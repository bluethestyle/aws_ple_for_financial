"""
Athena 쿼리 레이어 — S3의 Parquet 파일을 SQL로 조회합니다.
DuckDB(로컬)와 동일한 인터페이스를 제공하여 환경 전환이 쉽습니다.
"""

import logging
import time

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


class AthenaClient:
    """
    Example:
        client = AthenaClient(database="ml_platform", s3_output="s3://my-bucket/athena-results/")
        df = client.query("SELECT * FROM features WHERE ds = '2024-01-01' LIMIT 1000")
    """

    def __init__(self, database: str, s3_output: str, region: str = "ap-northeast-2"):
        self.database = database
        self.s3_output = s3_output
        self.client = boto3.client("athena", region_name=region)

    def query(self, sql: str, timeout: int = 300) -> pd.DataFrame:
        response = self.client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.s3_output},
        )
        execution_id = response["QueryExecutionId"]
        self._wait(execution_id, timeout)
        return self._fetch_results(execution_id)

    def _wait(self, execution_id: str, timeout: int) -> None:
        elapsed = 0
        while elapsed < timeout:
            status = self.client.get_query_execution(QueryExecutionId=execution_id)
            state = status["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                return
            elif state in ("FAILED", "CANCELLED"):
                reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
                raise RuntimeError(f"Athena query {state}: {reason}")
            time.sleep(2)
            elapsed += 2
        raise TimeoutError(f"Athena query timed out after {timeout}s")

    def _fetch_results(self, execution_id: str) -> pd.DataFrame:
        paginator = self.client.get_paginator("get_query_results")
        rows, header = [], None
        for page in paginator.paginate(QueryExecutionId=execution_id):
            result_rows = page["ResultSet"]["Rows"]
            if header is None:
                header = [col["VarCharValue"] for col in result_rows[0]["Data"]]
                result_rows = result_rows[1:]
            for row in result_rows:
                rows.append([col.get("VarCharValue", None) for col in row["Data"]])
        return pd.DataFrame(rows, columns=header)
