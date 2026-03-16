"""
S3DataStore — S3 읽기/쓰기 단일 인터페이스.

로컬 경로와 s3:// URI를 투명하게 처리합니다.
"""

import logging
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


class S3DataStore:
    """
    Example:
        store = S3DataStore(bucket="my-bucket", prefix="project/")
        store.upload("local/data.parquet", "data/train.parquet")
        store.download("data/train.parquet", "local/data.parquet")
        df = store.read_parquet("data/train.parquet")
    """

    def __init__(self, bucket: str, prefix: str = "", region: str = "ap-northeast-2"):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.s3 = boto3.client("s3", region_name=region)

    def _key(self, path: str) -> str:
        return f"{self.prefix}/{path}".lstrip("/") if self.prefix else path

    def upload(self, local_path: str, s3_path: str) -> str:
        key = self._key(s3_path)
        logger.info(f"Uploading {local_path} → s3://{self.bucket}/{key}")
        self.s3.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download(self, s3_path: str, local_path: str) -> None:
        key = self._key(s3_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading s3://{self.bucket}/{key} → {local_path}")
        self.s3.download_file(self.bucket, key, local_path)

    def upload_dir(self, local_dir: str, s3_prefix: str) -> list[str]:
        uploaded = []
        for p in Path(local_dir).rglob("*"):
            if p.is_file():
                rel = p.relative_to(local_dir)
                uri = self.upload(str(p), f"{s3_prefix}/{rel}")
                uploaded.append(uri)
        return uploaded

    def download_dir(self, s3_prefix: str, local_dir: str) -> None:
        paginator = self.s3.get_paginator("list_objects_v2")
        full_prefix = self._key(s3_prefix)
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(full_prefix):].lstrip("/")
                self.download(key, str(Path(local_dir) / rel))

    def read_parquet(self, s3_path: str):
        import pandas as pd
        import io
        key = self._key(s3_path)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def write_parquet(self, df, s3_path: str) -> str:
        import io
        key = self._key(s3_path)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.read())
        return f"s3://{self.bucket}/{key}"

    def exists(self, s3_path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(s3_path))
            return True
        except self.s3.exceptions.ClientError:
            return False

    @property
    def base_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}".rstrip("/")
