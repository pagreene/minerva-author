import json
import re
import uuid
from typing import Optional


class SessionManager:
    """This defines the API for managing user sessions."""

    def get_session(self, session_key: str) -> Optional[dict]:
        raise NotImplementedError

    @staticmethod
    def make_session() -> (str, dict):
        session_id = str(uuid.uuid4())
        session_data = {}
        return session_id, session_data

    def update_session(self, session_key: str, session_data: dict) -> None:
        raise NotImplementedError


class S3SessionManager(SessionManager):
    def __init__(self, root_path):
        m = re.match("^s3://(.*?)/(.*?)$", root_path)
        if m is None:
            raise ValueError(f"Invalid s3 path: {root_path}")
        self.bucket, self.prefix = m.groups()

    @property
    def __s3(self):
        import boto3

        session = boto3.Session()
        return session.client("s3")

    def __make_s3_key(self, session_key: str) -> str:
        """Make the full s3 key, combining the session key and the prefix."""
        return f"{self.prefix}{session_key}.json"

    def get_session(self, session_key: str) -> Optional[dict]:
        from botocore.exceptions import ClientError

        try:
            resp = self.__s3.get_object(
                Bucket=self.bucket, Key=self.__make_s3_key(session_key)
            )
        except ClientError as err:
            if err.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise
        return json.loads(resp["Body"].read())

    def update_session(self, session_key: str, session_data: dict) -> None:
        self.__s3.put_object(
            Bucket=self.bucket,
            Key=self.__make_s3_key(session_key),
            Body=json.dumps(session_data),
        )


class LocalSessionManager(SessionManager):
    def __init__(self):
        self.__sessions = {}

    def get_session(self, session_key: str) -> Optional[dict]:
        return self.__sessions.get(session_key)

    def update_session(self, session_key: str, session_data: dict) -> None:
        self.__sessions[session_key] = session_data
