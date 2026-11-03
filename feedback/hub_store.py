"""Write opted-in turns to a private Hugging Face dataset, never to local disk."""
import json
import os
import threading
import time
from io import BytesIO

from huggingface_hub import HfApi
from feedback.server import validate


class HubStore:
    def __init__(self, repo_id, api=None):
        self.repo_id = repo_id
        self.api = api or HfApi(token=os.environ["HF_TOKEN"])
        self.lock = threading.Lock()

    def save(self, payload):
        if payload.get("consent_version") != 2:
            raise ValueError("Current sharing consent is required")
        item = validate(payload)
        with self.lock:
            info = self.api.repo_info(self.repo_id, repo_type="dataset")
            if not info.private:
                raise RuntimeError("Refusing to write to a public dataset")
            item.update(schema_version=1, consent_version=2, updated_at=int(time.time()))
            content = json.dumps(item, ensure_ascii=False).encode("utf-8")
            self.api.upload_file(
                path_or_fileobj=BytesIO(content),
                path_in_repo=f"turns/{item['session_id']}/{item['turn_id']}.json",
                repo_id=self.repo_id, repo_type="dataset",
                commit_message="Save opted-in conversation turn",
            )
        return item["turn_id"]
