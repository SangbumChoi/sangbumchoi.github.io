import json
import unittest
import uuid
from types import SimpleNamespace

from feedback.hub_store import HubStore


class FakeApi:
    def __init__(self, private=True, fail=False):
        self.private = private
        self.fail = fail
        self.uploads = []

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private)

    def upload_file(self, **kwargs):
        if self.fail:
            raise OSError("offline")
        self.uploads.append(kwargs)


class HubStoreTest(unittest.TestCase):
    def setUp(self):
        self.payload = {"schema_version": 1, "consent": True, "consent_version": 2,
                        "session_id": str(uuid.uuid4()), "turn_id": str(uuid.uuid4()),
                        "question": "Hello?", "answer": "Hello.", "rating": None}

    def test_private_upload(self):
        api = FakeApi()
        self.assertEqual(HubStore("owner/private", api).save(self.payload), self.payload["turn_id"])
        self.assertEqual(json.load(api.uploads[0]["path_or_fileobj"])["consent_version"], 2)

    def test_public_dataset_rejected(self):
        api = FakeApi(private=False)
        with self.assertRaises(RuntimeError):
            HubStore("owner/public", api).save(self.payload)
        self.assertFalse(api.uploads)

    def test_consent_required_and_upload_errors_propagate(self):
        with self.assertRaises(ValueError):
            HubStore("owner/private", FakeApi()).save({**self.payload, "consent": False})
        with self.assertRaises(ValueError):
            HubStore("owner/private", FakeApi()).save({**self.payload, "consent_version": 1})
        with self.assertRaises(OSError):
            HubStore("owner/private", FakeApi(fail=True)).save(self.payload)


if __name__ == "__main__":
    unittest.main()
