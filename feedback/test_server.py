import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

from feedback.server import ConversationStore, handler_for


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.temp.name) / "feedback.sqlite3")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.store, {"http://localhost:4000"}))
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/conversations"
        self.payload = {"schema_version": 1, "consent": True, "session_id": str(uuid.uuid4()),
                        "turn_id": str(uuid.uuid4()), "question": "What is SAM2?", "answer": "Segmentation model.",
                        "source": "entity-index", "runtime": "profile-index", "app_version": "45", "rating": None}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.temp.cleanup()

    def send(self, payload, origin="http://localhost:4000"):
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Origin": origin})
        return urllib.request.urlopen(request, timeout=3)

    def test_session_groups_turns_and_resends_update_rating(self):
        with self.send(self.payload) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:4000")
        self.payload["rating"] = "helpful"
        self.send(self.payload).close()
        second = {**self.payload, "turn_id": str(uuid.uuid4()), "question": "What did Daniel contribute?"}
        self.send(second).close()
        rows = list(self.store.export())
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["session_id"] for r in rows}, {self.payload["session_id"]})
        self.assertEqual(rows[0]["rating"], "helpful")
        self.assertNotIn("ip", rows[0])

    def test_requires_consent_and_trusted_origin(self):
        for payload, origin, expected in [({**self.payload, "consent": False}, "http://localhost:4000", 400),
                                          (self.payload, "https://example.org", 403),
                                          ({**self.payload, "question": "x" * 2001}, "http://localhost:4000", 400)]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.send(payload, origin)
            self.assertEqual(error.exception.code, expected)
        self.assertEqual(list(self.store.export()), [])

    def test_cannot_overwrite_exchange_or_read_transcripts(self):
        self.send(self.payload).close()
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.send({**self.payload, "answer": "Different answer"})
        self.assertEqual(error.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url, timeout=3)
        self.assertEqual(error.exception.code, 404)
        self.assertEqual(list(self.store.export())[0]["answer"], self.payload["answer"])


if __name__ == "__main__":
    unittest.main()
