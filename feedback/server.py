"""Collect explicitly shared conversation turns; export privately as JSONL."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_DB = Path.home() / ".local/share/daniel-feedback/conversations.sqlite3"
MAX_BODY = 96 * 1024


def validate(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("consent") is not True:
        raise ValueError("Explicit consent and schema_version 1 are required")
    result = {}
    for key in ("session_id", "turn_id"):
        result[key] = str(uuid.UUID(str(data.get(key, ""))))
    for key, limit in (("question", 2000), ("answer", 24000), ("source", 120), ("runtime", 40), ("app_version", 40)):
        value = data.get(key, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"Invalid {key}")
        if key in ("question", "answer") and not value.strip():
            raise ValueError(f"Non-empty {key} is required")
        result[key] = value
    rating = data.get("rating")
    if rating not in (None, "helpful", "not_helpful"):
        raise ValueError("Invalid rating")
    result["rating"] = rating
    return result


class ConversationStore:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                question TEXT NOT NULL, answer TEXT NOT NULL, source TEXT NOT NULL,
                runtime TEXT NOT NULL, app_version TEXT NOT NULL, rating TEXT,
                consent_version INTEGER NOT NULL, created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, created_at)")
        self.path.chmod(0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def save(self, payload):
        item = validate(payload)
        now = int(time.time())
        with self.connect() as db:
            # A resend may update a rating, but cannot replace another exchange.
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT * FROM turns WHERE turn_id=?", (item["turn_id"],)).fetchone()
            if previous:
                if any(previous[key] != item[key] for key in ("session_id", "question", "answer")):
                    raise ValueError("Turn identity conflicts with an existing exchange")
                db.execute("UPDATE turns SET rating=?, updated_at=? WHERE turn_id=?",
                           (item["rating"], now, item["turn_id"]))
            else:
                db.execute("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    item["turn_id"], item["session_id"], item["question"], item["answer"],
                    item["source"], item["runtime"], item["app_version"], item["rating"], 1, now, now))
        return item["turn_id"]

    def export(self):
        with self.connect() as db:
            for row in db.execute("SELECT * FROM turns ORDER BY session_id, created_at, rowid"):
                yield {"schema_version": 1, **dict(row)}


def handler_for(store, origins):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # The database and request log do not collect IPs, headers, or URLs.
            pass

        def reply(self, code, payload=None):
            body = json.dumps(payload or {}).encode()
            self.send_response(code)
            origin = self.headers.get("Origin")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            if self.path != "/api/conversations" or self.headers.get("Origin") not in origins:
                return self.reply(403, {"error": "Origin not allowed"})
            self.reply(200)

        def do_GET(self):
            # No browser API can read or export saved transcripts.
            self.reply(200, {"status": "ok"}) if self.path == "/health" else self.reply(404)

        def do_POST(self):
            if self.path != "/api/conversations":
                return self.reply(404)
            if self.headers.get("Origin") not in origins:
                return self.reply(403, {"error": "Origin not allowed"})
            if self.headers.get_content_type() != "application/json":
                return self.reply(415, {"error": "Expected application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > MAX_BODY:
                    return self.reply(413, {"error": "Invalid body size"})
                turn_id = store.save(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError, json.JSONDecodeError):
                return self.reply(400, {"error": "Invalid conversation payload"})
            self.reply(200, {"saved": True, "turn_id": turn_id})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "export"), nargs="?", default="serve")
    parser.add_argument("--db", default=os.environ.get("CONVERSATION_DB", str(DEFAULT_DB)))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    store = ConversationStore(args.db)
    if args.command == "export":
        for item in store.export():
            print(json.dumps(item, ensure_ascii=False))
        return
    origins = set(os.environ.get("CONVERSATION_ORIGINS", "http://localhost:4000,http://127.0.0.1:4000").split(","))
    server = ThreadingHTTPServer((args.host, args.port), handler_for(store, origins))
    print(f"Conversation collector: http://{args.host}:{args.port} (write-only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
