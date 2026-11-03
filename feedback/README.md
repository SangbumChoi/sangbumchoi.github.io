# Conversation feedback

The portfolio collector is separate from the private knowledge API. It accepts
explicitly shared, completed question/answer turns through a write-only API.
No browser read/export endpoint exists. The client keeps unshared turns only
in page memory; reloading or starting a new conversation resets consent and
creates a fresh random session ID.

## Local use

```sh
python3 -m feedback.server
RUBYOPT=-r./scripts/jekyll_compat.rb bundle exec jekyll build --config _config.yml,_config.dev.yml
python3 -m http.server 4000 --bind 127.0.0.1 --directory _site
```

The development site points to `http://127.0.0.1:8767/api/conversations`.
Choose **Share conversation**, read the notice, and enable the checkbox.
The current session's completed turns are submitted; future turns and ratings
continue to save while sharing remains enabled. Disabling sharing aborts the
client queue but does not delete already submitted records. A request already
received by the server may finish saving before cancellation.

The default database is outside the public site:
`~/.local/share/daniel-feedback/conversations.sqlite3`.
Set `CONVERSATION_DB` or pass `--db` to use a different private location.

## Private export

```sh
python3 -m feedback.server export > ~/Downloads/daniel-conversations.jsonl
```

Each line contains schema version, session/turn IDs, question, answer, source,
runtime, app version, rating, consent version, and server timestamps. Uploads
are idempotent by turn ID; ratings update the same record. Review and redact
exports before using them as evaluation/training data. Sharing does not imply
the transcript is anonymous, and raw transcripts are not automatically uploaded
to Hugging Face or committed to Git.

## Hugging Face Space (experimental)

The Dockerfile in `feedback/space/` uses `CONVERSATION_HF_REPO` to select a
private dataset and `HF_TOKEN` as a server-side Space secret. Each explicitly
shared turn is uploaded before the server confirms it, and a public dataset or
an upload failure is rejected. The browser sends consent version 2, which
discloses Hugging Face storage; older consent versions are rejected. The
dataset is separate from the voice dataset. Revisions may remain in Hub history
even after a file is changed or removed; stopping sharing does not erase them.
Each upload creates a Hub commit, so rate limits and abuse controls matter.

This adapter is not deployed: protect the public ingress with independent rate
limits and request-size limits, configure retention and deletion, and verify
private storage and live integration before setting the production endpoint.
The in-process lock does not coordinate multiple Space instances. Use a single
instance or a transactional service. Do not treat CORS as authentication.

## Production connection

GitHub Pages cannot run this Python collector. Run it on the project server,
behind an HTTPS reverse proxy, with request size/time limits, rate limiting,
private database backups, and a retention policy appropriate to the site.
The origin allowlist is a browser control, not authentication against non-browser
clients; public ingestion needs server-side abuse controls at the proxy.

```sh
CONVERSATION_ORIGINS=https://sangbumchoi.github.io python3 -m feedback.server
```

Forward only `/api/conversations` to loopback port 8767, and set
`conversation_endpoint` in `_config.yml` to the resulting HTTPS URL. Do not put
an admin secret or database under `_site`. Until a deployed collector is
configured, production sharing is disabled and the site says it is not connected.

Validation: `python3 -m unittest feedback.test_server` and
`node --test scripts/test_conversation_feedback.mjs`.
