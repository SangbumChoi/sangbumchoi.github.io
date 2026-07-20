#!/usr/bin/env python3
"""Audit Daniel profile sources and produce a human-review evidence report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


USER_AGENT = "DanielOS-EvidenceAudit/1.0 (+https://sangbumchoi.github.io/)"
NAME_MARKERS = ("sangbum", "sangbum choi", "daniel choi", "최상범")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("assets/data/daniel-profile-sources.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/daniel-public-source-survey.json"),
    )
    parser.add_argument("--timeout", type=int, default=25)
    return parser.parse_args()


def visible_text(payload: bytes, content_type: str) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    if "json" in content_type:
        try:
            return json.dumps(json.loads(decoded), ensure_ascii=False)
        except json.JSONDecodeError:
            return decoded
    decoded = re.sub(r"(?is)<(script|style|svg).*?>.*?</\1>", " ", decoded)
    decoded = re.sub(r"(?s)<[^>]+>", " ", decoded)
    return " ".join(html.unescape(decoded).split())


def title_from_payload(payload: bytes, content_type: str, url: str) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    if "json" in content_type:
        try:
            data = json.loads(decoded)
            for key in ("name", "title", "full_name", "html_url"):
                if isinstance(data, dict) and data.get(key):
                    return str(data[key])[:240]
        except json.JSONDecodeError:
            pass
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", decoded)
    return " ".join(html.unescape(match.group(1)).split())[:240] if match else url


def relevant_excerpt(text: str, claim: str, radius: int = 220) -> str | None:
    lowered = text.lower()
    claim_terms = [
        token
        for token in re.findall(r"[a-z0-9가-힣]+", claim.lower())
        if len(token) >= 5
    ]
    markers = [*NAME_MARKERS, *claim_terms[:8]]
    positions = [lowered.find(marker) for marker in markers if lowered.find(marker) >= 0]
    if not positions:
        return None
    start = max(0, min(positions) - radius)
    end = min(len(text), min(positions) + radius)
    return text[start:end]


def fetch(url: str, claim: str, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(2_000_000)
            content_type = response.headers.get("Content-Type", "").lower()
            text = visible_text(payload, content_type)
            result = {
                "url": url,
                "final_url": response.geturl(),
                "domain": urlparse(response.geturl()).netloc,
                "status": response.status,
                "content_type": content_type.split(";")[0],
                "title": title_from_payload(payload, content_type, url),
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "downloaded_bytes": len(payload),
                "name_marker_present": any(marker in text.lower() for marker in NAME_MARKERS),
                "review_excerpt": relevant_excerpt(text, claim),
            }
            if "api.github.com/search/issues" in url:
                try:
                    result["github_total_count"] = json.loads(
                        payload.decode("utf-8")
                    ).get("total_count")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            return result
    except urllib.error.HTTPError as error:
        return {"url": url, "status": error.code, "error": str(error)}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"url": url, "status": None, "error": str(error)}


def main() -> None:
    args = parse_args()
    source_data = json.loads(args.sources.read_text(encoding="utf-8"))
    cache = {}
    claims = []
    for claim in source_data["claims"]:
        source_results = []
        for url in claim["sources"]:
            cache_key = (url, claim["claim"])
            if cache_key not in cache:
                cache[cache_key] = fetch(url, claim["claim"], args.timeout)
            source_results.append(cache[cache_key])
        healthy = sum(result.get("status") == 200 for result in source_results)
        claims.append(
            {
                "id": claim["id"],
                "claim": claim["claim"],
                "declared_status": claim["status"],
                "source_count": len(source_results),
                "healthy_source_count": healthy,
                "requires_human_review": claim["status"] != "not_verified",
                "sources": source_results,
            }
        )
    payload = {
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": {
            "automatic_use": "None. Network text never becomes an SFT target automatically.",
            "promotion": "A human must compare the claim with the source before editing the curated profile.",
            "failure": "An unreachable source is a review signal, not proof that a claim is false.",
        },
        "summary": {
            "claims": len(claims),
            "claims_with_sources": sum(item["source_count"] > 0 for item in claims),
            "claims_with_healthy_source": sum(item["healthy_source_count"] > 0 for item in claims),
        },
        "claims": claims,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
