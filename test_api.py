from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _http_get(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for PrexSyn API.")
    parser.add_argument("--base-url", default="http://localhost:8011")
    parser.add_argument("--smiles", default="CCO")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    health_url = f"{args.base_url.rstrip('/')}/health"
    predict_url = f"{args.base_url.rstrip('/')}/predict"
    payload = {
        "smiles": args.smiles,
        "top": args.top,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
    }

    try:
        health = _http_get(health_url, timeout=args.timeout)
        print("Health response:")
        print(json.dumps(health, indent=2))

        prediction = _http_post_json(predict_url, payload=payload, timeout=args.timeout)
        print("\nPredict response:")
        print(json.dumps(prediction, indent=2))
        return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
