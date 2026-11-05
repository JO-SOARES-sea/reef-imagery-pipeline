#!/usr/bin/env python3
import os
import sys
import json
import requests
from pathlib import Path

BASE = "https://cdd.dgterritorio.gov.pt"

def load_cookies_from_header(cookie_header: str):
    cookies = {}
    for chunk in cookie_header.split(";"):
        if "=" in chunk:
            k, v = chunk.strip().split("=", 1)
            cookies[k] = v
    return cookies

def session_from_env():
    s = requests.Session()
    ua = os.getenv("DGT_UA", "Mozilla/5.0")
    s.headers.update({
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Origin": BASE,
        "Referer": f"{BASE}/dgt-fe/downloads",
    })
    cookie_header = os.getenv("DGT_COOKIE", "").strip()
    if cookie_header:
        s.cookies.update(load_cookies_from_header(cookie_header))
    auth = os.getenv("DGT_AUTH", "").strip()
    if auth:
        s.headers["Authorization"] = auth
    return s

def dump_response(r):
    print("STATUS:", r.status_code)
    print("URL:", r.url)
    print("CONTENT-TYPE:", r.headers.get("content-type"))
    txt = r.text[:1200]
    print(txt)

def request_json(session, method, url, **kwargs):
    r = session.request(method, url, timeout=60, **kwargs)
    dump_response(r)
    r.raise_for_status()
    ct = r.headers.get("content-type", "")
    if "json" in ct:
        return r.json()
    return {"raw": r.text}

def download_file(session, url, out):
    with session.get(url, stream=True, timeout=300) as r:
        dump_response(r)
        r.raise_for_status()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"\nSaved to {out}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dgt_probe.py check")
        print("  python dgt_probe.py token <URL> [METHOD=GET] [JSON_BODY]")
        print("  python dgt_probe.py download <SIGNED_OR_TOKEN_URL> <output_file>")
        sys.exit(1)

    cmd = sys.argv[1]
    s = session_from_env()

    if cmd == "check":
        r = s.get(f"{BASE}/dgt-fe/downloads", timeout=60)
        dump_response(r)
        print("\nAuthenticated =", "Terminar sessão" in r.text)
        return

    if cmd == "token":
        if len(sys.argv) < 3:
            sys.exit("Missing token URL")
        url = sys.argv[2]
        method = sys.argv[3] if len(sys.argv) >= 4 else "GET"
        body = None
        if len(sys.argv) >= 5:
            body = json.loads(sys.argv[4])
        kwargs = {}
        if body is not None:
            kwargs["json"] = body
        data = request_json(s, method.upper(), url, **kwargs)
        print("\nJSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if cmd == "download":
        if len(sys.argv) < 4:
            sys.exit("Missing URL and output")
        download_file(s, sys.argv[2], sys.argv[3])
        return

    sys.exit(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
