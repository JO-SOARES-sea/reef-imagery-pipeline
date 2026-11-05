#!/usr/bin/env python3
"""
CDSE Sentinel-2 Downloader — VERSÃO CORRIGIDA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cena   : 2018-10-10 | Secchi 23.6m
Tile   : T29SNC  (Algarve ~37°N, 8°W)
Produto: S2A_MSIL2A_20181010T113321_N0500_R080_T29SNC_20230729T053438.SAFE
ID     : 3643e3cf-5f86-4cb6-95bb-6780878e3d6a
Granule: L2A_T29SNC_A017237_20181010T113823
Bandas : B02 + B03 @ 10m/pixel
Saída  : sentinel_images/20181010/
"""

import base64, requests
from pathlib import Path

# ── Credenciais lidas do ficheiro CMEMS (mesmo login) ─────────────────────────
def _read_credentials() -> tuple[str, str]:
    cred_file = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
    raw = base64.b64decode(cred_file.read_text().strip().rstrip('%')).decode()
    creds = {}
    for line in raw.splitlines():
        if '=' in line and not line.startswith('['):
            k, v = line.split('=', 1)
            creds[k.strip()] = v.strip()
    return creds["username"], creds["password"]

CDSE_USER, CDSE_PASSWORD = _read_credentials()

# ── Constantes descobertas via API ────────────────────────────────────────────
PRODUCT_ID    = "71bf5a6c-6a7c-483f-8801-7d95efd2e6bd"
PRODUCT_NAME  = "S2B_MSIL2A_20181009T110939_N0500_R137_T29SNB_20230715T052404.SAFE"
GRANULE_NAME  = ""   # resolved dynamically
TILE          = "T29SNB"
TARGET_DATE   = "2024-09-30"
OUT_DIR       = Path("sentinel_images/20240930")

TOKEN_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Autenticação OAuth2 ────────────────────────────────────────────────────
def get_token(user: str, pwd: str) -> str:
    resp = requests.post(TOKEN_URL, data={
        "client_id":  "cdse-public",
        "username":   user,
        "password":   pwd,
        "grant_type": "password",
    })
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print("✅ Token OAuth2 obtido.")
    return token


# ── 2. Pesquisa via contains(Name,tile) ──────────────────────────────────────
def search_scene(token: str, tile: str = TILE, date: str = TARGET_DATE) -> dict:
    start = f"{date}T00:00:00.000Z"
    end   = f"{date}T23:59:59.000Z"
    filt = (
        f"Collection/Name eq 'SENTINEL-2'"
        f" and contains(Name,'{tile}')"
        f" and ContentDate/Start gt {start}"
        f" and ContentDate/Start lt {end}"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"  att:att/Name eq 'productType'"
        f"  and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
    )
    resp = requests.get(
        CATALOGUE_URL,
        params={"$filter": filt, "$top": 5, "$orderby": "ContentDate/Start asc"},
        headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    products = resp.json().get("value", [])
    if not products:
        raise RuntimeError(f"Nenhuma cena L2A encontrada para tile={tile} data={date}")
    prod = products[0]
    print(f"✅ Cena encontrada : {prod['Name']}")
    print(f"   ID             : {prod['Id']}")
    print(f"   Online         : {prod.get('Online', '?')}")
    print(f"   Tamanho        : {prod['ContentLength'] / 1e6:.1f} MB")
    return prod


# ── 3. Quicklook TCI ─────────────────────────────────────────────────────────
def _scene_ts() -> str:
    """Extract scene timestamp (e.g. T110939) from PRODUCT_NAME."""
    parts = PRODUCT_NAME.split("_")
    return parts[2][8:] if len(parts) > 2 else "T113321"

def get_r10m_filenames(token: str) -> dict[str, str]:
    """Return dict of {keyword: exact_filename} from R10m node listing."""
    url = (
        f"{DOWNLOAD_BASE}/Products({PRODUCT_ID})"
        f"/Nodes({PRODUCT_NAME})"
        f"/Nodes(GRANULE)/Nodes({GRANULE_NAME})"
        f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes"
    )
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    nodes = resp.json().get("result", resp.json().get("value", []))
    return {n["Id"]: n["Id"] for n in nodes}

def download_by_keyword(token: str, keyword: str, label: str) -> Path | None:
    """Download a band/TCI file matching keyword from exact R10m listing."""
    url = (
        f"{DOWNLOAD_BASE}/Products({PRODUCT_ID})"
        f"/Nodes({PRODUCT_NAME})"
        f"/Nodes(GRANULE)/Nodes({GRANULE_NAME})"
        f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes"
    )
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    nodes = resp.json().get("result", resp.json().get("value", []))
    matches = [n["Id"] for n in nodes if keyword in n["Id"]]
    if not matches:
        print(f"  ⚠️  No file matching '{keyword}' in R10m")
        return None
    fname = matches[0]
    dl_url = (
        f"{DOWNLOAD_BASE}/Products({PRODUCT_ID})"
        f"/Nodes({PRODUCT_NAME})"
        f"/Nodes(GRANULE)/Nodes({GRANULE_NAME})"
        f"/Nodes(IMG_DATA)/Nodes(R10m)"
        f"/Nodes({fname})/$value"
    )
    out_path = OUT_DIR / fname
    _download_file(token, dl_url, out_path, label=label)
    return out_path

def download_tci(token: str) -> Path | None:
    return download_by_keyword(token, "TCI", "TCI (quicklook)")

def download_bands(token: str) -> None:
    for band in ["B02", "B03"]:
        download_by_keyword(token, band, f"Banda {band} @ 10m")


# ── Helper: download com progresso ───────────────────────────────────────────
def _download_file(token: str, url: str, out_path: Path, label: str = "") -> None:
    if out_path.exists():
        print(f"⏭  Já existe: {out_path.name}")
        return
    print(f"⬇️  {label}: {out_path.name}")
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True) as r:
        r.raise_for_status()
        total      = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb  = downloaded / 1e6
                    print(f"\r   {pct:5.1f}%  {mb:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print(f"\r✅ {out_path.name} guardado ({downloaded/1e6:.1f} MB)          ")


# ── 5. Lista R10m (debug) ─────────────────────────────────────────────────────
def list_r10m_bands(token: str) -> list[dict]:
    url = (
        f"{DOWNLOAD_BASE}/Products({PRODUCT_ID})"
        f"/Nodes({PRODUCT_NAME})"
        f"/Nodes(GRANULE)/Nodes({GRANULE_NAME})"
        f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes"
    )
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    nodes = resp.json().get("result", [])
    print(f"\n📂 Ficheiros em R10m ({len(nodes)} total):")
    for n in nodes:
        print(f"   {n['Id']:50s}  {n['ContentLength']/1e6:.1f} MB")
    return nodes


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print(" CDSE Sentinel-2 Downloader")
    print(f" Data: {TARGET_DATE}  |  Tile: {TILE}  |  Secchi: 23.6m")
    print(f" User: {CDSE_USER}")
    print(f"{'='*65}\n")

    token = get_token(CDSE_USER, CDSE_PASSWORD)

    product = search_scene(token)
    # Resolve IDs dynamically
    global PRODUCT_ID, PRODUCT_NAME, GRANULE_NAME
    PRODUCT_ID   = product["Id"]
    PRODUCT_NAME = product["Name"]

    # Discover granule name from SAFE structure
    nodes_url = f"{DOWNLOAD_BASE}/Products({PRODUCT_ID})/Nodes({PRODUCT_NAME})/Nodes(GRANULE)/Nodes"
    r = requests.get(nodes_url, headers={"Authorization": f"Bearer {token}"})
    if r.status_code == 200:
        gran_nodes = r.json().get("result", r.json().get("value", []))
        if gran_nodes:
            GRANULE_NAME = gran_nodes[0]["Id"]
            print(f"   Granule: {GRANULE_NAME}")
        else:
            print(f"   ⚠️  No granule nodes found — response: {r.text[:200]}")
    else:
        print(f"   ⚠️  Granule list {r.status_code} — will try without granule nav")

    list_r10m_bands(token)

    print("\n── Quicklook ─────────────────────────────────────────────────")
    download_tci(token)

    print("\n── Bandas B02 + B03 @ 10m ────────────────────────────────────")
    download_bands(token)

    print(f"\n{'='*65}")
    print(f"✅ Concluído! Ficheiros em: {OUT_DIR.resolve()}")
    print(f"{'='*65}")
    for f in sorted(OUT_DIR.glob("T29SNC*")):
        print(f"   {f.name}  ({f.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
