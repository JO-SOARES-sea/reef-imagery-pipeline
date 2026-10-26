#!/usr/bin/env python3
"""
scripts/index_local_sources.py — Índice dos dados externos já descarregados para o disco.

Muitos documentos e dados (EIA/AIA, COSMO, PSOEM, multifeixe, sonar, artigos) já foram obtidos
online e guardados em /Volumes/4G_WD_BLACK/BACKUP. Este script cataloga-os para que sejam
consultados localmente antes de voltar a pesquisar na internet.

Para cada ficheiro regista: caminho, tamanho, data, extensão, etiquetas temáticas e, quando
aplicável, extensão geográfica (WGS84) e CRS de rasters/vetores, nº de páginas e início do texto
de PDFs, título de .md/.txt e código de levantamento COSMO.

Saída:
    data/external/local_sources_index.csv   (uma linha por ficheiro; pesquisável)
    docs/external_data/LOCAL_SOURCES_INDEX.md  (resumo por pasta e por tema)

Uso:
    python scripts/index_local_sources.py
    python scripts/index_local_sources.py --root /Volumes/4G_WD_BLACK/BACKUP --root ~/dev/UnderwaterVisibility/reports
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import re
import subprocess
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index_local_sources")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = [Path("/Volumes/4G_WD_BLACK/BACKUP")]
OUT_CSV = PROJECT_ROOT / "data" / "external" / "local_sources_index.csv"
OUT_MD = PROJECT_ROOT / "docs" / "external_data" / "LOCAL_SOURCES_INDEX.md"
# Texto integral dos PDFs (pdftotext), para pesquisar com grep sem voltar à internet
TEXT_CACHE = PROJECT_ROOT / "data" / "external" / "local_sources_text"

# Etiqueta -> padrões (procurados no caminho e no início do texto, sem distinção de maiúsculas)
TOPICS: dict[str, list[str]] = {
    "greta": [r"greta", r"\bgrt\d?\b", r"pedra0"],
    "b24_aviao": [r"b-?24", r"liberator", r"pb4y", r"mar de faro"],
    "aia3587_quarteira_garrao": [r"aia ?3587", r"3587", r"quarteira.?garr[aã]o", r"t21\.031"],
    "quarteira_vilamoura": [r"quarteira", r"vilamoura", r"\bqf[-_]"],
    "garrao_ancao_quinta_lago": [r"garr[aã]o", r"anc[aã]o", r"quinta.?do.?lago", r"vale.?do.?lobo"],
    "praia_faro_ilha_faro": [r"praia.?de.?faro", r"ilha.?de.?faro", r"lthfa1", r"barreta", r"lthbaif"],
    "valado_albufeira": [r"valado", r"albufeira", r"lthpeal", r"armac[aã]o.?de.?p[eê]ra", r"benagil"],
    "recifes_artificiais": [r"recifes?.?artificia", r"artificial.?reef", r"complexos?.?recifa", r"cubos"],
    "multifeixe_mbes": [r"multifeixe", r"multibeam", r"\bmbes\b", r"\bcosmo\b", r"lth[a-z]{3,5}\d"],
    "sonar_lateral_sss": [r"sonar", r"side.?scan", r"varrimento.?lateral", r"\bsss\b"],
    "litologia_rocha": [r"afloramento", r"litol", r"rocha", r"rock.?outcrop", r"teixeira", r"plataforma.?de.?abras"],
    "patrimonio_naufragios": [r"naufr", r"wreck", r"patrim", r"arqueol", r"\bcns\b"],
    "psoem_dgrm": [r"psoem", r"dgrm", r"ordenamento.?do.?espa[cç]o.?mar"],
    "batimetria_emodnet_ih": [r"emodnet", r"batim", r"bathym", r"hidrogr", r"hidromar", r"\bih\b"],
    "satelite": [r"sentinel", r"\bs2[ab]?\b", r"ortosat", r"landsat", r"pl[eé]iades", r"planet"],
    "ccmar_ipma": [r"ccmar", r"ipma", r"ipimar", r"rensub"],
}
# Pastas sem relação com o projeto marinho (indexadas mas marcadas)
UNRELATED_HINTS = [r"/hager/", r"/re-cases/", r"/recovery/nsfw", r"\.mov$", r"acores_diveroid"]

COSMO_CODE = re.compile(r"\b(LTH[A-Z]{3,5}\d)\b", re.IGNORECASE)
TEXT_EXT = {".md", ".txt", ".csv", ".json", ".geojson", ".gpx", ".log", ".tsv", ".xml", ".html", ".py", ".sh"}
RASTER_EXT = {".tif", ".tiff", ".asc", ".img", ".vrt"}
VECTOR_EXT = {".geojson", ".shp", ".gpx", ".kml"}


def _head_text(path: Path, n_chars: int = 4000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(n_chars)
    except OSError:
        return ""


def _md_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()[:160]
        if line:
            return line[:160]
    return ""


def _pdf_info(path: Path, root: Path) -> tuple[str, str, str]:
    """Nº de páginas, título + início do texto e caminho do texto integral em cache (poppler)."""
    pages, title = "", ""
    try:
        info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=60).stdout
        for line in info.splitlines():
            if line.startswith("Pages:"):
                pages = f"{line.split(':', 1)[1].strip()} pág."
            elif line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass

    cache = TEXT_CACHE / (str(path.relative_to(root)).replace("/", "__") + ".txt")
    if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["pdftotext", "-layout", str(path), str(cache)], capture_output=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return pages, f"(PDF ilegível: {exc.__class__.__name__})", ""
    text = cache.read_text(encoding="utf-8", errors="ignore") if cache.exists() else ""
    if not text.strip():
        return pages, (title + " | " if title else "") + "(sem camada de texto: digitalizado?)", ""
    snippet = re.sub(r"\s+", " ", text[:3000]).strip()[:300]
    return pages, (title + " | " if title else "") + snippet, str(cache.relative_to(PROJECT_ROOT))


def _raster_info(path: Path) -> dict[str, str]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(path) as src:
            info = {
                "crs": str(src.crs) if src.crs else "",
                "detail": f"{src.width}x{src.height} px, {src.count} banda(s), res {abs(src.res[0]):.2f}",
            }
            if src.crs:
                w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                info["bbox_wgs84"] = f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}"
            return info
    except Exception as exc:
        return {"detail": f"(raster ilegível: {exc.__class__.__name__})"}


def _vector_info(path: Path) -> dict[str, str]:
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        info = {"crs": str(gdf.crs) if gdf.crs else "", "detail": f"{len(gdf)} entidades"}
        if len(gdf) and gdf.crs:
            w, s, e, n = gdf.to_crs("EPSG:4326").total_bounds
            info["bbox_wgs84"] = f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}"
        cols = [c for c in gdf.columns if c != "geometry"][:8]
        if cols:
            info["detail"] += f"; campos: {', '.join(map(str, cols))}"
        return info
    except Exception as exc:
        return {"detail": f"(vetor ilegível: {exc.__class__.__name__})"}


def _zip_info(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        return f"zip com {len(names)} ficheiros: " + "; ".join(names[:8])
    except Exception as exc:
        return f"(zip ilegível: {exc.__class__.__name__})"


def _tags(haystack: str) -> list[str]:
    low = haystack.lower()
    return [t for t, pats in TOPICS.items() if any(re.search(p, low) for p in pats)]


def index_file(path: Path, root: Path) -> dict[str, str]:
    stat = path.stat()
    ext = path.suffix.lower()
    rec = {
        "root": str(root),
        "path": str(path.relative_to(root)),
        "ext": ext,
        "size_mb": f"{stat.st_size / 1e6:.3f}",
        "modified": dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
        "title_or_snippet": "",
        "detail": "",
        "crs": "",
        "bbox_wgs84": "",
        "cosmo_code": "",
        "tags": "",
        "unrelated": "",
        "text_cache": "",
    }
    text = ""
    if ext in TEXT_EXT and stat.st_size < 50e6:
        text = _head_text(path)
        if ext in {".md", ".txt"}:
            rec["title_or_snippet"] = _md_title(text)
    if ext == ".pdf":
        rec["detail"], rec["title_or_snippet"], rec["text_cache"] = _pdf_info(path, root)
        text = (PROJECT_ROOT / rec["text_cache"]).read_text(encoding="utf-8", errors="ignore")[:20000] if rec["text_cache"] else ""
    if ext in RASTER_EXT:
        rec.update(_raster_info(path))
    if ext in VECTOR_EXT:
        rec.update(_vector_info(path))
    if ext == ".zip":
        rec["detail"] = _zip_info(path)
        text = rec["detail"]

    haystack = f"{rec['path']} {text[:20000]}"
    rec["cosmo_code"] = ",".join(sorted({m.upper() for m in COSMO_CODE.findall(haystack)}))
    rec["tags"] = ";".join(_tags(haystack))
    full = "/" + rec["path"].lower()
    rec["unrelated"] = "sim" if any(re.search(p, full) for p in UNRELATED_HINTS) else ""
    return rec


def write_markdown(records: list[dict[str, str]], roots: list[Path]) -> None:
    by_folder: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in records:
        top = r["path"].split("/", 1)[0] if "/" in r["path"] else "(raiz)"
        by_folder[f"{Path(r['root']).name}/{top}"].append(r)
    tag_counts = Counter(t for r in records for t in r["tags"].split(";") if t)

    lines = [
        "# Índice de fontes locais (dados já descarregados)",
        "",
        f"Gerado por `scripts/index_local_sources.py` em {dt.date.today().isoformat()} a partir de: "
        + ", ".join(f"`{r}`" for r in roots)
        + ".",
        "",
        "**Antes de pesquisar online, procure aqui.** O índice completo, uma linha por ficheiro com etiquetas, "
        "extensão WGS84, CRS e excerto do texto, está em `data/external/local_sources_index.csv`. "
        "O texto integral de todos os PDFs está em `data/external/local_sources_text/`. Exemplos:",
        "",
        "```bash",
        "grep -i greta data/external/local_sources_index.csv",
        "grep -ril 'recifes artificiais' data/external/local_sources_text/ | head",
        "grep -n -i 'B-24\\|B24' data/external/local_sources_text/*Arqueologia*.txt",
        "```",
        "",
        f"Total: {len(records)} ficheiros, {sum(float(r['size_mb']) for r in records) / 1e3:.1f} GB.",
        "",
        "## Ficheiros por tema",
        "",
        "| Tema | Ficheiros |",
        "|---|---|",
    ]
    lines += [f"| `{t}` | {n} |" for t, n in tag_counts.most_common()]
    lines += ["", "## Pastas", ""]
    for folder in sorted(by_folder):
        recs = by_folder[folder]
        size = sum(float(r["size_mb"]) for r in recs)
        exts = Counter(r["ext"] or "(sem)" for r in recs).most_common(6)
        tags = Counter(t for r in recs for t in r["tags"].split(";") if t).most_common(6)
        cosmo = sorted({c for r in recs for c in r["cosmo_code"].split(",") if c})
        unrelated = all(r["unrelated"] for r in recs)
        lines.append(f"### `{folder}`{' — sem relação com o projeto' if unrelated else ''}")
        lines.append("")
        lines.append(
            f"{len(recs)} ficheiros, {size:,.1f} MB. Tipos: "
            + ", ".join(f"{e} ({n})" for e, n in exts)
            + (f". Temas: {', '.join(f'{t} ({n})' for t, n in tags)}" if tags else "")
            + (f". Levantamentos COSMO: {', '.join(cosmo)}" if cosmo else "")
            + "."
        )
        docs = [r for r in recs if r["ext"] in {".md", ".pdf"} and r["title_or_snippet"]][:12]
        geo = [r for r in recs if r["bbox_wgs84"]][:8]
        if docs:
            lines.append("")
            for r in docs:
                lines.append(f"- `{r['path']}`: {r['title_or_snippet'][:140]}")
        if geo:
            lines.append("")
            for r in geo:
                lines.append(f"- 🗺 `{r['path']}`: bbox {r['bbox_wgs84']}; {r['crs']}; {r['detail'][:80]}")
        lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa dados externos já descarregados para o disco")
    parser.add_argument("--root", type=Path, action="append", help="Pasta a indexar (repetível).")
    args = parser.parse_args()
    roots = [r.expanduser() for r in (args.root or DEFAULT_ROOTS)]

    records = []
    for root in roots:
        files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("._") and p.name != ".DS_Store")
        logger.info(f"{root}: {len(files)} ficheiros")
        for i, path in enumerate(files, 1):
            try:
                records.append(index_file(path, root))
            except OSError as exc:
                logger.warning(f"{path}: {exc}")
            if i % 250 == 0:
                logger.info(f"  {i}/{len(files)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    write_markdown(records, roots)
    logger.info(f"Índice: {OUT_CSV} ({len(records)} linhas) e {OUT_MD}")


if __name__ == "__main__":
    main()
