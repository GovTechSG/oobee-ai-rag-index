#!/usr/bin/env python3
"""
Build precomputed RAG chunks for the WCAG 2.2 + DSS + Oobee-DETAILS corpus.

Faithfully ports oobee-desktop's build-wcag-index.js + build-dss-corpus.js to
Python so the same content can be produced inside the oobee-ai-rag-index CI
pipeline without Node.js.

Inputs (all fetched/cloned automatically):
  --wcag-src-dir   Local clone of https://github.com/w3c/wcag @ WCAG22-20241212
  --dss-cache-dir  Dir for scraped DSS JSON files (default: .cache/dss)
  --details-md     Path to oobee DETAILS.md (fetched if absent)

Namespace conventions (match wcagCorpus.js + ragIndex.ts):
  understanding/*  ->  wcag:understanding
  techniques/*/*   ->  wcag:techniques
  dss/*            ->  wcag:dss
  oobee-details/*  ->  wcag:oobee-details
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WCAG_REPO = "https://github.com/w3c/wcag.git"
WCAG_TAG = "WCAG22-20241212"
WCAG_BASE = "https://www.w3.org/WAI/WCAG22"
DETAILS_MD_URL = "https://raw.githubusercontent.com/GovTechSG/oobee/master/DETAILS.md"
DSS_BASE_URL = "https://info.standards.tech.gov.sg/control-catalog/dss"

DSS_CATEGORIES = [
    ("bd", "Baseline Design Practices"),
    ("pr", "Performance and Reliability"),
    ("tx", "Transactions and Payments"),
    ("tl", "Trust and Legitimacy"),
    ("uu", "Understand Users"),
    ("wo", "WCAG : Operable"),
    ("wp", "WCAG : Perceivable"),
    ("wr", "WCAG : Robust"),
    ("wu", "WCAG : Understandable"),
]

UNDERSTANDING_VERSIONS = ["20", "21", "22"]

INDEX_PAGE_NAMES = {
    "index.html", "intro.html", "about.html", "conformance.html",
    "documenting-accessibility-support.html", "refer-to-wcag.html",
    "understanding-act-rules.html", "understanding-metadata.html",
    "understanding-techniques.html", "understanding-template.html",
    "techniques.11tydata.js", "understanding.11tydata.js",
    "understanding.d.ts", "understanding.css",
    "technique-template.html", "changelog.html",
    "changelog.11tydata.json", "techniques.css",
}

OBSOLETED_UNDERSTANDING = {"parsing.html"}  # WCAG 4.1.1 removed in 2.2

TECHNIQUE_CATEGORIES = [
    "aria", "client-side-script", "css", "failures", "flash", "general",
    "html", "pdf", "server-side-script", "silverlight", "smil", "text",
]

MIN_CHUNK_LEN = 40  # characters — shorter chunks skipped

# ---------------------------------------------------------------------------
# HTML → text helpers (mirrors extractText + chunkPage in build-wcag-index.js)
# ---------------------------------------------------------------------------

def _bs4():
    try:
        from bs4 import BeautifulSoup, Tag, NavigableString
        return BeautifulSoup, Tag, NavigableString
    except ImportError:
        raise SystemExit("beautifulsoup4 is required: pip install beautifulsoup4 lxml")


def extract_text(soup_node) -> str:
    """Flatten an HTML subtree to markdown-ish plain text."""
    _, Tag, NavigableString = _bs4()
    buf: list[str] = []

    def walk(node):
        if isinstance(node, NavigableString):
            buf.append(str(node))
            return
        if not isinstance(node, Tag):
            return
        tag = (node.name or "").lower()
        if tag in ("script", "style", "link"):
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            buf.append("\n" + "#" * level + " ")
            for child in node.children:
                walk(child)
            buf.append("\n")
            return
        if tag == "li":
            buf.append("- ")
            for child in node.children:
                walk(child)
            buf.append("\n")
            return
        if tag in ("p", "dt", "dd", "figcaption"):
            for child in node.children:
                walk(child)
            buf.append("\n")
            return
        if tag == "pre":
            buf.append("\n```\n" + node.get_text() + "\n```\n")
            return
        if tag == "code":
            buf.append("`" + node.get_text() + "`")
            return
        for child in node.children:
            walk(child)

    walk(soup_node)
    text = "".join(buf)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_html_page(html: str, meta: dict) -> tuple[list[dict], str]:
    """
    Chunk a WCAG HTML page by top-level <section id="…"> elements.
    Returns (chunks, page_title).  Mirrors chunkPage() in build-wcag-index.js.
    """
    BeautifulSoup, Tag, _ = _bs4()
    soup = BeautifulSoup(html, "lxml")
    body = soup.body
    if not body:
        return [], ""

    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text().strip() if h1_tag else ""

    top_sections = [
        c for c in body.children
        if isinstance(c, Tag) and c.name == "section" and c.get("id")
    ]

    if not top_sections:
        text = extract_text(body)
        if len(text) > MIN_CHUNK_LEN:
            return [{
                "text": (f"# {h1}\n\n" if h1 else "") + text,
                "sectionId": "body",
                "sectionTitle": h1 or meta.get("slug", ""),
            }], h1
        return [], h1

    chunks: list[dict] = []
    for section in top_sections:
        classes = section.get("class") or []
        if "meta" in classes:
            continue
        sid = section.get("id", "")
        heading_tag = section.find(["h2", "h3", "h4", "h5", "h6"])
        heading = heading_tag.get_text().strip() if heading_tag else sid
        text = extract_text(section)
        if len(text) < MIN_CHUNK_LEN:
            continue
        chunks.append({
            "text": f"# {h1}\n\n## {heading or sid}\n\n{text}",
            "sectionId": sid,
            "sectionTitle": heading or sid,
        })
    return chunks, h1


# ---------------------------------------------------------------------------
# WCAG source helpers
# ---------------------------------------------------------------------------

def clone_wcag(dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        logger.info("Reusing existing wcag checkout at %s", dest)
        return
    logger.info("Cloning %s @ %s into %s …", WCAG_REPO, WCAG_TAG, dest)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", WCAG_TAG, "--depth", "1", WCAG_REPO, str(dest)],
        check=True,
    )


def collect_understanding_pages(wcag_src: Path) -> list[dict]:
    pages: list[dict] = []
    for version in UNDERSTANDING_VERSIONS:
        d = wcag_src / "understanding" / version
        if not d.is_dir():
            logger.warning("Missing understanding dir: %s", d)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != ".html":
                continue
            if f.name in INDEX_PAGE_NAMES or f.name in OBSOLETED_UNDERSTANDING:
                continue
            slug = f.stem
            minor = int(version) - 20  # 20→0, 21→1, 22→2
            pages.append({
                "file_path": f,
                "doc_type": "understanding",
                "slug": slug,
                "wcag_version": f"2.{minor}",
                "url": f"{WCAG_BASE}/Understanding/{slug}.html",
            })
    return pages


def collect_technique_pages(wcag_src: Path) -> list[dict]:
    pages: list[dict] = []
    for category in TECHNIQUE_CATEGORIES:
        d = wcag_src / "techniques" / category
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != ".html":
                continue
            if f.name in INDEX_PAGE_NAMES:
                continue
            pages.append({
                "file_path": f,
                "doc_type": "failure" if category == "failures" else "technique",
                "tech_id": f.stem,
                "category": category,
                "url": f"{WCAG_BASE}/Techniques/{category}/{f.name}",
            })
    return pages


def collect_sc_catalog(wcag_src: Path) -> dict:
    """Extract flat SC catalog from guidelines/wcag.json."""
    json_path = wcag_src / "guidelines" / "wcag.json"
    if not json_path.exists():
        logger.warning("wcag.json not found at %s — skipping SC catalog", json_path)
        return {"principles": [], "success_criteria": []}
    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse wcag.json: %s", exc)
        return {"principles": [], "success_criteria": []}

    principles = []
    success_criteria = []
    for p in doc.get("principles", []):
        p_out = {"num": p.get("num"), "handle": p.get("handle"),
                 "versions": p.get("versions"), "guidelines": []}
        for g in p.get("guidelines", []):
            g_out = {"num": g.get("num"), "handle": g.get("handle"),
                     "versions": g.get("versions"), "success_criteria": []}
            for sc in g.get("successcriteria", []):
                sc_id = sc.get("id", "")
                slug = sc_id[6:] if sc_id.startswith("WCAG2:") else None
                row = {
                    "num": sc.get("num"), "handle": sc.get("handle"),
                    "level": sc.get("level"), "versions": sc.get("versions"),
                    "slug": slug,
                    "url": f"{WCAG_BASE}/Understanding/{slug}.html" if slug else None,
                }
                g_out["success_criteria"].append(
                    {"num": row["num"], "handle": row["handle"], "level": row["level"]}
                )
                success_criteria.append(row)
            p_out["guidelines"].append(g_out)
        principles.append(p_out)
    return {"principles": principles, "success_criteria": success_criteria}


# ---------------------------------------------------------------------------
# DSS scraper (ports build-dss-corpus.js)
# ---------------------------------------------------------------------------

def _fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _node_to_md(node) -> str:
    """Mirrors nodeToMarkdown() from build-dss-corpus.js."""
    _, Tag, _ = _bs4()
    if not isinstance(node, Tag):
        return ""
    tn = (node.name or "").lower()
    if tn in ("h3", "h4"):
        return f"\n### {node.get_text().strip()}\n"
    if tn == "p":
        t = re.sub(r"\s+", " ", node.get_text()).strip()
        return f"\n{t}\n" if t else ""
    if tn in ("ul", "ol"):
        lines = [f"- {li.get_text().strip()}" for li in node.find_all("li", recursive=False)]
        return "\n" + "\n".join(lines) + "\n" if lines else ""
    if tn == "blockquote":
        t = re.sub(r"\s+", " ", node.get_text()).strip()
        return f"\n> {t}\n"
    t = re.sub(r"\s+", " ", node.get_text()).strip()
    return f"\n{t}\n" if t else ""


def scrape_dss_category(code: str, name: str) -> dict:
    url = f"{DSS_BASE_URL}/{code}/"
    logger.info("  Scraping DSS %s: %s", code, url)
    html = _fetch(url)
    BeautifulSoup, Tag, _ = _bs4()
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("h1", class_=re.compile(r"prose-display-lg"))
    category_title = title_tag.get_text().strip() if title_tag else name
    desc_tag = soup.find("p", class_=re.compile(r"prose-title-lg-regular"))
    category_desc = desc_tag.get_text().strip() if desc_tag else ""

    h2s = [
        t for t in soup.find_all("h2", class_=re.compile(r"prose-display-sm"))
        if re.match(r"^[A-Z]{2}-\d+:\s+", t.get_text().strip())
    ]

    controls: list[dict] = []
    for h2 in h2s:
        label = h2.get_text().strip()
        m = re.match(r"^([A-Z]{2}-\d+):\s+(.+)$", label)
        if not m:
            continue
        ctrl_code, ctrl_title = m.group(1), m.group(2).strip()
        anchor = h2.get("id", "")
        buf: list[str] = []
        cursor = h2.next_sibling
        while cursor:
            if isinstance(cursor, Tag):
                if (cursor.name or "").lower() == "h2":
                    break
                buf.append(_node_to_md(cursor))
            cursor = cursor.next_sibling
        body = re.sub(r"\n{3,}", "\n\n", "".join(buf)).strip()
        controls.append({
            "code": ctrl_code, "title": ctrl_title, "anchor": anchor,
            "url": f"{DSS_BASE_URL}/{code}/#{anchor}", "body": body,
        })

    if not controls:
        raise RuntimeError(
            f"No controls parsed for DSS category {code} — page shape may have changed"
        )
    logger.info("    %s: %d controls", code, len(controls))
    return {
        "code": code, "title": category_title, "description": category_desc,
        "url": f"{DSS_BASE_URL}/{code}/", "controls": controls,
    }


def ensure_dss_cache(dss_cache_dir: Path, force: bool = False) -> dict:
    import datetime
    manifest_path = dss_cache_dir / "manifest.json"
    if not force and manifest_path.exists():
        logger.info("Reusing cached DSS corpus at %s", dss_cache_dir)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    dss_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "fetchedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "categories": [],
    }
    for code, name in DSS_CATEGORIES:
        result = scrape_dss_category(code, name)
        out_path = dss_cache_dir / f"{code}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["categories"].append({
            "code": result["code"], "title": result["title"],
            "controlCount": len(result["controls"]), "file": f"{code}.json",
        })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    total = sum(c["controlCount"] for c in manifest["categories"])
    logger.info("DSS: wrote %d categories, %d controls", len(manifest["categories"]), total)
    return manifest


# ---------------------------------------------------------------------------
# Oobee DETAILS.md helpers (ports collectOobeeDetailsChunks)
# ---------------------------------------------------------------------------

def fetch_details_md(dest_path: Path) -> str:
    if dest_path.exists():
        logger.info("Reusing cached DETAILS.md at %s", dest_path)
        return dest_path.read_text(encoding="utf-8")
    logger.info("Fetching oobee DETAILS.md from %s …", DETAILS_MD_URL)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    text = _fetch(DETAILS_MD_URL)
    dest_path.write_text(text, encoding="utf-8")
    return text


def parse_coverage_table(md: str) -> list[dict]:
    """
    Port of parseCoverageTable() from build-wcag-index.js.

    Parses the master WCAG↔DSS mapping table in DETAILS.md.
    Row shape:  | WCAG 1.1.1 | WP-1 | A | Yes | | |
    Returns one dict per row:
      { wcag, dss, level, mustFix, goodToFix, needsReview, category }
    """
    row_re = re.compile(
        r"\|\s*WCAG\s+([\d.]+)\s*\|\s*([A-Z]{2}-\d+|—)\s*\|\s*(A{1,3})\s*"
        r"\|([^|\n]*)\|([^|\n]*)\|([^|\n]*)\|"
    )
    rows = []
    for m in row_re.finditer(md):
        must_fix    = bool(re.search(r"Yes", m.group(4)))
        good_to_fix = bool(re.search(r"Yes", m.group(5)))
        needs_review = bool(re.search(r"Yes", m.group(6)))
        category = (
            "needsReview" if needs_review
            else "mustFix" if must_fix
            else "goodToFix" if good_to_fix
            else None
        )
        rows.append({
            "wcag": m.group(1),
            "dss": None if m.group(2) == "—" else m.group(2),
            "level": m.group(3),
            "mustFix": must_fix,
            "goodToFix": good_to_fix,
            "needsReview": needs_review,
            "category": category,
        })
    return rows


def parse_dss_to_wcag_map(md: str) -> dict[str, list[str]]:
    """Derives {dss_code: [wcag_num, …]} from parse_coverage_table()."""
    mapping: dict[str, list[str]] = {}
    for row in parse_coverage_table(md):
        if not row["dss"]:
            continue
        lst = mapping.setdefault(row["dss"], [])
        if row["wcag"] not in lst:
            lst.append(row["wcag"])
    return mapping


def chunk_details_md(md: str) -> list[dict]:
    """Split DETAILS.md at ## headings. Mirrors chunkDetailsMd()."""
    lines = md.splitlines()
    sections: list[dict] = []
    current: Optional[dict] = None
    for line in lines:
        if re.match(r"^##\s+\S", line) and not line.startswith("###"):
            if current:
                sections.append(current)
            heading = re.sub(r"^##\s+", "", line).strip()
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            current = {"heading": heading, "slug": slug, "lines": [line]}
        elif current:
            current["lines"].append(line)
    if current:
        sections.append(current)
    return [
        {"slug": s["slug"], "heading": s["heading"], "text": "\n".join(s["lines"]).strip()}
        for s in sections
        if len("\n".join(s["lines"]).strip()) > 60
    ]


# ---------------------------------------------------------------------------
# Chunk-ID helper
# ---------------------------------------------------------------------------

def _chunk_id(namespace: str, source_file: str, index: int) -> str:
    from pathlib import PurePosixPath
    name = PurePosixPath(source_file).name
    return f"{namespace}::{name}::{index}"


# ---------------------------------------------------------------------------
# Main entry point called by build_local_index.py
# ---------------------------------------------------------------------------

def build_wcag_chunks(
    wcag_src_dir: Path,
    dss_cache_dir: Path,
    details_md_path: Path,
    chunk_records: list,
    all_texts: list,
    force_dss: bool = False,
) -> dict:
    """
    Append WCAG+DSS+DETAILS chunks to the shared lists used by
    build_local_index.py.  Returns a metadata dict for _meta.json.
    """
    # --- WCAG HTML pages ---------------------------------------------------
    understanding_pages = collect_understanding_pages(wcag_src_dir)
    technique_pages = collect_technique_pages(wcag_src_dir)
    logger.info(
        "WCAG: %d understanding pages, %d technique/failure pages",
        len(understanding_pages), len(technique_pages),
    )

    wcag_chunk_count = 0
    for page in understanding_pages + technique_pages:
        html = page["file_path"].read_text(encoding="utf-8", errors="replace")
        page_chunks, title = chunk_html_page(html, page)
        if not page_chunks:
            continue
        namespace = (
            "wcag:understanding" if page["doc_type"] == "understanding"
            else "wcag:techniques"
        )
        rel = str(page["file_path"])
        for i, chunk in enumerate(page_chunks):
            meta_fields: dict = {
                "docType": page["doc_type"], "title": title,
                "sectionId": chunk["sectionId"], "sectionTitle": chunk["sectionTitle"],
                "url": page["url"], "namespace": namespace,
            }
            if page["doc_type"] == "understanding":
                meta_fields["slug"] = page["slug"]
                meta_fields["wcagVersion"] = page["wcag_version"]
            else:
                meta_fields["techId"] = page["tech_id"]
                meta_fields["category"] = page["category"]
            text = chunk["text"]
            embed_text = "\n\n".join(filter(None, [title, chunk["sectionTitle"], text]))
            chunk_records.append({
                "id": _chunk_id(namespace, rel, i),
                "namespace": namespace, "sourceFile": rel,
                "heading": chunk["sectionTitle"], "text": text,
                "metadata": meta_fields,
            })
            all_texts.append(embed_text)
            wcag_chunk_count += 1

    logger.info("WCAG HTML -> %d chunks", wcag_chunk_count)

    # --- DSS controls -------------------------------------------------------
    dss_manifest = ensure_dss_cache(dss_cache_dir, force=force_dss)
    dss_chunk_count = 0
    for cat_entry in dss_manifest["categories"]:
        cat_data = json.loads(
            (dss_cache_dir / cat_entry["file"]).read_text(encoding="utf-8")
        )
        for ctrl in cat_data["controls"]:
            namespace = "wcag:dss"
            text = (
                f"# DSS {ctrl['code']}: {ctrl['title']}\n\n"
                f"**Category:** {cat_data['title']}\n\n" + ctrl["body"]
            )
            embed_text = "\n\n".join(filter(None, [ctrl["code"], ctrl["title"], text]))
            chunk_records.append({
                "id": _chunk_id(namespace, ctrl["code"], dss_chunk_count),
                "namespace": namespace,
                "sourceFile": f"dss/{cat_data['code']}/{ctrl['code']}",
                "heading": ctrl["title"], "text": text,
                "metadata": {
                    "docType": "dss", "techId": ctrl["code"], "title": ctrl["title"],
                    "sectionId": ctrl["anchor"], "sectionTitle": ctrl["title"],
                    "category": cat_data["code"], "categoryTitle": cat_data["title"],
                    "url": ctrl["url"], "namespace": namespace,
                },
            })
            all_texts.append(embed_text)
            dss_chunk_count += 1

    logger.info("DSS -> %d chunks", dss_chunk_count)


    # --- Oobee DETAILS.md ---------------------------------------------------
    details_text = fetch_details_md(details_md_path)
    details_sections = chunk_details_md(details_text)
    namespace = "wcag:oobee-details"
    details_chunk_count = 0
    for i, section in enumerate(details_sections):
        text = (
            f"# Oobee -- {section['heading']}\n\n"
            + re.sub(r"^##\s+.+\n", "", section["text"], count=1).strip()
        )
        embed_text = "\n\n".join(
            filter(None, ["Oobee -- Scan Issue Details", section["heading"], text])
        )
        chunk_records.append({
            "id": _chunk_id(namespace, "DETAILS.md", i),
            "namespace": namespace,
            "sourceFile": "oobee/DETAILS.md",
            "heading": section["heading"],
            "text": text,
            "metadata": {
                "docType": "oobee-details",
                "title": "Oobee -- Scan Issue Details",
                "sectionId": section["slug"],
                "sectionTitle": section["heading"],
                "url": f"{DETAILS_MD_URL}#{section['slug']}",
                "namespace": namespace,
            },
        })
        all_texts.append(embed_text)
        details_chunk_count += 1

    logger.info("Oobee DETAILS.md -> %d chunks", details_chunk_count)

    # --- coverage table (for oobeeDetails in _meta.json) --------------------
    coverage = parse_coverage_table(details_text)
    if not coverage:
        logger.warning(
            "DETAILS.md coverage table returned 0 rows — "
            "list_corpus_metadata(source='oobee-details') will be incomplete. "
            "The upstream table shape may have changed."
        )

    # --- SC catalog (for wcag in _meta.json) --------------------------------
    sc_catalog = collect_sc_catalog(wcag_src_dir)

    # --- techniques_by_category + failure_pages (for wcag in _meta.json) ---
    techniques_by_category: dict[str, int] = {}
    failure_pages = 0
    for p in technique_pages:
        techniques_by_category[p["category"]] = (
            techniques_by_category.get(p["category"], 0) + 1
        )
        if p.get("docType") == "failure":
            failure_pages += 1

    # --- understanding_by_version (for wcag in _meta.json) -----------------
    understanding_by_version: dict[str, int] = {}
    for p in understanding_pages:
        v = p.get("wcagVersion", "unknown")
        understanding_by_version[v] = understanding_by_version.get(v, 0) + 1

    # --- DSS catalog shaped for list_corpus_metadata ------------------------
    dss_catalog = {
        "fetchedAt": dss_manifest.get("fetchedAt"),
        "total_categories": len(dss_manifest["categories"]),
        "total_controls": sum(
            c["controlCount"] for c in dss_manifest["categories"]
        ),
        "categories": dss_manifest["categories"],
    }

    # --- coverage totals ----------------------------------------------------
    coverage_totals_by_level: dict[str, int] = {}
    coverage_totals_by_category: dict[str, int] = {}
    for row in coverage:
        lvl = row["level"]
        coverage_totals_by_level[lvl] = coverage_totals_by_level.get(lvl, 0) + 1
        cat = row["category"]
        if cat:
            coverage_totals_by_category[cat] = (
                coverage_totals_by_category.get(cat, 0) + 1
            )

    return {
        # -- chunk counts (for build_local_index.py log line) ----------------
        "wcag_understanding_count": len(understanding_pages),
        "wcag_technique_count": len(technique_pages),
        "wcag_chunk_count": wcag_chunk_count,
        "dss_chunk_count": dss_chunk_count,
        "details_chunk_count": details_chunk_count,
        # -- wcag key (mirrors Node builder's _meta.json shape) --------------
        "wcag": {
            "sourceTag": WCAG_TAG,
            "total_understanding_pages": len(understanding_pages),
            "understanding_by_version": understanding_by_version,
            "total_technique_pages": len(technique_pages),
            "techniques_by_category": techniques_by_category,
            "failure_pages": failure_pages,
            "total_success_criteria": len(sc_catalog["success_criteria"]),
            "success_criteria_totals_by_level": {
                lvl: sum(
                    1 for sc in sc_catalog["success_criteria"] if sc.get("level") == lvl
                )
                for lvl in ("A", "AA", "AAA")
            },
            "principles": sc_catalog["principles"],
            "success_criteria": sc_catalog["success_criteria"],
        },
        # -- dss key (mirrors Node builder's _meta.json shape) ---------------
        "dss": dss_catalog,
        # -- oobeeDetails key (mirrors Node builder's _meta.json shape) ------
        "oobeeDetails": {
            "sourceUrl": DETAILS_MD_URL,
            "total_sections": details_chunk_count,
            "sections": [
                {
                    "heading": s["heading"],
                    "slug": s["slug"],
                    "url": f"{DETAILS_MD_URL}#{s['slug']}",
                }
                for s in details_sections
            ],
            "total_coverage_rows": len(coverage),
            "coverage_totals_by_level": coverage_totals_by_level,
            "coverage_totals_by_category": coverage_totals_by_category,
            "coverage": coverage,
        },
    }

