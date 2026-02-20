#!/usr/bin/env python3
"""
Scraper for WCAG 2.2 success criteria.
Downloads the official W3C WCAG JSON and converts each success criterion
into a structured markdown file for embedding.
"""

import argparse
import hashlib
import html
import json
import logging
import re
import shutil
import urllib.request
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_wcag_json(url: str) -> tuple[dict, str]:
    """
    Download WCAG JSON from URL.

    Returns:
        Tuple of (parsed JSON dict, SHA256 hash of raw JSON content)
    """
    logger.info(f"Downloading WCAG JSON from {url}...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "oobee-ai-rag-index/1.0"}
    )
    with urllib.request.urlopen(req) as response:
        raw = response.read()
    content_hash = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    logger.info(f"Downloaded WCAG JSON, hash: {content_hash[:12]}")
    return data, content_hash


def html_to_markdown(html_content: str) -> str:
    """
    Convert simple HTML to markdown.
    Handles the limited tag set used in WCAG JSON content fields.
    """
    if not html_content:
        return ""
    text = html_content

    # Remove wrapping <p> tags, convert to double newlines
    text = re.sub(r'<p\b[^>]*>', '', text)
    text = re.sub(r'</p>', '\n\n', text)

    # Links: <a href="...">text</a> -> [text](url)
    text = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text)

    # Bold
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text)
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text)

    # Italic
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*', text)

    # Code
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text)

    # Line breaks
    text = re.sub(r'<br\s*/?>', '\n', text)

    # Definition lists
    text = re.sub(r'<dl\b[^>]*>', '', text)
    text = re.sub(r'</dl>', '', text)
    text = re.sub(r'<dt\b[^>]*>(.*?)</dt>', r'**\1**\n', text)
    text = re.sub(r'<dd\b[^>]*>(.*?)</dd>', r': \1\n', text)

    # Lists
    text = re.sub(r'<[uo]l\b[^>]*>', '', text)
    text = re.sub(r'</[uo]l>', '', text)
    text = re.sub(r'<li\b[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)

    # Strip remaining tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    text = html.unescape(text)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_introduced_version(criterion: dict) -> str:
    """Return the earliest WCAG version from the versions array."""
    versions = criterion.get("versions", ["2.0"])
    return min(versions, key=lambda v: [int(x) for x in v.split(".")])


def generate_criterion_filename(criterion: dict) -> str:
    """
    Generate filename for a criterion.
    Example: "1.1.1-non-text-content.md"
    """
    return f"{criterion['num']}-{criterion['id']}.md"


def format_single_technique(tech: dict) -> str:
    """Format a single technique entry, handling 'and' conjunctions and 'using' sub-arrays."""
    if "and" in tech:
        parts = [format_single_technique(t) for t in tech["and"]]
        return " **AND** ".join(parts)

    tech_id = tech.get("id", "")
    title = tech.get("title", "")
    technology = tech.get("technology", "")

    if tech_id and technology:
        url = f"https://www.w3.org/WAI/WCAG22/Techniques/{technology}/{tech_id}"
        line = f"- [{tech_id}]({url}): {title}"
    else:
        line = f"- {tech_id}: {title}" if tech_id else f"- {title}"

    # Handle 'using' sub-array
    using = tech.get("using", [])
    if using:
        for sub_tech in using:
            sub_id = sub_tech.get("id", "")
            sub_title = sub_tech.get("title", "")
            sub_technology = sub_tech.get("technology", "")
            if sub_id and sub_technology:
                sub_url = f"https://www.w3.org/WAI/WCAG22/Techniques/{sub_technology}/{sub_id}"
                line += f"\n  - [{sub_id}]({sub_url}): {sub_title}"
            else:
                line += f"\n  - {sub_id}: {sub_title}" if sub_id else f"\n  - {sub_title}"

    return line


def format_techniques_list(techniques: list[dict]) -> str:
    """Format a flat list of techniques (advisory or failure)."""
    lines = []
    for tech in techniques:
        lines.append(format_single_technique(tech))
    return "\n".join(lines)


def format_sufficient_techniques(sufficient: list) -> str:
    """
    Format the sufficient techniques array.
    Items can be either situation objects (with title/techniques/groups but no id)
    or simple technique objects (with id/technology/title).
    """
    lines = []
    for item in sufficient:
        # Simple technique object (has id) or conjunction (has "and")
        if "id" in item or "and" in item:
            lines.append(format_single_technique(item))
        # Situation object with title and nested techniques/groups
        elif "title" in item:
            title = html_to_markdown(item["title"])
            lines.append(f"### {title}")
            lines.append("")

            for tech in item.get("techniques", []):
                lines.append(format_single_technique(tech))

            for group in item.get("groups", []):
                group_title = group.get("title", "")
                if group_title:
                    lines.append(f"\n*{group_title}:*\n")
                for tech in group.get("techniques", []):
                    lines.append(format_single_technique(tech))

            lines.append("")

    return "\n".join(lines)


def format_details(details: list[dict]) -> str:
    """Convert the details/notes array to markdown."""
    lines = []
    for detail in details:
        detail_type = detail.get("type", "")

        if detail_type == "ulist":
            for item in detail.get("items", []):
                handle = item.get("handle", "")
                text = html_to_markdown(item.get("text", ""))
                if handle:
                    lines.append(f"- **{handle}:** {text}")
                else:
                    lines.append(f"- {text}")

        elif detail_type == "olist":
            for i, item in enumerate(detail.get("items", []), 1):
                handle = item.get("handle", "")
                text = html_to_markdown(item.get("text", ""))
                if handle:
                    lines.append(f"{i}. **{handle}:** {text}")
                else:
                    lines.append(f"{i}. {text}")

        elif detail_type == "note":
            text = html_to_markdown(detail.get("handle", ""))
            note_content = html_to_markdown(detail.get("text", ""))
            if note_content:
                lines.append(f"> **Note:** {note_content}")
            elif text:
                lines.append(f"> **Note:** {text}")

        else:
            text = html_to_markdown(
                detail.get("handle", detail.get("text", ""))
            )
            if text:
                lines.append(text)

    return "\n".join(lines)


def criterion_to_markdown(
    criterion: dict,
    principle: dict,
    guideline: dict
) -> str:
    """
    Convert a single success criterion to a structured markdown document.

    Args:
        criterion: Success criterion dict from JSON
        principle: Parent principle dict
        guideline: Parent guideline dict

    Returns:
        Markdown string
    """
    num = criterion["num"]
    handle = criterion["handle"]
    title = criterion.get("title", "")
    level = criterion.get("level", "")
    versions = criterion.get("versions", [])
    criterion_id = criterion["id"]

    ref_url = f"https://www.w3.org/WAI/WCAG22/Understanding/{criterion_id}.html"

    lines = []

    # Title
    lines.append(f"# {num} {handle}")
    lines.append("")

    # Metadata block
    lines.append(f"**Level:** {level}")
    lines.append(f"**Versions:** WCAG {', '.join(versions)}")
    lines.append(f"**Principle:** {principle['num']} {principle['handle']}")
    lines.append(f"**Guideline:** {guideline['num']} {guideline['handle']}")
    lines.append(f"**Reference:** [{num} {handle}]({ref_url})")
    lines.append("")

    # Description (use title for clean text; content HTML duplicates title + details)
    lines.append("## Description")
    lines.append("")
    if title:
        lines.append(html_to_markdown(title))
        lines.append("")

    # Details / Notes
    details = criterion.get("details", [])
    if details:
        lines.append("## Details")
        lines.append("")
        lines.append(format_details(details))
        lines.append("")

    # Techniques
    techniques = criterion.get("techniques", {})

    sufficient = techniques.get("sufficient", [])
    if sufficient:
        lines.append("## Sufficient Techniques")
        lines.append("")
        lines.append(format_sufficient_techniques(sufficient))
        lines.append("")

    advisory = techniques.get("advisory", [])
    if advisory:
        lines.append("## Advisory Techniques")
        lines.append("")
        lines.append(format_techniques_list(advisory))
        lines.append("")

    failure = techniques.get("failure", [])
    if failure:
        lines.append("## Failure Techniques")
        lines.append("")
        lines.append(format_techniques_list(failure))
        lines.append("")

    return "\n".join(lines)


def scrape_wcag(
    name: str,
    config: dict,
    output_base: Path
) -> dict:
    """
    Download WCAG JSON and generate markdown files.

    Args:
        name: Source name ("wcag")
        config: Static source config from config.yaml
        output_base: Base output directory (e.g., Path("docs/standards"))

    Returns:
        Dict matching scrape_source() return format:
        {
            "url": str,
            "commit": str (hash of JSON),
            "file_count": int,
            "files": {relative_path: content_hash}
        }
    """
    url = config["url"]
    data, json_hash = download_wcag_json(url)

    dest_dir = output_base / name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    file_hashes = {}
    file_count = 0

    for principle in data.get("principles", []):
        for guideline in principle.get("guidelines", []):
            for criterion in guideline.get("successcriteria", []):
                version = get_introduced_version(criterion)
                version_dir = dest_dir / version
                version_dir.mkdir(parents=True, exist_ok=True)

                filename = generate_criterion_filename(criterion)
                markdown = criterion_to_markdown(criterion, principle, guideline)

                file_path = version_dir / filename
                file_path.write_text(markdown, encoding="utf-8")

                # Relative path includes version subdirectory
                rel_path = f"{version}/{filename}"
                file_hashes[rel_path] = compute_file_hash(file_path)
                file_count += 1
                logger.debug(f"Generated: {rel_path}")

    logger.info(f"Generated {file_count} WCAG criterion files in {dest_dir}")

    return {
        "url": url,
        "commit": json_hash,
        "file_count": file_count,
        "files": file_hashes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scrape WCAG 2.2 success criteria from W3C JSON"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    wcag_config = config.get("static_sources", {}).get("wcag", {})
    if not wcag_config:
        logger.error("No wcag config found in static_sources")
        return 1

    output_base = Path(wcag_config.get("output_dir", "docs/standards"))
    result = scrape_wcag("wcag", wcag_config, output_base)

    print("\n--- WCAG Scrape Summary ---")
    print(f"  Files: {result['file_count']}")
    print(f"  JSON hash: {result['commit'][:12]}")

    # Show version breakdown
    version_counts = {}
    for rel_path in result["files"]:
        version = rel_path.split("/")[0]
        version_counts[version] = version_counts.get(version, 0) + 1
    for version in sorted(version_counts):
        print(f"  WCAG {version}: {version_counts[version]} criteria")

    return 0


if __name__ == "__main__":
    exit(main())
