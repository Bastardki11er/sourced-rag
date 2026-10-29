"""Builds eval/prometheus/corpus from the Prometheus docs at a pinned commit, or checks the committed copy.

    python eval/prometheus/prepare_corpus.py           # check corpus/ against manifest.json (offline)
    python eval/prometheus/prepare_corpus.py --fetch   # download the sources at COMMIT, convert, write manifest

The only change made to each file is removing its YAML front matter (the `---` block with `title:` and
`sort_rank:`), which the website uses and which would otherwise be indexed as text. Every file already
starts its body with a `# Title` heading; code blocks, tables, headings and inline HTML are left as they are.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
REPO = "prometheus/prometheus"
TAG = "v2.53.0"
COMMIT = "4c35b9250afefede41c5f5acd76191f90f625898"
# Every docs/*.md page at COMMIT except the four index.md navigation stubs and configuration/configuration.md
# (a single 160 KB reference page that would be most of the corpus on its own). Chosen before any question.
FILES = [
    "command-line/prometheus.md",
    "command-line/promtool.md",
    "configuration/alerting_rules.md",
    "configuration/https.md",
    "configuration/recording_rules.md",
    "configuration/template_examples.md",
    "configuration/template_reference.md",
    "configuration/unit_testing_rules.md",
    "feature_flags.md",
    "federation.md",
    "getting_started.md",
    "http_sd.md",
    "installation.md",
    "management_api.md",
    "migration.md",
    "querying/api.md",
    "querying/basics.md",
    "querying/examples.md",
    "querying/functions.md",
    "querying/operators.md",
    "querying/remote_read_api.md",
    "stability.md",
    "storage.md",
]
LEGAL = ["LICENSE", "NOTICE"]
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_front_matter(text: str) -> str:
    """The page without its leading `---` YAML block; a page without one is returned unchanged."""
    return _FRONT_MATTER.sub("", text, count=1)


def structure(text: str) -> dict[str, int]:
    """Counts that must survive conversion: headings, code fence lines, table rows."""
    return {
        "headings": len(re.findall(r"^#{1,6} ", text, re.M)),
        "fence_lines": len(re.findall(r"^\s*(```|~~~)", text, re.M)),
        "table_rows": len(re.findall(r"^\s*\|", text, re.M)),
    }


def fetch(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def build() -> None:
    manifest = {"repository": f"https://github.com/{REPO}", "tag": TAG, "commit": COMMIT, "files": [], "legal": []}
    for name in LEGAL:
        data = fetch(name)
        (HERE / name).write_bytes(data)
        manifest["legal"].append({"path": name, "sha256": sha256(data)})
    for name in FILES:
        source = fetch(f"docs/{name}")
        converted = strip_front_matter(source.decode("utf-8"))
        if structure(converted) != structure(source.decode("utf-8")):
            raise SystemExit(f"{name}: conversion changed headings, code fences or tables")
        if not converted.lstrip().startswith("# "):
            raise SystemExit(f"{name}: no '# Title' heading after the front matter")
        out = HERE / "corpus" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(converted, encoding="utf-8")
        manifest["files"].append(
            {
                "path": f"docs/{name}",
                "source_url": f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/docs/{name}",
                "source_sha256": sha256(source),
                "converted": f"corpus/{name}",
                "converted_sha256": sha256(converted.encode("utf-8")),
            }
        )
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(FILES)} files at {TAG} ({COMMIT[:12]}) and manifest.json")


def check() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text())
    problems = [
        f["converted"]
        for f in manifest["files"]
        if sha256((HERE / f["converted"]).read_bytes()) != f["converted_sha256"]
    ]
    problems += [f["path"] for f in manifest["legal"] if sha256((HERE / f["path"]).read_bytes()) != f["sha256"]]
    extra = {str(p.relative_to(HERE)) for p in (HERE / "corpus").rglob("*") if p.is_file()} - {
        f["converted"] for f in manifest["files"]
    }
    for problem in problems + sorted(extra):
        print(f"does not match manifest.json: {problem}")
    if not problems and not extra:
        print(f"corpus matches manifest.json: {len(manifest['files'])} files from {manifest['tag']}")
    return 1 if problems or extra else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="download and convert (network); default: check")
    if parser.parse_args().fetch:
        build()
    else:
        sys.exit(check())
