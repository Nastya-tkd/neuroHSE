"""
Walks OpenNeuro's S3 object-version history for a given key prefix and saves,
for each key, the most recent version that still has real content (i.e. not
a delete marker). Used by scripts/download_real_labels.py - see that
script's docstring for why this is needed (ds004873's current listing is
raw-only, but earlier snapshots of the same CC0 dataset had a full
`derivatives/` tree that was later removed via delete markers, which S3
keeps the prior content behind rather than erasing).

Usage: python list_versions.py <s3-key-prefix> <output.json>
  e.g. python list_versions.py ds004873/derivatives/sub-p019/ versions_sub-p019.json
"""

import urllib.request
import urllib.parse
import re
import sys
import json

S3_BASE = "https://s3.amazonaws.com/openneuro.org/"


def list_all_versions(prefix):
    marker, vmarker = "", ""
    entries = []  # (key, version_id, is_latest, is_delete, last_modified)
    pages = 0
    while True:
        qs = {"versions": "", "prefix": prefix, "max-keys": "1000"}
        if marker:
            qs["key-marker"] = marker
            qs["version-id-marker"] = vmarker
        url = S3_BASE + "?" + urllib.parse.urlencode(qs)
        with urllib.request.urlopen(url) as r:
            data = r.read().decode()
        for m in re.finditer(r"<(Version|DeleteMarker)>(.*?)</\1>", data, re.S):
            kind, block = m.groups()
            key = re.search(r"<Key>([^<]+)</Key>", block).group(1)
            vid = re.search(r"<VersionId>([^<]+)</VersionId>", block).group(1)
            lm = re.search(r"<LastModified>([^<]+)</LastModified>", block).group(1)
            entries.append((key, vid, kind == "DeleteMarker", lm))
        pages += 1
        trunc = re.search(r"<IsTruncated>(true|false)</IsTruncated>", data)
        if trunc and trunc.group(1) == "true":
            nm = re.search(r"<NextKeyMarker>([^<]*)</NextKeyMarker>", data)
            nvm = re.search(r"<NextVersionIdMarker>([^<]*)</NextVersionIdMarker>", data)
            marker = nm.group(1) if nm else ""
            vmarker = nvm.group(1) if nvm else ""
            if not marker:
                break
        else:
            break
        if pages > 40:  # safety cap
            break
    return entries


def latest_real_version_per_key(entries):
    """S3 lists versions most-recent-first per key, so the first
    non-delete-marker entry seen for a key is the one to keep."""
    seen = {}
    for key, vid, is_delete, lm in entries:
        if key in seen or is_delete:
            continue
        seen[key] = (vid, lm)
    return seen


def get_or_build_version_map(prefix, cache_path):
    """Returns {key: (version_id, last_modified)}, using cache_path if it
    already exists, else building and saving it."""
    import os
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    latest = latest_real_version_per_key(list_all_versions(prefix))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(latest, f, indent=1)
    return latest


if __name__ == "__main__":
    prefix = sys.argv[1]
    out_path = sys.argv[2]
    latest = latest_real_version_per_key(list_all_versions(prefix))
    with open(out_path, "w") as f:
        json.dump(latest, f, indent=1)
    print(f"{prefix}: {len(latest)} unique keys with a real (non-deleted) version, saved to {out_path}")
