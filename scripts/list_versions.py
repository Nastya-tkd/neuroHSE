"""
Проходит по истории версий объектов S3 OpenNeuro для заданного префикса
ключа и сохраняет для каждого ключа самую свежую версию, в которой ещё есть
реальное содержимое (т.е. не маркер удаления). Используется
scripts/download_real_labels.py - о том, зачем это нужно, см. докстринг
этого скрипта (текущий листинг ds004873 содержит только исходные данные,
но более ранние снимки того же датасета CC0 имели полное дерево
`derivatives/`, впоследствии удалённое через маркеры удаления, за которыми
S3 сохраняет прежнее содержимое, а не стирает его).

Использование: python list_versions.py <префикс-ключа-s3> <output.json>
  например: python list_versions.py ds004873/derivatives/sub-p019/ versions_sub-p019.json
"""

import urllib.request
import urllib.parse
import re
import sys
import json

S3_BASE = "https://s3.amazonaws.com/openneuro.org/"


def list_all_versions(prefix):
    marker, vmarker = "", ""
    entries = []  # (ключ, id_версии, последняя_ли, удалена_ли, дата_изменения)
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
        if pages > 40:  # защитный предел
            break
    return entries


def latest_real_version_per_key(entries):
    """S3 перечисляет версии по каждому ключу от самой свежей к самой
    старой, поэтому первая встреченная запись без маркера удаления для
    данного ключа - та, которую нужно оставить."""
    seen = {}
    for key, vid, is_delete, lm in entries:
        if key in seen or is_delete:
            continue
        seen[key] = (vid, lm)
    return seen


def get_or_build_version_map(prefix, cache_path):
    """Возвращает {key: (version_id, last_modified)}, используя cache_path,
    если он уже существует, иначе строит его и сохраняет."""
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
    print(f"{prefix}: {len(latest)} уникальных ключей с реальной (неудалённой) версией сохранено в {out_path}")
