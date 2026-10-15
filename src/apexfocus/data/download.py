"""Fetch an original-resolution FSOCO pilot using HTTP ZIP ranges."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import requests

URL = "http://fsoco.cs.uni-freiburg.de/datasets/fsoco_bounding_boxes_train.zip"
PILOT_GROUPS = ("amz", "ecurieaix", "frt", "ka", "mms", "tuwr")


class RemoteZip(io.RawIOBase):
    """Seekable archive reader; refuse servers that ignore byte ranges."""

    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.session.headers["Accept-Encoding"] = "identity"
        with self.session.head(url, timeout=30, allow_redirects=True) as r:
            r.raise_for_status()
            self.size = int(r.headers["Content-Length"])
            self.etag = r.headers.get("ETag")
        if not self.etag:
            raise ValueError("Archive must supply an ETag to detect changes during download")
        self.position = 0

    def close(self):
        self.session.close()
        super().close()

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        if base + offset < 0:
            raise ValueError("Negative archive offset")
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size - 1
        with self.session.get(self.url, headers={
            "Range": f"bytes={start}-{end}", "If-Match": self.etag,
        }, timeout=(10, 60), stream=True) as r:
            r.raise_for_status()
            if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("Server did not honor the requested byte range")
            data = r.raw.read(size + 1)
        if len(data) != size:
            raise ValueError("Truncated or oversized archive range")
        self.position += size
        return data


def rank(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def download(output, count=150, seed=42, min_width=1280, min_height=720, groups=PILOT_GROUPS):
    if count <= 0 or min_width <= 0 or min_height <= 0:
        raise ValueError("Count and minimum dimensions must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new directory")
    with RemoteZip(URL) as remote, zipfile.ZipFile(remote) as archive:
        names = set(archive.namelist())
        candidates = {}
        for name in sorted(names):
            parts = PurePosixPath(name).parts
            if len(parts) != 3 or parts[1] != "ann" or not name.endswith(".json"):
                continue
            if any(part in ("..", ".", "") for part in parts) or "\\" in name:
                raise ValueError(f"Unsafe archive member: {name}")
            image = f"{parts[0]}/img/{parts[2][:-5]}"
            if image not in names:
                raise ValueError(f"Missing image for {name}")
            candidates.setdefault(parts[0], []).append((name, image))
        for group in candidates:
            candidates[group].sort(key=lambda pair: rank(pair[0], seed), reverse=True)
        if not candidates:
            raise ValueError("No Supervisely team/ann/image.json pairs found")
        if not groups or len(set(groups)) != len(groups) or set(groups) - candidates.keys():
            raise ValueError("Requested source groups must be unique names present in the archive")
        candidates = {group: candidates[group] for group in groups}
        output.mkdir(parents=True)
        (output / "meta.json").write_bytes(archive.read("meta.json"))
        selected = []
        # Round-robin teams so the pilot is not dominated by a single recording source.
        while len(selected) < count and any(candidates.values()):
            for group in sorted(candidates):
                while candidates[group]:
                    annotation_name, image_name = candidates[group].pop()
                    annotation_bytes = archive.read(annotation_name)
                    annotation = json.loads(annotation_bytes)
                    size = annotation["size"]
                    if size["width"] < min_width or size["height"] < min_height:
                        continue
                    image_bytes = archive.read(image_name)  # ZipFile verifies member CRC.
                    for name, content in ((annotation_name, annotation_bytes), (image_name, image_bytes)):
                        path = output / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                    selected.append({
                        "image": image_name, "annotation": annotation_name,
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                    })
                    if len(selected) % 25 == 0:
                        print(f"Downloaded {len(selected)}/{count}", flush=True)
                    break
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise ValueError(f"Only {len(selected)} images met the requested dimensions")
        provenance = {
            "source": "FSOCO official public bounding-box training archive",
            "url": URL, "etag": remote.etag, "archive_bytes": remote.size,
            "seed": seed, "count": count, "min_width": min_width, "min_height": min_height,
            "groups": sorted(groups),
            "selection": "seeded SHA256 ordering within teams, round-robin across teams",
            "transport": "HTTP; CRC and local hashes detect corruption, not source authenticity",
            "files": selected,
        }
        (output / "download.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-width", type=int, default=1280)
    parser.add_argument("--min-height", type=int, default=720)
    parser.add_argument("--groups", nargs="+", default=PILOT_GROUPS)
    args = parser.parse_args()
    try:
        result = download(args.output, args.count, args.seed, args.min_width, args.min_height, args.groups)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Download failed: {exc}\nIncomplete output must not be used for preparation.\n")
    print(json.dumps({"output": str(args.output), "images": result["count"]}))


if __name__ == "__main__":
    main()
