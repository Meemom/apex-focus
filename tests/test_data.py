import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from apexfocus.data import download as downloader
from apexfocus.data.prepare import load_records, prepare, validate_annotation, yolo_lines
from apexfocus.data.splits import assign_splits


def annotation():
    return {"size": {"width": 80, "height": 60}, "tags": [{"name": "train"}],
            "objects": [{"id": 123, "classTitle": "blue_cone", "geometryType": "rectangle",
                         "points": {"exterior": [[8, 12], [24, 36]], "interior": []},
                         "tags": [{"name": "truncated"}]}]}


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "source"
    files = []
    for group in range(6):
        for index in range(2):
            image = f"team{group}/img/frame{index}.png"
            ann = f"team{group}/ann/frame{index}.png.json"
            (root / image).parent.mkdir(parents=True, exist_ok=True)
            (root / ann).parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(group * 2 + index)
            cv2.imwrite(str(root / image), rng.integers(0, 256, (60, 80, 3), dtype=np.uint8))
            value = annotation()
            if index == 1:
                value["objects"] = []
            (root / ann).write_text(json.dumps(value))
            files.append({"image": image, "annotation": ann,
                          "image_sha256": hashlib.sha256((root / image).read_bytes()).hexdigest(),
                          "annotation_sha256": hashlib.sha256((root / ann).read_bytes()).hexdigest()})
    (root / "download.json").write_text(json.dumps({"count": len(files), "files": files, "source": "synthetic fixture"}))
    (root / "meta.json").write_text("{}")
    return {"source_root": str(root), "output": str(tmp_path / "prepared"), "mode": "pilot",
            "seed": 42, "validation_fraction": .2, "test_fraction": 0,
            "pilot_manifest": None, "near_duplicate_hamming": 0, "small_area_quantile": .25,
            "checkpoint_provenance": {"status": "not_provided"}}


def test_known_box_conversion_preserves_metadata():
    boxes = validate_annotation(annotation(), 80, 60)
    assert boxes[0]["source_object"]["tags"] == [{"name": "truncated"}]
    line = yolo_lines({"width": 80, "height": 60, "boxes": boxes})
    assert line == "0 0.2000000000 0.4000000000 0.2000000000 0.4000000000\n"
    assert yolo_lines({"width": 80, "height": 60, "boxes": []}) == ""


@pytest.mark.parametrize("points", [
    [[-1, 0], [10, 10]], [[0, 0], [81, 10]], [[8, 8], [8, 20]],
    [[10, 10], [0, 0]], [[0, 0], [float("nan"), 20]], [[0, 0], [True, 20]],
    [None, [10, 20]],
])
def test_invalid_boxes_are_rejected(points):
    ann = annotation()
    ann["objects"][0]["points"]["exterior"] = points
    with pytest.raises(ValueError):
        validate_annotation(ann, 80, 60)


def test_unknown_class_and_image_dimensions_rejected():
    ann = annotation()
    ann["objects"][0]["classTitle"] = "unmapped"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_annotation(ann, 80, 60)
    with pytest.raises(ValueError, match="dimensions"):
        validate_annotation(annotation(), 160, 120)


def test_preparation_is_deterministic_and_preserves_empty_images(dataset, tmp_path):
    audit = prepare(dataset)
    output = Path(dataset["output"])
    assert audit["splits"]["train"]["images"] == 10
    assert audit["splits"]["val"]["images"] == 2
    assert audit["splits"]["test"]["images"] == 0
    assert audit["splits"]["train"]["empty_images"] == 5
    assert len(list((output / "labels").rglob("*.txt"))) == 12
    assert len(list((output / "overlays").glob("*.jpg"))) == 6
    records = [json.loads(line) for line in (output / "canonical.jsonl").read_text().splitlines()]
    for r in records:
        copied = output / "images" / r["split"] / f"{r['id']}.png"
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == r["image_sha256"]
        assert r["source_annotation"]["tags"] == [{"name": "train"}]
    alternate = dict(dataset, output=str(tmp_path / "second"))
    prepare(alternate)
    assert (output / "manifests/all.jsonl").read_bytes() == (tmp_path / "second/manifests/all.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(dataset)


def test_changed_source_fails_before_output_is_created(dataset):
    root = Path(dataset["source_root"])
    (root / "team0/ann/frame0.png.json").write_text("{}")
    with pytest.raises(ValueError, match="Source changed"):
        prepare(dataset)
    assert not Path(dataset["output"]).exists()


def test_missing_image_and_path_escape_are_rejected(dataset):
    root = Path(dataset["source_root"])
    manifest = json.loads((root / "download.json").read_text())
    first_image = root / manifest["files"][0]["image"]
    first_image.unlink()
    with pytest.raises(FileNotFoundError):
        load_records(root)
    manifest["files"][0]["image"] = "../../outside.png"
    (root / "download.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_records(root)


def test_duplicate_groups_stay_together_and_pilot_groups_never_enter_test(dataset):
    records, _ = load_records(Path(dataset["source_root"]))
    # Duplicate the content identity across two different team identifiers.
    records[2]["image_sha256"] = records[0]["image_sha256"]
    split, duplicates = assign_splits(records, test_fraction=.2, excluded_test_groups=["team0"], near_distance=0)
    assert duplicates
    assert split["team0"] == split["team1"] != "test"
    assert "test" in split.values()
    reverse, _ = assign_splits(list(reversed(records)), test_fraction=.2,
                               excluded_test_groups=["team0"], near_distance=0)
    assert split == reverse
    with pytest.raises(ValueError, match="non-pilot"):
        assign_splits(records, test_fraction=.2, excluded_test_groups=list(split), near_distance=0)


def test_study_requires_pilot_exclusions_and_pilot_cannot_create_test(dataset):
    with pytest.raises(ValueError, match="prior pilot"):
        prepare(dict(dataset, mode="study", test_fraction=.2))
    with pytest.raises(ValueError, match="development data"):
        prepare(dict(dataset, test_fraction=.2))


def test_study_excludes_relabelled_pilot_content(dataset, tmp_path):
    records, _ = load_records(Path(dataset["source_root"]))
    baseline, _ = assign_splits(records, test_fraction=.2, near_distance=0)
    test_group = next(group for group, split in baseline.items() if split == "test")
    previous = dict(next(r for r in records if r["group"] == test_group), group="old-pilot-team")
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(json.dumps(previous) + "\n")
    audit = prepare(dict(dataset, mode="study", test_fraction=.2, pilot_manifest=str(pilot)))
    assert test_group not in audit["splits"]["test"]["groups"]
    assert audit["pilot_matches"]


def test_download_reads_only_selected_pairs_and_records_hashes(dataset, tmp_path, monkeypatch):
    root = Path(dataset["source_root"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file() and path.name != "download.json":
                archive.write(path, path.relative_to(root).as_posix())

    class LocalArchive(io.BytesIO):
        def __init__(self, url):
            super().__init__(buffer.getvalue())
            self.size = len(buffer.getvalue())
            self.etag = '"fixture"'

    monkeypatch.setattr(downloader, "RemoteZip", LocalArchive)
    destination = tmp_path / "downloaded"
    manifest = downloader.download(destination, count=4, min_width=80, min_height=60, groups=["team0", "team1"])
    assert len(manifest["files"]) == 4
    records, _ = load_records(destination)
    assert {r["group"] for r in records} == {"team0", "team1"}
    with pytest.raises(FileExistsError):
        downloader.download(destination)


def test_range_reader_refuses_full_archive_response(monkeypatch):
    class Response(io.BytesIO):
        headers = {"Content-Length": "1000", "ETag": '"fixture"'}
        status_code = 200

        def raise_for_status(self):
            pass

    class Session:
        headers = {}

        def head(self, *args, **kwargs):
            return Response()

        def get(self, *args, **kwargs):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(downloader.requests, "Session", Session)
    with downloader.RemoteZip("http://fixture.invalid") as remote:
        with pytest.raises(ValueError, match="byte range"):
            remote.read(10)
