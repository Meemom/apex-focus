"""Train YOLO11n from random initialization on audited train data; select on val."""

import argparse
import json
import os
from pathlib import Path

import yaml

from apexfocus.data.prepare import CLASSES, write_json
from apexfocus.detection import require_device
from apexfocus.environment import installed_versions
from apexfocus.experiment_data import file_hash, load_development


def train(config):
    # Keep optional integration settings local to this project, including on EC2.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    from ultralytics import YOLO, settings

    require_device(config["device"])
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {output}")
    root = Path(config["dataset"]).resolve()
    records, _, fingerprint = load_development(root)
    # Ultralytics label caches use a weaker size/path hash; rebuild from verified labels.
    for split in ("train", "val"):
        (root / "labels" / f"{split}.cache").unlink(missing_ok=True)
    output.mkdir(parents=True)
    # Regenerate absolute paths on the current machine, including after transfer to EC2.
    data = {"path": str(root), "train": "images/train", "val": "images/val", "names": dict(enumerate(CLASSES))}
    (output / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    provenance = {"status": "started", "initialization": "random_yolo11n",
                  "classes": list(CLASSES), "dataset_fingerprint": fingerprint,
                  "train_ids": [r["id"] for r in records["train"]],
                  "validation_ids": [r["id"] for r in records["val"]],
                  "selection": "Ultralytics best validation fitness (mAP50-95)",
                  "config": config, "packages": installed_versions()}
    write_json(output / "provenance.json", provenance)
    # No pretrained cone weights or optional telemetry integrations.
    settings.update({"wandb": False, "mlflow": False, "clearml": False, "comet": False,
                     "dvc": False, "sync": False, "raytune": False})
    model = YOLO("yolo11n.yaml")
    model.train(
        data=str((output / "dataset.yaml").resolve()), pretrained=False,
        device=config["device"], epochs=config["epochs"], imgsz=640,
        batch=config["batch"], seed=config["seed"], deterministic=True,
        workers=0, cache=False, amp=False, optimizer="AdamW", lr0=.001,
        patience=0, plots=False, val=True, save=True, verbose=False,
        project=str(output.resolve()), name="fit", exist_ok=False,
        fliplr=0., flipud=0., hsv_h=0.,  # Cone colour and left/right semantics matter.
    )
    weights = Path(model.trainer.best)
    if not weights.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    provenance.update(status="trained", weights=str(weights.resolve()), weight_sha256=file_hash(weights),
                      resolved_training_args=json.loads(json.dumps(vars(model.trainer.args), default=str)))
    write_json(output / "provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    result = train(config)
    print(json.dumps({k: result[k] for k in ("status", "weights", "weight_sha256")}, indent=2))


if __name__ == "__main__":
    main()
