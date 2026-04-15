from pathlib import Path
import yaml

SRC = Path("test/prithvi_sen1floods11/config.yaml")
DST = Path("test/prithvi_sen1floods11/config_local.yaml")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source config: {SRC}")

    with SRC.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["init_args"]["model_args"]["backbone_pretrained"] = False

    with DST.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote: {DST}")
    print("Set model.init_args.model_args.backbone_pretrained = False")


if __name__ == "__main__":
    main()
