from __future__ import annotations

from pathlib import Path

from experiments.exp_token_export import run_token_export
from utils.io_utils import load_yaml_like


def main() -> None:
    # 脚本入口
    """作为脚本入口协调参数解析、数据处理和结果导出。"""
    fusion_path = Path("outputs/data/nbsdc_fusion/aligned_hourly_fusion.csv")
    legacy_fusion_path = Path("outputs/nbsdc_fusion/aligned_hourly_fusion.csv")
    if not fusion_path.exists() and legacy_fusion_path.exists():
        fusion_path = legacy_fusion_path
    if not fusion_path.exists():
        print(f"Missing fusion input: {fusion_path}")
        print("Please run: python run_nbsdc_fusion.py")
        raise SystemExit(1)

    config = load_yaml_like("config/token_export.yaml")
    outputs = run_token_export(fusion_path=fusion_path, config=config)
    print("Token export experiment finished.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
