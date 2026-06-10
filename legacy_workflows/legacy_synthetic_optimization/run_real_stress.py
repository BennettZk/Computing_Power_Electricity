from __future__ import annotations

from run_real_quick import run_real_scenario


def main() -> None:
    """作为脚本入口协调参数解析、数据处理和结果导出。"""
    run_real_scenario(scenario="full", include_full_algorithms=False, stress=True)


if __name__ == "__main__":
    main()
