from __future__ import annotations

from run_real_quick import run_real_scenario


def main() -> None:
    run_real_scenario(scenario="full", include_full_algorithms=False, stress=True)


if __name__ == "__main__":
    main()
