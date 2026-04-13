import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_hourly_data


def parse_server_plan(plan_str: str) -> np.ndarray:
    return np.array([int(x) for x in plan_str.split(",")], dtype=int)


def main():
    df = pd.read_csv("nsga2_results.csv")
    hourly_df = load_hourly_data("data/hourly_input.csv")

    # 帕累托散点
    plt.figure(figsize=(7, 5))
    plt.scatter(df["total_cost"], df["avg_delay_hours"], alpha=0.7)
    plt.xlabel("Total Cost")
    plt.ylabel("Average Delay (hours)")
    plt.title("Pareto Front")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=200)
    plt.close()

    # 选一个最便宜解和一个折中解
    cheapest = df.sort_values("total_cost").iloc[0]
    balanced = df.iloc[len(df) // 2]

    cheapest_plan = parse_server_plan(cheapest["server_plan"])
    balanced_plan = parse_server_plan(balanced["server_plan"])

    hours = hourly_df["hour"].to_numpy()
    prices = hourly_df["price"].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.step(hours, cheapest_plan, where="mid", label="Cheapest Plan")
    plt.step(hours, balanced_plan, where="mid", label="Balanced Plan")
    plt.xlabel("Hour")
    plt.ylabel("Number of Servers")
    plt.title("Server Scheduling Plans")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("server_plan_compare.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(hours, prices, marker="o")
    plt.xlabel("Hour")
    plt.ylabel("Price")
    plt.title("Hourly Electricity Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hourly_price.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()