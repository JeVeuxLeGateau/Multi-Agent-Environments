import time
import Environment_FTP as e
import pygame
import sys
import misc as m
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
import statistics


def toggle_eval(file_name, new_value1, new_value2):
    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        df_existing.loc[0, 'eval_true'] = new_value1
        df_existing.loc[0, 'render'] = new_value2
        df_existing.to_excel(file_name, index=False)

    else:
        print(f"File {file_name} does not exist.")


def run_sim(environment, time_stamps, episodes):
    initial_robots = environment.o["n_C"]
    initial_adversaries = environment.o["n_E"]

    environment.reset()

    time_log = 0

    survived = 0
    survived_list = []
    adversaries_defeated = 0
    defeated_list = []
    robots_unfrozen = 0
    time_steps = 0
    time_step_list = []
    timeouts = 0
    frozen_loss = 0
    ds_list = []

    offset_val = 0
    for i in range(episodes):
        last_info = None

        diameter = m.calc_diameter(environment.o)
        width = environment.o["width"]
        height = environment.o["height"]

        while diameter > width or diameter > height:
            environment.reset()
            diameter = m.calc_diameter(environment.o)

        dist = m.calc_ds(environment.o)
        ds_list.append(dist)

        for ts in range(time_stamps):
            # print("Timestep:", ts)
            actions = 1
            obs, reward, terminateds, truncateds, info = environment.step(actions)
            """
            if "TI" in info:
                TI_list.append(info["TI"].item())
            """
            last_info = info

            # print("REWARD", reward)

            if terminateds or truncateds:
                print("=================================")
                print("Episode:", i + 1)
                print("Final score:", reward)

                if info["val"] == 1:
                    print("Win!")
                    offset_val = info["offset"]

                else:
                    print("Loss!")
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            time_log += 1

            time.sleep(0.01)

        metric_values = m.process_info(last_info, initial_robots, initial_adversaries)
        # [termination_condition, adversaries_defeated, unfrozen_robots, time]

        if metric_values[0] == 1:
            survived += 1
            survived_list.append(1)
            time_steps += metric_values[3]
            print("ds", dist)
            print("time", (metric_values[3] * 10) + offset_val)
            print("Ratio:", ((metric_values[3] * 10) + offset_val) / dist)
            print()
        elif metric_values[0] == -1:
            timeouts += 1
            survived_list.append(0)
        elif metric_values[0] == -2:
            frozen_loss += 1
            survived_list.append(0)

        adversaries_defeated += metric_values[1]
        defeated_list.append(metric_values[1])
        robots_unfrozen += metric_values[2]
        time_step_list.append((metric_values[3] * 10) + offset_val)
        # TI_val += sum(TI_list)/len(TI_list)

        environment.reset()

    print()
    print("Robots", initial_robots)
    print("Adversaries:", initial_adversaries)
    print("Total Episodes:", episodes)
    print("Survival Rate:", survived / episodes)
    print("Timeout Rate:", timeouts / episodes)
    print("All Frozen Loss:", frozen_loss / episodes)
    print("Adversary Defeat Rate:", adversaries_defeated / episodes)
    print("Robot Unfreeze Rate:", robots_unfrozen / episodes)
    print("Average Time steps:", time_steps / episodes)

    X1 = np.array(defeated_list).reshape(-1, 1)
    y1 = np.array(survived_list)
    X2 = np.array(time_step_list).reshape(-1, 1)
    y2 = np.array(survived_list)

    B_0_a = "None"
    B_1_a = "None"
    B_0_t = "None"
    B_1_t = "None"

    if len(set(y1)) == 2:
        model = LogisticRegression()
        model.fit(X1, y1)
        B_0_a = model.intercept_[0]
        B_1_a = model.coef_[0][0]
        print("B_0_a:", B_0_a)
        print("B_1_a:", B_1_a)

    if len(set(y2)) == 2:
        model = LogisticRegression()
        model.fit(X2, y2)
        B_0_t = model.intercept_[0]
        B_1_t = model.coef_[0][0]
        print("B_0_t:", B_0_t)
        print("B_1_t:", B_1_t)

    MDS_ratios = []

    MDS = 0
    for i in range(len(time_step_list)):
        MDS += time_step_list[i] / ds_list[i]
        MDS_ratios.append(time_step_list[i] / ds_list[i])

    MDS = MDS / len(time_step_list)
    print("MDS:", MDS)
    print("median:", statistics.median(MDS_ratios))
    print("min_time:", min(MDS_ratios))
    print("max_time:", max(MDS_ratios))

    file_path = "DS_results_strat_1_fast.xlsx"
    results = {
        "Robots": initial_robots,
        "Adversaries": initial_adversaries,
        "Total Episodes": episodes,
        "Survival Rate": survived / episodes,
        "Timeout Rate": timeouts / episodes,
        "All Frozen Loss": frozen_loss / episodes,
        "Adversary Defeat Rate": adversaries_defeated / episodes,
        "Robot Unfreeze Rate": robots_unfrozen / episodes,
        "Average Time Steps": time_steps / episodes,
        "B_0_a": B_0_a,
        "B_1_a": B_1_a,
        "B_0_t": B_0_t,
        "B_1_t": B_1_t,
        "MDS": MDS,
        "median": statistics.median(MDS_ratios),
        "min_time": min(MDS_ratios),
        "max_time": max(MDS_ratios)
    }

    new_results_df = pd.DataFrame(results, index=[0]).transpose()
    if os.path.exists(file_path):
        # Load the existing Excel file
        existing_df = pd.read_excel(file_path, index_col=0)
        # Add new results as a new column
        new_column_name = f"Run {len(existing_df.columns) + 1}"
        existing_df[new_column_name] = new_results_df.iloc[:, 0]
        df = existing_df
    else:
        # Create a new DataFrame if the file does not exist
        new_results_df.columns = ["Run 1"]
        df = new_results_df

    df.to_excel(file_path)
    print("=================================")


def main():
    file_name = "eval_file.xlsx"
    toggle_eval(file_name, "True", "True")

    pygame.init()
    screen = pygame.display.set_mode([1000, 1000])

    configurations = [(2, 1), (20, 0), (50, 0)]

    time_steps = 15000
    episodes = 200

    for config in configurations:
        robots, adversaries = config
        overseer = e.Overseer(screen, robots, 0, adversaries, time_steps, 15, 10)

        print()
        run_sim(overseer, time_steps, episodes)


if __name__ == "__main__":
    main()
