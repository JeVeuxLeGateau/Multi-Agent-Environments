import math
from sklearn.preprocessing import StandardScaler
from vertex import *
import random
import pygame
import calc
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import misc as m
import create_pairings as cpair
import pandas as pd
import random

base_positions = {"p": ((50, 750), (50, 750)),
                  "d": ((50, 750), (50, 750)),
                  "e": ((50, 750), (50, 750))}

base_velocities = {"p": ((-5, 5), (-5, 5)),
                   "d": ((-5, 5), (-5, 5)),
                   "e": ((-5, 5), (-5, 5))}

# Colors
base_colors = {"p": (0, 0, 255),  # blue
               "d": (255, 0, 0),  # red
               "e": (255, 255, 0)}  # green

# Rewards for how "well" algorithm performs
rewards_dict = {
    "Destroy Destroyer": -100,
    "Destroy Evader": 100,
    "Wake Pursuer": 100,
    "closeness": 10,
    "timestep": 1,
    "victory": 200}

# Probabilities of winning
base_probabilities = {"pp": 1.0,
                      "pd": 0.50,
                      "pe": 0.50,
                      "dp": 1.0,
                      "dd": 0.0,
                      "de": 0.0,
                      "ep": 0.50,
                      "ed": 0.10,
                      "ee": 0.0}

# Attack radii
base_radii = {"p": 10,
              "d": 10,
              "e": 10}

# Angle in degrees they can move
base_maneuverability = {"p": 180,
                        "d": 180,
                        "e": 180}


def generate_agent(x, y, v_x, v_y, attack_radius, p1, p2, maneuverability, color, status, speed=5):
    agent_obj = Vertex_Dict()
    agent_obj.attributes["x"] = x
    agent_obj.attributes["y"] = y
    agent_obj.attributes["color"] = color
    agent_obj.attributes["v_x"] = v_x
    agent_obj.attributes["v_y"] = v_y
    agent_obj.attributes["r"] = attack_radius
    agent_obj.attributes["m"] = maneuverability
    agent_obj.attributes["c"] = speed
    agent_obj.attributes["p1"] = p1
    agent_obj.attributes["p2"] = p2
    agent_obj.attributes["status"] = status
    agent_obj.attributes["has_target"] = (None, 0)
    agent_obj.attributes["been_targeted"] = 0
    agent_obj.attributes["W"] = 1

    return agent_obj


def generate_random_agents(num_C, num_D, num_E, speed, positions_dict=None, velocities_dict=None,
                           radii_dict=None, prob_dict=None, maneuverability_dict=None, colors_dict=None):
    if positions_dict is None:
        positions_dict = base_positions

    if velocities_dict is None:
        velocities_dict = base_velocities

    if colors_dict is None:
        colors_dict = base_colors

    if maneuverability_dict is None:
        maneuverability_dict = base_maneuverability

    if radii_dict is None:
        radii_dict = base_radii

    if prob_dict is None:
        prob_dict = base_probabilities

    all_agents = {}

    for i in range(num_C):
        rand_x = random.uniform(positions_dict["p"][0][0], positions_dict["p"][0][1])
        rand_y = random.uniform(positions_dict["p"][1][0], positions_dict["p"][1][1])
        rand_vx = random.uniform(velocities_dict["p"][0][0], velocities_dict["p"][0][1])
        rand_vy = random.uniform(velocities_dict["p"][1][0], velocities_dict["p"][1][1])
        v_x, v_y = calc.individual_unit_velocity(rand_vx, rand_vy)

        if i == 0:
            all_agents["C" + str(i)] = generate_agent(rand_x, rand_y, v_x, v_y,
                                                      radii_dict["p"], prob_dict["pd"], prob_dict["pe"],
                                                      maneuverability_dict["p"], colors_dict["p"], 1, speed)
        else:
            all_agents["C" + str(i)] = generate_agent(rand_x, rand_y, 0, 0,
                                                      radii_dict["p"], prob_dict["pd"], prob_dict["pe"],
                                                      maneuverability_dict["p"], colors_dict["p"], 0, speed)
    for i in range(num_D):
        rand_x = random.uniform(positions_dict["d"][0][0], positions_dict["d"][0][1])
        rand_y = random.uniform(positions_dict["d"][1][0], positions_dict["d"][1][1])
        rand_vx = random.uniform(velocities_dict["d"][0][0], velocities_dict["d"][0][1])
        rand_vy = random.uniform(velocities_dict["d"][1][0], velocities_dict["d"][1][1])
        v_x, v_y = calc.individual_unit_velocity(rand_vx, rand_vy)

        all_agents["D" + str(i)] = generate_agent(rand_x, rand_y, v_x, v_y,
                                                  radii_dict["d"], prob_dict["dp"], prob_dict["de"],
                                                  maneuverability_dict["d"], colors_dict["d"], 1, speed)

    for i in range(num_E):
        rand_x = random.uniform(positions_dict["e"][0][0], positions_dict["e"][0][1])
        rand_y = random.uniform(positions_dict["e"][1][0], positions_dict["e"][1][1])
        rand_vx = random.uniform(velocities_dict["e"][0][0], velocities_dict["e"][0][1])
        rand_vy = random.uniform(velocities_dict["e"][1][0], velocities_dict["e"][1][1])
        v_x, v_y = calc.individual_unit_velocity(rand_vx, rand_vy)

        all_agents["E" + str(i)] = generate_agent(rand_x, rand_y, v_x, v_y,
                                                  radii_dict["e"], prob_dict["ep"], prob_dict["ed"],
                                                  maneuverability_dict["e"], colors_dict["e"], 1, speed)
    """

    all_agents["C0"] = generate_agent(500, 500, 1, 0,
                                      radii_dict["p"], prob_dict["pd"], prob_dict["pe"],
                                      maneuverability_dict["p"], colors_dict["p"], 1, speed)

    all_agents["C1"] = generate_agent(700, 700, 0, 0,
                                      radii_dict["p"], prob_dict["pd"], prob_dict["pe"],
                                      maneuverability_dict["p"], colors_dict["p"], 0, speed)

    all_agents["C2"] = generate_agent(700, 50, 0, 0,
                                      radii_dict["p"], prob_dict["pd"], prob_dict["pe"],
                                      maneuverability_dict["p"], colors_dict["p"], 0, speed)
    """
    return all_agents


def normalize_single_agent(obs, low, high, new_min=-3, new_max=3):
    """
    Normalize a single agent's observation.

    Parameters:
    obs (numpy array): The observation to normalize.
    low (numpy array): The lower bounds of the observation.
    high (numpy array): The upper bounds of the observation.
    new_min (float): Minimum of the target range (default: -3).
    new_max (float): Maximum of the target range (default: 3).

    Returns:
    numpy array: Normalized observation.
    """
    return new_min + (obs - low) * (new_max - new_min) / (high - low)


class Overseer(gym.Env):
    def __init__(self, screen, n_C, n_D, n_E, time_steps, size=5, speed=5, positions_dict=None,
                 velocities_dict=None, radii_dict=None, prob_dict=None, maneuverability_dict=None, colors_dict=None):

        super(Overseer, self).__init__()
        self.screen = screen

        self.time = None
        self.all_agents = generate_random_agents(n_C, n_D, n_E, speed, positions_dict, velocities_dict,
                                                 radii_dict, prob_dict, maneuverability_dict, colors_dict)
        self.n_C = n_C
        self.n_D = n_D
        self.n_E = n_E
        self.time_steps = time_steps
        self.size = size
        self.speed = speed
        self.positions_dict = positions_dict
        self.velocities_dict = velocities_dict
        self.radii_dict = radii_dict
        self.prob_dict = prob_dict
        self.maneuverability_dict = maneuverability_dict
        self.colors_dict = colors_dict
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.current_CDE = self.number_of_CDE()
        self.reward_sum = 0
        self.offset = 0
        self.counter = 0
        self.counter2 = 0
        self.render_status = True
        self.dist = None

        self.action_space = spaces.Box(low=np.float32(-1), high=np.float32(1), shape=(2 * n_C,))

        self.total_entities = n_C + n_D + n_E

        """
        # Observation space for a single agent
        agent_obs_space = spaces.Dict({
            "position": spaces.Box(low=np.array([0, 0]), high=np.array([1000, 1000]), dtype=np.float32),  # x, y
            "velocity": spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32),  # vx, vy
            "status": spaces.Discrete(2)  # awake-status is either 0 or 1
        })

        obs_dict = {}
        for agent in self.all_agents:
            obs_dict[agent] = agent_obs_space

        self.observation_space = spaces.Dict(obs_dict)
        """
        """
        agent_obs_space = spaces.Dict({
            "position": spaces.Box(low=np.array([0, 0]), high=np.array([1000, 1000]), dtype=np.float32),  # x, y
            "velocity": spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32),  # vx, vy
            "status": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # awake-status is either 0 or 1
        })

        # 2. Define the overall observation space as a Tuple of agent spaces
        self.observation_space = spaces.Tuple([agent_obs_space] * len(self.all_agents))
        """
        # Define single agent's limits
        # low_single_agent = np.array([0, 0, -1, -1, 0], dtype=np.float32)
        # high_single_agent = np.array([1000, 1000, 1, 1, 1], dtype=np.float32)

        width = 3
        low_single_agent = np.array([-width, -width, -width, -width, -width], dtype=np.float32)
        high_single_agent = np.array([width, width, width, width, width], dtype=np.float32)

        # Use np.tile to repeat this pattern for multiple agents
        low_multi_agent = np.tile(low_single_agent, self.total_entities).astype(np.float32)
        high_multi_agent = np.tile(high_single_agent, self.total_entities).astype(np.float32)

        # Define the Box space for multiple agents
        self.observation_space = spaces.Box(low=low_multi_agent, high=high_multi_agent, dtype=np.float32)

        self.observations = self.gather_observations()

        self.spec = gym.envs.registration.EnvSpec(id="CustomEnv-v0")
        self.spec.max_episode_steps = time_steps
        print("End of initialization")

    def get_state(self):
        state_dict = {
            "pursuers": {},
            "destroyers": {},
            "evaders": {}
        }

        for agent in self.all_agents:
            a = self.all_agents[agent].attributes
            if agent[0] == "C":
                state_dict["pursuers"][agent] = [a["x"], a["y"], a["v_x"], a["v_y"]]
            elif agent[0] == "D":
                state_dict["destroyers"][agent] = [a["x"], a["y"], a["v_x"], a["v_y"]]
            else:
                state_dict["evaders"][agent] = [a["x"], a["y"], a["v_x"], a["v_y"]]

        return state_dict

    def print_robots(self):
        for agent in self.all_agents:
            a = self.all_agents[agent].attributes
            print("\t" + agent + ":")
            print("\t\tx:", a["x"])
            print("\t\ty:", a["y"])
            print("\t\tvx:", a["v_x"])
            print("\t\tvy:", a["v_y"])
            print("\t\tstatus:", a["status"])

    def number_of_CDE(self):
        num_C = 0
        num_D = 0
        num_E = 0

        for agent in self.all_agents:
            if agent[0] == "C" and self.all_agents[agent].attributes["status"] == 1:
                num_C += 1
            elif agent[0] == "D" and self.all_agents[agent].attributes["status"] == 1:
                num_D += 1
            elif agent[0] == "E" and self.all_agents[agent].attributes["status"] == 1:
                num_E += 1

        return [num_C, num_D, num_E]

    def get_critic_values(self):
        model_config = {
            "fcnet_hiddens": [64, 64],  # Hidden layers for the critic network
            "fcnet_activation": "relu",  # Activation function
            "agents": self.n_C + self.n_D + self.n_E
            # Add any other custom model-specific parameters here
        }

        return self.observation_space, self.action_space, self.n_C, model_config, "Critic_name"

    def check_done(self):
        """
        -2 - Loss by done
        -1 - Loss by time
        0 - Not Done
        1 - Success
        :return:
        """

        if self.time == self.time_steps:
            return -1
        elif self.number_of_CDE()[0] == 0:
            return -2
        elif self.number_of_CDE()[0] == self.n_C:
            return 1
        else:
            return 0

    def get_alive(self):
        alive_agents = {}
        for agent in self.all_agents:
            if self.all_agents[agent].attributes["status"] == 1:
                alive_agents[agent] = self.all_agents[agent]
        return alive_agents

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.offset = 0
        self.reward_sum = 0
        self.all_agents = generate_random_agents(self.n_C, self.n_D, self.n_E, self.speed,
                                                 self.positions_dict, self.velocities_dict,
                                                 self.radii_dict, self.prob_dict,
                                                 self.maneuverability_dict, self.colors_dict)

        self.all_agents["C0"].attributes["status"] = 0

        file_name = "eval_file.xlsx"
        eval_data = pd.read_excel(file_name)
        evaluation_status = eval_data.loc[0, 'eval_true']
        self.render_status = eval_data.loc[0, 'render']

        diameter = m.calc_diameter_gym(self.all_agents)
        width = self.width
        height = self.height

        while diameter > width or diameter > height:
            self.all_agents = generate_random_agents(self.n_C, self.n_D, self.n_E, self.speed,
                                                     self.positions_dict, self.velocities_dict,
                                                     self.radii_dict, self.prob_dict,
                                                     self.maneuverability_dict, self.colors_dict)
            diameter = m.calc_diameter_gym(self.all_agents)

        self.dist = m.calc_ds_gym(self.all_agents)

        """
        if evaluation_status:
            self.all_agents["C0"].attributes["status"] = 1
        else:
            if self.counter % number_of_scenarios == 0:
                self.all_agents["E0"].attributes["status"] = 0

        self.all_agents["C0"].attributes["status"] = 1
        """

        number_of_scenarios = 8

        if evaluation_status:
            self.all_agents["C0"].attributes["status"] = 1
        else:
            if self.counter % number_of_scenarios == 0:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C1"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 1:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C2"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 2:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C3"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 3:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C1"].attributes["status"] = 1
                self.all_agents["C2"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 4:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C2"].attributes["status"] = 1
                self.all_agents["C3"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 5:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C1"].attributes["status"] = 1
                self.all_agents["C3"].attributes["status"] = 1
            else:
                self.all_agents["C0"].attributes["status"] = 1
            """
            if self.counter % number_of_scenarios == 0:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C1"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 1:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C2"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 2:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C3"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 3:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C1"].attributes["status"] = 1
                self.all_agents["C2"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 4:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C1"].attributes["status"] = 1
                self.all_agents["C3"].attributes["status"] = 1
            elif self.counter % number_of_scenarios == 5:
                self.all_agents["C0"].attributes["status"] = 1
                self.all_agents["C2"].attributes["status"] = 1
                self.all_agents["C3"].attributes["status"] = 1
            else:
                self.all_agents["C0"].attributes["status"] = 1
            """

        self.counter += 1

        """
        number_of_scenarios = 5

        available_agents = []

        for i in range(1, self.n_C):
            available_agents.append("C" + str(i))

        sample = []
        if evaluation_status:
            self.all_agents["C0"].attributes["status"] = 1
        
        else:
            if self.counter % number_of_scenarios == 0:
                sample = random.sample(available_agents, 1)
            if self.counter % number_of_scenarios == 1:
                sample = random.sample(available_agents, 2)
            if self.counter % number_of_scenarios == 2:
                sample = random.sample(available_agents, 3)

        self.all_agents["C0"].attributes["status"] = 1
        for agent in sample:
            self.all_agents[agent].attributes["status"] = 1
        
        
        self.counter += 1
        if self.counter % number_of_scenarios == 0:
            self.counter2 += 1
        """

        if self.render_status:
            self.render()
        self.current_CDE = self.number_of_CDE()
        observations = self.gather_observations()
        info = {}
        return observations, info

    def update_positions(self):
        for agent in self.get_alive():
            self.all_agents[agent].attributes["x"] += self.all_agents[agent].attributes["v_x"] * \
                                                      self.all_agents[agent].attributes["c"]
            self.all_agents[agent].attributes["y"] += self.all_agents[agent].attributes["v_y"] * \
                                                      self.all_agents[agent].attributes["c"]

            left = 25
            right = self.width - left
            down = 25
            up = self.height - down

            if right < self.all_agents[agent].attributes["x"]:
                self.all_agents[agent].attributes["x"] = right
            elif self.all_agents[agent].attributes["x"] < left:
                self.all_agents[agent].attributes["x"] = left

            if up < self.all_agents[agent].attributes["y"]:
                self.all_agents[agent].attributes["y"] = up
            elif self.all_agents[agent].attributes["y"] < down:
                self.all_agents[agent].attributes["y"] = down

            if agent[0] == "C":
                colors = base_colors["p"]
            elif agent[0] == "D":
                colors = base_colors["d"]
            else:
                colors = base_colors["e"]

            self.all_agents[agent].attributes["color"] = colors

    def update_velocities(self):
        for agent in self.all_agents:
            if self.all_agents[agent].attributes["status"] == 0:
                self.all_agents[agent].attributes["v_x"] = 0
                self.all_agents[agent].attributes["v_y"] = 0

    def update_alive_agents(self):
        dead_agents = []
        alive_agents = []

        for agent1 in self.all_agents:
            for agent2 in self.all_agents:
                a1 = self.all_agents[agent1].attributes
                a2 = self.all_agents[agent2].attributes
                if a1 is not a2:
                    # effective_dist = calc.euclidean_distance(a1["x"], a1["y"], a2["x"], a2["y"]) - (a1["r"] + a2["r"])
                    d1 = calc.euclidean_distance(a1["x"], a1["y"], a2["x"], a2["y"])
                    effective_dist = d1 - a1["r"]

                    r = random.uniform(0, 1)
                    if agent1[0] == "C" and agent2[0] == "C" and a1["status"] == 1 and a2["status"] == 0:
                        if effective_dist <= 0 and r <= a1["p1"]:
                            self.offset += d1 * self.speed
                            alive_agents.append(self.all_agents[agent2])
                            for agent in self.all_agents:
                                if agent[0] == "C" and self.all_agents[agent].attributes["has_target"] == agent2:
                                    self.all_agents[agent].attributes["has_target"] = (None, 0)
                    elif agent1[0] == "C" and agent2[0] == "E" and a1["status"] == 1 and a2["status"] == 1:
                        if effective_dist <= 0 and r <= a1["p2"]:
                            dead_agents.append(self.all_agents[agent2])
                    elif agent1[0] == "E" and agent2[0] == "C" and a1["status"] == 1 and a2["status"] == 1:
                        if effective_dist <= 0 and r <= a1["p1"]:
                            dead_agents.append(self.all_agents[agent2])
                            for agent in self.all_agents:
                                if agent[0] == "E" and self.all_agents[agent].attributes["has_target"] == agent2:
                                    self.all_agents[agent].attributes["has_target"] = (None, 0)

        for agent in dead_agents:
            agent.attributes["status"] = 0
            agent.attributes["v_x"] = 0
            agent.attributes["v_y"] = 0
            agent.attributes["has_target"] = (None, 0)
            agent.attributes["been_targeted"] = 0

        for agent in alive_agents:
            agent.attributes["status"] = 1
            agent.attributes["has_target"] = (None, 0)
            agent.attributes["been_targeted"] = 0

    """
    def gather_observations(self):
        observation = {}

        for agent in self.all_agents:
            a = self.all_agents[agent].attributes
            observation[agent] = {}
            observation[agent]["position"] = np.array([a["x"], a["y"]])
            observation[agent]["velocity"] = np.array([a["v_x"], a["v_y"]])
            observation[agent]["status"] = a["status"]
            # observation[agent]["W"] = a["W"]
            # observation[agent]["r"] = a["r"]
            # observation[agent]["c"] = a["c"]
            # observation[agent]["p1"] = a["p1"]
            # observation[agent]["p2"] = a["p2"]

        return observation
    """

    """
    def gather_observations(self):
        # Create an empty list to store agent observations
        observations = []

        # Iterate over the agents and collect their observations
        for agent in self.all_agents:
            a = self.all_agents[agent].attributes
            agent_obs = {
                "position": np.array([a["x"], a["y"]]),
                "velocity": np.array([a["v_x"], a["v_y"]]),
                "status": a["status"]
            }
            observations.append(agent_obs)  # Add this agent's observation to the list

        return tuple(observations)  # Return the observations as a tuple
    """

    def gather_observations(self):
        # Create an empty list to store agent observations
        observations = []

        # Iterate over the agents and collect their observations
        for agent in self.all_agents:
            a = self.all_agents[agent].attributes
            agent_obs = np.array([a["x"], a["y"], a["v_x"], a["v_y"], a["status"]])
            observations.append(agent_obs)  # Add this agent's observation to the list

        concatenated_obs = np.concatenate(observations).astype(np.float32)
        low = np.array([0, 0, -1, -1, 0], dtype=np.float32)
        high = np.array([1000, 1000, 1, 1, 1], dtype=np.float32)

        new_low = np.tile(low, self.total_entities).astype(np.float32)
        new_high = np.tile(high, self.total_entities).astype(np.float32)
        scaled_data = normalize_single_agent(concatenated_obs, new_low, new_high, new_min=-3, new_max=3)
        return scaled_data

    def calculate_rewards(self):
        """
        rewards = {
            "Destroy Destroyer": -100,
            "Destroy Evader": 100,
            "Wake Pursuer": 100,
            "closeness": 10,
            "timestep": 1,
            "victory": 200
        }
        """
        rewards = {
            "Destroy Destroyer": -100,
            "Destroy Evader": 100,
            "Wake Pursuer": 100,
            "closeness": 10,
            "timestep": 1,
            "victory": 100 * self.n_C
        }

        reward_calc = 0

        old = self.current_CDE
        new = self.number_of_CDE()

        del_P = new[0] - old[0]
        del_E = old[2] - new[2]

        # Reward for defeating Adversary
        r_del_E = del_E * rewards["Destroy Evader"]

        # Reward for unfreezing robot
        r_del_P = del_P * rewards["Wake Pursuer"]

        reward_calc += r_del_P + r_del_E

        for r in self.all_agents:
            if self.all_agents[r].attributes["status"] == 1 and r[0] == "C":
                # Reward for getting closer to frozen agent
                r_close = m.get_closest_robot_reward_non_tensor(self.all_agents, self.width, self.height, r)
                r_close_sqrt = math.sqrt(r_close)

                # Penalty for getting closer to adversary
                r_adv = m.get_closest_adv_reward_non_tensor(self.all_agents, self.width, self.height, r)

                r_heading = m.get_heading(self.all_agents, r)
                reward_calc += r_close - r_adv + r_heading

        self.current_CDE = self.number_of_CDE()

        done = self.check_done()
        if done == 1:
            reward_calc += rewards["victory"]
        elif done < 0:
            reward_calc += -rewards["victory"]
        return reward_calc

    def step(self, actions):
        self.time += 1
        robot_keys = []
        for agent in self.all_agents:
            if agent[0] == "C":
                robot_keys.append(agent)
        for i in range(len(robot_keys)):
            x = 2 * i
            y = 2 * i + 1
            if self.all_agents[robot_keys[i]].attributes["status"] == 1:
                vx, vy = calc.individual_unit_velocity(actions[x], actions[y])
                self.all_agents[robot_keys[i]].attributes["v_x"] = vx
                self.all_agents[robot_keys[i]].attributes["v_y"] = vy

        # print("Print in step")
        # self.print_robots()
        # cpair.strategy2(self.all_agents)
        v = cpair.adversary_test_policy_gym(self.all_agents, self.speed, self.time, self.time_steps)
        for adv in v:
            vx, vy = v[adv]
            self.all_agents[adv].attributes["v_x"] = vx
            self.all_agents[adv].attributes["v_y"] = vy

        self.update_velocities()
        self.update_positions()

        self.update_alive_agents()
        if self.render_status:
            self.render()
        self.observations = self.gather_observations()
        """
        done = self.check_done()

        if done == 1 or done == -1:
            return self.calculate_rewards(), done, pairings
        else:
            return 0, done, pairings
        """

        terminateds = self.check_done()

        """
        check_done()
        -2 - Loss by done
        -1 - Loss by time
        0 - Not Done
        1 - Success
        """

        reward = self.calculate_rewards()
        self.reward_sum += reward
        final_numbers = self.number_of_CDE()

        if terminateds == 1:
            return self.observations, reward, True, False, {"val": 1,
                                                            "n_C": final_numbers[0],
                                                            "n_D": final_numbers[1],
                                                            "n_E": final_numbers[2],
                                                            "current_time": self.time,
                                                            "offset": self.offset,
                                                            "dist": self.dist}
        elif terminateds == -2:
            return self.observations, reward, True, False, {"val": -2,
                                                            "n_C": final_numbers[0],
                                                            "n_D": final_numbers[1],
                                                            "n_E": final_numbers[2],
                                                            "current_time": self.time,
                                                            "offset": self.offset,
                                                            "dist": self.dist}
        elif terminateds == -1:
            return self.observations, reward, False, True, {"val": -1,
                                                            "n_C": final_numbers[0],
                                                            "n_D": final_numbers[1],
                                                            "n_E": final_numbers[2],
                                                            "current_time": self.time,
                                                            "offset": self.offset,
                                                            "dist": self.dist}
        else:
            return self.observations, reward, False, False, {"val": 0,
                                                             "n_C": final_numbers[0],
                                                             "n_D": final_numbers[1],
                                                             "n_E": final_numbers[2],
                                                             "current_time": self.time,
                                                             "offset": self.offset,
                                                             "dist": self.dist}
        # returns [obs], [rewards], [terminated], [truncated], [infos]

    def render(self):
        self.screen.fill((255, 255, 255))

        # radius of influence
        """
        for agent in self.all_agents:
            if self.all_agents[agent].attributes["status"] == 1:
                pygame.draw.circle(self.screen, (1, 0, 1),
                                   (self.all_agents[agent].attributes["x"],
                                    self.all_agents[agent].attributes["y"]),
                                   self.all_agents[agent].attributes["r"])
        """

        for agent in self.all_agents:
            a = self.all_agents[agent].attributes

            if self.all_agents[agent].attributes["status"] == 1:
                pygame.draw.circle(self.screen, self.all_agents[agent].attributes["color"],
                                   (self.all_agents[agent].attributes["x"],
                                    self.all_agents[agent].attributes["y"]), self.size)
                """
                pygame.draw.circle(self.screen, (0, 0, 0),
                                   (np.random.randint(0, 100), np.random.randint(0, 100)), self.size)
                                   """
            elif self.all_agents[agent].attributes["status"] == 0 and agent[0] == "C":
                pygame.draw.circle(self.screen, (0, 255, 255),
                                   (self.all_agents[agent].attributes["x"],
                                    self.all_agents[agent].attributes["y"]), self.size)
                """
                pygame.draw.circle(self.screen, (0, 0, 0),
                                   (np.random.randint(0, 100), np.random.randint(0, 100)), self.size)
                """
        pygame.display.flip()
