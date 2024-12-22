import calc
# import random
import Skills as s
import misc as m
import calc as c
import torch
# import Define_GNN as gnn
import copy
import define_TGN as tgn
from ray.rllib.algorithms.ppo import PPOConfig, PPO
import numpy as np
from CustomCallback import CustomCallback
import gym_FTP as e
import pygame
from ray.tune.registry import register_env

"""
def create_pairings(pairs, self_dict):
    pairings = pairs
    pairings2 = []
    pursuers = []

    # dict of agents in 200 radius
    def agents_in_radius(agent_current):
        cur = self_dict["current_CDE"][0] + self_dict["current_CDE"][1] + self_dict["current_CDE"][2]
        total = self_dict["n_C"] + self_dict["n_D"] + self_dict["n_E"]
        agent_dict = {agent_current: self_dict["all_agents"][agent_current]}
        scale = 1 - (cur / (total + 1))
        r = 500 * scale
        for agent_test in self_dict["all_agents"]:
            if (self_dict["all_agents"][agent_test].attributes["status"] == 1 and
                    calc.agent_d(self_dict["all_agents"][agent_current], self_dict["all_agents"][agent_test]) < r):
                agent_dict[agent_test] = self_dict["all_agents"][agent_test]

        return agent_dict

    # [Num_C, Num_D, Num_E]
    def get_numbers(agent_dict):
        numbers = [0, 0, 0]

        for key in agent_dict:
            if key[0] == "C":
                numbers[0] += 1
            elif key[0] == "D":
                numbers[1] += 1
            else:
                numbers[2] += 1

        return numbers

    def equal(a, b_azure):
        for i in range(len(a)):
            if a[i] != b_azure[i]:
                return False

        return True

    # get closet target
    def get_target(current_ag):
        def get_key_from_value(dictionary, target_value):
            for key, value in dictionary.items():
                if value == target_value:
                    return key

        un_targeted = []

        for ag in alive_agents:
            if (self_dict["all_agents"][ag].attributes["been_targeted"] == 0 and self_dict["all_agents"][ag]
                    is not current_ag and ag[0] == "E"):
                un_targeted.append(self_dict["all_agents"][ag])

        for i in range(len(un_targeted)):
            un_targeted[i] = (un_targeted[i], calc.agent_d(current_ag, un_targeted[i]))

        sorted_un_targeted = sorted(un_targeted, key=lambda x: x[1])
        # print("sorted", sorted_un_targeted)

        val = get_key_from_value(alive_agents, sorted_un_targeted[0][0])
        self_dict["all_agents"][val].attributes["been_targeted"] = 1
        return val

    # If every Evader has been targeted, allow them to be retargeted
    def reset_targets():
        for all_agent in alive_agents:
            if self_dict["all_agents"][all_agent].attributes["been_targeted"] == 0 and all_agent[0] == "E":
                return

        for all_agent in alive_agents:
            if all_agent[0] == "E":
                self_dict["all_agents"][all_agent].attributes["been_targeted"] = 0

    # reset all pursuer's targets
    def reset_has_target():
        for all_agent in alive_agents:
            if all_agent[0] == "E":
                self_dict["all_agents"][all_agent].attributes["has_target"] = (None, 0)

    # Get alive agents
    alive_agents = m.get_alive(self_dict)

    # populate pursuers
    for agent in alive_agents:
        if agent[0] == "C":
            pursuers.append(alive_agents[agent])

    # print(self.current_CDE, self.number_of_CDE(), len(pairings) != 0)
    # print(equal(self.current_CDE, self.number_of_CDE()))
    if equal(self_dict["current_CDE"], m.number_of_CDE(self_dict)) and len(pairings) != 0:
        # print("Using previous pairings")
        return pairings
    else:
        self_dict["current_CDE"] = m.number_of_CDE(self_dict)
        pairings = []
        for agent1 in alive_agents:
            reset_targets()
            if agent1[0] == "C" and alive_agents[agent1] in pursuers:
                agents = agents_in_radius(agent1)

                # check for distract
                b = get_numbers(agents)
                if len(agents) >= 4 and b[0] >= 2 and b[1] >= 1 and b[2] >= 1:
                    if random.uniform(0, 1) < 0.2:
                        pursuers_list = []
                        destroyers_list = []
                        evaders_list = []

                        for robot in agents:
                            if robot[0] == "C":
                                pursuers_list.append(self_dict["all_agents"][robot])
                            elif robot[0] == "D":
                                destroyers_list.append(self_dict["all_agents"][robot])
                            elif robot[0] == "E":
                                evaders_list.append(self_dict["all_agents"][robot])

                        random_pursuers = random.sample(range(len(pursuers_list)), 2)
                        random_destroyers = random.sample(range(len(destroyers_list)), 1)
                        random_evaders = random.sample(range(len(evaders_list)), 1)

                        p_counter = [pursuers_list[random_pursuers[0]], pursuers_list[random_pursuers[1]]]
                        d = destroyers_list[random_destroyers[0]]
                        e = evaders_list[random_evaders[0]]

                        pairings.append([p_counter, [d, e], s.distract])
                        pairings2.append([p_counter, ["d_str", "e_str"], "distract"])
                        e.attributes["been_targeted"] = 1
                        pursuers = [x for x in pursuers if x not in p_counter]
                    else:
                        target = get_target(self_dict["all_agents"][agent1])

                        destroyers = []
                        for d in self_dict["all_agents"]:
                            if d[0] == "D":
                                destroyers.append(self_dict["all_agents"][d])

                        pairings.append(
                            [self_dict["all_agents"][agent1], self_dict["all_agents"][target], destroyers, s.pursue2])
                        pairings2.append([agent1, target, "pursue2"])
                        pursuers.remove(self_dict["all_agents"][agent1])
                        self_dict["all_agents"][agent1].attributes["has_target"] = (target, 1)

                else:
                    target = get_target(self_dict["all_agents"][agent1])

                    destroyers = []
                    for d in self_dict["all_agents"]:
                        if d[0] == "D":
                            destroyers.append(self_dict["all_agents"][d])

                    pairings.append([self_dict["all_agents"][agent1], self_dict["all_agents"][target], s.MatthewAlgo])
                    pairings2.append([agent1, target, "p2"])
                    pursuers.remove(self_dict["all_agents"][agent1])
                    self_dict["all_agents"][agent1].attributes["has_target"] = (target, 1)

            elif agent1[0] == "D":
                for agent2 in alive_agents:
                    if agent2[0] == "C":
                        pairings.append([alive_agents[agent1], alive_agents[agent2], s.MatthewAlgo])
                        pairings2.append([agent1, agent2, "MattAlgo"])
                        break
            else:
                caps = []
                for agent in alive_agents:
                    if agent[0] == "C" and agent:
                        caps.append(alive_agents[agent])
                pairings.append([alive_agents[agent1], caps, s.evade_greedy])
                pairings2.append([agent1, caps, "evade"])

        # print("new pairings", pairings2)

        update_velocities_with_pairings(pairings, self_dict)


def update_velocities_with_pairings(pairings, self_dict):
    new_velocities = {}

    if pairings is not None:
        for pair in pairings:
            # agent1 = self.all_agents[pair[0]]
            # agent2 = self.all_agents[pair[1]]
            if len(pair) == 3:
                agent1 = pair[0]
                agent2 = pair[1]
                algorithm = pair[2]

            elif len(pair) == 4:
                pursuer = pair[0]
                target = pair[1]
                destroyers = pair[2]
                algorithm = pair[3]

            if algorithm == s.MatthewAlgo or algorithm == 'pursue':
                v_x_M, v_y_M = s.MatthewAlgo(agent1, agent2, self_dict["time"], self_dict["time_steps"])

               
                agent1.attributes["v_x"] = v_x_M
                agent1.attributes["v_y"] = v_y_M

            elif algorithm == s.evade_greedy or algorithm == 'distract':
                v_x_M, v_y_M = algorithm(agent1, agent2, 8, 100, self_dict["width"], self_dict["height"])
                
                agent1.attributes["v_x"] = v_x_M
                agent1.attributes["v_y"] = v_y_M

            elif algorithm == s.distract:
                pursuers = agent1
                destroyer = agent2[0]
                evader = agent2[1]
                t0 = self_dict["time"]
                tf = self_dict["time_steps"]
                n = 8
                threshold = 100
                width = self_dict["width"]
                height = self_dict["height"]

                vx1, vy1, vx2, vy2 = algorithm(pursuers, destroyer, evader, t0, tf, n, threshold, width, height)

                pursuers[0].attributes["v_x"] = vx1
                pursuers[0].attributes["v_y"] = vy1
                pursuers[1].attributes["v_x"] = vx2
                pursuers[1].attributes["v_y"] = vy2

            elif algorithm == s.destroy_target_attacker or algorithm == "destroy_target_attacker":
                v_x_M, v_y_M = s.destroy_target_attacker(agent1, agent2)
                agent1.attributes["v_x"] = v_x_M
                agent1.attributes["v_y"] = v_y_M

            elif algorithm == s.pursue2 or algorithm == "pursue2":
                v_x_M, v_y_M = s.pursue2(pursuer, target, destroyers)
                pursuer.attributes["v_x"] = v_x_M
                pursuer.attributes["v_y"] = v_y_M

    return new_velocities
"""


def adversary_test_policy(self_dict):
    for agent1 in m.get_alive(self_dict):
        alive_robots = []
        if agent1[0] == "E":
            for agent2 in m.get_alive(self_dict):
                if agent2[0] == "C":
                    alive_robots.append((self_dict["all_agents"][agent2],
                                         calc.agent_d(self_dict["all_agents"][agent1],
                                                      self_dict["all_agents"][agent2])))

            alive_robots.sort(key=lambda x: x[1])

            closest_robot = alive_robots[0][0]

            # new_vx, new_vy = s.destroy_target_attacker(self_dict["all_agents"][agent1], closest_robot)
            new_vx, new_vy = s.intercept(self_dict["all_agents"][agent1],
                                         closest_robot,
                                         self_dict["speed"],
                                         self_dict["time"],
                                         self_dict["time_steps"])

            self_dict["update_velocity"][agent1] = (new_vx, new_vy)


def adversary_test_policy_gym(self_dict, speed, time, time_steps):
    velocities = {}
    for agent1 in m.get_alive_gym(self_dict):
        alive_robots = []
        if agent1[0] == "E":
            for agent2 in m.get_alive_gym(self_dict):
                if agent2[0] == "C":
                    alive_robots.append((self_dict[agent2],
                                         calc.agent_d(self_dict[agent1],
                                                      self_dict[agent2])))

            alive_robots.sort(key=lambda x: x[1])

            closest_robot = alive_robots[0][0]

            # new_vx, new_vy = s.destroy_target_attacker(self_dict["all_agents"][agent1], closest_robot)
            new_vx, new_vy = s.intercept(self_dict[agent1],
                                         closest_robot,
                                         speed,
                                         time,
                                         time_steps)

            velocities[agent1] = (new_vx, new_vy)
    return velocities


def blob_PDE(self_dict):
    for agent1 in m.get_alive(self_dict):
        if agent1[0] == "C":
            evaders = []
            for agent2 in m.get_alive(self_dict):
                if agent2[0] == "E":
                    evaders.append((self_dict["all_agents"][agent2],
                                    calc.agent_d(self_dict["all_agents"][agent1], self_dict["all_agents"][agent2])))

            evaders.sort(key=lambda x: x[1])

            closest_evader = evaders[0][0]

            new_vx, new_vy = s.intercept(self_dict["all_agents"][agent1], closest_evader, self_dict["speed"],
                                         self_dict["time"], self_dict["time_steps"])

            self_dict["all_agents"][agent1].attributes["v_x"] = new_vx
            self_dict["all_agents"][agent1].attributes["v_y"] = new_vy


def blob_FTP(self_dict):
    for agent1 in m.get_alive(self_dict):
        if agent1[0] == "C":
            asleep_robots = []
            for agent2 in self_dict["all_agents"]:
                if (agent2[0] == "C" and agent1 is not agent2 and
                        self_dict["all_agents"][agent2].attributes["status"] == 0):
                    asleep_robots.append((self_dict["all_agents"][agent2],
                                          calc.agent_d(self_dict["all_agents"][agent1],
                                                       self_dict["all_agents"][agent2])))

            asleep_robots.sort(key=lambda x: x[1])

            closest_robot = asleep_robots[0][0]

            new_vx, new_vy = s.intercept(self_dict["all_agents"][agent1], closest_robot, self_dict["speed"],
                                         self_dict["time"], self_dict["time_steps"])

            self_dict["all_agents"][agent1].attributes["v_x"] = new_vx
            self_dict["all_agents"][agent1].attributes["v_y"] = new_vy

        else:
            alive_robots = []
            for agent2 in m.get_alive(self_dict):
                if agent2[0] == "C":
                    alive_robots.append((self_dict["all_agents"][agent2],
                                         calc.agent_d(self_dict["all_agents"][agent1],
                                                      self_dict["all_agents"][agent2])))

            alive_robots.sort(key=lambda x: x[1])

            closest_robot = alive_robots[0][0]

            new_vx, new_vy = s.intercept(self_dict["all_agents"][agent1], closest_robot, self_dict["speed"],
                                         self_dict["time"], self_dict["time_steps"])

            self_dict["all_agents"][agent1].attributes["v_x"] = new_vx
            self_dict["all_agents"][agent1].attributes["v_y"] = new_vy


def unique_intercept_FTP(self_dict, type1, type2):
    no_target_robots = m.get_agent_with_no_target(self_dict, type1)
    not_targeted = m.get_agent_not_targeted(self_dict, type1)

    while len(no_target_robots) > 0:
        if m.check_all_targeted(self_dict, type1):
            m.reset_agent_target_status(self_dict, type1)
        # print("Has no target - First while loop:", no_target_robots)
        # print("Not been targeted - First while loop:", not_targeted)
        asleep_robots = []
        for agent1 in no_target_robots:
            for agent2 in not_targeted:
                asleep_robots.append((agent1, agent2,
                                      calc.agent_d(self_dict["all_agents"][agent1], self_dict["all_agents"][agent2])))

        asleep_robots.sort(key=lambda x: x[2])

        while len(asleep_robots) > 0:
            # print("Asleep robots - start loop:", asleep_robots)
            pursuer_robot = asleep_robots[0][0]
            target_robot = asleep_robots[0][1]

            self_dict["all_agents"][pursuer_robot].attributes["has_target"] = asleep_robots[0][1]
            self_dict["all_agents"][target_robot].attributes["been_targeted"] = 1

            new_vx, new_vy = s.intercept(self_dict["all_agents"][pursuer_robot],
                                         self_dict["all_agents"][target_robot], self_dict["speed"],
                                         self_dict["time"], self_dict["time_steps"])

            self_dict["all_agents"][asleep_robots[0][0]].attributes["v_x"] = new_vx
            self_dict["all_agents"][asleep_robots[0][0]].attributes["v_y"] = new_vy

            new_list = []
            for val in asleep_robots:
                if val[0] != asleep_robots[0][0] and val[1] != asleep_robots[0][1]:
                    new_list.append(val)

            asleep_robots = new_list

            if m.check_all_targeted(self_dict, "C"):
                m.reset_robot_target_status(self_dict)

        no_target_robots = m.get_robots_with_no_target(self_dict)
        not_targeted = m.get_robots_not_targeted(self_dict)

    for agent1 in m.get_alive(self_dict):
        alive_robots = []
        if agent1[0] == type2:
            for agent2 in m.get_alive(self_dict):
                if agent2[0] == type1:
                    alive_robots.append((self_dict["all_agents"][agent2],
                                         calc.agent_d(self_dict["all_agents"][agent1],
                                                      self_dict["all_agents"][agent2])))

            alive_robots.sort(key=lambda x: x[1])

            closest_robot = alive_robots[0][0]

            """
            new_vx, new_vy = s.intercept(self_dict["all_agents"][agent1], closest_robot, self_dict["speed"],
                                         self_dict["time"], self_dict["time_steps"])
                                         """
            # new_vx, new_vy = s.destroy_target_attacker(self_dict["all_agents"][agent1], closest_robot)
            new_vx, new_vy = s.intercept(self_dict["all_agents"][agent1],
                                         closest_robot, self_dict["speed"],
                                         self_dict["time"], self_dict["time_steps"])
            self_dict["all_agents"][agent1].attributes["v_x"] = new_vx
            self_dict["all_agents"][agent1].attributes["v_y"] = new_vy


def MAPPO(self_dict):
    a = self_dict
    print(a)
    pass


def strategy2(self_dict):
    # m.print_agents(self_dict)
    # print("======================================================")
    # print()
    not_targeted_robots = m.get_robots_not_targeted(self_dict)
    # print("Robots not targeted:", not_targeted_robots)
    # print()

    unfrozen = m.get_unfrozen(self_dict)
    # print("Unfrozen robots:", unfrozen)
    # print()
    # print("Unfrozen robot targets")
    """
    for r in unfrozen:
        print(r + ":", unfrozen[r].attributes["has_target"])
    print()
    """

    has_no_targets = m.get_robots_with_no_target(self_dict)

    # This is the first part of the pseudocode
    adjacency_matrix = {}
    """
    Rows -> pursuer 
    Columns -> target
    
    Pursuer in row i attacks Target in column j
    
    get all distances and store them as each entry ij.
    => Get min dist
    ==> Have robot go there
    ==> Update adj matrix with inf so we get unique target
    ==> If A contains only inf, move on
    """

    for awake_r in unfrozen:
        row = {}
        for tar in not_targeted_robots:
            row[tar] = float("inf")
        adjacency_matrix[awake_r] = row

    """
    for row in adjacency_matrix:
        print(adjacency_matrix[row])
    """

    # If has_no_target is empty, every awake robot has a task
    # while has_no_targets:
    # print("have no target:", has_no_targets)

    # Go through the robots without a target to see which one is closest to a reachable robot
    for r in has_no_targets:
        r_cur = has_no_targets[r]
        # print("\tCurrent robot:", r)

        # Get available frozen targets for the current unfrozen robot
        available = m.get_reachable(r_cur, not_targeted_robots, self_dict)

        # update adjacency matrix with distance
        if available:
            for tar in available:
                t = not_targeted_robots[tar]
                d = c.agent_d(t, r_cur)
                adjacency_matrix[r][tar] = d

    # Check if correctly populated
    """
    for row in adjacency_matrix:
        print(adjacency_matrix[row])
    """

    # process the matrix

    while True:
        min_val = float("inf")
        min_p = None
        min_t = None

        # Find current smallest dist in matrix
        for p, t in adjacency_matrix.items():
            for tar, dist in t.items():
                if dist < min_val:
                    min_val = dist
                    min_p = p
                    min_t = tar

        # Check if done
        if min_val == float("inf"):
            break

        # Assign trajectory to robot
        new_vx, new_vy = s.intercept(self_dict["all_agents"][min_p],
                                     self_dict["all_agents"][min_t],
                                     self_dict["speed"], self_dict["time"], self_dict["time_steps"])

        self_dict["all_agents"][min_p].attributes["v_x"] = new_vx
        self_dict["all_agents"][min_p].attributes["v_y"] = new_vy
        self_dict["all_agents"][min_p].attributes["has_target"] = (min_t, min_val)
        self_dict["all_agents"][min_t].attributes["been_targeted"] = 1

        # Set distances for assigned pursuer to inf
        for pursuer in adjacency_matrix:
            for target in adjacency_matrix[pursuer]:
                if pursuer == min_p or target == min_t:
                    adjacency_matrix[pursuer][target] = float("inf")

        # Check if correctly updated
        """
        for row in adjacency_matrix:
            print(adjacency_matrix[row])
        """
    """
    print("Unfrozen robot targets after part 1")

    for r in unfrozen:
        print(r + ":", unfrozen[r].attributes["has_target"])
    print()
    """

    # Okay now if we get here, we have completed the first check

    # Part 2

    adjacency_matrix2 = {}

    has_no_targets = m.get_robots_with_no_target(self_dict)
    adversaries = m.get_alive_adversaries(self_dict)
    for awake_r in has_no_targets:
        row = {}
        for tar in adversaries:
            row[tar] = (float("inf"), float("inf"))
        adjacency_matrix2[awake_r] = row

    for r in has_no_targets:
        r_cur = has_no_targets[r]
        # print("\tCurrent robot:", r)

        # Get available frozen targets for the current unfrozen robot
        available = m.get_reachable_adversaries_for_r(r_cur, self_dict)

        # update adjacency matrix with distance
        if available:
            for tar in available:
                t = adversaries[tar]
                d = c.agent_d(t, r_cur)
                i = m.find_I_value(r, tar, self_dict)
                adjacency_matrix2[r][tar] = (d, i)

    # Check if correctly populated

    """
    for row in adjacency_matrix2:
        print(adjacency_matrix2[row])
    """

    while True:
        min_val = float("inf")
        min_p = None
        min_t = None

        # Find current smallest dist in matrix
        for p, t in adjacency_matrix2.items():
            for tar, val in t.items():
                dist, influence = val
                if dist < min_val and influence > 0:
                    min_val = dist
                    min_p = p
                    min_t = tar

        # Check if done
        if min_val == float("inf"):
            break

        # Assign trajectory to robot
        new_vx, new_vy = s.new_vx, new_vy = s.intercept(self_dict["all_agents"][min_p],
                                                        self_dict["all_agents"][min_t],
                                                        self_dict["speed"],
                                                        self_dict["time"],
                                                        self_dict["time_steps"])

        self_dict["update_velocity"][min_p] = (new_vx, new_vy)
        self_dict["all_agents"][min_p].attributes["has_target"] = (min_t, min_val)
        self_dict["all_agents"][min_t].attributes["been_targeted"] = 1

        # Set distances for assigned pursuer to inf
        for pursuer in adjacency_matrix2:
            for target in adjacency_matrix2[pursuer]:
                if pursuer == min_p or target == min_t:
                    adjacency_matrix2[pursuer][target] = (float("inf"), 0)

        # Check if correctly updated

        """
        for row in adjacency_matrix2:
            print(adjacency_matrix2[row])
        """

    """
    print("Unfrozen robot targets after part 2")
    for r in unfrozen:
        print(r + ":", unfrozen[r].attributes["has_target"])

    print()
    """

    # Okay now if we get here, we have completed the Second check

    # Part 3 going towards awake robots

    # Combine agents who are close before we populate adj matrix

    combine_dict = {}
    for awake_r in has_no_targets:
        combine_dict[awake_r] = []
        for test_combine in unfrozen:
            if awake_r is not test_combine:
                a1 = has_no_targets[awake_r].attributes
                a2 = unfrozen[test_combine].attributes
                d1 = calc.euclidean_distance(a1["x"], a1["y"], a2["x"], a2["y"])
                effective_dist = d1 - a1["r"]
                if effective_dist <= 0:
                    combine_dict[awake_r].append(test_combine)

    while len(combine_dict) > 0:
        max_val = 0
        agent = None

        for val in combine_dict:
            if max_val < len(combine_dict[val]):
                max_val = len(combine_dict[val])
                agent = val

        if agent is not None:
            for val in combine_dict[agent]:
                # print("Combining:", agent, "and", val)
                m.combine(self_dict, agent, val)
                if val in combine_dict:
                    del combine_dict[val]

                for key in combine_dict:
                    if val in combine_dict[key]:
                        combine_dict[key] = [value for value in combine_dict[key] if value != val]

            del combine_dict[agent]
        else:
            # print()
            break

    # print()

    adjacency_matrix3 = {}

    has_no_targets = m.get_robots_with_no_target(self_dict)

    unfrozen = m.get_unfrozen(self_dict)

    for awake_r in has_no_targets:
        row = {}
        for tar in unfrozen:
            row[tar] = float("inf")
        adjacency_matrix3[awake_r] = row

    for r in has_no_targets:
        r_cur = has_no_targets[r]
        # print("\tCurrent robot:", r)

        # Get available unfrozen targets for the current unfrozen robot
        available = m.get_reachable(r_cur, unfrozen, self_dict)

        # update adjacency matrix with distance
        if available:
            for tar in available:
                if tar is not r:
                    t = unfrozen[tar]
                    d = c.agent_d(t, r_cur)
                    adjacency_matrix3[r][tar] = d

    # Check if correctly populated

    """
    for row in adjacency_matrix3:
        print(adjacency_matrix3[row])

    print()
    """

    for r in has_no_targets:
        min_val = float("inf")
        min_p = None
        min_t = None

        # Find current smallest dist in matrix
        for t, val in adjacency_matrix3[r].items():
            if val < min_val:
                min_val = val
                min_p = r
                min_t = t

        # Assign trajectory to robot
        if min_p is not None and min_t is not None:
            new_vx, new_vy = s.new_vx, new_vy = s.intercept(self_dict["all_agents"][min_p],
                                                            self_dict["all_agents"][min_t],
                                                            self_dict["speed"],
                                                            self_dict["time"],
                                                            self_dict["time_steps"])

            self_dict["update_velocity"][min_p] = (new_vx, new_vy)
            self_dict["all_agents"][min_p].attributes["has_target"] = (min_t, min_val)
            self_dict["all_agents"][min_t].attributes["been_targeted"] = 1

    """
    print("Unfrozen robot targets after part 3")

    for r in unfrozen:
        print(r + ":", unfrozen[r].attributes["has_target"])

    print()

    print("Final check before we evade!")
    print()
    """

    has_no_targets = m.get_has_no_target(self_dict)
    adversaries = m.get_alive_adversaries(self_dict)

    for r in has_no_targets:
        vx, vy = s.evade_greedy(has_no_targets[r],
                                adversaries,
                                360,
                                100,
                                self_dict["width"],
                                self_dict["height"], self_dict["speed"])

        self_dict["update_velocity"][r] = (vx, vy)

    """
    print("Unfrozen robot targets after part 4")
    for r in unfrozen:
        print(r + ":", unfrozen[r].attributes["has_target"])

    print()
    """

    # adversaries
    # adversary_test_policy(self_dict)


def env_creator(config):
    # Access robots and adversaries from the config dictionary
    robots = config.get("robots", [])  # Provide a default value if not found
    adversaries = config.get("adversaries", [])

    if not pygame.get_init():
        pygame.init()
    screen = pygame.display.set_mode([1000, 1000])

    time_steps = 500
    overseer = e.Overseer(screen, robots, 0, adversaries, time_steps, 15)
    return overseer


