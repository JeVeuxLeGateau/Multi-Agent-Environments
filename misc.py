import math
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import calc
import calc as c
import random
import pandas as pd


def get_alive(self_dict):
    alive_agents = {}
    for agent in self_dict["all_agents"]:
        if self_dict["all_agents"][agent].attributes["status"] == 1:
            alive_agents[agent] = self_dict["all_agents"][agent]
    return alive_agents


def get_alive_gym(self_dict):
    alive_agents = {}
    for agent in self_dict:
        if self_dict[agent].attributes["status"] == 1:
            alive_agents[agent] = self_dict[agent]
    return alive_agents


def number_of_CDE(self_dict):
    num_C = 0
    num_D = 0
    num_E = 0

    for agent in self_dict["all_agents"]:
        if agent[0] == "C" and self_dict["all_agents"][agent].attributes["status"] == 1:
            num_C += self_dict["all_agents"][agent].attributes["W"]
        elif agent[0] == "D" and self_dict["all_agents"][agent].attributes["status"] == 1:
            num_D += self_dict["all_agents"][agent].attributes["W"]
        elif agent[0] == "E" and self_dict["all_agents"][agent].attributes["status"] == 1:
            num_E += self_dict["all_agents"][agent].attributes["W"]

    return [num_C, num_D, num_E]


def check_all_targeted(self_dict, agent_type):
    flag = True
    for agent in self_dict["all_agents"]:
        if (agent[0] == agent_type and
                self_dict["all_agents"][agent].attributes["been_targeted"] == 0 and
                self_dict["all_agents"][agent].attributes["status"] == 0):
            flag = False
    return flag


def reset_robot_target_status(self_dict):
    for agent in self_dict["all_agents"]:
        if agent[0] == "C" and self_dict["all_agents"][agent].attributes["status"] == 0:
            self_dict["all_agents"][agent].attributes["been_targeted"] = 0


def reset_agent_target_status(self_dict, agent_type):
    for agent in self_dict["all_agents"]:
        if agent[0] == agent_type and self_dict["all_agents"][agent].attributes["status"] == 0:
            self_dict["all_agents"][agent].attributes["been_targeted"] = 0


def check_all_has_target(self_dict):
    for agent in self_dict["all_agents"]:
        if agent[0] == "C" and self_dict["all_agents"][agent].attributes["has_target"] == (None, 0) and \
                self_dict["all_agents"][agent].attributes["status"] == 1:
            return False
    return True


def get_robots_with_no_target(self_dict):
    arr = {}
    alive_agents = get_alive(self_dict)
    for agent in alive_agents:
        if agent[0] == "C" and self_dict["all_agents"][agent].attributes["has_target"] == (None, 0):
            arr[agent] = self_dict["all_agents"][agent]
    return arr


def get_robots_not_targeted(self_dict):
    arr = {}
    for agent in self_dict["all_agents"]:
        if agent[0] == "C" and self_dict["all_agents"][agent].attributes["been_targeted"] == 0 and \
                self_dict["all_agents"][agent].attributes["status"] == 0:
            arr[agent] = self_dict["all_agents"][agent]
    return arr


def get_agent_with_no_target(self_dict, agent_type):
    arr = {}
    alive_agents = get_alive(self_dict)
    for agent in alive_agents:
        if agent[0] == agent_type and self_dict["all_agents"][agent].attributes["has_target"] == (None, 0):
            arr[agent] = self_dict["all_agents"][agent]
    return arr


def get_agent_not_targeted(self_dict, agent_type):
    arr = {}
    for agent in self_dict["all_agents"]:
        if agent[0] == agent_type and self_dict["all_agents"][agent].attributes["been_targeted"] == 0 and \
                self_dict["all_agents"][agent].attributes["status"] == 0:
            arr[agent] = self_dict["all_agents"][agent]
    return arr


def print_targets(self_dict):
    for agent in self_dict["all_agents"]:
        if self_dict["all_agents"][agent].attributes["status"] == 1:
            print("agent:", agent, "-", self_dict["all_agents"][agent].attributes["has_target"])


def print_agents(self_dict):
    for agent in self_dict["all_agents"]:
        print(agent + ":")
        for key, val in self_dict["all_agents"][agent].attributes.items():
            print(key + ":", val)

        print()


def prob_X_less_than_Y(p_1, p_2, X_0, Y_0):
    """
    Let X be the number of remaining number of robots
    Let Y be the number of remaining number of adversaries

    X_0 is the initial number of robots and Y_0 is the initial number of adversaries
    Let X_0 = 4 and Y_0 = 2

    To find P(X < Y)
    = P(X = 1, Y = 2) + P(X = 0, Y = 2) + P(X = 0, Y = 1)
    because these are independent
    = P(X = 1)P(Y = 2) + P(X = 0)P(Y = 2) + P(X = 0)P(Y = 1)

    let Binom(k, n, p) be the probability of getting k successes in n trials of
    probability p.

    P(X = n) = (X_0 C n) * ((1 - p2)^Y_0)^n * (1 - (1 - p2)^Y_0)^(X_0-n)
             = Binom(n, X_0, (1-p2)^Y_0)

    similarly
    P(Y = n) = Binom(n, Y_0, (1-p1)^X_0)
    """
    X_0 = int(X_0.item()) if isinstance(X_0, torch.Tensor) else X_0
    Y_0 = int(Y_0.item()) if isinstance(Y_0, torch.Tensor) else Y_0

    P_r = (1 - p_2) ** Y_0
    P_a = (1 - p_1) ** X_0

    total_prob = 0
    for x in range(X_0 + 1):
        for y in range(Y_0 + 1):
            if x < y:
                P_x = c.b(x, X_0, P_r)
                P_y = c.b(y, Y_0, P_a)
                total_prob += P_x * P_y
    return total_prob


"""
p1 = 0.2  # pe
p2 = 0.8  # ep
R_r = 2
A_r = 4
"""


def simulate(p1, p2, R_r, A_r):
    sims = 100000

    total = 0
    for _ in range(sims):
        R = []
        A = []

        for _ in range(R_r):
            R.append(1)

        for _ in range(A_r):
            A.append(1)

        # Sim robots turning off adversaries
        for i in range(R_r):
            for j in range(A_r):
                if random.uniform(0, 1) < p1:
                    A[j] = 0

        # Sim adversaries turning off robots
        for i in range(A_r):
            for j in range(R_r):
                if random.uniform(0, 1) < p2:
                    R[j] = 0

        if R.count(1) < A.count(1):
            total += 1

    print(total / sims)


# prob_X_less_than_Y(p1, p2, R_r, A_r)
# simulate(p1, p2, R_r, A_r)

def I_n(p1, p2, R_r, A_r):
    # p1 = 0.5
    # p2 = 0.5

    # prob we have more robots than adversaries
    P_Y_X = prob_X_less_than_Y(p2, p1, A_r, R_r)

    # prob we have more Adversaries than robots
    P_X_Y = prob_X_less_than_Y(p1, p2, R_r, A_r)

    return P_Y_X - P_X_Y


"""

Not_targeted = get_robots_not_targeted(self_dict) -> dict (Done)
R_unfrozen = get_unfrozen(self_dict) -> dict (Done)
R_reachable = get_reachable_robots_for_r(self_dict) -> dict (Done)
I_plus  = get_adversaries_in_I_plus(self_dict) -> dict
"""


def get_alive_adversaries(self_dict):
    adversaries = {}
    for agent in get_alive(self_dict):
        if agent[0] == "E":
            adversaries[agent] = self_dict["all_agents"][agent]

    return adversaries


def get_unfrozen(self_dict):
    unfrozen_robots = {}
    for agent in get_alive(self_dict):
        if agent[0] == "C":
            unfrozen_robots[agent] = self_dict["all_agents"][agent]

    return unfrozen_robots


def get_frozen(self_dict):
    frozen = {}
    for agent in self_dict["all_agents"]:
        if agent[0] == "C" and self_dict["all_agents"][agent].attributes["status"] == 0:
            frozen[agent] = self_dict["all_agents"][agent]

    return frozen


def get_has_no_target(self_dict):
    unfrozen = get_unfrozen(self_dict)
    no_target = {}
    for r in unfrozen:
        if unfrozen[r].attributes["has_target"] == (None, 0):
            no_target[r] = unfrozen[r]

    return no_target


# Good for now
def get_reachable(agent, test_set, self_dict):
    t = self_dict["time_steps"] - self_dict["time"]
    speed = self_dict["speed"]

    # agent we want to find who is reachable

    # Get adversaries
    alive_adversaries = get_alive_adversaries(self_dict)

    reachable = []

    for val in test_set:
        if val is not agent:
            d1 = c.agent_d(agent, test_set[val])
            # print("d1:", d1)
            flag = True
            for adv in alive_adversaries:
                d2 = c.agent_d(alive_adversaries[adv], test_set[val])
                # print("d2:", d2)
                if d2 < d1:
                    flag = False
                    break

            if flag:
                d3 = t * speed
                if d1 <= d3:
                    reachable.append(val)

    return reachable


def find_I_value(agent1, agent2, self_dict):
    """
    This differs from I_point because here were looking at what would happen if agent 1 were to go to
    the region controlled by agent 2.
    :param agent1:
    :param agent2:
    :param self_dict:
    :return:
    """
    alive = get_alive(self_dict)

    robots = 0
    adversaries = 0

    a1 = alive[agent1].attributes
    a2 = alive[agent2].attributes

    p1 = 0
    p2 = 0

    if agent1[0] == "C" and agent2[0] == "E":
        # robot prob
        p1 = self_dict["prob_dict"]["pe"]

        # adv prob
        p2 = self_dict["prob_dict"]["ep"]

        robots += a1["W"]
        adversaries += a2["W"]

    elif agent1[0] == "E" and agent2[0] == "C":
        # robot prob
        p1 = self_dict["prob_dict"]["ep"]

        # adv prob
        p2 = self_dict["prob_dict"]["pe"]

        robots += a2["W"]
        adversaries += a1["W"]

    for agent3 in alive:
        if agent1 is not agent3 and agent2 is not agent3:
            a3 = alive[agent3].attributes

            dist = c.euclidean_distance(a2["x"], a2["y"], a3["x"], a3["y"]) - a2["r"]
            if dist <= 0:
                if agent3[0] == "C":
                    robots += a3["W"]
                else:
                    adversaries += a3["W"]

    return I_n(p1, p2, robots, adversaries)


def get_adversaries_in_I_plus(agent, self_dict):
    t = self_dict["time_steps"] - self_dict["time"]
    speed = self_dict["speed"]

    # agent we want to find who is reachable
    a = agent.attributes

    # Get adversaries
    alive_adversaries = get_alive_adversaries(self_dict)

    adversaries_in_I_plus = {}
    for time in range(t):
        if alive_adversaries:
            for val in alive_adversaries:
                ts = alive_adversaries[val].attributes
                vx, vy = c.scaled_vector(a["x"], a["y"], ts["x"], ts["y"], 1)
                dist = (c.euclidean_distance(ts["x"], ts["y"], a["x"] + (vx * time), a["y"] + (vy * time)))

                if dist < speed / 2:
                    adversaries_in_I_plus[val] = alive_adversaries[val]
        else:
            break

    return adversaries_in_I_plus


def get_reachable_adversaries_for_r(agent, self_dict):
    t = self_dict["time_steps"] - self_dict["time"]
    speed = self_dict["speed"]

    # Get adversaries
    alive_adversaries = get_alive_adversaries(self_dict)

    reachable = []

    for adv in alive_adversaries:
        d1 = t * speed
        d2 = c.agent_d(alive_adversaries[adv], agent)
        if d2 <= d1:
            reachable.append(adv)

    return reachable


def combine(self_dict, a1, a2):
    self_dict["all_agents"][a1].attributes["W"] += self_dict["all_agents"][a2].attributes["W"]

    del self_dict["all_agents"][a2]

    new_update_velocity_dict = {}

    for agent in self_dict["update_velocity"]:
        if agent is not a2:
            new_update_velocity_dict[agent] = self_dict["update_velocity"][agent]

    self_dict["update_velocity"] = new_update_velocity_dict


def get_usable_agents(self_dict):
    usable = {}
    for agent in self_dict["all_agents"]:
        if agent[0] == "C":
            usable[agent] = self_dict["all_agents"][agent]
        elif agent[0] == "E" and self_dict["all_agents"][agent].attributes["status"] == 1:
            usable[agent] = self_dict["all_agents"][agent]

    return usable


def get_node_features(self_dict):
    """
        Agents: (Robot-0/Adversary-1, x, y, vx, vy, awake_status, speed, radius, weight, p1, p2)

        Robot - 0 / Adversary - 1
        x - [0, 1000]
        y - [0, 1000]
        vx - [-1, 1]
        vy - [-1, 1]
        awake status - 0/1
        speed - [0, inf]
        radius - [0, inf]
        weight - [1, inf]
        p1 - [0, 1]
        p2 - [0, 1]
        """
    """
            # Features that need requires_grad=True (positions and velocities)
            # 1
            x = torch.tensor([a["x"]], dtype=torch.float, requires_grad=True)
            # 2
            y = torch.tensor([a["y"]], dtype=torch.float, requires_grad=True)
            # 3
            vx = torch.tensor([a["v_x"]], dtype=torch.float, requires_grad=True)
            # 4
            vy = torch.tensor([a["v_y"]], dtype=torch.float, requires_grad=True)

            # Static features (no gradients needed)
            # 5
            status = torch.tensor([a["status"]], dtype=torch.float)
            # 6
            speed = torch.tensor([a["c"]], dtype=torch.float)
            # 7
            radius = torch.tensor([a["r"]], dtype=torch.float)
            # 8
            weight = torch.tensor([a["W"]], dtype=torch.float)
            # 9
            deaths = torch.tensor([a["deaths"]], dtype=torch.float)
            """

    usable = self_dict["all_agents"]

    node_features = []

    for agent in usable:
        a = usable[agent].attributes

        # Determine the agent type (0 = Robot, 1 = Adversary)

        agent_type = torch.tensor([0 if agent[0] == "C" else 1], dtype=torch.float)
        x = a["x"].unsqueeze(0)
        y = a["y"].unsqueeze(0)
        vx = a["v_x"].unsqueeze(0)
        vy = a["v_y"].unsqueeze(0)
        speed = a["c"].unsqueeze(0)
        status = a["status"].unsqueeze(0)
        radius = a["r"].unsqueeze(0)
        weight = a["W"].unsqueeze(0)
        deaths = a["deaths"].unsqueeze(0)

        # Collect all features into a single list
        node = torch.cat([agent_type, x, y, vx, vy, speed, status, radius, weight, deaths])

        node_features.append(node)

    # Stack all nodes into a tensor of shape (num_nodes, num_features)
    node_tensor = torch.stack(node_features)

    return node_tensor


def agents_as_numbers(self_dict):
    agents = get_usable_agents(self_dict)

    a_to_n = {}

    counter = 0
    for agent in agents:
        a_to_n[agent] = counter
        counter += 1

    return a_to_n


def numbers_as_agents(self_dict):
    agents = get_usable_agents(self_dict)

    n_to_a = {}

    counter = 0
    for agent in agents:
        n_to_a[counter] = agent
        counter += 1

    return n_to_a


def get_pursuer_indices(self_dict):
    agents = get_usable_agents(self_dict)
    a2n = agents_as_numbers(self_dict)

    p_i = []
    for agent in agents:
        if agent[0] == "C" and agents[agent].attributes["status"] == 1:
            p_i.append(a2n[agent])

    p_i_tensor = torch.tensor(p_i)
    return p_i_tensor


def get_edge_index_and_attr(self_dict):
    # Testing theory that the heap corruption issue for small numbers
    # of agents is partially caused by not having enough edges per node
    # So well test and see if we get an error before we leave and add
    # Edges until it works I guess?

    t = self_dict["time_steps"] - self_dict["time"]
    speed = self_dict["speed"]
    max_d = t * speed

    a2n = agents_as_numbers(self_dict)
    n2a = numbers_as_agents(self_dict)

    edge_index = []
    edge_attributes = []

    unfrozen_robots = []
    frozen_robots = []
    alive_adversaries = []

    for agent in self_dict["all_agents"]:
        a = self_dict["all_agents"][agent].attributes
        if agent[0] == "C":
            if a["status"] == 1:
                unfrozen_robots.append(agent)
            elif a["status"] == 0:
                frozen_robots.append(agent)
        elif agent[0] == "E":
            if a["status"] == 1:
                alive_adversaries.append(agent)

    # Get edges for unfrozen robots to frozen robots
    for agent1 in unfrozen_robots:
        a1 = self_dict["all_agents"][agent1]
        for agent2 in frozen_robots:
            a2 = self_dict["all_agents"][agent2]
            d = c.agent_d(a1, a2)
            if d <= max_d:
                edge_attributes.append([d])
                edge_index.append([a2n[agent1], a2n[agent2]])
                edge_attributes.append([d])
                edge_index.append([a2n[agent2], a2n[agent1]])

    # Get edges for adversaries to unfrozen robots
    for agent1 in alive_adversaries:
        a1 = self_dict["all_agents"][agent1]
        for agent2 in unfrozen_robots:
            a2 = self_dict["all_agents"][agent2]
            d = c.agent_d(a1, a2)
            if d <= max_d:
                edge_attributes.append([d])
                edge_index.append([a2n[agent1], a2n[agent2]])
                edge_attributes.append([d])
                edge_index.append([a2n[agent2], a2n[agent1]])

    error_preventing_edge_index = []
    if len(unfrozen_robots) + len(frozen_robots) < 5:
        for i in range(len(unfrozen_robots) + len(frozen_robots) + len(alive_adversaries)):
            for j in range(len(unfrozen_robots) + len(frozen_robots) + len(alive_adversaries)):
                if i != j:
                    error_preventing_edge_index.append([i, j])

        edge_index = error_preventing_edge_index

    tensor_index = torch.tensor(edge_index, dtype=torch.long)
    tensor_attr = torch.tensor(edge_attributes, dtype=torch.long)
    return tensor_index, tensor_attr


def get_data(self_dict):
    x = get_node_features(self_dict)
    idx, attr = get_edge_index_and_attr(self_dict)
    data = Data(x=x, edge_index=idx.t().contiguous(), edge_attr=attr)

    return data


def update_node_features(data, self_dict):
    for i, agent in enumerate(self_dict["all_agents"]):
        a = self_dict["all_agents"][agent].attributes

        # Use the current data.x[i] to preserve the graph structure
        new_x = data.x[i]  # No clone or detach, to keep gradients flowing

        # Update features with new values while preserving gradients

        new_x[1] = a["x"]
        new_x[2] = a["y"]
        new_x[3] = a["v_x"]
        new_x[4] = a["v_y"]
        new_x[6] = a["c"]
        new_x[7] = a["r"]
        new_x[8] = a["W"]
        new_x[9] = a["deaths"]

    """
    
   
    for i, agent in enumerate(usable):
        a = usable[agent].attributes

        # Update positions and velocities with new
        data.x[i][1] = a["x"]
        data.x[i][2] = a["y"]
        data.x[i][3] = a["v_x"]
        data.x[i][4] = a["v_y"]

        # Update other features as necessary (e.g., status, speed)
        data.x[i][5] = a["status"]
        data.x[i][6] = a["c"]
        data.x[i][7] = a["r"]
        data.x[i][8] = a["W"]
        data.x[i][9] = a["deaths"]
     """


def update_graph(data, self_dict):
    print(data.x)
    print(data.edge_index)
    print(data.edge_attr)

    update_node_features(data, self_dict)
    idx, attr = get_edge_index_and_attr(self_dict)
    data.edge_index = idx.t().contiguous()
    data.edge_attr = attr

    print(data.x)
    print(data.edge_index)
    print(data.edge_attr)


def get_endpoints(agent):
    x = agent["x"]
    y = agent["y"]
    r = agent["r"]

    top_left_x = int(x - r)
    top_left_y = int(y + r)
    bottom_right_x = int(x + r)
    bottom_right_y = int(y - r)

    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)


def I_point(self_dict, x, y):
    """
    As compared to find_I_value, this gets the influence at (x,y)
    """
    alive = get_alive(self_dict)

    robots = 0
    adversaries = 0

    p1 = self_dict["prob_dict"]["pe"]
    p2 = self_dict["prob_dict"]["ep"]

    for agent in alive:
        a = alive[agent].attributes

        dist = c.euclidean_distance(x, y, a["x"], a["y"]) - a["r"]

        if dist <= 0:
            if agent[0] == "C":
                robots += a["W"]
            else:
                adversaries += a["W"]

    return I_n(p1, p2, robots, adversaries)


def get_total_influence(self_dict):
    width = self_dict["width"]
    height = self_dict["height"]
    total_squares = width * height

    agents = get_alive(self_dict)
    non_empty_squares = {}
    seen_probs = {}

    for agent in agents:
        ul, br = get_endpoints(agents[agent].attributes)
        x1, y1 = ul
        x2, y2 = br
        for i in range(x1, x2 + 1):
            for j in range(y2, y1 + 1):
                if (i, j) not in non_empty_squares:
                    prob = I_point(self_dict, i, j)
                    prob_tensor = torch.tensor(prob, dtype=torch.float32, requires_grad=True)
                    non_empty_squares[(i, j)] = prob_tensor

                    if prob_tensor in seen_probs:
                        seen_probs[prob_tensor] += 1
                    else:
                        seen_probs[prob_tensor] = 1

    # scale values by total_squares
    scaled_values = {}
    for key in seen_probs:
        scaled_values[key] = seen_probs[key] / total_squares

    TI = torch.tensor(0, dtype=torch.float32, requires_grad=True)
    for key in scaled_values:
        TI = TI + key * torch.tensor(scaled_values[key], dtype=torch.float32, requires_grad=True)

    return TI


def get_closest_robot_reward(self_dict, cur, scale=1):
    dist = torch.tensor(float("inf"))

    current = self_dict["all_agents"][cur].attributes
    for agent in self_dict["all_agents"]:
        a = self_dict["all_agents"][agent].attributes
        if agent[0] == "C" and agent is not cur and a["status"] == 0:
            x1 = a["x"]
            y1 = a["y"]
            x2 = current["x"]
            y2 = current["y"]

            if not isinstance(a["x"], torch.Tensor):
                x1 = torch.tensor(x1)
            if not isinstance(a["y"], torch.Tensor):
                y1 = torch.tensor(y1)
            if not isinstance(current["x"], torch.Tensor):
                x2 = torch.tensor(x2)
            if not isinstance(current["y"], torch.Tensor):
                y2 = torch.tensor(y2)

            d = torch.sqrt(torch.pow(x1 - x2, 2) + torch.pow(y1 - y2, 2))
            if d < dist:
                dist = d

    width = torch.tensor(float(self_dict["width"]))
    height = torch.tensor(float(self_dict["height"]))
    max_d = torch.sqrt(torch.pow(width, 2) + torch.pow(height, 2))

    return scale * (1 - (dist / max_d))


def get_heading(self_dict, cur):
    dist = torch.tensor(float("inf"))
    B = None

    A = self_dict[cur].attributes
    for agent in self_dict:
        a = self_dict[agent].attributes
        if agent[0] == "C" and agent is not cur and a["status"] == 0:
            x1 = a["x"]
            y1 = a["y"]
            x2 = A["x"]
            y2 = A["y"]

            if not isinstance(a["x"], torch.Tensor):
                x1 = torch.tensor(x1)
            if not isinstance(a["y"], torch.Tensor):
                y1 = torch.tensor(y1)
            if not isinstance(A["x"], torch.Tensor):
                x2 = torch.tensor(x2)
            if not isinstance(A["y"], torch.Tensor):
                y2 = torch.tensor(y2)

            d = torch.sqrt(torch.pow(x1 - x2, 2) + torch.pow(y1 - y2, 2))
            if d < dist:
                dist = d
                B = a

    if B is not None:
        vx1, vy1 = c.scaled_vector(A["x"], A["y"], B["x"], B["y"], 1)
        vx2 = A["v_x"]
        vy2 = A["v_y"]

        vx1 = torch.tensor(vx1, dtype=torch.float32)
        vy1 = torch.tensor(vy1, dtype=torch.float32)
        vx2 = torch.tensor(vx2, dtype=torch.float32)
        vy2 = torch.tensor(vy2, dtype=torch.float32)

        v1 = torch.tensor([vx1, vy1], dtype=torch.float32)
        v2 = torch.tensor([vx2, vy2], dtype=torch.float32)

        dot_product = torch.dot(v1, v2)

        magnitude_v1 = torch.norm(v1)
        magnitude_v2 = torch.norm(v2)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0

        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return cos_theta.item()
    else:
        return 0


def get_closest_adv_reward(self_dict, cur, scale=1):
    dist = torch.tensor(float("inf"))

    current = self_dict["all_agents"][cur].attributes
    for agent in self_dict["all_agents"]:
        a = self_dict["all_agents"][agent].attributes
        if agent[0] == "E" and agent is not cur and a["status"] == 1:
            x1 = a["x"]
            y1 = a["y"]
            x2 = current["x"]
            y2 = current["y"]

            if not isinstance(a["x"], torch.Tensor):
                x1 = torch.tensor(x1)
            if not isinstance(a["y"], torch.Tensor):
                y1 = torch.tensor(y1)
            if not isinstance(current["x"], torch.Tensor):
                x2 = torch.tensor(x2)
            if not isinstance(current["y"], torch.Tensor):
                y2 = torch.tensor(y2)

            d = torch.sqrt(torch.pow(x1 - x2, 2) + torch.pow(y1 - y2, 2))

            if d < dist:
                dist = d

    width = torch.tensor(float(self_dict["width"]))
    height = torch.tensor(float(self_dict["height"]))
    max_d = torch.sqrt(torch.pow(width, 2) + torch.pow(height, 2))

    if torch.isinf(dist):
        return 0
    else:
        return scale * (1 - (dist / max_d))


def get_closest_robot_reward_non_tensor(self_dict, width, height, cur, scale=1):
    dist = float("inf")

    current = self_dict[cur].attributes
    for agent in self_dict:
        a = self_dict[agent].attributes
        if agent[0] == "C" and agent is not cur and a["status"] == 0:
            d = c.euclidean_distance(a["x"], a["y"], current["x"], current["y"])
            if d < dist:
                dist = d

    max_d = math.sqrt(width ** 2 + height ** 2)

    if math.isinf(dist):
        return 0
    else:
        return scale * (1 - (dist / max_d))


def get_closest_adv_reward_non_tensor(self_dict, width, height, cur, scale=1):
    dist = float("inf")

    current = self_dict[cur].attributes
    for agent in self_dict:
        a = self_dict[agent].attributes
        if agent[0] == "E" and agent is not cur and a["status"] == 1:
            d = c.euclidean_distance(a["x"], a["y"], current["x"], current["y"])
            if d < dist:
                dist = d

    max_d = math.sqrt(width ** 2 + height ** 2)

    if math.isinf(dist):
        return 0
    else:
        return scale * (1 - (dist / max_d))


def custom_reward(self_dict, TI, del_a):
    """
    Note the parameters seemed to perform a bit better when
    the values are larger like 200ish. Caution this was tested
    on single timesteps. Idk lets see.
    """

    # How much influence do we have over the board: A
    # - Scaled by number of robots

    # How many robots are awake at time t: B
    # - No scaling

    # How many adversaries are active at time t: C
    # - No scaling

    # How many adversaries were defeated at time t: D
    # - No scaling? test

    # Go towards closest robot reward: E
    # - Scaled between 0 and 1

    # Go away from adversary reward: F
    # - Scaled between -1 and 0

    loss = torch.tensor(0.0, requires_grad=True)

    for agent in self_dict["all_agents"]:
        a = self_dict["all_agents"][agent].attributes
        if agent[0] == "C" and a["status"] == 1:
            # Go towards closest robot // E
            r = get_closest_robot_reward(self_dict, agent)
            loss = loss + r

            # Go farther from adversary // F
            r = get_closest_adv_reward(self_dict, agent)
            loss = loss - r

            # Add 1 because we want +1 for each unfrozen robot // B
            loss = loss + 1

        elif agent[0] == "E" and a["status"] == 1:

            # Subtract 1 because we want -1 for each adversary // C
            loss = loss - 1

    # Add influence // A
    loss = loss + TI

    # Add # of defeated adversaries at current time step // D
    loss = loss + del_a

    """
    loss.backward()
    optimizer.step()

    print()
    print("Ep grad")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad}")
        else:
            print(f"No gradient for {name}")

    # optimizer.zero_grad()
    """
    return loss


# Normalizing function
def normalize_data(data):
    node_features = data.x

    # Compute mean and std directly from the tensor (along appropriate dims)
    mean = node_features.mean(dim=0, keepdim=True)
    std = node_features.std(dim=0, keepdim=True)

    # Normalize without breaking the graph (no detach, numpy, etc.)
    node_features_normalized = (node_features - mean) / (std + 1e-8)

    # Ensure the normalized tensor maintains gradient tracking
    data.x = node_features_normalized

    # Store mean and std for potential de-normalization later
    scaler = {"mean": mean, "std": std}

    return data, scaler


def un_normalize(data, scaler):
    node_features = data.x

    # Get the mean and standard deviation from the scaler dictionary
    mean = scaler['mean']
    std = scaler['std']

    # Perform the inverse transformation (create a new tensor)
    node_features_un_normalized = (node_features * std) + mean

    # Create a new Data object with the un-normalized features
    new_data = data.clone()  # Create a copy of the data object
    new_data.x = node_features_un_normalized

    return new_data  # Return the new data object


def process_info(info_dict, initial_robots, initial_adversaries):
    termination_condition = info_dict["val"]
    adversaries_defeated = initial_adversaries - info_dict["n_E"]
    time = info_dict["current_time"]
    unfrozen_robots = info_dict["n_C"]

    return [termination_condition, adversaries_defeated, unfrozen_robots, time]


def calc_ds(self_dict):
    agents = self_dict["all_agents"]
    ds = 0
    source = agents["C0"]

    for agent in agents:
        dist = calc.agent_d(source, agents[agent])
        if ds < dist:
            ds = dist

    return ds


def calc_diameter(self_dict):
    agents = self_dict["all_agents"]
    ds = 0

    for agent1 in agents:
        for agent2 in agents:
            dist = calc.agent_d(agents[agent1], agents[agent2])
            if ds < dist:
                ds = dist

    return ds


def calc_ds_gym(agents):
    ds = 0
    source = agents["C0"]

    for agent in agents:
        dist = calc.agent_d(source, agents[agent])
        if ds < dist:
            ds = dist

    return ds


def calc_diameter_gym(agents):
    ds = 0

    for agent1 in agents:
        for agent2 in agents:
            dist = calc.agent_d(agents[agent1], agents[agent2])
            if ds < dist:
                ds = dist

    return ds


def plot_results(episodes_list):
    # Create a 2x2 grid for subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    names = ["episode_reward_mean", "total_loss", "policy_loss", "vf_loss", "kl", "entropy"]
    df = pd.read_excel("Results_from_Training.xlsx", sheet_name="Sheet", names=names)

    erm = df["episode_reward_mean"]
    tl = df["total_loss"]
    pl = df["policy_loss"]
    vf_loss = df["vf_loss"]
    kl = df["kl"]
    entropy = df["entropy"]
    # KL_loss

    # Plot Episode Reward Mean in the first subplot
    axs[0, 0].plot(episodes_list, erm, label="Episode Reward Mean")
    axs[0, 0].set_xlabel("Episodes")
    axs[0, 0].set_ylabel("Mean Reward")
    axs[0, 0].set_title("Episode Reward Mean Over Episodes")
    axs[0, 0].legend()

    # Plot Total Loss in the second subplot
    axs[0, 1].plot(episodes_list, tl, label="Total Loss", color="orange")
    axs[0, 1].set_xlabel("Episodes")
    axs[0, 1].set_ylabel("Total Loss")
    axs[0, 1].set_title("Total Loss Over Episodes")
    axs[0, 1].legend()

    # Plot Policy Loss in the third subplot
    axs[1, 0].plot(episodes_list, pl, label="Policy Loss", color="blue")
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel("Policy Loss")
    axs[1, 0].set_title("Policy Loss Over Episodes")
    axs[1, 0].legend()

    # Plot Value Function Loss in the fourth subplot
    axs[1, 1].plot(episodes_list, vf_loss, label="Value Function Loss", color="green")
    axs[1, 1].set_xlabel("Episodes")
    axs[1, 1].set_ylabel("Value Function Loss")
    axs[1, 1].set_title("Value Function Loss Over Episodes")
    axs[1, 1].legend()

    # Plot KL Loss
    axs[2, 0].plot(episodes_list, kl, label="KL Divergence", color="purple")
    axs[2, 0].set_xlabel("Episodes")
    axs[2, 0].set_ylabel("KL Loss")
    axs[2, 0].set_title("KL Divergence Over Episodes")
    axs[2, 0].legend()

    # Plot Entropy
    axs[2, 1].plot(episodes_list, entropy, label="Entropy", color="red")
    axs[2, 1].set_xlabel("Episodes")
    axs[2, 1].set_ylabel("Entropy")
    axs[2, 1].set_title("Entropy Over Episodes")
    axs[2, 1].legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()
