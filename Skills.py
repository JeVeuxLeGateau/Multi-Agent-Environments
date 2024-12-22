import numpy as np
import calc
import math



def destroy_target_attacker(pursuer, target):
    return calc.get_unit_velocities(pursuer, target)


def proNav(pursuer, target, t0, tf):
    # beta - target flight path angle

    # RTx - horizontal position of target
    # RTy - vertical position of target
    # RMx - horizontal position of missile
    # RMy - vertical position of missile

    # VTx - horizontal position of target
    # VTy - vertical position of target
    # VMx - horizontal position of missile
    # VMy - vertical position of missile

    RTx = target.attributes["x"]
    RTy = target.attributes["y"]
    RMx = pursuer.attributes["x"]
    RMy = pursuer.attributes["y"]

    VTx = target.attributes["x"]
    VTy = target.attributes["y"]
    VMx = pursuer.attributes["x"]
    VMy = pursuer.attributes["y"]

    cT = target.attributes["c"]
    cM = pursuer.attributes["c"]

    # target acceleration
    aT = 0

    # heading angle
    HE_rad = math.radians(-20)

    # Time step size of integration
    h = 1

    # initial flight path angle (radians)
    beta_rad = 0

    # RTx, RTy, RMx, RMy - initials

    # VT target velocity magnitude (constant through sim)
    VT = cT * calc.d(VTx, VTy)

    # VM initial missile velocity magnitude
    VM = cM * calc.d(VMx, VMy)

    # navigation gain?
    Np = 4

    RTMx = RTx - RMx
    RTMy = RTy - RMy
    RTM = calc.d(RTMx, RTMy)

    lam = math.atan2(RTMy, RTMx)

    L = math.asin(VT * math.sin(beta_rad * lam) / VM)

    VMx = VM * math.cos(lam + L + HE_rad)
    VMy = VM * math.sin(lam + L + HE_rad)

    state = np.array([beta_rad, RTx, RTy, RMx, RMy, VTx, VTy, VMx, VMy])
    time_vector, state_trajectory = calc.discretized_integration(state, t0, tf, h, HE_rad, Np, VM, aT)
    next_state = state_trajectory[:, 1]

    new_vx1 = float(next_state[7])
    new_vy1 = float(next_state[8])
    new_vx2 = float(next_state[5])
    new_vy2 = float(next_state[6])

    p_vx, p_vy = calc.individual_unit_velocity(new_vx1, new_vy1)
    t_vy, t_vy = calc.individual_unit_velocity(new_vx2, new_vy2)

    return p_vx, p_vy


def MatthewAlgo(pursuer, target, speed, t0, tf):
    if pursuer.attributes["status"] == 1:
        P_x = pursuer.attributes["x"]
        P_y = pursuer.attributes["y"]

        E_x = target.attributes["x"]
        E_y = target.attributes["y"]
        E_vx = target.attributes["v_x"]
        E_vy = target.attributes["v_y"]

        for t in range(t0, tf):
            E_x += E_vx
            E_y += E_vy

            v_d = calc.d(E_x - P_x, E_y - P_y)

            # unit vector from P to E
            v_x = (E_x - P_x) / v_d
            v_y = (E_y - P_y) / v_d

            v = [v_x * (t - t0) + P_x, v_y * (t - t0) + P_y]

            d = calc.euclidean_distance(v[0], v[1], E_x, E_y) * speed

            if d <= speed * 2:
                return v_x, v_y

        # print("naive")
        return destroy_target_attacker(pursuer, target)
    else:
        return 0, 0


def pursue2(pursuer, target, destroyers):
    def closest_point_on_segment(A_p, B_p, P_p):
        A_p = np.array(A_p)
        B_p = np.array(B_p)
        P_p = np.array(P_p)

        # Vector AB and AP
        AB = B_p - A_p
        AP = P_p - A_p

        # Project AP onto AB
        AB_AB = np.dot(AB, AB)
        AP_AB = np.dot(AP, AB)
        t = AP_AB / AB_AB

        # Check if the projection is within the segment bounds
        if 0 <= t <= 1:
            # Closest point C
            C = A_p + t * AB
            # Check if C is perpendicular or on the segment
            if np.allclose(C, P_p):
                return P_p  # Point is exactly on the segment
            else:
                return tuple(C)  # Closest point on the segment
        else:
            return None

    def points_on_different_sides(A_p, B_p, P_p, Q_p):
        # Vector AB
        AB = (B_p[0] - A_p[0], B_p[1] - A_p[1])

        # Vectors AP and AQ
        AP = (P_p[0] - A_p[0], P_p[1] - A_p[1])
        AQ = (Q_p[0] - A_p[0], Q_p[1] - A_p[1])

        # Cross products
        cross1 = AB[0] * AP[1] - AB[1] * AP[0]
        cross2 = AB[0] * AQ[1] - AB[1] * AQ[0]

        # Check signs of cross products
        if cross1 * cross2 < 0:
            return True  # Points P and Q are on opposite sides of line AB
        else:
            return False  # Points P and Q are on the same side or exactly on line AB

    def midpoint(A_p, B_p):
        return (A_p[0] + B_p[0]) / 2, (A_p[1] + B_p[1]) / 2

    def point_position_relative_to_segment(A_p, B_p, P_p):
        x_a, y_a = A_p
        x_b, y_b = B_p
        x_p, y_p = P_p

        # Calculate the cross product
        cross_product = (x_b - x_a) * (y_p - y_a) - (y_b - y_a) * (x_p - x_a)

        if cross_product > 0:
            return -1
        elif cross_product < 0:
            return 1
        else:
            return 0

    px = pursuer.attributes["x"]
    py = pursuer.attributes["y"]

    tx = target.attributes["x"]
    ty = target.attributes["y"]

    points = []

    pursuer_point = np.array([px, py])
    target_point = np.array([tx, ty])

    for destroyer in destroyers:
        dx = destroyer.attributes["x"]
        dy = destroyer.attributes["y"]
        destroyer_point = np.array([dx, dy])

        point = closest_point_on_segment(pursuer_point, target_point, destroyer_point)

        if point is not None:
            x_val = point[0]
            y_val = point[1]

            d = calc.euclidean_distance(px, py, x_val, y_val)

            points.append((destroyer, (x_val, y_val), d))

    sorted_points = sorted(points, key=lambda c: c[2])

    if len(sorted_points) <= 1:
        return destroy_target_attacker(pursuer, target)
    else:
        d1 = sorted_points[0][0]
        d1x = d1.attributes["x"]
        d1y = d1.attributes["y"]

        d2 = sorted_points[1][0]
        d2x = d2.attributes["x"]
        d2y = d2.attributes["y"]

        A = (d1x, d1y)
        B = (d2x, d2y)

        if points_on_different_sides(pursuer_point, target_point, A, B):
            mid_x, mid_y = midpoint(A, B)
            return calc.scaled_vector(pursuer_point[0], pursuer_point[1], mid_x, mid_y, 1)
        else:
            distances = []
            for destroyer in destroyers:
                distances.append((destroyer, calc.agent_d(destroyer, pursuer)))

            sorted_distances = sorted(distances, key=lambda v: v[1])

            closest_destroyer = sorted_distances[0]
            d_check = closest_destroyer[1]
            d_check2 = pursuer.attributes["r"] + 200
            if d_check < d_check2:
                x = closest_destroyer[0].attributes["x"]
                y = closest_destroyer[0].attributes["y"]
                sign = point_position_relative_to_segment(pursuer_point, target_point, (x, y))

                vx, vy = calc.scaled_vector(pursuer_point[0], pursuer_point[1],
                                            target_point[0], target_point[1], 1)

                angle = 45 * (1 - calc.agent_d(pursuer, target) / 800)

                if sign < 0:
                    vx, vy = calc.rotation2D([vx, vy], -angle, 0, 0)
                    return vx, vy
                else:
                    vx, vy = calc.rotation2D([vx, vy], angle, 0, 0)
                    return vx, vy
            else:
                vx, vy = calc.scaled_vector(pursuer_point[0], pursuer_point[1],
                                            target_point[0], target_point[1], 1)
                return vx, vy


def evade_bad_attempt(pursuer, targets, width, height):
    vectors = []
    x = pursuer.attributes["x"]
    y = pursuer.attributes["y"]

    def compute_vectors(t):
        tx = t.attributes["x"]
        ty = t.attributes["y"]

        if calc.euclidean_distance(x, y, tx, ty) <= max(width / 2, height / 2):
            px, py = calc.scaled_vector(x, y, tx, ty, max(width / 2, height / 2))
            vx = px - tx
            vy = py - ty
            return [vx, vy]
        else:
            return [0, 0]

    def boundary_vectors():
        h = x - width / 2
        v = y - height / 2
        return [h, v]

    for target in targets:
        vectors.append(compute_vectors(target))

    b = boundary_vectors()
    vectors.append(b)

    r = [0, 0]

    for i in range(len(vectors)):
        for j in range(len(r)):
            r[j] += vectors[i][j]

    print("vectors:", vectors)
    print("results:", r)
    print()

    v_x, v_y = calc.normalize_vectors(r)

    return -v_x, -v_y


def evade_naive(pursuer, target, width, height):
    vx = -target[0].attributes["v_y"]
    vy = target[0].attributes["v_x"]
    return vx, vy


def evade_greedy(evader, pursuers, n, threshold, width, height, magnitude):
    # Goal minimize distance to walls
    # maximize distance to
    ps = []

    for pursuer in pursuers:
        if pursuers[pursuer].attributes["status"] == 1:
            if calc.agent_d(evader, pursuers[pursuer]) <= threshold:
                ps.append(pursuers[pursuer])

    def generate_n_directions(v_x, v_y):
        # v_new, y_new = calc.individual_unit_velocity(v_x, v_y)
        vectors = {0: [v_x, v_y]}

        angle = 360 / n

        for i in range(1, n):
            v_new, y_new = calc.rotation2D([vectors[i - 1][0], vectors[i - 1][1]], angle, 0, 0)
            vectors[i] = [v_new, y_new]

        for i in range(len(vectors)):
            vx_new, vy_new = calc.individual_unit_velocity(vectors[i][0], vectors[i][1])
            vectors[i][0] = vx_new
            vectors[i][1] = vy_new

        return vectors

    def calc_reward(x_start, y_start, v_x, v_y, magnitude):
        # TODO: Add SGD to rewards
        wx = 1
        wy = 1
        wd = 2
        reward = 0

        x_start = x_start + v_x * magnitude
        y_start = y_start + v_y * magnitude

        # print("new x", x_start)
        # print("new y", y_start)

        distance_reward = 0
        for p in ps:
            p_x = p.attributes["x"] + p.attributes["v_x"] * magnitude
            p_y = p.attributes["y"] + p.attributes["v_y"] * magnitude

            d = (500 - calc.euclidean_distance(x_start, y_start, p_x, p_y))

            # print("Closeness reward:", d)

            distance_reward += d

        x_reward = abs(x_start - (width / 2))
        y_reward = abs(y_start - (height / 2))

        # print("LR reward:", x_reward)
        # print("UD reward:", y_reward)

        reward += wy * x_reward + wx * y_reward + wd * distance_reward

        # print("total:", reward)
        # print()
        return reward

    x = evader.attributes["x"]
    y = evader.attributes["y"]
    vx = evader.attributes["v_x"]
    vy = evader.attributes["v_y"]

    directions = generate_n_directions(vx, vy)

    new_vx, new_vy = calc.rotation2D([vx, vy], 360 / n, 0, 0)

    test_direct = generate_n_directions(new_vx, new_vy)

    # goal is to minimize reward

    current_velocities = {}

    for i in range(len(test_direct)):
        if i == len(test_direct) - 1:
            current_velocities[i] = (len(test_direct) - 1, 0, calc_reward(x, y, test_direct[i][0], test_direct[i][1], magnitude),
                                     (test_direct[i][0], test_direct[i][1]))
        else:
            current_velocities[i] = (i, i + 1, calc_reward(x, y, test_direct[i][0], test_direct[i][1], magnitude),
                                     (test_direct[i][0], test_direct[i][1]))

    min_value = min(current_velocities, key=lambda k: current_velocities[k][2])

    initial_direction = current_velocities[min_value]

    right_vector = directions[initial_direction[0]]
    left_vector = directions[initial_direction[1]]

    best_reward = initial_direction[2]
    best_direction = initial_direction[3]

    flag = True

    while flag:
        angle = calc.dot(left_vector[0], left_vector[1], right_vector[0], right_vector[1]) / 2
        rotate_vx, rotate_vy = calc.rotation2D([right_vector[0], right_vector[1]], angle, 0, 0)
        rotate_vx_left, rotate_vy_left = calc.rotation2D([rotate_vx, rotate_vy], -angle / 2, 0, 0)
        rotate_vx_right, rotate_vy_right = calc.rotation2D([rotate_vx, rotate_vy], angle / 2, 0, 0)

        current_reward_vec = {(rotate_vx_left, rotate_vy_left): (
            left_vector, [rotate_vx, rotate_vy], calc_reward(x, y, rotate_vx_left, rotate_vy_left, magnitude),
            (rotate_vx_left, rotate_vy_left)),
            (rotate_vx_right, rotate_vy_right): ([rotate_vx, rotate_vy], right_vector,
                                                 calc_reward(x, y, rotate_vx_right, rotate_vy_right, magnitude),
                                                 (rotate_vx_left, rotate_vy_left))}

        new_reward = min(current_reward_vec, key=lambda b: current_reward_vec[b][2])

        if current_reward_vec[new_reward][2] < best_reward:
            best_reward = current_reward_vec[new_reward][2]
            best_direction = current_reward_vec[new_reward][3]
            left_vector = current_reward_vec[new_reward][0]
            right_vector = current_reward_vec[new_reward][1]
        else:
            return best_direction[0], best_direction[1]


def distract(pursuers, destroyer, evader, t0, tf, n, threshold, width, height):
    atk_def = {}

    d = []

    for i in range(len(pursuers)):
        atk_def[i] = [0, 0]
        d.append(calc.agent_d(pursuers[i], destroyer))

    closest = d.index(min(d))
    atk_def[closest] = [1, 0]

    # TODO change this to targeted: make it better
    if calc.agent_d(pursuers[closest], destroyer) <= 50:
        atk_def[closest][1] = 1

    for i in range(len(pursuers)):
        # print(atk_def[i][0])
        if atk_def[i][0] == 1:
            if atk_def[i][1] == 0:
                vx1, vy1 = MatthewAlgo(pursuers[i], destroyer, t0, tf)
            else:
                vx1, vy1 = evade_greedy(pursuers[i], [destroyer], n, threshold, width, height)

        else:
            if atk_def[closest][1] == 1:
                vx2, vy2 = MatthewAlgo(pursuers[i], evader, t0, tf)
            else:
                vx2, vy2 = evade_greedy(pursuers[i], [destroyer], n, threshold, width, height)

    return vx1, vy1, vx2, vy2


def intercept(pursuer, target, speed, time_remaining, time_final):
    p_x = pursuer.attributes["x"]
    p_y = pursuer.attributes["y"]
    t_x = target.attributes["x"]
    t_y = target.attributes["y"]
    t_vx = target.attributes["v_x"]
    t_vy = target.attributes["v_y"]

    # circles = []

    for t in range(time_final - time_remaining):
        radius = t * speed
        # circle = plt.Circle((x1, y1), radius, color='blue', fill=False, linewidth=2)
        # circles.append(circle)
        new_tar_x = t_x + (t_vx * radius)
        new_tar_y = t_y + (t_vy * radius)

        # plt.plot(new_tar_x, new_tar_y, 'ro')

        new_tar_x = max(0, min(new_tar_x, 1000))
        new_tar_y = max(0, min(new_tar_y, 1000))

        # vec towards target
        p_vx, p_vy = calc.scaled_vector(p_x, p_y, new_tar_x, new_tar_y, 1)

        new_p_x = p_x + radius * p_vx
        new_p_y = p_y + radius * p_vy

        # new_p_x = max(0, min(new_p_x, 1000))
        # new_p_y = max(0, min(new_p_y, 1000))

        if calc.euclidean_distance(new_p_x, new_p_y, new_tar_x, new_tar_y) < speed:
            return p_vx, p_vy

    return destroy_target_attacker(pursuer, target)


