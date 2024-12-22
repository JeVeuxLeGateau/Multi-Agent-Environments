import math
import numpy as np
from scipy.stats import binom


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def d(v1, v2):
    return math.sqrt(v1 ** 2 + v2 ** 2)


def agent_d(a1, a2):
    x1 = a1.attributes["x"]
    y1 = a1.attributes["y"]
    x2 = a2.attributes["x"]
    y2 = a2.attributes["y"]
    return euclidean_distance(x1, y1, x2, y2)


def get_unit_velocities(agent1, agent2):
    """
    Takes the position of robots 1 and 2 and computes the horizontal and vertical vectors v_x and v_y
    such that robot 1 move 1 unit towards robot 2
    """

    x1 = agent1.attributes["x"]
    y1 = agent1.attributes["y"]
    x2 = agent2.attributes["x"]
    y2 = agent2.attributes["y"]

    delta_x = x2 - x1
    delta_y = y2 - y1
    if euclidean_distance(x1, y1, x2, y2) == 0:
        return 0
    else:
        return delta_x / euclidean_distance(x1, y1, x2, y2), delta_y / euclidean_distance(x1, y1, x2, y2)


def individual_unit_velocity(v_x, v_y):
    mag = math.sqrt(v_x ** 2 + v_y ** 2)
    if mag == 0:
        return 0, 0
    return v_x / mag, v_y / mag


def translation2D(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])


def rotation2D(vector, angle, x=0, y=0):
    vector.append(1)

    angle = math.radians(angle)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0],
                         [0, 0, 1]])

    rotated_vector = translation2D(x, y) @ rotation @ translation2D(-x, -y) @ vector

    return float(rotated_vector[0]), float(rotated_vector[1])


def scaled_vector(x1, y1, x2, y2, scale):
    dx = x2 - x1
    dy = y2 - y1
    d = euclidean_distance(x1, y1, x2, y2)
    if d == 0:
        return 0, 0
    else:
        P_x = (dx / d) * scale
        P_y = (dy / d) * scale

    return P_x, P_y


def normalize_vectors(vec):
    length = np.linalg.norm(vec)
    if length == 0:
        return vec
    return vec / length


def dot(a1, a2, b1, b2):
    left = a1 * b1 + a2 * b2
    a = d(a1, a2)
    b = d(b1, b2)

    if round((a * b), 4) == 0:
        return 0
    else:
        angle = math.degrees(math.acos(round(left, 4) / round((a * b), 4)))
        return angle


def dot_radians(a1, a2, b1, b2):
    left = a1 * b1 + a2 * b2
    a = d(a1, a2)
    b = d(b1, b2)

    if round((a * b), 4) == 0:
        return 0
    else:
        angle = math.acos(round(left, 4) / round((a * b), 4))
        return angle


def nLinPro_sim(t, y, HE_rad, Np, aT, VM, PN_type):
    # I'm aware that python doesn't have pointers
    sel_beta = 0
    sel_RT1 = 1
    sel_RT2 = 2
    sel_RM1 = 3
    sel_RM2 = 4
    sel_VT1 = 5
    sel_VT2 = 6
    sel_VM1 = 7
    sel_VM2 = 8

    dy = np.zeros(9)

    # target velocity magnitude
    VT = d(y[sel_VT1], y[sel_VT2])

    # relative position and velocities
    RTM1 = y[sel_RT1] - y[sel_RM1]
    RTM2 = y[sel_RT2] - y[sel_RM2]
    VTM1 = y[sel_VT1] - y[sel_VM1]
    VTM2 = y[sel_VT2] - y[sel_VM2]

    # relative distance
    RTM = d(RTM1, RTM2)

    # line of sight angle and time derivative
    lam = -math.atan2(RTM2, RTM1)
    lam_dot = -(RTM1 * VTM2 - RTM2 * VTM1) / (RTM * 2)

    # Closing velocity
    VC = -(RTM1 * VTM1 + RTM2 * VTM2) / RTM

    # DE RHS comps y = [beta, RTx, RTy, RMx, RMy, VTx, VTy, VMx, VMy]
    dy[0] = aT / VT
    dy[1] = VT * math.cos(y[sel_beta])
    dy[2] = VT * math.sin(y[sel_beta])
    dy[3] = y[sel_VM1]
    dy[4] = y[sel_VM2]
    dy[5] = aT * math.sin(y[sel_beta])
    dy[6] = aT * math.cos(y[sel_beta])

    if PN_type == "True":
        nc = Np * VC * lam_dot
        dy[7] = nc * math.sin(lam)
        dy[8] = nc * math.cos(lam)

    elif PN_type == "Pure":
        heading_pursuer = math.atan2(y[sel_VM2], y[sel_VM1])
        nc = Np * VM * lam_dot
        dy[7] = nc * math.sin(heading_pursuer)
        dy[8] = nc * math.cos(heading_pursuer)

    return dy


def discretized_integration(y0, t0, tf, h, H_error, nP, VM, aT):
    t = np.arange(t0, tf + h, h)

    nt = len(t)

    y_out = np.zeros(shape=(len(y0), nt))

    y_out[:, 0] = y0

    y = y0.copy()

    # state = np.array([beta_rad, RTx, RTy, RMx, RMy, VTx, VTy, VMx, VMy])

    for j in range(1, nt):
        s1 = nLinPro_sim(t[j], y0, H_error, nP, aT, VM, "Pure")
        s2 = nLinPro_sim((t[j] * h) / 2, y0 + h * s1 / 2, H_error, nP, aT, VM, "Pure")
        s3 = nLinPro_sim((t[j] * h) / 2, y0 + h * s2 / 2, H_error, nP, aT, VM, "Pure")
        s4 = nLinPro_sim(t[j], y0 + h * s3 / 2, H_error, nP, aT, VM, "Pure")
        y = y + (h * (s1 + (2 * s2) + (2 * s3) + s4) / 6)
        y_out[:, j] = y

    return t, y_out


def b(k, n, p):
    return binom.pmf(k, n, p)

