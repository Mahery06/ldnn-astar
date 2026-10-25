import heapq
import math
import numpy as np


NEIGHBORS = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

PATH = "result/ablation_LSTM.txt"

def _is_collision(chosen_point_x, chosen_point_y, former_point_x, former_point_y, env: dict) -> int:
    vertical_barriers = env.get("Vertical Barriers", [])
    horizontal_barriers = env.get("Horizontal Barriers", [])
    tensor_test = np.array([[chosen_point_x - former_point_x], [chosen_point_y - former_point_y]])

    for vertical_barrier in vertical_barriers:
        if former_point_x == chosen_point_x == vertical_barrier[0]:
            if not (max(chosen_point_y, former_point_y) < vertical_barrier[1] or min(chosen_point_y, former_point_y) > vertical_barrier[2]):
                return 1
        end1 = [vertical_barrier[0], vertical_barrier[1]]
        end2 = [vertical_barrier[0], vertical_barrier[2]]

        tensor = np.array([[end1[0] - former_point_x, end2[0] - former_point_x], [end1[1] - former_point_y, end2[1] - former_point_y]])
        try:
            alpha, beta = np.linalg.solve(tensor, tensor_test)
            if alpha[0] >= 0 and beta[0] >= 0:
                if min(chosen_point_x, former_point_x) <= vertical_barrier[0] <= max(chosen_point_x, former_point_x):
                    return 1
        except np.linalg.LinAlgError:
            continue

    for horizontal_barrier in horizontal_barriers:
        if former_point_y == chosen_point_y == horizontal_barrier[0]:
            if not (max(chosen_point_x, former_point_x) < horizontal_barrier[1] or min(chosen_point_x, former_point_x) > horizontal_barrier[2]):
                return 1
        end1 = [horizontal_barrier[1], horizontal_barrier[0]]
        end2 = [horizontal_barrier[2], horizontal_barrier[0]]

        tensor = np.array([[end1[0] - former_point_x, end2[0] - former_point_x], [end1[1] - former_point_y, end2[1] - former_point_y]])
        try:
            alpha, beta = np.linalg.solve(tensor, tensor_test)
            if alpha[0] >= 0 and beta[0] >= 0:
                if min(chosen_point_y, former_point_y) <= horizontal_barrier[0] <= max(chosen_point_y, former_point_y):
                    return 1
        except np.linalg.LinAlgError:
            continue

    return 0


def on_barrier(x, y, env):
    for hb in env["Horizontal Barriers"]:
        if y == hb[0] and hb[1] <= x <= hb[2]:
            return True
    for vb in env["Vertical Barriers"]:
        if x == vb[0] and vb[1] <= y <= vb[2]:
            return True
    return False


def astar(start, goal, env, grid_w=50, grid_h=30, open_recorder=None):
    start = tuple(start)
    goal = tuple(goal)

    if on_barrier(goal[0], goal[1], env):
        return None, 0, 0

    def h(pos):
        return math.sqrt((pos[0] - goal[0])**2+ (pos[1] - goal[1])**2)

    def in_bounds(x, y):
        return 0 <= x <= grid_w and 0 <= y <= grid_h

    operation_cost=0
    open_set = [(h(start),0,start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_set:
        _,_,cur= heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        operation_cost += 1 # 由于每一次pop都会产生f-cost与neighbors的计算，operation cost只统计pop的次数

        if cur == goal:
            path = [list(cur)]
            while cur in came_from:
                cur = came_from[cur]
                path.append(list(cur))
            path.reverse()
            storage_cost = len(g_score)
            return path, operation_cost, storage_cost

        for dx, dy in NEIGHBORS:
            nx, ny = cur[0]+dx, cur[1]+dy
            nb = (nx, ny)
            if not in_bounds(nx, ny) or nb in closed or on_barrier(nx, ny, env):
                continue

            gt = g_score[cur] + dx ** 2 + dy ** 2
            if nb not in g_score or gt < g_score[nb]:
                came_from[nb] = cur
                g_score[nb] = gt
                heapq.heappush(open_set, (gt + h(nb), gt,nb))
                if open_recorder is not None:
                    open_recorder.add(nb)



    storage_cost = len(g_score)
    return None, operation_cost, storage_cost

def map_features(env, start, goal):

    hb=env["Horizontal Barriers"]
    vb=env["Vertical Barriers"]

    obstacles=0
    for h in hb:
        obstacles+=(h[2]-h[1]+1)
    for v in vb:
        obstacles+=(v[2]-v[1]+1)

    rho = obstacles / 1500
    D = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return rho, D


def adaptive_params(env, start, goal):
    """自适应计算 α、β"""
    RHO_REF = 0.15
    alpha0 = 1.5;beta0 = 1.0
    D_REF = 78
    rho, D = map_features(env, start, goal)
    alpha = alpha0 * (D / D_REF)
    beta  = beta0 * (rho / RHO_REF)
    alpha = max(1.5, min(2.0, alpha))
    beta  = max(0.75, min(1.0, beta))
    return alpha, beta

def llm_astar(start, goal, env, waypoints=None, grid_w=50, grid_h=30, open_recorder=None):

    start = tuple(start)
    goal = tuple(goal)

    alpha,beta = adaptive_params(env, start, goal)

    targets = []
    if waypoints:
        for wp in waypoints:
            w = (int(wp[0]), int(wp[1]))
            if w == start or w == goal:
                continue
            if not (0 <= w[0] <= grid_w and 0 <= w[1] <= grid_h):
                continue
            if on_barrier(w[0], w[1], env):
                continue
            if w not in targets:
                targets.append(w)
    targets.append(goal)

    if on_barrier(goal[0], goal[1], env):
        return None, 0, 0

    t_idx = [0]

    def h_goal(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) + (math.sqrt(2) - 2) * min(abs(pos[0] - goal[0]),abs(pos[1] - goal[1]))

    def h_target(pos):
        t = targets[t_idx[0]]
        return abs(pos[0] - t[0]) + abs(pos[1] - t[1])

    def in_bounds(x, y):
        return 0 <= x <= grid_w and 0 <= y <= grid_h

    operation_cost = 0
    open_set = [(alpha*h_goal(start) + beta*h_target(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        operation_cost += 1

        if cur == goal:
            path = [list(cur)]
            while cur in came_from:
                cur = came_from[cur]
                path.append(list(cur))
            path.reverse()
            storage_cost = len(g_score)
            return path, operation_cost, storage_cost

        for dx, dy in NEIGHBORS:
            nx, ny = cur[0] + dx, cur[1] + dy
            nb = (nx, ny)
            if not in_bounds(nx, ny):
                continue

            # 到达当前 waypoint：切到下一个目标并重建堆（对应 repo 的 _update_target / _update_queue）
            if targets[t_idx[0]] != goal and nb == targets[t_idx[0]]:
                t_idx[0] += 1
                rebuilt = []
                while open_set:
                    _, g, node = heapq.heappop(open_set)
                    heapq.heappush(rebuilt, (g + alpha*h_goal(node) + beta*h_target(node), g, node))
                open_set = rebuilt

            if nb in closed or on_barrier(nx, ny, env):
                continue

            gt = g_score[cur] + abs(dx)+abs(dy)
            if nb not in g_score or gt < g_score[nb]:
                came_from[nb] = cur
                g_score[nb] = gt
                heapq.heappush(open_set, (gt + alpha*h_goal(nb) + beta*h_target(nb), gt, nb))
                if open_recorder is not None:
                    open_recorder.add(nb)

    storage_cost = len(g_score)
    return None, operation_cost, storage_cost


# ─── Main ──────────────────────────────────────────────────────────

def run(env,full_waypoints):
    start=env["Start Point"]
    goal=env["Goal Point"]

    if (isinstance(full_waypoints, (list, tuple)) and len(full_waypoints) == 2
            and all(isinstance(value, (int, float)) for value in full_waypoints)):
        candidate_waypoints = [full_waypoints]
    elif isinstance(full_waypoints, (list, tuple)):
        candidate_waypoints = full_waypoints
    else:
        candidate_waypoints = []

    valid_waypoints = []
    for waypoint in candidate_waypoints:
        if (isinstance(waypoint, (list, tuple)) and len(waypoint) == 2
                and all(isinstance(value, (int, float)) for value in waypoint)):
            point = [int(waypoint[0]), int(waypoint[1])]
            if point not in valid_waypoints:
                valid_waypoints.append(point)
        else:
            print(f"Warning: ignoring malformed waypoint: {waypoint!r}")

    full_waypoints = [list(start)] + [
        wp for wp in valid_waypoints
        if wp != start and wp != goal and not on_barrier(wp[0], wp[1], env)
    ] + [list(goal)]

    with open(PATH, "a", encoding="utf-8") as f:
        print(file=f)
        print(f"Waypoints: {full_waypoints}",file=f)

        # ── Baseline: A* direct ──
        print(f"A* Direct:",file=f)
        direct_path, d_op, d_storage = astar(start, goal, env)


        if direct_path:
            d_len = sum(math.sqrt((direct_path[i][0]-direct_path[i-1][0])**2 +
                                  (direct_path[i][1]-direct_path[i-1][1])**2)
                        for i in range(1, len(direct_path)))

            print(f"    point-count={len(direct_path)} pts\t\tlength={d_len:.2f}", file=f)
            print(f"    operation cost   = {d_op}", file=f)
            print(f"    storage cost     = {d_storage}", file=f)
            print(f"    Path: {direct_path}",file=f)
        else:
            print(f"    Result: NO PATH (op={d_op}, storage={d_storage})",file=f)

        # ── A* with waypoints ──
        print(f"LDNN-A*:",file=f)
        wp_path, wp_op, wp_storage = llm_astar(start, goal, env, waypoints=full_waypoints)

        if wp_path:
            w_len = sum(math.sqrt((wp_path[i][0]-wp_path[i-1][0])**2 +
                                  (wp_path[i][1]-wp_path[i-1][1])**2)
                        for i in range(1, len(wp_path)))


            print(f"    point-count={len(wp_path)} pts\t\tlength={w_len:.2f}", file=f)
            print(f"    operation cost        = {wp_op}", file=f)
            print(f"    storage cost          = {wp_storage}", file=f)
            print(f"    Path: {wp_path}", file=f)
        else:
            print(f"\n    Result: FAILED",file=f)

        print(file=f)
        if wp_path:
            if d_storage > 0:
                ratio_s = wp_storage / d_storage
                ratio_o = wp_op / max(1, d_op)
                print(f"  vs Baseline:     storage {ratio_s*100:.1f}%   operation {ratio_o*100:.1f}%", file=f)
        print(file=f)

