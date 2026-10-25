import os
import json
import ast
from matplotlib import pyplot as plt
from ..tasks.tools.astar_with_waypoint import *

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_TXT = os.path.join(_BASE_DIR, "result", "f.txt")
ENV_JSON = os.path.join(_BASE_DIR, "problem", "q.json")
PLOT_DIR = os.path.join(_BASE_DIR, "result", "plots")


def _parse_hybrid(txt_path):
    comparisons, current = [], None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Waypoints:"):
                if current is not None:
                    comparisons.append(current)
                current = {"waypoints": ast.literal_eval(s[len("Waypoints:"):].strip()), "paths": []}
            elif s.startswith("Path:") and current is not None:
                current["paths"].append(ast.literal_eval(s[len("Path:"):].strip()))
        if current is not None:
            comparisons.append(current)
        return [{"waypoints": c["waypoints"],
               "direct_path": c["paths"][0] if len(c["paths"]) > 0 else None,
               "wp_path": c["paths"][1] if len(c["paths"]) > 1 else None} for c in comparisons]


def _load_envs(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_env(envs, start, goal):
    for e in envs:
        if tuple(e["Start Point"]) == start and tuple(e["Goal Point"]) == goal:
            return e
    return None



def _draw_barriers(ax, env):
    for y, x1, x2 in env.get("Horizontal Barriers", []):
        ax.plot([x1, x2], [y, y], color="black", linewidth=3, solid_capstyle="round")
    for x, y1, y2 in env.get("Vertical Barriers", []):
        ax.plot([x, x], [y1, y2], color="black", linewidth=3, solid_capstyle="round")


def _draw_scene(ax, env, path, open_nodes, waypoints, title, show_waypoints):
    start, goal = waypoints[0], waypoints[-1]
    _draw_barriers(ax, env)
    if open_nodes:
        ax.scatter([n[0] for n in open_nodes], [n[1] for n in open_nodes],
        s=20, marker="s", color="lightgray", zorder=1, label="OpenList")
    if show_waypoints:
        for wp in waypoints[1:-1]:
            ax.scatter(wp[0], wp[1], s=280, marker="*", color="yellow",
                        edgecolors="black", linewidths=0.6, zorder=4, label="Waypoint")
    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path],
                color="#1f77b4", linewidth=1.8, zorder=3, label="Path")
    ax.scatter(start[0], start[1], s=160, color="green", zorder=5, label="Start")
    ax.scatter(goal[0], goal[1], s=160, color="red", zorder=5, label="Goal")
    ax.annotate("Start", (start[0], start[1]), textcoords="offset points", xytext=(8, 8),
                  color="green", fontsize=10, fontweight="bold")
    ax.annotate("Goal", (goal[0], goal[1]), textcoords="offset points", xytext=(8, 8),
                  color="red", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-1, 51); ax.set_ylim(-1, 31); ax.set_aspect("equal")
    ax.set_xticks(range(0, 51, 5)); ax.set_yticks(range(0, 31, 5))
    ax.grid(True, color="gray", linewidth=0.4, alpha=0.35)
    h, l = ax.get_legend_handles_labels()
    by = dict(zip(l, h))
    ax.legend(by.values(), by.keys(), loc="best", fontsize=8, framealpha=0.9)


def plot_comparisons():
    os.makedirs(PLOT_DIR, exist_ok=True)
    envs = _load_envs(ENV_JSON)
    for idx, comp in enumerate(_parse_hybrid(HYBRID_TXT), start=1):
        start, goal = tuple(comp["waypoints"][0]), tuple(comp["waypoints"][-1])
        env = _find_env(envs, start, goal)
        if env is None:
            print(f"[warn] no env for {start}->{goal}"); continue
        direct_open = set(); astar(start, goal, env, open_recorder=direct_open)
        wp_open = set()
        llm_astar(start, goal, env, waypoints=comp["waypoints"], open_recorder=wp_open)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        _draw_scene(axes[0], env, comp["direct_path"], direct_open, comp["waypoints"],
                      "A* Direct", show_waypoints=False)
        _draw_scene(axes[1], env, comp["wp_path"], wp_open, comp["waypoints"],
                      "A* with Waypoints", show_waypoints=True)
        fig.suptitle(f"Comparison {idx}: Start {list(start)} -> Goal {list(goal)}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(PLOT_DIR, f"comparison_{idx:02d}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"saved: {out}")


if __name__ == "__main__":
    plot_comparisons()
