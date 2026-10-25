import re
import math
import json
import os
import numpy as np
import tensorflow as tf
from ...model.initial import deepseek
from ...resources.prompt import *
from LSTM.deter_weights import attribute_model
from LSTM.windows import win_model
from ...resources.data_1500 import standard_prompt

data=standard_prompt

RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "ablation_deepseek_flash.txt")


class CandidateParseError(RuntimeError):
    """模型输出无法解析出候选节点时抛出。"""


def extract_candidates(raw: str) -> list:
    candidates = []
    point_re = re.compile(
        r'"Point"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    )
    thought_re = re.compile(r'"Thought"\s*:\s*"')
    for pm in point_re.finditer(raw):
        x, y = float(pm.group(1)), float(pm.group(2))
        thought = ""
        t = thought_re.search(raw, pm.end())
        if t:
            i, buf = t.end(), []
            while i < len(raw):
                c = raw[i]
                if c == "\\" and i + 1 < len(raw):  # 跳过转义字符
                    buf.append(raw[i + 1])
                    i += 2
                    continue
                if c == '"':  # 结束引号
                    break
                buf.append(c)
                i += 1
            thought = "".join(buf)
        candidates.append({"Point": [int(x), int(y)], "Thought": thought})
    return candidates

def generate(state:list,env:dict,former_point:list,goal_point:list)->list:
    tensor_to_goal=np.array([goal_point[0]-former_point[0],goal_point[1]-former_point[1]])

    prompt=general_prompt.format(type_in=env,data=data,input=state,ts=tensor_to_goal)
    raw=deepseek(prompt)

    candidates = extract_candidates(raw)
    if not candidates:
        # 兜底：整个串本身可能是合法 JSON，再试一次
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                candidates = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                candidates = []
    if not candidates:
        with open(RESULT_PATH, "a", encoding="utf-8") as f:
            print(f"{env['Start Point']}  \"Goal Point\": {env['Goal Point']}        NA", file=f)
        raise CandidateParseError(f"无法从模型输出中解析出候选节点，原文前 300 字符：\n{raw[:300]}")

    return candidates


def is_collision(chosen_point_x,chosen_point_y,former_point_x,former_point_y,env:dict)->int:
    vertical_barriers = env["Vertical Barriers"]  # [[x，y_start ，y_end ],……]
    horizontal_barriers = env["Horizontal Barriers"]  # [[y，x_start ，x_end ],……]
    tensor_test=np.array([[chosen_point_x-former_point_x],[chosen_point_y-former_point_y]])

    for vertical_barrier in vertical_barriers:
        if former_point_x == chosen_point_x == vertical_barrier[0]:
            if not (max(chosen_point_y, former_point_y) < vertical_barrier[1] or min(chosen_point_y, former_point_y) > vertical_barrier[2]):
                return 1
        end1=[vertical_barrier[0],vertical_barrier[1]]
        end2=[vertical_barrier[0],vertical_barrier[2]]

        tensor=np.array([[end1[0]-former_point_x,end2[0]-former_point_x],[end1[1]-former_point_y,end2[1]-former_point_y]])
        try:
            alpha,beta=np.linalg.solve(tensor,tensor_test)

            if alpha[0]>=0 and beta[0]>=0:
                if min(chosen_point_x,former_point_x)<=vertical_barrier[0]<=max(chosen_point_x,former_point_x):
                    return 1
        except np.linalg.LinAlgError:
            continue

    for horizontal_barrier in horizontal_barriers:
        if former_point_y == chosen_point_y == horizontal_barrier[0]:
            if not (max(chosen_point_x, former_point_x) < horizontal_barrier[1] or min(chosen_point_x, former_point_x) > horizontal_barrier[2]):
                return 1
        end1=[horizontal_barrier[1],horizontal_barrier[0]]
        end2=[horizontal_barrier[2],horizontal_barrier[0]]

        tensor = np.array([[end1[0] - former_point_x, end2[0] - former_point_x],[end1[1] - former_point_y, end2[1] - former_point_y]])
        try:
            alpha,beta=np.linalg.solve(tensor,tensor_test)

            if alpha[0]>=0 and beta[0]>=0:
                if min(chosen_point_y,former_point_y)<=horizontal_barrier[0]<=max(chosen_point_y,former_point_y):
                    return 1
        except np.linalg.LinAlgError:
            continue

    return 0

def value(state:list,goal:list,env:dict)->float:
    """in terms of optimal solution as well as general trend,we incorporate element ProximityToGoal and
     element AwayFromFormer and element-heuristicDeath(to infer the probability of death)
    """
    # Initial states only contain [path, thought], so they have no previous
    # waypoint to compare against and must not be scored as generated candidates.
    if not isinstance(state, list) or len(state) < 4:
        return -2
    current_path = state[-2]
    former_path = state[-4]
    if (not isinstance(current_path, list) or not current_path or
            not isinstance(former_path, list) or not former_path):
        return -2

    chosen_point_x=current_path[-1][0]
    chosen_point_y=current_path[-1][1]

    former_point_x=former_path[-1][0]
    former_point_y=former_path[-1][1]

    goal_x=goal[0]
    goal_y=goal[1]

    start_x=state[0][0][0]
    start_y=state[0][0][1]

    if is_collision(chosen_point_x,chosen_point_y,former_point_x,former_point_y,env):
        return -2
    else:
        # stay away from the former chosen point
        heuristic=math.sqrt((former_point_x-goal_x)**2+(former_point_y-goal_y)**2)
        if math.sqrt((chosen_point_x-former_point_x)**2+(chosen_point_y-former_point_y)**2)<1:
            return -2
        else:
            AwayFromFormer=math.sqrt((chosen_point_x-former_point_x)**2+(chosen_point_y-former_point_y)**2)/heuristic

        #设置两个硬规则，使得绕路的内容直接舍弃
        path = state[-2]
        for p in path[:-1]:
            dx = chosen_point_x - p[0]
            dy = chosen_point_y - p[1]
            if abs(dx) <= 1 and abs(dy) <= 1:
                return -2


        dg_former = math.sqrt((former_point_x-goal_x)**2 + (former_point_y-goal_y)**2)
        dg_current = math.sqrt((chosen_point_x-goal_x)**2 + (chosen_point_y-goal_y)**2)
        delta_dg = dg_current - dg_former
        hard_penalty = 1.0
        if delta_dg > 5:
            return -2
        elif delta_dg > 2:
            hard_penalty = 0.2
        elif delta_dg > 1:
            hard_penalty = 0.5

        #Given to ProximityToGoal is sometimes less than zero,so we use BMW-TOPSIS to evaluate the result.
        #But only use Euclidean distance could not successfully denote proximity.We should use the trend signal-cosine to optimize.
        dg=math.sqrt((chosen_point_x-goal_x)**2+(chosen_point_y-goal_y)**2)
        ds=math.sqrt((chosen_point_x-start_x)**2+(chosen_point_y-start_y)**2)
        ProximityToGoal=1-dg/(ds+dg)

        x_test1=np.array([[AwayFromFormer,ProximityToGoal]])
        pre=attribute_model.predict(x_test1,verbose=0)
        score=pre[0][0]

        score = score * hard_penalty

        path=state[-2]

        if len(path)>=4:
            x_test2=tf.constant([[
                [path[len(path)-4][0],path[len(path)-4][1],path[len(path)-3][0],path[len(path)-3][1]],
                [path[len(path)-2][0],path[len(path)-2][1],path[len(path)-1][0],path[len(path)-1][1]]
            ]])
            ans=win_model.predict(x_test2,verbose=0)[0][0]
            score=math.sqrt(ans*score)

        return score

# return  of different paths
def final_evaluate(state_list,goal,env:dict)->list:
    if not state_list:
        return []

    score_data = []
    for state in state_list:
        score = value(state,goal,env)
        if score == -1:
            print("Score Error.")
            exit(1)
        else:
            score_data.append(score)
    idx = score_data.index(max(score_data))
    state = state_list[idx]
    temp = state[-2]
    return temp
