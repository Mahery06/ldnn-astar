from tools.modules import *
from tools.filter import filter
import time
from concurrent.futures import ThreadPoolExecutor
import heapq

def validate_waypoints(waypoints, source):
    if not isinstance(waypoints, (list, tuple)):
        raise TypeError(f"{source} must return a list of [x, y] points; got {waypoints!r}")

    for index, point in enumerate(waypoints):
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or not all(isinstance(value, (int, float)) for value in point)):
            raise ValueError(
                f"{source} returned malformed waypoint at index {index}: "
                f"{point!r}; full value={waypoints!r}"
            )


class Task:
    def __init__(self,env):
        self.env = env
        self.goal=env["Goal Point"]
        self.start=env["Start Point"]
        self.total=0
        self.t=0

    def layer(self, state_list):
        return_list = []
        expand = []
        for state in state_list:
            if self.goal == state[-2][-1]:
                return_list.append(state)  # 已到终点的保留原语义
            else:
                expand.append(state)

        def _gen(state):
            former_point = state[-2][-1]
            start=time.time()
            return state, generate(state, self.env, former_point, self.goal) ,time.time()-start

        temp_list, score_data = [], []
        sum_t=0
        with ThreadPoolExecutor(max_workers=min(8, len(expand))) as ex:
            for state, candidates, t in ex.map(_gen, expand):
                sum_t+=t
                for candidate in candidates:
                    temp_state = state.copy()
                    temp_path = state[-2].copy()
                    temp_path.append(candidate["Point"])
                    temp_state.append(temp_path)
                    temp_state.append(candidate["Thought"])

                    start=time.time()
                    score = value(temp_state, self.goal, self.env)
                    self.t += time.time() - start

                    if score == -1:
                        print("Score Error.");
                        exit(1)
                    elif score == -2:
                        continue
                    temp_list.append(temp_state)
                    score_data.append(score)

        if not temp_list:
            return state_list
        top3 = heapq.nlargest(3, [(i, v) for i, v in enumerate(score_data)], lambda x: x[1])
        return_list += [temp_list[i] for i, _ in top3]

        self.t+=(sum_t/min(8,len(expand)))
        return return_list

    def operate(self):

        path = [self.env["Start Point"]]
        state_initial = [path, "We should start from the Start Point."]
        state_list = [state_initial]
        # tree-graph generation
        # 总循环次数太多，会对结果有副作用
        for i in range(5):
            state_list=self.layer(state_list)


        temp=final_evaluate(state_list,self.goal,self.env)
        validate_waypoints(temp, "final_evaluate")

        s_waypoints = [self.start] + temp if self.start not in temp else temp
        full_waypoints = s_waypoints + [self.goal] if self.goal not in s_waypoints else s_waypoints


        GeneratedPath=filter(self.env,full_waypoints,self.goal)


        return GeneratedPath, self.t

