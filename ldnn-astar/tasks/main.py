from task import *
from tools.astar_with_waypoints import *
from tools.modules import CandidateParseError

with open("problem/2_标准格式.json") as f:
    questions = json.load(f)

if __name__ == '__main__':
    print("cooking……")
    for q in questions:
        try:
            task=Task(q)
            result,time=task.operate()


            run(q,result)
        except CandidateParseError:
            with open("result/ablation_LSTM.txt","a") as f:
                print(f"[NA] Start {q['Start Point']} -> Goal {q['Goal Point']}",file=f)

