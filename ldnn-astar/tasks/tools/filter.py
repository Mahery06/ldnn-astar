import math

from ...model.initial import deepseek
import tensorflow as tf
from modules import is_collision
from ...resources.data_1500 import standard_prompt
import json

data=standard_prompt


#prompt内一定保证start point是在temppath内部
def lm_filter(env:dict,path:list)->list:
    prompt=(
            f"现在，重新审核在{env}下生成的路径{path}([[x1,y1],[x2,y2],[x3,y3],……])"
            "我的需求是："
            "1.删除：删除路径中与之前节点行进方向相差超过90°的节点，路径序列长度不得大于5.如果长度大于5，相邻的节点只保留其中1个;"
            "2.条件：不能删除第一个选点；"
            "3.输出：直接返回路径([[x1,y1],[x2,y2],[x3,y3],……])，不需要思路、分析过程。")
    try:
        resp = deepseek(prompt)
        change_path=json.loads(resp)
        return change_path
    except json.decoder.JSONDecodeError as e:
        print(e)
        return path

def geo_filter(env,path:list)->list:
    if len(path)<4:
        return path
    else:
        d_list=[]
        for i in range(0,len(path)-2):
            if i in d_list:
                continue
            else:
                vec_1=tf.constant([path[i+1][0]-path[i][0],path[i+1][1]-path[i][1]],dtype=tf.float32)
                vec_2=tf.constant([path[i+2][0]-path[i+1][0],path[i+2][1]-path[i+1][1]],dtype=tf.float32)
                theta_cos=tf.tensordot(vec_1,vec_2,1)/(tf.norm(vec_1)*tf.norm(vec_2))
                if -1 <= theta_cos <= -math.sqrt(2)/2  :
                    d_list.append(i+1)
        for i in range(len(d_list)):
            path.pop(i)
        return path


#solve local-optimization problem -- aim to figure out the first point which could directly reach goal point.(plus,we could add another point to avoid redundant cost.)
def DirectToGoal(path,goal,env)->list:

    length=len(path)
    remain=[i for i in range(length)]
    fpath=[]
    if length < 4:
        return path
    else:
        for i in range(length-1,1,-1):
            if i in remain:
                for j in range(i-2,-1,-1):
                    if not is_collision(path[i][0],path[i][1],path[j][0],path[j][1],env):
                        for w in range(j+1,i):
                            remain.pop(w)
                        break
            else:
                continue
        for idx in remain:
            fpath.append(path[idx])
        return fpath


#use five layers of fiter to optimize generated path
def filter(env:dict,temppath:list,goal)->list:
    #first step:#avoid chosen points repetition
    path = []
    for obj in temppath:
        if obj not in path:
            path.append(obj)
    #second step:lm_filter——entirely depend on llm generation and evaluation
    temp_lm=lm_filter(env,path)
    temp_geo=geo_filter(env,temp_lm)
    final_path = []
    for obj in temp_geo:
        if obj not in final_path:
            final_path.append(obj)
    fpath = DirectToGoal(final_path,goal,env)
    for i in range(3):
        if len(fpath) > 4:
            fpath = DirectToGoal(fpath,goal,env)
        else:
            continue
    return fpath