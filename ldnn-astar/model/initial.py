from openai import OpenAI
import os

_client = None
def _get_client():
      global _client
      if _client is None:
          api_key = os.getenv("api_key")
          if not api_key:
              print("API key not found."); exit(1)
          _client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
      return _client


def deepseek(prompt:str)->str:

    client = _get_client()
    resp=client.chat.completions.create(
        model="deepseek-v4-pro",
        temperature=0.6,
        timeout=60,
        messages=[
            {"role":"system","content":"You are an expert specializing in computational geometry and path planning.Problems' Background Information:\n1.在一个大小为50*30的网格图总确定起点与终点之间的路径，以避开障碍物并找到到达目标的最短路线。\n2.Horizontal Barriers表示为[y，x_start ，x_end ]，Vertical Barriers表示为[x，y_start ，y_end ]."},
            {"role":"user","content":prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}}
    )
    return resp.choices[0].message.content

