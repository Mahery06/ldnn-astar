A* is a standard algorithm for grid-based path planning, but the number of nodes it expands and stores grows rapidly with map size and obstacle complexity, which limits its use on 
resource-constrained agents. Recent work reduces this overhead by letting large language models (LLMs) propose intermediate waypoints that split the search into shorter segments. 
However, LLMs reason poorly about space, so the proposed waypoints are often misplaced or redundant, which causes detours and, in many cases, more search effort than plain A*. 
We present LDNN-A*, a hybrid planner in which the LLM only proposes candidate waypoints, while lightweight neural evaluators decide which ones to keep. A deep neural network assesses 
the local geometry of each candidate, and a long short-term memory network assesses the waypoint sequence as a whole. The selected waypoints are then pruned by a three-stage filter and 
connected by an adaptive weighted A* search. We evaluate LDNN-A* on grid maps with different obstacle layouts against A*, LLM-A*, and iLLM-A*. LDNN-A* matches or outperforms all baselines 
in both node expansions and peak memory in most scenarios, while its paths remain within 7\% of the shortest path found by any method. Ablation studies show that waypoint filtering is the 
main factor in controlling search overhead.
