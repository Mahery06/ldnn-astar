general_prompt=(
                "Now the specific background of the problem is {type_in}."
                "Please refer to {data} for the specific method of node selection."
                "Your task: "
                "1. Based on the state ([[TempPath], 'content of thought', [TempPath], 'content of thought', ...]), select the possible next nodes;"
                "2. You need to generate 4 possible nodes based on {input}."
                "My Request:"
                "1. The selection of path points should be close to the obstacles, do not deliberately stay away from the obstacles."
                "2. Barriers do not necessarily have to be avoided. If the gaps between barriers are passable, you can go through them."
                "3. Each candidate node must differ from all existing points on the path by at least ≥2 grids in either the x or y direction (do not generate grids adjacent to existing points)."
                "4. The Thought content must be strictly output in the format [{{\"Point\":[x,y],\"Thought\":\"content of thought\"}},{{\"Point\":[x,y],\"Thought\":\"content of thought\"}}...}}], and it must be written."
                "and output as str.")

