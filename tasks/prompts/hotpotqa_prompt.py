hotpotqa_few_shots = ["""Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Thought 1: I need to identify who portrayed Corliss Archer in Kiss and Tell, then find that woman's government position.
Action 1: Search[Kiss and Tell (1945 film)]
Observation 1: Kiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film based on the play of the same name. The film stars Shirley Temple as Corliss Archer.
Thought 2: The film stars Shirley Temple, so I should search Shirley Temple to find any government position she held.
Action 2: Search[Shirley Temple]
Observation 2: Shirley Temple: Shirley Temple Black was an American actress, singer, dancer, businesswoman, and diplomat.
Thought 3: I should look up which government position she held.
Action 3: Lookup[diplomat]
Observation 3: (Result 1/1) She served as United States ambassador to Ghana and Czechoslovakia and as Chief of Protocol of the United States.
Thought 4: A government position she held was United States ambassador to Ghana.
Action 4: Finish[United States ambassador to Ghana]""",
"""Question: What is the nationality of the chef and restaurateur who wrote the cookbook that inspired the film Julie & Julia?
Thought 1: The cookbook that inspired Julie & Julia was written by Julia Child, so I need her nationality.
Action 1: Search[Julie & Julia]
Observation 1: Julie & Julia: The film is based in part on Julia Child's memoir My Life in France and on Julie Powell's blog.
Thought 2: I should search Julia Child to find her nationality.
Action 2: Search[Julia Child]
Observation 2: Julia Child: Julia Child was an American chef, author, and television personality.
Thought 3: Her nationality is American.
Action 3: Finish[American]"""]

# This prompt is intentionally left empty to use the AutoGen specific prompts instead
hotpotqa_solver_system_prompt = ""
