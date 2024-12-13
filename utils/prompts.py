prompt_template = """<image>
THE GIVEN QUESTION:
{}
Answer the question using a single word or phrase.

HISTORICAL REASONING STEPS:
{}

Your task is to predict the next step of reasoning or calculation based on THE GIVEN QUESTION and HISTORICAL REASONING STEPS. Ensure your prediction is a single atomic reasoning step, which should be small and focused. If the historical reasoning steps have already reached a conclusion, there is no need to predict the next step in reasoning; simply reply with \"To sum up, the final answer is: ...\".
"""

demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
"""


demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

---

Examples:

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Mike has built a construction, shown in the upper picture, from equal cubes. Lily has taken several cubes out of it, thus Mike's construction became such as we see in the lower picture. How many cubes has Lily taken?
[Standard Answer]: 10
[Model_answer] : 10 cubes
Judgement: 1

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 2 C. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[Standard Answer]: 18
[Model_answer] : 18 dollars
Judgement: 1
---

Now please provide your judgement:

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """


prompt_w_gt = """
You are an advanced multimodal large language model. Your task is to generate a Chain of Thought (CoT) reasoning for a question based on a provided image and a reference answer. Break down your reasoning into clear, logical steps that are easy to follow.

### Requirements:
1. Construct a logical, step-by-step thought process using information from the image and the reference answer, along with any relevant external knowledge.
2. Ensure each step builds on the previous one and leads to the final answer.
3. Make necessary **inferences** based on the image content and additional knowledge (e.g., science, mathematics, geography).
4. In the last step, provide a concise, well-supported answer to the question, concluding with "Step N: To sum up, the final answer is: xxx."

---

### Example Format:

Input:
Image: (A picture of an apple and an orange placed on a table)
Question: Which fruit contains more Vitamin C?
Reference Answer: Oranges are known for their high Vitamin C content compared to apples.

---

Output:
Step 1: The image shows an apple and an orange, both common fruits. We need to determine the level of vitamins they contain.
Step 2: Apples are known for their sweetness, while oranges are citrus fruits often associated with Vitamin C.
Step 3: Based on general knowledge, oranges are well-known to contain a high amount of Vitamin C, more than most other fruits, including apples.
Step 4: To sum up, the final answer is: orange.

---

### Note:
Ensure that your reasoning is **clear, logical, and complete**, with no omitted steps. This will demonstrate how the answer is derived from both the image content and the reference answer.

---

### Your Input:
"""
prompt_template = """<image>
THE GIVEN QUESTION:
{}
Answer the question using a single word or phrase.

HISTORICAL REASONING STEPS:
{}

Your task is to predict the next step of reasoning or calculation based on THE GIVEN QUESTION and HISTORICAL REASONING STEPS. Ensure your prediction is a single atomic reasoning step, which should be small and focused. If the historical reasoning steps have already reached a conclusion, there is no need to predict the next step in reasoning; simply reply with \"To sum up, the final answer is: ...\".
"""

demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
"""


demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

---

Examples:

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Mike has built a construction, shown in the upper picture, from equal cubes. Lily has taken several cubes out of it, thus Mike's construction became such as we see in the lower picture. How many cubes has Lily taken?
[Standard Answer]: 10
[Model_answer] : 10 cubes
Judgement: 1

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 2 C. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[Standard Answer]: 18
[Model_answer] : 18 dollars
Judgement: 1
---

Now please provide your judgement:

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """


prompt_w_gt = """
You are an advanced multimodal large language model. Your task is to generate a Chain of Thought (CoT) reasoning for a question based on a provided image and a reference answer. Break down your reasoning into clear, logical steps that are easy to follow.

### Requirements:
1. Construct a logical, step-by-step thought process using information from the image and the reference answer, along with any relevant external knowledge.
2. Ensure each step builds on the previous one and leads to the final answer.
3. Make necessary **inferences** based on the image content and additional knowledge (e.g., science, mathematics, geography).
4. In the last step, provide a concise, well-supported answer to the question, concluding with "Step N: To sum up, the final answer is: xxx."

---

### Example Format:

Input:
Image: (A picture of an apple and an orange placed on a table)
Question: Which fruit contains more Vitamin C?
Reference Answer: Oranges are known for their high Vitamin C content compared to apples.

---

Output:
Step 1: The image shows an apple and an orange, both common fruits. We need to determine the level of vitamins they contain.
Step 2: Apples are known for their sweetness, while oranges are citrus fruits often associated with Vitamin C.
Step 3: Based on general knowledge, oranges are well-known to contain a high amount of Vitamin C, more than most other fruits, including apples.
Step 4: To sum up, the final answer is: orange.

---

### Note:
Ensure that your reasoning is **clear, logical, and complete**, with no omitted steps. This will demonstrate how the answer is derived from both the image content and the reference answer.

---

### Your Input:
"""

prompt_make_wrong = """
### Instruction: Generating Incorrect Steps for Each Correct Step (Chain of Thought)

1. Introduce Errors:
Modify each subsequent step (Step 1 to Step N) to introduce intentional errors, such as:
     1) Factual mistakes (e.g., “Apples contain more Vitamin C than oranges.”).
     2) Calculation errors (e.g., 3 + 2 = 6).
     3) Flawed logic or incorrect conclusions.
     4) Empty value or repeated reasoning.
2. Ensure Each Incorrect Step is Self-Contained:
The incorrect step should not affect or rely on other steps—each incorrect version is a standalone replacement of the correct step.

---

### Example:
Your Input:
Step 1: Oranges contain more Vitamin C than apples.
Step 2: Thus, the orange is the better source of Vitamin C.
Step 3: The answer is: orange.

One of Your Valid Output:
Step 1: Apples contain more Vitamin C than oranges.
Step 2: Therefore, the oranges is the better source of Vitamin B.
Step 3: The answer is: banana.

---
### Note:
Your output should be consistent with the input format and should not return extra characters or explanations of incorrect steps.
---

### Your Input:

"""

str_check = """
Question: {}
Ground Truth: {}
The Response You Need to Check: {}
"""

prompt_check = """
### Instruction: Checking Answer Correctness

Given the question, image, and ground truth solution, follow these steps to determine if the provided answer is correct, including both the reasoning steps and final answer.

1. Analyze the Question and Image:
   - Ensure you understand the question statement and the context provided by the image.
2. Compare the Provided Answer with Ground Truth:
   - Verify that the steps, logic, and reasoning in the provided answer align with the ground truth solution.
   - Check for calculation correctness and factual consistency at every step of the answer.
3. Evaluate the Final Result:
   - Ensure the final answer matches the ground truth both in value and format.
4. Determine Correctness:
   - If all steps, logic, calculations, and the final result match the ground truth exactly, return `True`.
   - If there is **any discrepancy** in the process, logic, or result, return `False`.

---

### Example Format:
Your Input:
Image: (A picture of an apple and an orange placed on a table)
Question: Which fruit contains more Vitamin C?
Ground Truth: Orange
The Response You Need to Check:
Step 1: Apples are known for their sweetness, while oranges are citrus fruits often associated with Vitamin C.
Step 2: Based on general knowledge, oranges are well-known to contain a high amount of Vitamin C, more than most other fruits, including apples.
Step 3: Given that oranges generally contain more Vitamin C than apples, the answer to the question is that the orange contains more Vitamin C.
Step 4: To sum up, the final answer is: orange.

Your Output:
True

---

Your Input:
Image: (A picture of an apple and an orange placed on a table)
Question: Which fruit contains more Vitamin C?
Ground Truth: Orange
The Response You Need to Check:
Step 1: Apples contain more Vitamin C than oranges.
Step 2: Therefore, the oranges is the better source of Vitamin B.
Step 3: The answer is: banana.

Your Output:
False

---

### Note:
Only return True or False based on the correctness evaluation. Do not provide any additional comments, explanations, or intermediate outputs.

### Your Input:

"""

prompt_make_wrong = """
### Instruction: Generating Incorrect Steps for Each Correct Step (Chain of Thought)

1. Introduce Errors:
Modify each subsequent step (Step 1 to Step N) to introduce intentional errors, such as:
     1) Factual mistakes (e.g., “Apples contain more Vitamin C than oranges.”).
     2) Calculation errors (e.g., 3 + 2 = 6).
     3) Flawed logic or incorrect conclusions.
     4) Empty value or repeated reasoning.
2. Ensure Each Incorrect Step is Self-Contained:
The incorrect step should not affect or rely on other steps—each incorrect version is a standalone replacement of the correct step.

---

### Example:
Your Input:
Step 1: Oranges contain more Vitamin C than apples.
Step 2: Thus, the orange is the better source of Vitamin C.
Step 3: The answer is: orange.

One of Your Valid Output:
Step 1: Apples contain more Vitamin C than oranges.
Step 2: Therefore, the oranges is the better source of Vitamin B.
Step 3: The answer is: banana.

---
### Note:
Your output should be consistent with the input format and should not return extra characters or explanations of incorrect steps.
---

### Your Input:

"""

prompt_check = """
### Instruction: Checking Answer Correctness

Given the question, image, and ground truth solution, follow these steps to determine if the provided answer is correct, including both the reasoning steps and final answer.

1. Analyze the Question and Image:
   - Ensure you understand the question statement and the context provided by the image.
2. Compare the Provided Answer with Ground Truth:
   - Verify that the steps, logic, and reasoning in the provided answer align with the ground truth solution.
   - Check for calculation correctness and factual consistency at every step of the answer.
3. Evaluate the Final Result:
   - Ensure the final answer matches the ground truth both in value and format.
4. Determine Correctness:
   - If all steps, logic, calculations, and the final result match the ground truth exactly, return `True`.
   - If there is **any discrepancy** in the process, logic, or result, return `False`.

---

### Example Format:
Your Input:
Image: (A picture of an apple and an orange placed on a table)
Question: Which fruit contains more Vitamin C?
Ground Truth: Orange
The Response You Need to Check:
Step 1: Apples are known for their sweetness, while oranges are citrus fruits often associated with Vitamin C.
Step 2: Based on general knowledge, oranges are well-known to contain a high amount of Vitamin C, more than most other fruits, including apples.
Step 3: Given that oranges generally contain more Vitamin C than apples, the answer to the question is that the orange contains more Vitamin C.
Step 4: To sum up, the final answer is: orange.

Your Output:
True

---

Your Input:
Image: (A picture of an apple and an orange placed on a table)
Question: Which fruit contains more Vitamin C?
Ground Truth: Orange
The Response You Need to Check:
Step 1: Apples contain more Vitamin C than oranges.
Step 2: Therefore, the oranges is the better source of Vitamin B.
Step 3: The answer is: banana.

Your Output:
False

---

### Note:
Only return True or False based on the correctness evaluation. Do not provide any additional comments, explanations, or intermediate outputs.

### Your Input:

"""
