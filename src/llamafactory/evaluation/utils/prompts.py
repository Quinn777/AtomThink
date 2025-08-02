atomthink_prompt_template = """THE GIVEN QUESTION:
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