You are given a math or physics problem, a predicted answer, and the gold answer.

Task:
1) If both answers are numeric (possibly in scientific notation), judge correct if the predicted value matches the gold within 5% relative error (or absolute < 1e-3 when gold is 0).
2) Otherwise, judge correctness as a strict textual decision if the final boxed answer matches the gold.

Reply ONLY with a single uppercase letter:
- A if the predicted answer is correct
- B otherwise

Problem:
{question}

Gold:
{gold}

Predicted (extracted):
{pred}
