# Visual Columns Mapping Challenge 2024

This repository contains code for the third-place solution for the [Visual Columns Mapping Challenge](https://codex.flywire.ai/app/visual_columns_challenge) submitted by Xin Zheng in [Tatsuo Okubo Lab](https://cibr.ac.cn/science/team/detail/975?language=en) at the [Chinese Institute for Brain Research, Beijing (CIBR)](https://cibr.ac.cn/).

This data science challenge is about assigning neurons in the optic lobe of the Drosophila brain ([FlyWire dataset](https://codex.flywire.ai/)).

- Our score: 1,394,964 (`result.csv` attached)
- Number of assigned neurons: 23,247 neurons

We thank the FlyWire organizers at Princeton University for hosting this interesting challenge!

## General approach
Instead of directly solving the quadratic assignment problem (QAP), we broke down the problem into sub-problems, where each sub-problem was assigning neurons belonging to a single cell type.

1. We choose the baseline solution. We used two options:
    - [Benchmark solution prepared by the organizers](https://codex.flywire.ai/app/visual_columns_challenge)
    - We randomly picked two neurons of the same type, swapped their column assignments and if that lead to an increase in the score, we kept the swap.
2. For each cell type $t$, we calculated the hypothetical increase in the score ($\Delta_t$) as follows:
    - i. We first unassigned all the neurons of a given cell type $t$.
    - ii. We then calculated a score matrix $S$, with rows representing neurons and columns representing visual columns. Entry $S_{ij}$ of this score matrix represents the increase in the score if neuron $i$ was assigned to visual column $j$.
    - iii. Based on this score matrix $S$, we solve a assignment problem to determine the optimal neuron-to-column assignment for this given cell type $t$. This was done using the `linear_sum_assignment` function in [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html). The increase in the score after this optimal assignment is denoted as $\Delta_t$. Note that at this point, we are only calculating $\Delta_t$, but not performing the actual assignment yet, which will be done in the next step.
3. Once we calculated a table of cell type: $\Delta_t$, we pick the cell type based on the different strategies:
    - greedy: pick the cell type that gives the largest $\Delta_t$
    - $\varepsilon$-greedy: pick the cell type using greedy strategy with probability 1-$\varepsilon$ and pick a cell type at random with probability $\varepsilon$. 
4. Steps 2. and 3. were repeated until no increase in the score was observed (i.e. $\Delta$score is 0 for all the cell types). For our final submission, we used a high value of $\varepsilon=0.95$ making our algorithm very similar to randomly picking a cell type to reassign at each step.

## Code documentation
- We used Numpy 1.26.4 and Scipy 1.13.0, but it should work on most versions of Numpy and Scipy. For local swapping in step 1., we used [Julia](https://julialang.org/) 1.10.4  since we found it to be faster. No special optimization solver is necessary.
- `random_swap.ipynb` is a Jupyter notebook showing the random swapping algorithm we used (step 1). This provided us with a different starting point as the solution provided by the organizer, but it is not necessary. This is an inefficient algorithm that takes long time, and this notebook is mostly kept for the record. 
- `utils.py` contains utility functions including the function `diff_given_type()` that creates the score matrix and solves the assignment problem (step 2).
- `demo.ipynb` is a Jupyter notebook that shows how to use the functions in `utils.py` and shows a single step of the algorithm.
- `assign_prob_eps_greedy.py` is a Python script we used to run our algorithm

