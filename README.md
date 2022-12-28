# PopArt : efficient sparse regression and experimental design for optimal sparse linear bandits
This is the code for the paper 'PopArt : efficient sparse regression and experimental design for optimal sparse linear bandits' published in NeurIPS 2022.
(Under cleaning up the code)

## Requirement:
python 3.7 with numpy, scipy, sklearn, cvxpy, mosek, matplotlib

## To replicate the result:
For the l1 estimation error result of Figure 1 (row 1) in the paper, use

```
python 221222_CodeForPaper_final_version_l1reg.py
```
for Case 1 and

```
python 221222_CodeForPaper_final_version_l1reg.py -action_set=='uniform'
```
for Case 2. 


For the bandit result of Figure 1 (row 2) in the paper, use

```
python 221227_CodeForPaper_final_version_banditpart_theta_0.py
```
for Case 1 and

```
python 221227_CodeForPaper_final_version_banditpart_theta_0.py -action_set=='uniform'
```
for Case 2. 
