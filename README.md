# Tensor-Train-Orthogonal-Iteration (TTOI)

This repository contains Matlab functions and a tutorial for performing the Tensor-Train Orthogonal Iteration (TTOI) algorithm, proposed in the paper “Optimal High-order Tensor SVD via Tensor-Train Orthogonal Iteration” by Yuchen Zhou, Anru Zhang, Lili Zheng and Yazhen Wang. 

## Main Function
TTOI.m is a Matlab function for performing TTOI for any input tensor (noisy observation of a tensor with low TT-ranks), with specified TT-ranks, maximum number of iterations and tolerance. The output object includes estimated tensor at all iterations.

## Tutorial
Tutorial.mlx is a Matlab live script file, including some simulated examples for applying the TTOI function.
