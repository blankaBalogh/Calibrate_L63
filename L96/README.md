# L96 model

The ´data´ folder contains :
1. xt_truth.npz : the 'reference' dataset. 
2. values of metric $m$, computed on validation orbits of length 10, 15 and 25 Model Time Units (MTU, where 1 MTU = 200 integrations). They have been generated with 9 neural network (NN) models, for $F_b \in \lbrace 9.0, 9.25, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0 \rbrace$. 
There is one .npz file per validation length and value of $c$ used (i.e., c* or $c_0$). Each row corresponds to a given NN configuration, and each column to a value of $F_b$ (indexed from 9.0 to 11.0). 
3. Values of metric $m$ computed on the linear regression (baseline) model. 

The error files are used by Fig3.py. 