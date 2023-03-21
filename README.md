# global-active-subspace
## Introduction
Global active subspace (GAS) method is a generalization of active subspace (AS) method by replacing the matrix C with more "global" information. Theoretical results and experiments show that GAS is better when the test function has uncertainties, or it's discontinuous, and is giving similar results for ordinary cases with AS method.

Consider a square-integrable real-valued function $f(\pmb z)$ with domain $\Omega\subset\pmb R^d$ and finite second-order partial derivatives. Suppose that $\Omega$ is equipped with a probability measure with a cumulative distribution function in the form $\pmb F(\pmb z)=F_1(z_1)\cdot \ldots \cdot F_d(z_d)$, where $F_i$ are marginal distribution functions. 

Define a vector function $D_{\pmb z}f:\Omega\times\Omega\rightarrow\pmb R^d$ as follows: 
$$D_{\pmb z}f(\pmb v,\pmb z)=[D_{\pmb z,1}f(v_{1},\pmb z),...,D_{\pmb z,d}f(v_{d},\pmb z)]^T,$$
where
$$D_{\pmb z,i}f(v_{i},\pmb z)=(f(\pmb v_{\{i\}},\pmb z_{-\{i\}})-f(\pmb z))/(v_{i}-z_{i}).$$

Here $\pmb v_{\{i\}}$ corresponds to the $i$th input of vector $\pmb v$, and $\pmb z_{-\{i\}}$ is the vector of inputs corresponding to those indices in the complement of $\{i\}$. 

Define the $d\times d$ matrix $\pmb C$ by
$$\pmb C=E[E[(D_{\pmb z}f)(D_{\pmb z}f)^T|\pmb z]].$$

Do the eigenvalue decomposition of $\pmb C$ and partition the eigenvalues and eigenvectors into two bloacks, we get the global active subspace method.
$$\pmb C=\pmb U\Lambda \pmb U^T,\Lambda=diag(\lambda_1,...,\lambda_d), \lambda_1\geq...\geq \lambda_d\geq 0.$$

To estimate the matrix $\pmb C$, we use the following approximation:
$$\hat{\pmb C}=\frac1 {M_1M_2}\sum_{i=1}^{M_1}\sum_{j=1}^{M_2}(D_{\pmb z^{(i)}}f(\pmb v^{(i,j)}, \pmb z^{(i)}))(D_{\pmb z^{(i)}}f(\pmb v^{(i,j)}, \pmb z^{(i)}))^T.$$


## Usage
To use the global active subspace method, include the global_as.py and use the function GAS(). 

>GAS(Func, dim, chi, M1, M2=10, shiftedSobol):
>
>Func: an arbitrary function takes a chi $\times$ dim input matrix as input, and gives an output vector of length chi.
>
>dim: dimension of the input function, $d$.
>
>chi: sample size when approximating $\pmb C$, chi $=M_1M_2$.
>
>M1, M2: parameters used when approximating $\pmb C$.
>
>shiftedSobol: a bool value determines whether we use the shifted Sobol' sequence when generating $v^{(i,j)}$'s. 

To estimate $E[f(\pmb z)]$, use the function GAS_PCE().

>def GAS_PCE(Func, Num_exp, N, p, dim, dim1, u): 
>
>Func: the function $f(\pmb z)$.
>
>Num_exp: number of experiments. Usually takes value such as 40.
>
>N: sample size when constructing PCE.
>
>p: the highest degree of PCE.
>
>dim: dimension of the input function, $d$.
>
>dim1: dimension of the reduced space, $d_1$.
>
>u: eigenvector matrix derived from the function GAS(). 
