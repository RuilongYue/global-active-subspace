# The Global Active Subspace Method
## Introduction
### GAS method
Global active subspace (GAS) method <sup>[1]</sup> is a generalization of active subspace (AS) method by replacing the matrix C with more "global" information. Theoretical results and experiments show that GAS is better when the test function has uncertainties, or it's discontinuous, and is giving similar results for ordinary cases with AS method.

Consider a square-integrable real-valued function $f(\pmb z)$ with domain $\Omega\subset\pmb R^d$ and finite second-order partial derivatives. Suppose that $\Omega$ is equipped with a probability measure with a cumulative distribution function in the form $\pmb F(\pmb z)=F_1(z_1)\cdot \ldots \cdot F_d(z_d)$, where $F_i$ are marginal distribution functions. 

Define a vector function $D_{\pmb z}f:\Omega\times\Omega\rightarrow\pmb R^d$ as follows: 
$$D_{\pmb z}f(\pmb v,\pmb z)=[D_{\pmb z,1}f(v_{1},\pmb z),...,D_{\pmb z,d}f(v_{d},\pmb z)]^T,$$
where
$$D_{\pmb z,i}f(v_{i},\pmb z)=(f(\pmb v_{\lbrace i\rbrace},\pmb z_{-\lbrace i\rbrace})-f(\pmb z))/(v_{i}-z_{i}).$$

Here $\pmb v_{\lbrace i\rbrace}$ corresponds to the $i$ th input of vector $\pmb v$, and $\pmb z_{-\lbrace i\rbrace}$ is the vector of inputs corresponding to those indices in the complement of ${\lbrace i\rbrace}$. 

Define the $d\times d$ matrix $\pmb C$ by
$$\pmb C=E[E[(D_{\pmb z}f)(D_{\pmb z}f)^T|\pmb z]].$$

Do the eigenvalue decomposition of $\pmb C$ and partition the eigenvalues and eigenvectors into two bloacks, we get the global active subspace method.
$$\pmb C=\pmb U\Lambda \pmb U^T,\Lambda=diag(\lambda_1,...,\lambda_d), \lambda_1\geq...\geq \lambda_d\geq 0.$$

$$
\Lambda=\left[
\begin{array}{cc}
    \Lambda_1 &  \\
     & \Lambda_2
\end{array}
\right],   
\pmb U=\left[
\begin{array}{cc}
    \pmb{U}_1 &  \pmb{U}_2
\end{array}
\right].$$


To estimate the matrix $\pmb C$, we use the following approximation:
$$\pmb{\hat C}=\frac1 {M_1M_2}\sum_{i=1}^{M_1}\sum_{j=1}^{M_2}(D_{\pmb z^{(i)}}f(\pmb v^{(i,j)}, \pmb z^{(i)}))(D_{\pmb z^{(i)}}f(\pmb v^{(i,j)}, \pmb z^{(i)}))^T.$$

### PCE model
Polynomial chaos expansion (PCE) model is used as a surrogate model approximating functions by data. The PCE of a square-integrable variable $Y=f(\pmb z)$ is,
$$f(\pmb z)=\sum_{i=0}^{\infty}{k_i\phi_i(\pmb z)}.$$

$\lbrace\phi_i\rbrace_{i=0}^{\infty}$ is a family of multidimensional orthonormal polynomials with respect to a weight function. In practice, one needs to estimate the coefficients $k_i$, and compute 
$$\hat f_p(\pmb z)=\sum_{i=0}^{P-1}{\hat k_i\phi_i(\pmb z)},$$

where $P=\binom{d+p} d$ is the number of terms in the summation, and $\hat k_i$ is the estimated value for $k_i$. A popular method in the literature to estimate $\hat k_i$ is the least squares method. Let $\pmb{\Psi}\in\pmb{R}^{N\times P}$ be the coefficient matrix with $\Psi_{ni}=\phi_i(\pmb{z}^{(n)})$, and $\pmb{y}\in\pmb{R}^{N}$ be the response vector with $y_n=f(\pmb{z}^{(n)})$. The least squares approach computes 
$$\hat{\pmb{k}}=(\pmb{\Psi}^T\pmb{\Psi})^{-1}\pmb{\Psi}^T\pmb{y},$$

which minimizes the mean square error $||\pmb{y}-\pmb{\Psi}\pmb{k}||^2$. Given $\hat{\pmb{k}}$, the expectation $E[f(\pmb z)]$ is approximated by $\hat k_0$.


## Usage
In this repo, there is one .py file implementing Algorithm 2.1 in Yue and Okten <sup>[1]</sup>. The other four python notebook files cover the detailed implementation of the four numerical experiments in the paper.

To apply the global active subspace method, include the file global_as.py and use the function GAS(). 

>GAS(Func, dim, chi, M1, M2, shiftedSobol=True, distribution='normal'):
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
>
>distribution: default value is 'normal'. Can be changed to 'uniform' based on distribution in the problem.

To construct PCE model and estimate $E[f(\pmb z)]$, use the function GAS_PCE() in global_as.py.

>GAS_PCE(Func, Num_exp, z1, dim1, u, exponents, coefficients, P):
>
>Func: the function $f(\pmb z)$.
>
>Num_exp: number of experiments. Usually takes value such as $40$.
>
>z1: list of sampled points with length to be Num_exp.
>
>dim1: dimension of the reduced space, $d_1$.
>
>u: eigenvector matrix derived from the function GAS(). 
>
>exponents: exponent list of the polynomial chaos, see the package numpoly.
>
>coefficients: coefficient list of the polynomial chaos, see the package numpoly.
>
>P: number of terms of the polynomial chaos.


## References

[1]. The Global Active Subspace Method. (https://arxiv.org/abs/2304.14142)
