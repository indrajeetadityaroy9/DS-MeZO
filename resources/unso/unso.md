1

## UNSO: Unified Newton–Schulz Orthogonalization


Chen Hu, _Grad Student Member, IEEE_, Qianxi Zhao, Yuming Li, Mingyu Zhou, and Xiyin Li [*], _Member, IEEE_



_**Abstract**_ **—The Newton-Schulz (NS) iteration has gained in-**
**creasing interest for its role in the Muon optimizer and the**
**Stiefel manifold. However, the conventional NS iteration suffers**
**from inefficiency and instability. Although various improvements**
**have been introduced to NS iteration, they fail to deviate**
**from the conventional iterative paradigm, which could increase**
**computation burden largely due to the matrix products along**
**the long dimension repeatedly. To address this, we consolidate**
**the iterative structure into a unified framework, named Unified**
**Newton-Schulz Orthogonalization (UNSO). To do so, we could**
**avoid a polynomial expansion. Instead, we evaluate the role**
**of each matrix power, remove the insignificant terms, and**
**provide a recommended polynomial with learnable coefficients.**
**These learnable coefficients are then optimized, and achieve**
**an outstanding performance with stable convergence. The code**
**[of our method is available: https://github.com/greekinRoma/](https://github.com/greekinRoma/Unified_Newton_Schulz_Orthogonalization)**
**[Unified Newton Schulz Orthogonalization.](https://github.com/greekinRoma/Unified_Newton_Schulz_Orthogonalization)**


_**Index Terms**_ **—Newton-Schulz (NS) iteration, Unified Newton-**
**Schulz orthogonalization (UNSO), unified framework, Singular**
**value decomposition.**


I. INTRODUCTION

RTHOGONALITY is increasingly applied across various fields, including neural networks [1]–[3] and Rie# **O**
mannian optimization [4]–[7].
One of the key techniques for orthogonalization is the
Newton-Schulz (NS) iteration [8], [9], which replace singular
value decomposition (SVD) with the matrix multiplications.
The conventional Newton-Schulz iteration is given by


_Xk_ +1 = [1] �3 _I −_ _Xk_ _[⊤][X][k]_   - _,_ _X_ 0 _∈_ _R_ _[w][×][h]_ _._ (1)

2 _[X][k]_

_√_
When the largest singular value satisfies _σ_ max( _X_ 0) _<_ 3,

the iteration converges to the orthogonal factor of the polar
decomposition of _X_ 0, where _w_ and _h_ are the width and height
of _A_, respectively. Specifically, as _k_ +1 _→_ _N_, where _N_ is the
number of iterations, the sequence _{Xk}_ converges to _UV_ _[⊤]_ .
More generally, the NS iteration belongs to a broader family
of odd-degree polynomial iterations of order 2 _n_ +1, which can
be written as


_Xk_ +1 = _aXk_ + _bXkXk_ _[⊤][X][k]_ [+] _[ · · ·]_ [ +] _[ c]_ [(] _[X][k][X]_ _k_ _[⊤]_ [)] _[n][X][k][,]_ (2)


Manuscript received XXX XXX, XXX; revised XXX XXX, XXX.
_(Corresponding authors: Xiyin Li.)_
Chen Hu, Qianxi Zhao, Yuming Li, and Xiyin Li are with the School
of Intelligent Systems Engineering, the Sun Yat-sen University, Shenzhen, China (e-mail: zz8843157@gmail.com; zhaoqx9@mail2.sysu.edu.cn;
Liym297@mail2.sysu.edu.cn; stslxy@mail.sysu.edu.cn)
Mingyu Zhou is with the School of Information and Communication
Engineering, University of Electronic Science and Technology of China,
Chengdu 610054, China (e-mail: 202421011422@std.uestc.edu.cn).


|Col1|Iteration 1<br>.5 -0.5 3.4<br>+<br>Iteration k<br>Iteration N|Iteration 1<br>445 -4.775 2.0315 a<br>+<br>Iteraion k<br>Iteration N|Iteration 1<br>b c<br>+ P<br>Iteration k<br>Iteration N|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||Iterat|ion 1||
|**1**<br>|**1**<br>|**1**<br>|**1**<br>|Iteration k<br>**Computing Pk and**ek<br>**1-**ek<br>**1+**ek<br>**k**|||
|||||Iterat|ion N||
||||||||



Fig. 1: The existing NS iteration methods.


where the coefficients _a, b, . . ., c_ are chosen such that the
polynomial, as shown below:


_g_ ( _x_ ) = _ax_ + _bx_ [3] + _· · ·_ + _cx_ [2] _[n]_ [+1] _._ (3)


Let the singular value decomposition of _Xk_ be _Xk_ = _USkV_ _[⊤]_ .
Substituting this into Eq. (2) yields


_Xk_ +1 = _Ug_ ( _St_ ) _V_ _[⊤]_ = _Ug_ _[k]_ [+1] ( _S_ 0) _V_ _[⊤]_ _._ (4)


When _k_ + 1 _→_ _N_, the final function is shown below:


_XN_ = _Ug_ _[N]_ ( _S_ 0) _V_ _[⊤]_ = _U_ (msign( _S_ 0)) _V,_ (5)


where msign( _·_ ) is defined element-wise as


msign( _SN_ ) = diag(sign( _σ_ 1) _, . . .,_ sign( _σm_ )) _._ (6)


Here, _σ_ is the singular values of our input matrix _X_ 0 and _m_
denotes the number of singular values.
Building upon the original NS iteration, a number of studies
have been conducted to improve its performance. Higham
and Schreiber [10] proposed an adaptive hybrid iteration
that accelerates polar decomposition on modern hardware.
Subsequently, Nakatsukasa and Higham [11] point out NS
iteration is backward stable if the starting matrix ha _√_ s 2-norm
safely, but can be unstable if the norm is close to 3.

Based on above analysis, Keller Jordan [1] proposes a new
form of NS iteration:


_Xk_ +1 = _aXk_ + _b_ ( _XkXk_ _[⊤]_ [)] _[X][k]_ [+] _[ c]_ [(] _[X][k][X]_ _k_ _[⊤]_ [)][2] _[X][k][.]_ (7)


At first, he scales _A/∥A∥F_ so its singular values lie in [0 _,_ 1], to
ensure NS iteration converges to the polar factor _UV_ _[⊤]_ . Then,
Keller Jorden proposed a standard parameter set to achieve
quadratic convergence: ( _a, b, c_ ) = (3 _._ 4445 _, −_ 4 _._ 7750 _,_ 2 _._ 0315).
Finally, they repeat the iteration to approximate the _msign_ .
The Keller Ns iteration speed up the convergence of our matrix
but introduce high noise to the outputs.















(d) CANS.



(a) Origin NS iteration.



(b) Muon’s NS iteration.



(c) Cesista’s NS iteration.


2



**X**



II. METHOD





_A. Overall Structure_



The overall structure of the algorithm is shown in Fig. 2.
Firstly, we transform the input matrix _M ∈_ R _[h][×][w]_ into the
form:



_A_ =




_M_ _if h_ ⩽ _w_
(11)
_M_ _[⊤]_ _if h > w._


|Col1|Col2|
|---|---|
||**Transpose**|
|**SU**|**I**<br>**T**<br><br>**SUM**<br>**a1**<br>**a2**<br>**a3**<br>¼<br>**aN**<br>**M**|



Fig. 2: The overall of the unified NS iteration structure.


In order to solve this, the researchers [12] make coefficients
learnable and different per iteration step. The coefficients are
obtained by a reparameterized function:


_x_ k+1 = _xk_ + _γ ·_ ( _xk −f_ 0)( _xk −_ _f_ 1)( _xk −_ _f_ 2)( _xk −f_ 3)( _xk −f_ 4) _,_
(8)


_f_ = [ _−_ (1 + _r_ ) _, −_ (1 _−_ _l_ ) _,_ 0 _,_ 1 _−_ _l,_ 1 + _r_ ] _,_ (9)


where, _γ_, _r_, and _l_ are learnable weights, and optimized to
make the polynomial more efficient. Then, they compute _a_, _b_,
and _c_ using the obtained parameters _γ_, _r_, and _l_ .
Grishina et al. [13] propose a new polynomial derived from
the Chebyshev alternation theorem, and name it Chebyshevaccelerated Newton-Schulz (CANS). In addition, they also
utilize the Gelfand’s formula:


_σ_ 1( _X_ 0) ⩽ _||_ ( _X_ 0 _[⊤][X]_ [0][)] _[k][||]_ _F_ [1] _[/]_ [2] _[k]_ _,_ (10)


to approximate the maximum singular value and keep the
singular values falling into the desired range [0 _,_ 1].
In conclusion, the above methods are shown as in Fig. 1.
Although many algorithms have been proposed to improve
orthogonalization efficiency, they still rely on iterative structures, hindering the further improvements in efficiency due to
the repeated matrix multiplication along the long dimension.
We introduce a non-iterative method, named Unified Newton–Schulz Orthogonalization (UNSO), to improve efficiency
by replacing iterative steps with a single unified operation. To
avoid extra computation due to multiplying out, we analyze the
role of each matrix with varied power, ignore the unimportant
terms, and use learnable coefficients to get the optimized
polynomial. Our approach outperforms existing methods. To
sum up, the contributions of this manuscripts are below.


1. We replace iterative structures with a unified framework,
called UNSO.
2. We analyze the role of each matrix with varying powers,
remove negligible terms, and offer the recommended
polynomial to overcome the additional computation burdens introduced by the unified framework.
3. We set the coefficients of polynomial learnable and
demonstrate its outstanding performance compared with
other words.



In this way, the size of _A_ _[⊤]_ _A_ is minimized. Then, We firstly
scale our singular value of matrix into (0 _,_ 1), according to Eq.
(12).
_X_ = _A/||AA_ _[⊤]_ _||_ [1] _F_ _[/]_ [2] _[.]_ (12)


At last, we merge the multiple steps of the NS iteration into
a single, simplified operation, as shown in Eq. (13):



_Y_ = _X_ +



_N_ _−_ 1

- _ak_ ( _I −_ _XX_ _[⊤]_ ) [2] _[k][−]_ [1] _X_


_k_ =1



(13)




       + [ _e_ [1] _[/]_ [2][ �] 2 _[N/]_ [2] _−_ 1 _−_



_N_ _−_ 1

- _ak_ ]( _I −_ _XX_ _[⊤]_ ) [2] _[N]_ _[−]_ [1] _X._


_k_ =1



where _I_ denote the identify matrix and _N_ is the number of
terms of the polynomial.


_B. Unified Polynomial_


For all iterations of the NS method, the process can be expressed equivalently as a single unified operation by expanding
the original polynomial, shown as:


_XN_ = _aX_ 0 + _bX_ 0 _X_ 0 _[⊤][X]_ [0] [+] _[ · · ·]_ [ +] _[ c]_ [(] _[X]_ [0] _[X]_ 0 _[⊤]_ [)] _[n][X]_ [0] _[.]_ (14)


Furthermore, as designed, the expression approaches 1 when
the singular value _σ_ 0 of the input matrix tends to 1. To ensure
this behavior, we explicitly impose the following constraints:



_Y_ = _X_ +



_N_ _−_ 1

- _ak_ ( _I −_ _XX_ _[⊤]_ ) _[n][K]_ _X_ + _b_ ( _I −_ _XX_ _[⊤]_ ) _[n][N]_ _X._ (15)


_k_ =1



Each term in the Eq. (15) corresponds to repeatedly applying
the projection operator ( _I −_ _XX_ _[⊤]_ ) to _X_, yielding a polynomial structure. For the singular values, the contribution of the
_k_ -th term can be expressed as


_y_ = _fnk_ ( _x_ ) = _x_ (1 _−_ _x_ [2] ) _[n][k]_ _._ (16)


In terms of computational efficiency, this expansion reduces
the required multiplications along the long dimension from N
to 1. However, it adds significant new multiplications along
the short dimensions. To suppress this added expense, we
eliminate the insignificant terms.


_C. Term Selection_


We first examine the step-size selection for the parameter
_nk_ in Eq. (16), where _k ∈{_ 0 _, . . ., N_ _}_ . As shown in Fig. 3,
the induced curves exhibit substantially different behaviors
under two parameter growth strategies: linear increments and
exponential growth. In particular, exponential growth produces


3













|1.0|Col2|k=1 k=2|
|---|---|---|
|0.0<br>0.2<br>0.4<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>y||k=3<br>k=4<br>k=10<br>k=16|
|0.0<br>0.2<br>0.4<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>y||0.6<br>0.8<br>1.0|


(b)





(a)

















































(a) Linear Growth



(b) Exponential Growth



Fig. 3: Different parameter _nk_ pattern.


significantly faster and more pronounced shape variations than
linear increments.
Based on this observation, we adopt an exponential parameterization, defined as





_nk_ =




0 _,_ _k_ = 0 _,_
(17)
2 _[k][−]_ [1] _,_ _k_ = 1 _, . . ., N._



Under this parameterization, Eq. (16) can be rewritten as



_y_ = _fk_ ( _x_ ) =




- _x,_ _k_ = 0 _,_

(18)
_x_ �1 _−_ _x_ [2][�][2] _[k][−]_ [1] _,_ _k_ = 1 _, . . ., N._



We compute the gradient of Eq. (18). When _k_ = 0, the gradient
of _f_ 0( _x_ ) is 1. when _k >_ 0, the gradient is given by Eq. (19).

_fk_ _[′]_ [(] _[x]_ [) =] �1 _−_ _x_ [2][�][2] _[k][−]_ [1] _[−]_ [1][ �] 1 _−_ (2 _[k]_ + 1) _x_ [2][�] _._ (19)


For _k >_ 0, the function _fk_ ( _x_ ) has an extreme point within
the interval (0 _,_ 1), given by the expression



1
_x_ _[∗]_ = ~~_√_~~ _,_

2 _[k]_ + 1



Fig. 4: Family of Curves with Increasing _k_ . (a) Extreme point
values of _x∗_ and _y∗_ ; (b) The normalized _y_ .


**Algorithm 1:** Our Method

**Input:** Matrix _X ∈_ R _[n][×][n]_ ;
Polynomial order _N_ ;
Learned coefficients _{ak}_ _[N]_ _k_ =1 _[−]_ [1][.]
**Output:** Updated matrix _Y_
**Pre-compute coefficient:**
_c ←_ _e_ [1] _[/]_ [2][ �] 2 _[N/]_ [2] _−_ 1� _−_ [�] _[N]_ _k_ =1 _[−]_ [1] _[|][a][k][|]_
**Initialization:**
_I ←_ identity matrix of size _n_
_X_ 1 _←_ _I −_ _XX_ _[⊤]_

**Polynomial expansion:**
**for** _k_ = 2 **to** _N_ **do**

_Xk ←_ _Xk−_ 1 _· Xk−_ 1
**Polynomial aggregation:**
_Y ←_ _I_
**for** _k_ = 1 **to** _N −_ 1 **do**

_Y ←_ _Y_ + _akXk_
_Y ←_ _Y_ + _cXN_
**Final update:**
_Y ←_ _Y X_
**return** _Y_


For large _N_, this exact expression admits a simpler approximation:



1 2 _[k]_

(
2 _[k]_ + 1 2 _[k]_



_e_ _[−]_ 2 [1]
2 _[k]_ + 1



(20)

[1] 2 _._



1
_y_ _[∗]_ = ~~_√_~~



2 _[k]_ 1

~~_√_~~
2 _[k]_ + 1 [)][2] _[k][−]_ [1] _[ ≈]_ 2 _[k]_



When _k_ increases, the extreme point would be closer to
0, meaning the polynomial would be a sharper and more
localized peak, as shown in Fig. 4.


_D. Coefficient Optimization_


According to above analysis, the summation formula of each
term is




       _b ≈_ _e_ [1] _[/]_ [2][ �] 2 _[N/]_ [2] _−_ 1 _−_



_N_ _−_ 1

- _ak._ (24)


_k_ =1



_N_ _−_ 1

_f_ ( _x_ ) = _x_ + - _akx_ �1 _−_ _x_ [2][�][2] _[k][−]_ [1] + _bx_ (1 _−_ _x_ [2] ) [2] _[N]_ _[−]_ [1] _._ (21)


_k_ =1



The coefficients _ak_ are learnable through optimization, allowing the function _f_ ( _._ ) to closely approximate sign( _._ ) without
strong fluctuation.


_E. Matrix Iteration_


The polynomial in Eq. (21) is defined for scalar inputs and
can therefore be directly applied to the singular values of a
matrix, as shown in Eq. (4):


_Y_ = _Uf_ ( _S_ ) _V_ = _U_ (diag� _f_ ( _σ_ 0) _, . . ., f_ ( _σm_ )�) _V_ (25)



As illustrated in Fig. 4, the _N_ -th term is the first to converge
to 1. Hence, we set the _x_ -coordinate of the extreme point of
_fN_ ( _·_ ) as the location where the Eq. (21) first reaches 1, and
obtain the following constraints:


1
_f_ ( ~~_√_~~ ) = 1 (22)

2 _[N]_ + 1


According to Eq. (22) and (21), _b_ could be represented as
follows:





_b_ =  [�]



2 _[N]_ _[−]_ [1]




2 _[k][−]_ [1] []





 - 2 _N_ + 1




2 _[N]_



2 _[N]_ + 1 _−_ 1 _−_



_N_ _−_ 1

- _ak_


_k_ =1




- 2 _[N]_

2 _[N]_ + 1



_._



(23)



where _m_ is equal to _h_ . Consequently, Eq. (13) can be
obtained. In practice, the above formulation is implemented by _._
explicitly constructing each polynomial term through matrix
multiplications, as summarized in Algorithm 1.


Fig. 5: The optimized curve with the different _N_ .







Fig. 6: The curves based on different algorithms.


III. EXPERIMENT


_A. Experimental Setup_


All experiments are implemented in Python 3.10 using
PyTorch 2.1 and conducted on a single GPU. The experiments
are performed on a Windows system with CUDA 11.8 for
GPU acceleration. We adopt the Adam optimizer with an
initial learning rate of 10 _[−]_ [1] . The learning rate is decayed by
a factor of 0 _._ 5 every 10000 iterations using a step scheduler.
The model is trained for 20000 epochs. At each iteration, the
loss is computed by uniformly sampling 1000 points from the
interval (0 _,_ 1) to approximate the expectation.


_B. Ablation Study_


We study the effect of the polynomial depth _N_ on approximation performance, as shown in Fig. 5. By varying _N_
while keeping all other training settings fixed, we observe that
increasing _N_ consistently improves the expressive capacity
of the model, leading to lower approximation error on the
target interval (0 _,_ 1). However, beyond a certain depth, the
performance gain saturates, while the computation burden
gradually increases. Therefore, we set 14 as our default _N_
for the matrix. These results indicate that a moderate value of
_N_ provides the best trade-off between approximation accuracy
and stability.


_C. Comparative Experiments_


We compare our proposed methods against Origin NS,
Muon’s NS [1], Cesista’s NS [12], and the CANS algorithm

[13].
_1) Curve-Based Comparative Evaluation:_ We perform a
curve-based comparative evaluation to assess both the functional behavior and practical computational cost of different
curve construction methods As shown in Fig. 6, our curves are
compared with Origin other curve over the same input domain.
While all methods share similar global trends, our approach



4


TABLE I: Performance of the proposed method with increasing matrix size


**Method** **Size (** _h × w_ **)** **Error** _↓_ **FLOPs** _↑_


128 _×_ 128 0 _._ 4869 6 _._ 750 _×_ 10 [7]
Original NS 128 _×_ 512 0 _._ 4884 2 _._ 696 _×_ 10 [8]

128 _×_ 1024 0 _._ 4970 5 _._ 391 _×_ 10 [8]

128 _×_ 128 3 _._ 846 6 _._ 332 _×_ 10 ~~[7]~~
Muon’s NS 128 _×_ 512 3 _._ 838 2 _._ 533 _×_ 10 [8]

128 _×_ 1024 3 _._ 844 5 _._ 066 _×_ 10 [8]

128 _×_ 128 0 _._ 330 6 _._ 332 _×_ 10 ~~[7]~~
Cesista’s NS 128 _×_ 512 0 _._ 330 2 _._ 533 _×_ 10 [8]

128 _×_ 1024 0 _._ 330 5 _._ 066 _×_ 10 [8]

128 _×_ 128 1 _._ 311 6 _._ 332 _×_ 10 ~~[7]~~
CANS 128 _×_ 512 1 _._ 309 2 _._ 533 _×_ 10 [8]

128 _×_ 1024 1 _._ 311 5 _._ 066 _×_ 10 [8]

128 _×_ 128 0 _._ 040 6 _._ 314 _×_ 10 ~~[7]~~
Ours 128 _×_ 512 0 _._ 040 8 _._ 831 _×_ 10 [7]

128 _×_ 1024 0 _._ 040 1 _._ 219 _×_ 10 [8]


reach 1 when _x_ = 0 _._ 005 and have less fluctuation. Moreover,
it requires fewer matrix multiplications (14 and 9) than NSbased baselines (15–16), achieving comparable curve quality
with lower computational cost. Finally, our efficient method
achieves better performance, reaching 1 faster.

_2) Comparative Evaluation of Practical Matrix Operations:_
As illustrated in Tab. I, our method demonstrates superior efficiency. This advantage stems from its lower cost of floatingpoint computations, achieved by reducing the amount of
computation along the _w_ -axis. Specifically, as _w_ increases,
the computational burden of our method grows more slowly
compared to other methods. Additionally, we also examine the
effectivity among various algorithms by:



Our method also achieves superior orthogonalization performance.


IV. CONCLUSION


In this paper, we propose a new method, termed UNSO,
which reduces the repeated computation of _X ∈_ **R** _[h][×][w]_, a
significantly more expensive operation than matrix multiplication involving _X_ _[⊤]_ _X ∈_ **R** _[h][×][h]_, with _h_ ⩽ _w_ . Experiments
indicate our methods achieves the outstanding performance
with less computation burden. Despite these advantages, there
are several limitation that warrant further investigation. Firstly,
our method converge to 1 at a slower rate, compared with
other methods. Secondly, noticeable fluctuations are observe
when the propose method first converge to 1. Finally, the
matrix multiplications involving _X_ is still required, leading
to high computation cost when _w ≫_ _h_ . In the future, we
aim to overcome these limitation by introducing the dynamic
coefficient to enable a more flexible term structure. Besides,
we plan to explore neural network-based techniques to reduce
the dimensionality of _w_ .



_h_

- _Ei,j_ [2] _where_ _E_ = _Y_ _[⊤]_ _Y −_ _I._ (26)

_j_ =0



**Error** =




~~�~~

~~�~~ _h_
��



_i_ =0


5



REFERENCES


[1] K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and
J. Bernstein, “Muon: An optimizer for hidden layers in neural networks,”
[2024. [Online]. Available: https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)

[2] J. Bernstein and L. Newhouse, “Modular duality in deep learning,” _arXiv_
_preprint arXiv:2410.21265_, 2024.

[3] T. Pethick, W. Xie, K. Antonakopoulos, Z. Zhu, A. Silveti-Falls, and
V. Cevher, “Training deep learning models with norm-constrained lmos,”
_arXiv preprint arXiv:2502.07529_, 2025.

[4] P.-A. Absil, R. Mahony, and R. Sepulchre, _Optimization algorithms on_
_matrix manifolds_ . Princeton University Press, 2008.

[5] L. Tunc¸el, “Optimization algorithms on matrix manifolds,” 2009.

[6] J. Chen, H. Ye, M. Wang, T. Huang, G. Dai, I. W. Tsang, and Y. Liu,
“Decentralized riemannian conjugate gradient method on the stiefel
manifold,” _arXiv preprint arXiv:2308.10547_, 2023.

[7] B. Gao, N. T. Son, P.-A. Absil, and T. Stykel, “Riemannian optimization
on the symplectic stiefel manifold,” _SIAM Journal on Optimization_,
vol. 31, no. 2, pp. 1546–1575, 2021.

[8] ˚A. Bj¨orck and C. Bowie, “An iterative algorithm for computing the best
estimate of an orthogonal matrix,” _SIAM Journal on Numerical Analysis_,
vol. 8, no. 2, pp. 358–364, 1971.

[9] Z. Kovarik, “Some iterative methods for improving orthonormality,”
_SIAM Journal on Numerical Analysis_, vol. 7, no. 3, pp. 386–389, 1970.

[10] N. J. Higham and R. S. Schreiber, “Fast polar decomposition of an
arbitrary matrix,” _SIAM Journal on Scientific and Statistical Computing_,
vol. 11, no. 4, pp. 648–655, 1990.

[11] Y. Nakatsukasa and N. J. Higham, “Backward stability of iterations for
computing the polar decomposition,” _SIAM Journal on Matrix Analysis_
_and Applications_, vol. 33, no. 2, pp. 460–479, 2012.

[12] F. L. Cesista, Y. Jiacheng, and K. Jordan, “Squeezing 1-2% efficiency
gains out of Muon by optimizing the Newton-Schulz Coefficients,”
[February 2025. [Online]. Available: https://leloykun.github.io/ponder/](https://leloykun.github.io/ponder/muon-opt-coeffs/)
[muon-opt-coeffs/](https://leloykun.github.io/ponder/muon-opt-coeffs/)

[13] E. Grishina, M. Smirnov, and M. Rakhuba, “Accelerating newton-schulz
iteration for orthogonalization via chebyshev-type polynomials,” _arXiv_
_preprint arXiv:2506.10935_, 2025.


