## The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm

Noah Amsel [∗] David Persson [†] Christopher Musco [‡] Robert M. Gower [§]


September 29, 2025


**Abstract**


Computing the polar decomposition and the related matrix sign function has been
a well-studied problem in numerical analysis for decades. Recently, it has emerged as
an important subroutine within the `Muon` algorithm for training deep neural networks.
However, the requirements of this application differ sharply from classical settings:
deep learning demands GPU-friendly algorithms that prioritize high throughput over
high precision. We introduce `Polar Express`, a new method for computing the
polar decomposition. [1] Like Newton–Schulz and other classical polynomial methods,
our approach uses only matrix-matrix multiplications, making it very efficient on
GPUs. Inspired by earlier work of Chen & Chow and Nakatsukasa & Freund, `Polar`
`Express` adapts the update rule at each iteration by solving a minimax optimization
problem. We prove that this strategy minimizes error in a worst-case sense, allowing
`Polar Express` to converge as rapidly as possible both in the early iterations and
asymptotically. We also address finite-precision issues, making it practical to use in
`bfloat16` . When integrated into the `Muon` training framework, our method leads to
consistent improvements in validation loss when training a GPT-2 model on one billion
tokens from the FineWeb dataset, outperforming recent alternatives across a range of
learning rates.

### **1 Introduction**


Advanced linear algebra is making its way into deep learning. Efficient algorithms for
computing _matrix functions_ have found exciting new applications in training neural
networks. In particular, approximations to the matrix-inverse are used in the full Adagrad
method [15], the matrix square-root and quarter-root appear as subroutines in the Shampoo
and Soap optimizers [20, 48, 51], and most recently, the matrix sign function has become a
key ingredient of the `Muon` optimizer [5, 4, 25].
While the problem of computing these matrix functions has been studied by numerical
analysts for decades, applications in deep learning come with different requirements than
those in computational science. For deep learning, it is critical to take maximum advantage
of GPU-friendly operations like matrix-matrix products and to avoid less parallel operations.
Moreover, memory overhead must be small to handle large models. On the other hand,


∗New York University. `noah.amsel@nyu.edu`
†New York University and Flatiron Institute. `dup210@nyu.edu`, `dpersson@flatironinstitute.org`
‡New York University. `cmusco@nyu.edu`
§Flatiron Institute. `rgower@flatironinstitute.org`
1 `[https://github.com/NoahAmsel/PolarExpress](https://github.com/NoahAmsel/PolarExpress)`


1


**Algorithm 1** Python code for the `Polar Express` of degree = 5.

```
from itertools import repeat
import torch

coeffs_list = [
   (8.28721201814563, -23.595886519098837, 17.300387312530933),
   (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
   (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
   (3.3184196573706015, -2.488488024314874, 0.51004894012372),
   (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
   (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
   (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
   (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5)
          for (a, b, c) in coeffs_list [: -1]] + [coeffs_list [ -1]]

@torch. compile
def PolarExpress(G: torch.Tensor, steps: int ) -> torch.Tensor:
   assert G.ndim >= 2
   X = G.bfloat16 () # for speed
   if G.size (-2) > G.size (-1): X = X.mT # this reduces FLOPs
   X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 +1e-7)
   hs = coeffs_list [: steps] + list (
     repeat(coeffs_list [-1], steps - len (coeffs_list)))
   for a, b, c in hs:
     A = X @ X.mT
     B = b * A + c * A @ A
     X = a * X + B @ X # X <- aX + bXˆ3 + cXˆ5
   if G.size (-2) > G.size (-1): X = X.mT
   return X

```

high accuracy is typically less important; the gold standard of sixteen digits of accuracy is
overkill in deep learning.
Given these considerations, there is a need to develop new matrix function methods
that are tailor-made for deep learning applications. We take on this challenge by designing
a state-of-the-art, GPU-friendly algorithm for computing the matrix sign function, or
more generally, for computing the _polar decomposition_ of a rectangular matrix. We apply
our new `Polar Express` method (Algorithm 1) to compute the descent direction in the
increasingly popular `Muon` optimizer. In Figure 1, we show that using `Polar Express`
within `Muon` consistently results in lower validation loss across all learning rates when
training a GPT-2 model, as compared to other matrix sign methods [10, 36, 25].


2


6 _._ 5


6 _._ 0


5 _._ 5


5 _._ 0


4 _._ 5


4 _._ 0


3 _._ 5



3 _._ 46


3 _._ 44


3 _._ 42


3 _._ 40


3 _._ 38


3 _._ 36


3 _._ 34




|Col1|muon-You|
|---|---|
||~~muon-PolarExp~~<br>muon-Jordan|
|||
|||
|||
|||
|||



10 _[−]_ [2]



Learning Rate



0 _._ 0 0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8
Epoch



Figure 1: Training a GPT-2-Large model (774M params) on 1 billion tokens from the
FineWeb dataset [2]. The label muon- `<name>` refers to implementing `Muon` using `<name>` to
compute the polar factor. Left: final validation loss across learning rates. Right: validation
loss across epochs using the best learning rate. The best learning rate ( _lr_ ) and final
validation loss for each method was `adamw` ( _lr_ = 0 _._ 0001): 4 _._ 172, `muon-You` ( _lr_ = 0 _._ 02):
3 _._ 400, `muon-Jordan` ( _lr_ = 0 _._ 02): 3 _._ 398 and `muon-PolarExp` ( _lr_ = 0 _._ 02): 3 _._ 340.


**1.1** **The Muon Method**


The `Muon` optimizer has recently gained popularity for training large language models,
often outperforming state-of-the-art adaptive gradient methods like Adam and AdamW

[30, 35]. `Muon` has been used to set records for the NanoGPT speedrun [25], to expand the
Pareto frontier of performance versus training FLOPs for large language models [34, 47],
and even to train a 32-billion parameter frontier LLM [29].
The `Muon` update rule [5] is defined as follows. Let _λ, β >_ 0 be the learning rate and
momentum coefficient hyperparameters. (By default, _β_ = 0 _._ 9.) Let _**W**_ _t ∈_ R _[m][×][n]_ be the
weight matrix of a given neural network layer at iteration _t_, and let _**G**_ _t ∈_ R _[m][×][n]_ be its
(stochastic) gradient. Let _**M**_ _t ∈_ R _[m][×][n]_ be the running momentum estimate of the gradient,
where _**M**_ 0 = **0** . The `Muon` update is given by


_**M**_ _t_ = _β_ _**M**_ _t−_ 1 + (1 _−_ _β_ ) _**G**_ _t_
_**W**_ _t_ +1 = _**W**_ _t −_ _λ_ polar( _**M**_ _t_ ) _._


Whereas standard stochastic gradient descent (SGD) with momentum updates the weight
matrix by taking a step in the direction _−_ _**M**_ _t_, the `Muon` method steps in the direction

_−_ polar( _**M**_ _t_ ), where polar( _**M**_ ) denotes the closest semi-orthogonal matrix to _**M**_ [22,
Chapter 8]. Concretely, if _**M**_ = _**U**_ **Σ** _**V**_ [T] is the singular value decomposition (SVD) of _**M**_,
then
polar( _**M**_ ) := _**UV**_ [T] _._ (1)


The matrix polar( _**M**_ ) can be seen as a generalization of the matrix sign function to
rectangular matrices [3]. Indeed, when _**M**_ is square symmetric with eigendecomposition
_**M**_ = _**V**_ **Λ** _**V**_ [T], polar( _**M**_ ) exactly coincides with the matrix sign function sign( _**M**_ ) =
_**V**_ sign( **Λ** ) _**V**_ [T] [22, Chapter 5]. Equivalently, polar( _**M**_ ) is the left orthogonal factor of the
polar decomposition of _**M**_ [22, Chapter 8]. The motivation for `Muon` is that _−_ polar( _**M**_ )
gives the steepest-descent direction with respect to the _spectral norm_ (instead of the
Frobenius norm, as in standard SGD).


3


Recent work [45] shows that `Muon` can be viewed as a conditional gradient (Frank-Wolfe)
method with a trust region defined by the spectral norm. In the same work, the authors
also provide a convergence theory for the smooth and non-convex setting, as well as for the
stochastic non-convex case. The analysis of `Muon` was further refined in [46], which proves
convergence under a layerwise ( _L_ 0 _, L_ 1)-smoothness assumption, in both the stochastic
non-convex and stochastic Polyak–�Lojasiewicz settings. We also note earlier work that
anticipated `Muon` ’s use of the polar factor and its motivation as steepest descent under
the spectral norm [8, 9]. We refer the reader to [25] and [5] for further background. In
this paper, we take the `Muon` update rule as given and focus on the problem of efficiently
computing the polar decomposition polar( _**M**_ ).


**1.2** **Computing the Polar Factor**


Although polar( _**M**_ ) can be computed directly via an SVD in _O_ (min( _mn_ [2] _, nm_ [2] )) time,
doing so is prohibitively expensive in deep learning applications, especially as standard SVD
algorithms fail to take full advantage of the parallelism available on GPUs. There has been
significant work on highly-parallel methods for the SVD, but the most common approaches
actually require computing the matrix-sign function as a subroutine [38, 40]. Numerical
analysts have spent decades developing iterative methods for computing polar( _**M**_ ). This
rich line of work includes Newton–Schulz [22, Chapter 8], Pad´e iteration [28, 21], the
Newton and scaled Newton iterations [22, Chapter 8], the QWHD iteration [37, 40], and
_Zolo-pd_ (Zolotarev polar decomposition) [38]. Unfortunately, as discussed in Section 2, most
of these methods are based on rational approximations to the function sign( _x_ ) and require
computing matrix inverses or QR decompositions. Such methods are ill-suited to GPU
acceleration and deep learning applications. In contrast, the older Newton-Schulz method is
based on _polynomial_ approximation of sign( _x_ ) and uses only matrix-matrix products. Thus,
`Muon` initially used Newton-Schulz [4]. Indeed, `Muon` stands for “MomentUm Orthogonalized
by Newton-Schulz” [25].


**The Newton-Schulz methods.** Newton-Schulz constructs a sequence of approximations
_**X**_ _t ≈_ polar( _**M**_ ) as follows:



_**X**_ 0 = _**M**_ _/∥_ _**M**_ _∥_ F _**X**_ _t_ +1 = [3]




[3]

2 _**[X]**_ _[t][ −]_ [1] 2



_t_ _**[X]**_ _[t]_ (2)
2 _**[X]**_ _[t]_ _**[X]**_ _[⊤]_



At each iteration, this rule effectively applies the cubic polynomial _p_ ( _x_ ) = [3]




[3]

2 _[x][ −]_ [1] 2




[1]

2 _[x]_ [3]



to each singular value of _**X**_ _t_ . The scalar fixed-point iteration _xt_ +1 = _p_ ( _xt_ ) converges
to sign( _x_ 0) as _t →∞_, provided _|x_ 0 _| ≤_ 1. As a result, the matrix iteration satisfies
lim
_t→∞_ _**[X]**_ _[t]_ [ =] _**[ UV]**_ _[ ⊤]_ [=][ polar][(] _**[X]**_ [0][). Higher-degree versions of Newton-Schulz follow the same]

principle. For example, the degree-5 polynomial _p_ ( _x_ ) = (15 _x−_ 10 _x_ [3] +3 _x_ [5] ) _/_ 8 converges even
faster. The Newton-Schulz iterations converge super-exponentially when _**X**_ _t_ is sufficiently
close to polar( _**M**_ ), but they suffer from slow initial convergence; when _**X**_ 0 is far from
polar( _**M**_ ), the approximation improves slowly over the first few iterations.


**The Jordan and You methods.** In `Muon`, high accuracy approximations to polar( _**M**_ )
are usually not necessary. The primary goal is instead to compute a coarse approximation
in as few iterations as possible. To accelerate convergence in the low-accuracy regime,
Jordan recently proposed a fixed-point iteration based on the polynomial _p_ ( _x_ ) = 3 _._ 4445 _x −_


4


4 _._ 7750 _x_ [3] + 2 _._ 0315 _x_ [5] [25], which was found using a heuristic numerical search. Unlike
Newton-Schulz, the scheme that Jordan proposed does not converge to polar( _**M**_ ). Instead,
it plateaus at an error of _≈_ 0 _._ 3. However, it reaches this level of accuracy rapidly. As a
result, when the number of iterations is smaller than about 10, Jordan’s method outperforms
the Newton-Schulz iteration. Building on this idea, You [10] proposed a method that
applies six different polynomial updates in succession, which were again found by heuristic
search. This method achieves better accuracy than Jordan’s but still fails to converge.
We introduce a new method. In particular, we derive polynomial update rules that are
_optimal_ at every iteration, outperforming all previous polynomial methods in our setting.


**1.3** **Contributions**


We present `Polar Express` [2], an iterative method for approximating polar( _**M**_ ). Our
method dynamically adapts the polynomial update rule at each iteration, prioritizing rapid
progress in the initial stage and high accuracy in the later stage. `Polar Express` constructs
polynomials _p_ 1 _, . . ., pT_ so that the resulting composition is the optimal approximation to
the sign function with respect to the supremum ( _L_ _[∞]_ ) norm (Theorem 4.1). By iteratively
applying these polynomials to _**M**_, `Polar Express` computes an approximation to polar( _**M**_ )
that is optimal in the worst-case at every iteration. Our method converges to polar( _**M**_ )
super-exponentially (Theorem 4.3), and it quickly reaches a good approximation within just
five to ten iterations. This early-stage acceleration is especially valuable in deep learning
applications, where runtime efficiency takes precedence over high accuracy. In contrast,
classical methods like Newton-Schulz suffer from a slow initial convergence, while recent
heuristic proposals [25, 10] fail to converge. Our method is efficient to run on GPUs, using
only a few matrix-matrix products per iteration. [3]

We give an explicit instantiation of `Polar Express` in Algorithm 1, which incorporates
minor modifications to make it compatible with half-precision arithmetic (see Section 4.4).
Algorithm 1 can be used as a drop-in replacement for previous methods. In numerical
experiments, `Polar Express` outperforms previous methods on synthetic matrices and
gradient matrices from a GPT-2 transformer (Figure 4). We demonstrate the effectiveness
of using `Polar Express` within the `Muon` optimizer in Figure 1, showing that it consistently
improves the training of GPT-2 language models on 1 billion tokens of the FineWeb
dataset [2].


**Notation.** We let _∥_ _**M**_ _∥_ F and _∥_ _**M**_ _∥_ 2 denote the Frobenius norm and spectral norm
(largest singular value) of a matrix _**M**_, respectively. We denote the spectrum (set of
singular values) by _σ_ ( _**M**_ ).
Let P _d_ be the set of polynomials of degree at most _d_ . For odd _d_, P [odd] _d_ denotes the set of
polynomials of degree at most _d_ containing only odd-degree monomials. For a polynomial
_p_, deg( _p_ ) is its degree. Let sign( _x_ ) be the scalar sign function, which satisfies sign(0) = 0,
sign( _x_ ) = 1 if _x >_ 0 and sign( _x_ ) = _−_ 1 if _x <_ 0.
For a polynomial _p ∈_ P [odd] _d_ and a matrix _**M**_ with rank reduced SVD given by _**M**_ =
_**U**_ **Σ** _**V**_ [T] and positive singular values _σ_ 1 _≥· · · ≥_ _σ_ rank( _**M**_ ) _>_ 0, we define _p_ ( _**M**_ ) := _**U**_ _p_ ( **Σ** ) _**V**_ [T],
where _p_ ( **Σ** ) is the diagonal matrix with diagonal entries _p_ ( _σi_ ) for _i_ = 1 _, . . .,_ rank( _**M**_ ).


2 `[https://github.com/NoahAmsel/PolarExpress](https://github.com/NoahAmsel/PolarExpress)`
3In Appendices E and F, we describe two further algorithmic ideas that can be incorporated into `Polar`
`Express` . They are not used in our `Muon` experiments but they may be beneficial in other settings, and we
believe they merit further study.


5


### **2 Related Work**

Computing polar( _**M**_ ) is an important and longstanding problem in numerical linear
algebra, with applications spanning electronic structure calculations, lattice quantum
chromodynamics, orthogonal Procrustes analysis, parallel algorithms for computing the
SVD, and beyond; see e.g. [21, 26, 14, 18, 41, 49].


**Newton-Schulz and polynomial Pad´e methods.** The earliest methods in the literature are polynomial iterations like (2). Several nearly simultaneous papers introduced
the family of polynomial Pad´e iterations, comprising Newton-Schulz and its higher-degree
analogues [31, 6, 21, 33]. These higher-degree methods are also sometimes called “NewtonSchulz”; when doing so, we will specify the degree for clarity. In these methods, each
iteration refines the current approximation _**X**_ _t_ by applying a low-degree odd matrix polynomial, where any odd monomial _x �→_ _x_ [2] _[q]_ [+1] is defined for rectangular matrices by the

        -        - _q_
formula _**X**_ _t �→_ _**X**_ _t_ _**X**_ _t_ _[⊤]_ _**[X]**_ _[t]_ . Our `Polar Express` method also takes this form, though
unlike Newton-Schulz, it changes the polynomial at each iteration.
The polynomials used in Pad´e methods are chosen to match the value and first few
derivatives of sign( _x_ ) at the points _x_ = _±_ 1. For instance, the update rule of the third
method in this family is defined by _p_ ( _x_ ) = 1 �35 _x −_ 35 _x_ 3 + 21 _x_ 5 _−_ 5 _x_ 7�, which is the
16
unique degree-7 polynomial satisfying _p_ ( _±_ 1) = _±_ 1 and _p_ _[′]_ ( _±_ 1) = _p_ _[′′]_ ( _±_ 1) = _p_ _[′′′]_ ( _±_ 1) = 0.
These methods converge so long as all singular values of _**X**_ 0 lie in (0 _,_ 1], a condition
guaranteed by the initialization of (2). Furthermore, the order of convergence of the degree
2 _q_ + 1 method is _q_ + 1 [6]. In particular, the Newton-Schulz method ( _q_ = 1) converges
quadratically.


**Newton’s method and rational Pad´e.** In the numerical analysis literature, polynomial
methods were succeeded by rational iterations like Newton’s method [21], defined as follows [4] :



_**X**_ 0 = _**M**_ _**X**_ _t_ +1 = [1]

2




- _**X**_ _t_ + _**X**_ _t_ _[−⊤]_ (3)



Newton’s method also converges quadratically. Like Newton-Schulz, it works because the
rational function _r_ ( _x_ ) = [1]

2 [(] _[x]_ [ +] _[ x][−]_ [1][) has a stable fixed point at 1; unlike for Newton-Schulz,]
this point is a global attractor for the whole positive real line. At first glance, Newton’s
method has nothing to do with the Pad´e iterations discussed above. However, after a
change of variables _**Y**_ _t_ = _**X**_ _t_ _[−]_ [1], it can be reinterpreted as _**Y**_ _t_ +1 = 2 _**Y**_ _t_ ( _**I**_ + _**Y**_ _t_ _[⊤]_ _**[Y]**_ _[t]_ [)] _[−]_ [1][, which]
is sometimes called inverse Newton. Observing that _r_ ( _x_ ) = 1+2 _xx_ ~~[2]~~ [satisfies] _[ r]_ [(] _[±]_ [1) =] _[ ±]_ [1]
and _r_ _[′]_ ( _±_ 1) = 0, we see that (inverse) Newton is also a Pad´e method, though a rational
rather than polynomial one. In fact, given a odd degree 2 _qn_ + 1 for the numerator and
an even degree 2 _qd_ for the denominator, there is a unique rational function that matches
the value and first _qn_ + _qd_ derivatives of sign( _x_ ) at _x_ = _±_ 1. This directly yields a Pad´e
method for computing polar( _**M**_ ) whose order of convergence is _qn_ + _qd_ + 1. For instance,
_r_ ( _x_ ) = [3] 1+3 _[x]_ [+] _x_ _[x]_ ~~[2]~~ [3] [is called Halley’s method, which converges cubically. When] _[ q][d]_ [ = 0, we recover]

the polynomial Pad´e methods.


4Our description of Newton’s method and other rational methods assumes square non-singular _**M**_ .
Non-square problems can be reduced to the square case by an initial QR decomposition, but this is not an
option for purely polynomial methods like ours.


6


There are two main weakness of Newton’s method and the Pad´e iterations: slow
convergence in the initial phase and the need to compute explicit inverses. To accelerate
initial convergence, Higham popularized the technique of rescaling the matrix after every
Newton iteration [21]. Intuitively, rescaling _**X**_ _t_ so that _σ_ max = 1 _/σ_ min centers the spectrum
around 1, where convergence is fastest. Several easily-computable choices of scaling factor
exist to accomplish this approximately. Note that this rescaling scheme would fail for
Newton-Schulz, which likewise suffers from slow initial convergence but which would diverge
if _σ_ max _≫_ 1.
Computing matrix inverses is difficult to parallelize and to implement stably in low
precision arithmetic. However, a trick was developed for stably computing many rational
methods _without_ explicit inverses; QR decompositions can be used instead [37, 53]. Applying
this trick to Halley’s method and combining with a special rescaling scheme yields the
QDWH (QR-based dynamically weighted Halley) method, which converges in just six
iterations for any reasonably conditioned matrix [37].


**Adaptive rational methods from optimal approximations.** A landmark 2016 paper
introduced a new paradigm to design iterative methods for computing polar( _**M**_ ) [38]. We
describe this paradigm in more detail in Section 4, but the main insight is as follows.
Pad´e methods choose the update rule to be an approximation to sign( _x_ ) of a given degree
that is optimally accurate in the neighborhood of _x_ = 1. Instead, we should choose the
approximation to sign( _x_ ) that is optimal over an _interval_ [ _ℓ,_ 1] _⊂_ R _≥_ 0 that contains the
singular values. Moreover, after each step of the algorithm, the range of the singular values
changes; therefore, we adapt the update rule at each iteration to match the new interval.
When the range of the singular values is large, this approach ensures that the update rule
shrinks it as quickly as possible. As the algorithm proceeds and the interval shrinks to a
small neighborhood of 1, the update rule approaches that of a Pad´e method, maintaining
the same high order of convergence as it has.
Within the class of odd rational functions whose numerators and denominators have
degree 2 _q_ + 1 and 2 _q_, respectively, an explicit formula for this optimal approximation to
sign( _x_ ) on any interval [ _ℓ,_ 1] was found by Zolotarev. It was shown that these rationals have
remarkable convergence properties for any _q_ [38]. For _q_ = 1, this optimal approximation
coincides exactly with the dynamically weighted Halley’s method (QDWH) referenced
above. For even faster convergence than QDWH, [38] proposed the Zolo-pd method, which
uses _q_ = 17. Finally, these methods all admit the same QR-based implementation trick as
QDWH.


**Adaptive polynomial methods.** In this paper, we adopt the paradigm of Zolo-pd [38]
but with polynomials rather than rationals of degree (2 _q_ + 1 _,_ 2 _q_ ). This choice avoids the
need for QR factorizations, relying solely on GPU-friendly matrix-matrix multiplications in
low-precision arithmetic. While this class of methods has not been fully developed in the
numerical analysis literature, similar ideas have been rediscovered in different guises. In an
unpublished manuscript that predates Zolo-pd, Chen and Chow [12] describe a rescaling
strategy for Newton-Schulz. Though motivated differently, their method is equivalent to
ours for degree-3 polynomials (unlike our work, they do not consider general odd degree).
They also observe numerical instability that prevents the method from converging to all the
way to machine precision. Using the insights of [39], they propose a simple mitigation for
this issue that we adopt in Section 4.4. Our work gives the approach from [39] a stronger


7


theoretical foundation that connects to the paradigm of Zolo-pd. Concretely, we prove
that choosing an optimal polynomial at each iteration leads to a composed polynomial
that is _globally_ optimal in the sense of (6).
Independently, a group of cryptographers developed a similar method for approximating
the scalar function sign( _x_ ) in the context of homomorphic encryption schemes [32]. Their
focus is mainly on tuning the analogues in their setting of the polynomial degree and number
of iterations, whereas we focus on demonstrating optimality and efficiently constructing
the update polynomials for degree 3 and 5. In addition, we consider matrix-valued inputs
in low-precision arithmetic—not scalars in exact arithmetic—and we demonstrate our
method’s effectiveness within the `Muon` algorithm for training deep neural networks.


**Application within** **`Muon`** **.** The designers of `Muon` realized that, due to the extreme
efficiency requirements and lax accuracy requirements of their setting, rational-based
methods from the numerical analysis literature are inapplicable. However, polynomial-based
iteration schemes can take full advantage of GPUs because they use only matrix-matrix
products in half-precision arithmetic, not inverses or QR decompositions. The preference for
speed over accuracy motivates methods that aim to quickly produce coarse approximations,
even at the cost of asymptotic convergence. Examples include the proposals of Jordan [25]
and You [36, 10], as discussed in Section 1.2. Like Chen and Chow [12], Jordan found
that convergence in the initial phase can be accelerated by choosing update rules that
have a large derivative near zero, so as to increase the small singular values as much as
possible at each iteration. You furthermore chose to use different update rules at each
iteration, allowing extra flexibility to tune the trade-off between speed and accuracy. Both
used degree-5 polynomials that were found through gradient descent on heuristic objective
functions. These proposals were previously compared to Newton-Schultz [5], but never to
Chen and Chow’s method from [39]. We find that our method (which generalizes [39])
outperforms them all.
Finally, we remark that concurrent work of Grishina, Smirnov, and Rakhuba also
proposes an adaptive polynomial method that generalizes [39] and applies it to accelerating
Muon [19]. Like [39], this work does not establish global optimality of the composed
polynomial as we do in Section 4 or address finite precision considerations.

### **3 Approximations by Compositions of Polynomials**


To design a GPU-friendly method for computing polar( _**M**_ ), we limit ourselves to the
following GPU-friendly operations:


i) Linear combinations: given scalars _β, γ ∈_ R and matrices _**B**_ and _**C**_, compute
_β_ _**B**_ + _γ_ _**C**_,


ii) Matrix-matrix products: compute _**BC**_ .


While both these computational primitives are well-suited for parallel computing environments, matrix-matrix products come at a higher computational cost than linear
combinations. Therefore, our method attempts to minimize the number of matrix-matrix


5Jordan [25] actually compares to 2 _x −_ 32 _[x]_ [3][ +] [1] 2 _[x]_ [5][, whereas the true degree-5 Newton-Schulz polynomial]

is (15 _x −_ 10 _x_ [3] + 3 _x_ [5] ) _/_ 8. However, the difference in performance is negligible for the first few iterations.



3 [1]

2 _[x]_ [3][ +] 2



5Jordan [25] actually compares to 2 _x −_ 3



8


products. A key observation is that we can compute _odd_ monomials of _**M**_ = _**U**_ **Σ** _**V**_ [T] using
the following formula:


_**M**_ [2] _[q]_ [+1] := _**U**_ **Σ** [2] _[q]_ [+1] _**V**_ [T] = _**M**_ ( _**M**_ [T] _**M**_ ) _[q]_ _._


Hence, for an odd polynomial _p_ ( _x_ ) = _a_ 0 _x_ + _a_ 1 _x_ [3] + _· · ·_ + _aqx_ [2] _[q]_ [+1] we can compute


_p_ ( _**M**_ ) := _a_ 0 _**M**_ + _a_ 1 _**M**_ ( _**M**_ [T] _**M**_ ) + _· · ·_ + _aq_ _**M**_ ( _**M**_ [T] _**M**_ ) _[q]_ _._


It has been shown that for an arbitrary polynomial _p_, one requires Θ(deg( _p_ ) [1] _[/]_ [2] ) products
to compute _p_ ( _**M**_ ) [44]; see also [23] for related work. This compares favorably to the naive
approach that forms all monomials in _p_ and then sums them together, which requires
Ω(deg( _p_ )) products. However, if _p_ can be expressed as a composition of _T_ polynomials,
each of degree _d_
_p_ = _pT ◦_ _pT_ _−_ 1 _◦· · · ◦_ _p_ 1 _,_ (4)


then the degree of _p_ is _d_ _[T]_, and _p_ ( _**M**_ ) can be efficiently computed recursively by


_**X**_ 0 = _**M**_ _,_ _**X**_ _t_ = _pt_ ( _**X**_ _t−_ 1) for _t_ = 1 _,_ 2 _, . . ., T._ (5)



The final iterate is _**X**_ _T_ = _p_ ( _**M**_ ), which we compute with just _O_ ( _Td_ ) matrix-matrix
products.
Iterative methods for polar( _**M**_ ) can be seen in this light. For instance, the degree-5
Newton-Schulz method uses the polynomial update _pt_ ( _x_ ) = [15] _[x][ −]_ [10] _[x]_ [3][ +] [3] _[x]_ [5][ for each]



Newton-Schulz method uses the polynomial update _pt_ ( _x_ ) = [15] 8 _[x][ −]_ [10] 8 _[x]_ [3][ +] [3] 8 _[x]_ [5][ for each]

_t_ = 1 _, . . ., T_ . The composition _p_ = _pT ◦· · ·◦p_ 1 approximates sign( _x_ ), and the approximation
error goes to 0 as _T_ grows. In this paper, we ask the following question: what choice of
_pT ◦· · · ◦_ _p_ 1 gives the _best_ approximation to sign( _x_ )?
The method we will present is optimal in the following sense: given lower and upper
bounds _ℓ_ and _u_ on the singular values of _**M**_, an odd degree _d ∈_ N, and the number
of iterations _T ∈_ N, our method computes the composition _p_ _[⋆]_ ( _**M**_ ) that minimizes the
worst-case error in the spectral norm. That is,




[3]
8 _[x]_ [3][ +] 8



8 _[x][ −]_ [10] 8



_p_ _[⋆]_ = argmin
_p_ = _pT ◦pT −_ 1 _◦···◦p_ 1
_pt∈_ P [odd] _d_



_**M**_ max _∈_ R _[m][×][n]_ _∥_ polar( _**M**_ ) _−_ _p_ ( _**M**_ ) _∥_ 2 _._ (6)
_σ_ ( _**M**_ ) _⊂_ [ _ℓ,u_ ]



Given that polar( _**M**_ ) _−_ _p_ ( _**M**_ ) = _**U**_ ( _**I**_ _−_ _p_ ( **Σ** )) _**V**_ [T], and by the unitary invariance of the
spectral norm, we have that (6) is equivalent to



_p_ _[⋆]_ = argmin
_p_ = _pT ◦pT −_ 1 _◦···◦p_ 1
_pt∈_ P [odd] _d_



max (7)
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|][ .]_



For completeness, the equivalence between (6) and (7) is proven in Appendix C.
In other words, the problem given in (6) reduces to that of finding a “uniform” or
“minimax” approximation to the constant function _x �→_ 1 over the interval [ _ℓ, u_ ], as given
in (7). Uniform approximation on an interval by polynomials or rational functions of a
given degree is a central topic in approximation theory; see e.g. [50]. Here, we seek an
approximation of a particular form—a _composition_ of odd polynomials of fixed degrees. In
the next section, we solve the optimization problem of (7) and use the solution to create
`Polar Express` . Figure 2 (a) shows the resulting _p_ _[∗]_ polynomial labeled as `PolarExp`, as
compared to the `Jordan` ’s method in [25], and the six iterations of `You` ’s method in [10].


9


(a) The left figure compares the composition (for _T_ = 6 and _d_ = 5) of polynomials given by `Polar`
`Express` ( _ℓ_ = 0 _._ 001), You’s method (which is defined up to 6 iterations), Newton-Schulz, and
Jordan’s method for approximating sign( _x_ ). The right figure demonstrates the convergence of the
methods on [0 _._ 001 _,_ 1]. Note the slow initial convergence of Newton-Schulz.


(b) The evolution of the first three optimal polynomials _p_ 1, _p_ 2, and _p_ 3 and the corresponding
lower bounds _ℓt_ +1 = _pt_ ( _ℓt_ ) and upper bounds _ut_ +1 = 2 _−_ _ℓt_ +1, as described in Theorem 4.1. The
horizontal black line indicates _y_ = 1. The polynomial degree is _d_ = 5 and the number of iterations
is _T_ = 3. We set _ℓ_ 1 = 0 _._ 03 and _u_ 1 = 1.


Figure 2

### **4 The Polar Express**


**4.1** **Greedy is optimal**


The key observation is that the polynomial used in each iteration can be chosen greedily,
given the choice of polynomials from the previous iterations. For the first iteration, we
choose _p_ 1 so as to map the interval [ _ℓ, u_ ] as close to 1 as possible. That is, it minimizes
max _x∈_ [ _ℓ,u_ ] _|_ 1 _−_ _p_ 1( _x_ ) _|_ . The image of _p_ 1 will be a new interval [ _ℓ_ 2 _, u_ 2], where


_ℓ_ 2 = min _u_ 2 = max (8)
_x∈_ [ _ℓ,u_ ] _[p]_ [1][(] _[x]_ [)] _x∈_ [ _ℓ,u_ ] _[p]_ [1][(] _[x]_ [)]


We now pick _p_ 2 to map the interval [ _ℓ_ 2 _, u_ 2] as close to 1 as possible, obtaining a new
interval [ _ℓ_ 3 _, u_ 3] that is the image of [ _ℓ, u_ ] through _p_ 2 _◦_ _p_ 1. We continue this process for as
many iterations as desired.


10


The following theorem guarantees that this process finds the solution to (7), and thereby
also (6). The scheme is also outlined in Figure 2 (b), which demonstrates the evolution
of the lower bounds _ℓt_, the upper bounds _ut_, and the polynomials _pt_ across iterations.











_Proof._ See Appendix A.







Fortunately, (11) shows that once _pt_ has been found, we can compute the new lower and
upper bounds _ℓt_ +1 and _ut_ +1 and the approximation error simply by evaluating _pt_ ( _ℓt_ ). Hence,
for any _fixed_ upper and lower bounds on the singular values of _**M**_, we can _precompute_
the polynomials _p_ 1 _, . . ., pT_ and the bounds [ _ℓ_ 1 _, u_ 1] _, . . .,_ [ _ℓT_ +1 _, uT_ +1]. Then, applying the
iterative procedure of (5), the final iterate _**X**_ _T_ will satisfy the following error bound

_∥_ polar( _**M**_ ) _−_ _**X**_ _T ∥_ 2 = _∥_ polar( _**M**_ ) _−_ _p_ _[⋆]_ ( _**M**_ ) _∥_ 2 _≤_ 1 _−_ _ℓT_ +1 _._ (12)


From the optimality guarantee of Theorem 4.1, we know that our method converges
at least as fast as the Newton-Schulz iteration of the same degree. Combining this fact


11


with an existing analysis of Newton-Schulz, we immediately get the following convergence
guarantee showing that our method enjoys faster than exponential convergence.



_Proof._ See Appendix B.


In fact, Theorem 4.3 underestimates how fast our method converges. For degree _d_ = 5,
our method converges about twice as fast as Newton-Schulz (compare with [12, Section
3.1]). Furthermore, the same analysis applies even if _p_ _[∗]_ is constructed using a “lower
bound” _ℓ_ that was too high. That is, replacing _ℓ_ on the right-hand side of (13) by _σ_ min,
the theorem holds even if _p_ _[∗]_ is constructed to be optimal on the interval [ _ℓ,_ 1] for _ℓ> σ_ min.
Intuitively, when _ℓ_ = _u_ = 1, the polynomial _p_ _[∗]_ coincides exactly with the Newton-Schulz
method. Mistakenly setting _ℓ> σ_ min, we obtain a polynomial that converges slower than
the optimal polynomial but faster than Newton-Schulz, so the guarantee of Theorem 4.3
still holds (cf. [12, Theorem 3.3]).


**4.2** **Finding the optimal polynomial for each iteration**


Theorem 4.1 shows that we can solve (7) by greedily choosing the optimal approximation
_pt ∈_ P [odd] _d_ for each interval [ _ℓt, ut_ ] for _t_ = 1 _, . . ., T_ . In this section, we show how to find
each _pt_ . Since we are now focused on just one iteration, we drop the subscripts. Given _ℓ_
and _u_, we wish to solve the following optimization problem:


argmin max (14)
_p∈_ P [odd] _d_ _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_


That is, we seek a minimax or uniform approximation of the function _x �→_ 1 on [ _ℓ, u_ ] from
the set of odd polynomials. (Equivalently, we seek a minimax optimal approximation to
sign( _x_ ) on [ _−u, −ℓ_ ] _∪_ [ _ℓ, u_ ].)
Problems of the form (14) are well-studied in approximation theory and numerical
analysis. The key mathematical insight underlying the solution is the Equioscillation
Theorem, which we state formally for our setting in Lemma A.1. This theorem gives a
surprising characterization of the optimal solution of (14): an odd _p_ is optimal for degree
2 _q_ + 1 if and only if there is a set of _q_ + 2 equioscillating points. This is a set of points
at which _p_ achieves its maximum approximation error _±E_, and for which the sign of the
error alternates. Even if the optimal approximation error _E_ is not known in advance,
finding a set of _q_ + 2 equioscillating points for a given _E_ serves as a certificate that no
better approximation error is achievable. The Equioscillation Theorem is the basis of
the Remez algorithm [42, 43], a general tool that can be used to find (nearly) optimal
polynomial approximations of a given degree to _any_ function on any interval. With very
minor modifications to handle the constraint that _p_ be odd, Remez can be used to directly
solve (14).


12


However, the Remez algorithm is opaque, complex, and difficult to implement correctly.
Fortunately, we do not need the Remez algorithm in its full generality to solve our problem.
We seek only low degree polynomials, and the function we wish to approximate is the
constant function _f_ ( _x_ ) _≡_ 1. For _d_ = 3, we can derive an explicit, closed form solution to
(14) using the Equioscillation Theorem. Up to rescaling, the optimal polynomial turns out
to be the same one derived in Chen and Chow by different means [12]. For degree _d_ = 5, we
present Algorithm 3, a much simpler way of solving (14) that is mathematically equivalent
to Remez in our setting. This algorithm is implemented in its entirety in Appendix G.
We briefly describe the solution for _d_ = 3. We seek a polynomial of the form _p_ ( _x_ ) =
_ax_ + _bx_ [3] . The Equioscillation Theorem stipulates that _p_ must have an equioscillating set
of size 3. For _p_ to achieve its maximum error at a point _x_, _x_ must be a local extremum of
_p_ ( _x_ ) _−_ 1 on the interval [ _ℓ, u_ ]. Thus, for _x_ to be eligible for membership in the equioscillating
set, it must either be a true local extremum of _p_ ( _x_ ) _−_ 1 that happens to lie in [ _ℓ, u_ ], or else
one of the endpoints _ℓ, u_ . However, because _p_ is an odd cubic, it has at most one true local
extremum on R _≥_ 0. Thus, to build an equioscillating set of three points, we must include
_p_ ’s unique positive local extremum _and_ both endpoints. This local extremum of _p_ occurs
at ~~�~~ _−a_ [. Therefore, we seek] _[ a, b]_ [ such that]



_−a_



3 _b_ [. Therefore, we seek] _[ a, b]_ [ such that]







_p_ ( _ℓ_ ) = 1 _−_ _E,_ _p_



�� _−a_


3 _b_



= 1 + _E,_ _p_ ( _u_ ) = 1 _−_ _E_ (15)



for some _E_ . This is a system of three equations in three variables. The solution _p_ ( _x_ ) =
_ax_ + _bx_ [3] is most easily expressed as follows. Let _p_ NS( _x_ ) = [3] _[x][ −]_ [1] _[x]_ [3][. Then]




[3]

2 _[x][ −]_ [1] 2




[1]

2 _[x]_ [3][. Then]







3 4
and _β_ = _[.]_
_u_ [2] + _lu_ + _ℓ_ [2] 2 + _ℓu_ ( _ℓ_ + _u_ ) _α_ [3]



_p_ ( _x_ ) = _βp_ NS( _αx_ ) _,_ where _α_ =



We now turn to the degree-5 case. The intuition of Algorithm 3 is as follows. For
any fixed set of four points _ℓ< q < r < u_, we can find a degree-5 odd polynomial
_p_ ( _x_ ) = _ax_ + _bx_ [3] + _cx_ [5] that satisfies


_p_ ( _ℓ_ ) = 1 _−_ _E,_ _p_ ( _q_ ) = 1 + _E,_ _p_ ( _r_ ) = 1 _−_ _E,_ _p_ ( _u_ ) = 1 + _E_


for some _E_ by solving a 4 _×_ 4 linear system in _a, b, c_ and _E_ . Likewise, for any fixed degree-5
odd _p_, we can find its four (or fewer) local extrema on [ _ℓ, u_ ] as follows: they occur at _ℓ, u_
and the roots of _p_ _[′]_, which is an even degree-4 polynomial whose roots can easily be found
by the _quadratic_ formula. Algorithm 3 simply alternates between these two steps (solving
for _a, b, c, E_ and solving for _q, r_ ) until the points _q, r_ converge. Once they have converged,
_{ℓ, q, r, u}_ forms an equioscillating set, so _p_ is the optimal polynomial. For more details,
please see Appendix D.


**4.3** **Upper and lower bounds on the singular values**


To instantiate our method, we need upper and lower bounds _u_ and _ℓ_ on the singular values
of the input matrix _**M**_ . A trivial upper bound is given by _∥_ _**M**_ _∥_ F. For _**M**_ _∈_ R _[m][×][n]_ with
_n ≤_ _m_, this can overestimate _σ_ max( _**M**_ ) by a factor of _[√]_ ~~_n_~~ in the worst case. However in
practice, the gradient matrices of the weights of dense linear layers in neural networks
tend to have small effective rank [52]. Consequently, the Frobenius norm tends to be a


13


reasonably good bound on the spectral norm that is loose only by a small constant factor.
We rescale the input matrix by setting _**X**_ 0 = _**M**_ _/∥_ _**M**_ _∥_ F so that _u_ = 1.
It is difficult to efficiently find a good lower bound on the smallest singular value, so
we are forced to guess. Fortunately, the consequences of a bad guess are not severe. As
discussed above, the method will eventually converge for any _ℓ_ _∈_ (0 _, u_ ], and even an order of
magnitude error in our guess of _ℓ_ only delays convergence by a few iterations. For matrices
stored in floating point arithmetic, the singular values are usually larger than machine
precision _ϵ_ mach [7], so a good guess is to set _ℓ_ _≈_ _ϵ_ mach. In our numerical experiments we
work in `bfloat16` where _ϵ_ mach = 2 _[−]_ [7] = 0 _._ 0078125 _._ Hence we set _ℓ_ = 10 _[−]_ [3] and _u_ = 1. Since
we use these bounds for all input matrices, we can precompute the optimal polynomials
once and apply them to as many inputs as we want.


**4.4** **Finite precision considerations**



When working in finite-precision arithmetic,
especially the half-precision `bfloat16` format used in deep learning, we must take
some care to avoid blowups and other problems due to numerical error. To this end,
we make three small changes to the method.
These adjustments stabilize the algorithm
with a negligible effect on accuracy. Furthermore, these adjustments can be made
in the offline stage by modifying the coefficients of our optimal polynomials.
The first issue arises when numerical
round-off creates singular values that are
slightly larger than our current upper bound
_ut_ . Our optimal polynomials converge only
when the singular values of _**X**_ _t_ are less than
_ut_ . In some cases we have


_pt_ ( _ut_ + _ϵ_ ) _> ut_ +1 + _ϵ,_


|Col1|Col2|Col3|
|---|---|---|
||||
||||



Figure 3: Effects of stabilizing the update
rules with a safety factor and cushioning, as
described in Section 4.4. The blue curve is the
optimal degree-5 polynomial for the interval

[0 _._ 005 _,_ 1]. It is has numerical issues because it
maps singular values near 0 _._ 8 down to almost
zero and maps 1 + _ϵ_ to _≈_ _ut_ +1 + 25 _ϵ_ . The
stabilized version is better because it ensures
_pt_ ( _x_ ) _≥_ 0 _._ 236 and maps all _x ≤_ 1 _._ 01 to at

_x_
most _ut_ +1.



Before
stabilizing



After
0 _._ 236 _x_
stabilizing



2 _._ 0


1 _._ 5


1 _._ 0


0 _._ 5



0 _._ 0



0 _._ 0 0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
_x_



so over many iterations, a singular value

stabilized version is better because it ensures

that is slightly larger than _ut_ large could _pt_ ( _x_ )

_≥_ 0 _._ 236 and maps all _x ≤_ 1 _._ 01 to at

grow to _∞_ instead of converging to 1. _x_

most _ut_ +1.

To fix this issue, we simply replace each
polynomial _x �→_ _pt_ ( _x_ ) by _x �→_ _pt_ ( _x/_ 1 _._ 01).
This safety factor corrects for round-off errors in previous iterations while only slightly
changing the behavior of the polynomial on the interval [ _ℓt, ut_ ], though it does cause the
singular values to converge to 0 _._ 999998 instead of to 1. To correct for this, the safety factor
can be omitted in the final iteration.
The second issue was identified in [39] and addressed in the context of polynomial
iterations by Chen and Chow [12]. In general, iterative methods for polar( _**M**_ ) aim to
increase each singular value relative to the largest singular value; while _σ_ min( _**X**_ 0) _≪_
_σ_ max( _**X**_ 0), after enough iterations, _σ_ min( _**X**_ _t_ ) _≈_ _σ_ max( _**X**_ _t_ ) _≈_ 1. However, the convergence
of each singular value to _σ_ max may not be monotonic. Over the domain [ _ℓt, ut_ ], our optimal
polynomial _pt_ oscillates repeatedly between _ℓt_ +1 and _ut_ +1, so some singular values that are



14


near _ut_ may get mapped down to _ℓt_ +1. It so happens that this non-monotonicity—even at
a single iteration—can cause loss of precision. That is, problems occur if



max

( _σi_ ) _x∈_ [ _σ_ min _,σ_ max] _[p][t]_ [(] _[x]_ [)]

_≪_
_σi_ _σ_ max



_pt_ ( _σi_ )



_,_
_σ_ max



where 0 _≤_ _σ_ min _≤_ _σi ≤_ _σ_ max are singular values of _**X**_ _t_ [39]. In the extreme case _pt_ ( _σi_ ) _<_ 0,
the _i_ th singular vector will change sign, casuing the method to converge to the polar factor
of the wrong matrix. Unlike Newton-Schulz, unscaled Newton, or QDWH, our method is
affected by this loss of precision.
To mitigate this issue, [12] propose modifying their update polynomials to enforce a
lower bound on the ratio _[p][t]_ [(] _[σ][i]_ [)] [. This issue only occurs when] _[ ℓ][t][ ≪]_ _[u][t]_ [; as] _[ ℓ][t][ →]_ _[u][t]_ [, our]




_[i]_

_σi_ [. This issue only occurs when] _[ ℓ][t][ ≪]_ _[u][t]_ [; as] _[ ℓ][t][ →]_ _[u][t]_ [, our]



optimal polynomial approaches the Pad´e approximant and so _[p][t]_ [(] _[x]_ [)]



optimal polynomial approaches the Pad´e approximant and so _[t]_ _x_ _≥_ 1 for all _x ∈_ [0 _, ut_ ].

We could fully solve the problem by using the Pad´e approximant instead of our optimal
polynomial, but this would significantly slow down convergence. Instead we compromise.
When _ℓt ≥_ _ut/_ 10, we find that _[p][t]_ [(] _[x]_ [)] _≥_ 0 _._ 236. Therefore, whenever _ℓt < ut/_ 10 we select the



When _ℓt ≥_ _ut/_ 10, we find that _[t]_ _x_ _≥_ 0 _._ 236. Therefore, whenever _ℓt < ut/_ 10 we select the

update rule as though _ℓt_ = _ut/_ 10. This change slows convergence, but only very slightly.
(The choice of 10 is somewhat arbitrary. In Appendix G, we use a different factor.)
The third change is copied from the original `Muon` implementation: normalize _**M**_ by
_∥_ _**M**_ _∥_ F + 10 _[−]_ [2] instead of by _∥_ _**M**_ _∥_ F. As before, we set _u_ 1 = 1.



**4.5** **The algorithm**


**Algorithm 2** The General `Polar Express`
**input:** Matrix _**M**_, iteration count _T_, degree _d_,
approximate lower bound _ℓ_ .
**output:** An approximation _**X**_ _T_ to polar( _**M**_ ).



We give the pseudocode of our proposed method for any degree in Algorithm 2 and provide a complete im[plementation in our repository. We](https://github.com/NoahAmsel/PolarExpress)
give the specific version of the `Polar`
`Express` with degree _d_ = 5 and _ℓ_ =
10 _[−]_ [3] used in our GPT experiments
in Algorithm 1. Our algorithm first
computes the polynomials _p_ 1 _, . . ., pT_
of Theorem 4.1 in full precision using
the results of Section 4.2 (or the Remez algorithm for _d >_ 5). This stage
is offline because the coefficients of
the polynomials are only computed
and stored once. For every subsequent call to the algorithm, these coefficients are reused and the offline
stage is skipped. For instance, in
Algorithm 1 these polynomials have
been precomputed and stored in the
variable `coeffs` ~~`l`~~ `ist` .



1





8









The polynomial _p_ _[⋆]_ := _pT ◦· · · ◦_ _p_ 1 is then applied to the input matrix _**M**_ in the
online stage. The online stage can be performed in lower precision ( `bfloat16` ) for greater
speed on a GPU. Horner’s rule can be used to carry out each iteration. For instance, if


15


10 [0]


10 _[−]_ [1]


10 _[−]_ [2]


10 _[−]_ [3]














|Col1|Synthetic Matrix σmin/σmax = 10−6|Col3|Col4|
|---|---|---|---|
|||||
|0<br>5<br>10<br>Iteratio|0<br>5<br>10<br>Iteratio|0<br>5<br>10<br>Iteratio|0<br>5<br>10<br>Iteratio|
|Newton-Schul<br>Jordan|Newton-Schul<br>Jordan|z (_d_|= 5)<br>You<br>PolarExp,_ ℓ_= 10_−_8<br>PolarExp,_ ℓ_= 10_−_6<br>PolarExp,_ ℓ_= 10_−_4|
|||||



Figure 4: Convergence of various degree-5 polynomial methods in the spectral norm. When
tuned properly, Polar Express attains outperforms the other methods at every iteration.
Left panel: synthetic matrix with _σ_ max = 1, _σ_ min = 10 _[−]_ [6] . Right panel: gradient of a
certain weight matrix of a randomly-initialized GPT-2 architecture on a batch of language
modeling data, normalized by the Frobenius norm.


_pt_ = _ax_ + _bx_ [3] + _cx_ [5], then


_**X**_ _t_ = _**X**_ _t−_ 1 ( _a_ _**I**_ + _**Y**_ _t−_ 1 ( _b_ _**I**_ + _c_ _**Y**_ _t−_ 1))


where _**Y**_ _t−_ 1 = _**X**_ _t_ _[⊤]_ _−_ 1 _**[X]**_ _[t][−]_ [1][.]
A simple implementation of the offline stage of Algorithm 2 is given in Appendix G. For
deep learning applications, we recommend using _d_ = 5 and _T_ = 5 or 6 with _ℓ_ 1 = 10 _[−]_ [3] . With
these parameters, the offline stage as implemented in Appendix G gives the polynomials
encoded in `coeffs` ~~`l`~~ `ist` in Algorithm 1. All told, our proposal for `Muon` is to apply the
composition of these polynomials to _**M**_ _/_ ( _∥_ _**M**_ _∥F_ + 10 _[−]_ [2] ).

### **5 Numerical Experiments**


**5.1** **Convergence of** **`Polar Express`**


We compare the performance of `Polar Express` against degree-5 Newton-Schulz and the
methods of Chen and Chow [12], Jordan [25], and You [10].
We first study an idealized scenario where the spectrum of the input matrix is known
exactly. We generate a random matrix whose singular values are evenly spaced on a
logarithmic scale between 10 _[−]_ [6] and 1. The right and left singular vectors are chosen at
random. The left panel of Figure 4 shows the results. Since all the methods in this plot
use degree-5 polynomials, their computational and runtime costs are all proportional to
the number of iterations. As expected, Newton-Schulz converges but makes almost no
progress for the first 17 iterations. Jordan’s method rapidly achieves an error of _≈_ 0 _._ 3
after just 11 iterations, but ceases to converge further. You’s method, which is difficult to


16


1 _._ 00


0 _._ 75


0 _._ 50


0 _._ 25


0 _._ 00


#### GPT-2 Gradient Layer 1 attn.c a ttn

10 20 30
Matrix-Matrix Products



1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


#### GPT-2 Gradient Layer 3 attn.c p roj

10 20 30
Matrix-Matrix Products







Figure 5: Convergence of polynomial methods in the Frobenius norm on GPT-2 gradient
matrices. The number of matrix-matrix products is _T_ ( _d_ + 1) _/_ 2, where _d_ is the degree (3
for Chen & Chow; 5 for all others) and _T_ is the number of iterations.


see on the plot because it is only defined for six iterations, converges at a similar rate as
Jordan’s method. When `Polar Express` is instantiated with _ℓ_ = _σ_ min, it dominates the
other methods at every iteration, achieving excellent accuracy after just 11 iterations and
converging about twice as fast as Newton-Schulz to any given error. Even when the lower
bound on _σ_ min is wrong by two orders of magnitude in either direction, the method remains
competitive, though it does not actually outperform Jordan’s method until iteration 13 or
14.
Next we test the methods’ performance on a matrix from a real-world application,
namely, the gradient of a weight matrix from the fourth transformer block of a GPT-2
architecture with respect to a language modeling objective on a batch of text from the Tiny
Shakespeare dataset [27]. The right panel of Figure 4 shows the results. Once again, the
best-tuned version of `Polar Express` outperforms the other methods. This time, we see
that setting _ℓ_ to be many orders of magnitude too small can delay convergence significantly,
and make `Polar Express` less competitive as compared to Jordan’s method.
For most other weight matrices in this GPT-2 model, the methods all take more than
10 iterations to converge in the spectral norm. The spectral error is large if there is even
one outlying singular value that is far from 1. However, for some applications, we may be
satisfied with a weaker notion of convergence, like the relative Frobenius norm. Figure 5
shows the performance of various methods on this metric. We use gradient matrices of
the same model, but from two different layers. In addition, we compare the degree-5
methods to Chen and Chow’s degree-3 method. To make this comparison fair, we measure
the number of matrix-matrix products performed by each method instead the number
of iterations. We find that `Polar Express` can once again dominate the other methods
across iterations. Chen and Chow’s method is also quite competitive, and the remaining
methods behave much as in Figure 4.


17


4 _._ 4


4 _._ 2


4 _._ 0


3 _._ 8


3 _._ 6


4 _._ 4


4 _._ 2


4 _._ 0


3 _._ 8


3 _._ 6




|muon-Jordan<br>4.4 muon-You<br>adamw<br>muon-PolarExp<br>Loss<br>4.2<br>Validation<br>4.0<br>Final<br>3.8<br>3.6|Col2|Col3|Col4|
|---|---|---|---|
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Validation Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|||muon-Jordan<br>|
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Validation Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|||~~muon-You~~<br>adamw<br>muon-PolarExp|
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Validation Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Validation Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Validation Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Validation Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||



|10−3 10−2 Learning Rate|Col2|Col3|Col4|
|---|---|---|---|
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|||muon-Jordan<br>|
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp|||~~muon-You~~<br>adamw<br>muon-PolarExp|
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||
|3_._6<br>3_._8<br>4_._0<br>4_._2<br>4_._4<br>Final Loss<br>muon-Jordan<br>~~muon-You~~<br>adamw<br>muon-PolarExp||||


10 _[−]_ [3] 10 _[−]_ [2]

Learning Rate



0 _._ 0 0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8
Epoch


0 200 400 600 800 1000 1200
Time (s)





Figure 6: Training a GPT-2 (124M) model on 1 Billion tokens of the Fineweb data set [2].
The Legend muon- `<name>` refers to using muon with the `<name>` method for computing
polar( _**M**_ ) with weight decay zero. Top Left: The final validation loss vs. the learning
rate. The final best validation losses for each method were, in reverse order, `adamw` : 4 _._ 197,
`muon-Jordan` : 3 _._ 639, `muon-You` : 3 _._ 629 and `muon-PolarExp` : 3 _._ 588. Bottom Left: The final
training loss vs the learning rate. Top Right: Validation loss vs. number of iterations.
Bottom Left: validation loss vs. time, plotting each method with its best learning rate.


**5.2** **Training GPT-2**


In our final experiment, we compare the performance of using our `Polar Express` method
given in Algorithm 1 inside the `Muon` algorithm versus Jordan’s [25] and You’s [10] methods. [6]

We train two different GPT-2 models:


`GPT-Small` : _n_ embd = 768 _,_ _n_ layer = 12 _,_ _n_ head = 12

`GPT-Large` : _n_ embd = 1280 _,_ _n_ layer = 36 _,_ _n_ head = 20


and a vocabulary size of 50 _,_ 257, using a context length of 1024. Training is performed on
1B tokens from the FineWeb dataset [2], using a batch size of 32 and a single epoch. All
models are trained with mixed precision ( `bfloat16` ) on 4 H100 GPUs. For all methods
we use the learning rate schedule proposed in [24], consisting of a constant phase for the
first 40% of training steps followed by a linear decay. All methods for the matrix sign
computations are performed in `float16b` precision and use five iterations.


6Our code is available at `[https://github.com/modichirag/GPT-opt/tree/polar](https://github.com/modichirag/GPT-opt/tree/polar)`, in the `polar` branch.


18


|Col1|muon-You|
|---|---|
||muon-Jordan<br>muon-polarexpress|
|||
|||
|||
|||
|||



0 _._ 0 0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8
Epoch


0 2000 4000 6000 8000
Time (s)



6 _._ 5


6 _._ 0


5 _._ 5


5 _._ 0


4 _._ 5


4 _._ 0


3 _._ 5


6 _._ 5


6 _._ 0


5 _._ 5


5 _._ 0


4 _._ 5


4 _._ 0


3 _._ 5



3 _._ 46


3 _._ 44


3 _._ 42


3 _._ 40


3 _._ 38


3 _._ 36


3 _._ 34


3 _._ 46


3 _._ 44


3 _._ 42


3 _._ 40


3 _._ 38


3 _._ 36



10 _[−]_ [2]

|Col1|Col2|muon-You<br>muon-Jordan|
|---|---|---|
||muon-polarexpress|muon-polarexpress|
||||
||||
||||
||||
||||



10 _[−]_ [2]



Learning Rate


Learning Rate





Figure 7: Training a GPT-2-Large model (774M params) on 1 Billion tokens of the Fineweb
data set [2]. The Legend muon- `<name>` refers to using muon with the `<name>` method for
computing polar( _**M**_ ). We used weight decay 0 _._ 1 for all methods. Top Left: The final
validation loss vs. the learning rate. The final best validation losses for each method were,
in reverse order, `adamw` : 4 _._ 197, `muon-Jordan` : 3 _._ 639, `muon-You` : 3 _._ 629 and `muon-PolarExp` :
3 _._ 588. Bottom Left: The final training loss vs the learning rate. Top Right: Validation
loss vs. number of iterations. Bottom Left: validation loss vs. time, plotting each method
with its best learning rate.


We apply `Muon` selectively to certain layers of the model. Following the nano-gpt
implementation [24], we assign `Muon` to all parameters with at least two dimensions
(typically weight matrices, and excluding RMS norm parameters), excluding the embeddings,
unembeddings, and the positional encodings. These excluded parameters are instead
optimized with AdamW.

Figure 1 and Figure 6 shows the resulting runs of each method in terms of validation
loss and training loss on the `GPT-Large` and `GPT-Small` models, respectively. In both
figures we can see that `muon-PolarExp` achieves a better validation and training loss than
`muon-Jordan` or `muon-You` for every learning rate. Since each iteration of the different
matrix sign methods are equally expensive (since they all apply a degree 5 polynomial),
improved validation loss in terms of epochs also translates to an improved loss in terms of
wall clock time (see bottom right of Figure 6). The advantage is remarkably consistent
across all learning rates and epochs.
We also experimented with adding weight decay 0 _._ 1 to the model, keeping all else the


19


same, in Figure 7. Here we find again that `muon-PolarExp` achieves a better validation
and training loss for every learning rate, except one ( _lr_ = 10 _[−]_ [2] ) where its performance
matches that of `muon-Jordan` .

### **Acknowledgements**


This work was partially supported by NSF awards 2045590 and 2234660. Computations
were run at facilities supported by the Scientific Computing Core at the Flatiron Institute,
a division of the Simons Foundation.

### **References**


[1] N. I. Achieser. _Theory of approximation_ . Dover Publications, Inc., New York, 1992.
Translated from the Russian and with a preface by Charles J. Hyman, Reprint of the
1956 English translation.


[2] Samuel Aroca-Ouellette, Philippe Beaudoin, Guillaume Lajoie, Liam Paull, Joelle
Pineau, Pascal Vincent, and Anirudh Goyal. Fineweb: Learning language models
with high quality web data. In _NeurIPS Datasets and Benchmarks Track_, 2023. URL:
`[https://arxiv.org/abs/2306.03061](https://arxiv.org/abs/2306.03061)` .


[3] Michele Benzi and Ru Huang. Some matrix properties preserved by generalized matrix
functions. _Spec. Matrices_, 7:27–37, 2019. `[doi:10.1515/spma-2019-0003](https://doi.org/10.1515/spma-2019-0003)` .


[4] Jeremy Bernstein and Laker Newhouse. Modular duality in deep learning. _arXiv_
_preprint arXiv:2410.21265_, 2024. URL: `[https://arxiv.org/abs/2410.21265](https://arxiv.org/abs/2410.21265)` .


[5] Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology.
_arXiv preprint arXiv:2409.20325_, 2024. URL: `[https://arxiv.org/abs/2409.20325](https://arxiv.org/abs/2409.20325)` .


[6] A. Bj¨orck and C. Bowie. An iterative algorithm for computing the best estimate of an [˙]
orthogonal matrix. _SIAM J. Numer. Anal._, 8:358–364, 1971. `[doi:10.1137/0708036](https://doi.org/10.1137/0708036)` .


[7] Christos Boutsikas, Petros Drineas, and Ilse C. F. Ipsen. Small singular values can
increase in lower precision. _SIAM J. Matrix Anal. Appl._, 45(3):1518–1540, 2024.
`[doi:10.1137/23M1557209](https://doi.org/10.1137/23M1557209)` .


[8] David Carlson, Volkan Cevher, and Lawrence Carin. Stochastic Spectral Descent for
Restricted Boltzmann Machines. In Guy Lebanon and S. V. N. Vishwanathan, editors,
_Proceedings of the Eighteenth International Conference on Artificial Intelligence and_
_Statistics_, volume 38 of _Proceedings of Machine Learning Research_, pages 111–119,
San Diego, California, USA, 09–12 May 2015. PMLR. URL: `[https://proceedings.](https://proceedings.mlr.press/v38/carlson15.html)`
`[mlr.press/v38/carlson15.html](https://proceedings.mlr.press/v38/carlson15.html)` .


[9] David E Carlson, Edo Collins, Ya-Ping Hsieh, Lawrence Carin, and Volkan
Cevher. Preconditioned spectral descent for deep learning. In C. Cortes,
N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, _Advances in_
_Neural Information Processing Systems_, volume 28. Curran Associates, Inc.,
2015. URL: `[https://proceedings.neurips.cc/paper_files/paper/2015/file/](https://proceedings.neurips.cc/paper_files/paper/2015/file/f50a6c02a3fc5a3a5d4d9391f05f3efc-Paper.pdf)`
`[f50a6c02a3fc5a3a5d4d9391f05f3efc-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2015/file/f50a6c02a3fc5a3a5d4d9391f05f3efc-Paper.pdf)` .


20


[10] Franz Louis Cesista, You Jiacheng, and Keller Jordan. Squeezing 1-2% efficiency
gains out of muon by optimizing the newton-schulz coefficients, 2025. URL: `[http:](http://leloykun.github.io/ponder/muon-opt-coeffs/)`
`[//leloykun.github.io/ponder/muon-opt-coeffs/](http://leloykun.github.io/ponder/muon-opt-coeffs/)` .


[11] PL Chebyshev. Questions on smallest quantities connected with the approximate
representation of functions (1859). _Collected works_, 2:151–235, 1947.


[12] Jie Chen and Edmond Chow. A stable scaling of newton-schulz for improving the sign
function computation of a hermitian matrix. _Preprint ANL/MCS-P5059-0114_, 2014.
URL: `[https://www.mcs.anl.gov/papers/P5059-0114.pdf](https://www.mcs.anl.gov/papers/P5059-0114.pdf)` .


[13] E. W. Cheney. _Introduction to approximation theory_ . McGraw-Hill Book Co., New
York-Toronto-London, 1966.


[14] J. Douglas Carroll and Phipps Arabie. Chapter 3 - multidimensional scaling. In
Michael H. Birnbaum, editor, _Measurement, Judgment and Decision Making_, Handbook of Perception and Cognition (Second Edition), pages 179–250. Academic Press,
San Diego, 1998. URL: `[https://www.sciencedirect.com/science/article/pii/](https://www.sciencedirect.com/science/article/pii/B9780120999750500051)`
`[B9780120999750500051](https://www.sciencedirect.com/science/article/pii/B9780120999750500051)`, `[doi:10.1016/B978-012099975-0.50005-1](https://doi.org/10.1016/B978-012099975-0.50005-1)` .


[15] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online
learning and stochastic optimization. _J. Mach. Learn. Res._, 12:2121–2159, 2011.


[16] Alexandre Eremenko and Peter Yuditskii. Uniform approximation of sgn _x_ by polynomials and entire functions. _J. Anal. Math._, 101:313–324, 2007. `[doi:10.1007/](https://doi.org/10.1007/s11854-007-0011-3)`
`[s11854-007-0011-3](https://doi.org/10.1007/s11854-007-0011-3)` .


[17] Gene H. Golub and Charles F. Van Loan. _Matrix computations_ . Johns Hopkins
Studies in the Mathematical Sciences. Johns Hopkins University Press, Baltimore,
MD, fourth edition, 2013.


[18] J. C. Gower and G. B. Dijksterhuis. _Procrustes problems_, volume 30 of _Oxford_
_Statistical Science Series_ . Oxford University Press, Oxford, 2004. `[doi:10.1093/](https://doi.org/10.1093/acprof:oso/9780198510581.001.0001)`
`[acprof:oso/9780198510581.001.0001](https://doi.org/10.1093/acprof:oso/9780198510581.001.0001)` .


[19] Ekaterina Grishina, Matvey Smirnov, and Maxim Rakhuba. Accelerating newtonschulz iteration for orthogonalization via chebyshev-type polynomials, 2025. URL:
`[https://arxiv.org/abs/2506.10935](https://arxiv.org/abs/2506.10935)`, `[arXiv:2506.10935](https://arxiv.org/abs/2506.10935)` .


[20] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic
tensor optimization. In Jennifer Dy and Andreas Krause, editors, _Proceedings of_
_the 35th International Conference on Machine Learning_, volume 80 of _Proceedings_
_of Machine Learning Research_, pages 1842–1850. PMLR, 10–15 Jul 2018. URL:
`[https://proceedings.mlr.press/v80/gupta18a.html](https://proceedings.mlr.press/v80/gupta18a.html)` .


[21] Nicholas J. Higham. Computing the polar decomposition—with applications. _SIAM_
_J. Sci. Statist. Comput._, 7(4):1160–1174, 1986. `[doi:10.1137/0907079](https://doi.org/10.1137/0907079)` .


[22] Nicholas J. Higham. _Functions of matrices_ . SIAM, Philadelphia, PA, 2008. `[doi:](https://doi.org/10.1137/1.9780898717778)`
`[10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778)` .


21


[23] Elias Jarlebring and Gustaf Lorentzon. The polynomial set associated with a fixed
number of matrix-matrix multiplications. _arXiv preprint arXiv:2504.01500_, 2025.
URL: `[https://arxiv.org/abs/2504.01500](https://arxiv.org/abs/2504.01500)` .


[24] Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, @fernbear.bsky.social, Boza
Vlado, You Jiacheng, Franz Cesista, Braden Koszarsky, and @Grad62304977. moddednanogpt: Speedrunning the nanogpt baseline, 2024. URL: `[https://github.com/](https://github.com/KellerJordan/modded-nanogpt)`
`[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)` .


[25] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse,
and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024.
URL: `[https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)` .


[26] Tetsuya Kaneko, Simone Fiori, and Toshihisa Tanaka. Empirical arithmetic averaging
over the compact Stiefel manifold. _IEEE Trans. Signal Process._, 61(4):883–894, 2013.
`[doi:10.1109/TSP.2012.2226167](https://doi.org/10.1109/TSP.2012.2226167)` .


[27] Andrej Karpathy. char-rnn. `[https://github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn)`, 2015.


[28] Charles Kenney and Alan J. Laub. Rational iterative methods for the matrix sign
function. _SIAM J. Matrix Anal. Appl._, 12(2):273–291, 1991. `[doi:10.1137/0612020](https://doi.org/10.1137/0612020)` .


[29] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen,
Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, et al. Kimi
k2: Open agentic intelligence, 2025. URL: `[https://arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534)`,
`[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)` .


[30] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In
_International Conference on Learning Representations_, 2015. URL: `[http://arxiv.](http://arxiv.org/abs/1412.6980)`
`[org/abs/1412.6980](http://arxiv.org/abs/1412.6980)` .


[31] Zdislav Kov´aˇr´ık. Some iterative methods for improving orthonormality. _SIAM J._
_Numer. Anal._, 7:386–389, 1970. `[doi:10.1137/0707031](https://doi.org/10.1137/0707031)` .


[32] Eunsang Lee, Joon-Woo Lee, Jong-Seon No, and Young-Sik Kim. Minimax approximation of sign function by composite polynomial for homomorphic comparison.
_IEEE Transactions on Dependable and Secure Computing_, 19(6):3711–3727, 2022.

`[doi:10.1109/TDSC.2021.3105111](https://doi.org/10.1109/TDSC.2021.3105111)` .


[33] R. B. Leipnik. Rapidly convergent recursive solution of quadratic operator equations.
_Numer. Math._, 17:1–16, 1971. `[doi:10.1007/BF01395861](https://doi.org/10.1007/BF01395861)` .


[34] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao
Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for LLM training.
_arXiv preprint arXiv:2502.16982_, 2025. URL: `[https://arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982)` .


[35] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _Inter-_
_national Conference on Learning Representations_, 2019. URL: `[https://openreview.](https://openreview.net/forum?id=Bkg6RiCqY7)`
`[net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7)` .


[36] Modula. Newton-schulz algorithm — jiacheng’s six-step method. `[https://docs.](https://docs.modula.systems/algorithms/newton-schulz/#jiacheng-s-six-step)`
`[modula.systems/algorithms/newton-schulz/#jiacheng-s-six-step](https://docs.modula.systems/algorithms/newton-schulz/#jiacheng-s-six-step)`, 2024. Accessed: 2025-05-19.


22


[37] Yuji Nakatsukasa, Zhaojun Bai, and Fran¸cois Gygi. Optimizing Halley’s iteration for
computing the matrix polar decomposition. _SIAM J. Matrix Anal. Appl._, 31(5):2700–
2720, 2010. `[doi:10.1137/090774999](https://doi.org/10.1137/090774999)` .


[38] Yuji Nakatsukasa and Roland W. Freund. Computing fundamental matrix decompositions accurately via the matrix sign function in two iterations: the power of Zolotarev’s
functions. _SIAM Rev._, 58(3):461–493, 2016. `[doi:10.1137/140990334](https://doi.org/10.1137/140990334)` .


[39] Yuji Nakatsukasa and Nicholas J. Higham. Backward stability of iterations for
computing the polar decomposition. _SIAM J. Matrix Anal. Appl._, 33(2):460–479, 2012.
`[doi:10.1137/110857544](https://doi.org/10.1137/110857544)` .


[40] Yuji Nakatsukasa and Nicholas J. Higham. Stable and efficient spectral divide and
conquer algorithms for the symmetric eigenvalue decomposition and the SVD. _SIAM_
_J. Sci. Comput._, 35(3):A1325–A1349, 2013. `[doi:10.1137/120876605](https://doi.org/10.1137/120876605)` .


[41] Herbert Neuberger. Exactly massless quarks on the lattice. _Phys. Lett. B_, 417(12):141–144, 1998. `[doi:10.1016/S0370-2693(97)01368-3](https://doi.org/10.1016/S0370-2693(97)01368-3)` .


[42] Ricardo Pach´on and Lloyd N. Trefethen. Barycentric-Remez algorithms for best
polynomial approximation in the chebfun system. _BIT_, 49(4):721–741, 2009. `[doi:](https://doi.org/10.1007/s10543-009-0240-1)`
`[10.1007/s10543-009-0240-1](https://doi.org/10.1007/s10543-009-0240-1)` .


[43] T Parks and James McClellan. Chebyshev approximation for nonrecursive digital
filters with linear phase. _IEEE Transactions on circuit theory_, 19(2):189–194, 1972.
`[doi:10.1109/TCT.1972.1083419](https://doi.org/10.1109/TCT.1972.1083419)` .


[44] Michael S. Paterson and Larry J. Stockmeyer. On the number of nonscalar multiplications necessary to evaluate polynomials. _SIAM J. Comput._, 2:60–66, 1973.
`[doi:10.1137/0202007](https://doi.org/10.1137/0202007)` .


[45] Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio SilvetiFalls, and Volkan Cevher. Training deep learning models with norm-constrained lmos,
2025. URL: `[https://arxiv.org/abs/2502.07529](https://arxiv.org/abs/2502.07529)`, `[arXiv:2502.07529](https://arxiv.org/abs/2502.07529)` .


[46] Artem Riabinin, Egor Shulgin, Kaja Gruntkowska, and Peter Richt´arik. Gluon: Making
muon & scion great again! (bridging theory and practice of lmo-based optimizers for
llms), 2025. URL: `[https://arxiv.org/abs/2505.13416](https://arxiv.org/abs/2505.13416)`, `[arXiv:2505.13416](https://arxiv.org/abs/2505.13416)` .


[47] Ishaan Shah, Anthony M Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju,
Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, et al.
Practical efficiency of muon for pretraining. _arXiv preprint arXiv:2505.02222_, 2025.
URL: `[https://arxiv.org/abs/2505.02222](https://arxiv.org/abs/2505.02222)` .


[48] Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing
Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat. A distributed
data-parallel PyTorch implementation of the distributed Shampoo optimizer for
training neural networks at-scale. _arXiv preprint arXiv:2309.06497_, 2023. URL:
`[https://arxiv.org/abs/2309.06497](https://arxiv.org/abs/2309.06497)` .


[49] Attila Szabo and Neil S Ostlund. _Modern quantum chemistry: introduction to advanced_
_electronic structure theory_ . Courier Corporation, 1996.


23


[50] Lloyd N. Trefethen. _Approximation theory and approximation practice_ . Society for
Industrial and Applied Mathematics (SIAM), Philadelphia, PA, extended edition,
2020.


[51] Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas
Janson, and Sham M. Kakade. SOAP: Improving and stabilizing shampoo using
adam for language modeling. In _The Thirteenth International Conference on Learning_
_Representations_, 2025. URL: `[https://openreview.net/forum?id=IDxZhXrpNf](https://openreview.net/forum?id=IDxZhXrpNf)` .


[52] Greg Yang, James B. Simon, and Jeremy Bernstein. A spectral condition for feature
learning, 2024. URL: `[https://arxiv.org/abs/2310.17813](https://arxiv.org/abs/2310.17813)`, `[arXiv:2310.17813](https://arxiv.org/abs/2310.17813)` .


[53] Zhenyue Zhang, Hongyuan Zha, and Wenlong Ying. Fast parallelizable methods for
computing invariant subspaces of Hermitian matrices. _J. Comput. Math._, 25(5):583–
594, 2007. URL: `[http://www.jstor.org/stable/43693395](http://www.jstor.org/stable/43693395)` .

### **A Proof of Theorem 4.1**


The aim of this section is to prove Theorem 4.1. We begin with a result that provides a few
essential properties for the the polynomial solving (7) when _T_ = 1. This result is known as
Chebyshev’s theorem [11] or the equioscillation theorem [50, Chapter 10].



_Proof._ A discussion can be found in [16]. Here we include a formal proof for completeness.
By Chebyshev’s Theorem [1, 11, 13] it is sufficient to show that P [odd] _d_ satisfies the Haar
condition: any non-zero _p ∈_ P [odd] _d_ = span _{x, . . ., x_ [3] _, . . ., x_ [2] _[q]_ [+1] _}_ can have at most _q_ roots in

[ _ℓ, u_ ].
Since deg( _p_ ) = _d_ = 2 _q_ + 1 we know that _p_ can have at most 2 _q_ + 1 roots in R. However,
since _p_ (0) = 0 and _p_ ( _x_ ) = _−p_ ( _−x_ ) we know that _p_ has one root at zero, and the remaining
roots come in symmetric pairs ( _x, −x_ ). Because of this, _p_ can have at most _q_ roots in
the positive orthant, and thus it can have at most _q_ roots in [ _ℓ, u_ ] _⊂_ (0 _, ∞_ ). Hence, P [odd] _d_
satisfies the Haar condition, which yields the desired result.


The proof of Theorem 4.1 will be by induction on _T_ . We begin by establishing the base
case, _T_ = 1, which is handled by the following result.


24


_Proof._ Throughout the proof we assume _d_ = 2 _q_ + 1. We begin with proving


_p_ _[⋆]_ ( _ℓ_ ) = min
_x∈_ [ _ℓ,u_ ] _[p][⋆]_ [(] _[x]_ [)] _[.]_


Consider the polynomial _e_ ( _x_ ) := 1 _−_ _p_ _[⋆]_ ( _x_ ). The proof will contain three steps. We first rule
out the trivial case that _p_ _[⋆]_ = 0, since _p_ ( _x_ ) = 2
_ℓ_ + _u_ _[x]_ [ would then be a better approximation.]
Hence, _p_ _[⋆]_ cannot be the zero polynomial.
_Step 1: e_ ( _x_ ) _has exactly q stationary points inside the open interval_ ( _ℓ, u_ ) _._
Note that _e_ ( _x_ ) has at most 2 _q_ stationary points in R, since its derivative _e_ _[′]_ ( _x_ ) is a
polynomial of degree 2 _q_ . Furthermore, since _p_ _[⋆]_ is odd, we have that _e_ _[′]_ ( _x_ ) = _−p_ _[′]_ ( _x_ ) is even
of degree 2 _q_, and thus can have at most _q_ stationary points contained in (0 _,_ + _∞_ ). Hence,
there can be at _most q_ stationary points of _e_ ( _x_ ) inside the interval [ _ℓ, u_ ].
By Lemma A.1 there are _q_ + 2 points _x_ 0 _, . . ., xq_ +1 _∈_ [ _ℓ, u_ ] where _e_ ( _x_ ) is maximized
or minimized in [ _ℓ, u_ ]. These points are either stationary points or they are endpoints of
the interval [ _ℓ, u_ ]. Let _n_ ext be the number of stationary points and _n_ stat be the number
of endpoints in the set _{x_ 0 _, . . ., xq_ +1 _}_ . Since a point can be both a stationary point and
an endpoint we have _q_ + 2 _≤_ _n_ end + _n_ stat. However, _n_ end _≤_ 2 and _n_ stat _≤_ _q_, which follows
from the previous paragraph where we showed that there are at most _q_ stationary points
of _e_ ( _x_ ) in [ _ℓ, u_ ]. So _n_ end + _n_ stat _≤_ _q_ + 2, and consequently we must have _n_ end = 2 and
_n_ stat = _q_, as required.
_Step 2: x_ = _ℓ_ _is a maximum of e_ ( _x_ ) _on the interval_ [ _ℓ, u_ ]
By Lemma A.1 and the discussion from Step 1, we know that _|e_ ( _x_ ) _|_ is maximized at
_q_ + 2 points inside [ _ℓ, u_ ] and _q_ of these points are contained inside the open interval ( _ℓ, u_ ).
Hence, _x_ = _ℓ_ must either be a maximum or a minimum of _e_ ( _x_ ). We will show that _x_ = _ℓ_
must be a maximum by contradiction.
Suppose _x_ = _ℓ_ was a minimum of _e_ ( _x_ ) on [ _ℓ, u_ ]. First note that _p_ _[⋆]_ is trivially nonnegative on [ _ℓ, u_ ], or else _p_ ( _x_ ) = 0 would be a better polynomial. Hence, since _p_ _[⋆]_ (0) = 0
we must have _p_ _[∗′]_ ( _δ_ ) _>_ 0 for some _δ ∈_ [0 _, ℓ_ ], or else the zero polynomial _p_ ( _x_ ) = 0 would be
a better approximation. Hence, for some _δ ∈_ [0 _, ℓ_ ] we have _e_ _[′]_ ( _δ_ ) _<_ 0.
We must also have _e_ _[′]_ ( _ℓ_ ) _≥_ 0 or else _x_ = _ℓ_ is not a minimum of _e_ ( _x_ ). Since _e_ _[′]_ ( _δ_ ) _<_ 0
for some _δ ∈_ [0 _, ℓ_ ] and _e_ _[′]_ ( _ℓ_ ) _≥_ 0, by the intermediate value theorem there exists a point
_x_ _[∗]_ _∈_ [0 _, ℓ_ ] such that _e_ _[′]_ ( _x_ _[∗]_ ) = 0. However, by the discussion above we know that all
stationary points of _e_ are contained inside the open interval ( _ℓ, u_ ). Hence, _x_ = _ℓ_ cannot be
a minimum of _e_ ( _x_ ) on [ _ℓ, u_ ]. However, by Step 1 we know that the endpoints of [ _ℓ, u_ ] must
be either minima or maxima of _e_ ( _x_ ). Hence, _x_ = _ℓ_ is a maximum of _e_ ( _x_ ) on [ _ℓ, u_ ].
_Step 3: Obtaining the desired equalities_
Since _e_ ( _x_ ) has a maximum in [ _ℓ, u_ ] at _x_ = _ℓ_, we have _p_ _[⋆]_ ( _ℓ_ ) = min
_x∈_ [ _ℓ,u_ ] _[p][⋆]_ [(] _[x]_ [). The other]

two equalities are immediate consequences of the equioscillation property of _p_ _[⋆]_ Lemma A.1
and that _x_ = _ℓ_ is a minimum of _p_ _[⋆]_ over the set [ _ℓ, u_ ].


25


With the above-mentioned result in hand, we are ready to prove Theorem 4.1.











_Proof._ The proof of (11) is an immediate consequence of Lemma A.2, since for each
_t_ = 1 _, . . ., T_, _pt_ is the optimal approximation in P [odd] _d_ to _x �→_ 1.
We now proceed with the proof of (10), which will be by induction. The proof for
_T_ = 1 is an immediate consequence of Lemma A.2 and we also have _p_ _[⋆]_ ( _ℓ_ ) = _ℓ_ 2 by (11).
Now suppose the result is true for all _t ≤_ _T −_ 1. For _t_ = 1 _, . . ., T −_ 1, note that the image
of _pt_ on [ _ℓt, ut_ ] is exactly [ _ℓt_ +1 _, ut_ +1] by i). Hence, if we define _g_ ( _x_ ) := _pT_ _−_ 1 _◦· · · ◦_ _p_ 1( _x_ ),
then the image of _g_ on [ _ℓ, u_ ] is [ _ℓT, uT_ ]. Furthermore, by i) we also have _g_ ( _ℓ_ ) = _ℓT_ . Pick
any _f_ such that _f ̸_ = _g_ and
_f_ = � _pT_ _−_ 1 _◦· · · ◦_ _p_ �1 _,_

for some � _p_ 1 _, . . .,_ - _pT_ _−_ 1 _∈_ P [odd] _d_ . Let the image of _f_ on [ _ℓ, u_ ] be [ _a, b_ ]. We will prove that
_a_
_b_ _[≤]_ _u_ _[ℓ][T]_ _T_ [by contradiction.]

Suppose _[a]_ _[ℓ][T]_ 2

_b_ _[>]_ _uT_ [. Define] _[ c]_ [ =] _a_ + _b_ [. Then, the image of the scaled function] _[ cf]_ [ on [] _[ℓ, u]_ [] is]

[ _ca, cb_ ] and _cf_ satisfies




_[T]_

_uT_ [by contradiction.]




_[a]_ _[ℓ][T]_

_b_ _[>]_ _u_



Suppose _[a]_



max _[b][ −]_ _[a]_
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[cf]_ [(] _[x]_ [)] _[|]_ [ = max] _[ {]_ [1] _[ −]_ _[ca, cb][ −]_ [1] _[}]_ [ =] _a_ + _b_ _[.]_



Recall by our inductive hypothesis, we have max
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[g]_ [(] _[x]_ [)] _[|]_ [ = 1] _[ −]_ _[ℓ][T]_ [ =] _[ u][T][ −]_ [1 where the]

second equality holds by (11). It follows that



_a_



_a_ _[ℓ][T]_

_b_ _[>]_ _uT_



_uT_



_ℓT_

_⇔_ _[a]_

_b_ _[>]_ 2 _−_ _ℓT_



2 _a_
_⇔_ _ℓT <_
_a_ + _b_

_⇔_ 1 _−_ _ℓT >_ _[b][ −]_ _[a]_

_a_ + _b_
_⇔_ max
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[g]_ [(] _[x]_ [)] _[|][ >]_ [ max] _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[cf]_ [(] _[x]_ [)] _[|][,]_


26


which leads to a contradiction to our inductive hypothesis that _g_ is optimal. Hence, we
must have _[a]_

_b_ _[≤]_ _u_ _[ℓ][T]_ _T_ [.]

Consequently, using that _[a]_ _b_ _[≤]_ _u_ _[ℓ][T]_ _T_ [, we will show that for any][ �] _[p][T][ ∈]_ [P] _d_ [odd] and for any

_f_ = � _pT_ _−_ 1 _◦· · · ◦_ _p_ �1 � _pT ◦_ _f_ cannot be a better approximation than _pT ◦_ _g_ . In particular, we
have




_[a]_

_b_ _[≤]_ _u_ _[ℓ][T]_




_[T]_
_b_ _[≤]_ _uT_ [.]

Consequently, using that _[a]_




_[a]_

_b_ _[≤]_ _u_ _[ℓ][T]_



max max
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [�] _[T]_ [ (] _[f]_ [(] _[x]_ [))] _[| ≥]_ _p_ [min] _∈_ P _[∗]_ _d_ _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[f]_ [(] _[x]_ [))] _[|]_


= min max
_p∈_ P _[∗]_ _d_ _x∈_ [ _a,b_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_


= min max
_p∈_ P _[∗]_ _d_ _x∈_ [ _a/b,_ 1] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_

_≥_ min max
_p∈_ P _[∗]_ _d_ _x∈_ [ _ℓT /uT,_ 1] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_


= min max
_p∈_ P _[∗]_ _d_ _x∈_ [ _ℓT,uT_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_


= min max
_p∈_ P _[∗]_ _d_ _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[g]_ [(] _[x]_ [))] _[|]_


= max
_x∈_ [ _ℓT,uT_ ] _[|]_ [1] _[ −]_ _[p][T]_ [ (] _[g]_ [(] _[x]_ [))] _[|]_ [ = 1] _[ −]_ _[p][T]_ [ (] _[ℓ][T]_ [ ) = 1] _[ −]_ _[ℓ][T]_ [+1] _[,]_


where the second and third equality follow by changing variables _y_ = _x/b_ so that


min max max max
_p∈_ P _[∗]_ _d_ _x∈_ [ _a,b_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_ [ = min] _p∈_ P _[∗]_ _d_ _y∈_ [ _a/b,_ 1] _[|]_ [1] _[ −]_ _[p]_ [(] _[by]_ [)] _[|]_ [ = min] _p∈_ P _[∗]_ _d_ _y∈_ [ _a/b,_ 1] _[|]_ [1] _[ −]_ _[p]_ [(] _[y]_ [)] _[|]_


and this last equality follows because the space P _[∗]_ _d_ [is invariant under input rescaling; that]
is, for any _b ̸_ = 0, the map _x �→_ _bx_ preserves the space span _{x, x_ [3] _, . . ., x_ _[d]_ _}_ . This concludes
the proof.

### **B Proof of Theorem 4.3**


In this section we provide the proof of the convergence guarantee stated in Theorem 4.3.





_Proof._ Define
_p_ _[⋆]_ = argmin
_p_ = _pT ◦pT −_ 1 _◦···◦p_ 1
_pt∈_ P _[∗]_ _d_



max
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|][ .]_



Then Algorithm 2 returns _**X**_ _T_ = _p_ _[⋆]_ ( _**M**_ ). Let _h ∈_ P _q_ be [ _q/_ 0] Pad´e-approximant to
(1 _−_ _x_ ) _[−]_ [1] _[/]_ [2] [28, Section 3] and define _p_ ( _x_ ) = _xh_ (1 _−_ _x_ [2] ) _∈_ P [odd] _d_ . Define _f_ = _p ◦· · · ◦_ _p_ as


27


the composition of _p_ with itself _T_ times. Then, by Theorem 4.1, [28, Theorem 3.1], and
_f_ ( _x_ ) _≥_ 0 for _x ≥_ 0 we have


_∥_ sign( _**M**_ ) _−_ _**X**_ _T ∥_ 2 _≤_ max
_x∈_ [ _ℓ,_ 1] _[|]_ [1] _[ −]_ _[p][⋆]_ [(] _[x]_ [)] _[|]_


_≤_ max
_x∈_ [ _ℓ,_ 1] _[|]_ [1] _[ −]_ _[f]_ [(] _[x]_ [)] _[|]_







_≤_ max
_x∈_ [ _ℓ,_ 1]




- _|_ 1 _−_ _x_ [2] _|_ [(] _[d]_ [+1)] _[T]_


1 + _f_ ( _x_ )



_≤|_ 1 _−_ _ℓ_ [2] _|_ [(] _[d]_ [+1)] _[T]_ _,_


as required.

### C Proof of equivalence between (6) and (7)


In this section we provide a proof for the equivalence between (6) and (7). It is sufficient
to show that for any fixed polynomial _p_ we have


_ε_ 1 := _**M**_ max _∈_ R _[m][×][n]_ _∥_ polar( _**M**_ ) _−_ _p_ ( _**M**_ ) _∥_ 2 = max _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_ [ :=] _[ ε]_ [2] _[.]_
_σ_ ( _**M**_ ) _⊂_ [ _ℓ,u_ ]


For any fixed _**M**_, by the unitary invariance of the spectral norm we immediately have


_∥_ polar( _**M**_ ) _−_ _p_ ( _**M**_ ) _∥_ 2 = max
_σi∈σ_ ( _**M**_ ) _[|]_ [1] _[ −]_ _[p]_ [(] _[σ][i]_ [)] _[| ≤]_ _x_ [max] _∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|][ .]_


Consequently, _ε_ 1 _≤_ _ε_ 2.
Suppose that _x_ _[∗]_ _∈_ [ _ℓ, u_ ] is chosen so that _|_ 1 _−_ _p_ ( _x_ _[∗]_ ) _|_ = max _x∈_ [ _ℓ,u_ ] _|_ 1 _−_ _p_ ( _x_ ) _| ._ Without
loss of generality, assume _m ≥_ _n_ . Letting _**M**_ = _x_ _[∗]_ _**UV**_ [T], for any matrix _**U**_ _∈_ R _[m][×][n]_ and
_**V**_ _∈_ R _[n][×][n]_ with orthonormal columns, and noting polar( _**M**_ ) = _**UV**_ [T] yields


_ε_ 1 _≥∥_ polar( _**M**_ ) _−_ _p_ ( _**M**_ ) _∥_ 2
= _∥_ _**I**_ _n −_ _p_ ( _x_ _[∗]_ ) _**I**_ _n∥_ 2
= _|_ 1 _−_ _p_ ( _x_ _[∗]_ ) _|_


= max
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_ [ =] _[ ε]_ [2]


Consequently, _ε_ 1 _≥_ _ε_ 2. Hence, _ε_ 1 = _ε_ 2, as desired.

### **D Remez algorithm**


In this section, we show in detail how to solve (14). By Theorem 4.1, this also gives a
solution to (7). We give a closed form solution for _d_ = 3. We then describe how the
Remez algorithm [42, 43] can be used to approximate _pt_ for arbitrary _d_ . We finish with
Algorithm 3, a simplified version of Remez for solving (14) with _d_ = 5. Recall (14):


argmin max (17)
_p∈_ P [odd] _d_ _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_


We begin with the case when _d_ = 3. In this case, there is a simple closed form for
the optimal odd polynomial _p_ _[⋆]_ _∈_ P [odd] 3 as described in Section 4.2. On a given interval


28


[ _ℓ, u_ ], the optimal approximation to the constant function _x �→_ 1 is given by the scaled and
shifted Newton-Schulz polynomial _p_ NS( _x_ ) = [3] _[x][ −]_ [1] _[x]_ [3][:]




[3]

2 _[x][ −]_ [1] 2




[1]

2 _[x]_ [3][:]







3 4

[and] _[ β]_ [ =] _[.]_
_u_ [2] + _ℓu_ + _ℓ_ [2] 2 + _ℓu_ ( _ℓ_ + _u_ ) _α_ [3]



_p_ _[⋆]_ ( _x_ ) = _βp_ NS( _αx_ ) _,_ where _α_ =



One can verify that this polynomial satisfes the equioscillation condition from Lemma A.1
at _x_ = _ℓ,_ [1] _[, u]_ [ as given in][ (][15][)][, with] ~~�~~ _−a/_ (3 _b_ ) = 1 _/α_ and _E_ = _β −_ 1. Therefore, it must




[1] ~~�~~

_α_ _[, u]_ [ as given in][ (][15][)][, with]



at _x_ = _ℓ,_ ~~�~~ _−a/_ (3 _b_ ) = 1 _/α_ and _E_ = _β −_ 1. Therefore, it must

_α_ _[, u]_ [ as given in][ (][15][)][, with]

necessarily be the optimal approximation from P [odd] 3 . Note that when _u_ = 1, the function
_x �→_ _p_ NS( _αx_ ) is the same polynomial derived by Chen and Chow [12].
Unfortunately, for larger _d_, finding closed form expressions for optimal approximations
from P [odd] _d_ becomes challenging, and we know of no closed form solution. However, we can
approximate the optimal polynomial using the Remez algorithm. Let _d_ = 2 _q_ + 1. Again
recalling Lemma A.1, the optimal polynomial must satisfy the equioscillation property
at a set of _q_ + 2 points, as in (15). The Remez algorithm finds the equioscillation points
_A_ = _{x_ 0 _, . . ., xq_ +1 _}_ from Lemma A.1 by iteratively refining a sequence of trial points
_A_ [(] _[k]_ [)] = _{x_ [(] 0 _[k]_ [)] _[, . . ., x]_ _q_ [(] _[k]_ +1 [)] _[}]_ [ so that] _[ A]_ [(] _[k]_ [)][ converges to] _[ A]_ [. From the sequence of trial points]
_A_ [(] _[k]_ [)] the algorithm also finds a sequence of polynomials _p_ [(] _[k]_ [)] so that _p_ [(] _[k]_ [)] converges to the
optimal polynomial. The convergence is very fast, and usually 10 iterations is sufficient
to converge to the optimal polynomial up to double precision machine epsilon [42]. More
commonly, the Remez algorithm is used to find optimal polynomial approximations to
general continuous functions where _d ≈_ 100 or even _d ≈_ 1000. However, because the
polynomial we build to approximate sign( _x_ ) is a composition of polynomials, each of which
has a low degree, in our setting the degree _d_ is small, usually _d_ = 5. For _d_ = 5 the Remez
algorithm simplifies significantly. We now describe this simplified algorithm.
We first choose an initial set of trial points _A_ [(1)], which ideally should come close to
satisfying the equioscillation property. From Lemma A.1, the unique optimal approximation
_p_ _[⋆]_ _∈_ P [odd] 5 satisfies the equioscillation property at four points in [ _ℓ, u_ ]. Since the function
we wish to approximate is constant, the equioscillation points must be extrema of _p_ _[⋆]_ on

[ _ℓ, u_ ]. Because _p_ _[⋆]_ is a odd quintic, it can have at most two local extrema on the positive
real line, and thus at most two local extrema on [ _ℓ, u_ ]. The other two equioscillation points
must therefore be the endpoints _ℓ_ and _u_ . Since we know that _ℓ_ and _u_ must be among
the true equioscillation points, we always include them in our set of trial points. For
notational simplicity, we call the other two points _q_ and _r_ . We initialize _q_ 1 = [1] _[ℓ]_ [+] [3] _[u]_



notational simplicity, we call the other two points _q_ and _r_ . We initialize _q_ 1 = 4 _[ℓ]_ [+] 4 _[u]_

and _r_ 1 = [3] _[ℓ]_ [+] [1] _[u]_ [, since we observe that as] _[ ℓ]_ _[→]_ _[u]_ [ these are approximately the other two]




[3] [1]

4 _[ℓ]_ [+] 4




[1] [3]

4 _[ℓ]_ [+] 4



and _r_ 1 = 4 _[ℓ]_ [+] 4 _[u]_ [, since we observe that as] _[ ℓ]_ _[→]_ _[u]_ [ these are approximately the other two]

equioscillation points.
We now show how to refine a candidate set of trial points _A_ [(] _[k]_ [)] = _{ℓ, qk, rk, u}_ to produce
_A_ [(] _[k]_ [+1)] = _{ℓ, qk_ +1 _, rk_ +1 _, u}_ as well as an approximately equioscillating polynomial _pk_ . For
any fixed set of trial points, we can find a degree-5 odd polynomial _pk_ ( _x_ ) = _akx_ + _bkx_ [3] + _ckx_ [5]

that satisfies


_pk_ ( _ℓ_ ) = 1 _−_ _Ek,_ _pk_ ( _qk_ ) = 1 + _Ek,_ _pk_ ( _rk_ ) = 1 _−_ _Ek,_ _pk_ ( _u_ ) = 1 + _Ek_ (18)



for some _Ek_ by solving a linear system in _ak, bk, ck_ and _Ek_ . This can be rewritten as
follows:
 [3] [5]     



_ℓ_ _ℓ_ [3] _ℓ_ [5] 1
_qk_ _qk_ [3] _qk_ [5] _−_ 1
_rk_ _rk_ [3] _rk_ [5] 1
_u_ _u_ [3] _u_ [5] _−_ 1



























_ak_
_bk_
_ck_
_Ek_



=








1
1
1
1



_._ (19)












29


If _A_ [(] _[k]_ [)] were the extrema of the error function _ek_ ( _x_ ) = 1 _−_ _pk_ ( _x_ ) on [ _ℓ, u_ ], then they would
be an equioscillating set for _pk_, and _pk_ would be the solution. Therefore, to refine _A_ [(] _[k]_ [)],
we find the extrema of _ek_ ( _x_ ) = 1 _−_ _pk_ ( _x_ ). These can occur at _ℓ, u_ and the roots of _e_ _[′]_ _k_ [(] _[x]_ [).]
Setting _e_ _[′]_ _k_ [(] _[x]_ [) = 0 yields the quartic equation 5] _[c][k][x]_ [4][ + 3] _[b][k][x]_ [2][ +] _[ a][k]_ [ = 0, whose two solutions]
are given explicitly by the _quadratic_ formula after the substitution _y_ = _x_ [2] . We set _qk_ +1
and _rk_ +1 to be the solutions to this equation and let _A_ [(] _[k]_ [+1)] = _{ℓ, qk_ +1 _, rk_ +1 _, u}_ . We repeat
the procedure until _|Ek|_ := max
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p][k]_ [(] _[x]_ [)] _[| ≈]_ _x_ [max] _∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p][k]_ [+1][(] _[x]_ [)] _[|]_ [ =:] _[ |][E][k]_ [+1] _[|]_ [.]

We note that the matrix appearing in (19) is a Vandermonde matrix. Vandermonde
matrices become notoriously ill-conditioned as the degree grows large [17, Section 4.6].
However, since in our setting we choose _d_ to be small, there is no ill-conditioning due to
large degrees. Instead, we observe ill-conditioning when _ℓ_ _≈_ _u_ . However, as _ℓ/u →_ 1 the
optimal polynomial will converge to the polynomial _[x/u]_ �15 _−_ 10( _x/u_ )2 + 3( _x/u_ )4�, which



optimal polynomial will converge to the polynomial _[x/u]_ �15 _−_ 10( _x/u_ )2 + 3( _x/u_ )4�, which

8
can be verified by noting that as _ℓ/u →_ 1 all equioscillation points _x_ 0 _, x_ 1 _, x_ 2 _, x_ 3 must
converge to _u_ . For general _d_ = 2 _q_ + 1, the polynomial will converge to ( _x/ℓ_ ) _h_ (1 _−_ ( _x/ℓ_ ) [2] )
where _h ∈_ P _q_ is the [ _q/_ 0] Pad´e approximant to (1 _−_ _x_ ) [1] _[/]_ [2] [28]. In fact, this polynomial is
extremely close to the optimal polynomial for sufficiently large _ℓ_ . To see this, let _p_ _[⋆]_ be the
optimal approximation from P [odd] 5 and let _p_ ( _x_ ) = _[x/u]_ 8 �15 _−_ 10( _x/u_ )2 + 3( _x/u_ )4�. Then,


max
_x∈_ [ _ℓ,u_ ] _[|][p][⋆]_ [(] _[x]_ [)] _[ −]_ _[p]_ [(] _[x]_ [)] _[| ≤]_ _x_ [max] _∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_ [ + max] _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p][⋆]_ [(] _[x]_ [)] _[|]_


_≤_ 2 max
_x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_

_≤_ 2 (1 _−_ _ℓ/u_ ) [3] _._


where we invoked [28, Theorem 3.1] and the fact that _p_ _[⋆]_ is the optimal approximation
to _x �→_ 1 from P [odd] 5 . Hence, when _ℓ/u ≥_ 1 _−_ _ϵ_ [1] _d_ _[/]_ [3], where _ϵ_ double _≈_ 1 _._ 1 _×_ 10 _[−]_ [16] is the
double precision machine epsilon, then _|p_ _[⋆]_ ( _x_ ) _−_ _p_ ( _x_ ) _| ≤_ 2 _ϵ_ double. In other words, up to
double precision machine epsilon, _p_ _[⋆]_ is equal to _p_ . Therefore, whenever _ℓ/u ≥_ 1 _−_ _ϵ_ [1] double _[/]_ [3]
the algorithm simply returns the Pad´e approximant (that is, the scaled Newton-Schulz
polynomial).
The full algorithm is given in Algorithm 3. In our experiments, we never observed
Algorithm 3 taking more than five iterations to converge. This algorithm is implemented
in full in Appendix G.

### **E Initialization for Matrices with Large Spectral Gaps**


In Section 4, we constructed a sequence of polynomials that is adapted to the range of
the singular values [ _ℓ, u_ ]. Assuming nothing else about the input, these polynomials are
optimal since they provide a good approximation to 1 across the entire interval. However,
in many applications, the spectrum has large gaps; that is, there are several large outlying
singular values that are well-separated from the rest. For these matrices, it is not necessary
for the polynomial to be accurate on the entire interval [ _ℓ, u_ ], only on the range of the
small singular values plus a few other isolated points. In this section, we take advantage
of this structure to accelerate our method by preprocessing the matrix to eliminate the
largest singular values.
The first step is to find small intervals containing each of these large singular values. To
find lower bounds, we use subspace iteration, which is a generalization of the power method
that approximates multiple singular values simultaneously. Fix _k_, the number of singular


30



�15 _−_ 10( _x/u_ )2 + 3( _x/u_ )4�. Then,
8


**Algorithm 3** Remez algorithm (degree 5 approximation for sign( _x_ ))

**input:** interval [ _ℓ, u_ ] for _u > ℓ>_ 0.
**output:** Approximation _p ∈_ P [odd] 5 to _p_ _[⋆]_ = argmin max
_p∈_ P [odd] 5 _x∈_ [ _ℓ,u_ ] _[|]_ [1] _[ −]_ _[p]_ [(] _[x]_ [)] _[|]_ [.]



**define** _ϵ_ double = 1 _._ 11 _×_ 10 _[−]_ [16]



**if** _ℓ/u ≥_ 1 _−_ _ϵ_ [1] double _[/]_ [3] **[then]**
Return _p_ ( _x_ ) = _[x/u]_ �15



�15 _−_ 10( _x/u_ )2 + 3( _x/u_ )4�
8



**end if**
_q_ 1 = [1]




[3] 4 _[u,]_ _r_ 1 = [3] 4




[1] [3]

4 _[ℓ]_ [+] 4




[3] [1]

4 _[ℓ]_ [+] 4



_q_ 1 = 4 _[ℓ]_ [+] 4 _[u,]_ _r_ 1 = 4 _[ℓ]_ [+] 4 _[u]_ [.]

_E_ 0 = _∞,_ _E−_ 1 = _−∞_
_k ←_ 0
**while** _||Ek| −|Ek−_ 1 _|| > ϵ_ double **do**



_k ←_ _k_ + 1
  




_−_ 1 []















_ak_
_bk_
_ck_
_Ek_















_ℓ_ _ℓ_ [3] _ℓ_ [5] 1
_qk_ _qk_ [3] _qk_ [5] _−_ 1
_rk_ _rk_ [3] _r_ 1 [5] 1
_u_ _u_ [3] _u_ [5] _−_ 1







1
1
1
1











=





~~�~~



109 _bc_ [2] _kk_ _[−]_ [20] _[a][k][c][k]_ _,_ _rk_ +1 =




_−_ 3 _bk_ + ~~_[√]_~~



_qk_ +1 = _−_ 3 _bk−_ 109 _bc_ [2] _kk_ _[−]_ [20] _[a][k][c][k]_ _,_ _rk_ +1 = _−_ 3 _bk_ + 109 _bc_ [2] _kk_ _[−]_ [20] _[a][k][c][k]_

**end while**
Return _p_ ( _x_ ) = _akx_ + _bkx_ [3] + _ckx_ [5]




~~�~~



_qk_ +1 =




_−_ 3 _bk−_ ~~_[√]_~~



values we wish to eliminate. Letting _σ_ 1 _≥· · · ≥_ _σn_ denote the singular values of _**M**_,
subspace iteration produces estimates ˜ _σ_ 1 _≥· · · ≥_ _σ_ ˜ _k_ satisfying _σi ≥_ _σ_ ˜ _i_ for all _i ∈_ 1 _, . . ., k_ . [7]

To find upper bounds on each _σi_, we can use the fact that _∥_ _**M**_ _∥_ [2] F [=][ �] _j_ _[n]_ =1 _[σ]_ _j_ [2] [as follows:]



_n_



_j_ =1
_j_ = _i_



_σj_ [2] _[≤∥]_ _**[M]**_ _[∥]_ F [2] _[−]_






_σi_ [2] [=] _[ ∥]_ _**[M]**_ _[∥]_ F [2] _[−]_


That is, for each _i ∈_ [ _n_ ],



_σj_ [2] _[≤∥]_ _**[M]**_ _[∥]_ F [2] _[−]_






_k_



_j_ =1
_j_ = _i_



_k_



_j_ =1
_j_ = _i_



_σ_ ˜ _j_ [2] (20)



_k_



_j_ =1
_j_ = _i_




~~�~~

~~�~~



- F _[−]_

- _[∥]_ _**[M]**_ _[∥]_ [2]







_σi ∈_



_σ_ ˜ _i,_




_σ_ ˜ _j_ [2]



Setting _i_ = _k_ + 1, the above also provides an upper bound for the tail of the spectrum,
_σk_ +1 _, . . ., σn_ .
The second step is to find an odd polynomial that well-approximates the constant
function on each of these intervals and on the tail simultaneously. For simplicity, we treat
only the _k_ = 1 case here. Assume that _**M**_ is normalized to _∥_ _**M**_ _∥_ F = 1 and let _z_ = ˜ _σ_ 1
be the lower bound produced by subspace iteration (which reduces to the power method
_√_
in this case). Then (20) gives _σ_ 1 _∈_ [ _z,_ 1] and _σ_ 2 _, . . ., σn ≤_ 1 _−_ _z_ [2] . Assume that these



in this case). Then (20) gives _σ√_ 1 _∈_ [ _z,_ 1] and _σ_ 2 _, . . ., σ√n ≤_ 1 _−_ _z_ [2] . Assume that these

intervals do not overlap, that is, 1 _−_ _z_ [2] _≤_ _z ⇔_ _z ≥_ 1 _/_ 2. Then we construct the unique


7Let _**Q**_ 0 _∈_ R _n×k_ be a random matrix with orthonormal columns and define _**Q**_ _t_ +1 _,_ _**R**_ _t_ +1 = `qr` [�] _**M**_ _[⊤]_ _**MQ**_ _t_ �,
where `qr` is the QR decomposition. Subspace iteration outputs the singular values ˜ _σ_ 1 _, . . .,_ ˜ _σk_ of _**MQ**_ _T_,
_σ_ ˜1 _, . . .,_ ˜ _σk_ . By the Cauchy interlacing theorem, ˜ _σk ≤_ _σk_ .


31



2 _√n_

1 _−_ _z_ [2] _≤_ _z ⇔_ _z ≥_ 1 _/_



2. Then we construct the unique


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0





0 5 10 15 20 25
Iteration


Figure 8: Benefits of the spectrum-aware initialization scheme of Appendix E. Using this
scheme improves convergence of both Newton-Schulz and `Polar Express` on a synthetic
32 _×_ 32 matrix with _σj_ ( _**M**_ ) = _j_ _[−]_ [5] . Note that we count the spectrum-aware initialization
as an additional iteration.



_√_
odd cubic polynomial _p_ ( _x_ ) = _ax_ + _bx_ [3] that satisfies _p_ ( 1 _−_ _z_ [2] ) = 1 and _p_ ( _z_ ) = 1 by setting



_√_
_a_ = _[z]_ [2][(] _[z]_ [ +] ~~_√_~~



1 _−_ _z_ [2]



_√_



~~_√_~~
_z_



_b_ =
1 _−_ _z_ [2] (2 _z_ [2] _−_ 1)



_√_
1 _−_ _z_ [2] ) _−_



1 _−_ _z_ [2] _−_ _z_
~~_√_~~
_z_ 1 _−_ _z_ [2] (2 _z_ [2]



(21)
1 _−_ _z_ [2] (2 _z_ [2] _−_ 1)



Because _p_ (0) = 0 and _p_ has at most one local extremu _√_ m on R _≥_ 0, these conditions
immediately guarantee that _p_ is concave-increasing on [0 _,_ 1 _−_ _z_ [2] ], so it must lie above



immediately guarantee that _p_ is concave-increasing on [0 _,_ 1 _−_ _z_ [2] ], so it must lie above

_√_
the line _x �→_ _x/_ 1 _−_ _z_ [2] . Furthermore, _p_ is decreasing on [ _σ_ 1 _,_ 1], so it maps _σ_ 1 _∈_ [ _z,_ 1] to



the line _x �→_ _x/_ 1 _−_ _z_ [2] . Furthermore, _p_ is decreasing on [ _σ_ 1 _,_ 1], so it maps _σ√_ 1 _∈_ [ _z,_ 1] to

[ _p_ (1) _,_ 1]. By minimizing _p_ (1) over all valid _z_ (that is, over the interval _z ∈_ [1 _/_ 2 _,_ 1]), one

_√_
can further show that _p_ (1) _>_ 1 _/_ 2, so _σ_ 1 cannot be decreased very much by applying _p_ .

Thus, the largest singular value of _p_ ( _**M**_ ) is still at most 1, while the smaller singular values
_√_
have increased by a potentially large factor of 1 _/_ 1 _−_ _z_ [2] . When there is a large outlying



have increased by a potentially large factor of 1 _/_ 1 _−_ _z_ [2] . When there is a large outlying

singular value, _z_ is close to 1 and this initialization scheme makes much more progress
than a standard iteration of `PolarExpress` would have.
In Figure 8, we demonstrate the benefit of using the _p_ given by (21) on a synthetic
matrix whose spectrum follows a power law decay. That is, _σj_ ( _**M**_ ) = _j_ _[−]_ [5], so this matrix
has a large outlying singular value _σ_ 1 _≫_ _σ_ 2. Applying (21) costs almost as much as
performing an iteration of a degree-5 polynomial method, so for fair comparison, we count
it as an additional iteration in this plot. For both Newton-Schulz and `Polar Express`,
performing the extra spectrum-aware initialization step described in this section leads to
significant speedups in convergence.


### **F Fast Polynomial Iteration for Rectangular Matrices**

In this section, we describe a simple method for applying an iterative polynomial method to
a rectangular matrix. For matrices with a large aspect ratio, this method yields significant
computational savings. We emphasize that this method is applicable to _any_ computation
of the form ( _pT ◦· · · ◦_ _p_ 1)( _**X**_ ), where each _pt_ is an odd polynomial. Thus, it can be used to
apply Newton-Schulz or Jordan’s polynomials in addition to our own.
As a preliminary, we first describe the baseline approach. Let _**X**_ _∈_ R _[m][×][n]_ with
_m ≥_ _n_, where _α_ := _m/n ≥_ 1 is called the aspect ratio. Any odd polynomial _p_ of degree


32


_d_ = 2 _q_ + 1 can be represented as _p_ ( _x_ ) = _xh_ ( _x_ [2] ), where _h_ is a polynomial of degree _q_ . Thus,
_p_ ( _**X**_ ) = _**X**_ _h_ ( _**X**_ _[⊤]_ _**X**_ ). Furthermore, _h_ can be written in a factored form called Horner’s rule
to reduce the number of multiplications. For instance, if _h_ ( _y_ ) = _a_ + _by_ + _cy_ [2] + _dy_ [3], Horner’s
rule gives _h_ ( _y_ ) = _a_ + _y_ ( _b_ + _y_ ( _c_ + _dy_ )). For a matrix, _h_ ( _**Y**_ ) = _a_ _**I**_ + _**Y**_ ( _b_ _**I**_ + _**Y**_ ( _c_ _**I**_ + _d_ _**Y**_ )).
Thus for _**Y**_ _∈_ R _[n][×][n]_, computing _h_ ( _**Y**_ ) costs about (deg( _h_ ) _−_ 1) _· n_ [3] operations, and
computing _p_ ( _**X**_ ) = _**X**_ _h_ ( _**X**_ _[⊤]_ _**X**_ ) costs 2 _mn_ [2] + - _d−_ 1 _−_ 1� _· n_ [3] = - _d−_ 3 + 2 _α_ - _· n_ [3] operations.



computing _p_ ( _**X**_ ) = _**X**_ _h_ ( _**X**_ _[⊤]_ _**X**_ ) costs 2 _mn_ [2] + - _d−_ 1 _−_ 1� _· n_ [3] = - _d−_ 3 + 2 _α_ - _· n_ [3] operations.

2 2

This process could be repeated for each iteration _p_ 1 _, . . ., pT_ . Notice that if we instead
computed _h_ ( _**XX**_ _[⊤]_ ) _**X**_, the result would be the same but the cost would be higher.
A major drawback of this naive approach is that it has a strong dependence on _α_, since
two rectangular matrix multiplications must be performed in _each_ of the _T_ iterations. When
_m ≫_ _n_, these two multiplications dominate the cost. In Algorithm 4, we introduce a simple
trick that dramatically reduces this cost, using just two rectangular matrix multiplications
to compute _all T_ iterations.




_−_ 1 _−_ 1� _· n_ [3] = - _d−_ 3

2 2



**Algorithm 4** Fast Polynomial Iteration for Rectangular Matrices
**input:** _**X**_ _∈_ R _[m][×][n]_ with _m >_ 1 _._ 5 _n_, odd polynomials _p_ 1( _x_ ) = _xh_ 1( _x_ [2] ) _, . . ., pT_ ( _x_ ) = _xhT_ ( _x_ [2] ).
**output:** The matrix ( _pT ◦· · · ◦_ _p_ 1)( _**X**_ ).

_**Y**_ = _**X**_ _[⊤]_ _**X**_ _▷mn_ [2]
Let _**Q**_ 0 = _**I**_
**for** _t_ = 1 _,_ 2 _, . . ., T_ **do**

_**R**_ _t_ = _**Q**_ _[⊤]_ _t−_ 1 _**[Y Q]**_ _[t][−]_ [1] _▷_ 2 _n_ [3]
_**Q**_ _t_ = _**Q**_ _t−_ 1 _ht_ ( _**R**_ _t_ ) _▷_ Horner’s rule: deg( _ht_ ) _· n_ [3]
**end for**
**return** _**XQ**_ _T_ _▷mn_ [2]


To see why this works, define _q_ 0( _x_ ) = _x_,



_qt_ ( _x_ ) = [(] _[p][t][ ◦· · · ◦]_ _[p]_ [1][)(] _[x]_ [)]




_[p]_ [1][)(] _[x]_ [)]

= _[p][t]_ [ ((] _[p][t][−]_ [1] _[ ◦· · · ◦]_ _[p]_ [1][)(] _[x]_ [))]
_x_ _x_




_[p]_ [1][)(] _[x]_ [))]

= _[p][t]_ [ (] _[xq][t][−]_ [1][(] _[x]_ [))]
_x_ _x_



(22)
_x_




        - 2�
( _xqt−_ 1( _x_ ))          = _[xq][t][−]_ [1][(] _[x]_ [)] _[ ·][ h][t]_ = _qt−_ 1( _x_ ) _· ht_ _x_ [2] _· qt−_ 1( _x_ ) [2][�] (23)

_x_



and _rt_ ( _x_ ) = _x_ [2] _· qt−_ 1( _x_ ) [2] . It is clear by induction that _**R**_ _t_ = _rt_ ( _**X**_ ) _,_ _**Q**_ _t_ = _qt_ ( _**X**_ ), and
_**XQ**_ _T_ = ( _pt ◦· · · ◦_ _p_ 1)( _**X**_ ). As promised, this algorithm uses no rectangular multiplications
in the for-loop. If each _pt_ is degree _d_, then the total cost is - _d_ +32 _[T]_ [ + 2] _[α]_ - _· n_ [3] . When

_α >_ 1 _._ 5 _T_
_T_ _−_ 1 [, this is smaller than the naive method. We can use this criterion to select]
either Algorithm 4 or the baseline method at runtime.

Algorithm 4 can introduce numerical errors, especially when working in a low precision
format like `bfloat16` . We identify two sources of numerical trouble and propose remedies
for each. The first is due to the ill-conditioning of _**X**_ . Let _**X**_ = _**U**_ **Σ** _**V**_ _[⊤]_ be the SVD. For
large _T_, ( _pT ◦· · · p_ 1)( _**X**_ ) = _**XQ**_ _T ≈_ polar( _**X**_ ) = _**UV**_ _[⊤]_ . Thus, _**Q**_ _T ≈_ _**V**_ _[⊤]_ **Σ** _[−]_ [1] _**V**_ . When _**X**_
has very small singular values and the floating point precision is very low, instantiating
_**Q**_ _T_ may be unstable. To mitigate this issue, we use a restarting strategy. Notice that the
issue arises only for large _T_, for which ( _pT ◦· · · ◦_ _p_ 1)( _ϵ_ ) _≈_ 1. Limiting ourselves to _T_ = 3
iterations improves the conditioning of _**Q**_ _T_ because ( _pT ◦· · ·◦_ _p_ 1)( _ϵ_ ) _≪_ 1. Thus, to compute
_T >_ 3 iterations, we begin with _**X**_ 0 and apply Algorithm 4 with the first three polynomials,
producing _**X**_ 3. When then apply Algorithm 4 again with the next three polynomials to
_**X**_ 3, producing _**X**_ 6, and so on. As _**X**_ _t_ approaches convergence, its conditioning improves


33


10 [0]


10 _[−]_ [1]


10 _[−]_ [2]



_n_
1024
2048
4096
Restart Interval
1
3
6



10 [0] 10 [1]

Aspect Ratio ( _α_ = _m/n_ )


Figure 9: Effects of using Algorithm 4 on runtime on a GPU. We run _T_ = 6 iterations of a
degree-5 polynomial method on matrices with various dimensions _n_ and aspect ratios _α_ .
Restart interval = 6 is Algorithm 4, restart interval = 1 is equivalent to the baseline (that
is, not using Algorithm 4), and restart interval = 3 is an intermediate method that calls
Algorithm 4 once to do the first three iterations and again to do the last three iterations
for greater stability. When _α ≫_ 1, increasing the restart interval _significantly_ reduces the
runtime.


and we may no longer need to restart at all. Note that restarting Algorithm 4 after every
iteration is exactly the same as the baseline method.
Second, while the matrix _**Y**_ is positive definite in exact arithmetic, numerical round-off
can introduce spurious negative eigenvalues that cause the method to diverge to infinity.
To combat this issue, we instead set _**Y**_ = _**X**_ _[⊤]_ _**X**_ + 10 _[−]_ [3] _**I**_ during the first application of
Algorithm 4. (We also normalize by _∥_ _**X**_ _∥_ F +10 _[−]_ [3] instead of _∥_ _**X**_ _∥_ F.) In subsequent restarts
of Algorithm 4, we set _**Y**_ = _**X**_ _[⊤]_ _**X**_ as before. This is akin to slightly increasing each of the
singular values of _**X**_, but it does _not_ change the polar factor of _**X**_ . Thus, while the output
will be slightly different in the early iterations, the algorithm still converges to the correct
answer.

Figure 9 shows that using Algorithm 4 can significantly improve runtime on the GPU
when the aspect ratio is large enough. As expected, using Algorithm 4 for many iterations
significantly reduces the dependence of the runtime on the aspect ratio. Running six
iterations of a degree-5 polynomial method when _α_ = 4 (as with the linear transformations
in each MLP block of a transformer) we obtain almost a 2x speedup, and when _α_ = 32, we
obtain a 5x speedup. If we restart every three iterations, the trend is the same but the
runtime savings are somewhat smaller.


**F.1** **Application to** **`Muon`**


If these problems can be mitigated, the speed afforded by Algorithm 4 suggests an
improvement in the way `Muon` is applied to transformers. In sum, the idea is to replace one
large matrix with a small aspect ratio by many smaller matrices with large aspect ratios
and apply Algorithm 4 to all of them in parallel. Each multi-head attention layer contains
four square weight matrices _**W**_ _Q,_ _**W**_ _K,_ _**W**_ _V_ and _**W**_ _O ∈_ R _[d][×][d]_ . The orthogonalization step of


34


`Muon` is either applied separately to these four matrices or else to [ _**W**_ _Q |_ _**W**_ _K |_ _**W**_ _V_ ] and _**W**_ _O_,
since typical implementations of multi-head attention store the weights in this concatenated
form. However, we believe it is natural to consider each of these four weight matrices to
be a concatenation of many smaller linear transformations, each corresponding to a single
attention head. If _H_ is the number of heads, each of these smaller matrices has size _d ×_ _[d]_

_H_ [;]
that is, they have aspect ratio _α_ = _H_ . The gradient matrices of [ _**W**_ _Q |_ _**W**_ _K |_ _**W**_ _V_ ] and
_**W**_ _O_ can be reshaped into 3-tensors in which each slice is one of these smaller matrices.
Since typical transformers like GPT-3 can have as many as 96 heads, this variation of `Muon`
has the potential to reduce the runtime.
We use this idea to train a GPT-Small model on FineWeb1B. We compare four
conditions:


1. The baseline approach used in the rest of this paper


2. Splitting up the gradient matrices of [ _**W**_ _Q |_ _**W**_ _K |_ _**W**_ _V_ ] and _**W**_ _O_ by head and applying
Muon to each piece, as described above


3. Using Algorithm 4, restarted after three iterations


4. Splitting by head _and_ using Algorithm 4


We used `Polar Express` with weight decay of 0 _._ 1 for all conditions and swept learning
rates 0 _._ 003 _,_ 0 _._ 005 _,_ 0 _._ 01. Otherwise, all hyperparameters were the same as in Section 5.2.
Our results showed that these changes had a negligible effect in this setting. They did
not affect the optimization quality. Compared to the baseline, splitting by heads actually
reduced the final loss slightly from 3.59 to 3.55; using Algorithm 4 increased the loss very
slightly, from 3.59 to 3.60 when not splitting by head, and from 3.55 to 3.56 when we did
split. However, the runtimes of all 12 runs were nearly identical, showing that at this scale,
the FLOP savings of Algorithm 4 is not beneficial. The embedding size of GPT-Small is
just 768. These techniques may be more impactful when using a larger model. It may also
have more impact outside of deep learning, where `Polar Express` would be run for more
than the 5 iterations used in our experiments. We leave exploration of these settings to
future work.

### **G Code for Constructing Polynomials of Polar Express**


The following code gives a Python implementation of the offline stage of Algorithm 2. This
code was used to construct the coefficients of the polynomials given in (1), which in turn
were used in our `Muon` experiments (Section 5.2). It uses _ℓ_ = 10 _[−]_ [3] and _u_ = 1 by default. It
incorporates Algorithm 3 and the finite precision considerations described in Section 4.4.

```
from math import inf, sqrt
import numpy as np

def optimal_quintic (l, u):
   assert 0 <= l <= u
   if 1 - 5e-6 <= l / u:
     # Above this threshold, the equioscillating polynomials
     # is numerically equal to...
     return (15/8)/u, ( -10/8) /(u**3), (3/8) /(u**5)
   # This initialization becomes exact as l -> u

```

35


```
  q = (3*l + 1) / 4
  r = (l + 3) / 4
  E, old_E = inf, None
   while not old_E or abs (old_E - E) > 1e -15:
     old_E = E
     LHS = np.array ([
        [l, l**3, l**5, 1],
        [q, q**3, q**5, -1],
        [r, r**3, r**5, 1],
        [u, u**3, u**5, -1],
     ])
     a, b, c, E = np.linalg.solve(LHS, np.ones (4))
     q, r = np.sqrt ((-3*b + np.array ([-1, 1]) *
               sqrt (9*b**2 - 20*a*c)) / (10*c))
   return float (a), float (b), float (c)

def optimal_composition (l, num_iters, cushion =0.02407327424182761) :
  u = 1
   coefficients = []
   for _ in range (num_iters):
     a, b, c = optimal_quintic ( max (l, cushion*u), u)
     # Due to cushioning, this may be centered around 1 with
     # respect to 0.024*u, u. Recenter it around 1 with respect
     # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
     pl = a*l + b*l**3 + c*l**5
     pu = a*u + b*u**3 + c*u**5
     rescalar = 2/(pl + pu)
     a *= rescalar; b *= rescalar; c *= rescalar
     # Optionally incorporate safety factor here:
     # a /= 1.01; b /= 1.01**3; c /= 1.01**5
     coefficients.append ((a, b, c))
     l = a*l + b*l**3 + c*l**5
     u = 2 - l
   return coefficients

print (* optimal_composition (1e-3, 10), sep= "\n" )

```

36


