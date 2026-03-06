# Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning

^1 National University of Singapore
^2 Pennsylvania State University
^3 University of California, Los Angeles

## Abstract

While fine-tuning large language models (LLMs) for specific tasks often yields impressive results, it comes at the cost of memory inefficiency due to back-propagation in gradient-based training. Memory-efficient Zeroth-order (MeZO) optimizers, recently proposed to address this issue, only require forward passes during training, making them more memory-friendly. However, compared with exact gradients, ZO-based gradients usually exhibit an estimation error, which can significantly hurt the optimization process, leading to slower convergence and suboptimal solutions. In addition, we find that the estimation error will hurt more when adding to large weights instead of small weights. Based on this observation, this paper introduces Sparse MeZO, a novel memory-efficient zeroth-order optimization approach that applies ZO only to a carefully chosen subset of parameters. We propose a simple yet effective parameter selection scheme that yields significant performance gains with Sparse-MeZO. Additionally, we develop a memory-optimized implementation for sparse masking, ensuring the algorithm requires only inference-level memory consumption, allowing Sparse-MeZO to fine-tune LLaMA-30b on a single A100 GPU. Experimental results illustrate that Sparse-MeZO consistently improves both performance and convergence speed over MeZO without any overhead. For example, it achieves a 9% absolute accuracy improvement and 3.5x speedup over MeZO on the RTE task. Code is available at https://github.com/NUS-HPC-AI-Lab/SparseMeZO.

# Introduction

Fine-tuning large language models for specific tasks or datasets has become a prevalent practice in machine learning. However, a major obstacle in fine-tuning is the substantial memory requirements, which escalate as models increase in size and complexity, thereby limiting the scalability and accessibility for those with limited computational resources.

To mitigate the memory constraints, Parameter Efficient Fine-Tuning (PEFT) has been developed, allowing for the modification of only a subset of parameters and achieving comparable results to full model tuning. However, PEFT methods still necessitate the calculation of gradients for backpropagation and caching of numerous activations during training, which introduces additional memory overhead. Even with PEFT, training still requires approximately 6 times more memory than the memory cost for inference. This discrepancy raises a critical question: Can large language models be fine-tuned solely with the cost of inference?

In response to these challenges, zeroth-order (ZO) optimization presents a promising solution. ZO optimization is a gradient-free method that estimates gradients using only the forward pass of the model, eliminating the need for backpropagation and, consequently, reducing memory usage. MeZO is a recently proposed zeroth-order method for fine-tuning LLMs that has demonstrated impressive performance.

However, compared to exact gradients, ZO-based gradients usually exhibit an estimation error, which can be defined as noise. This noise can significantly hurt the optimization process, leading to slower convergence and suboptimal solutions. Moreover, we find that the estimated ZO gradient is difficult to generalize across batches. Specifically, while it can successfully reduce the training loss on the sampled batch with a high probability, it is more likely to increase the loss on other batches.

To address this challenge, we investigate the impact of gradient noise in zeroth-order optimization for LLM fine-tuning. We measure how the noise affects optimization by evaluating its effect on generalization performance across different data batches. Interestingly, our experiments reveal that the noise has a more significant impact when added to large weights compared to small weights. Based on this finding, we propose a novel sparse memory efficient zeroth-order method (Sparse-MeZO) to selectively optimize small weights, which are more resilient to noise perturbation. By focusing on these noise-resistant weights, we demonstrate that our method enables the use of larger learning rates, leading to improved performance and faster convergence. Our contributions can be summarized as follows:

- We investigate the impact of gradient noise in zeroth-order optimization for LLM fine-tuning. Our evaluations show that the gradient noise can make the estimated ZO gradient difficult to generalize across batches and the noise will hurt more when adding to large weights instead of small weights.

- Based on the above finding, we propose a sparse Memory-Efficient Zeroth-Order optimization method Sparse-MeZO (S-MeZO) for large language model fine-tuning. We also provide theoretical analysis to show the convergence of Sparse-MeZO.

- Different from the efficient implementation with random seed in MeZO, we propose a novel memory-efficient implementation of Sparse-MeZO, which can compute the sparse mask and perturb parameters in the forward pass. The technique enables fine-tuning LLaMA-30b with Sparse-MeZO on a single A100 GPU.

- We conduct empirical studies on LLaMA, OPT, and Mistral. The experimental results demonstrate that Sparse-MeZO can improve the fine-tuning performance and yield a faster convergence rate compared with vanilla MeZO across a wide range of natural language processing tasks. For example, it achieves a 9% absolute accuracy improvement and 3.5x speedup over MeZO on the RTE task.

# Preliminaries

## Parameter-Efficient Fine-Tuning

Parameter-Efficient Fine-Tuning (PEFT) is designed to facilitate efficient adaptation by updating only a subset of the model's parameters, rather than fine-tuning the entire model. These PEFT approaches can be categorized into selective and additive methods.

**Selective Methods.** Selective methods try to selectively fine-tune a portion of a model. For example, fine-tuning only bias terms can rival the results of fine-tuning the entire model, though the effectiveness of this approach diminishes with larger datasets. Beyond static parameter adjustments, dynamically modifying parts of the model has also been explored.

**Additive Methods.** Additive methods involve incorporating new layers into models, with the fine-tuning process focusing solely on these added layers. Traditional techniques such as adapters implemented layer additions in a sequential manner, which led to increased inference latency. LoRA has been proposed to mitigate this issue, which freezes the weights of the pre-trained model and introduces trainable matrices based on rank decomposition into each layer. IA3 introduced methods for adding parameters balancing parameter count with accuracy, while LST introduced a highway structure that learns only small, auxiliary channels.

## Zeroth-Order Optimization

Unlike traditional gradient-based optimization methods that rely on derivatives to guide the search for optimal solutions, Zeroth-Order (ZO) optimization techniques do not require derivatives for optimization. These methods utilize only the value of the objective function, denoted as $f(\bm{x})$, at any chosen point $\bm{x}$. To estimate the gradient in the direction of vector $\bm{z}$, the objective function is assessed at two points in close proximity, $f(\bm{x} + \epsilon \bm{z})$ and $f(\bm{x} - \epsilon \bm{z})$, with $\epsilon$ being a minimal value. Following this, conventional optimization algorithms, such as gradient descent or coordinate descent, are implemented using these approximated gradient values.

### MeZO

ZO-SGD employs SPSA to estimate the gradient. In general, conventional ZO-SGD algorithms utilizing SPSA consume twice the inference memory. MeZO is a memory-efficient variant of ZO-SGD. It circumvents the storage of gradients by saving the random seed and resampling the same random noise $\bm{z}$ with the seed during forward process. More specifically, to calculate $\mathcal{L}(\bm{\theta}+\epsilon \bm{z}) - \mathcal{L}(\bm{\theta}-\epsilon\bm{z})$, MeZO will sample a noise $\bm{z}$ to perturb $\bm{\theta}$ to $\bm{\theta+\epsilon \bm{z}}$ and then calculate $\mathcal{L}(\bm{\theta}+\epsilon \bm{z})$. Then it resamples the same noise $\bm{z}$ with the same seed and move the parameter back $\bm{\theta}-\epsilon \bm{z}$ and calculates the loss. As a result, the zeroth-order gradient estimator can be computed without any memory overhead.

### Sparsity for Zeroth-order Optimization

The lottery ticket hypothesis showed that within a densely connected neural network that is randomly initialized, there exists a subnetwork of sparse yet high-quality connections. Based on this hypothesis, model pruning aims to identify and preserve crucial sparse subnetworks within the larger neural network that can achieve comparable or even superior performance. Dynamic Sparse Training (DST) has been proposed to reduce the training and inference cost in first-order optimization. Several related works have applied sparsity to zeroth-order optimization, such as DeepZero which proposes a ZO training protocol with model pruning guided sparsity.

# Proposed Method

## Empirical Observation on MeZO

For large language models, zeroth-order optimization algorithms like MeZO are often necessary when exact gradients are unavailable or prohibitively expensive to compute. However, compared with exact gradients, these methods inherently introduce noise in the gradient estimates used for optimization. Specifically, the zeroth-order gradient $\bm{g_z(\theta)}$ is approximated as $\bm{g_z(\theta)} = \frac{\mathcal{L}(\bm{\theta} + \epsilon \bm{z}) - \mathcal{L}(\bm{\theta} - \epsilon \bm{z})}{2 \epsilon} \bm{z}$, where $\mathcal{L}$ is the loss function. MeZO exhibits extreme sensitivity to the choice of learning rate. Even a small increase from $1 \times 10^{-6}$ to $2 \times 10^{-6}$ causes divergence and instability, while this larger learning rate is totally fine when fine-tuning with first-order methods. This suggests that the gradient noise introduced by the zeroth-order approximation, defined as $\bm{\delta} = \bm{g(\theta)} - \bm{g_z(\theta)}$ where $\bm{g(\theta)}$ is the exact gradient, significantly hinders the optimization process when large step sizes are used.

To quantify how the gradient noise $\bm{\delta}$ hurts the optimization process, we evaluate its effect on the generalization performance of the estimated gradients. Specifically, we measure whether the zeroth-order gradient estimate computed on one batch can effectively reduce the loss on other held-out batches. For a batch $\mathcal{B}_{t} = \{\mathcal{B}_{t}^{1}, \mathcal{B}_{t}^{2}\}$ with 32 data points, we use 16 samples to estimate the zeroth-order gradient $\bm{g_z}(\bm{\theta}; \mathcal{B}_{t}^{1})$ on batch $\mathcal{B}_{t}^{1}$, and evaluate it on the remaining 16 held-out samples $\mathcal{B}_{t}^{2}$. While the estimated gradient $\bm{g_z}(\bm{\theta}; \mathcal{B}_{t}^{1})$ can reliably reduce the loss on the same batch $\mathcal{B}_{t}^{1}$ it was computed on (90% success rate), it only manages to decrease the loss on the new held-out batch $\mathcal{B}_{t}^{2}$ around 50% of the time. This suggests that the zeroth-order gradient estimates suffer from overfitting or noise that makes them less generalizable to unseen data samples. The gradient noise $\bm{\delta}$, while allowing descent on the current batch, appears to introduce errors that prevent reliable descent directions for unseen batches.

Next, we aim to understand if this effect is uniform across all model parameters or if certain parameter groups are more vulnerable to noise corruption. We notice the nature of vanilla MeZO, where $\frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta - \epsilon z)}{2\epsilon}$ is used to estimate the gradient, and all parameters share the same value of $\frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta - \epsilon z)}{2\epsilon}$. This means not all parameters are optimized in the true gradient direction, which could be a limitation. To analyze this, we divide the parameters into different groups based on their magnitude - the top 20% largest weights are considered "large", while the bottom 20% are "small". Our experiments reveal that the gradient noise $\bm{\delta}$ hurts optimization more when added to large weights compared to small weights. When continuing training from the point where test accuracy drops (due to noise), optimizing only the small weights can recover and further improve test accuracy. The noise $\bm{\delta}$ does not impact all parameters equally - it disproportionately disrupts the optimization of larger weights. Selectively optimizing smaller, noise-resilient weights may be a promising direction to mitigate the effects of gradient noise in zeroth-order optimization.

## Sparse-MeZO

Consider a labelled dataset $\mathcal{D}=\{(\bm{x}_{i}, \bm{y}_{i})\}_{i \in \mathcal{[|D|]}}$ and let $\mathcal{L}(\bm{\theta};\mathcal{B})$ denotes the loss on a mini-batch $\mathcal{B}$. We can define a sparse mask $\bm{m} \in \{0,1\}^{d}$ to selectively sample the random noise $\bm{z} \in \mathbb{R}^{d} \text{ with } \bm{z}\sim \mathcal{N}(\bm{0},\bm{I}_{d})$ on the sub-net of pre-trained model. A sparsified version of random perturbation can be defined as $\bm{\hat{z}} \in \mathbb{R}^{d}$:

$$
\bm{\hat{z}} = \bm{m} \odot \bm{z} .
$$

Based on this sparse perturbation $\bm{\hat{z}}$, we can redefine MeZO algorithm as Sparse-MeZO. The main difference is from the estimated gradient $\bm{g_{\hat{z}}(\bm{\theta})}$, which can be defined as:

$$
\begin{aligned}
    \bm{g_{\hat{z}}(\bm{\theta})} &= \frac{\mathcal{L}(\bm{\theta} + \epsilon \bm{\hat{z}};\mathcal{B}) - \mathcal{L}(\bm{\theta} - \epsilon \bm{\hat{z}};\mathcal{B})}{2\epsilon} \bm{\hat{z}} \\
    &= \frac{\mathcal{L}(\bm{\theta} + \epsilon \bm{m} \odot \bm{z};\mathcal{B}) - \mathcal{L}(\bm{\theta} - \epsilon \bm{m} \odot \bm{z};\mathcal{B})}{2\epsilon} \bm{\hat{z}},
\end{aligned}
$$

where $\epsilon$ represents the perturbation scale. Based on the observations from our motivation, we can create a sparse mask, $\bm{m}$, determined by parameter magnitudes. Specifically, we only update parameters of smaller magnitude. These targeted parameters are defined as $\bm{\hat{\theta}} = \bm{m} \odot \bm{\theta}$. We still preserve the complete set of parameters, but we apply sparse perturbations and gradient estimations only to the selected ones.

- **Constant Mask: Setting the Mask Before Training.** We compare the parameter values to a threshold for each layer to set the mask before training begins. However, a significant downside of this approach is the extra memory required to store a sparse mask, which is as large as the pre-trained model itself.

- **Dynamic Mask: Determining Mask at Each Iteration.** We can establish a threshold for each layer before training and then generate the mask by comparing parameter values to this threshold during each iteration. This method avoids the necessity of storing a large mask, $\bm{m}$.

```pseudocode
Algorithm: Sparse-MeZO (S-MeZO)

Require: theta represents pre-trained LLM weight, N is the number of layers in model,
         learning rate eta_t, s represents sparsification interval.
Initialize random seed s
Determine threshold h = {h_1, ..., h_N} of each layer with the sparsification interval

For t = 1 to T:
    Sample Minibatch B from X and random seed s.
    m <- GetMask(theta_t, h)
    theta_t <- PerturbParameters(theta_t, epsilon, s, m)
    l+ = L(theta_t; B)
    theta_t <- PerturbParameters(theta_t, -2*epsilon, s, m)
    l- = L(theta_t; B)
    theta_t <- PerturbParameters(theta, epsilon, s, m)
    proj_grad <- (l+ - l-) / (2*epsilon)
    Reset random seed s
    For theta_i in theta:
        z_i ~ N(0,1)
        theta_i <- theta_i - eta_t * proj_grad * m_i * z
    End For
End For
```

We employ a dynamic mask to choose which parameters to perturb and update, addressing the issue of memory constraints. We determine thresholds using a percentile-based method where the threshold is set based on a target sparsity level.

The algorithm outlines that we first establish the threshold $h_{i}$ for each layer before beginning training. We then use GetMask to compare each parameter against its threshold $h_{i}$ and create the mask $\bm{m}$. Following this, we introduce the function PerturbParameters to generate a Gaussian noise sample $\bm{z} \sim \mathcal{N}(\bm{0},\bm{I_{d}})$ and apply the mask $\bm{m}$ to produce a sparse perturbation $\bm{\hat{z}} = \bm{m} \odot \bm{z}$. With $\bm{\hat{z}}$, we perturb the current parameters $\bm{\theta_{t}}$ to get new parameters $\bm{\theta_{t} + \epsilon \hat{z}}$ and $\bm{\theta_{t} - \epsilon \hat{z}}$. This allows us to compute two distinct loss values: $l^{+} = \mathcal{L}(\bm{\theta_{t}} + \epsilon\bm{\hat{z}})$ and $l^{-} = \mathcal{L}(\bm{\theta_{t}} - \epsilon\bm{\hat{z}})$. From these losses, we calculate the estimated sparse gradient $\bm{g_{m}}(\bm{\theta_{t}}) = \text{proj\_grad} * \bm{\hat{z}}$, where $\text{proj\_grad} = \frac{l^{+} - l^{-}}{2\epsilon}$. Finally, this gradient can be used with a learning rate $\eta_{t}$ to update $\bm{\theta_{t}}$.

```pseudocode
Algorithm: PerturbParameters

Input: theta represents pre-trained LLM weight, perturbation scale epsilon,
       random seed s, mask m.
Reset random seed s
For theta_i in theta:
    z_i ~ N(0,1)
    theta_i <- theta_i + m_i * epsilon * z_i
End For
```

```pseudocode
Algorithm: GetMask

Input: theta represents pre-trained LLM weight, threshold h (h_i represents
       threshold of each layer).
Output: Mask m

For i = Layer 1 to Layer N:
    For theta_{i,j} in theta_i:
        If theta_{i,j} <= h_i:
            theta_{i,j} = 1
        Else:
            theta_{i,j} = 0
        End If
    End For
End For
```

## Memory-Efficient Implementation of Sparse-MeZO

Our approach involves perturbing the parameters $\bm{\theta_{t}}$ twice to generate two distinct sets of parameters, $\bm{\theta_{t}'} = \bm{\theta_{t}} + \epsilon \bm{z}$ and $\bm{\theta_{t}''} = \bm{\theta_{t}} - \epsilon \bm{z}$. We then use the estimated gradient to update the original parameters $\bm{\theta_{t}}$. This step typically requires storing two separate sets of parameters, leading to increased memory usage during fine-tuning.

MeZO conserves memory by saving random seeds $s$ and using it to resample $z$ for calculating $\bm{\theta_{t}'}$, $\bm{\theta_{t}''}$, and reconstructing $\bm{\theta_{t}}$ without needing extra memory. However, applying a sparse mask $\bm{m}$ for calculating sparse perturbation $\bm{\hat{z}}$ in MeZO poses a memory issue. We cannot simply reconstruct $\bm{\hat{z}}$ by saving the random seed because the sparse mask, determined by parameter magnitudes, changes when parameters are altered by the perturbation. To address this, we propose two solutions.

**1-bit Quantization:** We can apply 1-bit quantization to store the mask $\bm{m}$, as it consists solely of 0s and 1s. However, this method still increases memory usage.

**Calculating the Mask During the Forward Pass:** By computing the mask and perturbing parameters in the forward pass, we eliminate the need to store perturbed parameters $\bm{\theta_{t}'}$ and $\bm{\theta_{t}''}$. This means we only have to keep the original parameters $\bm{\theta_{t}}$ throughout training. For vanilla implementation, we first need to calculate the perturbed parameters with mask $\bm{m}$: $\bm{\theta_{t}'} = \bm{\theta_{t}} + \epsilon \bm{m} \odot \bm{z}$. After that, we can use perturbed parameters $\bm{\theta_{t}'}$ to calculate the loss value $l^{+}$ with the forward process. For example, the output of layer $i$ can be defined as $\bm{y^{(i)}} = \bm{\theta_{t}'^{(i)}} \bm{x^{(i)}} + \bm{b^{(i)}}$. The vanilla implementation requires saving both the vanilla parameters $\bm{\theta_{t}}$ and mask $\bm{m}$. However, for our proposed efficient implementation, we only need to save vanilla parameters $\bm{\theta_{t}}$. We can calculate the mask $\bm{m^{(i)}}$ of layer $i$ during the forward process and then obtain the output of this layer: $\bm{y^{(i)}} = (\bm{\theta_{t}^{(i)}} + \epsilon m(\bm{\theta_{t}}) \bm{z^{(i)}}) \bm{x^{(i)}} + \bm{b^{(i)}}$, where $m(\cdot)$ represents GetMask to calculate mask $\bm{m^{(i)}}$. Then, we can release the memory of mask $\bm{m}^{(i)}$ and calculate the output and mask of the next layer.

# Experiments

Following a similar setting to MeZO, we evaluate the performance of our proposed method on SuperGLUE.

## Experimental Setting

**Datasets.** We conduct experiments on various fine-tuning tasks including SST-2, RTE, BoolQ, WIC, MultiRC, and the multi-class task COPA.

**Models.** We primarily use pre-trained LLaMA-7b to evaluate the performance of our proposed method on downstream tasks. We also test with Mistral-7B-v0.1 and OPT-13b. Additionally, to examine scalability, we evaluate on LLaMA-30b.

**Baselines.** We compare our method to vanilla MeZO, R-MeZO (MeZO with a random mask), zero-shot learning, in-context learning, conventional full-parameter fine-tuning (FT) with Adam, and LoRA.

## Performance on SuperGLUE

On LLaMA-7b with 1,000 training examples, S-MeZO outperforms all other zeroth-order techniques. S-MeZO boosts MeZO's accuracy from 71.7% to 80.7% on RTE (up 9%) and from 75.9% to 80.9% on BoolQ (up 5%). The average accuracy across six SuperGLUE tasks improves from 76.6% (MeZO) to 80.3% (S-MeZO). All zeroth-order methods surpass zero-shot and in-context learning, and S-MeZO significantly narrows the gap to first-order fine-tuning (82.9%).

On Mistral-7B-v0.1, S-MeZO consistently improves over vanilla MeZO. For example, S-MeZO improves accuracy from 81.6 to 85.3 on BoolQ, achieving comparable performance with full fine-tuning.

On LLaMA2-7b, S-MeZO outperforms an expanded set of baselines including ZO-SGD-Cons, ZO-SGD-Sign, ZO-SGD-Adam, ZO-AdaMU, and AdaZeta. S-MeZO improves MeZO's accuracy from 78.8% to 82.2% on BoolQ (up 3.4%) and from 70.2% to 77.6% on RTE (up 7.4%), achieving an average of 80.0% compared to MeZO's 76.3%.

On OPT-13b, Sparse MeZO consistently improves MeZO's performance across BoolQ (72.1 to 73.8), RTE (75.5 to 77.6), and WIC (62.2 to 63.7).

## Performance on Commonsense Reasoning and Mathematics Tasks

On more challenging tasks using Mistral-7B, Sparse MeZO consistently outperforms MeZO. Substantial improvements are observed on the AQuA mathematics task (+2.6%, from 24.0 to 26.6) and the SIQA commonsense reasoning task (+1.7%, from 68.5 to 70.2). On PIQA, S-MeZO improves from 84.3 to 85.3, and on BoolQ from 76.6 to 79.2.

## Convergence Rate

S-MeZO converges faster than MeZO. S-MeZO only needs about 5,000 steps to achieve 70% accuracy on RTE but vanilla MeZO needs 17,500 steps. S-MeZO achieves about 3.5x speedup on RTE and 3x speedup on BoolQ.

## Memory Usage

S-MeZO with Efficient Implementation (S-MeZO-EI) does not require more memory than MeZO, both using 14.6 GB for LLaMA-7b fine-tuning across all tasks. This offers roughly 12 times less GPU memory compared to full-parameter fine-tuning (128.2 GB average). The vanilla S-MeZO implementation without efficient masking requires 28.3 GB, but the efficient implementation (calculating the mask during the forward pass) reduces this to the same 14.6 GB as MeZO.

## Sparse Rate

Experiments with various sparsity values (from 0.0 to 0.8) show that significant performance gains are obtained with sparsity values from 0.5 to 0.8. For most tasks, a sparsity value of 0.8 yields the best performance. For example, on RTE, S-MeZO improves accuracy from 71.7% to 82.3% at sparsity 0.8. On BoolQ, it achieves a 6.6% gain (from 75.9% to 82.5%). On WIC, improvements range from 2.0% to 2.9% across sparsity levels.

## Scalability

S-MeZO scales to larger language models. On LLaMA-30b, MeZO achieves 76.9% on RTE (up from 71.7% on LLaMA-7b), and S-MeZO further improves this to 82.1%. On BoolQ, S-MeZO reaches 85.7% on LLaMA-30b. On WIC, S-MeZO achieves 67.3% on LLaMA-30b compared to 64.9% on LLaMA-7b.

# Convergence Analysis of Sparse-MeZO

In this section, we explain why Sparse-MeZO can accelerate the convergence. We can define a sub-network in pre-trained large language models, which is determined by the sparse mask $\bm{m}$. The main idea of our proof is that if we follow the update rule in Sparse-MeZO, the gradient norm on the sub-network can be smaller than $\sigma^{2}$ after $\mathcal{O}(\frac{\hat{d}L}{\sigma^{2}})$ steps, where $\hat{d}$ is the number of parameters in the sub-network. Therefore, ZO can use fewer steps to converge when we only focus on a sub-network.

Firstly, we assume the loss function $\mathcal{L}(\bm{\theta};\bm{x})$ is Lipschitz Continuous:

**Assumption 1** (Lipschitz Continuous).

$$
\| \nabla \mathcal{L}(\bm{\theta};\bm{x}) - \nabla \mathcal{L}(\bm{\theta'}, \bm{x}) \| \leq \frac{L(l)}{2} \|\bm{\theta} - \bm{\theta'} \|^{2},
$$

where $\nabla \mathcal{L}(\bm{\theta};\bm{x})$ denotes the true first-order gradient of $\bm{\theta}$ on $\bm{x}$ and $L(l)$ represents the Lipschitz constant of $\mathcal{L}(\cdot)$. Given $\mathcal{L}_{\hat{z}}(\bm{\theta}) = \mathbb{E}_{\bm{\hat{z}}} [ \mathcal{L}(\bm{\theta} + \epsilon \bm{\hat{z}})]$ and the above Assumption 1, we can obtain the relationship between sparse gradient $\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta})$ and the expectation of estimated sparse ZO gradient $\bm{g_{\hat{z}}(\theta)}$:

**Lemma 1.** ZO gradient $\bm{g_{\hat{z}}(\bm{\theta})}$ is unbiased estimation of $\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta})$:

$$
\begin{aligned}
\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta}) &= \bm{m} \odot \nabla_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta}) \\
&= \bm{m} \odot \nabla_{\bm{\theta}} \mathbb{E}_{\bm{\hat{z}}}[\mathcal{L}(\bm{\theta} + \epsilon \bm{\hat{z}})] \\
&= \mathbb{E}_{\bm{\hat{z}}} [\frac{\mathcal{L}(\bm{\theta} + \epsilon \bm{\hat{z}}) - \mathcal{L}(\bm{\theta} - \epsilon \bm{\hat{z}})}{2\epsilon}\bm{\hat{z}}] \\
&= \mathbb{E}_{\bm{\hat{z}}}[\bm{g_{\hat{z}}(\theta)} ],
\end{aligned}
$$

where $\bm{g_{\hat{z}}(\theta)} = \frac{\mathcal{L}(\bm{\theta} + \epsilon \bm{\hat{z}}) - \mathcal{L}(\bm{\theta} - \epsilon \bm{\hat{z}})}{2\epsilon}\bm{\hat{z}}$. We can find that $\bm{g_{\hat{z}}(\theta)}$ is unbiased estimation of $\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta})$. Then, based on Lemma 1, we can use the distance $\|\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta}) - \nabla_{\bm{\theta}} \mathcal{L}_{m}(\bm{\theta})\|$ to analyze the relationship between the true sparse gradient $\nabla_{\bm{\theta}} \mathcal{L}_{m}(\bm{\theta}) = \bm{m} \odot \nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta})$ and sparse gradient $\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta})$:

**Lemma 2.** Let $\mathcal{L}$ be Lipschitz Continuous, we have:

$$
\| \nabla_{\bm{\theta}} \mathcal{L}_{m}(\bm{\theta}) \|^{2} \leq 2\|\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\bm{\theta})\|^{2} + \frac{\epsilon^{2}L^{2}(l)}{2}(\hat{d} + 4)^{3}.
$$

where $\nabla_{\bm{\theta}} \mathcal{L}_{m}(\bm{\theta}) = \bm{m} \odot \nabla_{\bm{\theta}} \mathcal{L}(\bm{\theta})$, $\hat{d} = \sum_{i=1}^{i=d} m_{i}$ is the number of selected parameters in mask $\bm{m}$, $L(l)$ represents the Lipschitz constant. Finally, we can obtain the convergence rate of Sparse-MeZO.

**Theorem 1.** Assuming a sequence of generated parameters $\{ \bm{\theta_{t}} \}_{t \geq 0}$ in Sparse-MeZO. We can have:

$$
\mathbb{E}_{\hat{z},x} [\| \nabla_{\bm{\theta}} \mathcal{L}_{m}(\bm{\theta_{T}})\|^{2}] \leq \sigma^{2}
$$

for any $T=\mathcal{O}(\frac{\hat{d}L}{\sigma^{2}})$

where $L(l) \leq L$ for all $\mathcal{L}(\bm{\theta_{t}})$. This theorem illustrates that the presence of pronounced sparsity patterns, along with the smoothness of the objective function, can significantly enhance the rate of convergence, potentially achieving a linear acceleration.

# The Proof of Lemma 1

Let $\mathcal{L}_{z}(\theta)$ be the expectation of $\mathcal{L}(\theta + \epsilon m \odot z)$:

$$
\begin{aligned}
\mathcal{L}_{\hat{z}}(\theta) :&= \mathbb{E}_{z}[\mathcal{L}(\theta + \epsilon m \odot z)] \\
&= \mathbb{E}_{\hat{z}}[\mathcal{L}(\theta + \epsilon \hat{z})]
\end{aligned}
$$

We can obtain the Lemma:

$$
\begin{aligned}
\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta) &= m \odot \nabla_{\theta}\mathcal{L}_{\hat{z}}(\theta) \\
&= m \odot \mathbb{E}_{z} [\nabla_{\theta} \mathcal{L}(\theta + \epsilon m \odot z)] \\
&= \mathbb{E}_{z}[\frac{\mathcal{L}(\theta + \epsilon m \odot z) - \mathcal{L}(\theta - \epsilon m \odot z)}{2\epsilon} m \odot z] \\
&= \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta - \epsilon \hat{z})}{2\epsilon}\hat{z}]
\end{aligned}
$$

Proof:

$$
\begin{aligned}
\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta) &= \widehat{\nabla}_{\theta} \mathbb{E}_{\hat{z}}[\mathcal{L}(\theta + \epsilon \hat{z})] \\
&= \widehat{\nabla}_{\theta} \int_{\hat{z}} \text{pdf}_{\hat{z}}(z) \mathcal{L}(\theta + \epsilon z) dz \\
&= m \odot \nabla_{\theta} \int_{\hat{z}} \text{pdf}_{\hat{z}}(z) \mathcal{L}(\theta + \epsilon z) dz \\
&= m \odot \int_{\hat{z}} \nabla_{\theta} \text{pdf}_{\hat{z}}(z) \mathcal{L}(\theta + \epsilon z) dz \\
&= \frac{1}{k} m \odot \int_{\hat{z}} \nabla_{\theta} e^{-\frac{1}{2}\|z\|^{2}}\mathcal{L}(\theta + \epsilon z) dz \\
&= \frac{1}{k} m \odot \int_{\hat{y}}\nabla_{\theta} e^{-\frac{1}{2}\|\frac{y-\theta}{\epsilon}\|^{2}}\mathcal{L}(y)\frac{1}{\epsilon^{n}} dy \\
&= \frac{1}{k} m \odot \int_{\hat{y}}\frac{y-\theta}{\epsilon^{2}}e^{-\frac{1}{2\epsilon^{2}}\|y-\theta\|^{2}} \mathcal{L}(y)\frac{1}{\epsilon^{n}} dy \\
&= \frac{1}{k} m \odot \int_{\hat{z}}\frac{z}{\epsilon} e^{-\frac{1}{2}\|z\|^{2}} \mathcal{L}(\theta + \epsilon z) dz \\
&= m \odot \int_{\hat{z}} \text{pdf}_{\hat{z}}(z) \mathcal{L}(\theta + \epsilon z) \frac{z}{\epsilon}dz \\
&= \mathbb{E}_{\hat{z}}[m \odot \frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}] \\
&= \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}]
\end{aligned}
$$

where we can define $y = \theta + \epsilon z$, $\hat{y} = \theta + \epsilon m \odot z$, $k = \sqrt{(2\pi)^{\hat{d}}}$ and $\hat{d}$ is the number of 1 in $\bm{m}$.

Therefore, we can obtain the gradient $\nabla_{\theta} \mathcal{L}_{m}(\theta)$ is equal to $\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}]$.

In addition, we prove $\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}]$ is also equal to $\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - L(\theta)}{\epsilon} \hat{z}]$:

$$
\begin{aligned}
&\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)}{\epsilon} \hat{z}] \\
&= \frac{1}{k} \int_{\hat{z}} \frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta)}{\epsilon} z e^{-\frac{1}{2}\|z\|^{2}} dz \\
&= \frac{1}{k} \int_{\hat{\epsilon}}\frac{\mathcal{L}(\theta + \epsilon z)}{\epsilon} z e^{-\frac{1}{2}\|z\|^{2}} dz - \frac{\mathcal{L}(\theta)}{\epsilon k} \int_{\hat{z}} z e^{-\frac{1}{2} \|z\|^{2}} dz \\
&= \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}]
\end{aligned}
$$

After that, we can get the relationship between $\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta) - \mathcal{L}(\theta - \epsilon \hat{z})}{\epsilon} \hat{z}]$ and $\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}]$:

$$
\begin{aligned}
\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta) - \mathcal{L}(\theta - \epsilon \hat{z})}{\epsilon} \hat{z}] &= \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon (-\hat{z})) - \mathcal{L}(\theta)}{\epsilon} (-\hat{z})] \\
&= \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z} - \mathcal{L}(\theta))}{\epsilon} \hat{z}] \\
&= \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}].
\end{aligned}
$$

Based on the equations above, we can obtain:

$$
\begin{aligned}
&\mathbb{E}_{\hat{z}} [\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta - \epsilon \hat{z})}{2\epsilon}\hat{z}] \\
&= \frac{1}{2} (\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z} - \frac{\mathcal{L}(\theta)}{\epsilon}\hat{z} + \frac{\mathcal{L}(\theta)}{\epsilon} \hat{z} - \frac{\mathcal{L}(\theta - \epsilon \hat{z})}{\epsilon} \hat{z}]) \\
&= \frac{1}{2} (\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)}{\epsilon} \hat{z}] + \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta) - \mathcal{L}(\theta - \epsilon \hat{z})}{\epsilon} \hat{z}]) \\
&= \frac{1}{2} (\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}] + \mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}]) \\
&=\mathbb{E}_{\hat{z}}[\frac{\mathcal{L}(\theta + \epsilon \hat{z})}{\epsilon} \hat{z}] \\
&= \widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta)
\end{aligned}
$$

Finally, we can obtain the relationship between $\mathbb{E}_{\hat{z}} [\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta - \epsilon \hat{z})}{2\epsilon}\hat{z}]$ and $\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta)$ and finish the proof.

# The Proof of Lemma 2

$$
\| \nabla_{\bm{\theta}} \mathcal{L}_{m}(\bm{\theta}) \|^{2} \leq 2\|\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta})\|^{2} + \frac{\epsilon^{2}L^{2}(l)}{2}(\hat{d} + 4)^{3}.
$$

Proof:

We can first define the distance between $\widehat{\nabla}_{\bm{\theta}} \mathcal{L}_{\hat{z}}(\bm{\theta}) = \mathbb{E}_{\hat{z}}[\bm{g_{\hat{z}}(\theta)}]$ and sparse FO gradient $\nabla \mathcal{L}_{m}(\theta)$ as:

$$
\begin{aligned}
&\|\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta) - \nabla_{\theta} \mathcal{L}_{m}(\theta) \| \\
&= \|\frac{1}{k} \int_{z} (\frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta - \epsilon z)}{2 \epsilon} - \langle \nabla_{\theta} \mathcal{L}_{m}(\theta), z \rangle) z e^{-\frac{1}{2}\|z\|^{2}} d\hat{z} \| \\
&= \|\frac{1}{k} \int_{z} (\frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta)}{\epsilon} - \langle m \odot \nabla_{\theta} \mathcal{L}(\theta), z \rangle) z e^{-\frac{1}{2}\|z\|^{2}} d\hat{z} \| \\
&\leq \frac{1}{k \epsilon} \int_{z}|\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta) - \epsilon \langle \nabla_{\theta} \mathcal{L}(\theta), \epsilon \rangle | \|m \odot z\| e^{-\frac{1}{2} \|z\|^{2}} d\hat{z} \\
&\leq \frac{\epsilon L(l)}{2k} \int_{\epsilon} \|z\|^{2} \|m \odot z\| e^{-\frac{1}{2}\|z\|^{2}} d\hat{z} \\
&= \frac{\epsilon L(l)}{2} \mathbb{E}_{\hat{z}}[\|\hat{z}\|^{3}] \\
&\leq \frac{\epsilon L(l)}{2} (\hat{d}+3)^{\frac{3}{2}}
\end{aligned}
$$

where $\hat{d}$ is the number of selected parameters with mask $\bm{m}$. The last inequality holds because $\mathbb{E}_{\hat{z}}[\|\hat{z}\|^{p}] \leq (\hat{d} + p)^{p/2}$ for $p \geq 2$. In addition, $\| \bm{a} + \bm{b}\|^{2} \leq 2 \|\bm{a}\|^{2} + 2 \|\bm{b}\|^{2}$, we can define $\bm{a} = \bm{a} - \bm{b}$ and obtain that $\|\bm{a}\|^{2} \leq 2 \|\bm{a} - \bm{b}\|^{2} + 2 \|\bm{b}\|^{2}$. Let $\bm{a} = \nabla_{\theta} \mathcal{L}_{m}(\theta)$ and $\bm{b} = \widehat{\nabla}_{\theta}\mathcal{L}_{\hat{z}}(\theta)$, we can obtain:

$$
\begin{aligned}
\|\nabla_{\theta} \mathcal{L}_{m}(\theta) \|^{2} &\leq 2\|\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta) - \nabla_{\theta} \mathcal{L}_{m}(\theta)\|^{2} + 2 \|\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta)\|^{2} \\
&\leq \frac{\epsilon^{2}L^{2}(l)}{2}(\hat{d} + 3)^{3} + 2 \|\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta)\|^{2} \\
&\leq \frac{\epsilon^{2}L^{2}(l)}{2} (\hat{d} + 4)^{3} + 2\|\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta)\|^{2}
\end{aligned}
$$

# The Proof of Theorem 1

Proof:

$$
\begin{aligned}
\mathcal{L}_{\hat{z}}(\theta) - \mathcal{L}(\theta) &= \mathbb{E}_{\hat{z}} [\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)] \\
&= \mathbb{E}_{\hat{z}}[\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta) - \epsilon \langle \nabla \mathcal{L}(\theta) , \hat{z} \rangle] \\
&= \frac{1}{k} \int_{\hat{z}} [\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta) - \epsilon \langle \nabla \mathcal{L}(\theta), z \rangle] e^{-\frac{1}{2}\| z \|^{2}} dz \\
&\leq \frac{1}{k} \int_{\hat{z}} \frac{\epsilon^{2}L(l)}{2} \| z \|^{2} e^{-\frac{1}{2}\| z \|^{2}} dz \\
&= \frac{\epsilon^{2}L(l)}{2}\mathbb{E}_{\hat{z}}[\|\hat{z}\|^{2}] \\
&\leq \frac{\epsilon^{2}L(l)}{2}\hat{d}
\end{aligned}
$$

The first inequality holds because Lipschitz Continuous: $|\mathcal{L}(\theta') - \mathcal{L}(\theta) - \langle \nabla \mathcal{L}(\theta), \theta' - \theta \rangle | \leq \frac{L(l)}{2}\|\theta' - \theta \|^{2}$, where $\theta' = \theta + \epsilon z$. The second inequality holds because $\mathbb{E}_{\hat{z}}[\|\hat{z}\|^{2}] = \hat{d}$, where $\hat{d}$ is the number of $1$ in mask $\bm{m}$.

$$
\begin{aligned}
&[(\mathcal{L}_{\hat{z}}(\theta) - \mathcal{L}(\theta)) - (\mathcal{L}_{\hat{z}}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta + \epsilon \hat{z}))]^{2} \\
&\leq 2[\mathcal{L}_{\hat{z}}(\theta) - \mathcal{L}(\theta)]^{2} + 2[\mathcal{L}_{\hat{z}}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta + \epsilon \hat{z})]^{2} \\
&\leq \frac{\epsilon^{4}L^{2}(l)}{2}\hat{d}^{2} + \frac{\epsilon^{4}L^{2}(l)}{2}\hat{d}^{2} \\
&= \epsilon^{4}L^{2}(l)\hat{d}^{2}
\end{aligned}
$$

The first inequality is due to $\|\bm{a} + \bm{b}\|^{2} \leq 2\|\bm{a}\|^{2} + 2\|\bm{b}\|^{2}$, where $\bm{a} = \mathcal{L}_{\hat{z}}(\theta) - \mathcal{L}(\theta), \bm{b} = \mathcal{L}_{\hat{z}}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta + \epsilon \hat{z})$. The second inequality is due to the equation above.

$$
\begin{aligned}
[\mathcal{L}_{\hat{z}}(\theta + \epsilon \hat{z}) - \mathcal{L}_{\hat{z}}(\theta)]^{2} &\leq 2[\mathcal{L}_{\hat{z}}(\theta + \epsilon \hat{z}) - \mathcal{L}_{\hat{z}}(\theta) - \epsilon \langle \widehat{\nabla}_{\theta}\mathcal{L}_{\hat{z}}(\theta), \hat{z} \rangle]^{2} + 2[\epsilon \langle \widehat{\nabla}_{\theta}\mathcal{L}_{\hat{z}}(\theta), \hat{z} \rangle]^{2} \\
&\leq \frac{\epsilon^{4}L^{2}(l)}{2} \|\hat{z}\|^{4} + 2 \epsilon^{2} \langle \widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta), \hat{z} \rangle^{2} \\
&\leq \frac{\epsilon^{4}L^{2}(l)}{2} \|\hat{z}\|^{4} + 2 \epsilon^{2} \| \widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta)\|^{2} \|\hat{z}\|^{2}
\end{aligned}
$$

The first inequality is due to $\|\bm{a} + \bm{b}\|^{2} \leq 2\|\bm{a}\|^{2} + 2\|\bm{b}\|^{2}$. The second inequality holds because Lipschitz Continuous: $|\mathcal{L}(\theta') - \mathcal{L}(\theta) - \langle \nabla \mathcal{L}(\theta), \theta' - \theta \rangle | \leq \frac{L(l)}{2}\|\theta' - \theta \|^{2}$, where $\theta' = \theta + \epsilon \hat{z}$.

$$
\begin{aligned}
&[\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)]^{2} \\
&\leq 2[(\mathcal{L}_{\hat{z}}(\theta) - \mathcal{L}(\theta)) - (\mathcal{L}_{\hat{z}}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta + \epsilon \hat{z}))]^{2} + 2[\mathcal{L}_{\hat{z}}(\theta + \epsilon \hat{z}) - \mathcal{L}_{\hat{z}}(\theta)]^{2} \\
&\leq 2\epsilon^{4} L^{2}(l)\hat{d}^{2} + \epsilon^{4}L^{2}(l)\|\hat{z}\|^{4} + 4\epsilon^{2}\|\widehat{\nabla}_{\theta} \mathcal{L}_{\hat{z}}(\theta) \|^{2}\|\hat{z}\|^{2}
\end{aligned}
$$

The first inequality is due to $\|\bm{a} + \bm{b}\|^{2} \leq 2\|\bm{a}\|^{2} + 2\|\bm{b}\|^{2}$. The last inequality holds because of the previous equations.

$$
\begin{aligned}
\mathbb{E}_{z, x} [\|g_{\hat{z}}(\theta)\|^{2}] &= \mathbb{E}_{\hat{z}}[\| \frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta - \epsilon \hat{z})}{2 \epsilon}\hat{z}\|^{2}] \\
&= \mathbb{E}_{\hat{z}}[\|\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)}{2 \epsilon} \hat{z} + \frac{\mathcal{L}(\theta) - \mathcal{L}(\theta - \epsilon \hat{z})}{2 \epsilon} \hat{z} \|^{2}] \\
&\leq \mathbb{E}_{\hat{z}}[2\|\frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)}{2 \epsilon} \hat{z} \|^{2} + 2 \| \frac{\mathcal{L}(\theta) - \mathcal{L}(\theta - \epsilon \hat{z})}{2 \epsilon} \hat{z} \|^{2}] \\
&= \mathbb{E}_{\hat{z}}[\frac{1}{2 \epsilon^{2}}[\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)]^{2} \cdot \|\hat{z}\|^{2} + \frac{1}{2 \epsilon^{2}} [\mathcal{L}(\theta) - \mathcal{L}(\theta - \epsilon \hat{z})]^{2} \cdot \|\hat{z}\|^{2}] \\
&\leq \mathbb{E}_{\hat{z}}[2\epsilon^{2}L^{2}(l)\hat{d}^{2}\|\hat{z}\|^{2} + \epsilon^{2}L^{2}(l)\|\hat{z}\|^{6}+4\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta)\|^{2}\|\hat{z}\|^{4}] \\
&\leq 2\epsilon^{2}L^{2}(l)\hat{d}^{3} + \epsilon^{2}L^{2}(l)(\hat{d}+6)^{3} + 4(\hat{d} + 4)^{2}\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta)\|^{2} \\
&\leq 3\epsilon^{2}L^{2}(l)(\hat{d}+4)^{3} + 4(\hat{d} + 4)^{2} \|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta)\|^{2}
\end{aligned}
$$

The first inequality holds because $\|\bm{a} + \bm{b}\|^{2} \leq 2\|\bm{a}\|^{2} + 2\|\bm{b}\|^{2}$, where $\bm{a} = \frac{\mathcal{L}(\theta + \epsilon \hat{z}) - \mathcal{L}(\theta)}{2 \epsilon} \hat{z}$, $\bm{b} = \frac{\mathcal{L}(\theta) - \mathcal{L}(\theta - \epsilon \hat{z})}{2 \epsilon} \hat{z}$. The second inequality is due to the previous equation. The third inequality holds because $\mathbb{E}_{\hat{z}}[\|\hat{z}\|^{2}] = \hat{d}$, $\mathbb{E}_{\hat{z}}[\|\hat{z}\|^{p}] \leq (\hat{d} + p)^{\frac{p}{2}}$ for $p \geq 2$. The last inequality holds because $2\hat{d}^{3} + (\hat{d} + 6)^{3} \leq 3(\hat{d} + 4)^{3}$.

Based on the assumption about Lipschitz Continuous, we can obtain: $|\mathcal{L}(\theta_{t+1}) - \mathcal{L}(\theta_{t}) - \langle \nabla \mathcal{L}(\theta_{t}), \theta_{t+1} - \theta_{t} \rangle | \leq \frac{L(l)}{2}\|\theta_{t+1} - \theta_{t} \|^{2}$.

Then, we can obtain:

$$
\begin{aligned}
\mathcal{L}_{\hat{z}}(\theta_{t+1}) - \mathcal{L}_{\hat{z}}(\theta_{t}) - \langle \widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t}), \theta_{t+1} - \theta_{t} \rangle
&\leq |\mathcal{L}_{\hat{z}}(\theta_{t+1}) - \mathcal{L}_{\hat{z}}(\theta_{t}) - \langle \widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t}), \theta_{t+1} - \theta_{t} \rangle | \\
&\leq \frac{L(l)}{2}\|\theta_{t+1} - \theta_{t} \|^{2}
\end{aligned}
$$

Based on the equation, we can follow the update rule: $\theta_{t+1} = \theta_{t} - \eta_{t} g_{\hat{z}}(\theta_{t})$ and we can find:

$$
\begin{aligned}
\mathcal{L}_{\hat{z}}(\theta_{t+1}) &\leq \mathcal{L}_{\hat{z}}(\theta_{t}) + \langle \widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t}), \theta_{t+1} - \theta_{t} \rangle + \frac{L(l)}{2} \|\theta_{t} - \theta_{t+1} \|^{2} \\
&= \mathcal{L}_{\hat{z}}(\theta_{t}) - \eta_{t} \langle \widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t}), g_{\hat{z}}(\theta_{t}) \rangle + \frac{(\eta_{t})^{2}L(l)}{2} \|g_{\hat{z}}(\theta_{t})\|^{2} \\
\end{aligned}
$$

where $\eta_{t}$ represents the learning rate at step $t$. Then, we can take the expectation for $\hat{z}$ and input $x$:

$$
\begin{aligned}
&\mathbb{E}_{\hat{z}, x}[\mathcal{L}_{\hat{z}}(\theta_{t+1})] \\
&\leq \mathbb{E}_{\hat{z}, x}[\mathcal{L}_{\hat{z}}(\theta_{t})] - \eta_{t} \mathbb{E}_{\hat{z}, x}[\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t})\|^{2}] + \frac{(\eta_{t})^{2}L(l)}{2} \mathbb{E}_{\hat{z}, x}[\|g_{z}(\theta_{t})\|^{2}] \\
&\leq \mathbb{E}_{\hat{z}, x}[\mathcal{L}_{\hat{z}}(\theta_{t})] - \eta_{t} \mathbb{E}_{\hat{z}, x}[\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t})\|^{2}] + \frac{(\eta_{t})^{2}L(l)}{2} (4(\hat{d}_{t} + 4)^{2} \mathbb{E}_{\hat{z},x}[\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t})\|^{2}] + 3\epsilon^{2}L^{2}(l)(\hat{d}_{t} + 4)^{3})
\end{aligned}
$$

The first inequality is due to the Lemma 1 result and the update rule equation. The second inequality holds because the bound on $\mathbb{E}_{\hat{z}, x}[\|g_{z}(\theta_{t})\|^{2}]$ derived above.

Then, we can select learning rate $\eta_{t} = \frac{1}{4(\hat{d}_{t} + 4)L(l)}$ and obtain:

$$
\mathbb{E}_{\hat{z}, x}[\mathcal{L}_{\hat{z}}(\theta_{t+1})] \leq \mathbb{E}_{\hat{z}, x}[\mathcal{L}_{\hat{z}}(\theta_{t})] - \frac{1}{8(\hat{d}_{t} + 4)L(l)}\mathbb{E}_{\hat{z},x}[\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{t})\|^{2}] + \frac{3\epsilon^{2}}{32}L(l)(\hat{d}_{t}+4)
$$

Then, taking the sum over the index from $T+1$ to $0$, we can have that:

$$
\mathbb{E}_{\hat{z}, x}[\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{T})\|^{2}] \leq 8(\hat{d}+4)L[\frac{\mathcal{L}_{\hat{z}}(\theta_{0}) - \mathcal{L}_{\hat{z}}^{*}}{T+1} + \frac{3\epsilon^{2}}{32}L(\hat{d}+4)]
$$

where $L(l) \leq L$ for all $\mathcal{L}(\bm{\theta_{t}})$. Thus, based on Lemma 2, we can have:

$$
\begin{aligned}
\mathbb{E}_{\hat{z},x}[\|\nabla \mathcal{L}_{m}(\theta_{T})\|^{2}] &\leq \frac{\epsilon^{2}L^{2}}{2}(\hat{d}+4)^{3} + 2\mathbb{E}_{\hat{z}, x}[\|\widehat{\nabla} \mathcal{L}_{\hat{z}}(\theta_{T})\|^{2}] \\
&\leq 16(\hat{d}+4)L\frac{\mathcal{L}_{\hat{z}}(\theta_{0}) - \mathcal{L}_{\hat{z}}^{*}}{T+1} + \frac{\epsilon^{2}L^{2}}{2}(\hat{d}+4)^{2}(\hat{d}+\frac{11}{2})
\end{aligned}
$$

The second inequality is due to the equation above. To obtain $\sigma$-accurate solution: $\mathbb{E}_{\hat{z},x}[\|\nabla \mathcal{L}_{m} (\theta_{T})\|^{2}] \leq \sigma^{2}$, we can define $\epsilon = \Omega(\frac{\sigma}{\hat{d}^{\frac{3}{2}}L})$.

$$
\begin{aligned}
16(\hat{d}+4)L\frac{\mathcal{L}_{\hat{z}}(\theta_{0}) - \mathcal{L}_{\hat{z}}^{*}}{T+1} + \mathcal{O}(\epsilon^{2}L^{2}\hat{d}^{3}) &= 16(\hat{d}+4)L\frac{\mathcal{L}_{\hat{z}}(\theta_{0} - \mathcal{L}_{\hat{z}}^{*})}{T+1} + \mathcal{O}(\sigma^{2}) \\
T &= \mathcal{O}(\frac{\hat{d}L}{\sigma^{2}})
\end{aligned}
$$

Finally, we can finish the proof of the theorem. This theorem illustrates that the presence of pronounced sparsity patterns, along with the smoothness of the objective function, can significantly enhance the rate of convergence, potentially achieving a linear acceleration.

# Conclusion

In this paper, we propose a novel memory-efficient zeroth-order fine-tuning method Sparse-MeZO, which can use a similar memory cost to the inference process. We evaluate the performance of fine-tuning LLaMA and OPT with Sparse-MeZO on SuperGLUE benchmark and the experimental results illustrate that Sparse-MeZO can achieve a higher accuracy and faster convergence. Finally, we can fine-tune LLaMA-30b on a single A100 GPU.

**Limitation**: There is still a performance gap between our proposed method Sparse-MeZO and first-order fine-tuning methods. We plan to address these limitations and enhance Sparse-MeZO's capabilities in future research and conduct more experiments on state-of-the-art pre-trained language models.
