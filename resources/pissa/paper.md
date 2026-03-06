# PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models

Institute for Artificial Intelligence, Peking University
State Key Laboratory of General Artificial Intelligence, Peking University
[https://github.com/GraphPKU/PiSSA](https://github.com/GraphPKU/PiSSA)

---

## Abstract

To parameter-efficiently fine-tune (PEFT) large language models (LLMs), the low-rank adaptation (LoRA) method approximates the model changes $\Delta W \in \mathbb{R}^{m \times n}$ through the product of two matrices $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$, where $r \ll \min(m, n)$, $A$ is initialized with Gaussian noise, and $B$ with zeros. LoRA **freezes the original model $W$** and **updates the "Noise & Zero" adapter**, which may lead to slow convergence. To overcome this limitation, we introduce **P**r**i**ncipal **S**ingular values and **S**ingular vectors **A**daptation (PiSSA). PiSSA shares the same architecture as LoRA, but initializes the adaptor matrices $A$ and $B$ with the principal components of the original matrix $W$, and put the remaining components into a residual matrix $W^{res} \in \mathbb{R}^{m \times n}$ which is frozen during fine-tuning.
Compared to LoRA, PiSSA **updates the principal components** while **freezing the "residual" parts**, allowing faster convergence and enhanced performance. Comparative experiments of PiSSA and LoRA across 11 different models, ranging from 184M to 70B, encompassing 5 NLG and 8 NLU tasks, reveal that PiSSA consistently outperforms LoRA under identical experimental setups. On the GSM8K benchmark, Gemma-7B fine-tuned with PiSSA achieves an accuracy of 77.7%, surpassing LoRA's 74.53% by 3.25%. Due to the same architecture, PiSSA is also compatible with quantization to further reduce the memory requirement of fine-tuning. Compared to QLoRA, QPiSSA (PiSSA with 4-bit quantization) exhibits smaller quantization errors in the initial stages.
Fine-tuning LLaMA-3-70B on GSM8K, QPiSSA attains an accuracy of 86.05%, exceeding the performance of QLoRA at 81.73%. Leveraging a fast SVD technique, PiSSA can be initialized in only a few seconds, presenting a negligible cost for transitioning from LoRA to PiSSA.

---

# Introduction

Fine-tuning large language models (LLMs) is a highly effective technique for boosting their capabilities in various tasks, ensuring models to follow instructions, and instilling models with desirable behaviors while eliminating undesirable ones. However, the fine-tuning process for very large models is accompanied by prohibitive costs. For example, regular 16-bit fine-tuning of a LLaMA 65B parameter model requires over 780 GB of GPU memory, and the VRAM consumption for training GPT-3 175B reaches 1.2TB.
Consequently, various parameter-efficient fine-tuning (PEFT) methods have been proposed to reduce the number of parameters and memory usage required for fine-tuning. Due to the ability to maintain the performance of full fine-tuning without adding additional inference latency, Low-Rank Adaptation (LoRA) has emerged as a popular PEFT method.

LoRA hypothesizes that the modifications to parameter matrices during fine-tuning exhibit low-rank properties. For a pre-trained weight matrix $W \in \mathbb{R}^{m \times n}$, LoRA substitutes the updates with a low-rank decomposition $\Delta W = AB$, where $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$, and the rank $r \ll \text{min}(m, n)$. For $Y = XW$, the modified forward pass is as follows:

$$
Y = X(W + \Delta W) = X(W+AB),
$$

A random Gaussian initialization is used for $A$ and zero for $B$, making $AB=0$ at the beginning of training, thereby the injection of adapters does not affect the model's output initially. LoRA avoids the need to compute gradients or maintain the optimizer states for the original matrix $W$, instead optimizing the injected, significantly smaller low-rank matrices $A,B$. Thus, it could reduce the number of trainable parameters by 10,000x and the GPU memory requirement by 3x. LoRA is capable of achieving comparable performance to full parameter fine-tuning.
By integrating the quantization of pre-trained matrices $W$, LoRA also enables reducing the average memory requirements by 16x. Meanwhile, the adapters can still utilize higher precision weights, thus, the quantization usually does not significantly degrade the performance of LoRA.

According to the LoRA equation, the gradients of A and B are $\frac{\partial L}{\partial A} = X^\top \left( \frac{\partial L}{\partial Y} \right) B^\top$ and $\frac{\partial L}{\partial B} = A^\top X^\top \left( \frac{\partial L}{\partial Y} \right)$. Compared to full fine-tuning, using LoRA initially does not change the output $Y$ for the same input $X$, so the magnitude and direction of gradient are primarily determined by the values of $A$ and $B$. Since $A$ and $B$ are initialized with Gaussian noise and zeros in LoRA, the gradients could be small and uninformative for a long time, leading to slow convergence in the fine-tuning process. We also observe this phenomenon empirically, as LoRA often wastes much time around the initial point.

Our **P**r**i**ncipal **S**ingular values and **S**ingular vectors **A**dapter (PiSSA) diverges from LoRA and its successors by focusing not on approximating $\Delta W$, but $W$.
We apply singular value decomposition (SVD) to matrix $W$. Based on the magnitude of the singular values, we partition $W$ into two parts: the principal low-rank matrix $W^{pri}$, comprising a few largest singular values, and the residual matrix $W^{res}$, which possesses the remaining smaller singular values (with a larger quantity, representing a possible long-tail distribution). The principal matrix $W^{pri}$ can be represented by the product of $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$, where $r \ll \min(m, n)$. $A$ and $B$ are initialized based on the principal singular values and singular vectors and are trainable. Conversely, $W^{res}$ is initialized with the product of the residual singular values and singular vectors and remains frozen during fine-tuning.
Since the principal singular vectors represent the directions in which the matrix $W$ has the most significant stretching or impact, by directly tuning these principal components, PiSSA is able to **fit the training data faster and better**.
Moreover, the loss and gradient norm curves of PiSSA often demonstrate a similar trend to those of full parameter fine-tuning in our experiments, indicating that fine-tuning the principal components matches the behavior of fine-tuning the full matrix to some degree.

Because the principal components $W^{pri}$ are preserved in the adapter at full precision, an additional benefit of PiSSA is that when applying quantization to the frozen part $W^{res}$, we can significantly **reduce the quantization error** compared to QLoRA (which quantizes the entire $W$). Therefore, PiSSA is even more compatible with quantization than LoRA, making it a superior plug-and-play substitution for LoRA.

Our paper makes several significant contributions:

- We analyze the initial gradient magnitude and direction in LoRA, demonstrating that $A$ initially has a zero gradient and $B$ has a random gradient, which slows down convergence and may lead to convergence at suboptimal local minima.
- We propose PiSSA initialization, a novel method that approximates the optimization direction of full-parameter fine-tuning by adapting a model's principal components. To our knowledge, PiSSA is the first to apply SVD to the original model, using principal singular values and vectors to initialize the adapter for fine-tuning, while keeping the residual components frozen. Experiments show that PiSSA converges faster and outperforms LoRA.
- We combine PiSSA with NF4 quantization to propose QPiSSA, which reduces quantization error by about 20% compared to QLoRA, while maintaining the fast convergence and high performance of PiSSA.

# Related Works

The vast complexity and computational needs of large language models (LLMs) with billions of parameters present significant hurdles in adapting them for specific downstream tasks. Parameter Efficient Fine-Tuning (PEFT) emerges as a compelling solution by minimizing the fine-tuning parameters and memory requirements while achieving comparable performance to full fine-tuning. PEFT encompasses strategies like partial fine-tuning, soft prompt fine-tuning, non-linear adapter fine-tuning, and low rank adapter based fine-tuning.

LoRA injects trainable adapters to the linear layers. After fine-tuning, these adaptations can be re-parameterized into the standard model structure, thus gaining widespread adoption due to their ability to maintain the model's original architecture while enabling efficient fine-tuning.
Following LoRA, AdaLoRA dynamically learns the rank size needed for LoRA in each layer of the model. DeltaLoRA updates the original weights of the model using parameters from adapter layers, enhancing LoRA's representational capacity. LoSparse incorporates LoRA to prevent pruning from eliminating too many expressive neurons. DoRA introduces a magnitude component to learn the scale of $\Delta W$ while utilizing the original AB as a direction component of $\Delta W$. Unlike LoRA and its successors, which focus on learning low-rank approximations of weight updates, our PiSSA directly tunes the essential low-rank parts of the model while keeping the noisier, high-rank, and nonessential parts frozen. Although our approach differs in philosophy from LoRA, it shares most of LoRA's structural benefits and can be extended by these methods to enhance its performance.

QLoRA integrates LoRA with 4-bit NormalFloat (NF4) quantization, along with Double Quantization and Paged Optimizers, enabling the fine-tuning of a 65B parameter model on a single 48GB GPU while preserving the performance of full 16-bit fine-tuning tasks. QA-LoRA introduces group-wise operators to increase the degree of freedom in low-bit quantization. LoftQ reduces quantization error by decomposing the quantization error matrix of QLoRA and retaining the principal components with an adapter. PiSSA can also be combined with quantization techniques, and we have found that PiSSA significantly reduces quantization error compared to QLoRA and LoftQ.

# PiSSA: Principal Singular Values and Singular Vectors Adaptation

This section formally presents our **P**r**i**ncipal **S**ingular values and **S**ingular vectors **A**daptation method. PiSSA computes the singular value decomposition (SVD) of matrices $W$ within the self-attention and multilayer perceptron (MLP) layers. The (economy size) SVD of a matrix $W \in \mathbb{R}^{m \times n}$ is given by $W = USV^\top$, where $U\in \mathbb{R}^{m \times \text{min}(m, n)}, V\in \mathbb{R}^{n \times \text{min}(m, n)}$ are the singular vectors with orthonormal columns, and $V^\top$ is the transpose of $V$.
$S =\text{diag}(\mathbf{s}) \in \mathbb{R}^{\text{min}(m, n) \times \text{min}(m, n)}$, where the operation $\text{diag}(\mathbf{s})$ transforms $\mathbf{s}$ to a diagonal matrix $S$, and $\mathbf{s}\in \mathbb{R}^{\text{min}(m, n)}_{\geq 0}$ represents the singular values arranged in descending order. When the top $r$ singular values $\mathbf{s}_{[:r]}$ are significantly larger than the remaining singular values $\mathbf{s}_{[r:]}$, we denote the intrinsic rank of $W$ as $r$. Consequently, $S$, along with $U$ and $V$, can be divided into two groups: the principal singular values and vectors -- $\{U_{[:,:r]}, S_{[:r,:r]}, V_{[:,:r]}\}$, and the residual singular values and vectors -- $\{U_{[:,r:]}, S_{[r:,r:]}, V_{[:,r:]}\}$, where the matrix slicing notations are the same as those in PyTorch and $[:r]$ denotes the first $r$ dimensions. The principal singular values and vectors are utilized to initialize the injected adapter consisting of $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$:

$$
A = U_{[:,:r]}\, S_{[:r,:r]}^{1/2} \in \mathbb{R}^{m \times r},
$$

$$
B = S_{[:r,:r]}^{1/2}\, V_{[:,:r]}^\top \in \mathbb{R}^{r \times n}.
$$

The residual singular values and vectors are used to build the residual matrix which is frozen during fine-tuning:

$$
W^{res} = U_{[:,r:]}\, S_{[r:,r:]}\, V_{[:,r:]}^\top \in \mathbb{R}^{m \times n}.
$$

As indicated by the following equation, the integration of $AB$ with the residual matrix also preserves the full capability of the pre-trained model in the beginning of fine-tuning:

$$
Y = XW = X(W^{res}+W^{pri}) = X(W^{res}+AB).
$$

Similar to LoRA, the gradients of $A$ and $B$ are also given by $\frac{\partial L}{\partial A} = X^\top \left( \frac{\partial L}{\partial Y} \right) B^\top$ and $\frac{\partial L}{\partial B} = A^\top X^\top \left( \frac{\partial L}{\partial Y} \right)$. Since elements of $\mathbf{s}_{[:r]}$ $\gg$ elements of $\mathbf{s}_{[r:]}$, the trainable adapter $W^{pri}=AB$ contains the most essential directions of $W$.
In the ideal case, training $AB$ mirrors the process of fine-tuning the entire model despite using fewer parameters. The ability to directly fine-tune the most essential part of a model enables PiSSA to converge faster and better.
In contrast, LoRA initializes the adapters $A$ and $B$ with Gaussian noise and zeros while keeping $W$ frozen. Consequently, the gradients are small or in random directions during the early stages of fine-tuning, possibly introducing much waste of gradient descent steps. Moreover, an inferior initialization might lead to suboptimal local minimum, resulting in worse generalization performance.

Since PiSSA shares the identical architecture with LoRA, it inherits most of LoRA's benefits.
These include but are not limited to the capability of fine-tuning a model with a reduced number of trainable parameters, quantizing the residual model to decrease memory consumption during forward propagation in training, and easy deployment.
The adapter's straightforward linear structure facilitates the integration of trainable matrices with the pre-trained weights upon deployment, thereby maintaining the original inference speed of a fully fine-tuned model.
Employing the Fast SVD technique allowed PiSSA to finish initialization in several seconds (see Appendix: Fast Singular Value Decomposition), which is a negligible cost.

For storage efficiency, we can choose not to store the dense parameter matrix $\Delta W$, but to store the low-rank matrices, $\Delta A$ and $\Delta B$ instead.
As shown in the Appendix (Equivalently Converting PiSSA into LoRA), leveraging solely the $\Delta A$ and $\Delta B$ facilitates their seamless integration with the original pre-trained models.
Finally, one pre-trained model can accommodate multiple $\Delta A,\Delta B$, fine-tuned by diverse PiSSA or LoRA procedures, which enables fast adaptation of the pre-trained model to different downstream applications.

# QPiSSA: An Extension Method with Lower Quantization Error

Quantization divides the value range of a matrix into several continuous regions, and maps all values falling inside a region into the same "quantized" value. It is an effective technique to reduce the memory consumption of forward propagation. At the same time, LoRA greatly reduces the backward memory requirement, making it highly suitable to use LoRA and quantization together, where the base model is quantized for memory-efficient forward propagation, and the LoRA adaptors are kept in full precision for accurate backward parameter updates.
One representative previous work, QLoRA, quantizes the base model to Normal Float 4-bit (NF4) and initializes the full-precision $A$ and $B$ with Gaussian-Zero initialization. Therefore, the overall error is given by:

$$
\text{Quantization Error of QLoRA} = ||W - \left(nf4(W) + AB\right)||_* = ||W - nf4(W)||_*,
$$

where $||M||_*$ denotes the nuclear norm (also known as the trace norm), defined as:

$$
\|M\|_{*}=\operatorname{trace}\left(\sqrt{M^{*}M}\right)=\sum_{i=1}^{\min\{m,n\}}\sigma_{i}(M),
$$

where $\sigma_{i}(M)$ is the $i^{\text{th}}$ singular value of $M$.
As we can see, the quantization error of QLoRA is the same as that of directly quantizing the base model.
Our QPiSSA, however, **does not quantize the base model but the residual model**. Therefore, its error is given by:

$$
\text{Quantization Error of QPiSSA} = ||W - \left(nf4(W^{res})+AB\right)||_* = ||W^{res} - nf4(W^{res})||_*.
$$

Since the residual model has removed the large-singular-value components, $W^{res}$ has a **narrower distribution** than that of $W$, which is **beneficial for reducing the quantization error**.
Moreover, given that NF4 is optimized for data with a normal distribution, $W^{res}$ aligns more closely with a Gaussian distribution and exhibits a smaller standard deviation, making it more suitable for applying NF4 than $W$. Both the above lead QPiSSA to achieve a significantly lower quantization error than QLoRA.

Besides the advantage of reducing quantization error, QPiSSA's gradient direction is similar to that of PiSSA, resulting in significantly better fine-tuning performance compared to QLoRA.

# Experiments

## Evaluating the Performance of PiSSA on both NLG and NLU Tasks

We compared PiSSA, LoRA, and full-parameter fine-tuning on natural language generation (NLG) tasks. We fine-tuned LLaMA 2-7B, Mistral-7B-v0.1, and Gemma-7B on the MetaMathQA dataset to assess their mathematical problem-solving capabilities on the GSM8K and MATH validation sets. Additionally, the models were fine-tuned on the CodeFeedback dataset and evaluated for coding proficiency using the HumanEval and MBPP datasets. Furthermore, the models were trained on the WizardLM-Evol-Instruct dataset and tested for conversational abilities on the MT-Bench dataset. All experiments were conducted using subsets containing 100K data points and were trained for only one epoch.

Across all models and tasks, fine-tuning with PiSSA consistently surpasses the performance of fine-tuning with LoRA. For example, on LLaMA 2-7B PiSSA achieved 53.22% on GSM8K versus LoRA's 42.85%; on Mistral-7B PiSSA reached 73.31% versus LoRA's 69.50%; and on Gemma-7B PiSSA scored 77.78% versus LoRA's 75.11%. This improvement is robust across various amounts of training data and epochs, including both 4-bit and full precision, different model sizes and types, and varying proportions of trainable parameters.

We also evaluated PiSSA's natural language understanding (NLU) capability on the GLUE benchmark with DeBERTa-v3-base. PiSSA outperforms LoRA in 7 out of 8 NLU tasks, achieving an overall average improvement of 1.21% (89.83 vs. 88.62 for LoRA with Kaiming initialization). Upon reviewing the training loss on the exceptional dataset, MNLI, we observed that PiSSA's average loss of $0.17$ was lower than LoRA's $0.24$ in the final epoch, indicating that the fitting ability of PiSSA remains stronger than that of LoRA.

## Experiments using Full Data and More Epochs

We finetuned LLaMA 2-7B on the complete MetaMathQA-395K dataset for 3 epochs to ensure thorough saturation.

According to the loss curves, the loss of PiSSA reduces rapidly during the first 100 steps, and the grad norm of PiSSA is significantly higher than that of LoRA, with a trend similar to full fine-tuning. Throughout the process, the loss of PiSSA remains lower than that of LoRA, indicating that PiSSA converges to a better local optimum.
PiSSA consistently achieves higher accuracy compared to LoRA, and in most cases also surpasses full parameters fine-tuning. We hypothesize that this is because PiSSA is a denoised version of full fine-tuning.
Comparing the grad norm and loss curves of PiSSA and full fine-tuning, we can see that the larger grad norm of full fine-tuning does not bring lower loss, indicating that a portion of the grad norm is spent on noisy directions not beneficial for loss reduction.

## Conducting 4-bit Quantization Experiments

We compared the initial quantization error reduction ratio of PiSSA, QLoRA, and LoftQ. This ratio is defined as $(1-\frac{||W-(nf4(W^{'})+AB)||_*}{||W-nf4(W)||_*})\times 100\%$, measuring the relative error decrease achieved by each method compared to directly quantizing the base model.

PiSSA reduces the quantization error by about 20% compared to directly quantizing the base model. The reduction is more significant for lower-rank matrices. For instance, in LLaMA-3-70B, all "Key" projection layers see a reduction of 49%. QLoRA does not reduce quantization error, while PiSSA significantly outperforms LoftQ in reducing quantization error.

We trained LLaMA 3-8B using LoRA/QLoRA, PiSSA/QPiSSA, LoftQ, and full fine-tuning on MetaMathQA-395K for 3 epochs. QPiSSA's loss reduction speed in the first 100 steps is even faster than PiSSA and full fine-tuning. Although LoftQ can reduce the quantization error, its loss convergence speed is not faster than LoRA and QLoRA, indicating that QPiSSA's ability to reduce the quantization error and its fast convergence might also be orthogonal capabilities. After sufficient training, QPiSSA's loss is also much lower than that of LoRA/QLoRA and LoftQ. In terms of fine-tuning performance, QPiSSA's accuracy is higher than that of QLoRA and LoftQ and even better than that of full-precision LoRA.

## Experiments Across Various Sizes and Types of Models

We compared (Q)PiSSA and (Q)LoRA across 9 models, ranging from 7-70B parameters, including LLaMA 2-7/13B, LLaMA-3-8/70B, Mistral-7B, Gemma-7B, Qwen1.5-7B, Yi-1.5-34B, and MoE models DeepSeek-MoE-16B and Mixtral-8x7B. (Q)PiSSA, compared to (Q)LoRA, shows improved accuracy across various sizes and types of models, demonstrating its consistent advantage.

## Experiments on Various Ranks

We explored the impact of incrementally increasing the rank of PiSSA/QPiSSA and LoRA/QLoRA from 1 to 128. The training was conducted using the MetaMathQA-100K dataset for 1 epoch, with validation on GSM8K and MATH.

QLoRA shows no reduction in quantization error, while QPiSSA consistently outperforms LoftQ in reducing quantization error across all ranks, with a particularly notable advantage at lower ranks. PiSSA and QPiSSA achieve a better fit to the training data compared to LoRA, QLoRA, and LoftQ. PiSSA consistently outperforms LoRA with the same amount of trainable parameters. As the rank increases, PiSSA reaches and surpasses the performance of full-parameter fine-tuning.

# Conclusion

This paper presents a PEFT technique that applies singular value decomposition (SVD) to the weight matrix of pre-trained models. The principal components obtained from the SVD are used to initialize a low-rank adapter named PiSSA, while the residual components are kept frozen, to achieve effective fine-tuning and parameter efficiency simultaneously.
Through extensive experiments, we found that PiSSA and its 4-bit quantization version QPiSSA significantly outperform LoRA and QLoRA in both NLG and NLU tasks,
across different training steps, various model sizes and types, and under various amount of trainable parameters.
As PiSSA shares the same architecture as LoRA, it can be seamlessly used in existing LoRA pipelines as an efficient alternative initialization method.

# Limitation

There are still some questions with PiSSA not addressed in this paper:

1. Besides language models, can PiSSA also be adapted to convolutional layers and enhance the performance of vision tasks?
2. Can PiSSA also benefit from some improvements to LoRA, such as AdaLoRA and DyLoRA which adaptively adjust the rank?
3. Can we provide more theoretical explanations for the advantages of PiSSA over LoRA?

We are actively exploring these questions.

---

# Appendix: Enhancing PiSSA with LoRA Improvement Methods

AdaLoRA introduces three improvements over LoRA:

- Trainable parameters in AdaLoRA are changed to $A, B$, and $E$. $A$ and $B$ are Gaussian-initialized, and $E$ is a zero-initialized $r$-dimensional vector, making $A \cdot \text{diag}(E) \cdot B = \Delta W$, similar to singular value decomposition.
- A regularization loss $|AA^T-I|+|B^TB-I|$ is used to make $A$ and $B$ gradually orthogonal during training, resembling the SVD of $\Delta W$.
- An initial large rank is set, and less important E values are gradually masked during training, resulting in different final ranks for each layer, achieving better performance with the same number of parameters.

Despite the extensive use of SVD terms, AdaLoRA **does not perform actual SVD on any matrix**. In the PEFT domain, terms like low-rank decomposition, and singular value decomposition often appear. They generally refer to products of low-dimensional matrices approximating an ideal $\Delta W$ without actual matrix decomposition. To our knowledge, PiSSA is the first to perform SVD on the original model, fine-tuning the principal component while keeping the residual model frozen.

PiSSA and AdaLoRA represent different improvements to LoRA, making them combinable. We additionally improved PiSSA based on AdaLoRA's innovations:

- After extracting the principal singular values and vectors of $W$, we use $S$ as an independent trainable vector instead of multiplying it into $U$ and $V$.
- Since PiSSA's $U$ and $V$ are orthogonal at the beginning, maintaining their orthogonality through orthogonal regularization is very easy.

DoRA adds a learnable magnitude module to LoRA, normalizing $W + AB$ at each update step and multiplying it by the magnitude module. This allows $A, B$ to learn the direction and the magnitude module to learn the magnitude of $\Delta W$.

PiSSA, with its intrinsic principal singular values and orthogonal singular vectors, is very suitable for combination with AdaLoRA. The performance of the improved PiSSA surpasses all other methods including PiSSA alone. PiSSA combined with DoRA significantly surpasses DoRA alone and also exceeds PiSSA alone, demonstrating the potential of PiSSA when integrated with other methods.

---

# Appendix: Fast Singular Value Decomposition

To speed up the decomposition of the pre-trained matrix $W$, we adopted the algorithm proposed by Halko et al. (denoted as Fast SVD), which introduces randomness to achieve an approximate matrix decomposition. We compared the initialization time, error, and training loss between SVD and Fast SVD.

The computation time of SVD is tens of times that of Fast SVD. SVD exhibits consistently high time consumption with minimal variation as the rank increases, while Fast SVD remains significantly lower throughout. As the rank increases, the initialization error initially rises gradually, with a slight decrease observed when the rank reaches 128. At the same rank, increasing the niter in Fast SVD leads to a gradual reduction in error. For training loss, as the rank increases, the training loss decreases gradually. At the same rank, with the increase of niter, the training loss of models initialized based on Fast SVD approaches that of models initialized based on SVD.

---

# Appendix: Equivalently Converting PiSSA into LoRA

The advantage of PiSSA lies in its ability to significantly enhance training outcomes during the fine-tuning phase. After training, it allows for the direct sharing of the trained matrices $A$ and $B$.
However, if we directly save $A,B$, users need to perform singular value decomposition on the original model to get $W^{res}$, which requires additional time. More importantly, such a way necessitates altering the parameters of the original model, which can be inconvenient when using multiple adapters. Therefore, we recommend converting the trained PiSSA module equivalently into a LoRA module, thereby eliminating the need to modify the original model's parameters during sharing and usage.
In the initialization phase, PiSSA decomposes the original matrix into principal components and a residual matrix: $W = W^{res} + A B$.
Upon completion of training, the model adjusts the weights as follows: $W + \Delta W = W^{res} + A' B'$.
Thus, the modification of the model weights by PiSSA is given by:

$$
\Delta W = A' B' - A B = \underbrace{[A'~ A]}_{\Delta A} \underbrace{\begin{bmatrix} B' \\ -B \end{bmatrix}}_{\Delta B}
$$

where $\Delta A \in \mathbb{R}^{m\times 2r}$ and $\Delta B\in \mathbb{R}^{2r\times n}$.
Therefore, we can store and share the new adaptor $\Delta A$ and $\Delta B$ instead of $A',B'$, which allows directly inserting the adaptor to the original matrix and avoids breaking $W$. Since $r$ is typically small, the twice storage overhead is still acceptable. This modification allows for plug-and-play usage without the need for singular value decomposition, saving time and avoiding computational errors associated with the SVD, without necessitating changes to the original model parameters.

---

# Appendix: Reducing Quantization Error through Multiple Iteration of SVD

When number of iterations $T>1$, LoftQ uses an $N$-bit quantized weight $Q \in \mathbb{R}_{N}^{m \times n}$ and low-rank approximations $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{n \times r}$ to minimize the following objective by alternating between quantization and singular value decomposition:

$$
\underset{Q, A, B}{\min} \| W - (Q + AB^{\top}) \|_{F},
$$

where $\| \cdot \|_{F}$ denotes the Frobenius norm, $A$ and $B$ are set to zero.

Inspired by LoftQ, our QPiSSA $T$-iter alternately minimizes the following objective:

$$
\underset{W_{res}, A, B}{\min} \| W - (nf4(W_{res}) + AB^{\top}) \|_{F},
$$

where $A$ and $B$ are initialized by the principal singular values and singular vectors.
The process is summarized in the following algorithm:

```pseudocode
Algorithm: QPiSSA-T-iters, T >= 2
Input: Pre-trained weight W, target rank r, 4-bit quantization function nf4(.), alternating step T
1. Initialize A_0, B_0 <- SVD(W)  (using principal singular values/vectors)
2. Initialize residual weight W_res <- W - A_0 * B_0^T
3. For t = 2 to T:
4.     Update A_t, B_t <- SVD(W - nf4(W_res))  (using principal singular values/vectors)
5.     Update residual weight W_res <- W - A_{t-1} * B_{t-1}^T
6. End For
Output: nf4(W_res), A_T, B_T
```

Multiple iterations can significantly reduce quantization error. For instance, using QPiSSA-r64 with 5-iter on LLaMA-3-8B reduces the quantization error nearly twice as much as with 1-iter. PiSSA consistently outperforms LoftQ in reducing quantization error across all ranks and iteration counts.

---

# Appendix: Conductive Experiments on Various SVD Components

To investigate the influence of singular values and vectors of varying magnitudes on fine-tuning performance, we initialized adapters with principal, medium, and minor singular values and vectors. Models were fine-tuned on the MetaMathQA dataset and evaluated against GSM8K and MATH.

The results highlight that initializing adapters with principal singular values and vectors consistently leads to reduced training loss and enhanced accuracy on both the GSM8K and MATH validation datasets across all three models. This underscores the efficacy of fine-tuning the model parameters based on the principal singular values.

---

# Appendix: The Residual Matrices having a Narrower Distribution

We applied PiSSA initialization to LLaMA-2-7B, Mistral-7B, Gemma-7B, and LLaMA-3-8B, and fit the values in every linear layer with Gaussian distribution and calculated their mu and sigma.

The results show that the residual models' means are closer to 0, and the standard deviations are smaller after PiSSA initialization. Thus, $W^{res}$ indeed has a narrower distribution than W in a statistical sense. The difference is not as large as that in the first layer after averaging all layers, which we suspect is because middle layers in a model tend to have more even eigenvalue distributions due to redundancy and insufficient training.

---

# Appendix: Comparing the Quantization Error of QLoRA, LoftQ and QPiSSA

This section provides a comprehensive comparison of the quantization errors associated with QLoRA, LoftQ, and QPiSSA.

The quantization error of QLoRA, which quantizes the base model to NF4 and initializes $A$ and $B$ with Gaussian-Zero initialization, is:

$$
\text{Quantization Error of QLoRA} = ||W - \left(nf4(W) + AB\right)||_* = ||W - nf4(W)||_*,
$$

QLoRA decomposes the original matrix into the sum of a quantized matrix and an error matrix. The magnitude of the error matrix is much smaller than that of the original matrix. Therefore, the benefit of preserving the principal components of the $W$ matrix with the adapter is greater than that of preserving the principal components of the error matrix with the adapter.

LoftQ, designed to preserve the principal components of the error matrix using the adapter, first performs singular value decomposition on the quantization error matrix of QLoRA:

$$
U^{err}S^{err}V^{err} = W - nf4(W),
$$

then uses the larger singular values to initialize $A$ and $B$, thereby reducing the quantization error to:

$$
LoftQ^{err} = ||W - \left(nf4(W) + AB\right)||_*= ||U^{err}_{[r:]}S^{err}_{[r:,r:]}V^{err}_{[r:]}||_*=\sum_{i=r}^{\min(m,n)}{S^{err}_{[i,i]}}.
$$

LoftQ eliminates only the largest $r$ singular values $S^{\text{err}}_{[:r]}$ from the QLoRA error matrix.

Our PiSSA, however, **does not quantize the base model but the residual model**:

$$
\text{Quantization Error of PiSSA} = ||W - \left(nf4(W^{res})+AB\right)||_* = ||W^{res} - nf4(W^{res})||_*,
$$

where $A$ and $B$ are initialized following the PiSSA initialization equations.
Since the residual model has removed the large-singular-value components, the value distribution of $W^{res}$ can be better fitted by a Student's t-distribution with higher degrees of freedom compared to $W$ and thus quantizing $W^{res}$ results in lower error using 4-bit NormalFloat.

---

# Appendix: Combining QPiSSA with Various Quantization Methods

In addition to NF4 quantization, QPiSSA can also be combined with GPTQ and INT8 quantization. PiSSA effectively reduces quantization error for several reasons:

- It reduces outlier values;
- It makes the value distribution more Gaussian-like;
- It preserves larger values in full precision, thereby narrowing the weight distribution in the quantized portion.

QPiSSA combined with INT8 reduces quantization error by 18.16% on LLaMA-3-8B and significantly outperforms QLoRA using INT8. The perplexity of LLaMA-3-8B increases to 20.79 after quantization with GPTQ-4bit, but when PiSSA is applied, the perplexity is reduced to 6.23. Overall, QPiSSA demonstrates a clear advantage over QLoRA when combined with various quantization methods, retaining the fast convergence and superior performance characteristics of PiSSA while minimizing quantization error.

---

# Appendix: Evaluating PiSSA on Mistral and Gemma with More Training Steps

We applied PiSSA, LoRA, and full parameter fine-tuning on the full MetaMathQA-395K dataset using Mistral-7B and Gemma-7B models, training for 3 epochs.

The loss for full parameter fine-tuning decreases sharply with each epoch, indicating overfitting to the training data. During the entire first epoch, the loss for full parameter fine-tuning on Mistral and Gemma is significantly higher than for LoRA and PiSSA, suggesting that full parameter fine-tuning has weaker generalization capabilities compared to LoRA and PiSSA on these models. The gradient norm for the first epoch fluctuates dramatically, further indicating instability. These experiments demonstrate that using parameter-efficient fine-tuning can prevent the over-fitting issue caused by over-parameters.

---

# Appendix: Supplementary Experiments on Various Ranks

PiSSA uses fewer trainable parameters compared to LoRA while achieving or even surpassing full-parameter fine-tuning on LLaMA-2-7B and Mistral-7B. On Gemma-7B, PiSSA exceeds full-parameter fine-tuning performance even at rank=1. However, as the rank increases to 128, the performance of PiSSA begins to decline, indicating that PiSSA over-parameterizes earlier than LoRA. This over-parameterization phenomenon does not occur on LLaMA-2-7B, suggesting that increasing the rank further might enable PiSSA to achieve even higher performance.

PiSSA consistently shows a faster initial loss reduction compared to LoRA across various ranks, and the final loss remains lower. This advantage is particularly pronounced when the rank is smaller. The gradient norm of PiSSA remains consistently higher than that of LoRA throughout the training process, indicating its efficient fitting of the training data. A closer look at the first few steps of LoRA's gradient norm reveals a trend of rising and then falling. LoRA's gradients are initially close to zero, leading to very slow model updates. This demonstrates the robustness of the faster convergence property of PiSSA across various ranks.

---

# Appendix: Comparison of Initial Gradient Subspaces

To compare the gradient subspaces of PiSSA and LoRA, we trained LLaMA-3-8B on the MetaMath dataset five times, initializing LoRA with different random seeds while using the same batch of 128 training examples to compute gradients.

We observed that the gradient of matrix $A$ remains consistently zero, while the gradient direction of matrix $B$ varies across initializations. This behavior arises because matrix $A$'s gradient depends on matrix $B$, which in LoRA is initialized to zero, resulting in a zero gradient for $A$. In contrast, matrix $B$ is initialized from a Gaussian distribution, leading to variation in its gradient direction across different seeds. PiSSA's gradient direction remains consistent across all five training runs, as it solely depends on the original model and the training data. This highlights the stability of PiSSA's optimization trajectory relative to LoRA's more variable directionality.

We also quantitatively compared the effect of updating along the principal singular value direction versus a "random" direction during the early stages of fine-tuning. Using LLaMA-3-8B on MetaMathQA, after just five updates, PiSSA reduced the loss from 0.8884 to 0.3346, while LoRA's loss reduction was more modest, dropping to only 0.5538. In the first step, matrix $A$ in LoRA exhibited a zero gradient and therefore did not update. Over the next four steps, it moved only 15.94% towards the target direction. Similarly, matrix $B$ in LoRA consistently moved less towards the target endpoint compared to PiSSA.
