# DoRA: Weight-Decomposed Low-Rank Adaptation

NVIDIA, HKUST

## Abstract

Among the widely used parameter-efficient fine-tuning (PEFT) methods, LoRA and its variants have gained considerable popularity because of avoiding additional inference costs. However, there still often exists an accuracy gap between these methods and full fine-tuning (FT). In this work, we first introduce a novel weight decomposition analysis to investigate the inherent differences between FT and LoRA. Aiming to resemble the learning capacity of FT from the findings, we propose Weight-**D**ecomposed L**o**w-**R**ank **A**daptation (**DoRA**). DoRA decomposes the pre-trained weight into two components, *magnitude* and *direction*, for fine-tuning, specifically employing LoRA for directional updates to efficiently minimize the number of trainable parameters. By employing DoRA, we enhance both the learning capacity and training stability of LoRA while avoiding any additional inference overhead. DoRA consistently outperforms LoRA on fine-tuning LLaMA, LLaVA, and VL-BART on various downstream tasks, such as commonsense reasoning, visual instruction tuning, and image/video-text understanding.

# Introduction

Models that are pre-trained with extensive general domain datasets have demonstrated remarkable generalization abilities, significantly benefiting a wide array of applications, from natural language processing (NLP) tasks to multi-modal tasks. To tailor these general models for specific downstream tasks, *full fine-tuning (FT)* is commonly employed, involving the retraining of all model parameters. Nevertheless, as the size of models and datasets expand in scale, the expense associated with fine-tuning the entire model becomes prohibitively large.

To address this issue, parameter-efficient fine-tuning (PEFT) methods have been introduced to fine-tune the pre-trained models with only a minimal number of parameters. Among these, LoRA, which does not change the model architecture, has become notably popular for its simplicity and efficacy. Nevertheless, there is still a capacity gap between LoRA and FT, which is often attributed to the limited number of trainable parameters without further exploration of other underlying causes.

Drawing on Weight Normalization, which achieves faster convergence via improving the conditioning of the gradient with weight reparameterization, we introduce a novel weight decomposition analysis that initially reparameterizes model weights into magnitude and directional components, subsequently examining the changes in magnitude and direction introduced by LoRA and FT. Our analysis reveals that LoRA and FT exhibit markedly distinct patterns of updates, leading us to surmise that these variations mirror the learning capability of each method. Inspired by our findings, we propose Weight-**D**ecomposed L**o**w-**R**ank **A**daptation (**DoRA**), which begins by decomposing the pre-trained weight into its magnitude and directional components, then fine-tunes both. Given the substantial size of the directional component in terms of parameters, we exploit LoRA for the directional adaptation to enable efficient fine-tuning.

Moreover, by showing a learning behavior similar to FT both empirically and mathematically, suggesting a learning capacity closely resembling FT, we have validated DoRA across a wide variety of tasks, from NLP to Vision-Language, and over various backbones, including LLM and LVLM. The experimental results show that DoRA consistently outperforms LoRA without sacrificing inference efficiency, such as commonsense reasoning (**+3.7**/**+1.0** on LLaMA-7B/13B, **+2.9** on LLaMA2-7B, and **+4.4** on LLaMA3-8B), visual instruction tuning (**+0.6** on LLaVA-7B), and image/video-text understanding (**+0.9**/**+1.9** on VL-BART).

The summary of our contributions is as follows:

- We introduce DoRA, a novel PEFT method that incorporates weight decomposition, achieving a learning capacity closely resembling FT without any additional inference latency over LoRA.
- We introduce a novel weight decomposition analysis to uncover the fundamental differences in the learning patterns of FT and different PEFT methods.
- DoRA consistently surpasses LoRA on various tasks, from NLP to Vision-Language benchmarks and across various backbones, including LLM and LVLM.

# Related Works

**Parameter-Efficient Fine-Tuning (PEFT)** methods are designed to reduce the high expense of fine-tuning large-scale models. They achieve this by training a relatively small subset of parameters, compared to the total number of parameters, for adapting to downstream tasks. Existing PEFT methods can be divided into three categories. The first category is *Adapter-based* methods, which involve introducing additional trainable modules into the original frozen backbone. For example, some approaches add linear modules in sequence to the existing layer, while others integrate these modules in parallel with the original layer to enhance performance. The second category is *Prompt-based* methods. These methods add extra soft tokens (prompts) to the initial input and focus solely on fine-tuning these trainable vectors. However, these approaches typically face challenges due to their sensitivity to initialization, affecting their overall effectiveness. These first two categories, whether altering the model's input or architecture, result in increased inference latency compared to the baseline model.

**LoRA and its variants** are among the third category of PEFT, notable for not adding any extra inference burden. These methods apply low-rank matrices to approximate weight changes during fine-tuning and can merge with pre-trained weights prior to inference. Variants include approaches using SVD decomposition with pruning of less significant singular values, low-rank Hadamard products, orthogonal factorization, weight tying to reduce trainable parameters, unified LoRA family frameworks, routing functions to combine different LoRAs for different tasks, and learnable scaling vectors with shared frozen random matrices across layers. Our research also falls within this third category.

# Pattern Analysis of LoRA and FT

## Low-Rank Adaptation (LoRA)

Building upon the hypothesis that updates made during the fine-tuning exhibit a low "intrinsic rank", LoRA proposes using the product of two low-rank matrices to update the pre-trained weights incrementally. For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA models the weight update $\Delta W \in \mathbb{R}^{d \times k}$ utilizing a low-rank decomposition, expressed as $BA$, where $B\in \mathbb{R}^{d \times r}$ and $A\in \mathbb{R}^{r \times k}$ represent two low-rank matrices, with $r \ll min(d,k)$. Consequently, the fine-tuned weight $W'$ can be represented as:

$$
W' = W_0 + \Delta W = W_0 + \underline{BA}
$$

where $W_0$ remains static during the fine-tuning process, and the underlined parameters are being trained. The matrix $A$ is initialized with uniform Kaiming distribution, while $B$ is initially set to zero, resulting in $\Delta W = BA$ being zero at the start of training. Notably, this decomposition of $\Delta W$ can be substituted with other LoRA variants, such as VeRA.

Additionally, based on the equation above, we can merge the learned $\Delta W$ with the pre-trained weight $W_0$ and obtain $W'$ in advance of deployment, and given that both $W'$ and $W_0$ both fall within the dimensionality of $\mathbb{R}^{d \times k}$, LoRA and its related variants do not introduce any extra latency during the inference compared to the original model.

## Weight Decomposition Analysis

The study presented in LoRA suggests that LoRA can be considered a general approximation of full fine-tuning. By gradually increasing the rank $r$ of LoRA to align with the rank of pre-trained weights, LoRA can attain a level of expressiveness akin to that of FT. Consequently, many previous studies have attributed the discrepancy in accuracy between LoRA and FT primarily to the limited number of trainable parameters, often without further analysis. Drawing inspiration from Weight Normalization, which reparameterizes the weight matrix into magnitude and direction for accelerating optimization, we introduce an innovative weight decomposition analysis. Our analysis restructures the weight matrix into two separate components, *magnitude* and *direction*, to reveal the inherent differences in LoRA and FT learning patterns.

**Analysis Method:** This analysis examines the updates in both magnitude and direction of the LoRA and FT weights relative to the pre-trained weights to reveal the fundamental differences in the learning behaviors of both. The weight decomposition of $W \in \mathbb{R}^{d \times k}$ can be formulated as:

$$
W = m\frac{V}{||V||_{c}} = ||W||_{c}\frac{W}{||W||_{c}}
$$

where $m \in \mathbb{R}^{1 \times k}$ is the magnitude vector, $V \in \mathbb{R}^{d \times k}$ is the directional matrix, with $||\cdot||_{c}$ being the vector-wise norm of a matrix across each column. This decomposition ensures that each column of $V/||V||_{c}$ remains a unit vector, and the corresponding scalar in $m$ defines the magnitude of each vector.

For our weight decomposition analysis, we select the VL-BART model fine-tuned on four image-text tasks for a case study, applying LoRA only to the query/value weight matrix in the self-attention module. We decompose the pre-trained weight $W_0$, the full fine-tuned weight $W_{\text{FT}}$, and the merged LoRA weight $W_{\text{LoRA}}$ of query/value weight matrix using the weight decomposition equation. The magnitude and directional variations between $W_0$ and $W_{\text{FT}}$ can be defined as follows:

$$
\Delta M_{\text{FT}}^{t} = \frac{\sum_{n = 1}^{k} |m_{\text{FT}}^{n,t} - m_0^{n}|}{k}
$$

$$
\Delta D_{\text{FT}}^{t} = \frac{\sum_{n = 1}^{k} (1- \mathbf{cos}(V_{\text{FT}}^{n,t}, W_{0}^{n}))}{k}
$$

Here, $\Delta M_{\text{FT}}^{t}$ and $\Delta D_{\text{FT}}^{t}$ represent the magnitude difference and directional difference between $W_0$ and $W_{\text{FT}}$ at $t$ training step respectively, with $\mathbf{cos}(\cdot,\cdot)$ being the cosine similarity function. $M_{\text{FT}}^{n,t}$ and $M_0^{n}$ are the $n^{th}$ scalars in their respective magnitude vectors, while $V_{\text{FT}}^{n,t}$ and $W_{0}^{n}$ are the $n^{th}$ columns in $V_{\text{FT}}^{t}$ and $W_{0}$. The magnitude and directional differences between $W_{\text{LoRA}}$ and $W_0$ are calculated similarly. We select checkpoints from four different training steps for analysis, comprising three intermediate steps and the final checkpoint from both FT and LoRA, and we perform weight decomposition analysis on each of these checkpoints to determine the $\Delta M$ and $\Delta D$ throughout different layers.

**Analysis Results:** Each point represents a ($\Delta D^{t}$, $\Delta M^{t}$) pair from query weight matrices across different layers and training steps. LoRA exhibits a consistent positive slope trend across all the intermediate steps, signifying a proportional relationship between the changes in direction and magnitude. In contrast, FT displays a more varied learning pattern with a relatively negative slope. This distinction between FT and LoRA likely mirrors their respective learning capability. While LoRA tends to either increase or decrease the magnitude and direction updates proportionally, it lacks the nuanced capability for more subtle adjustments. Specifically, LoRA does not show proficiency in executing slight directional changes alongside more significant magnitude alterations, or vice versa, a feature more characteristic of the FT method. We suspect that such limitation of LoRA might stem from the challenge of concurrent learning both magnitude and directional adaptation, which could be overly complex for LoRA. Consequently, in this work, we aim to propose a variant of LoRA that exhibits a learning pattern more closely resembling that of FT, and can improve the learning capacity over LoRA.

# Method

## Weight-Decomposed Low-Rank Adaptation

Drawing from the insights of our weight decomposition analysis, we introduce Weight-**D**ecomposed L**o**w-**R**ank **A**daptation (**DoRA**). DoRA initially decomposes the pre-trained weight into its magnitude and directional components and finetunes both of them. Because the directional component is large in terms of parameter numbers, we further decompose it with LoRA for efficient finetuning.

Our intuitions are two-fold. Firstly, we believe that limiting LoRA to concentrate exclusively on directional adaptation while also allowing the magnitude component to be tunable simplifies the task compared to the original approach, where LoRA is required to learn adjustments in both magnitude and direction. Secondly, the process of optimizing directional updates is made more stable through weight decomposition, which we delve into more thoroughly in the Gradient Analysis section. The main distinction between DoRA and weight normalization lies in their training approaches. Weight normalization trains both components from scratch, making the method sensitive to different initializations. Conversely, DoRA avoids such initialization concerns since both components begin with pre-trained weights. We initialize DoRA with pre-trained weight $W_0$ as outlined in the weight decomposition equation, where $m = ||W_0||_{c}$ and $V = W_0$ after initialization. We then keep $V$ frozen and $m$ a trainable vector. The directional component is then updated through LoRA. DoRA can be formulated similar to the LoRA equation as:

$$
W'= \underline{m}\frac{V+\Delta V}{||V+\Delta V||_{c}} = \underline{m}\frac{W_0+\underline{BA}}{||W_0+\underline{BA}||_{c}}
$$

where $\Delta V$ is the incremental directional update learned by multiplying two low-rank matrices $B$ and $A$, and the underlined parameters denote the trainable parameters. The matrices $B\in \mathbb{R}^{d \times r}$ and $A\in \mathbb{R}^{r \times k}$ are initialized in line with LoRA's strategy to ensure that $W'$ equals $W_0$ before the finetuning. Furthermore, DoRA can be merged with the pre-trained weight before inference, thereby not introducing any additional latency.

We visualize the magnitude and directional differences of the query weight matrix between the merged DoRA weight and $W_0$ in the same setting as for FT and LoRA. From the regression line for $(\Delta D, \Delta M)$ of both DoRA and FT, we reveal that in contrast to LoRA's pattern, DoRA, and FT are characterized by a distinct negative slope. We reason that FT tends towards a negative slope because pre-trained weights already possess substantial knowledge suitable for various downstream tasks. Therefore, when provided with adequate learning capacity, having a larger magnitude or direction alteration alone is sufficient enough for downstream adaptation. We additionally compute the correlation between $\Delta D$ and $\Delta M$ for FT, LoRA, and DoRA, and we found that both FT and DoRA exhibit negative correlation values of -0.62 and -0.31, respectively. In contrast, LoRA shows a positive correlation with a value of 0.83. The fact that DoRA demonstrates the ability to make only substantial directional adjustments with relatively minimal changes in magnitude or the reverse while showing learning patterns closer to FT's signifies its superior learning capacity over LoRA.

## Gradient Analysis of DoRA

In this section, we first derive the gradient of DoRA and illustrate how our proposed decomposition benefits the optimization of $\Delta V$. Subsequently, we analyze from the gradient's perspective to explicate the learning pattern of DoRA, which tends to have a negative slope.

From the DoRA equation, we can obtain the gradient of Loss $\mathcal{L}$ with respect to $m$ and $V' = V+\Delta V$ as:

$$
\nabla_{V'} \mathcal{L} = \frac{m}{||V'||_{c}}\left( I - \frac{V'V'^{\mathbf{T}}}{||V'||_{c}^2}  \right) \nabla_{W'} \mathcal{L}
$$

$$
\nabla_m \mathcal{L} =\frac{\nabla_{W'} \mathcal{L} \cdot V'}{||V'||_{c}}
$$

The first equation reveals that the weight gradient $\nabla_{W'} \mathcal{L}$ is scaled by $m/||V'||_{c}$ and is projected away from the current weight matrix. These two effects contribute to aligning the gradient's covariance matrix more closely with the identity matrix, which is advantageous for optimization. Additionally, given that $V' = V + \Delta V$, the gradient $\nabla_{V'} L$ is equivalent to $\nabla_{\Delta V} L$. Therefore, the optimization benefits derived from this decomposition are fully transferred to $\Delta V$, enhancing the learning stability of LoRA.

We can gain further insight into the learning pattern of DoRA by referring to the gradient of $m$. In the subsequent discussion, we represent vectors using lower-case letters instead of the previous matrix form notation. Consider $w'' = w' + \Delta w$ as the parameter update for a weight vector, where $\Delta w \propto \nabla_{w'} \mathcal{L}$. In two hypothetical update scenarios, $S1$ and $S2$, $S1$ involves a smaller directional update ($\Delta D_{S1}$), while $S2$ involves a larger one ($\Delta D_{S2}$). Assuming $||\Delta w_{S1}|| = ||\Delta w_{S2}||$, and at time 0, we have $\Delta v = 0$ and $v' = v$. From $\Delta D_{S1} < \Delta D_{S2}$, it follows that $|\mathbf{cos}(\Delta w_{S1}, w')| > |\mathbf{cos}(\Delta w_{S2}, w')|$. Since $\Delta w \propto \nabla_{w'} \mathcal{L}$, it implies $|\mathbf{cos}(\nabla_{w'}^{S1} \mathcal{L}, w')| > |\mathbf{cos}(\nabla_{w'}^{S2} \mathcal{L}, w')|$. With $v$ initialized as $v_0$ and $w' = w_0$ at time 0, we get $|\mathbf{cos}(\nabla_{w'} \mathcal{L}, w')| = |\mathbf{cos}(\nabla_{w'} \mathcal{L}, v')| = |\mathbf{cos}(\nabla_{w'} \mathcal{L}, v)|$. Using the cosine similarity equation with $\Delta v = 0$:

$$
cos(\nabla_{w'} \mathcal{L}, v') = cos(\nabla_{w'} \mathcal{L}, v) =  \frac{\nabla_{w'} \mathcal{L} \cdot v}{||\nabla_{w'} \mathcal{L}||||v||}
$$

denote $m_{*}$ as the magnitude scalar of vector $w'$, then the gradient with respect to $m_{*}$ can be rewritten to:

$$
\nabla_{m_{*}} \mathcal{L} = \frac{\nabla_{w'} \mathcal{L} \cdot v'}{||v'||} = ||\nabla_{w'} \mathcal{L}|| \cdot cos(\nabla_{w'} \mathcal{L}, v)
$$

Given that $||\Delta w_{S1}|| = ||\Delta w_{S2}||$ for $S1$ and $S2$, and $||\nabla_{w'}^{S1} \mathcal{L}|| = ||\nabla_{w'}^{S2} \mathcal{L}||$. Therefore, with:

$$
||\nabla_{w'}^{S1} \mathcal{L}|| \cdot |cos(\nabla_{w'}^{S1} \mathcal{L}, v)| > ||\nabla_{w'}^{S2} \mathcal{L}|| \cdot |cos(\nabla_{w'}^{S2} \mathcal{L}, v)|
$$

it can be inferred that $|\nabla_{m_{*}}^{S1} \mathcal{L}| > |\nabla_{m_{*}}^{S2} \mathcal{L}|$ which indicates that $S1$ has larger magnitude updates over $S2$ while having smaller directional alteration than that of $S2$. Our conclusion generally holds in practice, as evidenced by the empirical results. Consequently, we have effectively shown how DoRA can be utilized to adjust the learning pattern, diverging from that of LoRA and aligning more closely with the pattern of FT.

## Reduction of Training Overhead

In the LoRA equation, the gradients of $W'$ and $\Delta W$ are the same. However, with DoRA, which redirects the low-rank adaptation towards the directional component, the gradient of the low-rank updates differs from that of $W'$, as illustrated in the gradient equation for $V'$. This divergence necessitates extra memory during backpropagation. To address this, we suggest treating $||V + \Delta V||_{c}$ in the DoRA equation as a constant, thereby detaching it from the gradient graph. This means that while $||V + \Delta V||_{c}$ dynamically reflects the updates of $\Delta V$, it won't receive any gradient during backpropagation. With this modification, the gradient w.r.t $m$ remains unchanged, and $\nabla_{V'} \mathcal{L}$ is redefined as:

$$
\nabla_{V'} \mathcal{L} = \frac{m}{C} \nabla_{W'} \mathcal{L} \text{ where } C = ||V'||_{c}
$$

This approach reduces the gradient graph memory consumption drastically without a noticeable difference in accuracy. An ablation study evaluating the impact of this modification on fine-tuning LLaMA-7B and VL-BART shows that it leads to a training memory reduction of approximately 24.4% in fine-tuning LLaMA and 12.4% in VL-BART. Furthermore, the accuracy of DoRA with the modification remains unchanged for VL-BART and shows a negligible difference of only 0.2 compared to DoRA without the modification on LLaMA. Consequently, all subsequent experiments with DoRA incorporate this adjustment.

# Experiments

We conduct a variety of experiments to showcase the efficacy of DoRA on various tasks including language, image, and video domains. We evaluate DoRA against several PEFT methods by fine-tuning LLaMA-7B/13B, LLaMA2-7B, and LLaMA3-8B on commonsense reasoning tasks; compare DoRA with LoRA across multi-task image/video-text understanding tasks using VL-BART and visual instruction tuning with LLaVA-1.5-7B; explore the compatibility of DoRA with LoRA and VeRA for instruction-tuning; and perform ablation studies on training sample size, rank variations, and tuning granularity.

## Commonsense Reasoning

We evaluate DoRA against LoRA and several baseline methods including Prompt learning (Prefix), Series adapter, and Parallel adapter on LLaMA-7B/13B for commonsense reasoning tasks. The commonsense reasoning tasks comprise 8 sub-tasks (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA), each with a predefined training and testing set. Training datasets from all 8 tasks are amalgamated to create the final training dataset and evaluations are conducted on the individual testing dataset for each task. To ensure a fair comparison, DoRA follows the LoRA configuration maintaining the same rank while adjusting only the learning rate. The marginal increase of 0.01% in trainable parameters for DoRA over LoRA arises from the inclusion of learnable magnitude components (parameter of size $1 \times k$). We also halve the rank used in DoRA compared to LoRA and denote this adjusted configuration as DoRA-dagger.

DoRA consistently surpasses all baseline methods across LLaMA-7B/13B, LLaMA2-7B, and LLaMA3-8B. On LLaMA-7B, where LoRA exceeds other baselines at 74.7% average accuracy, DoRA further enhances accuracy to 78.4% (+3.7%), outstripping ChatGPT's 77.0%. For LLaMA-13B, where LoRA (80.5%) is inferior to the Parallel adapter (81.4%), DoRA achieves 81.5% with only a quarter of the Parallel adapter's trainable parameters and without extra inference overhead. DoRA surpasses LoRA on LLaMA2-7B by 2.1% (79.7% vs 77.6%) and on LLaMA3-8B by 4.4% (85.2% vs 80.8%). DoRA-dagger, with only half LoRA's trainable parameters, exceeds LoRA on LLaMA-7B by 2.8%, LLaMA-13B by 0.3%, LLaMA2-7B by 2.9%, and LLaMA3-8B by 4.2%. This suggests that DoRA's weight decomposition enhances learning capability, reducing the need for higher rank.

Additionally, we validated the hypothesis that a negative correlation between magnitude and directional updates is more optimal than a positive correlation, using LLaMA2-7B fine-tuned on commonsense reasoning datasets. The DoRA fine-tuned weights show less deviation from the pre-trained weights in both magnitude and direction, while the differences for LoRA fine-tuned weights are significantly larger. Coupled with DoRA's significantly better performance, this confirms that a robust foundation model does not require significant alterations for effective downstream adaptation, and having the ability to perform more fine-grained magnitude and directional updates explains DoRA's superiority over LoRA.

## Image/Video-Text Understanding

We compare DoRA with LoRA and full fine-tuning on VL-BART, which comprises a vision encoder (CLIP-ResNet101) and an encoder-decoder language model (BART-Base), across four image-text tasks (VQA-v2, GQA, NLVR2, MSCOCO captioning) and four video-text tasks from the VALUE Benchmark (TVQA, How2QA, TVC, YC2C).

On image-text tasks, DoRA (5.96% params) achieves an average of 77.4, surpassing LoRA (5.93% params, avg 76.5) by nearly 1% and matching FT (100% params, avg 77.3). On video-text tasks, DoRA (5.19% params) achieves an average of 85.4, surpassing LoRA (5.17% params, avg 83.5) by roughly 2%, approaching FT (100% params, avg 87.5).

## Visual Instruction Tuning

We scale up to LLaVA-1.5-7B, composed of Vicuna-1.5-7B and CLIP ViT-L/336px, and evaluate on seven vision-language benchmarks (VQA-v2, GQA, VisWiz, SQA, VQA-T, POPE, MMBench). DoRA follows the same LoRA configuration provided for LLaVA.

The average accuracy of LoRA (66.9%) already surpasses FT (66.5%), suggesting FT may be overfitting. In scenarios where FT is inferior to LoRA, DoRA's improvement over LoRA may be less pronounced. Nonetheless, DoRA achieves 67.6% average accuracy, improving over LoRA by 0.7% and over FT by 1.1%.

## Compatibility of DoRA with other LoRA variants

The concept of incremental directional update $\Delta V$ in DoRA can be replaced with alternative LoRA variants. We select VeRA as a case study, which freezes a unique pair of random low-rank matrices shared across all layers and employs only minimal layer-specific trainable scaling vectors, reducing trainable parameters by 10x compared to LoRA with minimal accuracy impact. Applying VeRA for the directional update in DoRA yields DVoRA.

We assess DVoRA and DoRA against VeRA and LoRA on LLaMA-7B and LLaMA2-7B, focusing on instruction tuning with the 10K subset of cleaned Alpaca dataset, evaluated on the MT-Bench benchmark. DoRA consistently improves over LoRA (5.5 vs 5.1 on LLaMA-7B, 6.0 vs 5.7 on LLaMA2-7B), and DVoRA consistently improves over VeRA (5.0 vs 4.3 on LLaMA-7B, 6.0 vs 5.5 on LLaMA2-7B). DVoRA achieves scores on par with or surpassing LoRA despite having significantly fewer parameters.

To further assess DoRA under varying training data amounts, we compare methods with sample sizes of 1000, 4000, 7000, and 10000. DoRA and DVoRA consistently outperform LoRA and VeRA across all training sample sizes. With 7000 samples, DoRA and DVoRA surpass LoRA and VeRA by 0.3 and 0.33 respectively. Even at 1000 samples, DoRA and DVoRA maintain leads of 0.29 and 0.22 over LoRA and VeRA.

## Robustness of DoRA towards different rank settings

We explore the impact of various rank configurations by adjusting $r$ within {4, 8, 16, 32, 64} on LLaMA-7B for commonsense reasoning tasks. DoRA consistently surpasses LoRA across all rank configurations. The performance gap widens for ranks below 8: LoRA drops to 40.7% for r=8 and 39.5% for r=4, while DoRA retains 78.0% for r=8 and 61.9% for r=4, demonstrating its resilience and consistently superior performance regardless of rank setting.

## Tuning Granularity Analysis

We investigate whether it is possible to decrease trainable parameters by updating only the magnitude components of specific modules while continuing to update both magnitude and directional components for the remaining linear modules. In contrast to the original LoRA configuration which requires updates to both Multi-head Attention and MLP layers, DoRA can achieve superior accuracy by updating only the directional and magnitude components of the multi-head attention layers and the magnitude of the MLP layers. By updating the directional and magnitude components of the QKV modules and only the magnitude of the rest of the layers, DoRA surpasses LoRA by 2.8% on LLaMA-7B (77.5% vs 74.7%) and 0.8% on LLaMA-13B (81.3% vs 80.5%), while utilizing less than half of the trainable parameters compared to LoRA.

## QDoRA: Enhancements to QLoRA

QLoRA quantizes the pretrained model to 4-bit and finetunes LoRA on top of the frozen low-bit backbone to further decrease memory demands. Substituting the LoRA component in QLoRA with DoRA yields QDoRA. Experiments on fine-tuning LLaMA2-7B/LLaMA3-8B using the Orca-Math dataset (100k training samples, 500 evaluation samples, exact match metric) show that QDoRA significantly surpasses QLoRA by 0.19/0.23 on LLaMA2-7B and LLaMA3-8B, and slightly outperforms FT on both models while using considerably less memory. This indicates that QDoRA can effectively combine the parameter efficiency of QLoRA with the more granular optimization of full finetuning.

## Text-to-Image Generation

We explore whether DoRA's advantages extend to text-to-image generation by fine-tuning SDXL via DreamBooth on two datasets (3D icons and Lego sets), keeping hyperparameter settings identical for LoRA and DoRA. DoRA achieves significantly better personalization than LoRA with identical training settings, more accurately reflecting training targets. For example, DoRA consistently captures distinctive features of the training data (such as unique visual elements in 3D icons and logos in Lego sets) that are absent in LoRA outputs.

# Conclusion

In this work, we first conduct a novel weight decomposition analysis to reveal the distinct learning patterns between LoRA and FT. Building on these insights, we introduce DoRA, a fine-tuning method that is compatible with LoRA and its variants and exhibits a closer resemblance to FT's learning behavior. DoRA consistently outperforms LoRA across various fine-tuning tasks and model architectures. Specifically, DoRA improves upon LoRA in commonsense reasoning and visual instruction tuning tasks. Furthermore, DoRA also shows compatibility with VeRA on the Alpaca instruction tuning task. Moreover, DoRA can be considered as a costless alternative to LoRA, as its decomposed magnitude and direction components can be merged back into the pre-trained weight after the training, ensuring that there is no extra inference overhead.
