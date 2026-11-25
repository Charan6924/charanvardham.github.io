# KL Divergence in LLM Post-Training

In machine learning, we often compare two probability distributions. Kullback–Leibler (KL) divergence, also called relative entropy, measures how much one distribution differs from another. 

Let $P$ and $Q$ be probability distributions on a discrete set $\mathcal{X}$. Their KL divergence is

$$
D_{\mathrm{KL}}(P\|Q) = \sum_{x\in\mathcal{X}} P(x)\log\frac{P(x)}{Q(x)}.
$$

If $P(x)>0$ and $Q(x)=0$ for any $x$, the divergence is infinite.

> [!note] Jensen's inequality
> A function $f:\mathbb{R}^k\to\mathbb{R}$ is convex if, for all $x,y\in\mathbb{R}^k$ and $\lambda\in[0,1]$,
>
> $$
> f(\lambda{x} + (1-\lambda)y \leq \lambda{f({x})} + (1-\lambda)f(y))
> $$
>
> It is strictly convex if the inequality is strict whenever $x\ne y$ and $0<\lambda<1$. For a random variable $X$ and a convex function $f$, Jensen's inequality states
>
> $$
> f\bigl(\mathbb{E}[X]\bigr)\leq\mathbb{E}[f(X)].
> $$
>
> If $f$ is strictly convex, equality holds only when $X$ is constant almost surely.

KL divergence is nonnegative. Let $S=\{x\in\mathcal{X}:P(x)>0\}$ and $r(x)=Q(x)/P(x)$ for $x\in S$. Since $\mathbb{E}_{P}[r]=\sum_{x\in S}Q(x)\leq 1$ and $-\log$ is convex, Jensen's inequality gives

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)&=\mathbb{E}_{P}[-\log r]\geq-\log\mathbb{E}_{P}[r]\\
&=-\log\left(\sum_{x\in S}Q(x)\right)\geq 0.
\end{aligned}
$$
Since $-\log$ is strictly convex, equality holds only when $P=Q$. Thus, $D_{\mathrm{KL}}(P\|Q)>0$ whenever $P\ne Q$.

## Probability Distributions in LLMs

An LLM generates text one token at a time. At each step, it assigns a probability to every token in its vocabulary, conditioned on the prompt and previously generated tokens.

Let $x$ be the prompt and $y=(y_1,\ldots,y_T)$ be a response. At position $t$, the model's next-token distribution is

$$
\pi_\theta(\cdot\mid x,y_{<t}).
$$

Here $\theta$ represents the model's parameters, and $y_{<t}=(y_1,\ldots,y_{t-1})$ are the previously generated tokens. The dot indicates a distribution over all possible next tokens.

The probability of a complete response follows the chain rule of probability:

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}).
$$

## Properties of KL Divergence

### Nonnegativity

$$
D_{\mathrm{KL}}(P\|Q)\geq 0.
$$

### Asymmetry

The order of the distributions matters. In general,

$$
D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P).
$$

### Zero when the distributions match

$$
D_{\mathrm{KL}}(P\|Q)=0\quad\Longleftrightarrow\quad P=Q.
$$

The divergence is zero exactly when $P$ and $Q$ are identical distributions.

### Support matters

If $P(i)>0$ but $Q(i)=0$ for any outcome $i$, then

$$
D_{\mathrm{KL}}(P\|Q)=\infty.
$$

In other words, $Q$ cannot assign zero probability to an outcome that $P$ considers possible without making the divergence infinite.


## Entropy, Cross-Entropy, and Relative Entropy

### *Surprise* function

$$
S(u)\triangleq\log\frac{1}{p(u)}.
$$
### Entropy

Let $U$ be a discrete random variable with values in an alphabet $\mathcal{U}$, and let $p(u)=\Pr(U=u)$. Its entropy is

$$
\begin{aligned}
H(U) &\triangleq \mathbb{E}[S(U)]\\[0.4em]
&= \mathbb{E}\left[\log\frac{1}{p(U)}\right]\\[0.4em]
&= \mathbb{E}[-\log p(U)]\\[0.4em]
&= -\sum_{u\in\mathcal{U}}p(u)\log p(u).
\end{aligned}
$$

Entropy is a property of the underlying distribution that measures the amount of randomness or surprise in the random variable. 

### Cross-Entropy

For distributions $p$ and $q$ on $\mathcal{X}$, cross-entropy is the expected negative log probability that $q$ assigns to outcomes drawn from $p$:

$$
\begin{aligned}
H(p,q)&=\mathbb{E}_{x\sim p}[-\log q(x)]\\[0.4em]
&=-\sum_{x\in\mathcal{X}}p(x)\log q(x).
\end{aligned}
$$

### Relative Entropy (KL Divergence)

For probability distributions $p$ and $q$ on $\mathcal{X}$, relative entropy can be written as

$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
&=\sum_{x\in\mathcal{X}}p(x)\log\frac{p(x)}{q(x)}\\[0.4em]
&=\mathbb{E}_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right]\\[0.4em]
&=\mathbb{E}_{x\sim p}[\log p(x)-\log q(x)]\\[0.4em]
&=\mathbb{E}_{x\sim p}[\log p(x)]-\mathbb{E}_{x\sim p}[\log q(x)]\\[0.4em]
&=H(p,q)-H(p).
\end{aligned}
$$

When $p$ is fixed, $H(p)$ is constant with respect to $q$. Minimizing $H(p,q)$ over $q$ also minimizes $D_{\mathrm{KL}}(p\|q)$. 

This is seen in Supervised Fine Tuning (SFT) where minimizing the negative log-likelihood is equivalent to minimizing empirical cross entropy against a fixed data distribution.

## Forward vs Reverse KL

We refer to $D_{KL}(P||Q)$ as the forward KL and $D_{KL}(Q||P)$ as the reverse KL between $P$ and $Q$. Forward KL encourages the student to cover the entire support of the teacher while Reverse KL tends to concentrate the student distribution on a subset of outputs that are assigned high probabilities by the teacher.

### Forward KL

The forward KL can be written as

$$
D_{\mathrm{KL}}(p\|q_\theta)=\mathbb{E}_{x\sim p}\left[\log p(x)-\log q_\theta(x)\right].
$$

Assuming the teacher $p(x)$ is fixed, differentiating with respect to $\theta$ gives

$$
\begin{aligned}
\nabla_\theta D_{\mathrm{KL}}(p\|q_\theta)
&=-\sum_x p(x)\nabla_\theta\log q_\theta(x)\\
&=-\mathbb{E}_{x\sim p}\left[\nabla_\theta\log q_\theta(x)\right].
\end{aligned}
$$

Each outcome's contribution to the gradient is weighted by its teacher probability $p(x)$. The student therefore receives a stronger training signal for outcomes the teacher considers more likely.

Forward KL encourages $q_\theta$ to cover the modes of $p$. In text classification, the output space contains a small, fixed set of classes, so covering those modes is manageable.

In open ended generation, the teacher distribution can have more modes than a smaller student can represent. Minimizing forward KL can then put too much probability in low probability regions of $p$ , and produce samples that are unlikely under $p$ during inference.

<img src="../assets/minillm-figure-2.png" alt="A Gaussian mixture fitted with a single Gaussian using forward and reverse KL divergence." width="538" style="max-width: 100%; height: auto;">

*Figure 2. A Gaussian mixture fitted with a single Gaussian using forward and reverse KL divergence. Recreated with Matplotlib from [Gu et al. (2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/8ac015d409635f196f9e3e9dcfb9a94e-Paper-Conference.pdf).*

### Reverse KL

The reverse KL on the other hand takes an expectation under the student,
$$D_{KL}(q_\theta||p) = \mathbb{E}_{x\sim q_{\theta}}[log \frac{q_\theta(x)}{p(x)}]$$
To derive its gradient we rewrite the expectation as a sum,
$$
D_{KL}(q_\theta||p) = \sum_{x}q_\theta(x)log \frac{q_\theta(x)}{p(x)}
$$
Applying the product rule with the teacher $p$ fixed,
$$
\nabla_\theta D_{KL}(q_\theta||p) = \sum_x\nabla_{\theta}q_\theta(x)[log\frac{q_\theta(x)}{p(x)} + 1]
$$

We also know that $q_\theta$ is a normalized distribution i.e it sums up to 1

$$
\sum_x\nabla_\theta q_\theta(x) = \nabla_\theta\sum_x q_\theta(x) = \nabla_\theta 1 = 0. $$

The +1 term vanishes. Using the log derivative trick, 
$$ \nabla_\theta q_\theta(x) = q_\theta(x)\nabla_\theta\log q_\theta(x)$$
We then obtain,

$$
\nabla_\theta D_{KL}(q_\theta||p) = \mathbb{E}_{x \sim q_{\theta}}\left[\log\frac{q_\theta(x)}{p(x)}\nabla_\theta\log q_\theta(x)\right]
$$
Here outcomes are weighted by the probability assigned by the student. Reverse KL penalizes the student for assigning a high probability to areas where the teacher assigns a low probability. When the student cannot fully represent the teacher, or optimization is limited, $q_\theta$ may concentrate on some modes of $p$ and assign little probability to others. It does not necessarily select the largest modes. If the student can represent the teacher, the global minimum is $q_\theta=p$, including its tail probabilities.

In open ended text generation, this mode seeking tendency can encourage a constrained student to concentrate on responses the teacher considers likely and avoid regions where the teacher assigns little probability. This does not guarantee correct generated content, but it helps the student learn to generate responses that better match the teacher's distribution.

## Token-Level and Sequence Level KL

An autoregressive language model defines both a probability distribution over the next token and a probability distribution over a complete response. We can compare a student and teacher at both levels.

Let $x$ be a prompt, $y = (y_1, y_2.....y_T)$ a response, $q_\theta$ a student policy and $p$ a teacher policy. For simplicity we will assume responses have a fixed length $T$. At position $t$, both policies are evaluated on the prefix $y_{<t}$

### Token Level KL

At a given prefix $y_{<t}$, token level reverse KL compares two distributions over the possible next token(s):

$$
d_{y_{<t}} = D_{KL}(q_\theta(.|x,y_{<t}) || p((q_\theta(.|x,y_{<t}))
$$
Expanding the divergence over the vocabulary $V$, 

$$
d_{y_{<t}} = \sum_{v \epsilon V}q_\theta(v | x, y_{<t})\log\frac{q_\theta(v|x, y_{<t})}{p(v|x,y_{<t})}
$$
This compares probabilities assigned to each token in the vocabulary conditioned on the given prefix. A token level distillation loss can sum or average these divergences over every token sampled in the response. In the case of On Policy Distillation (OPD), the responses are generated by the student and the teacher evaluates the prefixes the student generates.

### Sequence Level KL

Sequence level KL compares distributions over the entire response.

$$\mathcal{L}_{seq} = D_{KL}(q_\theta(y\mid x)||p(y \mid x))$$

Expanding the KL gives us
$$
\mathcal{L}_{seq} = \mathbb{E}_{y\sim q_\theta(.|x)}[\log \frac{q_\theta(y|x)}{p(y|x)}]
$$
Since both models are autoregressive,

$$ q_\theta(y\mid x) = \prod_{t=1}^{T} q_\theta(y_t\mid x,y_{<t}), \qquad p(y\mid x) = \prod_{t=1}^{T} p(y_t\mid x,y_{<t}) $$
 
 Substituting these products into the log ratio turns it into a sum:

$$\mathcal L_{\mathrm{seq}} = \mathbb E_{y\sim q_\theta} \left[ \sum_{t=1}^{T} \log \frac{ q_\theta(y_t\mid x,y_{<t}) }{ p(y_t\mid x,y_{<t}) } \right] $$

At each position, taking the expectation over the next token produces the KL between the two next token distributions:

$$
\mathcal{L}_{\mathrm{seq}}
= \mathbb{E}_{y\sim q_\theta(\cdot\mid x)}\left[
\sum_{t=1}^{T}
D_{\mathrm{KL}}\!\left(
q_\theta(\cdot\mid x,y_{<t})
\,\middle\|\,
p(\cdot\mid x,y_{<t})
\right)
\right]
$$


This is the chain rule for KL Divergence. The sequence level KL equals the expected sum of token level reverse KL terms conditioned on the prefix generated by the student. 

### The difference is seen in the gradient

If sequence level KL can be written as a sum of per-token KL terms why differentiate them in the first place?

The answer lies in how sampled prefixes are treated when calculating gradients.

A token level estimator samples a prefix from the student and then differentiates while treating the sampled prefix as fixed : 

$$ \mathcal L_{\mathrm{token}} = \mathbb E_{y\sim\operatorname{stopgrad}(q_\theta)} \left[ \sum_{t=1}^{T} d_t(y_{<t}) \right] $$
Here, $d_t(y_{<t})$ is the reverse KL divergence between the student and teacher distributions, conditioned on the prompt $x$ and the sampled prefix $y_{<t}$. This has the same numerical value as $\mathcal{L}_{seq}$. `stopgrad` however, prevents differentiation through the distribution that generated the prefix.

The sequence level gradient also accounts for the fact that changing the student changes which prefixes it visits. Let $q_\theta(y_{<t}\mid x)$ represent the probability of reaching a prefix. For a fixed teacher,

$$ \begin{aligned} \nabla_\theta \mathcal L_{\mathrm{seq}} = \mathbb E_{y\sim q_\theta} \left[ \sum_{t=1}^{T} \left( \nabla_\theta d_t(y_{<t}) + d_t(y_{<t}) \nabla_\theta\log q_\theta(y_{<t}\mid x) \right) \right] \end{aligned} $$

The first term updates the next token distribution at each sampled prefix. The second accounts for how the student’s earlier choices affect the prefix that is generated.


> [!note] KL direction
> The derivation above uses reverse KL. If we instead write a sequence-level forward KL,
>
> $$ D_{\mathrm{KL}}(p(y\mid x)\|q_\theta(y\mid x)) $$
>
> It's chain rule expectation is over teacher generated prefixes.
>
> The choice of KL direction and the choice of prefix distribution are separate. We can also calculate a forward KL between next token distributions at student generated prefixes at the token level.
****


## KL in RLHF

In Reinforcement Learning From Human Feedback (RLHF), a language model is trained to produce responses that receive higher scores from a reward model. The reward model is trained using human preferences and gives a higher reward when an output is more aligned with these preferences. However a reward model is a proxy metric, optimizing too aggressively can lead to a model forgetting previously learned behaviours or can cause it to exploit its weaknesses.

To limit this, RLHF uses KL Divergence to penalize the learned policy from drifting too far away from the reference policy. 

Let:

- $x$ be a prompt drawn from a prompt distribution $\mathcal D$.
- $y$ be a response sampled from the policy $\pi_\theta(\cdot\mid x)$.
- $r(x,y)$ be the reward model’s score.
- $\pi_{\mathrm{ref}}$ be the fixed reference policy.
- $\beta$ control the strength of the KL penalty.

A generalized RLHF objective is:
$$
 \max_\theta\; \mathbb E_{x\sim\mathcal D} \left[ \mathbb E_{y\sim\pi_\theta(\cdot\mid x)} \left[r(x,y)\right] - \beta D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid x) \| \pi_{\mathrm{ref}}(\cdot\mid x) \right) \right]
$$
The second term penalizes divergence from the reference and $\beta$ controls the tradeoff between reward and divergence

### Expanding the KL Penalty

Using the definition of KL Divergence,
$$D_{KL}(\pi_\theta||\pi_{ref}) := \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\left[\log \frac{\pi_\theta(y\mid x)}{\pi_{ref}(y \mid x)}\right]$$
Substituting it into the RLHF objective, we get
$$
 \max_\theta\; \mathbb E_{\substack{x\sim\mathcal D\\y\sim\pi_\theta(\cdot\mid x)}} \left[ r(x,y) - \beta\log \frac{ \pi_\theta(y\mid x) }{ \pi_{\mathrm{ref}}(y\mid x) } \right]
$$
The log ratio compares the probability with the current policy against the probability under the reference

### From sequences to tokens

Both policies are autoregressive. Therefore, for $y=(y_1,\ldots,y_T)$,

$$
\begin{aligned}
\log \frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
&= \log \frac{\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t})}{\prod_{t=1}^{T}\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}\\
&= \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}.
\end{aligned}
$$
This lets us express the sampled sequence log ratio as the sum of token log ratios
## KL in Distillation

Distillation is used to transfer knowledge from a teacher model to a student by training the student to mimic the teacher's predictions. Typically the student has fewer parameters than the teacher and distillation is used to improve task specific performance while maintaining lower inference costs and memory footprint.
### On Policy Distillation

Typical distillation approaches either sample responses from the teacher model or use a fixed set of responses drawn from a dataset. These approaches suffer from distribution mismatch between sequences seen during training and those generated by the student during training. Moreover, the student model may not be expressive enough to reasonably fit the teacher's distribution.

To solve this, on policy distillation samples sequences from the student and uses the teacher model to assign labels on those sequences. Given an input $x$, the student generates an output sequence $y$ and imitates the teacher's token-level distributions, $p_T(\cdot\mid x,y_{<t})$, at the generated prefixes. Specifically, the on-policy loss $\mathcal L_{\mathrm{OD}}$ is given by

$$
\mathcal L_{\mathrm{OD}}(\theta)
:= \mathbb E_{x\sim X}\mathbb E_{y\sim p_S(\cdot\mid x)}
\left[D_{\mathrm{KL}}\!\left(p_T\|p_S^\theta\right)(y\mid x)\right].
$$
$$ D_{\mathrm{KL}}(p_T\|p_S^\theta)(y\mid x) := \frac{1}{|y|} \sum_{t=1}^{|y|} D_{\mathrm{KL}} \left( p_T(\cdot\mid x,y_{<t}) \| p_S^\theta(\cdot\mid x,y_{<t}) \right)$$

Here the KL Divergence is used to compare the student generated and teacher generated distributions over the next token conditioned on the input $x$ and generated prefix $y_{<t}$. Both forward and reverse KL can be used. Reverse KL is preferred for open ended generation. 

### SDPO

Self Distillation Policy Optimization (SDPO) samples rollouts from the student, collects rich environment feedback and minimizes a logit level loss that matches the current policy's distribution to that of the teacher's. The teacher comprises of the current policy prompted with the question $x$ and the rich feedback $f$. The feedback can include any environment output and a sample solution if $x$ was already solved during a previous rollout. 

The KL divergence is used here to compare the next-token distributions of the student and self-teacher. 

$$
\mathcal L_{\mathrm{SDPO}}(\theta)
:= \sum_t \mathrm{KL}\!\left(
\pi_\theta(\cdot\mid x,y_{<t})
\,\middle\|\,
\operatorname{stopgrad}\!\left(\pi_\theta(\cdot\mid x,f,y_{<t})\right)
\right).
$$

#### MOPD

Multi Teacher On Policy Distillation (MOPD) trains experts in multiple domains and transfers each experts capabilities into a single student policy. Each prompt is routed to it's corresponding domain's teacher which provides a training signal for the student generated response. the student is optimised against a per-token reverse KL toward the dispatched teacher:

$$
\mathcal L_{\mathrm{rev\text{-}KL}}
= \mathbb E_{x,y\sim\pi_\theta}\left[
\frac{1}{|y|}\sum_t\sum_v
\pi_\theta(v)\log\frac{\pi_\theta(v)}{\pi_{\phi_d}(v)}
\right].
$$


## Alternative

KL Divergence is not the only way to compare two distributions. One alternative is Jensen-Shannon divergence. For teacher distribution $p_T$ and student distribution $p_S^\theta$, define their mixture as

$$ m=\frac{1}{2}\left(p_T+p_S^\theta\right)$$
The Jensen-Shannon divergence is then given by:

$$  D_{\mathrm{JS}}(p_T\|p_S^\theta) = \frac{1}{2}D_{\mathrm{KL}}(p_T\|m) + \frac{1}{2}D_{\mathrm{KL}}(p_S^\theta\|m) $$

The generalized form uses a weighted mixture:

$$
\begin{aligned}
m_\beta &= \beta p_T+(1-\beta)p_S^\theta,\\
D_{\mathrm{JS}}^{(\beta)}(p_T\|p_S^\theta)
&= \beta D_{\mathrm{KL}}(p_T\|m_\beta)
+(1-\beta)D_{\mathrm{KL}}(p_S^\theta\|m_\beta),
\qquad 0<\beta<1.
\end{aligned}
$$

Setting $\beta=\frac{1}{2}$ recovers the symmetric equation above. Unlike KL Divergence, Jensen-Shannon divergence is bounded.

## Conclusion

KL Divergence appears throughout LLM post training. It's role depends on the distributions being compared. In RLHF, it used used to penalize deviation from the baseline policy. In distillation it is used to compared student and teacher distributions and provide a learning signal.

The direction of the divergence changes how distributions are compared. Forward KL takes an expectation over the teacher while reverse KL takes an expectation over the student. It also matters compare next token distributions or distributions over an entire sequence. Learning these distinctions is paramount to understand what a post training method is asking the model to learn.

## Citations

- Weissman, T. (2024). [*Lecture 2: Entropy, Relative Entropy, and Mutual Information*](https://web.stanford.edu/class/ee276/files/previous_notes/lecture_2.pdf). Stanford University, EE276 lecture notes.
- MacCartney, B. (2005). [*CS224N Section 1*](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1056/handouts/section1.pdf). Stanford University course handout.
- Duchi, J. (2023). [*Lecture Notes on Statistics and Information Theory*](https://web.stanford.edu/class/stats311/lecture-notes.pdf). Stanford University, STATS 311 lecture notes.
- Agarwal, R., et al. (2024). [*On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf). ICLR 2024. [arXiv:2306.13649](https://arxiv.org/pdf/2306.13649).
- Gu, Y., Dong, L., Wei, F., & Huang, M. (2024). [*MiniLLM: Knowledge Distillation of Large Language Models*](https://proceedings.iclr.cc/paper_files/paper/2024/file/8ac015d409635f196f9e3e9dcfb9a94e-Paper-Conference.pdf). ICLR 2024. [arXiv:2306.08543](https://arxiv.org/pdf/2306.08543).
- Hübotter, J., et al. (2026). [*Reinforcement Learning via Self-Distillation*](https://arxiv.org/html/2601.20802#A1.SS1). arXiv:2601.20802.
- Ouyang, L., et al. (2022). [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/html/2203.02155). arXiv:2203.02155.
- GX-Chen, A., et al. (2025). [*KL-Regularized Reinforcement Learning is Designed to Mode Collapse*](https://arxiv.org/abs/2510.20817). arXiv:2510.20817.
- Stiennon, N., et al. (2020). [*Learning to Summarize from Human Feedback*](https://arxiv.org/html/2009.01325). arXiv:2009.01325.
- Jin, W., et al. (2026). [*Entropy-Aware On-Policy Distillation of Language Models*](https://arxiv.org/pdf/2603.07079). ICML 2026. arXiv:2603.07079.
- Chen, Y. (2025). [*Introduction to Kullback-Leibler Divergence*](https://medium.com/@yian.chen261/introduction-to-kullback-leibler-divergence-2d76979d1d8c). Medium.
- DeepSeek-AI. (2025). [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/pdf/2501.12948). arXiv:2501.12948.
- Malinin, A., & Gales, M. (2019). [*Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness*](https://proceedings.neurips.cc/paper/2019/hash/7dd2ae7db7d18ee7c9425e38df1af5e2-Abstract.html). NeurIPS 2019.
- Huszár, F. (2015). [*How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?*](https://arxiv.org/abs/1511.05101). arXiv:1511.05101.
- Rege, A. (2026). [*RL for LLM Post-Training: Foundations of Policy Gradients*](https://hackmd.io/@uJBQ39TYQW-gQckeNsmyIg/policy_gradient_algorithms). HackMD.
- Lambert, N. (2025). [*Instruction Fine-Tuning*](https://rlhfbook.com/c/04-instruction-tuning). In *Reinforcement Learning from Human Feedback*.
- Strauss, H., Fowl, L., & Razin, N. (2026). [*Which Tokens, in What Order, and How Fast? A Theoretical Analysis of On-Policy Distillation Objectives*](https://arxiv.org/pdf/2611.23397). arXiv:2611.23397.
- Ma, W., et al. (2026). [*MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training*](https://arxiv.org/html/2606.30406#S3). arXiv:2606.30406.

