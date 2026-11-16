# Self-Distillation Policy Optimization (SDPO)

SDPO turns feedback about a model's answer into a dense, token-level learning signal. It needs neither an external teacher nor an explicit reward model: the same model, prompted with feedback, acts as its own teacher.

## A model as its own teacher

Let $x$ be a question and $f$ the feedback on an initial answer. The **self-teacher** is $\pi_\theta(\cdot \mid x, f)$: the current policy prompted with both $x$ and $f$. Feedback can include environment output, such as a runtime error, or a successful solution from an earlier rollout.

Since it sees this additional information, we expect the self-teacher $\pi_\theta(\cdot \mid x, f)$ to make better-informed predictions than the student $\pi_\theta(\cdot \mid x)$.

SDPO uses the same policy in both roles: the student generates an initial attempt, and the self-teacher evaluates its token choices in hindsight.

## Distillation objective

For each question $x$, we sample an answer $y$ from the student and obtain feedback $f$. We then compare the student and self-teacher next-token distributions at each position in that same answer. The teacher does not need to generate a second answer.

We measure the difference between two distributions with the KL divergence:

$$
\mathrm{KL}(p\|q) = \sum_i p(i)\log\frac{p(i)}{q(i)}
$$

The SDPO objective sums this divergence across the answer's token positions:

$$
\mathcal{L}_{\mathrm{SDPO}}(\theta) := \sum_{t=1}^{|y|} \mathrm{KL}\left(\pi_\theta(\cdot \mid x, y_{<t}) \,\middle\|\, \operatorname{stopgrad}\left(\pi_\theta(\cdot \mid x, f, y_{<t})\right)\right)
$$

The $\operatorname{stopgrad}$ operator blocks gradients through the self-teacher. This keeps the teacher from moving toward the student's original predictions and ignoring the feedback.

## Computing the SDPO loss

The implementation scores the same sampled answer twice: once as the student and once with feedback added to the self-teacher's context. In pseudocode:

```python
def compute_sdpo_loss(batch, teacher_context, loss_mask):
    logprobs_student = compute_log_prob(batch)  # (T, V)

    teacher_batch = reprompt(batch, teacher_context)
    logprobs_teacher = compute_log_prob(teacher_batch).detach()  # (T, V)

    per_token_loss = divergence(logprobs_student, logprobs_teacher)  # (T,)
    return agg_loss(per_token_loss, loss_mask, loss_agg_mode="token-mean")
```

Here, `detach()` implements $\operatorname{stopgrad}$, and `loss_mask` restricts the average to answer tokens. The divergence produces one loss value per token position.

The KL divergence above explains the basic objective. In the experiments, the authors use symmetric Jensen–Shannon divergence and a regularized self-teacher for training stability. They also use top-$K$ distillation with a tail term to avoid storing the full student and teacher logits in memory.

## Gradient computation

For a fixed sampled answer $y$ and its feedback $f$, let $T=|y|$. We begin by expanding the KL divergence over possible next tokens $\hat{y}_t$:

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{\mathrm{SDPO}}(\theta)
&= \nabla_\theta \sum_{t=1}^{T} \mathrm{KL}\left(\pi_\theta(\cdot \mid x, y_{<t}) \,\middle\|\, \operatorname{stopgrad}\left(\pi_\theta(\cdot \mid x, f, y_{<t})\right)\right) \\
&= \nabla_\theta \sum_{t=1}^{T} \sum_{\hat{y}_t} \pi_\theta(\hat{y}_t \mid x, y_{<t}) \log\frac{\pi_\theta(\hat{y}_t \mid x, y_{<t})}{\operatorname{stopgrad}\left(\pi_\theta(\hat{y}_t \mid x, f, y_{<t})\right)}.
\end{aligned}
$$

Define the token-level advantage as:

$$
A_t(\hat{y}_t) := \log\frac{\operatorname{stopgrad}\left(\pi_\theta(\hat{y}_t \mid x, f, y_{<t})\right)}{\pi_\theta(\hat{y}_t \mid x, y_{<t})}.
$$

Then, applying the product rule of differentiation, we get:

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{\mathrm{SDPO}}(\theta)
&= -\nabla_\theta \sum_{t=1}^{T} \sum_{\hat{y}_t}
\pi_\theta(\hat{y}_t \mid x, y_{<t}) A_t(\hat{y}_t) \\
&= -\sum_{t=1}^{T} \sum_{\hat{y}_t} \Bigl[\pi_\theta(\hat{y}_t \mid x, y_{<t})\nabla_\theta A_t(\hat{y}_t) + A_t(\hat{y}_t)\nabla_\theta \pi_\theta(\hat{y}_t \mid x, y_{<t})\Bigr].
\end{aligned}
$$
The derivative of $A_t(\hat{y}_t)$ is the negative score function:

$$
\nabla_\theta A_t(\hat{y}_t) = -\nabla_\theta \log \pi_\theta(\hat{y}_t \mid x, y_{<t}).
$$

Using the log-derivative trick,

$$
\pi_\theta(\hat{y}_t \mid x, y_{<t})\nabla_\theta \log \pi_\theta(\hat{y}_t \mid x, y_{<t})
= \nabla_\theta \pi_\theta(\hat{y}_t \mid x, y_{<t}).
$$

<aside class="note">
<strong>Why the log-derivative trick works</strong>
<p>Let $p_\theta = \pi_\theta(\hat{y}_t \mid x, y_{<t})$. By the chain rule, $\nabla_\theta \log p_\theta = (\nabla_\theta p_\theta)/p_\theta$. Multiplying both sides by $p_\theta$ gives the equation above.</p>
</aside>

The first term then simplifies to:

$$
\begin{aligned}
-\sum_{t=1}^{T}\sum_{\hat{y}_t}
\pi_\theta(\hat{y}_t \mid x, y_{<t})\nabla_\theta A_t(\hat{y}_t)
&= \sum_{t=1}^{T}\sum_{\hat{y}_t}
\nabla_\theta \pi_\theta(\hat{y}_t \mid x, y_{<t}) \\
&= \sum_{t=1}^{T}\nabla_\theta
\underbrace{\sum_{\hat{y}_t}\pi_\theta(\hat{y}_t \mid x, y_{<t})}_{=1} \\
&= 0.
\end{aligned}
$$

Thus, the gradient of $\mathcal{L}_{\mathrm{SDPO}}$ is:

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{\mathrm{SDPO}}(\theta)
&= -\sum_{t=1}^{T}\sum_{\hat{y}_t}
A_t(\hat{y}_t)\nabla_\theta \pi_\theta(\hat{y}_t \mid x, y_{<t}) \\
&= -\sum_{t=1}^{T}\sum_{\hat{y}_t}
\pi_\theta(\hat{y}_t \mid x, y_{<t})
\left[A_t(\hat{y}_t)\nabla_\theta \log \pi_\theta(\hat{y}_t \mid x, y_{<t})\right] \\
&= -\sum_{t=1}^{T}
\mathbb{E}_{\hat{y}_t \sim \pi_\theta(\cdot \mid x, y_{<t})}
\left[A_t(\hat{y}_t)\nabla_\theta \log \pi_\theta(\hat{y}_t \mid x, y_{<t})\right].
\end{aligned}
$$

During training, this gradient is averaged over student rollouts $y\sim\pi_\theta(\cdot\mid x)$.

## Using SDPO with policy gradient algorithms

The SDPO gradient has the same form as a policy gradient: an advantage weights the gradient of a token's log probability. SDPO obtains that advantage by comparing the student and self-teacher predictions.

Let $y_i$ be the $i$th rollout for question $x$, with feedback $f_i$. The SDPO advantage is computed as follows:

$$
A_{i,t}^{\mathrm{SDPO}}(\hat{y}_{i,t})
= \log\frac{\pi_\theta(\hat{y}_{i,t} \mid x, f_i, y_{i,<t})}{\pi_\theta(\hat{y}_{i,t} \mid x, y_{i,<t})}.
$$

This has the same value as $A_t(\hat{y}_t)$ for rollout $i$. Since the extra derivative term vanished above, the equivalent policy gradient update treats the entire advantage as fixed during backpropagation.

A policy gradient has the general form:

$$
g = \nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim p_\theta}\left[
\sum_{t=0}^{T-1}\Psi_t \cdot \nabla_\theta\log\pi_\theta(a_t \mid s_t)
\right].
$$

Here, $s_t$ is a state, $a_t$ is an action, and $\Psi_t$ is an advantage weight.

For SDPO, the state is the question and sampled prefix, $s_{i,t}=(x,y_{i,<t})$.

At prefix $s_{i,t}$, $a_t$ corresponds to a candidate next token $\hat{y}_{i,t}$.

Since SDPO minimizes $\mathcal{L}_{\mathrm{SDPO}}$, its ascent direction is $-\nabla_\theta\mathcal{L}_{\mathrm{SDPO}}$. SDPO can assign a separate advantage to every candidate next token at each sampled prefix. Its full update averages over those candidates; using only the generated token gives a sampled approximation.

## Comparison with other algorithms

The methods differ in how they sample answers and how much information each training signal provides:

| Method                     | Sampling   | Learning signal                     | Feedback or supervision                             |
| -------------------------- | ---------- | ----------------------------------- | --------------------------------------------------- |
| SFT / teacher distillation | Off policy | Dense targets for each token        | Labeled data or an external teacher                 |
| On policy distillation     | On policy  | Dense targets for each token        | External teacher                                    |
| RLVR, such as GRPO         | On policy  | Sparse outcome reward               | Environment or verifier                             |
| SDPO                       | On policy  | Dense credit for each token         | Environment feedback or successful student rollouts |

Here, *dense* means the method scores each token individually, while *sparse* assigns a single scalar per completion.

## Limitations

SDPO is more computationally expensive than algorithms such as GRPO. Naively computing the KL divergence between the student and teacher distributions requires holding two minibatches of logits in memory.

SDPO's performance depends on the model's ability to learn from feedback. It tends to work better with larger base models and can underperform with smaller ones. Feedback quality also matters: vague or misleading feedback gives the model little guidance.

## Conclusion

SDPO produces substantially shorter generations than GRPO while achieving higher accuracy. Its responses are three times shorter than GRPO's on average. It also avoids circular reasoning and filler phrases such as "hmm" and "wait".

While increasing response length is a widely accepted way to improve reasoning, SDPO suggests that effective reasoning does not always have to be verbose. More efficient reasoning brings several benefits, including faster responses, lower inference costs, and more context space.

SDPO's central contribution lies in turning feedback on a model's own output into a token-level learning signal. This gives the policy a way to learn where an answer went wrong, not just whether the final answer succeeded.

## References

- Hübotter, J., et al. (2026). [*Reinforcement Learning via Self-Distillation*](https://arxiv.org/abs/2601.20802). arXiv:2601.20802.
- Lambert, N. (2026). [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/).
