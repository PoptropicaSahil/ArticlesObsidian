## [Introduction](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#introduction)

> **build our way up from raw bits to a production fine-tuning workflow**

![The journey of this post](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora-journey.svg)


All the code in this post lives in this repo: [here](https://github.com/debnsuma/fine-tuning-lora).

If you’d like a refresher on the memory side of training, the [static and dynamic memory sections](https://debnsuma.github.io/my-blog/posts/distributed-training-from-scratch/#bottlenecks-in-single-gpu-training) of my distributed training post cover it in detail 

> From *SumanDebnath_DistributedTraining* notes

> *parameters*, *gradients*, and *optimizer states*) are **static**; the *activations* are **dynamic**, depending on your batch and sequence length, and often become the main bottleneck for large-scale training.


**Full precision training - STATIC memory**

| Component | Precision | Size  |  Details |
| --- | --- | --- | --- |
| Model Parameters | BF32 (4 bytes) | 4⁢Ψ | Used for fwd/back passes |
| Parameter Gradients | BF32 (4 bytes) | 4⁢Ψ | For backpropagation |
| Optimizer States (Adam) | FP32 (4+4 bytes) | 8⁢Ψ | 1st and 2nd moment estimates |
| **Total Static Memory** |  | **16⁢Ψ** | **Absolute minimum for static storage** |

*DYNAMIC memory (my guess)*

|Component     |   Precision  | Size |
| --- | --- | --- |
| Activations   | BF32    |  4⁢Ψ |


**Mixed precision training - STATIC memory**


| Component | Precision | Size  |  Details |
| --- | --- |--- | --- |
| Model Parameters | BF16 (2 bytes) | 2⁢Ψ | For forward/backward passes |
| Parameter Gradients | BF16 (2 bytes) | 2⁢Ψ | Backpropagation |
| **Master Weights** (Adam) | FP32 (4 bytes) | 4⁢Ψ | For weight updates |
| Optimizer States (Adam) | FP32 (4+4 bytes) | 8⁢Ψ | 1st/2nd moment estimates |
| **Total Static Memory** |  | **16⁢Ψ** | **Unchanged overall** |

**DYNAMIC memory (my guess)**

|Component     |   Precision  | Size |
| --- | --- | --- |
| Activations   | BF16    |  2⁢Ψ |

> Mixed precision does NOT take less STATIC memory, but it is *faster* (activations use half the memory). *Remember to store master weights by the optimiser even during mixed precision*

## [Why Fine-Tune at All ?](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#why-fine-tune-at-all)


> The training loop is exactly next-token prediction with cross-entropy loss, just on our curated examples 

![Supervised fine-tuning: a chat-format training example flows through the LLM, and the loss is computed only on the assistant tokens](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/sft-overview.svg)

The _algorithm_ is nothing new. The problem, as always, is memory.

## [The Real Cost of Full Fine-Tuning](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-real-cost-of-full-fine-tuning)

For fine-tuning a modern 8B model, say [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). Here, Ψ is the number of parameters, and with standard **mixed-precision training + Adam**, every parameter costs us:

![](Pasted%20image%2020261017071108.png)


For `Qwen3-8B`, where Ψ=8×109:

|     |     |    |
| --- | --- |--- |
| **Model weights** | BF16 | 16 GB |
| **Gradients** | BF16 | 16 GB |
| **Optimizer states** | FP32 | 96 GB |
| **Activations** (with checkpointing) | BF16 | ~18 GB |
| **Total** |  | **~146 GB** |

![Memory comparison: full fine-tuning vs LoRA vs QLoRA for an 8B model](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/memory_8b_comparison.png)

Now, this is really bad. An 80 GB H100 cannot even hold the _optimizer states_, let alone the rest

> The **weights themselves are only ~11% of the bill**. The other ~89% is _training overhead_

That giant orange block in the chart above exists to update **all 8 billion parameters**. But do we _need_ to update all of them ? 

1. **Can we make the frozen part smaller ?** →`quantization`
2. **Can we make the trainable part smaller ?** → `LoRA`

> Added the Quantisation section in the file `Quantisation Sub Blog`

…

Enter LoRA.

# LoRA: The Low-Rank Idea

## [A Strange Discovery: Intrinsic Dimensionality(]https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#a-strange-discovery-intrinsic-dimensionality)

> *Downstream tasks are inherently low rank* YESSSIRR

In 2018, [Li et al.](https://arxiv.org/abs/1804.08838) tried something odd: instead of training a network in its full parameter space, they froze the network and only allowed updates inside a small ***random*** subspace of dimension 𝑑. Then they asked: how big does 𝑑 need to be before the network trains almost as well as normal ? The answer was *shockingly small*, they called it the **intrinsic dimension** of the task.

Two years later, [Aghajanyan et al.](https://arxiv.org/abs/2012.13255) ran the same experiment on _fine-tuning_ pretrained language models. They show that, for many natural language understanding (NLU) tasks, you can recover most of full fine-tuning performance while optimizing in a space with only hundreds to low thousands of degrees of freedom, even if the model has hundreds of millions of trainable weights. *They concluded that fine-tuning updates behave like they live in a small subspace, and that subspace tends to shrink as models scale.*

![Illustrative curve: percent of full fine-tuning performance vs number of trainable directions, crossing 90 percent at roughly 200 directions](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/intrinsic_dimension.png)

And here is the real data from the paper, two tasks, four models, accuracy against subspace dimension 𝑑, where each dotted horizontal line marks 90% of that model’s full fine-tuning performance:

![Figure 1 from Aghajanyan et al., 2020: evaluation accuracy on MRPC and QQP for BERT and RoBERTa models across subspace dimensions d, with dotted lines marking 90 percent of each full model's performance](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/intrinsic_dimension2.png)

_Figure 1 from [Aghajanyan et al., 2020](https://arxiv.org/abs/2012.13255)_

> Note: Look at where each curve crosses its dotted line in the figure above: RoBERTa-Large (red) gets there at a far smaller 𝑑 than BERT-Base (blue). *That’s the part that still amazes me*: **the bigger the pretrained model, the _lower_ its intrinsic dimension.** Pre-training doesn’t just teach the model facts, it builds such good general-purpose features that adapting to a new task becomes a _simpler_ problem, not a harder one. The 8B model isn’t 16x harder to fine-tune than a 0.5B model, in this sense, it’s _easier_.

## [The LoRA Hypothesis](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-lora-hypothesis)

Usual gradient descent 

$W \leftarrow W - \eta \frac{\partial J}{\partial W}$

![[debnsuma-github-io-my-blog-posts-lora-serverless-fine-tuning 2026-10-18 11.28.41.excalidraw]]

We can always write the final weights as the pretrained weights plus one update matrix:

$W' = W_0 + \Delta W$

> Now comes the shift in perspective. 

We don’t have to fold Δ⁢𝑊 _into_ 𝑊0 at all, matrix multiplication distributes over addition, so the pretrained weights can stay untouched while the *update rides alongside as its own matrix*:

$(W_0 + \Delta W)x = W_0x + \Delta Wx$

It isolates the thing we actually need to learn (Δ⁢𝑊) from the thing we already have (𝑊0). LoRA’s hypothesis: since the change we need is intrinsically low-dimensional, let’s constrain Δ⁢𝑊 to be a **low-rank** matrix, the product of two thin matrices:

 $W' = W_0 + BA,\quad B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k},\quad r \ll \min(d, k)$

Where:

- 𝑊0: the pretrained weights, completely **frozen** (no gradients, no optimizer states!)
- 𝐴: a thin matrix that projects the input _down_ into 𝑟 dimensions
- 𝐵: a thin matrix that projects back _up_ to the output dimension
- 𝑟: the **rank**, our knob for how expressive the update can be

![LoRA decomposition: frozen W plus low-rank B times A](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora.svg)

Count parameters for one typical attention projection in an 8B-class model, a 4096×4096 matrix:


|  What we train   |  Parameters   | Share   |
| --- | --- |--- |
| Full Δ⁢𝑊 (4096×4096) | 16,777,216 | 100% |
| LoRA, 𝑟=64 | 524,288 | 3.1% |
| LoRA, 𝑟=8 (4096×8+8×4096) | 65,536 | **0.39%** |
| LoRA, 𝑟=1 *(one column + one row!)* | 8,192 | 0.05% |

The parameter count went from $𝑑×𝑘$ to $𝑟×(𝑑+𝑘)$, it now grows _**linearly**_ with the matrix size instead of quadratically

For one of GPT-3 175B’s actual $12288×12288$ attention projections (~151M parameters), $𝑟=1$ means ~24.5K parameters, **~6100x smaller**

> the LoRA paper found that even rank 1 is surprisingly competitive

And remember from our memory math: gradients and optimizer states are proportional to _trainable_ parameters. Shrink the trainable set by 250x, and Adam’s 12-bytes-per-parameter tax shrinks by 250x too.

That giant orange block from our memory chart, the 112 GB of gradients and optimizer states, almost vanishes.

## [Scaling and Initialization](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#scaling-and-initialization)

Two small details in the LoRA recipe carry a lot of weight (pun intended). The actual forward pass is:

$h=W_0x + \dfrac{\alpha}{r} BAx$

> Observe that $r$ shows up (!!)

Where:

- $\dfrac{\alpha}{r}$: a fixed scaling factor; 𝛼 is a hyperparameter (a common heuristic is $𝛼=2⁢𝑟$)

> Dividing by $𝑟$ keeps the *update’s magnitude roughly stable* if we change the rank, so $𝛼$ and the learning rate don’t have to be re-tuned for every $𝑟$ (consistent with [Schulman et al., 2025](https://thinkingmachines.ai/blog/lora/), who find the optimal learning rate shifts less than 2x across ranks 4 to 512)

***MOST IMPORTANT***

And the initialization is sneaky-clever:

- 𝐴 starts as **small random Gaussian** noise
- 𝐵 starts as **exactly zero**

Which means at step 0, $𝐵⁢𝐴=0$, and the model is _bit-for-bit identical_ to the pretrained model. Fine-tuning starts from a perfect no-op and gradually grows a change. No cliff, no cold start, no risk of scrambling the model before the first gradient step.

Why not initialize _both_ matrices to zero ? Because then nothing would ever train
> One of the two must start non-zero, and LoRA picks 𝐴

## [Building LoRA from Scratch](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#building-lora-from-scratch)

(from [`02_lora_and_qlora.ipynb`](https://github.com/debnsuma/fine-tuning-lora/blob/main/01-into-lora-fine-tuning/02_lora_and_qlora.ipynb))

We’ll wrap a 768×768 linear layer with a rank-8 adapter and count what’s actually trainable:

```py
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=4, alpha=8):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)           # 1. freeze the pretrained weights
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.scaling = alpha / rank                      # 2. the alpha/r scaling factor

        # 3. A: small random init, B: zeros  =>  BA = 0 at step 0
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base_out = self.base(x)                          # 4. frozen path
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T   # 5. low-rank detour
        return base_out + self.scaling * lora_out        # 6. sum the two paths

base_layer = nn.Linear(768, 768)
lora_layer = LoRALinear(base_layer, rank=8, alpha=16)

total_params = sum(p.numel() for p in lora_layer.parameters())
trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

print("total parameters:", total_params)
print("trainable parameters:", trainable_params)
print("trainable share:", round(trainable_params / total_params * 100, 2), "percent")
```

**Output:**

```bash
total parameters: 602880
trainable parameters: 12288
trainable share: 2.04 percent
```

Check that the wrapped layer should behave identically to the plain frozen layer before any training happens.

```py
x = torch.randn(4, 768)
with torch.no_grad():
    difference = (lora_layer(x) - base_layer(x)).abs().max()

print("max difference before training:", difference.item())
```

**Output:**

```sh
max difference before training: 0.0
```

> Not approximately identical, _exactly_

## [Training Our Adapter](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#training-our-adapter)

Now let’s train it. We’ll set up a toy problem where we _know_ the answer: create a “target” layer whose weights differ from our base layer by a **rank-4** perturbation, and see if our rank-8 adapter (which has more than enough capacity) can learn that difference. Notice that the optimizer receives only 𝐴 and 𝐵, the base layer never sees it:

```py
torch.manual_seed(0)

# The task's ideal update really is low rank: a rank 4 change on top of the
# frozen weight, which the rank 8 adapter has more than enough capacity for.
true_rank = 4
low_rank_update = torch.randn(768, true_rank) @ torch.randn(true_rank, 768) * 0.1

target_layer = nn.Linear(768, 768)
with torch.no_grad():
    target_layer.weight.copy_(base_layer.weight + low_rank_update)
    target_layer.bias.copy_(base_layer.bias)

optimizer = torch.optim.Adam([lora_layer.lora_A, lora_layer.lora_B], lr=1e-2)

losses = []
for step in range(300):
    x = torch.randn(32, 768)
    target = target_layer(x).detach()

    prediction = lora_layer(x)
    loss = ((prediction - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
```

**Output:**

```
step 0, loss 23.9552
step 50, loss 8.6610
step 100, loss 1.4803
step 150, loss 0.3001
step 200, loss 0.0901
step 250, loss 0.0301
```

![Toy LoRA training loss falling three orders of magnitude](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora_toy_loss.png)

And on data the adapter has never seen:

```py
x_eval = torch.randn(256, 768)
with torch.no_grad():
    eval_loss = ((lora_layer(x_eval) - target_layer(x_eval)) ** 2).mean()

print("evaluation loss on unseen data:", eval_loss.item())
```

**Output:**

```sh
evaluation loss on unseen data: 0.01526715513318777
```

The loss falls by **three orders of magnitude**, and generalizes to unseen inputs, while 98% of the parameters never moved.

## [Core Principle: Singular Value Decomposition (SVD)](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#core-principle-singular-value-decomposition-svd)

Any matrix 𝑀 can be factored as:

$𝑀=𝑈⁢Σ⁢𝑉^𝑇$

Where:

- 𝑈 and 𝑉: orthogonal matrices, they reorient space but don’t stretch it
- Σ: a diagonal matrix whose entries, sorted largest to smallest, are the **singular values**

> Each singular values tells us how much the matrix stretches space along one particular direction. A matrix’s **rank** is exactly the count of non-zero singular values, the number of directions that survive.

`lora_B @ lora_A` is a 768×768 matrix, the same shape as a full weight update, but it was built from a (768,8) and an (8,768) matrix. So it cannot have rank higher than 8. [`torch.linalg.matrix_rank`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html) confirms this directly by counting singular values above a small threshold:

```py
with torch.no_grad():
    composite_update = lora_layer.lora_B @ lora_layer.lora_A

rank = torch.linalg.matrix_rank(composite_update)
singular_values = torch.linalg.svdvals(composite_update)   # used for the plot below

print("rank of the trained update:", rank.item())
```

**Output:**

```sh
rank of the trained update: 8
```

And plotting the singular values shows why: at most 8 of them can be meaningfully non-zero, everything else has nowhere to come from and gets crushed to numerical noise.

![Singular values of the trained update BA: first 12 of 768 on a log scale, only the first 8 are meaningfully non-zero](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora_svd_rank.png)

Watching this evolve _during_ training makes the point concrete:  exactly **four** singular values pull away from the rest, the true rank of the update this particular toy task needed. *The other four stay small (unused adapter capacity), and everything beyond the adapter’s rank is numerical noise.*

![Animated bar chart showing the singular value spectrum concentrating into 4 dominant values over training](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora_training.gif)

> **Tip: This is the magic of LoRA**

> > We didn’t approximate the update _after_ training (like classical SVD compression would). We ***constrained the search space before training***, and gradient descent found the best answer _inside the box_. The *intrinsic dimensionality* research is why the box is almost always big enough. (Downstream tasks are low rank)

## [Patching a Real Model](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#patching-a-real-model)

Our toy layer works. Scaling up to a real LLM is just a tree walk: find every module whose name we target, here the four attention projections, and swap in a `LoRALinear`. The interesting part is just this:

```py
def apply_lora(model, target_names, rank=8, alpha=16):
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))

apply_lora(model, target_names=["q_proj", "k_proj", "v_proj", "o_proj"], rank=8, alpha=16)
```

**Output:**

```sh
total parameters: 495114112
trainable parameters: 1081344
trainable share: 0.2184 percent
max logit difference after wrapping: 0.0
```


## [Merging: The Free Lunch at Inference](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#merging-the-free-lunch-at-inference)

After training we can **fold it back into the base weights** with a single addition:

$𝑊_{merged}=𝑊_0+\dfrac{𝛼}{𝑟⁢}𝐵⁢𝐴$

```py
with torch.no_grad():
    merged_weight = base_layer.weight + lora_layer.scaling * (lora_layer.lora_B @ lora_layer.lora_A)

merged_layer = nn.Linear(768, 768)
with torch.no_grad():
    merged_layer.weight.copy_(merged_weight)
    merged_layer.bias.copy_(base_layer.bias)

x = torch.randn(4, 768)
with torch.no_grad():
    difference = (merged_layer(x) - lora_layer(x)).abs().max()

print("max difference between merged and adapter form:", difference.item())
```

**Output:**

```sh
max difference between merged and adapter form: 1.049041748046875e-05
```

>  *zero extra inference latency* ofcourse. Adapter is tiny and portable. The LoRA paper reports that for GPT-3 175B, a fine-tuning checkpoint shrinks from **~350 GB to ~35 MB**, a **10,000x** reduction.


![Multi-tenant LoRA serving: one shared frozen base model plus one tiny adapter pair per user](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora-multitenant.svg)

One central frozen $𝑊_0$ serves everyone, and each customer owns only their tiny $(𝐵_𝑖,𝐴_𝑖)$ pair, typically tens to a couple hundred megabytes that cost almost nothing to store while idle.

## [Which Layers Should We Adapt ?](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#which-layers-should-we-adapt)

The original paper adapted only the attention query and value projections $(𝑊_𝑞, 𝑊_𝑣)$, and that recipe is still widely repeated. The evidence has moved on, and it’s worth being blunt about it:

- The [QLoRA paper](https://arxiv.org/abs/2305.14314) found adapting **all linear layers** was needed to match full fine-tuning.
- [Thinking Machines’ “LoRA Without Regret”](https://thinkingmachines.ai/blog/lora/) (2025) found **attention-only LoRA significantly underperforms**, even at matched parameter counts, while MLP-only performs about as well as MLP+attention.

Their training curves make the second point hard to argue with, on both a dense model and a sparse MoE, the attention-only run (red) sits clearly above everything else, while MLP-only is indistinguishable from adapting everything:

![Figure 4 from LoRA Without Regret: test NLL curves on Tulu3 for Llama-3.1-8B and Qwen3-30B-A3B, showing attention-only LoRA underperforming while MLP-only matches MLP plus attention](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora-adapt.png)

_Figure 4 from [Schulman et al., 2025](https://thinkingmachines.ai/blog/lora/)_

###### IMP
> Interestingly, a related lesson, that adapting _more matrices at low rank_ beats adapting _one matrix at high rank_, was hiding in the original paper all along. Here is a slice of its rank-sweep on GPT-3 (validation accuracy on WikiSQL, [Table 6 of Hu et al.](https://arxiv.org/abs/2106.09685)):

|Weight types adapted|𝑟 =1|𝑟 =8|𝑟 =64|
|---|---|---|---|
|𝑊𝑞 only|68.8|70.4|*70.0*|
|𝑊𝑞,𝑊𝑣|73.4|73.8|73.5|
|𝑊𝑞,𝑊𝑘,𝑊𝑣,𝑊𝑜|***74.1***|74.0|73.9|

Compare the corners. Spreading a small parameter budget across _more_ matrices wins over concentrating it in one expressive update.

###### **CRUICIAL**
And notice how little rank matters once the coverage is right: on the two multi-matrix rows, 𝑟=1 is within noise of 𝑟=64, while on the 𝑊𝑞-only row rank still visibly helps, exactly what you’d expect if _which matrices you adapt_, not rank, is the binding constraint.

###### **IMPORTANT CRITICAL INSIGHT**

> `q_proj + v_proj only` is the _historical_ recipe, not the _recommended_ one.
> 
> The modern default is simple: **adapt every linear layer** (in `peft`, `target_modules="all-linear"`). If you have to trim the budget, drop the attention adapters and keep the MLP ones, not the other way around, attention-only is the one configuration the evidence consistently argues against.

## [When LoRA Fails](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#when-lora-fails)

![Learning-forgetting tradeoff from Biderman et al.: LoRA runs cluster near the base model with less forgetting but less learning, full fine-tuning runs learn more but forget more](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/paper_biderman_intro.png)

_Figure from [Biderman et al., 2024](https://arxiv.org/abs/2405.09673)_

Read it as a tradeoff frontier: up means more learned (target domain), right means more retained (source domain). Full fine-tuning’s points climb high but drift left, LoRA’s cluster hugs the right edge near the base models but stays low.
> ?!

|Setting|What the paper found|
|---|---|
|**Instruction tuning** (~100K examples)|LoRA ≈ full fine-tuning. This is the regime most of us live in.|
|**Continued pre-training**(~billions of tokens)|LoRA **substantially underperforms**: full fine-tuning’s weight changes have 10–100x higher rank than any practical adapter can represent.|
|**Forgetting**|LoRA **forgets less** of the base model’s abilities, the constraint that limits learning also acts as a regularizer.|

> ***BEHAVIORS (a format, a style, a task) ARE LOW-RANK, KNOWLEDGE (a new language, a whole domain corpus) IS NOT***


# QLoRA: LoRA Meets Quantization

**[QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)**

## The Insight [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-insight)


- `Quantization` handles the FROZEN BASE: store it in **NF4**, dequantize each layer to BF16 on the fly for compute.
- `LoRA` handles the trainable part: **adapters** stay in **BF16**, small enough that their optimizer states are pocket change.

The only new engineering is the seam between them: gradients must flow _through_ the quantized frozen layers but only _into_ the adapters.

![Figure 1 from the QLoRA paper comparing full fine-tuning, LoRA, and QLoRA](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/paper_qlora_fig1.png)

_Figure 1 from [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)_

The result made single-GPU history: the QLoRA authors fine-tuned **LLaMA-65B on a single 48 GB GPU**, a job that needs **~780 GB** the traditional way.

> But there is no free lunch here either, we pay the **dequantize-on-the-fly compute tax** (typically 10–30% slower per step than BF16 LoRA), and we **accept the small NF4 rounding noise** in the frozen base.

## [From Scratch to `peft`: The Mapping](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#from-scratch-to-peft-the-mapping)

[`peft`](https://huggingface.co/docs/peft/en/index) (Parameter-Efficient Fine-Tuning):


|What we built by hand|Library equivalent|
|---|---|
|`quantize_absmax_int8`/ NF4 idea|`BitsAndBytesConfig(load_in_4bit=True, ...)`|
|`LoRALinear` + `apply_lora`|`peft.LoraConfig` + `get_peft_model`|
|Our toy `training loop`|`transformers.Trainer`|
|`W + scaling * (B @ A)` merge|`model.merge_and_unload()`|

Let’s fine-tune [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) on a slice of [OpenAssistant-Guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) (300 train / 30 eval examples, deliberately tiny so it runs anywhere).

## [Loading and Preparing](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#loading-and-preparing)

Step one, load the base in 4-bit, this is the _exact_`bnb_config` from the quantization section, reused verbatim. Step two, one new helper:

```py
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 1. store weights in 4-bit
    bnb_4bit_quant_type="nf4",              # 2. use NF4, not FP4 (default is fp4!)
    bnb_4bit_use_double_quant=True,         # 3. quantize the quantization constants
    bnb_4bit_compute_dtype=torch.bfloat16,  # 4. dequantize to BF16 for the actual math
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,          # dtype for the layers that do NOT get quantized
    device_map={"": 0},
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable parameters right after prepare_model_for_kbit_training:", trainable_before)
```

**Output:**

```sh
trainable parameters right after prepare_model_for_kbit_training: 0
```

This helper does the unglamorous safety work of training on a quantized base: freezes every quantized weight, casts the few remaining non-quantized bits (like layer norms) to FP32 for numerical stability, and enables gradient checkpointing, the activation-recomputation trick [we met in the distributed training post](https://debnsuma.github.io/my-blog/posts/distributed-training-from-scratch/#solution-1-activation-recomputation).

Note that at this point **zero** parameters are trainable, the model is fully frozen, waiting for adapters.

## [Attaching the Adapter](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#attaching-the-adapter)


```py
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                                                    # rank (same as our from-scratch run)
    lora_alpha=16,                                          # alpha = 2r heuristic
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,                                      # dropout on the adapter path only
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**Output:**

```sh
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
```

Look at that number. **1,081,344**, _exactly_ the count we computed with our hand-rolled `apply_lora` a few sections ago, down to the last parameter.

> Crazy only

`peft` is doing precisely what we did, walking the module tree and wrapping the four attention projections with rank-8 adapters. *This is my favourite moment in the whole post: the library stopped being a black box.*

###### Note

> Where does `target_modules` come from ?
> If you don’t pass `target_modules`, `peft` looks up a per-architecture default:
```py 
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
print(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["qwen2"]) # ['q_proj', 'v_proj']
```

> That’s the _conservative 2021 recipe_ (see [Which Layers Should We Adapt ?](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/#which-layers-should-we-adapt)). For real work, pass `target_modules="all-linear"` and let `peft` wrap every linear layer, including the MLP projections, which is where the capacity that matters lives.

## [Training](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#training)


```py 
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs/qlora-run",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # effective batch = 8
    num_train_epochs=1,
    learning_rate=2e-4,              # ~10x a full fine-tuning LR
    logging_steps=5,
    eval_strategy="epoch",
    bf16=True,                       # compute dtype
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
)

result = trainer.train()
print(f"final training loss: {result.training_loss:.4f}")
```

**Output:**

```sh
final training loss: 5.8718
```


> 300 examples for one epoch is a _demonstration_, not a fine-tune

## [Merging] (https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#merging)


```py 
merged_model = model.merge_and_unload()   # W' = W0 + (alpha/r) * BA, adapters folded in
```

**Output:**

```sh 
UserWarning: Merge lora module to 4-bit linear may get different generations due to rounding errors.
```

> *That warning is worth reading, it’s our whole post in one sentence :)*

###### **IMPORTANT**
> Merging adds BF16 adapter weights into an NF4-quantized base, so the result must be re-quantized, and the rounding noise can slightly shift generations. In practice, for deployment you merge into a _full-precision_ copy of the base weights instead, or skip merging entirely and serve base + adapter separately.

## [The Scoreboard](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-scoreboard)

- **LoRA** wipes out the **Grads + Optimizer** column
- **Quantization** then shrinks the **Weights** column

And neither touches the other’s work:

|Strategy|Weights|Grads + Optimizer|Approx. total (8B)|Fits on|
|---|---|---|---|---|
|**Full fine-tuning**(BF16 + Adam)|16 GB|112 GB|**~146 GB**|multi-GPU only|
|**+ LoRA**(freeze base, r=16)|16 GB|~1 GB|**~25 GB**|one A100 40GB|
|**+ QLoRA**(NF4 base, r=16)|~4.5 GB|~1 GB|**~11 GB**|one consumer GPU!|

From “we need a cluster and FSDP” to “my desktop GPU can do it”, with two ideas we built from scratch in an afternoon.

# Introduction to Serverless Fine-Tuning with Crusoe

> Have omitted this section for now, perhaps because understanding the Crusoe API isn’t the objective


# References

- **Suman’s blog!**