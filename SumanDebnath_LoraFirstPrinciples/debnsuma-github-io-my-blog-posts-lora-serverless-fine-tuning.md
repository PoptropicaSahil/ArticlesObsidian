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

$W \leftarrow W - \eta \frac{\partial J}{\partial W}$

That subtracted term is a weight update. Accumulate every step of a fine-tuning run, and we can always write the final weights as the pretrained weights plus one update matrix:

$W' = W_0 + \Delta W$


Now comes the shift in perspective. We don’t have to fold Δ⁢𝑊 _into_ 𝑊0 at all, matrix multiplication distributes over addition, so the pretrained weights can stay untouched while the update rides alongside as its own matrix:


![](Pasted%20image%2020261018101524.png)

By itself, this rearrangement changes nothing, Δ⁢𝑊 still has the full 𝑑×𝑘 shape, and a second matrix as large as the first is no cheaper to train or store. But it isolates the thing we actually need to learn (Δ⁢𝑊) from the thing we already have (𝑊0), and that’s the opening the intrinsic dimensionality result walks through. Full fine-tuning lets Δ⁢𝑊 be _anything_, a full 𝑑×𝑘 matrix. LoRA’s hypothesis: since the change we need is intrinsically low-dimensional, let’s constrain Δ⁢𝑊 to be a **low-rank** matrix, the product of two thin matrices:

𝑊′=𝑊0+𝐵⁢𝐴,𝐵∈ℝ𝑑×𝑟,𝐴∈ℝ𝑟×𝑘,𝑟≪min⁡(𝑑,𝑘)

Where:

- 𝑊0: the pretrained weights, completely **frozen** (no gradients, no optimizer states!)
- 𝐴: a thin matrix that projects the input _down_ into 𝑟 dimensions
- 𝐵: a thin matrix that projects back _up_ to the output dimension
- 𝑟: the **rank**, our knob for how expressive the update can be

![LoRA decomposition: frozen W plus low-rank B times A](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora.svg)

Why is this such a big deal ? Let’s count parameters for one typical attention projection in an 8B-class model, a 4096×4096 matrix:

| Full Δ⁢𝑊 (4096×4096) | 16,777,216 | 100% |
| LoRA, 𝑟=64 | 524,288 | 3.1% |
| LoRA, 𝑟=8 (4096×8+8×4096) | 65,536 | **0.39%** |
| LoRA, 𝑟=1 (one column + one row!) | 8,192 | 0.05% |

The parameter count went from 𝑑×𝑘 to 𝑟×(𝑑+𝑘), it now grows _linearly_ with the matrix size instead of quadratically. And it’s worth pausing on that last row: at 𝑟=1, 𝐴 is literally a single row and 𝐵 a single column.

For one of GPT-3 175B’s actual 12288×12288 attention projections (~151M parameters), 𝑟=1 means ~24.5K parameters, **~6100x smaller**, and as we’ll see in a moment, the LoRA paper found that even rank 1 is surprisingly competitive. And remember from our memory math: gradients and optimizer states are proportional to _trainable_ parameters. Shrink the trainable set by 250x, and Adam’s 12-bytes-per-parameter tax shrinks by 250x too.

That giant orange block from our memory chart, the 112 GB of gradients and optimizer states, almost vanishes.

## Scaling and Initialization [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#scaling-and-initialization)

Two small details in the LoRA recipe carry a lot of weight (pun intended). The actual forward pass is:

ℎ=𝑊0⁢𝑥+𝛼𝑟𝐵⁢𝐴⁢𝑥

Where:

- 𝛼𝑟: a fixed scaling factor; 𝛼 is a hyperparameter (a common heuristic is 𝛼=2⁢𝑟)
- Dividing by 𝑟 keeps the update’s magnitude roughly stable if we change the rank, so 𝛼 and the learning rate don’t have to be re-tuned for every 𝑟 (consistent with [Schulman et al., 2025](https://thinkingmachines.ai/blog/lora/), who find the optimal learning rate shifts less than 2x across ranks 4 to 512)

And the initialization is sneaky-clever:

- 𝐴 starts as **small random Gaussian** noise
- 𝐵 starts as **exactly zero**

Which means at step 0, 𝐵⁢𝐴=0, and the model is _bit-for-bit identical_ to the pretrained model. Fine-tuning starts from a perfect no-op and gradually grows a change. No cliff, no cold start, no risk of scrambling the model before the first gradient step.

Why not initialize _both_ matrices to zero ? Because then nothing would ever train: the gradient of 𝐴 flows through 𝐵, so a zero 𝐵 means a zero gradient for 𝐴, forever. One of the two must start non-zero, and LoRA picks 𝐴. It doesn’t matter that 𝐴 is full of random values, because 𝐵 is zero, so nothing 𝐴 produces reaches the output anyway.

## Building LoRA from Scratch [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#building-lora-from-scratch)

Enough theory, let’s build it. Here is a complete LoRA layer in plain PyTorch, no `peft`, no tricks (from [`02_lora_and_qlora.ipynb`](https://github.com/debnsuma/fine-tuning-lora/blob/main/01-into-lora-fine-tuning/02_lora_and_qlora.ipynb)).

We’ll wrap a 768×768 linear layer with a rank-8 adapter and count what’s actually trainable:

```
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

```
total parameters: 602880
trainable parameters: 12288
trainable share: 2.04 percent
```

That’s the whole thing. And since `lora_B` starts at zero, the wrapped layer should behave identically to the plain frozen layer before any training happens. Check that directly:

```
x = torch.randn(4, 768)
with torch.no_grad():
    difference = (lora_layer(x) - base_layer(x)).abs().max()

print("max difference before training:", difference.item())
```

**Output:**

```
max difference before training: 0.0
```

Two things worth noticing here:

- **We didn’t _replace_ the layer.** We added a parallel low-rank detour around it, the frozen path still does all the heavy lifting.
- That `max difference before training: 0.0` is the zero-init guarantee we just discussed, our wrapped layer is _provably_ identical to the original before training starts. Not approximately identical, _exactly_: 𝐵 is true zeros, so the detour contributes nothing at all.

## Training Our Adapter [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#training-our-adapter)

Now let’s train it. We’ll set up a toy problem where we _know_ the answer: create a “target” layer whose weights differ from our base layer by a **rank-4** perturbation, and see if our rank-8 adapter (which has more than enough capacity) can learn that difference. Notice that the optimizer receives only 𝐴 and 𝐵, the base layer never sees it:

```
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

```
x_eval = torch.randn(256, 768)
with torch.no_grad():
    eval_loss = ((lora_layer(x_eval) - target_layer(x_eval)) ** 2).mean()

print("evaluation loss on unseen data:", eval_loss.item())
```

**Output:**

```
evaluation loss on unseen data: 0.01526715513318777
```

The loss falls by **three orders of magnitude**, and generalizes to unseen inputs, while 98% of the parameters never moved.

## Core Principle: Singular Value Decomposition (SVD) [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#core-principle-singular-value-decomposition-svd)

How do we _know_ the learned update is really low-rank ? The tool for answering that is the **Singular Value Decomposition (SVD)**, and it’s worth a short detour, because it is the mathematical backbone of everything low-rank in this post. Any matrix 𝑀 can be factored as:

𝑀=𝑈⁢Σ⁢𝑉𝑇

Where:

- 𝑈 and 𝑉: orthogonal matrices, they reorient space but don’t stretch it
- Σ: a diagonal matrix whose entries, sorted largest to smallest, are the **singular values**

The singular values have a concrete geometric meaning: each one tells us how much the matrix stretches space along one particular direction. A singular value of 40 means some direction gets stretched 40x when we multiply by 𝑀. A singular value of 0 means inputs along that direction get crushed to nothing, they contribute no information to the output. A matrix’s **rank** is exactly the count of non-zero singular values, the number of directions that survive.

Now, `lora_B @ lora_A` is a 768×768 matrix, the same shape as a full weight update, but it was built from a (768,8) and an (8,768) matrix. Multiplying through two skinny matrices like that can only ever produce 8 independent directions, no matter how large the resulting matrix looks, so it cannot have rank higher than 8. [`torch.linalg.matrix_rank`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html) confirms this directly by counting singular values above a small threshold:

```
with torch.no_grad():
    composite_update = lora_layer.lora_B @ lora_layer.lora_A

rank = torch.linalg.matrix_rank(composite_update)
singular_values = torch.linalg.svdvals(composite_update)   # used for the plot below

print("rank of the trained update:", rank.item())
```

**Output:**

```
rank of the trained update: 8
```

And plotting the singular values shows why: at most 8 of them can be meaningfully non-zero, everything else has nowhere to come from and gets crushed to numerical noise.

![Singular values of the trained update BA: first 12 of 768 on a log scale, only the first 8 are meaningfully non-zero](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora_svd_rank.png)

Watching this evolve _during_ training makes the point concrete: the spectrum starts flat at zero (nothing learned yet) and, step by step, exactly **four** singular values pull away from the rest, the true rank of the update this particular toy task needed. The other four stay small (unused adapter capacity), and everything beyond the adapter’s rank is numerical noise.

![Animated bar chart showing the singular value spectrum concentrating into 4 dominant values over training](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora_training.gif)

TipThis is the magic of LoRA

We didn’t approximate the update _after_ training (like classical SVD compression would). We **constrained the search space before training**, and gradient descent found the best answer _inside the box_. The intrinsic dimensionality research is why the box is almost always big enough.

## Patching a Real Model [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#patching-a-real-model)

Our toy layer works. Scaling up to a real LLM is just a tree walk: find every module whose name we target, here the four attention projections, and swap in a `LoRALinear`. The full cell (loading [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) in BF16, freezing every parameter, and verifying the outputs) is in the [notebook](https://github.com/debnsuma/fine-tuning-lora/blob/main/01-into-lora-fine-tuning/02_lora_and_qlora.ipynb), the interesting part is just this:

```
def apply_lora(model, target_names, rank=8, alpha=16):
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))

apply_lora(model, target_names=["q_proj", "k_proj", "v_proj", "o_proj"], rank=8, alpha=16)
```

**Output:**

```
total parameters: 495114112
trainable parameters: 1081344
trainable share: 0.2184 percent
max logit difference after wrapping: 0.0
```

Read that `trainable parameters` line again: on a real 0.5B model, we are training **0.22% of the parameters**. And the last line is our zero-init guarantee scaled up, wrapping every attention layer in the model changed its outputs by exactly nothing.

## Merging: The Free Lunch at Inference [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#merging-the-free-lunch-at-inference)

One more property makes LoRA special among adapter methods. Because the detour is just an additive matrix, after training we can **fold it back into the base weights** with a single addition:

𝑊merged=𝑊0+𝛼𝑟⁢𝐵⁢𝐴

```
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

```
max difference between merged and adapter form: 1.049041748046875e-05
```

The merged model is a plain `nn.Linear` again, same architecture, same speed, **zero extra inference latency**. Compare that with adapter methods that insert new layers (permanent inference tax) and you see why LoRA won.

And if we _don’t_ merge, we get a different superpower: the adapter is tiny and portable. The LoRA paper reports that for GPT-3 175B, a fine-tuning checkpoint shrinks from **~350 GB to ~35 MB**, a **10,000x** reduction.

To appreciate why this number changed the industry, put yourself in the shoes of anyone serving fine-tuned models for many teams or customers (we’ll be on the customer side of exactly this arrangement in the finale). Without LoRA, every fine-tune is a complete private copy of the model, hundreds of gigabytes that must sit on disk, and ideally on warm GPUs, whether or not that fine-tune ever receives traffic. A thousand fine-tunes, a thousand copies. With LoRA, the picture inverts:

![Multi-tenant LoRA serving: one shared frozen base model plus one tiny adapter pair per user](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora-multitenant.svg)

One central frozen 𝑊0 serves everyone, and each customer owns only their tiny (𝐵𝑖,𝐴𝑖) pair, typically tens to a couple hundred megabytes that cost almost nothing to store while idle.

## Which Layers Should We Adapt ? [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#which-layers-should-we-adapt)

The original paper adapted only the attention query and value projections (𝑊𝑞, 𝑊𝑣), and that recipe is still widely repeated. The evidence has moved on, and it’s worth being blunt about it:

- The [QLoRA paper](https://arxiv.org/abs/2305.14314) found adapting **all linear layers** was needed to match full fine-tuning.
- [Thinking Machines’ “LoRA Without Regret”](https://thinkingmachines.ai/blog/lora/) (2025) found **attention-only LoRA significantly underperforms**, even at matched parameter counts, while MLP-only performs about as well as MLP+attention.

Their training curves make the second point hard to argue with, on both a dense model and a sparse MoE, the attention-only run (red) sits clearly above everything else, while MLP-only is indistinguishable from adapting everything:

![Figure 4 from LoRA Without Regret: test NLL curves on Tulu3 for Llama-3.1-8B and Qwen3-30B-A3B, showing attention-only LoRA underperforming while MLP-only matches MLP plus attention](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/lora-adapt.png)

_Figure 4 from [Schulman et al., 2025](https://thinkingmachines.ai/blog/lora/)_

Interestingly, a related lesson, that adapting _more matrices at low rank_ beats adapting _one matrix at high rank_, was hiding in the original paper all along. Here is a slice of its rank-sweep on GPT-3 (validation accuracy on WikiSQL, [Table 6 of Hu et al.](https://arxiv.org/abs/2106.09685)):

| 𝑊𝑞 only | 68.8 | 70.4 | 70.0 |
| 𝑊𝑞,𝑊𝑣 | 73.4 | 73.8 | 73.5 |
| 𝑊𝑞,𝑊𝑘,𝑊𝑣,𝑊𝑜 | **74.1** | 74.0 | 73.9 |

Compare the corners: adapting **all four attention matrices at rank 1** (74.1) beats adapting **𝑊𝑞 alone at rank 64** (70.0) by a wide margin. Spreading a small parameter budget across _more_ matrices wins over concentrating it in one expressive update.

And notice how little rank matters once the coverage is right: on the two multi-matrix rows, 𝑟=1 is within noise of 𝑟=64, while on the 𝑊𝑞-only row rank still visibly helps, exactly what you’d expect if _which matrices you adapt_, not rank, is the binding constraint.

ImportantCritical Insight

`q_proj + v_proj only` is the _historical_ recipe, not the _recommended_ one.

The modern default is simple: **adapt every linear layer** (in `peft`, `target_modules="all-linear"`). If you have to trim the budget, drop the attention adapters and keep the MLP ones, not the other way around, attention-only is the one configuration the evidence consistently argues against.

## When LoRA Fails [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#when-lora-fails)

There is no free lunch, so let’s be honest about where the low-rank box is too small. [Biderman et al., 2024, “LoRA Learns Less and Forgets Less”](https://arxiv.org/abs/2405.09673) ran a careful head-to-head:

![Learning-forgetting tradeoff from Biderman et al.: LoRA runs cluster near the base model with less forgetting but less learning, full fine-tuning runs learn more but forget more](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/paper_biderman_intro.png)

_Figure from [Biderman et al., 2024](https://arxiv.org/abs/2405.09673)_

Read it as a tradeoff frontier: up means more learned (target domain), right means more retained (source domain). Full fine-tuning’s points climb high but drift left, LoRA’s cluster hugs the right edge near the base models but stays low.

The picture that emerges:

| **Instruction tuning** (~100K examples) | LoRA ≈ full fine-tuning. This is the regime most of us live in. |
| **Continued pre-training** (~billions of tokens) | LoRA **substantially underperforms**: full fine-tuning’s weight changes have 10–100x higher rank than any practical adapter can represent. |
| **Forgetting** | LoRA **forgets less** of the base model’s abilities, the constraint that limits learning also acts as a regularizer. |

The mental model I like: LoRA is perfect for teaching a behavior (a format, a style, a task), and wrong for teaching a body of knowledge (a new language, a whole domain corpus). Behaviors are low-rank, knowledge is not.

# QLoRA: LoRA Meets Quantization

We now have two independent tricks:

- **Quantization** shrinks the _frozen_ parameters (fewer bits each)
- **LoRA** shrinks the _trainable_ parameters (fewer of them)

Notice they attack different parts of the memory bill, which means we can stack them. That stack has a name: **[QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)**.

## The Insight [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-insight)

Each lever keeps its own job.

- `Quantization` handles the frozen base: store it in **NF4**, dequantize each layer to BF16 on the fly for compute.
- `LoRA` handles the trainable part: adapters stay in **BF16**, small enough that their optimizer states are pocket change.

The only new engineering is the seam between them: gradients must flow _through_ the quantized frozen layers but only _into_ the adapters.

![Figure 1 from the QLoRA paper comparing full fine-tuning, LoRA, and QLoRA](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/paper_qlora_fig1.png)

_Figure 1 from [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)_

The result made single-GPU history: the QLoRA authors fine-tuned **LLaMA-65B on a single 48 GB GPU**, a job that needs **~780 GB** the traditional way.

But there is no free lunch here either, we pay the dequantize-on-the-fly compute tax (typically 10–30% slower per step than BF16 LoRA), and we accept the small NF4 rounding noise in the frozen base.

## From Scratch to `peft`: The Mapping [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#from-scratch-to-peft-the-mapping)

Everything we hand-built has a direct equivalent in the Hugging Face ecosystem, and because we built the internals ourselves, none of these are magic anymore. The adapter side lives in a library called [`peft`](https://huggingface.co/docs/peft/en/index) (Parameter-Efficient Fine-Tuning):

| `quantize_absmax_int8` / NF4 idea | `BitsAndBytesConfig(load_in_4bit=True, ...)` |
| `LoRALinear` \+ `apply_lora` | `peft.LoraConfig` \+ `get_peft_model` |
| Our toy `training loop` | `transformers.Trainer` |
| `W + scaling * (B @ A)` merge | `model.merge_and_unload()` |

Time to see the mapping in action. Let’s fine-tune [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) on a slice of [OpenAssistant-Guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) (300 train / 30 eval examples, deliberately tiny so it runs anywhere).

## Loading and Preparing [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#loading-and-preparing)

Step one, load the base in 4-bit, this is the _exact_`bnb_config` from the quantization section, reused verbatim. Step two, one new helper:

```
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

```
trainable parameters right after prepare_model_for_kbit_training: 0
```

This helper does the unglamorous safety work of training on a quantized base: freezes every quantized weight, casts the few remaining non-quantized bits (like layer norms) to FP32 for numerical stability, and enables gradient checkpointing, the activation-recomputation trick [we met in the distributed training post](https://debnsuma.github.io/my-blog/posts/distributed-training-from-scratch/#solution-1-activation-recomputation).

Note that at this point **zero** parameters are trainable, the model is fully frozen, waiting for adapters.

## Attaching the Adapter [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#attaching-the-adapter)

The model is frozen and waiting, so let’s give it its adapters:

```
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

```
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
```

Look at that number. **1,081,344**, _exactly_ the count we computed with our hand-rolled `apply_lora` a few sections ago, down to the last parameter.

`peft` is doing precisely what we did, walking the module tree and wrapping the four attention projections with rank-8 adapters. This is my favorite moment in the whole post: the library stopped being a black box.

NoteWhere does `target_modules` come from ?

If you don’t pass `target_modules`, `peft` looks up a per-architecture default:

```
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

print(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["qwen2"])
```

**Output:**

```
['q_proj', 'v_proj']
```

That’s the _conservative 2021 recipe_ (see [Which Layers Should We Adapt ?](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/#which-layers-should-we-adapt)). For real work, pass `target_modules="all-linear"` and let `peft` wrap every linear layer, including the MLP projections, which is where the capacity that matters lives.

## Training [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#training)

The training loop is the standard `Trainer`, and every argument in it is a concept we’ve already earned:

```
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

```
final training loss: 5.8718
```

NoteAn honest caveat about this run

300 examples for one epoch is a _demonstration_, not a fine-tune. If you generate from this model before and after, the outputs are nearly identical.

Loss went down, behavior barely moved. Teaching a model something real takes thousands of examples, which is exactly what we’ll do in the final project, where the same recipe, at 2,500 examples, produces a model that beats a 70B on its task.

## Merging [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#merging)

And the final step, the same algebra as our from-scratch merge, one method call:

```
merged_model = model.merge_and_unload()   # W' = W0 + (alpha/r) * BA, adapters folded in
```

**Output:**

```
UserWarning: Merge lora module to 4-bit linear may get different generations due to rounding errors.
```

That warning is worth reading, it’s our whole post in one sentence :)

Merging adds BF16 adapter weights into an NF4-quantized base, so the result must be re-quantized, and the rounding noise can slightly shift generations. In practice, for deployment you merge into a _full-precision_ copy of the base weights instead, or skip merging entirely and serve base + adapter separately.

## The Scoreboard [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-scoreboard)

Time to close the loop we opened with the memory math. Read the table row by row and you can watch each lever land:

- **LoRA** wipes out the **Grads + Optimizer** column
- **Quantization** then shrinks the **Weights** column

And neither touches the other’s work:

| **Full fine-tuning** (BF16 + Adam) | 16 GB | 112 GB | **~146 GB** | multi-GPU only |
| **\+ LoRA** (freeze base, r=16) | 16 GB | ~1 GB | **~25 GB** | one A100 40GB |
| **\+ QLoRA** (NF4 base, r=16) | ~4.5 GB | ~1 GB | **~11 GB** | one consumer GPU! |

From “we need a cluster and FSDP” to “my desktop GPU can do it”, with two ideas we built from scratch in an afternoon.

# Introduction to Serverless Fine-Tuning with Crusoe

Now that we’ve learned the mechanics of LoRA and QLoRA and how they work under the hood, it’s time to fine-tune a model that solves a real-world problem.

In production, though, we may not want to manage all of that ourselves, the GPUs, the training run, the serving stack. It would be great to have a managed service handle the infrastructure so we can focus on the actual problem.

That’s where [Crusoe’s serverless fine-tuning](https://www.crusoe.ai/cloud/serverless-fine-tuning) comes in: we can fine-tune open models for our domain or task without provisioning a single GPU, and once we’re happy with the training results, deploy the model to a dedicated inference endpoint with [self-serve deployments](https://www.crusoe.ai/resources/blog/self-serve-deployments-dedicated-inference).

I recently joined Crusoe and worked on the serverless fine-tuning service as we built it, so I thought it would be great to share my experience and walk you through how to use it.

# Project: PII Redaction at Scale with Serverless Fine-Tuning

Let’s now solve a real problem, with a real model, the way we’d do it in production. At a high level, this is what we’ll do:

1. **Build a training set** from a public Hugging Face dataset
2. **Fine-tune Qwen3-8B with LoRA** on Crusoe Serverless Fine-Tuning
3. **Deploy the checkpoint** as a dedicated endpoint
4. **Measure it** against a general-purpose 70B model on entity-level F1

**The problem:** financial documents are full of PII, names, account numbers, addresses, SWIFT codes. We want a model that takes any document and returns strict JSON: the redacted text plus a list of every entity it found.

This is a _behavior_ task (exact format, every time), which our mental model says is **exactly what LoRA is for**.

**The model:** [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), the 8B model whose memory bill we computed at the start of this post.

![End-to-end PII redaction fine-tuning pipeline](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/pii-redaction-pipeline.svg)

## The Dataset [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-dataset)

We’ll use [gretelai’s synthetic\_pii\_finance\_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) dataset from Hugging Face: financial documents with labeled PII spans, fully synthetic, so no actual PII touches our training run.

The preparation script ( [`prepare_dataset.py`](https://github.com/debnsuma/fine-tuning-lora/blob/main/02-serverless-fine-tuning-pii-redaction/data/prepare_dataset.py)) converts each document into a chat-format training example. Two functions do the real work:

```
def redact(text, spans):
    # Replace right to left so earlier span offsets stay valid.
    for s in sorted(spans, key=lambda s: s["start"], reverse=True):
        text = text[: s["start"]] + f"[{s['label'].upper()}]" + text[s["end"] :]
    return text

def to_example(row):
    text = row["generated_text"]
    spans = json.loads(row["pii_spans"])
    target = {
        "redacted_text": redact(text, spans),
        "entities": [{"type": s["label"], "value": text[s["start"] : s["end"]]}\
                     for s in sorted(spans, key=lambda s: s["start"])],
    }
    return {"messages": [\
        {"role": "system", "content": SYSTEM_PROMPT},\
        {"role": "user", "content": text},\
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},\
    ]}
```

After filtering for quality we get **2,500 train / 150 validation / 100 held-out test** examples covering **29 PII types**.

**Output:**

```
9705 of 55940 rows pass the filters (English, quality >= 80, <= 1200 chars, >= 2 spans)
train.jsonl: 2500 examples
validation.jsonl: 150 examples
test.jsonl: 100 examples

29 PII types in the selection. Most common: name (4409), date (4282), street_address (2543), company (2509), email (955), time (650), phone_number (533), bban (161), customer_id (159), swift_bic_code (158)
```

TipValidate before you upload

Every training example ends with the **assistant** turn, because that’s where the loss is computed, and every assistant message must parse as exactly our target JSON schema. A tiny local `validate_dataset()` pass (valid JSON per line, correct roles, parseable targets) before uploading has saved me from more failed jobs than any other habit. The [repo’s `utils.py`](https://github.com/debnsuma/fine-tuning-lora/blob/main/02-serverless-fine-tuning-pii-redaction/utils.py) has the full checker.

The **system prompt baked into the training data must match the one you send at inference time, verbatim**. The model learns the _pairing_ of prompt and behavior.

```
import utils
from pathlib import Path

DATA_DIR = Path("data")
for path in (DATA_DIR / "train.jsonl", DATA_DIR / "validation.jsonl", DATA_DIR / "test.jsonl"):
    utils.validate_dataset(path)
```

**Output:**

```
data/train.jsonl: 2500 rows, all checks passed (~1,580,989 tokens, chars/4 heuristic)
data/validation.jsonl: 150 rows, all checks passed (~92,815 tokens, chars/4 heuristic)
data/test.jsonl: 100 rows, all checks passed (~62,292 tokens, chars/4 heuristic)
```

## Setup [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#setup)

The platform is fully compatible with the OpenAI API, so the client is the `openai` SDK pointed at two base URLs:

- _Control plane_: files, jobs, model registry
- _Data plane_: chat completions

```
import os, httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
CRUSOE_API_KEY = os.environ["CRUSOE_API_KEY"]

# Control plane
INTELLIGENCE_BASE_URL = "https://api.intelligence.crusoecloud.com/v1"
# Data plane
INFERENCE_BASE_URL = "https://api.inference.crusoecloud.com/v1"

client = OpenAI(api_key=CRUSOE_API_KEY, base_url=INTELLIGENCE_BASE_URL)
```

Which models can we fine-tune ? Let’s ask the registry, using the `models.list()` API:

```
models = client.models.list().data
fine_tunable = [m for m in models if getattr(m, "fine_tuning_available", False)]

print(f"{len(fine_tunable)} fine-tunable base models available:\n")
print(f"{'Model ID':<52}  HuggingFace name")
print("-" * 92)
for m in sorted(fine_tunable, key=lambda m: m.model_name.lower()):
    print(f"{m.id:<52}  {m.model_name}")
```

**Output:**

```
13 fine-tunable base models available:

Model ID                                              HuggingFace name
--------------------------------------------------------------------------------------------
ftmodel-54a373c9-4007-406d-887b-e8a888dccb78          crusoeai/DeepSeek-V4-Flash
model-crusoeai-gpt-oss-120b-bf16-359a43a3             crusoeai/gpt-oss-120b-BF16
model-crusoeai-gpt-oss-20b-bf16-27df53a7              crusoeai/gpt-oss-20b-BF16
model-google-gemma-4-31b-it-b8c649b7                  google/gemma-4-31B-it
model-meta-llama-llama-3-1-8b-instruct-2e441db2       meta-llama/Llama-3.1-8B-Instruct
model-meta-llama-llama-3-3-70b-instruct-49dda8b3      meta-llama/Llama-3.3-70B-Instruct
ftmodel-4c2da1f5-bcb8-4c5c-88cc-9c839a7484f9          nvidia/GLM-5.2-NVFP4
model-qwen-qwen3-235b-a22b-instruct-2507-1b577be0     Qwen/Qwen3-235B-A22B-Instruct-2507
model-qwen-qwen3-8b-2662a271                          Qwen/Qwen3-8B
model-qwen-qwen3-5-2b-9ffa090d                        Qwen/Qwen3.5-2B
ftmodel-d8cb63e4-fe11-44e1-b4d8-628c26a0c2e2          Qwen/Qwen3.5-4B
model-qwen-qwen3-5-9b-2ce94bd8                        Qwen/Qwen3.5-9B
ftmodel-bb55f628-5e73-4541-a0ae-85c5163222fd          Qwen/Qwen3.6-35B-A3B
```

As new models are added this list will grow, so `models.list()` is always the source of truth. Let’s pick our base model, `Qwen3-8B`:

```
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
BASE_MODEL_ID = next(m.id for m in fine_tunable if m.model_name == BASE_MODEL_NAME)
print(f"Base model    :    {BASE_MODEL_NAME}")
print(f"Base model ID :    {BASE_MODEL_ID}")
```

**Output:**

```
Base model    :    Qwen/Qwen3-8B
Base model ID :    model-qwen-qwen3-8b-2662a271
```

## Upload and Launch [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#upload-and-launch)

Next, upload the two dataset files:

```
train_file = client.files.create(file=open("data/train.jsonl", "rb"), purpose="fine-tune")
val_file   = client.files.create(file=open("data/validation.jsonl", "rb"), purpose="fine-tune")
```

We can now set the hyperparameters and launch the fine-tuning job.

```
HYPERPARAMETERS = {
    "n_epochs": 2,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
}

job = client.fine_tuning.jobs.create(
    model=BASE_MODEL_ID,
    training_file=train_file.id,
    validation_file=val_file.id,
    suffix="pii-redaction-qwen",
    method={
        "type": "supervised",
        "supervised": {"hyperparameters": HYPERPARAMETERS},
    },
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")
```

**Output:**

```
Job ID: ftjob-588078bcd97544189b51240835ca59fc
Status: validating_files
```

As you can see, we set every hyperparameter explicitly (this post spent a lot of words earning them), but you don’t have to. Each of these fields also accepts `"auto"`, and there is real engineering behind that word: no single recipe fine-tunes every architecture well (mixture-of-experts models, hybrid-attention designs, and per-family chat templates each need their own treatment), so the Crusoe team tuned and verified a recipe for every model in the catalog. `"auto"` starts you from those defaults, and you can override any subset, as we did, while the server fills in the rest.

`lora_rank` and `lora_alpha` are Crusoe extensions to the OpenAI schema, and more dials like `warmup_ratio` and `early_stopping_patience` are there when you want them; the full set lives in the [API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning/operation/createFineTuningJob). For a workflow-focused walkthrough of the service, see the official [practical guide](https://www.crusoe.ai/resources/blog/a-practical-guide-to-serverless-fine-tuning).

That’s it! The job is queued, and you can follow it from the console or the API.

## Monitoring the Job [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#monitoring-the-job)

A fine-tuning job walks through a little state machine, which we can watch via the events API:

![Training job states](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/job-states.svg)

```
import time

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

def wait_for_job(job_id, poll_seconds=15):
    last_status, started = None, time.time()
    while True:
        j = client.fine_tuning.jobs.retrieve(job_id)
        if j.status != last_status:
            print(f"[{int(time.time() - started):>4}s] status -> {j.status}")
            last_status = j.status
        if j.status in TERMINAL_STATUSES:
            return j
        time.sleep(poll_seconds)

job = wait_for_job(job.id)
assert job.status == "succeeded", f"Job ended as {job.status}, see events below or the console"
print("\nTraining complete")
```

**Output:**

```
[   0s] status -> queued
[  30s] status -> running
[ 413s] status -> succeeded

Training complete
```

Read that timestamp again: **~7 minutes** to fine-tune an 8B model on 2,500 examples for 2 epochs.

And we now know _why_ it’s that fast: only ~0.5% of the model is trainable, the base never changes.

We can visualize the loss curves while the training job is running, or after it has completed, from the console:

![Training and validation loss curves in the Crusoe console](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/job-monitor.png)

And checkpoints are first-class citizens, each one is a _deployable model_ with its own metrics, we can list them via the API as well:

```
ckpts = client.fine_tuning.jobs.checkpoints.list(job.id)
for c in ckpts.data:
    m = c.metrics
    print(f"step {c.step_number:>5}  train_loss={m.train_loss:.4f}  valid_loss={m.valid_loss:.4f}")
```

**Output:**

```
step   100  train_loss=0.0164  valid_loss=0.0298
step   200  train_loss=0.0153  valid_loss=0.0216
step   300  train_loss=0.0141  valid_loss=0.0204
step   314  train_loss=0.0125  valid_loss=0.0203
```

The last checkpoint, step 314, is the final model, and train and validation loss fall together the whole way. (If validation loss had turned upward, we’d simply deploy an earlier checkpoint, this is exactly why they exist.)

TipDownload the adapter

Once the job is done, the model is yours to keep: download it, and what you get is just a zip of LoRA adapters (`.safetensors`).

```
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# the fine_tuned_model ID registers shortly after the job succeeds
FINE_TUNED_MODEL = job.fine_tuned_model
while FINE_TUNED_MODEL is None:
    time.sleep(5)
    FINE_TUNED_MODEL = client.fine_tuning.jobs.retrieve(job.id).fine_tuned_model

zip_path = out_dir / f"{FINE_TUNED_MODEL}.zip"
with httpx.stream(
    "GET",
    f"{INTELLIGENCE_BASE_URL}/models/{FINE_TUNED_MODEL}/download",
    headers={"Authorization": f"Bearer {CRUSOE_API_KEY}"},
    timeout=300,
) as r:
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_bytes(8192):
            f.write(chunk)
print(f"Saved {zip_path} ({zip_path.stat().st_size:,} bytes)")
```

**Output:**

```
Saved outputs/ftmodel-31e71c5f-3983-49d1-b1de-77e5512c1225.zip (161,510,890 bytes)
```

The entire fine-tune of an 8B model is a ~ **162 MB** zip of LoRA adapter weights (rank 16, all layers, plus configs).

You can deploy it on Crusoe, or download it and serve it yourself with vLLM anywhere. No lock-in, it’s just the 𝐵 and 𝐴 matrices :)

## Deploy and Infer [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#deploy-and-infer)

Let’s deploy the model via the [console](https://docs.crusoecloud.com/self-serve-deployments/overview): pick the job (or any checkpoint), choose an optimization profile, set a replica count, and pick a deployment name.

- **Responsiveness**: optimize time-to-first-token, for user-facing apps
- **Throughput**: optimize tokens/sec/dollar, for batch pipelines like ours
- **Balanced**: a blend of the two, the default for general purpose production traffic

![deployment](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/Deployment.png)

Once the model is deployed we can use the **deployment name** (`"test-qwen3-8B-ft"`), and perform the inference:

```
DEPLOYMENT_ALIAS = "test-qwen3-8B-ft"

inference = OpenAI(
    api_key=CRUSOE_API_KEY,
    base_url=INFERENCE_BASE_URL,
)

demo_document = """Subject: Wire transfer confirmation

Hi Suman,

The transfer of $18,400 to Meridian Holdings LLC was initiated on March 3, 2026
from account 4432-889912. Please confirm receipt with our treasury desk at
treasury@meridianholdings.com, or call Daniel Okafor at (415) 555-0173.

Thanks,
Elena Vasquez
Senior Treasury Analyst"""

response = inference.chat.completions.create(
    model=DEPLOYMENT_ALIAS,
    messages=[\
        {"role": "system", "content": utils.SYSTEM_PROMPT},\
        {"role": "user", "content": demo_document},\
    ],
    temperature=0,
)

prediction = utils.parse_prediction(response.choices[0].message.content)
print(prediction["redacted_text"])
print("\nEntities found:")
for e in prediction["entities"]:
    print(f"  {e['type']:<18} {e['value']}")
```

**Output:**

```
Subject: Wire transfer confirmation

Hi [NAME],

The transfer of $18,400 to [COMPANY] was initiated on [DATE] from account [ACCOUNT_PIN]. Please confirm receipt with our treasury desk at [EMAIL], or call [NAME] at [PHONE_NUMBER].

Thanks,
[NAME]
Senior Treasury Analyst

Entities found:
  name               Suman
  company            Meridian Holdings LLC
  date               March 3, 2026
  account_pin        4432-889912
  email              treasury@meridianholdings.com
  name               Daniel Okafor
  phone_number       (415) 555-0173
  name               Elena Vasquez
```

## The Payoff: Our 8B Beats a 70B [Anchor](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-payoff-our-8b-beats-a-70b)

Does fine-tuning actually buy anything over prompting a big general model ? We run both over the 100 held-out test documents with the identical system prompt and score them at the entity level:

- **Detection**: the model found the PII string, whatever type it assigned. Precision, recall, and F1 over (document, value) pairs.
- **Typed**: the value and its type label both match the reference. This is where format and taxonomy training shows.
- **Bad JSON**: replies that could not be parsed into the output contract at all. In a production redaction pipeline this is the metric that pages someone.

The baseline is **Llama 3.3 70B Instruct** on **Crusoe Serverless Inference** (~9x larger than our fine-tuned Qwen3-8B), which we prompt with the same system message.

```
BASELINE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # serverless, pay per token
EVAL_N = 100

test_rows = [json.loads(line) for line in open("data/test.jsonl")]

contenders = {
    "Fine-tuned Qwen3 8B": DEPLOYMENT_ALIAS,
    "Llama 3.3 70B (serverless)": BASELINE_MODEL,
}

def predict(model_name, document):
    r = inference.chat.completions.create(
        model=model_name,
        messages=[\
            {"role": "system", "content": utils.SYSTEM_PROMPT},\
            {"role": "user", "content": document},\
        ],
        temperature=0,
        max_tokens=1500,
    )
    return utils.parse_prediction(r.choices[0].message.content)

scores = {label: [] for label in contenders}
parse_failures = {label: 0 for label in contenders}

for row in test_rows[:EVAL_N]:
    document = row["messages"][1]["content"]
    reference = json.loads(row["messages"][2]["content"])["entities"]
    for label, model_name in contenders.items():
        pred = predict(model_name, document)
        if pred is None:
            parse_failures[label] += 1
            pred = {"entities": []}
        scores[label].append(utils.score_entities(reference, pred["entities"]))

summaries = {label: utils.aggregate_scores(rows) for label, rows in scores.items()}
utils.report_eval(summaries, parse_failures, min(EVAL_N, len(test_rows)))
```

**Output:**

```
Entity level scores on 100 held-out documents

Model                                    Det P   Det R  Det F1  Typed F1  Bad JSON
----------------------------------------------------------------------------------
Fine-tuned Qwen3 8B                       86%    75%    80%      79%         3
Llama 3.3 70B (serverless)                36%    62%    46%      36%        13
```

![Evaluation: fine-tuned Qwen3-8B vs Llama 3.3 70B on PII redaction](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/eval_ft_vs_70b.png)

The gap is bigger than I expected. Looking at individual documents makes it concrete: on one representative example, the fine-tune returns the 6 entities in the reference and nothing else, valid JSON, correct type labels.

The 70B finds 21 entities in the same document, half of them duplicates or fragments, labels them with type names it made up (`ACCOUNT_NUMBER` where the schema says `account_number`), and across the full test set fails to produce parseable JSON 13 times out of 100.

None of this means the 70B is a worse model. It means it never saw our schema. It’s guessing at the output contract from a system prompt, and a system prompt turns out to be a weak way to hold a model to a format across a hundred documents. Two epochs of LoRA on 2,500 examples is a much stronger one: that’s the 79% vs 36% typed F1 in the table.

The cost side points the same direction. The fine-tune is ~9x smaller, so each document is cheaper and faster to process, and because it runs on a dedicated deployment, the latency is predictable and the bill is a flat GPU-hour rate rather than per-token, a better fit for a pipeline that redacts documents all day.

# Conclusion

We covered a lot of ground in this post, so let’s recap the journey. We started with the uncomfortable arithmetic of full fine-tuning, ~146 GB to fine-tune an 8B model, of which nearly 90% is training overhead rather than the model itself. We attacked the bill from two directions: **quantization** (built an INT8 quantizer by hand, met the outlier problem, and understood why NF4’s Gaussian-shaped bins work), and **LoRA** (built a 20-line adapter, trained it, and proved with SVD that the update stayed inside its low-rank box). We stacked them into **QLoRA** and watched `peft` reproduce our hand-rolled parameter count to the digit.

Then we put it to work: **Qwen3-8B fine-tuned into a PII redaction engine in ~7 minutes** on Crusoe’s Serverless Fine-Tuning, deployed on a dedicated endpoint, and beating a 70B generalist by 43 typed-F1 points on the task it was trained for, with a 162 MB adapter we can download and take anywhere.

The trade-offs are real and worth restating: LoRA teaches _behaviors_, not _bodies of knowledge_, continued pre-training still wants full fine-tuning and the FSDP machinery from the last post. QLoRA pays a compute tax for its memory savings. And a managed service trades fine-grained control (custom loss functions, exotic architectures) for never thinking about CUDA drivers again, a trade I happily make for the 95% of fine-tunes that are standard SFT.

There’s plenty we haven’t covered, a proper tour of the LoRA variants (DoRA, PiSSA, VeRA and friends), preference fine-tuning (DPO/RLHF) on top of LoRA, multi-adapter serving, distillation into even smaller models, and I hope to write about those as I learn them :).

# References

- **Papers**
  - [Measuring the Intrinsic Dimension of Objective Landscapes (Li et al., 2018)](https://arxiv.org/abs/1804.08838)
  - [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (Aghajanyan et al., 2020)](https://arxiv.org/abs/2012.13255)
  - [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
  - [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339)
  - [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
  - [LoRA Learns Less and Forgets Less (Biderman et al., 2024)](https://arxiv.org/abs/2405.09673)
- **Blogs and books**
  - [LoRA Without Regret (Schulman et al., Thinking Machines, 2025)](https://thinkingmachines.ai/blog/lora/)
  - [Build a Large Language Model (From Scratch), Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  - [A Hands-On Guide to Fine-Tuning Large Language Models, Daniel Voigt Godoy](https://leanpub.com/finetuning)
- **Libraries and code**
  - [This post’s repo: debnsuma/fine-tuning-lora](https://github.com/debnsuma/fine-tuning-lora)
  - [Hugging Face PEFT](https://huggingface.co/docs/peft) · [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) · [Transformers](https://huggingface.co/docs/transformers)
  - [gretelai/synthetic\_pii\_finance\_multilingual dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual)
- **Crusoe**
  - [Introducing Serverless Fine-Tuning (launch blog)](https://crusoe.ai/resources/blog/crusoe-introduces-serverless-fine-tuning)
  - [Self-Serve Deployments (launch blog)](https://crusoe.ai/resources/blog/crusoe-self-serve-deployments)
  - [Serverless Fine-Tuning docs](https://docs.crusoecloud.com/serverless-fine-tuning/overview) · [Self-Serve Deployments docs](https://docs.crusoecloud.com/self-serve-deployments/overview) · [Fine-tuning API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- **My previous posts in this series**
  - [From Single GPU to Clusters: A Practical Journey into Distributed Training](https://debnsuma.github.io/my-blog/posts/distributed-training-from-scratch/)
  - [Inside FSDP with PyTorch and Ray](https://debnsuma.github.io/my-blog/posts/fsdp-ray-train/)