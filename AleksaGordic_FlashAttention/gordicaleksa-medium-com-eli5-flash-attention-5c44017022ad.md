
![](https://miro.medium.com/v2/resize:fit:1000/1*i4tDdwgvGtXuTIyJpFUn8A.png)

>  Should understand this diagram perfectly by the end of this blog

Without further ado let’s start by breaking down the paper title:

“FlashAttention: **Fast** and **Memory-Efficient** **Exact** Attention with **IO-Awareness**”

- **Fast**— excerpt from the paper: *“We train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1, GPT2 (seq. length 1K) 3x faster than baseline implementations from HuggingFace and Megatron-LM, and long-range arena (seq. length 1K-4K) 2.4x faster than baselines.”*
- **Memory-efficient** — vanilla attention is quadratic in sequence length, _O(N²)_, this method is sub-quadratic/linear in N ( _O(N)_)
- **Exact**— **not an approximation** of the attention mechanism
- **IO aware** — compared to vanilla attention, flash attention is **sentient** (aware)

> Why sub-quadratic/linear →Interesting wording


> [!NOTE] Relevant excerpt from the paper:
> “Although these \[approximate\] methods reduce the compute requirements to linear or near-linear in sequence length, many of them do not display wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that *they focus on FLOP reduction* (which may not correlate with wall-clock speed) and *tend to ignore overheads from memory access (IO).*”


![](https://miro.medium.com/v2/resize:fit:700/0*ykyAH9UomvFs-WeJ.png)

Over the years GPUs have been adding compute capacity (FLOPS) at a faster pace than increasing the memory throughput (TB/s)

**It doesn’t matter if you can compute at exaFLOPS speeds if there is no data to be processed.** These 2 need to be closely aligned, and since the hardware lost that balance we have to make our software compensate for it.

> Hence “IO-aware”

 - **compute-bound** (example: matrix multiplication)
 - OR **memory-bound** (examples: elementwise ops (activation, dropout, masking), reduction ops (softmax, layer norm, sum, etc.)…)

> Note on the terminology: this ratio is commonly measured by the **arithmetic intensity**, which is the number of arithmetic operations per byte of memory access.
>
> Note 2: Aleksa strongly recommends reading Horace’s blog post [https://horace.io/brrr\_intro.html](https://horace.io/brrr_intro.html)

It turns out **attention is** (on current AI accelerators) **memory-bound**.

Why?

Because it “mostly consists of elementwise ops” or more accurately the arithmetic density of attention is not very high.

![](https://miro.medium.com/v2/resize:fit:700/1*QOpKA6V8ZQ4X_s5lD0zybQ.png)

Masking, softmax & dropout are the (elementwise) ops that are taking the bulk of the time and NOT matrix multiplication (even though bulk of the FLOPS is in matmul)

Memory is not a monolithic artifact, it’s hierarchical in its nature!
- Faster memory = more expensive = smaller capacity

![](https://miro.medium.com/v2/resize:fit:700/1*wf7fUgSOUz-9X3-mUWkYmA.png)

Memory is hierarchical in its nature. We could continue the pyramid by further adding SSD (even higher capacity and even slower), HDD (hard drives), etc. (AWS S3? :) ). lol. 

 “IO-aware” → exploiting the fact that SRAM is so much faster than HBM (“high bandwidth memory” — unfortunate name) by making sure to reduce the communication between the two.


> [!NOTE] concrete example
> A100 GPU has **40–80GB** HBM with a bandwidth of **1.5–2.0 TB/s** and **192KB**of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around **19TB/s.**

> Similar ratios still hold for H100 and other accelerators

### Computations behind the standard attention implementation:

![](https://miro.medium.com/v2/resize:fit:1000/1*9CnVGnPdRrXMQ4VtSmpWtg.png)

Standard implementation shows the *UTMOST DISRESPECT* for the way HW operates → treating HBM load/store ops as 0 cost (it’s not “IO-aware”)

The lowest hanging fruit is to **remove redundant HBM reads/writes**

Why write **_S_** back to HBM only to (re)load it again in order to compute the softmax? Let’s keep it in SRAM instead, perform all of the intermediate steps, and only then write the final result back to HBM

This is what compilers folks refer to as **“kernel fusion”**, one of the most important low-level optimizations in deep learning:

![](https://miro.medium.com/v2/resize:fit:700/0*0Yn1aLye8s6_WTOu.png)

> **_kernel_**is a fancy way of saying “a GPU operation”. **_Fusion_** means  fusing/combining multiple ops together.  **“materialization”** means we’ve **allocated** full **NxN** matrices ( **_S_**, **_P_**)

So, you are loading from the HBM only **once,** you execute the fused op, and only then write the results back. By doing this you reduce the communication overhead.


### Flash attention boils down to 2 main ideas

1. **Tiling**(used during **both forward & backward passes**) — chunking the NxN softmax/scores matrix into blocks
2.  **Recomputation**(used in the **backward pass only** — if you’re familiar with activation/gradient checkpointing, this will be trivial to understand)

That’s it.

Here is the algorithm:

![](https://miro.medium.com/v2/resize:fit:700/1*OJBNIb3fz6JEuisrLfSW6g.png)

My job here is done. Hope you enjoyed this blog post, subscribe for more content in the future! 🚀

Just kidding.

> This guyyyy. Legenddddd

## FlashAttention algorithm

> Consider one vector. Apply softmax to every element

**The main hurdle in getting the tiling approach to work is softmax.**

![](https://miro.medium.com/v2/resize:fit:285/1*sOvo7mEBzW60Dchup8mDLA.png)

$z\_i$ is the $i$-th score (key-query dot product), and the output is the $i$-th token’s probability that we later use to weight the value vector

> Denominator is the issue

To compute _$i$-th_ token attention score, need to have all others’ scores readily available ($z\_j$_) in SRAM

But SRAM is severely limited in its capacity. Cannot just load the whole thing. N (sequence length) can be 1000 or [even 100.000](https://www.anthropic.com/index/100k-context-windows) tokens. So _N²_ explodes 

### Trick
Chop the softmax computation into smaller blocks → still end up with precisely the same result

![](https://miro.medium.com/v2/resize:fit:1000/1*8nxJfYSXZvev0NE_jC5WBQ.png)

Formula 1. *Partial* computation of softmax

Grab only the first _B_ scores ( _x\_1_ through _x\_B_) and compute the softmax for them.

> **These numbers are, at least for now, incorrect** 


> Note: also check out the original papers that introduced the online softmax: [https://arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)
>
> [https://arxiv.org/abs/2112.05682](https://arxiv.org/abs/2112.05682)

![](https://miro.medium.com/v2/resize:fit:1000/1*ViyWK0-Nc6twYFpxR45OqQ.png)

Formula 2. Core idea behind softmax tiling → by recursively repeating this computation across all of the blocks we end up with the correct softmax output

$x^{(1)}, x^{(2)}$ are each blocks of size *B*

Keep track of 2 statistics for each of the blocks
- **_m(x)_** (maximum score)
- **_l(x)_** (sum of exp scores)

Continue recursively up to the last, _($N/B$)-th,_ block → get the N-dimensional correct softmax output!


> [!NOTE] MATHEMATICAL EXPLANATION 
> Check my notes from Umar Jamil’s video [here](https://github.com/PoptropicaSahil/FlashAttention-from-scratch#algorithm)


Ok, we now have all of the ingredients we need in order to understand the forward pass of the flash attention algorithm.

> Note: the algo below assumes (a) batch size 1 (i.e. single sequence), (b) single attention head, (c) ignore dropout & masking for now

![](https://miro.medium.com/v2/resize:fit:1000/1*OJBNIb3fz6JEuisrLfSW6g.png)

Notation remark: d — attention head dimension, B\_r ( **row** block size), B\_c ( **column** block size) — the rest of the notation was introduced before.

Let’s now break it down step by step!

S **tep 0:** HBM’s capacity is measured in GBs (e.g. RTX 3090 has 24 GBs of VRAM/HBM, A100 has 40–80 GB, etc.) so allocating **_Q_**, **_K_**, and **_V_** is not an issue.

**Step 1:** Let’s compute the row/column block sizes. Why _ceil(M/4d)_? Because query, key, and value vectors are d-dimensional, and, we also need to combine them into the output d-dimensional vector. So this size basically allows us to max out SRAM capacity with _q_, _k_, _v_, and _o_ vectors.

Toy example: assume M = 1000, d = 5. In this example, the block size is (1000/4\*5) = 50. So in this example, we would load blocks of 50 _q, k, v, o_ vectors at a time, to make sure we’re reducing the number of reads/writes between HBM/SRAM.

Worth keeping this image in your mind (it will make more sense soon):

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*0OV7ituGlfv9EaIfmtqbmg.png)

As for _B\_r_, I’m not exactly sure why do they perform a min op with _d_? If anyone knows feel free to leave a comment!

**Step 2:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*zkv3FSgGyZ7-iG_Q4k8GxQ.png)

We initialize the output matrix **_O_** with all 0s. It’ll act as an accumulator hence that init value. Similarly for **_l_**(remember: its purpose is to hold the cumulative denominator for the softmax - the sum of exp scores). **_m_**(that holds row-wise maximum scores)is initialized with _-inf_ because we’ll be doing a max operator over it so whatever the first block’s max is — it’ll certainly be larger than _-inf —_ hence this is the natural init value.

**Step 3:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*VG8bCa1mhiXx6D32wvk7Zw.png)

We split the **_Q, K,_** and **_V_** into blocks using the block sizes from Step 1. See also the diagram above.

**Step 4:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*wTgKZnd1nFJ1SkRpZ9nj8Q.png)

Similarly split **_O, l, m_** into blocks (same block size as **_Q_**).

**Step 5:**

![](https://miro.medium.com/v2/resize:fit:493/1*e0_mx0OSzVNIOE-4ZIq-zw.png)

Let’s start looping across the columns i.e. across key/value vectors ( **outer loop** in the diagram above).

**Step 6:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*PaxaJ90G6aHmKP-0RYVrtw.png)

Let’s load the **_K\_j_** and **_V\_j_** blocks from HBM to SRAM. Remember because of the way we constructed the block sizes we still have 50% of the SRAM unoccupied at this point in time (dedicated to **Q** and **O**).

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/0*Ywl4arvMvN6HCFwZ.png)

Roughly & abstractly. The GPU memory layout will obviously be different.

**Step 7:**

![](https://miro.medium.com/v2/resize:fit:520/1*mTCUNX239DWlG-i5LGll9w.png)

Start the **inner loop** across the rows i.e. across query vectors (again, see the diagram).

## Get Aleksa Gordić’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

Subscribe

Remember me for faster sign in

**Step 8:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*y0gyIcenDedgWkmr_s2Bug.png)

Load **_Q\_i_** ( _B\_r x d_) and **_O\_i (_** _B\_r x d_ **_)_** blocks, as well as **_l\_i_** (_B\_r_) & **_m\_i (_**_B\_r_**_)_**into SRAM.

How do **_l\_i_** & **_m\_i_** fit into the SRAM (including all of the intermediate variables) when we computed block size in such a way that we only have enough space for **_K\_j_**, **_V\_j_**, **_Q\_i_** & **_O\_i_**? I think the answer is: registers (see [this CUDA video series](https://www.youtube.com/watch?v=4APkMJdiudU&list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe) to get some intuition on GPU memory hierarchy). But I might be wrong, someone who’s actually implemented this in CUDA please correct me. 🙏 I’m sure I’m missing out on important implementation details by just analyzing the pseudo-algorithm.

**Step 9:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*LMZEvals-Shf7giT8slQmw.png)

Compute the dot product between **_Q\_i_** ( _B\_r x d_) and **_K\_j_** transposed ( _d x B\_c_) to get the scores ( _B\_r x B\_c_). As you can see we don’t have the whole _NxN_ **_S_** (scores) matrix “materialized”. Only a fraction of it (**_S\_i\_j_**)!

Toy example: assuming the outer loop index is _j_ (j=3), inner loop index is _i_ (i=2), _N_ is 25 and the block size is 5 this is what we just computed (assuming 1-based indexing):

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/0*m0pdO7LvXYrZ503F.png)

Basically the attention scores for tokens 6–10 with tokens 11–15 of our input sequence. But, importantly, these are exact scores, they’ll never change (as opposed to softmax results that will gradually get refined).

**Step 10:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*F4LNBA_VsPhqNk4VkmNrOw.png)

Compute **_m~\_i\_j_**, **_l~\_i\_j_**, and **_P~\_i\_j_** using the scores computed in the previous step. It’s trivial.

**_m~\_i\_j_** is computed row-wise, find the max element for each of the above rows.

We get **_P~\_i\_j_** by applying elementwise ops:

1. Normalization — take the row max and subtract it from row scores
2. Exp

**_l~\_i\_j_** is simply a row-wise sum of the matrix P.

**Step 11:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*jFlLbR8DS5Ei5NpwgIIvWA.png)

Compute **_m\_new\_i_** and **_l\_new\_i_**. Again fairly simple, let’s reuse the diagram from above:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/0*yHc4h9UJ3AiHQ7fn.png)

**_m\_i_** contains row-wise maximums for all of the blocks that came before (j=1 & j=2, colored in green). **_m~\_i\_j_** contains the row-wise maximums for the current block (colored in yellow). To get the **_m\_new\_i_** we just have to apply a max between **_m~\_i\_j_**& **_m\_i_**. Similarly for **_l\_new\_i_**(it additionally requires multiplying by coefficients as we saw previously in _formula 2_).

**Step 12 (the most important step):**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*KNZwy8pvXo-AwcogXBETng.png)

This is the hardest part of the algorithm but still not that complicated, esp. once you internalize the _formulas 1 & 2_ for partial softmax computation.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/0*2DlPHK6iEltHdQxW.png)

Let’s break down the **_diag(l)_** part first.

It basically just allows us to do row-wise scalar multiplication in a matrix form. If you have a list of scalars **_s_**( _N_)and a matrix **_A (_** _NxN_ **_)_**, if you do **_diag(s)_**\* **_A_** you’re basically doing elementwise multiplication of rows of **_A_** with those scalars.

Next up notice the similarity between _step 12_ and _formula 1_ (pasting it here again for convenience):

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*XlUH2kUNmtZzCXMfQ9Q65A.png)

So what the 1st term of step 12 does (underlined in green) is it updates the current softmax estimate for the blocks before the current block in the same row of blocks. In case j=1 (that is the first block in this row) the 1st term will be 0 and we’ll just end up with the 2nd term.

The multiplication of the 1st term by **_diag(l\_i)_** is there to cancel the division by that same constant from the previous iteration (this constant is hidden inside of **_O\_i_**).

The 2nd term of the expression (underlined in yellow) doesn’t require this canceling of terms because as you can see we’re directly multiplying the **_P~\_i\_j_** matrix with the block of **_V_** vectors ( **_V\_j_**).

The **_e^x_** terms are there to modify the matrix **_P~\_i\_j_** & **_O\_i_** by canceling out the **_m_** from the previous iteration and instead updating it with the latest estimate ( **_m\_new\_i_**) that contains the row-wise max so far.

The easiest way to convince yourself this makes sense is to just simulate a couple of iterations yourself — in case you still didn’t quite get it.

It literally takes 5 minutes. Here is my step-by-step analysis (hope it helps!):

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*RuLi2fMrkxeCkPo6_bttvw.png)

As you can see the main point is that these e terms outside cancel out with e terms inside of P/O matrices and we always end up having the latest m\_new\_1 estimate!

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*a4Fu3h_eL9LMLIkLgXW-1g.png)

The third iteration is similar and we end up with the correct and final result!

Recall: this is just a current estimate of the final **_O\_i_**. Only after we iterated through all of the red blocks in the diagram above will we end up having the exact result. And that’s it!

**Step 13:**

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*fMJPUiQQ-JhGpZFFOFTNWw.png)

Write the newest cumulative statistics ( **_l\_i_** & **_m\_i_**) back to HBM. Notice these are of dimension _B\_r_.

**Steps 14, 15, 16:**

![](https://miro.medium.com/v2/resize:fit:382/1*GNaqyihDrF2wo3eaFPUCLg.png)

Once the nested for loop is over, **_O_** (Nxd) will contain the final result: attention-weighted value vectors for each of the input tokens!

That’s it guys. That’s the forward pass of the flash attention!

This algorithm can easily be extended to “block-sparse FlashAttention”, a sparse attention algorithm that is 2–4 faster than even FlashAttention, scaling up to a sequence length of 64k! The idea is we use a block form mask matrix and we simply skip certain loads/stores from the above nested for loop and by doing so we can save proportionally to the sparsity coefficient.

![](https://miro.medium.com/v2/resize:fit:274/0*jBov-kkiUvAv1cA0.png)

Example of a block form mask matrix. Assuming our block length is 3 we’d have only 3/9 iterations in this toy example. Thus 3x faster!

Now let’s briefly touch on the complexity.

### Complexity analysis

**Space:** We’ve allocated **_Q_**, **_K_**, **_V_**, **_O_** ( _Nxd_), **_l_** & **_m_** ( _N_) in HBM. That’s 4\*N\*d + 2\*N. Dropping the constants (big _O_ stuff), and knowing that _d_ is also a constant and usually much smaller than N (e.g. usually d={32, 64, 128}, N={1024, …, 100k}) we get _O(N)_ for space. Big win! That helps us scale transformers to 64k sequence lengths “easily” (add to that a couple of other “tricks” like ALiBi which I’ll cover in one of the follow-up blog posts).

**Time:** we won’t strictly do a time complexity analysis, instead we’ll use a good proxy: **the number of HBM accesses**.

Here is an excerpt from the paper:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*blllQ-SaDeNQoUOilvUhTg.png)

How do they get to that number? Well, let’s analyze the nested for loop:

- Our block size is _M/4d_. This means that the vectors are split into N/(M/4d) blocks.
- Raise that to the power of _2_ (because we’re looping across row/column blocks) and we get _O(N² d² / M²)_
- Now here I have _M²_ and they only have _M¹_— my hypothesis is that we probably need _M/4d_ memory accesses inside of the loop to actually fetch all the vectors, i.e. we can’t fetch the whole block all at once. That would explain the missing _M_? (_d_ can be ignored as a constant in big _O_).

If we were to do a big _O_ analysis that could lead us to think this is not much better than standard attention, but for typical numbers, this leads to up to 9x fewer accesses (as per the excerpt above).

And that’s it, you now (hopefully) understand the flash attention!

Let’s wrap it up by closing the gap with the real world. So far we were analyzing the pseudo algorithm focusing on a single attention head assuming a batch size of 1. And we also glossed over the backward pass.

### batch\_size > 1, num\_heads > 1, backward pass

Let’s start with the low-hanging fruit. Extending the implementation we saw to support _batch\_size_ \> 1 and the _num\_heads_ \> 1 is actually not that hard.

So far the algorithm we saw is basically handled by a single **_thread block_** (CUDA programming lingo). This thread block is executed on a single **_streaming multiprocessor_** (SM) (e.g. there are 108 of these on A100). To parallelize our computation we just run _batch\_size_ \\* _num\_heads_ threadblocks in parallel on different SMs. The closer that number is to the number of available SMs on the system the higher the utilization will be (ideally a multiple as each SM can run multiple thread blocks).

What happens when that number is bigger than the number of available SMs? I’m not sure but I assume there is a queue that keeps track of the waiting kernels (update: [apparently](https://www.youtube.com/watch?v=xwbD6fL5qC8&t=760s&ab_channel=TomNurkkala) the CUDA runtime takes care of that and it is using some sort of queues to implement that logic).

Next up let’s briefly address the backward pass.

The backward pass relies on the same set of concepts + **_recomputation_**.

To demonstrate the concept of recomputation I’ll use the example of “ _activation/gradient checkpointing_” method.

We know that we need to have the activations computed during the forward pass readily available during the backward pass in order to compute the gradients w.r.t. our loss function.

The trick here is to not store them during the fwd pass (as they have a huge memory footprint), but instead, recompute them de novo during the backward pass. There is a built-in **tradeoff** here: we’re slowing down the backward pass in order to reduce the memory footprint.

> Note: This tradeoff is a spectrum, e.g. you can store the activations every _n_ layers, and then when computing the activations for the i-th layer you don’t have to start from the input but instead from the closest stored activations.

The same concept of recomputation is re-used here — but with a twist! Luckily for the flash attention, we don’t have to sacrifice neither runtime nor memory!

By storing the output **_O_** ( _Nxd_) and the softmax normalization statistics ( _N_) we can recompute the attention matrices **_S_** ( _NxN_) and **_P_** ( _NxN_) in the backward pass directly from blocks of **_Q_**, **_K_**, and **_V_** ( _Nxd_) in SRAM! Thus keeping the memory at _O(N)_. I encourage you to read the paper if you’re curious about the details, but I assure you, you’re equipped with all of the tools you need to understand it now.

Lastly, let’s see some of the issues one could expect implementing flash attention.

## The real world is…messy

The same thing that gives flash attention its power is the root cause of its issues. Let’s see this excerpt from the paper:

“Our current approach to building IO-aware implementations of attention requires writing a new CUDA kernel for each new attention implementation. This requires writing the attention algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort. Implementations may also not be transferrable across GPU architectures. These limitations suggest the need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and compiling to IO-aware implementations in CUDA…”

As a consequence, the original flash attention supported only a subset of GPUs. For example, V100 is not supported. See this issue on their GitHub:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*WAgDijytJXfwz-_u1geEIg.png)

[https://github.com/HazyResearch/flash-attention/issues/148](https://github.com/HazyResearch/flash-attention/issues/148)

> Note: here is one more relevant issue: [https://github.com/HazyResearch/flash-attention/issues/190](https://github.com/HazyResearch/flash-attention/issues/190)

To make this issue a bit more visceral, here is the actual CUDA code from the original implementation:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*MscGQcD2J3L9KxiZQA6TnA.png)

a snippet of CUDA implementation from the original codebase [https://github.com/HazyResearch/flash-attention/blob/main/csrc/flash\_attn/src/fmha\_fprop\_kernel\_1xN.h#L383](https://github.com/HazyResearch/flash-attention/blob/main/csrc/flash_attn/src/fmha_fprop_kernel_1xN.h#L383)

As you can see, writing CUDA is…messy. Additionally — this is a research codebase which doesn’t help, but even if it wasn’t for someone coming from an ML research background, who is only comfortable with Python this is (potentially) a deal breaker.

This is where projects like [OpenAI’s Triton](https://openai.com/research/triton) could be game changers (see their [FlashAttention implementation](https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py)). Triton is basically a **_DSL_**(domain-specific language) between CUDA & other DSLs (e.g. **_TVM_**) in its level of abstraction. You can write Python code that’s super optimized (once compiled) instead of having to deal directly with CUDA. That very same Python code could then be deployed on an arbitrary accelerator (that responsibility lies on Triton devs and HW manufacturers).

Triton has recently been integrated with PyTorch 2.0 so definitely keep an eye out for this project! Shout out to [Philippe Tillet](https://github.com/ptillet/) who started building Triton during his PhD and later became a part of OpenAI.

Finally, it’s worth mentioning that for certain use cases, you might still prefer other methods. E.g. for sequence lengths beyond 1K, some approximate attention methods (e.g., Linformer) start to become faster. But ultimately, to the best of my understanding, the block-sparse implementation of the flash attention outperforms all other methods.

## Outro

You might ask yourself: why didn’t anyone invent FlashAttention before? Given how crucial this piece of computation is to all modern ML workloads. Given how many engineering hours are spent, across different tech organizations, trying to squeeze the last ounce of performance out of these systems. Why did a [Stanford student](https://tridao.me/) (shout out to Tri Dao for the amazing work!) come up with this and not e.g. NVIDIA engineers?

There is a couple of possible explanations that I can see:

- FlashAttention is, as far as I can tell, easier/only possible to implement on the latest GPUs (hence V100 is not supported in the original codebase). Couple this with the phenomenon that oftentimes “outsiders” are those who see the problems with beginner’s eyes and tackle issues from the first principles and we get to the possible explanation.
- Ultimately I really like [Nat Friedman’s](https://nat.org/) perspective on the efficiency of the world:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*V5FdAlPuQgFFjCxRjcLVjg.png)

[https://nat.org/](https://nat.org/)

Finally, let me wrap up with some food for thought:

Given how much it costs to train these models (see [this blog](https://www.mosaicml.com/blog/gpt-3-quality-for-500k) from MosaicML for the most optimistic estimates) being able to shave off 15% from BERT-large training, or speed up GPT training 2/3x has such a tremendous economic impact when considering all of the current, and future, model trainings at a global scale.

Imagine if we lived in a world where **_Y_** (value captured) was actually correlated with **_X_** (value produced). The author would be a trillionaire over the next few years. :)

Alas, researchers rarely capture that value, for better or worse (imagine if Pythagoras patented his theorem :P). Yet people who oftentimes just repackage stuff are those who capture most of the value. Going on a complete tangent here, until next time! ;)

### Further resources

- Flashier Attention blog — [https://www.adept.ai/blog/flashier-attention](https://www.adept.ai/blog/flashier-attention) -\> They show how to further optimize flash attention for highly distributed settings where batch sizes become super small (pipeline parallelism) and sequence length super long.
- Tri Dao’s talk: [FlashAttention — Tri Dao \| Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4&ab_channel=StanfordMLSysSeminars)
- Tri Dao’s talk: [MedAI #54: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness \| Tri Dao](https://www.youtube.com/watch?v=FThvfkXWqtE&ab_channel=StanfordMedAI)
- Paper: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

### Acknowledgments

Thanks to [Tri Dao](https://twitter.com/tri_dao), [Horace He](https://twitter.com/cHHillee), and [Amrit Sahu](https://twitter.com/AmritSa51993336) for reading earlier drafts of this blog post and providing feedback!

### BibTeX Citation

```
@article{gordiceli5flashattention,
  author={Aleksa Gordic},
  title={ELI5: FlashAttention},
  year={2023},
  url={https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad},
}
```

### Connect with me

Last but not least feel free to drop me a message or:

1. Connect and reach out on [**LinkedIn**](https://www.linkedin.com/in/aleksagordic/) and/or [**Twitter**](https://twitter.com/gordic_aleksa)
2. Subscribe to my [**YouTube channel**](https://www.youtube.com/c/TheAiEpiphany) for more ML
3. Follow me on [**Medium**](https://medium.com/@gordicaleksa) and [**GitHub**](https://github.com/gordicaleksa)
4. Subscribe to my [**monthly AI newsletter**](https://aiepiphany.substack.com/) and join the [**Discord**](https://discord.gg/peBrCpheKE) community!
5. 📄 [**Website**](https://gordicaleksa.com/)

And if you find the content I create useful [**consider becoming a Patreon!**](https://www.patreon.com/theaiepiphany)

Much love ❤️

[Machine Learning](https://medium.com/tag/machine-learning?source=post_page---footer_tags--5c44017022ad-----------------------------------------)

[Large Language Models](https://medium.com/tag/large-language-models?source=post_page---footer_tags--5c44017022ad-----------------------------------------)

[Attention Is All You Need](https://medium.com/tag/attention-is-all-you-need?source=post_page---footer_tags--5c44017022ad-----------------------------------------)

[Mlsys](https://medium.com/tag/mlsys?source=post_page---footer_tags--5c44017022ad-----------------------------------------)

[![Aleksa Gordić](https://miro.medium.com/v2/resize:fill:96:96/1*q54RAGVbYTZtIBJfGC3goA.png)](https://gordicaleksa.medium.com/?source=post_page---post_author_info--5c44017022ad-----------------------------------------)

[![Aleksa Gordić](https://miro.medium.com/v2/resize:fill:128:128/1*q54RAGVbYTZtIBJfGC3goA.png)](https://gordicaleksa.medium.com/?source=post_page---post_author_info--5c44017022ad-----------------------------------------)

Follow

[**Written by Aleksa Gordić**](https://gordicaleksa.medium.com/?source=post_page---post_author_info--5c44017022ad-----------------------------------------)

[5K followers](https://gordicaleksa.medium.com/followers?source=post_page---post_author_info--5c44017022ad-----------------------------------------)

· [4 following](https://gordicaleksa.medium.com/following?source=post_page---post_author_info--5c44017022ad-----------------------------------------)

scaling LLMs. ex-DeepMind/Microsoft; [https://www.aleksagordic.com/](https://www.aleksagordic.com/)

Follow

[Help](https://help.medium.com/hc/en-us?source=post_page-----5c44017022ad-----------------------------------------)

[Status](https://status.medium.com/?source=post_page-----5c44017022ad-----------------------------------------)

[About](https://medium.com/about?autoplay=1&source=post_page-----5c44017022ad-----------------------------------------)

[Careers](https://medium.com/jobs-at-medium/work-at-medium-959d1a85284e?source=post_page-----5c44017022ad-----------------------------------------)

[Press](mailto:pressinquiries@medium.com)

[Blog](https://blog.medium.com/?source=post_page-----5c44017022ad-----------------------------------------)

[Store](https://medium.com/store)

[Privacy](https://policy.medium.com/medium-privacy-policy-f03bf92035c9?source=post_page-----5c44017022ad-----------------------------------------)

[Rules](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page-----5c44017022ad-----------------------------------------)

[Terms](https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----5c44017022ad-----------------------------------------)

[Text to speech](https://speechify.com/medium?source=post_page-----5c44017022ad-----------------------------------------)

reCAPTCHA

Recaptcha requires verification.

protected by **reCAPTCHA**