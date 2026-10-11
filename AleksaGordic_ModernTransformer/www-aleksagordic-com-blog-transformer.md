[← Back to blog](https://www.aleksagordic.com/blog)

> Deep dive into the internals of a *modern dense transformer* [\[1\]](https://www.aleksagordic.com/blog/transformer#ref-1). Focus exclusively on the forward pass on a single GPU, as if we were about to perform a training step, while *ignoring the backward pass* and distributed systems details

> [Rnj 1.5](https://huggingface.co/EssentialAI/rnj-1.5-instruct) model is being used as a running example

## Transformer forward pass

As a running example, assume we sample 2 "documents" from a dataset, with:

- batch size = 1
- sequence length = 16
- document packing enabled

![](Pasted%20image%2020261011074441.jpg)

Figure 1: Tokenization stage

> *Tokenise, then pack the two documents into a single sequence*

> Good tokenizer has several desirable properties; for example, representing digits as individual tokens can help with numerical reasoning

- inputs positions - used by the positional embedding module (YaRN)
- segmentation mask - used in attention for masking

Next, we use the input tokens to *index into* the embedding table (get a sequence of 16 4096-dim `bf16` vectors)

> wonder why *index into* is used

![Figure 2: Embedding stage](https://www.aleksagordic.com/blog/transformer/flow_pro.png)

Figure 2: Embedding stage

> *Special tokens don't naturally appear during tokenization - no text maps to token IDs >= 128,000.* They're injected during training (and later used at inference) to improve performance (e.g. FIM, repo packing, etc.) or to enforce specific behaviors (e.g. end of generation / turn, tool calls)

### FIM [\[3\]](https://www.aleksagordic.com/blog/transformer#ref-3) (fill-in-the-middle) special tokens

During (pre)training, documents are split it into prefix, middle (infix), and suffix —>

```
<FIM_PRE> prefix <FIM_SUF> suffix <FIM_MID> middle
```

The model is trained to *predict the middle* given the prefix and suffix

For example, an IDE autocomplete - the cursor naturally splits the code into a prefix and suffix, with the middle missing. By inserting FIM tokens and ending with `<FIM_MID>`, you *prompt the model* to generate a completion for the gap.

### Transformer layer 

> In Rnj 1.5 each layer has its own learnable weights (!!)

> also note that some transformer implementations do weight sharing or partial weight sharing between layers (there are many variations) but here we're focusing on Rnj 1.5.

![Figure 3: Forward pass through transformer blocks](https://www.aleksagordic.com/blog/transformer/flow_main.png)

Figure 3: Forward pass through transformer blocks

> Not sure why *four RMSNorm* submodules - maybe just a variation

![](Pasted%20image%2020261011134453.jpg)

**NOTE USING CLAUDE:**
> You're right about the shape flow: n × d_model → linear → n × vocab. The "embedding table" in the note is that linear layer. With tied embeddings, the **output projection** *reuses* the **input embedding matrix instead of learning a separate one.** There's no extra embedding step here.

> **Why it works:** the input embedding matrix is (128256 × 4096), one 4096-dim row per token. At the input, you look up a token's row. *At the output, you multiply the final hidden vector (4096) by that same matrix transposed, giving 128256 numbers. Each logit is the dot product of the hidden vector with one token's embedding row, so it measures how well the model's output "points at" that token. The highest-scoring token is the one whose embedding is most aligned with the hidden state.*

> Why tie them: it saves about 525M parameters (128256 × 4096) by according a new table and ties the input and output meaning of each token together.

> One caveat: this diagram is for Rnj-1.5. Llama 3 8B and 70B do not tie embeddings; they have a separate learned output layer (lm_head). Only the smaller Llama 3.2 models (1B and 3B) tie them

## RMSNorm (Root Mean Square Layer Normalization)

![Figure 5: RMSNorm](https://www.aleksagordic.com/blog/transformer/RMSNorm.png)

Figure 5: RMSNorm

## GeGLU MLP (GELU-gated linear unit)

```
 `W2` @ GELU(W0@X)*(W1@X)`
```

![](Pasted%20image%2020261011143637.jpg)

Figure 6: GeGLU MLP

- ReLU - output is non negative
- GELU - gated outputs can be negative

## MHA / GQA

> Multiple queries (group) attend to the same key

![](Pasted%20image%2020261011161611.jpg)

Figure 7: MHA - multi head attention

## YaRN

But why do we need positional embeddings in the first place?
![](Pasted%20image%2020261011174145.jpg)

Figure 8: The WHY behind positional embeddings

> ***Only read till here for now. Will continue***

<hr>

<hr>

RoPE working:

![Figure 9: YaRN frequency table](https://www.aleksagordic.com/blog/transformer/yarn2.png)

Figure 9: YaRN frequency table

Here is a visualization showing how different YaRN frequencies behave. Notice that our slowest frequency does 1 cycle every 1.088M positions!

![Figure 10: YaRN frequencies](https://www.aleksagordic.com/blog/transformer/yarn3.png)

Figure 10: YaRN frequencies

> **Dotted is yarn!**

With this we're ready to see how positional embeddings are injected during the forward pass:

![Figure 11: YaRN - forward pass](https://www.aleksagordic.com/blog/transformer/yarn4.png)

Figure 11: YaRN - forward pass

That wraps up the YaRN forward pass.

Now that we understand the mechanics of it you might still be wondering: how does YaRN encode relative positional information via pairwise coordinate rotations of the query and key vectors, followed by a dot product?

![Figure 12: How does RoPE encode relative positional information?](https://www.aleksagordic.com/blog/transformer/yarn5.png)

Figure 12: How does RoPE encode relative positional information?

## Core Attention

Finally, let’s analyze the core attention mechanism. In practice, we use FlashAttention, which deserves a separate blog post (I actually wrote one back in ’23, [check it out](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) [\[11\]](https://www.aleksagordic.com/blog/transformer#ref-11)). Here, I’ll walk through vanilla attention.

Core attention is the mechanism that models relationships between tokens in a sequence. Take some time to analyze this:

![Figure 13: Computing (seqlen, seqlen) matrix of attention scores](https://www.aleksagordic.com/blog/transformer/attn1.png)

Figure 13: Computing (seqlen, seqlen) matrix of attention scores

Now if we just stopped here we'd have a situation where:

1. tokens from document 1 could attend to tokens from document 2 (and vice versa)
2. token `i` could attend to token `i+1` (future token) which breaks the causality

In order to prevent this we need to introduce masking!

![Figure 14: Attention masking & value vector aggregation](https://www.aleksagordic.com/blog/transformer/attn2.png)

Figure 14: Attention masking & value vector aggregation

Now imagine our sequence length is 32,768 instead of 16. For simplicity, assume a single document with no padding. What would the mask look like?

![Figure 15: Hybrid attention: block local + global](https://www.aleksagordic.com/blog/transformer/attn3.png)

Figure 15: Hybrid attention: block local + global

Here’s another way to visualize the layout, focusing on tokens at positions 9,000 and 10,000:

![Figure 16: Hybrid attention layout](https://www.aleksagordic.com/blog/transformer/attn4.png)

Figure 16: Hybrid attention layout

You can see that in most layers (block-local), these two tokens cannot attend to positions beyond 4,096. In the remaining eight layers (global), they can attend all the way back to position 0.

## Transformer math

Finally, I want to briefly touch on the KV cache, as it’s an extremely important concept for understanding inference. So far, we’ve looked at the forward pass during training.

During inference, transformers are autoregressive - we generate one token at a time. It would be extremely inefficient to recompute the keys and values for all previous tokens at every step. Fortunately, there’s no need: in a causal transformer, they remain unchanged. Instead, we compute them once and store them in a cache.

Let’s go through the basic KV cache storage requirements:

![Figure 17: KV cache calculation](https://www.aleksagordic.com/blog/transformer/calc1.png)

Figure 17: KV cache calculation

Let's now also calculate how many learnable parameters Rnj 1.5 has. We just need to go through the architecture and account for all the learnable weights.

![Figure 18: Number of learnable parameters calculation](https://www.aleksagordic.com/blog/transformer/calc2.png)

Figure 18: Number of learnable parameters calculation

**For quick mental math** notice that you only need to account for 3 matrices inside MLP and 4 matrices inside attention and you can ignore everything else.

Let's now calculate how much compute (FLOPs) we need per token. This is extremely valuable when it comes to planning cluster sizing - more on that after this section.

![Figure 19: FLOPs/token calculation](https://www.aleksagordic.com/blog/transformer/calc3.png)

Figure 19: FLOPs/token calculation

It's worth remembering the `6N` formula. It's also worth remembering the setting under which it holds (i.e. seqlen << inner model dimension).

Finally let's see how we can use the above formula for cluster sizing:

![Figure 20: Cluster sizing calculation](https://www.aleksagordic.com/blog/transformer/calc4.png)

Figure 20: Cluster sizing calculation

You now go to Masayoshi Son and ask for a $1B seed round.

![Figure 21: Profit](https://www.aleksagordic.com/blog/transformer/sp.png)

Figure 21: Profit

## Epilogue

We've seen how a single token flows through the transformer and how all the subcomponents work together.

We've explored YaRN and attention in depth, and derived some of the most important transformer formulas.


## References

1. Aleksa’s blog!