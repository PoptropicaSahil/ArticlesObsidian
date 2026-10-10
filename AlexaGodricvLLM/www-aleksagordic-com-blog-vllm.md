
## LLM Engine & Engine Core

LLM engine enables high-throughput inference - but only in an offline setting. You can't serve it to customers over the web yet

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

📝Environment vars:

- VLLM\_USE\_V1="1" # we're using engine V1
- VLLM\_ENABLE\_V1\_MULTIPROCESSING="0" # we're running in a single process

> This configuration is: offline, synchronous, single-GPU, using standard transformer


## LLM Engine constructor


The KV-cache manager maintains a `free_block_queue` \- a pool of available KV-cache blocks (often on the order of hundreds of thousands, depending on VRAM size and block size). **During paged attention, the blocks serve as the indexing structure that map tokens to their computed KV cache blocks.**

![LLM engine constructor](https://www.aleksagordic.com/blog/vllm/engine_constructor.png) 

> Block size for a standard transformer layer (non-MLA [\[4\]](https://www.aleksagordic.com/blog/vllm#ref-4)) is computed as follows:

2 (key/value) \* `block_size` (default=16) \* `num_kv_heads` \*  `head_size` \* `dtype_num_bytes` (e.g. 2 for bf16)


> Unless `--enforce-eager` is provided, for each of warmup batch sizes do a dummy run and capture CUDA graphs. **CUDA graphs record the whole sequence of GPU work into a DAG.** Later during fwd pass we launch/replay pre-baked graphs and cut on kernel launch overhead and thus improve latency.

## Generate function

The first step is to validate and feed requests into the engine. For each prompt we:

1. Create a unique request ID, capture its arrival time
2. Call an input preprocessor that tokenizes the prompt and returns a dictionary containing `prompt`, `prompt_token_ids`, and a `type` (text, tokens, embeds, etc.)
3. Pack this info into an `EngineCoreRequest`, adding priority, sampling params, and other metadata
4. Pass the request into the engine core, which wraps it in a `Request` object and sets its status to `WAITING`. This request is then added to the scheduler's `waiting` queue (append if FCFS, or heap-push if priority)

> For now — there's no mechanism to inject new requests mid-run, but **asynchronous engine** (aka **continuous batching** [\[6\]](https://www.aleksagordic.com/blog/vllm#ref-6)): after each step, both new and old requests are considered

**Because the forward pass flattens the batch into a single sequence and custom kernels handle it efficiently, continuous batching is fundamentally supported even in the synchronous engine.**

> since it’s paged attention I guess?!

Next, as long as there are requests to process, the engine repeatedly calls its `step()` function. Each step has three stages:

1. Schedule: select which requests to run in this step **(decode, and/or (chunked) prefill)**
2. Forward pass: run the model and sample tokens
3. Postprocess: append sampled token IDs to each `Request`, detokenize, and check stop conditions. *If a request is finished, clean up (e.g. return its KV-cache blocks to `free_block_queue`) and return the output early*


![Engine loop](https://www.aleksagordic.com/blog/vllm/engine_loop.png)

Engine loop.

## Scheduler

There are two main types of workloads an inference engine handles:

1. **Prefill** requests — a forward pass over all prompt tokens.**compute-bound** 
2. **Decode** requests — a forward pass to generate the most recent token. **memory-bandwidth-bound**, since we still need to load all LLM weights (and KV caches)

V1 scheduler: either prefill or decode 
V2 scheduler: both in same step


`allocate_slots`:

1. Determines how many new KV-cache blocks (`n`) must be allocated. Each block stores 16 tokens by default. For example, if a prefill request has 17 new tokens, we need `ceil(17/16) = 2` blocks
2. **Checks availability** — exit early if not enough blocks in the manager's pool; evict low-priority requests, or skip scheduling and continue execution
3. **Allocates blocks**  stores to `req_to_blocks`, the dictionary mapping each `request_id` to its list of KV-cache blocks

 > ***clearly the takeaway here!***

![KV cache blocks](https://www.aleksagordic.com/blog/vllm/kv_cache_blocks.png)

> ***CRUX — list of KV cache blocks***

## Run forward pass

We call model executor's `execute_model`, which delegates to the `Worker`, which in turn delegates to the model runner.

Here are the main steps:

1. **Update states** — prune finished requests from `input_batch`; update misc fwd pass related metadata *(e.g., KV cache blocks per request that will be used to index into paged KV cache memory).*
2. **Prepare inputs** — copy buffers from CPU→GPU etc
3. **Forward pass** — **run the model with custom paged attn kernels. All sequences are flattened and concatenated into one long "super sequence". Position indices and attention masks ensure each sequence only attends to its own tokens, which enables continuous batching without right-padding.**


Forward-pass step itself has two execution modes:

1. **Eager mode** — standard forward pass 
2. **"Captured" mode** — execute/replay the pre-captured CUDA Graph

Here is a concrete example that should make continuous batching and paged attention clear:

![[Pasted image 20261010143928.jpg]]

> Good goood

## Advanced Features — extending the core engine logic


## Chunked prefill

Chunked prefill is a technique for handling long prompts by splitting their prefill step into smaller chunks. Without it, we could end up with a single very long request monopolizing one engine step disallowing other prefill requests to run. That would postpone all other requests and increase their latency.

For example, let each chunk contain `n` (=8) tokens, labeled with lowercase letters separated by "-". A long prompt `P` could look like `x-y-z`, where `z` is an incomplete chunk (e.g. 2 toks). Executing the full prefill for `P` would then take ≥ 3 engine steps (> can happen if it's not scheduled for execution in one of the steps), and only in the last chunked prefill step would we sample one new token.

Here is that same example visually:

![Chunked prefilling - pt 1](https://www.aleksagordic.com/blog/vllm/chunked_pt1.png)

Implementation is straightforward: cap the number of new tokens per step. If the requested number exceeds `long_prefill_token_threshold`, reset it to exactly that value

> slightly sus

## Prefix Caching

To explain how prefix caching works, let's take the original code example and tweak it a bit:

```python
from vllm import LLM, SamplingParams

long_prefix = "<a piece of text that is encoded into more than block_size tokens>"

prompts = [\
    "Hello, my name is",\
    "The president of the United States is",\
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(long_prefix + prompts[0], sampling_params)
    outputs = llm.generate(long_prefix + prompts[1], sampling_params)

if __name__ == "__main__":
    main()
```

Prefix caching avoids recomputing tokens that multiple prompts share at the beginning - hence **prefix**

> ***Hence the caching!***

`long_prefix`: any prefix longer than a KV-cache block (16 tokens by default). Say `n x block_size` (where `n ≥ 1`) i.e. it perfectly aligns with block boundary - otherwise we'd have to recompute `long_prefix_len % block_size` tokens as we can't cache incomplete blocks.


With prefix caching, those `n x block_size` tokens are computed once (their KVs stored in KV cache paged memory) and then reused, so only the new prompt tokens need processing. This speeds up **prefill requests** 

> though it **doesn't help with decode**

How does this work in vLLM?

During the first `generate` call, in the scheduling stage, inside `kv_cache_manager.get_computed_blocks`, the engine invokes `hash_request_tokens`

1. This function splits the `long_prefix + prompts[0]` into 16-token chunks
2. For each complete chunk, it computes a hash (combining the previous block's hash, the current tokens, and optional metadata)
3. Each result is stored as a `BlockHash` object containing both the hash and its token IDs. We return a list of block hashes.

The list is stored in `self.req_to_block_hashes[request_id]`.

**Next, the engine calls `find_longest_cache_hit` to check if any of these hashes already exist in `cached_block_hash_to_block`.** On the first request, no hits are found.

![Prefix caching logic - pt 1](https://www.aleksagordic.com/blog/vllm/prefix_pt1.png)

New `BlockHash` entries with allocated KV blocks are recorded in `cached_block_hash_to_block` 

> *(as in the image above)*

***Forward pass will populate KVs in paged KV cache memory corresponding to KV cache blocks that we allocated above!!***

After many engine steps it'll allocate more KV cache blocks but it doesn't matter for our example because the prefix has diverged immediately after `long_prefix`.

![Prefix caching logic - pt 2](https://www.aleksagordic.com/blog/vllm/prefix_pt2.png)

On a second `generate` call with the same prefix, steps 1-3 repeat, but now `find_longest_cache_hit` finds matches for all `n` blocks (via linear search). The engine can reuse those KV blocks directly.

![Prefix caching logic - pt 3](https://www.aleksagordic.com/blog/vllm/prefix_pt3.png)

If the original request were still alive, the reference count for those blocks would increment (e.g. to 2). In this example, the first request has already completed, so the blocks were freed back to the pool and their reference counts set back to 0. Because we were able to retrieve them from `cached_block_hash_to_block` we know they're valid (the logic of the KV cache manager is setup in such a way), so we just remove them from `free_block_queue` again. (#revisit)


> **The gist of prefix caching: don't recompute prefixes you've already seen — just reuse their KV cache!**


## Guided Decoding (FSM)

At each decoding step, the logits are constrained by a grammar-based finite state machine. *only tokens allowed by the grammar can be sampled.*

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

prompts = [\
    "This sucks",\
    "The weather is beautiful",\
]

guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

In the toy example I gave (assume character-level tokenization): at **prefill**, the FSM masks logits so only "P" or "N" are viable. *If "P" is sampled, the FSM moves to the "Positive" branch; next step only "o" is allowed, and so on.*

![FSM](https://www.aleksagordic.com/blog/vllm/fsm.png)

Toy example FSM

> libraries like `xgrammar` compile the ask, vLLM does the scheduling, forward pass, making disallowed logits to  –∞ etc.



Further clarification on a bit masking step — 
If `vocab_size = 32`, `_grammar_bitmask` is a single integer; its *binary representation* encodes which tokens are allowed ("1") vs disallowed ("0"). For example, "101…001" expands to a length-32 array `[1, 0, 1, …, 0, 0, 1]`; **positions with 0 get logits set to –∞.** For larger vocabularies, multiple 32-bit words are used and expanded/concatenated accordingly. The backend (e.g., `xgrammar`) is responsible for producing these bit patterns using the current FSM state.

![FSM](https://www.aleksagordic.com/blog/vllm/fsm2.png)

Toy example


## Speculative Decoding

Steps:

1. **Draft:** run the small model on the current context and propose `k` tokens
2. **Verify:** run the large model once on context + `k` draft tokens. This produces probabilities for those `k` positions plus one extra (so we get `k+1` candidates) —> *potentially one free*
3. **Accept/reject:** going from left to right over the `k` draft tokens:
   - If the large model's probability for the draft token ≥ the draft's probability, accept it
   - Otherwise, accept it with probability `p_large(token)/p_draft(token)`
   - Stop at the first rejection, or accept all `k` draft tokens
   
     - If all `k` draft tokens are accepted, also sample the extra `(k+1)`-th token **"for free"** from the large model (we already computed that distribution)
     - If there was a rejection create a new rebalanced distribution at that position (`p_large - p_draft`, clamp min at 0, normalise to sum to 1) and sample the last token from it

> **YESSS!**

>> **Why this works:** Although we use the small model to propose candidates, the accept/reject rule guarantees that in expectation the sequence is distributed exactly as if we had sampled token by token from the large model. This means speculative decoding is statistically equivalent to standard autoregressive decoding — but potentially much faster, **since a single large-model pass can yield up to `k+1` tokens**

> Alexa recommends looking at [gpt-fast](https://github.com/meta-pytorch/gpt-fast) for a simple implementation

vLLM V1 does not support the LLM draft model method, instead it implements faster—but less accurate—proposal schemes: n-gram, EAGLE [\[9\]](https://www.aleksagordic.com/blog/vllm#ref-9), and Medusa [\[10\]](https://www.aleksagordic.com/blog/vllm#ref-10)

One-liners on each:

1. **n-gram:** take the last `prompt_lookup_max` tokens; find a prior match in the sequence; if found, propose the `k` tokens that followed that match; otherwise decrement the window and retry down to `prompt_lookup_min` ( sus but okay)

The current implementation returns `k` tokens after the **first** match. It feels more natural to introduce a recency bias and reverse the search direction? (i.e. last match)

3. **Eagle:** keep embeddings and LM head of large LM, replace the transformer stack with a lightweight MLP; fine-tune that as a cheap draft
4. **Medusa:** train auxiliary linear heads on top (embeddings before LM head) of the large model to predict the next `k` tokens in parallel; use these heads to propose tokens more efficiently than running a separate small LM 

> (!!)

Here's how to invoke speculative decoding in vLLM using `ngram` as the draft method:

```python
from vllm import LLM, SamplingParams

prompts = [\
    "Hello, my name is",\
    "The president of the United States is",\
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config={
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", speculative_config=speculative_config)

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```


> The best way to internalise this is to fire up your debugger and step through the codebase, but this section hopefully gives you a taste for it. This as well:

![[Pasted image 20261010213332.jpg]]
![[Pasted image 20261010213351.jpg]]

## Disaggregated P/D

> *haven’t read this section nicely*

I've already previously hinted at the motivation behind disaggregated P/D (prefill/decode).

Prefill and decode have very different performance profiles (compute-bound vs. memory-bandwidth-bound), so separating their execution is a sensible design. It gives tighter control over latency — both `TTFT` (time-to-first-token) and `ITL` (inter-token latency) — more on this in the [benchmarking](https://www.aleksagordic.com/blog/vllm#cpt5) section.

How does this work in vLLM?

For clarity, the example below relies on `SharedStorageConnector`, a debugging connector implementation used to illustrate the mechanics.

Connector is vLLM's abstraction for handling the exchange of KVs between instances. Connector interface is not yet stable, there are some near-term improvements planned which will involve changes, some potentially breaking.

We launch 2 vLLM instances (GPU 0 for prefill and GPU 1 for decode), and then transfer the KV cache between them:

```python

import os
import time
from multiprocessing import Event, Process
import multiprocessing as mp

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

prompts = [\
    "Hello, my name is",\
    "The president of the United States is",\
]

def run_prefill(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)
  llm.generate(prompts, sampling_params)

  prefill_done.set()  # notify decode instance that KV cache is ready

  # To keep the prefill node running in case the decode node is not done;
  # otherwise, the script might exit prematurely, causing incomplete decoding.
  try:
      while True:
          time.sleep(1)
  except KeyboardInterrupt:
      print("Script stopped by user.")

def run_decode(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"

  sampling_params = SamplingParams(temperature=0, top_p=0.95)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)

  prefill_done.wait()  # block waiting for KV cache from prefill instance

  # Internally it'll first fetch KV cache before starting the decoding loop
  outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
  prefill_done = Event()
  prefill_process = Process(target=run_prefill, args=(prefill_done,))
  decode_process = Process(target=run_decode, args=(prefill_done,))

  prefill_process.start()
  decode_process.start()

  decode_process.join()
  prefill_process.terminate()
```

📝Note:

I've also experimented with `LMCache` [\[11\]](https://www.aleksagordic.com/blog/vllm#ref-11), the fastest production-ready connector (uses NVIDIA's NIXL as the backend), but it's still at the bleeding edge and I ran into some bugs. Since much of its complexity lives in an external repo, `SharedStorageConnector` is a better choice for explanation.

These are the steps in vLLM:

1. **Instantiation**— During engine construction, connectors are created in two places:
   - Inside the worker's init device procedure (under init worker distributed environment function), with role "worker".
   - Inside the scheduler constructor, with role "scheduler".
2. **Cache lookup** — When the scheduler processes prefill requests from the `waiting` queue (after local prefix-cache checks), it calls connector's `get_num_new_matched_tokens`. This checks for externally cached tokens in the KV-cache server. Prefill always sees 0 here; decode may have a cache hit. The result is added to the local count before calling `allocate_slots`.
3. **State update** — The scheduler then calls `connector.update_state_after_alloc`, which records requests that had a cache (no-op for prefill).
4. **Meta build** — At the end of scheduling, the scheduler calls `meta = connector.build_connector_meta`:
   - Prefill adds all requests with `is_store=True` (to upload KV).
   - Decode adds requests with `is_store=False` (to fetch KV).
5. **Context manager**— Before the forward pass, the engine enters a KV-connector context manager:
   - On enter: `kv_connector.start_load_kv` is called. For decode, this loads KV from the external server and injects it into paged memory. For prefill, it's a no-op.
   - On exit: `kv_connector.wait_for_save` is called. For prefill, this blocks until KV is uploaded to the external server. For decode, it's a no-op.

Here is a visual example:

![disaggregated P/D](https://www.aleksagordic.com/blog/vllm/pd.png)

disaggregated P/D

📝Additional notes:

- For `SharedStorageConnector` "external server" is just a local file system.
- Depending on configuration, KV transfers can also be done layer-by-layer (before/after each attention layer).
- Decode loads external KV only once, on the first step of its requests; afterwards it computes/stores locally.

> there are other deployment-specific notes in the main blog which are skipped here. Alexa is a genius to be knowing the details of everything!




## References

1. Alexa Goodrich’s blog on vLLM