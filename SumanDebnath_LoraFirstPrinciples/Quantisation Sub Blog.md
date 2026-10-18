
> The quantisation section from the LoRA blog, added here for ease of reading

# Quantization: Doing More with Fewer Bits

## [How Numbers Live in Memory](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#how-numbers-live-in-memory)

Every parameter in our model is a `floating-point` number, and a `floating-point` number is just a fixed budget of bits split three ways:

![FP32, FP16, BF16, INT8 bit layouts](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/quantization_formats.svg)


|Format|Bits|Exponent|Mantissa|Max value|Personality|
|---|---|---|---|---|---|
|**FP32**|32|8|23|~3.4e38|The gold standard, big range _and_ fine precision|
|**FP16**|16|5|10|**65,504**|Precise but _fragile_, tiny range|
|**BF16**|16|8|7|~3.4e38|*FP32’s range, chunky precision*|

> That FP16 max value of **65,504** is not a footnote, it’s a *landmine*.

```py
x = torch.tensor(60000.0, dtype=torch.float16)
print(x * 2)          # 120000 > 65504, past FP16's ceiling
# tensor(inf, dtype=torch.float16)
```

With BF16:

```py 
x = torch.tensor(60000.0, dtype=torch.bfloat16)
print(x * 2)          #  BF16 has FP32's exponent range
# tensor(119808., dtype=torch.bfloat16)
```

> Nice!


Wait, 60000 × 2 is 120000, so where did **119808** come from ? This one number demonstrates both sides of the trade. With only 7 mantissa bits, BF16 can’t represent 60000 exactly: the nearest values it _can_ store *AT THAT MAGNITUDE* are 256 apart, so 60000 gets rounded down to 59904 the moment we create the tensor. Doubling that gives 119808. So BF16 survived the multiplication that killed FP16 (range), but it couldn’t even hold our input exactly (precision).

And that’s precisely the trade we want for training. FP16 silently overflows to `inf`, and one `inf` in a forward pass poisons everything downstream (this is exactly why FP16 training needs `loss scaling`, while BF16 mostly doesn’t). A rounding error of 96 in 60000, on the other hand, is the kind of small, well-behaved noise neural networks shrug off. This is why **BF16 has become the default training dtype** on modern GPUs, it trades precision we can usually afford to lose for range we absolutely cannot.

**Note**
Mixed precision is NOT quantization

It’s easy to conflate the two, both involve “using fewer bits”. **Mixed precision** (`torch.autocast`) *`keeps the weights in FP32 but runs the math in BF16`* where it’s safe, it’s a speed optimization. We can verify this ourselves:

```py
import time
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2000, 2000)
        self.b = nn.Linear(2000, 2000)

    def forward(self, x):
        return self.b(self.a(x))

def timeit_forward(model, x, n=200):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n):
        model(x)
    torch.cuda.synchronize()
    return (time.time() - start) / n * 1e6

model = TinyModel().to("cuda")                      # weights stay FP32 the whole time
x = torch.randn(2000, 2000, device="cuda")

print(f"Plain FP32 forward pass: {timeit_forward(model, x):.0f} microseconds")

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    print(f"FP32 model under BF16 autocast: {timeit_forward(model, x):.0f} microseconds")

print(f"Model weights are still: {model.a.weight.dtype}")
```

**Output:**

```
Plain FP32 forward pass:        1645 microseconds
FP32 model under BF16 autocast:  996 microseconds
Model weights are still:        torch.float32
```

The weights never changed, only the compute did. **Quantization**, which we do next, changes the _storage format of the weights themselves_, it’s a _memory_ optimization. LoRA fine-tuning will end up using both.




---
---



## [What Quantization Actually Does](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#what-quantization-actually-does)

The core idea of quantization is simple: **snap every weight to the nearest point on a coarse grid, and remember the grid.**

> Remember the grid to dequantise (!!)

![Animation: continuous weights snapping to discrete quantization bins](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/quantization_bins.gif)

Let’s build the standard **absmax INT8** quantizer by hand. An INT8 holds integers from −128 to 127; we use the symmetric range \[−127, 127\] so one positive scale covers both sides. The recipe: find the biggest weight (in absolute value), stretch it to 127, and scale everything else proportionally:

![](Pasted%20image%2020261017074945.png)

Where:

- max⁡(\|𝑊\|): the largest magnitude in the tensor
- scale: the single FP32 number we must remember to undo the mapping
- 𝑊int8: the weights, now 1 byte each instead of 4


```py
def quantize_absmax_int8(x):
    scale = x.abs().max() / 127
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.to(torch.float32) * scale

weight = torch.randn(768, 768)
q, scale = quantize_absmax_int8(weight)
restored = dequantize(q, scale)

error = (weight - restored).pow(2).mean().item()
print("Original size (in bytes):", weight.numel() * weight.element_size())
print("Quantized size (in bytes):", q.numel() * q.element_size())
print("Mean Squared Error:", error)
```

**Output:**

```
Original size (in bytes):  2359296
Quantized size (in bytes): 589824
Mean Squared Error:        0.00011198947322554886
```

> **4x smaller, with an error in the 4th decimal place.** But neural networks turn out to be remarkably tolerant of this kind of *uniform, small noise.*

## [The Outlier Problem](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#the-outlier-problem)

The absmax scheme has one weakness: the scale is computed from the **single largest magnitude** in the tensor. If one row of the matrix is an outlier, it forces the scale up for every other value too, and the *resolution available to the majority of the weights collapses.*

**Real transformer weight matrices contain exactly this kind of outlier**, systematically: [Dettmers et al., 2022](https://arxiv.org/abs/2208.07339) **found that a small number of feature dimensions in large models carry disproportionately large values, and built `LLM.int8()` around that observation.**

> Crazy crazy

Let’s simulate an outlier channel, and quantize it two ways:
1. with a single scale for the whole tensor (`per-tensor`)
2. with a separate scale per row (`per-channel`)

```py
def quantize_per_channel(x):
    scale = x.abs().amax(dim=1, keepdim=True) / 127
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize_per_channel(q, scale):
    return q.to(torch.float32) * scale

weight_with_outlier = weight.clone()
weight_with_outlier[3] *= 50          # one row now behaves like an outlier channel

q_tensor, scale_tensor = quantize_absmax_int8(weight_with_outlier)
q_channel, scale_channel = quantize_per_channel(weight_with_outlier)

error_tensor = (weight_with_outlier - dequantize(q_tensor, scale_tensor)).pow(2).mean().item()
error_channel = (weight_with_outlier - dequantize_per_channel(q_channel, scale_channel)).pow(2).mean().item()

print("per tensor scale, mean squared error:", error_tensor)
print("per channel scale, mean squared error:", error_channel)
```

**Output:**

```
per tensor scale, mean squared error: 0.21627412736415863
per channel scale, mean squared error: 0.00033763001556508243
```

![Per-tensor vs per-channel quantization error with an outlier row](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/quant_outlier_mse.png)

This is why *production quantization schemes use per-channel or per-block scales* (one scale per small group of values) instead of one scale for an entire weight matrix: an outlier can then only affect its own block.


## [Going to 4 Bits: NF4](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#going-to-4-bits-nf4)

 **4 bits** i.e. just **16 distinct values** per weight. *With 16 levels, where we place them suddenly matters a lot.* A grid that ignores the data distribution, wastes levels where almost no weights live.

But neural network weights are approximately **normally distributed**, most weights are near zero, few are large. The **NF4 (NormalFloat4)** format, introduced in the [QLoRA paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314), places its 16 levels at the **quantiles of a Gaussian** so dense levels near zero get where the crowd is, sparse levels in the tails

![FP4 vs NF4 quantization levels against a normal distribution](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/assets/fp4_vs_nf4_levels.svg)

> **TIP** Double Quantization: compressing the compression

Block-wise quantization has a hidden cost: **every block** of 64 weights **needs its own FP32 scale**, that’s 32/64=0.5 extra bits per parameter. The QLoRA authors’ fix is delightfully recursive: **quantize the scales themselves** (`double quantization`), saving about **0.4 bits per parameter**. When you have 8 billion parameters, 0.4 bits each is ~400 MB, real money.

> Recursive? One step recursive yes. Real money oh yessss

## [Quantizing a Real Model](https://debnsuma.github.io/my-blog/posts/lora-serverless-fine-tuning/\#quantizing-a-real-model)

In practice [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) implements block-wise NF4 quantization (including double quantization)

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

model_bf16 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to("cuda")
memory_bf16 = torch.cuda.memory_allocated() / 1e6
print(f"BF16 memory used: {memory_bf16:.0f} MB")

del model_bf16
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 1. store weights in 4-bit
    bnb_4bit_quant_type="nf4",              # 2. use NF4, not FP4 (default is fp4!)
    bnb_4bit_use_double_quant=True,         # 3. quantize the quantization constants
    bnb_4bit_compute_dtype=torch.bfloat16,  # 4. dequantize to BF16 for the actual math
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,          # dtype for the layers that do NOT get quantized
    device_map={"": 0},
)
memory_4bit = torch.cuda.memory_allocated() / 1e6

print(f"4 bit memory used: {memory_4bit:.0f} MB")
print(f"Reduction: {memory_bf16 / memory_4bit:.2f}x")
```

**Output:**

```
BF16 memory used: 1453 MB
4 bit memory used: 643 MB
Reduction: 2.26x
```

***`IMPORTANT`*** 

The most important line in the `bnb_config` is `bnb_4bit_compute_dtype=torch.bfloat16`. 4-bit is a **storage** format, not a compute format, ***no GPU multiplies NF4 numbers directly.*** ***Each layer dequantizes its weights to BF16 on the fly, does the matmul in BF16, and throws the BF16 copy away.*** We pay some compute (typically a 10–30% slowdown) to save a lot of memory.

***Note***

Wait, why only 2x and not 4x ?

Good catch! If we peek inside, **only the `nn.Linear` layers became 4-bit:**

```py
print(model_4bit.model.layers[0].self_attn.q_proj)
# Linear4bit(in_features=896, out_features=896, bias=True)
```

*The embedding table and output head stay in BF16*, and on a _small_ 0.5B model those make up a big share of the parameters. On an 8B+ model, linear layers dominate and the reduction gets much closer to the ideal ~4x

> **A nice reminder that memory math always depends on the _shape_ of the model, not just its size**.

So we can now store the linear layers, the bulk of any large model, in a quarter of the memory. But storage was only half the bill, if we fine-tune this model the old way, the gradients and optimizer states come roaring back at full size. Fewer _bits_ per parameter is not enough. We need fewer _trainable parameters_. Enter LoRA.
