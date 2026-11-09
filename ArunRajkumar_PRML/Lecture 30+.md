
## Supervised Learning: Introduction & Linear Regression

### Types of supervised learning

Given data ${(x_1, y_1), \ldots, (x_n, y_n)}$ with $x_i \in \mathbb{R}^d$:

|Problem type|$y_i$ takes values in|Example|
|---|---|---|
|Binary classification|${+1, -1}$ or ${0,1}$|Spam detection|
|Multi-class classification|${0, 1, \ldots, K}$|Digit recognition|
|Regression|$\mathbb{R}$|Rainfall prediction|

---

### Linear Regression

**Goal:** learn $h: \mathbb{R}^d \to \mathbb{R}$ that predicts $y$ from $x$.

**Error measure:** squared error over training set:

$$\text{err}(h) = \sum_{i=1}^n (h(x_i) - y_i)^2$$

**Why not just minimise this over all functions?** Setting $h(x_i) = y_i$ for all $i$ achieves zero training error — but this is pure memorisation. It overfits: good on training data, useless on new inputs.

**Fix: restrict the search space.** Assume a linear relationship:

$$\mathcal{H}_{\text{linear}} = {h_w : x \mapsto w^\top x \mid w \in \mathbb{R}^d}$$

The problem becomes:

$$\min_{w \in \mathbb{R}^d} \sum_{i=1}^n (w^\top x_i - y_i)^2$$

This is **linear regression**.

---

### Closed-form solution

Stack data into matrix form: $X^\top \in \mathbb{R}^{n \times d}$ (rows = data points), $y \in \mathbb{R}^n$ (labels). Then:

$$\min_w |X^\top w - y|^2 = \min_w (X^\top w - y)^\top(X^\top w - y)$$

Take gradient and set to zero:

$$\nabla_w = 2XX^\top w - 2Xy = 0 \implies XX^\top w = Xy$$

$$\boxed{w^* = (XX^\top)^+ Xy}$$

where $(XX^\top)^+$ is the **pseudoinverse** (reduces to regular inverse when $XX^\top$ is full rank; otherwise gives the minimum-norm solution).

---

### Geometric interpretation

![](Pasted%20image%2020261109151037.png)

Think of each feature (e.g. height of all $n$ people) as an $n$-dimensional vector $F_k$. The label vector $y \in \mathbb{R}^n$ may not lie in the subspace spanned by the features.

**$w^_$ is the solution because $X^\top w^_$ is the orthogonal projection of $y$ onto $\text{span}{F_1, \ldots, F_d}$.**

The residual $y - X^\top w^*$ is perpendicular to every feature vector:

$$(y - X^\top w^_)^\top X^\top w^_ = 0$$

You can verify $w^* = (XX^\top)^+ Xy$ satisfies this. Special case: if $y$ already lies in the feature subspace, the projection is $y$ itself and training error is 0 — but this is _not_ overfitting, it means the data is perfectly linearly explainable.

---

### Four open questions

|Question|Status|
|---|---|
|Geometric meaning of $w^*$?|Answered — projection of labels onto feature subspace|
|Computational considerations?|Up next|
|Nonlinear feature-label relationships?|Up next|
|Probabilistic view / noise model?|Up next|