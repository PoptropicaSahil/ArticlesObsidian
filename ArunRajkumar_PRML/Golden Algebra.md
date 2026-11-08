> Tips for understanding the math

---

## For PCA derivation

> Remember $c$ is a scalar

$\Vert{}x - cw\Vert{}^2$ <br>
$= (x - cw)^\top (x - cw)$ <br>
$= (x^\top - cw^\top)(x - cw)$ <br>
$= x^\top x - x^\top(cw) - (cw^\top)x + (cw^\top)(cw)$ <br>
$= x^\top x - c(x^\top w) - c(w^\top x) + c^2(w^\top w)$

For vectors  $x^\top w = w^\top x = scalar$

$= x^\top x - 2c(x^\top w) + c^2(w^\top w)$ <br>

Derivative wrt variable $c$ and set it to zero

$⇒ -2(x^\top w) + 2c(w^\top w) = 0$ <br>
$⇒ c = \dfrac{x^\top w}{\Vert{}w\Vert{}^2}$
$⇒ c^* = x^\top w$ , if $w$ is a unit vector


> [!NOTE] Good to Remember
> Norm is a Dot Product: $\Vert{}v\Vert{}^2 = v^\top v$ <br>
>  $x^\top y = y^\top x$ <br>
>  $(AB)^\top = B^\top A^\top$ <br>
>  Scalars Float Freely: If c is a scalar number, $(cx)^\top y = c(x^\top y) = x^\top(cy)$


---

## More gradients 

> [!NOTE] **THE GOLDEN RULE OF MATRIX DERIVATIVES: SHAPE MATCHING**
> When you take the derivative (the gradient) of a scalar loss $L$ wrt vector $w$ → resulting gradient must have the exact same shape as $w$. If $w$ is a column vector, $\dfrac{\partial L}{\partial w}$ must also be a column vector

$\dfrac{\partial}{\partial w} (x^\top w) = x \dfrac{\partial}{\partial w} (w^\top x) = x$

$\dfrac{\partial}{\partial w} (w^\top A w) = 2Aw$ (Quadratic rule)


---

### Ordinary Least Squares (Linear Regression)

1. Input data matrix $X \in  \mathbb{R}^{n× d}$ (row convention)
2. Output vector $y  \in  \mathbb{R}^{n}$
3. Weight vector $w  \in  \mathbb{R}^{d}$
4. Loss (to minimise) $L = \Vert{}Xw - y\Vert{}^2$

> If it was column convention *(stack datapoints as columns)* then $X \in  \mathbb{R}^{d × n}$  and $L = \Vert{}X^\top w - y\Vert{}^2$


$L = (Xw - y)^\top (Xw - y)$ <br>
$L = (w^\top X^\top - y^\top) (Xw - y)$ <br>
$L = w^\top X^\top X w - w^\top X^\top y - y^\top X w + y^\top y$ <br>

> LHS scalar ⇒ RHS scalar ⇒ every term scalar ⇒ $w^\top X^\top y$ scalar and  $y^\top X w$ *is the same* scalar

$L = w^\top X^\top X w - 2 y^\top X w + y^\top y$

Derivative wrt variable $w$ and set it to zero

$⇒ 2 X^\top X w - 2 X^\top y = 0$ <br>
$⇒ X^\top X w = X^\top y$


---

## Covariance and Eigenvalues

1. **Covariance Matrix: "Shape" of Data**

Dataset with $n$ features ⇒ Covariance matrix $\Sigma \in \mathbb{R}^{n× n}$ 

- Diagonals: variance of each feature
- Off-Diagonals: (co-)variance with other features

2. **Eigenvectors: "Stubborn Skeleton" of Data**

> The word *eigen* is German for "own" or "characteristic" ⇒ Eigenvectors are vectors fundamentally characteristic to a specific matrix

Matrix usually rotates and stretches the vector into a new position. But for its special **eigenvectors** $v$, matrix cannot change their direction ($v$ can only get stretched or squished → by magnitude **eigenvalue**)

$\Sigma v = \lambda v$ ($\lambda$ is a scalar → eigenvalue)

3. **Eigenvalues: "Importance Score"**

Eigenvalue ($\lambda$) is exactly how much it (eigenvector)gets stretched.

> [!NOTE] Takeaway
> Eigenvalue = how much variance exists along that specific eigenvector's line

```
Putting It All Together: Why ML Cares (PCA)

Dataset with 100 features (a 100-dimensional space). Most of that space is probably empty, and the data is likely clustered in a shape that looks like a flat pancake tilted at a weird angle.

You want to compress this down to just 2 features to visualize it, but you don't want to lose the important information. Here is the exact recipe for PCA using what you just learned:

1. Calculate the Covariance Matrix: Figure out the exact multidimensional "shape" of your 100-feature data.
2. Find the Eigenvectors: Find the rigid 100 axes (the skeleton) that define the orientation of that shape.
3. Find the Eigenvalues: Look at the stretch factor for each axis.
4. Sort and Slice: Sort the eigenvectors by their eigenvalues from largest to smallest. The eigenvector with the absolute biggest eigenvalue is your Principal Component 1. It is the single line through your data that captures the most spread (information).

If you drop the 98 eigenvectors with the smallest eigenvalues and only keep the top 2, you have successfully compressed your data from 100 dimensions down to a 2D plane, while mathematically guaranteeing you preserved as much of the original data's shape as absolutely possible!
```

---


1. The Algebra of the Covariance Matrix

Dataset matrix $X$, where every row is a data point and every column is a feature
> (Assume we have centered the data so the mean is at the origin, zero)

$\Sigma = \dfrac{1}{n} X^\top X$

> [!NOTE] Looks familiar? 
> It is the same $X^\top X$ matrix from the Linear Regression derivative. <br>
> Algebraically, $X^\top X$ =  dot product of every feature column with every other feature column

2. The Algebra of Variance (The Quadratic Rule)

PCA ⇒ Reducing reconstruction error from projection (point $x$ onto a line $w$ gives scalar $c = x^\top w$)

PCA ⇒ **find the line $w$** that **maximizes the variance** of *all* our projected points

$\text{Variance} = w^\top \Sigma w$ 

Recollect $w^\top w = 1$ 

**Lagrange multipliers** → Maximise

$L = w^\top \Sigma w - \lambda(w^\top w - 1)$

$⇒ 2\Sigma w - 2\lambda w = 0$

$⇒ \Sigma w = \lambda w$

> *Mic Drop*

---

## Second derivatives needed?

1. The Matrix Second Derivative (The Hessian)

In matrix calculus, the second derivative is a matrix of second derivatives called the Hessian ($H$).

Variance function $f(w) = w^\top \Sigma w$ → second derivative wrt $w$ ⇒ $H = 2\Sigma$

 $\Sigma$ Covariance Matrix is **Positive Semi-Definite (PSD)**. By definition, a PSD matrix guarantees that $w^\top \Sigma w \ge 0$ for any vector $w$

> Geometrically, this means variance function is shaped like a multi-dimensional bowl opening upwards (it is **strictly convex**)

2. The Sphere Constraint

Lagrange multiplier:  $w$ is a unit vector $(\Vert{}w\Vert{}^2 = 1)$

> Geometrically, this means we are walking along a perfect circle (or sphere in higher dimensions) drawn inside that bowl, and we are looking for the highest point on that specific path.

3. The Ultimate Shortcut: Variance = Eigenvalue

$\text{Variance} = w^\top (\Sigma w)$

$\text{Variance} = w^\top (\lambda w)$ … (substitute main equation)

$\text{Variance} = \lambda (w^\top w) = \lambda$


> This is one of the most beautiful results in linear algebra

When you set the first derivative to zero, you don't just find one critical point; you find n critical points (the eigenvectors). No need of a second derivative test because, just look at your list of eigenvalues:

- The eigenvector with the largest $\lambda$ is the global maximum (Principal Component 1)
- The eigenvector with the smallest $\lambda$ is the global minimum
- Any eigenvectors in between are saddle points


---

## PCS reconstruction error

**Minimizing the reconstruction error is exactly the same as maximizing the variance**

$\Vert{}x_i - (x_i^\top w)w\Vert{}^2$ … (per-point reconstruction error)

$\Vert{}x - cw\Vert{}^2$ … (drop the subscript $i$ and $c=x^\top w$ )

$\Vert{}x - cw\Vert{}^2 = (x - cw)^\top (x - cw) = (x^\top - cw^\top) (x - cw)$

$= x^\top x - x^\top(cw) - (cw^\top)x + (cw^\top)(cw)$

$= x^\top x - c(x^\top w) - c(w^\top x) + c^2(w^\top w)$

$= \Vert{}x\Vert{}^2 - 2c(x^\top w) + c^2$

$= \Vert{}x\Vert{}^2 - 2(x^\top w)(x^\top w) + (x^\top w)^2$

$= \Vert{}x\Vert{}^2 - 2(x^\top w)^2 + (x^\top w)^2$

$= \Vert{}x\Vert{}^2 - (x^\top w)^2$

$= \frac{1}{n} \sum_{i=1}^n [\Vert{}x_i\Vert{}^2 - (x_i^\top w)^2]$

>  ML Optimization "Aha!" Moment

$f(w) = \text{Average}( \Vert{}x_i\Vert{}^2 ) - \text{Average}( (x_i^\top w)^2 )$ … (to minimise )

- $\Vert{}x_i\Vert{}^2$ is just the squared length of your original data points → constant
- $(x_i^\top w)^2$ is the squared length of the projections (the variance!)

$⇒ \arg\min_w (\text{Reconstruction Error}) \equiv \arg\max_w (\text{Variance})$


---

## Kernel trick

> [!NOTE] Convention switch
> Columns are data points here ⇒ $X \in \mathbb{R}^{d × n}$ and $C = \dfrac{1}{n}XX^\top$ ($d × d$) <br>
> Same as the column-convention note from OLS

**Why?** $d = 10{,}000$ features, $n = 500$ points ⇒ $C$ is $10{,}000 × 10{,}000$ → nightmare to eigen-decompose

**Trick:** solve an $n × n$ problem instead of $d × d$

---

### 1. Birth of $\alpha$ (Dual representation)

$Cw_k = \lambda_k w_k$ <br>
$⇒ \dfrac{1}{n}XX^\top w_k = \lambda_k w_k$ <br>
$⇒ w_k = X \left[ \dfrac{1}{n\lambda_k} X^\top w_k \right]$

Shape check inside bracket: $X^\top (n × d)$ times $w_k (d × 1)$ ⇒ $(n × 1)$ vector, times a scalar ⇒ still an $(n × 1)$ vector. Give it a name

$\alpha_k = \dfrac{1}{n\lambda_k} X^\top w_k$ <br>
$⇒ w_k = X\alpha_k$

> **Every eigenvector is a weighted mix (linear combination) of the data points.** $\alpha_k$ = the weights


### 2. Solve for $\alpha_k$ (Pre-multiply trick)

Substitute $w_k = X\alpha_k$ in the eigenvector equation

$\dfrac{1}{n}XX^\top (X\alpha_k) = \lambda_k (X\alpha_k)$ <br>
$⇒ XX^\top X\alpha_k = n\lambda_k X\alpha_k$

Annoying $X$ sitting at the front of both sides → **pre-multiply by $X^\top$**

$⇒ (X^\top X)(X^\top X)\alpha_k = n\lambda_k (X^\top X)\alpha_k$


### 3. Gram / Kernel matrix

$K = X^\top X \in \mathbb{R}^{n × n}$ ($500 × 500$, not $10{,}000 × 10{,}000$)

$K^2\alpha_k = n\lambda_k K\alpha_k$ <br>
$⇒ K\alpha_k = (n\lambda_k)\alpha_k$ (drop one $K$, fine for $\lambda_k \neq 0$) <br>
$⇒ \text{similar to~} Cw_k = \lambda_k w_k$

> [!NOTE] Takeaway
> $\alpha_k$ is an eigenvector of the small matrix $K$ (eigenvalue $n\lambda_k$) <br>
> 1. Build $K = X^\top X$ <br>
> 2. Eigen-decompose $K$ → $\alpha_k$ <br>
> 3. $w_k = X\alpha_k$ → true principal components

Normalisation (need $w_k$ unit vector)

$\Vert{}w_k\Vert{}^2 = \alpha_k^\top X^\top X\alpha_k = \alpha_k^\top K\alpha_k = n\lambda_k (\alpha_k^\top\alpha_k) = 1$ <br>
$⇒$ rescale $\alpha_k$ so that $\Vert{}\alpha_k\Vert{}^2 = \dfrac{1}{n\lambda_k}$

---

## Kernel PCA

Map data to a huge space: $x_i → \phi(x_i)$. Two matrices to keep apart

**1. Matrix we avoid: Covariance $C$**

$C = \dfrac{1}{n} \sum_{i=1}^n \phi(x_i)\phi(x_i)^\top$

Infinite-dimensional space ⇒ $\infty × \infty$ matrix → never computed

**2. Matrix we build: Kernel matrix $K$**

$K \in \mathbb{R}^{n × n}$ (data points, not features) <br>
$K_{ij} = \phi(x_i)^\top\phi(x_j) = k(x_i, x_j)$

> Kernel function $k$ = shortcut. Gives the exact high-dim dot product using only the original low-dim $x_i, x_j$ → never compute $\phi$ itself

Fill the $n × n$ grid with $k(x_i, x_j)$ ⇒ captured how the infinite-dim vectors relate to each other, without ever seeing them

---

### The steps

1. **Compute and center $K$**

$K^c_{ij} = K_{ij} - \theta_i - \theta_j + \bar{K}$ ($\theta_i$ = mean of row $i$, $\bar{K}$ = mean of all entries)

> Standard PCA needs mean = 0. This centers the data in high-dim space using only values of $K$

2. **Eigen-decompose $K^c$** → eigenvectors $\beta_k$ and eigenvalues (cheap, only $n × n$)

3. **Normalise** eigenvectors $\beta_k$ → $\alpha_k$ (same normalisation as above)

4. **Project** a point $x$ onto $w_k$

$w_k = \sum_j \alpha_{kj}\phi(x_j)$ <br>
$⇒ w_k^\top\phi(x) = \sum_j \alpha_{kj}\, k(x_j, x)$

> Only dot products needed → swap in the kernel function

---

### Reconstruction $\hat{x}_i$: Not possible

Going back needs

- exact coordinates of $w_k$ in the high-dim space
- inverse function $\phi^{-1}$

Never computed either. We only ever had dot products (overlaps)

> Fine for classification / clustering. Only the projected coordinates matter, not getting back the original data

