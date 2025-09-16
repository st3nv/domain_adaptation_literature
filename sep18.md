---
marp: true
theme: default
paginate: true
footer: 'Presented by [Tengyu Song](http://st3nv.github.io/)'
header: ''
style: |
    .columns {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap');
    body, p, h1, h2, h3, h4, h5, h6, li, blockquote, table, tr, td, th, a {
        font-family: 'Lato', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
    }
math: mathjax
---
<!-- paginate: skip -->

# Sep 10 presentation
## Lipton et al. (2018): Detecting and Correcting for Label Shift with Black Box Predictors (BBSE)

## Azizzadenesheli et al. (2020): Regularized Learning for Domain Adaptation under Label Shifts (RLLS)

<!-- -->

<style scoped>
  section{justify-content: center;}
</style>

<style>
img {
  display: block;
  margin: 0 auto 10px auto;
}
</style>



---
<!-- paginate: true -->
<!-- header: '' -->
<!-- footer: '' -->

## Notations and Problem Setup

- $x \in \mathcal{X}=\mathbb{R}^d$ denotes the features, $y \in \mathcal{Y}$ to denote the label variables. For simplicity, we assume that $\mathcal{Y}$ is a **discrete domain** equivalent to $\{1, 2, ..., k\}$.


- Data: $\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \ldots,\left(\boldsymbol{x}_n, y_n\right)$ drawn iid from a training (or source) distribution $P$, and $X^{\prime}=\left[\boldsymbol{x}_1^{\prime} ; \ldots ; \boldsymbol{x}_m^{\prime}\right]$ drawn iid from a test (or target) distribution $Q$. 


---

## Assumptions 

- A1: The label shift (also known as target shift) assumption

$$
p(\boldsymbol{x} \mid y)=q(\boldsymbol{x} \mid y) \quad \forall x \in \mathcal{X}, y \in \mathcal{Y} .
$$

- A2: For every $y \in \mathcal{Y}$ with $q(y)>0$ we require $p(y)>0.$

- A3: Access to a black box predictor $f: \mathcal{X} \rightarrow \mathcal{Y}$ where the expected confusion matrix $\mathbf{C}_p(f)$ is invertible.

$$
\mathbf{C}_P(f):=p(f(\boldsymbol{x}), y) \in \mathbb{R}^{|\mathcal{Y}| \times|\mathcal{Y}|}
$$

Note: Assumption A3 requires that the expected predictor outputs for each class be linearly independent. This assumption

---
## Idea 
Let $\hat{y}=f(\boldsymbol{x})$, where $f: \mathcal{X} \rightarrow \mathcal{Y}$ is a fixed function.

By the law of total probability and under assumption A1 (label shift) and A2 (common support)
$$
\begin{aligned}
q(\hat{y}) & =\sum_{y \in \mathcal{Y}} q(\hat{y} \mid y) q(y) \\
& =\sum_{y \in \mathcal{Y}} p(\hat{y} \mid y) q(y)=\sum_{y \in \mathcal{Y}} p(\hat{y}, y) \frac{q(y)}{p(y)} .
\end{aligned}
$$

- $p(\hat{y} \mid y)$ and $p(\hat{y}, y)$ can be estimated using $f$ and data from source distribution $P$, 

- $q(\hat{y})$ can be estimated with unlabeled test data drawn from target distribution $Q$. 



---
## BBSE: Black Box Shift Estimation
$$
\begin{array}{ll}
{\left[\boldsymbol{\nu}_y\right]_i=p(y=i)} & {\left[\boldsymbol{\mu}_y\right]_i=q(y=i)} \\
{\left[\boldsymbol{\nu}_{\hat{y}}\right]_i=p(f(\boldsymbol{x})=i)} & {\left[\boldsymbol{\mu}_{\hat{y}}\right]_i=q(f(\boldsymbol{x})=i)} \\
{\left[\hat{\boldsymbol{\nu}}_{\hat{y}}\right]_i=\frac{\sum_j \mathbb{1}\left\{f\left(\boldsymbol{x}_j\right)=i\right\}}{n}} & {\left[\hat{\boldsymbol{\mu}}_{\hat{y}}\right]_i=\frac{\sum_j \mathbb{1}\left\{f\left(\boldsymbol{x}_j^{\prime}\right)=i\right\}}{m}}
\end{array}
$$

and $[\boldsymbol{w}]_i=q(y=i) / p(y=i)$. Lastly define the covariance matrices $\mathbf{C}_{\hat{y}, y}, \mathbf{C}_{\hat{y} \mid y}$ and $\hat{\mathbf{C}}_{\hat{y}, y}$ in $\mathbb{R}^{k \times k}$ via

$$
\begin{aligned}
& {\left[\mathbf{C}_{\hat{y}, y}\right]_{i j}=p(f(\boldsymbol{x})=i, y=j)} \\
& {\left[\mathbf{C}_{\hat{y} \mid y}\right]_{i j}=p(f(\boldsymbol{x})=i \mid y=j)} \\
& {\left[\hat{\mathbf{C}}_{\hat{y}, y}\right]_{i j}=\frac{1}{n} \sum_l \mathbb{1}\left\{f\left(\boldsymbol{x}_l\right)=i \text { and } y_l=j\right\}}
\end{aligned}
$$

---

We can now rewrite Equation (1) in matrix form:

$$
\boldsymbol{\mu}_{\hat{y}}=\mathbf{C}_{\hat{y} \mid y} \boldsymbol{\mu}_y=\mathbf{C}_{\hat{y}, y} \boldsymbol{w}
$$


Using plug-in maximum likelihood estimates of the above quantities yields the estimators

$$
\hat{\boldsymbol{w}}=\hat{\mathbf{C}}_{\hat{y}, y}^{-1} \hat{\boldsymbol{\mu}}_{\hat{y}} \text { and } \hat{\boldsymbol{\mu}}_y=\operatorname{diag}\left(\hat{\boldsymbol{\nu}}_y\right) \hat{\boldsymbol{w}},
$$

The weight vector $\hat{\boldsymbol{w}}$ can be used to reweight the training data and obtain a consistent estimate of the target distribution $Q$.

--- 
## Theoretical Guarantees

The authors showed that the estimator performs well in high probability when the number of samples $n$ and $m$ are large.

Theorem (Error bounds). Assume that A3 holds robustly. Let $\sigma_{\min }$ be the smallest eigenvalue of $\mathbf{C}_{\hat{y}, y}$. There exists a constant $C>0$ such that for all $n>80 \log (n) \sigma_{\min }^{-2}$, with probability at least $1-3 \mathrm{kn}^{-10}-2 \mathrm{~km}^{-10}$ we have

$$
\begin{aligned}
\|\hat{\boldsymbol{w}}-\boldsymbol{w}\|_2^2 & \leq \frac{C}{\sigma_{\min }^2}\left(\frac{\|\boldsymbol{w}\|^2 \log n}{n}+\frac{k \log m}{m}\right) \\
\left\|\hat{\boldsymbol{\mu}}_y-\boldsymbol{\mu}_y\right\|^2 & \leq \frac{C\|\boldsymbol{w}\|^2 \log n}{n}+\left\|\boldsymbol{\nu}_y\right\|_{\infty}^2\|\hat{\boldsymbol{w}}-\boldsymbol{w}\|_2^2
\end{aligned}
$$

---

## Thoughts

- The method relies on **finite sample estimation** of confusion matrix $\mathbf{C}_{\hat{y}, y}$ and the distribution of predicted labels on target domain $\boldsymbol{\mu}_{\hat{y}}$, which can have **high variance** when $k$ is large and $n, m$ are small.
-  If the **classifier** $f$ is poor, the confusion matrix may be close to **singular** and estimation can be arbitrarily bad.
-  From the theorem, the error is **linear** in $k$.

---

## Regularized Learning for Domain Adaptation under Label Shifts
