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

# Oct 30 update


---

<!-- paginate: true -->
<!-- header: '' -->
<!-- footer: '' -->

## Renyi divergence calculation (fixed code)

- Rényi divergence for 2D continuous distributions

The formula is given by

$$
D_\alpha(P \| Q)=\frac{1}{\alpha-1} \log \int_{\mathbb{R}^2} p(x, y)^\alpha q(x, y)^{1-\alpha} d x d y
$$


For Monte Carlo estimation:

$$
D_\alpha(P \| Q) \approx \frac{1}{\alpha-1} \log \frac{1}{n} \sum_{i=1}^n \frac{p\left(x_i, y_i\right)^{\alpha-1}}{q\left(x_i, y_i\right)^{\alpha-1}}, \quad\left(x_i, y_i\right) \sim P
$$

---

- If $y$ is binary, then the joint distribution $p(x, y)$ can be decomposed as $p(y|x)p(x)$, and we have

$$
D_\alpha(P \| Q)=\frac{1}{\alpha-1} \log \sum_y p(y)^\alpha q(y)^{1-\alpha} \int p(x \mid y)^\alpha q(x \mid y)^{1-\alpha} d x .
$$

- In implementation, we use gaussian_kde from scipy to estimate the density functions and sample from the estimated distributions to compute the divergence.