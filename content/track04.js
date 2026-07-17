/* ============================================================
   TRACK 4 — Linear Algebra III — Norms, Distances & Inverses
   4.1 Vector Norms · 4.2 Matrix Norms · 4.3 Inverse & Pseudo-inverse ·
   4.4 Solving Ax=b, LU & Conditioning · 4.E Track Exam
   ============================================================ */
(window.LESSON_CONTENT ||= {})["4.1"] = {
  subtitle: "How big is a vector? Different answers give L1 sparsity and L2 weight decay.",

  aiMoment: String.raw`<p>Add $\lambda\lVert\mathbf w\rVert_2^2$ to a loss and you get <strong>weight decay</strong> — weights
  shrink toward zero but stay nonzero. Add $\lambda\lVert\mathbf w\rVert_1$ instead and you get <strong>Lasso</strong> —
  many weights become <em>exactly</em> zero, a sparse model. The only difference is which "norm" measures the size of
  $\mathbf w$. Norms also define distance in embedding space and the threshold in gradient clipping. Picking a norm is a
  modeling decision with real consequences.</p>`,

  plainEnglish: String.raw`<p>A <strong>norm</strong> is a way to measure the length, or size, of a vector — one number
  saying "how big." There's more than one sensible way to measure: straight-line distance (L2), total city-block
  distance (L1), or just the single biggest component (L∞).</p>`,

  intuition: String.raw`<p>Draw all vectors of "size 1" under each norm and you get a different shape — the
  <strong>unit ball</strong>. L2 is a round circle; L1 is a diamond with corners poking out along the axes; L∞ is a
  square. Those L1 corners, sitting exactly on the axes, are the geometric reason L1 produces sparse (axis-aligned)
  solutions.</p>
  <figure class="figure">
  <svg viewBox="0 0 360 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Unit balls of the L2, L1, and Linfinity norms">
    <g stroke="#cbd5e1"><line x1="20" y1="70" x2="100" y2="70"/><line x1="60" y1="30" x2="60" y2="110"/></g>
    <circle cx="60" cy="70" r="36" fill="#eef0ff" stroke="#4f46e5" stroke-width="2"/>
    <text x="38" y="135" font-size="12" fill="#4f46e5" font-family="sans-serif">L2 (circle)</text>
    <g stroke="#cbd5e1"><line x1="140" y1="70" x2="220" y2="70"/><line x1="180" y1="30" x2="180" y2="110"/></g>
    <polygon points="180,34 216,70 180,106 144,70" fill="#f0fdfa" stroke="#0d9488" stroke-width="2"/>
    <text x="158" y="135" font-size="12" fill="#0d9488" font-family="sans-serif">L1 (diamond)</text>
    <g stroke="#cbd5e1"><line x1="260" y1="70" x2="340" y2="70"/><line x1="300" y1="30" x2="300" y2="110"/></g>
    <rect x="266" y="36" width="68" height="68" fill="#fff7ed" stroke="#d97706" stroke-width="2"/>
    <text x="280" y="135" font-size="12" fill="#d97706" font-family="sans-serif">L∞ (square)</text>
  </svg>
  <figcaption>Same "size 1," three shapes. L1's corners lie on the axes — that's where sparse solutions land.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $\mathbf x\in\mathbb{R}^n$:</p>
  $$\lVert\mathbf x\rVert_1=\sum_i|x_i|,\qquad \lVert\mathbf x\rVert_2=\sqrt{\sum_i x_i^2},\qquad
  \lVert\mathbf x\rVert_\infty=\max_i|x_i|,\qquad \lVert\mathbf x\rVert_p=\Big(\sum_i|x_i|^p\Big)^{1/p}.$$
  <p>Every norm satisfies: nonnegativity (zero only for $\mathbf 0$), scaling $\lVert c\mathbf x\rVert=|c|\lVert\mathbf x\rVert$,
  and the triangle inequality $\lVert\mathbf x+\mathbf y\rVert\le\lVert\mathbf x\rVert+\lVert\mathbf y\rVert$.
  <strong>Distance</strong> between points is the norm of their difference, $\lVert\mathbf x-\mathbf y\rVert$.</p>`,

  derivation: String.raw`<p><strong>Why L2 shrinks but L1 zeroes — from the gradients.</strong> Add a penalty to the loss and
  take a gradient-descent step.</p>
  <p><strong>L2 penalty</strong> $\tfrac12\lambda\lVert\mathbf w\rVert_2^2=\tfrac12\lambda\sum_i w_i^2$. Its gradient is
  $\lambda\mathbf w$, so the penalty's step is $w_i\leftarrow w_i-\eta\lambda w_i=(1-\eta\lambda)w_i$.</p>
  <p><strong>Plain English:</strong> L2 multiplies each weight by a factor just below 1 — it shrinks proportionally and
  <em>never reaches exactly 0</em> (halving repeatedly never hits zero).</p>
  <p><strong>L1 penalty</strong> $\lambda\lVert\mathbf w\rVert_1=\lambda\sum_i|w_i|$. Its (sub)gradient is
  $\lambda\,\operatorname{sign}(w_i)$, so the step is $w_i\leftarrow w_i-\eta\lambda\,\operatorname{sign}(w_i)$.</p>
  <p><strong>Plain English:</strong> L1 subtracts a <em>constant</em> $\eta\lambda$ toward zero regardless of how small
  $w_i$ is. Once a weight is within that constant of 0, the step pushes it to <em>exactly</em> 0 and it stays — that's
  sparsity. $\blacksquare$ Geometrically: minimizing the loss subject to an L1 budget, the diamond's corner (on an axis,
  where coordinates are 0) is the first point the loss contour touches.</p>`,

  code: [
    { label: "The three norms", src: String.raw`
import numpy as np

x = np.array([3.0, -4.0, 0.0, 1.0])
print("L1   =", np.linalg.norm(x, 1))        # 8.0
print("L2   =", np.linalg.norm(x, 2))        # 5.099
print("Linf =", np.linalg.norm(x, np.inf))   # 4.0

# distance between two points = norm of the difference
a, b = np.array([1.,2.]), np.array([4.,6.])
print("L2 distance =", np.linalg.norm(a - b))   # 5.0  (a 3-4-5 triangle)
` },
    { label: "L2 shrinks, L1 zeroes", src: String.raw`
import numpy as np

w_l2 = np.array([1.0]); w_l1 = np.array([1.0])
lr, lam = 0.1, 0.5
for _ in range(40):
    w_l2 = (1 - lr*lam) * w_l2                       # L2: multiplicative shrink
    w_l1 = w_l1 - lr*lam*np.sign(w_l1)               # L1: constant push toward 0
    w_l1 = np.where(np.abs(w_l1) < lr*lam, 0.0, w_l1)  # soft-threshold to exactly 0
print("L2 weight after 40 steps:", w_l2)   # tiny but NONZERO
print("L1 weight after 40 steps:", w_l1)   # exactly 0.0  -> sparse
` }
  ],

  keyPoints: [
    "A norm measures vector size; distance is the norm of the difference.",
    "L2 = straight-line $\\sqrt{\\sum x_i^2}$; L1 = city-block $\\sum|x_i|$; L∞ = max $|x_i|$.",
    "L2 regularization (weight decay) shrinks weights proportionally, never to exactly 0.",
    "L1 regularization subtracts a constant, driving small weights to exactly 0 → sparsity.",
    "Unit balls: L2 circle, L1 diamond (corners on axes), L∞ square."
  ],

  commonMistakes: [
    { wrong: "Believing L2 regularization produces sparse weights.", why: "L2 shrinks toward 0 but never reaches it; you get many small nonzero weights. For exact zeros (feature selection) you need L1." },
    { wrong: "Forgetting to take absolute values in L1.", why: "$\\lVert\\mathbf x\\rVert_1=\\sum|x_i|$, not $\\sum x_i$. Signs would let positives and negatives cancel, which a size measure must not do." },
    { wrong: "Comparing raw L2 norms across different dimensions.", why: "L2 norm grows with dimension for random vectors ($\\sim\\sqrt n$); to compare sizes fairly, normalize or use a mean-based measure." }
  ],

  quiz: [
    { q: "$\\lVert[3,-4]\\rVert_2 = ?$", options: ["5", "7", "1", "12"], answer: 0,
      explain: "$\\sqrt{9+16}=\\sqrt{25}=5$. Choice 7 is the L1 norm $|3|+|-4|$." },
    { q: "$\\lVert[1,-6,2]\\rVert_\\infty = ?$", options: ["6", "9", "3", "2"], answer: 0,
      explain: "L∞ is the largest absolute component: $\\max(1,6,2)=6$." },
    { q: "Which regularizer drives weights to exactly zero?", options: ["L1", "L2", "L∞", "Frobenius"], answer: 0,
      explain: "L1's constant-size gradient push reaches 0; L2 only shrinks proportionally." },
    { q: "$\\lVert[3,-4]\\rVert_1 = ?$", options: ["7", "5", "1", "12"], answer: 0,
      explain: "$|3|+|-4|=7$ (city-block). The straight-line L2 distance would be 5." },
    { q: "For a fixed vector, order the norms by size (generally).", options: ["$\\lVert x\\rVert_\\infty\\le\\lVert x\\rVert_2\\le\\lVert x\\rVert_1$", "$\\lVert x\\rVert_1\\le\\lVert x\\rVert_2\\le\\lVert x\\rVert_\\infty$", "they're always equal", "$\\lVert x\\rVert_2$ is always largest"], answer: 0,
      explain: "The max component ≤ Euclidean ≤ sum of magnitudes: $\\lVert x\\rVert_\\infty\\le\\lVert x\\rVert_2\\le\\lVert x\\rVert_1$." }
  ],

  practice: [
    { level: "easy", prompt: "Compute all three norms of $[2,-2,1]$.", solution: "L1 $=2+2+1=5$; L2 $=\\sqrt{4+4+1}=3$; L∞ $=\\max(2,2,1)=2$." },
    { level: "easy", prompt: "What is the L2 distance between $[0,0]$ and $[6,8]$?", solution: "$\\lVert[6,8]\\rVert_2=\\sqrt{36+64}=\\sqrt{100}=10$." },
    { level: "med", prompt: "Show $\\lVert\\mathbf x\\rVert_\\infty\\le\\lVert\\mathbf x\\rVert_2$.", solution: "Let $M=\\max_i|x_i|$. Then $\\lVert\\mathbf x\\rVert_2^2=\\sum_i x_i^2\\ge M^2$ (the max term alone is $M^2$), so $\\lVert\\mathbf x\\rVert_2\\ge M=\\lVert\\mathbf x\\rVert_\\infty$." },
    { level: "hard", prompt: "AI task: gradient clipping rescales $\\mathbf g$ to $\\mathbf g\\cdot\\min(1, c/\\lVert\\mathbf g\\rVert_2)$. Explain what it does and why the L2 norm is used.", solution: "If the gradient's L2 norm exceeds the threshold $c$, it's scaled down to have norm exactly $c$ (direction preserved); otherwise it's left alone. This caps the step size during spikes (exploding gradients in RNNs/transformers) without changing direction. L2 is used because it measures the true Euclidean magnitude of the whole gradient vector — the actual distance the step would move — so capping it bounds the update length directly." }
  ],

  deepDive: String.raw`<p><strong>The sparsity of L1, seen as constrained optimization.</strong></p>
  <p>Reframe regularization as: minimize the loss subject to a budget on the weights — $\lVert\mathbf w\rVert_1\le t$ for
  L1, or $\lVert\mathbf w\rVert_2\le t$ for L2. The solution is where the loss's contour lines first touch the budget
  region. For L2 the region is a round ball, and a contour can touch it anywhere — generically at a point with all
  coordinates nonzero. For L1 the region is a <em>diamond</em> whose corners stick out along the axes, and a contour
  sliding in is most likely to first hit a <strong>corner</strong> — where some coordinates are exactly 0.</p>
  <p>That's the whole story of why Lasso selects features: the geometry of the L1 ball makes axis-aligned (sparse)
  solutions the natural meeting point. The same corners explain why L1 is non-smooth (the diamond has kinks), which is
  why it needs sub-gradients or proximal methods rather than plain gradients. Norm choice isn't cosmetic — it shapes the
  solution set. You'll see L2 reappear as a Gaussian prior (MAP, Track 11) and L1 as a Laplace prior, tying these
  geometric pictures to probability.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["4.2"] = {
  subtitle: "How much can a matrix stretch a vector? That cap is the spectral norm.",

  aiMoment: String.raw`<p>A layer $\mathbf x\mapsto W\mathbf x$ can amplify its input. The most it can stretch any vector
  — the <strong>spectral norm</strong> $\lVert W\rVert_2$ — is the layer's <strong>Lipschitz constant</strong>, and
  controlling it keeps signals and gradients from exploding. <strong>Spectral normalization</strong> divides $W$ by this
  number to stabilize GAN training; the <strong>Frobenius norm</strong> shows up in weight decay on matrices and in the
  error of low-rank approximation (Eckart–Young, Track 5).</p>`,

  plainEnglish: String.raw`<p>A matrix norm measures the "size" of a matrix. The <strong>Frobenius norm</strong> treats
  the matrix as one long vector and takes its L2 length. The <strong>spectral norm</strong> measures the matrix's
  maximum amplification — the biggest factor by which it can lengthen any vector.</p>`,

  intuition: String.raw`<p>Feed every unit vector through a matrix and the outputs trace an ellipse. The matrix stretches
  most along the ellipse's longest axis — that maximum stretch is the spectral norm. The Frobenius norm instead bundles
  <em>all</em> the stretching directions together.</p>
  <figure class="figure">
  <svg viewBox="0 0 366 160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A matrix maps the unit circle to an ellipse">
    <circle cx="70" cy="80" r="36" fill="#eef0ff" stroke="#4f46e5" stroke-width="2"/>
    <text x="45" y="135" font-size="11" fill="#4f46e5" font-family="sans-serif">unit circle</text>
    <line x1="120" y1="80" x2="168" y2="80" stroke="#64748b" stroke-width="1.6" marker-end="url(#k1)"/>
    <text x="132" y="72" font-size="11" fill="#64748b" font-family="sans-serif">A</text>
    <ellipse cx="252" cy="80" rx="58" ry="28" fill="#f0fdfa" stroke="#0d9488" stroke-width="2"/>
    <line x1="252" y1="80" x2="310" y2="80" stroke="#dc2626" stroke-width="2" marker-end="url(#k2)"/>
    <line x1="252" y1="80" x2="252" y2="52" stroke="#7c3aed" stroke-width="2" marker-end="url(#k3)"/>
    <text x="286" y="74" font-size="11" fill="#dc2626" font-family="sans-serif">σ₁</text>
    <text x="232" y="58" font-size="11" fill="#7c3aed" font-family="sans-serif">σ₂</text>
    <text x="214" y="135" font-size="11" fill="#0d9488" font-family="sans-serif">ellipse: σ₁ = spectral norm</text>
    <defs>
      <marker id="k1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#64748b"/></marker>
      <marker id="k2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#dc2626"/></marker>
      <marker id="k3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#7c3aed"/></marker>
    </defs>
  </svg>
  <figcaption>The unit circle becomes an ellipse; its longest semi-axis σ₁ is the spectral norm (max stretch).</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>Frobenius norm</strong> sums every entry squared:
  $\lVert A\rVert_F=\sqrt{\sum_{i,j}A_{ij}^2}=\sqrt{\operatorname{tr}(A^\top A)}$. The <strong>spectral norm</strong> is
  the maximum stretch:</p>
  $$\lVert A\rVert_2=\max_{\mathbf x\neq\mathbf 0}\frac{\lVert A\mathbf x\rVert_2}{\lVert\mathbf x\rVert_2}=\sigma_{\max}(A),$$
  <p>the largest <strong>singular value</strong>. It gives the Lipschitz bound $\lVert A\mathbf x\rVert\le\lVert A\rVert_2\,\lVert\mathbf x\rVert$.
  Both relate to singular values $\sigma_i$: $\lVert A\rVert_2=\sigma_{\max}$ and $\lVert A\rVert_F=\sqrt{\sum_i\sigma_i^2}$.</p>`,

  derivation: String.raw`<p><strong>Spectral norm $=\sqrt{\lambda_{\max}(A^\top A)}=\sigma_{\max}$.</strong></p>
  <p><strong>Step 1 — maximize the squared stretch.</strong> Maximizing $\lVert A\mathbf x\rVert/\lVert\mathbf x\rVert$ is
  the same as maximizing its square. Write
  $\dfrac{\lVert A\mathbf x\rVert^2}{\lVert\mathbf x\rVert^2}=\dfrac{\mathbf x^\top A^\top A\,\mathbf x}{\mathbf x^\top\mathbf x}.$</p>
  <p><strong>Step 2 — recognize the Rayleigh quotient.</strong> The matrix $A^\top A$ is symmetric and positive
  semidefinite. For such a matrix, the quotient $\dfrac{\mathbf x^\top M\mathbf x}{\mathbf x^\top\mathbf x}$ is maximized
  by the top eigenvector and equals the largest eigenvalue $\lambda_{\max}(M)$ (Track 5).</p>
  <p><strong>Step 3 — take the square root.</strong> So
  $\lVert A\rVert_2=\sqrt{\lambda_{\max}(A^\top A)}$. Defining singular values as $\sigma_i=\sqrt{\lambda_i(A^\top A)}$,
  this is exactly $\sigma_{\max}$. $\blacksquare$ Plain English: the worst-case stretch of $A$ is the square root of the
  worst-case stretch of $A^\top A$, which is its biggest eigenvalue.</p>`,

  code: [
    { label: "Frobenius vs spectral, via singular values", src: String.raw`
import numpy as np

A = np.array([[2., 0.],
              [0., 1.],
              [1., 1.]])
fro  = np.linalg.norm(A, 'fro')
spec = np.linalg.norm(A, 2)
s = np.linalg.svd(A, compute_uv=False)     # singular values
print("Frobenius          =", round(float(fro), 4))
print("spectral (||A||_2) =", round(float(spec), 4))
print("max singular value =", round(float(s.max()), 4))   # equals spectral
print("Frob == sqrt(Σσ²)? ", np.isclose(fro, np.sqrt((s**2).sum())))
` },
    { label: "Spectral norm is the Lipschitz bound", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)
A = rng.normal(size=(3, 2)); spec = np.linalg.norm(A, 2)

ok = True
for _ in range(1000):
    x = rng.normal(size=2)
    if np.linalg.norm(A @ x) > spec * np.linalg.norm(x) + 1e-9:
        ok = False
print("||Ax|| <= ||A||_2 ||x|| for all tested x:", ok)   # True
` }
  ],

  keyPoints: [
    "Frobenius norm $=\\sqrt{\\sum A_{ij}^2}=\\sqrt{\\operatorname{tr}(A^\\top A)}=\\sqrt{\\sum\\sigma_i^2}$.",
    "Spectral norm $\\lVert A\\rVert_2=\\sigma_{\\max}$ — the maximum stretch / Lipschitz constant.",
    "$\\lVert A\\mathbf x\\rVert\\le\\lVert A\\rVert_2\\lVert\\mathbf x\\rVert$ bounds how much a layer amplifies.",
    "Spectral norm $=\\sqrt{\\lambda_{\\max}(A^\\top A)}$ via the Rayleigh quotient.",
    "Spectral normalization divides $W$ by $\\sigma_{\\max}$ to control amplification."
  ],

  commonMistakes: [
    { wrong: "Treating the Frobenius norm as the spectral norm.", why: "They differ: Frobenius sums <em>all</em> singular values squared, spectral takes only the largest. Frobenius $\\ge$ spectral, and equality holds only for rank-1 matrices." },
    { wrong: "Thinking the largest entry of $A$ is its spectral norm.", why: "Spectral norm is about stretching whole vectors, not single entries; it depends on how entries combine (the singular values), not the max element." },
    { wrong: "Ignoring spectral norm when stacking many layers.", why: "Lipschitz constants multiply through depth: if each layer can stretch by 1.5, ten layers can stretch by $1.5^{10}\\approx58$ — a recipe for exploding activations." }
  ],

  quiz: [
    { q: "The Frobenius norm of $\\begin{bmatrix}3&0\\\\0&4\\end{bmatrix}$ is…", options: ["5", "7", "4", "12"], answer: 0,
      explain: "$\\sqrt{3^2+0+0+4^2}=\\sqrt{25}=5$." },
    { q: "The spectral norm of $\\begin{bmatrix}3&0\\\\0&4\\end{bmatrix}$ is…", options: ["4", "5", "3", "7"], answer: 0,
      explain: "For a diagonal matrix the singular values are $|3|,|4|$; the max is 4 (the spectral norm)." },
    { q: "$\\lVert A\\rVert_2$ equals the…", options: ["largest singular value", "largest entry", "sum of eigenvalues", "trace"], answer: 0,
      explain: "Spectral norm $=\\sigma_{\\max}=\\sqrt{\\lambda_{\\max}(A^\\top A)}$." },
    { q: "If $\\lVert W\\rVert_2=2$, the most $\\lVert W\\mathbf x\\rVert$ can be for a unit $\\mathbf x$ is…", options: ["2", "1", "4", "$\\sqrt2$"], answer: 0,
      explain: "By definition the max stretch of a unit vector is $\\lVert W\\rVert_2=2$." },
    { q: "Five layers each with spectral norm 1.2 can amplify a signal by up to…", options: ["$1.2^5\\approx2.49$", "$1.2\\times5=6$", "$1.2$", "$5$"], answer: 0,
      explain: "Lipschitz constants multiply: $1.2^5\\approx2.49$. This compounding is why spectral control matters in deep nets." }
  ],

  practice: [
    { level: "easy", prompt: "Compute the Frobenius norm of $\\begin{bmatrix}1&2\\\\2&0\\end{bmatrix}$.", solution: "$\\sqrt{1+4+4+0}=\\sqrt9=3$." },
    { level: "med", prompt: "Why is $\\lVert A\\rVert_F\\ge\\lVert A\\rVert_2$ always?", solution: "$\\lVert A\\rVert_F=\\sqrt{\\sum_i\\sigma_i^2}\\ge\\sqrt{\\sigma_{\\max}^2}=\\sigma_{\\max}=\\lVert A\\rVert_2$, since adding the other (nonnegative) $\\sigma_i^2$ only increases the sum. Equality holds when there's a single nonzero singular value (rank 1)." },
    { level: "med", prompt: "A diagonal matrix has entries $\\{0.5, 3, -2\\}$. Give its spectral and Frobenius norms.", solution: "Singular values are the absolute diagonal entries $\\{0.5,3,2\\}$. Spectral $=\\max=3$; Frobenius $=\\sqrt{0.25+9+4}=\\sqrt{13.25}\\approx3.64$." },
    { level: "hard", prompt: "AI task: spectral normalization replaces $W$ with $W/\\sigma_{\\max}(W)$. What does the new layer's spectral norm become, and why does this stabilize training?", solution: "The new spectral norm is $\\sigma_{\\max}(W)/\\sigma_{\\max}(W)=1$, so the layer is 1-Lipschitz — it can never stretch a vector by more than its length. Across depth, products of 1-Lipschitz maps stay 1-Lipschitz, so activations and gradients can't explode multiplicatively. In GANs this bounds the discriminator's sensitivity, smoothing the loss landscape and preventing the runaway gradients that cause mode collapse / instability." }
  ],

  deepDive: String.raw`<p><strong>Lipschitz constants, depth, and why norms are the language of stability.</strong></p>
  <p>A function is $L$-Lipschitz if $\lVert f(\mathbf x)-f(\mathbf y)\rVert\le L\lVert\mathbf x-\mathbf y\rVert$ — it can't
  amplify differences by more than $L$. For a linear layer, that constant is exactly the spectral norm $\lVert W\rVert_2$.
  The crucial fact is that Lipschitz constants <strong>multiply</strong> under composition: a network of layers with
  spectral norms $L_1,L_2,\dots,L_k$ is at most $\big(\prod L_i\big)$-Lipschitz. If the product is much greater than 1,
  small input changes blow up (exploding activations/gradients); much less than 1 and signals vanish.</p>
  <p>This single observation drives a lot of practice: spectral normalization pins each layer near 1; orthogonal
  initialization (Track 3) starts every layer at exactly 1; residual connections add an identity path so the effective
  Lipschitz constant stays near 1 even when a block's is small. It also gives a clean way to reason about robustness — a
  network with a small overall Lipschitz constant can't change its output much under a tiny input perturbation, which is
  the basis of certified defenses against adversarial examples. Norms turn "the model is unstable" into a number you can
  bound.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["4.3"] = {
  subtitle: "Undoing a matrix when you can — and the best 'undo' when you can't.",

  aiMoment: String.raw`<p>The closed-form least-squares solution is $\mathbf x=(A^\top A)^{-1}A^\top\mathbf b$ — an
  <strong>inverse</strong> wrapped around $A$. But $A$ is usually tall (more data than features), so it has no inverse;
  the combination $(A^\top A)^{-1}A^\top$ is the <strong>pseudo-inverse</strong> $A^+$, the right tool for non-square or
  rank-deficient problems. And in practice we almost never <em>form</em> these inverses — we solve. Knowing when an
  inverse exists, and what to use when it doesn't, is everyday linear algebra.</p>`,

  plainEnglish: String.raw`<p>The <strong>inverse</strong> $A^{-1}$ is the matrix that undoes $A$: do $A$, then $A^{-1}$,
  and you're back where you started. It only exists for square matrices that don't collapse any direction. The
  <strong>pseudo-inverse</strong> $A^+$ is a generalized "best undo" that works for any matrix — it gives the
  least-squares answer.</p>`,

  intuition: String.raw`<p>If a matrix keeps every direction (no collapse), you can run it backward — that's the
  inverse, a clean round trip. If it squashes a direction to zero (a null space, Lesson 3.2), information is lost and
  there's no exact way back; the pseudo-inverse returns the closest recoverable answer instead.</p>
  <figure class="figure">
  <svg viewBox="0 0 360 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Inverse as a round trip there and back">
    <text x="18" y="55" font-size="14" font-family="sans-serif" fill="#20242c">x</text>
    <line x1="30" y1="50" x2="78" y2="50" stroke="#94a3b8" marker-end="url(#l1)"/>
    <rect x="80" y="32" width="46" height="36" rx="6" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="97" y="55" font-size="13" fill="#4f46e5" font-family="sans-serif">A</text>
    <line x1="126" y1="50" x2="174" y2="50" stroke="#94a3b8" marker-end="url(#l1)"/>
    <text x="140" y="42" font-size="11" fill="#64748b" font-family="sans-serif">Ax</text>
    <rect x="176" y="32" width="52" height="36" rx="6" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="190" y="55" font-size="13" fill="#0d9488" font-family="sans-serif">A⁻¹</text>
    <line x1="228" y1="50" x2="276" y2="50" stroke="#94a3b8" marker-end="url(#l1)"/>
    <text x="284" y="55" font-size="14" font-family="sans-serif" fill="#20242c">x  ✓</text>
    <text x="40" y="100" font-size="11" fill="#dc2626" font-family="sans-serif">singular A: a direction collapses → no exact way back</text>
    <defs><marker id="l1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>An invertible matrix is a reversible round trip; a singular one loses information on the way.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>inverse</strong> $A^{-1}$ satisfies $AA^{-1}=A^{-1}A=I$. It exists <em>iff</em> $A$ is
  square and full rank ($\det A\neq0$ — no null space). Useful rule: $(AB)^{-1}=B^{-1}A^{-1}$ (order reverses).
  The <strong>Moore–Penrose pseudo-inverse</strong> for a full-column-rank $A$ is</p>
  $$A^{+}=(A^\top A)^{-1}A^\top,\qquad \mathbf x=A^{+}\mathbf b\ \text{ solves least-squares}.$$
  <p>When $A$ is square and invertible, $A^{+}=A^{-1}$. Plain English: the pseudo-inverse is the inverse when one
  exists, and the least-squares "best fit" undo when it doesn't.</p>`,

  derivation: String.raw`<p><strong>The pseudo-inverse falls out of least-squares.</strong> Minimize
  $f(\mathbf x)=\lVert A\mathbf x-\mathbf b\rVert^2$.</p>
  <p><strong>Step 1 — expand</strong> using $\lVert\mathbf v\rVert^2=\mathbf v^\top\mathbf v$:
  $f=(A\mathbf x-\mathbf b)^\top(A\mathbf x-\mathbf b)=\mathbf x^\top A^\top A\,\mathbf x-2\mathbf b^\top A\,\mathbf x+\mathbf b^\top\mathbf b.$</p>
  <p><strong>Step 2 — set the gradient to zero</strong> (Lesson 6.3 rules: $\nabla(\mathbf x^\top M\mathbf x)=2M\mathbf x$
  for symmetric $M=A^\top A$, and $\nabla(\mathbf c^\top\mathbf x)=\mathbf c$):
  $\nabla f=2A^\top A\,\mathbf x-2A^\top\mathbf b=\mathbf 0.$</p>
  <p><strong>Step 3 — the normal equations:</strong> $A^\top A\,\mathbf x=A^\top\mathbf b.$</p>
  <p><strong>Step 4 — solve</strong> (valid when $A^\top A$ is invertible, i.e. $A$ has independent columns):
  $\mathbf x=(A^\top A)^{-1}A^\top\mathbf b=A^{+}\mathbf b.\;\blacksquare$ Plain English: the best-fit weights are the
  pseudo-inverse times the targets.</p>`,

  code: [
    { label: "Inverse, pseudo-inverse, and least-squares", src: String.raw`
import numpy as np

A = np.array([[1.,1.],[2.,1.],[3.,1.]])   # tall (3x2): no true inverse
b = np.array([2., 3., 5.])

Aplus = np.linalg.pinv(A)                  # Moore-Penrose pseudo-inverse
x_pinv = Aplus @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
print("pseudo-inverse solution:", np.round(x_pinv, 4))
print("matches lstsq?         :", np.allclose(x_pinv, x_lstsq))

S = np.array([[2.,1.],[1.,1.]])            # square, invertible
print("S @ inv(S) == I ?", np.allclose(S @ np.linalg.inv(S), np.eye(2)))
` },
    { label: "Solve, don't invert (same answer, better way)", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)
A = rng.normal(size=(4,4)); b = rng.normal(size=4)

x_inv   = np.linalg.inv(A) @ b      # forms the full inverse (wasteful, less stable)
x_solve = np.linalg.solve(A, b)     # factor-and-solve (preferred)
print("same result?", np.allclose(x_inv, x_solve))
# np.linalg.solve never builds A^{-1}; it factors A and back-substitutes.
` }
  ],

  keyPoints: [
    "$A^{-1}$ undoes $A$ ($AA^{-1}=I$); it exists iff $A$ is square and full rank.",
    "$(AB)^{-1}=B^{-1}A^{-1}$ — the order reverses, like the transpose rule.",
    "The pseudo-inverse $A^{+}=(A^\\top A)^{-1}A^\\top$ gives the least-squares solution $\\mathbf x=A^{+}\\mathbf b$.",
    "When $A$ is invertible, $A^{+}=A^{-1}$.",
    "Prefer <code>solve</code> over forming <code>inv</code>: cheaper and more numerically stable."
  ],

  commonMistakes: [
    { wrong: "Computing $A^{-1}\\mathbf b$ to solve $A\\mathbf x=\\mathbf b$.", why: "Forming the inverse costs more and is less accurate than factor-and-solve (<code>np.linalg.solve</code>). Inverses are for analysis, not computation." },
    { wrong: "Expecting a singular (rank-deficient) matrix to have an inverse.", why: "If $\\det A=0$ there's a null space — multiple inputs map to the same output — so no unique undo exists. Use the pseudo-inverse." },
    { wrong: "Writing $(AB)^{-1}=A^{-1}B^{-1}$.", why: "The order must reverse: $(AB)^{-1}=B^{-1}A^{-1}$ (undo the last operation first)." }
  ],

  quiz: [
    { q: "A matrix is invertible iff…", options: ["it is square and full rank", "it is symmetric", "it has positive entries", "it is tall"], answer: 0,
      explain: "Square + full rank ($\\det\\neq0$, no null space) is exactly the condition for a unique inverse." },
    { q: "For a tall full-column-rank $A$, the least-squares solution is…", options: ["$(A^\\top A)^{-1}A^\\top\\mathbf b$", "$A^{-1}\\mathbf b$", "$A^\\top\\mathbf b$", "$(AA^\\top)^{-1}\\mathbf b$"], answer: 0,
      explain: "That's the pseudo-inverse $A^+\\mathbf b$; $A^{-1}$ doesn't exist for a non-square $A$." },
    { q: "$(AB)^{-1}$ equals…", options: ["$B^{-1}A^{-1}$", "$A^{-1}B^{-1}$", "$(BA)^{-1}$", "$A^{-1}B$"], answer: 0,
      explain: "Reverse the order, mirroring the transpose rule $(AB)^\\top=B^\\top A^\\top$." },
    { q: "Why use <code>solve(A,b)</code> instead of <code>inv(A)@b</code>?", options: ["cheaper and more numerically stable", "it gives a different answer", "it works for singular A", "inv is undefined in NumPy"], answer: 0,
      explain: "Factor-and-solve avoids building the full inverse — fewer operations and less rounding error." },
    { q: "If $A$ is $5\\times3$ with rank 3, then $A^\\top A$ is…", options: ["$3\\times3$ and invertible", "$5\\times5$ and invertible", "singular", "$5\\times3$"], answer: 0,
      explain: "$A^\\top A$ is $3\\times3$; full column rank makes it positive definite, hence invertible — so the pseudo-inverse exists." }
  ],

  practice: [
    { level: "easy", prompt: "Find the inverse of $\\begin{bmatrix}2&0\\\\0&5\\end{bmatrix}$.", solution: "Diagonal inverts entrywise: $\\begin{bmatrix}1/2&0\\\\0&1/5\\end{bmatrix}$." },
    { level: "med", prompt: "Invert $\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}$ using the 2×2 formula.", solution: "$\\det=1\\cdot4-2\\cdot3=-2$. $A^{-1}=\\frac{1}{-2}\\begin{bmatrix}4&-2\\\\-3&1\\end{bmatrix}=\\begin{bmatrix}-2&1\\\\1.5&-0.5\\end{bmatrix}$." },
    { level: "med", prompt: "Why does $\\begin{bmatrix}1&2\\\\2&4\\end{bmatrix}$ have no inverse?", solution: "$\\det=1\\cdot4-2\\cdot2=0$. Its columns are dependent (rank 1, nonzero null space), so it collapses a direction and can't be undone." },
    { level: "hard", prompt: "AI task: linear regression has design matrix $A$ ($n\\times d$, $n\\gg d$). Why is the solution $A^+\\mathbf b$ rather than $A^{-1}\\mathbf b$, and what breaks if two features are perfectly correlated?", solution: "$A$ is tall and non-square, so $A^{-1}$ doesn't exist; the least-squares fit uses the pseudo-inverse $A^+=(A^\\top A)^{-1}A^\\top$. If two features are perfectly correlated, $A$ loses column rank, $A^\\top A$ becomes singular (a zero eigenvalue), and $(A^\\top A)^{-1}$ blows up — infinitely many equally-good solutions. Ridge regression fixes it by using $(A^\\top A+\\lambda I)^{-1}$, restoring full rank (Lesson 3.1 deep dive)." }
  ],

  deepDive: String.raw`<p><strong>"Solve, don't invert" — and the SVD form of the pseudo-inverse.</strong></p>
  <p>Forming $A^{-1}$ explicitly is almost always the wrong move. To solve $A\mathbf x=\mathbf b$, a library factors $A$
  once (LU, Lesson 4.4) and back-substitutes — roughly $\tfrac13 n^3$ operations and good stability. Computing $A^{-1}$
  and then multiplying costs more (about $n^3$ plus the multiply) and accumulates extra rounding error, because the
  inverse's entries can be huge for ill-conditioned $A$. The inverse is a conceptual object; the solve is the
  computation.</p>
  <p>The most robust pseudo-inverse doesn't use $(A^\top A)^{-1}$ at all. Via the SVD $A=U\Sigma V^\top$ (Track 5),
  $A^{+}=V\Sigma^{+}U^\top$, where $\Sigma^{+}$ just inverts the nonzero singular values and leaves the zeros alone. This
  form is defined even when $A^\top A$ is singular, and it makes the pseudo-inverse's behavior transparent: it inverts
  the directions the matrix can stretch and ignores the directions it collapses. Tiny singular values are where things
  get dangerous — inverting a near-zero $\sigma$ amplifies noise enormously — which is exactly the conditioning story of
  the next lesson.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["4.4"] = {
  subtitle: "Factor once, substitute fast — and the number that says how much you can trust the answer.",

  aiMoment: String.raw`<p>Under the hood, <code>solve</code> doesn't invert — it computes an <strong>LU
  factorization</strong> and runs two quick triangular solves. Whether the answer is trustworthy is governed by the
  <strong>condition number</strong> $\kappa(A)$: a big $\kappa$ means tiny input errors (or float rounding) explode in
  the solution. The same idea explains why training in float16 can diverge while bfloat16 survives, and why softmax
  needs the log-sum-exp trick — numerical stability is a first-class concern, not an afterthought.</p>`,

  plainEnglish: String.raw`<p><strong>LU</strong> splits a matrix into a Lower-triangular and an Upper-triangular factor;
  triangular systems are trivial to solve by substitution, so this makes solving fast. The <strong>condition number</strong>
  is an error-amplification factor: it tells you how many digits of accuracy a solve can lose.</p>`,

  intuition: String.raw`<p>Gaussian elimination (Lesson 3.1), remembered as a product, <em>is</em> LU. Once you have
  triangular factors, solving is one easy sweep down and one back up. Conditioning is separate: a well-conditioned
  matrix barely magnifies error; an ill-conditioned one turns a rounding speck into a wrong answer.</p>
  <figure class="figure">
  <svg viewBox="0 0 330 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A equals L times U">
    <rect x="20" y="22" width="64" height="76" rx="4" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="46" y="65" font-size="16" fill="#4f46e5" font-family="sans-serif">A</text>
    <text x="96" y="65" font-size="18" font-family="sans-serif">=</text>
    <rect x="120" y="22" width="76" height="76" rx="4" fill="#f0fdfa" stroke="#0d9488"/>
    <path d="M126,28 L126,92 L184,92 Z" fill="#cdeee9"/>
    <text x="134" y="84" font-size="12" fill="#0d9488" font-family="sans-serif">L</text>
    <text x="208" y="65" font-size="18" font-family="sans-serif">·</text>
    <rect x="226" y="22" width="80" height="76" rx="4" fill="#fff7ed" stroke="#d97706"/>
    <path d="M232,28 L298,28 L298,92 Z" fill="#fde9cf"/>
    <text x="270" y="46" font-size="12" fill="#d97706" font-family="sans-serif">U</text>
  </svg>
  <figcaption>A = L·U: a lower-triangular factor times an upper-triangular one. Triangular = easy to solve.</figcaption>
  </figure>`,

  formalism: String.raw`<p>An <strong>LU factorization</strong> writes $A=LU$ with $L$ lower-triangular (1's on its
  diagonal) and $U$ upper-triangular. To solve $A\mathbf x=\mathbf b$: solve $L\mathbf y=\mathbf b$ by forward
  substitution, then $U\mathbf x=\mathbf y$ by back substitution. The <strong>condition number</strong> is</p>
  $$\kappa(A)=\lVert A\rVert_2\,\lVert A^{-1}\rVert_2=\frac{\sigma_{\max}}{\sigma_{\min}}\ \ge 1,$$
  <p>and it bounds error growth: a relative perturbation of size $\varepsilon$ in the data can become up to
  $\kappa(A)\cdot\varepsilon$ in the solution. Plain English: $\kappa$ is how many times bigger your answer's error can
  be than your input's.</p>`,

  derivation: String.raw`<p><strong>Solving with LU, and why triangular is easy.</strong> Take $A=LU$.</p>
  <p><strong>Step 1 — substitute:</strong> $A\mathbf x=\mathbf b$ becomes $L(U\mathbf x)=\mathbf b$. Name $\mathbf y=U\mathbf x$.</p>
  <p><strong>Step 2 — forward solve</strong> $L\mathbf y=\mathbf b$. Because $L$ is lower-triangular, the first equation
  has one unknown ($y_1$), the second has two ($y_1$ known, solve $y_2$), and so on — each step a single division.</p>
  <p><strong>Step 3 — back solve</strong> $U\mathbf x=\mathbf y$ the same way from the bottom up. Two triangular sweeps,
  each $O(n^2)$, instead of re-doing elimination. $\blacksquare$</p>
  <hr class="soft">
  <p><strong>Why conditioning bites.</strong> Suppose $A\mathbf x=\mathbf b$ and we perturb $\mathbf b$ by $\delta\mathbf b$.
  The solution shifts by $\delta\mathbf x=A^{-1}\delta\mathbf b$. Taking norms and using the spectral-norm bounds
  $\lVert\mathbf b\rVert\le\lVert A\rVert\lVert\mathbf x\rVert$ and $\lVert\delta\mathbf x\rVert\le\lVert A^{-1}\rVert\lVert\delta\mathbf b\rVert$,
  divide to get</p>
  $$\frac{\lVert\delta\mathbf x\rVert}{\lVert\mathbf x\rVert}\le \underbrace{\lVert A\rVert\,\lVert A^{-1}\rVert}_{\kappa(A)}\,\frac{\lVert\delta\mathbf b\rVert}{\lVert\mathbf b\rVert}.$$
  <p>So $\kappa(A)$ is exactly the worst-case factor by which relative error grows. A $\kappa$ of $10^{8}$ in float32
  (≈7 digits) means the answer can be meaningless.</p>`,

  code: [
    { label: "A 2×2 LU by hand, then solve", src: String.raw`
import numpy as np

A = np.array([[2., 1.],
              [4., 3.]])
# Doolittle: L has unit diagonal
L = np.array([[1., 0.],
              [2., 1.]])         # multiplier 4/2 = 2
U = np.array([[2., 1.],
              [0., 1.]])         # row2 - 2*row1
print("L @ U == A ?", np.allclose(L @ U, A))

b = np.array([1., 7.])
y = np.array([b[0], b[1] - 2*b[0]])     # forward solve L y = b
x = np.array([(y[0]-y[1])/2, y[1]])     # back solve U x = y
print("solution x =", x, " check:", np.allclose(A @ x, b))
` },
    { label: "Condition number predicts lost accuracy", src: String.raw`
import numpy as np

A = np.array([[1., 1.],
              [1., 1.0001]])     # nearly singular
print("condition number kappa(A) =", round(float(np.linalg.cond(A)), 1))

b  = np.array([2., 2.0001])
x  = np.linalg.solve(A, b)
b2 = np.array([2., 2.0002])      # tiny change in b
x2 = np.linalg.solve(A, b2)
print("x  =", x)
print("x2 =", x2, " <- a small b change moved x a lot (ill-conditioned)")
` }
  ],

  keyPoints: [
    "$A=LU$: solve $L\\mathbf y=\\mathbf b$ then $U\\mathbf x=\\mathbf y$ by substitution — fast and stable.",
    "LU is Gaussian elimination stored as factors; symmetric positive-definite matrices use cheaper Cholesky.",
    "Condition number $\\kappa(A)=\\sigma_{\\max}/\\sigma_{\\min}\\ge1$ amplifies relative error.",
    "Large $\\kappa$ ⇒ lost digits; $\\kappa\\approx10^{k}$ costs about $k$ digits of accuracy.",
    "Catastrophic cancellation: subtracting nearly-equal numbers destroys precision."
  ],

  commonMistakes: [
    { wrong: "Ignoring the condition number when a solve looks fine.", why: "An ill-conditioned system can return a confident, completely wrong answer. Always sanity-check $\\kappa(A)$ for near-collinear data." },
    { wrong: "Computing variance as $E[x^2]-E[x]^2$ in float.", why: "For data with a large mean and small variance, the two terms are nearly equal — catastrophic cancellation gives a tiny or even negative 'variance'. Use a stable (two-pass or Welford) formula." },
    { wrong: "Pivoting blindly past a tiny pivot.", why: "A near-zero pivot magnifies error. Partial pivoting (swap in the largest available pivot) is what makes LU numerically reliable." }
  ],

  quiz: [
    { q: "In $A=LU$, $L$ and $U$ are…", options: ["lower- and upper-triangular", "both diagonal", "both orthogonal", "both symmetric"], answer: 0,
      explain: "$L$ lower-triangular, $U$ upper-triangular — the point is that triangular systems solve by substitution." },
    { q: "A condition number $\\kappa(A)=10^{6}$ in float32 (~7 digits) leaves about…", options: ["1 reliable digit", "7 reliable digits", "0 — always wrong", "13 reliable digits"], answer: 0,
      explain: "You lose about $\\log_{10}\\kappa=6$ digits, leaving roughly $7-6=1$ — a fragile solve." },
    { q: "The condition number equals…", options: ["$\\sigma_{\\max}/\\sigma_{\\min}$", "$\\sigma_{\\max}\\cdot\\sigma_{\\min}$", "$\\det(A)$", "$\\operatorname{tr}(A)$"], answer: 0,
      explain: "$\\kappa=\\lVert A\\rVert_2\\lVert A^{-1}\\rVert_2=\\sigma_{\\max}/\\sigma_{\\min}$." },
    { q: "Why factor with LU instead of recomputing elimination for each new $\\mathbf b$?", options: ["Reuse the factors: each new solve is just $O(n^2)$ substitution", "LU gives a different answer", "LU needs no pivoting", "It avoids floating point"], answer: 0,
      explain: "The $O(n^3)$ factorization is done once; many right-hand sides then cost only $O(n^2)$ each." },
    { q: "Subtracting $1.00000001$ from $1.00000002$ in float32 is risky because…", options: ["catastrophic cancellation destroys significant digits", "the result overflows", "floats can't store 1.0", "it underflows to 0"], answer: 0,
      explain: "The leading digits cancel, leaving only the noisy trailing bits — relative error explodes." }
  ],

  practice: [
    { level: "easy", prompt: "What does it mean for $\\kappa(A)=1$?", solution: "Perfectly conditioned: $\\sigma_{\\max}=\\sigma_{\\min}$, so the matrix scales all directions equally (e.g. an orthogonal matrix). No error amplification." },
    { level: "med", prompt: "Give the LU factors of $\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}$ (unit-diagonal $L$).", solution: "Multiplier $3/1=3$: $L=\\begin{bmatrix}1&0\\\\3&1\\end{bmatrix}$, and $U=\\begin{bmatrix}1&2\\\\0&4-3\\cdot2\\end{bmatrix}=\\begin{bmatrix}1&2\\\\0&-2\\end{bmatrix}$. Check $LU$ returns the original." },
    { level: "med", prompt: "An orthogonal matrix $Q$ has what condition number, and why is that ideal?", solution: "$\\kappa(Q)=1$: all singular values equal 1 ($Q^\\top Q=I$). It preserves lengths and never amplifies error — which is exactly why orthogonal initialization and QR-based solves are numerically friendly." },
    { level: "hard", prompt: "AI task: training diverges in float16 but is stable in bfloat16 at the same learning rate. Using range vs precision, explain why.", solution: "float16 and bfloat16 are both 16-bit, but split the bits differently: float16 has more mantissa (precision) yet only ~5 exponent bits (small range, overflow near 65504); bfloat16 keeps float32's 8 exponent bits (huge range) at the cost of precision. Training has occasional large gradients/activations; in float16 these <em>overflow to inf/NaN</em> and training diverges, while bfloat16's wide range absorbs them. The lesson: for deep-learning dynamics, dynamic range matters more than mantissa precision — which is why bfloat16 became the default (Track 13)." }
  ],

  deepDive: String.raw`<p><strong>Conditioning is a property of the problem, not the algorithm.</strong></p>
  <p>A key distinction: an <em>ill-conditioned problem</em> is inherently sensitive — no algorithm can produce an
  accurate answer from inaccurate data, because the true answer really does swing wildly with the input. An
  <em>unstable algorithm</em> is one that adds extra error beyond what the conditioning forces. The condition number
  $\kappa(A)$ measures the first; a good algorithm (LU with partial pivoting, QR, Cholesky for SPD matrices) avoids the
  second. You can't fix bad conditioning by being clever in the solve — you fix it by changing the problem
  (regularization adds $\lambda I$, raising $\sigma_{\min}$ and lowering $\kappa$; better features reduce collinearity).</p>
  <p>This reframes several ML practices. Batch/layer normalization improves the conditioning of the optimization by
  rescaling activations so the loss surface is rounder (smaller effective $\kappa$ of the Hessian). Weight decay and
  ridge nudge $\sigma_{\min}$ up. Mixed-precision training picks formats whose range survives the dynamics. The unifying
  idea — which you'll formalize in Track 13 — is that real numerical computing lives in finite precision, and the
  matrices and losses we build are only as trustworthy as their conditioning allows. The math on paper assumes infinite
  precision; the engineering is about surviving the gap.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["4.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 60 minutes.",

  intro: String.raw`<p>This exam spans all of Track 4: vector norms and regularization, matrix norms and Lipschitz
  bounds, inverse and pseudo-inverse, and LU / conditioning. <strong>Give yourself 60 minutes</strong>, produce each
  answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "For $\\mathbf x=[1,-2,2]$ give $\\lVert\\mathbf x\\rVert_1$, $\\lVert\\mathbf x\\rVert_2$, $\\lVert\\mathbf x\\rVert_\\infty$.",
      solution: "$\\lVert\\mathbf x\\rVert_1=1+2+2=5$; $\\lVert\\mathbf x\\rVert_2=\\sqrt{1+4+4}=3$; $\\lVert\\mathbf x\\rVert_\\infty=\\max(1,2,2)=2$." },
    { level: "easy", prompt: "Which regularizer yields exactly-zero weights, and one sentence on why?",
      solution: "L1. Its gradient is a constant $\\lambda\\,\\operatorname{sign}(w)$ that keeps pushing toward 0 regardless of size, so small weights reach exactly 0; L2's proportional shrink never does." },
    { level: "med", prompt: "Compute the Frobenius and spectral norms of $\\begin{bmatrix}3&0\\\\0&-4\\end{bmatrix}$.",
      solution: "Singular values are $|3|,|4|$. Frobenius $=\\sqrt{9+16}=5$; spectral $=\\max(3,4)=4$." },
    { level: "med", prompt: "A layer has spectral norm $\\lVert W\\rVert_2=1.5$. Bound $\\lVert W\\mathbf x\\rVert$ for $\\lVert\\mathbf x\\rVert=2$, and the amplification of 8 stacked such layers.",
      solution: "$\\lVert W\\mathbf x\\rVert\\le1.5\\cdot2=3$. Eight layers: Lipschitz constants multiply, $1.5^{8}\\approx25.6\\times$ — illustrating why spectral control matters with depth." },
    { level: "med", prompt: "Invert $\\begin{bmatrix}4&3\\\\6&5\\end{bmatrix}$.",
      solution: "$\\det=4\\cdot5-3\\cdot6=2$. $A^{-1}=\\frac12\\begin{bmatrix}5&-3\\\\-6&4\\end{bmatrix}=\\begin{bmatrix}2.5&-1.5\\\\-3&2\\end{bmatrix}$." },
    { level: "med", prompt: "Give the LU factors (unit-diagonal $L$) of $\\begin{bmatrix}2&4\\\\1&5\\end{bmatrix}$.",
      solution: "Multiplier $1/2$: $L=\\begin{bmatrix}1&0\\\\0.5&1\\end{bmatrix}$, $U=\\begin{bmatrix}2&4\\\\0&5-0.5\\cdot4\\end{bmatrix}=\\begin{bmatrix}2&4\\\\0&3\\end{bmatrix}$. Check $LU$ equals the original." },
    { level: "hard", prompt: "Why does a library solve least-squares with $A^+=(A^\\top A)^{-1}A^\\top$ rather than $A^{-1}$, and what conditioning trap does forming $A^\\top A$ create?",
      solution: "A tall design matrix $A$ ($n>d$) isn't square, so $A^{-1}$ doesn't exist; the pseudo-inverse gives the least-squares fit. But forming $A^\\top A$ squares the condition number ($\\kappa(A^\\top A)=\\kappa(A)^2$), so with correlated features the solve loses twice as many digits — which is why QR (Track 3) is preferred over the explicit normal equations." },
    { level: "hard", prompt: "A solve has $\\kappa(A)=10^{5}$. In float32 (~7 digits) and float64 (~16 digits), roughly how many accurate digits remain?",
      solution: "You lose about $\\log_{10}\\kappa=5$ digits. float32: $\\approx7-5=2$ reliable digits (risky). float64: $\\approx16-5=11$ (fine). This is why ill-conditioned problems are often run in double precision." },
    { level: "hard", prompt: "Explain catastrophic cancellation in the variance formula $E[x^2]-E[x]^2$ and give the fix.",
      solution: "For data with a large mean $\\mu$ and small spread, $E[x^2]\\approx\\mu^2\\approx E[x]^2$, so the subtraction cancels the leading digits and the tiny true variance is swamped by rounding noise (it can even come out negative). Fix: subtract the mean first and compute $\\frac1n\\sum(x_i-\\bar x)^2$ (two-pass), or use Welford's online algorithm — both avoid differencing two large nearly-equal numbers." },
    { level: "hard", prompt: "AI task: ridge regression solves $(A^\\top A+\\lambda I)\\mathbf x=A^\\top\\mathbf b$. Explain, in conditioning terms, why $\\lambda$ both regularizes and stabilizes.",
      solution: "$A^\\top A$ has eigenvalues $\\{\\sigma_i^2\\}$; near-collinear features make $\\sigma_{\\min}\\approx0$, so $\\kappa=\\sigma_{\\max}^2/\\sigma_{\\min}^2$ is enormous and the solve is fragile. Adding $\\lambda I$ shifts every eigenvalue to $\\sigma_i^2+\\lambda$, lifting the smallest off zero, so $\\kappa$ drops to $(\\sigma_{\\max}^2+\\lambda)/(\\sigma_{\\min}^2+\\lambda)$ — a far better-conditioned, uniquely-solvable system. Statistically the same $\\lambda$ shrinks the weights (a Gaussian prior / MAP, Track 11); numerically it rescues the inverse. One knob, two benefits." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Norms, inverses, and conditioning are solid. On to Track 5 (Eigenvalues, PCA & SVD).</li>
    <li><strong>7–8:</strong> Strong. Revisit spectral norm ↔ singular values if that slipped.</li>
    <li><strong>5–6:</strong> Re-derive the L1/L2 gradient distinction and the condition-number bound; redo Lessons 4.1 and 4.4.</li>
    <li><strong>Below 5:</strong> Rework the track — these ideas (norms, conditioning, pseudo-inverse) underpin regularization, SVD, and stable training throughout.</li>
  </ul>`
};
