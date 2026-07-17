/* ============================================================
   TRACK 14 — Matrix Calculus & Backpropagation
   The bridge from "I know calculus" to "I can derive the gradient
   of a real layer." Every result here is checked numerically.
   ============================================================ */
window.LESSON_CONTENT ||= {};

/* ------------------------------------------------------------------ 14.1 */
window.LESSON_CONTENT["14.1"] = {
  subtitle: "Scalars, vectors, matrices — what a derivative even *is* when the inputs and outputs aren't single numbers, and the one rule that keeps the shapes straight.",

  aiMoment: String.raw`When you write <code>W -= lr * W.grad</code> in a training loop, you are subtracting one array from another elementwise — which only makes sense because <code>W.grad</code> has the <em>exact same shape</em> as <code>W</code>. That is not a coincidence; it is the <strong>shape rule</strong> of matrix calculus. Before you can derive a single backprop equation, you need a firm grip on what "the derivative" means when your variable is a length-1000 vector or a 512×512 weight matrix — and how to keep every index in its place.`,

  plainEnglish: String.raw`A derivative answers "if I nudge the input a little, how much does the output move?" When the input is a whole vector, you have one such answer <em>per input component</em>, so the derivative is itself a vector. When both input and output are vectors, you have one answer for every (output, input) pair, so the derivative is a grid of numbers — a matrix. The trick is just bookkeeping: keep track of which slot each number belongs in.`,

  intuition: String.raw`Think of three rungs on a ladder. <strong>Scalar → scalar</strong>: the derivative is a single number (the slope). <strong>Vector → scalar</strong> (a loss depending on many weights): the derivative is the <em>gradient</em>, a vector with one slope per input — and by convention we give it the <em>same shape as the input</em>, so it can be subtracted from the input during an update. <strong>Vector → vector</strong> (a layer mapping activations to activations): the derivative is the <em>Jacobian</em>, a matrix whose $(i,j)$ entry is "how much output $i$ moves when input $j$ is nudged." Everything in backprop is built from these three shapes.`,

  formalism: String.raw`For $f:\mathbb R^n\to\mathbb R$ (a scalar output), the <strong>gradient</strong> collects all partials:
$$\nabla f(\mathbf x)=\Big[\tfrac{\partial f}{\partial x_1},\dots,\tfrac{\partial f}{\partial x_n}\Big]^\top\in\mathbb R^{n}.$$
For $f:\mathbb R^n\to\mathbb R^m$ (a vector output), the <strong>Jacobian</strong> is the $m\times n$ matrix
$$\big(J_f\big)_{ij}=\frac{\partial f_i}{\partial x_j},\qquad J_f\in\mathbb R^{m\times n}.$$
The <strong>shape rule</strong> we adopt (the machine-learning convention): the gradient of a scalar with respect to <em>any</em> array has the <em>same shape as that array</em>. So $\partial L/\partial W$ is the same size as $W$, and $\partial L/\partial \mathbf x$ the same size as $\mathbf x$.`,

  derivation: String.raw`<strong>Building a Jacobian one entry at a time.</strong>
<ol>
<li><strong>Pick a concrete map.</strong> Let $f:\mathbb R^3\to\mathbb R^2$ with $f(\mathbf x)=\big(x_1x_2,\ \sin x_3\big)$. Two outputs, three inputs, so the Jacobian is $2\times 3$.</li>
<li><strong>Fill row 1</strong> (output $f_1=x_1x_2$): $\frac{\partial f_1}{\partial x_1}=x_2,\ \frac{\partial f_1}{\partial x_2}=x_1,\ \frac{\partial f_1}{\partial x_3}=0.$</li>
<li><strong>Fill row 2</strong> (output $f_2=\sin x_3$): $\frac{\partial f_2}{\partial x_1}=0,\ \frac{\partial f_2}{\partial x_2}=0,\ \frac{\partial f_2}{\partial x_3}=\cos x_3.$</li>
<li><strong>Assemble.</strong> $J_f=\begin{bmatrix}x_2 & x_1 & 0\\[2pt]0 & 0 & \cos x_3\end{bmatrix}.$ Rows are outputs, columns are inputs — read "$J_{ij}$ = how output $i$ responds to input $j$."</li>
<li><strong>The gradient is the one-row special case.</strong> If $m=1$ (scalar output), the Jacobian is a single row $1\times n$; transpose it and you get the gradient column $\nabla f\in\mathbb R^n$. Gradient and Jacobian are the same object seen at different output sizes.</li>
</ol>`,

  code: [
    { label: "Gradients and Jacobians, checked by finite differences",
      src: String.raw`import numpy as np

# A reusable numeric-gradient helper: nudge each entry of x and watch f.
def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):              # works for vectors AND matrices
        old = x[i]
        x[i] = old + h; fp = f(x)
        x[i] = old - h; fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2*h)
    return g

# --- vector -> scalar: gradient of f(x) = 1/2 ||x||^2 is x itself ---
x = np.array([1.0, -2.0, 3.0, 0.5, -1.5])
f = lambda x: 0.5 * (x @ x)
grad = numeric_grad(f, x.copy())
print("numeric gradient:", np.round(grad, 4))
print("analytic (= x)  :", x)
print("same shape as x :", grad.shape == x.shape)      # the shape rule

# --- vector -> vector: f: R^3 -> R^2 has a 2x3 Jacobian ---
g = lambda x: np.array([x[0]*x[1], np.sin(x[2])])
x = np.array([2.0, 3.0, 0.0])
J = np.zeros((2, 3)); h = 1e-6
for j in range(3):
    e = np.zeros(3); e[j] = h
    J[:, j] = (g(x+e) - g(x-e)) / (2*h)         # column j = d(all outputs)/d x_j
print("Jacobian (2x3):\n", np.round(J, 4))       # -> [[3, 2, 0], [0, 0, 1]]` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 210" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <text x="360" y="20" text-anchor="middle" font-weight="700" fill="#1f2a44">the shape of a derivative follows its inputs and outputs</text>
  <text x="115" y="48" text-anchor="middle" fill="#4a5878">scalar → scalar</text>
  <rect x="98" y="92" width="34" height="34" rx="4" fill="#eef2f7" stroke="#33415c"/>
  <text x="115" y="114" text-anchor="middle" fill="#1f2a44" font-size="12">f ′</text>
  <text x="115" y="150" text-anchor="middle" fill="#5a6b8c" font-size="10.5">one number</text>
  <text x="115" y="166" text-anchor="middle" fill="#5a6b8c" font-size="10.5">(the slope)</text>
  <text x="355" y="48" text-anchor="middle" fill="#4a5878">vector → scalar  (gradient)</text>
  <g fill="#dbe8fb" stroke="#2a6f97">
    <rect x="338" y="74" width="34" height="17"/><rect x="338" y="91" width="34" height="17"/>
    <rect x="338" y="108" width="34" height="17"/><rect x="338" y="125" width="34" height="17"/>
  </g>
  <text x="355" y="162" text-anchor="middle" fill="#2a6f97" font-size="10.5">∇f ∈ ℝⁿ</text>
  <text x="355" y="178" text-anchor="middle" fill="#5a6b8c" font-size="10.5">same shape as x</text>
  <text x="600" y="48" text-anchor="middle" fill="#4a5878">vector → vector  (Jacobian)</text>
  <g fill="#e7f0e8" stroke="#3a7d44">
    <rect x="556" y="80" width="26" height="20"/><rect x="583" y="80" width="26" height="20"/><rect x="610" y="80" width="26" height="20"/>
    <rect x="556" y="101" width="26" height="20"/><rect x="583" y="101" width="26" height="20"/><rect x="610" y="101" width="26" height="20"/>
  </g>
  <text x="600" y="162" text-anchor="middle" fill="#3a7d44" font-size="10.5">J ∈ ℝᵐˣⁿ</text>
  <text x="600" y="178" text-anchor="middle" fill="#5a6b8c" font-size="10.5">Jᵢⱼ = ∂fᵢ / ∂xⱼ</text>
</svg>`,

  keyPoints: [
    String.raw`A gradient is a vector (one slope per input); a Jacobian is a matrix (one slope per output–input pair). Same idea, different output size.`,
    String.raw`<strong>The shape rule:</strong> the gradient of a scalar loss w.r.t. any array has the <em>same shape</em> as that array. That's what makes <code>W -= lr * W.grad</code> a legal elementwise update.`,
    String.raw`Jacobian convention: rows index outputs, columns index inputs — $J_{ij}=\partial f_i/\partial x_j$, so $J$ is $m\times n$ for $f:\mathbb R^n\to\mathbb R^m$.`,
    String.raw`Finite differences ($\frac{f(x+h)-f(x-h)}{2h}$) give a cheap ground truth for <em>any</em> gradient — your first move whenever you hand-derive one.`,
    String.raw`The gradient is a Jacobian with one row; transposing that row gives the gradient column. Keep the two straight and half of backprop's index confusion disappears.`
  ],

  commonMistakes: [
    { wrong: "Mixing up the two Jacobian layout conventions and ending up with everything transposed.",
      why: String.raw`"Numerator layout" and "denominator layout" put outputs/inputs on opposite axes. Pick one (this course uses $J_{ij}=\partial f_i/\partial x_j$, plus the shape rule for gradients) and stay consistent. Most 'my gradient is transposed' bugs are a silent layout switch.` },
    { wrong: "Expecting the gradient of a scalar w.r.t. a matrix to be a matrix of a different shape.",
      why: String.raw`By the shape rule, $\partial L/\partial W$ is exactly the size of $W$. If your derivation produces something a different shape, an index is misplaced — recheck before moving on.` },
    { wrong: "Trusting a hand-derived gradient without a numeric check.",
      why: String.raw`A finite-difference check costs three lines and catches almost every sign/transpose slip. Skipping it is the most expensive shortcut in ML engineering.` }
  ],

  quiz: [
    { q: "f : ℝ⁴ → ℝ³. What is the shape of its Jacobian?",
      options: ["3 × 4", "4 × 3", "4 × 4", "3 × 3"], answer: 0,
      explain: String.raw`Rows = outputs (3), columns = inputs (4), so $J$ is $3\times4$: $J_{ij}=\partial f_i/\partial x_j$.` },
    { q: "L is a scalar loss and W is a 128×256 weight matrix. What shape is ∂L/∂W?",
      options: ["128 × 256 (same as W)", "256 × 128", "1 × 32768", "a single number"], answer: 0,
      explain: String.raw`The shape rule: the gradient of a scalar w.r.t. an array matches that array's shape, so the update $W-\eta\,\partial L/\partial W$ is elementwise.` },
    { q: "For f(x) = ½‖x‖² with x ∈ ℝⁿ, the gradient is…",
      options: ["x", "2x", "‖x‖", "the identity matrix"], answer: 0,
      explain: String.raw`$\frac{\partial}{\partial x_k}\tfrac12\sum_i x_i^2 = x_k$, so $\nabla f=\mathbf x$ — same shape as $\mathbf x$, as the code confirms.` },
    { q: "In the Jacobian of f(x) = (x₁x₂, sin x₃), what is entry J₁₂ (row 1, col 2)?",
      options: ["x₁", "x₂", "0", "cos x₃"], answer: 0,
      explain: String.raw`Row 1 is output $f_1=x_1x_2$; column 2 is $\partial/\partial x_2$, giving $x_1$.` },
    { q: "Why is the central difference (f(x+h) − f(x−h))/(2h) preferred over (f(x+h) − f(x))/h for checking gradients?",
      options: ["Its error is O(h²) instead of O(h), so it's far more accurate",
                "It is faster to compute", "It needs no evaluation of f", "It works only for linear f"], answer: 0,
      explain: String.raw`Taylor expansion shows the symmetric difference cancels the first-order error term, leaving $O(h^2)$ — much tighter for the same $h$.` }
  ],

  practice: [
    { level: "easy", prompt: "Write down the Jacobian of f(x) = (x₁ + x₂, x₁ − x₂, 3x₁). What are its dimensions?",
      solution: String.raw`It is $3\times2$: $J=\begin{bmatrix}1&1\\1&-1\\3&0\end{bmatrix}$. Three outputs, two inputs.` },
    { level: "easy", prompt: "For the affine map f(x) = Ax + b with A an m×n matrix, what is the Jacobian ∂f/∂x?",
      solution: String.raw`$J=A$ (the constant $b$ drops out). Differentiating a linear map just returns the matrix — the single most-used Jacobian in all of deep learning.` },
    { level: "med", prompt: "Use the numeric_grad helper to check that the gradient of f(x) = log(∑ eˣⁱ) (log-sum-exp) is the softmax of x.",
      solution: String.raw`$\frac{\partial}{\partial x_k}\log\sum_i e^{x_i}=\frac{e^{x_k}}{\sum_i e^{x_i}}=\mathrm{softmax}(\mathbf x)_k$. The numeric gradient matches <code>np.exp(x)/np.exp(x).sum()</code>. (You'll meet this again in 14.5.)` },
    { level: "med", prompt: "Extend numeric_grad to return the full Jacobian of a vector-valued f by stacking the per-output gradients.",
      solution: String.raw`Loop over output components: for each $i$, compute the gradient of $f(\cdot)_i$ and place it in row $i$. Equivalently perturb each input and read the whole output vector into column $j$, as the lesson code does.` },
    { level: "hard", prompt: "The gradient of f(x) = ½ xᵀA x (A not necessarily symmetric) is (A + Aᵀ)x/... wait — derive it and explain why the symmetric part appears.",
      solution: String.raw`$\frac{\partial}{\partial x_k}\tfrac12\sum_{i,j}A_{ij}x_ix_j=\tfrac12\sum_j(A_{kj}+A_{jk})x_j=\tfrac12((A+A^\top)\mathbf x)_k$. Only the symmetric part $\tfrac12(A+A^\top)$ survives because $x_ix_j=x_jx_i$ can't tell $A_{ij}$ from $A_{ji}$. (We meet the un-halved version $x^\top A x$ next lesson.)` },
    { level: "hard", prompt: "Explain, using the shape rule, why the gradient of a scalar w.r.t. a rank-3 tensor (e.g. a batch of images, N×H×W) is itself N×H×W.",
      solution: String.raw`The shape rule is dimension-agnostic: there is one partial derivative per scalar entry of the variable, and they're collected into the same index structure. A loss depending on an $N\times H\times W$ tensor has $NHW$ partials, arranged back into an $N\times H\times W$ gradient — which is exactly what an optimizer needs to update it in place.` }
  ],

  deepDive: String.raw`<p><strong>Numerator vs denominator layout.</strong> Textbooks disagree on whether $\partial \mathbf y/\partial \mathbf x$ is $m\times n$ or $n\times m$. Neither is "wrong"; they're transposes. The machine-learning world sidesteps the fight with the <em>shape rule</em> for gradients (gradient matches the variable's shape) and keeps full Jacobians $m\times n$ only when they're genuinely needed. When you read a paper's derivation and the transposes look off, suspect a layout difference before suspecting an error.</p>
<p><strong>Why we almost never build the Jacobian.</strong> A layer mapping $\mathbb R^{1000}\to\mathbb R^{1000}$ has a $1000\times1000$ Jacobian — a million numbers you'd have to store and multiply. Backprop's whole cleverness (next two lessons) is to <em>never</em> form these matrices: it multiplies a Jacobian by a vector on the fly (a "vector–Jacobian product"), which costs the same as the forward pass. The shapes you learned here are the language that trick is written in.</p>
<p><strong>Finite differences in practice.</strong> The $O(h^2)$ central difference is the gold-standard gradient check, but it costs two forward passes <em>per parameter</em> — fine for a unit test on a small layer, hopeless for a real network. That asymmetry (cheap to check one gradient, ruinous to compute all of them numerically) is exactly why reverse-mode autodiff exists.</p>`
};

/* ------------------------------------------------------------------ 14.2 */
window.LESSON_CONTENT["14.2"] = {
  subtitle: "A short table of vector/matrix derivative identities — and the index-notation trick that lets you re-derive any of them in ten seconds.",

  aiMoment: String.raw`The very first gradient every ML student derives is the one for linear regression / a linear layer with squared loss: $\nabla_{\mathbf w}\lVert X\mathbf w-\mathbf y\rVert^2 = 2X^\top(X\mathbf w-\mathbf y)$. Set it to zero and you get the <strong>normal equations</strong> that solve least squares in closed form. That single derivation uses three reusable identities, and those identities are the vocabulary you'll use to differentiate <em>any</em> layer. Learn the handful here and matrix calculus stops being a lookup table you fear and becomes something you can do on a whiteboard.`,

  plainEnglish: String.raw`You don't re-derive derivatives of $x^2$ every time — you recognize the pattern and write $2x$. Matrix calculus is the same: a small set of patterns covers almost everything you meet. "See $\mathbf a^\top\mathbf x$, gradient is $\mathbf a$." "See $\mathbf x^\top A\mathbf x$, gradient is $(A+A^\top)\mathbf x$." Memorize a handful and you can differentiate a loss by pattern-matching instead of grinding.`,

  intuition: String.raw`Every one of these identities is really a statement in plain calculus, hidden by matrix shorthand. The way to <em>recover</em> it — and never have to trust your memory — is <strong>index notation</strong>: write the scalar as an explicit sum over indices, differentiate with respect to one component $x_k$ using ordinary rules, then look at the resulting expression and recognize which matrix–vector product it is. The matrices are just bookkeeping over the sums.`,

  formalism: String.raw`The workhorse identities (with $\mathbf a,\mathbf x$ vectors, $A$ a matrix):
$$\frac{\partial(\mathbf a^\top\mathbf x)}{\partial\mathbf x}=\mathbf a,\qquad
\frac{\partial(\mathbf x^\top\mathbf x)}{\partial\mathbf x}=2\mathbf x,\qquad
\frac{\partial(\mathbf x^\top A\,\mathbf x)}{\partial\mathbf x}=(A+A^\top)\mathbf x.$$
$$\frac{\partial}{\partial\mathbf w}\lVert X\mathbf w-\mathbf y\rVert^2=2X^\top(X\mathbf w-\mathbf y),\qquad
\frac{\partial\,\mathrm{tr}(AX)}{\partial X}=A^\top,\qquad
\frac{\partial\,\mathrm{tr}(X^\top A X)}{\partial X}=(A+A^\top)X.$$
When $A$ is symmetric, $(A+A^\top)\mathbf x=2A\mathbf x$.`,

  derivation: String.raw`<strong>Two derivations by index notation — the only technique you need.</strong>
<ol>
<li><strong>The quadratic form.</strong> Write $\mathbf x^\top A\mathbf x=\sum_{i}\sum_{j}A_{ij}x_ix_j$. Differentiate w.r.t. a single $x_k$; the sum only survives where $i=k$ or $j=k$:
$$\frac{\partial}{\partial x_k}\sum_{i,j}A_{ij}x_ix_j=\sum_{j}A_{kj}x_j+\sum_{i}A_{ik}x_i=(A\mathbf x)_k+(A^\top\mathbf x)_k.$$
Stacking over all $k$: $\;\partial(\mathbf x^\top A\mathbf x)/\partial\mathbf x=(A+A^\top)\mathbf x.$</li>
<li><strong>Least squares.</strong> Expand the loss: $\lVert X\mathbf w-\mathbf y\rVert^2=\mathbf w^\top X^\top X\mathbf w-2\mathbf y^\top X\mathbf w+\mathbf y^\top\mathbf y.$ Differentiate term by term. Term 1 is a quadratic form with the symmetric matrix $X^\top X$, giving $2X^\top X\mathbf w$. Term 2 is linear, $\mathbf a^\top\mathbf w$ with $\mathbf a=2X^\top\mathbf y$, giving $-2X^\top\mathbf y$. Term 3 is constant. Sum: $2X^\top X\mathbf w-2X^\top\mathbf y=2X^\top(X\mathbf w-\mathbf y).$</li>
<li><strong>Read off the optimum.</strong> Set the gradient to $\mathbf 0$: $X^\top X\mathbf w=X^\top\mathbf y$ — the <em>normal equations</em>. That's least squares solved in closed form, straight out of one gradient.</li>
</ol>`,

  code: [
    { label: "Every identity, checked numerically",
      src: String.raw`import numpy as np
rng = np.random.default_rng(0)

def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]; x[i]=old+h; fp=f(x); x[i]=old-h; fm=f(x); x[i]=old
        g[i] = (fp - fm) / (2*h)
    return g

# Identity:  d/dx (xᵀ A x) = (A + Aᵀ) x     (= 2Ax when A is symmetric)
A = rng.standard_normal((4, 4)); x = rng.standard_normal(4)
print("xᵀAx  :", np.allclose(numeric_grad(lambda x: x @ A @ x, x.copy()), (A + A.T) @ x))

# Least squares:  d/dw ||Xw - y||²  =  2 Xᵀ(Xw - y)
X = rng.standard_normal((6, 3)); y = rng.standard_normal(6); w = rng.standard_normal(3)
print("lstsq :", np.allclose(numeric_grad(lambda w: ((X@w - y)**2).sum(), w.copy()),
                             2 * X.T @ (X @ w - y)))

# Zeroing that gradient gives the normal equations  XᵀX w = Xᵀy
w_star = np.linalg.solve(X.T @ X, X.T @ y)
print("grad at optimum ≈ 0:", np.round(2 * X.T @ (X @ w_star - y), 6))

# Trace identity:  d/dX tr(A X) = Aᵀ
A = rng.standard_normal((3, 5)); M = rng.standard_normal((5, 3))
print("trace :", np.allclose(numeric_grad(lambda M: np.trace(A @ M), M.copy()), A.T))` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 226" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="12">
  <rect x="20" y="20" width="680" height="30" fill="#33415c"/>
  <text x="185" y="40" text-anchor="middle" fill="#fff" font-weight="700">expression  f</text>
  <text x="520" y="40" text-anchor="middle" fill="#fff" font-weight="700">gradient  ∂f / ∂(·)</text>
  <line x1="360" y1="20" x2="360" y2="220" stroke="#c7d0df"/>
  <g fill="#1f2a44">
    <rect x="20" y="50" width="680" height="28" fill="#f4f7fb" stroke="#e0e6ef"/>
    <text x="185" y="69" text-anchor="middle">aᵀx</text><text x="520" y="69" text-anchor="middle" fill="#2a6f97">a</text>
    <rect x="20" y="78" width="680" height="28" fill="#fff" stroke="#e0e6ef"/>
    <text x="185" y="97" text-anchor="middle">xᵀx</text><text x="520" y="97" text-anchor="middle" fill="#2a6f97">2x</text>
    <rect x="20" y="106" width="680" height="28" fill="#f4f7fb" stroke="#e0e6ef"/>
    <text x="185" y="125" text-anchor="middle">xᵀ A x</text><text x="520" y="125" text-anchor="middle" fill="#2a6f97">(A + Aᵀ) x</text>
    <rect x="20" y="134" width="680" height="28" fill="#fff" stroke="#e0e6ef"/>
    <text x="185" y="153" text-anchor="middle">‖Xw − y‖²</text><text x="520" y="153" text-anchor="middle" fill="#2a6f97">2 Xᵀ(Xw − y)</text>
    <rect x="20" y="162" width="680" height="28" fill="#f4f7fb" stroke="#e0e6ef"/>
    <text x="185" y="181" text-anchor="middle">tr(A X)</text><text x="520" y="181" text-anchor="middle" fill="#2a6f97">Aᵀ</text>
    <rect x="20" y="190" width="680" height="28" fill="#fff" stroke="#e0e6ef"/>
    <text x="185" y="209" text-anchor="middle">tr(Xᵀ A X)</text><text x="520" y="209" text-anchor="middle" fill="#2a6f97">(A + Aᵀ) X</text>
  </g>
</svg>`,

  keyPoints: [
    String.raw`Differentiate a linear form $\mathbf a^\top\mathbf x\to\mathbf a$; a quadratic form $\mathbf x^\top A\mathbf x\to(A+A^\top)\mathbf x$. Those two cover a huge fraction of ML losses.`,
    String.raw`The least-squares gradient $2X^\top(X\mathbf w-\mathbf y)$ set to zero gives the normal equations $X^\top X\mathbf w=X^\top\mathbf y$ — closed-form regression from one derivative.`,
    String.raw`<strong>Index notation is the escape hatch:</strong> write the scalar as $\sum_{ij}(\dots)$, differentiate one component, and re-recognize the matrix product. You never have to <em>memorize</em> an identity you can rebuild.`,
    String.raw`Trace shows up because it turns a matrix-valued expression into a scalar you can differentiate: $\partial\,\mathrm{tr}(AX)/\partial X=A^\top$ is the matrix analogue of $\partial(\mathbf a^\top\mathbf x)/\partial\mathbf x=\mathbf a$.`,
    String.raw`When $A$ is symmetric (like $X^\top X$), $(A+A^\top)\mathbf x=2A\mathbf x$ — the case that appears in nearly every quadratic loss.`
  ],

  commonMistakes: [
    { wrong: "Writing d/dx (xᵀA x) = A x, forgetting the transpose term.",
      why: String.raw`That's only right when $A$ is symmetric. In general the correct gradient is $(A+A^\top)\mathbf x$, because $x_ix_j$ pairs $A_{ij}$ with $A_{ji}$. Dropping the $A^\top$ is a classic exam-losing slip.` },
    { wrong: "Losing the factor of 2 in the least-squares gradient.",
      why: String.raw`$\lVert X\mathbf w-\mathbf y\rVert^2$ is a <em>squared</em> norm; its gradient carries the 2 from the power rule. (Some texts define the loss as $\tfrac12\lVert\cdot\rVert^2$ precisely to cancel it — know which convention you're in.)` },
    { wrong: "Treating Xᵀy and yᵀX as interchangeable scalars in a derivation.",
      why: String.raw`They're transposes of each other and only equal because the expression is a scalar. Keeping shapes explicit (is this $n\times1$ or $1\times n$?) prevents a transpose creeping into the final gradient.` }
  ],

  quiz: [
    { q: "d/dx (aᵀx) equals…",
      options: ["a", "aᵀ", "x", "aᵀx"], answer: 0,
      explain: String.raw`$\mathbf a^\top\mathbf x=\sum_i a_ix_i$, so $\partial/\partial x_k=a_k$, i.e. the gradient is $\mathbf a$ (same shape as $\mathbf x$).` },
    { q: "For symmetric A, d/dx (xᵀA x) is…",
      options: ["2Ax", "Ax", "(A − Aᵀ)x", "Aᵀx"], answer: 0,
      explain: String.raw`In general it's $(A+A^\top)\mathbf x$; symmetry ($A=A^\top$) collapses that to $2A\mathbf x$.` },
    { q: "Setting the least-squares gradient 2Xᵀ(Xw − y) to zero gives which equation?",
      options: ["XᵀX w = Xᵀy (the normal equations)", "Xw = y", "XXᵀ w = y", "w = Xᵀy"], answer: 0,
      explain: String.raw`$2X^\top(X\mathbf w-\mathbf y)=\mathbf 0\Rightarrow X^\top X\mathbf w=X^\top\mathbf y$ — the closed-form least-squares solution (when $X^\top X$ is invertible).` },
    { q: "X is 100×5. What are the shapes of XᵀX and the gradient 2Xᵀ(Xw − y)?",
      options: ["5×5 and 5×1", "5×5 and 100×1", "100×100 and 5×1", "5×100 and 5×1"], answer: 0,
      explain: String.raw`$X^\top X$ is $5\times5$; the gradient has the shape of $\mathbf w\in\mathbb R^5$, i.e. $5\times1$ — the shape rule again.` },
    { q: "d/dX tr(AX) with A being 3×5 and X being 5×3 equals…",
      options: ["Aᵀ (5×3)", "A (3×5)", "AᵀA", "the identity"], answer: 0,
      explain: String.raw`$\partial\,\mathrm{tr}(AX)/\partial X=A^\top$, which is $5\times3$ — the same shape as $X$, as required.` }
  ],

  practice: [
    { level: "easy", prompt: "Differentiate f(w) = wᵀc + 5 with respect to w.",
      solution: String.raw`$\nabla_{\mathbf w}f=\mathbf c$. The linear term contributes $\mathbf c$; the constant 5 contributes nothing.` },
    { level: "easy", prompt: "For the ridge loss ‖Xw − y‖² + λ‖w‖², write the gradient.",
      solution: String.raw`$2X^\top(X\mathbf w-\mathbf y)+2\lambda\mathbf w$. Setting it to zero gives the ridge normal equations $(X^\top X+\lambda I)\mathbf w=X^\top\mathbf y$ — the $\lambda I$ is what makes the inverse always exist.` },
    { level: "med", prompt: "Derive d/dx ‖x‖ (the ordinary 2-norm, not squared) for x ≠ 0.",
      solution: String.raw`$\lVert\mathbf x\rVert=(\mathbf x^\top\mathbf x)^{1/2}$; chain rule gives $\frac{1}{2\lVert\mathbf x\rVert}\cdot2\mathbf x=\mathbf x/\lVert\mathbf x\rVert$ — the unit vector pointing along $\mathbf x$. (Undefined at $\mathbf x=\mathbf 0$, which is why L2-norm penalties are smooth but the norm itself isn't at the origin.)` },
    { level: "med", prompt: "Use index notation to derive d/dx (Ax)ᵀ(Ax) and confirm it equals 2AᵀAx.",
      solution: String.raw`$(A\mathbf x)^\top(A\mathbf x)=\mathbf x^\top A^\top A\mathbf x$, a quadratic form with symmetric matrix $A^\top A$, so the gradient is $2A^\top A\mathbf x$. This is the least-squares gradient with $\mathbf y=\mathbf 0$.` },
    { level: "hard", prompt: "Derive d/dX ‖X‖_F² (squared Frobenius norm) and relate it to the trace identity.",
      solution: String.raw`$\lVert X\rVert_F^2=\mathrm{tr}(X^\top X)=\sum_{ij}X_{ij}^2$, so $\partial/\partial X_{ij}=2X_{ij}$, i.e. $\partial\lVert X\rVert_F^2/\partial X=2X$. It's the matrix version of $\partial\lVert\mathbf x\rVert^2/\partial\mathbf x=2\mathbf x$, and it's the gradient of weight decay applied to a whole weight matrix.` },
    { level: "hard", prompt: "The multivariate-Gaussian log-likelihood contains −½(x−μ)ᵀ Σ⁻¹(x−μ). Derive its gradient w.r.t. μ.",
      solution: String.raw`Let $\mathbf u=\mathbf x-\boldsymbol\mu$. The term is $-\tfrac12\mathbf u^\top\Sigma^{-1}\mathbf u$; since $\Sigma^{-1}$ is symmetric, $\partial/\partial\mathbf u=-\Sigma^{-1}\mathbf u$, and $\partial\mathbf u/\partial\boldsymbol\mu=-I$, so the gradient is $\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$. Setting it to zero over a dataset recovers the MLE $\hat{\boldsymbol\mu}=\bar{\mathbf x}$ — connecting straight back to the Gaussian-Bayes capstone.` }
  ],

  deepDive: String.raw`<p><strong>Why trace is everywhere in matrix calculus.</strong> To differentiate a scalar that's built from matrices, it helps to write it as a trace, because trace is linear and has the cyclic property $\mathrm{tr}(ABC)=\mathrm{tr}(CAB)$. That lets you shuffle a matrix expression until the variable $X$ sits alone, at which point $\partial\,\mathrm{tr}(BX)/\partial X=B^\top$ reads the gradient straight off. The "matrix cookbook" is essentially this one move applied over and over.</p>
<p><strong>From normal equations to gradient descent.</strong> Least squares has a closed form, so why iterate? Because $X^\top X$ is $d\times d$: for $d$ in the thousands or millions (every deep net), forming and inverting it is impossible, and it's terribly conditioned. So we keep the <em>gradient</em> $2X^\top(X\mathbf w-\mathbf y)$ and walk downhill with it instead — which is exactly the optimizer capstone. The identity you derived here is the thing an optimizer consumes a million times.</p>
<p><strong>The bridge to backprop.</strong> Notice every gradient here was "some matrix times the error." $2X^\top(X\mathbf w-\mathbf y)$ is $X^\top$ times the residual; a layer's weight gradient will turn out to be its input times the upstream error. That recurring "$\text{input}^\top\times\text{error}$" shape is not a coincidence — it's the chain rule, which the next lesson makes explicit.</p>`
};

/* ------------------------------------------------------------------ 14.3 */
window.LESSON_CONTENT["14.3"] = {
  subtitle: "The chain rule when everything is a vector: multiply Jacobians — and the ordering trick that turns that product into backpropagation.",

  aiMoment: String.raw`Backpropagation <em>is</em> the vector chain rule, run in a specific order. A deep network is a composition $L(f_n(\dots f_2(f_1(\mathbf x))))$; each layer contributes a Jacobian, and the gradient is the product of all of them. The reason <code>loss.backward()</code> costs about the same as the forward pass — rather than exploding — is a single associativity trick: multiply those Jacobians <em>output-side first</em>, always against a vector. Get this lesson and backprop is no longer an algorithm to memorize; it's a consequence you can derive.`,

  plainEnglish: String.raw`To differentiate a chain of nested functions, you multiply how sensitive each step is to the one before it. With vectors, "how sensitive" is a Jacobian matrix, and "multiply" is matrix multiplication. The clever part is the order: if you start from the loss and work backward, every multiplication is a matrix times a <em>vector</em> (cheap), and you never build a giant matrix in between.`,

  intuition: String.raw`Line up the maps $\mathbf x\to\mathbf y\to\mathbf z$. Going forward you compute values. Going backward you compute gradients by stacking Jacobians: $J_{z/x}=J_{z/y}J_{y/x}$. Here's the whole efficiency of backprop in one observation: a row-vector times a matrix, $\mathbf v^\top J$, is another row-vector — cheap. But matrix times matrix, $J_2J_1$, is expensive and huge. So reverse mode <em>never</em> forms $J_2J_1$; it carries the vector $\mathbf v$ leftward, multiplying $\mathbf v^\top J_2$, then $(\mathbf v^\top J_2)J_1$, one cheap vector–Jacobian product at a time.`,

  formalism: String.raw`For $\mathbf z=g(\mathbf y)$ and $\mathbf y=f(\mathbf x)$, the chain rule composes Jacobians:
$$J_{z/x}=J_{z/y}\,J_{y/x}\qquad(\text{output-side Jacobian on the left}).$$
For a scalar loss $L$, transpose to get the gradient (a "vector–Jacobian product", VJP):
$$\frac{\partial L}{\partial\mathbf x}=J_{y/x}^\top\,\frac{\partial L}{\partial\mathbf y}.$$
For a chain of maps $\mathbf x_0\to\mathbf x_1\to\dots\to\mathbf x_n\to L$, the gradient is
$$\frac{\partial L}{\partial\mathbf x_0}=J_1^\top J_2^\top\cdots J_n^\top\,\frac{\partial L}{\partial\mathbf x_n},$$
evaluated <strong>right to left</strong> so each step is a matrix times a vector.`,

  derivation: String.raw`<strong>From entrywise chain rule to backprop's ordering.</strong>
<ol>
<li><strong>Entrywise chain rule.</strong> With $\mathbf z=g(\mathbf y)$, $\mathbf y=f(\mathbf x)$: $\frac{\partial z_i}{\partial x_k}=\sum_j\frac{\partial z_i}{\partial y_j}\frac{\partial y_j}{\partial x_k}.$ That sum over the shared index $j$ is <em>exactly</em> the definition of the matrix product $(J_{z/y}J_{y/x})_{ik}$. So the vector chain rule is just matrix multiplication of Jacobians.</li>
<li><strong>Transpose to a gradient.</strong> For scalar $L$, $\frac{\partial L}{\partial\mathbf x}=J_{y/x}^\top\frac{\partial L}{\partial\mathbf y}$. The transpose appears because the gradient (shape rule!) is a column matching $\mathbf x$, so we hit the upstream gradient with $J^\top$, not $J$.</li>
<li><strong>Associativity is the whole game.</strong> The gradient of a deep net is $J_1^\top J_2^\top\cdots J_n^\top\mathbf g$ (with $\mathbf g=\partial L/\partial\mathbf x_n$). Matrix multiplication is associative, so you may bracket it either way:
$$\underbrace{\big(J_1^\top(J_2^\top(\cdots(J_n^\top\mathbf g)))\big)}_{\text{reverse mode: vectors}}\quad=\quad\underbrace{\big(J_1^\top J_2^\top\cdots J_n^\top\big)\mathbf g}_{\text{forward: build the matrix}}.$$</li>
<li><strong>Reverse wins for ML.</strong> Bracketing right-to-left keeps a <em>vector</em> the whole way, so each step costs $O(\text{layer size}^2)$ — the same as the forward pass — and there's one such pass total. Building the full product first would cost matrix–matrix products and enormous memory. One scalar output, many parameters $\Rightarrow$ reverse mode. That is backpropagation.</li>
</ol>`,

  code: [
    { label: "Chain rule = Jacobian product; backprop = vector–Jacobian product",
      src: String.raw`import numpy as np
rng = np.random.default_rng(1)

# A 2-step map:   x  ->  y = tanh(W1 x)  ->  z = W2 y
W1 = rng.standard_normal((4, 3)); W2 = rng.standard_normal((2, 4))
f = lambda x: W2 @ np.tanh(W1 @ x)
x = rng.standard_normal(3)

# local Jacobians
a  = W1 @ x
J1 = np.diag(1 - np.tanh(a)**2) @ W1        # dy/dx  (4x3): tanh'(a) scales each row of W1
J2 = W2                                      # dz/dy  (2x4): a linear map's Jacobian is itself
J_total = J2 @ J1                            # chain rule: output-side Jacobian on the LEFT (2x3)

# check against a numeric Jacobian
Jn = np.zeros((2, 3)); h = 1e-6
for j in range(3):
    e = np.zeros(3); e[j] = h
    Jn[:, j] = (f(x+e) - f(x-e)) / (2*h)
print("J2 @ J1 matches numeric Jacobian:", np.allclose(J_total, Jn))

# Reverse mode never forms J_total. It carries a VECTOR backward:  x.grad = J1ᵀ (J2ᵀ v)
v = rng.standard_normal(2)                   # an upstream gradient dL/dz
grad_x = J1.T @ (J2.T @ v)                    # right-to-left, one matrix-times-vector at a time
print("VJP equals J_totalᵀ v :", np.allclose(grad_x, J_total.T @ v))` }
  ],

  diagram: String.raw`<svg viewBox="0 0 660 196" width="100%" style="max-width:660px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <defs><marker id="t3f" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#8a97b3"/></marker>
        <marker id="t3b" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#c1121f"/></marker></defs>
  <rect x="30" y="66" width="92" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="76" y="91" text-anchor="middle" fill="#1f2a44">x ∈ ℝ³</text>
  <rect x="284" y="66" width="92" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="330" y="91" text-anchor="middle" fill="#1f2a44">y ∈ ℝ⁴</text>
  <rect x="538" y="66" width="92" height="40" rx="6" fill="#e7f0e8" stroke="#3a7d44"/><text x="584" y="91" text-anchor="middle" fill="#245030">z ∈ ℝ²</text>
  <line x1="122" y1="78" x2="282" y2="78" stroke="#8a97b3" marker-end="url(#t3f)"/>
  <line x1="376" y1="78" x2="536" y2="78" stroke="#8a97b3" marker-end="url(#t3f)"/>
  <text x="202" y="70" text-anchor="middle" fill="#5a6b8c">J₁ = ∂y/∂x</text>
  <text x="456" y="70" text-anchor="middle" fill="#5a6b8c">J₂ = ∂z/∂y</text>
  <line x1="282" y1="98" x2="122" y2="98" stroke="#c1121f" stroke-dasharray="4 3" marker-end="url(#t3b)"/>
  <line x1="536" y1="98" x2="376" y2="98" stroke="#c1121f" stroke-dasharray="4 3" marker-end="url(#t3b)"/>
  <text x="202" y="120" text-anchor="middle" fill="#c1121f">J₁ᵀ(J₂ᵀv) = ∇ₓL</text>
  <text x="456" y="120" text-anchor="middle" fill="#c1121f">J₂ᵀ v</text>
  <text x="584" y="128" text-anchor="middle" fill="#c1121f">v = ∇_z L</text>
  <text x="330" y="150" text-anchor="middle" fill="#4a5878">forward: compute values (grey) → &#160;&#160; backward: carry a vector (red) ←</text>
  <text x="330" y="170" text-anchor="middle" fill="#5a6b8c" font-size="10.5">reverse mode never builds J₂J₁ — it multiplies a vector by one Jacobian at a time</text>
</svg>`,

  keyPoints: [
    String.raw`The vector chain rule is matrix multiplication of Jacobians: $J_{z/x}=J_{z/y}J_{y/x}$ (output-side Jacobian on the left).`,
    String.raw`For a scalar loss, propagate gradients with the transpose: $\partial L/\partial\mathbf x=J_{y/x}^\top(\partial L/\partial\mathbf y)$ — a <strong>vector–Jacobian product</strong>.`,
    String.raw`Backprop = evaluating $J_1^\top J_2^\top\cdots J_n^\top\mathbf g$ <strong>right to left</strong>. Associativity lets you keep a vector throughout instead of forming giant matrices.`,
    String.raw`Cost: reverse mode gets <em>all</em> input gradients in one pass ≈ the forward cost. That's the payoff of one scalar output and many inputs.`,
    String.raw`You rarely need a full Jacobian — you need $J^\top\mathbf v$. Every autodiff op is implemented as this VJP, not as a materialized matrix.`
  ],

  commonMistakes: [
    { wrong: "Multiplying the Jacobians in forward order, J₁ J₂, and getting a shape error.",
      why: String.raw`The output-side Jacobian goes on the <em>left</em>: $J_{z/x}=J_{z/y}J_{y/x}=J_2J_1$. With $J_2$ being $2\times4$ and $J_1$ being $4\times3$, only $J_2J_1$ conforms. Reversed, the shapes don't even match.` },
    { wrong: "Propagating gradients with J instead of Jᵀ.",
      why: String.raw`The forward Jacobian maps input-perturbations to output-perturbations; the gradient flows the other way, so you multiply the upstream gradient by $J^\top$. Using $J$ leaves you transposed (and often shape-broken).` },
    { wrong: "Materializing the full Jacobian of a big layer to 'be safe'.",
      why: String.raw`A $d\times d$ Jacobian for $d=10^4$ is $10^8$ numbers — you'll run out of memory. The entire point is the VJP $J^\top\mathbf v$, which needs only the vector. Think in vector–Jacobian products, not matrices.` }
  ],

  quiz: [
    { q: "J₂ is 2×4 and J₁ is 4×3. What is the shape of the chained Jacobian ∂z/∂x?",
      options: ["2×3", "3×2", "4×4", "2×4"], answer: 0,
      explain: String.raw`$J_{z/x}=J_2J_1$ is $(2\times4)(4\times3)=2\times3$ — rows = final outputs (2), columns = original inputs (3).` },
    { q: "To send a scalar loss's gradient from y back to x, you compute…",
      options: ["J_{y/x}ᵀ · (∂L/∂y)", "J_{y/x} · (∂L/∂y)", "(∂L/∂y) · J_{y/x}", "J_{y/x}ᵀ · J_{y/x}"], answer: 0,
      explain: String.raw`Gradients flow via the transpose: $\partial L/\partial\mathbf x=J_{y/x}^\top(\partial L/\partial\mathbf y)$, the vector–Jacobian product.` },
    { q: "Why does reverse mode bracket the product J₁ᵀJ₂ᵀ⋯Jₙᵀ g from the right?",
      options: ["So every step is a matrix times a vector (cheap), never matrix times matrix",
                "Because matrix multiplication isn't associative",
                "To compute the loss faster in the forward pass",
                "It doesn't matter; both orders cost the same"], answer: 0,
      explain: String.raw`Right-to-left keeps a vector at every stage — $O(\text{size}^2)$ per step and $O(1)$ passes. Left-to-right would build huge intermediate matrices. Associativity guarantees the same answer.` },
    { q: "A layer maps ℝ¹⁰⁰⁰ → ℝ¹⁰⁰⁰. Its Jacobian has how many entries, and what does backprop store instead?",
      options: ["1,000,000 entries; backprop stores only the length-1000 gradient vector",
                "1000 entries; it stores the matrix",
                "2000 entries; it stores both",
                "1,000,000 entries; it stores all of them"], answer: 0,
      explain: String.raw`The Jacobian is $1000\times1000=10^6$ numbers, but the VJP needs only $J^\top\mathbf v$ — a length-1000 vector. That memory saving is why deep nets are trainable.` },
    { q: "Forward-mode autodiff would be the better choice when…",
      options: ["there are few inputs and many outputs (a tall Jacobian)",
                "there is one scalar loss and millions of parameters",
                "always — it's strictly faster",
                "the function is linear"], answer: 0,
      explain: String.raw`Forward mode costs one pass per input; reverse costs one per output. Few inputs / many outputs favors forward. ML is the opposite extreme (one loss, many params), so it uses reverse.` }
  ],

  practice: [
    { level: "easy", prompt: "For y = Wx (W an m×n matrix), what is the Jacobian ∂y/∂x, and how does a gradient ∂L/∂y flow back to x?",
      solution: String.raw`$\partial\mathbf y/\partial\mathbf x=W$; the gradient flows as $\partial L/\partial\mathbf x=W^\top(\partial L/\partial\mathbf y)$. This one VJP is the backbone of every linear layer (next lesson).` },
    { level: "easy", prompt: "For an elementwise activation y = σ(x), what does its Jacobian look like, and why is the VJP just an elementwise multiply?",
      solution: String.raw`The Jacobian is diagonal, $\mathrm{diag}(\sigma'(x_1),\dots)$. Multiplying a vector by a diagonal matrix is an elementwise product, so backprop through an activation is simply $\partial L/\partial\mathbf x=\sigma'(\mathbf x)\odot(\partial L/\partial\mathbf y)$ — no matrix needed.` },
    { level: "med", prompt: "Add a third layer to the code (z → w = W3 z) and verify J3 J2 J1 matches the numeric Jacobian, then that J1ᵀJ2ᵀJ3ᵀv equals its transpose times v.",
      solution: String.raw`Stack another linear map; the total Jacobian is $J_3J_2J_1$ and the VJP is $J_1^\top(J_2^\top(J_3^\top\mathbf v))$. Both match their numeric/explicit counterparts, showing the pattern chains to any depth.` },
    { level: "med", prompt: "Count the multiply–adds to compute the gradient of a 3-layer net two ways: (a) build J₃J₂J₁ then transpose-multiply, vs (b) three VJPs. Assume each layer is d×d.",
      solution: String.raw`(a) Two matrix–matrix products cost $\sim2d^3$ plus a matvec; (b) three matrix–vector products cost $\sim3d^2$. For $d=1000$ that's ~$2\times10^9$ vs ~$3\times10^6$ — a thousandfold gap that grows with depth and width.` },
    { level: "hard", prompt: "Explain why the forward pass must cache its intermediate activations for the backward pass, and what that implies for memory.",
      solution: String.raw`Each layer's local Jacobian depends on its input/output (e.g. $\tanh'(a)$ needs $a$; a linear layer's $\partial L/\partial W$ needs the layer input). So reverse mode must <em>store</em> every forward activation until it's used on the way back — memory grows with depth×width. That's exactly what gradient checkpointing trades away by recomputing activations instead of storing them.` },
    { level: "hard", prompt: "Reverse mode gives J^T v for a chosen v. How would you recover a specific single row of the Jacobian, and a single column, using VJPs and JVPs?",
      solution: String.raw`Row $i$ of $J$ is $\mathbf e_i^\top J=(J^\top\mathbf e_i)^\top$ — one reverse-mode VJP with $\mathbf v=\mathbf e_i$. Column $j$ is $J\mathbf e_j$ — one <em>forward</em>-mode Jacobian–vector product (JVP) with the input perturbation $\mathbf e_j$. Whole Jacobians are built by looping one or the other, which is why materializing them is expensive.` }
  ],

  deepDive: String.raw`<p><strong>Two modes, one chain rule.</strong> Forward-mode autodiff pushes a perturbation $\dot{\mathbf x}$ forward, computing $J\dot{\mathbf x}$ (a Jacobian–vector product) in lockstep with the function; reverse-mode pulls a cotangent $\bar{\mathbf y}$ backward, computing $J^\top\bar{\mathbf y}$ (a VJP). Both are the same chain rule bracketed differently. Real systems (JAX especially) expose both — <code>jvp</code> and <code>vjp</code> — and build everything else (Hessians, Jacobians, Hessian–vector products) by composing them.</p>
<p><strong>The activation-cache tax.</strong> Because each layer's VJP needs that layer's forward values, training memory scales with the number of stored activations, not just parameters. This is why batch size and sequence length hit memory limits, why <em>gradient checkpointing</em> (recompute instead of store) exists, and why inference — which needs no backward pass — is so much lighter than training on the same model.</p>
<p><strong>Why "vector–Jacobian product" is the right primitive.</strong> Frameworks never register a "Jacobian" for an operation; they register its VJP rule. Add, matmul, softmax, convolution — each ships a function that takes the upstream gradient and returns the downstream one, without ever forming a matrix. When you write a custom autograd op (as in the tiny-autograd capstone), the one thing you must supply is precisely this VJP.</p>`
};

/* ------------------------------------------------------------------ 14.4 */
window.LESSON_CONTENT["14.4"] = {
  subtitle: "Derive the three gradients of a dense layer — ∂L/∂W, ∂L/∂b, ∂L/∂X — by hand, and watch the 'input times error' pattern fall out of the identities.",

  aiMoment: String.raw`<code>nn.Linear</code> is the most common layer in deep learning — it's every fully-connected layer, every projection in an attention block, the readout head on top of almost any model. When you call <code>.backward()</code>, PyTorch runs exactly the three gradient formulas you're about to derive. Deriving them once, by hand, is the moment backprop stops being a black box: you'll see that a weight's gradient is just its input times the incoming error, and you'll recognize that same shape everywhere.`,

  plainEnglish: String.raw`A dense layer multiplies its input by a weight matrix and adds a bias. To train it you need three things: how to nudge the weights, how to nudge the bias, and how to pass the blame back to the input. All three come from one quantity — how wrong the output was (the upstream gradient) — combined with the layer's own input. In words: <em>weight gradient = input × error</em>, <em>bias gradient = sum of errors</em>, <em>input gradient = error × weights</em>.`,

  intuition: String.raw`Write the layer as $Y=XW+b$ with a batch of $N$ rows. The loss $L$ sits downstream and hands the layer an upstream gradient $\delta=\partial L/\partial Y$ (same shape as $Y$). Each of the three gradients is $\delta$ combined with whatever it was multiplied against in the forward pass: the weights saw the input $X$, so $\partial L/\partial W$ pairs $\delta$ with $X$; the bias was just added to every row, so its gradient sums $\delta$ down the batch; the input passed through $W$, so its gradient sends $\delta$ back through $W^\top$. Nothing here is new — it's the chain rule plus last lesson's linear-map Jacobian.`,

  formalism: String.raw`For $Y=XW+b$ with $X\in\mathbb R^{N\times d_\text{in}}$, $W\in\mathbb R^{d_\text{in}\times d_\text{out}}$, $b\in\mathbb R^{d_\text{out}}$, and upstream gradient $\delta=\dfrac{\partial L}{\partial Y}\in\mathbb R^{N\times d_\text{out}}$:
$$\frac{\partial L}{\partial W}=X^\top\delta,\qquad
\frac{\partial L}{\partial b}=\sum_{n=1}^{N}\delta_{n,:},\qquad
\frac{\partial L}{\partial X}=\delta\,W^\top.$$
Check the shapes: $X^\top\delta$ is $d_\text{in}\times d_\text{out}$ (matches $W$); the bias sum is length $d_\text{out}$ (matches $b$); $\delta W^\top$ is $N\times d_\text{in}$ (matches $X$). The shape rule audits every one.`,

  derivation: String.raw`<strong>All three gradients from the chain rule.</strong> Work per-element, then recognize the matrix form.
<ol>
<li><strong>Set up.</strong> $Y_{nk}=\sum_i X_{ni}W_{ik}+b_k$, and we're given $\delta_{nk}=\partial L/\partial Y_{nk}$.</li>
<li><strong>Weights.</strong> $W_{ik}$ affects every row's output $k$: $\frac{\partial L}{\partial W_{ik}}=\sum_n\frac{\partial L}{\partial Y_{nk}}\frac{\partial Y_{nk}}{\partial W_{ik}}=\sum_n\delta_{nk}X_{ni}=(X^\top\delta)_{ik}.$ So $\partial L/\partial W=X^\top\delta$ — literally input$^\top$ times error.</li>
<li><strong>Bias.</strong> $b_k$ is added to output $k$ in every row: $\frac{\partial L}{\partial b_k}=\sum_n\delta_{nk}\frac{\partial Y_{nk}}{\partial b_k}=\sum_n\delta_{nk}.$ So $\partial L/\partial b$ is $\delta$ summed over the batch dimension.</li>
<li><strong>Input.</strong> $X_{ni}$ affects output $k$ through $W_{ik}$: $\frac{\partial L}{\partial X_{ni}}=\sum_k\delta_{nk}\frac{\partial Y_{nk}}{\partial X_{ni}}=\sum_k\delta_{nk}W_{ik}=(\delta W^\top)_{ni}.$ So $\partial L/\partial X=\delta W^\top$ — the error sent back through the transposed weights, exactly the VJP of last lesson.</li>
<li><strong>Stack layers.</strong> Feed $\partial L/\partial X$ as the next layer's $\delta$ and repeat. That handoff, layer by layer, <em>is</em> backpropagation through a whole network.</li>
</ol>`,

  code: [
    { label: "A dense layer's forward + backward, gradient-checked",
      src: String.raw`import numpy as np
rng = np.random.default_rng(2)

# One linear layer on a batch:  Y = X W + b
N, d_in, d_out = 5, 3, 2
X = rng.standard_normal((N, d_in)); W = rng.standard_normal((d_in, d_out)); b = rng.standard_normal(d_out)
T = rng.standard_normal((N, d_out))               # a target for a simple MSE loss

def forward(X, W, b): return X @ W + b
def loss(X, W, b):    return 0.5 * ((forward(X, W, b) - T)**2).sum()

# Backward: from the upstream gradient dL/dY, the three gradients fall right out.
Y  = forward(X, W, b)
dY = Y - T                                         # dL/dY for this MSE
dW = X.T @ dY                                       # input^T @ error   -> shape of W  (in x out)
db = dY.sum(0)                                      # sum errors over batch -> shape of b (out,)
dX = dY @ W.T                                       # error @ W^T        -> shape of X  (N x in)

# gradient-check every one against finite differences
def numeric_grad(f, P, h=1e-6):
    g = np.zeros_like(P)
    for i in np.ndindex(P.shape):
        old = P[i]; P[i]=old+h; fp=f(P); P[i]=old-h; fm=f(P); P[i]=old
        g[i] = (fp - fm) / (2*h)
    return g

print("dW correct:", np.allclose(dW, numeric_grad(lambda W: loss(X, W, b), W.copy())))
print("db correct:", np.allclose(db, numeric_grad(lambda b: loss(X, W, b), b.copy())))
print("dX correct:", np.allclose(dX, numeric_grad(lambda X: loss(X, W, b), X.copy())))` }
  ],

  diagram: String.raw`<svg viewBox="0 0 680 212" width="100%" style="max-width:680px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <defs><marker id="t4f" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#8a97b3"/></marker>
        <marker id="t4b" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#c1121f"/></marker></defs>
  <rect x="24" y="70" width="94" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="71" y="95" text-anchor="middle" fill="#1f2a44">X (N×in)</text>
  <rect x="266" y="62" width="150" height="56" rx="8" fill="#dbe8fb" stroke="#2a6f97"/><text x="341" y="95" text-anchor="middle" fill="#123a5a" font-weight="700">Y = X·W + b</text>
  <rect x="560" y="70" width="96" height="40" rx="6" fill="#e7f0e8" stroke="#3a7d44"/><text x="608" y="95" text-anchor="middle" fill="#245030">Y (N×out)</text>
  <line x1="118" y1="82" x2="264" y2="82" stroke="#8a97b3" marker-end="url(#t4f)"/>
  <line x1="416" y1="82" x2="558" y2="82" stroke="#8a97b3" marker-end="url(#t4f)"/>
  <line x1="558" y1="100" x2="418" y2="100" stroke="#c1121f" stroke-dasharray="4 3" marker-end="url(#t4b)"/>
  <line x1="264" y1="100" x2="120" y2="100" stroke="#c1121f" stroke-dasharray="4 3" marker-end="url(#t4b)"/>
  <text x="490" y="118" text-anchor="middle" fill="#c1121f">δ = ∂L/∂Y</text>
  <text x="188" y="118" text-anchor="middle" fill="#c1121f">∂L/∂X = δ Wᵀ</text>
  <text x="341" y="150" text-anchor="middle" fill="#8f2233" font-weight="700">the three gradients from one error δ</text>
  <text x="341" y="172" text-anchor="middle" fill="#1f2a44">∂L/∂W = Xᵀδ  &#160;(input × error)</text>
  <text x="341" y="190" text-anchor="middle" fill="#1f2a44">∂L/∂b = Σₙ δ  &#160;(sum over batch)</text>
  <text x="341" y="208" text-anchor="middle" fill="#1f2a44">∂L/∂X = δWᵀ  &#160;(error back through Wᵀ)</text>
</svg>`,

  keyPoints: [
    String.raw`For $Y=XW+b$ with upstream $\delta=\partial L/\partial Y$: $\ \partial L/\partial W=X^\top\delta,\ \ \partial L/\partial b=\sum_n\delta,\ \ \partial L/\partial X=\delta W^\top.$`,
    String.raw`<strong>Weight gradient = input$^\top$ × error.</strong> That "outer-product of activations and errors" is the single most repeated computation in all of training.`,
    String.raw`The bias gradient sums the error over the batch, because one bias is broadcast to every example — backprop through a broadcast is a sum.`,
    String.raw`The input gradient $\delta W^\top$ is exactly the vector–Jacobian product from 14.3 (the layer's Jacobian is $W$). Passing it on as the next $\delta$ is the whole backward pass.`,
    String.raw`Every shape is forced by the shape rule: $\partial L/\partial W$ matches $W$, $\partial L/\partial X$ matches $X$. If a shape is off, an index is misplaced.`
  ],

  commonMistakes: [
    { wrong: "Writing ∂L/∂W = δᵀX or δ Xᵀ instead of Xᵀδ.",
      why: String.raw`Only $X^\top\delta$ has the shape of $W$ ($d_\text{in}\times d_\text{out}$). The others are transposed or non-conforming. When unsure, let the shape rule pick the arrangement that matches $W$.` },
    { wrong: "Forgetting to sum the bias gradient over the batch.",
      why: String.raw`The bias is added identically to all $N$ rows, so its gradient collects contributions from all of them: $\partial L/\partial b=\sum_n\delta_{n,:}$. Using a single row's $\delta$ under-counts the gradient by a factor of the batch size.` },
    { wrong: "Reusing X from the wrong step when computing ∂L/∂W deep in a network.",
      why: String.raw`$\partial L/\partial W$ needs <em>this layer's own input</em>, cached from the forward pass. Grabbing a different layer's activations is a subtle bug that a per-parameter finite-difference check catches instantly.` }
  ],

  quiz: [
    { q: "For Y = XW + b, the weight gradient ∂L/∂W (with δ = ∂L/∂Y) is…",
      options: ["Xᵀδ", "δXᵀ", "Wᵀδ", "δWᵀ"], answer: 0,
      explain: String.raw`$\partial L/\partial W=X^\top\delta$ — input$^\top$ times error, the only arrangement with $W$'s shape $d_\text{in}\times d_\text{out}$.` },
    { q: "X is 64×256, W is 256×10. What shape is ∂L/∂W?",
      options: ["256×10 (same as W)", "64×10", "256×256", "64×256"], answer: 0,
      explain: String.raw`Gradients match their variable: $\partial L/\partial W$ is $256\times10$. Concretely $X^\top\delta=(256\times64)(64\times10)$.` },
    { q: "Why is the bias gradient a sum over the batch dimension?",
      options: ["The bias is broadcast to every example, and backprop through a broadcast is a sum",
                "Because bias is a scalar", "To normalize the learning rate", "It isn't — it's a mean"], answer: 0,
      explain: String.raw`Adding $b$ to all $N$ rows is a broadcast; each row contributes a gradient, and they add: $\partial L/\partial b=\sum_n\delta_{n,:}$.` },
    { q: "The input gradient ∂L/∂X = δWᵀ is an instance of which idea from the previous lesson?",
      options: ["A vector–Jacobian product (the layer's Jacobian is W)",
                "The normal equations", "The shape rule failing", "Forward-mode autodiff"], answer: 0,
      explain: String.raw`A linear layer's Jacobian is $W$, so propagating the gradient is $W^\top$ applied to $\delta$ — exactly the VJP $J^\top\mathbf v$.` },
    { q: "In a 2-layer net, what becomes the first layer's upstream gradient δ₁?",
      options: ["The second layer's input gradient ∂L/∂X₂ (passed back)",
                "The final loss value", "The labels", "Zero"], answer: 0,
      explain: String.raw`Each layer's $\partial L/\partial X$ is handed to the layer below as its $\delta$. That chaining of input-gradients is backprop through the network.` }
  ],

  practice: [
    { level: "easy", prompt: "For a single example (N = 1), what does ∂L/∂W = Xᵀδ become, and why is it called an outer product?",
      solution: String.raw`With $N=1$, $X$ is a row $\mathbf x^\top$ and $\delta$ a row $\boldsymbol\delta^\top$, so $\partial L/\partial W=\mathbf x\,\boldsymbol\delta^\top$ — the outer product of the input vector and the error vector, a full $d_\text{in}\times d_\text{out}$ matrix from two vectors.` },
    { level: "easy", prompt: "If you scale the loss by 1/N (a mean instead of a sum), how do the three gradients change?",
      solution: String.raw`Every gradient is divided by $N$: $\partial L/\partial W=\tfrac1N X^\top\delta$, etc. That's why switching between 'sum' and 'mean' reductions effectively rescales your learning rate by the batch size.` },
    { level: "med", prompt: "Add a ReLU after the layer (A = relu(Y)) and give the new backward pass from ∂L/∂A.",
      solution: String.raw`First backprop through ReLU: $\delta=\partial L/\partial Y=\partial L/\partial A\odot\mathbf 1[Y>0]$ (zero out where the pre-activation was negative). Then apply the same three formulas with this $\delta$. The elementwise mask is the activation's diagonal-Jacobian VJP.` },
    { level: "med", prompt: "Implement a reusable Linear class with forward(X) that caches X, and backward(dY) that returns dX and stores dW, db. Gradient-check it.",
      solution: String.raw`<code>forward</code> saves <code>self.X = X</code> and returns <code>X@W+b</code>; <code>backward(dY)</code> sets <code>self.dW = self.X.T@dY</code>, <code>self.db = dY.sum(0)</code>, and returns <code>dY@self.W.T</code>. Chaining several of these reproduces a full MLP's backward pass — the design every framework uses.` },
    { level: "hard", prompt: "Stack two linear layers with a tanh between them and derive the full gradient ∂L/∂W₁ for a squared loss.",
      solution: String.raw`With $H=\tanh(XW_1+b_1)$, $Y=HW_2+b_2$, $L=\tfrac12\lVert Y-T\rVert^2$: let $\delta_2=Y-T$; then $\delta_1=(\delta_2 W_2^\top)\odot(1-H^2)$ and $\partial L/\partial W_1=X^\top\delta_1$. The tanh derivative $1-H^2$ appears as the elementwise mask between the two linear backward steps — exactly the autograd capstone's math, now derived on paper.` },
    { level: "hard", prompt: "Show that the weight update W ← W − η Xᵀδ is a sum of rank-1 updates, and interpret what each training example contributes.",
      solution: String.raw`$X^\top\delta=\sum_n \mathbf x_n\boldsymbol\delta_n^\top$, a sum over the batch of outer products. Each example contributes a rank-1 nudge that raises the weights aligned with its input in the direction that reduces its own error. The batch gradient is the average of these per-example rank-1 corrections — the microscopic picture of what SGD actually does to a weight matrix.` }
  ],

  deepDive: String.raw`<p><strong>This one layer is 90% of a network's compute.</strong> Transformers are mostly big matrix multiplies: the QKV projections, the attention output projection, and the MLP blocks are all $Y=XW+b$. So the forward matmul $XW$ and its two backward matmuls ($X^\top\delta$ and $\delta W^\top$) dominate training FLOPs. The rule of thumb "backward costs about twice the forward" comes straight from these two gradient matmuls per one forward matmul.</p>
<p><strong>Why activations must be cached.</strong> Notice $\partial L/\partial W=X^\top\delta$ needs the layer's forward input $X$. Every layer must therefore keep its input around until the backward pass reaches it — which is why training memory scales with depth and batch size, and why inference (no backward) is so much lighter. Gradient checkpointing recomputes these $X$'s instead of storing them.</p>
<p><strong>The pattern generalizes.</strong> "Gradient = (thing it multiplied)$^\top$ × (upstream error)" recurs for convolutions (a structured matmul), embeddings (a gather, whose backward is a scatter-add), and attention. Once the dense layer clicks, new layers are just new Jacobians slotted into the same $J^\top\delta$ machinery — which is exactly how you'd add an op to the autograd engine from the capstones.</p>`
};

/* ------------------------------------------------------------------ 14.5 */
window.LESSON_CONTENT["14.5"] = {
  subtitle: "Softmax has a messy Jacobian and cross-entropy has a divide-by-p — yet together their gradient collapses to the cleanest formula in deep learning: p − y.",

  aiMoment: String.raw`The last step of almost every classifier, and the per-token loss of every language model, is softmax followed by cross-entropy. When you backprop through it, the gradient that flows into the logits is simply $\mathbf p-\mathbf y$ — the predicted distribution minus the one-hot (or target) distribution. That astonishing simplicity is not luck: it's a deliberate pairing, and it's why frameworks fuse "softmax + cross-entropy" into a single op. Deriving the cancellation once shows you why the two belong together.`,

  plainEnglish: String.raw`Softmax turns raw scores (logits) into probabilities. Cross-entropy measures how far those probabilities are from the truth. To improve, you nudge each logit by exactly "how much probability it has minus how much it should have." Too much probability on a wrong class? Push that logit down. Not enough on the right class? Push it up. That's the entire gradient: <em>predicted minus target</em>.`,

  intuition: String.raw`Taken separately these look painful: the softmax Jacobian is a full matrix $\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$, and cross-entropy's derivative has a $-y_i/p_i$ with a $p$ in the denominator. But when you chain them, the $p_i$ from the Jacobian cancels the $1/p_i$ from the loss, and everything collapses to $p_j-y_j$. The picture below shows it: the gradient is positive on the classes that stole probability (push their logits down) and negative on the true class (push its logit up), each in exact proportion to the error.`,

  formalism: String.raw`Softmax and its Jacobian:
$$p_i=\frac{e^{z_i}}{\sum_k e^{z_k}},\qquad \frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),$$
where $\delta_{ij}$ is 1 if $i=j$ else 0. Cross-entropy against a target distribution $\mathbf y$ (one-hot for a hard label):
$$L=-\sum_i y_i\log p_i.$$
Chaining them gives the headline result:
$$\boxed{\ \frac{\partial L}{\partial z_j}=p_j-y_j\ }\qquad(\text{for a batch with a mean loss, divide by }N).$$`,

  derivation: String.raw`<strong>The famous cancellation, step by step.</strong>
<ol>
<li><strong>Softmax Jacobian.</strong> For $i=j$: $\frac{\partial p_i}{\partial z_i}=p_i(1-p_i)$. For $i\ne j$: $\frac{\partial p_i}{\partial z_j}=-p_ip_j$. Both cases are $p_i(\delta_{ij}-p_j)$ (quotient rule on $e^{z_i}/\sum_k e^{z_k}$).</li>
<li><strong>Loss derivative w.r.t. the probabilities.</strong> $\frac{\partial L}{\partial p_i}=-\frac{y_i}{p_i}.$</li>
<li><strong>Chain them.</strong> $\displaystyle\frac{\partial L}{\partial z_j}=\sum_i\frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j}=\sum_i\Big(-\frac{y_i}{p_i}\Big)\,p_i(\delta_{ij}-p_j)=-\sum_i y_i(\delta_{ij}-p_j).$ The $p_i$ and $1/p_i$ <em>cancel</em> — this is the whole trick.</li>
<li><strong>Finish the sum.</strong> $-\sum_i y_i\delta_{ij}+p_j\sum_i y_i=-y_j+p_j\cdot 1=p_j-y_j,$ using $\sum_i y_i=1$. Hence $\partial L/\partial\mathbf z=\mathbf p-\mathbf y$.</li>
</ol>
The same cancellation makes 2-class softmax collapse to the logistic (sigmoid) gradient $p-y$ — softmax+cross-entropy is the multi-class generalization of logistic regression.`,

  code: [
    { label: "Softmax + cross-entropy, and the p − y gradient",
      src: String.raw`import numpy as np

def softmax(z):
    z = z - z.max()                 # subtract the max for numerical stability (exp overflow)
    e = np.exp(z); return e / e.sum()

# logits z, and a one-hot target y (true class = index 2)
z = np.array([2.0, 1.0, 0.1, -0.5, 1.5]); y = np.zeros(5); y[2] = 1.0
p = softmax(z)
L = -(y * np.log(p)).sum()
print("probabilities:", np.round(p, 3), " loss:", round(float(L), 4))

# The famous result:  dL/dz = p - y   (predicted minus target)
grad = p - y

# verify against finite differences
def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]; x[i]=old+h; fp=f(x); x[i]=old-h; fm=f(x); x[i]=old
        g[i] = (fp - fm) / (2*h)
    return g
loss = lambda z: -(y * np.log(softmax(z))).sum()
print("dL/dz = p - y correct:", np.allclose(grad, numeric_grad(loss, z.copy())))
print("p - y =", np.round(grad, 3))       # + on wrong classes, - on the true class` }
  ],

  diagram: String.raw`<svg viewBox="0 0 520 222" width="100%" style="max-width:520px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <text x="260" y="18" text-anchor="middle" font-weight="700" fill="#1f2a44">the gradient into the logits is  ∂L/∂z = p − y</text>
  <line x1="40" y1="120" x2="470" y2="120" stroke="#33415c" stroke-width="1"/>
  <rect x="67"  y="87"  width="46" height="33" fill="#e4909b" stroke="#c1121f"/>
  <rect x="147" y="108" width="46" height="12" fill="#e4909b" stroke="#c1121f"/>
  <rect x="227" y="120" width="46" height="69" fill="#8fc7a0" stroke="#2f7d4f"/>
  <rect x="307" y="117" width="46" height="3"  fill="#e4909b" stroke="#c1121f"/>
  <rect x="387" y="100" width="46" height="20" fill="#e4909b" stroke="#c1121f"/>
  <text x="90"  y="80"  text-anchor="middle" fill="#8f2233">+0.45</text>
  <text x="170" y="102" text-anchor="middle" fill="#8f2233">+0.17</text>
  <text x="250" y="203" text-anchor="middle" fill="#2f7d4f">−0.93</text>
  <text x="330" y="110" text-anchor="middle" fill="#8f2233">+0.04</text>
  <text x="410" y="93"  text-anchor="middle" fill="#8f2233">+0.28</text>
  <text x="90"  y="136" text-anchor="middle" fill="#5a6b8c" font-size="10">class 0</text>
  <text x="170" y="136" text-anchor="middle" fill="#5a6b8c" font-size="10">class 1</text>
  <text x="250" y="136" text-anchor="middle" fill="#2f7d4f" font-size="10" font-weight="700">class 2 (true)</text>
  <text x="330" y="136" text-anchor="middle" fill="#5a6b8c" font-size="10">class 3</text>
  <text x="410" y="136" text-anchor="middle" fill="#5a6b8c" font-size="10">class 4</text>
  <text x="260" y="216" text-anchor="middle" fill="#4a5878" font-size="10.5">red: p&gt;y → push that logit DOWN&#160;&#160;·&#160;&#160;green: p&lt;y → push the TRUE logit UP</text>
</svg>`,

  keyPoints: [
    String.raw`Softmax+cross-entropy has the cleanest gradient in ML: $\partial L/\partial\mathbf z=\mathbf p-\mathbf y$ (divide by $N$ for a batch mean).`,
    String.raw`The $p_i$ in the softmax Jacobian cancels the $1/p_i$ in the cross-entropy derivative — that cancellation is <em>why</em> the two are always paired.`,
    String.raw`The softmax Jacobian itself is $\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$: diagonal term $p_i(1-p_i)$, off-diagonal $-p_ip_j$.`,
    String.raw`Interpretation: the gradient pushes down every logit holding too much probability and pushes up the true logit, each proportional to the error $p-y$.`,
    String.raw`Frameworks fuse softmax and cross-entropy into one op (e.g. <code>cross_entropy(logits, y)</code>) — both for this clean gradient and for numerical stability (log-sum-exp).`
  ],

  commonMistakes: [
    { wrong: "Computing softmax without subtracting the max, then overflowing to inf/nan.",
      why: String.raw`$e^{z}$ overflows for large logits. Subtracting $\max_k z_k$ first is exact (it cancels in the ratio) and keeps every exponent $\le 0$. This is the log-sum-exp trick, and it's why you pass <em>logits</em>, not probabilities, to the loss.` },
    { wrong: "Taking softmax then log separately (log(softmax(z))) instead of a fused log-softmax.",
      why: String.raw`Doing it in two steps can log a value that underflowed to 0, giving $-\infty$. <code>log_softmax</code> computes $z_i-\log\sum_k e^{z_k}$ directly — stable, and it's what a fused cross-entropy uses internally.` },
    { wrong: "Forgetting the 1/N when the loss is a batch mean but the gradient uses a sum.",
      why: String.raw`If $L=\frac1N\sum_n\text{CE}_n$, then $\partial L/\partial Z=\frac1N(P-Y)$. Mismatching the reduction inflates the gradient by a factor of $N$ and effectively multiplies your learning rate by the batch size.` }
  ],

  quiz: [
    { q: "The gradient of softmax+cross-entropy w.r.t. the logits is…",
      options: ["p − y", "y − p", "p(1 − p)", "−log p"], answer: 0,
      explain: String.raw`$\partial L/\partial\mathbf z=\mathbf p-\mathbf y$ — predicted minus target. (Some texts write the negative because they ascend the log-likelihood; here we descend the loss.)` },
    { q: "For the true class t (yₜ = 1), what is the gradient component ∂L/∂zₜ?",
      options: ["pₜ − 1 (negative, so the logit is pushed up)", "pₜ", "1 − pₜ", "0"], answer: 0,
      explain: String.raw`$\partial L/\partial z_t=p_t-y_t=p_t-1\le 0$. Negative gradient means gradient <em>descent</em> increases $z_t$ — it raises the true class's score.` },
    { q: "The diagonal of the softmax Jacobian, ∂pᵢ/∂zᵢ, equals…",
      options: ["pᵢ(1 − pᵢ)", "−pᵢ²", "pᵢ", "1"], answer: 0,
      explain: String.raw`From $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ with $i=j$: $p_i(1-p_i)$. Off-diagonal is $-p_ip_j$.` },
    { q: "Why can softmax subtract maxₖ zₖ from all logits with no effect on the probabilities?",
      options: ["A constant shift multiplies numerator and denominator by the same eᶜ, which cancels",
                "Because probabilities must sum to 1", "It changes them but negligibly", "Only if the logits are positive"], answer: 0,
      explain: String.raw`$e^{z_i-c}/\sum_k e^{z_k-c}=e^{-c}e^{z_i}/(e^{-c}\sum_k e^{z_k})$; the $e^{-c}$ cancels. Exact invariance, used for numerical stability.` },
    { q: "Logits are z with softmax p = (0.7, 0.2, 0.1) and the true class is 0. The gradient p − y is…",
      options: ["(−0.3, 0.2, 0.1)", "(0.7, 0.2, 0.1)", "(0.3, −0.2, −0.1)", "(0, 0, 0)"], answer: 0,
      explain: String.raw`$\mathbf y=(1,0,0)$, so $\mathbf p-\mathbf y=(0.7-1,\,0.2,\,0.1)=(-0.3,0.2,0.1)$: push class 0's logit up, the others down.` }
  ],

  practice: [
    { level: "easy", prompt: "Verify numerically that log_softmax(z) = z − logsumexp(z) matches log(softmax(z)) but is stabler for large z.",
      solution: String.raw`Compute <code>z - (z.max() + np.log(np.exp(z - z.max()).sum()))</code>. It equals <code>np.log(softmax(z))</code> on ordinary inputs but stays finite when <code>softmax(z)</code> would underflow a class to 0.0 and log it to $-\infty$.` },
    { level: "easy", prompt: "For a batch of N examples with logits Z (N×C) and one-hot targets Y, write the gradient of the mean cross-entropy w.r.t. Z.",
      solution: String.raw`$\partial L/\partial Z=(P-Y)/N$, where $P$ is the row-wise softmax of $Z$. One line: <code>(softmax_rows(Z) - Y) / N</code>. This is the exact tensor a classifier backprops from.` },
    { level: "med", prompt: "Derive the off-diagonal softmax Jacobian ∂pᵢ/∂zⱼ (i ≠ j) from the quotient rule and confirm it's −pᵢpⱼ.",
      solution: String.raw`$p_i=e^{z_i}/S$ with $S=\sum_k e^{z_k}$. For $j\ne i$, only $S$ depends on $z_j$: $\partial p_i/\partial z_j=e^{z_i}\cdot(-S^{-2})\cdot e^{z_j}=-p_ip_j$. For $i=j$ add the numerator's derivative to get $p_i-p_i^2=p_i(1-p_i)$.` },
    { level: "med", prompt: "With label smoothing, the target is y_smooth = (1−ε)·onehot + ε/C. What does the gradient p − y_smooth do differently?",
      solution: String.raw`The target is no longer 0 on wrong classes, so at the optimum the model is pushed toward $p_i=\varepsilon/C>0$ everywhere rather than a one-hot. The gradient stops driving the true logit to $+\infty$, which caps confidence and improves calibration — an information-theoretic regularizer (a KL-to-uniform term).` },
    { level: "hard", prompt: "Show that binary softmax (C = 2) reduces to the logistic/sigmoid gradient, i.e. that softmax+CE is multi-class logistic regression.",
      solution: String.raw`With two logits, $p_1=\frac{e^{z_1}}{e^{z_0}+e^{z_1}}=\sigma(z_1-z_0)$. Cross-entropy in terms of the single score $u=z_1-z_0$ becomes the logistic loss, and its gradient is $\sigma(u)-y=p_1-y_1$ — the sigmoid gradient. Softmax+CE is exactly the $C$-way generalization of logistic regression.` },
    { level: "hard", prompt: "Add a temperature T (p = softmax(z/T)) and derive ∂L/∂z. What happens to gradients as T → ∞?",
      solution: String.raw`By the chain rule $\partial L/\partial z_j=(p_j-y_j)/T$ with $p=\mathrm{softmax}(z/T)$. As $T\to\infty$, $p\to$ uniform and the gradient magnitude $\to 0$ (like $1/T$) — the model becomes maximally unsure and learns slowly. Temperature is the knob distillation uses to expose a teacher's 'dark knowledge' in the soft targets.` }
  ],

  deepDive: String.raw`<p><strong>Why the pairing is not arbitrary.</strong> Softmax is the exponential-family link that turns logits into a categorical distribution, and cross-entropy is that distribution's negative log-likelihood. Maximum-likelihood theory guarantees the gradient of a log-likelihood w.r.t. the natural parameters is "observed minus expected" — here $\mathbf y-\mathbf p$. So $\mathbf p-\mathbf y$ isn't a happy accident; it's the generic form of a maximum-likelihood gradient, which is also why it never saturates the way a mismatched loss (e.g. softmax with squared error) would.</p>
<p><strong>Stability by fusion.</strong> Because the clean gradient needs $\mathbf p$ but the stable forward needs log-sum-exp, frameworks implement one fused <code>cross_entropy(logits, target)</code> that computes $\log\text{-softmax}$ for the forward and returns $\mathbf p-\mathbf y$ for the backward. Passing already-softmaxed probabilities into such a function double-applies softmax and is a classic bug — always hand it raw logits.</p>
<p><strong>Where you've seen the pieces.</strong> The cross-entropy loss itself is the information-theory capstone (C.4); the $\mathbf p-\mathbf y$ gradient is what the autograd capstone (C.5) would compute if you added a softmax-CE op; and the linear layer of 14.4 sits right underneath, turning this $\delta=\mathbf p-\mathbf y$ into weight gradients $X^\top\delta$. This lesson is where the whole track meets the rest of the course: probability picks the loss, matrix calculus turns it into gradients, and the optimizer walks them downhill.</p>`
};

/* ------------------------------------------------------------------ 14.E */
window.LESSON_CONTENT["14.E"] = {
  exam: true,
  intro: String.raw`Ten problems on matrix calculus and backprop, roughly easy → hard. Budget about <strong>75 minutes</strong>. Work on paper first; keep a Python REPL open only to <em>check</em> your answers with finite differences (the same <code>numeric_grad</code> from the lessons), not to find them. A correct gradient always has the right <em>shape</em> — audit every answer against the shape rule before you check the numbers. Solutions are written out; grade yourself with the rubric at the end.`,
  problems: [
    { level: "easy", prompt: "Shapes. (a) f : ℝ⁷ → ℝ² — what shape is its Jacobian? (b) L is a scalar loss and W is 32×64 — what shape is ∂L/∂W? (c) Why does (b) make W − η·∂L/∂W a legal update?",
      solution: String.raw`(a) $2\times7$ (rows = outputs, cols = inputs). (b) $32\times64$, the same shape as $W$ (shape rule). (c) Because the gradient matches $W$ elementwise, the subtraction is well-defined entry by entry.` },
    { level: "easy", prompt: "Basic identities. Give ∂/∂x of (a) aᵀx, (b) xᵀx, (c) the constant c.",
      solution: String.raw`(a) $\mathbf a$. (b) $2\mathbf x$. (c) $\mathbf 0$ (a constant has zero gradient).` },
    { level: "med", prompt: "Quadratic form. For A = [[1, 2], [0, 3]] (not symmetric) and x = (x₁, x₂), compute ∂/∂x (xᵀAx) as a vector, and state the general rule.",
      solution: String.raw`General rule: $(A+A^\top)\mathbf x$. Here $A+A^\top=\begin{bmatrix}2&2\\2&6\end{bmatrix}$, so the gradient is $(2x_1+2x_2,\ 2x_1+6x_2)$. (Check: $x^\top Ax=x_1^2+2x_1x_2+3x_2^2$, whose partials are exactly these.)` },
    { level: "med", prompt: "Least squares. Derive ∂/∂w ‖Xw − y‖² and state the equation you get by setting it to zero.",
      solution: String.raw`Expand $\mathbf w^\top X^\top X\mathbf w-2\mathbf y^\top X\mathbf w+\mathbf y^\top\mathbf y$; differentiate to $2X^\top X\mathbf w-2X^\top\mathbf y=2X^\top(X\mathbf w-\mathbf y)$. Zeroing gives the normal equations $X^\top X\mathbf w=X^\top\mathbf y$.` },
    { level: "med", prompt: "Trace / Frobenius. Give (a) ∂/∂X tr(AX) and (b) ∂/∂X ‖X‖_F². State the shape of each.",
      solution: String.raw`(a) $A^\top$ (shape of $X$). (b) $\lVert X\rVert_F^2=\mathrm{tr}(X^\top X)=\sum_{ij}X_{ij}^2$, so the gradient is $2X$ (shape of $X$) — the matrix version of $2\mathbf x$.` },
    { level: "med", prompt: "Chain rule + VJP. A network step has local Jacobians J₁ (dy/dx, 4×3) and J₂ (dz/dy, 2×4). (a) What is the shape of ∂z/∂x and how is it formed? (b) Given an upstream gradient v = ∂L/∂z, write ∂L/∂x and give its shape.",
      solution: String.raw`(a) $J_{z/x}=J_2J_1$, shape $2\times3$ (output-side Jacobian on the left). (b) $\partial L/\partial\mathbf x=J_1^\top(J_2^\top\mathbf v)$, shape $3$ — a vector–Jacobian product evaluated right-to-left.` },
    { level: "hard", prompt: "Linear layer. For Y = XW + b with X (N×dᵢₙ), W (dᵢₙ×dₒᵤₜ), b (dₒᵤₜ) and upstream δ = ∂L/∂Y, derive ∂L/∂W, ∂L/∂b, ∂L/∂X and verify each shape.",
      solution: String.raw`$\partial L/\partial W=X^\top\delta$ ($d_\text{in}\times d_\text{out}$ ✓), $\partial L/\partial b=\sum_n\delta_{n,:}$ (length $d_\text{out}$ ✓), $\partial L/\partial X=\delta W^\top$ ($N\times d_\text{in}$ ✓). Each follows from $Y_{nk}=\sum_i X_{ni}W_{ik}+b_k$ and the chain rule; the bias sums over the batch because it's broadcast to every row.` },
    { level: "hard", prompt: "Softmax + cross-entropy. Show that for p = softmax(z) and one-hot y, ∂L/∂z = p − y, and explain the cancellation in one sentence.",
      solution: String.raw`$\frac{\partial L}{\partial z_j}=\sum_i(-y_i/p_i)\,p_i(\delta_{ij}-p_j)=-\sum_i y_i(\delta_{ij}-p_j)=-y_j+p_j\sum_i y_i=p_j-y_j$ (using $\sum_i y_i=1$). The cancellation: the $p_i$ in the softmax Jacobian kills the $1/p_i$ from the cross-entropy derivative.` },
    { level: "med", prompt: "Gradient checking. (a) Why is (f(x+h) − f(x−h))/(2h) preferred over the one-sided difference? (b) How do you numerically check a gradient w.r.t. a matrix parameter?",
      solution: String.raw`(a) The symmetric (central) difference cancels the first-order Taylor error, giving $O(h^2)$ accuracy vs $O(h)$ for one-sided. (b) Perturb each entry $W_{ij}$ by $\pm h$ in turn, recompute the scalar loss, and place $(f_+-f_-)/(2h)$ into position $(i,j)$ — exactly what <code>numeric_grad</code> does by looping over <code>np.ndindex</code>.` },
    { level: "hard", prompt: "Put it together. For H = tanh(XW₁ + b₁), Y = HW₂ + b₂, L = ½‖Y − T‖², derive ∂L/∂W₁.",
      solution: String.raw`Let $\delta_2=Y-T$. Backprop through layer 2's input: $\partial L/\partial H=\delta_2 W_2^\top$. Through tanh (diagonal Jacobian): $\delta_1=(\delta_2 W_2^\top)\odot(1-H^2)$. Through layer 1's weights: $\partial L/\partial W_1=X^\top\delta_1$. This is precisely the backward pass the tiny-autograd capstone runs — now derived by hand.` }
  ],
  rubric: String.raw`<ul>
<li><strong>9–10:</strong> You can derive a layer's gradients from scratch and audit them by shape and by finite differences. You're ready to read/implement backprop in any framework — and to add a custom autograd op.</li>
<li><strong>7–8:</strong> Strong. Revisit whichever slipped: the quadratic-form transpose (14.2), the VJP ordering (14.3), or the softmax cancellation (14.5).</li>
<li><strong>5–6:</strong> The pieces are there but not yet fluent. Redo the linear-layer derivation (14.4) and the chain-rule/VJP argument (14.3) until the shapes are automatic.</li>
<li><strong>Below 5:</strong> Rework the track in order. Matrix calculus is pure pattern practice — do every "Practice" exercise with a finite-difference check and it will click.</li>
</ul>`
};

