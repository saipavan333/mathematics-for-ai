/* ============================================================
   CAPSTONES — five projects that actually run in the browser.
   Each is written to the full lesson standard, but the star of
   every one is the CODE section: press Run and watch it work.
   (Pyodide auto-loads NumPy / scikit-learn from the imports.)
   ============================================================ */
window.LESSON_CONTENT ||= {};

/* ------------------------------------------------------------------ C.1 */
window.LESSON_CONTENT["C.1"] = {
  subtitle: "Compress 64-dimensional handwritten digits down to a handful of numbers — with nothing but NumPy — then rebuild them.",

  aiMoment: String.raw`Every embedding you have ever used — word vectors, the 768-dimensional <em>[CLS]</em> vector from BERT, the penultimate layer of a vision model — lives in a space too big to see or store cheaply. <strong>Principal Component Analysis (PCA)</strong> is the oldest and still one of the most used tools for shrinking that space: it finds the few directions the data actually varies along and throws the rest away. It is how people first visualized word embeddings in 2013, how "eigenfaces" recognized faces in the 1990s, and how practitioners still <em>whiten</em> features before a k-NN or a linear probe. In this capstone you build PCA from the eigenvectors up and run it on real handwritten digits.`,

  plainEnglish: String.raw`Imagine a swarm of points shaped like a squashed, tilted rugby ball. Most of the spread is along the long axis; very little is along the thin ones. PCA rotates your coordinate axes to line up with the ball: the first new axis points along the longest direction of spread, the second along the next longest, and so on. Keep the first few axes and you keep almost all the shape while using far fewer numbers.`,

  intuition: String.raw`A digit image is 64 numbers (an 8×8 grid of pixel brightnesses), so each digit is one point in 64-dimensional space. Those points are not scattered everywhere — pixels near the center are almost always on, corners are almost always off, and neighboring pixels move together. That correlation means the cloud is a thin, tilted ellipsoid, not a round ball. PCA finds the axes of that ellipsoid, longest first. Projecting each digit onto the top two axes already separates the tens of thousands of pixel-patterns into visible clusters by digit — with two numbers instead of sixty-four.`,

  formalism: String.raw`Center the data by subtracting the mean image $\boldsymbol\mu=\frac1n\sum_i \mathbf x_i$, giving rows $\tilde{\mathbf x}_i=\mathbf x_i-\boldsymbol\mu$ stacked into $\tilde X\in\mathbb R^{n\times d}$. The <strong>covariance matrix</strong> is
$$C=\frac{1}{n-1}\tilde X^\top \tilde X\in\mathbb R^{d\times d}.$$
Its eigen-decomposition $C=V\Lambda V^\top$ gives orthonormal eigenvectors (the columns of $V$, the <em>principal components</em>) and eigenvalues $\lambda_1\ge\lambda_2\ge\dots\ge0$ (the variance along each). Equivalently and more stably, the thin SVD $\tilde X=U\Sigma V^\top$ gives the same $V$, with $\lambda_k=\sigma_k^2/(n-1)$. The fraction of variance kept by component $k$ is $\lambda_k/\sum_j\lambda_j=\sigma_k^2/\sum_j\sigma_j^2$.`,

  derivation: String.raw`<strong>Why the top eigenvector is the direction of maximum variance.</strong>
<ol>
<li><strong>Set up the objective.</strong> The variance of the centered data projected onto a unit direction $\mathbf w$ is $\operatorname{Var}(\tilde X\mathbf w)=\mathbf w^\top C\,\mathbf w$. We want the $\mathbf w$ that maximizes it, subject to $\lVert\mathbf w\rVert^2=1$ (otherwise we could cheat by scaling $\mathbf w$ up).</li>
<li><strong>Form the Lagrangian.</strong> $\mathcal L(\mathbf w,\lambda)=\mathbf w^\top C\mathbf w-\lambda(\mathbf w^\top\mathbf w-1).$</li>
<li><strong>Differentiate and set to zero.</strong> Using $\nabla_{\mathbf w}(\mathbf w^\top C\mathbf w)=2C\mathbf w$ and $\nabla_{\mathbf w}(\mathbf w^\top\mathbf w)=2\mathbf w$: $\;2C\mathbf w-2\lambda\mathbf w=\mathbf 0\;\Rightarrow\; C\mathbf w=\lambda\mathbf w.$</li>
<li><strong>Read off the answer.</strong> So the optimal $\mathbf w$ is an <em>eigenvector</em> of $C$. Left-multiplying by $\mathbf w^\top$ gives $\mathbf w^\top C\mathbf w=\lambda$ — the variance captured equals the eigenvalue. To maximize it, pick the eigenvector with the largest eigenvalue. The second-best direction, forced orthogonal to the first, is the next eigenvector, and so on. That is PCA.</li>
</ol>
In plain words: the best axis to keep is the one the data spreads out along the most, and "how much it spreads" is exactly that axis's eigenvalue.`,

  code: [
    { label: "1 · PCA from scratch, checked against scikit-learn",
      src: String.raw`import numpy as np
from sklearn.datasets import load_digits

d = load_digits(); X = d.data.astype(float); labels = d.target   # X: (1797, 64)
mu = X.mean(0); Xc = X - mu                                       # 1) center the data

# 2) PCA from scratch via the SVD:  Xc = U @ diag(S) @ Vt
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
comps  = Vt                     # each ROW of Vt is a principal direction (64-D)
evr    = (S**2) / (S**2).sum()  # fraction of variance explained by each PC
scores = Xc @ comps.T           # 3) project: coordinates of each digit in PC space
cum    = np.cumsum(evr)
k90 = int(np.searchsorted(cum, 0.90) + 1)

print("variance explained by the first 5 PCs:", np.round(evr[:5], 4))
print("components needed for 90% of the variance:", k90, "of 64")

# sanity check: our from-scratch numbers should match the library
from sklearn.decomposition import PCA
sk = PCA().fit(Xc)
print("matches scikit-learn:", np.allclose(evr, sk.explained_variance_ratio_, atol=1e-8))` },
    { label: "2 · Look at the components + how many you need",
      src: String.raw`import matplotlib.pyplot as plt

# The top-16 principal components, each reshaped back to an 8x8 "eigen-digit".
fig, axs = plt.subplots(2, 8, figsize=(11, 3.0))
for i, ax in enumerate(axs.ravel()):
    ax.imshow(comps[i].reshape(8, 8), cmap='RdBu_r')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"PC{i+1}", fontsize=7)
fig.suptitle("Top-16 principal components (each an 8x8 eigen-digit)", fontsize=10)
fig.tight_layout()

# Cumulative variance: the elbow tells you how many components are 'enough'.
plt.figure(figsize=(6, 3.4))
plt.plot(range(1, 65), cum, '-o', ms=3, color='#2a6f97')
plt.axhline(0.90, ls='--', c='gray'); plt.axvline(k90, ls='--', c='#d1495b')
plt.text(k90 + 1, 0.55, f"{k90} PCs -> 90%", color='#d1495b')
plt.xlabel("number of components"); plt.ylabel("cumulative variance explained")
plt.title("How many components do we actually need?"); plt.tight_layout()` },
    { label: "3 · Project to 2-D, and reconstruct from k numbers",
      src: String.raw`import matplotlib.pyplot as plt

# 2-D projection: 64 numbers -> 2 numbers, colored by the true digit.
plt.figure(figsize=(6.2, 5))
sc = plt.scatter(scores[:, 0], scores[:, 1], c=labels, cmap='tab10', s=9, alpha=.85)
plt.colorbar(sc, label='digit', ticks=range(10))
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.title("Every digit projected onto just the first two PCs"); plt.tight_layout()

# Reconstruct each digit from only k components:  x ~= mu + scores[:, :k] @ comps[:k]
def reconstruct(k): return scores[:, :k] @ comps[:k] + mu
ks = [2, 8, 16, 32]
fig, axs = plt.subplots(len(ks) + 1, 8, figsize=(11, 6.2))
for j in range(8):
    axs[0, j].imshow(X[j].reshape(8, 8), cmap='gray'); axs[0, j].axis('off')
axs[0, 0].set_title("original", fontsize=8, loc='left')
for r, k in enumerate(ks):
    Xk = reconstruct(k)
    for j in range(8):
        axs[r+1, j].imshow(Xk[j].reshape(8, 8), cmap='gray'); axs[r+1, j].axis('off')
    axs[r+1, 0].set_title(f"k = {k}", fontsize=8, loc='left')
fig.suptitle("Rebuilding digits from k principal components (top row = original 64-D)", fontsize=10)
fig.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 760 190" width="100%" style="max-width:760px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <defs><marker id="c1arr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <rect x="12" y="30" width="128" height="46" rx="6" fill="#eef2f7" stroke="#33415c"/>
  <text x="76" y="50" text-anchor="middle" fill="#1f2a44">64-D digit</text>
  <text x="76" y="66" text-anchor="middle" fill="#5a6b8c">(8×8 pixels)</text>
  <rect x="168" y="30" width="118" height="46" rx="6" fill="#eef2f7" stroke="#33415c"/>
  <text x="227" y="50" text-anchor="middle" fill="#1f2a44">subtract mean</text>
  <text x="227" y="66" text-anchor="middle" fill="#5a6b8c">(center)</text>
  <rect x="314" y="30" width="132" height="46" rx="6" fill="#eef2f7" stroke="#33415c"/>
  <text x="380" y="50" text-anchor="middle" fill="#1f2a44">SVD → directions V</text>
  <text x="380" y="66" text-anchor="middle" fill="#5a6b8c">+ eigenvalues</text>
  <rect x="474" y="30" width="120" height="46" rx="6" fill="#eef2f7" stroke="#33415c"/>
  <text x="534" y="50" text-anchor="middle" fill="#1f2a44">keep top k</text>
  <text x="534" y="66" text-anchor="middle" fill="#5a6b8c">directions</text>
  <rect x="622" y="30" width="126" height="46" rx="6" fill="#dbe8fb" stroke="#2a6f97"/>
  <text x="685" y="50" text-anchor="middle" fill="#123a5a">k-D code</text>
  <text x="685" y="66" text-anchor="middle" fill="#2a6f97">(scores)</text>
  <line x1="140" y1="53" x2="166" y2="53" stroke="#6b7a99" marker-end="url(#c1arr)"/>
  <line x1="286" y1="53" x2="312" y2="53" stroke="#6b7a99" marker-end="url(#c1arr)"/>
  <line x1="446" y1="53" x2="472" y2="53" stroke="#6b7a99" marker-end="url(#c1arr)"/>
  <line x1="594" y1="53" x2="620" y2="53" stroke="#6b7a99" marker-end="url(#c1arr)"/>
  <rect x="474" y="128" width="274" height="44" rx="6" fill="#e7f0e8" stroke="#3a7d44"/>
  <text x="611" y="148" text-anchor="middle" fill="#204d2a">reconstruct ≈ original digit</text>
  <text x="611" y="164" text-anchor="middle" fill="#3a7d44">mean + scores · directions</text>
  <line x1="685" y1="76" x2="685" y2="126" stroke="#3a7d44" marker-end="url(#c1arr)"/>
  <text x="700" y="104" fill="#3a7d44">rebuild</text>
</svg>`,

  keyPoints: [
    String.raw`PCA = rotate the axes to line up with the data's spread, then keep the few longest axes. The axes are eigenvectors of the covariance; the variance kept along each is its eigenvalue.`,
    String.raw`<strong>Always center first.</strong> PCA is about variance <em>around the mean</em>; skip the centering and the first "component" just points at the mean.`,
    String.raw`Use the <strong>SVD of the centered data</strong>, not an explicit eigen-decomposition of $\tilde X^\top\tilde X$ — same answer, far better numerics, and you never form the $d\times d$ covariance.`,
    String.raw`The cumulative-variance curve turns "how many dimensions?" into a decision you can see: here 21 of 64 components already carry 90% of the signal.`,
    String.raw`Reconstruction $\hat{\mathbf x}=\boldsymbol\mu+\text{scores}_{:k}\,V_{:k}$ is the best possible rank-$k$ approximation (Eckart–Young) — PCA is lossy compression with a provable optimality guarantee.`
  ],

  commonMistakes: [
    { wrong: "Running PCA on raw features of wildly different scales (e.g. pixels 0–16 next to a feature in the thousands).",
      why: String.raw`Variance is scale-dependent, so the biggest-unit feature hijacks PC 1. Standardize (divide by each feature's std) when units differ. Digit pixels share a scale, so we only centered — but for mixed features, standardize.` },
    { wrong: "Treating a principal component's sign or exact orientation as meaningful.",
      why: String.raw`Eigenvectors are defined only up to sign (and up to rotation within an equal-eigenvalue subspace). $\mathbf w$ and $-\mathbf w$ are the same component; don't read meaning into which way it points.` },
    { wrong: "Assuming the directions of largest variance are the directions that matter for your label.",
      why: String.raw`PCA is <em>unsupervised</em> — it never sees the labels. High-variance directions are often useful, but a tiny-variance direction can be the one that separates your classes. When you need discriminative axes, use LDA or just train a classifier.` }
  ],

  quiz: [
    { q: "A dataset has eigenvalues (variances) [6, 3, 1, 0]. What fraction of variance do the top two components capture?",
      options: ["90%", "60%", "100%", "50%"], answer: 0,
      explain: String.raw`$(6+3)/(6+3+1+0)=9/10=90\%$. Eigenvalues are additive variance, so you just sum and divide.` },
    { q: "Why prefer the SVD of Xc over eigendecomposing Xcᵀ Xc?",
      options: ["It avoids squaring the condition number (better numerical stability) and never forms the d×d matrix",
                "It gives a different, better set of components",
                "It is the only way to get eigenvalues",
                "It requires the data to be square"], answer: 0,
      explain: String.raw`Forming $\tilde X^\top\tilde X$ squares the condition number, losing precision; the SVD gets the same $V$ directly from $\tilde X$. For a tall-thin data matrix it is also cheaper.` },
    { q: "You forget to subtract the mean before the SVD. What most likely happens to PC 1?",
      options: ["It points toward the data's mean/offset instead of its direction of spread",
                "Nothing — centering is optional",
                "The eigenvalues become negative",
                "The components stay identical but reversed"], answer: 0,
      explain: String.raw`Uncentered, the largest singular direction is dominated by the offset of the cloud from the origin, not its internal spread. Centering is what makes PCA about <em>variance</em>.` },
    { q: "In an 8×8 digit (64-D), you keep k=16 components. How many numbers now represent each digit, and what are the eigen-digits?",
      options: ["16 numbers per digit; the eigen-digits are the 16 principal directions, each itself an 8×8 image",
                "64 numbers; the eigen-digits are the originals",
                "16 numbers; the eigen-digits are 16 of the original training images",
                "8 numbers; one per row of the image"], answer: 0,
      explain: String.raw`Each digit becomes its 16 scores. Every principal direction lives in the same 64-D pixel space, so it reshapes to an 8×8 image — the 'eigen-digit' you plotted.` },
    { q: "Reconstruction error when keeping k components equals…",
      options: ["The sum of the discarded eigenvalues, $\\sum_{j>k}\\lambda_j$",
                "Zero, always",
                "The largest eigenvalue $\\lambda_1$",
                "The number of discarded components"], answer: 0,
      explain: String.raw`Mean squared reconstruction error is exactly the leftover variance you threw away, $\sum_{j>k}\lambda_j$. That is why the variance curve <em>is</em> the error curve, flipped.` }
  ],

  practice: [
    { level: "easy", prompt: "Change k in the reconstruction cell to 1, 4, and 64. At k=64, what should the reconstruction look like, and why?",
      solution: String.raw`At $k=64$ you keep every component, so $\hat{\mathbf x}=\mathbf x$ exactly (up to floating error) — the reconstruction is pixel-perfect because you've thrown nothing away. Small $k$ shows progressively blurrier digits.` },
    { level: "easy", prompt: "Print the smallest k whose cumulative variance reaches 95%. (Hint: np.searchsorted(cum, 0.95)+1.)",
      solution: String.raw`It is 29 of 64 for this dataset. Code: <code>int(np.searchsorted(cum, 0.95) + 1)</code>. So 95% of the signal survives at less than half the dimensions.` },
    { level: "med", prompt: "Add whitening: divide each score column by its standard deviation so every kept direction has unit variance. Re-plot the 2-D scatter.",
      solution: String.raw`Whitened scores: <code>Z = scores[:, :k] / scores[:, :k].std(0)</code>. The eigenvalue spectrum is flattened to all-ones, so the cloud becomes spherical; distances no longer over-weight PC 1. This is the standard preprocessing before a k-NN or a linear probe.` },
    { level: "med", prompt: "Verify Eckart–Young numerically: compute the mean squared reconstruction error for k = 1…64 and overlay it on the discarded-variance curve.",
      solution: String.raw`Error at $k$: <code>((X - reconstruct(k))**2).mean()</code>. Plotted against $\sum_{j>k}\lambda_j\,(d)$ they coincide — the least-squares rank-$k$ approximation error equals the tail of the spectrum.` },
    { level: "hard", prompt: "Replace the SVD with an explicit eigendecomposition of the covariance (np.linalg.eigh) and confirm you get the same components (up to sign).",
      solution: String.raw`<code>vals, vecs = np.linalg.eigh(np.cov(Xc, rowvar=False))</code>; sort descending. Compare <code>abs(vecs[:, ::-1][:, :k])</code> to <code>abs(comps[:k].T)</code> — equal up to column sign. <code>eigh</code> is for symmetric matrices and returns ascending eigenvalues, hence the reversal.` },
    { level: "hard", prompt: "PCA assumes variance = information. Construct a 2-class toy set where the class-separating direction has the SMALLEST variance, so PC 1 is useless for classification.",
      solution: String.raw`Two thin, long clusters offset by a small gap along the short axis: e.g. class A ~ N([0,0], diag(9, 0.01)), class B ~ N([0,0.4], diag(9, 0.01)). PC 1 runs along the long (x) axis — shared by both classes — while the tiny-variance y-axis is the one that separates them. It's the canonical case where you want LDA, not PCA.` }
  ],

  deepDive: String.raw`<p><strong>PCA is a linear autoencoder.</strong> A single-hidden-layer linear autoencoder with a $k$-unit bottleneck, trained with squared error, learns exactly the PCA subspace (Baldi & Hornik, 1989). Encoder = project onto $V_{:k}$, decoder = multiply back by $V_{:k}^\top$. Nonlinear autoencoders generalize this by bending the subspace into a curved manifold — which is why "autoencoder" and "PCA" belong in the same sentence.</p>
<p><strong>Where it breaks.</strong> PCA's three assumptions are: (1) directions that matter are high-variance, (2) the interesting structure is linear, and (3) the mean and covariance summarize the data. Each fails somewhere — swiss-roll manifolds (use t-SNE/UMAP/kernel-PCA), heavy-tailed data where covariance is unstable (use robust PCA), or when the label lives in a low-variance direction (use LDA). Knowing PCA's failure modes is what tells you which fancier method to reach for.</p>
<p><strong>Cost.</strong> The thin SVD of an $n\times d$ matrix is $O(nd^2)$. For huge $d$ you never want the full SVD — randomized SVD or an iterative solver (power iteration on the covariance) gets the top $k$ components in $O(ndk)$, which is what scikit-learn's <code>svd_solver='randomized'</code> does under the hood.</p>`
};

/* ------------------------------------------------------------------ C.2 */
window.LESSON_CONTENT["C.2"] = {
  subtitle: "Implement gradient descent, momentum, and Adam from scratch, then watch all three race across a non-convex valley.",

  aiMoment: String.raw`When you type <code>optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)</code>, you are choosing <em>how</em> every weight in your network moves in response to its gradient. That one line decides whether training converges in an hour or diverges in ten steps. Yet most people set the learning rate and the $\beta$s by superstition. In this capstone you implement plain gradient descent, momentum, and Adam in a dozen lines each and run them on the same loss surface, so you can <em>see</em> what each knob does.`,

  plainEnglish: String.raw`Think of the loss as a hilly landscape and your parameters as a ball you want to roll to the lowest point. <strong>Gradient descent</strong> takes a small step straight downhill every time. <strong>Momentum</strong> gives the ball weight, so it builds up speed down long slopes and coasts through small bumps. <strong>Adam</strong> gives each direction its own step size, taking big steps where the ground is gently sloped and small careful ones where it is steep.`,

  intuition: String.raw`Our test landscape is the Rosenbrock "banana" — a long, curved, narrow valley with the minimum at $(1,1)$. It is deliberately nasty: the floor of the valley is nearly flat while the walls are steep, so the gradient points mostly <em>across</em> the valley (toward the walls) and only slightly <em>along</em> it (toward the goal). Plain gradient descent therefore zig-zags between the walls and crawls forward. Momentum accumulates the small along-valley pushes into real speed. Adam divides out the huge across-valley gradient, so its steps point more usefully forward. You'll watch exactly this play out.`,

  formalism: String.raw`With gradient $g_t=\nabla f(w_t)$ and learning rate $\eta$:
$$\textbf{GD:}\quad w_{t+1}=w_t-\eta\,g_t.$$
$$\textbf{Momentum:}\quad v_{t}=\beta v_{t-1}+g_t,\qquad w_{t+1}=w_t-\eta\,v_t.$$
$$\textbf{Adam:}\quad m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad s_t=\beta_2 s_{t-1}+(1-\beta_2)g_t^2,$$
$$\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\quad \hat s_t=\frac{s_t}{1-\beta_2^{\,t}},\qquad w_{t+1}=w_t-\eta\,\frac{\hat m_t}{\sqrt{\hat s_t}+\varepsilon}.$$
All operations on $m_t,s_t$ are elementwise, so Adam maintains a separate running mean and running "size" of the gradient for <em>every</em> coordinate.`,

  derivation: String.raw`<strong>Why momentum accelerates, and why Adam's bias correction exists.</strong>
<ol>
<li><strong>Momentum is a running sum of gradients.</strong> Unrolling $v_t=\beta v_{t-1}+g_t$ gives $v_t=\sum_{k=0}^{t}\beta^{\,k}g_{t-k}$ — an exponentially weighted sum of past gradients with horizon $\approx 1/(1-\beta)$. At $\beta=0.9$ that averages the last ~10 gradients.</li>
<li><strong>In a valley, consistent components survive; oscillating ones cancel.</strong> The along-valley gradient always points the same way, so its contributions <em>add</em> across steps ($\sum\beta^k\approx\tfrac1{1-\beta}=10\times$ boost). The across-valley gradient flips sign every step, so successive terms <em>cancel</em>. Momentum amplifies progress and damps zig-zag — exactly what the flat-floor/steep-wall geometry needs.</li>
<li><strong>Adam normalizes per coordinate.</strong> Dividing by $\sqrt{\hat s_t}$ (the running root-mean-square of that coordinate's gradient) makes every coordinate's effective step size about the same, regardless of how steep it is. That is why Adam barely cares about the valley's ill-conditioning.</li>
<li><strong>Bias correction fixes the cold start.</strong> $m_0=s_0=0$, so early on $m_t,s_t$ are biased toward zero: $\mathbb E[s_t]=(1-\beta_2^{\,t})\,\mathbb E[g^2]$. Dividing by $1-\beta_2^{\,t}$ removes that factor, so the very first steps aren't artificially tiny. Without it, Adam stalls for dozens of steps at the start.</li>
</ol>`,

  code: [
    { label: "1 · The surface and the three optimizers",
      src: String.raw`import numpy as np

# Rosenbrock: a non-convex 'banana' valley. Global minimum at (1, 1), where f = 0.
def f(p):
    x, y = p
    return (1 - x)**2 + 100*(y - x*x)**2
def grad(p):
    x, y = p
    return np.array([-2*(1 - x) - 400*x*(y - x*x), 200*(y - x*x)])

def optimize(kind, lr, steps=3000, b1=0.9, b2=0.999, eps=1e-8):
    w = np.array([-1.5, 1.6]); path = [w.copy()]
    v = np.zeros(2); m = np.zeros(2); s = np.zeros(2)          # momentum / Adam state
    for t in range(1, steps + 1):
        g = grad(w)
        if   kind == 'gd':   w = w - lr*g
        elif kind == 'mom':  v = b1*v + g;  w = w - lr*v
        elif kind == 'adam':
            m = b1*m + (1 - b1)*g
            s = b2*s + (1 - b2)*g*g
            mhat = m/(1 - b1**t); shat = s/(1 - b2**t)         # bias correction
            w = w - lr*mhat/(np.sqrt(shat) + eps)
        path.append(w.copy())
    return np.array(path)

print("gradient at the start point (-1.5, 1.6):", grad(np.array([-1.5, 1.6])))
print("notice it is huge in y and small in x -- that is the ill-conditioning.")` },
    { label: "2 · Race them: trajectories + loss curves",
      src: String.raw`import matplotlib.pyplot as plt

runs = {'GD  (lr=7e-4)':             (optimize('gd',   7e-4), '#d1495b'),
        'Momentum  (lr=7e-4, β=0.9)': (optimize('mom',  7e-4), '#2a6f97'),
        'Adam  (lr=0.02)':            (optimize('adam', 0.02), '#3a7d44')}
for name, (path, _) in runs.items():
    print(f"{name:28s} final = ({path[-1,0]:.3f}, {path[-1,1]:.3f})   loss = {f(path[-1]):.2e}")

xs = np.linspace(-2, 1.6, 300); ys = np.linspace(-0.4, 2.2, 300)
Xg, Yg = np.meshgrid(xs, ys); Zg = (1 - Xg)**2 + 100*(Yg - Xg*Xg)**2
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].contour(Xg, Yg, Zg, levels=np.logspace(-1, 3.3, 18), cmap='Greys', linewidths=.6)
for name, (path, c) in runs.items():
    ax[0].plot(path[:, 0], path[:, 1], color=c, lw=1.5, label=name)
    ax[0].plot(path[-1, 0], path[-1, 1], 'o', color=c, ms=6)
ax[0].plot(1, 1, 'k*', ms=14)
ax[0].set_title("Trajectories on Rosenbrock (minimum at ★)"); ax[0].legend(fontsize=7, loc='lower right')
for name, (path, c) in runs.items():
    ax[1].semilogy(np.maximum([f(w) for w in path], 1e-16), color=c, lw=1.5, label=name)
ax[1].set_title("Loss vs iteration (log scale)"); ax[1].set_xlabel("step"); ax[1].legend(fontsize=7)
fig.tight_layout()` },
    { label: "3 · The learning rate is the whole ball game",
      src: String.raw`import matplotlib.pyplot as plt

def gd_losses(lr, steps=1500):
    w = np.array([-1.2, 1.0]); L = [f(w)]
    for _ in range(steps):
        w = w - lr*grad(w); val = f(w); L.append(val)
        if not np.isfinite(val) or val > 1e6: break           # bail out if it explodes
    return L

plt.figure(figsize=(7.2, 4))
for lr, tag in [(1e-4, 'too small'), (1e-3, 'just right'), (7e-3, 'too big')]:
    L = gd_losses(lr)
    end = 'DIVERGED' if (not np.isfinite(L[-1]) or L[-1] > 1e6) else f'loss={L[-1]:.2e}'
    plt.semilogy(np.maximum(L, 1e-16), lw=1.6, label=f"lr={lr}  ({tag}) → {end}")
plt.xlabel("step"); plt.ylabel("loss (log scale)")
plt.title("Same optimizer (GD), three learning rates"); plt.legend(); plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 760 196" width="100%" style="max-width:760px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="12">
  <rect x="10" y="20" width="230" height="160" rx="8" fill="#fbeaed" stroke="#d1495b"/>
  <text x="125" y="42" text-anchor="middle" font-weight="700" fill="#8f2233">Gradient Descent</text>
  <text x="125" y="82" text-anchor="middle" fill="#3a2327">g = ∇f(w)</text>
  <text x="125" y="108" text-anchor="middle" fill="#3a2327">w ← w − η·g</text>
  <text x="125" y="150" text-anchor="middle" fill="#a05563" font-size="11">one step straight downhill</text>
  <rect x="264" y="20" width="230" height="160" rx="8" fill="#e7eff6" stroke="#2a6f97"/>
  <text x="379" y="42" text-anchor="middle" font-weight="700" fill="#1c4e70">Momentum</text>
  <text x="379" y="82" text-anchor="middle" fill="#22303a">v ← β·v + g</text>
  <text x="379" y="108" text-anchor="middle" fill="#22303a">w ← w − η·v</text>
  <text x="379" y="150" text-anchor="middle" fill="#3d6a89" font-size="11">a heavy ball builds speed (β≈0.9)</text>
  <rect x="518" y="20" width="232" height="160" rx="8" fill="#e7f0e8" stroke="#3a7d44"/>
  <text x="634" y="42" text-anchor="middle" font-weight="700" fill="#245030">Adam</text>
  <text x="634" y="76" text-anchor="middle" fill="#23331f" font-size="11">m ← β₁m + (1−β₁)g</text>
  <text x="634" y="98" text-anchor="middle" fill="#23331f" font-size="11">s ← β₂s + (1−β₂)g²</text>
  <text x="634" y="120" text-anchor="middle" fill="#23331f" font-size="11">w ← w − η·m / (√s + ε)</text>
  <text x="634" y="152" text-anchor="middle" fill="#4a7d55" font-size="11">a private step size per coordinate</text>
</svg>`,

  keyPoints: [
    String.raw`All three optimizers are tiny: GD is one line, momentum adds a velocity vector, Adam adds a per-coordinate second moment plus bias correction. Nothing is magic.`,
    String.raw`Momentum is an exponentially weighted average of past gradients ($v_t=\sum_k\beta^k g_{t-k}$). It amplifies consistent directions and cancels oscillating ones — perfect for narrow valleys.`,
    String.raw`Adam rescales each coordinate by its gradient's running RMS, so it is nearly immune to ill-conditioning. That robustness is why it's the default for transformers and most large models.`,
    String.raw`Bias correction ($1-\beta^t$ denominators) matters only for the first dozens of steps, but without it Adam starts far too slowly.`,
    String.raw`The learning rate dominates everything: too small crawls, too big diverges in a handful of steps. There is a usable band, and finding it is most of the tuning battle.`
  ],

  commonMistakes: [
    { wrong: "Reusing Adam's lr=3e-4 for plain SGD, or SGD's lr=0.1 for Adam.",
      why: String.raw`Adam's update is normalized to roughly unit scale per coordinate, so its good learning rates are much smaller than SGD's. The optimizers live on different lr scales — always retune when you switch.` },
    { wrong: "Blaming the model when loss goes to NaN after a few steps.",
      why: String.raw`That is the classic 'lr too big' signature: the step overshoots, the gradient grows, the next step overshoots more. Before touching the architecture, divide the learning rate by 10.` },
    { wrong: "Assuming Adam always beats momentum/SGD.",
      why: String.raw`Adam converges fast and robustly, but well-tuned SGD+momentum often <em>generalizes</em> better on vision tasks (a documented gap). 'Best optimizer' depends on the problem; here momentum actually reaches the minimum first.` }
  ],

  quiz: [
    { q: "With β = 0.9, roughly how many past gradients does momentum effectively average over?",
      options: ["About 10", "About 90", "About 2", "All of them equally"], answer: 0,
      explain: String.raw`The effective horizon of an exponential average is $1/(1-\beta)=1/0.1=10$. Older gradients are down-weighted by $\beta^k$.` },
    { q: "Adam divides the update by √ŝ, the running RMS of each coordinate's gradient. What is the main effect?",
      options: ["Every coordinate gets a similar-sized step regardless of its gradient magnitude",
                "It guarantees convergence to the global minimum",
                "It removes the need for a learning rate",
                "It makes the optimizer identical to GD"], answer: 0,
      explain: String.raw`Normalizing by the gradient's own scale equalizes step sizes across coordinates — that is the cure for ill-conditioning. You still need a learning rate, and convergence is not guaranteed.` },
    { q: "On Rosenbrock, the gradient at the start is huge in y and small in x. Why does that hurt plain GD?",
      options: ["A single lr can't be both small enough to stay stable in y and large enough to progress in x",
                "The gradient is wrong",
                "GD ignores the y-coordinate",
                "The minimum is unreachable by any method"], answer: 0,
      explain: String.raw`One scalar $\eta$ must serve both directions. Small enough for the steep $y$ wall means far too small for the flat $x$ floor — so GD zig-zags and crawls. Adam's per-coordinate scaling is exactly the fix.` },
    { q: "You run GD with lr = 7e-3 and the loss becomes inf within a few steps. The fix is most likely to…",
      options: ["Lower the learning rate", "Raise the learning rate", "Add more parameters", "Train for more steps"], answer: 0,
      explain: String.raw`Divergence to inf is the 'lr too big' signature — the standard first move is to cut the learning rate (often by 10×). More steps or parameters won't help an unstable step size.` },
    { q: "Why does Adam need the bias-correction terms 1 − βᵗ?",
      options: ["Because m and s start at 0, biasing early estimates toward 0; dividing by 1−βᵗ removes that",
                "To make the learning rate adaptive",
                "To guarantee the gradient is unbiased",
                "They are optional and never matter"], answer: 0,
      explain: String.raw`With $m_0=s_0=0$, early averages are shrunk toward zero; $\mathbb E[s_t]=(1-\beta_2^t)\mathbb E[g^2]$. Dividing by $1-\beta_2^t$ undoes the shrinkage so the first steps aren't tiny.` }
  ],

  practice: [
    { level: "easy", prompt: "In cell 2, set Adam's lr to 0.001 and rerun. What happens to its trajectory and final loss, and why?",
      solution: String.raw`Adam becomes much slower and won't reach $(1,1)$ in 3000 steps — its steps are now ~20× smaller. It shows that even the 'robust' optimizer still needs a sensible learning rate; robustness widens the good band, it doesn't remove it.` },
    { level: "easy", prompt: "Turn momentum off by setting β = 0 in the 'mom' branch. What optimizer do you get back?",
      solution: String.raw`With $\beta=0$, $v_t=g_t$ and the update is $w-\eta g_t$ — plain gradient descent. Momentum contains GD as its zero-memory special case.` },
    { level: "med", prompt: "Add a fourth optimizer, 'Nesterov' momentum: evaluate the gradient at the look-ahead point w − η·β·v instead of at w. Compare its path to plain momentum.",
      solution: String.raw`Compute <code>g = grad(w - lr*b1*v)</code> before the velocity update. Nesterov 'peeks ahead', so it corrects sooner as it approaches curves — usually slightly less overshoot than classical momentum on the valley bends.` },
    { level: "med", prompt: "Plot the distance to the minimum ‖wₜ − (1,1)‖ versus step for all three, on a log scale. Which decreases most steadily?",
      solution: String.raw`<code>np.linalg.norm(path - np.array([1,1]), axis=1)</code> per optimizer, then <code>semilogy</code>. GD's curve is the shallowest and bumpiest; momentum and Adam fall faster, momentum often the steepest here.` },
    { level: "hard", prompt: "Switch the surface to an ill-conditioned quadratic f(w) = ½(w₁² + κ·w₂²) with κ = 100. Derive GD's exact convergence factor and confirm it empirically.",
      solution: String.raw`For a quadratic with condition number $\kappa$, optimal-lr GD contracts the error by $\frac{\kappa-1}{\kappa+1}$ per step $=99/101\approx0.980$. Measure the ratio of successive $\lVert w_t\rVert$ — it should sit near 0.98. Momentum improves this to $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\approx0.818$.` },
    { level: "hard", prompt: "Add isotropic noise to the gradient (g += σ·randn) to simulate stochastic minibatches. How large can σ get before each optimizer stops converging, and which tolerates the most?",
      solution: String.raw`Add <code>np.random.randn(2)*sigma</code> to <code>g</code>. Small $\sigma$ just jitters the path; past a threshold the optimizers hover in a noise ball of radius $\propto\eta\sigma$ around the minimum instead of settling. Momentum averages noise away (more tolerant); large fixed-lr Adam tends to rattle the most. The practical fix is a decaying learning rate.` }
  ],

  deepDive: String.raw`<p><strong>Convergence rates, concretely.</strong> On a convex quadratic with condition number $\kappa=\lambda_{\max}/\lambda_{\min}$, gradient descent contracts error by $\frac{\kappa-1}{\kappa+1}$ per step; heavy-ball momentum improves that to $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ — a square-root speedup that is enormous when $\kappa$ is large. This is <em>the</em> reason momentum exists: it turns $\kappa$ into $\sqrt\kappa$.</p>
<p><strong>Why Adam won the deep-learning era.</strong> Deep nets have wildly different gradient scales across layers and parameters (embedding rows seen once vs. a bias seen every batch). Per-coordinate normalization means you can pick one learning rate and have it be roughly right everywhere — a huge practical convenience at billion-parameter scale. The cost is a documented generalization gap versus tuned SGD+momentum on some vision benchmarks, which is why both are still in use.</p>
<p><strong>Stochastic reality.</strong> Real training uses <em>minibatch</em> gradients — noisy estimates of the true gradient. That noise is not purely a nuisance: it helps escape sharp minima and saddle points. But it also means a fixed learning rate leaves you rattling in a noise ball around the optimum, which is why every serious run uses a schedule (warmup then cosine/step decay) to shrink the step as you close in.</p>`
};

/* ------------------------------------------------------------------ C.3 */
window.LESSON_CONTENT["C.3"] = {
  subtitle: "Learn what each class 'looks like' as a Gaussian, classify new points with Bayes' rule, and watch a prior rescue you when data is scarce.",

  aiMoment: String.raw`Before deep learning ate everything, the spam filter in your inbox was almost certainly a <strong>Naive Bayes</strong> classifier, and it worked shockingly well. The idea — model what each class generates, then invert with Bayes' rule — is <em>generative</em> classification, and it still shows up wherever data is scarce or you need calibrated probabilities and a prior you can defend. This capstone builds a Gaussian Bayes classifier from scratch, derives why its decision boundary curves, and shows the single most important reason to be Bayesian: a sensible <strong>prior</strong> keeps you accurate when you've barely seen any data.`,

  plainEnglish: String.raw`Instead of drawing a line between the classes, you learn a little "blob" for each one — where its points sit and how spread out they are. To label a new point, you ask each blob "how likely were <em>you</em> to have produced this point?", multiply by how common that class is overall (its prior), and pick the winner. That multiply-and-pick is Bayes' rule.`,

  intuition: String.raw`Each class becomes a Gaussian bump over feature space. A new point is scored by every bump; the bump that assigns it the highest value — after weighting by the class's prior — wins. Where two bumps are equally high, you get the decision boundary. Because the bumps can have different widths, that boundary is generally a <em>curve</em> (a conic), not a straight line — which is exactly what separates this generative model from a linear classifier like logistic regression. The diagram shows the one-dimensional version: two weighted bells, and the boundary sitting precisely where they cross.`,

  formalism: String.raw`Bayes' rule turns a generative story into a classifier:
$$P(y=c\mid \mathbf x)=\frac{P(y=c)\,p(\mathbf x\mid y=c)}{\sum_{c'}P(y=c')\,p(\mathbf x\mid y=c')}.$$
The denominator is the same for every class, so we classify by the numerator alone. <strong>Gaussian Naive Bayes</strong> models each class-conditional as a Gaussian with a <em>diagonal</em> covariance (features independent given the class):
$$p(\mathbf x\mid y=c)=\prod_{j=1}^d \frac{1}{\sqrt{2\pi\sigma_{cj}^2}}\exp\!\Big(-\frac{(x_j-\mu_{cj})^2}{2\sigma_{cj}^2}\Big).$$
Taking logs (to avoid underflow and turn products into sums), the score for class $c$ is
$$\log P(y=c)+\sum_{j}\Big[-\tfrac12\log(2\pi\sigma_{cj}^2)-\tfrac{(x_j-\mu_{cj})^2}{2\sigma_{cj}^2}\Big].$$`,

  derivation: String.raw`<strong>Where the parameters come from, and why the boundary is quadratic.</strong>
<ol>
<li><strong>MLE for the Gaussian.</strong> Maximizing the log-likelihood of class $c$'s data gives exactly the sample statistics: $\hat\mu_{cj}=\frac1{n_c}\sum_{i\in c}x_{ij}$ and $\hat\sigma_{cj}^2=\frac1{n_c}\sum_{i\in c}(x_{ij}-\hat\mu_{cj})^2$. The prior is the class frequency $\hat P(y=c)=n_c/n$.</li>
<li><strong>MAP shrinks the variance.</strong> With a prior that variances shouldn't be tiny, the estimate becomes $\hat\sigma^2_{cj}=\frac{\sum_{i\in c}(x_{ij}-\hat\mu_{cj})^2+\nu\,\sigma_0^2}{n_c+\nu}$. With $\nu=0$ this is the MLE; with $\nu>0$ it is pulled toward $\sigma_0^2$. When $n_c$ is tiny this regularization is the difference between a usable model and a divide-by-almost-zero disaster.</li>
<li><strong>The boundary between two classes.</strong> Set their log-scores equal. Each score is quadratic in $\mathbf x$ (from the $(x_j-\mu_{cj})^2/\sigma_{cj}^2$ terms), so their difference is quadratic too: the boundary $\{\mathbf x: \text{score}_a(\mathbf x)=\text{score}_b(\mathbf x)\}$ is a conic — ellipse, parabola, or hyperbola.</li>
<li><strong>The special case that goes linear.</strong> If both classes share the <em>same</em> covariance, the quadratic terms cancel and only a linear term survives — the boundary becomes a straight line. That special case is Linear Discriminant Analysis, and it's why LDA is linear while general Gaussian Bayes is curved.</li>
</ol>`,

  code: [
    { label: "1 · Gaussian Bayes from scratch, checked against scikit-learn",
      src: String.raw`import numpy as np
rng = np.random.default_rng(1)

# Three 2-D Gaussian classes with DIFFERENT shapes (covariances).
means = np.array([[-2.5, -1.0], [2.5, -1.0], [0.0, 2.8]])
covs  = [np.array([[1.3,  .5], [ .5,  .7]]),
         np.array([[ .8, -.35], [-.35, 1.4]]),
         np.array([[1.1,  0.], [ 0.,  .7]])]
def sample(n):
    X, y = [], []
    for k in range(3):
        X.append(rng.multivariate_normal(means[k], covs[k], n)); y += [k]*n
    return np.vstack(X), np.array(y)
Xtr, ytr = sample(120); Xte, yte = sample(400)

class GaussianBayes:
    def fit(self, X, y, nu=0.0, var_prior=1.0):
        self.classes = np.unique(y); self.prior = {}; self.mean = {}; self.var = {}
        for c in self.classes:
            Xc = X[y == c]
            self.prior[c] = len(Xc) / len(X)          # class frequency
            self.mean[c]  = Xc.mean(0)                 # MLE mean
            # MAP variance: pull toward var_prior with strength nu (nu=0 -> plain MLE)
            self.var[c] = (((Xc - self.mean[c])**2).sum(0) + nu*var_prior) / (len(Xc) + nu) + 1e-9
        return self
    def log_score(self, X):                            # log prior + log Gaussian likelihood
        return np.array([
            (-0.5*np.log(2*np.pi*self.var[c]) - 0.5*(X - self.mean[c])**2/self.var[c]).sum(1)
            + np.log(self.prior[c]) for c in self.classes]).T
    def predict(self, X): return self.classes[self.log_score(X).argmax(1)]
    def proba(self, X):                                # normalize scores into probabilities
        L = self.log_score(X); L = L - L.max(1, keepdims=True)
        P = np.exp(L); return P / P.sum(1, keepdims=True)

gb = GaussianBayes().fit(Xtr, ytr)
print(f"train acc = {(gb.predict(Xtr)==ytr).mean():.3f}    test acc = {(gb.predict(Xte)==yte).mean():.3f}")

from sklearn.naive_bayes import GaussianNB
sk = GaussianNB().fit(Xtr, ytr)
print("matches scikit-learn:", (sk.predict(Xte) == gb.predict(Xte)).mean() > 0.99)` },
    { label: "2 · Decision regions + posterior probability surfaces",
      src: String.raw`import matplotlib.pyplot as plt
cols = np.array(['#1f77b4', '#ff7f0e', '#2ca02c'])
xx, yy = np.meshgrid(np.linspace(-6.5, 6.5, 300), np.linspace(-5, 7, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# where each class wins -> curved boundaries
plt.figure(figsize=(6.2, 5))
plt.contourf(xx, yy, gb.predict(grid).reshape(xx.shape), levels=[-.5, .5, 1.5, 2.5], colors=cols, alpha=.25)
for k in range(3):
    plt.scatter(Xtr[ytr==k, 0], Xtr[ytr==k, 1], s=12, color=cols[k], edgecolors='k', lw=.2)
plt.title("Gaussian Bayes decision regions (boundaries curve)"); plt.tight_layout()

# how CONFIDENT the model is, everywhere: P(class k | x)
P = gb.proba(grid)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for k in range(3):
    im = ax[k].contourf(xx, yy, P[:, k].reshape(xx.shape), levels=np.linspace(0, 1, 11), cmap='viridis')
    ax[k].scatter(Xtr[ytr==k, 0], Xtr[ytr==k, 1], s=6, color='w', edgecolors='k', lw=.2)
    ax[k].set_title(f"P(class {k} | x)", fontsize=9)
fig.colorbar(im, ax=ax, fraction=.02); fig.suptitle("Posterior probability surfaces", fontsize=10)` },
    { label: "3 · The payoff: a prior beats MLE when data is scarce; and vs logistic regression",
      src: String.raw`import matplotlib.pyplot as plt

# ---- MLE vs MAP as the training set grows (average over many random draws) ----
Ns = [2, 3, 5, 8, 12, 20, 40, 80]; trials = 60
curve = {'MLE': [], 'MAP': []}
for N in Ns:
    for tag, nu in [('MLE', 0.0), ('MAP', 6.0)]:
        accs = []
        for _ in range(trials):
            Xs, ys = [], []
            for k in range(3):
                Xs.append(rng.multivariate_normal(means[k], covs[k], N)); ys += [k]*N
            m = GaussianBayes().fit(np.vstack(Xs), np.array(ys), nu=nu, var_prior=1.5)
            accs.append((m.predict(Xte) == yte).mean())
        curve[tag].append(np.mean(accs))
print(f"N=2  per class:  MLE={curve['MLE'][0]:.3f}   MAP={curve['MAP'][0]:.3f}")
print(f"N=80 per class:  MLE={curve['MLE'][-1]:.3f}   MAP={curve['MAP'][-1]:.3f}")

# ---- logistic regression from scratch (softmax), for a discriminative contrast ----
def softmax_lr(X, y, K=3, epochs=400, lr=0.2):
    Xb = np.c_[X, np.ones(len(X))]; W = np.zeros((Xb.shape[1], K)); Y = np.eye(K)[y]
    for _ in range(epochs):
        Z = Xb @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (Xb.T @ (P - Y) / len(Xb))          # gradient of cross-entropy
    return W
W = softmax_lr(Xtr, ytr)
lr_predict = lambda X: (np.c_[X, np.ones(len(X))] @ W).argmax(1)
print(f"logistic regression test acc = {(lr_predict(Xte) == yte).mean():.3f}")

plt.figure(figsize=(6, 3.8))
plt.plot(Ns, curve['MLE'], '-o', color='#d1495b', label='MLE (no prior)')
plt.plot(Ns, curve['MAP'], '-o', color='#2a6f97', label='MAP (variance prior)')
plt.xlabel("training points per class"); plt.ylabel("test accuracy")
plt.title("MLE vs MAP as data grows"); plt.legend(); plt.tight_layout()

cols = np.array(['#1f77b4', '#ff7f0e', '#2ca02c'])
xx, yy = np.meshgrid(np.linspace(-6.5, 6.5, 300), np.linspace(-5, 7, 300)); grid = np.c_[xx.ravel(), yy.ravel()]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, pred, title in [(ax[0], gb.predict, "Gaussian Bayes (curved)"),
                       (ax[1], lr_predict, "Logistic Regression (straight)")]:
    a.contourf(xx, yy, pred(grid).reshape(xx.shape), levels=[-.5, .5, 1.5, 2.5], colors=cols, alpha=.25)
    for k in range(3):
        a.scatter(Xtr[ytr==k, 0], Xtr[ytr==k, 1], s=10, color=cols[k], edgecolors='k', lw=.2)
    a.set_title(title)
fig.suptitle("Generative vs discriminative decision boundaries", fontsize=10); fig.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 420 198" width="100%" style="max-width:460px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <line x1="30" y1="160" x2="392" y2="160" stroke="#33415c" stroke-width="1"/>
  <polyline fill="none" stroke="#c1121f" stroke-width="2" points="30,157 42,154 54,149 66,142 78,132 90,119 102,105 114,89 126,76 138,66 150,61 162,63 174,71 186,84 198,99 210,114 222,127 234,138 246,147 258,152 270,156 282,158 294,159 306,160 318,160 330,160 342,160 354,160 366,160 378,160 390,160"/>
  <polyline fill="none" stroke="#2a6f97" stroke-width="2" points="30,160 42,160 54,160 66,160 78,160 90,160 102,160 114,160 126,160 138,160 150,159 162,159 174,156 186,152 198,144 210,131 222,113 234,92 246,72 258,59 270,55 282,63 294,80 306,101 318,121 330,137 342,148 354,154 366,157 378,159 390,160"/>
  <line x1="217" y1="48" x2="217" y2="160" stroke="#6b7a99" stroke-width="1.3" stroke-dasharray="4 3"/>
  <text x="217" y="42" text-anchor="middle" fill="#4a5878">decision boundary (curves cross)</text>
  <text x="112" y="60" text-anchor="middle" fill="#c1121f" font-weight="700">class A</text>
  <text x="300" y="52" text-anchor="middle" fill="#2a6f97" font-weight="700">class B</text>
  <text x="120" y="178" text-anchor="middle" fill="#8a3b46">← predict A</text>
  <text x="315" y="178" text-anchor="middle" fill="#1c4e70">predict B →</text>
  <text x="210" y="193" text-anchor="middle" fill="#5a6b8c">each curve = prior × class-likelihood — classify by whichever is taller</text>
</svg>`,

  keyPoints: [
    String.raw`Generative classification models $p(\mathbf x\mid y)$ per class and inverts with Bayes' rule, rather than modeling the boundary directly. The class you predict is the numerator's argmax: prior × likelihood.`,
    String.raw`Fit is just counting and averaging: the MLE mean/variance are the sample mean/variance, and the prior is the class frequency. No iterative training needed.`,
    String.raw`Different per-class covariances ⇒ a <strong>quadratic</strong> (curved) boundary. Shared covariance collapses it to a straight line — that's LDA hiding inside the same math.`,
    String.raw`A <strong>MAP prior on the variance</strong> is the headline result: at 2 points per class it lifted accuracy from ~75% (MLE) to ~96%. Priors <em>are</em> regularization, and they matter most exactly when data is scarce.`,
    String.raw`Always work in log-space: products of many small densities underflow to zero, but sums of log-densities are stable.`
  ],

  commonMistakes: [
    { wrong: "Multiplying raw probabilities/densities instead of adding their logs.",
      why: String.raw`A product of dozens of densities underflows to 0.0 in floating point, and every class ties at zero. Summing $\log$-densities is numerically stable and monotonic — always classify in log-space.` },
    { wrong: "Letting a class's estimated variance go to (near) zero with few samples.",
      why: String.raw`With one or two points, the MLE variance can be ~0, so the Gaussian becomes a spike and the log-likelihood blows up. That's the exact failure the MAP prior (or a variance floor) prevents.` },
    { wrong: "Believing the 'naive' feature-independence assumption has to be true for the classifier to work.",
      why: String.raw`It's almost never true, yet Naive Bayes is often a strong classifier: even when the estimated <em>probabilities</em> are miscalibrated, the <em>argmax</em> (the actual decision) is frequently still correct. Useful ≠ literally true.` }
  ],

  quiz: [
    { q: "Two classes have priors P(A)=0.2, P(B)=0.8. For a point x, the likelihoods are p(x|A)=0.9, p(x|B)=0.3. Which class does Bayes pick?",
      options: ["B — because 0.8·0.3 = 0.24 > 0.2·0.9 = 0.18", "A — its likelihood is higher",
                "A — because 0.2·0.9 = 0.18 wins", "It's a tie"], answer: 0,
      explain: String.raw`Compare posteriors ∝ prior×likelihood: $A:0.2\cdot0.9=0.18$, $B:0.8\cdot0.3=0.24$. B wins. The strong prior for B overrides A's higher likelihood — that's the prior doing its job.` },
    { q: "A 1-D two-class problem: μ_A = 0, μ_B = 4, equal variance σ² = 1 and equal priors. Where is the decision boundary?",
      options: ["x = 2 (the midpoint)", "x = 0", "x = 4", "There is no boundary"], answer: 0,
      explain: String.raw`With equal variance and priors, the boundary is where $(x-\mu_A)^2=(x-\mu_B)^2$, i.e. the midpoint $x=2$. Equal variances make it linear (here, a single point in 1-D).` },
    { q: "Why is a general Gaussian Bayes boundary curved while logistic regression's is straight?",
      options: ["Different per-class covariances leave a quadratic term in the score difference; LR is linear in x by construction",
                "Because Gaussian Bayes uses more data",
                "Because logistic regression has no prior",
                "They are actually both always straight"], answer: 0,
      explain: String.raw`Each class score has a $(x-\mu)^2/\sigma^2$ term; unequal $\sigma$ leaves a quadratic in the difference ⇒ a conic boundary. Logistic regression models $\log$-odds as linear in $\mathbf x$, so its boundary is a hyperplane.` },
    { q: "In the experiment, at N = 2 points/class the MAP model scored ~0.96 while MLE scored ~0.75. The reason is:",
      options: ["MLE variance estimates are terrible with 2 points; the prior regularizes them toward a sane value",
                "MAP saw more data than MLE",
                "MLE is simply a buggy method",
                "The prior added new training points"], answer: 0,
      explain: String.raw`Two points give a wild (often near-zero) variance estimate. The MAP prior pulls it toward $\sigma_0^2$, stabilizing the Gaussians. As $N$ grows the data outvotes the prior and the two converge.` },
    { q: "You have 3 classes and features x. Which quantity do you NOT need to compute to make a prediction?",
      options: ["The denominator ∑ P(c')p(x|c') — it's identical across classes",
                "Each class prior P(c)",
                "Each class-conditional likelihood p(x|c)",
                "The argmax over classes"], answer: 0,
      explain: String.raw`The evidence (denominator) is the same for every class, so it cancels in the argmax. You only need it if you want normalized posterior <em>probabilities</em>, not for the decision itself.` }
  ],

  practice: [
    { level: "easy", prompt: "Change the class priors: pass a heavily imbalanced training set (e.g. 200 of class 0, 20 each of 1 and 2). How do the decision regions shift?",
      solution: String.raw`Class 0's region grows: its larger prior $P(y{=}0)$ adds a bigger constant to its log-score, so ties break in its favor further from its mean. Rare classes shrink. This is Bayes correctly encoding base rates.` },
    { level: "easy", prompt: "Set nu = 0 (MLE) and retrain on just 3 points per class a few times. Describe how unstable the boundaries look versus nu = 6 (MAP).",
      solution: String.raw`MLE boundaries jump around wildly run-to-run and sometimes collapse (near-zero variance spikes). MAP boundaries stay smooth and sensible because the prior anchors the variances. It's the visual version of the accuracy curve.` },
    { level: "med", prompt: "Make it LDA: force all classes to share one pooled covariance (average their variances). Confirm the boundaries become straight lines.",
      solution: String.raw`Pool: <code>vbar = np.mean([m.var[c] for c in classes], axis=0)</code> and use <code>vbar</code> for every class. Equal covariance cancels the quadratic term, leaving linear boundaries — that's Linear Discriminant Analysis.` },
    { level: "med", prompt: "Drop the naive assumption: fit a full 2×2 covariance per class (np.cov) and use the multivariate-normal log-density. Does accuracy change on this correlated data?",
      solution: String.raw`Use $-\tfrac12\log\det(2\pi\Sigma_c)-\tfrac12(\mathbf x-\mu_c)^\top\Sigma_c^{-1}(\mathbf x-\mu_c)$. Since these classes have correlated (tilted) covariances, the full model fits the tilt and usually edges out the diagonal 'naive' version. This is QDA.` },
    { level: "hard", prompt: "Compare calibration: bin test points by predicted P(class) and plot observed accuracy per bin (a reliability curve) for Gaussian Bayes vs logistic regression.",
      solution: String.raw`For each bin of predicted confidence, plot mean predicted vs actual correctness. Generative Gaussian Bayes is often over-confident under model misspecification; logistic regression, trained on the calibration objective (cross-entropy), tends to be better calibrated here.` },
    { level: "hard", prompt: "Derive the exact posterior odds log P(A|x)/P(B|x) for the 1-D equal-variance case and show it is linear in x. What is its slope?",
      solution: String.raw`$\log\frac{P(A|x)}{P(B|x)}=\log\frac{P(A)}{P(B)}+\frac{(\mu_A-\mu_B)}{\sigma^2}x-\frac{\mu_A^2-\mu_B^2}{2\sigma^2}$. The $x^2$ terms cancel (equal $\sigma$), leaving a line with slope $(\mu_A-\mu_B)/\sigma^2$ — exactly the logistic-regression log-odds form, which is why equal-variance Gaussian Bayes and LR agree in shape.` }
  ],

  deepDive: String.raw`<p><strong>Generative vs discriminative.</strong> Gaussian Bayes (generative) and logistic regression (discriminative) can describe the <em>same</em> linear boundary in the equal-covariance case, yet they estimate it differently. Ng & Jordan (2001) showed the trade-off precisely: the generative model has higher asymptotic error but <em>approaches</em> it faster — it needs fewer examples to get going. That is the mathematical version of what you saw: with tiny data, the model that builds in more assumptions (a prior, a Gaussian form) wins; with abundant data, the discriminative model that commits to fewer assumptions catches up and passes it.</p>
<p><strong>Priors are regularization.</strong> The MAP variance estimate is formally a posterior mode under an inverse-Gamma prior, but operationally it's just smoothing — the same move as Laplace ("add-one") smoothing in a text Naive Bayes, or an $L_2$ penalty in a regression. Every regularizer is a prior wearing an engineering hat, and this capstone is where that equivalence becomes concrete.</p>
<p><strong>Calibration caveat.</strong> Because the naive independence assumption is usually false, Gaussian Naive Bayes tends to produce <em>over-confident</em> probabilities (pushed toward 0 or 1) even when its top-1 decision is right. If you need the probability itself — for thresholding, ranking, or downstream expected-cost decisions — calibrate it (Platt scaling / isotonic) or prefer a discriminative model trained on the log-loss.</p>`
};

/* ------------------------------------------------------------------ C.4 */
window.LESSON_CONTENT["C.4"] = {
  subtitle: "Build a tiny language model, then measure it with the exact quantities that train GPT: cross-entropy, KL divergence, mutual information — and prove CE = H + KL numerically.",

  aiMoment: String.raw`The loss every language model minimizes is <strong>cross-entropy</strong>. The headline number in every LM paper is <strong>perplexity</strong>, which is just cross-entropy exponentiated. The penalty that keeps an RLHF-tuned model from drifting too far from its base model is a <strong>KL divergence</strong>. The objective behind contrastive pretraining (SimCLR, CLIP, InfoNCE) is a bound on <strong>mutual information</strong>. These four quantities <em>are</em> modern machine learning's scoreboard. In this capstone you build a character-level bigram model and compute all of them by hand, then verify the identity CE = H + KL that ties them together.`,

  plainEnglish: String.raw`<strong>Entropy</strong> is how surprised you are on average by the next symbol when you know the true odds. <strong>Cross-entropy</strong> is how surprised you are when you use the <em>wrong</em> odds (your model's) to place your bets. <strong>KL divergence</strong> is the extra surprise you eat for being wrong — the gap between the two. <strong>Mutual information</strong> is how much knowing the previous symbol cuts your surprise about the next one. Training a model is the act of shrinking that KL gap to zero.`,

  intuition: String.raw`Picture encoding text with a codebook that gives short codes to likely symbols and long codes to rare ones. Entropy is the fewest bits-per-symbol any codebook can achieve if you know the true distribution. If you build your codebook from a <em>model</em> $q$ that's slightly off, you pay cross-entropy bits — always at least entropy, with the surplus being exactly the KL divergence. Perplexity re-expresses those bits as an effective number of equally-likely choices: perplexity 8 means "as unsure as rolling a fair 8-sided die at each step." A better model = fewer bits = lower perplexity = a smaller die.`,

  formalism: String.raw`For distributions $p$ (truth) and $q$ (model) over the same alphabet:
$$H(p)=-\sum_x p(x)\log p(x),\qquad H(p,q)=-\sum_x p(x)\log q(x),\qquad D_{\mathrm{KL}}(p\Vert q)=\sum_x p(x)\log\frac{p(x)}{q(x)}.$$
These obey the central identity
$$\boxed{\,H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)\,}\qquad\text{and since }D_{\mathrm{KL}}\ge 0,\;\; H(p,q)\ge H(p).$$
<strong>Perplexity</strong> is $\mathrm{PPL}=2^{H(p,q)}$ in bits (or $e^{H(p,q)}$ in nats). <strong>Mutual information</strong> between the previous symbol $X$ and the next $Y$ is
$$I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}=H(X)+H(Y)-H(X,Y)=D_{\mathrm{KL}}\big(p(x,y)\,\Vert\,p(x)p(y)\big)\ge 0.$$`,

  derivation: String.raw`<strong>Two identities, proven by pushing symbols around.</strong>
<ol>
<li><strong>CE = H + KL.</strong> Start from cross-entropy and multiply inside the log by $p(x)/p(x)=1$:
$$H(p,q)=-\sum_x p(x)\log q(x)=-\sum_x p(x)\log\Big(p(x)\cdot\tfrac{q(x)}{p(x)}\Big).$$
Split the log of a product into a sum: $=-\sum_x p(x)\log p(x)-\sum_x p(x)\log\tfrac{q(x)}{p(x)}$. The first term is $H(p)$; the second is $+\sum_x p(x)\log\tfrac{p(x)}{q(x)}=D_{\mathrm{KL}}(p\Vert q)$. Hence $H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)$.</li>
<li><strong>Why training works.</strong> $H(p)$ depends only on the data, not on your model — it's a constant you cannot beat. So minimizing cross-entropy $H(p,q)$ over $q$ is <em>identical</em> to minimizing $D_{\mathrm{KL}}(p\Vert q)$: every gradient step on the LM loss is a step that makes the model's distribution closer to the data's, in the KL sense.</li>
<li><strong>MI as a KL divergence.</strong> Substituting $p(x,y)$ for "truth" and the independence model $p(x)p(y)$ for "$q$" in the KL formula gives $I(X;Y)=D_{\mathrm{KL}}(p(x,y)\Vert p(x)p(y))$. Because KL is always $\ge 0$, mutual information is always $\ge 0$: knowing $X$ can never <em>increase</em> your average uncertainty about $Y$. It's zero exactly when $X$ and $Y$ are independent.</li>
</ol>`,

  code: [
    { label: "1 · A bigram language model, and its cross-entropy + perplexity",
      src: String.raw`import numpy as np

corpus = ("the cat sat on the mat . the cat ate the rat . a rat sat on a mat . "
          "the fat cat sat on the mat . a cat ate a rat . the rat ran . ") * 6
chars = sorted(set(corpus)); V = len(chars); idx = {c: i for i, c in enumerate(chars)}
print("vocabulary size:", V, "->", ''.join(chars).replace(' ', '_'))

# Count every adjacent character pair: this IS the bigram model's raw data.
counts = np.zeros((V, V))
for a, b in zip(corpus[:-1], corpus[1:]):
    counts[idx[a], idx[b]] += 1

# Model q(next | prev), with add-alpha smoothing so nothing has probability exactly 0.
alpha = 0.1
Q = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha*V)

# Cross-entropy of the model on the corpus, in nats and bits, and perplexity.
logprob = sum(np.log(Q[idx[a], idx[b]]) for a, b in zip(corpus[:-1], corpus[1:]))
ce_nats = -logprob / (len(corpus) - 1)
ce_bits = ce_nats / np.log(2)
ppl = np.exp(ce_nats)
print(f"cross-entropy = {ce_nats:.4f} nats = {ce_bits:.4f} bits")
print(f"perplexity    = {ppl:.3f}  (as unsure as a fair {ppl:.1f}-sided die at each step)")` },
    { label: "2 · Prove CE = H + KL, and measure mutual information",
      src: String.raw`# The identity behind every LM loss:  CE(p, q) = H(p) + KL(p || q).
context = ' '                                        # true vs model 'next char' after a space
p = counts[idx[context]] / counts[idx[context]].sum() # empirical (true) distribution
q = Q[idx[context]]                                   # model's distribution
m = p > 0
H  = -(p[m] * np.log2(p[m])).sum()                    # entropy of the truth (bits)
KL =  (p[m] * np.log2(p[m] / q[m])).sum()             # extra bits for using q instead of p
CE = -(p[m] * np.log2(q[m])).sum()                    # cross-entropy
print(f"H(p) = {H:.4f}   KL(p||q) = {KL:.4f}   H + KL = {H+KL:.4f}   CE(p,q) = {CE:.4f}")
print("CE == H + KL ?", np.isclose(CE, H + KL))

# Mutual information between consecutive characters: how much does 'prev' tell you about 'next'?
P = counts / counts.sum()                             # joint P(prev, next)
px = P.sum(1); py = P.sum(0)                           # marginals
MI = sum(P[i, j] * np.log2(P[i, j] / (px[i]*py[j]))
         for i in range(V) for j in range(V) if P[i, j] > 0)
Hx = -(px[px>0]*np.log2(px[px>0])).sum()
Hy = -(py[py>0]*np.log2(py[py>0])).sum()
Pf = P[P>0]; Hxy = -(Pf*np.log2(Pf)).sum()
print(f"MI(prev; next) = {MI:.4f} bits   (check H(X)+H(Y)-H(X,Y) = {Hx+Hy-Hxy:.4f})")` },
    { label: "3 · See it: the model, and which pairs carry information",
      src: String.raw`import matplotlib.pyplot as plt
labels = [c.replace(' ', '_') for c in chars]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

im = ax[0].imshow(Q, cmap='viridis'); ax[0].set_title("model  q(next | prev)")
ax[0].set_xticks(range(V)); ax[0].set_xticklabels(labels, fontsize=7)
ax[0].set_yticks(range(V)); ax[0].set_yticklabels(labels, fontsize=7)
ax[0].set_xlabel("next"); ax[0].set_ylabel("prev"); fig.colorbar(im, ax=ax[0], fraction=.046)

# pointwise mutual information: red = pair co-occurs more than chance, blue = less
pmi = np.where(P > 0, np.log2((P + 1e-12) / (np.outer(px, py) + 1e-12)), 0)
im2 = ax[1].imshow(pmi, cmap='RdBu_r', vmin=-4, vmax=4)
ax[1].set_title("pointwise MI:  log2  P(x,y) / P(x)P(y)")
ax[1].set_xticks(range(V)); ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_yticks(range(V)); ax[1].set_yticklabels(labels, fontsize=7)
ax[1].set_xlabel("next"); ax[1].set_ylabel("prev"); fig.colorbar(im2, ax=ax[1], fraction=.046)
fig.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 210" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <text x="175" y="26" text-anchor="middle" font-weight="700" fill="#1f2a44">cross-entropy = H(p) + KL(p‖q)</text>
  <rect x="20" y="52" width="210" height="42" fill="#cfe6d6" stroke="#3a7d44"/>
  <rect x="230" y="52" width="96" height="42" fill="#f4c9cf" stroke="#c1121f"/>
  <text x="125" y="78" text-anchor="middle" fill="#204d2a">H(p)</text>
  <text x="278" y="78" text-anchor="middle" fill="#8f2233">KL</text>
  <text x="125" y="112" text-anchor="middle" fill="#3a7d44" font-size="10.5">unavoidable surprise</text>
  <text x="278" y="112" text-anchor="middle" fill="#c1121f" font-size="10.5">wasted bits</text>
  <text x="173" y="140" text-anchor="middle" fill="#4a5878" font-size="10.5">training drives KL → 0, so CE → H</text>
  <text x="173" y="162" text-anchor="middle" fill="#4a5878" font-size="10.5">perplexity = 2&#8319;&#8309; (bits) = effective #choices</text>
  <text x="560" y="26" text-anchor="middle" font-weight="700" fill="#1f2a44">mutual information = shared entropy</text>
  <circle cx="520" cy="120" r="58" fill="#2a6f97" fill-opacity="0.20" stroke="#2a6f97"/>
  <circle cx="600" cy="120" r="58" fill="#3a7d44" fill-opacity="0.20" stroke="#3a7d44"/>
  <text x="486" y="124" text-anchor="middle" fill="#1c4e70">H(X)</text>
  <text x="634" y="124" text-anchor="middle" fill="#245030">H(Y)</text>
  <text x="560" y="118" text-anchor="middle" fill="#3a2327" font-weight="700">I(X;Y)</text>
  <text x="560" y="134" text-anchor="middle" fill="#5a6b8c" font-size="9.5">overlap</text>
  <text x="560" y="196" text-anchor="middle" fill="#4a5878" font-size="10.5">prev X — next Y;   I = H(X)+H(Y)−H(X,Y)</text>
</svg>`,

  keyPoints: [
    String.raw`Cross-entropy is the LM loss; perplexity $=2^{\text{CE bits}}$ is that loss made interpretable as an "effective number of choices." Lower is better; they carry the same information.`,
    String.raw`$H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)$ and $D_{\mathrm{KL}}\ge0$, so cross-entropy can never beat the data's own entropy $H(p)$. Minimizing CE = minimizing KL to the data.`,
    String.raw`Bits vs nats is only the log base: $\log_2$ gives bits, $\ln$ gives nats, and $1\text{ nat}=1/\ln 2\approx1.443$ bits. Frameworks report nats; papers often report bits. Convert, don't confuse.`,
    String.raw`KL is <strong>not symmetric</strong>: $D_{\mathrm{KL}}(p\Vert q)\ne D_{\mathrm{KL}}(q\Vert p)$ in general. "Forward" vs "reverse" KL give different-behaving objectives (mean-seeking vs mode-seeking).`,
    String.raw`Mutual information $=D_{\mathrm{KL}}(\text{joint}\Vert\text{product of marginals})\ge0$, zero iff independent. It's the principled measure of "how much X tells you about Y."`
  ],

  commonMistakes: [
    { wrong: "Reporting or comparing perplexities computed in different bases or over different tokenizers.",
      why: String.raw`Perplexity depends on the log base <em>and</em> the unit (chars vs subwords vs words). A char-level PPL and a word-level PPL aren't comparable, and bits-PPL ≠ nats-PPL. Only compare like with like.` },
    { wrong: "Assuming KL(p‖q) = KL(q‖p).",
      why: String.raw`KL is asymmetric. Forward KL $D(p\Vert q)$ (used in max-likelihood) is mean-seeking and punishes $q$ for missing mass $p$ has; reverse KL (used in some variational/RL objectives) is mode-seeking. Picking the wrong direction changes what your model does.` },
    { wrong: "Letting a model assign probability 0 to something that then occurs.",
      why: String.raw`If $q(x)=0$ but $x$ appears, $\log q(x)=-\infty$ and cross-entropy/perplexity blow up. That's why we smoothed ($\alpha>0$); real LMs use a softmax, which is strictly positive everywhere for the same reason.` }
  ],

  quiz: [
    { q: "A language model reaches a cross-entropy of 3 bits per token on held-out text. What is its perplexity?",
      options: ["8", "3", "9", "6"], answer: 0,
      explain: String.raw`$\mathrm{PPL}=2^{\text{bits}}=2^3=8$. The model is, on average, as uncertain as choosing uniformly among 8 options each step.` },
    { q: "p = (0.5, 0.5), q = (0.25, 0.75). Compute KL(p‖q) in bits.",
      options: ["≈ 0.207 bits", "0 bits", "1 bit", "≈ 0.5 bits"], answer: 0,
      explain: String.raw`$0.5\log_2\frac{0.5}{0.25}+0.5\log_2\frac{0.5}{0.75}=0.5(1)+0.5(-0.585)=0.5-0.293\approx0.207$ bits.` },
    { q: "Cross-entropy H(p,q) is always at least H(p). Which fact guarantees this?",
      options: ["KL(p‖q) ≥ 0 and H(p,q) = H(p) + KL(p‖q)", "H(p) is always negative",
                "q is always uniform", "Perplexity is exponential"], answer: 0,
      explain: String.raw`Since $H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)$ and KL is non-negative, the extra term can only add. The data's entropy is the floor no model can beat.` },
    { q: "If the previous character were statistically independent of the next, what would the mutual information be?",
      options: ["Exactly 0", "Exactly 1 bit", "Equal to the entropy H(Y)", "Infinite"], answer: 0,
      explain: String.raw`Independence means $p(x,y)=p(x)p(y)$, so the KL between them is 0, hence $I(X;Y)=0$. Any positive MI means the previous symbol carries predictive signal.` },
    { q: "Your loss prints 1.5 (nats). What is it in bits?",
      options: ["≈ 2.16 bits", "≈ 1.04 bits", "0.75 bits", "3 bits"], answer: 0,
      explain: String.raw`bits $=$ nats$/\ln 2 = 1.5/0.693\approx2.16$. Multiply nats by $1.443$ to get bits.` }
  ],

  practice: [
    { level: "easy", prompt: "Compute the entropy of a fair six-sided die in bits, by hand and in code. Why is it not an integer?",
      solution: String.raw`$H=\log_2 6\approx2.585$ bits. Code: <code>np.log2(6)</code>. It isn't an integer because 6 isn't a power of 2 — you'd need a fractional number of bits per roll on average (achievable by coding many rolls together).` },
    { level: "easy", prompt: "Raise the smoothing alpha from 0.1 to 5.0 and rerun. What happens to perplexity, and why?",
      solution: String.raw`Perplexity rises: heavy smoothing flattens $q$ toward uniform, so the model is less confident on the true next characters and pays more bits. Smoothing trades sharpness for safety against zeros.` },
    { level: "med", prompt: "Split the corpus into train/test halves. Fit Q on train only and compute perplexity on the held-out half. How does it compare to the training perplexity?",
      solution: String.raw`Held-out perplexity is higher — the model memorized train-specific bigrams that don't all transfer. That gap between train and test perplexity is exactly the generalization gap LMs fight with more data and regularization.` },
    { level: "med", prompt: "Build a unigram model (ignore the previous char) and compare its perplexity to the bigram model. Relate the improvement to the mutual information you computed.",
      solution: String.raw`Bigram perplexity is lower. In fact the average per-symbol reduction in cross-entropy from conditioning on the previous character equals the mutual information $I(X;Y)$ — MI is literally the bits/character the context buys you.` },
    { level: "hard", prompt: "Construct two independent random symbol streams and verify their mutual information is ≈ 0 (up to finite-sample noise). Why isn't it exactly 0?",
      solution: String.raw`Sample $X,Y$ independently; estimate $\hat I$ from counts. It's small but positive because finite samples show spurious correlations — the plug-in MI estimator is biased upward by roughly $\frac{(V-1)^2}{2N\ln 2}$ bits. It shrinks as $N\to\infty$.` },
    { level: "hard", prompt: "Demonstrate KL's asymmetry: pick p sharp and q broad, compute KL(p‖q) and KL(q‖p), and explain which penalizes 'missing a mode' vs 'covering non-modes'.",
      solution: String.raw`With $p$ sharp, $q$ broad: $D(p\Vert q)$ stays modest (p's mass sits where q is nonzero), while $D(q\Vert p)$ is large (q puts mass where $p\approx0$, and $\log\frac{q}{p}\to\infty$). Forward KL $D(p\Vert q)$ is mean-seeking (covers all of p's mass); reverse KL $D(q\Vert p)$ is mode-seeking (concentrates on one peak). This is the crux of variational inference and distillation design.` }
  ],

  deepDive: String.raw`<p><strong>The LM loss is maximum likelihood.</strong> Minimizing average cross-entropy $-\frac1N\sum\log q(x_i)$ is exactly maximizing the log-likelihood of the data under the model — and, by CE = H + KL, is exactly minimizing $D_{\mathrm{KL}}(\text{data}\Vert\text{model})$. Three names, one gradient. This is why "train to predict the next token" and "fit the data distribution" are the same sentence.</p>
<p><strong>Where each quantity shows up.</strong> KL is the leash in RLHF/PPO (a $\beta\,D_{\mathrm{KL}}(\pi\Vert\pi_{\text{ref}})$ term keeps the policy near the reference model) and the objective in variational inference and knowledge distillation (match a student's distribution to a teacher's). Mutual information underlies contrastive learning: InfoNCE is a tractable lower bound on $I(\text{view}_1;\text{view}_2)$, so SimCLR/CLIP are, formally, MI-maximizers. Even <em>label smoothing</em> is an information-theoretic tweak: it adds a KL-to-uniform term that stops the model from becoming infinitely confident.</p>
<p><strong>Estimation is subtle.</strong> Plug-in entropy and MI estimates from counts are biased on small samples (entropy biased low, MI biased high, by $O(V/N)$ terms). Serious measurements use bias corrections (Miller–Madow) or dedicated estimators (KSG for continuous MI). The identities you verified are exact; the <em>estimates</em> of them from finite data are where the care goes.</p>`
};

/* ------------------------------------------------------------------ C.5 */
window.LESSON_CONTENT["C.5"] = {
  subtitle: "Build reverse-mode automatic differentiation — the engine inside PyTorch — in about 40 lines, check it against finite differences, then train a neural net with it.",

  aiMoment: String.raw`Every time you call <code>loss.backward()</code> in PyTorch, a <strong>reverse-mode automatic differentiation</strong> engine walks the computation graph and hands you a gradient for every parameter. That one mechanism — not any particular architecture — is what makes training billion-parameter models possible. Andrej Karpathy's <em>micrograd</em> famously showed the whole idea fits on a napkin. In this capstone you build that engine from a single scalar <code>Value</code> class, prove it correct against finite differences, and then use it to train a 2-layer network that learns a curved decision boundary.`,

  plainEnglish: String.raw`Every number in your calculation remembers the two things: what it equals, and how it was built from other numbers. Once you have the final loss, you walk <em>backwards</em> through that history. At each step you ask "if I nudge this input a bit, how much does the output change?" — the chain rule — and multiply those local answers together. When you reach the parameters, you've got their gradients. That backward walk is the whole trick.`,

  intuition: String.raw`Think of the computation as a graph: leaves are inputs and weights, internal nodes are operations (+, ×, tanh), the root is the loss. The <strong>forward pass</strong> fills in each node's value bottom-up. The <strong>backward pass</strong> fills in each node's gradient top-down: the root's gradient is 1 (the loss with respect to itself), and every edge multiplies by its <em>local</em> derivative on the way down. When a value feeds into several places, its gradient is the <em>sum</em> of what comes back along each path — that plus-sign is the multivariable chain rule, and it's why we accumulate with <code>+=</code>.`,

  formalism: String.raw`Reverse-mode AD computes $\frac{\partial L}{\partial v}$ for every node $v$ using the chain rule. If $v$ feeds operations that produce nodes $u_1,\dots,u_k$, then
$$\frac{\partial L}{\partial v}=\sum_{i=1}^{k}\frac{\partial L}{\partial u_i}\,\frac{\partial u_i}{\partial v}.$$
Each operation contributes a known <em>local</em> derivative $\partial u_i/\partial v$:
$$\frac{\partial (a+b)}{\partial a}=1,\quad \frac{\partial (ab)}{\partial a}=b,\quad \frac{\partial\,a^{p}}{\partial a}=p\,a^{p-1},\quad \frac{d}{dx}\tanh x=1-\tanh^2 x,\quad \frac{d}{dx}\operatorname{relu}(x)=\mathbf 1[x>0].$$
Processing nodes in reverse topological order guarantees that by the time we handle $v$, every $\partial L/\partial u_i$ downstream is already known.`,

  derivation: String.raw`<strong>Why the algorithm is exactly these three rules.</strong>
<ol>
<li><strong>Seed the root.</strong> $\partial L/\partial L=1$. That's the base case the whole backward pass grows from.</li>
<li><strong>Local rule per op.</strong> For a node $u=f(v,\dots)$, calculus gives $\partial u/\partial v$ in closed form (the list above). Reverse-mode multiplies the incoming $\partial L/\partial u$ by this local factor to push the gradient onto $v$: the operation's <code>_backward</code> does <code>v.grad += (∂u/∂v) · u.grad</code>.</li>
<li><strong>Accumulate over fan-out (the <code>+=</code>).</strong> If $v$ is used in several operations, each sends back a contribution and they <em>add</em> — that's the summation in the chain-rule formula above. Using <code>=</code> instead of <code>+=</code> would keep only the last path and silently compute the wrong gradient.</li>
<li><strong>Order matters.</strong> A node's gradient is only correct once <em>all</em> its consumers have contributed. Visiting nodes in reverse topological order (build a topo sort of the graph, then walk it backwards) guarantees exactly that. This is why we sort before we accumulate.</li>
<li><strong>Why reverse, not forward.</strong> With one scalar loss and millions of parameters, reverse-mode computes <em>all</em> parameter gradients in a single backward pass — cost proportional to the forward pass. Forward-mode would need one pass per parameter. That asymmetry is the entire reason deep learning is affordable.</li>
</ol>`,

  code: [
    { label: "1 · The entire autograd engine: a scalar Value with backward()",
      src: String.raw`import numpy as np, math, random
random.seed(1); np.random.seed(1)

class Value:
    """A scalar that remembers how it was computed, so it can backprop through itself."""
    def __init__(self, data, _children=()):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None; self._prev = set(_children)
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data + o.data, (self, o))
        def _b(): self.grad += out.grad; o.grad += out.grad          # d(a+b) = 1, 1
        out._backward = _b; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data * o.data, (self, o))
        def _b(): self.grad += o.data * out.grad; o.grad += self.data * out.grad   # d(ab) = b, a
        out._backward = _b; return out
    def __pow__(self, p):
        out = Value(self.data ** p, (self,))
        def _b(): self.grad += p * self.data ** (p - 1) * out.grad   # d(a^p) = p a^(p-1)
        out._backward = _b; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def _b(): self.grad += (1 - t * t) * out.grad               # d tanh = 1 - tanh^2
        out._backward = _b; return out
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def _b(): self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _b; return out
    def __neg__(s):        return s * -1
    def __radd__(s, o):    return s + o
    def __rmul__(s, o):    return s * o
    def __sub__(s, o):     return s + (-o)
    def __rsub__(s, o):    return (-s) + o
    def __truediv__(s, o): return s * (o ** -1 if isinstance(o, Value) else o ** -1)
    def backward(self):
        topo, seen = [], set()                      # build reverse topological order
        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev: build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0                             # seed: dL/dL = 1
        for v in reversed(topo): v._backward()      # chain rule, downstream first

print("Autograd engine ready. That class is the whole thing.")` },
    { label: "2 · Prove it's correct: autograd vs finite differences",
      src: String.raw`# A deliberately messy expression touching every operation.
a = Value(2.0); b = Value(1.5); c = Value(0.8)
f = (a*b + (b*c).tanh() + a**2).relu() + c/b
f.backward()

# Finite differences: nudge each input by h and watch f, using ordinary floats.
def f_numeric(av, bv, cv):
    r = av*bv + math.tanh(bv*cv) + av**2
    r = r if r > 0 else 0.0
    return r + cv/bv

h = 1e-6
for name, var in [('a', a), ('b', b), ('c', c)]:
    base = [2.0, 1.5, 0.8]; i = 'abc'.index(name)
    hi = base.copy(); hi[i] += h
    lo = base.copy(); lo[i] -= h
    numeric = (f_numeric(*hi) - f_numeric(*lo)) / (2*h)
    print(f"d/d{name}:  autograd = {var.grad:+.6f}   finite-diff = {numeric:+.6f}   match = {np.isclose(var.grad, numeric, atol=1e-4)}")` },
    { label: "3 · Train a 2-layer MLP with it (takes ~15–20s in the browser)",
      src: String.raw`import matplotlib.pyplot as plt, time

# two-moons dataset, built by hand (no sklearn), then standardized
def moons(n=80, noise=0.10):
    t = np.linspace(0, np.pi, n//2)
    outer = np.c_[np.cos(t), np.sin(t)]
    inner = np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5]
    X = np.vstack([outer, inner]) + noise*np.random.randn(n, 2)
    y = np.array([1.0]*(n//2) + [-1.0]*(n//2)); return X, y
X, y = moons(80); X = (X - X.mean(0)) / X.std(0)

# a 2-layer MLP (2 -> 8 tanh -> 1), where every weight is a Value
class Neuron:
    def __init__(s, nin): s.w = [Value(np.random.randn()*0.8) for _ in range(nin)]; s.b = Value(0.0)
    def __call__(s, x):   return sum((wi*xi for wi, xi in zip(s.w, x)), s.b)
    def params(s):        return s.w + [s.b]
class Layer:
    def __init__(s, nin, nout, act): s.neurons = [Neuron(nin) for _ in range(nout)]; s.act = act
    def __call__(s, x):
        out = [n(x) for n in s.neurons]
        return [v.tanh() for v in out] if s.act == 'tanh' else out
    def params(s): return [p for n in s.neurons for p in n.params()]
class MLP:
    def __init__(s): s.l1 = Layer(2, 8, 'tanh'); s.l2 = Layer(8, 1, 'linear')
    def __call__(s, x): return s.l2(s.l1(x))[0]
    def params(s): return s.l1.params() + s.l2.params()

model = MLP(); print("trainable parameters:", len(model.params()))

def loss_and_acc(reg=1e-3):
    scores = [model([Value(a), Value(b)]) for a, b in X]
    data_loss = sum(((1 + (-yi)*s).relu() for yi, s in zip(y, scores))) * (1.0/len(scores))  # SVM hinge
    reg_loss = reg * sum((p*p for p in model.params()))                                        # L2
    acc = np.mean([(s.data > 0) == (yi > 0) for yi, s in zip(y, scores)])
    return data_loss + reg_loss, acc

t0 = time.time(); history = []
for k in range(150):
    L, acc = loss_and_acc()
    for p in model.params(): p.grad = 0.0      # zero the grads (they accumulate!)
    L.backward()                                # our engine fills every p.grad
    lr = 1.0 - 0.9*k/150                         # simple learning-rate decay
    for p in model.params(): p.data -= lr * p.grad
    history.append((L.data, acc))
print(f"trained 150 steps in {time.time()-t0:.1f}s   final loss = {history[-1][0]:.3f}   accuracy = {history[-1][1]:.1%}")

# pull trained weights into numpy for a fast decision-boundary sweep
W1 = np.array([[w.data for w in n.w] for n in model.l1.neurons]); b1 = np.array([n.b.data for n in model.l1.neurons])
W2 = np.array([[w.data for w in n.w] for n in model.l2.neurons]); b2 = np.array([n.b.data for n in model.l2.neurons])
forward = lambda P: (np.tanh(P @ W1.T + b1) @ W2.T + b2).ravel()
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.6, X[:,0].max()+.6, 220),
                     np.linspace(X[:,1].min()-.6, X[:,1].max()+.6, 220))
Z = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].contourf(xx, yy, (Z > 0).astype(float), levels=[-.1,.5,1.1], colors=['#f3d9dd','#d6e4ef'])
ax[0].contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.3)
ax[0].scatter(X[y>0,0], X[y>0,1], c='#c1121f', s=20, edgecolors='k', lw=.3, label='class +1')
ax[0].scatter(X[y<0,0], X[y<0,1], c='#2a6f97', s=20, edgecolors='k', lw=.3, label='class -1')
ax[0].set_title(f"Learned decision boundary (acc {history[-1][1]:.0%})"); ax[0].legend(fontsize=8, loc='upper right')
Ls = [hh[0] for hh in history]; As = [hh[1] for hh in history]
ax[1].plot(Ls, color='#3a0ca3'); ax[1].set_xlabel("step"); ax[1].set_ylabel("loss", color='#3a0ca3')
axb = ax[1].twinx(); axb.plot(As, color='#2a9d8f'); axb.set_ylabel("accuracy", color='#2a9d8f'); axb.set_ylim(0, 1.05)
ax[1].set_title("Training loss & accuracy"); fig.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 660 220" width="100%" style="max-width:660px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <defs><marker id="c5f" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#8a97b3"/></marker>
        <marker id="c5b" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#c1121f"/></marker></defs>
  <text x="330" y="20" text-anchor="middle" font-weight="700" fill="#1f2a44">forward computes values (grey) · backward computes gradients (red)</text>
  <rect x="26" y="40" width="120" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/>
  <text x="86" y="65" text-anchor="middle" fill="#1f2a44">a = 2</text>
  <rect x="26" y="132" width="120" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/>
  <text x="86" y="157" text-anchor="middle" fill="#1f2a44">b = −3</text>
  <rect x="252" y="78" width="150" height="40" rx="6" fill="#e7eff6" stroke="#2a6f97"/>
  <text x="327" y="103" text-anchor="middle" fill="#1c4e70">e = a·b = −6</text>
  <rect x="252" y="160" width="120" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/>
  <text x="312" y="185" text-anchor="middle" fill="#1f2a44">c = 10</text>
  <rect x="500" y="98" width="140" height="40" rx="6" fill="#e7f0e8" stroke="#3a7d44"/>
  <text x="570" y="123" text-anchor="middle" fill="#245030">L = e + c = 4</text>
  <line x1="146" y1="58" x2="250" y2="90" stroke="#8a97b3" marker-end="url(#c5f)"/>
  <line x1="146" y1="150" x2="250" y2="104" stroke="#8a97b3" marker-end="url(#c5f)"/>
  <line x1="402" y1="98" x2="498" y2="112" stroke="#8a97b3" marker-end="url(#c5f)"/>
  <line x1="372" y1="178" x2="498" y2="126" stroke="#8a97b3" marker-end="url(#c5f)"/>
  <text x="86" y="98" text-anchor="middle" fill="#c1121f" font-size="10.5">∂L/∂a = b = −3</text>
  <text x="86" y="190" text-anchor="middle" fill="#c1121f" font-size="10.5">∂L/∂b = a = 2</text>
  <text x="327" y="136" text-anchor="middle" fill="#c1121f" font-size="10.5">∂L/∂e = 1</text>
  <text x="312" y="216" text-anchor="middle" fill="#c1121f" font-size="10.5">∂L/∂c = 1</text>
  <text x="570" y="156" text-anchor="middle" fill="#c1121f" font-size="10.5">∂L/∂L = 1 (seed)</text>
  <line x1="250" y1="70" x2="150" y2="52" stroke="#c1121f" stroke-dasharray="4 3" marker-end="url(#c5b)"/>
  <line x1="250" y1="112" x2="150" y2="150" stroke="#c1121f" stroke-dasharray="4 3" marker-end="url(#c5b)"/>
  <line x1="498" y1="108" x2="404" y2="94" stroke="#c1121f" stroke-dasharray="4 3" marker-end="url(#c5b)"/>
</svg>`,

  keyPoints: [
    String.raw`Reverse-mode AD is three ideas: seed $\partial L/\partial L=1$, multiply by each op's local derivative on the way back, and <strong>accumulate</strong> contributions where a value fans out.`,
    String.raw`The <code>+=</code> in every <code>_backward</code> is the multivariable chain rule. Overwrite with <code>=</code> and you silently compute wrong gradients — this is the single most common autograd bug.`,
    String.raw`You must process nodes in reverse topological order so a node's gradient is complete before it's used. That's why <code>backward()</code> topo-sorts first.`,
    String.raw`Always zero gradients before each backward pass. Because grads accumulate, skipping <code>p.grad = 0</code> sums gradients across steps — exactly why PyTorch makes you call <code>optimizer.zero_grad()</code>.`,
    String.raw`Reverse-mode gives <em>all</em> parameter gradients in one backward pass at the cost of one forward pass. That one-scalar-out, many-inputs asymmetry is why it, not forward-mode, powers deep learning.`
  ],

  commonMistakes: [
    { wrong: "Forgetting to zero p.grad between steps.",
      why: String.raw`Gradients accumulate by design (the fan-out sum). Without zeroing, step $t$ uses the sum of gradients from steps $1..t$, and training destabilizes. PyTorch's <code>zero_grad()</code> exists for exactly this reason.` },
    { wrong: "Writing v.grad = local * out.grad instead of v.grad += local * out.grad.",
      why: String.raw`When a value is used more than once (e.g. a weight feeding several neurons, or <code>x</code> in <code>x*x</code>), only the accumulated sum is correct. Assignment keeps just one path and gives wrong gradients that finite-difference checks immediately expose.` },
    { wrong: "Trusting an unverified gradient implementation.",
      why: String.raw`A gradient bug can still let a model train 'okay-ish', hiding the error. The finite-difference check in cell 2 is cheap insurance and should be your reflex whenever you hand-write a backward pass.` }
  ],

  quiz: [
    { q: "For L = a·b + c with a = 3, b = 4, what is ∂L/∂a?",
      options: ["4 (the value of b)", "3", "7", "12"], answer: 0,
      explain: String.raw`$\partial L/\partial a=b=4$. The $+c$ contributes nothing to $\partial/\partial a$, and the product rule gives $b$.` },
    { q: "In the expression y = x·x, why must x.grad accumulate from both factors?",
      options: ["x fans out into two inputs of the multiply, so the chain rule sums both paths: dy/dx = 2x",
                "Because x is negative", "It shouldn't — one path is enough", "To make it run faster"], answer: 0,
      explain: String.raw`Both operands are the same node, so backward hits <code>x.grad += </code> twice, giving $x+x=2x$ — the correct derivative. With <code>=</code> you'd get just $x$.` },
    { q: "Reverse-mode does a single backward pass to get gradients for how many parameters?",
      options: ["All of them at once", "Exactly one", "Two", "Half of them"], answer: 0,
      explain: String.raw`One backward pass yields $\partial L/\partial\theta$ for every parameter simultaneously, at roughly the cost of the forward pass. That's the whole efficiency argument for backprop.` },
    { q: "Why does backward() sort the graph topologically before accumulating?",
      options: ["So each node's gradient is fully accumulated before it's passed further back",
                "To make the graph smaller", "To randomize the order", "Topological order is irrelevant"], answer: 0,
      explain: String.raw`A node may feed several downstream nodes; its gradient is only complete once all of them have reported back. Reverse topological order guarantees that ordering.` },
    { q: "Through f = relu(z), the incoming gradient is 2.0 and z = −0.5. What gradient flows to z?",
      options: ["0 (relu is flat for z < 0)", "2.0", "−1.0", "1.0"], answer: 0,
      explain: String.raw`$\operatorname{relu}'(z)=\mathbf 1[z>0]=0$ for $z=-0.5$, so $0\times2.0=0$. Negative pre-activations pass no gradient — the origin of 'dying ReLU'.` }
  ],

  practice: [
    { level: "easy", prompt: "Add an exp() method to Value with the correct backward, and finite-difference-check d/dx exp(x).",
      solution: String.raw`<code>def exp(self): t=math.exp(self.data); out=Value(t,(self,)); out._backward=lambda: setattr(self,'grad',self.grad+t*out.grad) or None; return out</code> (or the closure form). Since $\frac{d}{dx}e^x=e^x$, the local factor is the output value <code>t</code>. Verify against $(e^{x+h}-e^{x-h})/2h$.` },
    { level: "easy", prompt: "Remove the p.grad = 0 line from the training loop. Describe what happens to the loss and why.",
      solution: String.raw`Gradients accumulate across steps, so the effective step size explodes and the loss diverges (or oscillates wildly). It's the concrete demonstration of why zeroing grads is mandatory.` },
    { level: "med", prompt: "Swap the hinge loss for logistic (binary cross-entropy) using a sigmoid built from exp. Does the boundary change much?",
      solution: String.raw`Add <code>sigmoid</code> via <code>1/(1+(-x).exp())</code> and loss $-[y\log p+(1-y)\log(1-p)]$ with $y\in\{0,1\}$. The learned boundary is similar (both are max-margin-ish), but logistic gives calibrated probabilities and a smoother loss surface.` },
    { level: "med", prompt: "Add a second hidden layer (2 → 8 → 8 → 1). How do parameter count, training time, and the boundary change?",
      solution: String.raw`Insert another <code>Layer(8, 8, 'tanh')</code>. Parameters jump from 33 to ~105; each scalar-graph step is bigger so training is slower; the boundary can bend more flexibly (helpful on harder data, prone to overfit on easy moons).` },
    { level: "hard", prompt: "Verify a full network gradient: pick one weight, compute its autograd gradient, then finite-difference the whole loss w.r.t. that weight. Do they match?",
      solution: String.raw`Save <code>w.grad</code> after backward; then set <code>w.data += h</code>, recompute loss (call <code>loss_and_acc()[0].data</code>), set <code>w.data -= 2h</code>, recompute, and form the central difference. They should agree to ~1e-4 — end-to-end proof the engine is correct on the real network.` },
    { level: "hard", prompt: "Explain why scalar autograd is too slow for real nets, and what changes to make it a tensor engine like PyTorch.",
      solution: String.raw`Each scalar op is a Python object and a closure — millions of them per step. Real engines store <em>tensors</em> at each node and give each op a vectorized backward (a vector-Jacobian product) so one node handles a whole layer's arithmetic in optimized C/CUDA. Same graph, same chain rule — just batched, and with a tape instead of per-scalar closures.` }
  ],

  deepDive: String.raw`<p><strong>Reverse vs forward mode, precisely.</strong> Automatic differentiation comes in two flavors. Forward mode propagates derivatives input→output and costs one pass <em>per input</em>; reverse mode propagates output→input and costs one pass <em>per output</em>. Deep learning has one output (the scalar loss) and millions of inputs (parameters), so reverse mode wins overwhelmingly — you pay for the one output, not the millions of inputs. Flip the ratio (many outputs, few inputs, like a Jacobian row) and forward mode would win. Knowing which regime you're in is a genuinely useful engineering instinct.</p>
<p><strong>From micrograd to PyTorch.</strong> The engine you built is complete but scalar. Production frameworks keep the identical graph-and-chain-rule structure and change one thing: each node holds a <em>tensor</em>, and each operation ships a vectorized backward (a vector–Jacobian product) that runs in fused C/CUDA kernels. They also record operations on a <em>tape</em> during the forward pass rather than via Python closures, and they add <em>gradient checkpointing</em> — recomputing some forward values during backward to trade compute for memory on very deep graphs.</p>
<p><strong>Why this is the capstone.</strong> Linear algebra gave you the layers, calculus gave you the derivatives, optimization gave you the update rule, probability gave you the loss. Autograd is the machine that connects them: it turns "here is a loss built from those pieces" into "here is how to change every parameter to reduce it." Build it once by hand and <code>loss.backward()</code> stops being magic for good.</p>`
};

