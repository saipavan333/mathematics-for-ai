/* ============================================================
   TRACK 5 — Linear Algebra IV — Eigenvalues, PCA & SVD
   Opener: 5.1 Eigenvalues & Eigenvectors.
   ============================================================ */
(window.LESSON_CONTENT ||= {})["5.1"] = {
  subtitle: "The special directions a matrix only stretches — and the language of loss curvature.",

  aiMoment: String.raw`<p>The <strong>Hessian</strong> of a loss has eigenvalues that <em>are</em> the curvature of the
  loss surface: big eigenvalues mean steep, narrow valleys; near-zero ones mean flat plateaus; negative ones mean you're
  at a saddle, not a minimum. PCA finds the eigenvectors of a covariance matrix; PageRank is an eigenvector of the web;
  spectral norm is the largest singular value (a cousin of eigenvalues). "What are the eigenvalues?" is one of the most
  useful questions you can ask about any matrix in ML.</p>`,

  plainEnglish: String.raw`<p>A matrix is a machine that moves vectors around — usually rotating <em>and</em> stretching
  them. But a few special directions come out pointing the exact same way they went in, only longer or shorter. Those
  directions are <strong>eigenvectors</strong>, and the stretch factor is the <strong>eigenvalue</strong>.</p>`,

  intuition: String.raw`<p>Apply the matrix to lots of arrows. Most get knocked off their line (rotated). An
  eigenvector is the stubborn arrow that stays on its own line — the matrix can only lengthen, shrink, or flip it.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Eigenvector stays on its line while a generic vector rotates">
    <line x1="20" y1="150" x2="300" y2="150" stroke="#eef0f4"/>
    <line x1="160" y1="20" x2="160" y2="210" stroke="#eef0f4"/>
    <line x1="160" y1="150" x2="250" y2="80" stroke="#cbd5e1" stroke-dasharray="4 4"/>
    <line x1="160" y1="150" x2="214" y2="108" stroke="#4f46e5" stroke-width="3" marker-end="url(#e1)"/>
    <line x1="160" y1="150" x2="268" y2="66" stroke="#818cf8" stroke-width="3" marker-end="url(#e2)"/>
    <text x="210" y="120" font-size="10" fill="#4f46e5" font-family="sans-serif">v</text>
    <text x="262" y="58" font-size="10" fill="#818cf8" font-family="sans-serif">Av = λv</text>
    <line x1="160" y1="150" x2="96" y2="110" stroke="#94a3b8" stroke-width="3" marker-end="url(#e3)"/>
    <line x1="160" y1="150" x2="92" y2="158" stroke="#cbd5e1" stroke-width="3" marker-end="url(#e4)"/>
    <text x="78" y="104" font-size="10" fill="#94a3b8" font-family="sans-serif">u</text>
    <text x="60" y="170" font-size="10" fill="#94a3b8" font-family="sans-serif">Au (rotated)</text>
    <defs>
      <marker id="e1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="e2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#818cf8"/></marker>
      <marker id="e3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker>
      <marker id="e4" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#cbd5e1"/></marker>
    </defs>
  </svg>
  <figcaption>v keeps its direction (only scaled by λ); the generic u gets rotated off its line.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A nonzero vector $\mathbf v$ is an <strong>eigenvector</strong> of a square matrix $A$ with
  <strong>eigenvalue</strong> $\lambda$ if</p>
  $$A\mathbf v=\lambda\mathbf v,\qquad \mathbf v\neq\mathbf 0.$$
  <p>Plain English: acting with $A$ is the same as just scaling by the number $\lambda$. To find the $\lambda$'s, rewrite
  as $(A-\lambda I)\mathbf v=\mathbf 0$. A nonzero solution exists only if $A-\lambda I$ is singular, i.e.</p>
  $$\det(A-\lambda I)=0\qquad\text{(the characteristic equation).}$$
  <p>Two handy checks: the eigenvalues sum to the trace, $\sum_i\lambda_i=\operatorname{tr}(A)$, and multiply to the
  determinant, $\prod_i\lambda_i=\det(A)$.</p>`,

  derivation: String.raw`<p><strong>Find the eigenpairs of $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ from scratch.</strong></p>
  <p><strong>Step 1 — form $A-\lambda I$:</strong> $\begin{bmatrix}2-\lambda&1\\1&2-\lambda\end{bmatrix}$.</p>
  <p><strong>Step 2 — set its determinant to zero:</strong>
  $\det=(2-\lambda)^2-(1)(1)=\lambda^2-4\lambda+3=0.$</p>
  <p><strong>Step 3 — solve the characteristic equation:</strong> $(\lambda-1)(\lambda-3)=0$, so $\lambda=1$ and
  $\lambda=3$. (Check: they sum to $4=\operatorname{tr}A$ and multiply to $3=\det A$.)</p>
  <p><strong>Step 4 — eigenvector for $\lambda=3$:</strong> solve $(A-3I)\mathbf v=\mathbf 0$:
  $\begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf v=\mathbf 0\Rightarrow -v_1+v_2=0\Rightarrow \mathbf v\propto[1,1].$</p>
  <p><strong>Step 5 — eigenvector for $\lambda=1$:</strong>
  $\begin{bmatrix}1&1\\1&1\end{bmatrix}\mathbf v=\mathbf 0\Rightarrow v_1+v_2=0\Rightarrow \mathbf v\propto[1,-1].$
  $\blacksquare$ Notice the two eigenvectors are orthogonal — no accident for a symmetric matrix (Deep Dive). Plain
  English: along $[1,1]$ the matrix triples; along $[1,-1]$ it leaves length unchanged.</p>`,

  code: [
    { label: "Eigenvalues and a verification of Av = λv", src: String.raw`
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eig(A)
print("eigenvalues :", np.round(vals, 4))      # [3. 1.] (order may vary)

for i in range(2):
    lam, v = vals[i], vecs[:, i]
    print(f"λ={lam:.1f}:  Av =", np.round(A@v,4), "  λv =", np.round(lam*v,4))
print("trace == sum(λ)? ", np.isclose(np.trace(A), vals.sum()))
print("det   == prod(λ)?", np.isclose(np.linalg.det(A), vals.prod()))
` },
    { label: "Power iteration finds the top eigenvector", src: String.raw`
import numpy as np
A = np.array([[2.0,1.0],[1.0,2.0]])

v = np.random.default_rng(0).normal(size=2)
for _ in range(50):
    v = A @ v
    v = v / np.linalg.norm(v)         # repeatedly apply A and renormalize
lam = (v @ A @ v)                     # Rayleigh quotient ≈ top eigenvalue
print("dominant eigenvector ≈", np.round(v,3), " (∝ [1,1])")
print("dominant eigenvalue  ≈", round(float(lam),3))   # ≈ 3
` }
  ],

  keyPoints: [
    "$A\\mathbf v=\\lambda\\mathbf v$: an eigenvector keeps its direction; $\\lambda$ is the stretch factor.",
    "Eigenvalues solve the characteristic equation $\\det(A-\\lambda I)=0$.",
    "$\\sum\\lambda_i=\\operatorname{tr}(A)$ and $\\prod\\lambda_i=\\det(A)$ — quick sanity checks.",
    "Hessian eigenvalues are loss curvature: positive = bowl, negative = saddle direction.",
    "Eigenvectors are defined only up to scale (and sign)."
  ],

  commonMistakes: [
    { wrong: "Thinking an eigenvector is a single unique vector.", why: "Any nonzero multiple of an eigenvector is also one — they define a <em>direction</em>/line, not a point. Libraries return unit-length representatives, but the sign is arbitrary." },
    { wrong: "Assuming every matrix has real eigenvalues.", why: "A rotation matrix has complex eigenvalues — it turns every real vector, so no real direction is preserved. Real eigenvalues are guaranteed only for symmetric matrices." },
    { wrong: "Reading a near-zero eigenvalue as harmless.", why: "A tiny Hessian eigenvalue means an almost-flat direction — slow learning and a near-singular, ill-conditioned problem. Zero/negative eigenvalues are exactly the saddle structure of deep nets." }
  ],

  quiz: [
    { q: "Eigenvalues of $\\begin{bmatrix}3&0\\\\0&5\\end{bmatrix}$?", options: ["3 and 5", "0 and 0", "8 and 15", "4 and 4"], answer: 0,
      explain: "A diagonal matrix has its diagonal entries as eigenvalues, with eigenvectors the coordinate axes." },
    { q: "For $A=\\begin{bmatrix}2&1\\\\1&2\\end{bmatrix}$, an eigenvector for $\\lambda=3$ is…", options: ["$[1,1]$", "$[1,-1]$", "$[1,0]$", "$[0,1]$"], answer: 0,
      explain: "$(A-3I)[1,1]^\\top=[-1+1,\\,1-1]=\\mathbf 0$. $[1,-1]$ is the $\\lambda=1$ eigenvector." },
    { q: "$\\operatorname{tr}(A)=10$ and one eigenvalue of the $2\\times2$ matrix is $4$. The other is…", options: ["6", "4", "14", "2.5"], answer: 0,
      explain: "Eigenvalues sum to the trace: $4+\\lambda_2=10\\Rightarrow\\lambda_2=6$." },
    { q: "A Hessian has eigenvalues $\\{5,-2\\}$ at a critical point. That point is a…", options: ["saddle", "minimum", "maximum", "flat region"], answer: 0,
      explain: "Mixed signs mean up in one direction, down in another — a saddle. All-positive would be a minimum, all-negative a maximum." },
    { q: "Which matrix is guaranteed real eigenvalues?", options: ["a symmetric matrix", "a rotation matrix", "any square matrix", "any invertible matrix"], answer: 0,
      explain: "The spectral theorem guarantees real eigenvalues (and orthogonal eigenvectors) for symmetric matrices; rotations have complex ones." }
  ],

  practice: [
    { level: "easy", prompt: "Find the eigenvalues of $\\begin{bmatrix}4&0\\\\0&-1\\end{bmatrix}$.", solution: "Diagonal ⇒ eigenvalues are $4$ and $-1$; eigenvectors $[1,0]$ and $[0,1]$." },
    { level: "easy", prompt: "If $A\\mathbf v=\\lambda\\mathbf v$, what is $A(3\\mathbf v)$?", solution: "$A(3\\mathbf v)=3A\\mathbf v=3\\lambda\\mathbf v=\\lambda(3\\mathbf v)$ — still an eigenvector with the same $\\lambda$. Eigenvectors are scale-free." },
    { level: "med", prompt: "Compute the eigenvalues of $\\begin{bmatrix}0&1\\\\1&0\\end{bmatrix}$ and interpret geometrically.", solution: "$\\det\\begin{bmatrix}-\\lambda&1\\\\1&-\\lambda\\end{bmatrix}=\\lambda^2-1=0\\Rightarrow\\lambda=\\pm1$. This matrix swaps coordinates: along $[1,1]$ it does nothing ($\\lambda=1$); along $[1,-1]$ it flips sign ($\\lambda=-1$)." },
    { level: "hard", prompt: "AI task: gradient descent on a quadratic loss has Hessian eigenvalues $\\lambda_{\\max}=20$, $\\lambda_{\\min}=0.5$. What learning rate range converges, and what does the ratio tell you?", solution: "Stability needs $0\\lt\\eta\\lt 2/\\lambda_{\\max}=0.1$. But progress along the flat direction is governed by $\\eta\\lambda_{\\min}$, which is tiny, so convergence speed is set by the condition number $\\kappa=\\lambda_{\\max}/\\lambda_{\\min}=40$: the bigger $\\kappa$, the more zig-zag and the slower the crawl along the shallow valley. This is exactly why preconditioning / Adam (Track 8) help — they flatten that ratio." }
  ],

  deepDive: String.raw`<p><strong>Why symmetric matrices are the well-behaved ones (the spectral theorem).</strong></p>
  <p>Covariance matrices, Hessians, Gram matrices $A^\top A$ — the matrices we actually meet in ML — are almost always
  <strong>symmetric</strong>. That's a gift, because the spectral theorem guarantees a symmetric real matrix has (1)
  all <em>real</em> eigenvalues and (2) a full set of <em>orthogonal</em> eigenvectors. You saw it concretely above:
  $[1,1]\perp[1,-1]$.</p>
  <p>Geometrically this means a symmetric matrix acts as a pure stretch along perpendicular axes — no rotation, no
  shear, once you look in the eigenvector frame. That is the whole basis of PCA (the eigenvectors of the covariance are
  orthogonal principal axes, Lesson 5.5) and of understanding the loss surface: near a minimum the Hessian's orthogonal
  eigen-directions are independent "valleys," each with its own curvature $\lambda_i$. Diagonalizing in that frame
  (Lesson 5.2) is what turns a coupled, messy optimization into a set of clean, independent 1-D problems — and it's why
  eigenvalues, not raw matrix entries, are the right vocabulary for curvature.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["5.2"] = {
  subtitle: "Change to the eigenbasis and a matrix becomes pure, independent stretches.",

  aiMoment: String.raw`<p>A covariance matrix is symmetric, and PCA works by <strong>diagonalizing</strong> it — rotating
  to the basis where it becomes a simple list of variances along perpendicular axes. The same trick computes matrix
  powers (the steady state of a Markov chain, repeated linear dynamics) and decouples a tangled system into independent
  one-dimensional modes. "Diagonalize it" is one of the most powerful moves in applied linear algebra.</p>`,

  plainEnglish: String.raw`<p>To <strong>diagonalize</strong> a matrix is to switch to the coordinate system of its
  eigenvectors, where it stops mixing dimensions and just scales each axis by an eigenvalue. <strong>Symmetric</strong>
  matrices are the friendliest: their eigenvectors are perpendicular and their eigenvalues are real, so the new axes are
  a clean rotation of the old ones.</p>`,

  intuition: String.raw`<p>In the eigenbasis a matrix is a diagonal of stretch factors. So acting with $A$ is really
  three steps: rotate into the eigenbasis, stretch each axis, rotate back — exactly what $A=PDP^{-1}$ says.</p>
  <figure class="figure">
  <svg viewBox="0 0 390 90" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A equals P D P inverse as rotate, scale, rotate back">
    <g font-family="sans-serif" font-size="12">
    <text x="8" y="50" fill="#20242c">x</text>
    <line x1="20" y1="46" x2="56" y2="46" stroke="#94a3b8" marker-end="url(#s1)"/>
    <rect x="58" y="28" width="64" height="36" rx="6" fill="#f7f3ff" stroke="#7c3aed"/>
    <text x="66" y="51" fill="#7c3aed">P⁻¹ rotate</text>
    <line x1="122" y1="46" x2="150" y2="46" stroke="#94a3b8" marker-end="url(#s1)"/>
    <rect x="152" y="28" width="62" height="36" rx="6" fill="#fff7ed" stroke="#d97706"/>
    <text x="162" y="51" fill="#d97706">D scale</text>
    <line x1="214" y1="46" x2="242" y2="46" stroke="#94a3b8" marker-end="url(#s1)"/>
    <rect x="244" y="28" width="74" height="36" rx="6" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="252" y="51" fill="#0d9488">P rotate back</text>
    <line x1="318" y1="46" x2="346" y2="46" stroke="#94a3b8" marker-end="url(#s1)"/>
    <text x="352" y="50" fill="#20242c">Ax</text>
    </g>
    <defs><marker id="s1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>A = P D P⁻¹: rotate into the eigenbasis, scale each axis by an eigenvalue, rotate back.</figcaption>
  </figure>`,

  formalism: String.raw`<p>If $A$ ($n\times n$) has $n$ independent eigenvectors, collect them as columns of $P$ and the
  eigenvalues on the diagonal of $D$. Then</p>
  $$A=PDP^{-1},\qquad A^{k}=PD^{k}P^{-1}.$$
  <p>For a <strong>symmetric</strong> $A=A^\top$, the eigenvectors can be chosen orthonormal, so $P$ becomes an
  orthogonal $Q$ ($Q^{-1}=Q^\top$) and $A=Q\Lambda Q^\top$ — the <strong>spectral theorem</strong>. Plain English:
  symmetric matrices diagonalize with a pure rotation.</p>`,

  derivation: String.raw`<p><strong>Part 1 — why $A=PDP^{-1}$.</strong> Stack the eigen-equations $A\mathbf v_i=\lambda_i\mathbf v_i$
  as columns: $A[\mathbf v_1\,\cdots\,\mathbf v_n]=[\lambda_1\mathbf v_1\,\cdots\,\lambda_n\mathbf v_n]=[\mathbf v_1\,\cdots\,\mathbf v_n]\,\mathrm{diag}(\lambda_i).$
  That is $AP=PD$. If the eigenvectors are independent, $P$ is invertible, so $A=PDP^{-1}.\;\blacksquare$ Powers follow:
  $A^2=PDP^{-1}PDP^{-1}=PD^2P^{-1}$, and by induction $A^k=PD^kP^{-1}$ — raise the diagonal, that's all.</p>
  <hr class="soft">
  <p><strong>Part 2 — symmetric ⇒ orthogonal eigenvectors.</strong> Take eigenpairs $(\lambda_i,\mathbf v_i)$,
  $(\lambda_j,\mathbf v_j)$ with $\lambda_i\neq\lambda_j$.</p>
  <p><strong>Step 1.</strong> Compute $\mathbf v_i^\top A\mathbf v_j$ two ways. Using $A\mathbf v_j=\lambda_j\mathbf v_j$:
  $\mathbf v_i^\top A\mathbf v_j=\lambda_j\,\mathbf v_i^\top\mathbf v_j$.</p>
  <p><strong>Step 2.</strong> Using symmetry $A=A^\top$ and $A\mathbf v_i=\lambda_i\mathbf v_i$:
  $\mathbf v_i^\top A\mathbf v_j=(A\mathbf v_i)^\top\mathbf v_j=\lambda_i\,\mathbf v_i^\top\mathbf v_j$.</p>
  <p><strong>Step 3.</strong> Subtract: $(\lambda_i-\lambda_j)\,\mathbf v_i^\top\mathbf v_j=0$. Since $\lambda_i\neq\lambda_j$,
  we must have $\mathbf v_i^\top\mathbf v_j=0$ — the eigenvectors are orthogonal. $\blacksquare$ Plain English: for a
  symmetric matrix, different eigen-directions are automatically perpendicular, which is why PCA's axes come out
  orthogonal for free.</p>`,

  code: [
    { label: "Diagonalize, reconstruct, and take a power", src: String.raw`
import numpy as np

A = np.array([[2., 1.],
              [1., 2.]])
vals, P = np.linalg.eig(A)
D = np.diag(vals)
print("P D P^-1 == A ?", np.allclose(P @ D @ np.linalg.inv(P), A))

# A^5 the easy way: raise the eigenvalues
A5 = P @ np.diag(vals**5) @ np.linalg.inv(P)
print("A^5 via eig == A@A@A@A@A ?", np.allclose(A5, np.linalg.matrix_power(A, 5)))
` },
    { label: "Symmetric matrices: orthogonal eigenvectors", src: String.raw`
import numpy as np

S = np.array([[2., 1.],
              [1., 2.]])
vals, Q = np.linalg.eigh(S)        # eigh for symmetric: returns orthonormal Q
print("eigenvalues:", np.round(vals, 4))      # [1. 3.]
print("Q^T Q = I ? ", np.allclose(Q.T @ Q, np.eye(2)))   # orthonormal
print("Q Λ Q^T == S?", np.allclose(Q @ np.diag(vals) @ Q.T, S))
` }
  ],

  keyPoints: [
    "Diagonalizing = changing to the eigenbasis where $A$ is a diagonal of stretch factors.",
    "$A=PDP^{-1}$ (eigenvectors in $P$, eigenvalues in $D$); $A^k=PD^kP^{-1}$.",
    "Needs $n$ independent eigenvectors; otherwise the matrix is 'defective' (not diagonalizable).",
    "Symmetric ⇒ spectral theorem: $A=Q\\Lambda Q^\\top$ with orthonormal $Q$, real eigenvalues.",
    "Different eigenvalues of a symmetric matrix have orthogonal eigenvectors."
  ],

  commonMistakes: [
    { wrong: "Assuming every matrix is diagonalizable.", why: "A 'defective' matrix like $\\begin{bmatrix}1&1\\\\0&1\\end{bmatrix}$ lacks a full set of independent eigenvectors and cannot be written $PDP^{-1}$ (you need the Jordan form or SVD)." },
    { wrong: "Computing $A^k$ by repeated multiplication when eigen is available.", why: "$PD^kP^{-1}$ raises only the diagonal — far cheaper and the way to find steady states / limits as $k\\to\\infty$." },
    { wrong: "Using <code>eig</code> for a symmetric matrix.", why: "<code>eigh</code> exploits symmetry: it's faster, more accurate, returns sorted real eigenvalues and genuinely orthonormal eigenvectors." }
  ],

  quiz: [
    { q: "If $A=PDP^{-1}$ with $D=\\mathrm{diag}(2,3)$, what is $A^{3}$ similar to?", options: ["$\\mathrm{diag}(8,27)$", "$\\mathrm{diag}(6,9)$", "$\\mathrm{diag}(2,3)$", "$\\mathrm{diag}(5,6)$"], answer: 0,
      explain: "$A^3=PD^3P^{-1}$ and $D^3=\\mathrm{diag}(2^3,3^3)=\\mathrm{diag}(8,27)$." },
    { q: "For a symmetric matrix, $P^{-1}$ equals…", options: ["$P^\\top$", "$P$", "$-P$", "$D$"], answer: 0,
      explain: "Symmetric ⇒ orthonormal eigenvectors ⇒ $P=Q$ orthogonal ⇒ $Q^{-1}=Q^\\top$." },
    { q: "Which matrix is NOT diagonalizable?", options: ["$\\begin{bmatrix}1&1\\\\0&1\\end{bmatrix}$", "$\\begin{bmatrix}2&0\\\\0&3\\end{bmatrix}$", "$\\begin{bmatrix}0&1\\\\1&0\\end{bmatrix}$", "$\\begin{bmatrix}2&1\\\\1&2\\end{bmatrix}$"], answer: 0,
      explain: "The shear has a repeated eigenvalue 1 but only one independent eigenvector — defective, not diagonalizable. The others are fine (two are symmetric)." },
    { q: "Eigenvectors of a symmetric matrix with distinct eigenvalues are…", options: ["orthogonal", "parallel", "complex", "equal"], answer: 0,
      explain: "The subtraction argument forces $\\mathbf v_i^\\top\\mathbf v_j=0$ when $\\lambda_i\\neq\\lambda_j$." },
    { q: "Why is diagonalization useful for a Markov chain's long-run behavior?", options: ["$A^k=PD^kP^{-1}$ makes the $k\\to\\infty$ limit easy", "it makes $A$ symmetric", "it removes eigenvalues", "it lowers the rank"], answer: 0,
      explain: "Raising eigenvalues to the $k$ shows which modes survive (eigenvalue 1) and which decay ($|\\lambda|<1$), giving the steady state directly." }
  ],

  practice: [
    { level: "easy", prompt: "Diagonalize $\\begin{bmatrix}3&0\\\\0&5\\end{bmatrix}$.", solution: "Already diagonal: $P=I$, $D=\\begin{bmatrix}3&0\\\\0&5\\end{bmatrix}$. Eigenvectors are the axes." },
    { level: "med", prompt: "Using $A=\\begin{bmatrix}2&1\\\\1&2\\end{bmatrix}$ (eigenvalues 1, 3; eigenvectors $[1,-1],[1,1]$), write $A^{10}$ as $P D^{10}P^{-1}$ conceptually and give the eigenvalues of $A^{10}$.", solution: "$A^{10}=P\\,\\mathrm{diag}(1^{10},3^{10})\\,P^{-1}=P\\,\\mathrm{diag}(1,59049)\\,P^{-1}$. The eigenvectors are unchanged; only the eigenvalues are raised to the 10th power." },
    { level: "med", prompt: "Show that if $A=Q\\Lambda Q^\\top$ (symmetric), then $A^{-1}=Q\\Lambda^{-1}Q^\\top$ when all $\\lambda_i\\neq0$.", solution: "$Q\\Lambda Q^\\top\\cdot Q\\Lambda^{-1}Q^\\top=Q\\Lambda(Q^\\top Q)\\Lambda^{-1}Q^\\top=Q\\Lambda\\Lambda^{-1}Q^\\top=QQ^\\top=I$. So inverting a symmetric matrix is just inverting its eigenvalues in the eigenbasis." },
    { level: "hard", prompt: "AI task: near a minimum the loss is $\\approx\\tfrac12(\\mathbf w-\\mathbf w^*)^\\top H(\\mathbf w-\\mathbf w^*)$ with symmetric Hessian $H=Q\\Lambda Q^\\top$. Explain why gradient descent decouples into independent 1-D problems in the eigenbasis.", solution: "Substitute coordinates $\\mathbf z=Q^\\top(\\mathbf w-\\mathbf w^*)$. Then the loss becomes $\\tfrac12\\sum_i\\lambda_i z_i^2$ — a sum of independent 1-D parabolas, one per eigen-direction. Gradient descent updates each $z_i$ separately as $z_i\\leftarrow(1-\\eta\\lambda_i)z_i$, converging at rate $|1-\\eta\\lambda_i|$. Convergence is fast where $\\lambda_i$ is moderate and crawls where $\\lambda_i$ is tiny — the per-axis view that explains the condition-number slowdown and motivates Adam's per-coordinate scaling (Track 8)." }
  ],

  deepDive: String.raw`<p><strong>The spectral theorem, and why real symmetric eigenvalues are real.</strong></p>
  <p>We used "symmetric ⇒ real eigenvalues" above; here's the proof. Let $A$ be real symmetric with eigenpair
  $(\lambda,\mathbf v)$, allowing complex $\mathbf v$. Consider the scalar $\bar{\mathbf v}^\top A\mathbf v$, where the
  bar is complex conjugate. On one hand $A\mathbf v=\lambda\mathbf v$ gives $\bar{\mathbf v}^\top A\mathbf v=\lambda\,\bar{\mathbf v}^\top\mathbf v=\lambda\lVert\mathbf v\rVert^2$.
  On the other hand, this scalar equals its own conjugate: $(\bar{\mathbf v}^\top A\mathbf v)^{*}=\mathbf v^\top A^\top\bar{\mathbf v}=\mathbf v^\top A\bar{\mathbf v}=\bar{\mathbf v}^\top A\mathbf v$
  (using $A^\top=A$, real). A number equal to its conjugate is real, and $\lVert\mathbf v\rVert^2>0$ is real, so
  $\lambda$ is real. $\blacksquare$</p>
  <p>Combined with the orthogonality proof from the derivation, this is the full spectral theorem: every real symmetric
  matrix factors as $Q\Lambda Q^\top$ with real $\Lambda$ and orthonormal $Q$. It's the quiet workhorse behind PCA,
  Gaussian covariances, kernel methods, and second-order optimization — every time you see a covariance, a Gram matrix
  $A^\top A$, or a Hessian, the spectral theorem guarantees a clean orthogonal eigenbasis to think in. The next lesson
  adds one more condition — positive eigenvalues — to capture "this matrix is a bowl."</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["5.3"] = {
  subtitle: "The matrices that curve upward everywhere — bowls, not saddles.",

  aiMoment: String.raw`<p>A critical point is a true <strong>minimum</strong> exactly when the Hessian is
  <strong>positive definite</strong> — the loss curves up in every direction. Covariance matrices and kernel (Gram)
  matrices are positive semidefinite by construction; Newton's method needs a positive-definite Hessian to point
  downhill; and the Cholesky factorization of a PD covariance is how you sample from a multivariate Gaussian (Track 10).
  "Is this matrix positive definite?" is the question behind a lot of optimization and probability.</p>`,

  plainEnglish: String.raw`<p>A symmetric matrix is <strong>positive definite</strong> if the quadratic it defines always
  comes out positive: $\mathbf x^\top A\mathbf x\gt0$ for every nonzero $\mathbf x$. Geometrically that's a bowl that
  curves up no matter which way you walk. The cleanest test: all its eigenvalues are positive.</p>`,

  intuition: String.raw`<p>The function $\mathbf x^\top A\mathbf x$ is a surface. If $A$ is positive definite it's a bowl
  with a single lowest point; if eigenvalues have mixed signs it's a saddle — up one way, down another.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Positive-definite bowl versus indefinite saddle contours">
    <g>
    <ellipse cx="80" cy="75" rx="60" ry="40" fill="none" stroke="#0d9488"/>
    <ellipse cx="80" cy="75" rx="40" ry="26" fill="none" stroke="#0d9488"/>
    <ellipse cx="80" cy="75" rx="20" ry="13" fill="none" stroke="#0d9488"/>
    <circle cx="80" cy="75" r="3" fill="#0d9488"/>
    <text x="34" y="135" font-size="11" fill="#0d9488" font-family="sans-serif">PD: bowl, xᵀAx&gt;0</text>
    </g>
    <g transform="translate(180,0)">
    <path d="M20,30 Q60,75 20,120" fill="none" stroke="#dc2626"/>
    <path d="M100,30 Q60,75 100,120" fill="none" stroke="#dc2626"/>
    <path d="M20,40 Q60,75 100,40" fill="none" stroke="#7c3aed"/>
    <path d="M20,110 Q60,75 100,110" fill="none" stroke="#7c3aed"/>
    <text x="20" y="135" font-size="11" fill="#dc2626" font-family="sans-serif">indefinite: saddle</text>
    </g>
  </svg>
  <figcaption>Positive definite = concentric bowl (one minimum). Mixed-sign eigenvalues = saddle.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A symmetric $A$ is <strong>positive definite (PD)</strong> if $\mathbf x^\top A\mathbf x\gt0$ for
  all $\mathbf x\neq\mathbf 0$, and <strong>positive semidefinite (PSD)</strong> if $\ge0$. Equivalent tests:
  (1) all eigenvalues $\lambda_i\gt0$ (PD) or $\ge0$ (PSD); (2) all leading principal minors $\gt0$ (Sylvester's
  criterion); (3) a Cholesky factorization $A=LL^\top$ exists with positive diagonal. Any $A^\top A$ is automatically
  PSD. Plain English: PD means the quadratic bowl opens strictly upward in every direction.</p>`,

  derivation: String.raw`<p><strong>Positive definite ⇔ all eigenvalues positive.</strong> Use the spectral theorem
  $A=Q\Lambda Q^\top$ (symmetric, Lesson 5.2).</p>
  <p><strong>Step 1 — rewrite the quadratic in the eigenbasis.</strong> Let $\mathbf y=Q^\top\mathbf x$ (a rotation, so
  $\mathbf y\neq\mathbf 0$ whenever $\mathbf x\neq\mathbf 0$). Then
  $\mathbf x^\top A\mathbf x=\mathbf x^\top Q\Lambda Q^\top\mathbf x=\mathbf y^\top\Lambda\mathbf y=\sum_i\lambda_i y_i^2.$</p>
  <p><strong>Step 2 — read off the sign.</strong> This is a weighted sum of squares. If every $\lambda_i\gt0$, the sum
  is $\gt0$ for any nonzero $\mathbf y$ → PD. Conversely, if some $\lambda_k\le0$, pick $\mathbf y=\mathbf e_k$ (i.e.
  $\mathbf x=Q\mathbf e_k=\mathbf q_k$): then $\mathbf x^\top A\mathbf x=\lambda_k\le0$, so not PD.</p>
  <p><strong>Step 3 — conclude.</strong> $A\succ0\iff$ all $\lambda_i\gt0$. $\blacksquare$ Plain English: rotating to the
  eigenbasis turns the quadratic into $\sum\lambda_i y_i^2$, which is positive everywhere exactly when every curvature
  $\lambda_i$ is positive.</p>`,

  code: [
    { label: "Test positive-definiteness; Cholesky", src: String.raw`
import numpy as np

A = np.array([[2., 1.],
              [1., 2.]])
vals = np.linalg.eigvalsh(A)
print("eigenvalues:", vals, " -> PD?", np.all(vals > 0))   # [1,3] -> True

L = np.linalg.cholesky(A)         # succeeds only if PD
print("L L^T == A ?", np.allclose(L @ L.T, A))

B = np.array([[1., 2.],
              [2., 1.]])           # eigenvalues 3, -1  -> indefinite (saddle)
print("B eigenvalues:", np.linalg.eigvalsh(B))
` },
    { label: "A^T A is always PSD; quadratic form stays >= 0", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

M = rng.normal(size=(3, 2))
G = M.T @ M                         # Gram matrix -> PSD
print("eigenvalues of M^T M:", np.round(np.linalg.eigvalsh(G), 4), ">= 0")

worst = min((x @ G @ x) for x in rng.normal(size=(2000, 2)))
print("min x^T (M^T M) x over random x:", round(float(worst), 4), ">= 0")
` }
  ],

  keyPoints: [
    "Positive definite: $\\mathbf x^\\top A\\mathbf x\\gt0$ for all $\\mathbf x\\neq\\mathbf 0$ — a strict bowl.",
    "PD ⇔ all eigenvalues positive; PSD ⇔ all eigenvalues $\\ge0$.",
    "A PD Hessian marks a local minimum; mixed signs mark a saddle.",
    "Any $A^\\top A$ (Gram matrix) and any covariance is PSD.",
    "PD matrices have a Cholesky factor $A=LL^\\top$ — used to sample Gaussians."
  ],

  commonMistakes: [
    { wrong: "Checking only the diagonal entries for positive-definiteness.", why: "Positive diagonal is necessary but not sufficient. $\\begin{bmatrix}1&2\\\\2&1\\end{bmatrix}$ has positive diagonal yet an eigenvalue $-1$ — it's indefinite." },
    { wrong: "Confusing positive definite with 'all entries positive'.", why: "PD is about $\\mathbf x^\\top A\\mathbf x$, not the entries. A PD matrix can have negative off-diagonal entries, and an all-positive matrix can fail to be PD." },
    { wrong: "Assuming a symmetric matrix is automatically PD.", why: "Symmetry only guarantees real eigenvalues; their <em>signs</em> determine definiteness. You still must check $\\lambda_i\\gt0$." }
  ],

  quiz: [
    { q: "$\\begin{bmatrix}3&0\\\\0&-2\\end{bmatrix}$ is…", options: ["indefinite (a saddle)", "positive definite", "positive semidefinite", "negative definite"], answer: 0,
      explain: "Eigenvalues $3$ and $-2$ have mixed signs → indefinite. $\\mathbf x^\\top A\\mathbf x$ is positive along $x$, negative along $y$." },
    { q: "Which guarantees a matrix is PSD?", options: ["it equals $M^\\top M$ for some $M$", "it has positive trace", "it is symmetric", "its determinant is positive"], answer: 0,
      explain: "$\\mathbf x^\\top M^\\top M\\mathbf x=\\lVert M\\mathbf x\\rVert^2\\ge0$ always. Positive trace/determinant don't rule out a mixed-sign pair." },
    { q: "A Hessian with eigenvalues $\\{4,0.1,2\\}$ at a critical point indicates a…", options: ["local minimum", "saddle", "local maximum", "plateau only"], answer: 0,
      explain: "All eigenvalues positive ⇒ positive definite ⇒ curves up in every direction ⇒ local minimum." },
    { q: "Cholesky $A=LL^\\top$ succeeds iff $A$ is…", options: ["positive definite", "orthogonal", "singular", "upper triangular"], answer: 0,
      explain: "A real Cholesky factor with positive diagonal exists exactly for symmetric positive-definite matrices — which is how libraries test PD." },
    { q: "If $A$ is PD, what about $A^{-1}$?", options: ["also PD", "indefinite", "PSD but not PD", "not invertible"], answer: 0,
      explain: "PD means all $\\lambda_i\\gt0$; $A^{-1}$ has eigenvalues $1/\\lambda_i\\gt0$, so it's PD too." }
  ],

  practice: [
    { level: "easy", prompt: "Is $\\begin{bmatrix}5&0\\\\0&2\\end{bmatrix}$ positive definite?", solution: "Eigenvalues are the diagonal $5,2$, both positive → yes, PD." },
    { level: "med", prompt: "Determine the definiteness of $\\begin{bmatrix}2&2\\\\2&2\\end{bmatrix}$.", solution: "Eigenvalues: trace $4$, det $0$, so $\\lambda=4,0$. All $\\ge0$ but one is $0$ → positive semidefinite (not PD). It has a null direction $[1,-1]$." },
    { level: "med", prompt: "Show $\\mathbf x^\\top(M^\\top M)\\mathbf x\\ge0$ for any $M$.", solution: "$\\mathbf x^\\top M^\\top M\\mathbf x=(M\\mathbf x)^\\top(M\\mathbf x)=\\lVert M\\mathbf x\\rVert^2\\ge0$. It's a squared length, hence nonnegative — so every Gram matrix is PSD." },
    { level: "hard", prompt: "AI task: Newton's method updates $\\mathbf w\\leftarrow\\mathbf w-H^{-1}\\nabla\\mathcal L$. Why must $H$ be positive definite for this to be a descent step, and what's done when it isn't?", solution: "The step direction is $-H^{-1}\\mathbf g$. Its dot with the gradient is $-\\mathbf g^\\top H^{-1}\\mathbf g$; this is negative (a true descent direction) exactly when $H^{-1}$ — equivalently $H$ — is positive definite, since then $\\mathbf g^\\top H^{-1}\\mathbf g\\gt0$. If $H$ is indefinite (near a saddle), the Newton step can point <em>uphill</em>. Fixes: add $\\tau I$ to make $H+\\tau I$ PD (damped/Levenberg–Marquardt), use a trust region, or use a PSD approximation like Gauss–Newton ($J^\\top J$, always PSD)." }
  ],

  deepDive: String.raw`<p><strong>Positive-definiteness is the bridge from linear algebra to convex optimization.</strong></p>
  <p>A twice-differentiable function is <strong>convex</strong> exactly when its Hessian is positive semidefinite
  everywhere, and <strong>strictly convex</strong> when it's positive definite (Track 8). That's why PD matters so much:
  it's the local certificate that you're in a bowl, so a critical point is a genuine minimum and there's nothing better
  hiding nearby. For a quadratic loss $\tfrac12\mathbf w^\top A\mathbf w-\mathbf b^\top\mathbf w$, PD-ness of $A$ is what
  guarantees a unique global minimum at $A^{-1}\mathbf b$ — and it's exactly the full-rank condition that made the normal
  equations solvable in Track 3.</p>
  <p>The concept threads through ML: kernels must be PSD for the kernel trick to correspond to a real inner product
  (Mercer's condition); covariance matrices are PSD so that every variance $\mathbf a^\top\Sigma\mathbf a\ge0$; Gaussian
  log-densities need a PD covariance to be well-defined; and second-order optimizers spend real effort keeping their
  curvature estimates PD so their steps point downhill. Positive-definiteness is how "curves upward" becomes a checkable
  matrix property — and the SVD in the next lesson will let us handle the matrices that aren't even square.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["5.4"] = {
  subtitle: "Every matrix, of any shape, is a rotation then a stretch then a rotation.",

  aiMoment: String.raw`<p>The <strong>SVD</strong> is the most useful matrix factorization in all of ML. It powers
  recommendation systems (users × items factored into latent tastes), latent semantic analysis, image and model
  compression, the robust pseudo-inverse (Track 4), and it's the precise math behind low-rank adapters like LoRA. Where
  eigen-decomposition only works for square matrices, SVD works for <em>any</em> matrix — which is why it's everywhere.</p>`,

  plainEnglish: String.raw`<p>The SVD says any matrix, even a non-square one, does exactly three things to space: a
  rotation, then a stretch along perpendicular axes, then another rotation. The three pieces are $U$ (output rotation),
  $\Sigma$ (the stretch factors, called singular values), and $V^\top$ (input rotation).</p>`,

  intuition: String.raw`<p>Feed the SVD an orthonormal set of input directions (the columns of $V$). It stretches each
  by a singular value $\sigma_i$ and sends it to an orthonormal output direction (a column of $U$). A unit circle of
  inputs becomes an ellipse whose semi-axes are the $\sigma_i$.</p>
  <figure class="figure">
  <svg viewBox="0 0 330 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SVD maps orthonormal v's to stretched orthonormal u's">
    <circle cx="70" cy="78" r="38" fill="#eef0ff" stroke="#4f46e5" stroke-width="1.6"/>
    <line x1="70" y1="78" x2="108" y2="78" stroke="#4f46e5" stroke-width="2.4" marker-end="url(#v1)"/>
    <line x1="70" y1="78" x2="70" y2="40" stroke="#0d9488" stroke-width="2.4" marker-end="url(#v2)"/>
    <text x="92" y="94" font-size="11" fill="#4f46e5" font-family="sans-serif">v₁</text>
    <text x="52" y="46" font-size="11" fill="#0d9488" font-family="sans-serif">v₂</text>
    <line x1="120" y1="78" x2="168" y2="78" stroke="#64748b" stroke-width="1.6" marker-end="url(#v3)"/>
    <text x="132" y="70" font-size="11" fill="#64748b" font-family="sans-serif">A</text>
    <ellipse cx="252" cy="78" rx="56" ry="26" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.6" transform="rotate(-18 252 78)"/>
    <line x1="252" y1="78" x2="305" y2="61" stroke="#4f46e5" stroke-width="2.6" marker-end="url(#v4)"/>
    <line x1="252" y1="78" x2="244" y2="53" stroke="#0d9488" stroke-width="2.6" marker-end="url(#v5)"/>
    <text x="290" y="54" font-size="11" fill="#4f46e5" font-family="sans-serif">σ₁u₁</text>
    <text x="214" y="52" font-size="11" fill="#0d9488" font-family="sans-serif">σ₂u₂</text>
    <defs>
      <marker id="v1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="v2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#0d9488"/></marker>
      <marker id="v3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#64748b"/></marker>
      <marker id="v4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="v5" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#0d9488"/></marker>
    </defs>
  </svg>
  <figcaption>Orthonormal inputs vᵢ map to orthonormal outputs uᵢ, stretched by σᵢ — the circle becomes an ellipse.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Every $A\in\mathbb{R}^{m\times n}$ factors as</p>
  $$A=U\Sigma V^\top,$$
  <p>where $U\in\mathbb{R}^{m\times m}$ and $V\in\mathbb{R}^{n\times n}$ are orthogonal ($U^\top U=I$, $V^\top V=I$) and
  $\Sigma$ is diagonal with <strong>singular values</strong> $\sigma_1\ge\sigma_2\ge\dots\ge0$. The columns of $V$ are
  <em>right</em> singular vectors, the columns of $U$ are <em>left</em> singular vectors, and $A\mathbf v_i=\sigma_i\mathbf u_i$.
  The number of nonzero $\sigma_i$ is the rank. Relation to eigen: $\mathbf v_i$ are eigenvectors of $A^\top A$ and
  $\sigma_i=\sqrt{\lambda_i(A^\top A)}$.</p>`,

  derivation: String.raw`<p><strong>Construct the SVD from $A^\top A$.</strong></p>
  <p><strong>Step 1 — diagonalize $A^\top A$.</strong> It's symmetric PSD (Lesson 5.3), so by the spectral theorem
  $A^\top A=V\Lambda V^\top$ with orthonormal columns $\mathbf v_i$ and eigenvalues $\lambda_i\ge0$. Define singular
  values $\sigma_i=\sqrt{\lambda_i}$, ordered largest first.</p>
  <p><strong>Step 2 — define the output directions.</strong> For each $\sigma_i\gt0$ set $\mathbf u_i=\dfrac{A\mathbf v_i}{\sigma_i}$.</p>
  <p><strong>Step 3 — show the $\mathbf u_i$ are orthonormal.</strong>
  $\mathbf u_i^\top\mathbf u_j=\dfrac{(A\mathbf v_i)^\top(A\mathbf v_j)}{\sigma_i\sigma_j}=\dfrac{\mathbf v_i^\top(A^\top A)\mathbf v_j}{\sigma_i\sigma_j}
  =\dfrac{\lambda_j\,\mathbf v_i^\top\mathbf v_j}{\sigma_i\sigma_j}=\dfrac{\sigma_j^2}{\sigma_i\sigma_j}\delta_{ij}=\delta_{ij}.$
  (Used $A^\top A\mathbf v_j=\lambda_j\mathbf v_j$ and $\mathbf v_i^\top\mathbf v_j=\delta_{ij}$.)</p>
  <p><strong>Step 4 — verify $A\mathbf v_i=\sigma_i\mathbf u_i$ for all $i$.</strong> For $\sigma_i\gt0$ it's the
  definition; for $\sigma_i=0$, $\lVert A\mathbf v_i\rVert^2=\mathbf v_i^\top A^\top A\mathbf v_i=\lambda_i=0$, so
  $A\mathbf v_i=\mathbf 0=\sigma_i\mathbf u_i$.</p>
  <p><strong>Step 5 — assemble.</strong> Stacking $A\mathbf v_i=\sigma_i\mathbf u_i$ as columns: $AV=U\Sigma$. Since $V$
  is orthogonal, $A=U\Sigma V^\top.\;\blacksquare$ Plain English: the eigenvectors of $A^\top A$ are the input axes, their
  square-rooted eigenvalues are the stretches, and $A$ maps each input axis to an output axis.</p>`,

  code: [
    { label: "SVD: factor, reconstruct, read singular values", src: String.raw`
import numpy as np

A = np.array([[3., 1., 1.],
              [-1., 3., 1.]])
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print("singular values:", np.round(S, 4))
print("U orthonormal? ", np.allclose(U.T @ U, np.eye(U.shape[1])))
print("V orthonormal? ", np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0])))
print("U Σ V^T == A ? ", np.allclose(U @ np.diag(S) @ Vt, A))
` },
    { label: "Singular values = sqrt of eigenvalues of A^T A; A v = σ u", src: String.raw`
import numpy as np

A = np.array([[3., 1., 1.],
              [-1., 3., 1.]])
U, S, Vt = np.linalg.svd(A, full_matrices=False)
eig = np.sqrt(np.sort(np.linalg.eigvalsh(A.T @ A))[::-1][:len(S)])
print("σ from svd        :", np.round(S, 4))
print("sqrt(eig of A^T A):", np.round(eig, 4))    # match
v1 = Vt[0]; print("A v1 == σ1 u1 ?", np.allclose(A @ v1, S[0]*U[:,0]))
` }
  ],

  keyPoints: [
    "$A=U\\Sigma V^\\top$ for ANY matrix: orthogonal $U,V$, diagonal singular values $\\sigma_i\\ge0$.",
    "Geometry: rotate ($V^\\top$), stretch ($\\Sigma$), rotate ($U$) — a circle becomes an ellipse.",
    "$\\mathbf v_i$ are eigenvectors of $A^\\top A$; $\\sigma_i=\\sqrt{\\lambda_i(A^\\top A)}$; $A\\mathbf v_i=\\sigma_i\\mathbf u_i$.",
    "Number of nonzero singular values = rank; $\\sigma_{\\max}=\\lVert A\\rVert_2$.",
    "SVD works where eigendecomposition can't (non-square, defective) — the universal factorization."
  ],

  commonMistakes: [
    { wrong: "Confusing singular values with eigenvalues.", why: "Singular values are $\\ge0$ and exist for any matrix; eigenvalues can be negative/complex and need a square matrix. They coincide (in absolute value) only for symmetric PSD matrices." },
    { wrong: "Forgetting $V^\\top$ is transposed in $U\\Sigma V^\\top$.", why: "NumPy returns <code>Vt</code> already transposed; reconstruct with <code>U @ diag(S) @ Vt</code>, not <code>@ Vt.T</code>." },
    { wrong: "Expecting $U=V$ for a non-symmetric matrix.", why: "Input and output rotations differ in general. They coincide only for symmetric matrices (where SVD aligns with the eigendecomposition up to signs)." }
  ],

  quiz: [
    { q: "The SVD exists for…", options: ["any matrix, any shape", "only square matrices", "only symmetric matrices", "only invertible matrices"], answer: 0,
      explain: "Universality is the whole point: $A=U\\Sigma V^\\top$ holds for every real matrix." },
    { q: "Singular values are always…", options: ["nonnegative", "positive", "real or complex", "integers"], answer: 0,
      explain: "$\\sigma_i=\\sqrt{\\lambda_i(A^\\top A)}\\ge0$; some may be exactly 0 (rank-deficient)." },
    { q: "The number of nonzero singular values equals…", options: ["the rank of $A$", "the trace", "the dimension $n$", "the determinant"], answer: 0,
      explain: "Rank = count of independent directions = count of nonzero stretches $\\sigma_i$." },
    { q: "$\\lVert A\\rVert_2$ (spectral norm) equals…", options: ["$\\sigma_{\\max}$", "$\\sigma_{\\min}$", "$\\sum\\sigma_i$", "$\\prod\\sigma_i$"], answer: 0,
      explain: "The maximum stretch is the largest singular value (Lesson 4.2)." },
    { q: "The right singular vectors $\\mathbf v_i$ are eigenvectors of…", options: ["$A^\\top A$", "$A$", "$AA^\\top$", "$A+A^\\top$"], answer: 0,
      explain: "$A^\\top A=V\\Sigma^2 V^\\top$, so the $\\mathbf v_i$ diagonalize $A^\\top A$ (and the $\\mathbf u_i$ diagonalize $AA^\\top$)." }
  ],

  practice: [
    { level: "easy", prompt: "What are the singular values of $\\begin{bmatrix}3&0\\\\0&2\\end{bmatrix}$?", solution: "For a diagonal matrix with nonnegative entries, the singular values are the entries: $3$ and $2$." },
    { level: "med", prompt: "How do you get the rank of a matrix from its SVD, and why is that robust?", solution: "Count singular values above a small tolerance. It's robust because exact zeros rarely occur in floating point; thresholding tiny $\\sigma_i$ separates true signal directions from numerical noise — a far more reliable rank than checking $\\det=0$." },
    { level: "med", prompt: "If $A=U\\Sigma V^\\top$, express $A^\\top A$ and $AA^\\top$ in terms of the SVD factors.", solution: "$A^\\top A=V\\Sigma^\\top U^\\top U\\Sigma V^\\top=V\\Sigma^2 V^\\top$ and $AA^\\top=U\\Sigma^2 U^\\top$. Both are symmetric PSD with eigenvalues $\\sigma_i^2$; $V$ and $U$ are their eigenvectors." },
    { level: "hard", prompt: "AI task: a recommender factors a (users × items) rating matrix $R\\approx U\\Sigma V^\\top$ and keeps the top $k$ singular values. What do $U$, $V$, and $\\Sigma$ mean, and why does truncation generalize?", solution: "$U$'s columns are latent user-taste directions, $V$'s columns are latent item-feature directions, and $\\Sigma$ weights how strongly each latent factor matters. Keeping the top $k$ keeps the strongest shared patterns and discards the long tail of tiny singular values, which mostly encode noise and per-user idiosyncrasies. The rank-$k$ reconstruction fills in unseen entries by these dominant patterns — that's collaborative filtering, and truncation is what makes it generalize instead of memorizing the sparse observed ratings." }
  ],

  deepDive: String.raw`<p><strong>SVD as a sum of rank-1 pieces — the unifying view.</strong></p>
  <p>Write the SVD as a sum: $A=\sum_{i=1}^{r}\sigma_i\,\mathbf u_i\mathbf v_i^\top$. Each term is a rank-1 matrix (an
  outer product), scaled by its singular value, and the terms are ordered by importance. This single expression unifies
  most of what we've built. The <strong>spectral norm</strong> is the first term's weight, $\sigma_1$ (Track 4). The
  <strong>pseudo-inverse</strong> is $A^+=\sum_{\sigma_i>0}\frac1{\sigma_i}\mathbf v_i\mathbf u_i^\top$ — invert the
  stretches you can, ignore the collapsed directions (Track 4). The <strong>rank</strong> is how many terms are nonzero
  (Track 3). And, as the next lesson shows, truncating the sum after $k$ terms gives the <em>provably best</em> rank-$k$
  approximation.</p>
  <p>This is also why SVD is the safe way to do almost anything fragile with a matrix: it never forms $A^\top A$
  explicitly in a way that squares the conditioning, it exposes exactly which directions are near-singular (small
  $\sigma_i$), and it degrades gracefully. When a numerical method "just works" on rank-deficient or ill-conditioned
  data, an SVD is usually doing the quiet work underneath. It is, in a real sense, the master factorization the rest of
  this track was building toward.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["5.5"] = {
  subtitle: "Find the directions data spreads most — and keep only what matters.",

  aiMoment: String.raw`<p><strong>PCA</strong> is the workhorse for dimensionality reduction: compress 1000-dimensional
  embeddings to 50 while keeping the variance that matters, visualize high-dimensional data in 2-D, or whiten features
  before training. It's the same math as <strong>low-rank approximation</strong> — keeping the top singular values to
  compress a matrix — which underlies model compression and is the principle behind LoRA. This lesson is the direct
  setup for the PCA-from-scratch capstone.</p>`,

  plainEnglish: String.raw`<p><strong>PCA</strong> finds the directions along which your data spreads out the most — the
  principal components — and lets you describe each point by just a few of them. <strong>Low-rank approximation</strong>
  is the same idea for a matrix: keep the few strongest patterns (top singular values) and throw away the rest, getting
  the best possible compression for that size.</p>`,

  intuition: String.raw`<p>A cloud of points has a long axis (most spread) and shorter ones. PCA rotates to those axes
  and orders them by spread. Describe points by their position along the top axes and you've compressed the data with
  minimal loss.</p>
  <figure class="figure">
  <svg viewBox="0 0 316 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A point cloud with its principal components">
    <g fill="#94a3b8">
    <circle cx="70" cy="120" r="3"/><circle cx="95" cy="112" r="3"/><circle cx="110" cy="100" r="3"/>
    <circle cx="130" cy="96" r="3"/><circle cx="150" cy="84" r="3"/><circle cx="165" cy="78" r="3"/>
    <circle cx="185" cy="66" r="3"/><circle cx="205" cy="58" r="3"/><circle cx="120" cy="108" r="3"/>
    <circle cx="160" cy="92" r="3"/><circle cx="140" cy="74" r="3"/><circle cx="175" cy="88" r="3"/>
    </g>
    <line x1="137" y1="89" x2="210" y2="55" stroke="#4f46e5" stroke-width="2.6" marker-end="url(#pc1)"/>
    <line x1="137" y1="89" x2="118" y2="68" stroke="#0d9488" stroke-width="2.6" marker-end="url(#pc2)"/>
    <text x="190" y="48" font-size="11" fill="#4f46e5" font-family="sans-serif">PC1 (most spread)</text>
    <text x="74" y="62" font-size="11" fill="#0d9488" font-family="sans-serif">PC2</text>
    <defs>
      <marker id="pc1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="pc2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#0d9488"/></marker>
    </defs>
  </svg>
  <figcaption>PC1 is the direction of greatest variance; PC2 is perpendicular with the next-most spread.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Center the data: $\tilde X=X-\bar{\mathbf x}$ (subtract the mean row). The
  <strong>covariance</strong> is $C=\tfrac1n\tilde X^\top\tilde X$ (symmetric PSD). The <strong>principal components</strong>
  are its top eigenvectors — equivalently the right singular vectors of $\tilde X=U\Sigma V^\top$, with variance along
  $\mathbf v_i$ equal to $\sigma_i^2/n$. The <strong>rank-$k$ approximation</strong> is
  $A_k=\sum_{i=1}^{k}\sigma_i\mathbf u_i\mathbf v_i^\top$. <strong>Eckart–Young:</strong> $A_k$ is the closest rank-$k$
  matrix to $A$, with error $\lVert A-A_k\rVert_F=\sqrt{\sum_{i>k}\sigma_i^2}$.</p>`,

  derivation: String.raw`<p><strong>Part 1 — PCA maximizes variance.</strong> Project centered data onto a unit direction
  $\mathbf w$; the projected values are $\tilde X\mathbf w$, with variance
  $\tfrac1n\lVert\tilde X\mathbf w\rVert^2=\mathbf w^\top C\mathbf w$.</p>
  <p><strong>Step 1.</strong> Maximize $\mathbf w^\top C\mathbf w$ subject to $\lVert\mathbf w\rVert=1$. This is a
  Rayleigh quotient; for the symmetric PSD $C$ it's maximized by the top eigenvector, with value $\lambda_{\max}(C)$.</p>
  <p><strong>Step 2.</strong> So the first principal component is the top eigenvector of $C$, and the variance it
  captures is $\lambda_{\max}$. The next PC is the top eigenvector orthogonal to it, and so on.</p>
  <p><strong>Step 3 — connect to SVD.</strong> With $\tilde X=U\Sigma V^\top$, $C=\tfrac1n V\Sigma^2 V^\top$, so the
  eigenvectors of $C$ are exactly the right singular vectors $\mathbf v_i$, with variance $\sigma_i^2/n$. PCA <em>is</em>
  the SVD of the centered data. $\blacksquare$</p>
  <hr class="soft">
  <p><strong>Part 2 — why the top-$k$ truncation is optimal (Eckart–Young).</strong> Among all rank-$k$ matrices $B$,
  the SVD truncation $A_k=\sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^\top$ minimizes $\lVert A-B\rVert$ (both Frobenius
  and spectral norms). The leftover error is the energy in the discarded singular values:
  $\lVert A-A_k\rVert_F^2=\sum_{i>k}\sigma_i^2$. Plain English: each singular value is a "chunk" of the matrix's energy;
  keep the biggest chunks and you lose the least.</p>`,

  code: [
    { label: "PCA from scratch (NumPy only) via SVD", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

# correlated 2-D data
X = rng.normal(size=(200, 2)) @ np.array([[3., 1.], [1., 1.]])
Xc = X - X.mean(axis=0)                 # center

U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
pcs = Vt                                # principal components (rows)
var_explained = S**2 / np.sum(S**2)
print("principal components:\n", np.round(pcs, 3))
print("variance explained  :", np.round(var_explained, 3))

# project onto the top component (1-D compression)
z = Xc @ pcs[0]
print("compressed shape:", z.shape, " (from 2-D to 1-D)")
` },
    { label: "Low-rank approximation + Eckart–Young error", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)
A = rng.normal(size=(8, 6))

U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = 3
A_k = (U[:, :k] * S[:k]) @ Vt[:k]      # rank-3 truncation
err = np.linalg.norm(A - A_k, 'fro')
predicted = np.sqrt(np.sum(S[k:]**2))  # Eckart-Young
print("rank-k Frobenius error :", round(float(err), 6))
print("sqrt(Σ dropped σ²)     :", round(float(predicted), 6), "-> match")
` }
  ],

  keyPoints: [
    "PCA = SVD of the mean-centered data; principal components are the right singular vectors.",
    "Variance captured by component $i$ is $\\sigma_i^2/n$ — order components by singular value.",
    "Always center the data first; PCA on uncentered data finds the mean, not the spread.",
    "Rank-$k$ truncation $A_k=\\sum_{i\\le k}\\sigma_i\\mathbf u_i\\mathbf v_i^\\top$ is the best rank-$k$ approximation (Eckart–Young).",
    "Compression error is $\\sqrt{\\sum_{i>k}\\sigma_i^2}$ — the discarded singular-value energy."
  ],

  commonMistakes: [
    { wrong: "Running PCA without centering the data.", why: "The first component then points toward the mean, not the direction of greatest variance. Always subtract the mean (and often standardize) first." },
    { wrong: "Treating PCA as feature selection.", why: "Principal components are <em>linear combinations</em> of all original features, not a subset. PCA reduces dimension by mixing, not by dropping columns." },
    { wrong: "Keeping components by index instead of variance.", why: "Choose $k$ from the variance-explained / singular-value spectrum (e.g. keep 95% of variance), not an arbitrary count — the spectrum tells you where the signal ends." }
  ],

  quiz: [
    { q: "PCA's principal components are the eigenvectors of…", options: ["the covariance matrix", "the data matrix $X$", "$XX^\\top$", "the identity"], answer: 0,
      explain: "PCs are the top eigenvectors of the covariance $C=\\tfrac1n\\tilde X^\\top\\tilde X$, equivalently the right singular vectors of centered $\\tilde X$." },
    { q: "Before PCA you must…", options: ["subtract the mean (center)", "normalize to unit norm rows", "square the data", "take the log"], answer: 0,
      explain: "Centering is essential so the covariance measures spread, not offset from the origin." },
    { q: "The best rank-$k$ approximation of $A$ in Frobenius norm is…", options: ["the top-$k$ SVD truncation", "the top-$k$ rows", "$A^\\top A$", "the diagonal of $A$"], answer: 0,
      explain: "Eckart–Young: keep the $k$ largest singular triplets; nothing rank-$k$ does better." },
    { q: "Singular values are $[5,4,2,1]$. Fraction of variance kept by the top 2 (using $\\sigma^2$)?", options: ["$41/46\\approx0.89$", "$9/12$", "$0.5$", "$2/4$"], answer: 0,
      explain: "Energy $=\\sigma^2$: top-2 $=25+16=41$, total $=25+16+4+1=46$, ratio $\\approx0.89$." },
    { q: "Rank-$k$ truncation error in Frobenius norm equals…", options: ["$\\sqrt{\\sum_{i>k}\\sigma_i^2}$", "$\\sigma_{k}$", "$\\sum_{i>k}\\sigma_i$", "$\\sigma_1-\\sigma_k$"], answer: 0,
      explain: "The discarded energy is the sum of squared dropped singular values; its square root is the Frobenius error." }
  ],

  practice: [
    { level: "easy", prompt: "Singular values $[6,3,1]$. What fraction of variance does the top component explain?", solution: "$6^2/(6^2+3^2+1^2)=36/46\\approx0.78$, about 78%." },
    { level: "med", prompt: "Why does centering matter? Describe what PC1 becomes if you skip it on data far from the origin.", solution: "Uncentered, the covariance-like matrix $\\tfrac1n X^\\top X$ is dominated by the squared mean. PC1 then points roughly toward the data's mean location (the biggest 'pull' from the origin) rather than the direction of greatest spread, so the reduction captures position instead of structure." },
    { level: "med", prompt: "Data has singular values $[10, 1, 0.5, 0.1]$. Recommend $k$ and justify.", solution: "Energies $\\sigma^2=[100,1,0.25,0.01]$; the first holds $100/101.26\\approx98.8\\%$. A sharp drop after $\\sigma_1$ suggests $k=1$ captures nearly everything; choose $k$ where the cumulative variance crosses your threshold (here $k=1$ for ~99%)." },
    { level: "hard", prompt: "AI task: outline implementing PCA on MNIST (784-dim images) with only NumPy, and how to visualize the top 16 components as images.", solution: "(1) Load images as an $n\\times784$ matrix $X$; (2) center: $\\tilde X=X-\\bar{\\mathbf x}$; (3) compute $U,\\Sigma,V^\\top=\\texttt{np.linalg.svd}(\\tilde X,\\,\\text{full\\_matrices=False})$; (4) the rows of $V^\\top$ (or columns of $V$) are the principal components, each a 784-vector; (5) reshape each of the top 16 to $28\\times28$ and display with <code>imshow</code> — they look like smooth 'eigen-digit' strokes; (6) project with $Z=\\tilde X V_k$ to compress, and reconstruct via $\\tilde X\\approx Z V_k^\\top+\\bar{\\mathbf x}$. The variance-explained curve $\\sigma_i^2/\\sum\\sigma_j^2$ tells you how many components to keep. This is exactly the Linear Algebra capstone." }
  ],

  deepDive: String.raw`<p><strong>Eckart–Young: why "keep the biggest singular values" is provably optimal.</strong></p>
  <p>The theorem says no rank-$k$ matrix approximates $A$ better than its own truncated SVD $A_k$ — in <em>both</em> the
  Frobenius and spectral norms simultaneously. The intuition comes from the rank-1 expansion
  $A=\sum_i\sigma_i\mathbf u_i\mathbf v_i^\top$: the terms are mutually orthogonal "energy packets," and the total
  energy is $\sum_i\sigma_i^2$. Any rank-$k$ matrix can capture at most $k$ orthogonal directions of that energy, and the
  greedy choice — the $k$ largest $\sigma_i^2$ — leaves the least behind, $\sum_{i>k}\sigma_i^2$. There's nothing
  cleverer to do.</p>
  <p>This one result quietly justifies a surprising amount of practice. Image and embedding <strong>compression</strong>
  keeps top singular directions. <strong>Denoising</strong> drops the tail, assuming noise lives in the small singular
  values. <strong>LoRA</strong> bets that a weight update's important action is low-rank, so $\Delta W=BA$ with small $r$
  captures it. Even spectral-norm regularization and PCA whitening are reading the same singular spectrum. Once you see a
  matrix as an ordered sum of rank-1 pieces, "how much can I throw away?" has an exact, computable answer — and that is
  the through-line connecting eigenvalues, SVD, compression, and modern parameter-efficient fine-tuning.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["5.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 70 minutes.",

  intro: String.raw`<p>This exam spans all of Track 5: eigenvalues and diagonalization, the spectral theorem,
  positive-definiteness, the SVD, and PCA / low-rank approximation. <strong>Give yourself 70 minutes</strong>, produce
  each answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "Find the eigenvalues of $\\begin{bmatrix}4&1\\\\0&2\\end{bmatrix}$.",
      solution: "Triangular, so eigenvalues are the diagonal: $4$ and $2$. (Check: $\\det(A-\\lambda I)=(4-\\lambda)(2-\\lambda)=0$.)" },
    { level: "easy", prompt: "If $A=PDP^{-1}$ with $D=\\mathrm{diag}(1,2,3)$, give the eigenvalues of $A^{4}$.",
      solution: "$A^4=PD^4P^{-1}$, so eigenvalues are $1^4,2^4,3^4=1,16,81$; eigenvectors unchanged." },
    { level: "med", prompt: "Is $\\begin{bmatrix}2&-1\\\\-1&2\\end{bmatrix}$ positive definite? Justify two ways.",
      solution: "Eigenvalues: trace 4, det $4-1=3$, so $\\lambda=1,3\\gt0$ → PD. Also Sylvester: leading minors $2\\gt0$ and $\\det=3\\gt0$ → PD. (And $\\mathbf x^\\top A\\mathbf x=2x_1^2-2x_1x_2+2x_2^2=(x_1-x_2)^2+x_1^2+x_2^2\\gt0$.)" },
    { level: "med", prompt: "A symmetric matrix has eigenvalues $\\{3,-1\\}$. Classify the critical point of $\\tfrac12\\mathbf x^\\top A\\mathbf x$ at the origin.",
      solution: "Mixed signs → indefinite → saddle point: the quadratic curves up along one eigenvector and down along the other." },
    { level: "med", prompt: "Why are the eigenvectors of a symmetric matrix (distinct eigenvalues) orthogonal? Sketch the argument.",
      solution: "Compute $\\mathbf v_i^\\top A\\mathbf v_j$ two ways: $=\\lambda_j\\mathbf v_i^\\top\\mathbf v_j$ (use $A\\mathbf v_j=\\lambda_j\\mathbf v_j$) and $=\\lambda_i\\mathbf v_i^\\top\\mathbf v_j$ (use symmetry and $A\\mathbf v_i=\\lambda_i\\mathbf v_i$). Subtracting, $(\\lambda_i-\\lambda_j)\\mathbf v_i^\\top\\mathbf v_j=0$, and since $\\lambda_i\\neq\\lambda_j$, $\\mathbf v_i^\\top\\mathbf v_j=0$." },
    { level: "med", prompt: "A matrix has singular values $[4,3,0]$. Give its rank and spectral norm.",
      solution: "Rank = number of nonzero singular values = 2. Spectral norm $=\\sigma_{\\max}=4$." },
    { level: "hard", prompt: "Relate the SVD $A=U\\Sigma V^\\top$ to the eigendecompositions of $A^\\top A$ and $AA^\\top$.",
      solution: "$A^\\top A=V\\Sigma^2 V^\\top$ and $AA^\\top=U\\Sigma^2 U^\\top$. So $V$ holds the eigenvectors of $A^\\top A$, $U$ holds those of $AA^\\top$, and both share eigenvalues $\\sigma_i^2$. The singular values are $\\sigma_i=\\sqrt{\\lambda_i}$." },
    { level: "hard", prompt: "Singular values are $[8,4,2,1]$. What fraction of variance/energy do the top 2 components keep, and what is the rank-2 Frobenius error?",
      solution: "Energy $=\\sigma^2=[64,16,4,1]$, total $85$. Top-2 keep $80/85\\approx94.1\\%$. Rank-2 Frobenius error $=\\sqrt{4+1}=\\sqrt5\\approx2.236$ (the dropped singular-value energy)." },
    { level: "hard", prompt: "Why is PCA on un-centered data wrong, and what exactly does it compute instead?",
      solution: "PCA wants directions of greatest <em>variance</em>, which is defined about the mean. Skipping centering replaces the covariance $\\tfrac1n\\tilde X^\\top\\tilde X$ with $\\tfrac1n X^\\top X$, whose dominant eigenvector is pulled toward the data's mean offset from the origin. So it computes the direction of greatest <em>second moment</em> (spread + squared mean), and PC1 ends up pointing roughly at the centroid rather than along the true spread." },
    { level: "hard", prompt: "AI task: explain why Eckart–Young makes low-rank truncation optimal, and how the same fact justifies LoRA.",
      solution: "Write $A=\\sum_i\\sigma_i\\mathbf u_i\\mathbf v_i^\\top$: orthogonal rank-1 packets with energies $\\sigma_i^2$. Any rank-$k$ matrix captures at most $k$ orthogonal directions, so keeping the $k$ largest $\\sigma_i$ minimizes leftover energy $\\sum_{i>k}\\sigma_i^2$ — that's Eckart–Young, optimal in Frobenius and spectral norm. LoRA applies the same logic to a weight <em>update</em>: it assumes $\\Delta W$ has low intrinsic rank, so a rank-$r$ factorization $BA$ captures the important directions of change with $2dr\\ll d^2$ parameters, discarding the negligible tail — a deliberate, justified low-rank approximation of the fine-tuning signal." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> You command eigen/SVD/PCA — the heart of applied linear algebra. On to Calculus (Track 6) and, when ready, the PCA capstone.</li>
    <li><strong>7–8:</strong> Strong. Revisit the SVD ↔ $A^\\top A$ link and Eckart–Young if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive the SVD construction and the PD ⇔ positive-eigenvalues argument; redo Lessons 5.3 and 5.4.</li>
    <li><strong>Below 5:</strong> Rework the track in order — eigen/SVD/PCA recur in optimization, probability, and compression throughout the rest of the course.</li>
  </ul>`
};
