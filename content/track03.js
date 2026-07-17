/* ============================================================
   TRACK 3 — Linear Algebra II — Systems, Spaces & Rank
   3.1 Gaussian Elimination · 3.2 Rank/Column/Null Space ·
   3.3 Independence/Basis/Dimension · 3.4 Orthogonality/Projections/Gram–Schmidt ·
   3.5 QR Decomposition · 3.E Track Exam
   ============================================================ */
(window.LESSON_CONTENT ||= {})["3.1"] = {
  subtitle: "Solving Ax = b by organized bookkeeping — the engine under least-squares.",

  aiMoment: String.raw`<p>Fit a line to data and you're solving a linear system. Linear regression's "normal equations"
  $A^\top A\,\mathbf x=A^\top\mathbf b$ are exactly an $A\mathbf x=\mathbf b$ to be solved, and the workhorse for solving
  any such system by hand or in a solver is <strong>Gaussian elimination</strong>. It's also the cleanest way to see
  <em>why</em> a system has one solution, none, or infinitely many — which becomes rank in the next lesson.</p>`,

  plainEnglish: String.raw`<p>A system of linear equations asks: what values make all the equations true at once —
  the point where several lines (or planes) meet? <strong>Gaussian elimination</strong> is organized substitution: use
  one equation to remove a variable from the others, repeat until one variable is alone, then back-substitute.</p>`,

  intuition: String.raw`<p>Two equations in two unknowns are two lines. They either cross at one point (one solution),
  run parallel (no solution), or are the same line (infinitely many). Elimination is the algebra that finds the
  crossing point without drawing.</p>
  <figure class="figure">
  <svg viewBox="0 0 240 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two lines crossing at the solution">
    <line x1="20" y1="165" x2="225" y2="165" stroke="#eef0f4"/>
    <line x1="30" y1="15" x2="30" y2="175" stroke="#eef0f4"/>
    <line x1="35" y1="40" x2="210" y2="150" stroke="#4f46e5" stroke-width="2.6"/>
    <line x1="35" y1="160" x2="210" y2="45" stroke="#0d9488" stroke-width="2.6"/>
    <circle cx="123" cy="95" r="5" fill="#dc2626"/>
    <text x="132" y="92" font-size="11" fill="#dc2626" font-family="sans-serif">solution</text>
    <text x="160" y="142" font-size="11" fill="#4f46e5" font-family="sans-serif">eq 1</text>
    <text x="160" y="60" font-size="11" fill="#0d9488" font-family="sans-serif">eq 2</text>
  </svg>
  <figcaption>One crossing point = a unique solution. Parallel lines = none; identical lines = infinitely many.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Write the system as $A\mathbf x=\mathbf b$ and stack into the <strong>augmented matrix</strong>
  $[\,A\mid\mathbf b\,]$. Three <strong>elementary row operations</strong> leave the solution unchanged: swap two rows,
  scale a row by a nonzero number, add a multiple of one row to another. Apply them until the matrix is in
  <strong>row echelon form</strong> (each leading nonzero entry, a <em>pivot</em>, sits to the right of the one above).
  Then <strong>back-substitute</strong> from the bottom up.</p>`,

  derivation: String.raw`<p><strong>Solve</strong> $\;x+y+z=6,\;\;2x+5y+z=15,\;\;2x+3y+8z=32.$</p>
  <p><strong>Step 1 — clear $x$ from rows 2 and 3</strong> using row 1.
  $R_2\leftarrow R_2-2R_1:\;(2x+5y+z)-2(x+y+z)=15-12\Rightarrow 3y-z=3.$
  $R_3\leftarrow R_3-2R_1:\;(2x+3y+8z)-2(x+y+z)=32-12\Rightarrow y+6z=20.$</p>
  <p><strong>Step 2 — the reduced system</strong> is now
  $x+y+z=6,\;\;3y-z=3,\;\;y+6z=20.$</p>
  <p><strong>Step 3 — clear $y$ from row 3</strong> (scale to stay in integers):
  $R_3\leftarrow 3R_3-R_2:\;3(y+6z)-(3y-z)=3(20)-3\Rightarrow 19z=57\Rightarrow z=3.$</p>
  <p><strong>Step 4 — back-substitute.</strong> Row 2: $3y-z=3\Rightarrow 3y-3=3\Rightarrow y=2$. Row 1:
  $x+2+3=6\Rightarrow x=1.$</p>
  <p><strong>Step 5 — solution:</strong> $(x,y,z)=(1,2,3).\;\blacksquare$ Plain English: each step trades a messy
  equation for a simpler one until a single variable is pinned down, then the rest fall out in reverse.</p>`,

  code: [
    { label: "Solve a system, two ways", src: String.raw`
import numpy as np

A = np.array([[1.,1.,1.],
              [2.,5.,1.],
              [2.,3.,8.]])
b = np.array([6., 15., 32.])

x = np.linalg.solve(A, b)        # solver does elimination internally
print("solution x =", np.round(x, 6))         # [1. 2. 3.]
print("check A x == b:", np.allclose(A @ x, b))
` },
    { label: "When there is no unique solution", src: String.raw`
import numpy as np

# parallel-line system: x+y=1 and x+y=2 (inconsistent)
A = np.array([[1.,1.],[1.,1.]]); b = np.array([1., 2.])
print("rank A   =", np.linalg.matrix_rank(A))             # 1
print("rank [A|b]=", np.linalg.matrix_rank(np.c_[A, b]))  # 2  -> inconsistent
# rank(A) < rank([A|b])  means NO solution exists.
` }
  ],

  keyPoints: [
    "A linear system is $A\\mathbf x=\\mathbf b$; solve it with elementary row operations.",
    "Row operations (swap, scale, add-multiple) never change the solution set.",
    "Reduce to row echelon form, then back-substitute from the bottom up.",
    "Pivots determine the outcome: a system has one, none, or infinitely many solutions.",
    "Inconsistent ⇔ $\\text{rank}(A)\\lt\\text{rank}([A\\mid\\mathbf b])$."
  ],

  commonMistakes: [
    { wrong: "Applying a row operation to $A$ but not to $\\mathbf b$.", why: "The augmented column must travel with the rows. Forgetting it solves a different system." },
    { wrong: "Dividing by a zero pivot.", why: "If a pivot position is 0 you must swap in a row with a nonzero entry first (partial pivoting) — also the key to numerical stability." },
    { wrong: "Assuming every square system has a unique solution.", why: "If a pivot is missing (a row becomes all zeros on the left), there are free variables → infinitely many solutions, or none if the right side is inconsistent." }
  ],

  quiz: [
    { q: "Solve $x+y=5,\\;x-y=1$.", options: ["$(3,2)$", "$(2,3)$", "$(4,1)$", "$(1,4)$"], answer: 0,
      explain: "Add the equations: $2x=6\\Rightarrow x=3$, then $y=2$. Choice $(2,3)$ swaps them." },
    { q: "Which operation is NOT a valid elementary row operation?", options: ["Multiply a row by 0", "Swap two rows", "Add 2× one row to another", "Multiply a row by 3"], answer: 0,
      explain: "Scaling by 0 destroys information (the row vanishes) and is not reversible, so it's not allowed; scaling by any nonzero number is fine." },
    { q: "If elimination yields a row $[\\,0\\;0\\;0\\mid 4\\,]$, the system is…", options: ["inconsistent (no solution)", "uniquely solvable", "has infinitely many solutions", "homogeneous"], answer: 0,
      explain: "That row says $0=4$, a contradiction — no solution. (A row $[0\\,0\\,0\\mid 0]$ instead would signal free variables.)" },
    { q: "A $3\\times3$ system reduces to 2 pivots. The solution set is…", options: ["a line (infinitely many)", "a single point", "empty", "a plane"], answer: 0,
      explain: "2 pivots in 3 unknowns leaves 1 free variable, so (if consistent) the solutions form a 1-parameter line." },
    { q: "Back-substitution into $2z=6$, then $y+z=5$, gives $y=$…", options: ["2", "3", "5", "8"], answer: 0,
      explain: "$z=3$, then $y=5-3=2$." }
  ],

  practice: [
    { level: "easy", prompt: "Solve $2x+y=7,\\;x-y=-1$.", solution: "Add: $3x=6\\Rightarrow x=2$; then $y=7-2x=3$. Solution $(2,3)$." },
    { level: "med", prompt: "Put $\\begin{bmatrix}1&2&|&3\\\\2&5&|&8\\end{bmatrix}$ in row echelon form and solve.", solution: "$R_2\\leftarrow R_2-2R_1=[0\\,1\\mid2]$ → $y=2$; back-sub $x+2(2)=3\\Rightarrow x=-1$. Solution $(-1,2)$." },
    { level: "med", prompt: "For what $b$ does $x+y=1,\\;2x+2y=b$ have solutions, and how many?", solution: "The second row is twice the first only if $b=2$. If $b=2$: infinitely many (one free variable). If $b\\neq2$: inconsistent, none." },
    { level: "hard", prompt: "AI task: ridge regression solves $(A^\\top A+\\lambda I)\\mathbf x=A^\\top\\mathbf b$. Explain why adding $\\lambda I$ guarantees a unique solution even when $A^\\top A$ is singular.", solution: "$A^\\top A$ can be singular (a zero pivot / dependent columns), making the plain normal equations have no unique solution. Adding $\\lambda I$ (with $\\lambda\\gt0$) shifts every eigenvalue up by $\\lambda$, so the matrix becomes positive definite — full rank, all pivots nonzero — hence invertible and the system has exactly one solution. This is the linear-algebra reason $L_2$ regularization also stabilizes the fit (and connects to MAP in Track 11)." }
  ],

  deepDive: String.raw`<p><strong>Existence and uniqueness, read off the pivots.</strong></p>
  <p>After elimination, count pivots. For an $n$-unknown system: if there are $n$ pivots (one per column) and no
  contradiction row, the solution is <strong>unique</strong>. If a left-hand row goes all-zero while its right side is
  nonzero ($0=c$), the system is <strong>inconsistent</strong> — no solution. If some columns have no pivot (free
  variables) and there's no contradiction, there are <strong>infinitely many</strong> solutions, parameterized by the
  free variables.</p>
  <p>This pivot count is the same number we'll call the <strong>rank</strong> in Lesson 3.2. It also explains a
  practical ML point: the normal equations $A^\top A\mathbf x=A^\top\mathbf b$ have a unique least-squares solution only
  when $A$ has full column rank (independent features). Collinear features create a missing pivot — infinitely many
  equally-good fits — which is exactly the instability that ridge regularization cures by restoring full rank. Solving
  systems and understanding rank are two views of one idea.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["3.2"] = {
  subtitle: "Reachable outputs, crushed inputs, and the count of truly independent directions.",

  aiMoment: String.raw`<p>LoRA — the dominant way to fine-tune large models — assumes the weight <em>update</em> is
  <strong>low rank</strong>: $\Delta W=BA$ where $B,A$ are skinny. It works because a big update often has only a few
  independent directions of change. "Rank" is the precise count of those directions, and "column space" / "null space"
  are the geometry of what a weight matrix can produce and what it throws away.</p>`,

  plainEnglish: String.raw`<p>Think of a matrix as a machine that turns input vectors into output vectors. The
  <strong>column space</strong> is everything it can possibly output. The <strong>null space</strong> is the set of
  inputs it crushes to zero. The <strong>rank</strong> is the number of genuinely independent directions the machine
  keeps — its true "width" of information.</p>`,

  intuition: String.raw`<p>A matrix can flatten space. If it squashes a 3-D input so all outputs land on a 2-D plane,
  its rank is 2, and the squashed-away direction is the null space — inputs along it come out as zero.</p>
  <figure class="figure">
  <svg viewBox="0 0 330 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A matrix maps inputs; the null direction collapses to zero">
    <circle cx="60" cy="95" r="4" fill="#475569"/>
    <line x1="60" y1="95" x2="120" y2="55" stroke="#4f46e5" stroke-width="2.6" marker-end="url(#r1)"/>
    <line x1="60" y1="95" x2="40" y2="150" stroke="#94a3b8" stroke-width="2.6" marker-end="url(#r2)"/>
    <text x="120" y="48" font-size="11" fill="#4f46e5" font-family="sans-serif">v</text>
    <text x="20" y="150" font-size="11" fill="#94a3b8" font-family="sans-serif">n (null)</text>
    <text x="40" y="170" font-size="10" fill="#94a3b8" font-family="sans-serif">input space</text>
    <line x1="150" y1="95" x2="200" y2="95" stroke="#64748b" stroke-width="1.6" marker-end="url(#r3)"/>
    <text x="160" y="86" font-size="11" fill="#64748b" font-family="sans-serif">A</text>
    <circle cx="270" cy="95" r="4" fill="#475569"/>
    <line x1="270" y1="95" x2="320" y2="60" stroke="#4f46e5" stroke-width="2.6" marker-end="url(#r4)"/>
    <circle cx="270" cy="95" r="8" fill="none" stroke="#dc2626" stroke-dasharray="3 3"/>
    <text x="300" y="52" font-size="11" fill="#4f46e5" font-family="sans-serif">Av</text>
    <text x="232" y="120" font-size="10" fill="#dc2626" font-family="sans-serif">An = 0</text>
    <text x="250" y="170" font-size="10" fill="#94a3b8" font-family="sans-serif">column space</text>
    <defs>
      <marker id="r1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="r2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker>
      <marker id="r3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#64748b"/></marker>
      <marker id="r4" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
    </defs>
  </svg>
  <figcaption>The null-space direction n maps to 0; reachable outputs (like Av) fill the column space.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $A\in\mathbb{R}^{m\times n}$: the <strong>column space</strong> $\operatorname{col}(A)$
  is the span of its columns (all $A\mathbf x$). The <strong>null space</strong> $\operatorname{null}(A)=\{\mathbf x:A\mathbf x=\mathbf 0\}$.
  The <strong>rank</strong> $r=\dim\operatorname{col}(A)$ equals the number of pivots after elimination. The
  <strong>rank–nullity theorem</strong> ties them together:</p>
  $$\operatorname{rank}(A)+\dim\operatorname{null}(A)=n\quad(\text{number of columns}).$$
  <p>Plain English: every input dimension is either kept (counts toward rank) or collapsed (counts toward nullity).</p>`,

  derivation: String.raw`<p><strong>Find the rank and null space of</strong> $A=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}.$</p>
  <p><strong>Step 1 — eliminate.</strong> $R_2\leftarrow R_2-4R_1=[0,-3,-6]$; $R_3\leftarrow R_3-7R_1=[0,-6,-12]$.
  Then $R_3\leftarrow R_3-2R_2=[0,0,0]$.</p>
  <p><strong>Step 2 — count pivots.</strong> Row-echelon form is
  $\begin{bmatrix}1&2&3\\0&-3&-6\\0&0&0\end{bmatrix}$: pivots in columns 1 and 2, none in column 3. So
  $\operatorname{rank}(A)=2$.</p>
  <p><strong>Step 3 — rank–nullity.</strong> $n=3$ columns, so $\dim\operatorname{null}(A)=3-2=1$: one collapsed
  direction.</p>
  <p><strong>Step 4 — find that direction.</strong> Solve $A\mathbf x=\mathbf 0$. From the reduced rows,
  $-3y-6z=0\Rightarrow y=-2z$, and $x+2y+3z=0\Rightarrow x=-2y-3z=4z-3z=z$. With $z=1$: $\mathbf x=[1,-2,1]$.
  Check: $A[1,-2,1]^\top=[1-4+3,\,4-10+6,\,7-16+9]=[0,0,0].\;\checkmark\;\blacksquare$</p>`,

  code: [
    { label: "Rank, and verifying a null-space vector", src: String.raw`
import numpy as np

A = np.array([[1.,2,3],[4,5,6],[7,8,9]])
print("rank(A)   =", np.linalg.matrix_rank(A))     # 2
print("nullity   =", A.shape[1] - np.linalg.matrix_rank(A))  # 1

n = np.array([1., -2., 1.])          # claimed null-space vector
print("A @ n =", np.round(A @ n, 6)) # [0 0 0]  -> collapsed to zero
` },
    { label: "LoRA: a low-rank update", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

d, r = 100, 4                       # full dim 100, adapter rank 4
B = rng.normal(size=(d, r)) * 0.1
A = rng.normal(size=(r, d)) * 0.1
dW = B @ A                          # the weight update
print("dW shape:", dW.shape, " rank:", np.linalg.matrix_rank(dW))  # 4
# 100x100 = 10,000 numbers, but only 2*100*4 = 800 are trained.
print("params stored:", B.size + A.size, "vs full", d*d)
` }
  ],

  keyPoints: [
    "Column space = all reachable outputs $A\\mathbf x$; null space = inputs sent to $\\mathbf 0$.",
    "Rank = number of pivots = dimension of the column space = independent directions kept.",
    "Rank–nullity: $\\operatorname{rank}(A)+\\operatorname{nullity}(A)=n$ (columns).",
    "Rank $\\le\\min(m,n)$; 'full rank' means as large as that allows.",
    "LoRA exploits low rank: a big update $\\Delta W=BA$ needs only $r$ independent directions."
  ],

  commonMistakes: [
    { wrong: "Counting rows instead of independent directions.", why: "Rank is the number of <em>independent</em> rows (equivalently columns). The $3\\times3$ example has 3 rows but rank 2 because the rows are dependent." },
    { wrong: "Thinking rank–nullity uses $m$ (rows).", why: "It uses $n$, the number of <em>columns</em> (input dimensions). Each input dimension is either kept or collapsed." },
    { wrong: "Assuming a square matrix is always full rank / invertible.", why: "A square matrix with dependent columns has rank $\\lt n$, a nonzero null space, and no inverse — exactly the singular case." }
  ],

  quiz: [
    { q: "$A=\\begin{bmatrix}1&2\\\\2&4\\end{bmatrix}$ has rank…", options: ["1", "2", "0", "4"], answer: 0,
      explain: "Row 2 = 2× row 1, so only one independent direction: rank 1. Its null space is spanned by $[2,-1]$." },
    { q: "If $A$ is $4\\times6$ with rank 4, its nullity is…", options: ["2", "0", "4", "6"], answer: 0,
      explain: "Rank–nullity: $4+\\text{nullity}=6\\Rightarrow\\text{nullity}=2$ (columns $n=6$)." },
    { q: "The maximum possible rank of a $3\\times5$ matrix is…", options: ["3", "5", "8", "15"], answer: 0,
      explain: "$\\operatorname{rank}\\le\\min(m,n)=\\min(3,5)=3$." },
    { q: "A nonzero null space means the matrix is…", options: ["singular (not invertible)", "full rank", "orthogonal", "symmetric"], answer: 0,
      explain: "If some nonzero $\\mathbf x$ gives $A\\mathbf x=\\mathbf 0$, the map isn't one-to-one, so no inverse exists." },
    { q: "A LoRA update $\\Delta W=BA$ with $B\\in\\mathbb{R}^{d\\times r}$, $A\\in\\mathbb{R}^{r\\times d}$ has rank at most…", options: ["$r$", "$d$", "$d^2$", "$2r$"], answer: 0,
      explain: "A product through an $r$-dimensional middle can carry at most $r$ independent directions: $\\operatorname{rank}(BA)\\le r$." }
  ],

  practice: [
    { level: "easy", prompt: "What are the rank and nullity of the $2\\times2$ identity?", solution: "Two pivots → rank 2; nullity $=2-2=0$ (only $\\mathbf 0$ maps to $\\mathbf 0$)." },
    { level: "med", prompt: "Find the rank of $\\begin{bmatrix}1&0&1\\\\0&1&1\\\\1&1&2\\end{bmatrix}$.", solution: "Row 3 = row 1 + row 2, so it's dependent. Rows 1 and 2 are independent → rank 2 (nullity 1)." },
    { level: "med", prompt: "Give a nonzero vector in the null space of $\\begin{bmatrix}1&1\\\\1&1\\end{bmatrix}$.", solution: "Solve $x+y=0$: any multiple of $[1,-1]$. Check: $[1,1]\\cdot[1,-1]=0$ for both rows." },
    { level: "hard", prompt: "AI task: a $1000\\times1000$ weight update is approximated by rank-8 LoRA. How many trainable parameters, versus full fine-tuning, and what assumption makes this OK?", solution: "LoRA stores $B\\,(1000\\times8)$ and $A\\,(8\\times1000)$: $2\\cdot1000\\cdot8=16{,}000$ parameters, versus $1000^2=1{,}000{,}000$ for the full matrix — a $62.5\\times$ reduction. It works under the assumption that the adaptation $\\Delta W$ has low <em>intrinsic rank</em>: the fine-tuning change really lives in a handful of directions, so rank 8 captures it with negligible loss." }
  ],

  deepDive: String.raw`<p><strong>Why low rank is everywhere: the rank of a product is a bottleneck.</strong></p>
  <p>A key fact: $\operatorname{rank}(BA)\le\min(\operatorname{rank}(B),\operatorname{rank}(A))$. Information must pass
  through the smaller of the two, so a product can never have higher rank than its skinniest factor. That single
  inequality drives a lot of modern ML. LoRA forces $\Delta W=BA$ through an $r$-dimensional waist, so the update is
  rank-$\le r$ by construction — cheap to store and train. The same bottleneck explains autoencoders (a narrow latent
  layer caps the rank of what flows through), and it's why stacking linear layers can't raise expressivity
  (Lesson 2.4): the effective matrix is rank-limited by the thinnest layer.</p>
  <p>It also warns of a failure mode: <strong>rank collapse</strong>. If representations across tokens or layers drift
  toward the same few directions, the effective rank drops and the model loses capacity — a real concern in very deep
  transformers, mitigated by residual connections and normalization. Rank isn't an abstraction; it's the currency of
  how much independent information a layer can carry.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["3.3"] = {
  subtitle: "No-redundancy directions, a minimal spanning set, and the size of a space.",

  aiMoment: String.raw`<p>An embedding layer learns a <strong>basis</strong> for meaning: each of the (say) 768
  dimensions is a direction, and a token is coordinates in that basis. You want those directions
  <strong>independent</strong> — if two carry the same information, you've wasted capacity. Independence, basis, and
  dimension are how we measure whether a representation is using its space or quietly collapsing.</p>`,

  plainEnglish: String.raw`<p>A set of vectors is <strong>independent</strong> if none of them can be built from the
  others — no redundancy. A <strong>basis</strong> is a smallest set of independent vectors that still reaches every
  point in the space. The <strong>dimension</strong> is simply how many vectors a basis needs.</p>`,

  intuition: String.raw`<p>Two arrows pointing different ways let you reach any point in a plane — they're
  independent. But if a third arrow already lies in that plane, it adds nothing new; it's dependent. Independence is
  "each vector opens a new direction."</p>
  <figure class="figure">
  <svg viewBox="0 0 380 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Independent vs dependent vectors">
    <line x1="70" y1="130" x2="135" y2="80" stroke="#4f46e5" stroke-width="2.8" marker-end="url(#i1)"/>
    <line x1="70" y1="130" x2="55"  y2="60" stroke="#0d9488" stroke-width="2.8" marker-end="url(#i2)"/>
    <text x="40" y="155" font-size="11" fill="#16a34a" font-family="sans-serif">independent → plane</text>
    <line x1="240" y1="120" x2="305" y2="85" stroke="#4f46e5" stroke-width="2.8" marker-end="url(#i3)"/>
    <line x1="240" y1="120" x2="287" y2="95" stroke="#7c3aed" stroke-width="2.8" marker-end="url(#i4)"/>
    <text x="225" y="150" font-size="11" fill="#dc2626" font-family="sans-serif">dependent → just a line</text>
    <defs>
      <marker id="i1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="i2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#0d9488"/></marker>
      <marker id="i3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="i4" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#7c3aed"/></marker>
    </defs>
  </svg>
  <figcaption>Left: two directions span a plane. Right: two near-parallel vectors only reach a line.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Vectors $\mathbf v_1,\dots,\mathbf v_k$ are <strong>linearly independent</strong> if the only
  way to combine them to zero is the trivial way:</p>
  $$c_1\mathbf v_1+\dots+c_k\mathbf v_k=\mathbf 0\;\Longrightarrow\;c_1=\dots=c_k=0.$$
  <p>A <strong>basis</strong> of a space is an independent set that <em>spans</em> it. Every basis of a given space has
  the same size, called the <strong>dimension</strong>. Given a basis, each vector has <strong>unique
  coordinates</strong> (the weights $c_i$). Plain English: a basis is a complete, non-redundant coordinate system.</p>`,

  derivation: String.raw`<p><strong>Claim.</strong> $\mathbf b_1=[1,0]$, $\mathbf b_2=[1,1]$ form a basis of $\mathbb{R}^2$,
  and coordinates are unique.</p>
  <p><strong>Step 1 — independence.</strong> Suppose $c_1[1,0]+c_2[1,1]=[0,0]$. Componentwise: $c_1+c_2=0$ and
  $c_2=0$. The second gives $c_2=0$, then $c_1=0$. Only the trivial combination → independent.</p>
  <p><strong>Step 2 — spanning.</strong> Given any $[t_1,t_2]$, solve $c_1+c_2=t_1$, $c_2=t_2$: $c_2=t_2$,
  $c_1=t_1-t_2$. A solution always exists, so the pair spans $\mathbb{R}^2$.</p>
  <p><strong>Step 3 — it's a basis</strong> (independent + spanning), and since $\dim\mathbb{R}^2=2$, two vectors is
  exactly right.</p>
  <p><strong>Step 4 — uniqueness of coordinates.</strong> If $[t_1,t_2]$ had two representations, their difference would
  be a nontrivial combination giving $\mathbf 0$ — impossible by independence. So the coordinates $(t_1-t_2,\,t_2)$ are
  unique. $\blacksquare$ Plain English: in a basis, every vector has exactly one "address."</p>`,

  code: [
    { label: "Test independence with rank; find coordinates", src: String.raw`
import numpy as np

V = np.array([[1.,1.],
              [0.,1.]]).T          # columns are the basis vectors [1,0],[1,1]
print("rank =", np.linalg.matrix_rank(V), "-> independent?" , np.linalg.matrix_rank(V)==2)

t = np.array([5., 2.])             # express t in the basis: solve V c = t
c = np.linalg.solve(V, t)
print("coordinates of t:", c)      # [3. 2.]  -> t = 3*[1,0] + 2*[1,1]
print("rebuild:", V @ c)
` },
    { label: "Dependent vectors have a deficient rank", src: String.raw`
import numpy as np

W = np.array([[1.,2.,3.],
              [2.,4.,6.]]).T       # [1,2,3] and 2x that... actually columns:
cols = np.array([[1.,2.],[2.,4.],[3.,6.]])   # second column = 2 * first
print("rank =", np.linalg.matrix_rank(cols), "(only 1 independent direction)")
` }
  ],

  keyPoints: [
    "Independent ⇔ the only zero-combination is all-zero weights (no redundancy).",
    "A basis is a minimal spanning, independent set; dimension is its size.",
    "Coordinates in a basis are unique — one address per vector.",
    "$k$ vectors in $\\mathbb{R}^n$ with $k>n$ are always dependent.",
    "Testing independence = checking the matrix of vectors has full rank."
  ],

  commonMistakes: [
    { wrong: "Calling orthogonal the same as independent.", why: "Orthogonal vectors are always independent, but independent vectors need not be perpendicular. $[1,0]$ and $[1,1]$ are independent yet not orthogonal." },
    { wrong: "Thinking any spanning set is a basis.", why: "A spanning set can have extras. A basis must <em>also</em> be independent — the minimal spanning set." },
    { wrong: "Believing more vectors always means more dimensions.", why: "Three vectors in $\\mathbb{R}^2$ must be dependent — you can't exceed the space's dimension. Extra vectors are redundant." }
  ],

  quiz: [
    { q: "Are $[1,2]$ and $[2,4]$ independent?", options: ["No — one is twice the other", "Yes", "Only in 3-D", "Yes if normalized"], answer: 0,
      explain: "$[2,4]=2[1,2]$, so they're dependent and span only a line." },
    { q: "What is the dimension of the plane $\\{(x,y,0)\\}$ in $\\mathbb{R}^3$?", options: ["2", "3", "1", "0"], answer: 0,
      explain: "It needs two basis vectors (e.g. $[1,0,0],[0,1,0]$): dimension 2, a plane sitting inside 3-space." },
    { q: "Coordinates of $[5,2]$ in basis $\\{[1,0],[1,1]\\}$ are…", options: ["$(3,2)$", "$(5,2)$", "$(2,3)$", "$(7,2)$"], answer: 0,
      explain: "$c_2=2$, $c_1=5-2=3$: $3[1,0]+2[1,1]=[5,2]$." },
    { q: "Five vectors in $\\mathbb{R}^4$ are…", options: ["necessarily dependent", "necessarily independent", "a basis", "orthogonal"], answer: 0,
      explain: "More vectors than the dimension (4) forces dependence; at most 4 can be independent." },
    { q: "An embedding matrix's columns being independent means…", options: ["no embedding dimension is redundant", "the embeddings are orthogonal", "the model is overfit", "the rank is 1"], answer: 0,
      explain: "Independence means each dimension adds new information (full column rank); it does not require orthogonality." }
  ],

  practice: [
    { level: "easy", prompt: "Are $[1,0,0]$, $[0,1,0]$, $[0,0,1]$ a basis of $\\mathbb{R}^3$?", solution: "Yes — the standard basis: independent (each opens a new axis) and spanning. Dimension 3." },
    { level: "med", prompt: "Show $[2,1]$ and $[1,3]$ are independent.", solution: "$c_1[2,1]+c_2[1,3]=\\mathbf0$ gives $2c_1+c_2=0$ and $c_1+3c_2=0$. From the first $c_2=-2c_1$; sub: $c_1-6c_1=-5c_1=0\\Rightarrow c_1=0$, then $c_2=0$. Only trivial → independent." },
    { level: "med", prompt: "Write $[4,5]$ in the basis $\\{[2,1],[1,3]\\}$.", solution: "Solve $2c_1+c_2=4$, $c_1+3c_2=5$. From the first $c_2=4-2c_1$; sub: $c_1+12-6c_1=5\\Rightarrow-5c_1=-7\\Rightarrow c_1=1.4$, $c_2=1.2$. So $[4,5]=1.4[2,1]+1.2[1,3]$." },
    { level: "hard", prompt: "AI task: a 512-dim embedding layer is found to have rank 300 across the vocabulary. What does that imply, and is it good or bad?", solution: "Only 300 independent directions are actually used — 212 dimensions are linear combinations of the rest, so the embeddings live in a 300-dimensional subspace of the 512-dim space. It signals wasted capacity / partial rank collapse: the model could (in principle) represent the same information in 300 dims. Mild deficiency is normal and sometimes fine, but severe collapse hurts expressivity and is worth watching (it connects to anisotropy in language-model embeddings)." }
  ],

  deepDive: String.raw`<p><strong>Change of basis — the same vector, different coordinates.</strong></p>
  <p>A vector is a fixed arrow; its <em>coordinates</em> depend on the basis you measure with. If $B$ is the matrix
  whose columns are a basis, then a coordinate vector $\mathbf c$ in that basis corresponds to the actual vector
  $\mathbf v=B\mathbf c$, and going back is $\mathbf c=B^{-1}\mathbf v$. Choosing a clever basis can make a hard problem
  easy: this is the entire idea behind diagonalization and PCA in Track 5, where we switch to the
  <strong>eigenbasis</strong> so a matrix acts as simple independent stretches, and behind whitening, where we change to
  a basis that makes features uncorrelated.</p>
  <p>It also reframes what a neural network layer does: $W\mathbf x$ can be read as re-expressing $\mathbf x$ in a new,
  learned coordinate system. The "best" representations are ones where the basis aligns with the structure of the data —
  independent, well-spread directions. When those directions become redundant (rank drops, Lesson 3.2), the coordinate
  system folds in on itself and information is lost. Independence, basis, and dimension are the vocabulary for "is my
  representation actually using its space?"</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["3.4"] = {
  subtitle: "Perpendicularity, shadows onto a subspace, and how to straighten any basis.",

  aiMoment: String.raw`<p>Least-squares regression is a <strong>projection</strong>: it finds the point in the column
  space of $A$ closest to your data $\mathbf b$ — the shadow of $\mathbf b$ on what the model can represent.
  <strong>Orthogonal</strong> weight initialization keeps signal lengths stable through deep networks, and PCA's axes
  are orthogonal by construction. Projection and orthogonality are the geometry of "best approximation."</p>`,

  plainEnglish: String.raw`<p><strong>Orthogonal</strong> just means perpendicular. A <strong>projection</strong> of a
  vector onto a line or plane is its shadow — the closest point in that subspace, found by dropping a perpendicular.
  <strong>Gram–Schmidt</strong> is a procedure that takes any basis and straightens it into a perpendicular one.</p>`,

  intuition: String.raw`<p>To approximate $\mathbf b$ using only directions in a subspace, drop a perpendicular from
  $\mathbf b$ to the subspace. The foot of that perpendicular is the projection $\mathbf p$; the leftover
  $\mathbf b-\mathbf p$ is perpendicular to the subspace and is the smallest possible error.</p>
  <figure class="figure">
  <svg viewBox="0 0 280 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Projection of b onto a subspace with perpendicular residual">
    <line x1="30" y1="150" x2="250" y2="150" stroke="#0d9488" stroke-width="2.4"/>
    <text x="232" y="143" font-size="11" fill="#0d9488" font-family="sans-serif">subspace</text>
    <line x1="40" y1="150" x2="160" y2="50" stroke="#4f46e5" stroke-width="3" marker-end="url(#j1)"/>
    <text x="92" y="92" font-size="12" fill="#4f46e5" font-family="sans-serif">b</text>
    <line x1="40" y1="150" x2="160" y2="150" stroke="#7c3aed" stroke-width="3" marker-end="url(#j2)"/>
    <text x="96" y="166" font-size="12" fill="#7c3aed" font-family="sans-serif">p = proj b</text>
    <line x1="160" y1="150" x2="160" y2="50" stroke="#dc2626" stroke-width="2" stroke-dasharray="4 4"/>
    <text x="166" y="100" font-size="11" fill="#dc2626" font-family="sans-serif">b − p</text>
    <rect x="148" y="138" width="12" height="12" fill="none" stroke="#dc2626"/>
    <defs>
      <marker id="j1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="j2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#7c3aed"/></marker>
    </defs>
  </svg>
  <figcaption>The residual b − p meets the subspace at a right angle — that's what makes p the closest point.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Vectors are <strong>orthogonal</strong> when $\mathbf a\cdot\mathbf b=0$; <strong>orthonormal</strong>
  if also unit length. The <strong>projection of $\mathbf b$ onto the line through $\mathbf a$</strong> is</p>
  $$\operatorname{proj}_{\mathbf a}\mathbf b=\frac{\mathbf a\cdot\mathbf b}{\mathbf a\cdot\mathbf a}\,\mathbf a.$$
  <p>A matrix $Q$ with orthonormal columns satisfies $Q^\top Q=I$ and <strong>preserves length</strong>:
  $\lVert Q\mathbf x\rVert=\lVert\mathbf x\rVert$. <strong>Gram–Schmidt</strong> turns a basis $\{\mathbf a_1,\mathbf a_2,\dots\}$
  into an orthogonal one by subtracting off the projections onto what's already been built.</p>`,

  derivation: String.raw`<p><strong>Part 1 — the projection formula from "closest point."</strong> Project $\mathbf b$ onto
  the line $\{t\mathbf a\}$. We want the $t$ minimizing the error, equivalently making the residual perpendicular.</p>
  <p><strong>Step 1.</strong> Residual $\mathbf r=\mathbf b-t\mathbf a$ must satisfy $\mathbf a\cdot\mathbf r=0$ (perpendicular to the line).</p>
  <p><strong>Step 2.</strong> $\mathbf a\cdot(\mathbf b-t\mathbf a)=0\Rightarrow \mathbf a\cdot\mathbf b-t(\mathbf a\cdot\mathbf a)=0.$</p>
  <p><strong>Step 3.</strong> Solve: $t=\dfrac{\mathbf a\cdot\mathbf b}{\mathbf a\cdot\mathbf a}$, so
  $\mathbf p=t\mathbf a=\dfrac{\mathbf a\cdot\mathbf b}{\mathbf a\cdot\mathbf a}\mathbf a.\;\blacksquare$</p>
  <hr class="soft">
  <p><strong>Part 2 — Gram–Schmidt on two vectors.</strong> Given independent $\mathbf a_1,\mathbf a_2$:</p>
  <p><strong>Step 1.</strong> Keep the first: $\mathbf u_1=\mathbf a_1$.</p>
  <p><strong>Step 2.</strong> Remove from $\mathbf a_2$ its shadow along $\mathbf u_1$:
  $\mathbf u_2=\mathbf a_2-\dfrac{\mathbf u_1\cdot\mathbf a_2}{\mathbf u_1\cdot\mathbf u_1}\mathbf u_1.$ By Part 1 the
  leftover is perpendicular to $\mathbf u_1$.</p>
  <p><strong>Step 3.</strong> Normalize: $\mathbf q_i=\mathbf u_i/\lVert\mathbf u_i\rVert$. Now $\{\mathbf q_1,\mathbf q_2\}$
  is orthonormal and spans the same space. $\blacksquare$</p>`,

  code: [
    { label: "Projection and a perpendicular residual", src: String.raw`
import numpy as np

a = np.array([1., 0.])       # the line (x-axis)
b = np.array([2., 3.])
p = (a @ b) / (a @ a) * a     # projection of b onto a
r = b - p                     # residual
print("projection p =", p)            # [2. 0.]
print("residual  r  =", r)            # [0. 3.]
print("r ⟂ a ?      ", np.isclose(a @ r, 0.0))   # True
` },
    { label: "Gram–Schmidt → orthonormal columns", src: String.raw`
import numpy as np

a1 = np.array([1., 1., 0.])
a2 = np.array([1., 0., 1.])

u1 = a1
u2 = a2 - (u1 @ a2)/(u1 @ u1) * u1
q1, q2 = u1/np.linalg.norm(u1), u2/np.linalg.norm(u2)

Q = np.column_stack([q1, q2])
print("Q^T Q =\n", np.round(Q.T @ Q, 6))   # identity -> orthonormal
print("lengths preserved:", np.allclose(np.linalg.norm(Q @ np.array([2.,-1.])),
                                         np.linalg.norm(np.array([2.,-1.]))))
` }
  ],

  keyPoints: [
    "Orthogonal = perpendicular ($\\mathbf a\\cdot\\mathbf b=0$); orthonormal adds unit length.",
    "$\\operatorname{proj}_{\\mathbf a}\\mathbf b=\\frac{\\mathbf a\\cdot\\mathbf b}{\\mathbf a\\cdot\\mathbf a}\\mathbf a$ is the closest point on the line.",
    "The residual $\\mathbf b-\\mathbf p$ is perpendicular to the subspace — that's why it's the least error.",
    "Orthogonal matrices ($Q^\\top Q=I$) preserve length and angles.",
    "Gram–Schmidt straightens any basis into an orthonormal one."
  ],

  commonMistakes: [
    { wrong: "Forgetting to divide by $\\mathbf a\\cdot\\mathbf a$ in the projection.", why: "The formula needs the normalization $\\mathbf a\\cdot\\mathbf a=\\lVert\\mathbf a\\rVert^2$. Only if $\\mathbf a$ is already unit length does it drop out." },
    { wrong: "Assuming any square matrix preserves length.", why: "Only orthogonal matrices ($Q^\\top Q=I$) do. A general matrix stretches and skews — which is why orthogonal init is special." },
    { wrong: "Running Gram–Schmidt naively in finite precision.", why: "Classical Gram–Schmidt loses orthogonality to rounding error; the modified version (or a QR routine, Lesson 3.5) is the numerically stable choice." }
  ],

  quiz: [
    { q: "Project $\\mathbf b=[3,4]$ onto $\\mathbf a=[1,0]$.", options: ["$[3,0]$", "$[0,4]$", "$[3,4]$", "$[1,0]$"], answer: 0,
      explain: "$\\frac{\\mathbf a\\cdot\\mathbf b}{\\mathbf a\\cdot\\mathbf a}\\mathbf a=\\frac{3}{1}[1,0]=[3,0]$ — the shadow on the x-axis." },
    { q: "If $Q^\\top Q=I$, then $\\lVert Q\\mathbf x\\rVert$ equals…", options: ["$\\lVert\\mathbf x\\rVert$", "$\\lVert\\mathbf x\\rVert^2$", "$1$", "$\\lVert Q\\rVert\\,\\lVert\\mathbf x\\rVert$"], answer: 0,
      explain: "$\\lVert Q\\mathbf x\\rVert^2=\\mathbf x^\\top Q^\\top Q\\mathbf x=\\mathbf x^\\top\\mathbf x=\\lVert\\mathbf x\\rVert^2$ — length is preserved." },
    { q: "The residual $\\mathbf b-\\operatorname{proj}\\mathbf b$ is…", options: ["orthogonal to the subspace", "parallel to the subspace", "always zero", "longer than $\\mathbf b$"], answer: 0,
      explain: "By construction the residual is perpendicular to the subspace; that orthogonality is what minimizes its length." },
    { q: "After Gram–Schmidt, $\\mathbf u_2=\\mathbf a_2-\\operatorname{proj}_{\\mathbf u_1}\\mathbf a_2$ satisfies…", options: ["$\\mathbf u_1\\cdot\\mathbf u_2=0$", "$\\mathbf u_2=\\mathbf a_2$", "$\\mathbf u_2=\\mathbf 0$", "$\\lVert\\mathbf u_2\\rVert=1$"], answer: 0,
      explain: "Subtracting the projection removes the component along $\\mathbf u_1$, leaving $\\mathbf u_2\\perp\\mathbf u_1$ (not yet unit length)." },
    { q: "Why are orthogonal matrices used for weight init in deep nets?", options: ["They preserve signal norm, avoiding vanishing/exploding activations", "They are faster to multiply", "They have rank 1", "They make the loss convex"], answer: 0,
      explain: "Length preservation keeps activation and gradient norms stable across many layers, mitigating vanishing/exploding signals." }
  ],

  practice: [
    { level: "easy", prompt: "Are $[1,1]$ and $[1,-1]$ orthogonal? Normalize them.", solution: "$[1,1]\\cdot[1,-1]=1-1=0$ → orthogonal. Each has length $\\sqrt2$, so the orthonormal pair is $\\frac{1}{\\sqrt2}[1,1]$ and $\\frac{1}{\\sqrt2}[1,-1]$." },
    { level: "med", prompt: "Project $[2,2]$ onto the line through $[3,1]$.", solution: "$\\frac{[3,1]\\cdot[2,2]}{[3,1]\\cdot[3,1]}[3,1]=\\frac{6+2}{9+1}[3,1]=\\frac{8}{10}[3,1]=[2.4,0.8]$." },
    { level: "med", prompt: "Apply Gram–Schmidt to $\\mathbf a_1=[3,0]$, $\\mathbf a_2=[1,2]$ (don't normalize).", solution: "$\\mathbf u_1=[3,0]$. $\\mathbf u_2=[1,2]-\\frac{[3,0]\\cdot[1,2]}{9}[3,0]=[1,2]-\\frac{3}{9}[3,0]=[1,2]-[1,0]=[0,2]$. Check $\\mathbf u_1\\cdot\\mathbf u_2=0.\\;\\checkmark$" },
    { level: "hard", prompt: "AI task: least-squares fits $\\hat{\\mathbf b}=A\\mathbf x$ closest to $\\mathbf b$. Explain why the optimal $\\mathbf x$ satisfies $A^\\top(\\mathbf b-A\\mathbf x)=0$ and what it means geometrically.", solution: "The best $A\\mathbf x$ is the projection of $\\mathbf b$ onto $\\operatorname{col}(A)$, so the residual $\\mathbf b-A\\mathbf x$ must be orthogonal to every column of $A$ — i.e. $A^\\top(\\mathbf b-A\\mathbf x)=\\mathbf 0$. Rearranged, that's the normal equations $A^\\top A\\mathbf x=A^\\top\\mathbf b$. Geometrically: you can't reduce the error by moving within the column space, because the leftover error points straight out of it." }
  ],

  deepDive: String.raw`<p><strong>Least-squares <em>is</em> projection — and why orthonormal bases make it trivial.</strong></p>
  <p>Fitting $A\mathbf x\approx\mathbf b$ when there are more equations than unknowns has no exact answer, so we settle
  for the closest reachable point: the projection of $\mathbf b$ onto $\operatorname{col}(A)$. Demanding the residual be
  orthogonal to that subspace gives the normal equations $A^\top A\mathbf x=A^\top\mathbf b$ — the same equations from
  Lesson 3.1. So "best linear fit" and "drop a perpendicular" are literally the same computation.</p>
  <p>Here's the payoff of orthonormality. If the columns of $A$ are orthonormal (call it $Q$), then $Q^\top Q=I$ and the
  solution collapses to $\mathbf x=Q^\top\mathbf b$ — no matrix to invert, each coordinate is just a dot product. That's
  why algorithms first orthonormalize (Gram–Schmidt → QR, Lesson 3.5) and why orthonormal bases are the comfortable
  place to compute. It's also the deeper reason PCA uses orthogonal axes: in an orthonormal basis, projecting,
  reconstructing, and measuring variance all become independent one-dimensional operations instead of a coupled mess.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["3.5"] = {
  subtitle: "Split any matrix into a clean rotation and an upper-triangular mixer — Gram–Schmidt as a factorization.",

  aiMoment: String.raw`<p>When a library fits a linear regression, it usually does <strong>not</strong> form the normal
  equations $A^\top A\mathbf x=A^\top\mathbf b$ — that squares the conditioning and loses precision. Instead it computes
  the <strong>QR decomposition</strong> $A=QR$ and solves a triangular system. QR is also the heart of the standard
  algorithm for computing eigenvalues. It's Gram–Schmidt (Lesson 3.4) repackaged into two clean factors.</p>`,

  plainEnglish: String.raw`<p>QR breaks a matrix into two pieces: $Q$, whose columns are perpendicular unit vectors
  (a pure rotation/reflection), and $R$, an upper-triangular matrix that records how to mix those clean directions back
  into the originals. Because $Q$ is orthonormal and $R$ is triangular, anything built on $A$ becomes easy and stable.</p>`,

  intuition: String.raw`<p>Run Gram–Schmidt on the columns of $A$: you get orthonormal directions $Q$, and the
  bookkeeping of "how much of each clean direction rebuilds each original column" is exactly the triangular $R$.</p>
  <figure class="figure">
  <svg viewBox="0 0 330 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A equals Q times R">
    <rect x="20" y="25" width="64" height="80" rx="4" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="46" y="70" font-size="16" fill="#4f46e5" font-family="sans-serif">A</text>
    <text x="96" y="70" font-size="18" font-family="sans-serif">=</text>
    <rect x="120" y="25" width="64" height="80" rx="4" fill="#f0fdfa" stroke="#0d9488"/>
    <line x1="128" y1="33" x2="128" y2="97" stroke="#0d9488"/><line x1="142" y1="33" x2="142" y2="97" stroke="#0d9488"/>
    <text x="138" y="118" font-size="10" fill="#0d9488" font-family="sans-serif">Q  (⟂ cols)</text>
    <text x="196" y="70" font-size="18" font-family="sans-serif">·</text>
    <rect x="214" y="25" width="80" height="80" rx="4" fill="#fff7ed" stroke="#d97706"/>
    <path d="M222,33 L286,33 L286,97 Z" fill="#fde9cf"/>
    <text x="236" y="52" font-size="10" fill="#d97706" font-family="sans-serif">≠0</text>
    <text x="232" y="92" font-size="10" fill="#94a3b8" font-family="sans-serif">0s</text>
    <text x="226" y="118" font-size="10" fill="#d97706" font-family="sans-serif">R (upper △)</text>
  </svg>
  <figcaption>Q holds orthonormal directions; R (upper-triangular, zeros below the diagonal) mixes them back into A.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $A\in\mathbb{R}^{m\times n}$ with independent columns, $A=QR$ where $Q\in\mathbb{R}^{m\times n}$
  has orthonormal columns ($Q^\top Q=I$) and $R\in\mathbb{R}^{n\times n}$ is upper triangular with $R_{ij}=\mathbf q_i\cdot\mathbf a_j$.
  To solve least-squares $A\mathbf x\approx\mathbf b$: substitute $QR\mathbf x=\mathbf b$, left-multiply by $Q^\top$
  (using $Q^\top Q=I$) to get</p>
  $$R\mathbf x=Q^\top\mathbf b,$$
  <p>a triangular system solved instantly by back-substitution. Plain English: rotate the problem with $Q$, then solve
  a tidy triangular system.</p>`,

  derivation: String.raw`<p><strong>QR is Gram–Schmidt written as a product.</strong> Take $A$ with columns $\mathbf a_1,\mathbf a_2$.</p>
  <p><strong>Step 1 — orthonormalize</strong> (Lesson 3.4): get $\mathbf q_1,\mathbf q_2$ orthonormal spanning the same
  space.</p>
  <p><strong>Step 2 — express each original column in the new basis.</strong> Since $\mathbf a_1$ points along
  $\mathbf q_1$: $\mathbf a_1=(\mathbf q_1\cdot\mathbf a_1)\,\mathbf q_1$. And $\mathbf a_2$ lies in
  $\operatorname{span}(\mathbf q_1,\mathbf q_2)$: $\mathbf a_2=(\mathbf q_1\cdot\mathbf a_2)\mathbf q_1+(\mathbf q_2\cdot\mathbf a_2)\mathbf q_2$.</p>
  <p><strong>Step 3 — read off $R$.</strong> Collect the coefficients:
  $R=\begin{bmatrix}\mathbf q_1\cdot\mathbf a_1 & \mathbf q_1\cdot\mathbf a_2\\ 0 & \mathbf q_2\cdot\mathbf a_2\end{bmatrix}.$
  The lower-left is $0$ because $\mathbf a_1$ has no $\mathbf q_2$ component — that's <em>why</em> $R$ is upper
  triangular.</p>
  <p><strong>Step 4 — assemble.</strong> Stacking the two column equations is precisely $A=QR$. $\blacksquare$ Plain
  English: orthonormal directions go in $Q$; the triangular recipe to rebuild $A$ from them is $R$.</p>`,

  code: [
    { label: "QR factorization and reconstruction", src: String.raw`
import numpy as np

A = np.array([[1., 1.],
              [1., 0.],
              [0., 1.]])
Q, R = np.linalg.qr(A)
print("Q^T Q =\n", np.round(Q.T @ Q, 6))   # identity -> orthonormal cols
print("R (upper triangular) =\n", np.round(R, 4))
print("A == Q R ?", np.allclose(A, Q @ R))
` },
    { label: "Solve least-squares via QR (stable)", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

A = rng.normal(size=(20, 3))      # 20 equations, 3 unknowns (over-determined)
b = rng.normal(size=20)

Q, R = np.linalg.qr(A)
x_qr = np.linalg.solve(R, Q.T @ b)        # R x = Q^T b  (back-substitution)
x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
print("QR solution    :", np.round(x_qr, 4))
print("matches lstsq? :", np.allclose(x_qr, x_ref))
` }
  ],

  keyPoints: [
    "$A=QR$: $Q$ has orthonormal columns ($Q^\\top Q=I$), $R$ is upper triangular.",
    "QR is Gram–Schmidt packaged as a factorization; $R_{ij}=\\mathbf q_i\\cdot\\mathbf a_j$.",
    "Solve least-squares by $R\\mathbf x=Q^\\top\\mathbf b$ (back-substitution) — no $A^\\top A$.",
    "QR avoids squaring the condition number, so it's the stable way to fit regression.",
    "The QR algorithm (repeated QR) is how eigenvalues are computed in practice."
  ],

  commonMistakes: [
    { wrong: "Defaulting to the normal equations $A^\\top A\\mathbf x=A^\\top\\mathbf b$.", why: "Forming $A^\\top A$ squares the condition number, amplifying rounding error. QR keeps the conditioning of $A$ itself — far safer for nearly-collinear features." },
    { wrong: "Expecting $R$ to be lower triangular.", why: "$R$ is <em>upper</em> triangular: column $\\mathbf a_j$ only uses $\\mathbf q_1,\\dots,\\mathbf q_j$, so entries below the diagonal are 0." },
    { wrong: "Forgetting $Q$ may be 'thin'.", why: "For a tall $A$ ($m>n$), reduced QR gives $Q\\in\\mathbb{R}^{m\\times n}$ with orthonormal columns — $Q Q^\\top\\neq I$, only $Q^\\top Q=I$." }
  ],

  quiz: [
    { q: "In $A=QR$, the matrix $R$ is…", options: ["upper triangular", "orthogonal", "diagonal", "lower triangular"], answer: 0,
      explain: "$R$ is upper triangular; $Q$ is the one with orthonormal columns." },
    { q: "Which property does $Q$ satisfy?", options: ["$Q^\\top Q=I$", "$Q^\\top Q=0$", "$Q=Q^\\top$", "$Q^2=I$"], answer: 0,
      explain: "Orthonormal columns mean $Q^\\top Q=I$ (length- and angle-preserving)." },
    { q: "To solve least-squares with QR you solve…", options: ["$R\\mathbf x=Q^\\top\\mathbf b$", "$Q\\mathbf x=R\\mathbf b$", "$A^\\top A\\mathbf x=\\mathbf b$", "$R^\\top\\mathbf x=Q\\mathbf b$"], answer: 0,
      explain: "From $QR\\mathbf x=\\mathbf b$, multiply by $Q^\\top$: $R\\mathbf x=Q^\\top\\mathbf b$, then back-substitute." },
    { q: "Why prefer QR over the normal equations numerically?", options: ["It avoids squaring the condition number", "It is always faster", "It gives a different answer", "It needs no memory"], answer: 0,
      explain: "Normal equations use $A^\\top A$, whose condition number is $\\kappa(A)^2$; QR works with $\\kappa(A)$, preserving precision." },
    { q: "If $A$ is $5\\times2$ with independent columns, the shape of (reduced) $Q$ is…", options: ["$5\\times2$", "$2\\times2$", "$5\\times5$", "$2\\times5$"], answer: 0,
      explain: "Reduced QR: $Q$ matches $A$'s shape $5\\times2$ with orthonormal columns; $R$ is $2\\times2$." }
  ],

  practice: [
    { level: "easy", prompt: "What does $Q^\\top Q=I$ tell you about the columns of $Q$?", solution: "They are orthonormal: each has unit length and every pair is orthogonal (off-diagonal dot products are 0)." },
    { level: "med", prompt: "Given the QR of $A$, write the least-squares solution and the one matrix operation that dominates the cost.", solution: "$\\mathbf x=R^{-1}Q^\\top\\mathbf b$, computed as: form $Q^\\top\\mathbf b$ (a matrix–vector product), then back-substitute through the triangular $R$. Back-substitution on $R$ is cheap ($O(n^2)$); the QR factorization itself ($O(mn^2)$) dominates." },
    { level: "med", prompt: "For $A=\\begin{bmatrix}1&1\\\\0&1\\end{bmatrix}$, find $Q$ and $R$ by Gram–Schmidt.", solution: "$\\mathbf a_1=[1,0]$ is already unit: $\\mathbf q_1=[1,0]$. $\\mathbf a_2=[1,1]$; subtract $(\\mathbf q_1\\cdot\\mathbf a_2)\\mathbf q_1=1\\cdot[1,0]=[1,0]$ → $\\mathbf u_2=[0,1]$, unit already: $\\mathbf q_2=[0,1]$. So $Q=I$ and $R=\\begin{bmatrix}1&1\\\\0&1\\end{bmatrix}=A$ (A was already orthonormal-columned)." },
    { level: "hard", prompt: "AI task: features in a regression are nearly collinear, giving condition number $\\kappa(A)\\approx10^{6}$. Explain in numbers why QR beats the normal equations in float64.", solution: "float64 has ~16 significant digits. The normal equations operate on $A^\\top A$ with condition number $\\kappa(A)^2\\approx10^{12}$, so you can lose ~12 of 16 digits — only ~4 reliable digits remain. QR works with $\\kappa(A)\\approx10^{6}$, losing ~6 digits and keeping ~10 — a far more accurate fit. The lesson: never square the conditioning if you can factor instead." }
  ],

  deepDive: String.raw`<p><strong>Conditioning: the precise reason solvers avoid $A^\top A$.</strong></p>
  <p>The <em>condition number</em> $\kappa(A)$ measures how much relative error a linear solve can amplify (Track 4
  makes it formal). The normal equations require $A^\top A$, and a basic fact is $\kappa(A^\top A)=\kappa(A)^2$. So if
  $A$ is even moderately ill-conditioned — say $\kappa(A)=10^4$, common with correlated features — then $A^\top A$ has
  $\kappa=10^8$, and in float32 (≈7 digits) the solve can be garbage. QR sidesteps this: it never forms $A^\top A$, so
  it inherits $\kappa(A)$, not its square.</p>
  <p>This is a recurring theme you'll see again in Track 13: the <em>mathematically</em> cleanest formula (here, the
  normal equations) is often the <em>numerically</em> worst. The same matrix can be factored many ways — LU, Cholesky,
  QR, SVD — and choosing the right factorization for the conditioning at hand is a real engineering skill. QR's blend of
  stability and modest cost is why it's the default workhorse behind least-squares, and why the iterative QR algorithm,
  applied over and over, is how production libraries find eigenvalues (Track 5).</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["3.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 65 minutes.",

  intro: String.raw`<p>This exam spans all of Track 3: solving systems, rank / column space / null space, independence
  and basis, orthogonality / projection / Gram–Schmidt, and QR. <strong>Give yourself 65 minutes</strong> and produce
  each answer before checking. About half are calculation. Score with the rubric at the end.</p>`,

  problems: [
    { level: "easy", prompt: "Solve $x+2y=4,\\;3x-y=5$ by elimination.",
      solution: "From the first, $x=4-2y$; sub: $3(4-2y)-y=5\\Rightarrow12-7y=5\\Rightarrow y=1$, $x=2$. Solution $(2,1)$." },
    { level: "easy", prompt: "State the rank and nullity of $\\begin{bmatrix}1&2\\\\3&6\\end{bmatrix}$.",
      solution: "Row 2 = 3× row 1 → rank 1. Columns $n=2$, so nullity $=2-1=1$ (null space spanned by $[2,-1]$)." },
    { level: "med", prompt: "Find a basis for the null space of $A=\\begin{bmatrix}1&1&1\\\\0&1&2\\end{bmatrix}$.",
      solution: "Free variable $z$. Row 2: $y+2z=0\\Rightarrow y=-2z$. Row 1: $x+y+z=0\\Rightarrow x=-y-z=2z-z=z$. So $\\mathbf x=z[1,-2,1]$; basis $\\{[1,-2,1]\\}$, nullity 1 (rank 2)." },
    { level: "med", prompt: "Are $[1,2,3]$, $[2,4,6]$, $[0,1,0]$ independent? Give the rank.",
      solution: "The second is twice the first, so they're dependent. Independent ones: $[1,2,3]$ and $[0,1,0]$ → rank 2." },
    { level: "med", prompt: "Project $\\mathbf b=[1,2,2]$ onto $\\mathbf a=[0,3,4]$.",
      solution: "$\\frac{\\mathbf a\\cdot\\mathbf b}{\\mathbf a\\cdot\\mathbf a}\\mathbf a=\\frac{0+6+8}{0+9+16}[0,3,4]=\\frac{14}{25}[0,3,4]=[0,1.68,2.24]$." },
    { level: "med", prompt: "Orthogonalize $\\mathbf a_1=[2,0]$, $\\mathbf a_2=[3,4]$ with Gram–Schmidt (no normalizing).",
      solution: "$\\mathbf u_1=[2,0]$. $\\mathbf u_2=[3,4]-\\frac{[2,0]\\cdot[3,4]}{4}[2,0]=[3,4]-\\frac{6}{4}[2,0]=[3,4]-[3,0]=[0,4]$. Orthogonal: $[2,0]\\cdot[0,4]=0.\\;\\checkmark$" },
    { level: "med", prompt: "For which $k$ is $\\{[1,2],[2,k]\\}$ a basis of $\\mathbb{R}^2$?",
      solution: "They're a basis iff independent iff not parallel iff the determinant $\\ne0$: $1\\cdot k-2\\cdot2=k-4\\ne0$, so all $k\\ne4$. (At $k=4$ the second is twice the first.)" },
    { level: "hard", prompt: "An over-determined system $A\\mathbf x=\\mathbf b$ ($A$ is $100\\times3$) has no exact solution. What does least-squares return, and what equation characterizes it?",
      solution: "It returns the $\\mathbf x$ whose $A\\mathbf x$ is the projection of $\\mathbf b$ onto $\\operatorname{col}(A)$ — the closest reachable point. It's characterized by an orthogonal residual: $A^\\top(\\mathbf b-A\\mathbf x)=\\mathbf 0$, i.e. the normal equations $A^\\top A\\mathbf x=A^\\top\\mathbf b$." },
    { level: "hard", prompt: "Explain why a library solves regression with QR ($R\\mathbf x=Q^\\top\\mathbf b$) instead of the normal equations.",
      solution: "Normal equations require $A^\\top A$, whose condition number is $\\kappa(A)^2$ — squaring any ill-conditioning and destroying precision with correlated features. QR factors $A=QR$ directly and solves a triangular system, working with $\\kappa(A)$ itself. Same answer in exact arithmetic, far more accurate in floating point." },
    { level: "hard", prompt: "AI task: a $4096\\times4096$ weight matrix is fine-tuned with rank-16 LoRA. Give the parameter count vs full fine-tuning and the rank fact that justifies it.",
      solution: "LoRA stores $B\\,(4096\\times16)$ and $A\\,(16\\times4096)$: $2\\cdot4096\\cdot16=131{,}072$ parameters, versus $4096^2\\approx16.8$M for the full matrix — about a $128\\times$ reduction. Justification: $\\operatorname{rank}(BA)\\le16$, and the assumption is that the fine-tuning update $\\Delta W$ has low intrinsic rank, so 16 independent directions capture it." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Spaces and rank are second nature. On to Track 4 (Norms, Distances & Inverses).</li>
    <li><strong>7–8:</strong> Strong. Revisit the projection ↔ least-squares link if that one slipped.</li>
    <li><strong>5–6:</strong> Re-derive rank–nullity and the projection formula by hand; redo Lessons 3.2 and 3.4.</li>
    <li><strong>Below 5:</strong> Rework the track in order — these ideas (rank, projection, orthogonality) recur in PCA, SVD, and least-squares everywhere ahead.</li>
  </ul>`
};
