/* ============================================================
   TRACK 2 — Linear Algebra I — Vectors & Matrices
   Openers: 2.1 Vectors, 2.3 The Dot Product.
   ============================================================ */
(window.LESSON_CONTENT ||= {})["2.1"] = {
  subtitle: "A vector is a list of numbers and an arrow in space — the atom of every model.",

  aiMoment: String.raw`<p>When GPT reads the word "king", it looks up a <strong>vector</strong> — an ordered list of
  maybe 768 numbers — in its embedding table. Words with similar meaning land near each other, which is why the
  famous arithmetic <em>king − man + woman ≈ queen</em> works: it's literal vector subtraction and addition. A whole
  batch of token vectors stacked together is a <strong>matrix</strong>; a batch of batches is a <strong>tensor</strong>.
  Everything a model touches is one of these objects.</p>`,

  plainEnglish: String.raw`<p>A <strong>scalar</strong> is a single number. A <strong>vector</strong> is an ordered list
  of numbers — and you can also picture it as an arrow pointing from the origin to a spot in space. A
  <strong>matrix</strong> is a grid of numbers (a stack of vectors). That's the whole vocabulary.</p>`,

  intuition: String.raw`<p>The vector $[3,2]$ is an arrow: walk 3 right, then 2 up. Its two ingredients are a
  <strong>direction</strong> (which way it points) and a <strong>length</strong> (how far). In a model's embedding
  space the direction carries the meaning.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A 2-D vector as an arrow with components">
    <line x1="30" y1="180" x2="240" y2="180" stroke="#cbd5e1"/>
    <line x1="30" y1="180" x2="30" y2="20" stroke="#cbd5e1"/>
    <line x1="30" y1="180" x2="180" y2="80" stroke="#4f46e5" stroke-width="3" marker-end="url(#v1)"/>
    <line x1="30" y1="180" x2="180" y2="180" stroke="#94a3b8" stroke-dasharray="4 4"/>
    <line x1="180" y1="180" x2="180" y2="80" stroke="#94a3b8" stroke-dasharray="4 4"/>
    <text x="100" y="120" fill="#4f46e5" font-family="sans-serif" font-size="13">v = [3, 2]</text>
    <text x="98" y="197" fill="#64748b" font-family="sans-serif" font-size="11">vₓ = 3</text>
    <text x="184" y="135" fill="#64748b" font-family="sans-serif" font-size="11">v_y = 2</text>
    <defs><marker id="v1" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker></defs>
  </svg>
  <figcaption>Components are the shadow on each axis; together they pin down the arrow.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A vector in $\mathbb{R}^n$ is written $\mathbf{v}=[v_1,v_2,\dots,v_n]$, where each $v_i$ is a
  <em>component</em> and $n$ is the <em>dimension</em>. We treat vectors as <strong>columns</strong> by default:</p>
  $$\mathbf{v}=\begin{bmatrix}v_1\\v_2\\\vdots\\v_n\end{bmatrix}\in\mathbb{R}^{n}, \qquad
  A=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{bmatrix}\in\mathbb{R}^{2\times2}.$$
  <p>$A_{ij}$ is the entry in row $i$, column $j$. A matrix in $\mathbb{R}^{m\times n}$ has $m$ rows and $n$ columns;
  its shape is the pair $(m,n)$. A <em>tensor</em> just adds more axes (e.g. a batch of images is $4$-D:
  batch × channel × height × width).</p>`,

  derivation: String.raw`<p><strong>The two pictures of a vector agree.</strong> We claim that adding vectors
  "component by component" is the same as the geometric "tip-to-tail" rule. Take $\mathbf{a}=[a_1,a_2]$ and
  $\mathbf{b}=[b_1,b_2]$.</p>
  <p><strong>Step 1 — algebra.</strong> Define the sum componentwise: $\mathbf{a}+\mathbf{b}=[a_1+b_1,\;a_2+b_2]$.</p>
  <p><strong>Step 2 — geometry.</strong> Tip-to-tail says: walk along $\mathbf{a}$, then from there walk along
  $\mathbf{b}$. Walking $\mathbf{a}$ moves you $(a_1,a_2)$. Continuing along $\mathbf{b}$ adds another $(b_1,b_2)$ to
  your position.</p>
  <p><strong>Step 3 — compare.</strong> Your final position is $(a_1+b_1,\;a_2+b_2)$ — the displacements on each axis
  simply add. That is exactly the Step-1 vector. $\blacksquare$ Plain English: the arrow rule and the list rule are
  the same arithmetic seen two ways. Scalar multiplication $c\mathbf{a}=[ca_1,ca_2]$ likewise stretches the arrow's
  length by $|c|$ (and flips it if $c\lt0$).</p>`,

  code: [
    { label: "Vectors, matrices, and shapes in NumPy", src: String.raw`
import numpy as np

v = np.array([3.0, 2.0])               # a vector in R^2
A = np.array([[1.0, 2.0],
              [3.0, 4.0]])             # a 2x2 matrix
print("v      =", v, " shape", v.shape)      # (2,)
print("A shape =", A.shape)                  # (2, 2)
print("A[1,0] =", A[1, 0])                   # row 1, col 0  -> 3.0

# word arithmetic is just vector arithmetic
king, man, woman = np.array([0.7,0.2]), np.array([0.6,0.1]), np.array([0.1,0.9])
print("king - man + woman =", king - man + woman)   # ~ 'queen'
` },
    { label: "An embedding table is a matrix you index into", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

vocab, d = 5, 4
E = rng.normal(size=(vocab, d))     # 5 tokens, each a length-4 vector
token_ids = [0, 3, 3, 1]            # "the cat cat sat"
batch = E[token_ids]               # fancy indexing -> shape (4, d)
print("embedding matrix E shape:", E.shape)
print("batch of token vectors  :", batch.shape)
` }
  ],

  keyPoints: [
    "A vector is an ordered list of numbers and equivalently an arrow (direction + length).",
    "A matrix is a grid / stack of vectors; a tensor adds more axes (e.g. a batch).",
    "Shape $(m,n)$ = $m$ rows by $n$ columns; $A_{ij}$ is row $i$, column $j$.",
    "Vector addition is componentwise and matches the geometric tip-to-tail rule.",
    "Embeddings are vectors; semantic relationships show up as vector arithmetic."
  ],

  commonMistakes: [
    { wrong: "Treating shape $(n,)$ and $(n,1)$ as interchangeable.", why: "A 1-D array $(n,)$ has no second axis; an $(n,1)$ column does. Mixing them triggers broadcasting surprises in matrix math — a frequent source of silent bugs." },
    { wrong: "Confusing a Python list with a NumPy array.", why: "<code>[1,2]+[3,4]</code> concatenates to $[1,2,3,4]$; <code>np.array([1,2])+np.array([3,4])</code> adds to $[4,6]$. Vector math needs arrays." },
    { wrong: "Forgetting whether your vectors are rows or columns.", why: "It flips which dimension a matrix multiply contracts. Fix a convention (columns here) and keep shapes explicit." }
  ],

  quiz: [
    { q: "What is $2[1,-3]+[0,4]$?", options: ["$[2,-2]$", "$[2,1]$", "$[1,1]$", "$[2,-6]$"], answer: 0,
      explain: "Scale first: $[2,-6]$, then add $[0,4]$ componentwise: $[2,-2]$. Choice $[2,-6]$ forgot to add." },
    { q: "A matrix has shape $(3,5)$. How many rows and columns?", options: ["3 rows, 5 columns", "5 rows, 3 columns", "3 rows, 3 columns", "15 of each"], answer: 0,
      explain: "Convention is (rows, columns): 3 rows, 5 columns, 15 entries total." },
    { q: "An embedding table for 50k tokens with dimension 768 has what shape?", options: ["$(50000,768)$", "$(768,50000)$", "$(50000,)$", "$(768,)$"], answer: 0,
      explain: "One row per token, each row a 768-vector: $(50000,768)$. Indexing a token id returns a length-768 vector." },
    { q: "Which equals the arrow from the tip of $\\mathbf a$ to the tip of $\\mathbf b$ (both from origin)?", options: ["$\\mathbf b-\\mathbf a$", "$\\mathbf a-\\mathbf b$", "$\\mathbf a+\\mathbf b$", "$-(\\mathbf a+\\mathbf b)$"], answer: 0,
      explain: "To go from $\\mathbf a$'s tip to $\\mathbf b$'s tip you add $\\mathbf b-\\mathbf a$ (since $\\mathbf a+(\\mathbf b-\\mathbf a)=\\mathbf b$)." },
    { q: "In NumPy, <code>np.array([1,2,3]).shape</code> is…", options: ["$(3,)$", "$(1,3)$", "$(3,1)$", "$3$"], answer: 0,
      explain: "A 1-D array reports a one-element shape tuple $(3,)$ — not a 2-D row or column." }
  ],

  practice: [
    { level: "easy", prompt: "Write the vector that moves 5 left and 2 up, then give its components.", solution: "$[-5,2]$: $v_x=-5$ (left is negative $x$), $v_y=2$." },
    { level: "easy", prompt: "Compute $[2,0,-1]+[1,1,1]$ and $3[1,2]$.", solution: "$[3,1,0]$ (componentwise) and $[3,6]$ (scale each entry)." },
    { level: "med", prompt: "Stack the column vectors $[1,2]$ and $[3,4]$ into a matrix two ways; give both shapes.", solution: "As columns: $\\begin{bmatrix}1&3\\\\2&4\\end{bmatrix}$, shape $(2,2)$. As rows: $\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}$, also $(2,2)$ but a different matrix — column-stacking is the transpose of row-stacking." },
    { level: "hard", prompt: "AI task: you have 32 sentences, each padded to 128 tokens, embedded in dimension 512. Give the shape of the batch tensor and the meaning of each axis.", solution: "$(32,128,512)$: axis 0 = which sentence (batch), axis 1 = token position, axis 2 = the 512 embedding features. This is the exact shape that enters a transformer block." }
  ],

  deepDive: String.raw`<p><strong>The $(n,)$ vs $(n,1)$ trap, and why broadcasting both saves and bites you.</strong></p>
  <p>NumPy lets a $(3,)$ vector and a $(3,1)$ column interact, but the result can surprise you. Subtracting a $(n,)$
  row-shaped mean from an $(m,n)$ data matrix is what you want (it centers each column). But accidentally giving the
  mean shape $(m,1)$ makes NumPy broadcast it against the wrong axis and silently produce a different $(m,n)$ result —
  no error, just wrong numbers.</p>
  <p>The rule: broadcasting compares shapes from the <em>right</em>, stretching any axis of length 1. $(m,n)$ with
  $(n,)$ aligns $n$ with $n$ — correct. $(m,n)$ with $(m,1)$ aligns over rows — also valid but different. Whenever a
  result has an unexpected shape, print <code>.shape</code> at every step; in real models a misaligned broadcast is
  the bug that trains to a plausible-but-wrong loss. You'll formalize these rules in Lesson 2.5.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["2.3"] = {
  subtitle: "Multiply matching entries, add them up — the single number behind every attention score.",

  aiMoment: String.raw`<p>In a transformer, how much token $i$ "attends to" token $j$ is a <strong>dot product</strong>
  of a query vector $\mathbf q_i$ with a key vector $\mathbf k_j$. Bigger dot product → more alignment → more
  attention. The same operation powers semantic search (cosine similarity between embeddings) and the logits of a
  linear classifier ($\mathbf w\cdot\mathbf x$). Learn this one number and a surprising amount of deep learning
  stops being mysterious.</p>`,

  plainEnglish: String.raw`<p>The dot product takes two equal-length vectors, multiplies them entry by entry, and adds
  up the results into a single number. That number is large and positive when the vectors point the same way, zero
  when they're perpendicular, and negative when they point apart. It measures <strong>alignment</strong>.</p>`,

  intuition: String.raw`<p>Geometrically, the dot product is "how much of one vector lies along the other," scaled by
  their lengths. Drop a perpendicular from $\mathbf a$ onto $\mathbf b$: the length of that shadow, times $|\mathbf b|$,
  is the dot product.</p>
  <figure class="figure">
  <svg viewBox="0 0 280 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dot product as projection">
    <line x1="40" y1="160" x2="240" y2="160" stroke="#0d9488" stroke-width="3" marker-end="url(#dd1)"/>
    <line x1="40" y1="160" x2="170" y2="70" stroke="#4f46e5" stroke-width="3" marker-end="url(#dd2)"/>
    <line x1="170" y1="70" x2="170" y2="160" stroke="#94a3b8" stroke-dasharray="4 4"/>
    <path d="M40,160 A28,28 0 0,1 66,150" fill="none" stroke="#64748b"/>
    <text x="70" y="150" font-size="12" fill="#64748b" font-family="sans-serif">θ</text>
    <text x="214" y="176" font-size="13" fill="#0d9488" font-family="sans-serif">b</text>
    <text x="150" y="58" font-size="13" fill="#4f46e5" font-family="sans-serif">a</text>
    <text x="92" y="150" font-size="11" fill="#64748b" font-family="sans-serif">shadow of a on b</text>
    <defs>
      <marker id="dd1" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#0d9488"/></marker>
      <marker id="dd2" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
    </defs>
  </svg>
  <figcaption>a·b = (length of a's shadow on b) × |b| = |a||b|cos θ.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $\mathbf a,\mathbf b\in\mathbb{R}^n$ the dot product has two equal definitions:</p>
  $$\mathbf a\cdot\mathbf b=\sum_{i=1}^{n} a_i b_i \qquad\text{(algebraic)},\qquad
  \mathbf a\cdot\mathbf b=\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert\cos\theta \qquad\text{(geometric)}.$$
  <p>Here $\lVert\mathbf a\rVert=\sqrt{\sum_i a_i^2}$ is the length (Track 4) and $\theta$ is the angle between the
  vectors. Setting $\mathbf a\cdot\mathbf b=0$ means $\cos\theta=0$, i.e. the vectors are <strong>orthogonal</strong>
  (perpendicular). <strong>Cosine similarity</strong> divides out the lengths:
  $\cos\theta=\dfrac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$.</p>`,

  derivation: String.raw`<p><strong>Why the two definitions are the same thing.</strong> We prove
  $\sum_i a_i b_i=\lVert\mathbf a\rVert\lVert\mathbf b\rVert\cos\theta$ using the law of cosines.</p>
  <p><strong>Step 1 — a triangle.</strong> Place $\mathbf a$ and $\mathbf b$ from the same origin with angle $\theta$
  between them. The side opposite $\theta$ is the vector $\mathbf a-\mathbf b$, with length $\lVert\mathbf a-\mathbf b\rVert$.</p>
  <p><strong>Step 2 — law of cosines</strong> (a geometric fact):
  $\lVert\mathbf a-\mathbf b\rVert^2=\lVert\mathbf a\rVert^2+\lVert\mathbf b\rVert^2-2\lVert\mathbf a\rVert\lVert\mathbf b\rVert\cos\theta.$</p>
  <p><strong>Step 3 — expand the left side algebraically</strong> using $\lVert\mathbf v\rVert^2=\mathbf v\cdot\mathbf v$
  and distributivity:
  $\lVert\mathbf a-\mathbf b\rVert^2=(\mathbf a-\mathbf b)\cdot(\mathbf a-\mathbf b)=\lVert\mathbf a\rVert^2-2(\mathbf a\cdot\mathbf b)+\lVert\mathbf b\rVert^2.$</p>
  <p><strong>Step 4 — match the two expressions.</strong> Both equal $\lVert\mathbf a-\mathbf b\rVert^2$, so cancel the
  common $\lVert\mathbf a\rVert^2+\lVert\mathbf b\rVert^2$:
  $-2(\mathbf a\cdot\mathbf b)=-2\lVert\mathbf a\rVert\lVert\mathbf b\rVert\cos\theta.$</p>
  <p><strong>Step 5 — divide by $-2$:</strong> $\mathbf a\cdot\mathbf b=\lVert\mathbf a\rVert\lVert\mathbf b\rVert\cos\theta.\;\blacksquare$
  Plain English: the entry-wise sum secretly encodes the angle between the arrows.</p>`,

  code: [
    { label: "Dot product, cosine similarity, orthogonality", src: String.raw`
import numpy as np

a = np.array([2.0, 1.0, 0.0])
b = np.array([1.0, 3.0, 0.0])

dot = a @ b                       # 2*1 + 1*3 + 0 = 5
cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
print("a·b               =", dot)
print("cosine similarity =", round(float(cos), 4))

u, w = np.array([1.0,0.0]), np.array([0.0,1.0])
print("perpendicular? u·w =", u @ w, "-> orthogonal")   # 0
` },
    { label: "Attention scores are a matrix of dot products", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

n, d = 3, 4
Q = rng.normal(size=(n, d))      # queries
K = rng.normal(size=(n, d))      # keys
scores = Q @ K.T                 # every query · every key  -> (n, n)
print("score matrix shape:", scores.shape)
print(np.round(scores, 2))
# softmax over each row turns these alignments into attention weights
` }
  ],

  keyPoints: [
    "Dot product = sum of entrywise products = $\\lVert a\\rVert\\lVert b\\rVert\\cos\\theta$.",
    "It measures alignment: positive (same way), zero (perpendicular), negative (opposing).",
    "Dot product $=0$ ⇔ vectors are orthogonal.",
    "Cosine similarity is the dot product with lengths divided out — angle only.",
    "An attention score matrix $QK^\\top$ is every query dotted with every key."
  ],

  commonMistakes: [
    { wrong: "Using the dot product when you meant the elementwise product.", why: "$\\mathbf a\\cdot\\mathbf b$ returns one scalar; $\\mathbf a\\odot\\mathbf b$ returns a vector. In NumPy, <code>a@b</code> vs <code>a*b</code> — mixing them changes shapes and breaks the next layer." },
    { wrong: "Comparing raw dot products as if they were similarities.", why: "A large dot product can come from long vectors, not aligned ones. For similarity, normalize to cosine; otherwise magnitude dominates direction." },
    { wrong: "Forgetting the $1/\\sqrt d$ scale in attention.", why: "Dot products of $d$-dim vectors grow like $\\sqrt d$; unscaled, softmax saturates. That's why it's $QK^\\top/\\sqrt d$ (see Deep Dive)." }
  ],

  quiz: [
    { q: "Compute $[1,2,3]\\cdot[4,0,-1]$.", options: ["1", "7", "0", "-1"], answer: 0,
      explain: "$1\\cdot4+2\\cdot0+3\\cdot(-1)=4+0-3=1$. Choice 7 forgot the negative term." },
    { q: "Two unit vectors have dot product $0$. The angle between them is…", options: ["90°", "0°", "45°", "180°"], answer: 0,
      explain: "$\\cos\\theta=0\\Rightarrow\\theta=90°$ — orthogonal. $0°$ would give dot product $1$." },
    { q: "$\\mathbf a\\cdot\\mathbf b=-6$, $\\lVert\\mathbf a\\rVert=2$, $\\lVert\\mathbf b\\rVert=3$. Find $\\cos\\theta$.", options: ["$-1$", "$1$", "$-0.5$", "$0$"], answer: 0,
      explain: "$\\cos\\theta=-6/(2\\cdot3)=-1$, so they point in exactly opposite directions ($\\theta=180°$)." },
    { q: "Cosine similarity of $[3,4]$ with itself?", options: ["1", "5", "25", "0"], answer: 0,
      explain: "Any nonzero vector is perfectly aligned with itself: $\\cos0=1$. (The raw dot product is $25$, but cosine normalizes it.)" },
    { q: "Why scale attention scores by $1/\\sqrt d$?", options: ["Dot products grow like $\\sqrt d$, so this keeps softmax from saturating", "It makes them integers", "It changes the ranking of scores", "It is required for orthogonality"], answer: 0,
      explain: "With variance growing in $d$, large logits push softmax to one-hot and kill gradients; dividing by $\\sqrt d$ restores a usable scale without changing the ranking." }
  ],

  practice: [
    { level: "easy", prompt: "Compute $[2,2]\\cdot[3,-1]$ and say whether the angle is acute, right, or obtuse.", solution: "$2\\cdot3+2\\cdot(-1)=6-2=4\\gt0$, so the angle is acute (they broadly point the same way)." },
    { level: "easy", prompt: "Are $[1,2]$ and $[2,-1]$ orthogonal?", solution: "$1\\cdot2+2\\cdot(-1)=0$, yes — perpendicular." },
    { level: "med", prompt: "Find $k$ so that $[1,k,2]$ is orthogonal to $[3,-1,1]$.", solution: "Dot $=3-k+2=5-k$. Set to $0$: $k=5$. Then the vectors are perpendicular." },
    { level: "med", prompt: "Show the dot product distributes: $\\mathbf a\\cdot(\\mathbf b+\\mathbf c)=\\mathbf a\\cdot\\mathbf b+\\mathbf a\\cdot\\mathbf c$.", solution: "$\\sum_i a_i(b_i+c_i)=\\sum_i(a_ib_i+a_ic_i)=\\sum_i a_ib_i+\\sum_i a_ic_i$. Distributivity of numbers lifts to vectors term by term." },
    { level: "hard", prompt: "AI task: a unit query $\\mathbf q$ and two unit keys give $\\mathbf q\\cdot\\mathbf k_1=0.9$, $\\mathbf q\\cdot\\mathbf k_2=0.2$. Which token gets more attention, and what does scaling by $1/\\sqrt d$ change?", solution: "Higher score $0.9$ → token 1 wins after softmax. Dividing by $\\sqrt d$ shrinks <em>both</em> scores equally, making the softmax less peaked — it rescales confidence but preserves that token 1 ranks above token 2." }
  ],

  deepDive: String.raw`<p><strong>Why attention divides by $\sqrt d$ — a variance argument.</strong></p>
  <p>Suppose query and key components are independent with mean $0$ and variance $1$. Their dot product is
  $\mathbf q\cdot\mathbf k=\sum_{i=1}^{d} q_i k_i$. Each term $q_ik_i$ has mean $0$ and variance $1$, and the $d$
  terms are independent, so variances add: $\operatorname{Var}(\mathbf q\cdot\mathbf k)=d$, giving a typical magnitude
  of $\sqrt d$.</p>
  <p>Feed scores of size $\sim\sqrt d$ into softmax and, as $d$ grows, the largest logit dominates: softmax collapses
  toward one-hot and its gradient vanishes. Dividing by $\sqrt d$ rescales the variance back to $1$, keeping the
  distribution soft and the gradients alive. This is the one line in "Attention Is All You Need" —
  $\operatorname{softmax}(QK^\top/\sqrt d)$ — and now you can derive the constant rather than memorize it. (Variance
  is made formal in Track 9, softmax stability in Track 13.)</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["2.2"] = {
  subtitle: "Add vectors, scale them, mix them — the only moves a linear layer ever makes.",

  aiMoment: String.raw`<p>A <strong>residual connection</strong> — the idea that lets us train 100-layer networks — is
  literally vector addition: the block computes $\mathbf x+f(\mathbf x)$. A gradient-descent step,
  $\mathbf w-\eta\nabla\mathcal L$, is a scaled vector subtracted from another. And every neuron's pre-activation is a
  <strong>linear combination</strong> $\sum_i w_i x_i$. Addition, scaling, and combining are the entire vocabulary of a
  linear layer.</p>`,

  plainEnglish: String.raw`<p>To <strong>add</strong> two vectors, add their matching components (or, with arrows, walk
  one then the other). To <strong>scale</strong> a vector, multiply every component by a number — it stretches or
  shrinks, and flips if the number is negative. A <strong>linear combination</strong> is just a weighted mix:
  scale a few vectors and add the results.</p>`,

  intuition: String.raw`<p>Addition is the diagonal of a parallelogram: lay $\mathbf b$ at the tip of $\mathbf a$ and
  the sum reaches the far corner. Every point you can reach by mixing $\mathbf a$ and $\mathbf b$ with all possible
  weights is their <strong>span</strong>.</p>
  <figure class="figure">
  <svg viewBox="0 0 250 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Vector addition as a parallelogram">
    <line x1="40" y1="180" x2="150" y2="140" stroke="#4f46e5" stroke-width="3" marker-end="url(#p1)"/>
    <line x1="40" y1="180" x2="80"  y2="70"  stroke="#0d9488" stroke-width="3" marker-end="url(#p2)"/>
    <line x1="150" y1="140" x2="190" y2="30" stroke="#0d9488" stroke-width="1.6" stroke-dasharray="4 4"/>
    <line x1="80"  y1="70"  x2="190" y2="30" stroke="#4f46e5" stroke-width="1.6" stroke-dasharray="4 4"/>
    <line x1="40" y1="180" x2="190" y2="30" stroke="#7c3aed" stroke-width="3" marker-end="url(#p3)"/>
    <text x="92"  y="172" font-size="13" fill="#4f46e5" font-family="sans-serif">a</text>
    <text x="48"  y="120" font-size="13" fill="#0d9488" font-family="sans-serif">b</text>
    <text x="120" y="92"  font-size="13" fill="#7c3aed" font-family="sans-serif">a + b</text>
    <defs>
      <marker id="p1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="p2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#0d9488"/></marker>
      <marker id="p3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#7c3aed"/></marker>
    </defs>
  </svg>
  <figcaption>The sum is the parallelogram's diagonal; dashed edges are copies of a and b.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $\mathbf a,\mathbf b\in\mathbb{R}^n$ and scalar $c$:</p>
  $$\mathbf a+\mathbf b=[a_1+b_1,\dots,a_n+b_n],\qquad c\mathbf a=[ca_1,\dots,ca_n].$$
  <p>A <strong>linear combination</strong> of $\mathbf v_1,\dots,\mathbf v_k$ with weights $c_1,\dots,c_k$ is
  $\sum_{i=1}^{k} c_i\mathbf v_i$. The set of <em>all</em> such combinations is the <strong>span</strong>. The
  <strong>zero vector</strong> $\mathbf 0$ is the additive identity ($\mathbf a+\mathbf 0=\mathbf a$), and $-\mathbf a$
  is the additive inverse. Addition is commutative and associative — it inherits those from ordinary numbers, component
  by component.</p>`,

  derivation: String.raw`<p><strong>Two independent vectors span a whole plane.</strong> We show every point of
  $\mathbb{R}^2$ is a linear combination of $\mathbf e_1=[1,0]$ and $\mathbf e_2=[0,1]$.</p>
  <p><strong>Step 1 — pick any target</strong> $\mathbf t=[t_1,t_2]$.</p>
  <p><strong>Step 2 — propose weights.</strong> Try $c_1=t_1$, $c_2=t_2$.</p>
  <p><strong>Step 3 — combine and check:</strong>
  $c_1\mathbf e_1+c_2\mathbf e_2=t_1[1,0]+t_2[0,1]=[t_1,0]+[0,t_2]=[t_1,t_2]=\mathbf t.\;\checkmark$</p>
  <p><strong>Step 4 — conclude.</strong> Every $\mathbf t$ is reachable, so $\{\mathbf e_1,\mathbf e_2\}$ spans
  $\mathbb{R}^2$. $\blacksquare$ Plain English: with the right weights, two independent directions can take you
  anywhere in the plane — the seed of "basis" in Track 3.</p>
  <hr class="soft">
  <p><strong>Why this matters for training.</strong> A gradient-descent trajectory $\mathbf w_{t+1}=\mathbf w_t-\eta\,\mathbf g_t$
  is a running linear combination of gradients: unrolling gives $\mathbf w_T=\mathbf w_0-\eta\sum_{t<T}\mathbf g_t$. The
  weights are all $\eta$ here; <strong>momentum</strong> (Track 8) simply uses a smarter weighted combination of past
  gradients. Optimizers are recipes for choosing the coefficients in a linear combination.</p>`,

  code: [
    { label: "Add, scale, combine — and a residual connection", src: String.raw`
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([0.5, 0.5, 0.5])

print("a + b      =", a + b)            # componentwise
print("3 * a      =", 3 * a)            # scale
print("2a - b     =", 2*a - b)          # linear combination

# a residual block: output = input + f(input)
def f(x): return np.tanh(x) * 0.1       # some learned transform
x = np.array([0.2, -0.4, 1.0])
print("residual x + f(x) =", x + f(x))  # literally vector addition
` },
    { label: "A linear combination spans a plane", src: String.raw`
import numpy as np

e1 = np.array([1.0, 0.0]); e2 = np.array([0.0, 1.0])
target = np.array([3.0, -2.0])
combo = target[0]*e1 + target[1]*e2     # weights = coordinates
print("reconstructed target:", combo, "->", np.allclose(combo, target))
` }
  ],

  keyPoints: [
    "Vector addition and scaling are componentwise; addition is the parallelogram diagonal.",
    "A linear combination is a weighted sum $\\sum_i c_i\\mathbf v_i$; the set of all of them is the span.",
    "Residual connections are vector addition: $\\mathbf x+f(\\mathbf x)$.",
    "Gradient descent accumulates a linear combination of gradients.",
    "$\\mathbf 0$ is the additive identity; $-\\mathbf a$ undoes $\\mathbf a$."
  ],

  commonMistakes: [
    { wrong: "Adding vectors of different lengths.", why: "Addition is defined only for equal-dimensional vectors. NumPy may <em>broadcast</em> instead of erroring, silently producing a wrong-shaped result (Lesson 2.5)." },
    { wrong: "Thinking scaling changes a vector's direction.", why: "A positive scalar only changes length; direction is unchanged. Only a negative scalar flips it to the opposite direction." },
    { wrong: "Confusing the weights with the vectors in a combination.", why: "In $c_1\\mathbf v_1+c_2\\mathbf v_2$ the $c_i$ are scalars (knobs) and the $\\mathbf v_i$ are vectors. Swapping the roles is a shape error." }
  ],

  quiz: [
    { q: "Compute $2[1,-1,0]-[0,2,3]$.", options: ["$[2,-4,-3]$", "$[2,0,3]$", "$[2,-4,3]$", "$[1,-3,-3]$"], answer: 0,
      explain: "$[2,-2,0]-[0,2,3]=[2,-4,-3]$. Choice $[2,-4,3]$ flips the sign on the last component." },
    { q: "Which is a linear combination of $\\mathbf u,\\mathbf v$?", options: ["$3\\mathbf u-2\\mathbf v$", "$\\mathbf u\\cdot\\mathbf v$", "$\\mathbf u/\\mathbf v$", "$\\lVert\\mathbf u\\rVert$"], answer: 0,
      explain: "A linear combination scales and adds vectors: $3\\mathbf u-2\\mathbf v$. The dot product and norm return scalars; vector division isn't defined." },
    { q: "The span of $[1,0]$ and $[2,0]$ is…", options: ["the x-axis (a line)", "all of $\\mathbb{R}^2$", "just the origin", "a plane in 3-D"], answer: 0,
      explain: "Both point along $x$, so every combination stays on the x-axis. They're dependent, so they span only a line, not the plane." },
    { q: "A residual block outputs $\\mathbf x+f(\\mathbf x)$ with $\\mathbf x=[1,2]$, $f(\\mathbf x)=[0.1,-0.2]$. Output?", options: ["$[1.1,1.8]$", "$[1.1,2.2]$", "$[0.1,-0.4]$", "$[1,2]$"], answer: 0,
      explain: "Add componentwise: $[1+0.1,\\,2-0.2]=[1.1,1.8]$." },
    { q: "After 3 steps $\\mathbf w\\leftarrow\\mathbf w-\\eta\\mathbf g_t$ from $\\mathbf w_0$, $\\mathbf w_3=$?", options: ["$\\mathbf w_0-\\eta(\\mathbf g_0+\\mathbf g_1+\\mathbf g_2)$", "$\\mathbf w_0-\\eta\\mathbf g_2$", "$\\mathbf w_0-3\\eta\\mathbf g_0$", "$\\mathbf w_0+\\eta\\sum\\mathbf g_t$"], answer: 0,
      explain: "Each step subtracts $\\eta\\mathbf g_t$; unrolling sums them: $\\mathbf w_0-\\eta\\sum_{t=0}^{2}\\mathbf g_t$ — a linear combination of gradients." }
  ],

  practice: [
    { level: "easy", prompt: "Compute $[2,3]+[-1,4]$ and $-2[1,-1]$.", solution: "$[1,7]$ and $[-2,2]$." },
    { level: "easy", prompt: "Is $[0,0]$ in the span of any set of vectors? Why?", solution: "Yes — choose all weights $0$. The zero vector is in every span." },
    { level: "med", prompt: "Write $[5,1]$ as a linear combination of $[1,1]$ and $[1,-1]$.", solution: "Solve $c_1[1,1]+c_2[1,-1]=[5,1]$: $c_1+c_2=5$, $c_1-c_2=1$ ⇒ $c_1=3,c_2=2$. So $3[1,1]+2[1,-1]=[5,1]$." },
    { level: "hard", prompt: "AI task: explain why momentum $\\mathbf m_t=\\beta\\mathbf m_{t-1}+\\mathbf g_t$ makes the update a weighted linear combination of past gradients, and what $\\beta$ controls.", solution: "Unrolling, $\\mathbf m_t=\\sum_{k=0}^{t}\\beta^{k}\\mathbf g_{t-k}$ — recent gradients get weight near 1, older ones decay as $\\beta^k$. The step $-\\eta\\mathbf m_t$ is thus a linear combination with exponentially-decaying coefficients. $\\beta$ sets the memory: larger $\\beta$ averages over more past gradients, smoothing the zig-zag from Lesson 6.3's Deep Dive." }
  ],

  deepDive: String.raw`<p><strong>The residual stream as a sum of contributions.</strong></p>
  <p>In a transformer, every attention head and MLP block <em>adds</em> its output into a shared running vector — the
  "residual stream." Because addition is associative and commutative, the stream at any layer is just the input plus
  the sum of all contributions written so far: $\mathbf h_\ell=\mathbf x+\sum_{b<\ell}f_b(\cdot)$. This is more than
  notation. It means each component can be studied as an additive term, which is the foundation of mechanistic
  interpretability: researchers literally subtract one head's contribution to see what it wrote, or project the stream
  onto a direction to read out a concept. The humble "+" of a residual connection is what makes a network's internal
  computation decomposable. Linear combinations also set up <strong>basis</strong> and <strong>rank</strong> in Track
  3: the span of a set of vectors is exactly the reach of their linear combinations.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["2.4"] = {
  subtitle: "The forward pass of every dense layer — and a machine that transforms space.",

  aiMoment: String.raw`<p>A fully-connected layer computes $\mathbf y=W\mathbf x+\mathbf b$. That $W\mathbf x$ is a
  <strong>matrix–vector multiply</strong>, the single most executed operation in deep learning — GPUs are essentially
  matrix-multiply machines. Stacking layers <em>composes</em> their matrices; attention is a stack of matrix multiplies;
  even a convolution is a matrix multiply in disguise. Understanding this operation two ways — as row·column arithmetic
  and as a transformation of space — is the core skill of the whole track.</p>`,

  plainEnglish: String.raw`<p>Matrix multiplication has two readings. Mechanically, each output entry is a
  <strong>dot product</strong> of a row of the first matrix with a column of the second. Conceptually, a matrix is a
  <strong>machine that transforms space</strong> — it moves every vector to a new place — and multiplying two matrices
  means doing one transformation after the other.</p>`,

  intuition: String.raw`<p>A matrix is defined by where it sends the basis vectors: those landing spots are its
  <strong>columns</strong>. Knowing where $\hat\imath$ and $\hat\jmath$ go tells you where every vector goes, because
  every vector is a linear combination of the basis (Lesson 2.2).</p>
  <figure class="figure">
  <svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A matrix sends basis vectors to its columns">
    <line x1="30" y1="160" x2="290" y2="160" stroke="#eef0f4"/>
    <line x1="60" y1="190" x2="60" y2="20" stroke="#eef0f4"/>
    <line x1="60" y1="160" x2="110" y2="160" stroke="#cbd5e1" stroke-width="3" marker-end="url(#m0)"/>
    <line x1="60" y1="160" x2="60"  y2="110" stroke="#cbd5e1" stroke-width="3" marker-end="url(#m0)"/>
    <text x="86" y="176" font-size="11" fill="#94a3b8" font-family="sans-serif">î</text>
    <text x="40" y="132" font-size="11" fill="#94a3b8" font-family="sans-serif">ĵ</text>
    <line x1="60" y1="160" x2="210" y2="160" stroke="#4f46e5" stroke-width="3" marker-end="url(#m1)"/>
    <line x1="60" y1="160" x2="135" y2="85"  stroke="#0d9488" stroke-width="3" marker-end="url(#m2)"/>
    <text x="150" y="176" font-size="11" fill="#4f46e5" font-family="sans-serif">A î = col₁ = [2,0]</text>
    <text x="120" y="74" font-size="11" fill="#0d9488" font-family="sans-serif">A ĵ = col₂ = [1,1]</text>
    <defs>
      <marker id="m0" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#cbd5e1"/></marker>
      <marker id="m1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="m2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#0d9488"/></marker>
    </defs>
  </svg>
  <figcaption>The columns of A are the images of the basis vectors. For A = [[2,1],[0,1]]: î→[2,0], ĵ→[1,1].</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $A\in\mathbb{R}^{m\times k}$ and $B\in\mathbb{R}^{k\times n}$, the product $C=AB\in\mathbb{R}^{m\times n}$ has</p>
  $$C_{ij}=\sum_{p=1}^{k} A_{ip}B_{pj}.$$
  <p>The <strong>inner dimensions must match</strong> ($k=k$); the result takes the outer dimensions $m\times n$. A
  matrix–vector product is the special case $n=1$. Crucially, $A\mathbf x$ is also a <strong>linear combination of the
  columns</strong> of $A$, weighted by the entries of $\mathbf x$. Matrix multiplication is <em>not</em> commutative:
  in general $AB\neq BA$.</p>`,

  derivation: String.raw`<p><strong>The two views are the same.</strong> Let $A$ have columns
  $\mathbf a_1,\dots,\mathbf a_k$ and let $\mathbf x=[x_1,\dots,x_k]$.</p>
  <p><strong>Step 1 — the row·column rule</strong> gives the $i$-th output entry:
  $(A\mathbf x)_i=\sum_{p=1}^{k}A_{ip}x_p$.</p>
  <p><strong>Step 2 — read $A_{ip}$ as "the $i$-th entry of column $\mathbf a_p$"</strong>, i.e. $A_{ip}=(\mathbf a_p)_i$.
  Substitute: $(A\mathbf x)_i=\sum_{p}x_p(\mathbf a_p)_i$.</p>
  <p><strong>Step 3 — recognize the right side</strong> as the $i$-th entry of $\sum_p x_p\mathbf a_p$. Since this holds
  for every $i$, $A\mathbf x=\sum_{p=1}^{k}x_p\mathbf a_p.\;\blacksquare$</p>
  <p>Plain English: multiplying by $A$ mixes its columns with weights $\mathbf x$. The row·column arithmetic and the
  "combine the columns" picture are one and the same.</p>
  <hr class="soft">
  <p><strong>Multiplication = composition.</strong> For matrices, $(AB)\mathbf x=A(B\mathbf x)$: apply $B$ to
  $\mathbf x$, then $A$ to the result. So the product matrix $AB$ <em>is</em> the single transformation equivalent to
  "do $B$, then $A$." That's why associativity $A(BC)=(AB)C$ is automatic — it's just regrouping the same chain of
  transformations.</p>`,

  code: [
    { label: "Matrix multiply, three equivalent ways", src: String.raw`
import numpy as np

A = np.array([[2.0, 1.0],
              [0.0, 1.0]])
x = np.array([3.0, 4.0])

print("A @ x (row·col)      :", A @ x)                      # [10, 4]
print("combine columns      :", x[0]*A[:,0] + x[1]*A[:,1])  # same thing
print("shapes: (2x2)(2,) -> ", (A @ x).shape)
` },
    { label: "A dense layer's forward pass, and composition", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

W1 = rng.normal(size=(4, 3)); b1 = np.zeros(4)
W2 = rng.normal(size=(2, 4)); b2 = np.zeros(2)
x  = rng.normal(size=3)

# two layers applied in sequence
h = W1 @ x + b1
y = W2 @ h + b2
print("forward pass y =", np.round(y, 3))

# composition: (W2 W1) is ONE matrix doing both (ignoring biases)
W = W2 @ W1
print("same via composed matrix:", np.round(W @ x, 3))
print("match?", np.allclose(W @ x, W2 @ (W1 @ x)))
` }
  ],

  keyPoints: [
    "$C_{ij}=\\sum_p A_{ip}B_{pj}$: each output is a row·column dot product.",
    "Inner dims must match; $(m\\times k)(k\\times n)=(m\\times n)$.",
    "$A\\mathbf x$ is a linear combination of $A$'s columns, weighted by $\\mathbf x$.",
    "A matrix transforms space; multiplying matrices composes the transformations.",
    "Matrix multiplication is associative but NOT commutative ($AB\\ne BA$)."
  ],

  commonMistakes: [
    { wrong: "Using <code>*</code> for matrix multiply in NumPy.", why: "<code>A*B</code> is elementwise (Hadamard); matrix multiply is <code>A@B</code> or <code>np.matmul</code>. Mixing them gives wrong results or a broadcast surprise." },
    { wrong: "Assuming $AB=BA$.", why: "Order is a transformation order — rotate-then-scale differs from scale-then-rotate. Even the shapes may only conform one way." },
    { wrong: "Mismatching inner dimensions.", why: "$(m\\times k)(n\\times p)$ needs $k=n$. A shape error here is the most common bug when wiring layers; print shapes." }
  ],

  quiz: [
    { q: "$\\begin{bmatrix}1&2\\\\0&1\\end{bmatrix}\\begin{bmatrix}3\\\\4\\end{bmatrix}=$?", options: ["$[11,4]$", "$[3,4]$", "$[7,4]$", "$[11,7]$"], answer: 0,
      explain: "Row 1: $1\\cdot3+2\\cdot4=11$; row 2: $0\\cdot3+1\\cdot4=4$. So $[11,4]$." },
    { q: "Shapes $(8\\times16)$ times $(16\\times4)$ give…", options: ["$(8\\times4)$", "$(16\\times16)$", "$(4\\times8)$", "error"], answer: 0,
      explain: "Inner dims $16$ match; result is outer dims $8\\times4$." },
    { q: "$A\\mathbf x$ where $\\mathbf x=[0,1]$ returns…", options: ["the 2nd column of $A$", "the 2nd row of $A$", "the sum of $A$'s columns", "a scalar"], answer: 0,
      explain: "$A\\mathbf x=0\\cdot\\mathbf a_1+1\\cdot\\mathbf a_2=\\mathbf a_2$, the second column — picking out a column is just multiplying by a basis vector." },
    { q: "Which product is generally valid AND equals the other order?", options: ["neither order is generally equal", "$AB=BA$ always", "only square $AB=BA$", "$AB=BA$ if same shape"], answer: 0,
      explain: "Matrix multiply isn't commutative; equality $AB=BA$ holds only in special cases, never in general." },
    { q: "Two linear layers $W_2(W_1\\mathbf x)$ (no activation) are equivalent to…", options: ["one linear layer $(W_2W_1)\\mathbf x$", "a nonlinear layer", "an attention block", "a residual connection"], answer: 0,
      explain: "Composition of linear maps is linear: $W_2W_1$ is a single matrix. This is why depth needs nonlinearities (Deep Dive)." }
  ],

  practice: [
    { level: "easy", prompt: "Multiply $\\begin{bmatrix}2&0\\\\0&3\\end{bmatrix}\\begin{bmatrix}1\\\\1\\end{bmatrix}$.", solution: "Diagonal scaling: $[2\\cdot1,\\,3\\cdot1]=[2,3]$." },
    { level: "med", prompt: "Compute $\\begin{bmatrix}1&1\\\\0&1\\end{bmatrix}\\begin{bmatrix}1&0\\\\1&1\\end{bmatrix}$ and the reverse order; are they equal?", solution: "First: $\\begin{bmatrix}2&1\\\\1&1\\end{bmatrix}$. Reverse: $\\begin{bmatrix}1&0\\\\1&1\\end{bmatrix}\\begin{bmatrix}1&1\\\\0&1\\end{bmatrix}=\\begin{bmatrix}1&1\\\\1&2\\end{bmatrix}$. Not equal — confirming non-commutativity." },
    { level: "med", prompt: "If $A$ is $(3\\times3)$ and $\\mathbf x=[1,0,0]$, what is $A\\mathbf x$? Explain via the columns view.", solution: "$A\\mathbf x=1\\cdot\\mathbf a_1+0+0=\\mathbf a_1$, the first column of $A$. Multiplying by a standard basis vector extracts the matching column." },
    { level: "hard", prompt: "AI task: a batch $X$ has shape $(B,d_{in})$ and a layer weight $W$ has shape $(d_{in},d_{out})$. Give the matmul that produces the batch of outputs and its shape, and say why we put the batch axis first.", solution: "Outputs $=X W$ with shape $(B,d_{out})$ — each row of $X$ (one example) is transformed independently. Putting batch first ($B$ as axis 0) lets one matrix multiply handle the whole batch in parallel, which is exactly what GPUs are optimized for; the contraction happens over $d_{in}$, the shared inner dimension." }
  ],

  deepDive: String.raw`<p><strong>Why a deep <em>linear</em> network is secretly shallow — and the case for activations.</strong></p>
  <p>Stack $L$ linear layers with no nonlinearity: $\mathbf y=W_L(W_{L-1}(\cdots W_1\mathbf x))$. By associativity this
  collapses to $\mathbf y=(W_LW_{L-1}\cdots W_1)\,\mathbf x=W_{\text{eff}}\,\mathbf x$ — a single matrix. No matter how
  many layers you stack, a purely linear network can only ever represent <em>one</em> linear map; depth buys exactly
  nothing in expressive power.</p>
  <p>That collapse is the mathematical reason every useful network inserts a <strong>nonlinearity</strong> (ReLU, GELU,
  …) between matrix multiplies. The nonlinearity breaks the "product of matrices is one matrix" chain, so each layer can
  reshape space in a genuinely new way and the composition becomes vastly more expressive. It also connects to rank
  (Track 3): the product $W_{\text{eff}}$ has rank at most the smallest layer width, so a thin linear layer is a
  literal information bottleneck — the principle behind LoRA's low-rank updates. Matrix multiplication being composition
  is not a metaphor; it dictates architecture.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["2.5"] = {
  subtitle: "Four everyday operations: flip, do-nothing, multiply-elementwise, and auto-stretch.",

  aiMoment: String.raw`<p>Attention scores use $K^\top$ — a <strong>transpose</strong>. Backprop is full of transposes:
  the gradient flowing back through $\mathbf y=W\mathbf x$ is $W^\top$ times the upstream gradient. Gating in LSTMs and
  GLUs is a <strong>Hadamard</strong> (elementwise) product. And adding a bias vector to a batch relies on
  <strong>broadcasting</strong>. These four small operations show up in every model you'll build.</p>`,

  plainEnglish: String.raw`<p>The <strong>transpose</strong> flips a matrix across its diagonal (rows become columns).
  The <strong>identity</strong> matrix is the do-nothing transform. The <strong>Hadamard product</strong> multiplies two
  same-shaped arrays entry by entry. <strong>Broadcasting</strong> is NumPy automatically stretching a smaller array so
  its shape lines up with a bigger one.</p>`,

  intuition: String.raw`<p>Transpose is a mirror across the main diagonal: the entry at row $i$, column $j$ trades
  places with row $j$, column $i$. Diagonal entries stay put; off-diagonal ones swap.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Transpose flips a matrix across its diagonal">
    <g font-family="sans-serif" font-size="15">
    <rect x="30" y="25" width="92" height="92" rx="6" fill="#f7f3ff" stroke="#7c3aed"/>
    <line x1="30" y1="25" x2="122" y2="117" stroke="#c4b5fd" stroke-dasharray="4 4"/>
    <text x="55" y="60" fill="#20242c">a</text><text x="97" y="60" fill="#4f46e5">b</text>
    <text x="55" y="102" fill="#0d9488">c</text><text x="97" y="102" fill="#20242c">d</text>
    <text x="68" y="16" fill="#7c3aed" font-size="12">A</text>
    <line x1="138" y1="71" x2="186" y2="71" stroke="#94a3b8" stroke-width="2" marker-end="url(#t1)"/>
    <text x="150" y="62" fill="#64748b" font-size="13">ᵀ</text>
    <rect x="198" y="25" width="92" height="92" rx="6" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="223" y="60" fill="#20242c">a</text><text x="265" y="60" fill="#0d9488">c</text>
    <text x="223" y="102" fill="#4f46e5">b</text><text x="265" y="102" fill="#20242c">d</text>
    <text x="232" y="16" fill="#0d9488" font-size="12">Aᵀ</text>
    </g>
    <defs><marker id="t1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>b and c swap across the diagonal; a and d stay. That's the transpose.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Transpose:</strong> $(A^\top)_{ij}=A_{ji}$. <strong>Identity</strong> $I$ has 1's on
  the diagonal, 0's elsewhere, and $IA=AI=A$. <strong>Hadamard product:</strong> $(A\odot B)_{ij}=A_{ij}B_{ij}$ (same
  shapes, entrywise — NumPy <code>*</code>). <strong>Broadcasting</strong> aligns shapes from the right and stretches
  any axis of length 1: an $(m,n)$ array plus an $(n,)$ vector adds that vector to every row.</p>`,

  derivation: String.raw`<p><strong>The transpose-of-a-product rule:</strong> $(AB)^\top=B^\top A^\top$ — note the order
  flips. This exact identity is why gradients propagate with transposed weights.</p>
  <p><strong>Step 1 — left side, by definition of transpose:</strong> $\big((AB)^\top\big)_{ij}=(AB)_{ji}=\sum_k A_{jk}B_{ki}.$</p>
  <p><strong>Step 2 — right side, expand $B^\top A^\top$:</strong>
  $\big(B^\top A^\top\big)_{ij}=\sum_k (B^\top)_{ik}(A^\top)_{kj}=\sum_k B_{ki}A_{jk}.$</p>
  <p><strong>Step 3 — compare.</strong> Both equal $\sum_k A_{jk}B_{ki}$ (numbers commute, so $A_{jk}B_{ki}=B_{ki}A_{jk}$).
  Hence $(AB)^\top=B^\top A^\top.\;\blacksquare$</p>
  <p>Plain English: transposing a product reverses the factors. In backprop, if a forward step is $\mathbf y=W\mathbf x$,
  the gradient w.r.t. the input is $W^\top$ times the gradient w.r.t. the output — the transpose appears precisely
  because of this rule (made rigorous in Track 13).</p>`,

  code: [
    { label: "Transpose, identity, Hadamard", src: String.raw`
import numpy as np

A = np.array([[1.,2.,3.],
              [4.,5.,6.]])
print("A shape   :", A.shape, " A.T shape:", A.T.shape)   # (2,3) -> (3,2)
print("I @ A == A:", np.allclose(np.eye(2) @ A, A))

B = np.array([[1.,0.,1.],
              [2.,2.,0.]])
print("Hadamard A*B:\n", A * B)        # elementwise, NOT matmul
print("(A B^T)^T == B A^T ? ",
      np.allclose((A @ B.T).T, B @ A.T))   # transpose-of-product rule
` },
    { label: "Broadcasting a bias — and a shape-bug", src: String.raw`
import numpy as np

X = np.array([[1.,2.,3.],
              [4.,5.,6.]])      # (2,3): 2 examples, 3 features
b = np.array([10., 20., 30.])  # (3,): one bias per feature

print("X + b (correct, per-feature):\n", X + b)   # b added to every row

bad = np.array([[100.],[200.]])  # (2,1): wrong — broadcasts over columns!
print("X + bad (silent wrong shape):\n", X + bad)  # adds per-ROW instead
` }
  ],

  keyPoints: [
    "$(A^\\top)_{ij}=A_{ji}$ flips across the diagonal; $(AB)^\\top=B^\\top A^\\top$ (order reverses).",
    "The identity $I$ leaves vectors unchanged: $IA=AI=A$.",
    "Hadamard $\\odot$ is entrywise (NumPy <code>*</code>); it is not matrix multiplication.",
    "Broadcasting aligns shapes from the right and stretches length-1 axes.",
    "Transposes appear in backprop: the gradient through $W\\mathbf x$ uses $W^\\top$."
  ],

  commonMistakes: [
    { wrong: "Writing $(AB)^\\top=A^\\top B^\\top$.", why: "The order must reverse: $(AB)^\\top=B^\\top A^\\top$. The wrong order usually won't even have conforming shapes." },
    { wrong: "Using <code>*</code> expecting matrix multiply.", why: "<code>*</code> is Hadamard (elementwise). For matrix multiply use <code>@</code>. This is the #1 NumPy/PyTorch beginner bug." },
    { wrong: "Trusting broadcasting blindly.", why: "An $(m,1)$ vs $(n,)$ mix can broadcast to a valid-but-wrong shape with no error — the model trains to a plausible but incorrect loss. Always print shapes." }
  ],

  quiz: [
    { q: "If $A$ is $(3\\times5)$, what is the shape of $A^\\top$?", options: ["$(5\\times3)$", "$(3\\times5)$", "$(5\\times5)$", "$(3\\times3)$"], answer: 0,
      explain: "Transpose swaps the dimensions: $(3\\times5)\\to(5\\times3)$." },
    { q: "$(AB)^\\top$ equals…", options: ["$B^\\top A^\\top$", "$A^\\top B^\\top$", "$(BA)^\\top$", "$AB$"], answer: 0,
      explain: "Transpose of a product reverses the order: $B^\\top A^\\top$." },
    { q: "In NumPy, <code>A * B</code> for same-shape $A,B$ computes…", options: ["the Hadamard (elementwise) product", "the matrix product", "the dot product", "an error"], answer: 0,
      explain: "<code>*</code> is elementwise. Matrix product is <code>@</code>; the elementwise one is the Hadamard product." },
    { q: "$X$ is $(32,10)$ and $b$ is $(10,)$. <code>X + b</code> gives shape…", options: ["$(32,10)$", "$(10,10)$", "error", "$(32,1)$"], answer: 0,
      explain: "Broadcasting stretches $b$ across all 32 rows, adding it per feature; shape stays $(32,10)$." },
    { q: "Which leaves any conformable matrix unchanged under multiplication?", options: ["the identity $I$", "the zero matrix", "$A^\\top$", "a diagonal of 2's"], answer: 0,
      explain: "$IA=A$. The zero matrix maps everything to $0$; a diagonal of 2's doubles." }
  ],

  practice: [
    { level: "easy", prompt: "Write the transpose of $\\begin{bmatrix}1&2&3\\\\4&5&6\\end{bmatrix}$.", solution: "$\\begin{bmatrix}1&4\\\\2&5\\\\3&6\\end{bmatrix}$ — rows become columns, shape $(2,3)\\to(3,2)$." },
    { level: "easy", prompt: "Compute the Hadamard product $[1,2,3]\\odot[4,0,2]$.", solution: "Entrywise: $[4,0,6]$." },
    { level: "med", prompt: "Verify $(AB)^\\top=B^\\top A^\\top$ for $A=\\begin{bmatrix}1&2\\\\0&1\\end{bmatrix}$, $B=\\begin{bmatrix}1&0\\\\3&1\\end{bmatrix}$.", solution: "$AB=\\begin{bmatrix}7&2\\\\3&1\\end{bmatrix}$, so $(AB)^\\top=\\begin{bmatrix}7&3\\\\2&1\\end{bmatrix}$. And $B^\\top A^\\top=\\begin{bmatrix}1&3\\\\0&1\\end{bmatrix}\\begin{bmatrix}1&0\\\\2&1\\end{bmatrix}=\\begin{bmatrix}7&3\\\\2&1\\end{bmatrix}.\\;\\checkmark$" },
    { level: "hard", prompt: "AI task: a layer computes $\\mathbf y=W\\mathbf x$ with $W$ of shape $(d_{out},d_{in})$. Backprop receives $\\frac{\\partial \\mathcal L}{\\partial\\mathbf y}$ of shape $(d_{out},)$ and must produce $\\frac{\\partial\\mathcal L}{\\partial\\mathbf x}$. Give the formula and the shape check.", solution: "$\\frac{\\partial\\mathcal L}{\\partial\\mathbf x}=W^\\top\\frac{\\partial\\mathcal L}{\\partial\\mathbf y}$. Shapes: $W^\\top$ is $(d_{in},d_{out})$ times $(d_{out},)$ gives $(d_{in},)$ — matching $\\mathbf x$. The transpose is exactly what makes the dimensions line up, and it follows from differentiating a matrix–vector product (Track 13's reverse-mode autodiff)." }
  ],

  deepDive: String.raw`<p><strong>Broadcasting rules in full — and the bug they hide.</strong></p>
  <p>NumPy compares two shapes <em>from the right</em>. For each aligned pair of axes, they are compatible if they're
  equal or one of them is 1 (which gets stretched). Missing leading axes are treated as 1. So $(32,10)+(10,)$ works
  (the $(10,)$ becomes $(1,10)$ then stretches to $(32,10)$), and $(32,10)+(32,1)$ also works — but they do
  <em>different</em> things: the first adds a per-feature bias, the second a per-example bias.</p>
  <p>The danger is that both are legal. Suppose you mean to subtract a feature mean of shape $(10,)$ to center your
  data, but a stray reshape makes it $(32,1)$. No error fires; the code centers by the wrong axis and your model quietly
  learns from corrupted inputs. There's no exception to catch — only a wrong number. The professional habit: assert the
  shape you expect (<code>assert mean.shape == (10,)</code>) at boundaries, and print <code>.shape</code> whenever a
  result looks off. Broadcasting is a superpower for writing batch math without loops, and a footgun for exactly the
  same reason. This sets up the formal treatment of axes, norms, and distances in Track 4.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["2.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 60 minutes, no calculator beyond arithmetic.",

  intro: String.raw`<p>This exam covers all of Track 2: vectors and shapes, addition and linear combinations, the dot
  product and attention scores, matrix multiplication (both views), and transpose / identity / Hadamard / broadcasting.
  <strong>Give yourself 60 minutes.</strong> Work each problem fully before opening its solution — the goal is to
  <em>produce</em> the reasoning, not recognize it. Roughly half require calculation. Score yourself with the rubric at
  the bottom.</p>`,

  problems: [
    { level: "easy", prompt: "Let $\\mathbf a=[3,-1,2]$, $\\mathbf b=[1,4,0]$. Compute $2\\mathbf a-\\mathbf b$ and $\\mathbf a\\cdot\\mathbf b$.",
      solution: "$2\\mathbf a=[6,-2,4]$, so $2\\mathbf a-\\mathbf b=[5,-6,4]$. Dot product $=3\\cdot1+(-1)\\cdot4+2\\cdot0=3-4+0=-1$." },
    { level: "easy", prompt: "Give the shape of $A^\\top$ if $A$ is $(7\\times2)$, and state what $I_7 A$ equals.",
      solution: "$A^\\top$ is $(2\\times7)$. $I_7 A=A$ (the $7\\times7$ identity leaves a matrix with 7 rows unchanged)." },
    { level: "med", prompt: "Compute $\\begin{bmatrix}2&1\\\\1&3\\end{bmatrix}\\begin{bmatrix}1\\\\-1\\end{bmatrix}$ two ways: row·column and combine-the-columns.",
      solution: "Row·column: row1 $=2\\cdot1+1\\cdot(-1)=1$; row2 $=1\\cdot1+3\\cdot(-1)=-2$ → $[1,-2]$. Columns: $1\\cdot[2,1]+(-1)\\cdot[1,3]=[2,1]-[1,3]=[1,-2]$. Same result." },
    { level: "med", prompt: "Are $\\mathbf u=[2,1,-1]$ and $\\mathbf v=[1,-1,1]$ orthogonal? Find their cosine similarity.",
      solution: "$\\mathbf u\\cdot\\mathbf v=2-1-1=0$ → orthogonal. Cosine $=0/(\\lVert u\\rVert\\lVert v\\rVert)=0$, confirming a 90° angle regardless of the (nonzero) lengths." },
    { level: "med", prompt: "Queries $Q$ are $(n\\times d)$ and keys $K$ are $(n\\times d)$ with $d=64$. What operation gives the score matrix, what is its shape, and why divide by $8$?",
      solution: "Scores $=QK^\\top$, shape $(n\\times n)$ (inner dim $d$ contracts). Divide by $\\sqrt d=\\sqrt{64}=8$ because dot products of $d$-dim vectors have standard deviation $\\sim\\sqrt d$; scaling keeps the softmax from saturating (Lesson 2.3 Deep Dive)." },
    { level: "med", prompt: "Show $(AB)^\\top=B^\\top A^\\top$ for $A=\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}$, $B=\\begin{bmatrix}0&1\\\\1&0\\end{bmatrix}$.",
      solution: "$AB=\\begin{bmatrix}2&1\\\\4&3\\end{bmatrix}$ so $(AB)^\\top=\\begin{bmatrix}2&4\\\\1&3\\end{bmatrix}$. $B^\\top=\\begin{bmatrix}0&1\\\\1&0\\end{bmatrix}$, $A^\\top=\\begin{bmatrix}1&3\\\\2&4\\end{bmatrix}$, product $=\\begin{bmatrix}2&4\\\\1&3\\end{bmatrix}.\\;\\checkmark$" },
    { level: "med", prompt: "Write $[7,4]$ as a linear combination of $[1,2]$ and $[3,1]$.",
      solution: "Solve $c_1+3c_2=7$, $2c_1+c_2=4$. From the second, $c_2=4-2c_1$; sub: $c_1+3(4-2c_1)=7\\Rightarrow c_1+12-6c_1=7\\Rightarrow -5c_1=-5\\Rightarrow c_1=1$, $c_2=2$. So $[7,4]=1\\cdot[1,2]+2\\cdot[3,1]$." },
    { level: "hard", prompt: "$X$ has shape $(B,d)$ (a batch). You want to add a per-feature bias $\\mathbf b$ of shape $(d,)$. Explain why $X+\\mathbf b$ is correct and why reshaping $\\mathbf b$ to $(B,1)$ would be a silent bug.",
      solution: "Broadcasting aligns from the right: $(B,d)$ with $(d,)$ matches the $d$ axis and stretches the bias across all $B$ rows — exactly 'add this feature bias to every example.' Reshaping to $(B,1)$ instead matches the first axis and stretches across columns, adding a per-<em>example</em> constant to all features. Both are legal shapes, so NumPy raises no error — the second just computes the wrong thing." },
    { level: "hard", prompt: "Explain, with a one-line algebra argument, why stacking two dense layers with no activation cannot be more expressive than one.",
      solution: "$W_2(W_1\\mathbf x)=(W_2W_1)\\mathbf x=W_{\\text{eff}}\\mathbf x$ by associativity — the two weight matrices collapse into a single matrix $W_{\\text{eff}}$. So the composition is just one linear map; depth adds nothing without a nonlinearity between the multiplies." },
    { level: "hard", prompt: "AI task: after $T$ gradient steps $\\mathbf w\\leftarrow\\mathbf w-\\eta\\mathbf g_t$, express $\\mathbf w_T$ in terms of $\\mathbf w_0$ and the gradients, and state how momentum changes the coefficients.",
      solution: "Unrolling: $\\mathbf w_T=\\mathbf w_0-\\eta\\sum_{t=0}^{T-1}\\mathbf g_t$ — a linear combination of all gradients with equal weights $\\eta$. Momentum replaces those equal weights with exponentially-decaying ones ($\\mathbf m_t=\\sum_k\\beta^{k}\\mathbf g_{t-k}$), so recent gradients dominate and old ones fade as $\\beta^{k}$, smoothing the path (Track 8)." }
  ],

  rubric: String.raw`<p>Count the problems you solved correctly <em>before</em> opening the solution.</p>
  <ul>
    <li><strong>9–10:</strong> You own Linear Algebra I. Move to Track 3 (Systems, Spaces & Rank) with confidence.</li>
    <li><strong>7–8:</strong> Solid. Re-read the Deep Dive of any lesson you missed, then advance.</li>
    <li><strong>5–6:</strong> The mechanics are landing but the "two views" aren't yet automatic. Redo Lessons 2.3 and 2.4 and their practice.</li>
    <li><strong>Below 5:</strong> Re-work the track in order; focus on shapes — most errors here are shape errors in disguise.</li>
  </ul>
  <p>Be honest about the calculation problems especially: an answer you can <em>derive</em> under time pressure is the
  real test of fluency.</p>`
};
