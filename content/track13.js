/* ============================================================
   TRACK 13 — Numerical Methods & Computational Math
   13.1 Floating Point · 13.2 Stability & Log-Sum-Exp ·
   13.3 Matrix Factorizations · 13.4 Automatic Differentiation ·
   13.5 Gradient Checking & Mixed Precision · 13.E Track Exam
   ============================================================ */
(window.LESSON_CONTENT ||= {})["13.1"] = {
  subtitle: "Computers store reals with finite bits — so precision and range are budgets you must manage.",

  aiMoment: String.raw`<p>Training a large model in <strong>float16</strong> can diverge to <code>NaN</code> while
  <strong>bfloat16</strong> sails through — same 16 bits, split differently between range and precision. Weight updates
  smaller than machine epsilon silently vanish; activations above 65504 overflow float16. Understanding floating point is
  the difference between a model that trains and one that quietly produces garbage.</p>`,

  plainEnglish: String.raw`<p>A computer can't store most real numbers exactly — it keeps a fixed number of digits, like
  scientific notation with a budget. That budget splits into <strong>range</strong> (how big/small a number can be) and
  <strong>precision</strong> (how many significant digits). Rounding to fit the budget introduces small errors that can
  accumulate.</p>`,

  intuition: String.raw`<p>Representable numbers aren't evenly spaced — they cluster densely near zero and spread out for
  large magnitudes (like scientific notation, where the gap grows with the exponent). Between two representable values,
  everything rounds to the nearest one.</p>
  <figure class="figure">
  <svg viewBox="0 0 300 90" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Floating point numbers are denser near zero">
    <line x1="20" y1="55" x2="285" y2="55" stroke="#94a3b8"/>
    <g stroke="#4f46e5">
    <line x1="30" y1="48" x2="30" y2="62"/><line x1="42" y1="48" x2="42" y2="62"/><line x1="56" y1="48" x2="56" y2="62"/><line x1="74" y1="48" x2="74" y2="62"/>
    <line x1="98" y1="48" x2="98" y2="62"/><line x1="130" y1="48" x2="130" y2="62"/><line x1="172" y1="48" x2="172" y2="62"/><line x1="226" y1="48" x2="226" y2="62"/>
    </g>
    <text x="24" y="78" font-size="10" fill="#64748b" font-family="sans-serif">0</text>
    <text x="150" y="30" font-size="10" fill="#4f46e5" font-family="sans-serif">gaps grow with magnitude →</text>
  </svg>
  <figcaption>Representable floats are dense near 0, sparse for large values — precision is relative, not absolute.</figcaption>
  </figure>`,

  formalism: String.raw`<p>IEEE 754 stores a number as $\pm\, m\times 2^{e}$: a <strong>sign</strong>, an <strong>exponent</strong>
  $e$ (sets range), and a <strong>mantissa</strong> $m$ (sets precision). The formats:</p>
  <div class="tag-note"><strong>float32</strong>: 1 sign, 8 exponent, 23 mantissa — range $\sim10^{\pm38}$, $\varepsilon\approx2^{-23}\approx1.2\times10^{-7}$.
  &nbsp;<strong>float16</strong>: 1/5/10 — range only $\sim\pm65504$, $\varepsilon\approx2^{-10}\approx10^{-3}$.
  &nbsp;<strong>bfloat16</strong>: 1/8/7 — float32's range, but $\varepsilon\approx2^{-7}\approx8\times10^{-3}$.</div>
  <p><strong>Machine epsilon</strong> $\varepsilon$ is the gap between 1.0 and the next representable number — the
  relative rounding error.</p>`,

  derivation: String.raw`<p><strong>Why float16 overflows but bfloat16 doesn't — it's the exponent bits.</strong></p>
  <p><strong>Step 1 — range comes from the exponent.</strong> With $E$ exponent bits, the largest exponent is about
  $2^{E-1}$. float16 has $E=5$ → max exponent $\approx15$ → largest value $\approx2^{16}\approx65504$. bfloat16 has $E=8$
  (same as float32) → max $\approx2^{128}\approx3.4\times10^{38}$.</p>
  <p><strong>Step 2 — precision comes from the mantissa.</strong> With $M$ mantissa bits, $\varepsilon=2^{-M}$. float16
  ($M=10$) is more precise than bfloat16 ($M=7$), but bfloat16's extra exponent bits give it vastly more range.</p>
  <p><strong>Step 3 — the tradeoff.</strong> Both are 16 bits, so more exponent means less mantissa. Deep-learning values
  (gradients, activations) span many orders of magnitude, so <em>range matters more than precision</em> — which is why
  bfloat16, despite being <em>less</em> precise, trains more robustly than float16. $\blacksquare$</p>`,

  code: [
    { label: "Rounding error is real: 0.1 + 0.2 ≠ 0.3", src: String.raw`
import numpy as np
print("0.1 + 0.2 ==", 0.1 + 0.2, " (not exactly 0.3)")
print("equal to 0.3?", (0.1 + 0.2) == 0.3)         # False!
print("machine epsilon (float64):", np.finfo(np.float64).eps)
print("machine epsilon (float32):", np.finfo(np.float32).eps)
` },
    { label: "float16 overflows; bfloat16 keeps the range", src: String.raw`
import numpy as np
big = np.float16(70000)                 # exceeds float16 max (65504)
print("float16(70000)  =", big)         # inf  -> overflow

x = np.float16(1.0) + np.float16(1e-4)  # below float16 precision
print("1 + 1e-4 in fp16 =", x, "(the tiny term vanishes)")

print("float16  max =", np.finfo(np.float16).max)     # 65504
print("float32  max =", np.finfo(np.float32).max)     # 3.4e38
# bfloat16 shares float32's exponent range (~3.4e38) with less mantissa
` }
  ],

  keyPoints: [
    "Floats store $\\pm m\\times2^{e}$: exponent = range, mantissa = precision.",
    "Machine epsilon $\\varepsilon=2^{-M}$ is the relative rounding error ($\\sim10^{-7}$ for float32).",
    "float16: high precision, tiny range (overflow at 65504); bfloat16: float32 range, low precision.",
    "For deep learning, range usually matters more than precision — hence bfloat16.",
    "Representable numbers are dense near 0 and sparse for large magnitudes (relative precision)."
  ],

  commonMistakes: [
    { wrong: "Testing floats for exact equality.", why: "$0.1+0.2\\neq0.3$ in binary floating point. Compare with a tolerance (<code>abs(a-b)&lt;1e-6</code>), never <code>==</code>." },
    { wrong: "Assuming float16 and bfloat16 are interchangeable.", why: "They trade range vs precision oppositely. float16 needs loss scaling to avoid gradient underflow; bfloat16 usually doesn't, because it keeps float32's range." },
    { wrong: "Adding a tiny number to a large one and expecting it to count.", why: "If the addend is below the large number's epsilon-scaled gap, it's rounded away entirely — the source of vanishing updates and why optimizer state is often kept in float32." }
  ],

  quiz: [
    { q: "In IEEE 754, the exponent bits control…", options: ["the range (magnitude)", "the precision", "the sign", "rounding mode"], answer: 0,
      explain: "Exponent = range; mantissa = precision." },
    { q: "float16 overflows to infinity above roughly…", options: ["65504", "3.4e38", "1e7", "1.0"], answer: 0,
      explain: "Its 5 exponent bits cap the max at 65504 — a real training hazard." },
    { q: "bfloat16 vs float16: bfloat16 has…", options: ["more range, less precision", "less range, more precision", "identical", "more of both"], answer: 0,
      explain: "bfloat16 keeps float32's 8 exponent bits (range) at the cost of mantissa (precision)." },
    { q: "Does $0.1+0.2==0.3$ in float64?", options: ["no", "yes", "only in float32", "only in float16"], answer: 0,
      explain: "Binary floating point can't represent these decimals exactly, so the sum is slightly off." },
    { q: "Machine epsilon for float32 is about…", options: ["$10^{-7}$", "$10^{-16}$", "$10^{-3}$", "$1$"], answer: 0,
      explain: "$2^{-23}\\approx1.2\\times10^{-7}$ — ~7 significant decimal digits." }
  ],

  practice: [
    { level: "easy", prompt: "Why can't you use $==$ to compare two computed floats?", solution: "Rounding makes mathematically-equal expressions differ in their last bits (e.g. $0.1+0.2\\neq0.3$). Use $|a-b|<\\text{tol}$ instead." },
    { level: "med", prompt: "A model's gradients are around $10^{-8}$. Which 16-bit format risks losing them, and what's the fix?", solution: "float16's smallest normal is ~$6\\times10^{-5}$, so $10^{-8}$ gradients underflow to 0. Fix: <em>loss scaling</em> (multiply the loss by a large constant so gradients land in range, then unscale), or use bfloat16, which has float32's range and won't underflow there." },
    { level: "med", prompt: "Why is optimizer state (e.g. Adam's moments) often kept in float32 even during mixed-precision training?", solution: "Moment updates accumulate many tiny increments; in float16 each increment can be below epsilon relative to the running value and vanish, so the state stops updating (stale). Keeping a float32 'master' copy preserves the small updates while the forward/backward pass runs fast in 16-bit." },
    { level: "hard", prompt: "AI task: explain why bfloat16 became the default for training large models despite being less precise than float16.", solution: "Training values — activations, gradients, and especially their products across many layers — span a huge dynamic range, occasionally spiking large or shrinking tiny. float16's narrow range ($\\pm65504$, smallest normal ~$6\\times10^{-5}$) causes overflow (→NaN) and underflow (→0), requiring fragile loss-scaling tuning. bfloat16 keeps float32's 8 exponent bits, so it has the <em>same range</em> and rarely overflows/underflows; its lower precision (7 mantissa bits) is tolerable because deep-learning training is robust to small per-value noise (SGD is already noisy). So bfloat16 trades precision you can afford to lose for range you can't — making it a near drop-in for float32 at half the memory/bandwidth, which is why hardware and frameworks standardized on it." }
  ],

  deepDive: String.raw`<p><strong>Why deep learning tolerates low precision but not low range.</strong></p>
  <p>It seems paradoxical that networks train fine in 16 bits (or even 8) when scientific computing often demands 64. The
  reason is that SGD is a <em>noisy, iterative, self-correcting</em> process. Each step already injects gradient noise
  from minibatch sampling (Track 7), so a little extra rounding noise in the weights is just more of the same — averaged
  away over thousands of steps, and often even mildly regularizing. Precision loss degrades gracefully. <strong>Range</strong>
  loss does not: a single overflow produces <code>inf</code>, one <code>inf−inf</code> gives <code>NaN</code>, and NaN
  contaminates every subsequent computation, destroying the run instantly. There's no averaging away a NaN.</p>
  <p>This asymmetry shapes modern numerics. Mixed-precision training runs matmuls in 16-bit for speed but accumulates in
  float32 to protect sums from precision loss; it keeps a float32 master copy of weights so tiny updates survive; and it
  uses loss scaling (float16) or wide-range formats (bfloat16) to protect gradients from underflow. Newer 8-bit formats
  (FP8) push the same logic further with per-tensor scaling factors. The through-line: guard the range at all costs,
  spend precision freely. Knowing which bits buy range and which buy precision — this lesson's core — is what lets you
  read a training divergence and know whether to reach for loss scaling, a wider format, or float32 accumulation. The
  next lesson turns to the classic instability that even float32 doesn't save you from: exponentials in softmax.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["13.2"] = {
  subtitle: "The same math can be safe or explosive in floating point — softmax and the log-sum-exp trick.",

  aiMoment: String.raw`<p>Compute softmax naively on a logit of 1000 and you get <code>NaN</code> — $e^{1000}=\infty$.
  Every deep-learning framework's softmax, <code>log_softmax</code>, and cross-entropy quietly use the <strong>log-sum-exp
  trick</strong> to avoid this. Numerical stability — choosing an algorithm that doesn't amplify rounding error or
  overflow — is the difference between a loss that computes and one that returns garbage.</p>`,

  plainEnglish: String.raw`<p>Two formulas can be mathematically identical yet behave completely differently on a
  computer: one stays accurate, the other overflows or loses all its digits. A <strong>numerically stable</strong>
  algorithm is written to avoid those traps — most famously by <strong>subtracting the maximum</strong> before
  exponentiating in softmax.</p>`,

  intuition: String.raw`<p>The exponentials in softmax blow up for large inputs. But softmax is unchanged if you shift all
  logits by the same constant — so subtract the largest logit first. Then every exponent is $\le0$, every $e^{(\cdot)}\in(0,1]$,
  and nothing overflows.</p>
  <figure class="figure">
  <svg viewBox="0 0 300 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Subtract the max so exponents are non-positive">
    <text x="20" y="34" font-size="11" fill="#dc2626" font-family="sans-serif">naive: e^1000 = ∞  → NaN</text>
    <line x1="30" y1="50" x2="120" y2="50" stroke="#94a3b8" marker-end="url(#ns)"/>
    <text x="130" y="54" font-size="11" fill="#0d9488" font-family="sans-serif">subtract M = max(z)</text>
    <text x="20" y="90" font-size="11" fill="#0d9488" font-family="sans-serif">stable: e^(z−M) ≤ 1  → safe</text>
    <defs><marker id="ns" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>Shifting logits by their max leaves softmax unchanged but keeps every exponential in a safe range.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>log-sum-exp</strong> identity, with $M=\max_i z_i$:</p>
  $$\log\sum_i e^{z_i}=M+\log\sum_i e^{z_i-M}.$$
  <p>Stable <strong>softmax</strong>: $\sigma(z)_i=\dfrac{e^{z_i-M}}{\sum_j e^{z_j-M}}$. Stable
  <strong>cross-entropy</strong> combines the log into the sum (never forming the raw probability):
  $-\log\sigma(z)_y=-(z_y-M)+\log\sum_j e^{z_j-M}$. <strong>Catastrophic cancellation</strong> is the other classic trap:
  subtracting two nearly-equal numbers loses significant digits.</p>`,

  derivation: String.raw`<p><strong>The log-sum-exp trick, rigorously.</strong> Let $M=\max_i z_i$.</p>
  <p><strong>Step 1 — add and subtract $M$ in each exponent:</strong> $\sum_i e^{z_i}=\sum_i e^{(z_i-M)+M}=e^{M}\sum_i e^{z_i-M}.$</p>
  <p><strong>Step 2 — take the log</strong> and use $\log(e^M\cdot S)=M+\log S$:
  $\log\sum_i e^{z_i}=M+\log\sum_i e^{z_i-M}.$ $\blacksquare$</p>
  <p><strong>Why it's safe:</strong> every $z_i-M\le0$, so each $e^{z_i-M}\in(0,1]$ — no overflow. The largest term is
  exactly $e^0=1$, so the sum is $\ge1$ and its log is well-defined — no underflow-to-$-\infty$ either.</p>
  <hr class="soft">
  <p><strong>Catastrophic cancellation example.</strong> Computing variance as $E[x^2]-(E[x])^2$ for data near a large
  mean subtracts two big, nearly-equal numbers, so the small true variance is swamped by rounding noise (it can even come
  out negative). The stable fix subtracts the mean <em>first</em>: $\tfrac1n\sum(x_i-\bar x)^2$ — no large cancellation.</p>`,

  code: [
    { label: "Naive softmax overflows; the stable one doesn't", src: String.raw`
import numpy as np
z = np.array([1000.0, 1001.0, 1002.0])

naive = np.exp(z) / np.exp(z).sum()          # e^1000 = inf -> nan
print("naive softmax :", naive)

M = z.max()
stable = np.exp(z - M) / np.exp(z - M).sum()  # subtract the max first
print("stable softmax:", np.round(stable, 4))  # [0.09, 0.2447, 0.6652]
` },
    { label: "Log-sum-exp and catastrophic cancellation", src: String.raw`
import numpy as np
z = np.array([1000.0, 1001.0, 1002.0])
M = z.max()
lse = M + np.log(np.exp(z - M).sum())         # stable log-sum-exp
print("log-sum-exp   :", round(float(lse), 4))   # ~1002.407, finite

# catastrophic cancellation in variance
x = np.array([1e8 + 1, 1e8 + 2, 1e8 + 3], dtype=np.float32)
unstable = np.mean(x**2) - np.mean(x)**2      # subtract huge near-equal numbers
stable   = np.mean((x - x.mean())**2)         # center first
print("unstable var  :", unstable, " stable var:", round(float(stable), 4))
` }
  ],

  keyPoints: [
    "Numerical stability = an algorithm that doesn't amplify rounding error or overflow.",
    "Softmax is shift-invariant; subtract $\\max_i z_i$ so every exponent is $\\le0$.",
    "$\\log\\sum_i e^{z_i}=M+\\log\\sum_i e^{z_i-M}$ — the trick inside every log-softmax.",
    "Fuse log into softmax (log-sum-exp) rather than computing log(softmax(x)).",
    "Catastrophic cancellation: subtracting nearly-equal numbers destroys precision — restructure to avoid it."
  ],

  commonMistakes: [
    { wrong: "Computing <code>log(softmax(x))</code> in two steps.", why: "softmax can underflow to 0, then log gives $-\\infty$. Use <code>log_softmax</code> (log-sum-exp) which never forms the raw 0." },
    { wrong: "Exponentiating raw logits.", why: "Large logits overflow ($e^{1000}=\\infty$). Always subtract the max before <code>exp</code> in softmax/cross-entropy." },
    { wrong: "Computing variance as $E[x^2]-E[x]^2$ on large-mean data.", why: "Catastrophic cancellation: the two terms are nearly equal and huge, so the small variance drowns in rounding error. Center the data first." }
  ],

  quiz: [
    { q: "Naive softmax on logits $[1000,1001,1002]$ gives…", options: ["NaN (overflow)", "$[0.09,0.24,0.67]$", "$[1,1,1]$", "an error message"], answer: 0,
      explain: "$e^{1000}=\\infty$, and $\\infty/\\infty=$ NaN. The stable version subtracts the max first." },
    { q: "The log-sum-exp trick subtracts…", options: ["the max logit", "the mean", "the min", "zero"], answer: 0,
      explain: "Subtracting $M=\\max z_i$ makes every exponent $\\le0$, avoiding overflow." },
    { q: "Softmax is unchanged if you…", options: ["shift all logits by a constant", "scale all logits", "square them", "negate them"], answer: 0,
      explain: "$\\frac{e^{z_i-c}}{\\sum e^{z_j-c}}=\\frac{e^{z_i}}{\\sum e^{z_j}}$ — the shift cancels, which is why subtracting the max is free." },
    { q: "Catastrophic cancellation happens when you…", options: ["subtract two nearly-equal large numbers", "add two small numbers", "multiply by zero", "take a log"], answer: 0,
      explain: "The leading digits cancel, leaving only noisy trailing bits — relative error explodes." },
    { q: "Why use <code>log_softmax</code> instead of <code>log(softmax(x))</code>?", options: ["it folds the log into the sum, avoiding underflow of the raw probability", "it's a different function", "it changes the gradient", "it's only for GPUs"], answer: 0,
      explain: "Fusing avoids forming a possibly-zero probability that log would send to $-\\infty$." }
  ],

  practice: [
    { level: "easy", prompt: "Rewrite $\\log(e^{a}+e^{b})$ in a numerically stable form.", solution: "With $M=\\max(a,b)$: $M+\\log(e^{a-M}+e^{b-M})$. One of the exponents is $e^0=1$, the other $\\le1$ — no overflow." },
    { level: "med", prompt: "Show softmax is invariant to subtracting a constant $c$ from every logit.", solution: "$\\frac{e^{z_i-c}}{\\sum_j e^{z_j-c}}=\\frac{e^{-c}e^{z_i}}{e^{-c}\\sum_j e^{z_j}}=\\frac{e^{z_i}}{\\sum_j e^{z_j}}$ — the $e^{-c}$ cancels. So choosing $c=\\max_i z_i$ is free and prevents overflow." },
    { level: "med", prompt: "Why can $E[x^2]-E[x]^2$ produce a negative 'variance' in floating point?", solution: "For data with a large mean, $E[x^2]$ and $E[x]^2$ are both huge and nearly equal, so their difference (the tiny true variance) is computed from the noisy trailing bits — rounding can make it slightly negative. Variance is provably $\\ge0$ mathematically; the negative result is pure floating-point cancellation error." },
    { level: "hard", prompt: "AI task: derive the numerically stable cross-entropy from logits (no intermediate softmax).", solution: "For true class $y$, cross-entropy $=-\\log\\text{softmax}(z)_y=-\\big(z_y-\\log\\sum_j e^{z_j}\\big)$. Apply log-sum-exp with $M=\\max_j z_j$: $=-(z_y-M)+\\log\\sum_j e^{z_j-M}$. This never exponentiates a raw large logit and never forms the softmax probability (which could underflow to 0 and break the log). It's exactly what <code>cross_entropy(logits, y)</code> computes — the fused, stable path — which is why you pass <em>logits</em>, not probabilities, to the loss." }
  ],

  deepDive: String.raw`<p><strong>Stability is a property of the algorithm, not the problem — and it's everywhere in ML.</strong></p>
  <p>The same mathematical function has many algorithmic implementations, and they differ wildly in floating point. This
  is why frameworks obsess over 'fused' numerically-stable kernels: <code>log_softmax</code>, <code>logsumexp</code>,
  <code>log1p</code>/<code>expm1</code> (accurate for small arguments), stable <code>sigmoid</code> and
  <code>softplus</code>, and cross-entropy-from-logits. Each avoids a specific trap — overflow, underflow, or
  cancellation — that the naive expression would hit. Passing <em>logits</em> (not probabilities) to a loss function is a
  stability decision: it lets the framework take the fused, safe path.</p>
  <p>The discipline generalizes far beyond softmax. Layer/batch normalization must compute variance stably (center
  first, or use Welford's online formula) or risk negative variances. Attention scales by $1/\sqrt d$ partly to keep
  logits in a safe range before the softmax. Log-space accumulation (summing log-probabilities instead of multiplying
  probabilities, Track 1) is stability by design. Even the choice to accumulate matmuls in float32 (Lesson 13.1) is a
  cancellation-avoidance move. The meta-lesson: when a computation misbehaves, don't just add precision — ask whether a
  mathematically-equivalent reformulation is stable. That instinct, plus knowing the three classic failure modes, is
  what separates code that works on toy inputs from code that survives real logits, real gradients, and real scale. It
  also sets up the crown jewel of computational ML — automatic differentiation — where a careful algorithm computes exact
  gradients at a fraction of the naive cost.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["13.3"] = {
  subtitle: "Break a matrix into simpler factors to solve fast and stably — and know which factorization to reach for.",

  aiMoment: String.raw`<p>You almost never invert a matrix in practice (Track 4.3) — you <strong>factorize and
  solve</strong>. Which factorization depends on the matrix: <strong>LU</strong> for a general system, <strong>Cholesky</strong>
  for a symmetric positive-definite one (covariances, Gram matrices, Gaussian sampling — at half the cost), and
  <strong>QR</strong> for least-squares regression (the stable choice, Track 3.5). Picking the right one is everyday
  numerical practice.</p>`,

  plainEnglish: String.raw`<p>Solving $A\mathbf x=\mathbf b$ directly is expensive and fragile. Instead, factor $A$ once
  into simple pieces (triangular or orthogonal), then solving becomes a couple of easy sweeps. Different matrix shapes
  and properties call for different factorizations — each a different tradeoff of speed and stability.</p>`,

  intuition: String.raw`<p>Think of a decision tree. Is the matrix symmetric and positive definite? Use Cholesky — it's
  the fastest. Just square and general? LU. Non-square or least-squares (more equations than unknowns)? QR — the most
  numerically stable.</p>
  <figure class="figure">
  <svg viewBox="0 0 340 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Decision flow for choosing LU, Cholesky, or QR">
    <g font-family="sans-serif" font-size="11">
    <rect x="120" y="12" width="100" height="30" rx="6" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="132" y="31" fill="#4f46e5">solve A x = b</text>
    <line x1="170" y1="42" x2="60" y2="70" stroke="#94a3b8"/><line x1="170" y1="42" x2="170" y2="70" stroke="#94a3b8"/><line x1="170" y1="42" x2="285" y2="70" stroke="#94a3b8"/>
    <rect x="14" y="72" width="96" height="42" rx="6" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="22" y="90" fill="#0d9488">SPD square</text><text x="30" y="106" fill="#0d9488">→ Cholesky</text>
    <rect x="122" y="72" width="96" height="42" rx="6" fill="#fff7ed" stroke="#d97706"/>
    <text x="130" y="90" fill="#d97706">general square</text><text x="150" y="106" fill="#d97706">→ LU</text>
    <rect x="238" y="72" width="96" height="42" rx="6" fill="#faf5ff" stroke="#7c3aed"/>
    <text x="246" y="90" fill="#7c3aed">tall / lst-sq</text><text x="256" y="106" fill="#7c3aed">→ QR</text>
    <text x="20" y="140" font-size="10" fill="#64748b">≈ ⅓n³ (fastest)</text>
    <text x="132" y="140" font-size="10" fill="#64748b">≈ ⅔n³</text>
    <text x="248" y="140" font-size="10" fill="#64748b">≈ 4/3n³ (stablest)</text>
    </g>
  </svg>
  <figcaption>Match the factorization to the matrix: Cholesky (SPD, cheapest), LU (general), QR (least-squares, stablest).</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>LU:</strong> $A=LU$ (lower × upper triangular), any square $A$; solve by forward then back
  substitution; cost $\approx\tfrac23 n^3$. <strong>Cholesky:</strong> $A=LL^\top$ for symmetric positive-definite $A$;
  cost $\approx\tfrac13 n^3$ (half of LU) and it doubles as a positive-definiteness test. <strong>QR:</strong> $A=QR$
  (orthonormal $Q$, upper-triangular $R$); solve least-squares via $R\mathbf x=Q^\top\mathbf b$; cost $\approx\tfrac43 n^3$
  but avoids squaring the condition number (no $A^\top A$).</p>`,

  derivation: String.raw`<p><strong>Why Cholesky is half the cost of LU — and why it needs positive-definiteness.</strong></p>
  <p><strong>Step 1 — LU wastes symmetry.</strong> A general LU computes both $L$ and a separate $U$ ($n^2$ unknowns of
  work, $\approx\tfrac23 n^3$ flops). For a symmetric matrix, $L$ and $U$ carry the same information twice.</p>
  <p><strong>Step 2 — Cholesky exploits it.</strong> Writing $A=LL^\top$ finds a single triangular factor and its
  transpose, halving the work to $\approx\tfrac13 n^3$. The entries come from
  $L_{jj}=\sqrt{A_{jj}-\sum_{k<j}L_{jk}^2}$ and $L_{ij}=\big(A_{ij}-\sum_{k<j}L_{ik}L_{jk}\big)/L_{jj}$.</p>
  <p><strong>Step 3 — the square root demands positive-definiteness.</strong> $L_{jj}$ takes a square root of a diagonal
  quantity; that quantity is positive for <em>every</em> $j$ exactly when $A$ is positive definite (Track 5.3). If it ever
  goes $\le0$, Cholesky fails — which is precisely how libraries <em>test</em> for positive-definiteness. $\blacksquare$
  Plain English: symmetry buys you a $2\times$ speedup, but only if the matrix is a genuine 'bowl'.</p>`,

  code: [
    { label: "Three factorizations, one solve", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

M = rng.normal(size=(4,4)); A = M @ M.T + 4*np.eye(4)   # symmetric positive-definite
b = rng.normal(size=4)

x_solve = np.linalg.solve(A, b)                 # LU under the hood
L = np.linalg.cholesky(A)                        # A = L L^T (SPD only)
y = np.linalg.solve(L, b); x_chol = np.linalg.solve(L.T, y)
Q, R = np.linalg.qr(A); x_qr = np.linalg.solve(R, Q.T @ b)
print("LU/solve :", np.round(x_solve, 4))
print("Cholesky :", np.round(x_chol,  4), " match:", np.allclose(x_solve, x_chol))
print("QR       :", np.round(x_qr,    4), " match:", np.allclose(x_solve, x_qr))
` },
    { label: "Cholesky powers Gaussian sampling", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

Sigma = np.array([[3.0, 1.5],[1.5, 1.0]])       # target covariance (SPD)
L = np.linalg.cholesky(Sigma)                    # Sigma = L L^T
z = rng.normal(size=(200000, 2))                 # standard normal
x = z @ L.T                                       # now Cov(x) ≈ Sigma
print("empirical covariance:\n", np.round(np.cov(x.T), 2))
# x = μ + L z is the multivariate reparameterization trick (Track 10.4)
` }
  ],

  keyPoints: [
    "Factorize-and-solve, don't invert: cheaper and more stable.",
    "LU ($\\approx\\tfrac23 n^3$): general square systems.",
    "Cholesky ($\\approx\\tfrac13 n^3$): symmetric positive-definite — covariances, and a PD test.",
    "QR ($\\approx\\tfrac43 n^3$): least-squares / tall matrices; avoids squaring the condition number.",
    "$A=LL^\\top$ (Cholesky) enables Gaussian sampling: $\\boldsymbol\\mu+L\\mathbf z$."
  ],

  commonMistakes: [
    { wrong: "Using Cholesky on a non-SPD matrix.", why: "It requires symmetric positive-definite input; on anything else the square root fails. (That failure is also the standard way to <em>check</em> positive-definiteness.)" },
    { wrong: "Solving least-squares via the normal equations $A^\\top A$.", why: "That squares the condition number (Track 4.4). QR works with $\\kappa(A)$ directly — prefer it for regression." },
    { wrong: "Re-factorizing for each new right-hand side.", why: "Factor once ($O(n^3)$); then every new $\\mathbf b$ is just two triangular solves ($O(n^2)$). Reuse the factorization." }
  ],

  quiz: [
    { q: "For a symmetric positive-definite system, the fastest factorization is…", options: ["Cholesky", "LU", "QR", "SVD"], answer: 0,
      explain: "Cholesky exploits symmetry for ~half the cost of LU (and requires SPD)." },
    { q: "QR is preferred for least-squares because it…", options: ["avoids squaring the condition number", "is the cheapest", "needs no memory", "requires SPD"], answer: 0,
      explain: "It works with $\\kappa(A)$ instead of $\\kappa(A)^2$ from the normal equations." },
    { q: "Cholesky requires the matrix to be…", options: ["symmetric positive-definite", "square only", "orthogonal", "diagonal"], answer: 0,
      explain: "The square-root steps need a genuine 'bowl' (all positive pivots)." },
    { q: "After factoring $A$ once, solving for a new $\\mathbf b$ costs about…", options: ["$O(n^2)$ (two triangular solves)", "$O(n^3)$", "$O(n)$", "$O(1)$"], answer: 0,
      explain: "Reuse the factors; only the substitution ($O(n^2)$) repeats." },
    { q: "A Cholesky factorization fails (negative under the square root). This tells you…", options: ["the matrix isn't positive definite", "the matrix is singular only", "you need more precision", "nothing"], answer: 0,
      explain: "A failed Cholesky is the standard positive-definiteness test." }
  ],

  practice: [
    { level: "easy", prompt: "Which factorization for solving a general (non-symmetric) square system?", solution: "LU — it works for any square matrix (with pivoting for stability)." },
    { level: "med", prompt: "You must sample from $\\mathcal N(\\boldsymbol\\mu,\\Sigma)$. Which factorization and how?", solution: "Cholesky: $\\Sigma=LL^\\top$. Draw $\\mathbf z\\sim\\mathcal N(\\mathbf0,I)$ and set $\\mathbf x=\\boldsymbol\\mu+L\\mathbf z$; then $\\operatorname{Cov}(\\mathbf x)=L L^\\top=\\Sigma$. This is the multivariate reparameterization trick." },
    { level: "med", prompt: "Why is factor-once-then-substitute much better than computing $A^{-1}$ for many right-hand sides?", solution: "Both start with an $O(n^3)$ step, but forming $A^{-1}$ then multiplying is more work and less stable, and you still pay $O(n^2)$ per solve. Factoring once and back-substituting is the same $O(n^2)$ per solve but avoids ever building the error-prone inverse — cheaper and more accurate." },
    { level: "hard", prompt: "AI task: Gaussian-process regression needs to solve $(\\mathbf K+\\sigma^2 I)^{-1}\\mathbf y$ where $\\mathbf K$ is an SPD kernel matrix. Which factorization, and what extra quantity does it give you for free?", solution: "Use Cholesky: $\\mathbf K+\\sigma^2 I=LL^\\top$ (the added $\\sigma^2 I$ keeps it SPD and well-conditioned). Solve $L\\mathbf a'=\\mathbf y$ then $L^\\top\\boldsymbol\\alpha=\\mathbf a'$ for the predictive mean, at $\\tfrac13 n^3$. For free, Cholesky gives the <em>log-determinant</em>: $\\log\\det(\\mathbf K+\\sigma^2 I)=2\\sum_i\\log L_{ii}$, which is exactly the term needed for the GP's marginal likelihood (used to tune kernel hyperparameters). One factorization serves both the solve and the likelihood — which is why Cholesky is the backbone of GP libraries." }
  ],

  deepDive: String.raw`<p><strong>Factorizations are the hidden engine under 'just call solve'.</strong></p>
  <p>Every high-level linear-algebra call — <code>np.linalg.solve</code>, <code>lstsq</code>, a Gaussian's log-density, a
  Kalman filter update, a Newton step — dispatches to one of these factorizations underneath, chosen for the matrix's
  structure. The art is that structure buys speed and stability: symmetry halves the work (Cholesky), orthogonality
  preserves conditioning (QR), triangularity makes solves trivial (LU's $L$ and $U$). Libraries like LAPACK are, in
  essence, a giant catalog of 'if your matrix looks like <em>this</em>, factor it like <em>that</em>.' Knowing the catalog
  lets you write code that's both faster and more robust than the naive formula.</p>
  <p>The theme also connects the whole numerical-methods track. Conditioning (Lesson 13.1's cousin, Track 4.4) decides
  <em>whether</em> a factorization will be accurate; stability (Lesson 13.2) decides <em>how</em> to compute each step;
  and the SVD (Track 5.4) is the ultimate factorization for the hardest, rank-deficient cases. In ML specifically,
  Cholesky underlies Gaussian processes and natural-gradient/second-order methods (factoring curvature), QR underlies
  stable least-squares and some orthogonalization tricks, and LU underlies general implicit solves. But the single most
  important 'factorization-like' algorithm in deep learning isn't any of these — it's the decomposition of a gradient
  computation into a graph of local derivatives, reused with perfect efficiency. That is automatic differentiation, the
  next and final core lesson.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["13.4"] = {
  subtitle: "Exact gradients by applying the chain rule to a computation graph — this IS backpropagation.",

  aiMoment: String.raw`<p><strong>Backpropagation is reverse-mode automatic differentiation.</strong> Every framework's
  autograd — PyTorch, JAX, TensorFlow — builds a graph of your computation and walks it backward, applying the chain rule
  at each node to get exact gradients of the loss with respect to millions of parameters, all in about the cost of one
  forward pass. It's neither symbolic math (no formula explosion) nor finite differences (no approximation error) — it's
  the algorithm that makes deep learning trainable.</p>`,

  plainEnglish: String.raw`<p><strong>Automatic differentiation</strong> computes exact derivatives by breaking your
  function into elementary operations, recording them as a graph, and mechanically applying the chain rule. Do it
  <strong>forward</strong> (efficient when there are few inputs) or <strong>backward</strong> from the output (efficient
  when there's one output — like a loss). Backward is backprop.</p>`,

  intuition: String.raw`<p>Your computation is a graph: inputs flow through operations to an output. The
  <strong>forward pass</strong> fills in each node's value. The <strong>backward pass</strong> starts from the output with
  gradient 1 and flows gradients back, each node multiplying by its local derivative and adding into its inputs.</p>
  <figure class="figure">
  <svg viewBox="0 0 340 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Computation graph with forward values and backward gradients">
    <g font-family="sans-serif" font-size="11">
    <circle cx="40" cy="45" r="16" fill="#eef0ff" stroke="#4f46e5"/><text x="34" y="49" fill="#4f46e5">a</text>
    <circle cx="40" cy="105" r="16" fill="#eef0ff" stroke="#4f46e5"/><text x="34" y="109" fill="#4f46e5">b</text>
    <circle cx="150" cy="75" r="16" fill="#fff7ed" stroke="#d97706"/><text x="146" y="79" fill="#d97706">×</text>
    <circle cx="260" cy="75" r="16" fill="#f0fdfa" stroke="#0d9488"/><text x="256" y="79" fill="#0d9488">+</text>
    <line x1="56" y1="50" x2="135" y2="70" stroke="#94a3b8" marker-end="url(#ad)"/>
    <line x1="56" y1="100" x2="135" y2="82" stroke="#94a3b8" marker-end="url(#ad)"/>
    <line x1="166" y1="75" x2="243" y2="75" stroke="#94a3b8" marker-end="url(#ad)"/>
    <line x1="276" y1="75" x2="320" y2="75" stroke="#94a3b8" marker-end="url(#ad)"/>
    <text x="316" y="79" fill="#20242c">L</text>
    <text x="70" y="34" font-size="9" fill="#16a34a">forward → values</text>
    <text x="150" y="128" font-size="9" fill="#dc2626">backward ← gradients (×local deriv)</text>
    </g>
    <defs><marker id="ad" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>Forward fills node values; backward flows the loss gradient through each node's local derivative.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Represent the function as a <strong>computational graph</strong> of elementary ops.
  <strong>Forward mode</strong> propagates a derivative $\dot v_i=\partial v_i/\partial x$ alongside each value (a
  Jacobian–vector product, JVP) — one pass per input. <strong>Reverse mode</strong> propagates an adjoint
  $\bar v_i=\partial L/\partial v_i$ backward (a vector–Jacobian product, VJP): seed $\bar L=1$, and for each node
  $\bar v=\sum_{\text{children }c}\bar c\,\dfrac{\partial c}{\partial v}$. For a scalar $L$ and many inputs, reverse mode
  gets <em>all</em> gradients in one backward pass.</p>`,

  derivation: String.raw`<p><strong>Reverse mode is the chain rule, bookkept backward.</strong> Take $L=f(g(h(x)))$ with
  intermediates $u=h(x)$, $v=g(u)$, $L=f(v)$.</p>
  <p><strong>Step 1 — the chain rule:</strong> $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial v}\dfrac{\partial v}{\partial u}\dfrac{\partial u}{\partial x}.$</p>
  <p><strong>Step 2 — evaluate right-to-left, storing partial products.</strong> Start with $\bar L=1$. Then
  $\bar v=\bar L\cdot f'(v)$, then $\bar u=\bar v\cdot g'(u)$, then $\bar x=\bar u\cdot h'(x)$. Each step reuses the
  accumulated adjoint from the step before.</p>
  <p><strong>Step 3 — count the cost.</strong> Every node is visited once on the forward pass and once on the backward
  pass, so the whole gradient costs $O(1)$ times the forward evaluation — <em>independent of the number of inputs</em>.
  Contrast forward mode / finite differences, which need one pass per input ($n$ passes for $n$ parameters). $\blacksquare$
  Plain English: because a network has one loss and millions of parameters, walking backward once beats walking forward a
  million times.</p>`,

  code: [
    { label: "A tiny reverse-mode autograd (≈20 lines)", src: String.raw`
class Value:
    def __init__(self, data, _children=()):
        self.data = data; self.grad = 0.0
        self._backward = lambda: None; self._prev = set(_children)
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def _bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = _bw; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def _bw(): self.grad += o.data*out.grad; o.grad += self.data*out.grad
        out._backward = _bw; return out
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()

a, b = Value(2.0), Value(3.0)
L = a*b + a                                  # L = a*b + a
L.backward()
print("L =", L.data, " dL/da =", a.grad, " dL/db =", b.grad)   # 8, 4, 2
` },
    { label: "Autograd gradient matches finite differences", src: String.raw`
import numpy as np
def f(a, b): return a*b + a                   # same function, plain floats
a, b, h = 2.0, 3.0, 1e-6
da = (f(a+h,b) - f(a-h,b)) / (2*h)            # numeric ∂/∂a
db = (f(a,b+h) - f(a,b-h)) / (2*h)            # numeric ∂/∂b
print("finite-diff dL/da =", round(da,5), " dL/db =", round(db,5))   # 4, 2
print("autograd said     dL/da = 4.0        dL/db = 2.0")
` }
  ],

  keyPoints: [
    "Autodiff applies the chain rule to a computation graph — exact, not approximate or symbolic.",
    "Forward mode (JVP): one pass per input — good when inputs are few.",
    "Reverse mode (VJP) = backprop: one pass per output — good when the output (loss) is scalar.",
    "Reverse mode computes ALL parameter gradients in ~one forward-pass cost.",
    "Each node stores a local derivative and accumulates the upstream gradient (its adjoint)."
  ],

  commonMistakes: [
    { wrong: "Confusing autodiff with numerical (finite-difference) differentiation.", why: "Autodiff is <em>exact</em> (to floating-point) and costs one backward pass; finite differences approximate and need a pass per parameter — useless at scale (only for gradient checking, Lesson 13.5)." },
    { wrong: "Thinking autodiff is symbolic differentiation.", why: "It doesn't build a giant closed-form expression (which explodes in size); it evaluates derivatives numerically along the graph, reusing intermediate values." },
    { wrong: "Using forward mode to train a network.", why: "With millions of parameters and one loss, forward mode needs millions of passes. Reverse mode needs one — the whole reason backprop, not forward-mode AD, powers deep learning." }
  ],

  quiz: [
    { q: "Backpropagation is…", options: ["reverse-mode automatic differentiation", "forward-mode AD", "finite differences", "symbolic differentiation"], answer: 0,
      explain: "It propagates adjoints backward from the scalar loss — reverse-mode AD." },
    { q: "Reverse mode is efficient when there are…", options: ["few outputs, many inputs", "few inputs, many outputs", "equal inputs and outputs", "no inputs"], answer: 0,
      explain: "One backward pass yields all input gradients — ideal for a scalar loss over many parameters." },
    { q: "The cost of reverse-mode AD relative to the forward pass is about…", options: ["$O(1)\\times$", "$O(n)\\times$ (n = #params)", "$O(n^2)\\times$", "$O(\\log n)\\times$"], answer: 0,
      explain: "Each node is visited once forward and once backward — a constant factor, independent of parameter count." },
    { q: "Compared to finite differences, autodiff is…", options: ["exact (to floating point)", "always slower", "an approximation", "symbolic"], answer: 0,
      explain: "It computes true derivatives via the chain rule, with no step-size error." },
    { q: "Forward-mode AD to get all gradients of a scalar loss over $n$ parameters needs…", options: ["$n$ passes", "1 pass", "2 passes", "$\\log n$ passes"], answer: 0,
      explain: "Forward mode differentiates w.r.t. one input per pass — hence reverse mode wins for training." }
  ],

  practice: [
    { level: "easy", prompt: "For $L=a\\cdot b$ at $a=4,b=5$, give $\\partial L/\\partial a$ and $\\partial L/\\partial b$.", solution: "$\\partial L/\\partial a=b=5$, $\\partial L/\\partial b=a=4$ — each input's gradient is the other factor (the product rule / autodiff's local derivative)." },
    { level: "med", prompt: "Why is reverse-mode preferred over forward-mode for neural networks?", solution: "A network maps millions of parameters to a single scalar loss. Reverse mode's cost scales with the number of <em>outputs</em> (one), giving all gradients in one backward pass. Forward mode's cost scales with the number of <em>inputs</em> (millions), needing a pass each. So reverse mode is millions of times cheaper here." },
    { level: "med", prompt: "In the tiny autograd, why does <code>_backward</code> use <code>+=</code> for grads, not <code>=</code>?", solution: "A value can feed into multiple downstream operations (be used more than once). Its total gradient is the sum of contributions from every path to the loss (multivariable chain rule). Accumulating with <code>+=</code> collects all those contributions; overwriting with <code>=</code> would keep only the last path and give wrong gradients." },
    { level: "hard", prompt: "AI task: outline how you'd extend the tiny autograd to train a 2-layer network, and what each op needs.", solution: "Add the ops the network uses, each with a forward value and a <code>_backward</code> encoding its local derivative: <code>+</code> and <code>*</code> (have them), a matmul/linear (grad w.r.t. weights is inputᵀ·upstream, w.r.t. input is upstreamᵀ·W — the transpose rule, Track 2.5), a nonlinearity like <code>tanh</code>/<code>relu</code> (local deriv $1-\\tanh^2$ or the 0/1 mask), and a loss (e.g. MSE or cross-entropy) as the scalar root. Forward: compute $\\hat y=\\text{relu}(xW_1+b_1)W_2+b_2$ and the loss. Backward: call <code>loss.backward()</code>, which topologically orders the graph and runs each node's VJP, filling every parameter's <code>.grad</code>. Then update $W\\leftarrow W-\\eta\\,W.\\text{grad}$ (Track 8). That's exactly the Numerical Methods capstone — a working autograd trains a real MLP in ~100 lines." }
  ],

  deepDive: String.raw`<p><strong>The two orders of the chain rule, and why one built modern AI.</strong></p>
  <p>The chain rule for a deep composition is a product of Jacobians, $J_L\cdots J_2 J_1$. Matrix multiplication is
  associative, so you may evaluate this product in any order — and the order is <em>exactly</em> the choice between
  forward and reverse mode. Multiply right-to-left, starting from the input side, and you compute Jacobian–vector
  products (forward mode); the work scales with the number of inputs. Multiply left-to-right, starting from the scalar
  loss, and you compute vector–Jacobian products (reverse mode); the work scales with the number of outputs. For a
  network — enormous fan-in of parameters, single scalar loss — reverse mode is the difference between one pass and a
  billion. That associativity choice, nothing more, is why training deep nets is feasible.</p>
  <p>This reframes the whole enterprise. A framework's <code>backward</code> for each operation is just its VJP — a rule
  for turning an output gradient into input gradients — and 'building a model' is composing ops whose VJPs the framework
  already knows. Memory is the price: reverse mode must store intermediate activations from the forward pass to use on the
  way back, which is why activation memory dominates training and why gradient checkpointing (recompute instead of store)
  exists. Higher-order derivatives, Hessian–vector products, and <code>jax.grad(jax.grad(...))</code> all fall out of
  composing these modes. Once you see backprop as 'the chain rule, associated in the cheap direction, bookkept over a
  graph,' the magic dissolves into an algorithm — and you can build it yourself, which is precisely the capstone that
  closes this course.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["13.5"] = {
  subtitle: "Catch backprop bugs with finite differences, and train fast in low precision without breaking.",

  aiMoment: String.raw`<p>When you implement a gradient by hand (a custom layer, a research op), <strong>gradient
  checking</strong> against finite differences is how you verify it before trusting a training run. And
  <strong>mixed-precision training</strong> — 16-bit for speed, float32 for the parts that need it, plus
  <strong>loss scaling</strong> — is how modern models train 2–3× faster without diverging. Both are everyday numerical
  hygiene.</p>`,

  plainEnglish: String.raw`<p><strong>Gradient checking</strong> compares your analytic gradient to a slow-but-simple
  finite-difference estimate; if they match closely, your backprop is probably correct. <strong>Mixed precision</strong>
  runs most computation in fast 16-bit numbers while keeping a high-precision copy for the delicate accumulation, using
  <strong>loss scaling</strong> to stop tiny gradients from vanishing.</p>`,

  intuition: String.raw`<p>To check a gradient, nudge an input by a tiny $h$ both ways and measure the slope — that
  central difference should match your analytic value. To train in float16, gradients can be too small to represent, so
  multiply the loss by a big factor first (shifting gradients into range), then divide it back out before updating.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Loss scaling shifts gradients into the representable range">
    <text x="16" y="28" font-size="11" fill="#dc2626" font-family="sans-serif">grad ≈ 1e-8  → underflows to 0 in fp16</text>
    <line x1="30" y1="44" x2="120" y2="44" stroke="#94a3b8" marker-end="url(#gs)"/>
    <text x="128" y="48" font-size="11" fill="#0d9488" font-family="sans-serif">scale loss ×S (e.g. 2¹⁶)</text>
    <text x="16" y="82" font-size="11" fill="#0d9488" font-family="sans-serif">grad ×S ≈ 6e-4  → representable</text>
    <line x1="30" y1="98" x2="120" y2="98" stroke="#94a3b8" marker-end="url(#gs)"/>
    <text x="128" y="102" font-size="11" fill="#4f46e5" font-family="sans-serif">unscale (÷S) before the step</text>
    <defs><marker id="gs" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>Loss scaling multiplies gradients into float16's range, then divides them out before the optimizer step.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Central difference:</strong> $g_{\text{num}}=\dfrac{f(x+h)-f(x-h)}{2h}$, accurate to
  $O(h^2)$ (vs the one-sided forward difference's $O(h)$). The check passes when the <strong>relative error</strong></p>
  $$\frac{\lvert g_{\text{analytic}}-g_{\text{num}}\rvert}{\max(\lvert g_{\text{analytic}}\rvert,\lvert g_{\text{num}}\rvert,\ \epsilon)}$$
  <p>is tiny (below $\sim10^{-5}$ in float64). <strong>Mixed precision:</strong> forward/backward in 16-bit, master
  weights and accumulation in float32, gradients scaled by $S$ (statically or dynamically) to avoid underflow.</p>`,

  derivation: String.raw`<p><strong>Why the central difference is $O(h^2)$ — and why $h$ can't be too small.</strong></p>
  <p><strong>Step 1 — Taylor-expand both sides:</strong>
  $f(x\pm h)=f(x)\pm f'(x)h+\tfrac12 f''(x)h^2\pm\tfrac16 f'''(x)h^3+\cdots$</p>
  <p><strong>Step 2 — subtract:</strong> the even terms cancel, leaving $f(x+h)-f(x-h)=2f'(x)h+\tfrac13 f'''(x)h^3+\cdots$</p>
  <p><strong>Step 3 — divide by $2h$:</strong> $\dfrac{f(x+h)-f(x-h)}{2h}=f'(x)+\tfrac16 f'''(x)h^2+\cdots=f'(x)+O(h^2).$
  So halving $h$ cuts the truncation error fourfold. $\blacksquare$</p>
  <p><strong>The catch (Lessons 13.1–13.2):</strong> too-small $h$ makes $f(x+h)\approx f(x-h)$, so their difference
  suffers catastrophic cancellation. Truncation error falls as $h^2$ but rounding error grows as $\varepsilon/h$; the
  sweet spot is around $h\sim\varepsilon^{1/3}$ (roughly $10^{-5}$ in float64). There's a best $h$ — smaller is not
  better.</p>`,

  code: [
    { label: "Gradient checking: analytic vs finite difference", src: String.raw`
import numpy as np

def f(x):  return np.sum(x**3) + 2*np.sum(x)        # scalar function
def grad_analytic(x): return 3*x**2 + 2             # hand-derived gradient

x = np.array([1.0, -2.0, 0.5])
g = grad_analytic(x)
h = 1e-5
g_num = np.array([(f(x+h*e)-f(x-h*e))/(2*h) for e in np.eye(len(x))])
rel = np.abs(g-g_num) / np.maximum(np.abs(g)+np.abs(g_num), 1e-12)
print("analytic:", np.round(g, 5))
print("numeric :", np.round(g_num, 5))
print("max relative error:", f"{rel.max():.2e}", "-> PASS" if rel.max()<1e-6 else "-> FAIL")
` },
    { label: "Loss scaling rescues tiny float16 gradients", src: String.raw`
import numpy as np
grad = np.float16(1e-8)                    # a realistic small gradient
print("fp16 grad as-is      :", grad)      # 0.0  -> underflow, update lost

S = np.float32(2**16)                       # loss-scale factor
scaled = np.float16(np.float32(1e-8) * S)  # scale up before casting to fp16
print("fp16 scaled grad     :", scaled)    # representable (~6.5e-4)
recovered = np.float32(scaled) / S         # unscale in float32 before the step
print("recovered in fp32    :", recovered) # back to ~1e-8, update preserved
` }
  ],

  keyPoints: [
    "Central difference $\\frac{f(x+h)-f(x-h)}{2h}$ is $O(h^2)$ accurate — use it to check gradients.",
    "Pass a gradient check when the relative error is $\\lesssim10^{-5}$ (float64).",
    "There's an optimal $h\\sim\\varepsilon^{1/3}$; too small triggers cancellation.",
    "Mixed precision: 16-bit compute, float32 master weights and accumulation.",
    "Loss scaling multiplies the loss (and gradients) by $S$ to escape float16 underflow, then unscales before the step."
  ],

  commonMistakes: [
    { wrong: "Using a one-sided difference for gradient checking.", why: "Forward difference is only $O(h)$ accurate and gives more false failures. Use the central (two-sided) difference — $O(h^2)$." },
    { wrong: "Making $h$ as small as possible.", why: "Below $\\sim\\varepsilon^{1/3}$, $f(x+h)$ and $f(x-h)$ round to nearly the same value and their difference is noise (catastrophic cancellation). Accuracy gets <em>worse</em>." },
    { wrong: "Forgetting to unscale gradients after loss scaling.", why: "You must divide gradients by $S$ (in float32) before the optimizer step, or the effective learning rate is off by $S$. Also skip the step if scaled gradients overflowed to inf/NaN (dynamic loss scaling)." }
  ],

  quiz: [
    { q: "The central difference has truncation error of order…", options: ["$O(h^2)$", "$O(h)$", "$O(h^3)$", "$O(1)$"], answer: 0,
      explain: "The even Taylor terms cancel, leaving an $h^2$ leading error — more accurate than the one-sided $O(h)$." },
    { q: "A gradient check passes when the relative error is…", options: ["very small (e.g. <1e-5)", "exactly 0", "less than 1", "greater than 1"], answer: 0,
      explain: "Floating point prevents an exact match; a tiny relative error indicates a correct gradient." },
    { q: "Making $h$ extremely small in a finite difference…", options: ["hurts accuracy via cancellation", "always helps", "has no effect", "removes rounding error"], answer: 0,
      explain: "Subtracting near-equal values loses digits; there's an optimal $h\\sim\\varepsilon^{1/3}$." },
    { q: "Loss scaling exists to prevent float16 gradients from…", options: ["underflowing to zero", "overflowing to inf", "becoming negative", "being biased"], answer: 0,
      explain: "Tiny gradients fall below float16's smallest normal; scaling the loss lifts them into range." },
    { q: "In mixed-precision training, what is typically kept in float32?", options: ["master weights and accumulation", "the input images", "nothing", "only the loss value"], answer: 0,
      explain: "A float32 master copy and float32 accumulation preserve small updates that 16-bit would drop." }
  ],

  practice: [
    { level: "easy", prompt: "Write the central-difference estimate of $f'(x)$ and its error order.", solution: "$\\frac{f(x+h)-f(x-h)}{2h}$, with error $O(h^2)$." },
    { level: "med", prompt: "Your analytic gradient is 3.0000 and the finite-difference gives 3.0002. Should you trust it?", solution: "Relative error $\\approx0.0002/3\\approx7\\times10^{-5}$ — small but a bit high for float64. Likely fine (or $h$ slightly off); if you want confidence, tune $h$ toward $10^{-5}$ and re-check. A gross mismatch (e.g. 3.0 vs 1.5) would indicate a real backprop bug." },
    { level: "med", prompt: "Why keep a float32 master copy of weights during float16 training?", solution: "Weight updates $\\eta\\cdot\\text{grad}$ are often tiny relative to the weight; in float16 the update can be below the weight's epsilon-gap and round away, so the weight never changes ('stalled' training). A float32 master accumulates these small updates faithfully; the float16 copy is refreshed from it for the fast forward/backward pass." },
    { level: "hard", prompt: "AI task: describe the full mixed-precision training loop with dynamic loss scaling.", solution: "(1) Keep float32 master weights; cast a float16 copy for the forward pass. (2) Multiply the loss by a scale $S$. (3) Backprop in float16 — gradients are effectively $\\times S$, lifted above underflow. (4) Check gradients for inf/NaN: if any, the step overflowed — <em>skip</em> the update and halve $S$. (5) Otherwise unscale (divide gradients by $S$) and update the <em>float32</em> master weights with the optimizer (whose state is float32). (6) Periodically increase $S$ if many steps succeed. This keeps gradients in float16's range (avoiding underflow) while catching overflow, giving float32-like results at 16-bit speed and memory — the standard recipe behind fast large-model training. bfloat16 simplifies this by having float32's range, often removing the need for loss scaling entirely (Lesson 13.1)." }
  ],

  deepDive: String.raw`<p><strong>Numerical hygiene is the unglamorous half of making models work.</strong></p>
  <p>Everything in this track is about the gap between mathematics on paper (infinite precision, exact derivatives,
  well-conditioned problems) and computation in the real world (finite bits, rounding, overflow). Gradient checking is
  the bridge you build when you can't fully trust your own math — it's how research code earns confidence before a
  week-long training run, and it caught countless bugs in the era before autodiff frameworks made hand-derived gradients
  rare. The technique itself is a microcosm of the track: it pits truncation error (Taylor, Track 6) against rounding
  error (floating point, Lesson 13.1) and finds the sweet spot, exactly the kind of trade that defines numerical
  computing.</p>
  <p>Mixed precision, meanwhile, is where all five lessons converge in a single training loop. It exploits low precision
  for speed (13.1), guards range with loss scaling and float32 accumulation to dodge overflow/underflow and cancellation
  (13.2), relies on stable factorized/fused kernels underneath (13.3), and threads the whole thing through reverse-mode
  autodiff (13.4). The payoff is enormous — 2–3× faster training, half the memory — and it's invisible when it works and
  catastrophic (silent NaNs, stalled updates) when the numerics are wrong. That's the quiet truth of applied ML: the
  elegant math gets the headlines, but whether a model actually trains often comes down to whether someone respected the
  finite precision it runs on. With this, the mathematical foundation is complete — from arithmetic to autodiff — and
  what remains is to <em>build</em>: the capstones put every track to work in code that runs.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["13.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 65 minutes.",

  intro: String.raw`<p>This exam spans all of Track 13: floating point, numerical stability / log-sum-exp, matrix
  factorizations, automatic differentiation, and gradient checking / mixed precision. <strong>Give yourself 65
  minutes</strong>, produce each answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "Does $0.1+0.2==0.3$ in IEEE-754 double precision? Why?",
      solution: "No — none of $0.1,0.2,0.3$ is exactly representable in binary, so the rounded sum differs from the rounded $0.3$ in the last bits. Compare with a tolerance, not $==$." },
    { level: "easy", prompt: "float16 vs bfloat16: which has more range, which more precision?",
      solution: "bfloat16 has more range (8 exponent bits, like float32, max ~$3.4\\times10^{38}$); float16 has more precision (10 mantissa bits vs 7) but tiny range (max 65504)." },
    { level: "med", prompt: "Rewrite $\\log(e^{800}+e^{801})$ so it doesn't overflow, and give its approximate value.",
      solution: "With $M=801$: $801+\\log(e^{-1}+e^{0})=801+\\log(1+e^{-1})=801+\\log(1.3679)\\approx801.313$. No exponentiation of a raw large number." },
    { level: "med", prompt: "Which factorization for (a) a general square solve, (b) a symmetric positive-definite solve, (c) least-squares?",
      solution: "(a) LU, (b) Cholesky (half the cost of LU), (c) QR (stable; avoids squaring the condition number)." },
    { level: "med", prompt: "State the central-difference formula and its error order, and why $h$ shouldn't be too small.",
      solution: "$\\frac{f(x+h)-f(x-h)}{2h}$, error $O(h^2)$. Too-small $h$ makes $f(x+h)\\approx f(x-h)$, so their difference loses significant digits (catastrophic cancellation); the optimal $h\\sim\\varepsilon^{1/3}$." },
    { level: "med", prompt: "For $L=a\\cdot b + a$ at $a=2,b=3$, give $L$, $\\partial L/\\partial a$, $\\partial L/\\partial b$.",
      solution: "$L=2\\cdot3+2=8$; $\\partial L/\\partial a=b+1=4$; $\\partial L/\\partial b=a=2$ (autodiff's local derivatives summed over paths)." },
    { level: "hard", prompt: "Why is backprop reverse-mode AD and not forward-mode, in cost terms?",
      solution: "The gradient is a product of Jacobians; reverse mode evaluates it from the scalar loss outward (vector–Jacobian products), so its cost scales with the number of <em>outputs</em> (one). Forward mode scales with the number of <em>inputs</em> (millions of parameters). For a scalar loss over many parameters, reverse mode computes all gradients in ~one forward-pass cost; forward mode would need one pass per parameter." },
    { level: "hard", prompt: "Derive the stable softmax and explain why subtracting the max is free.",
      solution: "Softmax is shift-invariant: $\\frac{e^{z_i-c}}{\\sum_j e^{z_j-c}}=\\frac{e^{z_i}}{\\sum_j e^{z_j}}$ for any $c$ (the $e^{-c}$ cancels). Choosing $c=\\max_i z_i$ makes every exponent $\\le0$, so all $e^{(\\cdot)}\\in(0,1]$ — no overflow — and the largest term is 1, so the denominator is $\\ge1$ — no underflow. The result is identical to the naive softmax, hence 'free'." },
    { level: "hard", prompt: "Explain loss scaling in float16 training and one failure it prevents.",
      solution: "Multiply the loss by a large $S$ before backprop; by linearity every gradient is scaled by $S$, lifting tiny values (e.g. $10^{-8}$) above float16's underflow threshold so they're representable. Before the optimizer step, divide gradients by $S$ (in float32) to recover true magnitudes. It prevents small gradients from underflowing to zero (which would stall learning); dynamic loss scaling also detects overflow (inf/NaN) and backs off $S$." },
    { level: "hard", prompt: "AI task: you implement a custom layer and training loss is NaN after a few steps. Give a numerical debugging checklist.",
      solution: "(1) Gradient-check the layer's backward against a central difference on a tiny input — a mismatch means a backprop bug. (2) Look for exp/log without stabilization (use log-sum-exp / pass logits to the loss, not probabilities). (3) Check for division by near-zero or $\\log 0$ (clamp or add $\\varepsilon$). (4) In float16, suspect overflow/underflow — try bfloat16 or add/adjust loss scaling; watch for activations exceeding 65504. (5) Reduce the learning rate / add gradient clipping to rule out divergence vs a numerics bug. (6) Verify variance/normalization is computed stably (center before squaring). The NaN's first appearance (which op, which step) usually localizes the culprit." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Numerical methods are solid — you can debug precision, stability, and gradients. The mathematical course is complete; on to the capstones.</li>
    <li><strong>7–8:</strong> Strong. Revisit the log-sum-exp derivation or forward-vs-reverse AD if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive the stable softmax and the central-difference error order; redo Lessons 13.2 and 13.4.</li>
    <li><strong>Below 5:</strong> Rework the track — floating point, stability, and autodiff are what make the beautiful math actually run.</li>
  </ul>`
};
