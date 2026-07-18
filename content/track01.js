/* ============================================================
   TRACK 1 — Arithmetic & Algebra Refresher
   Full lesson standard. All strings use String.raw so that
   LaTeX backslashes and Python escapes survive verbatim.
   ============================================================ */
(window.LESSON_CONTENT ||= {})["1.1"] = {
  subtitle: "The undo-button for exponentials — and why every model trains in log-space.",

  aiMoment: String.raw`<p>When a language model scores the sentence <em>"the cat sat"</em>, it multiplies the
  probability of each token: maybe $0.04 \times 0.0007 \times 0.02 \times \dots$. Over a few hundred tokens the
  product is so small it <strong>underflows to exactly 0.0</strong> in 32-bit floating point — and you can't take a
  gradient of zero.</p>
  <p>The fix is everywhere in ML: work with <strong>log-probabilities</strong>. The logarithm turns that
  fragile product into a sturdy <em>sum</em>. This is why losses are reported in "nats" or "bits", why PyTorch
  ships <code>log_softmax</code> and <code>nll_loss</code> instead of raw probabilities, and why the next two
  tracks lean on logs constantly.</p>`,

  plainEnglish: String.raw`<p>An <strong>exponent</strong> is repeated multiplication: $2^3$ means "multiply three 2's".
  A <strong>logarithm</strong> is the reverse question — "two to <em>what</em> power gives me this number?"</p>
  <p>The one fact that makes logs matter for AI: <strong>a log turns multiplication into addition.</strong>
  Multiplying a thousand tiny probabilities is numerically dangerous; adding a thousand logs is calm and stable.</p>`,

  intuition: String.raw`<p>Picture exponential growth and a logarithm as <strong>mirror images</strong> across the
  diagonal line $y=x$. The exponential rockets upward; the log is its lazy reflection, rising ever more slowly.
  Whatever the exponential <em>does</em>, the log <em>undoes</em>.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Exponential and logarithm reflected across y=x">
    <line x1="40" y1="280" x2="40" y2="30" stroke="#94a3b8" stroke-width="1.5"/>
    <line x1="40" y1="280" x2="300" y2="280" stroke="#94a3b8" stroke-width="1.5"/>
    <line x1="40" y1="280" x2="280" y2="40" stroke="#cbd5e1" stroke-width="1.4" stroke-dasharray="5 5"/>
    <polyline points="40,240 60,223 80,200 100,167 120,120 140,54 143,40" fill="none" stroke="#d97706" stroke-width="2.6"/>
    <polyline points="80,280 100,257 120,240 160,217 200,200 280,177" fill="none" stroke="#4f46e5" stroke-width="2.6"/>
    <text x="150" y="58" font-size="12" fill="#d97706" font-family="sans-serif">y = 2ˣ  (grows fast)</text>
    <text x="214" y="168" font-size="12" fill="#4f46e5" font-family="sans-serif">y = log₂x</text>
    <text x="232" y="58" font-size="11" fill="#94a3b8" font-family="sans-serif">y = x</text>
  </svg>
  <figcaption>The log is the exponential viewed in the mirror of the line y = x.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Exponent.</strong> For a base $b\gt 0$ and integer $n$, $b^n = \underbrace{b\cdot b \cdots b}_{n}$.
  The laws extend to all real exponents:</p>
  $$b^{m}\,b^{n}=b^{m+n},\qquad \frac{b^{m}}{b^{n}}=b^{m-n},\qquad (b^{m})^{n}=b^{mn},\qquad b^{0}=1,\qquad b^{-n}=\frac{1}{b^{n}}.$$
  <p><strong>Logarithm.</strong> The log base $b$ is defined as the inverse of $b^{(\cdot)}$:</p>
  $$\log_b(x)=y \quad\Longleftrightarrow\quad b^{y}=x \qquad (x\gt 0,\; b\gt 0,\; b\neq 1).$$
  <p>Here $b$ is the <em>base</em>, $x$ the <em>argument</em> (must be positive), $y$ the resulting exponent.
  Two bases dominate ML: $e\approx 2.718$ giving the <strong>natural log</strong> $\ln x=\log_e x$, and base 2 giving
  <strong>bits</strong>. In one plain sentence: <em>a logarithm is the exponent you were looking for.</em></p>`,

  derivation: String.raw`<p><strong>Claim.</strong> The product rule of logs: $\log_b(xy)=\log_b x+\log_b y$.
  This single identity is the reason log-space turns products into sums.</p>
  <p><strong>Step 1 — name the two logs.</strong> Let $m=\log_b x$ and $n=\log_b y$. We just gave names to the
  answers we're after.</p>
  <p><strong>Step 2 — rewrite as exponentials.</strong> By the definition of log, $m=\log_b x$ means $b^{m}=x$, and
  likewise $b^{n}=y$. We've moved from log-form to exponential-form.</p>
  <p><strong>Step 3 — multiply.</strong> Multiply the two equations: $xy=b^{m}\cdot b^{n}$.</p>
  <p><strong>Step 4 — use the exponent law.</strong> Same base, so add exponents: $b^{m}\cdot b^{n}=b^{m+n}$.
  Therefore $xy=b^{m+n}$.</p>
  <p><strong>Step 5 — take the log of both sides.</strong> $\log_b(xy)=\log_b\!\big(b^{m+n}\big)=m+n$, because $\log_b$
  and $b^{(\cdot)}$ cancel.</p>
  <p><strong>Step 6 — substitute back.</strong> Replace $m,n$: $\log_b(xy)=\log_b x+\log_b y.\;\blacksquare$</p>
  <p>The same three lines give the quotient and power rules:
  $\log_b\frac{x}{y}=\log_b x-\log_b y$ and $\log_b(x^{k})=k\log_b x$.</p>
  <hr class="soft">
  <p><strong>Change of base.</strong> Computers only store $\ln$, yet papers quote bits. Convert with one formula.
  Let $y=\log_b x$, so $b^{y}=x$. Take $\ln$ of both sides: $y\ln b=\ln x$, hence</p>
  $$\log_b x=\frac{\ln x}{\ln b}.$$
  <p>Plain English: any log is the natural log divided by the natural log of the base.</p>`,

  code: [
    { label: "Logs turn products into sums", src: String.raw`
import numpy as np

probs = np.array([0.04, 0.0007, 0.02, 0.5, 0.11, 0.9, 0.003])

# Direct product — fine here, but underflows for long sequences.
prod = np.prod(probs)

# Same quantity via logs: log(prod) = sum(log p)
log_sum = np.sum(np.log(probs))

print("product          :", prod)
print("exp(sum of logs) :", np.exp(log_sum))   # should match
print("sum of log-probs :", log_sum, "nats")
` },
    { label: "Why log-space survives where products die", src: String.raw`
import numpy as np

# 5000 small probabilities, like scoring a long document.
rng = np.random.default_rng(0)
p = rng.uniform(0.001, 0.05, size=5000)

direct = np.prod(p)               # collapses to 0.0 (underflow)
logp   = np.sum(np.log(p))        # stays a sensible finite number

print("naive product :", direct)            # 0.0  -> gradient is dead
print("log-likelihood:", logp, "nats")      # finite, usable
print("recovered?    :", np.isfinite(logp))
` },
    { label: "Plot: exponential vs logarithm", src: String.raw`
import numpy as np, matplotlib.pyplot as plt

x = np.linspace(0.1, 4, 200)
plt.figure(figsize=(5,4))
plt.plot(x, 2**x, label="2^x  (exponential)", lw=2)
plt.plot(x, np.log2(x), label="log2(x)", lw=2)
plt.plot(x, x, "--", color="gray", label="y = x")
plt.axhline(0, color="k", lw=.5); plt.legend(); plt.title("Mirror images across y = x")
plt.tight_layout(); plt.show()
` }
  ],

  keyPoints: [
    "A logarithm answers “the base to what power gives this number?” — it is the inverse of exponentiation.",
    "$\\log(xy)=\\log x+\\log y$: logs convert multiplication into addition, the backbone of log-likelihoods.",
    "Work in log-space to avoid underflow when multiplying many small probabilities.",
    "Change of base: $\\log_b x=\\ln x/\\ln b$; bits and nats differ only by the constant $\\ln 2$.",
    "The argument of a log must be positive — there is no real $\\log$ of $0$ or a negative number."
  ],

  commonMistakes: [
    { wrong: "Writing $\\log(x+y)=\\log x+\\log y$.", why: "The product rule is about multiplication. $\\log(x+y)$ has no simple expansion; only $\\log(xy)$ splits into a sum." },
    { wrong: "Treating cross-entropy in 'bits' and 'nats' as the same number.", why: "They measure the same information but in different units: nats use $\\ln$, bits use $\\log_2$. They differ by the factor $\\ln 2\\approx0.693$." },
    { wrong: "Taking $\\log$ of a probability that could be exactly 0.", why: "$\\log 0=-\\infty$ and breaks training. Frameworks clamp with an $\\varepsilon$ or use $\\texttt{log\\_softmax}$, which never forms the raw 0." }
  ],

  quiz: [
    { q: "Simplify $\\log_2(8\\cdot 4)$.", options: ["5", "32", "12", "6"], answer: 0,
      explain: "$\\log_2(8\\cdot4)=\\log_2 8+\\log_2 4=3+2=5$. Choice 32 is the product $8\\cdot4$ itself; 6 forgets to add." },
    { q: "A model assigns token probabilities $0.5, 0.25, 0.1$. What is the log-likelihood in nats?", options: ["$\\ln(0.0125)\\approx-4.38$", "$0.85$", "$-0.85$", "$\\ln 3$"], answer: 0,
      explain: "Sum of logs $=\\ln0.5+\\ln0.25+\\ln0.1=\\ln(0.0125)\\approx-4.38$. Choice 2 adds the probabilities; choice 3 negates that incorrect sum." },
    { q: "Which equals $\\log_b(x^3/y)$?", options: ["$3\\log_b x-\\log_b y$", "$3\\log_b x/\\log_b y$", "$\\log_b(3x)-\\log_b y$", "$(\\log_b x)^3-\\log_b y$"], answer: 0,
      explain: "Power rule then quotient rule: $\\log_b x^3-\\log_b y=3\\log_b x-\\log_b y$. The others misapply the rules." },
    { q: "Convert $\\log_2 10$ to natural logs.", options: ["$\\ln 10/\\ln 2$", "$\\ln 2/\\ln 10$", "$\\ln 10\\cdot\\ln 2$", "$\\ln(10/2)$"], answer: 0,
      explain: "Change of base divides by $\\ln$ of the base: $\\log_2 10=\\ln10/\\ln2\\approx3.32$." },
    { q: "Why does $\\texttt{log\\_softmax}$ exist instead of $\\texttt{log(softmax(x))}$?", options: ["It avoids overflow/underflow by never forming the raw probabilities", "It is mathematically different", "It is faster only on GPUs", "It changes the gradient direction"], answer: 0,
      explain: "Same math, safer numerics: it folds the log into the softmax (log-sum-exp) so it never computes $e^{x}$ that overflows or a $0$ it must log." }
  ],

  practice: [
    { level: "easy", prompt: "Evaluate $\\log_{10}(1000)$ and $\\ln(e^4)$.", solution: "$\\log_{10}1000=\\log_{10}10^3=3$. $\\ln(e^4)=4$ because $\\ln$ and $e^{(\\cdot)}$ are inverses." },
    { level: "easy", prompt: "Rewrite $\\log(abc)$ as a sum.", solution: "Apply the product rule twice: $\\log(abc)=\\log a+\\log b+\\log c$." },
    { level: "med", prompt: "A sequence has probabilities $0.2,0.2,0.05$. Give the likelihood and the log-likelihood; explain which you'd accumulate in code.", solution: "Likelihood $=0.2\\cdot0.2\\cdot0.05=0.002$. Log-likelihood $=\\ln0.2+\\ln0.2+\\ln0.05=-1.609-1.609-2.996=-6.215$ nats. Accumulate the <em>log</em>-likelihood: summing logs avoids the underflow that kills the raw product on long sequences." },
    { level: "med", prompt: "Show $\\log_b(1/x)=-\\log_b x$ from the rules.", solution: "$\\log_b(1/x)=\\log_b(x^{-1})=-1\\cdot\\log_b x=-\\log_b x$ by the power rule (or quotient rule with numerator 1, since $\\log_b 1=0$)." },
    { level: "hard", prompt: "Cross-entropy loss for one example with true class $c$ is $-\\ln q_c$, where $q_c$ is the predicted probability. If a model is 'confidently wrong' with $q_c=10^{-6}$, what is the loss in nats and in bits? Why does this dominate the average?", solution: "Nats: $-\\ln(10^{-6})=6\\ln10\\approx13.8$. Bits: divide by $\\ln2$, $\\approx19.9$ bits. Because the loss grows like $-\\ln q_c$, a single near-zero probability contributes a huge term, so one confident mistake can outweigh many correct, mildly-confident predictions — which is exactly why label smoothing and gradient clipping help." },
    { level: "hard", prompt: "An AI task: you store token log-probs and need the sentence probability ratio $P(A)/P(B)$ without ever leaving log-space until the end. Give the recipe.", solution: "Keep $\\ell_A=\\sum\\ln p_i^{A}$ and $\\ell_B=\\sum\\ln p_j^{B}$. The ratio is $P(A)/P(B)=e^{\\ell_A-\\ell_B}$. Subtract first (a stable difference of moderate numbers), exponentiate once at the very end. Never exponentiate $\\ell_A$ and $\\ell_B$ separately — each may underflow." }
  ],

  deepDive: String.raw`<p><strong>The log-sum-exp trick — your first taste of numerical stability.</strong></p>
  <p>Softmax needs $\sum_i e^{z_i}$. If some logit $z_i$ is, say, $1000$, then $e^{1000}=\infty$ in float32 and the
  whole thing becomes <code>NaN</code>. The cure uses one log/exp identity. Let $M=\max_i z_i$. Then</p>
  $$\log\sum_i e^{z_i}=\log\sum_i e^{(z_i-M)+M}=\log\!\Big(e^{M}\sum_i e^{z_i-M}\Big)=M+\log\sum_i e^{z_i-M}.$$
  <p><strong>Why it works, step by step.</strong> (1) Add and subtract $M$ inside the exponent — legal, changes
  nothing. (2) Factor out $e^{M}$ using $e^{a+b}=e^a e^b$. (3) Pull $e^{M}$ out of the log as $+M$ using
  $\log(e^M\cdot S)=M+\log S$. Now every remaining exponent $z_i-M\le 0$, so each $e^{z_i-M}\in(0,1]$: no overflow,
  and the largest term is exactly $1$, so no underflow-to-zero either.</p>
  <p>This is the identity hiding inside every framework's <code>logsumexp</code>, <code>log_softmax</code>, and
  cross-entropy. You will re-derive it for real in Track 13. The lesson: logs don't just tidy the algebra —
  they make the computation <em>possible</em>.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["1.2"] = {
  subtitle: "Σ is a for-loop that adds; Π is a for-loop that multiplies. Read them and you read ML papers.",

  aiMoment: String.raw`<p>The single most common formula in deep learning is the cross-entropy loss:
  $\;\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{K} y_{ic}\,\log q_{ic}.$ Two stacked $\Sigma$'s — one over
  the $N$ examples in a batch, one over the $K$ classes. The softmax that produced $q$ has a $\Sigma$ in its
  denominator. If $\Sigma$ and $\Pi$ read like noise, every paper reads like noise; if they read like loops,
  the math reads like code.</p>`,

  plainEnglish: String.raw`<p>$\Sigma$ (capital sigma) means "add up a list of things." $\Pi$ (capital pi) means
  "multiply a list of things." The little symbols underneath and on top just say where the list starts and where
  it stops.</p>`,

  intuition: String.raw`<p>$\sum_{i=1}^{n} a_i$ is exactly this loop: <code>total = 0; for i in 1..n: total += a_i</code>.
  Each item pours into a running total.</p>
  <figure class="figure">
  <svg viewBox="0 0 420 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Summation as a fan-in accumulator">
    <g font-family="sans-serif" font-size="13">
    <rect x="14"  y="24" width="58" height="34" rx="6" fill="#eef0ff" stroke="#4f46e5"/>
    <rect x="92"  y="24" width="58" height="34" rx="6" fill="#eef0ff" stroke="#4f46e5"/>
    <rect x="170" y="24" width="58" height="34" rx="6" fill="#eef0ff" stroke="#4f46e5"/>
    <rect x="248" y="24" width="58" height="34" rx="6" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="43"  y="46" text-anchor="middle" fill="#4f46e5">a₁</text>
    <text x="121" y="46" text-anchor="middle" fill="#4f46e5">a₂</text>
    <text x="199" y="46" text-anchor="middle" fill="#4f46e5">a₃</text>
    <text x="277" y="46" text-anchor="middle" fill="#4f46e5">aₙ</text>
    <text x="328" y="46" fill="#94a3b8">· · ·</text>
    <line x1="43"  y1="58" x2="196" y2="120" stroke="#94a3b8"/>
    <line x1="121" y1="58" x2="198" y2="120" stroke="#94a3b8"/>
    <line x1="199" y1="58" x2="202" y2="120" stroke="#94a3b8"/>
    <line x1="277" y1="58" x2="206" y2="120" stroke="#94a3b8"/>
    <circle cx="200" cy="134" r="22" fill="#4f46e5"/>
    <text x="200" y="140" text-anchor="middle" fill="#fff" font-size="18">Σ</text>
    <line x1="222" y1="134" x2="288" y2="134" stroke="#94a3b8" marker-end="url(#ar)"/>
    <text x="298" y="139" fill="#20242c">= S</text>
    </g>
    <defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>Every term feeds one running total. Π is the same picture with × instead of +.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Sum.</strong> $\displaystyle\sum_{i=1}^{n} a_i = a_1+a_2+\dots+a_n.$ Here $i$ is the
  <em>index</em> (a throwaway counter), $1$ and $n$ are the lower and upper <em>bounds</em>, and $a_i$ is the term.
  The name $i$ is irrelevant — $\sum_i a_i=\sum_j a_j$.</p>
  <p><strong>Product.</strong> $\displaystyle\prod_{i=1}^{n} a_i = a_1\cdot a_2\cdots a_n.$ By convention the
  <em>empty sum is $0$</em> and the <em>empty product is $1$</em> (the identity element of each operation).</p>`,

  derivation: String.raw`<p><strong>Two facts you will use constantly.</strong></p>
  <p><strong>(A) Sums are linear.</strong> For a constant $c$,
  $\displaystyle\sum_{i=1}^{n}(c\,a_i+b_i)=c\sum_{i=1}^{n}a_i+\sum_{i=1}^{n}b_i.$</p>
  <p><strong>Step 1.</strong> Write the left side out: $(ca_1+b_1)+(ca_2+b_2)+\dots+(ca_n+b_n)$.
  <strong>Step 2.</strong> Regroup the $a$-terms and $b$-terms (addition is commutative):
  $(ca_1+\dots+ca_n)+(b_1+\dots+b_n)$. <strong>Step 3.</strong> Factor $c$ out of the first group:
  $c(a_1+\dots+a_n)+(b_1+\dots+b_n)=c\sum a_i+\sum b_i.\;\blacksquare$ Plain English: you can pull constants out of
  a sum and split a sum of sums.</p>
  <hr class="soft">
  <p><strong>(B) Log of a product is a sum of logs.</strong> This links Lesson 1.1 to every likelihood:</p>
  $$\log\prod_{i=1}^{n} p_i=\sum_{i=1}^{n}\log p_i.$$
  <p><strong>Step 1.</strong> $\log(p_1 p_2\cdots p_n)$. <strong>Step 2.</strong> Apply $\log(xy)=\log x+\log y$ to peel
  off $p_1$: $=\log p_1+\log(p_2\cdots p_n)$. <strong>Step 3.</strong> Repeat $n-1$ times (induction) until every
  factor is its own log term: $=\log p_1+\log p_2+\dots+\log p_n=\sum_i\log p_i.\;\blacksquare$ This is precisely how a
  product-likelihood becomes the summed log-likelihood that we actually optimize.</p>`,

  code: [
    { label: "Sum, product, and axis-wise reduction", src: String.raw`
import numpy as np

a = np.array([2.0, 3.0, 4.0])
print("sum  Σa =", np.sum(a))      # 9.0
print("prod Πa =", np.prod(a))     # 24.0

# A batch loss is a SUM over rows. axis controls which Σ you take.
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]])
row_sums = logits.sum(axis=1)      # Σ over classes, per example
print("Σ over classes (axis=1):", row_sums)
print("Σ over everything       :", logits.sum())
` },
    { label: "Cross-entropy as a double sum", src: String.raw`
import numpy as np

# one-hot labels y and predicted probabilities q for a batch of 2, K=3
y = np.array([[1,0,0],
              [0,1,0]], dtype=float)
q = np.array([[0.7,0.2,0.1],
              [0.1,0.8,0.1]])

# L = -(1/N) ΣΣ y*log q   — the two Σ's are the two array axes
N = y.shape[0]
L = -(1/N) * np.sum(y * np.log(q))
print("cross-entropy loss:", round(float(L), 4), "nats")
` }
  ],

  keyPoints: [
    "$\\Sigma$ adds a list, $\\Pi$ multiplies it; the bounds are just loop start/stop values.",
    "The index name is meaningless: $\\sum_i a_i=\\sum_j a_j$.",
    "Sums are linear — pull out constants, split sums of sums.",
    "$\\log\\prod p_i=\\sum\\log p_i$ turns a likelihood (product) into a log-likelihood (sum).",
    "Empty sum $=0$, empty product $=1$."
  ],

  commonMistakes: [
    { wrong: "Reusing the same index for an inner and outer sum, e.g. $\\sum_i\\sum_i$.", why: "The inner loop needs its own counter. Stomping on $i$ silently computes the wrong thing — use $\\sum_i\\sum_j$." },
    { wrong: "Summing over the wrong axis in NumPy.", why: "<code>X.sum(axis=0)</code> collapses rows (sum down columns); <code>axis=1</code> collapses columns. Picking the wrong one swaps a per-example sum for a per-feature sum." },
    { wrong: "Reading $\\sum_{i=1}^{n}$ as excluding $n$.", why: "The upper bound is inclusive: $\\sum_{i=1}^{n}$ runs $i=1,2,\\dots,n$ — that's $n$ terms, not $n-1$." }
  ],

  quiz: [
    { q: "Compute $\\sum_{i=1}^{4} (2i-1)$.", options: ["16", "10", "20", "9"], answer: 0,
      explain: "Terms are $1,3,5,7$, summing to $16$ (the sum of the first 4 odd numbers, $=4^2$). Choice 10 sums $1+2+3+4$." },
    { q: "What is $\\prod_{k=1}^{3} 2^{k}$?", options: ["$2^{6}=64$", "$2^{3}=8$", "$6$", "$2^{9}$"], answer: 0,
      explain: "$2^1\\cdot2^2\\cdot2^3=2^{1+2+3}=2^6=64$. The exponents add (Lesson 1.1)." },
    { q: "$\\log\\prod_{i=1}^{n}p_i$ equals…", options: ["$\\sum_i\\log p_i$", "$\\prod_i\\log p_i$", "$n\\log p_1$", "$\\log\\sum_i p_i$"], answer: 0,
      explain: "Product rule of logs applied $n$ times. The others confuse product with sum or assume all $p_i$ equal." },
    { q: "For $X$ of shape $(N,K)$, which gives one loss value per example?", options: ["$X.\\text{sum(axis=1)}$", "$X.\\text{sum(axis=0)}$", "$X.\\text{sum()}$", "$X.\\text{sum(axis=1).T}$"], answer: 0,
      explain: "axis=1 collapses the $K$ classes, leaving one number per row (example). axis=0 would give one number per class." },
    { q: "$\\sum_{i=1}^{3}\\sum_{j=1}^{2} ij$ = ?", options: ["18", "12", "9", "36"], answer: 0,
      explain: "$=\\big(\\sum_i i\\big)\\big(\\sum_j j\\big)=(1+2+3)(1+2)=6\\cdot3=18$, since the term factorizes." }
  ],

  practice: [
    { level: "easy", prompt: "Write 'multiply $x_1$ through $x_5$' in $\\Pi$ notation, then 'add their squares' in $\\Sigma$ notation.", solution: "$\\prod_{i=1}^{5}x_i$ and $\\sum_{i=1}^{5}x_i^2$." },
    { level: "med", prompt: "Show $\\sum_{i=1}^{n} c = nc$ for a constant $c$.", solution: "The term doesn't depend on $i$, so you're adding $c$ to itself $n$ times: $c+c+\\dots+c=nc$. (A constant 'sum' is just multiplication.)" },
    { level: "med", prompt: "Express the average $\\bar a$ of $a_1,\\dots,a_n$ and explain why it is linear in the data.", solution: "$\\bar a=\\frac1n\\sum_{i=1}^n a_i$. By linearity, scaling all data by $c$ scales $\\bar a$ by $c$, and the mean of a sum of two datasets is the sum of their means — both follow from pulling constants out of $\\Sigma$." },
    { level: "hard", prompt: "Softmax: write $\\sigma(z)_k=\\dfrac{e^{z_k}}{\\sum_{j} e^{z_j}}$ and show $\\sum_k\\sigma(z)_k=1$.", solution: "Sum over $k$: $\\sum_k\\sigma(z)_k=\\dfrac{\\sum_k e^{z_k}}{\\sum_j e^{z_j}}$. Numerator and denominator are the same sum (different index letters), so the ratio is $1$. That's why softmax outputs a probability distribution." },
    { level: "hard", prompt: "AI task: the log-likelihood of a sequence under an autoregressive model is $\\log\\prod_{t=1}^{T}p(x_t\\mid x_{&lt;t})$. Rewrite it as the quantity a trainer accumulates, and say what dividing by $T$ gives.", solution: "$\\sum_{t=1}^{T}\\log p(x_t\\mid x_{&lt;t})$ — the summed token log-probabilities. Dividing by $T$ gives the average log-likelihood per token; its negative is the per-token cross-entropy, and $e$ raised to it is the perplexity (Track 12)." }
  ],

  deepDive: String.raw`<p><strong>Double sums, and why you should swap them for vectorization.</strong></p>
  <p>A double sum $\sum_{i=1}^{m}\sum_{j=1}^{n} A_{ij}$ adds every entry of an $m\times n$ grid. Two facts matter.
  <strong>First, order doesn't change the result</strong> (add row-by-row or column-by-column):
  $\sum_i\sum_j A_{ij}=\sum_j\sum_i A_{ij}$. <strong>Second, if the term factorizes</strong> as $A_{ij}=u_i v_j$, the
  double sum collapses: $\sum_i\sum_j u_i v_j=\big(\sum_i u_i\big)\big(\sum_j v_j\big)$.</p>
  <p>That factorization is why we replace nested Python loops with a single matrix operation. The expression
  $\sum_j A_{ij}x_j$ — a sum for each $i$ — is exactly the matrix–vector product $Ax$ from Track 2. Recognizing a
  double sum as a matrix multiply is the difference between code that runs in milliseconds on a GPU and code that
  crawls.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["1.3"] = {
  subtitle: "A function is a machine; a neural network is many machines wired in series.",

  aiMoment: String.raw`<p>A neural network is one big composed function: $\hat y=f_L(\dots f_2(f_1(x)))$. Each layer
  $f_\ell$ is a machine that takes the previous machine's output. An activation like the sigmoid is a function with a
  specific <strong>domain</strong> (all real numbers) and <strong>range</strong> (only values strictly between 0 and
  1). Knowing ranges tells you why a sigmoid output can be read as a probability, and why feeding it somewhere that
  needs an input $\gt 1$ would break.</p>`,

  plainEnglish: String.raw`<p>A function is a rule that turns each input into exactly one output — a machine with a
  slot and a chute. <strong>Composition</strong> means feeding one machine's output into the next.
  An <strong>inverse</strong> is the same machine run backward, recovering the input from the output.</p>`,

  intuition: String.raw`<p>Think pipeline. Push $x$ into machine $g$, take its output $g(x)$, push that into machine
  $f$. The whole pipeline is the single function $f\circ g$.</p>
  <figure class="figure">
  <svg viewBox="0 0 440 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Function composition pipeline">
    <g font-family="sans-serif" font-size="14">
    <text x="20" y="80" fill="#20242c">x</text>
    <line x1="34" y1="75" x2="78" y2="75" stroke="#94a3b8" marker-end="url(#a2)"/>
    <rect x="80" y="52" width="78" height="46" rx="8" fill="#f7f3ff" stroke="#7c3aed"/>
    <text x="119" y="80" text-anchor="middle" fill="#7c3aed">g</text>
    <line x1="158" y1="75" x2="206" y2="75" stroke="#94a3b8" marker-end="url(#a2)"/>
    <text x="182" y="66" text-anchor="middle" font-size="12" fill="#64748b">g(x)</text>
    <rect x="208" y="52" width="78" height="46" rx="8" fill="#fff7ed" stroke="#d97706"/>
    <text x="247" y="80" text-anchor="middle" fill="#d97706">f</text>
    <line x1="286" y1="75" x2="338" y2="75" stroke="#94a3b8" marker-end="url(#a2)"/>
    <text x="392" y="80" text-anchor="middle" fill="#20242c">f(g(x))</text>
    </g>
    <defs><marker id="a2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>Composition wires machines in series. Order matters: f∘g is usually not g∘f.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A <strong>function</strong> $f:X\to Y$ assigns to each $x$ in the <em>domain</em> $X$ exactly
  one $y=f(x)$ in the <em>codomain</em> $Y$. The <strong>range</strong> is the set of outputs actually hit.
  <strong>Composition:</strong> $(f\circ g)(x)=f(g(x))$, defined when $g$'s outputs lie in $f$'s domain.
  An <strong>inverse</strong> $f^{-1}$ satisfies $f^{-1}(f(x))=x$; it exists exactly when $f$ is one-to-one (no two
  inputs share an output).</p>
  <p><strong>Inequalities</strong> obey the usual algebra with one twist: multiplying or dividing by a negative number
  <em>flips</em> the direction, e.g. $-2x\lt 6\Rightarrow x\gt -3$.</p>`,

  derivation: String.raw`<p><strong>Deriving the quadratic formula by completing the square.</strong> Solve
  $ax^2+bx+c=0$ with $a\neq0$.</p>
  <p><strong>Step 1 — divide by $a$:</strong> $x^2+\frac{b}{a}x+\frac{c}{a}=0$.</p>
  <p><strong>Step 2 — move the constant:</strong> $x^2+\frac{b}{a}x=-\frac{c}{a}$.</p>
  <p><strong>Step 3 — complete the square.</strong> Add $\left(\frac{b}{2a}\right)^2$ to both sides so the left becomes
  a perfect square: $x^2+\frac{b}{a}x+\frac{b^2}{4a^2}=\frac{b^2}{4a^2}-\frac{c}{a}.$</p>
  <p><strong>Step 4 — factor the left, combine the right:</strong>
  $\left(x+\frac{b}{2a}\right)^2=\frac{b^2-4ac}{4a^2}.$</p>
  <p><strong>Step 5 — take square roots</strong> (the $\pm$ appears because both roots square to the same thing):
  $x+\frac{b}{2a}=\pm\frac{\sqrt{b^2-4ac}}{2a}.$</p>
  <p><strong>Step 6 — isolate $x$:</strong> $\displaystyle x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}.\;\blacksquare$
  Plain English: completing the square rebuilds the equation as "(something)$^2$ = number", which a square root always
  solves.</p>`,

  code: [
    { label: "Sigmoid and its inverse (the logit)", src: String.raw`
import numpy as np

def sigmoid(x):  return 1.0 / (1.0 + np.exp(-x))   # domain: all R, range: (0,1)
def logit(p):    return np.log(p / (1.0 - p))      # inverse; domain: (0,1)

x = np.array([-2.0, 0.0, 1.5])
p = sigmoid(x)
print("sigmoid(x)        =", p)            # all strictly between 0 and 1
print("logit(sigmoid(x)) =", logit(p))     # recovers x  ->  f^{-1}(f(x)) = x
` },
    { label: "Composition is not commutative", src: String.raw`
f = lambda x: x + 1
g = lambda x: x * 2
print("f(g(3)) =", f(g(3)))   # 2*3=6 then +1 -> 7
print("g(f(3)) =", g(f(3)))   # 3+1=4 then *2 -> 8   (different!)
` }
  ],

  keyPoints: [
    "A function maps each input to exactly one output; domain = allowed inputs, range = produced outputs.",
    "Composition $f\\circ g$ feeds $g$'s output into $f$ and is generally not equal to $g\\circ f$.",
    "An inverse exists only for one-to-one (monotonic) functions — that's why activations are usually monotonic.",
    "Sigmoid maps $\\mathbb{R}\\to(0,1)$; its inverse is the logit, mapping $(0,1)\\to\\mathbb{R}$.",
    "Multiplying an inequality by a negative number flips its direction."
  ],

  commonMistakes: [
    { wrong: "Calling $f^{-1}$ the reciprocal $1/f$.", why: "The inverse <em>function</em> undoes $f$; the reciprocal is just one over its value. $\\text{logit}=\\sigma^{-1}$, not $1/\\sigma$." },
    { wrong: "Assuming $f\\circ g=g\\circ f$.", why: "Order matters: a linear map then a nonlinearity is not the reverse — that's the whole point of layer ordering." },
    { wrong: "Feeding values outside the domain (e.g. $\\log$ of a negative, logit of $p\\ge 1$).", why: "These are undefined and produce NaN/inf. Clamp inputs to the valid domain first." }
  ],

  quiz: [
    { q: "If $f(x)=x^2$ and $g(x)=x-3$, what is $(f\\circ g)(5)$?", options: ["4", "22", "2", "25"], answer: 0,
      explain: "$g(5)=2$, then $f(2)=4$. Choice 22 computes $f(5)-3$; choice 25 forgets $g$." },
    { q: "What is the range of $\\sigma(x)=1/(1+e^{-x})$?", options: ["$(0,1)$", "$[0,1]$", "$(-1,1)$", "$(0,\\infty)$"], answer: 0,
      explain: "Sigmoid never actually reaches 0 or 1, so the range is the open interval $(0,1)$ — which is why a true 0 or 1 probability can't be produced." },
    { q: "Solve $-3x+1\\ge 7$.", options: ["$x\\le -2$", "$x\\ge -2$", "$x\\le 2$", "$x\\ge 2$"], answer: 0,
      explain: "$-3x\\ge 6\\Rightarrow x\\le -2$: dividing by $-3$ flips the inequality. Forgetting the flip gives $x\\ge -2$." },
    { q: "Which has no inverse function on all of $\\mathbb{R}$?", options: ["$f(x)=x^2$", "$f(x)=2x+1$", "$f(x)=x^3$", "$f(x)=e^x$"], answer: 0,
      explain: "$x^2$ maps $2$ and $-2$ to the same output, so it isn't one-to-one on $\\mathbb{R}$ and has no inverse there. The others are monotonic." },
    { q: "Roots of $x^2-5x+6=0$?", options: ["2 and 3", "−2 and −3", "1 and 6", "5 and 6"], answer: 0,
      explain: "Discriminant $25-24=1$; $x=(5\\pm1)/2=3,2$. Choice '1 and 6' multiplies to 6 but sums to 7, not 5." }
  ],

  practice: [
    { level: "easy", prompt: "Give the domain of $h(x)=\\ln(x-2)$.", solution: "The log needs a positive argument: $x-2\\gt0$, so the domain is $x\\gt2$." },
    { level: "easy", prompt: "With $f(x)=3x$ and $g(x)=x+4$, compute $f\\circ g$ and $g\\circ f$ as formulas.", solution: "$(f\\circ g)(x)=3(x+4)=3x+12$; $(g\\circ f)(x)=3x+4$. Different — composition isn't commutative." },
    { level: "med", prompt: "Find the inverse of $f(x)=2x-3$ and verify $f^{-1}(f(x))=x$.", solution: "Set $y=2x-3$, solve for $x$: $x=(y+3)/2$, so $f^{-1}(y)=(y+3)/2$. Check: $f^{-1}(2x-3)=((2x-3)+3)/2=x.\\;\\checkmark$" },
    { level: "med", prompt: "Show the logit is the inverse of the sigmoid.", solution: "Let $p=\\sigma(x)=\\frac{1}{1+e^{-x}}$. Then $1-p=\\frac{e^{-x}}{1+e^{-x}}$, so $\\frac{p}{1-p}=e^{x}$. Taking $\\ln$: $\\ln\\frac{p}{1-p}=x$. That function of $p$ is the logit, hence $\\text{logit}=\\sigma^{-1}$." },
    { level: "hard", prompt: "AI task: a 2-layer net computes $\\hat y=\\sigma(w_2\\,\\mathrm{relu}(w_1 x))$. Identify the composition chain and each piece's range.", solution: "Chain: $x\\to f_1(x)=w_1x$ (range $\\mathbb{R}$) $\\to \\mathrm{relu}$ (range $[0,\\infty)$) $\\to f_2(\\cdot)=w_2(\\cdot)$ (range $\\mathbb{R}$) $\\to\\sigma$ (range $(0,1)$). The final $(0,1)$ range lets $\\hat y$ be read as a probability; the ReLU's flat negative side is why 'dead' units stop learning." }
  ],

  deepDive: String.raw`<p><strong>Invertibility, monotonicity, and why activations are shaped the way they are.</strong></p>
  <p>A function is invertible exactly when it is <em>strictly monotonic</em> — always increasing or always decreasing
  — because then each output came from a unique input. Sigmoid and tanh are strictly increasing, so they're
  invertible (their inverses, logit and $\operatorname{artanh}$, appear in flow models that run the network backward).
  ReLU is <em>not</em> strictly monotonic: it's flat for $x\lt0$, so every negative input maps to $0$ and the
  information there is lost forever — the "dying ReLU" problem.</p>
  <p>This is also why <strong>normalizing flows</strong> (and the change-of-variables trick in Track 10) insist on
  invertible layers: to compute how a probability density transforms, you must run the function backward and measure
  how it stretches space. Monotonicity isn't a stylistic choice — it's the price of being able to undo a layer.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["1.4"] = {
  subtitle: "How cost grows when the input grows — the reason long-context attention hurts.",

  aiMoment: String.raw`<p>Self-attention compares every token with every other token. For a sequence of length $n$
  that's $n\times n$ comparisons: the cost scales like $n^2$. Double the context and you roughly quadruple the
  compute and memory. This single fact — attention is $O(n^2)$ — is why 100k-token context windows are a research
  problem, why FlashAttention fights to cut the memory constant, and why "linear attention" papers exist. Big-O is the
  language that statement is written in.</p>`,

  plainEnglish: String.raw`<p>Big-O describes how an algorithm's cost grows as the input gets bigger, ignoring
  constant factors and small print. It answers one question: <strong>if I double the input, what happens to the
  time (or memory)?</strong> Stays the same? Doubles? Quadruples?</p>`,

  intuition: String.raw`<p>Plot cost against input size and you get a family of curves. Constant time is flat.
  Linear is a straight ramp. Quadratic curves sharply upward — it's the one that bites first as $n$ grows.</p>
  <figure class="figure">
  <svg viewBox="0 0 380 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Growth rate curves">
    <line x1="44" y1="230" x2="44" y2="24" stroke="#94a3b8" stroke-width="1.4"/>
    <line x1="44" y1="230" x2="356" y2="230" stroke="#94a3b8" stroke-width="1.4"/>
    <text x="12" y="30" font-size="11" fill="#64748b" font-family="sans-serif">cost</text>
    <text x="322" y="248" font-size="11" fill="#64748b" font-family="sans-serif">n →</text>
    <polyline points="44,222 356,216" fill="none" stroke="#16a34a" stroke-width="2.2"/>
    <polyline points="44,224 90,196 150,180 240,170 356,164" fill="none" stroke="#0d9488" stroke-width="2.2"/>
    <polyline points="44,230 356,96" fill="none" stroke="#2563eb" stroke-width="2.2"/>
    <polyline points="44,230 140,178 240,120 320,74 356,60" fill="none" stroke="#7c3aed" stroke-width="2.2"/>
    <polyline points="44,230 130,206 190,168 234,126 268,80 298,42 314,26" fill="none" stroke="#dc2626" stroke-width="2.6"/>
    <g font-size="11" font-family="sans-serif">
    <text x="316" y="212" fill="#16a34a">O(1)</text>
    <text x="296" y="158" fill="#0d9488">O(log n)</text>
    <text x="326" y="90"  fill="#2563eb">O(n)</text>
    <text x="304" y="70"  fill="#7c3aed">O(n log n)</text>
    <text x="276" y="32"  fill="#dc2626">O(n²)</text>
    </g>
  </svg>
  <figcaption>Same axes, wildly different fates. The quadratic curve is what makes long sequences expensive.</figcaption>
  </figure>`,

  formalism: String.raw`<p>We say $f(n)=O\big(g(n)\big)$ if there exist constants $C\gt0$ and $n_0$ such that</p>
  $$f(n)\le C\,g(n)\quad\text{for all } n\ge n_0.$$
  <p>Plain English: past some point, $f$ never grows faster than a constant multiple of $g$. Two cousins:
  $\Omega(g)$ is a lower bound ("at least this fast"), and $\Theta(g)$ means both — a tight rate.
  In Big-O we <strong>drop constants and lower-order terms</strong>: $3n^2+5n+7$ and $\tfrac12 n^2$ are both
  $O(n^2)$, because for large $n$ the $n^2$ term dominates everything else.</p>`,

  derivation: String.raw`<p><strong>Claim.</strong> $3n^2+5n+7=O(n^2)$. We exhibit explicit constants.</p>
  <p><strong>Step 1 — bound each term by an $n^2$ term</strong> for $n\ge1$: since $n\le n^2$ and $1\le n^2$,
  we have $5n\le 5n^2$ and $7\le 7n^2$.</p>
  <p><strong>Step 2 — add the bounds:</strong> $3n^2+5n+7\le 3n^2+5n^2+7n^2=15n^2$.</p>
  <p><strong>Step 3 — read off the constants:</strong> with $C=15$ and $n_0=1$, the definition
  $f(n)\le Cn^2$ holds for all $n\ge n_0$. Therefore $f(n)=O(n^2).\;\blacksquare$ Plain English: the messy lower terms
  can each be absorbed into a slightly bigger constant on $n^2$.</p>
  <hr class="soft">
  <p><strong>Why attention is $O(n^2 d)$.</strong> Queries $Q$ and keys $K$ are $n\times d$. The scores
  $QK^\top$ form an $n\times n$ matrix, and each of its $n^2$ entries is a dot product costing $d$ multiply-adds.
  Total: $n^2\cdot d$ operations, i.e. $O(n^2 d)$. The $n^2$ is unavoidable if every token must see every token — that
  is the cost the whole "efficient attention" literature is trying to dodge.</p>`,

  code: [
    { label: "Watch attention scale like n²", src: String.raw`
import numpy as np, time

d = 64
for n in [128, 256, 512, 1024]:
    Q = np.random.randn(n, d); K = np.random.randn(n, d)
    t = time.time()
    scores = Q @ K.T        # the n x n score matrix
    dt = time.time() - t
    print(f"n={n:5d}  scores={scores.shape}  time={dt*1e3:7.2f} ms")
# Each time n doubles, work ~quadruples: the signature of O(n^2).
` },
    { label: "Plot measured time vs an n² reference", src: String.raw`
import numpy as np, time, matplotlib.pyplot as plt

ns = [64,128,256,512,768,1024]; d=64; times=[]
for n in ns:
    Q=np.random.randn(n,d); K=np.random.randn(n,d)
    t=time.time(); _=Q@K.T; times.append(time.time()-t)

ns=np.array(ns); times=np.array(times)
ref = times[0]*(ns/ns[0])**2          # pure n^2 curve, anchored at first point
plt.figure(figsize=(5,4))
plt.plot(ns, times, "o-", label="measured")
plt.plot(ns, ref, "--", label="proportional to n^2")
plt.xlabel("sequence length n"); plt.ylabel("seconds"); plt.legend()
plt.title("Attention scores scale quadratically"); plt.tight_layout(); plt.show()
` }
  ],

  keyPoints: [
    "Big-O is about growth rate, not exact operation counts — constants and lower terms are dropped.",
    "$f=O(g)$ means $f\\le C\\,g$ beyond some $n_0$; $\\Theta$ is a tight (upper and lower) bound.",
    "Self-attention is $O(n^2 d)$ in sequence length $n$ — the core scaling pain of long context.",
    "Lower-order terms vanish asymptotically: $3n^2+5n+7=O(n^2)$.",
    "Time complexity and memory complexity are separate budgets — track both."
  ],

  commonMistakes: [
    { wrong: "Believing a smaller Big-O is always faster in practice.", why: "Constants matter for real inputs. An $O(n\\log n)$ method with a huge constant can lose to an $O(n^2)$ one for small $n$ — FlashAttention is still $O(n^2)$ but wins by shrinking the memory constant." },
    { wrong: "Confusing compute cost with memory cost.", why: "Vanilla attention is $O(n^2)$ in <em>both</em>, but they're optimized differently; FlashAttention keeps $O(n^2)$ compute while cutting memory to $O(n)$." },
    { wrong: "Reading $O$ as an exact count or a lower bound.", why: "$O$ is only an upper bound. Use $\\Theta$ when you mean exact growth and $\\Omega$ for a lower bound." }
  ],

  quiz: [
    { q: "Simplify the Big-O of $7n^3+1000n^2+n$.", options: ["$O(n^3)$", "$O(n^2)$", "$O(1000n^2)$", "$O(n^3+n^2)$"], answer: 0,
      explain: "Highest-order term wins and constants drop: $O(n^3)$. The $1000n^2$ looks scary but is asymptotically smaller than $n^3$." },
    { q: "Attention over length $n$ uses how much score memory?", options: ["$O(n^2)$", "$O(n)$", "$O(n d)$", "$O(d^2)$"], answer: 0,
      explain: "The score matrix is $n\\times n$, so $O(n^2)$ memory — the reason long sequences blow up VRAM." },
    { q: "If an $O(n^2)$ job takes 1s at $n=1000$, estimate it at $n=4000$.", options: ["≈16s", "≈4s", "≈8s", "≈64s"], answer: 0,
      explain: "Cost scales as $(4000/1000)^2=16$, so about 16s. Choice 4s wrongly assumes linear scaling." },
    { q: "Which grows fastest as $n\\to\\infty$?", options: ["$2^n$", "$n^{10}$", "$n\\log n$", "$n^2$"], answer: 0,
      explain: "Exponential beats every polynomial eventually: $2^n$ dominates $n^{10}$ for large $n$." },
    { q: "Doubling context from 2k to 4k tokens multiplies attention compute by about…", options: ["4×", "2×", "1×", "8×"], answer: 0,
      explain: "$O(n^2)$ means $2^2=4\\times$. This is exactly why doubling context is so costly." }
  ],

  practice: [
    { level: "easy", prompt: "Give the Big-O of $50n+9$.", solution: "$O(n)$ — linear; the constant 50 and the $+9$ are dropped." },
    { level: "easy", prompt: "Rank by growth for large $n$: $n^2$, $\\log n$, $n$, $2^n$.", solution: "$\\log n \\lt n \\lt n^2 \\lt 2^n$." },
    { level: "med", prompt: "Prove $n^2+n=O(n^2)$ with explicit $C,n_0$.", solution: "For $n\\ge1$, $n\\le n^2$, so $n^2+n\\le n^2+n^2=2n^2$. Take $C=2,n_0=1$. Hence $O(n^2)$." },
    { level: "med", prompt: "A model multiplies $A_{n\\times n}B_{n\\times n}$. State the compute Big-O and why.", solution: "$O(n^3)$: the result has $n^2$ entries and each is a length-$n$ dot product ($n$ multiply-adds), giving $n^2\\cdot n=n^3$. (Sub-cubic algorithms exist but aren't used in practice.)" },
    { level: "hard", prompt: "AI task: vanilla attention is $O(n^2 d)$ compute; a 'linear attention' claims $O(n d^2)$. For $n=8192$, $d=128$, which is cheaper, and by roughly what factor?", solution: "Vanilla: $n^2 d=8192^2\\cdot128\\approx8.6\\times10^{9}$. Linear: $n d^2=8192\\cdot128^2\\approx1.34\\times10^{8}$. Ratio $\\approx 64\\times$ cheaper — and the gap widens with $n$ because one term scales with $n^2$, the other with $n$. That difference is the entire selling point of linear attention." }
  ],

  deepDive: String.raw`<p><strong>Why generation is $O(n)$ per token but $O(n^2)$ overall — and what the KV-cache buys.</strong></p>
  <p>When a transformer generates text one token at a time, step $t$ must attend to all $t$ previous tokens. Naively
  re-running attention from scratch each step costs $O(t^2)$, and summing $t=1\dots n$ gives $O(n^3)$ — catastrophic.
  The <strong>KV-cache</strong> fixes this: keys and values for past tokens are stored, so step $t$ only computes the
  new query against $t$ cached keys, costing $O(t\,d)$. Summing $\sum_{t=1}^{n} t\,d=O(n^2 d)$ total — the same
  quadratic as processing the whole sequence at once, but now spread one $O(n)$ step at a time.</p>
  <p>The catch is memory: the cache stores $O(n)$ key/value vectors per layer, so a long conversation's KV-cache can
  dwarf the model weights. That tradeoff is the reason multi-query and grouped-query attention exist — they shrink the
  cache's constant. Reading these as Big-O statements turns hand-wavy "it's expensive" into an exact budget.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["1.E"] = {
  exam: true,
  subtitle: "Ten mixed problems on logs, Σ/Π, functions, and Big-O. Target: 50 minutes.",

  intro: String.raw`<p>This exam covers all of Track 1. <strong>Give yourself 50 minutes</strong> and resist opening a
  solution until you've committed to an answer — recognizing a result is not the same as producing it. About half the
  problems are pure calculation. Tally your score against the rubric at the end.</p>`,

  problems: [
    { level: "easy", prompt: "Evaluate $\\log_2 32$, $\\ln(e^{3})$, and $\\log_{10}(0.001)$.",
      solution: "$\\log_2 32=\\log_2 2^5=5$. $\\ln(e^3)=3$. $\\log_{10}(10^{-3})=-3$." },
    { level: "easy", prompt: "Expand $\\log\\dfrac{x^2 y}{z}$ into a sum/difference of logs.",
      solution: "$\\log x^2+\\log y-\\log z=2\\log x+\\log y-\\log z$ (product, quotient, then power rules)." },
    { level: "med", prompt: "A model assigns token probabilities $0.5,\\,0.1,\\,0.2$. Give the log-likelihood in nats and explain why you'd accumulate this rather than the raw product.",
      solution: "$\\ln0.5+\\ln0.1+\\ln0.2=-0.693-2.303-1.609=-4.605$ nats. Summing logs avoids the underflow that destroys the raw product $0.5\\cdot0.1\\cdot0.2=0.01$ once sequences get long." },
    { level: "med", prompt: "Express $\\log_2 1000$ using natural logs and give a 2-decimal value.",
      solution: "$\\log_2 1000=\\dfrac{\\ln 1000}{\\ln 2}=\\dfrac{6.9078}{0.6931}\\approx 9.97$." },
    { level: "med", prompt: "Compute $\\sum_{i=1}^{5} 2i$ and, for a constant $c$, $\\sum_{i=1}^{n} c$.",
      solution: "$\\sum_{i=1}^{5}2i=2(1+2+3+4+5)=2\\cdot15=30$. $\\sum_{i=1}^{n}c=nc$ (adding $c$ to itself $n$ times)." },
    { level: "med", prompt: "Show $\\log\\prod_{i=1}^{n} p_i=\\sum_{i=1}^{n}\\log p_i$ and name where this is used in ML.",
      solution: "Apply $\\log(xy)=\\log x+\\log y$ repeatedly to peel each factor into its own log term. It turns a likelihood (a product over examples/tokens) into the summed log-likelihood that training actually maximizes (equivalently, the negative is cross-entropy)." },
    { level: "med", prompt: "Give the domain of $h(x)=\\ln(x-3)+\\dfrac{1}{x-5}$.",
      solution: "Need $x-3\\gt0$ (so $x\\gt3$) and $x\\neq5$. Domain: $(3,5)\\cup(5,\\infty)$." },
    { level: "hard", prompt: "With $f(x)=2x+1$ and $g(x)=x^2$, compute $(f\\circ g)(x)$ and $(g\\circ f)(x)$, and find $f^{-1}$.",
      solution: "$(f\\circ g)(x)=2x^2+1$; $(g\\circ f)(x)=(2x+1)^2=4x^2+4x+1$ — different, since composition isn't commutative. Inverse: $y=2x+1\\Rightarrow x=(y-1)/2$, so $f^{-1}(x)=(x-1)/2$." },
    { level: "hard", prompt: "Simplify the Big-O of $4n^2+3n\\log n+100$, and state the compute Big-O of self-attention over length $n$, dimension $d$.",
      solution: "$O(n^2)$ — the $n^2$ term dominates $n\\log n$ and the constant. Self-attention is $O(n^2 d)$: the $n\\times n$ score matrix has $n^2$ entries, each a length-$d$ dot product." },
    { level: "hard", prompt: "AI task: an $O(n^2)$ attention layer takes 2s at context $n=1024$. Estimate the time at $n=8192$, and explain what the KV-cache changes during generation.",
      solution: "Scale factor $(8192/1024)^2=8^2=64$, so about $2\\times64=128$ s. During autoregressive generation, the KV-cache stores past keys/values so each new token costs $O(n d)$ instead of recomputing $O(n^2)$ from scratch — total work stays $O(n^2 d)$ but is spread one $O(n)$ step per token, at the cost of $O(n)$ memory per layer." }
  ],

  rubric: String.raw`<p>Count the problems you solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Fluent. The algebra won't slow you down anywhere later — proceed to Track 2.</li>
    <li><strong>7–8:</strong> Strong. Revisit any missed log or Big-O item; these recur constantly.</li>
    <li><strong>5–6:</strong> Shaky on one area. Redo the relevant lesson's Practice section before moving on.</li>
    <li><strong>Below 5:</strong> Re-read the track top to bottom — every later track leans on logs, sums, and Big-O.</li>
  </ul>`
};
