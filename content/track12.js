/* ============================================================
   TRACK 12 — Information Theory
   Opener: 12.2 Cross-Entropy & KL Divergence (the loss itself).
   ============================================================ */
(window.LESSON_CONTENT ||= {})["12.2"] = {
  subtitle: "The classification loss, decoded: average surprise, split into what's unavoidable and what's your model's fault.",

  aiMoment: String.raw`<p>Every classifier and language model you have ever trained minimizes
  <strong>cross-entropy</strong>. For a one-hot label it collapses to the famous $-\log q_{\text{correct}}$: punish the
  model by how surprised it was at the right answer. <strong>KL divergence</strong> is its close cousin — the term a
  VAE adds to keep its latent code near a Gaussian, and the quantity RLHF constrains so a fine-tuned model doesn't
  drift too far from the base. The identity $\text{cross-entropy}=\text{entropy}+\text{KL}$ ties them together and
  explains what the loss is really measuring.</p>`,

  plainEnglish: String.raw`<p><strong>Surprise</strong> at an event of probability $q$ is $-\log q$: rare events are
  surprising, certain ones aren't. <strong>Cross-entropy</strong> is your average surprise when reality follows $p$
  but you predicted with $q$. <strong>KL divergence</strong> is the <em>extra</em> surprise you pay for using the wrong
  distribution $q$ instead of the true $p$ — it's zero only when you're exactly right.</p>`,

  intuition: String.raw`<p>True distribution $p$ is the data; your prediction $q$ is the guess. Where $q$ puts too
  little mass on something that actually happens a lot, surprise spikes. KL adds up that mismatch, weighted by how
  often each outcome really occurs.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="True distribution p versus predicted q as bars">
    <line x1="40" y1="160" x2="300" y2="160" stroke="#cbd5e1"/>
    <g font-family="sans-serif" font-size="11">
    <rect x="60"  y="60"  width="20" height="100" fill="#4f46e5"/>
    <rect x="84"  y="96"  width="20" height="64"  fill="#a5b4fc"/>
    <rect x="140" y="120" width="20" height="40"  fill="#4f46e5"/>
    <rect x="164" y="100" width="20" height="60"  fill="#a5b4fc"/>
    <rect x="220" y="100" width="20" height="60"  fill="#4f46e5"/>
    <rect x="244" y="80"  width="20" height="80"  fill="#a5b4fc"/>
    <text x="62"  y="174" fill="#64748b">cat</text>
    <text x="142" y="174" fill="#64748b">dog</text>
    <text x="218" y="174" fill="#64748b">bird</text>
    <rect x="210" y="22" width="12" height="12" fill="#4f46e5"/><text x="226" y="32" fill="#20242c">p (true)</text>
    <rect x="210" y="38" width="12" height="12" fill="#a5b4fc"/><text x="226" y="48" fill="#20242c">q (predicted)</text>
    </g>
  </svg>
  <figcaption>KL measures the gap between the bars, weighted by the true probabilities p.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For distributions $p,q$ over the same outcomes:</p>
  $$\underbrace{H(p)=-\sum_i p_i\log p_i}_{\text{entropy}},\qquad
  \underbrace{H(p,q)=-\sum_i p_i\log q_i}_{\text{cross-entropy}},\qquad
  \underbrace{D_{\mathrm{KL}}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}}_{\text{KL divergence}}.$$
  <p>$H(p)$ is the average surprise of the <em>true</em> distribution — the irreducible floor. $H(p,q)$ is the average
  surprise when you score true-$p$ data with model $q$. $D_{\mathrm{KL}}$ is the gap between them. Base-2 logs give
  <strong>bits</strong>; natural logs give <strong>nats</strong>.</p>`,

  derivation: String.raw`<p><strong>Part 1 — the identity $H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)$.</strong></p>
  <p><strong>Step 1.</strong> Start from KL and split the log of a ratio:
  $D_{\mathrm{KL}}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}=\sum_i p_i\log p_i-\sum_i p_i\log q_i.$</p>
  <p><strong>Step 2.</strong> Recognize the two sums: the first is $-H(p)$, the second is $-H(p,q)$. So
  $D_{\mathrm{KL}}(p\Vert q)=-H(p)+H(p,q).$</p>
  <p><strong>Step 3.</strong> Rearrange: $H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q).\;\blacksquare$ Plain English: your loss
  (cross-entropy) is the unavoidable noise in the data $H(p)$ plus a penalty for being wrong $D_{\mathrm{KL}}$. Since
  $H(p)$ doesn't depend on your model, <em>minimizing cross-entropy is exactly minimizing KL to the truth.</em></p>
  <hr class="soft">
  <p><strong>Part 2 — why $D_{\mathrm{KL}}(p\Vert q)\ge 0$ (Gibbs' inequality).</strong> We use one fact:
  $\log x\le x-1$ for all $x\gt0$, with equality only at $x=1$.</p>
  <p><strong>Step 1.</strong> Look at the negative KL and flip the ratio:
  $-D_{\mathrm{KL}}(p\Vert q)=\sum_i p_i\log\frac{q_i}{p_i}.$</p>
  <p><strong>Step 2.</strong> Apply $\log x\le x-1$ with $x=q_i/p_i$:
  $\log\frac{q_i}{p_i}\le\frac{q_i}{p_i}-1.$ Multiply by $p_i\ge0$ (preserves the inequality):
  $p_i\log\frac{q_i}{p_i}\le q_i-p_i.$</p>
  <p><strong>Step 3.</strong> Sum over $i$:
  $-D_{\mathrm{KL}}(p\Vert q)\le\sum_i(q_i-p_i)=\sum_i q_i-\sum_i p_i=1-1=0.$</p>
  <p><strong>Step 4.</strong> Therefore $-D_{\mathrm{KL}}\le 0$, i.e. $D_{\mathrm{KL}}(p\Vert q)\ge0$. Equality needs
  $q_i/p_i=1$ for every $i$ — that is, $q=p$. $\blacksquare$ Plain English: you can never be <em>less</em> surprised by
  using the wrong distribution, and you're only un-penalized when your prediction is exactly the truth.</p>`,

  code: [
    { label: "Entropy, cross-entropy, KL — and the identity", src: String.raw`
import numpy as np

p = np.array([0.7, 0.2, 0.1])     # true distribution
q = np.array([0.5, 0.3, 0.2])     # model's prediction

H   = -np.sum(p * np.log(p))              # entropy of p
CE  = -np.sum(p * np.log(q))              # cross-entropy H(p,q)
KL  =  np.sum(p * np.log(p / q))          # KL(p || q)

print("H(p)        =", round(float(H), 4))
print("H(p,q)      =", round(float(CE), 4))
print("KL(p||q)    =", round(float(KL), 4))
print("H + KL      =", round(float(H + KL), 4), " == H(p,q)?", np.isclose(CE, H + KL))
print("KL >= 0 ?   ", KL >= 0)
` },
    { label: "For one-hot labels, cross-entropy = −log q(correct)", src: String.raw`
import numpy as np

# true class is index 2 (one-hot p), model probabilities q
p = np.array([0.0, 0.0, 1.0])
q = np.array([0.1, 0.3, 0.6])

CE = -np.sum(p * np.log(q))
print("cross-entropy      =", round(float(CE), 4))
print("-log q[correct]    =", round(float(-np.log(q[2])), 4))   # identical
# entropy of a one-hot p is 0, so here CE == KL: all loss is 'your fault'
` }
  ],

  keyPoints: [
    "Surprise $=-\\log q$; cross-entropy is average surprise of true data scored by model $q$.",
    "$H(p,q)=H(p)+D_{\\mathrm{KL}}(p\\Vert q)$: loss = irreducible entropy + model penalty.",
    "Minimizing cross-entropy = minimizing KL to the truth (since $H(p)$ is fixed).",
    "$D_{\\mathrm{KL}}\\ge0$, and $=0$ only when $q=p$ (Gibbs' inequality).",
    "KL is asymmetric: $D_{\\mathrm{KL}}(p\\Vert q)\\ne D_{\\mathrm{KL}}(q\\Vert p)$ — it is not a distance."
  ],

  commonMistakes: [
    { wrong: "Calling KL a distance / metric.", why: "It's asymmetric and violates the triangle inequality. $D_{\\mathrm{KL}}(p\\Vert q)$ and $D_{\\mathrm{KL}}(q\\Vert p)$ can differ wildly — which is why 'forward' vs 'reverse' KL matters (Deep Dive)." },
    { wrong: "Expecting cross-entropy loss to reach 0 on noisy labels.", why: "The floor is $H(p)\\gt0$ whenever the true labels are uncertain. You can only drive the KL term to 0, not the irreducible entropy." },
    { wrong: "Mixing log bases between loss and reported 'bits'.", why: "Training usually uses nats ($\\ln$); perplexity and 'bits per token' use $\\log_2$. Compare like with like or divide by $\\ln 2$." }
  ],

  quiz: [
    { q: "A one-hot label with predicted prob $0.25$ on the true class. Cross-entropy (nats)?", options: ["$-\\ln 0.25\\approx1.386$", "$0.25$", "$\\ln 0.25$", "$0$"], answer: 0,
      explain: "One-hot CE $=-\\log q_{\\text{correct}}=-\\ln0.25\\approx1.386$. Choice $\\ln0.25$ drops the minus sign (loss must be positive)." },
    { q: "If $q=p$, then $D_{\\mathrm{KL}}(p\\Vert q)$ is…", options: ["0", "1", "$H(p)$", "undefined"], answer: 0,
      explain: "KL is zero exactly when the distributions match — there's no extra surprise. Gibbs' inequality makes 0 the minimum." },
    { q: "Cross-entropy equals entropy plus…", options: ["$D_{\\mathrm{KL}}(p\\Vert q)$", "$D_{\\mathrm{KL}}(q\\Vert p)$", "mutual information", "variance"], answer: 0,
      explain: "$H(p,q)=H(p)+D_{\\mathrm{KL}}(p\\Vert q)$. The order matters: it's the forward KL from truth $p$ to model $q$." },
    { q: "Which makes KL undefined / infinite?", options: ["$q_i=0$ where $p_i>0$", "$p_i=0$ where $q_i>0$", "$p=q$", "all $q_i$ equal"], answer: 0,
      explain: "If the model assigns 0 probability to something that truly occurs ($p_i>0$), the $\\log(p_i/q_i)$ term blows up — the model is infinitely surprised. The reverse ($p_i=0$) contributes 0." },
    { q: "True $p=[0.5,0.5]$, model $q=[0.5,0.5]$. The cross-entropy in bits is…", options: ["1 bit", "0 bits", "0.5 bit", "2 bits"], answer: 0,
      explain: "$q=p$ so KL$=0$ and CE$=H(p)=-2\\cdot0.5\\log_2 0.5=1$ bit — the irreducible cost of a fair coin." }
  ],

  practice: [
    { level: "easy", prompt: "Compute the entropy of a fair 4-sided die in bits.", solution: "$H=-\\sum_{i=1}^4\\tfrac14\\log_2\\tfrac14=\\log_2 4=2$ bits. Uniform over $n$ outcomes has entropy $\\log_2 n$." },
    { level: "easy", prompt: "One-hot label, model puts $0.9$ on the correct class. Give the loss in nats.", solution: "$-\\ln 0.9\\approx0.105$ nats — small, because the model was nearly right." },
    { level: "med", prompt: "With $p=[0.5,0.5]$, $q=[0.9,0.1]$, compute $D_{\\mathrm{KL}}(p\\Vert q)$ in nats.", solution: "$0.5\\ln\\frac{0.5}{0.9}+0.5\\ln\\frac{0.5}{0.1}=0.5(-0.5878)+0.5(1.6094)=0.5108$ nats. Positive, as Gibbs guarantees." },
    { level: "med", prompt: "Show that for a one-hot $p$, $H(p)=0$ and therefore cross-entropy equals KL.", solution: "A one-hot $p$ has a single $p_k=1$; $H(p)=-1\\cdot\\ln1=0$ (other terms are $0\\ln0=0$). Then $H(p,q)=H(p)+D_{\\mathrm{KL}}=0+D_{\\mathrm{KL}}$, so the entire loss is the KL penalty — every nat is 'the model's fault'." },
    { level: "hard", prompt: "AI task: a language model's average cross-entropy on held-out text is $2.0$ nats/token. State the perplexity, and explain why lower cross-entropy means lower perplexity.", solution: "Perplexity $=e^{\\text{CE}}=e^{2.0}\\approx7.39$ — the model is as confused as if choosing uniformly among ~7.4 tokens each step. Since perplexity is a monotonic ($\\exp$) function of cross-entropy, minimizing CE during training directly minimizes perplexity; they carry the same information in different units." }
  ],

  deepDive: String.raw`<p><strong>Forward vs reverse KL: mode-covering vs mode-seeking.</strong></p>
  <p>Because KL is asymmetric, <em>which</em> argument is the truth changes the behavior of the fit.
  <strong>Forward KL</strong>, $D_{\mathrm{KL}}(p\Vert q)$ — the one inside cross-entropy — weights the error by the
  true $p$. Wherever $p$ has mass, $q$ had better not be near zero (or the $\log(p/q)$ term explodes). So forward KL is
  <strong>mode-covering</strong>: $q$ stretches to cover every bump of $p$, even if it smears mass into the valleys.
  This is what maximum-likelihood training does, and why an under-powered model produces blurry, averaged outputs.</p>
  <p><strong>Reverse KL</strong>, $D_{\mathrm{KL}}(q\Vert p)$, weights by $q$ instead. Now $q$ is punished for putting
  mass where $p$ is small, but pays nothing for ignoring a mode of $p$. So reverse KL is <strong>mode-seeking</strong>:
  $q$ locks onto one peak and ignores the rest. Variational inference and some RL objectives optimize reverse KL,
  which is exactly why they can collapse onto a single mode. Same formula, opposite personalities — and the choice is a
  real modeling decision, not a technicality. (Entropy itself is built in Lesson 12.1; Jensen's inequality, the engine
  behind these bounds, is Lesson 12.4.)</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["12.1"] = {
  subtitle: "How surprised you are on average — the measure of uncertainty behind perplexity.",

  aiMoment: String.raw`<p>A language model's <strong>perplexity</strong> is two-to-the-<strong>entropy</strong> of its
  next-token predictions — the effective number of choices it's wavering between. Entropy quantifies uncertainty, sets
  the floor of the cross-entropy loss (Lesson 12.2), and underlies max-entropy modeling and information gain in decision
  trees. It's the root concept of the whole track.</p>`,

  plainEnglish: String.raw`<p><strong>Surprise</strong> at an outcome of probability $p$ is $-\log p$: rare things are
  surprising, certain things aren't. <strong>Entropy</strong> is your <em>average</em> surprise over a distribution — how
  uncertain you are before seeing the outcome. A fair coin is maximally uncertain; a rigged one is predictable.</p>`,

  intuition: String.raw`<p>Entropy is highest when every outcome is equally likely (you can't guess) and drops to zero
  when one outcome is certain (no surprise). For a coin, it peaks at a 50/50 split and falls off toward either extreme.</p>
  <figure class="figure">
  <svg viewBox="0 0 280 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Entropy of a coin peaks at p=0.5">
    <line x1="40" y1="140" x2="260" y2="140" stroke="#94a3b8"/>
    <line x1="40" y1="140" x2="40" y2="20" stroke="#94a3b8"/>
    <path d="M40,140 C90,42 110,30 150,30 C190,30 210,42 260,140" fill="none" stroke="#4f46e5" stroke-width="2.4"/>
    <line x1="150" y1="140" x2="150" y2="30" stroke="#cbd5e1" stroke-dasharray="3 3"/>
    <text x="20" y="34" font-size="10" fill="#64748b" font-family="sans-serif">1 bit</text>
    <text x="132" y="156" font-size="10" fill="#64748b" font-family="sans-serif">p=0.5</text>
    <text x="44" y="156" font-size="10" fill="#64748b" font-family="sans-serif">0</text>
    <text x="252" y="156" font-size="10" fill="#64748b" font-family="sans-serif">1</text>
    <text x="90" y="24" font-size="10" fill="#4f46e5" font-family="sans-serif">H(p) = entropy of a coin</text>
  </svg>
  <figcaption>Maximum uncertainty (1 bit) at a fair 50/50 split; zero at a certain outcome.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>Shannon entropy</strong> of a distribution $p$ over outcomes is</p>
  $$H(p)=-\sum_i p_i\log p_i=E_{p}[-\log p].$$
  <p>Base-2 logs give <strong>bits</strong>, natural logs give <strong>nats</strong>. Entropy is maximized by the
  <strong>uniform</strong> distribution ($H=\log n$ over $n$ outcomes) and is $0$ when one outcome has probability 1. A
  model's <strong>perplexity</strong> is $2^{H}$ (bits) or $e^{H}$ (nats) — the effective number of equally-likely
  options.</p>`,

  derivation: String.raw`<p><strong>Entropy of a coin, and why it peaks at $p=\tfrac12$.</strong></p>
  <p><strong>Step 1 — write it out</strong> (in bits): $H(p)=-p\log_2 p-(1-p)\log_2(1-p).$</p>
  <p><strong>Step 2 — differentiate</strong> (natural log for ease, same maximizer):
  $\dfrac{dH}{dp}=-\log p-1+\log(1-p)+1=\log\dfrac{1-p}{p}.$</p>
  <p><strong>Step 3 — set to zero:</strong> $\log\dfrac{1-p}{p}=0\Rightarrow\dfrac{1-p}{p}=1\Rightarrow p=\tfrac12.$</p>
  <p><strong>Step 4 — evaluate:</strong> $H(\tfrac12)=-\tfrac12\log_2\tfrac12-\tfrac12\log_2\tfrac12=1$ bit — the most
  uncertainty a single yes/no can hold. $\blacksquare$</p>
  <p><strong>Uniform is the max in general:</strong> for $n$ equally-likely outcomes, $H=-\sum\tfrac1n\log\tfrac1n=\log n$,
  and no other distribution over $n$ outcomes beats it — spreading probability out maximizes average surprise.</p>`,

  code: [
    { label: "Entropy, in bits, and perplexity", src: String.raw`
import numpy as np
def entropy_bits(p): p=np.asarray(p); p=p[p>0]; return -np.sum(p*np.log2(p))

print("fair coin      :", entropy_bits([0.5,0.5]), "bit")
print("biased 0.9/0.1 :", round(entropy_bits([0.9,0.1]),3), "bit (less uncertain)")
print("fair 4-sided   :", entropy_bits([0.25]*4), "bits = log2(4)")
print("certain        :", entropy_bits([1.0,0.0]), "bits (no surprise)")
# perplexity of an LM with per-token entropy 2 bits:
print("perplexity     :", 2**2, "(as confused as choosing among 4)")
` },
    { label: "The Bernoulli entropy curve", src: String.raw`
import numpy as np, matplotlib.pyplot as plt
p = np.linspace(1e-6, 1-1e-6, 200)
H = -p*np.log2(p) - (1-p)*np.log2(1-p)
plt.figure(figsize=(5,3.5))
plt.plot(p, H, lw=2); plt.axvline(0.5, ls="--", color="gray")
plt.xlabel("p"); plt.ylabel("entropy (bits)"); plt.title("Coin entropy peaks at p=0.5")
plt.tight_layout(); plt.show()
` }
  ],

  keyPoints: [
    "Surprise $=-\\log p$; entropy $H(p)=-\\sum p_i\\log p_i$ is average surprise.",
    "Entropy measures uncertainty: max for uniform ($\\log n$), zero for a certain outcome.",
    "Bits use $\\log_2$, nats use $\\ln$; they differ by the factor $\\ln 2$.",
    "A fair coin has entropy exactly 1 bit; a fair die $\\log_2 6\\approx2.585$ bits.",
    "Language-model perplexity is $2^{H}$ — the effective number of choices."
  ],

  commonMistakes: [
    { wrong: "Including zero-probability terms in the sum.", why: "$0\\log 0$ is taken as 0 (the limit), so drop zero-probability outcomes; naively computing $\\log 0$ gives $-\\infty$/NaN." },
    { wrong: "Mixing bits and nats.", why: "Entropy and perplexity must use a consistent base. A '2.0' entropy is very different in bits ($2^2=4$ perplexity) vs nats ($e^2\\approx7.4$)." },
    { wrong: "Thinking lower entropy is always 'better'.", why: "Low output entropy means a confident model, which is good only if it's <em>right</em>. Overconfident-and-wrong has low entropy too; calibration matters." }
  ],

  quiz: [
    { q: "The entropy of a fair coin is…", options: ["1 bit", "0 bits", "2 bits", "0.5 bit"], answer: 0,
      explain: "$-2\\cdot\\tfrac12\\log_2\\tfrac12=1$ bit — maximum for two outcomes." },
    { q: "Entropy is maximized by which distribution over $n$ outcomes?", options: ["uniform", "a one-hot (certain) one", "the most skewed", "any"], answer: 0,
      explain: "Uniform spreads probability evenly, giving max average surprise $\\log n$." },
    { q: "A distribution with a single certain outcome has entropy…", options: ["0", "1", "$\\log n$", "$\\infty$"], answer: 0,
      explain: "No surprise: $-1\\log 1=0$." },
    { q: "An LM has per-token entropy of 3 bits. Its perplexity is…", options: ["8", "3", "6", "9"], answer: 0,
      explain: "Perplexity $=2^{H}=2^3=8$ — as uncertain as choosing among 8 equally-likely tokens." },
    { q: "The entropy of a fair 8-sided die (bits) is…", options: ["3", "8", "6", "2"], answer: 0,
      explain: "$\\log_2 8=3$ bits — three yes/no questions pin down one of eight outcomes." }
  ],

  practice: [
    { level: "easy", prompt: "Compute the entropy (bits) of a distribution $[0.5, 0.25, 0.25]$.", solution: "$-(0.5\\log_2 0.5+0.25\\log_2 0.25+0.25\\log_2 0.25)=0.5+0.5+0.5=1.5$ bits." },
    { level: "med", prompt: "Which has higher entropy: $[0.9,0.1]$ or $[0.6,0.4]$? Why?", solution: "$[0.6,0.4]$ — it's closer to uniform, so more uncertain. $[0.9,0.1]$ is more predictable (lower entropy $\\approx0.47$ bit vs $\\approx0.97$ bit)." },
    { level: "med", prompt: "Show the entropy of a uniform distribution over $n$ outcomes is $\\log n$.", solution: "$H=-\\sum_{i=1}^n\\tfrac1n\\log\\tfrac1n=-n\\cdot\\tfrac1n\\log\\tfrac1n=-\\log\\tfrac1n=\\log n$." },
    { level: "hard", prompt: "AI task: two language models have held-out per-token cross-entropies of 1.5 and 2.0 nats. Give their perplexities and say what perplexity intuitively means.", solution: "Perplexity $=e^{\\text{CE}}$: $e^{1.5}\\approx4.48$ and $e^{2.0}\\approx7.39$. Intuitively, each model is on average as uncertain as if it were choosing uniformly among that many tokens at each step — 4.5 vs 7.4 'effective options'. The lower-perplexity model is more confident and (on held-out data) more accurate. Since cross-entropy is entropy plus a KL penalty (Lesson 12.2), lower perplexity means the model's predicted distribution is closer to the true next-token distribution." }
  ],

  deepDive: String.raw`<p><strong>Entropy as the ultimate compression limit — Shannon's source coding theorem.</strong></p>
  <p>Entropy isn't just a convenient uncertainty score; it's a hard physical limit. Shannon's <strong>source coding
  theorem</strong> says the entropy $H(p)$ (in bits) is the minimum average number of bits needed to encode outcomes
  drawn from $p$ — no lossless code can do better, and codes approaching it exist (Huffman, arithmetic coding). Surprise
  $-\log p$ literally is the ideal code length for an outcome: likely things get short codes, rare things long ones. So
  "average surprise" and "best achievable compression" are the same number.</p>
  <p>This is why entropy underpins so much of ML. A language model that predicts the next token well is, by this
  theorem, a good <strong>compressor</strong> of text — and indeed, cross-entropy loss measured in bits-per-token is
  exactly a compression rate, which is why "compression = intelligence" arguments carry weight. Information gain in
  decision trees is a reduction in entropy (bits saved). Max-entropy modeling picks the least-committal distribution
  consistent with known constraints — the most honest default (and the reason the Gaussian, Track 10, is the max-entropy
  distribution for a fixed variance). Once you see entropy as "the bits reality actually costs," cross-entropy, KL, and
  mutual information all become bookkeeping of who pays how many bits — which is exactly the rest of this track.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["12.3"] = {
  subtitle: "How much knowing one variable tells you about another — the currency of representation learning.",

  aiMoment: String.raw`<p>Contrastive methods like <strong>InfoNCE</strong> (SimCLR, CLIP) train representations by
  maximizing a lower bound on the <strong>mutual information</strong> between different views of the same data. The
  <strong>information bottleneck</strong> frames learning as keeping MI with the label while discarding MI with the
  input. Feature selection ranks features by MI with the target. It's the measure of shared information between
  variables.</p>`,

  plainEnglish: String.raw`<p><strong>Mutual information</strong> answers: how much does knowing $Y$ reduce your
  uncertainty about $X$? If they're independent, knowing one tells you nothing (MI = 0). If one determines the other,
  knowing it removes all uncertainty (MI = the full entropy).</p>`,

  intuition: String.raw`<p>Picture each variable's uncertainty as a circle of information. Where the circles overlap is
  what they share — the mutual information. Independent variables have non-overlapping circles; tightly-coupled ones
  overlap almost entirely.</p>
  <figure class="figure">
  <svg viewBox="0 0 300 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mutual information as the overlap of two entropy circles">
    <circle cx="120" cy="75" r="55" fill="#4f46e5" fill-opacity="0.18" stroke="#4f46e5"/>
    <circle cx="180" cy="75" r="55" fill="#0d9488" fill-opacity="0.18" stroke="#0d9488"/>
    <path d="M150,32 A55,55 0 0,1 150,118 A55,55 0 0,1 150,32" fill="#7c3aed" fill-opacity="0.30"/>
    <text x="86" y="80" font-size="12" fill="#4f46e5" font-family="sans-serif">H(X)</text>
    <text x="196" y="80" font-size="12" fill="#0d9488" font-family="sans-serif">H(Y)</text>
    <text x="128" y="140" font-size="10" fill="#7c3aed" font-family="sans-serif">I(X;Y) = overlap</text>
  </svg>
  <figcaption>Mutual information is the shared region — uncertainty in X that Y also explains.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>mutual information</strong> between $X$ and $Y$:</p>
  $$I(X;Y)=H(X)-H(X\mid Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)}=D_{\mathrm{KL}}\big(p(x,y)\,\Vert\,p(x)p(y)\big).$$
  <p>It's symmetric ($I(X;Y)=I(Y;X)$), always $\ge0$, and $=0$ exactly when $X$ and $Y$ are independent. In words: how
  much the joint distribution differs from "independent" — the KL divergence from the product of marginals.</p>`,

  derivation: String.raw`<p><strong>MI is the KL divergence from independence (hence $\ge0$).</strong></p>
  <p><strong>Step 1 — start from uncertainty reduction:</strong> $I(X;Y)=H(X)-H(X\mid Y)$, the drop in uncertainty about
  $X$ once you know $Y$.</p>
  <p><strong>Step 2 — expand both entropies</strong> using $p(x)=\sum_y p(x,y)$:
  $H(X)=-\sum_{x,y}p(x,y)\log p(x)$ and $H(X\mid Y)=-\sum_{x,y}p(x,y)\log p(x\mid y)$.</p>
  <p><strong>Step 3 — subtract:</strong> $I=\sum_{x,y}p(x,y)\log\dfrac{p(x\mid y)}{p(x)}=\sum_{x,y}p(x,y)\log\dfrac{p(x,y)}{p(x)p(y)}$
  (using $p(x\mid y)=p(x,y)/p(y)$).</p>
  <p><strong>Step 4 — recognize the KL:</strong> that sum is exactly $D_{\mathrm{KL}}(p(x,y)\,\Vert\,p(x)p(y))$. Since KL
  $\ge0$ (Lesson 12.2), $I(X;Y)\ge0$, with equality iff $p(x,y)=p(x)p(y)$ — i.e. independence. $\blacksquare$ Plain
  English: mutual information measures how far two variables are from being independent, in bits.</p>`,

  code: [
    { label: "Mutual information from a joint table", src: String.raw`
import numpy as np
def mutual_info(P):
    P=np.asarray(P); Px=P.sum(1,keepdims=True); Py=P.sum(0,keepdims=True)
    m=P>0
    return float(np.sum(P[m]*np.log2(P[m]/(Px@Py)[m])))

indep = np.array([[0.25,0.25],[0.25,0.25]])          # independent
coupled = np.array([[0.5,0.0],[0.0,0.5]])            # Y = X (perfectly coupled)
print("I(indep)   =", round(mutual_info(indep),3), "bits (independent -> 0)")
print("I(coupled) =", round(mutual_info(coupled),3), "bits (Y determines X -> 1)")
` },
    { label: "MI = H(X) − H(X|Y), verified", src: String.raw`
import numpy as np
P = np.array([[0.4,0.1],[0.1,0.4]])                  # correlated
Px = P.sum(1);
Hx = -np.sum(Px*np.log2(Px))
Hxy = -np.sum(P[P>0]*np.log2((P/P.sum(0,keepdims=True))[P>0]))   # H(X|Y)
print("H(X)      =", round(Hx,3))
print("H(X|Y)    =", round(Hxy,3))
print("I=H(X)-H(X|Y) =", round(Hx-Hxy,3), "bits (>0: they share info)")
` }
  ],

  keyPoints: [
    "$I(X;Y)=H(X)-H(X\\mid Y)$: uncertainty about $X$ removed by knowing $Y$.",
    "$I(X;Y)=D_{\\mathrm{KL}}(p(x,y)\\Vert p(x)p(y))\\ge0$, symmetric, $=0$ iff independent.",
    "MI captures ANY dependence (nonlinear too), unlike correlation which sees only linear.",
    "Contrastive learning (InfoNCE) maximizes a lower bound on MI between views.",
    "Estimating MI in high dimensions is hard — hence variational lower bounds."
  ],

  commonMistakes: [
    { wrong: "Equating zero correlation with zero mutual information.", why: "Correlation only measures linear association; MI is zero <em>only</em> under full independence. $Y=X^2$ (symmetric $X$) has zero correlation but positive MI." },
    { wrong: "Assuming MI is easy to estimate from samples.", why: "In high dimensions, plug-in MI estimates are badly biased and high-variance. Practical methods use bounds (InfoNCE/MINE), not direct estimation." },
    { wrong: "Forgetting MI is bounded by the entropies.", why: "$I(X;Y)\\le\\min(H(X),H(Y))$ — you can't share more information than a variable contains. A deterministic $Y=f(X)$ gives $I=H(Y)$." }
  ],

  quiz: [
    { q: "If $X$ and $Y$ are independent, $I(X;Y)$ is…", options: ["0", "1", "$H(X)$", "$\\infty$"], answer: 0,
      explain: "Independence means the joint equals the product of marginals, so the KL — and MI — is 0." },
    { q: "Mutual information equals…", options: ["$H(X)-H(X\\mid Y)$", "$H(X)+H(Y)$", "$H(X\\mid Y)$", "$H(X)\\cdot H(Y)$"], answer: 0,
      explain: "It's the reduction in uncertainty about $X$ from knowing $Y$." },
    { q: "MI is $D_{\\mathrm{KL}}$ between the joint and…", options: ["the product of marginals", "the uniform", "the prior", "the posterior"], answer: 0,
      explain: "$I(X;Y)=D_{\\mathrm{KL}}(p(x,y)\\Vert p(x)p(y))$ — distance from independence." },
    { q: "Compared to correlation, mutual information…", options: ["captures nonlinear dependence too", "only captures linear dependence", "is always smaller", "requires Gaussianity"], answer: 0,
      explain: "MI detects any statistical dependence; correlation sees only the linear part." },
    { q: "For a deterministic $Y=f(X)$, $I(X;Y)$ equals…", options: ["$H(Y)$", "0", "$\\infty$", "$H(X)+H(Y)$"], answer: 0,
      explain: "Knowing $X$ removes all uncertainty in $Y$, so they share $Y$'s full entropy $H(Y)$." }
  ],

  practice: [
    { level: "easy", prompt: "Two independent fair coins. What is $I(X;Y)$?", solution: "Independent ⇒ $I(X;Y)=0$ bits: knowing one coin says nothing about the other." },
    { level: "med", prompt: "$Y$ is an exact copy of a fair coin $X$. Compute $I(X;Y)$.", solution: "$Y=X$, so knowing $Y$ removes all uncertainty in $X$: $I=H(X)-H(X\\mid Y)=1-0=1$ bit — they share $X$'s full entropy." },
    { level: "med", prompt: "Explain why $I(X;Y)\\le\\min(H(X),H(Y))$.", solution: "$I(X;Y)=H(X)-H(X\\mid Y)$ and $H(X\\mid Y)\\ge0$, so $I\\le H(X)$; by symmetry $I\\le H(Y)$. You can't extract more shared information than either variable individually contains." },
    { level: "hard", prompt: "AI task: SimCLR/CLIP maximize mutual information between two views (or image–text pairs). Why maximize MI, and why use a bound like InfoNCE instead of MI directly?", solution: "Maximizing MI between two views of the same underlying content forces the representation to keep what's <em>shared</em> (the semantic content) and discard view-specific nuisance (crop, color, wording) — exactly the invariances we want. But MI is intractable to compute/optimize directly in high dimensions (the densities are unknown and estimates are badly biased). InfoNCE is a <em>tractable lower bound</em> on MI built from a contrastive classification task: pull matching pairs together and push mismatches apart across a batch. Maximizing the bound pushes up the true MI, giving representations where paired views are highly informative about each other — the basis of modern self-supervised and multimodal learning." }
  ],

  deepDive: String.raw`<p><strong>The information bottleneck: learning as squeezing information.</strong></p>
  <p>Mutual information gives a strikingly clean theory of what a good representation is. The <strong>information
  bottleneck</strong> principle says: given input $X$ and label $Y$, learn a representation $Z$ that <em>maximizes</em>
  $I(Z;Y)$ (keep everything predictive of the label) while <em>minimizing</em> $I(Z;X)$ (throw away everything else).
  The objective $\max\,I(Z;Y)-\beta\,I(Z;X)$ formalizes "compress the input, keep the signal." It reframes generalization
  as forgetting: a representation that keeps only label-relevant information can't overfit to input noise, because it
  literally doesn't retain it.</p>
  <p>This lens unifies a lot. Contrastive self-supervision maximizes $I$ between views to find semantic content;
  supervised bottlenecks trade off the two MIs; disentanglement and fairness constraints are MI conditions ("$Z$ should
  share no information with a sensitive attribute"). The catch, as the mistakes noted, is that MI is brutally hard to
  estimate in high dimensions, so practice runs on <em>bounds</em> — InfoNCE, MINE, variational bounds — that make the
  quantity trainable without computing it exactly. Mutual information is thus more of a north star than a number you read
  off: it tells you what representations <em>should</em> do (share the right information, discard the rest), and the
  engineering is in bounding it well. Jensen's inequality, next, is the tool that turns these intractable
  information quantities into optimizable bounds.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["12.4"] = {
  subtitle: "Convexity turns an intractable log-of-an-average into an optimizable bound — the VAE's ELBO.",

  aiMoment: String.raw`<p>The <strong>ELBO</strong> — the objective every VAE maximizes — is derived in one line from
  <strong>Jensen's inequality</strong>. The same move powers the EM algorithm and all of variational inference: when a
  quantity involves the log of an intractable integral (a marginal likelihood, a partition function), Jensen converts it
  into a <em>lower bound</em> you can actually optimize. It's also why $D_{\mathrm{KL}}\ge0$ and mutual information
  $\ge0$.</p>`,

  plainEnglish: String.raw`<p><strong>Jensen's inequality</strong> says: for a curve that bows upward (convex), the value
  at the average input is at most the average of the values. For a curve that bows downward (concave) like the logarithm,
  it flips: the log of an average is at least the average of the logs. That gap is exactly what lets us bound hard
  quantities.</p>`,

  intuition: String.raw`<p>Take two points on the concave $\log$ curve and connect them with a straight chord. The chord
  sits <em>below</em> the curve. Averaging the two $\log$ values lands you on the chord; taking $\log$ of the averaged
  input lands you on the (higher) curve. So $\log$ of the average $\ge$ average of the logs.</p>
  <figure class="figure">
  <svg viewBox="0 0 280 160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Concave log curve with a chord below it (Jensen)">
    <line x1="30" y1="135" x2="265" y2="135" stroke="#94a3b8"/>
    <path d="M40,120 C90,60 160,38 255,28" fill="none" stroke="#4f46e5" stroke-width="2.4"/>
    <line x1="70" y1="93" x2="220" y2="41" stroke="#dc2626" stroke-width="1.8"/>
    <circle cx="70" cy="93" r="3" fill="#dc2626"/><circle cx="220" cy="41" r="3" fill="#dc2626"/>
    <circle cx="145" cy="67" r="3" fill="#0d9488"/><circle cx="145" cy="55" r="3" fill="#4f46e5"/>
    <line x1="145" y1="67" x2="145" y2="55" stroke="#94a3b8" stroke-dasharray="2 2"/>
    <text x="150" y="70" font-size="9" fill="#0d9488" font-family="sans-serif">E[log X] (chord)</text>
    <text x="150" y="50" font-size="9" fill="#4f46e5" font-family="sans-serif">log E[X] (curve)</text>
    <text x="200" y="150" font-size="10" fill="#4f46e5" font-family="sans-serif">log is concave</text>
  </svg>
  <figcaption>Chord below curve: averaging then logging beats logging then averaging — log E[X] ≥ E[log X].</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Jensen's inequality.</strong> For a convex $f$ and random $X$: $f(E[X])\le E[f(X)]$.
  For a concave $f$ (like $\log$): $f(E[X])\ge E[f(X)]$, so $\log E[X]\ge E[\log X]$. The <strong>evidence lower bound
  (ELBO)</strong> for a latent-variable model $p(x,z)$ with any distribution $q(z)$:</p>
  $$\log p(x)\ \ge\ \underbrace{E_{q}\!\left[\log\frac{p(x,z)}{q(z)}\right]}_{\text{ELBO}}=E_q[\log p(x\mid z)]-D_{\mathrm{KL}}\big(q(z)\,\Vert\,p(z)\big).$$
  <p>The gap between $\log p(x)$ and the ELBO is exactly $D_{\mathrm{KL}}(q(z)\,\Vert\,p(z\mid x))\ge0$.</p>`,

  derivation: String.raw`<p><strong>Deriving the ELBO from Jensen.</strong></p>
  <p><strong>Step 1 — write the marginal as an expectation.</strong> Introduce any $q(z)>0$:
  $\log p(x)=\log\!\int p(x,z)\,dz=\log\!\int q(z)\dfrac{p(x,z)}{q(z)}\,dz=\log\,E_{q}\!\Big[\dfrac{p(x,z)}{q(z)}\Big].$</p>
  <p><strong>Step 2 — apply Jensen</strong> ($\log$ is concave, so $\log E[\cdot]\ge E[\log\cdot]$):
  $\log p(x)\ge E_{q}\!\Big[\log\dfrac{p(x,z)}{q(z)}\Big]=\text{ELBO}.$</p>
  <p><strong>Step 3 — split the ELBO:</strong> $E_q[\log p(x,z)-\log q(z)]=E_q[\log p(x\mid z)]+E_q[\log p(z)-\log q(z)]=E_q[\log p(x\mid z)]-D_{\mathrm{KL}}(q(z)\Vert p(z)).$
  Plain English: reconstruct the data well (first term) while keeping $q$ near the prior (second term).</p>
  <p><strong>Step 4 — the gap is a KL.</strong> Since $\log p(x)-\text{ELBO}=D_{\mathrm{KL}}(q(z)\Vert p(z\mid x))\ge0$,
  maximizing the ELBO both tightens the bound and pushes $q$ toward the true posterior. $\blacksquare$</p>`,

  code: [
    { label: "Jensen: log E[X] ≥ E[log X]", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(0.1, 5.0, size=100000)           # positive samples
print("log E[X] =", round(float(np.log(x.mean())), 4))
print("E[log X] =", round(float(np.mean(np.log(x))), 4))
print("log E[X] >= E[log X] ?", np.log(x.mean()) >= np.mean(np.log(x)))   # True
` },
    { label: "ELBO is a lower bound on log p(x)", src: String.raw`
import numpy as np
# toy: p(x,z) over a discrete z; compute true log p(x) vs ELBO for a chosen q
pxz = np.array([0.2, 0.05, 0.15])                # p(x, z) for z in {0,1,2}
logpx = np.log(pxz.sum())                         # true log marginal
for q in [np.array([1/3,1/3,1/3]), pxz/pxz.sum()]:  # arbitrary q, then optimal q
    elbo = np.sum(q * np.log(pxz / q))
    print(f"ELBO = {elbo:.4f}  <=  log p(x) = {logpx:.4f}   gap(KL) = {logpx-elbo:.4f}")
# the optimal q = p(z|x) makes the gap zero (ELBO tight)
` }
  ],

  keyPoints: [
    "Jensen: convex $f$ ⇒ $f(E[X])\\le E[f(X)]$; concave (log) ⇒ $\\log E[X]\\ge E[\\log X]$.",
    "The ELBO is a lower bound on $\\log p(x)$, obtained by applying Jensen to the log-marginal.",
    "ELBO $=E_q[\\log p(x\\mid z)]-D_{\\mathrm{KL}}(q(z)\\Vert p(z))$ — reconstruction minus a KL regularizer.",
    "The bound's gap is $D_{\\mathrm{KL}}(q\\Vert\\text{posterior})$; tightening it fits $q$ to the posterior.",
    "Jensen is also why $D_{\\mathrm{KL}}\\ge0$ and mutual information $\\ge0$."
  ],

  commonMistakes: [
    { wrong: "Getting the inequality direction backwards for log.", why: "$\\log$ is <em>concave</em>, so $\\log E[X]\\ge E[\\log X]$ (not $\\le$). Convex functions go the other way. Mixing them flips your bound." },
    { wrong: "Thinking the ELBO equals the log-likelihood.", why: "It's a lower bound; they're equal only when $q=p(z\\mid x)$ exactly. With an approximate $q$, there's a KL gap you can't see from the ELBO alone." },
    { wrong: "Treating the VAE's KL term as optional regularization.", why: "The $D_{\\mathrm{KL}}(q\\Vert p(z))$ term isn't a bolt-on — it falls directly out of the ELBO derivation. It's the price of the variational bound, not an ad-hoc penalty." }
  ],

  quiz: [
    { q: "For the concave $\\log$, Jensen gives…", options: ["$\\log E[X]\\ge E[\\log X]$", "$\\log E[X]\\le E[\\log X]$", "$\\log E[X]=E[\\log X]$", "no relation"], answer: 0,
      explain: "Concave functions satisfy $f(E[X])\\ge E[f(X)]$." },
    { q: "The ELBO is…", options: ["a lower bound on $\\log p(x)$", "an upper bound on $\\log p(x)$", "exactly $\\log p(x)$", "the KL divergence"], answer: 0,
      explain: "Jensen applied to $\\log p(x)=\\log E_q[\\cdot]$ yields a lower bound." },
    { q: "The gap between $\\log p(x)$ and the ELBO equals…", options: ["$D_{\\mathrm{KL}}(q(z)\\Vert p(z\\mid x))$", "0 always", "the entropy", "the reconstruction error"], answer: 0,
      explain: "It's the KL from $q$ to the true posterior; zero only when they match." },
    { q: "The ELBO splits into reconstruction minus…", options: ["$D_{\\mathrm{KL}}(q(z)\\Vert p(z))$", "the entropy of $x$", "mutual information", "the prior mean"], answer: 0,
      explain: "$E_q[\\log p(x\\mid z)]-D_{\\mathrm{KL}}(q(z)\\Vert p(z))$ — fit the data, stay near the prior." },
    { q: "Jensen's inequality also proves that…", options: ["$D_{\\mathrm{KL}}\\ge0$", "entropy is negative", "correlation $=$ MI", "variance $=0$"], answer: 0,
      explain: "Applying Jensen to $-\\log$ (convex) gives the non-negativity of KL (Gibbs, Lesson 12.2)." }
  ],

  practice: [
    { level: "easy", prompt: "Is $\\log$ convex or concave, and which way does Jensen point for it?", solution: "Concave ($\\frac{d^2}{dx^2}\\log x=-1/x^2<0$). So $\\log E[X]\\ge E[\\log X]$." },
    { level: "med", prompt: "Use Jensen to show $E[X^2]\\ge(E[X])^2$.", solution: "$f(x)=x^2$ is convex, so $f(E[X])\\le E[f(X)]$: $(E[X])^2\\le E[X^2]$. Their difference is the variance $\\ge0$ (Lesson 9.4) — a one-line proof via Jensen." },
    { level: "med", prompt: "Write the ELBO's two terms and say what each encourages.", solution: "$\\text{ELBO}=E_{q}[\\log p(x\\mid z)]-D_{\\mathrm{KL}}(q(z)\\Vert p(z))$. The first term encourages latents $z$ that reconstruct $x$ well; the second keeps the approximate posterior $q$ close to the prior $p(z)$ (regularizing the latent space, e.g. toward $\\mathcal N(0,I)$)." },
    { level: "hard", prompt: "AI task: explain why VAEs maximize the ELBO instead of the true log-likelihood, and what the reparameterization trick contributes.", solution: "The true $\\log p(x)=\\log\\int p(x,z)dz$ is intractable (the integral over latents has no closed form for a neural decoder). Jensen converts it into the ELBO — a tractable lower bound you can compute and differentiate for any chosen $q(z)$. Maximizing the ELBO simultaneously (a) raises a bound on the data likelihood and (b) shrinks the gap $D_{\\mathrm{KL}}(q\\Vert p(z\\mid x))$, fitting the approximate posterior. The reconstruction term is an expectation over $z\\sim q$, which isn't directly differentiable in $q$'s parameters — so the reparameterization trick (Track 10.5) writes $z=\\mu+\\sigma\\varepsilon$ to push gradients through the sample, while the KL term has a closed form for Gaussians. Together, Jensen makes the objective tractable and reparameterization makes it trainable." }
  ],

  deepDive: String.raw`<p><strong>Variational inference: turning integration into optimization.</strong></p>
  <p>The ELBO is one instance of a sweeping idea. Exact Bayesian inference requires an intractable integral — the
  evidence $p(x)=\int p(x,z)\,dz$, or a posterior $p(z\mid x)$ with no closed form. <strong>Variational inference</strong>
  sidesteps it: pick a tractable family of distributions $q_\phi(z)$, and <em>optimize</em> $\phi$ to make $q$ as close as
  possible to the true posterior — which, by the derivation above, is exactly maximizing the ELBO. A hard integration
  problem becomes a gradient-descent problem. This is the engine behind VAEs, Bayesian neural networks, topic models,
  and much of probabilistic programming.</p>
  <p>The same Jensen move recurs whenever a "log of an expectation" blocks progress. The EM algorithm alternates between
  tightening the bound (E-step: set $q=p(z\mid x)$) and maximizing it (M-step). Importance-weighted and multi-sample
  bounds (IWAE) tighten the ELBO by averaging inside the log before applying Jensen. Even the non-negativity of KL and
  MI — the backbone of the previous lessons — are Jensen in disguise. So this one inequality is the quiet workhorse that
  makes intractable probabilistic quantities <em>optimizable</em>: it converts "I can't compute this integral" into "I
  can maximize this bound," which is the difference between an elegant theory and a trainable model. It closes the
  information-theory track by giving you the tool that turns entropy, cross-entropy, KL, and mutual information from
  quantities you admire into objectives you can descend.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["12.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 65 minutes.",

  intro: String.raw`<p>This exam spans all of Track 12: entropy, cross-entropy and KL divergence, mutual information, and
  Jensen's inequality / the ELBO. <strong>Give yourself 65 minutes</strong>, produce each answer before checking, and
  score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "Give the entropy (bits) of a fair 4-sided die.",
      solution: "$\\log_2 4=2$ bits (uniform over 4 outcomes)." },
    { level: "easy", prompt: "A language model has per-token cross-entropy 2 nats. Give its perplexity.",
      solution: "Perplexity $=e^{2}\\approx7.39$ — as uncertain as choosing among ~7.4 tokens." },
    { level: "med", prompt: "Compute the entropy of $[0.5, 0.25, 0.125, 0.125]$ in bits.",
      solution: "$-(0.5\\log_2 0.5+0.25\\log_2 0.25+2\\cdot0.125\\log_2 0.125)=0.5+0.5+2(0.375)=1.75$ bits." },
    { level: "med", prompt: "State the identity linking cross-entropy, entropy, and KL, and what it implies for training.",
      solution: "$H(p,q)=H(p)+D_{\\mathrm{KL}}(p\\Vert q)$. Since $H(p)$ (the label entropy) is fixed, minimizing cross-entropy is exactly minimizing the KL from the truth $p$ to the model $q$ — you can never beat the floor $H(p)$." },
    { level: "med", prompt: "Two independent variables have what mutual information? Give the general formula for MI as a KL.",
      solution: "$I(X;Y)=0$ for independent variables. In general $I(X;Y)=D_{\\mathrm{KL}}(p(x,y)\\Vert p(x)p(y))$ — the KL from the joint to the product of marginals, which is 0 iff independent." },
    { level: "med", prompt: "Which way does Jensen point for $\\log$, and use it to prove $E[X^2]\\ge(E[X])^2$.",
      solution: "$\\log$ concave ⇒ $\\log E[X]\\ge E[\\log X]$. For the second part use convex $f(x)=x^2$: $f(E[X])\\le E[f(X)]$ gives $(E[X])^2\\le E[X^2]$ (difference $=$ variance $\\ge0$)." },
    { level: "hard", prompt: "Why is $D_{\\mathrm{KL}}(p\\Vert q)\\ge0$, and why is it asymmetric?",
      solution: "Non-negativity is Gibbs' inequality, provable via Jensen on the convex $-\\log$: $D_{\\mathrm{KL}}(p\\Vert q)=E_p[-\\log(q/p)]\\ge-\\log E_p[q/p]=-\\log 1=0$. It's asymmetric because it weights the log-ratio by $p$ (the first argument): swapping $p$ and $q$ changes both the weighting and the ratio, so $D_{\\mathrm{KL}}(p\\Vert q)\\ne D_{\\mathrm{KL}}(q\\Vert p)$ in general — forward KL is mode-covering, reverse is mode-seeking." },
    { level: "hard", prompt: "Derive the ELBO for a latent-variable model in two lines.",
      solution: "$\\log p(x)=\\log\\int q(z)\\frac{p(x,z)}{q(z)}dz=\\log E_q[\\frac{p(x,z)}{q(z)}]\\ \\ge\\ E_q[\\log\\frac{p(x,z)}{q(z)}]$ (Jensen, log concave). The right side is the ELBO $=E_q[\\log p(x\\mid z)]-D_{\\mathrm{KL}}(q(z)\\Vert p(z))$, with gap $D_{\\mathrm{KL}}(q(z)\\Vert p(z\\mid x))\\ge0$." },
    { level: "hard", prompt: "A classifier is 'confidently wrong': true class prob 0.001. Give the cross-entropy loss (nats) and explain its effect.",
      solution: "Loss $=-\\ln(0.001)=\\ln 1000\\approx6.91$ nats — huge. Because the loss grows like $-\\log q_{\\text{true}}$, one confident mistake contributes an enormous term that dominates the batch average and produces a large gradient, which is why confident errors are heavily penalized (and why label smoothing / clipping help stabilize training)." },
    { level: "hard", prompt: "AI task: explain why 'a good language model is a good compressor,' linking entropy and cross-entropy.",
      solution: "By Shannon's source coding theorem, the entropy $H(p)$ (bits) is the minimum average code length for data from $p$, with the ideal code length of an outcome being $-\\log_2 p$. A language model $q$ used to code text achieves an average length equal to the cross-entropy $H(p,q)=H(p)+D_{\\mathrm{KL}}(p\\Vert q)$ bits per token — its 'bits-per-token' loss <em>is</em> a compression rate. Minimizing cross-entropy shrinks the KL to the true distribution, so a better predictor codes text in fewer bits: better modeling = better compression. This is the formal basis for 'compression ≈ understanding' arguments about LLMs." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Information theory is solid — you can read losses as bits and derive the ELBO. On to Numerical Methods (Track 13) and the information-theory capstone.</li>
    <li><strong>7–8:</strong> Strong. Revisit the CE = H + KL identity or the ELBO derivation if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive KL ≥ 0 via Jensen and MI as a KL; redo Lessons 12.2 and 12.4.</li>
    <li><strong>Below 5:</strong> Rework the track — entropy, KL, and the ELBO are the language of modern generative and self-supervised learning.</li>
  </ul>`
};
