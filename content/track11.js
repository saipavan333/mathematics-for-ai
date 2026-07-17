/* ============================================================
   TRACK 11 — Statistics & Estimation
   11.1 Point Estimation / Bias & Variance · 11.2 MLE · 11.3 MAP & Priors ·
   11.4 Bias–Variance Tradeoff · 11.5 Hypothesis Testing / CIs / Bootstrap ·
   11.E Track Exam
   ============================================================ */
(window.LESSON_CONTENT ||= {})["11.1"] = {
  subtitle: "An estimator turns data into a guess — judged by how off-center (bias) and how jittery (variance) it is.",

  aiMoment: String.raw`<p>A model's test error decomposes into three pieces — <strong>bias²</strong>, <strong>variance</strong>,
  and irreducible noise — and that decomposition is the lens behind under/overfitting, model-size choices, and
  regularization. Before the famous tradeoff (Lesson 11.4), you need the atoms: what an <strong>estimator</strong> is,
  and what its bias and variance mean.</p>`,

  plainEnglish: String.raw`<p>An <strong>estimator</strong> is a recipe that turns a dataset into a guess for some
  unknown quantity (a mean, a weight, an accuracy). Run it on many datasets and two things matter: is it centered on the
  right answer (<strong>bias</strong>), and how much does the guess jump around from dataset to dataset
  (<strong>variance</strong>)?</p>`,

  intuition: String.raw`<p>Think of throwing darts at a target. <strong>Bias</strong> is how far the cluster's center is
  from the bullseye; <strong>variance</strong> is how spread out the darts are. You can be tightly clustered but
  off-center (biased, low variance) or centered on average but scattered (unbiased, high variance).</p>
  <figure class="figure">
  <svg viewBox="0 0 320 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dartboards showing bias and variance">
    <g>
    <circle cx="80" cy="80" r="52" fill="none" stroke="#cbd5e1"/><circle cx="80" cy="80" r="34" fill="none" stroke="#cbd5e1"/><circle cx="80" cy="80" r="16" fill="none" stroke="#cbd5e1"/>
    <circle cx="80" cy="80" r="3" fill="#94a3b8"/>
    <g fill="#dc2626"><circle cx="120" cy="52" r="3"/><circle cx="126" cy="60" r="3"/><circle cx="118" cy="64" r="3"/><circle cx="128" cy="50" r="3"/></g>
    <text x="36" y="152" font-size="11" fill="#dc2626" font-family="sans-serif">high bias, low variance</text>
    </g>
    <g transform="translate(160,0)">
    <circle cx="80" cy="80" r="52" fill="none" stroke="#cbd5e1"/><circle cx="80" cy="80" r="34" fill="none" stroke="#cbd5e1"/><circle cx="80" cy="80" r="16" fill="none" stroke="#cbd5e1"/>
    <circle cx="80" cy="80" r="3" fill="#94a3b8"/>
    <g fill="#0d9488"><circle cx="60" cy="60" r="3"/><circle cx="104" cy="72" r="3"/><circle cx="78" cy="104" r="3"/><circle cx="96" cy="52" r="3"/><circle cx="58" cy="96" r="3"/></g>
    <text x="34" y="152" font-size="11" fill="#0d9488" font-family="sans-serif">low bias, high variance</text>
    </g>
  </svg>
  <figcaption>Bias = distance of the cluster's center from the bullseye; variance = how scattered the throws are.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For an estimator $\hat\theta$ of a true value $\theta$:</p>
  $$\operatorname{bias}(\hat\theta)=E[\hat\theta]-\theta,\qquad \operatorname{Var}(\hat\theta)=E\big[(\hat\theta-E[\hat\theta])^2\big],$$
  $$\operatorname{MSE}(\hat\theta)=E\big[(\hat\theta-\theta)^2\big]=\operatorname{bias}(\hat\theta)^2+\operatorname{Var}(\hat\theta).$$
  <p>The expectations are over random datasets. $\hat\theta$ is <strong>unbiased</strong> if $\operatorname{bias}=0$, and
  <strong>consistent</strong> if $\hat\theta\to\theta$ as the sample size grows.</p>`,

  derivation: String.raw`<p><strong>The MSE splits into bias² + variance.</strong> Write $\mu=E[\hat\theta]$.</p>
  <p><strong>Step 1 — add and subtract $\mu$:</strong> $\operatorname{MSE}=E[(\hat\theta-\theta)^2]=E\big[((\hat\theta-\mu)+(\mu-\theta))^2\big].$</p>
  <p><strong>Step 2 — expand the square:</strong> $=E[(\hat\theta-\mu)^2]+2E[(\hat\theta-\mu)](\mu-\theta)+(\mu-\theta)^2.$</p>
  <p><strong>Step 3 — the cross term vanishes:</strong> $E[\hat\theta-\mu]=E[\hat\theta]-\mu=0$, so the middle term is 0.</p>
  <p><strong>Step 4 — name the pieces:</strong> $\operatorname{MSE}=\underbrace{E[(\hat\theta-\mu)^2]}_{\text{variance}}+\underbrace{(\mu-\theta)^2}_{\text{bias}^2}.\;\blacksquare$
  Plain English: total squared error is exactly how scattered you are plus how off-center you are — you can trade one
  against the other.</p>`,

  code: [
    { label: "Bias of the naive variance estimator (1/n vs 1/(n-1))", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

true_var = 4.0                                   # population variance (σ²=4)
biased, unbiased = [], []
for _ in range(20000):
    x = rng.normal(0, 2.0, size=5)               # small sample of 5
    biased.append(x.var())                       # divides by n  -> biased low
    unbiased.append(x.var(ddof=1))               # divides by n-1 -> unbiased
print("E[1/n   estimator] =", round(np.mean(biased),   3), " (true 4.0, biased low)")
print("E[1/n-1 estimator] =", round(np.mean(unbiased), 3), " (true 4.0, unbiased)")
` },
    { label: "MSE = bias² + variance, verified", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)
theta = 3.0
ests = np.array([rng.normal(0,1,size=10).mean()+theta+0.5 for _ in range(50000)])  # biased by +0.5
bias = ests.mean() - theta
var  = ests.var()
mse  = np.mean((ests - theta)**2)
print("bias² + var =", round(bias**2 + var, 4), " vs MSE =", round(mse, 4))
` }
  ],

  keyPoints: [
    "An estimator maps data → a guess for an unknown parameter.",
    "Bias $=E[\\hat\\theta]-\\theta$ (systematic offset); variance $=$ jitter across datasets.",
    "$\\operatorname{MSE}=\\operatorname{bias}^2+\\operatorname{Var}$ — the master decomposition.",
    "Unbiased means centered on the truth; consistent means it converges as $n\\to\\infty$.",
    "Lower bias often costs higher variance and vice versa (Lesson 11.4)."
  ],

  commonMistakes: [
    { wrong: "Thinking unbiased is always best.", why: "A slightly biased estimator with much lower variance can have smaller MSE. Regularized (biased) estimators routinely beat unbiased ones — the whole point of the tradeoff." },
    { wrong: "Dividing by $n$ for a sample variance when you need an unbiased estimate.", why: "The $1/n$ version underestimates the population variance; use $1/(n-1)$ (Bessel's correction) for an unbiased estimate." },
    { wrong: "Confusing the variance of the data with the variance of the estimator.", why: "Data variance $\\sigma^2$ is a property of the distribution; estimator variance is how much your <em>estimate</em> wobbles across samples (often $\\sigma^2/n$)." }
  ],

  quiz: [
    { q: "An estimator has $E[\\hat\\theta]=5$ and true $\\theta=4$. Its bias is…", options: ["1", "−1", "0", "9"], answer: 0,
      explain: "Bias $=E[\\hat\\theta]-\\theta=5-4=1$." },
    { q: "If bias $=2$ and variance $=3$, the MSE is…", options: ["7", "5", "13", "6"], answer: 0,
      explain: "$\\operatorname{MSE}=\\operatorname{bias}^2+\\operatorname{Var}=4+3=7$." },
    { q: "An unbiased estimator is one where…", options: ["$E[\\hat\\theta]=\\theta$", "$\\operatorname{Var}=0$", "$\\hat\\theta=\\theta$ always", "MSE $=0$"], answer: 0,
      explain: "Unbiased means centered on the truth in expectation, not zero variance." },
    { q: "The sample variance with $1/n$ (not $1/(n-1)$) tends to…", options: ["underestimate the true variance", "overestimate it", "be unbiased", "be zero"], answer: 0,
      explain: "Dividing by $n$ biases the estimate low; $1/(n-1)$ corrects it." },
    { q: "A tightly-clustered but off-center estimator has…", options: ["high bias, low variance", "low bias, high variance", "zero MSE", "high both"], answer: 0,
      explain: "Clustered = low variance; off-center = high bias." }
  ],

  practice: [
    { level: "easy", prompt: "An estimator gives values with mean 10 across datasets; the true value is 12. Bias?", solution: "$E[\\hat\\theta]-\\theta=10-12=-2$." },
    { level: "med", prompt: "Show the sample mean $\\bar X$ is an unbiased estimator of $\\mu$.", solution: "$E[\\bar X]=E[\\tfrac1n\\sum X_i]=\\tfrac1n\\sum E[X_i]=\\tfrac1n\\cdot n\\mu=\\mu$. So bias $=0$." },
    { level: "med", prompt: "The sample mean of $n$ iid samples (variance $\\sigma^2$) has what variance?", solution: "$\\operatorname{Var}(\\bar X)=\\sigma^2/n$ (independent terms, Lesson 9.5). So its MSE is $\\sigma^2/n$ (unbiased), shrinking with more data." },
    { level: "hard", prompt: "AI task: a regularized model has bias 0.3 and variance 0.05; an unregularized one has bias 0 and variance 0.5. Which generalizes better by MSE, and what does this say about regularization?", solution: "Regularized MSE $=0.3^2+0.05=0.14$; unregularized MSE $=0+0.5=0.5$. The regularized model wins despite being biased — regularization deliberately adds bias to cut variance, and when variance dominates (small data, big models), that trade lowers total error. This is exactly why weight decay/early stopping help generalization even though they pull the fit away from the training data." }
  ],

  deepDive: String.raw`<p><strong>Why 'wrong on purpose' can be right: the James–Stein shock.</strong></p>
  <p>Intuition says an unbiased estimator should be best, but statistics has a famous counterexample. When estimating
  three or more means at once, the <strong>James–Stein estimator</strong> — which deliberately shrinks each estimate
  toward a common point, introducing bias — has provably <em>lower</em> total MSE than the obvious unbiased choice (the
  raw sample means). Shrinkage trades a little bias for a large drop in variance, and wins. This was so counterintuitive
  it was initially disbelieved.</p>
  <p>That result is the theoretical ancestor of nearly every regularizer in ML. Weight decay shrinks weights toward
  zero; early stopping keeps them near their small initial values; ensembling and dropout average away variance. All are
  biased on purpose because, in the high-variance regime where modern over-parameterized models live, cutting variance
  matters more than eliminating bias. The bias–variance decomposition isn't just bookkeeping — it's the reason
  "regularize" is the default advice, and the next few lessons make the specific trades (MLE vs MAP, model capacity)
  precise.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["11.2"] = {
  subtitle: "Pick the parameters that make your data most probable — and rediscover cross-entropy and least squares.",

  aiMoment: String.raw`<p>The <strong>cross-entropy loss</strong> you minimize for every classifier <em>is</em> the
  negative log-likelihood of the data under the model — training is <strong>maximum likelihood estimation</strong>.
  Fitting a Gaussian by MLE gives least squares; fitting a categorical gives softmax + cross-entropy. MLE is the single
  principle behind most supervised training objectives.</p>`,

  plainEnglish: String.raw`<p><strong>Maximum likelihood</strong> asks: of all possible parameter settings, which one
  makes the data I actually observed the most probable? You write down the probability of the data as a function of the
  parameters, then turn the knobs to maximize it.</p>`,

  intuition: String.raw`<p>Slide and stretch a distribution over your data points. Where it sits so the points fall in
  its high-probability region, the likelihood is largest. MLE finds that best-fitting position.</p>
  <figure class="figure">
  <svg viewBox="0 0 300 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A distribution fitted to data points by maximum likelihood">
    <line x1="20" y1="120" x2="285" y2="120" stroke="#94a3b8"/>
    <path d="M60,120 C110,120 120,40 155,40 C190,40 200,120 250,120" fill="none" stroke="#4f46e5" stroke-width="2"/>
    <g fill="#dc2626"><circle cx="120" cy="120" r="3"/><circle cx="140" cy="120" r="3"/><circle cx="155" cy="120" r="3"/><circle cx="165" cy="120" r="3"/><circle cx="185" cy="120" r="3"/></g>
    <text x="96" y="30" font-size="11" fill="#4f46e5" font-family="sans-serif">p(x | θ) fit to maximize likelihood</text>
    <text x="150" y="140" font-size="10" fill="#dc2626" font-family="sans-serif">observed data</text>
  </svg>
  <figcaption>MLE positions the distribution so the observed points land where the density is high.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Given iid data $x_1,\dots,x_n$, the <strong>likelihood</strong> is $L(\theta)=\prod_i p(x_i\mid\theta)$.
  We maximize its log (Lesson 1.2) or minimize the <strong>negative log-likelihood</strong>:</p>
  $$\hat\theta_{\text{MLE}}=\arg\max_\theta\sum_{i}\log p(x_i\mid\theta)=\arg\min_\theta\underbrace{\Big(-\sum_i\log p(x_i\mid\theta)\Big)}_{\text{NLL}}.$$
  <p>For a categorical model with predicted class probabilities $q$ and one-hot labels $y$, the per-example NLL is
  $-\sum_c y_c\log q_c$ — exactly the <strong>cross-entropy</strong> (Track 12).</p>`,

  derivation: String.raw`<p><strong>Part 1 — MLE for a coin (Bernoulli).</strong> Observe $k$ heads in $n$ flips.</p>
  <p><strong>Step 1 — log-likelihood:</strong> $\ell(\theta)=k\log\theta+(n-k)\log(1-\theta).$</p>
  <p><strong>Step 2 — differentiate, set to 0:</strong> $\ell'(\theta)=\dfrac{k}{\theta}-\dfrac{n-k}{1-\theta}=0.$</p>
  <p><strong>Step 3 — solve:</strong> $k(1-\theta)=(n-k)\theta\Rightarrow k=n\theta\Rightarrow \hat\theta=\dfrac{k}{n}.\;\blacksquare$
  The MLE of a probability is just the observed frequency.</p>
  <hr class="soft">
  <p><strong>Part 2 — MLE for a Gaussian mean = least squares.</strong> With $x_i\sim\mathcal N(\mu,\sigma^2)$,
  $\ell(\mu)=-\dfrac{1}{2\sigma^2}\sum_i(x_i-\mu)^2+\text{const}.$ Maximizing $\ell$ over $\mu$ is <em>minimizing</em>
  $\sum_i(x_i-\mu)^2$ — a sum of squared errors. Setting the derivative to zero gives $\hat\mu=\bar x$ (Step: $\sum(x_i-\mu)=0$).
  $\blacksquare$ So Gaussian MLE is exactly least squares, and the quadratic loss is a log-density, not an arbitrary choice.</p>`,

  code: [
    { label: "MLE for Bernoulli and Gaussian", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

flips = rng.random(200) < 0.7            # true p = 0.7
print("Bernoulli MLE  p̂ = k/n =", round(float(flips.mean()), 3))

x = rng.normal(5.0, 2.0, size=500)
print("Gaussian MLE   μ̂ = x̄  =", round(float(x.mean()), 3))
print("Gaussian MLE   σ²̂ (1/n) =", round(float(x.var()), 3))
` },
    { label: "Cross-entropy IS the negative log-likelihood", src: String.raw`
import numpy as np

q = np.array([0.1, 0.7, 0.2])            # model's predicted class probabilities
true_class = 1                            # one-hot label = [0,1,0]

nll = -np.log(q[true_class])              # negative log-likelihood of the label
ce  = -np.sum(np.eye(3)[true_class] * np.log(q))   # cross-entropy with one-hot y
print("NLL           =", round(float(nll), 4))
print("cross-entropy =", round(float(ce),  4), "-> identical")
` }
  ],

  keyPoints: [
    "MLE picks $\\theta$ maximizing $\\prod_i p(x_i\\mid\\theta)$ — equivalently minimizing the NLL.",
    "Cross-entropy loss = negative log-likelihood of the labels; training a classifier is MLE.",
    "Bernoulli MLE: $\\hat\\theta=k/n$ (observed frequency).",
    "Gaussian-mean MLE = least squares; the squared loss is a Gaussian log-density.",
    "Work in log-space: products of probabilities become stable sums (Lesson 1.1)."
  ],

  commonMistakes: [
    { wrong: "Maximizing the likelihood product directly.", why: "The product underflows for large $n$. Always maximize the <em>log</em>-likelihood (a sum), which is numerically stable and has the same maximizer." },
    { wrong: "Thinking cross-entropy and NLL are different losses.", why: "They're the same quantity for a one-hot label: $-\\sum_c y_c\\log q_c=-\\log q_{\\text{true}}$. 'Minimize cross-entropy' and 'do maximum likelihood' are identical instructions." },
    { wrong: "Forgetting MLE can overfit.", why: "MLE fits the data as hard as possible with no prior; for flexible models it overfits. Adding a prior gives MAP (next lesson) — i.e. regularization." }
  ],

  quiz: [
    { q: "You observe 8 heads in 10 flips. The Bernoulli MLE for $p$ is…", options: ["0.8", "0.5", "0.2", "8"], answer: 0,
      explain: "$\\hat p=k/n=8/10=0.8$ — the observed frequency." },
    { q: "Minimizing cross-entropy is the same as…", options: ["maximum likelihood", "maximizing variance", "minimizing bias only", "MAP with a strong prior"], answer: 0,
      explain: "Cross-entropy is the negative log-likelihood, so minimizing it maximizes likelihood." },
    { q: "Gaussian-mean MLE minimizes…", options: ["$\\sum_i(x_i-\\mu)^2$ (least squares)", "$\\sum_i|x_i-\\mu|$", "$\\max_i|x_i-\\mu|$", "the variance only"], answer: 0,
      explain: "The Gaussian log-density is a negative sum of squares, so MLE = least squares, giving $\\hat\\mu=\\bar x$." },
    { q: "Why maximize the log-likelihood instead of the likelihood?", options: ["it turns a product into a stable sum with the same maximizer", "it changes the answer", "it removes the parameters", "logs are faster to multiply"], answer: 0,
      explain: "$\\log$ is monotonic (same argmax) and converts the underflow-prone product into a sum." },
    { q: "The per-example NLL for a one-hot label with predicted prob $0.2$ on the true class is…", options: ["$-\\ln 0.2\\approx1.61$", "$0.2$", "$\\ln 0.2$", "$0.8$"], answer: 0,
      explain: "$-\\log q_{\\text{true}}=-\\ln0.2\\approx1.61$ — exactly the cross-entropy." }
  ],

  practice: [
    { level: "easy", prompt: "Observe 3 heads in 12 flips. Give the MLE for the coin's bias.", solution: "$\\hat p=3/12=0.25$." },
    { level: "med", prompt: "Show the Gaussian-mean MLE is $\\hat\\mu=\\bar x$.", solution: "$\\ell(\\mu)=-\\tfrac{1}{2\\sigma^2}\\sum(x_i-\\mu)^2+\\text{const}$. $\\frac{d\\ell}{d\\mu}=\\tfrac{1}{\\sigma^2}\\sum(x_i-\\mu)=0\\Rightarrow\\sum x_i=n\\mu\\Rightarrow\\hat\\mu=\\bar x$." },
    { level: "med", prompt: "Why is the quadratic (MSE) loss the 'natural' loss for regression?", solution: "Assuming Gaussian noise $y=f(x)+\\varepsilon$, $\\varepsilon\\sim\\mathcal N(0,\\sigma^2)$, the log-likelihood of the data is $-\\tfrac{1}{2\\sigma^2}\\sum(y_i-f(x_i))^2+\\text{const}$. Maximizing it is minimizing the sum of squared errors — so MSE <em>is</em> the MLE objective under Gaussian noise, not an arbitrary choice." },
    { level: "hard", prompt: "AI task: a classifier outputs logits, softmax makes probabilities $q$, and you minimize cross-entropy. Explain the full chain from MLE to this training loss.", solution: "Model $P(\\text{class}=c\\mid x)=q_c=\\text{softmax}(\\text{logits})_c$. The likelihood of the dataset is $\\prod_i q_{i,y_i}$ (each example's probability under its true label). MLE maximizes the log: $\\sum_i\\log q_{i,y_i}$, equivalently minimizes $-\\sum_i\\log q_{i,y_i}=\\sum_i(-\\sum_c y_{ic}\\log q_{ic})$ — the average cross-entropy. So 'softmax + cross-entropy' is precisely maximum-likelihood estimation of a categorical model, which is why it's the universal classification loss." }
  ],

  deepDive: String.raw`<p><strong>Why maximum likelihood is a principled default — and its guarantees.</strong></p>
  <p>MLE isn't just convenient; it has strong theoretical backing. Under mild conditions it is <strong>consistent</strong>
  (converges to the true parameter as $n\to\infty$) and <strong>asymptotically efficient</strong> (achieves the smallest
  possible variance, the Cramér–Rao bound, in the large-data limit). Even more tellingly, maximizing likelihood is
  equivalent to <em>minimizing the KL divergence</em> from the true data distribution to the model (Track 12): MLE makes
  your model as close as possible, in an information-theoretic sense, to the process that generated the data. That's the
  deep reason cross-entropy is the right loss — it's the sample estimate of that KL, up to the data's own entropy.</p>
  <p>The flip side is MLE's Achilles' heel: with limited data and flexible models it <strong>overfits</strong>, because it
  trusts the data completely and has no opinion about what parameters are reasonable. A single unseen class gets
  probability zero; a separable dataset sends logistic-regression weights to infinity. The cure is to add a
  <strong>prior</strong> — a belief about plausible parameters — which turns MLE into MAP estimation and, as the next
  lesson shows, is exactly where $L_2$ and $L_1$ regularization come from. Maximum likelihood is the honest starting
  point; priors are how we inject the humility that finite data demands.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["11.3"] = {
  subtitle: "Maximum likelihood plus a prior belief — and out falls L2 (weight decay) and L1.",

  aiMoment: String.raw`<p>Add $\lambda\lVert\mathbf w\rVert_2^2$ to your loss — <strong>weight decay</strong> — and you're
  doing <strong>MAP estimation with a Gaussian prior</strong> on the weights. Use $\lambda\lVert\mathbf w\rVert_1$ instead
  and it's a Laplace prior (Lasso). Every regularizer you add is secretly a statement about what weights you expected
  before seeing data. MAP is where "regularization" and "Bayesian prior" become the same thing.</p>`,

  plainEnglish: String.raw`<p><strong>MAP</strong> (maximum a posteriori) is maximum likelihood with an added
  <strong>prior</strong>: a belief about which parameter values are plausible before you look at the data. It balances
  fitting the data against staying near what you expected — which is exactly what regularization does.</p>`,

  intuition: String.raw`<p>The <strong>likelihood</strong> pulls the estimate toward whatever best fits the data; the
  <strong>prior</strong> pulls it toward your prior belief (often zero). The <strong>posterior</strong> — and its peak,
  the MAP estimate — sits in a compromise between them.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Prior, likelihood, and posterior curves">
    <line x1="20" y1="120" x2="300" y2="120" stroke="#94a3b8"/>
    <path d="M40,120 C80,120 90,55 120,55 C150,55 160,120 200,120" fill="none" stroke="#0d9488" stroke-width="2"/>
    <path d="M140,120 C180,120 190,40 220,40 C250,40 260,120 300,120" fill="none" stroke="#4f46e5" stroke-width="2"/>
    <path d="M100,120 C150,120 160,72 175,72 C195,72 205,120 255,120" fill="none" stroke="#dc2626" stroke-width="2.4"/>
    <text x="44" y="48" font-size="10" fill="#0d9488" font-family="sans-serif">prior (near 0)</text>
    <text x="228" y="36" font-size="10" fill="#4f46e5" font-family="sans-serif">likelihood (data)</text>
    <text x="150" y="140" font-size="10" fill="#dc2626" font-family="sans-serif">posterior → MAP (compromise)</text>
  </svg>
  <figcaption>The MAP estimate is the posterior's peak — pulled between the prior and the data likelihood.</figcaption>
  </figure>`,

  formalism: String.raw`<p>By Bayes (Track 9.3), the posterior is $p(\theta\mid x)\propto p(x\mid\theta)\,p(\theta)$. The
  <strong>MAP estimate</strong> maximizes it:</p>
  $$\hat\theta_{\text{MAP}}=\arg\max_\theta\Big[\log p(x\mid\theta)+\log p(\theta)\Big]=\arg\min_\theta\Big[\text{NLL}(\theta)-\log p(\theta)\Big].$$
  <p>A <strong>Gaussian</strong> prior $\theta\sim\mathcal N(0,\tau^2)$ adds $\tfrac{1}{2\tau^2}\lVert\theta\rVert_2^2$
  ($L_2$ / weight decay); a <strong>Laplace</strong> prior adds $\tfrac1b\lVert\theta\rVert_1$ ($L_1$). As data grows,
  the likelihood dominates and MAP $\to$ MLE.</p>`,

  derivation: String.raw`<p><strong>A Gaussian prior gives exactly $L_2$ regularization.</strong></p>
  <p><strong>Step 1 — MAP objective:</strong> maximize $\log p(x\mid\theta)+\log p(\theta)$.</p>
  <p><strong>Step 2 — plug in a Gaussian prior</strong> $p(\theta)=\mathcal N(0,\tau^2 I)$:
  $\log p(\theta)=-\dfrac{1}{2\tau^2}\lVert\theta\rVert_2^2+\text{const}.$</p>
  <p><strong>Step 3 — combine and flip to a minimization</strong> (NLL $=-\log p(x\mid\theta)$):
  $\hat\theta_{\text{MAP}}=\arg\min_\theta\Big[\text{NLL}(\theta)+\underbrace{\tfrac{1}{2\tau^2}}_{\lambda}\lVert\theta\rVert_2^2\Big].$</p>
  <p><strong>Step 4 — read it off:</strong> that's the regularized loss with $\lambda=\tfrac{1}{2\tau^2}$. $\blacksquare$
  A tighter prior (small $\tau$) means a bigger $\lambda$ — stronger shrinkage. A <strong>Laplace</strong> prior
  $p(\theta)\propto e^{-\lVert\theta\rVert_1/b}$ gives the $L_1$ penalty the same way, which is why $L_1$ yields sparsity
  (Track 4.1).</p>`,

  code: [
    { label: "MAP = MLE + prior: ridge shrinks the estimate", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

# estimate a mean from few noisy samples; Gaussian prior N(0, tau²) pulls toward 0
x = rng.normal(4.0, 3.0, size=6)
n, sigma2 = len(x), 9.0
for tau2 in [np.inf, 4.0, 1.0]:               # prior variance: ∞ = no prior (MLE)
    lam = 0 if np.isinf(tau2) else sigma2/(n*tau2)
    map_est = x.mean() / (1 + lam)            # MAP for a mean with Gaussian prior
    print(f"prior var={tau2!s:>4}:  estimate = {map_est:6.3f}",
          "(MLE)" if np.isinf(tau2) else "(shrunk toward 0)")
` },
    { label: "As data grows, MAP → MLE", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)
tau2, sigma2 = 1.0, 9.0
for n in [3, 30, 3000]:
    x = rng.normal(4.0, 3.0, size=n)
    lam = sigma2/(n*tau2)
    print(f"n={n:5d}: MAP={x.mean()/(1+lam):.3f}  MLE={x.mean():.3f}  (gap shrinks)")
` }
  ],

  keyPoints: [
    "MAP maximizes the posterior: $\\log p(x\\mid\\theta)+\\log p(\\theta)$ (likelihood + prior).",
    "Gaussian prior ⇒ $L_2$ penalty (weight decay); Laplace prior ⇒ $L_1$ penalty (sparsity).",
    "The regularization strength $\\lambda$ is set by the prior's tightness ($\\lambda=1/2\\tau^2$).",
    "As data grows, the likelihood dominates and MAP converges to MLE.",
    "MLE vs MAP vs full Bayesian: point estimate, point estimate with prior, or the whole posterior."
  ],

  commonMistakes: [
    { wrong: "Thinking regularization is an ad-hoc trick.", why: "Weight decay <em>is</em> MAP with a Gaussian prior — a principled probabilistic statement, not a hack. The $\\lambda$ you tune is a prior variance in disguise." },
    { wrong: "Expecting a strong prior to matter with huge data.", why: "The likelihood grows with $n$ while the prior stays fixed, so MAP $\\to$ MLE. Priors matter most when data is scarce (few-shot, rare classes)." },
    { wrong: "Confusing MAP with the full Bayesian posterior.", why: "MAP is a single point (the posterior's mode); Bayesian inference keeps the whole distribution to quantify uncertainty. MAP can be a poor summary of a skewed posterior." }
  ],

  quiz: [
    { q: "Adding $\\lambda\\lVert\\mathbf w\\rVert_2^2$ to a loss corresponds to a MAP estimate with a…", options: ["Gaussian prior", "Laplace prior", "uniform prior", "no prior"], answer: 0,
      explain: "A Gaussian prior contributes a squared-$L_2$ term; its precision is $\\lambda$." },
    { q: "An $L_1$ penalty corresponds to which prior?", options: ["Laplace", "Gaussian", "uniform", "exponential"], answer: 0,
      explain: "The Laplace prior $\\propto e^{-|\\theta|/b}$ gives the $L_1$ term and induces sparsity." },
    { q: "As the dataset grows, MAP…", options: ["approaches MLE", "approaches the prior mean", "diverges", "equals the prior"], answer: 0,
      explain: "The likelihood dominates the fixed prior, so MAP $\\to$ MLE." },
    { q: "A uniform (flat) prior makes MAP equal to…", options: ["MLE", "the prior mean", "zero", "the posterior variance"], answer: 0,
      explain: "A flat prior adds a constant, so maximizing the posterior is just maximizing the likelihood." },
    { q: "Stronger weight decay (larger $\\lambda$) corresponds to a prior that is…", options: ["tighter around 0 (smaller $\\tau$)", "wider", "uniform", "centered away from 0"], answer: 0,
      explain: "$\\lambda=1/2\\tau^2$, so bigger $\\lambda$ means smaller prior variance $\\tau^2$ — more confident the weights are near 0." }
  ],

  practice: [
    { level: "easy", prompt: "What prior does ordinary (unregularized) MLE correspond to?", solution: "A flat/uniform prior — every parameter value is a priori equally plausible, so only the likelihood matters." },
    { level: "med", prompt: "Write the MAP objective for logistic regression with a Gaussian prior on weights.", solution: "$\\hat{\\mathbf w}=\\arg\\min_{\\mathbf w}\\sum_i\\log(1+e^{-y_i\\mathbf w^\\top\\mathbf x_i})+\\lambda\\lVert\\mathbf w\\rVert_2^2$ — cross-entropy (the NLL) plus an $L_2$ term from the Gaussian prior. It also fixes MLE's blow-up on separable data (weights no longer run to infinity)." },
    { level: "med", prompt: "Why does $L_1$ give sparse solutions but $L_2$ does not?", solution: "The Laplace prior has a sharp peak at 0 (heavy mass exactly at zero-ish), so its penalty $\\propto|\\theta|$ has a constant gradient that pushes small weights to exactly 0 (Track 4.1). The Gaussian prior's penalty $\\propto\\theta^2$ shrinks proportionally and never reaches 0. The prior shape dictates the sparsity." },
    { level: "hard", prompt: "AI task: you have 5 training examples for a rare class. Explain, via MAP, why a prior helps and what happens with millions of examples.", solution: "With only 5 examples the likelihood is weak and noisy, so MLE overfits (e.g. assigns extreme probabilities). A prior (weight decay, or a smoothing pseudo-count / Dirichlet prior on class probabilities) pulls the estimate toward reasonable defaults, drastically cutting variance at the cost of a little bias — usually a big MSE win in this low-data regime. With millions of examples the likelihood overwhelms the fixed prior, so MAP $\\to$ MLE and the regularization's effect fades — which is why large-data training uses light regularization while few-shot settings lean on strong priors." }
  ],

  deepDive: String.raw`<p><strong>MLE vs MAP vs Bayesian: three answers to 'what are the parameters?'</strong></p>
  <p>These form a ladder of how much you model your uncertainty. <strong>MLE</strong> returns a single best-fit point and
  no prior — maximally data-driven, prone to overfitting. <strong>MAP</strong> returns a single point but regularized by
  a prior — the workhorse of practical deep learning, since every weight-decayed network is a MAP estimate.
  <strong>Full Bayesian inference</strong> refuses to collapse to a point at all: it keeps the entire posterior
  $p(\theta\mid x)$ and predicts by averaging over it, $p(y\mid x)=\int p(y\mid x,\theta)\,p(\theta\mid x)\,d\theta$. That
  integral is what lets a model say "I don't know" in unfamiliar regions — genuine uncertainty, not just a confident
  point guess.</p>
  <p>The catch is tractability. MLE and MAP are optimizations (find one $\theta$); Bayesian inference is an integration
  over all $\theta$, which for neural networks is intractable and drives the approximations of Track 12 — variational
  inference, Monte-Carlo dropout, deep ensembles. So the ladder is also a cost curve: more honesty about uncertainty
  costs more computation. Knowing where you are on it tells you what your model's outputs actually mean — a calibrated
  probability, a regularized point estimate, or just the mode of a likelihood — and that distinction matters enormously
  the moment a model faces data unlike its training set.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["11.4"] = {
  subtitle: "Too simple misses the pattern; too complex chases the noise. Total error is a U.",

  aiMoment: String.raw`<p>Every choice of model size, regularization strength, and training time is navigating the
  <strong>bias–variance tradeoff</strong>. Underfitting (high bias) and overfitting (high variance) are its two failure
  modes, and the classic U-shaped test-error curve is the map. It's the single most useful mental model for "why is my
  model not generalizing?"</p>`,

  plainEnglish: String.raw`<p>A model that's too simple can't capture the real pattern — it's systematically wrong
  (<strong>high bias</strong>). A model that's too flexible fits the training noise and changes wildly with the data
  (<strong>high variance</strong>). The best generalization sits in between, where the total error is lowest.</p>`,

  intuition: String.raw`<p>As you increase model complexity, bias falls (the model can fit more) but variance rises (it
  fits noise). Their sum — the expected test error — first drops, then climbs: a U. The bottom of the U is the sweet
  spot.</p>
  <figure class="figure">
  <svg viewBox="0 0 300 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="U-shaped test error from bias plus variance">
    <line x1="40" y1="150" x2="285" y2="150" stroke="#94a3b8"/>
    <line x1="40" y1="150" x2="40" y2="20" stroke="#94a3b8"/>
    <path d="M50,40 C110,120 150,150 260,158" fill="none" stroke="#0d9488" stroke-width="2"/>
    <path d="M50,158 C150,150 190,120 260,40" fill="none" stroke="#dc2626" stroke-width="2"/>
    <path d="M50,80 C120,140 180,140 260,80" fill="none" stroke="#4f46e5" stroke-width="2.6"/>
    <circle cx="150" cy="128" r="3.5" fill="#4f46e5"/>
    <text x="52" y="34" font-size="10" fill="#0d9488" font-family="sans-serif">bias²</text>
    <text x="238" y="34" font-size="10" fill="#dc2626" font-family="sans-serif">variance</text>
    <text x="120" y="96" font-size="10" fill="#4f46e5" font-family="sans-serif">total error</text>
    <text x="120" y="172" font-size="10" fill="#64748b" font-family="sans-serif">model complexity →</text>
  </svg>
  <figcaption>Bias falls and variance rises with complexity; their sum is the U-shaped test error.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For a target $y=f(x)+\varepsilon$ with noise $\varepsilon$ (mean 0, variance $\sigma^2$), the
  expected squared test error of a fitted $\hat f$ (averaged over datasets) decomposes as</p>
  $$E\big[(y-\hat f(x))^2\big]=\underbrace{\big(f(x)-E[\hat f(x)]\big)^2}_{\text{bias}^2}+\underbrace{E\big[(\hat f(x)-E[\hat f(x)])^2\big]}_{\text{variance}}+\underbrace{\sigma^2}_{\text{irreducible}}.$$
  <p>The noise $\sigma^2$ is a floor no model can beat. Complexity trades bias for variance.</p>`,

  derivation: String.raw`<p><strong>Deriving the decomposition.</strong> Fix $x$; expectations are over datasets and the
  noise $\varepsilon$. Let $\bar f=E[\hat f(x)]$.</p>
  <p><strong>Step 1 — insert the noise model:</strong> $E[(y-\hat f)^2]=E[(f+\varepsilon-\hat f)^2]$, and since
  $\varepsilon$ is independent with mean 0, it splits off: $=E[(f-\hat f)^2]+\sigma^2.$</p>
  <p><strong>Step 2 — add and subtract $\bar f$:</strong> $E[(f-\hat f)^2]=E[((f-\bar f)+(\bar f-\hat f))^2].$</p>
  <p><strong>Step 3 — expand; the cross term dies</strong> because $E[\bar f-\hat f]=0$:
  $=(f-\bar f)^2+E[(\hat f-\bar f)^2].$</p>
  <p><strong>Step 4 — collect:</strong> $E[(y-\hat f)^2]=\underbrace{(f-\bar f)^2}_{\text{bias}^2}+\underbrace{E[(\hat f-\bar f)^2]}_{\text{variance}}+\sigma^2.\;\blacksquare$
  Plain English: error = how wrong on average + how much you wobble + unavoidable noise.</p>`,

  code: [
    { label: "Polynomial fits: the U-curve in action", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

f = lambda x: np.sin(2*np.pi*x)                 # true function
def experiment(deg, n=20):
    x = rng.uniform(0,1,n); y = f(x) + rng.normal(0,0.3,n)
    xt = np.linspace(0,1,200); yt = f(xt)
    c = np.polyfit(x, y, deg)                    # fit polynomial of degree deg
    train = np.mean((np.polyval(c,x)-y)**2)
    test  = np.mean((np.polyval(c,xt)-yt)**2)
    return train, test
for deg in [1, 3, 5, 9, 15]:
    tr, te = np.mean([experiment(deg) for _ in range(200)], axis=0)
    print(f"degree {deg:2d}: train={tr:.3f}  test={te:.3f}")
# test error dips then rises: underfit (low deg) -> sweet spot -> overfit (high deg)
` }
  ],

  keyPoints: [
    "Test error $=$ bias² $+$ variance $+$ irreducible noise $\\sigma^2$.",
    "Underfitting = high bias (too simple); overfitting = high variance (too flexible).",
    "Increasing complexity lowers bias but raises variance — a U-shaped test curve.",
    "Regularization, more data, and ensembling reduce variance; more capacity reduces bias.",
    "The noise floor $\\sigma^2$ can't be beaten by any model."
  ],

  commonMistakes: [
    { wrong: "Reading a low training error as good generalization.", why: "Training error mostly reflects bias; overfit models have near-zero training error but high <em>test</em> error from variance. Always judge on held-out data." },
    { wrong: "Believing bigger is always worse (overfitting).", why: "The classic U isn't the whole story: very over-parameterized models can enter a 'double descent' regime where test error falls again — modern deep nets often live there." },
    { wrong: "Trying to drive test error to zero.", why: "The irreducible noise $\\sigma^2$ is a hard floor. Chasing below it means you're fitting noise (variance), which hurts generalization." }
  ],

  quiz: [
    { q: "A model with high training error and high test error is likely…", options: ["underfitting (high bias)", "overfitting (high variance)", "perfect", "at the noise floor"], answer: 0,
      explain: "High error everywhere means the model is too simple to capture the pattern — bias." },
    { q: "Low training error but high test error indicates…", options: ["overfitting (high variance)", "underfitting", "high bias", "low variance"], answer: 0,
      explain: "It fit the training data (incl. noise) but doesn't generalize — variance." },
    { q: "Test error $=$ bias² $+$ variance $+$ …", options: ["irreducible noise $\\sigma^2$", "0", "the training error", "the bias"], answer: 0,
      explain: "The noise variance is the third, unavoidable term." },
    { q: "Increasing model complexity generally…", options: ["lowers bias, raises variance", "lowers both", "raises both", "raises bias, lowers variance"], answer: 0,
      explain: "More flexibility fits more (less bias) but chases noise (more variance)." },
    { q: "Adding more training data primarily reduces…", options: ["variance", "bias", "the noise floor", "the true function"], answer: 0,
      explain: "More data stabilizes the fit across samples (less variance); it doesn't change a model's inherent bias or the noise floor." }
  ],

  practice: [
    { level: "easy", prompt: "Name the bias/variance character of (a) a constant predictor and (b) a lookup table that memorizes training data.", solution: "(a) High bias, low variance — it ignores inputs, systematically wrong but stable. (b) Low bias, high variance — it fits training data perfectly but changes entirely with new data and fails to generalize." },
    { level: "med", prompt: "You observe train error 0.02 and test error 0.40. Diagnose and give two fixes.", solution: "Large gap ⇒ overfitting (high variance). Fixes: reduce capacity or add regularization (stronger weight decay, dropout), and/or get more data; early stopping also helps." },
    { level: "med", prompt: "Why does the noise term $\\sigma^2$ set a floor on test error?", solution: "$y=f(x)+\\varepsilon$ contains randomness independent of $x$; even the true $f$ can't predict $\\varepsilon$. So $E[(y-f(x))^2]=\\sigma^2>0$ — no model, however good, can do better on average." },
    { level: "hard", prompt: "AI task: modern huge networks often generalize well despite fitting training data exactly. Reconcile this with the classic U-curve.", solution: "The classic U assumes increasing complexity monotonically raises variance. Empirically, past the interpolation threshold (enough parameters to fit the data exactly), test error can <em>descend a second time</em> — 'double descent'. Over-parameterization plus implicit regularization from SGD selects flat, low-norm solutions among the many that fit, keeping effective variance low. So the U is the small/medium-model story; very large models live on the far side where more parameters help again. The bias–variance <em>concept</em> still holds, but 'complexity' must be measured by effective capacity (norm/flatness), not raw parameter count." }
  ],

  deepDive: String.raw`<p><strong>Double descent: where the textbook U-curve meets deep learning.</strong></p>
  <p>For decades the bias–variance U was the whole story: add capacity, watch test error dip then rise, stop at the
  bottom. Deep learning broke the picture. As models grow past the <strong>interpolation threshold</strong> — the point
  where they have just enough parameters to fit the training set exactly — test error, instead of exploding, often
  <em>peaks and then descends again</em>. Plotted against model size, the curve looks like the classic U followed by a
  second downward slope: <strong>double descent</strong>. Modern networks deliberately operate far to the right, hugely
  over-parameterized, and generalize well.</p>
  <p>The resolution is that raw parameter count is the wrong axis. Among the infinitely many parameter settings that
  interpolate the data, <strong>SGD's implicit bias</strong> prefers low-norm, flat-minimum solutions (Track 8), which
  behave like a strongly-regularized — hence low-variance — model. So the honest statement is: test error still equals
  bias² + variance + noise, but a model's <em>effective</em> complexity is set by the norm and geometry of the solution
  the optimizer finds, not by how many weights it has. This is why over-parameterization is safe in practice, why weight
  decay and early stopping still help (they nudge toward lower-norm solutions), and why "just make it bigger" has been a
  winning strategy. The bias–variance tradeoff remains the right frame — you just have to measure complexity by what the
  optimizer actually uses.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["11.5"] = {
  subtitle: "Is that improvement real or just noise? p-values, confidence intervals, and resampling.",

  aiMoment: String.raw`<p>Model B scores 91% and model A scores 90% — is B actually better, or did it get lucky on
  this test set? That's a <strong>hypothesis test</strong>. Reporting a metric with a <strong>confidence interval</strong>
  instead of a single number, and using the <strong>bootstrap</strong> to get that interval without distributional
  assumptions, is how you make honest, reproducible claims about model performance and A/B experiments.</p>`,

  plainEnglish: String.raw`<p>A <strong>hypothesis test</strong> asks whether an observed effect is bigger than what
  random chance would produce. The <strong>p-value</strong> is the probability of seeing a result at least this extreme
  if there were truly no effect. A <strong>confidence interval</strong> is a plausible range for the quantity. The
  <strong>bootstrap</strong> estimates that uncertainty by resampling your own data.</p>`,

  intuition: String.raw`<p>Imagine the world where there's no real effect (the "null"). Simulate the statistic many
  times under it — that's the null distribution. If your observed value lands way out in its tail, chance alone rarely
  produces it, so the effect is probably real.</p>
  <figure class="figure">
  <svg viewBox="0 0 300 160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Null distribution with observed statistic in the tail">
    <line x1="20" y1="120" x2="285" y2="120" stroke="#94a3b8"/>
    <path d="M40,120 C90,120 100,40 150,40 C200,40 210,120 260,120" fill="none" stroke="#4f46e5" stroke-width="2"/>
    <path d="M222,120 C232,108 240,96 246,80 C250,100 254,112 260,120 Z" fill="#fdd" stroke="#dc2626"/>
    <line x1="228" y1="120" x2="228" y2="52" stroke="#dc2626" stroke-dasharray="3 3"/>
    <text x="120" y="30" font-size="10" fill="#4f46e5" font-family="sans-serif">null distribution (no effect)</text>
    <text x="196" y="140" font-size="10" fill="#dc2626" font-family="sans-serif">observed → p-value = tail area</text>
  </svg>
  <figcaption>The p-value is the tail probability of a result this extreme under the null.</figcaption>
  </figure>`,

  formalism: String.raw`<p>State a <strong>null hypothesis</strong> $H_0$ (no effect) and an <strong>alternative</strong> $H_1$.
  Compute a <strong>test statistic</strong> from the data; the <strong>p-value</strong> is $P(\text{statistic at least this extreme}\mid H_0)$.
  Reject $H_0$ if $p<\alpha$ (e.g. $\alpha=0.05$). A <strong>confidence interval</strong> is a range that would contain
  the true value in $(1-\alpha)$ of repeated experiments. The <strong>bootstrap</strong> resamples the data with
  replacement $B$ times and uses the spread of the recomputed statistic as its sampling distribution.</p>`,

  derivation: String.raw`<p><strong>The bootstrap, and why it works.</strong> You have one dataset but want the sampling
  distribution of a statistic $\hat\theta$ (say, accuracy).</p>
  <p><strong>Step 1 — resample.</strong> Draw a new dataset of the same size $n$ by sampling your data <em>with
  replacement</em>. Compute $\hat\theta^*_1$.</p>
  <p><strong>Step 2 — repeat</strong> $B$ times (e.g. $B=10{,}000$), getting $\hat\theta^*_1,\dots,\hat\theta^*_B$.</p>
  <p><strong>Step 3 — read off uncertainty.</strong> The standard deviation of the $\hat\theta^*_b$ estimates the
  standard error; the 2.5th and 97.5th percentiles give a 95% confidence interval.</p>
  <p><strong>Why it works:</strong> the empirical data distribution is our best estimate of the true one, so resampling
  from it mimics drawing fresh datasets — approximating the sampling variability we can't observe directly. $\blacksquare$
  Plain English: shake up your own data many times to see how much the answer would wobble.</p>`,

  code: [
    { label: "Bootstrap confidence interval for a mean", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

data = rng.normal(5.0, 2.0, size=60)             # one sample
boot = np.array([rng.choice(data, size=len(data), replace=True).mean()
                 for _ in range(10000)])          # resample & recompute
lo, hi = np.percentile(boot, [2.5, 97.5])
print("sample mean       =", round(float(data.mean()), 3))
print("bootstrap 95% CI  = [", round(float(lo),3), ",", round(float(hi),3), "]")
print("bootstrap std err =", round(float(boot.std()), 3), " vs σ/√n =", round(2/np.sqrt(60),3))
` },
    { label: "Is model B really better? A bootstrap test", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

# per-example correctness (1/0) for two models on the same 500-example test set
A = (rng.random(500) < 0.90).astype(int)
B = (rng.random(500) < 0.91).astype(int)
diffs = []
n = len(A)
for _ in range(10000):
    idx = rng.integers(0, n, n)                  # paired resample
    diffs.append(B[idx].mean() - A[idx].mean())
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print("acc(B)-acc(A)      =", round(float(B.mean()-A.mean()), 4))
print("95% CI for the gap = [", round(float(lo),4), ",", round(float(hi),4), "]")
print("gap significant?    ", not (lo <= 0 <= hi))   # CI excludes 0?
` }
  ],

  keyPoints: [
    "A p-value is $P(\\text{data this extreme}\\mid H_0)$ — NOT the probability the null is true.",
    "Reject $H_0$ when $p<\\alpha$; $\\alpha$ is your tolerated false-positive rate.",
    "A confidence interval is a plausible range; report metrics with one, not as bare numbers.",
    "The bootstrap resamples your data with replacement to estimate uncertainty assumption-free.",
    "If a 95% CI for a difference excludes 0, the difference is significant at ~5%."
  ],

  commonMistakes: [
    { wrong: "Reading the p-value as $P(H_0\\text{ is true})$.", why: "It's the probability of the data given $H_0$, not the probability of $H_0$ given the data. A small p-value means 'surprising under the null', not '5% chance the null holds'." },
    { wrong: "Running many comparisons and reporting the best p-value.", why: "Testing 20 things at $\\alpha=0.05$ yields ~1 false positive by chance. Correct for multiple comparisons (Bonferroni, FDR) or you'll 'discover' noise." },
    { wrong: "Comparing two models' accuracies without accounting for test-set variance.", why: "A 1% gap on a small test set is often within noise. Use a paired bootstrap / test and report a CI; otherwise you'll chase illusory improvements." }
  ],

  quiz: [
    { q: "A p-value of 0.03 means…", options: ["a 3% chance of data this extreme if the null were true", "a 3% chance the null is true", "the effect is 97% real", "the alternative is proven"], answer: 0,
      explain: "p-value = P(data at least this extreme | H₀), a statement about the data under the null, not about H₀'s truth." },
    { q: "The bootstrap estimates uncertainty by…", options: ["resampling the data with replacement", "assuming a Gaussian", "removing outliers", "increasing the sample size"], answer: 0,
      explain: "It treats the sample as the population and resamples to mimic sampling variability." },
    { q: "A 95% CI for a model-accuracy difference is $[-0.01, 0.03]$. You conclude…", options: ["the difference is not significant (CI includes 0)", "B is significantly better", "A is significantly better", "the p-value is 0"], answer: 0,
      explain: "Since 0 is inside the interval, you can't rule out no difference at the 5% level." },
    { q: "Testing 20 independent hypotheses at $\\alpha=0.05$, the expected number of false positives is about…", options: ["1", "0", "5", "20"], answer: 0,
      explain: "$20\\times0.05=1$ — why multiple-comparison correction matters." },
    { q: "A confidence interval is best described as…", options: ["a range that covers the true value in (1−α) of repeated experiments", "the range containing 95% of the data", "where the mean must be", "the model's error bar on one point"], answer: 0,
      explain: "It's a statement about the procedure's long-run coverage of the true parameter." }
  ],

  practice: [
    { level: "easy", prompt: "Your bootstrap gives estimates with mean 0.82 and std 0.03. Give an approximate 95% CI.", solution: "Roughly mean ± 2·std $=0.82\\pm0.06=[0.76,0.88]$ (or use the 2.5/97.5 percentiles of the bootstrap distribution directly)." },
    { level: "med", prompt: "Explain why a paired (same test examples) comparison of two models is more powerful than comparing independent runs.", solution: "Pairing cancels the per-example difficulty variance: examples that are hard for both models contribute similarly, so you measure the <em>difference</em> directly with much lower variance than comparing two separately-sampled accuracies. This makes real gaps easier to detect (higher statistical power) for the same test-set size." },
    { level: "med", prompt: "A model's accuracy on 100 test examples is 88%. Roughly, what's the standard error?", solution: "For a proportion, $\\text{SE}=\\sqrt{p(1-p)/n}=\\sqrt{0.88\\cdot0.12/100}\\approx\\sqrt{0.001056}\\approx0.0325$, so ~±3.2% — a 95% CI of about $[81.6\\%, 94.4\\%]$. Small test sets have wide error bars." },
    { level: "hard", prompt: "AI task: you fine-tune a model and it beats the baseline by 0.4% on a 1,000-example benchmark. Design a check for whether to believe it.", solution: "Compute a per-example paired difference and bootstrap it: resample the 1,000 (example-level) results with replacement ~10,000 times, recompute the accuracy gap each time, and form a 95% CI for the gap. If the CI comfortably excludes 0, the improvement is likely real; if it straddles 0 (very possible for 0.4% on n=1,000, where SE is ~1%), the result is within noise. Also test on multiple seeds/splits and correct for the number of variants you tried — a 0.4% gap 'discovered' after many attempts is exactly the multiple-comparisons trap. Report the CI, not just the point gap." }
  ],

  deepDive: String.raw`<p><strong>The replication crisis, and what it teaches ML.</strong></p>
  <p>Across science, a wave of published results failed to replicate — and the statistical culprits are precisely the
  mistakes above. <strong>p-hacking</strong>: trying many analyses and reporting the one that crossed $p<0.05$.
  <strong>Multiple comparisons</strong>: testing dozens of hypotheses and celebrating the inevitable false positive.
  <strong>Ignoring effect size and uncertainty</strong>: a 'significant' result on a huge sample can be trivially small,
  while a bare point metric hides enormous variance. These aren't esoteric — they're everyday hazards in ML
  benchmarking, where leaderboards, many model variants, and small test sets create the same pressures.</p>
  <p>The defenses are cultural as much as technical: pre-register or fix your evaluation before peeking; report
  <strong>confidence intervals and multiple seeds</strong>, not single numbers; correct for the number of things you
  tried; and prefer effect sizes ("+0.4% ± 1.1%") over bare significance. A model that's "state of the art by 0.2%" on
  one split with one seed is often statistical noise dressed as progress. Treating evaluation with the same rigor as
  training — quantifying uncertainty, resisting the temptation to cherry-pick — is what separates a reproducible result
  from a lucky one, and it's the quiet skill that makes ML claims trustworthy. This closes the statistics track: you can
  now not only <em>fit</em> models (MLE/MAP) but honestly <em>judge</em> them.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["11.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 65 minutes.",

  intro: String.raw`<p>This exam spans all of Track 11: estimators and bias/variance, MLE, MAP and priors, the
  bias–variance tradeoff, and hypothesis testing / CIs / bootstrap. <strong>Give yourself 65 minutes</strong>, produce
  each answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "An estimator has $E[\\hat\\theta]=6$, true $\\theta=5$, variance 4. Give the bias and the MSE.",
      solution: "Bias $=6-5=1$; MSE $=\\text{bias}^2+\\text{Var}=1+4=5$." },
    { level: "easy", prompt: "Observe 15 heads in 25 flips. Give the Bernoulli MLE.",
      solution: "$\\hat p=k/n=15/25=0.6$." },
    { level: "med", prompt: "Show minimizing cross-entropy equals maximum likelihood for a classifier.",
      solution: "The dataset likelihood is $\\prod_i q_{i,y_i}$ (predicted prob of each true label). MLE maximizes $\\sum_i\\log q_{i,y_i}$, i.e. minimizes $-\\sum_i\\log q_{i,y_i}=\\sum_i(-\\sum_c y_{ic}\\log q_{ic})$ — the total cross-entropy. Same objective." },
    { level: "med", prompt: "What regularizer corresponds to a MAP estimate with a Gaussian prior on the weights, and what sets its strength?",
      solution: "An $L_2$ penalty $\\lambda\\lVert\\mathbf w\\rVert_2^2$ (weight decay). The strength is $\\lambda=1/2\\tau^2$ where $\\tau^2$ is the prior variance — a tighter prior (small $\\tau$) means larger $\\lambda$." },
    { level: "med", prompt: "Train error 0.01, test error 0.35. Diagnose and give two remedies.",
      solution: "Big gap ⇒ overfitting (high variance). Remedies: add regularization (weight decay / dropout) or reduce capacity; also more data and early stopping." },
    { level: "med", prompt: "Decompose expected test error into its three parts and say which one more data reduces.",
      solution: "Error $=$ bias² $+$ variance $+$ irreducible noise $\\sigma^2$. More data primarily reduces <em>variance</em> (stabilizes the fit); it doesn't change model bias or the noise floor." },
    { level: "hard", prompt: "Derive the Gaussian-mean MLE and explain its connection to least squares.",
      solution: "Log-likelihood $\\ell(\\mu)=-\\tfrac{1}{2\\sigma^2}\\sum(x_i-\\mu)^2+\\text{const}$. $\\frac{d\\ell}{d\\mu}=\\tfrac{1}{\\sigma^2}\\sum(x_i-\\mu)=0\\Rightarrow\\hat\\mu=\\bar x$. Maximizing this log-likelihood is minimizing $\\sum(x_i-\\mu)^2$, so Gaussian MLE = least squares — the squared loss is the Gaussian log-density." },
    { level: "hard", prompt: "A p-value is 0.02. State precisely what it means and one thing it does NOT mean.",
      solution: "It means: if the null hypothesis were true, there's a 2% probability of observing data at least as extreme as this. It does NOT mean there's a 2% probability the null is true (that would require a prior and Bayes), nor that the effect is 98% real or important." },
    { level: "hard", prompt: "You compare two models on a 400-example test set; B beats A by 1.5%. Outline a bootstrap check.",
      solution: "Take the per-example correctness of both models on the same 400 examples. Bootstrap: resample the 400 indices with replacement ~10,000 times, recompute acc(B)−acc(A) each time, and take the 2.5/97.5 percentiles for a 95% CI on the gap. If the CI excludes 0, the 1.5% gap is likely real; if it includes 0, it's within noise (SE for a proportion on n=400 is ~2.5%, so 1.5% is borderline). Also test multiple seeds and correct for how many variants you tried." },
    { level: "hard", prompt: "AI task: explain the MLE→MAP→Bayesian ladder in terms of what each returns and when you'd want each.",
      solution: "MLE returns a single best-fit parameter with no prior — use with abundant data and simple models, accept overfitting risk. MAP returns a single regularized point (MLE + prior) — the practical default (every weight-decayed net is MAP); use when you want regularization and a point prediction. Full Bayesian keeps the whole posterior and predicts by integrating over it — use when calibrated uncertainty matters (safety, active learning, small data), at higher compute cost. The ladder trades computation for honesty about uncertainty; as data grows, MAP→MLE and the prior's influence fades." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> You can fit models (MLE/MAP) and judge them (CIs/tests). On to Information Theory (Track 12) and the probability capstone.</li>
    <li><strong>7–8:</strong> Strong. Revisit the MAP↔regularization derivation or the p-value definition if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive MSE = bias² + variance and the Gaussian-MLE = least-squares link; redo Lessons 11.1 and 11.2.</li>
    <li><strong>Below 5:</strong> Rework the track — estimation and evaluation underpin how every model is trained and compared.</li>
  </ul>`
};
