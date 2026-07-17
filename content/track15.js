/* ============================================================
   TRACK 15 — Bayesian Inference & Probabilistic ML
   Exact inference (conjugacy → Bayesian regression), then
   approximate inference (MCMC, variational), then nonparametric
   (Gaussian processes). Every result checked numerically.
   ============================================================ */
window.LESSON_CONTENT ||= {};

/* ------------------------------------------------------------------ 15.1 */
window.LESSON_CONTENT["15.1"] = {
  subtitle: "Start with a belief, see some data, update — and when the prior and likelihood are a matched (conjugate) pair, that update is a one-line formula.",

  aiMoment: String.raw`Your product ships an A/B test: version B got 8 clicks out of 10 impressions. Is B really better, or did it get lucky? A point estimate ("80%!") throws away exactly the thing you need — <em>how sure are we?</em> Bayesian inference keeps a full distribution over the click-through rate and sharpens it as data arrives. That same machinery powers Thompson sampling in bandits, uncertainty estimates in Bayesian neural nets, and every "we're 95% confident" claim. This lesson builds the cleanest case: <strong>conjugate priors</strong>, where updating your belief is pure arithmetic.`,

  plainEnglish: String.raw`You hold a belief about some unknown (say, a coin's bias) as a <em>distribution</em>, not a single number. Then you see data. Bayes' rule tells you how to combine the two into an updated belief — the <strong>posterior</strong> — which lands somewhere between what you believed and what the data says. See more data and the posterior leans further toward the data and gets narrower (more certain).`,

  intuition: String.raw`Represent "how likely each bias $\theta$ is" with a Beta distribution — a bump on $[0,1]$. Flip a coin a few times: each head nudges the bump toward 1, each tail toward 0. Remarkably, after any number of flips the bump is <em>still</em> a Beta — just with updated parameters. That "same family in, same family out" property is <strong>conjugacy</strong>, and it turns Bayes' rule from an integral into "add the heads to one parameter, the tails to the other." Early on the prior matters; pile on data and it washes out.`,

  formalism: String.raw`Bayes' rule for a parameter $\theta$ given data $\mathcal D$:
$$p(\theta\mid\mathcal D)=\frac{p(\mathcal D\mid\theta)\,p(\theta)}{p(\mathcal D)},\qquad p(\mathcal D)=\int p(\mathcal D\mid\theta)\,p(\theta)\,d\theta.$$
For a coin with a $\mathrm{Beta}(a,b)$ prior and $k$ heads in $n$ flips (a Binomial likelihood), the posterior is again Beta:
$$p(\theta)=\mathrm{Beta}(a,b)\ \xrightarrow{\ k\text{ heads},\ n-k\text{ tails}\ }\ p(\theta\mid\mathcal D)=\mathrm{Beta}(a+k,\ b+n-k).$$
Its mean, $\dfrac{a+k}{a+b+n}$, is the empirical frequency $k/n$ smoothed by $a{+}b$ "pseudo-flips" of prior — MAP/MLE from the Bayes capstone, now with the whole distribution attached.`,

  derivation: String.raw`<strong>Why Binomial × Beta stays Beta.</strong>
<ol>
<li><strong>Write the two pieces.</strong> Likelihood (Binomial in $\theta$): $p(\mathcal D\mid\theta)\propto\theta^{k}(1-\theta)^{n-k}$. Prior (Beta): $p(\theta)\propto\theta^{a-1}(1-\theta)^{b-1}$.</li>
<li><strong>Multiply (Bayes' numerator).</strong> $p(\theta\mid\mathcal D)\propto\theta^{k}(1-\theta)^{n-k}\cdot\theta^{a-1}(1-\theta)^{b-1}=\theta^{(a+k)-1}(1-\theta)^{(b+n-k)-1}.$ The exponents just <em>add</em>.</li>
<li><strong>Recognize the family.</strong> That is exactly the kernel of a $\mathrm{Beta}(a+k,\ b+n-k)$. The normalizing constant is whatever makes it integrate to 1 — and for a Beta we know it in closed form, so we never compute the scary integral $p(\mathcal D)$.</li>
<li><strong>Read the mean.</strong> $\mathbb E[\theta\mid\mathcal D]=\dfrac{a+k}{(a+k)+(b+n-k)}=\dfrac{a+k}{a+b+n}$. As $n\to\infty$ this $\to k/n$ (the data wins); with $n=0$ it's $a/(a+b)$ (the prior). The prior behaves like having already seen $a$ heads and $b$ tails.</li>
</ol>`,

  code: [
    { label: "One-line Bayesian updating with a conjugate Beta prior",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt

# Beta is conjugate to coin-flip (Bernoulli/Binomial) data: the posterior stays Beta.
def beta_pdf(x, a, b):
    from math import lgamma
    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    return np.exp((a-1)*np.log(x) + (b-1)*np.log(1-x) - logB)

a0, b0 = 2, 2                  # prior Beta(2,2): "probably fairish"
k, n   = 8, 10                 # data: 8 heads in 10 flips
aN, bN = a0 + k, b0 + (n - k)  # THE UPDATE: add heads to a, tails to b

print(f"prior mean     = {a0/(a0+b0):.3f}")
print(f"data frequency = {k/n:.3f}")
print(f"posterior Beta({aN},{bN}), mean = {aN/(aN+bN):.3f}   (between prior and data)")

x = np.linspace(1e-3, 1-1e-3, 400)
plt.figure(figsize=(6.4, 3.8))
plt.plot(x, beta_pdf(x, a0, b0), '--', color='#8a97b3', label=f'prior Beta({a0},{b0})')
plt.plot(x, beta_pdf(x, aN, bN), color='#2a6f97', lw=2, label=f'posterior Beta({aN},{bN})')
plt.axvline(k/n, color='#d1495b', ls=':', label='data frequency')
plt.xlabel('θ = P(heads)'); plt.ylabel('density'); plt.legend(); plt.title('Bayesian update: prior → posterior')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 176" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <rect x="18" y="44" width="176" height="52" rx="8" fill="#eef2f7" stroke="#33415c"/>
  <text x="106" y="68" text-anchor="middle" fill="#1f2a44" font-weight="700">prior belief</text>
  <text x="106" y="86" text-anchor="middle" fill="#5a6b8c">Beta(a, b)</text>
  <text x="212" y="76" text-anchor="middle" fill="#8f2233" font-size="20">×</text>
  <rect x="230" y="44" width="196" height="52" rx="8" fill="#fbeaed" stroke="#c1121f"/>
  <text x="328" y="68" text-anchor="middle" fill="#8f2233" font-weight="700">likelihood (the data)</text>
  <text x="328" y="86" text-anchor="middle" fill="#a05563">k heads, n−k tails</text>
  <text x="446" y="76" text-anchor="middle" fill="#245030" font-size="17">∝</text>
  <rect x="466" y="44" width="236" height="52" rx="8" fill="#e7f0e8" stroke="#3a7d44"/>
  <text x="584" y="68" text-anchor="middle" fill="#204d2a" font-weight="700">posterior belief</text>
  <text x="584" y="86" text-anchor="middle" fill="#3a7d44">Beta(a + k, b + n − k)</text>
  <text x="360" y="132" text-anchor="middle" fill="#1f2a44">posterior mean = (a + k) / (a + b + n)</text>
  <text x="360" y="152" text-anchor="middle" fill="#5a6b8c" font-size="10.5">the prior acts like (a + b) coin-flips you've already seen; data eventually outweighs it</text>
</svg>`,

  keyPoints: [
    String.raw`Bayes' rule: posterior $\propto$ likelihood $\times$ prior. The hard part is the normalizer $p(\mathcal D)=\int\!p(\mathcal D\mid\theta)p(\theta)d\theta$ — conjugacy lets you skip it.`,
    String.raw`<strong>Conjugate prior</strong> = the posterior is in the same family as the prior. Beta–Bernoulli: $\mathrm{Beta}(a,b)\to\mathrm{Beta}(a+k,\,b+n-k)$. Updating is arithmetic.`,
    String.raw`The posterior <em>mean</em> $\frac{a+k}{a+b+n}$ interpolates prior mean and data frequency; the prior's weight is $a+b$ pseudo-observations.`,
    String.raw`More data $\Rightarrow$ posterior concentrates and forgets the prior. With little data, the prior (your regularizer) matters most — exactly the MLE-vs-MAP story, now with full uncertainty.`,
    String.raw`Bayesian updating is <strong>sequential = batch</strong>: feeding data one point at a time (posterior becomes next prior) gives the same result as one big update.`
  ],

  commonMistakes: [
    { wrong: "Reporting only the posterior mean and dropping the spread.",
      why: String.raw`The whole point of going Bayesian is the <em>distribution</em>. A posterior mean of 0.7 from 10 flips and from 10,000 flips are worlds apart in certainty; the variance (or a credible interval) is what tells you which.` },
    { wrong: "Confusing a 95% credible interval with a 95% confidence interval.",
      why: String.raw`A credible interval genuinely says "given the data and prior, $\theta$ is in here with 95% probability" — a statement about $\theta$. A frequentist confidence interval is a statement about the <em>procedure</em> over hypothetical repeats. Different meanings; don't swap the interpretations.` },
    { wrong: "Using a strong prior and then being surprised the data can't move it.",
      why: String.raw`A prior like $\mathrm{Beta}(100,100)$ is worth 200 pseudo-flips; ten real flips barely dent it. Match prior strength to how much you actually know — and remember you can always check sensitivity by trying a weaker prior.` }
  ],

  quiz: [
    { q: "Prior Beta(2, 2). You observe 8 heads and 2 tails. What is the posterior?",
      options: ["Beta(10, 4)", "Beta(8, 2)", "Beta(10, 6)", "Beta(2, 2)"], answer: 0,
      explain: String.raw`Add heads to $a$ and tails to $b$: $\mathrm{Beta}(2+8,\ 2+2)=\mathrm{Beta}(10,4)$.` },
    { q: "For that Beta(10, 4) posterior, what is the posterior mean?",
      options: ["10/14 ≈ 0.714", "0.8", "0.5", "10/4"], answer: 0,
      explain: String.raw`$\mathbb E[\theta]=\frac{a}{a+b}=\frac{10}{14}\approx0.714$ — between the prior mean 0.5 and the data frequency 0.8.` },
    { q: "What makes a prior 'conjugate' to a likelihood?",
      options: ["The posterior is in the same distributional family as the prior",
                "The prior is uniform", "The prior has zero variance", "The likelihood is Gaussian"], answer: 0,
      explain: String.raw`Conjugacy means prior and posterior share a family (Beta$\to$Beta here), so Bayes' rule reduces to updating parameters instead of computing an integral.` },
    { q: "As the number of observations n → ∞, the posterior mean approaches…",
      options: ["The data frequency k/n", "The prior mean", "Always 0.5", "Infinity"], answer: 0,
      explain: String.raw`$\frac{a+k}{a+b+n}\to\frac{k}{n}$ as $n\to\infty$: the fixed prior counts $a,b$ become negligible next to the data. The data wins in the limit.` },
    { q: "Why can we avoid computing the evidence integral p(𝒟) in the conjugate case?",
      options: ["The posterior's functional form is a known distribution, so its normalizer is known in closed form",
                "p(𝒟) is always 1", "The prior cancels it", "We can't — we always integrate"], answer: 0,
      explain: String.raw`Once we recognize the posterior kernel as a $\mathrm{Beta}(a+k,b+n-k)$, its normalizing constant is the known Beta function — no integral needed. (When the posterior <em>isn't</em> a known family, that integral is the whole problem — see the next lessons.)` }
  ],

  practice: [
    { level: "easy", prompt: "Prior Beta(1, 1) (uniform). Observe 3 heads, 1 tail. Give the posterior and its mean.",
      solution: String.raw`$\mathrm{Beta}(1+3,\,1+1)=\mathrm{Beta}(4,2)$, mean $=4/6\approx0.667$. Note that with a uniform prior the posterior mean is $(k+1)/(n+2)$ — Laplace's 'add-one' smoothing, which is this exact update.` },
    { level: "easy", prompt: "Show that updating on 8 heads then 2 tails gives the same posterior as updating on all 10 flips at once.",
      solution: String.raw`After 8 heads: $\mathrm{Beta}(a+8,b)$. Treat that as the new prior; add 2 tails: $\mathrm{Beta}(a+8,\,b+2)$ — identical to the single batch update. Conjugate updates commute and accumulate, which is why online and batch Bayesian learning agree.` },
    { level: "med", prompt: "Compute a 95% posterior credible interval for Beta(10, 4) numerically and interpret it.",
      solution: String.raw`Use the Beta quantile function (e.g. sample many draws and take the 2.5% and 97.5% percentiles, or invert the CDF): roughly $[0.48, 0.90]$. Interpretation: given prior and data, there's a 95% probability the true click-through rate lies in that range — a direct probability statement about $\theta$.` },
    { level: "med", prompt: "The Normal–Normal conjugate pair: with known variance σ², a Normal prior on the mean gives a Normal posterior. Write the posterior mean as a precision-weighted average.",
      solution: String.raw`Posterior precision $=$ prior precision $+$ data precision, and posterior mean $=\frac{\tau_0^{-2}\mu_0+n\sigma^{-2}\bar x}{\tau_0^{-2}+n\sigma^{-2}}$ — a precision-weighted blend of prior mean $\mu_0$ and sample mean $\bar x$. It's the Gaussian analogue of the Beta pseudo-count formula, and it's exactly the machinery of the next lesson.` },
    { level: "hard", prompt: "Derive the posterior predictive probability that the NEXT flip is heads for a Beta(a, b) posterior, and explain why it isn't just the posterior-mean plug-in… (it is, here — show why).",
      solution: String.raw`$P(\text{next}=H)=\int_0^1\theta\,p(\theta\mid\mathcal D)\,d\theta=\mathbb E[\theta\mid\mathcal D]=\frac{a}{a+b}$. For a Bernoulli next-flip the predictive equals the posterior mean because the likelihood is linear in $\theta$. In general the posterior predictive $\int p(\text{new}\mid\theta)p(\theta\mid\mathcal D)d\theta$ integrates over uncertainty and is <em>wider</em> than a mean plug-in — the Bayesian way to avoid overconfident predictions.` },
    { level: "hard", prompt: "A Beta(0.5, 0.5) 'Jeffreys' prior is popular for coins. What's special about it, and how does it behave with zero data on one side (e.g. 0 tails)?",
      solution: String.raw`Jeffreys priors are invariant to reparameterization (built from the Fisher information) and here put mass near 0 and 1, encoding "the coin might be nearly deterministic." With $k$ heads and $0$ tails the posterior is $\mathrm{Beta}(0.5+k,\,0.5)$ — still proper and not collapsing to $\theta=1$, avoiding the MLE's overconfident "100% heads" from a small sample.` }
  ],

  deepDive: String.raw`<p><strong>Priors are regularizers with a probability interpretation.</strong> The pseudo-count view ($a$ prior heads, $b$ prior tails) is the same idea as Laplace smoothing in a text model and the same idea as an $L_2$ penalty in regression: all three inject prior belief to stabilize estimates when data is scarce. Bayesian inference just makes the belief explicit and hands you calibrated uncertainty for free.</p>
<p><strong>The exponential family is why conjugacy exists.</strong> Bernoulli, Poisson, Gaussian, Gamma and friends are exponential-family distributions, and each has a conjugate prior built from the same sufficient statistics. That's not a coincidence — it's a structural property that makes a handful of models solvable in closed form. Those closed forms are the sanity checks you use to validate the approximate methods (MCMC, variational) that handle everything else.</p>
<p><strong>Where the closed form dies.</strong> Change the coin to a neural network, or use a non-conjugate prior, and the posterior is no longer a named distribution — the evidence integral $p(\mathcal D)$ becomes a high-dimensional integral with no formula. Everything after this lesson is about that: how to <em>sample</em> from such a posterior (MCMC), how to <em>approximate</em> it with something tractable (variational inference), and how to put a prior directly over functions (Gaussian processes).</p>`
};

/* ------------------------------------------------------------------ 15.2 */
window.LESSON_CONTENT["15.2"] = {
  subtitle: "Keep a whole distribution over the weights instead of one best fit — and get predictions that come with honest error bars that widen where you have no data.",

  aiMoment: String.raw`A model that quietly extrapolates a confident straight line into a region it has never seen is a liability — it's how systems make bold, wrong calls off-distribution. <strong>Bayesian linear regression</strong> fixes that: it returns a predictive distribution whose error bars grow exactly where the data thins out. That "know what you don't know" is the engine of Bayesian optimization (which hyperparameters to try next), active learning (which example to label next), and safe decision-making. It's also the exact finite-dimensional cousin of the Gaussian process you'll meet at the end of this track.`,

  plainEnglish: String.raw`Ordinary regression picks the single best-fitting curve. Bayesian regression keeps <em>all</em> curves that are plausible given the data, weighted by how plausible they are. To predict at a new $x$, it averages those curves — and reports how much they <em>disagree</em> there. Where you have lots of data the curves agree (tight error bar); far from any data they fan out (wide error bar).`,

  intuition: String.raw`Put a Gaussian prior on the weights: before seeing data, small weights are more plausible than huge ones. Each data point rules out weight combinations that miss it, shrinking the cloud of plausible weights into a tighter Gaussian — the posterior. Prediction has <em>two</em> sources of uncertainty: the irreducible observation noise, and how much the surviving weights still disagree at that input. The second term is small near data and large in the gaps, which is why the band pinches at the points and balloons between and beyond them.`,

  formalism: String.raw`With feature map $\Phi$ (row $n$ is $\phi(x_n)^\top$), Gaussian prior $\mathbf w\sim\mathcal N(\mathbf 0,\alpha^{-1}I)$ and Gaussian noise $y\mid\mathbf w\sim\mathcal N(\Phi\mathbf w,\beta^{-1}I)$, the posterior over weights is Gaussian:
$$\mathbf w\mid\mathcal D\sim\mathcal N(\mathbf m_N,\,S_N),\qquad S_N^{-1}=\alpha I+\beta\,\Phi^\top\Phi,\qquad \mathbf m_N=\beta\,S_N\,\Phi^\top\mathbf y.$$
The <strong>posterior precision is prior precision plus data precision</strong>. The predictive distribution at a new input $x_\*$ is Gaussian with
$$\text{mean}=\phi(x_\*)^\top\mathbf m_N,\qquad \text{variance}=\underbrace{\beta^{-1}}_{\text{noise}}+\underbrace{\phi(x_\*)^\top S_N\,\phi(x_\*)}_{\text{weight uncertainty}}.$$`,

  derivation: String.raw`<strong>Gaussian × Gaussian = Gaussian, via completing the square.</strong>
<ol>
<li><strong>Log-posterior.</strong> $\log p(\mathbf w\mid\mathcal D)=\log p(\mathbf y\mid\mathbf w)+\log p(\mathbf w)+\text{const}=-\tfrac{\beta}{2}\lVert\mathbf y-\Phi\mathbf w\rVert^2-\tfrac{\alpha}{2}\lVert\mathbf w\rVert^2+\text{const}.$</li>
<li><strong>It's quadratic in $\mathbf w$.</strong> Expand: the $\mathbf w^\top(\cdot)\mathbf w$ term is $-\tfrac12\mathbf w^\top(\alpha I+\beta\Phi^\top\Phi)\mathbf w$, and the linear term is $\beta\,\mathbf w^\top\Phi^\top\mathbf y$. A quadratic log-density <em>is</em> a Gaussian.</li>
<li><strong>Read off precision and mean.</strong> Matching to $-\tfrac12(\mathbf w-\mathbf m_N)^\top S_N^{-1}(\mathbf w-\mathbf m_N)$ gives $S_N^{-1}=\alpha I+\beta\Phi^\top\Phi$ and (from the linear term) $\mathbf m_N=\beta S_N\Phi^\top\mathbf y$. This is the least-squares/normal-equations gradient from 14.2, now with a prior term $\alpha I$ added — Bayesian regression <em>is</em> ridge regression with uncertainty attached.</li>
<li><strong>Predict by integrating out the weights.</strong> $p(y_\*\mid x_\*,\mathcal D)=\int\mathcal N(y_\*;\phi_\*^\top\mathbf w,\beta^{-1})\,\mathcal N(\mathbf w;\mathbf m_N,S_N)\,d\mathbf w=\mathcal N\big(\phi_\*^\top\mathbf m_N,\ \beta^{-1}+\phi_\*^\top S_N\phi_\*\big).$ The two variance terms are the noise and the propagated weight uncertainty — the second is what widens the band away from data.</li>
</ol>`,

  code: [
    { label: "Bayesian linear regression: posterior weights + predictive band",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def phi(x): return np.stack([np.ones_like(x), x, x**2, x**3], 1)   # cubic basis functions
Xtr = np.sort(rng.uniform(-1, 1, 12)); ytr = np.sin(3*Xtr) + 0.1*rng.standard_normal(12)

alpha, beta = 2.0, 25.0                # prior precision on weights, noise precision (1/σ²)
P  = phi(Xtr)
SN = np.linalg.inv(alpha*np.eye(4) + beta * P.T @ P)   # posterior covariance of the weights
mN = beta * SN @ P.T @ ytr                             # posterior mean of the weights

xg = np.linspace(-1.3, 1.3, 200); Pg = phi(xg)
mean = Pg @ mN                                          # predictive mean
var  = 1/beta + np.einsum('ij,jk,ik->i', Pg, SN, Pg)   # noise + weight-uncertainty
sd = np.sqrt(var)
print(f"predictive sd at center = {sd[100]:.3f},  at the edge = {sd[-1]:.3f}   (wider away from data)")

plt.figure(figsize=(6.6, 4))
plt.fill_between(xg, mean-2*sd, mean+2*sd, color='#2a6f97', alpha=.18, label='±2σ predictive')
plt.plot(xg, mean, color='#2a6f97', lw=2, label='posterior mean')
plt.scatter(Xtr, ytr, color='#d1495b', zorder=5, label='data')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(fontsize=8); plt.title('Bayesian linear regression: mean + uncertainty')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 208" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <rect x="18" y="30" width="330" height="150" rx="8" fill="#eef2f7" stroke="#2a6f97"/>
  <text x="183" y="52" text-anchor="middle" font-weight="700" fill="#1c4e70">posterior over weights</text>
  <text x="183" y="80" text-anchor="middle" fill="#1f2a44">prior:  w ~ N(0, α⁻¹I)</text>
  <text x="183" y="106" text-anchor="middle" fill="#1f2a44">Sₙ⁻¹ = αI + β ΦᵀΦ</text>
  <text x="183" y="124" text-anchor="middle" fill="#5a6b8c" font-size="10">(prior precision + data precision)</text>
  <text x="183" y="150" text-anchor="middle" fill="#1f2a44">mₙ = β Sₙ Φᵀy</text>
  <text x="183" y="170" text-anchor="middle" fill="#5a6b8c" font-size="10">= ridge solution (MAP)</text>
  <rect x="372" y="30" width="330" height="150" rx="8" fill="#e7f0e8" stroke="#3a7d44"/>
  <text x="537" y="52" text-anchor="middle" font-weight="700" fill="#245030">predict at a new x</text>
  <text x="537" y="80" text-anchor="middle" fill="#1f2a44">mean = φ(x)ᵀ mₙ</text>
  <text x="537" y="112" text-anchor="middle" fill="#1f2a44">var = 1/β + φ(x)ᵀ Sₙ φ(x)</text>
  <text x="470" y="140" text-anchor="middle" fill="#8f2233" font-size="10">↑ noise</text>
  <text x="600" y="140" text-anchor="middle" fill="#1c4e70" font-size="10">↑ weight uncertainty</text>
  <text x="537" y="166" text-anchor="middle" fill="#5a6b8c" font-size="10">second term grows where data is sparse</text>
</svg>`,

  keyPoints: [
    String.raw`Gaussian prior + Gaussian likelihood $\Rightarrow$ Gaussian posterior over weights: $S_N^{-1}=\alpha I+\beta\Phi^\top\Phi$, $\mathbf m_N=\beta S_N\Phi^\top\mathbf y$. Exact, closed-form, no sampling.`,
    String.raw`<strong>Posterior precision = prior precision + data precision.</strong> Each observation adds information ($\beta\Phi^\top\Phi$) that tightens the weight cloud.`,
    String.raw`Predictive variance has two parts: irreducible noise $\beta^{-1}$ plus weight uncertainty $\phi_\*^\top S_N\phi_\*$. The second term is what makes error bars widen away from data.`,
    String.raw`The posterior mean $\mathbf m_N$ is exactly the <strong>ridge-regression</strong> solution — Bayesian linear regression is ridge with calibrated uncertainty bolted on. Priors are penalties.`,
    String.raw`This is the finite-basis version of a Gaussian process: replace $\phi(x)^\top S_N\phi(x')$ with a kernel and you get 15.5.`
  ],

  commonMistakes: [
    { wrong: "Reporting only the noise term β⁻¹ as the predictive uncertainty.",
      why: String.raw`That ignores $\phi_\*^\top S_N\phi_\*$ — the model's uncertainty about the weights — which is the part that grows off-distribution. Drop it and your model is falsely confident exactly where it should hesitate.` },
    { wrong: "Forgetting that the α I prior term is what keeps the posterior well-defined.",
      why: String.raw`Without the prior ($\alpha=0$), $S_N^{-1}=\beta\Phi^\top\Phi$ can be singular (fewer/rank-deficient data than features) and the posterior is improper. The prior precision $\alpha I$ is exactly the $\lambda I$ that makes ridge invertible.` },
    { wrong: "Standardizing features after fitting, or mixing feature scales without adjusting the prior.",
      why: String.raw`An isotropic prior $\alpha^{-1}I$ assumes all weights are on a comparable scale. If your basis functions have wildly different magnitudes, that prior penalizes them unequally — standardize features (or use a scaled prior) so the prior means what you intend.` }
  ],

  quiz: [
    { q: "In Bayesian linear regression, the posterior precision S_N⁻¹ equals…",
      options: ["αI + βΦᵀΦ", "βΦᵀΦ only", "αI only", "Φᵀy"], answer: 0,
      explain: String.raw`Posterior precision = prior precision $\alpha I$ + data precision $\beta\Phi^\top\Phi$ — information adds up.` },
    { q: "The predictive variance at a new point is β⁻¹ + φ(x)ᵀS_N φ(x). Which term grows far from the training data?",
      options: ["φ(x)ᵀS_N φ(x) (weight uncertainty)", "β⁻¹ (noise)", "Both equally", "Neither — it's constant"], answer: 0,
      explain: String.raw`The noise $\beta^{-1}$ is constant; the weight-uncertainty term grows where features $\phi(x)$ point in directions the data didn't constrain — i.e. away from the data.` },
    { q: "The posterior mean m_N coincides with which classical estimator?",
      options: ["Ridge regression (L2-penalized least squares)", "Ordinary least squares with no penalty",
                "The sample mean of y", "The maximum of the likelihood only"], answer: 0,
      explain: String.raw`$\mathbf m_N=(\alpha I+\beta\Phi^\top\Phi)^{-1}\beta\Phi^\top\mathbf y$ is the ridge solution with $\lambda=\alpha/\beta$. Bayesian regression = ridge + uncertainty.` },
    { q: "You have 4 basis functions and 12 data points. What is the shape of the posterior covariance S_N?",
      options: ["4×4 (over the weights)", "12×12", "12×4", "A single number"], answer: 0,
      explain: String.raw`$S_N$ is the covariance of the weight vector $\mathbf w\in\mathbb R^4$, so it's $4\times4$ regardless of how many data points there are.` },
    { q: "Increasing the prior precision α (a stronger prior toward w = 0) does what to the fitted curve?",
      options: ["Shrinks the weights toward zero — a smoother, flatter fit", "Makes it interpolate every point",
                "Removes the noise term", "Has no effect"], answer: 0,
      explain: String.raw`Larger $\alpha$ is a heavier penalty on weight size (stronger regularization), pulling the fit toward the prior mean (flat) and reducing overfitting — the bias–variance knob.` }
  ],

  practice: [
    { level: "easy", prompt: "Set α very small (e.g. 1e-6) and refit. What happens to the fit and why?",
      solution: String.raw`With almost no prior, the posterior mean approaches ordinary least squares — the cubic wiggles more to chase the noise, and the predictive band can behave badly where $\Phi^\top\Phi$ is nearly singular. It shows the prior's role as a stabilizer.` },
    { level: "easy", prompt: "Draw 10 sample curves from the posterior over weights and overlay them. What do you expect to see relative to the ±2σ band?",
      solution: String.raw`Sample $\mathbf w^{(s)}\sim\mathcal N(\mathbf m_N,S_N)$ and plot $\Phi_g\mathbf w^{(s)}$. The sampled curves cluster tightly near the data and spread out in the gaps — their envelope traces the same shape as the analytic ±2σ predictive band (minus the noise term).` },
    { level: "med", prompt: "Show that the predictive-mean formula reduces to ridge regression by writing m_N in terms of λ = α/β.",
      solution: String.raw`$\mathbf m_N=(\alpha I+\beta\Phi^\top\Phi)^{-1}\beta\Phi^\top\mathbf y=(\lambda I+\Phi^\top\Phi)^{-1}\Phi^\top\mathbf y$ with $\lambda=\alpha/\beta$ — the textbook ridge estimator. The Bayesian ratio of prior-to-noise precision <em>is</em> the ridge penalty.` },
    { level: "med", prompt: "Use the predictive variance to pick the next x to sample (active learning). Which x does 'maximum variance' choose, and why is that sensible?",
      solution: String.raw`It picks the $x$ where $\beta^{-1}+\phi_\*^\top S_N\phi_\*$ is largest — typically the region farthest from existing data. Labelling there removes the most uncertainty per query, which is the core idea of uncertainty-sampling active learning and Bayesian optimization's exploration.` },
    { level: "hard", prompt: "Add hyperparameter learning: sketch how you'd choose α and β by maximizing the marginal likelihood (evidence) instead of guessing them.",
      solution: String.raw`The evidence $p(\mathbf y\mid\alpha,\beta)=\int p(\mathbf y\mid\mathbf w,\beta)p(\mathbf w\mid\alpha)d\mathbf w$ is a Gaussian integral with a closed form (it's $\mathcal N(\mathbf y;\mathbf 0,\beta^{-1}I+\alpha^{-1}\Phi\Phi^\top)$). Maximizing its log over $\alpha,\beta$ ('type-II maximum likelihood' / evidence approximation) automatically balances fit and complexity — Occam's razor falling out of the math, and the same objective GPs optimize.` },
    { level: "hard", prompt: "Derive why φ(x)ᵀS_N φ(x) can be written using only inner products of features, hinting at the kernel trick.",
      solution: String.raw`Using the matrix identity for $S_N$, the predictive variance and mean can be re-expressed entirely through the Gram matrix $\Phi\Phi^\top$ and the vectors $\phi(x_\*)^\top\Phi^\top$ — i.e. only through inner products $\phi(x)^\top\phi(x')$. Replace every such inner product with a kernel $k(x,x')$ and you never need the features explicitly: that's the Gaussian process (15.5), Bayesian regression in an infinite-dimensional basis.` }
  ],

  deepDive: String.raw`<p><strong>Two uncertainties, and why the split matters.</strong> The predictive variance separates <em>aleatoric</em> uncertainty (the $\beta^{-1}$ noise — irreducible randomness in the data) from <em>epistemic</em> uncertainty (the $\phi_\*^\top S_N\phi_\*$ term — ignorance about the weights, which more data can reduce). Modern deep-learning uncertainty work (deep ensembles, MC-dropout, Bayesian last layers) is largely about recovering that epistemic term, which ordinary networks throw away — and it's the term that flags out-of-distribution inputs.</p>
<p><strong>Evidence and Occam's razor.</strong> The marginal likelihood $p(\mathcal D\mid\alpha,\beta)$ you'd maximize to set the hyperparameters automatically penalizes overly flexible models: a model that can fit anything spreads its probability thin and scores low evidence on the data actually seen. This "Bayesian Occam's razor" is why evidence maximization resists overfitting without a separate validation set — the same principle underlies GP kernel selection.</p>
<p><strong>The bridge to Gaussian processes.</strong> Everything here was written in terms of a fixed, finite feature map $\phi$. The kernel trick lets you take $\phi$ to be <em>infinite-dimensional</em> while only ever computing inner products $k(x,x')=\phi(x)^\top\phi(x')$. Do that and Bayesian linear regression becomes a Gaussian process — a prior directly over functions. That's the final lesson of this track; this one is its skeleton.</p>`
};

/* ------------------------------------------------------------------ 15.3 */
window.LESSON_CONTENT["15.3"] = {
  subtitle: "When the posterior isn't a tidy formula, you can't write it down — but you can draw samples from it, using only the unnormalized density. That's MCMC.",

  aiMoment: String.raw`Real Bayesian models — hierarchical models, Bayesian neural nets, anything non-conjugate — have posteriors with no closed form and an intractable normalizing constant. <strong>Markov chain Monte Carlo</strong> is the workhorse that made applied Bayesian statistics possible (it's what PyMC and Stan run under the hood): it produces samples from the posterior using <em>only</em> the unnormalized density, which is the one thing you can always compute (likelihood × prior). The very same "sample from an unnormalized distribution" problem reappears in diffusion models, so this idea reaches well beyond classical stats.`,

  plainEnglish: String.raw`You want samples from a distribution but you only know its <em>shape</em>, not its total area (the normalizer). MCMC builds a wanderer that steps around the parameter space and, over time, spends time in each region in proportion to its probability. Collect everywhere the wanderer goes and their histogram <em>is</em> the posterior. The trick that makes it work: the wanderer only ever compares two spots by ratio, and in a ratio the unknown normalizer cancels.`,

  intuition: String.raw`From the current $\theta$, propose a nearby $\theta'$. If $\theta'$ is more probable, move there. If it's less probable, still move there sometimes — with probability equal to how much less probable it is. Uphill moves are always accepted; downhill moves are accepted in proportion to the drop. That gentle willingness to go downhill is what lets the walker explore the whole distribution instead of getting stuck on the nearest peak, and the long-run fraction of time spent anywhere equals its posterior probability.`,

  formalism: String.raw`To sample a target $\pi(\theta)=\tilde p(\theta)/Z$ known only up to $Z$, <strong>Metropolis–Hastings</strong> uses a proposal $q(\theta'\mid\theta)$ and accepts with probability
$$\alpha=\min\!\Big(1,\ \frac{\tilde p(\theta')\,q(\theta\mid\theta')}{\tilde p(\theta)\,q(\theta'\mid\theta)}\Big).$$
For a symmetric proposal ($q(\theta'\mid\theta)=q(\theta\mid\theta')$, e.g. a Gaussian random walk) this simplifies to
$$\alpha=\min\!\Big(1,\ \frac{\tilde p(\theta')}{\tilde p(\theta)}\Big)\qquad\Longleftarrow\ Z\text{ cancels}.$$
The resulting Markov chain has $\pi$ as its stationary distribution, so after a burn-in the states are (correlated) samples from the posterior.`,

  derivation: String.raw`<strong>Why the chain lands on the posterior: detailed balance.</strong>
<ol>
<li><strong>The sufficient condition.</strong> A chain with transition density $T(\theta\to\theta')$ has $\pi$ as a stationary distribution if it satisfies <em>detailed balance</em>: $\pi(\theta)\,T(\theta\to\theta')=\pi(\theta')\,T(\theta'\to\theta)$ for all $\theta,\theta'$. (Sum both sides over $\theta$ to see $\pi$ is preserved.)</li>
<li><strong>MH transition.</strong> A move $\theta\to\theta'$ happens by proposing (prob. $q(\theta'\mid\theta)$) then accepting (prob. $\alpha(\theta,\theta')$), so $T(\theta\to\theta')=q(\theta'\mid\theta)\,\alpha(\theta,\theta')$ for $\theta'\ne\theta$.</li>
<li><strong>Plug in the acceptance rule.</strong> With $\alpha=\min\!\big(1,\frac{\pi(\theta')q(\theta\mid\theta')}{\pi(\theta)q(\theta'\mid\theta)}\big)$, check the product $\pi(\theta)q(\theta'\mid\theta)\,\alpha(\theta,\theta')$. In the case the min picks the ratio, it equals $\pi(\theta')q(\theta\mid\theta')\cdot 1$ — which is exactly $\pi(\theta')T(\theta'\to\theta)$. Detailed balance holds. (The other case is symmetric.)</li>
<li><strong>The normalizer never appears.</strong> Everywhere $\pi$ shows up it's inside a ratio $\pi(\theta')/\pi(\theta)=\tilde p(\theta')/\tilde p(\theta)$, so $Z$ cancels. You only ever evaluate the unnormalized $\tilde p=\text{likelihood}\times\text{prior}$ — which is why MCMC works when nothing else does.</li>
</ol>`,

  code: [
    { label: "Metropolis–Hastings from scratch, checked against a known posterior",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# Infer the mean θ of N(θ, 1) data with a N(0, 100) prior.
# (This case is conjugate, so we KNOW the true posterior and can check MH against it.)
data = rng.normal(1.5, 1.0, 40); n = len(data); tau2 = 100.0
def log_p(t):                                     # UNNORMALIZED log posterior = log lik + log prior
    return -0.5*((data - t)**2).sum() - 0.5*t**2/tau2

# Metropolis-Hastings: random-walk propose, accept with prob min(1, ratio).
theta = 0.0; samples = []; step = 0.4; accepts = 0
for _ in range(20000):
    prop = theta + rng.normal(0, step)            # symmetric Gaussian proposal
    if np.log(rng.uniform()) < log_p(prop) - log_p(theta):   # accept? (compare in log-space)
        theta = prop; accepts += 1
    samples.append(theta)
samples = np.array(samples[2000:])                # discard burn-in

# analytic (conjugate) posterior, for ground truth
prec = n + 1/tau2; m = (n*data.mean())/prec; sd = np.sqrt(1/prec)
print(f"MH:       mean={samples.mean():.3f}  sd={samples.std():.3f}   accept rate={accepts/20000:.2f}")
print(f"analytic: mean={m:.3f}  sd={sd:.3f}")

plt.figure(figsize=(6.6, 3.8))
plt.hist(samples, bins=40, density=True, color='#8fb0c9', edgecolor='w', label='MH samples')
tg = np.linspace(m-4*sd, m+4*sd, 200)
plt.plot(tg, np.exp(-0.5*((tg-m)/sd)**2)/(sd*np.sqrt(2*np.pi)), color='#d1495b', lw=2, label='true posterior')
plt.xlabel('θ'); plt.ylabel('density'); plt.legend(); plt.title('Metropolis–Hastings recovers the posterior')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 172" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="t53" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <rect x="12" y="42" width="140" height="48" rx="8" fill="#eef2f7" stroke="#33415c"/><text x="82" y="70" text-anchor="middle" fill="#1f2a44">current θ</text>
  <rect x="182" y="42" width="164" height="48" rx="8" fill="#eef2f7" stroke="#33415c"/><text x="264" y="64" text-anchor="middle" fill="#1f2a44">propose θ′ = θ + step·ε</text><text x="264" y="80" text-anchor="middle" fill="#5a6b8c" font-size="10">(random walk)</text>
  <rect x="376" y="38" width="172" height="56" rx="8" fill="#fff5e6" stroke="#b8860b"/><text x="462" y="60" text-anchor="middle" fill="#7a5b00" font-weight="700">r = p̃(θ′) / p̃(θ)</text><text x="462" y="78" text-anchor="middle" fill="#9a7b20" font-size="10">normalizer Z cancels</text>
  <rect x="578" y="42" width="130" height="48" rx="8" fill="#e7f0e8" stroke="#3a7d44"/><text x="643" y="64" text-anchor="middle" fill="#245030">accept with</text><text x="643" y="80" text-anchor="middle" fill="#245030">prob min(1, r)</text>
  <line x1="152" y1="66" x2="180" y2="66" stroke="#6b7a99" marker-end="url(#t53)"/>
  <line x1="346" y1="66" x2="374" y2="66" stroke="#6b7a99" marker-end="url(#t53)"/>
  <line x1="548" y1="66" x2="576" y2="66" stroke="#6b7a99" marker-end="url(#t53)"/>
  <line x1="643" y1="90" x2="643" y2="132" stroke="#6b7a99"/><line x1="643" y1="132" x2="82" y2="132" stroke="#6b7a99"/><line x1="82" y1="132" x2="82" y2="92" stroke="#6b7a99" marker-end="url(#t53)"/>
  <text x="360" y="152" text-anchor="middle" fill="#4a5878" font-size="10.5">record θ, then repeat — uphill moves always accepted, downhill sometimes (that's how it explores)</text>
</svg>`,

  keyPoints: [
    String.raw`MCMC samples a posterior known only up to its normalizer, using the unnormalized $\tilde p=\text{likelihood}\times\text{prior}$ — the one quantity you can always evaluate.`,
    String.raw`Metropolis acceptance for a symmetric proposal is $\min(1,\tilde p(\theta')/\tilde p(\theta))$. Because it's a ratio, the intractable $Z$ cancels — that's the whole magic.`,
    String.raw`<strong>Detailed balance</strong> $\pi(\theta)T(\theta\to\theta')=\pi(\theta')T(\theta'\to\theta)$ guarantees $\pi$ is the chain's stationary distribution.`,
    String.raw`Samples are <em>correlated</em> and the chain needs a <strong>burn-in</strong> before it forgets its start. You summarize with means/quantiles over post-burn-in draws.`,
    String.raw`Step size is a tuning knob: too small $\Rightarrow$ tiny steps, slow mixing; too big $\Rightarrow$ most proposals rejected. A ~20–50% acceptance rate is a healthy target for random-walk Metropolis.`
  ],

  commonMistakes: [
    { wrong: "Multiplying raw likelihood × prior instead of adding log-likelihood + log-prior.",
      why: String.raw`Products of many densities underflow to 0, making every ratio $0/0$. Work in log-space and compare $\log\tilde p(\theta')-\log\tilde p(\theta)$ against $\log u$ — exactly what the code does.` },
    { wrong: "Using the samples without discarding burn-in or checking mixing.",
      why: String.raw`Early samples reflect the arbitrary starting point, not the posterior. And highly correlated chains have a small <em>effective</em> sample size — 20,000 draws might be worth only a few hundred independent ones. Check trace plots and autocorrelation before trusting a summary.` },
    { wrong: "Assuming a good acceptance rate means the chain has converged.",
      why: String.raw`Acceptance rate tunes step size; it says nothing about whether the chain has explored all the posterior's modes. A walker can happily accept 40% of moves while trapped in one mode of a multimodal posterior. Run multiple chains from different starts and compare.` }
  ],

  quiz: [
    { q: "For a symmetric random-walk proposal, the Metropolis acceptance probability is…",
      options: ["min(1, p̃(θ′)/p̃(θ))", "p̃(θ′) − p̃(θ)", "always 1", "min(1, Z)"], answer: 0,
      explain: String.raw`Symmetry cancels the proposal terms, leaving $\min(1,\tilde p(\theta')/\tilde p(\theta))$ — only the unnormalized density ratio.` },
    { q: "Why does MCMC only need the UNNORMALIZED posterior p̃(θ) = likelihood × prior?",
      options: ["The acceptance rule uses a ratio p̃(θ′)/p̃(θ), so the normalizer Z cancels",
                "Because Z is always 1", "Because priors are normalized", "It actually needs Z"], answer: 0,
      explain: String.raw`Every appearance of the target is in a ratio, and $Z/Z=1$. That's precisely why MCMC handles intractable posteriors.` },
    { q: "A proposed θ′ has p̃(θ′) = 0.6·p̃(θ). What is the probability of accepting it?",
      options: ["0.6", "1.0", "0.4", "0"], answer: 0,
      explain: String.raw`It's a downhill move, so $\alpha=\min(1,0.6)=0.6$ — accepted 60% of the time.` },
    { q: "What property guarantees the chain's stationary distribution is the target π?",
      options: ["Detailed balance", "A high acceptance rate", "A Gaussian proposal", "A large step size"], answer: 0,
      explain: String.raw`Detailed balance $\pi(\theta)T(\theta\to\theta')=\pi(\theta')T(\theta'\to\theta)$ implies $\pi$ is stationary; the Metropolis rule is constructed to satisfy it.` },
    { q: "Your random-walk step size is tiny and the acceptance rate is 98%. What's the likely problem?",
      options: ["Steps are so small the chain barely moves — slow mixing, high autocorrelation",
                "The chain has converged perfectly", "The normalizer is wrong", "The prior is too weak"], answer: 0,
      explain: String.raw`Near-100% acceptance means proposals barely change $\theta$, so the walker inches along and takes forever to explore. Increase the step size toward a ~20–50% acceptance rate.` }
  ],

  practice: [
    { level: "easy", prompt: "Change the step size to 0.02 and to 5.0. Report the acceptance rate in each case and which mixes better.",
      solution: String.raw`0.02 gives a very high acceptance rate but a chain that crawls (high autocorrelation); 5.0 gives a low acceptance rate (most proposals rejected) so it also mixes poorly. A middle value (~0.4 here) balances the two — the classic step-size sweet spot.` },
    { level: "easy", prompt: "Plot the trace (θ vs iteration) for the first 500 steps. What does burn-in look like?",
      solution: String.raw`Starting from $\theta=0$, the trace drifts upward toward ~1.6 over the first few dozen steps, then hovers there. That initial transient is burn-in; discarding it removes the bias from the arbitrary start.` },
    { level: "med", prompt: "Estimate the effective sample size from the lag-1 autocorrelation ρ using ESS ≈ N(1−ρ)/(1+ρ). Why is it far below N?",
      solution: String.raw`Random-walk samples are positively correlated ($\rho$ near, say, 0.9), so $\mathrm{ESS}\approx N\frac{1-\rho}{1+\rho}$ is a small fraction of $N$ — thousands of draws may be worth only hundreds of independent ones. ESS, not raw count, is what determines Monte Carlo error.` },
    { level: "med", prompt: "Make the target bimodal (e.g. p̃(θ) ∝ exp(−(θ²−4)²)) and watch random-walk Metropolis struggle to jump between modes.",
      solution: String.raw`With two well-separated modes, a small-step walker gets stuck in whichever mode it started in for long stretches; crossing the low-probability valley is rare. It demonstrates why multimodal posteriors need tempering, multiple chains, or smarter samplers.` },
    { level: "hard", prompt: "Derive the general (non-symmetric) Metropolis–Hastings acceptance ratio and show it reduces to Metropolis when q is symmetric.",
      solution: String.raw`$\alpha=\min\!\big(1,\frac{\tilde p(\theta')q(\theta\mid\theta')}{\tilde p(\theta)q(\theta'\mid\theta)}\big)$. The extra $q$ ratio (the 'Hastings correction') compensates for asymmetric proposals so detailed balance still holds. If $q(\theta'\mid\theta)=q(\theta\mid\theta')$, that ratio is 1 and you recover $\min(1,\tilde p(\theta')/\tilde p(\theta))$.` },
    { level: "hard", prompt: "Explain why gradient-based samplers (HMC/NUTS) scale to high dimensions where random-walk Metropolis fails.",
      solution: String.raw`Random-walk steps that keep acceptance reasonable must shrink like $1/\sqrt{d}$ as dimension $d$ grows, so exploration slows dramatically. Hamiltonian Monte Carlo uses the gradient $\nabla\log\tilde p$ to propose long, informed trajectories that stay in high-probability regions, giving distant, high-acceptance moves. That gradient — the same one autodiff computes — is why HMC (and NUTS, Stan's default) dominate for continuous, high-dimensional posteriors.` }
  ],

  deepDive: String.raw`<p><strong>A family, not a single algorithm.</strong> Metropolis–Hastings is the ancestor; its descendants specialize the proposal. Gibbs sampling proposes from exact conditional distributions (acceptance always 1) and shines when those conditionals are conjugate. Hamiltonian Monte Carlo and NUTS use $\nabla\log\tilde p$ to glide along the posterior's contours, which is what makes Bayesian inference tractable for thousands of parameters — and why probabilistic programming languages default to them.</p>
<p><strong>Diagnostics are not optional.</strong> MCMC gives <em>asymptotically</em> correct samples, but on a finite run you must check it actually explored the posterior: trace plots, the $\hat R$ statistic across multiple chains (should be ≈1), and effective sample size. A chain can look busy while being wrong (stuck in one mode), so "it ran" is never the same as "it converged."</p>
<p><strong>The unexpected reach.</strong> "Sample from a distribution you only know up to a constant, using its (log-)gradient" is exactly the problem a diffusion model solves at generation time — Langevin/score-based sampling is MCMC with the score $\nabla\log p$ supplied by a neural network. So this classical-statistics lesson is also a direct on-ramp to the generative-models track: the machinery you built for a coin's posterior is the machinery that samples images.</p>`
};

/* ------------------------------------------------------------------ 15.4 */
window.LESSON_CONTENT["15.4"] = {
  subtitle: "Turn inference into optimization: pick a simple distribution q and tune it to hug the true posterior. The objective you maximize — the ELBO — is exactly the loss that trains a VAE.",

  aiMoment: String.raw`MCMC is accurate but can be slow, and it doesn't scale gracefully to millions of latent variables. <strong>Variational inference</strong> trades a little accuracy for enormous speed by turning "find the posterior" into "optimize a distribution" — something SGD is great at. Its objective, the <strong>ELBO</strong>, is not a niche tool: it <em>is</em> the training loss of a variational autoencoder (reconstruction minus a KL term), the KL leash in RLHF, and the "free energy" of statistical physics. If you understand the ELBO, you understand the backbone of modern deep generative models.`,

  plainEnglish: String.raw`The true posterior is too complicated to write down. So instead of chasing it exactly, you choose a simple, friendly distribution $q$ (say a Gaussian) and adjust its parameters until it looks as much like the posterior as possible. "As much like" is measured by KL divergence. You can't compute that KL directly (it hides the intractable normalizer), so you maximize a clever stand-in — the ELBO — and it turns out that pushing the ELBO up automatically pulls $q$ toward the posterior.`,

  intuition: String.raw`Here's the exact bookkeeping. The log-evidence $\log p(x)$ splits, with no approximation, into two non-negative-gap pieces: the ELBO plus $\mathrm{KL}\big(q\Vert p(z\mid x)\big)$. Crucially, $\log p(x)$ doesn't depend on $q$ — it's a fixed ceiling. So raising the ELBO must lower the KL by the same amount, squeezing $q$ toward the true posterior. And since KL $\ge 0$, the ELBO is always $\le\log p(x)$ — an <em>Evidence Lower BOund</em>, which is where the name comes from.`,

  formalism: String.raw`For any distribution $q(z)$, the log-evidence decomposes exactly as
$$\log p(x)=\underbrace{\mathbb E_{q}\!\big[\log p(x,z)-\log q(z)\big]}_{\text{ELBO}(q)}+\underbrace{\mathrm{KL}\big(q(z)\,\Vert\,p(z\mid x)\big)}_{\ge\,0}.$$
Since the left side is fixed and the KL is non-negative,
$$\text{ELBO}(q)=\mathbb E_q[\log p(x,z)]+\mathbb H[q]\ \le\ \log p(x),$$
with equality iff $q=p(z\mid x)$. We maximize the ELBO over $q$'s parameters. To get low-variance gradients when $q_\phi=\mathcal N(\mu,\sigma^2)$, use the <strong>reparameterization trick</strong>: write $z=\mu+\sigma\,\varepsilon$ with $\varepsilon\sim\mathcal N(0,1)$, so the randomness is independent of $\phi$ and $\nabla_\phi$ passes straight through the sample.`,

  derivation: String.raw`<strong>The decomposition, and why maximizing the ELBO is inference.</strong>
<ol>
<li><strong>Start from an identity.</strong> For any $q(z)$ with the same support, $\log p(x)=\log p(x)\int q(z)\,dz=\int q(z)\log p(x)\,dz=\mathbb E_q[\log p(x)].$</li>
<li><strong>Insert the posterior.</strong> $p(x)=\dfrac{p(x,z)}{p(z\mid x)}$, so $\log p(x)=\mathbb E_q\big[\log\tfrac{p(x,z)}{p(z\mid x)}\big]=\mathbb E_q\big[\log\tfrac{p(x,z)}{q(z)}\cdot\tfrac{q(z)}{p(z\mid x)}\big].$</li>
<li><strong>Split the log.</strong> $=\underbrace{\mathbb E_q\big[\log\tfrac{p(x,z)}{q(z)}\big]}_{\text{ELBO}}+\underbrace{\mathbb E_q\big[\log\tfrac{q(z)}{p(z\mid x)}\big]}_{\mathrm{KL}(q\Vert p(z\mid x))\ \ge\ 0}.$ Exact — no approximation yet.</li>
<li><strong>Conclusion.</strong> $\log p(x)$ is constant in $q$, so $\max_q\text{ELBO}(q)\equiv\min_q\mathrm{KL}(q\Vert p(z\mid x))$. Maximizing the (computable) ELBO minimizes the (uncomputable) KL to the posterior. The reparameterization $z=\mu+\sigma\varepsilon$ then makes $\nabla_{\mu,\sigma}\,\mathbb E_q[\cdot]$ an ordinary expectation of a gradient you can estimate with a few samples — the trick that lets a VAE train by backprop.</li>
</ol>`,

  code: [
    { label: "Maximize the ELBO to fit a Gaussian q to an unnormalized target",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# An unnormalized, non-Gaussian target we want to approximate with a Gaussian q.
log_p  = lambda z: -0.5*z**2 - 0.1*z**4 + 0.3*z         # log p~(z)
dlog_p = lambda z: -z - 0.4*z**3 + 0.3                  # its derivative (for reparam gradients)

# The true log-evidence, by numerical integration -- the ceiling the ELBO can't exceed.
zg = np.linspace(-5, 5, 4000); pz = np.exp(log_p(zg))
logZ = np.log(np.sum((pz[:-1] + pz[1:]) / 2 * np.diff(zg)))     # trapezoid, no scipy needed

# Fit q = N(mu, sig^2) by ELBO ascent with the reparameterization trick z = mu + sig*eps.
eps = rng.standard_normal(600)                          # fixed samples -> smooth objective
mu, sig, lr, elbos = 0.0, 1.0, 0.02, []
for _ in range(400):
    z = mu + sig*eps
    elbo = np.mean(log_p(z)) + 0.5*np.log(2*np.pi*np.e*sig**2)   # E_q[log p~] + entropy of q
    elbos.append(elbo)
    mu  += lr * np.mean(dlog_p(z))                       # reparam gradient w.r.t. mu
    sig += lr * (np.mean(eps*dlog_p(z)) + 1/sig)         # ... and w.r.t. sigma
    sig = max(sig, 1e-3)

print(f"final ELBO = {elbos[-1]:.4f}   log Z = {logZ:.4f}   gap = KL(q‖p) = {logZ - elbos[-1]:.4f}")
print(f"ELBO stays below log Z (it's a lower bound): {elbos[-1] <= logZ + 1e-9}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(elbos, color='#3a7d44'); ax[0].axhline(logZ, ls='--', color='#d1495b', label='log evidence (ceiling)')
ax[0].set_xlabel('step'); ax[0].set_ylabel('ELBO'); ax[0].legend(); ax[0].set_title('ELBO rises toward log Z')
qz = np.exp(-0.5*((zg-mu)/sig)**2) / (sig*np.sqrt(2*np.pi))
ax[1].plot(zg, pz/np.exp(logZ), color='#2a6f97', lw=2, label='true p(z)')
ax[1].plot(zg, qz, '--', color='#3a7d44', lw=2, label='fitted q(z)')
ax[1].set_xlim(-4, 4); ax[1].set_xlabel('z'); ax[1].legend(); ax[1].set_title('q approximates p')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 178" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <text x="360" y="26" text-anchor="middle" font-weight="700" fill="#1f2a44">log p(x)  =  ELBO(q)  +  KL(q ‖ posterior)</text>
  <rect x="40" y="52" width="640" height="34" fill="none" stroke="#33415c" stroke-dasharray="3 3"/>
  <text x="360" y="46" text-anchor="middle" fill="#5a6b8c" font-size="10">log-evidence — a fixed ceiling (does not depend on q)</text>
  <rect x="40" y="52" width="470" height="34" fill="#cfe6d6" stroke="#3a7d44"/>
  <rect x="510" y="52" width="170" height="34" fill="#f4c9cf" stroke="#c1121f"/>
  <text x="275" y="74" text-anchor="middle" fill="#204d2a" font-weight="700">ELBO(q)  — maximize this</text>
  <text x="595" y="74" text-anchor="middle" fill="#8f2233">KL ≥ 0</text>
  <line x1="510" y1="92" x2="510" y2="112" stroke="#6b7a99"/>
  <text x="360" y="132" text-anchor="middle" fill="#4a5878">raising the ELBO must shrink the KL gap (the ceiling is fixed) → q is pushed toward the true posterior</text>
  <text x="360" y="154" text-anchor="middle" fill="#5a6b8c" font-size="10.5">ELBO ≤ log p(x) always, with equality only when q = p(z | x)  ⇒  "Evidence Lower BOund"</text>
</svg>`,

  keyPoints: [
    String.raw`Exact identity: $\log p(x)=\text{ELBO}(q)+\mathrm{KL}(q\Vert p(z\mid x))$. The evidence is fixed, so maximizing the ELBO minimizes the KL to the posterior.`,
    String.raw`ELBO $=\mathbb E_q[\log p(x,z)]+\mathbb H[q]\le\log p(x)$ — a computable lower bound. Variational inference is optimization, not sampling.`,
    String.raw`The <strong>reparameterization trick</strong> $z=\mu+\sigma\varepsilon$ moves the randomness off the parameters, so gradients flow through the samples — this is what lets a VAE train by ordinary backprop.`,
    String.raw`ELBO $=\mathbb E_q[\log p(x\mid z)]-\mathrm{KL}(q(z)\Vert p(z))$ — read as <em>reconstruction minus a prior-matching penalty</em>, which is precisely the VAE loss.`,
    String.raw`VI is fast and scalable but <em>biased</em>: $q$'s limited family can't match every posterior. Reverse-KL VI is mode-seeking, so it tends to under-cover (too-narrow uncertainty).`
  ],

  commonMistakes: [
    { wrong: "Thinking the ELBO can exceed the log-evidence.",
      why: String.raw`It can't: $\log p(x)-\text{ELBO}=\mathrm{KL}(q\Vert p(z\mid x))\ge 0$. If your ELBO ever prints above $\log p(x)$, there's a bug (a wrong sign, a missing entropy term, or a biased estimator).` },
    { wrong: "Sampling z ~ q inside the loss without the reparameterization trick, then backpropagating.",
      why: String.raw`If $z$ is sampled directly from $q_\phi$, the sampling op blocks the gradient to $\phi$. Rewriting $z=\mu+\sigma\varepsilon$ with $\varepsilon$ fixed makes $z$ a differentiable function of $\phi$, so $\nabla_\phi$ works. (The alternative score-function/REINFORCE estimator works too but has much higher variance.)` },
    { wrong: "Treating VI's narrow uncertainty as the truth.",
      why: String.raw`Minimizing reverse KL $\mathrm{KL}(q\Vert p)$ is mode-seeking: $q$ locks onto one mode and <em>underestimates</em> variance. Great for a fast point-plus-error-bar, but don't report VI's error bars as calibrated the way you would MCMC's — check against a gold-standard sampler when it matters.` }
  ],

  quiz: [
    { q: "The gap between the log-evidence and the ELBO is exactly…",
      options: ["KL(q ‖ p(z|x))", "the entropy of q", "the reconstruction loss", "zero, always"], answer: 0,
      explain: String.raw`$\log p(x)-\text{ELBO}(q)=\mathrm{KL}\big(q(z)\Vert p(z\mid x)\big)\ge 0$. Closing that gap is the goal of VI.` },
    { q: "Maximizing the ELBO over q is equivalent to…",
      options: ["Minimizing KL(q ‖ p(z|x))", "Maximizing the log-evidence", "Minimizing the entropy of q", "Maximizing the prior"], answer: 0,
      explain: String.raw`Because $\log p(x)$ is constant in $q$, pushing the ELBO up pulls the KL down by the same amount — maximizing ELBO = minimizing KL to the posterior.` },
    { q: "Why is the ELBO called a 'lower bound'?",
      options: ["Because ELBO ≤ log p(x) (the KL gap is ≥ 0)", "Because it's always negative",
                "Because it lower-bounds the KL", "Because q is Gaussian"], answer: 0,
      explain: String.raw`ELBO $=\log p(x)-\mathrm{KL}\le\log p(x)$. It's a bound on the (log) evidence from below — Evidence Lower BOund.` },
    { q: "The reparameterization trick writes z = μ + σε. Its purpose is to…",
      options: ["Let gradients flow through the sampling step to μ and σ",
                "Make z deterministic", "Remove the need for a prior", "Normalize the ELBO"], answer: 0,
      explain: String.raw`With $\varepsilon\sim\mathcal N(0,1)$ fixed, $z$ is a differentiable function of $\mu,\sigma$, so $\nabla_{\mu,\sigma}\mathbb E_q[\cdot]$ becomes an expectation of gradients — trainable by backprop.` },
    { q: "Written as ELBO = E_q[log p(x|z)] − KL(q(z) ‖ p(z)), the two terms are…",
      options: ["reconstruction quality and a prior-matching penalty (the VAE loss)",
                "prior and posterior", "bias and variance", "mean and variance"], answer: 0,
      explain: String.raw`The first term rewards $z$'s that reconstruct $x$; the second keeps $q(z)$ near the prior $p(z)$. That's exactly a VAE's objective — reconstruction minus a KL regularizer.` }
  ],

  practice: [
    { level: "easy", prompt: "Run the code and confirm the printed ELBO ends just below log Z. What does the small remaining gap represent?",
      solution: String.raw`The gap $\log Z-\text{ELBO}\approx 0.003$ is $\mathrm{KL}(q\Vert p)$ — the residual mismatch because a Gaussian $q$ can't perfectly match the slightly non-Gaussian target. If the target were Gaussian, the gap would go to 0 and the ELBO would reach log Z.` },
    { level: "easy", prompt: "Drop the entropy term 0.5·log(2πe σ²) from the ELBO. What goes wrong during optimization?",
      solution: String.raw`Without the entropy reward, nothing stops $q$ from collapsing to a spike ($\sigma\to0$) at the target's mode — the ELBO would chase the highest density point instead of matching the whole distribution. The entropy term is what keeps $q$ appropriately spread.` },
    { level: "med", prompt: "Contrast reverse-KL VI (min KL(q‖p)) with forward-KL (min KL(p‖q)) for a bimodal target. Which under-covers?",
      solution: String.raw`Reverse KL is mode-seeking: $q$ picks one mode and ignores the other, underestimating spread. Forward KL is mean-seeking: $q$ stretches to cover both modes, placing mass in the low-probability valley between them. Standard VI/VAEs use reverse KL, hence their tendency to under-cover.` },
    { level: "med", prompt: "Write the VAE ELBO for one datapoint and identify each term in terms of an encoder q_φ(z|x) and decoder p_θ(x|z).",
      solution: String.raw`$\text{ELBO}=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-\mathrm{KL}\big(q_\phi(z\mid x)\Vert p(z)\big)$. First term = expected reconstruction log-likelihood (decoder); second = KL of the encoder's latent to the prior $\mathcal N(0,I)$. Maximizing it over $\theta,\phi$ jointly trains the autoencoder — 'amortized' VI because one network $q_\phi$ infers $z$ for every $x$.` },
    { level: "hard", prompt: "Show that the EM algorithm is coordinate ascent on the ELBO, alternating an E-step and an M-step.",
      solution: String.raw`Hold model parameters $\theta$ fixed and maximize the ELBO over $q$: the optimum is $q(z)=p(z\mid x,\theta)$ (E-step, KL$=0$). Then hold $q$ fixed and maximize over $\theta$: that maximizes $\mathbb E_q[\log p(x,z\mid\theta)]$ (M-step). Alternating the two is exactly EM — so EM is VI where the E-step can be done exactly, and VI is EM when it can't.` },
    { level: "hard", prompt: "The reparameterization estimator and the score-function (REINFORCE) estimator both give unbiased ELBO gradients. Why is reparameterization preferred?",
      solution: String.raw`REINFORCE uses $\nabla_\phi\mathbb E_q[f]=\mathbb E_q[f\,\nabla_\phi\log q]$, which multiplies by the (high-variance) score and ignores $f$'s smoothness. Reparameterization pushes the gradient <em>through</em> $f$ via $z=\mu+\sigma\varepsilon$, exploiting $\nabla f$ and yielding far lower variance for continuous, differentiable $f$ — which is why VAEs and normalizing flows use it and discrete-latent models (which can't reparameterize easily) must fall back to score-function or relaxations.` }
  ],

  deepDive: String.raw`<p><strong>The VAE is amortized variational inference.</strong> Classic VI fits a separate $q$ for each datapoint; a VAE trains one <em>encoder network</em> $q_\phi(z\mid x)$ that outputs the variational parameters for any $x$ in a single forward pass. The decoder $p_\theta(x\mid z)$ plays the role of the likelihood, and the whole thing is trained by maximizing the ELBO with the reparameterization trick. Reconstruction minus KL — the loss you'd write for a VAE — is this lesson's equation with a neural net plugged in.</p>
<p><strong>Bound-tightening is a research engine.</strong> Because the ELBO is a lower bound, much of modern generative modeling is about making it tighter or richer: importance-weighted bounds (IWAE), normalizing flows for a more flexible $q$, and the observation that a hierarchy of latents with better bounds leads, in the limit, to diffusion models — whose training objective is itself an ELBO over a long chain of latents. The next track picks up exactly there.</p>
<p><strong>Speed vs. fidelity, made explicit.</strong> MCMC (last lesson) is asymptotically exact but sequential and slow; VI is fast, parallel, and differentiable but biased by $q$'s family and by reverse-KL's mode-seeking. Real systems mix them: VI to get close fast, then a few MCMC steps to correct, or MCMC to validate VI's error bars. Knowing which failure mode you're buying — slow-but-right vs. fast-but-narrow — is the practical skill this pair of lessons is really teaching.</p>`
};

/* ------------------------------------------------------------------ 15.5 */
window.LESSON_CONTENT["15.5"] = {
  subtitle: "Put a prior directly over functions instead of over weights. Given data, you get a mean curve plus an uncertainty band that pinches at the data and grows in the gaps.",

  aiMoment: String.raw`How do you tune the learning rate, weight decay, and architecture of an expensive model when each trial costs hours? You fit a <strong>Gaussian process</strong> to the results you have, and it tells you not just its best guess of the loss surface but <em>where it's most uncertain</em> — so you can try the most informative next configuration. That's Bayesian optimization, the standard engine of hyperparameter tuning and AutoML. GPs are also the exact limit of an infinitely-wide neural network, making them a theoretical lens on deep learning itself. And they're the natural finale of this track: a GP is Bayesian linear regression taken to an infinite basis via the kernel trick.`,

  plainEnglish: String.raw`Instead of choosing basis functions and putting a prior on their weights, a Gaussian process puts a prior straight on the <em>function</em>. You only have to say how much you expect nearby inputs to have similar outputs — that's the kernel. Feed it data and it returns, at every input, a best-guess value and an honest error bar: tiny where you have observations, wide where you don't.`,

  intuition: String.raw`A GP models the function's values at any set of points as one big joint Gaussian, with the correlation between $f(x)$ and $f(x')$ set by a kernel $k(x,x')$ — large when $x,x'$ are close, decaying to zero as they separate. Sampling from the prior gives smooth random curves. Conditioning on your data pins those curves down at the observed points (uncertainty collapses there) while leaving them free to wander far away (uncertainty returns to the prior). No weights, no basis choice — the kernel <em>is</em> the model.`,

  formalism: String.raw`Write $f\sim\mathcal{GP}(0,k)$: for any inputs, the function values are jointly Gaussian with covariance $k$. A common choice is the RBF kernel $k(x,x')=\sigma^2\exp\!\big(-\lVert x-x'\rVert^2/2\ell^2\big)$. With training inputs $X$, noisy targets $\mathbf y$ (noise variance $\sigma_n^2$), and test inputs $X_\*$, the posterior is Gaussian with
$$\boldsymbol\mu_\*=K_{\*X}\,(K_{XX}+\sigma_n^2 I)^{-1}\mathbf y,\qquad \Sigma_\*=K_{\*\*}-K_{\*X}\,(K_{XX}+\sigma_n^2 I)^{-1}K_{X\*},$$
where $K_{AB}$ is the kernel evaluated between sets $A$ and $B$. The predictive variance $\mathrm{diag}(\Sigma_\*)$ shrinks near data and returns to the prior $\sigma^2$ far from it.`,

  derivation: String.raw`<strong>The posterior is just Gaussian conditioning.</strong>
<ol>
<li><strong>Joint prior.</strong> Under the GP, the training targets $\mathbf y$ and a test value $f_\*$ are jointly Gaussian: $\begin{bmatrix}\mathbf y\\ f_\*\end{bmatrix}\sim\mathcal N\!\Big(\mathbf 0,\ \begin{bmatrix}K_{XX}+\sigma_n^2 I & K_{X\*}\\ K_{\*X} & K_{\*\*}\end{bmatrix}\Big).$</li>
<li><strong>Condition.</strong> For a joint Gaussian $\begin{bmatrix}\mathbf a\\ \mathbf b\end{bmatrix}\sim\mathcal N\big(\mathbf 0,\begin{bmatrix}A&C\\ C^\top&B\end{bmatrix}\big)$, the conditional is $\mathbf b\mid\mathbf a\sim\mathcal N\big(C^\top A^{-1}\mathbf a,\ B-C^\top A^{-1}C\big).$</li>
<li><strong>Substitute.</strong> With $\mathbf a=\mathbf y$, $\mathbf b=f_\*$, $A=K_{XX}+\sigma_n^2 I$, $C=K_{X\*}$, $B=K_{\*\*}$: mean $=K_{\*X}(K_{XX}+\sigma_n^2 I)^{-1}\mathbf y$ and variance $=K_{\*\*}-K_{\*X}(K_{XX}+\sigma_n^2 I)^{-1}K_{X\*}$ — exactly the boxed formulas.</li>
<li><strong>Why it's kernelized Bayesian regression.</strong> Compare to 15.2: the predictive there used $\phi(x)^\top S_N\phi(x')$, an inner product of features. Replace every feature inner product $\phi(x)^\top\phi(x')$ with a kernel $k(x,x')$ and the finite-basis formulas become these. The GP is Bayesian linear regression in a (possibly infinite-dimensional) feature space you never build explicitly — the kernel trick.</li>
</ol>`,

  code: [
    { label: "Gaussian process regression from scratch",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def rbf(A, B, l=0.4, s=1.0):                     # RBF kernel: nearby x -> highly correlated f
    d = A[:, None] - B[None, :]
    return s * np.exp(-0.5 * (d/l)**2)

X = np.array([-2, -1.5, -0.5, 0.3, 1.0, 2.2])    # training inputs
y = np.sin(X) + 0.05*rng.standard_normal(len(X)) # noisy targets
noise = 0.05**2
K = rbf(X, X) + noise*np.eye(len(X))             # kernel matrix at the data (+ noise)

xs = np.linspace(-3.5, 3.5, 200)                 # test inputs
Ks, Kss = rbf(xs, X), rbf(xs, xs)
L = np.linalg.cholesky(K)                         # stable solve of (K + σ²I)⁻¹ via Cholesky
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
mean = Ks @ alpha                                 # posterior mean  = K*X (K+σ²I)⁻¹ y
v = np.linalg.solve(L, Ks.T)
var = np.diag(Kss) - np.sum(v**2, 0)              # posterior variance (shrinks near data)
sd = np.sqrt(np.maximum(var, 0))
print(f"predictive sd near a data point = {sd[np.argmin(abs(xs-0.3))]:.3f},  far away = {sd[-1]:.3f}  (→ prior sd 1.0)")

plt.figure(figsize=(6.8, 4))
plt.fill_between(xs, mean-2*sd, mean+2*sd, color='#6a51a3', alpha=.18, label='±2σ')
plt.plot(xs, mean, color='#6a51a3', lw=2, label='GP mean')
plt.scatter(X, y, color='#d1495b', zorder=5, label='data')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(fontsize=8); plt.title('Gaussian process regression')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 200" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11.5">
  <rect x="18" y="30" width="316" height="146" rx="8" fill="#eef2f7" stroke="#6a51a3"/>
  <text x="176" y="52" text-anchor="middle" font-weight="700" fill="#4b2e83">the model: a prior over functions</text>
  <text x="176" y="80" text-anchor="middle" fill="#1f2a44">f ~ GP(0, k)</text>
  <text x="176" y="108" text-anchor="middle" fill="#1f2a44">k(x, x′) = σ² exp(−‖x−x′‖² / 2ℓ²)</text>
  <text x="176" y="134" text-anchor="middle" fill="#5a6b8c" font-size="10.5">nearby x ⇒ correlated f(x)</text>
  <text x="176" y="156" text-anchor="middle" fill="#5a6b8c" font-size="10.5">ℓ sets smoothness, σ² the amplitude</text>
  <rect x="358" y="30" width="344" height="146" rx="8" fill="#e7f0e8" stroke="#3a7d44"/>
  <text x="530" y="52" text-anchor="middle" font-weight="700" fill="#245030">posterior at test points X*</text>
  <text x="530" y="82" text-anchor="middle" fill="#1f2a44">mean = K*ₓ (Kₓₓ + σ²ₙI)⁻¹ y</text>
  <text x="530" y="112" text-anchor="middle" fill="#1f2a44">cov = K** − K*ₓ (Kₓₓ + σ²ₙI)⁻¹ Kₓ*</text>
  <text x="530" y="140" text-anchor="middle" fill="#8f2233" font-size="10.5">variance → 0 at data</text>
  <text x="530" y="158" text-anchor="middle" fill="#1c4e70" font-size="10.5">variance → prior σ² far from data</text>
</svg>`,

  keyPoints: [
    String.raw`A GP is a prior over <em>functions</em>: any finite set of function values is jointly Gaussian, with correlations set by the kernel $k(x,x')$.`,
    String.raw`The posterior is plain Gaussian conditioning: mean $=K_{\*X}(K_{XX}+\sigma_n^2 I)^{-1}\mathbf y$, variance $=K_{\*\*}-K_{\*X}(K_{XX}+\sigma_n^2 I)^{-1}K_{X\*}$.`,
    String.raw`Predictive uncertainty <strong>shrinks to (near) zero at observed points and grows back to the prior far away</strong> — calibrated 'I don't know' off-distribution, for free.`,
    String.raw`A GP is <strong>kernelized Bayesian linear regression</strong>: replace feature inner products $\phi(x)^\top\phi(x')$ with $k(x,x')$ and 15.2 becomes 15.5 — an infinite basis you never build.`,
    String.raw`Cost is $O(n^3)$ from inverting the $n\times n$ kernel matrix, which is why exact GPs suit small/medium data and sparse/inducing-point approximations exist for large $n$.`
  ],

  commonMistakes: [
    { wrong: "Forgetting the noise term σ²ₙI and inverting a singular (or ill-conditioned) kernel matrix.",
      why: String.raw`Without observation noise (or a tiny 'jitter'), $K_{XX}$ can be numerically singular when inputs are close, and the Cholesky fails. The $\sigma_n^2 I$ both models noise and regularizes the inverse — always include it.` },
    { wrong: "Picking the kernel lengthscale ℓ by eye and treating the uncertainty as gospel.",
      why: String.raw`Too-small $\ell$ overfits (band collapses everywhere and wiggles); too-large $\ell$ oversmooths. The right move is to learn $\ell$ (and $\sigma^2,\sigma_n^2$) by maximizing the marginal likelihood — otherwise the calibrated-looking band can be badly miscalibrated.` },
    { wrong: "Inverting (K + σ²I) directly with np.linalg.inv.",
      why: String.raw`Explicit inversion is less stable and slower than solving via the Cholesky factor $L$ (as the code does). Compute $L=\mathrm{chol}(K+\sigma_n^2 I)$ once and reuse it for both the mean solve and the variance — standard practice in every GP library.` }
  ],

  quiz: [
    { q: "A Gaussian process is a probability distribution over…",
      options: ["functions", "weight vectors", "single scalars", "datasets"], answer: 0,
      explain: String.raw`A GP places a prior directly over functions; any finite collection of function values is jointly Gaussian.` },
    { q: "At a noise-free training input, the GP posterior variance is approximately…",
      options: ["≈ 0 (the function is pinned to the observation)", "≈ the prior variance σ²",
                "infinite", "exactly 1"], answer: 0,
      explain: String.raw`Conditioning on an observation collapses uncertainty there, so $\mathrm{var}\to 0$ (up to the noise $\sigma_n^2$). Far from data it returns to the prior $\sigma^2$.` },
    { q: "The GP posterior mean K*ₓ(Kₓₓ + σ²ₙI)⁻¹y is the same as which model, kernelized?",
      options: ["Bayesian / ridge linear regression with feature inner products replaced by k(x,x′)",
                "k-means", "logistic regression", "a decision tree"], answer: 0,
      explain: String.raw`It's Bayesian linear regression (15.2) with $\phi(x)^\top\phi(x')$ swapped for the kernel — the kernel trick applied to Bayesian regression.` },
    { q: "The dominant computational cost of an exact GP with n training points is…",
      options: ["O(n³), from the n×n kernel matrix inverse/Cholesky", "O(n)", "O(n log n)", "O(1)"], answer: 0,
      explain: String.raw`Factorizing $K_{XX}+\sigma_n^2 I$ is $O(n^3)$ time and $O(n^2)$ memory — the reason exact GPs are used for small-to-medium data and approximated otherwise.` },
    { q: "Decreasing the RBF lengthscale ℓ makes the fitted functions…",
      options: ["wigglier — correlations decay faster, so far points are nearly independent",
                "smoother and flatter", "exactly linear", "independent of the data"], answer: 0,
      explain: String.raw`Small $\ell$ means $k$ drops off quickly, so only very nearby points are correlated; the posterior can change rapidly between them — wigglier fits and uncertainty that snaps back up between points.` }
  ],

  practice: [
    { level: "easy", prompt: "Set the lengthscale ℓ to 0.1 and to 2.0 and compare the fits. Describe the bias–variance trade-off you see.",
      solution: String.raw`$\ell=0.1$: wiggly interpolation, uncertainty collapses only right at points and shoots up between them (low bias, high variance). $\ell=2.0$: nearly a straight, smooth trend that may miss structure (high bias, low variance). The lengthscale is the GP's smoothness / complexity knob.` },
    { level: "easy", prompt: "Draw three sample functions from the GP posterior and overlay them on the ±2σ band.",
      solution: String.raw`Sample $f\sim\mathcal N(\boldsymbol\mu_\*,\Sigma_\*)$ using a Cholesky of $\Sigma_\*$ (add a tiny jitter for stability). The sampled curves pass through the data and spread apart in the gaps, filling the ±2σ envelope — a picture of 'plausible functions given the data'.` },
    { level: "med", prompt: "Swap the RBF kernel for a periodic kernel k(x,x′)=σ²exp(−2sin²(π|x−x′|/p)/ℓ²). What prior assumption does that encode?",
      solution: String.raw`It encodes that the function repeats with period $p$ — points one period apart are strongly correlated. GP priors are chosen by kernel: RBF for smooth trends, periodic for seasonality, linear for trends, and sums/products of kernels to combine assumptions (e.g. trend + seasonality).` },
    { level: "med", prompt: "Write the log marginal likelihood of a GP and explain how maximizing it selects ℓ, σ², σ²ₙ.",
      solution: String.raw`$\log p(\mathbf y\mid X)=-\tfrac12\mathbf y^\top(K+\sigma_n^2I)^{-1}\mathbf y-\tfrac12\log\lvert K+\sigma_n^2I\rvert-\tfrac{n}{2}\log 2\pi$. The first term rewards fit, the second penalizes model complexity (a built-in Occam's razor). Maximizing it over the kernel hyperparameters trades off fit vs. smoothness automatically — the principled alternative to eyeballing $\ell$.` },
    { level: "hard", prompt: "Use the GP's mean and variance to define an acquisition function for Bayesian optimization and explain the exploration–exploitation trade-off.",
      solution: String.raw`E.g. Upper Confidence Bound: pick the next $x$ maximizing $\mu(x)+\kappa\,\sigma(x)$. The mean term $\mu$ exploits (go where the model predicts good values); the variance term $\sigma$ explores (go where it's uncertain). $\kappa$ tunes the balance — the reason GPs are the classic surrogate for optimizing expensive black-box functions like model hyperparameters.` },
    { level: "hard", prompt: "State the sense in which an infinitely-wide neural network is a Gaussian process, and why that matters.",
      solution: String.raw`For a network with i.i.d. random weights and a hidden layer of width $\to\infty$, the Central Limit Theorem makes the pre-activation outputs jointly Gaussian, with a kernel (the 'NNGP kernel') determined by the architecture and nonlinearity (Neal, 1996). So a wide Bayesian net's prior over functions <em>is</em> a GP, and (via the Neural Tangent Kernel) gradient training of a wide net behaves like GP/kernel regression. It connects the two halves of this course: deep networks and Bayesian nonparametrics are the same object in a limit.` }
  ],

  deepDive: String.raw`<p><strong>The kernel is the model.</strong> A GP has no weights to interpret — all of its inductive bias lives in the kernel. RBF assumes smoothness; a periodic kernel assumes seasonality; a linear kernel recovers Bayesian linear regression; and you can add and multiply kernels to compose assumptions (trend × seasonality, etc.). Choosing and learning kernels is to GPs what choosing architectures is to neural nets — and marginal-likelihood maximization is the principled way to do it.</p>
<p><strong>The $O(n^3)$ wall, and how it's scaled.</strong> Exact GPs invert an $n\times n$ matrix, so they're happiest for $n$ up to a few thousand. Sparse GPs summarize the data with $m\ll n$ 'inducing points' ($O(nm^2)$), and structured-kernel and iterative (conjugate-gradient) methods push exact GPs to large $n$ on modern hardware. Knowing the cubic cost — and that it comes from the kernel inverse — tells you when to reach for these approximations.</p>
<p><strong>Where this track lands.</strong> You now have the full Bayesian toolkit: exact conjugate updates (15.1–15.2), sampling when exact fails (15.3), optimization-based approximation and the ELBO (15.4), and a nonparametric prior over functions (15.5). The ELBO in particular is the springboard to the next track — diffusion and generative models are, at heart, deep latent-variable models trained by maximizing a (very cleverly constructed) evidence lower bound.</p>`
};

/* ------------------------------------------------------------------ 15.E */
window.LESSON_CONTENT["15.E"] = {
  exam: true,
  intro: String.raw`Ten problems spanning conjugate inference, Bayesian regression, MCMC, variational inference, and Gaussian processes — roughly easy → hard. Budget about <strong>75 minutes</strong>. Reason on paper; a Python REPL is for <em>checking</em> (a conjugate posterior, an acceptance ratio, a GP variance), not for finding answers. Watch the recurring theme: every method is a different way to cope with the same intractable integral, the evidence $p(\mathcal D)$. Solutions and a rubric follow.`,
  problems: [
    { level: "easy", prompt: "Conjugate update. Prior Beta(3, 1). You observe 5 heads and 5 tails. (a) Posterior? (b) Posterior mean? (c) Prior mean, for comparison.",
      solution: String.raw`(a) $\mathrm{Beta}(3+5,\,1+5)=\mathrm{Beta}(8,6)$. (b) $8/14\approx0.571$. (c) Prior mean $3/4=0.75$; the data (frequency 0.5) pulled the posterior mean down between prior and data.` },
    { level: "easy", prompt: "MCMC essentials. Why can Metropolis–Hastings sample a posterior when you can only compute likelihood × prior (not the normalizer)?",
      solution: String.raw`The acceptance rule depends on the target only through the ratio $\tilde p(\theta')/\tilde p(\theta)$, in which the normalizing constant $Z$ cancels. So the unnormalized $\tilde p=\text{likelihood}\times\text{prior}$ is all you ever evaluate.` },
    { level: "med", prompt: "Bayesian linear regression. (a) Write the posterior precision Sₙ⁻¹. (b) Name the two terms of the predictive variance and say which grows away from data. (c) Which classical estimator equals the posterior mean?",
      solution: String.raw`(a) $S_N^{-1}=\alpha I+\beta\Phi^\top\Phi$ (prior precision + data precision). (b) $\beta^{-1}$ (observation noise, constant) $+\ \phi_\*^\top S_N\phi_\*$ (weight uncertainty, grows away from data). (c) Ridge regression with $\lambda=\alpha/\beta$.` },
    { level: "med", prompt: "Metropolis step. A symmetric proposal gives a candidate with p̃(θ′) = 0.3·p̃(θ). (a) Acceptance probability? (b) If instead p̃(θ′) = 2·p̃(θ)?",
      solution: String.raw`(a) $\min(1,0.3)=0.3$. (b) $\min(1,2)=1$ — an uphill move is always accepted. Downhill moves are accepted in proportion to the density drop; that willingness to go downhill is what lets the chain explore.` },
    { level: "med", prompt: "The ELBO. State the exact decomposition of log p(x), explain what maximizing the ELBO over q achieves, and give the inequality that names it.",
      solution: String.raw`$\log p(x)=\text{ELBO}(q)+\mathrm{KL}(q(z)\Vert p(z\mid x))$. Since $\log p(x)$ is fixed in $q$, maximizing the ELBO minimizes $\mathrm{KL}(q\Vert p(z\mid x))$, pushing $q$ toward the true posterior. Because KL $\ge 0$, $\text{ELBO}\le\log p(x)$ — the Evidence Lower BOund.` },
    { level: "med", prompt: "Reparameterization. Why sample z = μ + σε (ε ~ N(0,1)) rather than z ~ N(μ, σ²) directly when optimizing the ELBO by gradient ascent?",
      solution: String.raw`Sampling directly blocks the gradient from reaching $\mu,\sigma$ (the sampling op isn't differentiable in them). Writing $z=\mu+\sigma\varepsilon$ with $\varepsilon$ fixed makes $z$ a differentiable function of the parameters, so $\nabla_{\mu,\sigma}\mathbb E_q[\cdot]$ becomes an expectation of gradients — trainable by backprop, and much lower variance than the score-function estimator.` },
    { level: "hard", prompt: "Gaussian process. (a) Write the GP posterior mean and variance at test inputs X*. (b) What is the variance at a noise-free training point, and far from all data? (c) In what sense is a GP kernelized Bayesian regression?",
      solution: String.raw`(a) $\mu_\*=K_{\*X}(K_{XX}+\sigma_n^2I)^{-1}\mathbf y$, $\Sigma_\*=K_{\*\*}-K_{\*X}(K_{XX}+\sigma_n^2I)^{-1}K_{X\*}$. (b) ≈0 at a training point (uncertainty collapses); →prior $\sigma^2$ far away. (c) Replace feature inner products $\phi(x)^\top\phi(x')$ in Bayesian linear regression with $k(x,x')$; the GP is that model in an implicit, possibly infinite, feature space.` },
    { level: "hard", prompt: "Conjugacy derivation. Show that a Beta(a,b) prior times a Binomial(k heads, n−k tails) likelihood is a Beta, and give its parameters.",
      solution: String.raw`Prior $\propto\theta^{a-1}(1-\theta)^{b-1}$; likelihood $\propto\theta^{k}(1-\theta)^{n-k}$. Product $\propto\theta^{(a+k)-1}(1-\theta)^{(b+n-k)-1}$ — the kernel of $\mathrm{Beta}(a+k,\,b+n-k)$. The exponents add, and the (known) Beta normalizer replaces the intractable evidence integral.` },
    { level: "med", prompt: "Convergence. (a) What condition guarantees a Markov chain's stationary distribution is the target π? (b) Why must you discard burn-in and check effective sample size?",
      solution: String.raw`(a) Detailed balance: $\pi(\theta)T(\theta\to\theta')=\pi(\theta')T(\theta'\to\theta)$. (b) Early samples reflect the arbitrary start, not $\pi$ (discard burn-in), and successive samples are correlated, so the <em>effective</em> sample size (which sets Monte Carlo error) is much smaller than the raw count — always check trace/autocorrelation/$\hat R$.` },
    { level: "hard", prompt: "Method choice. You must do Bayesian inference in four settings: (i) a coin's bias, (ii) a 50-parameter hierarchical model, (iii) a deep latent-variable generative model on millions of images, (iv) tuning 5 expensive hyperparameters. Which of {conjugate, MCMC, variational, GP} fits each, and why?",
      solution: String.raw`(i) <strong>Conjugate</strong> (Beta–Bernoulli): exact and instant. (ii) <strong>MCMC</strong> (HMC/NUTS): moderate dimension, want accurate posterior, no closed form. (iii) <strong>Variational</strong> (a VAE-style ELBO): millions of latents demand fast, amortized, differentiable inference; accept some bias for scale. (iv) <strong>Gaussian process</strong> Bayesian optimization: few expensive evaluations, need calibrated uncertainty to pick the next trial. The through-line: pick the cheapest method whose accuracy your problem can tolerate.` }
  ],
  rubric: String.raw`<ul>
<li><strong>9–10:</strong> You can update a conjugate posterior in your head, derive the ELBO decomposition, and pick the right inference tool for a new problem. You're ready to read a Bayesian deep-learning or diffusion paper's math.</li>
<li><strong>7–8:</strong> Strong. Revisit whichever slipped: the predictive-variance split (15.2), the ELBO = evidence − KL identity (15.4), or GP conditioning (15.5).</li>
<li><strong>5–6:</strong> The concepts are landing but not yet fluent. Re-run the MCMC (15.3) and VI (15.4) code, changing one knob at a time until the behavior is predictable.</li>
<li><strong>Below 5:</strong> Rework the track in order. It's one idea — posterior ∝ likelihood × prior — attacked four ways; anchor on the conjugate case (15.1) and build out.</li>
</ul>`
};

