/* ============================================================
   TRACK 10 — Probability II — Continuous Distributions & Theorems
   10.1 PDFs/CDFs/Uniform · 10.2 Gaussian · 10.3 Exponential/Beta/Dirichlet ·
   10.4 Multivariate Gaussian · 10.5 Change of Variables & Reparameterization ·
   10.6 LLN & CLT · 10.E Track Exam
   ============================================================ */
(window.LESSON_CONTENT ||= {})["10.1"] = {
  subtitle: "For continuous outcomes, probability is area under a density — not the height.",

  aiMoment: String.raw`<p>Continuous randomness is everywhere in ML: the uniform noise that builds a dropout mask, the
  random crop offset in augmentation, the $u\sim\text{Uniform}(0,1)$ that seeds inverse-transform sampling. To reason
  about these you need <strong>probability density functions</strong> — where probability is the <em>area</em> under a
  curve, not a value at a point — and their running total, the <strong>CDF</strong>.</p>`,

  plainEnglish: String.raw`<p>For a continuous variable, any single exact value has probability zero — there are
  infinitely many. Instead a <strong>density</strong> $p(x)$ tells you probability <em>per unit length</em>, and the
  probability of landing in a range is the <strong>area</strong> under the density over that range. The
  <strong>CDF</strong> accumulates that area from the left.</p>`,

  intuition: String.raw`<p>Think of the density as how thickly probability is spread along the line. Thick regions are
  likely; thin ones rare. The chance of falling between $a$ and $b$ is the shaded area, and the total area is always 1.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 140" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A density with the area between a and b shaded">
    <line x1="20" y1="110" x2="240" y2="110" stroke="#94a3b8"/>
    <path d="M30,110 C80,110 90,35 130,35 C170,35 180,110 230,110 Z" fill="none" stroke="#4f46e5" stroke-width="2"/>
    <path d="M90,110 C100,72 112,52 130,52 C150,52 160,78 172,110 Z" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="86" y="124" font-size="11" fill="#64748b" font-family="sans-serif">a</text>
    <text x="170" y="124" font-size="11" fill="#64748b" font-family="sans-serif">b</text>
    <text x="112" y="92" font-size="10" fill="#4f46e5" font-family="sans-serif">P(a&lt;X&lt;b)</text>
  </svg>
  <figcaption>Probability is the shaded area; the height p(x) is density, not probability.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A <strong>probability density function</strong> $p(x)$ satisfies $p(x)\ge0$ and $\int_{-\infty}^{\infty}p(x)\,dx=1$.
  Probabilities are integrals: $P(a&lt;X&lt;b)=\int_a^b p(x)\,dx$. The <strong>CDF</strong> is
  $F(x)=P(X\le x)=\int_{-\infty}^{x}p(t)\,dt$, so $p=F'$ (Track 7.2). The <strong>uniform</strong> $\mathrm{U}(a,b)$ has
  constant density $p(x)=\tfrac{1}{b-a}$ on $[a,b]$.</p>`,

  derivation: String.raw`<p><strong>Mean and variance of the uniform $\mathrm{U}(a,b)$.</strong></p>
  <p><strong>Step 1 — mean.</strong> $E[X]=\int_a^b \frac{x}{b-a}\,dx=\frac{1}{b-a}\cdot\frac{b^2-a^2}{2}=\frac{a+b}{2}$
  (the midpoint, by symmetry).</p>
  <p><strong>Step 2 — second moment.</strong> $E[X^2]=\int_a^b\frac{x^2}{b-a}\,dx=\frac{b^3-a^3}{3(b-a)}=\frac{a^2+ab+b^2}{3}.$</p>
  <p><strong>Step 3 — variance.</strong> $\operatorname{Var}(X)=E[X^2]-\big(\tfrac{a+b}{2}\big)^2=\frac{a^2+ab+b^2}{3}-\frac{a^2+2ab+b^2}{4}=\frac{(b-a)^2}{12}.\;\blacksquare$</p>
  <hr class="soft">
  <p><strong>Inverse-transform sampling.</strong> The uniform CDF is $F(x)=\frac{x-a}{b-a}$; inverting, $F^{-1}(u)=a+u(b-a)$.
  More generally, for any distribution, if $U\sim\mathrm{U}(0,1)$ then $F^{-1}(U)$ has CDF $F$ — the trick behind drawing
  samples from a target using only uniform noise (Track 7.2 deep dive).</p>`,

  code: [
    { label: "A uniform density integrates to 1; sample it", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

a, b = 2.0, 5.0
x = rng.uniform(a, b, size=1_000_000)
print("sample mean ≈", round(float(x.mean()), 3), " vs (a+b)/2 =", (a+b)/2)
print("sample var  ≈", round(float(x.var()),  3), " vs (b-a)²/12 =", round((b-a)**2/12, 3))
` },
    { label: "Inverse-transform sampling from an exponential", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

lam = 1.5
u = rng.uniform(size=1_000_000)         # uniform noise
x = -np.log(1 - u) / lam                # F^{-1}(u) for Exponential(λ)
print("sample mean ≈", round(float(x.mean()), 3), " vs 1/λ =", round(1/lam, 3))
` }
  ],

  keyPoints: [
    "A density $p(x)\\ge0$ integrates to 1; probability is area, $P(a&lt;X&lt;b)=\\int_a^b p$.",
    "For continuous $X$, $P(X=x)=0$; only intervals have positive probability.",
    "Density is not probability — $p(x)$ can exceed 1 (it's probability per unit length).",
    "CDF $F(x)=\\int_{-\\infty}^x p$, and $p=F'$.",
    "Uniform $\\mathrm{U}(a,b)$: mean $\\tfrac{a+b}{2}$, variance $\\tfrac{(b-a)^2}{12}$; inverse-CDF turns uniform noise into any distribution."
  ],

  commonMistakes: [
    { wrong: "Reading $p(x)$ as a probability.", why: "It's a density — probability per unit length — and can exceed 1 (e.g. $\\mathrm{U}(0,0.5)$ has $p=2$). Only its integral over an interval is a probability." },
    { wrong: "Asking for $P(X=c)$ of a continuous variable.", why: "It's exactly 0. Continuous questions must be about ranges: $P(c-\\epsilon&lt;X&lt;c+\\epsilon)$." },
    { wrong: "Forgetting to normalize a hand-built density.", why: "Any nonnegative function becomes a density only after dividing by its integral (the normalizing constant); otherwise the 'probabilities' won't sum to 1." }
  ],

  quiz: [
    { q: "For a continuous $X$, $P(X=3)$ equals…", options: ["0", "$p(3)$", "1", "undefined"], answer: 0,
      explain: "A single point has zero area under the density; only intervals carry probability." },
    { q: "$\\mathrm{U}(0,4)$ has density…", options: ["$1/4$ on $[0,4]$", "$1$ on $[0,4]$", "$4$", "$1/2$"], answer: 0,
      explain: "Constant $1/(b-a)=1/4$ so the area is 1." },
    { q: "The variance of $\\mathrm{U}(0,6)$ is…", options: ["3", "36", "6", "1"], answer: 0,
      explain: "$(b-a)^2/12=36/12=3$." },
    { q: "The CDF relates to the density by…", options: ["$p=F'$", "$F=p'$", "$p=1-F$", "$F=\\int p'$"], answer: 0,
      explain: "Differentiate the accumulated area to recover the density (FTC, Track 7.2)." },
    { q: "Inverse-transform sampling maps $u\\sim\\mathrm{U}(0,1)$ to a target via…", options: ["$F^{-1}(u)$", "$F(u)$", "$p(u)$", "$1-u$"], answer: 0,
      explain: "Applying the inverse CDF to uniform noise yields samples with CDF $F$." }
  ],

  practice: [
    { level: "easy", prompt: "Give the density of $\\mathrm{U}(1,3)$ and $P(1.5&lt;X<2)$.", solution: "Density $=1/(3-1)=1/2$ on $[1,3]$. $P(1.5&lt;X<2)=(2-1.5)\\cdot\\tfrac12=0.25$." },
    { level: "med", prompt: "A density is $p(x)=cx$ on $[0,2]$, else 0. Find $c$.", solution: "Normalize: $\\int_0^2 cx\\,dx=c\\cdot\\tfrac{4}{2}=2c=1\\Rightarrow c=\\tfrac12$." },
    { level: "med", prompt: "Derive $F^{-1}$ for $\\mathrm{U}(a,b)$ and use it to sample.", solution: "$F(x)=\\frac{x-a}{b-a}$; set $u=F(x)$ and solve: $x=a+u(b-a)$. So $a+u(b-a)$ with $u\\sim\\mathrm{U}(0,1)$ samples $\\mathrm{U}(a,b)$." },
    { level: "hard", prompt: "AI task: to sample $\\mathrm{Exponential}(\\lambda)$ from uniform noise, derive $F^{-1}$ and state the formula.", solution: "The exponential CDF is $F(x)=1-e^{-\\lambda x}$ for $x\\ge0$. Invert: $u=1-e^{-\\lambda x}\\Rightarrow e^{-\\lambda x}=1-u\\Rightarrow x=-\\tfrac{1}{\\lambda}\\ln(1-u)$. So $x=-\\ln(1-u)/\\lambda$ (or equivalently $-\\ln(u)/\\lambda$ since $1-u$ is also uniform) transforms uniform noise into exponential samples — exactly how libraries generate waiting times." }
  ],

  deepDive: String.raw`<p><strong>Density vs probability, and why the distinction matters for likelihoods.</strong></p>
  <p>The fact that a density isn't a probability trips up more than beginners — it shapes how continuous likelihoods
  behave. Because $p(x)$ can be arbitrarily large, a <strong>log-likelihood</strong> $\log p(x)$ can be positive, and it
  is <em>not</em> bounded above the way a discrete log-probability (always $\le0$) is. So when you train a continuous
  density model (a VAE decoder, a normalizing flow, a diffusion model) by maximizing log-likelihood, "bits per dimension"
  can look strange, and comparisons across models must fix the same base measure — a density is only defined relative to
  how you measure volume.</p>
  <p>This is also why <strong>change of variables</strong> (Lesson 10.5) carries a Jacobian factor: transform the
  variable and the density must be rescaled by how much the transform stretches space, precisely so that areas — the
  actual probabilities — are preserved. Keep the mental rule "probability = area, density = height," and continuous
  probability stops being paradoxical: the point mass that seems to vanish is really just spread infinitesimally thin,
  and every real question is about how that area accumulates.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["10.2"] = {
  subtitle: "The bell curve: two numbers — mean and variance — and it shows up everywhere.",

  aiMoment: String.raw`<p>Weight initialization (Xavier, He) draws from a carefully scaled <strong>Gaussian</strong>; the
  noise in a VAE's latent and in diffusion's forward process is Gaussian; and the Central Limit Theorem (Lesson 10.6)
  makes Gaussians the default model for aggregated noise. Its two parameters — mean $\mu$ and variance $\sigma^2$ — and
  its standardization rule are among the most-used facts in all of ML.</p>`,

  plainEnglish: String.raw`<p>The <strong>Normal (Gaussian)</strong> distribution is the symmetric bell curve. Its
  <strong>mean</strong> $\mu$ sets where it's centered and its <strong>standard deviation</strong> $\sigma$ how wide it
  spreads. Any Gaussian can be shifted and scaled into the <em>standard</em> one, $\mathcal N(0,1)$.</p>`,

  intuition: String.raw`<p>Almost all the mass sits within a few standard deviations of the mean: about 68% within
  $\pm1\sigma$, 95% within $\pm2\sigma$, 99.7% within $\pm3\sigma$. That "68–95–99.7" rule is the Gaussian's fingerprint.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 140" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bell curve with 68-95-99.7 bands">
    <line x1="20" y1="110" x2="240" y2="110" stroke="#94a3b8"/>
    <path d="M30,110 C80,110 95,30 130,30 C165,30 180,110 230,110 Z" fill="#eef0ff" stroke="#4f46e5" stroke-width="2"/>
    <line x1="130" y1="110" x2="130" y2="30" stroke="#dc2626" stroke-dasharray="3 3"/>
    <line x1="103" y1="110" x2="103" y2="66" stroke="#94a3b8"/><line x1="157" y1="110" x2="157" y2="66" stroke="#94a3b8"/>
    <text x="120" y="124" font-size="10" fill="#dc2626" font-family="sans-serif">μ</text>
    <text x="88" y="124" font-size="10" fill="#64748b" font-family="sans-serif">−σ</text>
    <text x="150" y="124" font-size="10" fill="#64748b" font-family="sans-serif">+σ</text>
    <text x="94" y="56" font-size="10" fill="#4f46e5" font-family="sans-serif">68% within ±1σ</text>
  </svg>
  <figcaption>The bell: centered at μ, width σ, with the 68–95–99.7 rule.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>Normal</strong> density is</p>
  $$p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),\qquad X\sim\mathcal N(\mu,\sigma^2).$$
  <p>The constant $\sqrt{2\pi\sigma^2}$ is the Gaussian integral (Track 7.4). <strong>Standardization:</strong>
  $Z=\dfrac{X-\mu}{\sigma}\sim\mathcal N(0,1)$. A linear function of a Gaussian is Gaussian: $aX+b\sim\mathcal N(a\mu+b,\,a^2\sigma^2)$.</p>`,

  derivation: String.raw`<p><strong>Standardization: why $Z=(X-\mu)/\sigma$ is standard normal.</strong></p>
  <p><strong>Step 1 — it's a linear transform</strong> of $X$, and linear transforms of a Gaussian are Gaussian, so $Z$
  is normal; we just need its mean and variance.</p>
  <p><strong>Step 2 — mean.</strong> $E[Z]=E\big[\tfrac{X-\mu}{\sigma}\big]=\tfrac{E[X]-\mu}{\sigma}=\tfrac{\mu-\mu}{\sigma}=0.$</p>
  <p><strong>Step 3 — variance.</strong> $\operatorname{Var}(Z)=\operatorname{Var}\big(\tfrac{X-\mu}{\sigma}\big)=\tfrac{1}{\sigma^2}\operatorname{Var}(X)=\tfrac{\sigma^2}{\sigma^2}=1.$</p>
  <p><strong>Step 4 — conclude:</strong> $Z\sim\mathcal N(0,1).\;\blacksquare$ This is why a single standard-normal table
  (or one $\texttt{erf}$) answers questions about <em>every</em> Gaussian: shift by $\mu$, scale by $\sigma$.</p>`,

  code: [
    { label: "Sample a Gaussian; check 68–95–99.7", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

mu, sigma = 3.0, 2.0
x = rng.normal(mu, sigma, size=1_000_000)
for k in [1, 2, 3]:
    frac = np.mean(np.abs(x - mu) < k*sigma)
    print(f"within ±{k}σ: {frac:.3f}")     # ≈ 0.68, 0.95, 0.997
` },
    { label: "Standardization maps any Gaussian to N(0,1)", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

x = rng.normal(10.0, 4.0, size=1_000_000)
z = (x - 10.0) / 4.0                        # standardize
print("z mean ≈", round(float(z.mean()), 3), " z std ≈", round(float(z.std()), 3))
` }
  ],

  keyPoints: [
    "$\\mathcal N(\\mu,\\sigma^2)$ is the bell curve centered at $\\mu$ with spread $\\sigma$.",
    "Standardize with $Z=(X-\\mu)/\\sigma\\sim\\mathcal N(0,1)$ — one table fits all Gaussians.",
    "68–95–99.7 rule: mass within 1, 2, 3 standard deviations.",
    "Linear maps of Gaussians are Gaussian: $aX+b\\sim\\mathcal N(a\\mu+b,a^2\\sigma^2)$.",
    "The normalizing constant $\\sqrt{2\\pi\\sigma^2}$ is the Gaussian integral (Track 7.4)."
  ],

  commonMistakes: [
    { wrong: "Confusing variance $\\sigma^2$ and standard deviation $\\sigma$.", why: "The density uses $\\sigma^2$ in the exponent's denominator; the 68–95–99.7 bands use $\\sigma$. Mixing them mis-scales everything (e.g. He init specifies a variance, not a std)." },
    { wrong: "Assuming a sum of Gaussians standardizes trivially.", why: "For <em>independent</em> Gaussians variances add: $\\mathcal N(\\mu_1,\\sigma_1^2)+\\mathcal N(\\mu_2,\\sigma_2^2)=\\mathcal N(\\mu_1+\\mu_2,\\sigma_1^2+\\sigma_2^2)$. With correlation there's a covariance term (Lesson 9.5)." },
    { wrong: "Treating any bell-shaped data as exactly Gaussian.", why: "Real data has heavier tails; assuming Gaussianity underestimates rare events. Check with a QQ-plot before trusting Gaussian tail probabilities." }
  ],

  quiz: [
    { q: "For $\\mathcal N(0,1)$, roughly what fraction is within $\\pm2$?", options: ["95%", "68%", "99.7%", "50%"], answer: 0,
      explain: "The 68–95–99.7 rule: ±2σ captures about 95%." },
    { q: "If $X\\sim\\mathcal N(5,9)$, its standard deviation is…", options: ["3", "9", "81", "$\\sqrt5$"], answer: 0,
      explain: "Variance is 9, so $\\sigma=\\sqrt9=3$." },
    { q: "Standardizing $X\\sim\\mathcal N(10,4)$: $Z=$", options: ["$(X-10)/2$", "$(X-10)/4$", "$(X-4)/10$", "$X/10$"], answer: 0,
      explain: "$\\sigma=\\sqrt4=2$, so $Z=(X-\\mu)/\\sigma=(X-10)/2$." },
    { q: "$2X+1$ where $X\\sim\\mathcal N(3,4)$ is distributed as…", options: ["$\\mathcal N(7,16)$", "$\\mathcal N(7,4)$", "$\\mathcal N(7,8)$", "$\\mathcal N(4,16)$"], answer: 0,
      explain: "Mean $2\\cdot3+1=7$; variance $2^2\\cdot4=16$." },
    { q: "A value 3σ above the mean is…", options: ["very rare (~0.15% in the upper tail)", "common", "impossible", "the mean"], answer: 0,
      explain: "Beyond +3σ is about 0.15% of the mass — the '3-sigma event' idea." }
  ],

  practice: [
    { level: "easy", prompt: "$X\\sim\\mathcal N(100,225)$. Give its mean and standard deviation.", solution: "Mean 100, $\\sigma=\\sqrt{225}=15$." },
    { level: "med", prompt: "Roughly what fraction of $\\mathcal N(50,100)$ lies between 40 and 60?", solution: "$\\sigma=10$, so 40 and 60 are $\\mu\\pm1\\sigma$: about 68%." },
    { level: "med", prompt: "If $X\\sim\\mathcal N(2,1)$ and $Y\\sim\\mathcal N(3,4)$ are independent, what is the distribution of $X+Y$?", solution: "Sum of independent Gaussians: mean $2+3=5$, variance $1+4=5$, so $X+Y\\sim\\mathcal N(5,5)$." },
    { level: "hard", prompt: "AI task: He initialization sets weights $\\sim\\mathcal N(0,\\,2/n_{\\text{in}})$ for a ReLU layer with $n_{\\text{in}}$ inputs. Explain why the variance scales as $1/n_{\\text{in}}$.", solution: "A pre-activation is $z=\\sum_{i=1}^{n_{\\text{in}}} w_i x_i$. With independent zero-mean weights of variance $\\sigma_w^2$ and inputs of variance $\\sigma_x^2$, $\\operatorname{Var}(z)=n_{\\text{in}}\\sigma_w^2\\sigma_x^2$ (variances add over independent terms, Lesson 9.5). To keep $\\operatorname{Var}(z)\\approx\\operatorname{Var}(x)$ across layers (so signals neither vanish nor explode), you need $n_{\\text{in}}\\sigma_w^2\\approx1$, i.e. $\\sigma_w^2\\propto1/n_{\\text{in}}$. The extra factor 2 in He init compensates for ReLU zeroing ~half the inputs, halving the effective variance. So the Gaussian's variance is chosen to preserve signal scale through depth." }
  ],

  deepDive: String.raw`<p><strong>Why the Gaussian, of all shapes?</strong></p>
  <p>The Gaussian isn't an arbitrary choice; several deep principles single it out. By the <strong>Central Limit
  Theorem</strong> (Lesson 10.6), sums and averages of many independent effects converge to it — so any quantity built
  from lots of small contributions (measurement noise, aggregated features, initialization sums) is approximately
  Gaussian. By the <strong>maximum-entropy principle</strong>, among all distributions with a given mean and variance,
  the Gaussian is the one that assumes the <em>least</em> beyond those two facts — the most honest default when you only
  know the first two moments. And it's uniquely convenient: closed under linear maps, conditioning, and marginalization,
  with a log-density that's just a quadratic (Track 5's positive-definite forms), which is why Gaussian assumptions make
  so much math tractable.</p>
  <p>Those properties compound in ML. Least-squares regression is exactly maximum likelihood under Gaussian noise (the
  quadratic loss is the Gaussian log-density, Track 11). Weight initializations use Gaussians because their variance
  algebra (this lesson's practice) cleanly controls signal propagation. VAEs and diffusion models use Gaussians for
  their latent and noise because the reparameterization trick (Lesson 10.5) makes them differentiable and their KL
  divergences have closed forms (Track 12). The bell curve earns its ubiquity — it's the distribution that shows up
  when you add things up, assume the least, and want the algebra to work out.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["10.3"] = {
  subtitle: "Waiting times, a distribution over a probability, and a distribution over probability vectors.",

  aiMoment: String.raw`<p>The <strong>exponential</strong> models waiting times (between events, between failures). The
  <strong>Beta</strong> is a distribution <em>over a probability</em> — the natural prior for a click-through rate or a
  coin's bias, and the engine of Thompson sampling and Bayesian A/B testing. The <strong>Dirichlet</strong> generalizes
  Beta to a probability <em>vector</em> on the simplex — it generates the topic mixtures in LDA and priors over softmax
  outputs. Together they're the conjugate-prior toolkit.</p>`,

  plainEnglish: String.raw`<p>The <strong>exponential</strong> is the time until the next random event, and it's
  "memoryless" — having waited a while tells you nothing about how much longer. The <strong>Beta</strong> is a bump on
  $[0,1]$: a distribution describing how uncertain you are about a probability. The <strong>Dirichlet</strong> does the
  same for a whole set of probabilities that must add to 1.</p>`,

  intuition: String.raw`<p>Beta's shape is set by two counts $(\alpha,\beta)$ — roughly "prior heads and tails." Equal
  and small is flat (know nothing); large means confident and peaked; lopsided leans toward 0 or 1. It's the shape of
  your belief about a probability.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Beta distribution shapes on the unit interval">
    <line x1="20" y1="105" x2="240" y2="105" stroke="#94a3b8"/>
    <line x1="20" y1="105" x2="240" y2="105" stroke="#16a34a" stroke-width="2"/>
    <path d="M20,60 C70,20 110,18 130,30 C150,42 160,80 175,105" fill="none" stroke="#4f46e5" stroke-width="2"/>
    <path d="M85,105 C100,80 110,42 130,30 C150,18 190,20 240,60" fill="none" stroke="#dc2626" stroke-width="2"/>
    <text x="146" y="100" font-size="9" fill="#16a34a" font-family="sans-serif">Beta(1,1) flat</text>
    <text x="42" y="28" font-size="9" fill="#4f46e5" font-family="sans-serif">Beta(2,5)</text>
    <text x="190" y="34" font-size="9" fill="#dc2626" font-family="sans-serif">Beta(5,2)</text>
    <text x="18" y="122" font-size="9" fill="#64748b" font-family="sans-serif">0</text>
    <text x="232" y="122" font-size="9" fill="#64748b" font-family="sans-serif">1</text>
  </svg>
  <figcaption>Beta describes uncertainty about a probability: flat = ignorant, peaked = confident.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Exponential:</strong> $p(x)=\lambda e^{-\lambda x}$ for $x\ge0$, mean $1/\lambda$.
  <strong>Beta:</strong> $p(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}$ on $[0,1]$, mean $\dfrac{\alpha}{\alpha+\beta}$.
  <strong>Dirichlet:</strong> $p(\boldsymbol\theta)\propto\prod_k\theta_k^{\alpha_k-1}$ on the simplex ($\theta_k\ge0$,
  $\sum_k\theta_k=1$) — the multivariate Beta. Beta and Dirichlet are <strong>conjugate priors</strong> for the
  Bernoulli/binomial and categorical/multinomial likelihoods.</p>`,

  derivation: String.raw`<p><strong>Part 1 — the exponential is memoryless.</strong> Its survival function is
  $P(X>t)=\int_t^\infty\lambda e^{-\lambda s}\,ds=e^{-\lambda t}$. Then</p>
  <p>$P(X>s+t\mid X>s)=\dfrac{P(X>s+t)}{P(X>s)}=\dfrac{e^{-\lambda(s+t)}}{e^{-\lambda s}}=e^{-\lambda t}=P(X>t).\;\blacksquare$
  Having already waited $s$ doesn't change the remaining wait — the defining property of the exponential.</p>
  <hr class="soft">
  <p><strong>Part 2 — Beta is conjugate to the Bernoulli.</strong> Prior $p(\theta)=\mathrm{Beta}(\alpha,\beta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}$.
  Observe $k$ heads in $n$ flips: likelihood $\propto\theta^{k}(1-\theta)^{n-k}$.</p>
  <p><strong>Step 1 — Bayes (Track 9.3):</strong> posterior $\propto$ likelihood $\times$ prior
  $=\theta^{k}(1-\theta)^{n-k}\cdot\theta^{\alpha-1}(1-\theta)^{\beta-1}$.</p>
  <p><strong>Step 2 — collect exponents:</strong> $=\theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}$.</p>
  <p><strong>Step 3 — recognize the form:</strong> that's $\mathrm{Beta}(\alpha+k,\ \beta+n-k)$. $\blacksquare$ Plain
  English: update a Beta belief by just adding the observed heads to $\alpha$ and tails to $\beta$ — Bayesian learning
  becomes counting.</p>`,

  code: [
    { label: "Beta posterior: Bayesian updating is counting", src: String.raw`
import numpy as np

alpha, beta = 1.0, 1.0                  # uniform prior on the coin's bias
flips = [1,1,0,1,1,1,0,1,1,1]           # observed heads(1)/tails(0)
alpha += sum(f==1 for f in flips)       # add heads
beta  += sum(f==0 for f in flips)       # add tails
print("posterior Beta(α,β) =", (alpha, beta))
print("posterior mean bias =", round(alpha/(alpha+beta), 3))   # ~0.75
` },
    { label: "Exponential mean and Dirichlet samples sum to 1", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

x = rng.exponential(scale=1/1.5, size=1_000_000)   # rate λ=1.5 -> mean 1/λ
print("exp sample mean ≈", round(float(x.mean()), 3), " vs 1/λ =", round(1/1.5, 3))

d = rng.dirichlet([2., 3., 5.], size=5)            # 5 probability vectors
print("Dirichlet samples (each row sums to 1):\n", np.round(d, 3))
print("row sums:", np.round(d.sum(axis=1), 6))
` }
  ],

  keyPoints: [
    "Exponential $\\lambda e^{-\\lambda x}$: waiting times, memoryless, mean $1/\\lambda$.",
    "Beta$(\\alpha,\\beta)$ is a distribution over a probability in $[0,1]$, mean $\\alpha/(\\alpha+\\beta)$.",
    "Dirichlet is the multivariate Beta — a distribution over probability vectors on the simplex.",
    "Beta/Dirichlet are conjugate priors: posterior stays Beta/Dirichlet, so updating is adding counts.",
    "Conjugacy powers Thompson sampling, Bayesian A/B testing, and LDA topic models."
  ],

  commonMistakes: [
    { wrong: "Confusing the exponential's rate $\\lambda$ with its mean.", why: "The mean is $1/\\lambda$, not $\\lambda$. A high rate means short waits (small mean). NumPy's <code>exponential</code> even takes the <em>scale</em> $=1/\\lambda$." },
    { wrong: "Treating a Beta sample as a probability of a specific event.", why: "A Beta value <em>is</em> a candidate probability (the bias $\\theta$); the distribution expresses uncertainty <em>about</em> that probability — a probability of a probability." },
    { wrong: "Using a Dirichlet with all $\\alpha_k<1$ expecting a spread-out vector.", why: "$\\alpha_k<1$ pushes mass to the simplex corners (sparse, near one-hot vectors); $\\alpha_k>1$ concentrates near the center (uniform mixtures). Pick $\\alpha$ for the sparsity you want." }
  ],

  quiz: [
    { q: "An exponential with rate $\\lambda=4$ has mean…", options: ["0.25", "4", "16", "0.5"], answer: 0,
      explain: "Mean $=1/\\lambda=1/4=0.25$." },
    { q: "The mean of $\\mathrm{Beta}(3,1)$ is…", options: ["0.75", "0.25", "0.5", "3"], answer: 0,
      explain: "$\\alpha/(\\alpha+\\beta)=3/4=0.75$." },
    { q: "Start with $\\mathrm{Beta}(1,1)$; observe 7 heads, 3 tails. Posterior is…", options: ["$\\mathrm{Beta}(8,4)$", "$\\mathrm{Beta}(7,3)$", "$\\mathrm{Beta}(1,1)$", "$\\mathrm{Beta}(4,8)$"], answer: 0,
      explain: "Add heads to $\\alpha$, tails to $\\beta$: $\\mathrm{Beta}(1+7,\\,1+3)=\\mathrm{Beta}(8,4)$." },
    { q: "The 'memoryless' property means…", options: ["$P(X>s+t\\mid X>s)=P(X>t)$", "$X$ has no variance", "$X$ is discrete", "$P(X=x)=0$"], answer: 0,
      explain: "Past waiting time doesn't change the remaining wait — unique to the exponential (continuous case)." },
    { q: "A Dirichlet distribution lives on…", options: ["the probability simplex ($\\theta_k\\ge0$, $\\sum\\theta_k=1$)", "all of $\\mathbb{R}^k$", "$[0,1]$", "the integers"], answer: 0,
      explain: "It's a distribution over probability vectors — the simplex." }
  ],

  practice: [
    { level: "easy", prompt: "Give the mean of $\\mathrm{Beta}(2,8)$ and say which way it leans.", solution: "$2/(2+8)=0.2$ — leans toward 0 (more prior 'tails' than 'heads')." },
    { level: "med", prompt: "You model a coin with a $\\mathrm{Beta}(2,2)$ prior and see 3 heads, 1 tail. Give the posterior and its mean.", solution: "$\\mathrm{Beta}(2+3,\\,2+1)=\\mathrm{Beta}(5,3)$; mean $5/8=0.625$. The prior's pseudo-counts pull the raw estimate $3/4$ toward $0.5$." },
    { level: "med", prompt: "Show the exponential's survival function is $e^{-\\lambda t}$.", solution: "$P(X>t)=\\int_t^\\infty\\lambda e^{-\\lambda s}\\,ds=[-e^{-\\lambda s}]_t^\\infty=0-(-e^{-\\lambda t})=e^{-\\lambda t}$." },
    { level: "hard", prompt: "AI task: Thompson sampling for a two-armed bandit keeps a Beta per arm. Describe one round and why conjugacy makes it efficient.", solution: "Each arm $i$ has a $\\mathrm{Beta}(\\alpha_i,\\beta_i)$ over its reward probability. One round: (1) sample a candidate probability $\\hat\\theta_i$ from each arm's Beta; (2) pull the arm with the largest $\\hat\\theta_i$; (3) observe reward $r\\in\\{0,1\\}$ and update that arm — $\\alpha_i{+}{=}r$, $\\beta_i{+}{=}1-r$. Conjugacy makes the posterior update a single integer increment (no integration), so the whole algorithm is just sampling from Betas and adding counts. Sampling naturally balances exploration (wide Betas get optimistic draws) and exploitation (confident arms are picked more), which is why Thompson sampling is both simple and near-optimal." }
  ],

  deepDive: String.raw`<p><strong>Conjugacy: why these particular distributions, and where it breaks.</strong></p>
  <p>A prior is <em>conjugate</em> to a likelihood when the posterior belongs to the same family — so Bayesian updating
  reduces to updating parameters (here, adding counts) instead of computing an intractable integral. The
  Beta–Bernoulli, Dirichlet–Multinomial, and Gaussian–Gaussian pairs are the classic conjugate families, and they're
  beloved because they make sequential Bayesian learning $O(1)$ per observation. This is why they anchor bandit
  algorithms, online A/B testing, and the collapsed Gibbs samplers of topic models: the math stays closed-form no matter
  how much data streams in.</p>
  <p>The catch is that conjugacy is a luxury of simple models. The moment you want a neural network's weights as the
  "parameter," no conjugate prior exists, and the posterior has no closed form — which is precisely why <strong>Bayesian
  deep learning</strong> needs approximations: variational inference (turn the integral into an optimization, Track 12),
  Monte Carlo (sample it, Track 7), or Laplace approximations (fit a Gaussian at the mode). So the Beta and Dirichlet are
  more than textbook distributions; they're the tractable ideal that approximate inference is trying to recover at scale.
  Understanding them tells you both what clean Bayesian updating looks like and why the deep-learning version has to work
  so much harder.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["10.4"] = {
  subtitle: "The bell curve in many dimensions: a mean vector and a covariance matrix that tilts the ellipse.",

  aiMoment: String.raw`<p>A VAE's latent prior is a <strong>multivariate Gaussian</strong> $\mathcal N(\mathbf 0,I)$;
  Gaussian processes are infinite-dimensional versions; whitening and Mahalanobis distance use its covariance. Its
  contour lines are ellipses whose axes are the eigenvectors of the covariance matrix — the very same principal
  components from PCA (Track 5). This distribution ties probability, geometry, and linear algebra into one object.</p>`,

  plainEnglish: String.raw`<p>The <strong>multivariate Gaussian</strong> is a bell spread over several dimensions at
  once. A <strong>mean vector</strong> says where it's centered and a <strong>covariance matrix</strong> says how it's
  stretched and tilted. Its equal-probability contours are ellipses, aligned to the directions the data varies most.</p>`,

  intuition: String.raw`<p>In 2-D, points cluster in a tilted elliptical cloud. The ellipse's long axis is the direction
  of greatest variance (the top eigenvector of the covariance), the short axis the least. Circular contours mean
  uncorrelated, equal-variance features; a tilt means correlation.</p>
  <figure class="figure">
  <svg viewBox="0 0 240 160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Tilted elliptical contours of a 2D Gaussian">
    <line x1="20" y1="130" x2="220" y2="130" stroke="#e2e8f0"/>
    <line x1="60" y1="20" x2="60" y2="150" stroke="#e2e8f0"/>
    <g transform="rotate(-28 130 80)">
    <ellipse cx="130" cy="80" rx="78" ry="34" fill="none" stroke="#4f46e5"/>
    <ellipse cx="130" cy="80" rx="52" ry="22" fill="none" stroke="#4f46e5"/>
    <ellipse cx="130" cy="80" rx="26" ry="11" fill="none" stroke="#4f46e5"/>
    <line x1="130" y1="80" x2="208" y2="80" stroke="#dc2626" stroke-width="2" marker-end="url(#mv1)"/>
    <line x1="130" y1="80" x2="130" y2="46" stroke="#0d9488" stroke-width="2" marker-end="url(#mv2)"/>
    </g>
    <text x="140" y="28" font-size="10" fill="#dc2626" font-family="sans-serif">top eigenvector</text>
    <defs>
      <marker id="mv1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#dc2626"/></marker>
      <marker id="mv2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#0d9488"/></marker>
    </defs>
  </svg>
  <figcaption>Contours are ellipses; their axes are the covariance's eigenvectors, radii ∝ √eigenvalues.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $\mathbf x\in\mathbb{R}^d$ with mean $\boldsymbol\mu$ and covariance $\Sigma$ (symmetric PD):</p>
  $$p(\mathbf x)=\frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\!\left(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\right).$$
  <p>The exponent is $-\tfrac12$ the squared <strong>Mahalanobis distance</strong> $(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$.
  Contours of constant density are ellipses with axes along the eigenvectors of $\Sigma$ and radii $\propto\sqrt{\lambda_i}$.
  Marginals and conditionals of a Gaussian are Gaussian.</p>`,

  derivation: String.raw`<p><strong>Why the contours are ellipses aligned to $\Sigma$'s eigenvectors.</strong></p>
  <p><strong>Step 1 — a contour is constant Mahalanobis.</strong> The density is constant where the exponent is constant,
  i.e. $(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)=c.$ Center coordinates: let $\mathbf y=\mathbf x-\boldsymbol\mu$,
  so $\mathbf y^\top\Sigma^{-1}\mathbf y=c$.</p>
  <p><strong>Step 2 — diagonalize.</strong> $\Sigma=Q\Lambda Q^\top$ (spectral theorem, Track 5.2), so
  $\Sigma^{-1}=Q\Lambda^{-1}Q^\top$. In the eigenbasis coordinates $\mathbf z=Q^\top\mathbf y$,
  $\mathbf y^\top\Sigma^{-1}\mathbf y=\sum_i \dfrac{z_i^2}{\lambda_i}=c.$</p>
  <p><strong>Step 3 — read the shape.</strong> $\sum_i z_i^2/\lambda_i=c$ is the equation of an <strong>ellipse</strong>
  with semi-axes $\sqrt{c\lambda_i}$ along the eigenvectors $\mathbf q_i$. $\blacksquare$ Plain English: the covariance's
  eigenvectors are the ellipse's axes and its eigenvalues set their lengths — the Gaussian's geometry <em>is</em> the
  spectral decomposition of $\Sigma$, which is exactly why PCA falls out of it.</p>`,

  code: [
    { label: "Sample a correlated 2-D Gaussian; recover its axes", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

mu = np.array([0., 0.])
Sigma = np.array([[3.0, 1.5],
                  [1.5, 1.0]])
X = rng.multivariate_normal(mu, Sigma, size=200000)

emp = np.cov(X.T)                        # empirical covariance ≈ Sigma
vals, vecs = np.linalg.eigh(emp)         # axes of the ellipse
print("empirical Σ:\n", np.round(emp, 2))
print("eigenvalues (axis variances):", np.round(vals, 2))
print("top eigenvector (long axis)  :", np.round(vecs[:, -1], 3))
` },
    { label: "Mahalanobis distance accounts for correlation", src: String.raw`
import numpy as np

Sigma = np.array([[4., 0.],
                  [0., 1.]])            # x varies 4x more than y
Si = np.linalg.inv(Sigma)
for p in ([2., 0.], [0., 2.]):
    p = np.array(p)
    d2 = p @ Si @ p                      # squared Mahalanobis distance
    print(f"point {p.tolist()}: Euclidean={np.linalg.norm(p):.2f}  Mahalanobis={np.sqrt(d2):.2f}")
# same Euclidean distance, different 'how surprising' after scaling by variance
` }
  ],

  keyPoints: [
    "$\\mathcal N(\\boldsymbol\\mu,\\Sigma)$: mean vector centers it, covariance $\\Sigma$ stretches and tilts it.",
    "The exponent is $-\\tfrac12$ the squared Mahalanobis distance $(\\mathbf x-\\boldsymbol\\mu)^\\top\\Sigma^{-1}(\\mathbf x-\\boldsymbol\\mu)$.",
    "Contours are ellipses; axes = eigenvectors of $\\Sigma$, radii $\\propto\\sqrt{\\lambda_i}$ — PCA's directions.",
    "Marginals and conditionals of a Gaussian are Gaussian (closed under both).",
    "$\\det\\Sigma$ in the normalizer measures the ellipse's volume."
  ],

  commonMistakes: [
    { wrong: "Using Euclidean distance in a correlated Gaussian.", why: "Points equally far in Euclidean terms can be very differently likely; Mahalanobis distance ($\\Sigma^{-1}$-weighted) is the right 'how many standard deviations' measure." },
    { wrong: "Assuming a singular $\\Sigma$ still gives a valid density.", why: "If $\\Sigma$ is not full rank (collinear dimensions), $\\det\\Sigma=0$ and $\\Sigma^{-1}$ doesn't exist — the mass collapses onto a lower-dimensional subspace and the density formula breaks." },
    { wrong: "Thinking uncorrelated components of a multivariate Gaussian might still be dependent.", why: "For <em>jointly Gaussian</em> variables, zero covariance <em>does</em> imply independence — a special property (the general counterexample $Y=X^2$ isn't jointly Gaussian)." }
  ],

  quiz: [
    { q: "The contours of a 2-D Gaussian are…", options: ["ellipses", "circles only", "straight lines", "hyperbolas"], answer: 0,
      explain: "Constant Mahalanobis distance gives ellipses (circles only when $\\Sigma\\propto I$)." },
    { q: "The ellipse axes align with…", options: ["the eigenvectors of $\\Sigma$", "the coordinate axes always", "the mean vector", "the rows of $\\Sigma$"], answer: 0,
      explain: "Diagonalizing $\\Sigma$ shows the axes are its eigenvectors, radii $\\propto\\sqrt{\\lambda_i}$." },
    { q: "Mahalanobis distance squared is…", options: ["$(\\mathbf x-\\boldsymbol\\mu)^\\top\\Sigma^{-1}(\\mathbf x-\\boldsymbol\\mu)$", "$\\lVert\\mathbf x-\\boldsymbol\\mu\\rVert^2$", "$(\\mathbf x-\\boldsymbol\\mu)^\\top\\Sigma(\\mathbf x-\\boldsymbol\\mu)$", "$\\mathbf x^\\top\\mathbf x$"], answer: 0,
      explain: "It weights by $\\Sigma^{-1}$, stretching distance by the inverse covariance." },
    { q: "For jointly Gaussian variables, zero covariance implies…", options: ["independence", "nothing", "perfect correlation", "equal variance"], answer: 0,
      explain: "A special Gaussian property: uncorrelated + jointly Gaussian ⇒ independent." },
    { q: "A VAE's standard latent prior is…", options: ["$\\mathcal N(\\mathbf 0, I)$", "$\\mathcal N(\\mathbf 0,\\Sigma)$ with dense $\\Sigma$", "uniform", "Dirichlet"], answer: 0,
      explain: "An isotropic unit Gaussian — spherical contours, independent components." }
  ],

  practice: [
    { level: "easy", prompt: "What does $\\Sigma=I$ imply about a 2-D Gaussian's contours?", solution: "Circular contours: equal variance in all directions and zero correlation (isotropic)." },
    { level: "med", prompt: "For $\\Sigma=\\mathrm{diag}(9,1)$, along which axis is the Gaussian more spread, and by what factor in standard deviations?", solution: "Along the first axis (variance 9 vs 1). Standard deviations are $3$ vs $1$, so it's 3× wider in that direction." },
    { level: "med", prompt: "Why is $\\det\\Sigma$ in the normalizing constant?", solution: "$\\det\\Sigma=\\prod_i\\lambda_i$ is the product of the axis variances — proportional to the squared volume of the probability ellipse. Larger volume means the density is spread over more space, so the normalizer $\\sqrt{(2\\pi)^d\\det\\Sigma}$ grows to keep the total integral at 1 (the multivariate Gaussian integral, Track 7.4)." },
    { level: "hard", prompt: "AI task: connect the multivariate Gaussian to PCA. If data is $\\mathcal N(\\mathbf 0,\\Sigma)$, what do PCA's components and variances correspond to?", solution: "PCA diagonalizes the (empirical) covariance $\\Sigma=Q\\Lambda Q^\\top$. The principal components are exactly the eigenvectors $\\mathbf q_i$ — the axes of the Gaussian's contour ellipses — and the variance captured by component $i$ is the eigenvalue $\\lambda_i$, the spread along that axis. So fitting a Gaussian and running PCA are the same computation: PCA is estimating the shape of the data's Gaussian and reporting its principal axes ordered by how far the bell stretches. Projecting onto the top-$k$ eigenvectors keeps the directions of largest variance — the fattest parts of the ellipsoid (Track 5.5)." }
  ],

  deepDive: String.raw`<p><strong>The multivariate Gaussian is where probability and linear algebra become one.</strong></p>
  <p>Everything about this distribution is governed by the eigen-structure of $\Sigma$, so it inherits the whole toolkit
  of Track 5. Sampling uses the <strong>Cholesky</strong> factor: to draw $\mathbf x\sim\mathcal N(\boldsymbol\mu,\Sigma)$,
  compute $\Sigma=LL^\top$ and set $\mathbf x=\boldsymbol\mu+L\boldsymbol\varepsilon$ with $\boldsymbol\varepsilon\sim\mathcal N(\mathbf0,I)$
  — the reparameterization trick's multivariate form (next lesson). <strong>Whitening</strong> uses $\Sigma^{-1/2}$ to make
  the cloud spherical (Track 9.5). <strong>Conditioning</strong> on some coordinates yields another Gaussian with a
  Schur-complement covariance — the exact operation a <strong>Gaussian process</strong> performs to predict from observed
  points. And the log-density being a quadratic form $-\tfrac12\mathbf y^\top\Sigma^{-1}\mathbf y$ is why Gaussian MLE,
  least squares, and ridge regression all reduce to solving linear systems (Tracks 3–4).</p>
  <p>This unification is why the Gaussian is the backbone of tractable probabilistic modeling. When a method needs a
  continuous distribution that stays closed-form under the operations of ML — linear transforms, conditioning,
  marginalizing, KL divergence — the multivariate Gaussian is almost always the answer, precisely because those
  operations are matrix operations on $\boldsymbol\mu$ and $\Sigma$. The geometry you can see (tilted ellipses), the
  linear algebra you can compute (eigenvectors, Cholesky, inverses), and the probability you want (densities, samples,
  conditionals) are three faces of the same object.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["10.5"] = {
  subtitle: "Transform a random variable and its density rescales by the stretch — the trick that lets VAEs backprop through sampling.",

  aiMoment: String.raw`<p>The <strong>reparameterization trick</strong> is what makes VAEs trainable: it moves the
  randomness out of the parameters so gradients can flow through a sampling step. Underneath it is the
  <strong>change-of-variables</strong> formula, which also defines <strong>normalizing flows</strong> — invertible
  networks that reshape a simple density into a complex one while tracking exactly how probability mass is stretched.</p>`,

  plainEnglish: String.raw`<p>When you pass a random variable through a function, its <strong>density</strong> changes —
  not just its values. Squeeze the variable into a smaller range and the density piles up (grows); stretch it and the
  density thins out. The <strong>reparameterization trick</strong> is a specific, friendly change of variables: write a
  Gaussian sample as $z=\mu+\sigma\varepsilon$ with fixed noise $\varepsilon$, so $\mu,\sigma$ become ordinary
  differentiable knobs.</p>`,

  intuition: String.raw`<p>Probability is area (Lesson 10.1), and it must stay 1. So if a transform stretches a region
  by a factor, the density there must shrink by the same factor to conserve area. That conservation factor is the
  derivative of the transform (the Jacobian in higher dimensions).</p>
  <figure class="figure">
  <svg viewBox="0 0 332 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A narrow density transformed into a wider, shorter one">
    <line x1="16" y1="105" x2="130" y2="105" stroke="#94a3b8"/>
    <path d="M40,105 C55,105 58,35 70,35 C82,35 85,105 100,105 Z" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="34" y="122" font-size="9" fill="#4f46e5" font-family="sans-serif">p_Z(z): tall, narrow</text>
    <line x1="140" y1="60" x2="176" y2="60" stroke="#64748b" marker-end="url(#cv1)"/>
    <text x="140" y="52" font-size="9" fill="#64748b" font-family="sans-serif">x=g(z)</text>
    <line x1="150" y1="105" x2="270" y2="105" stroke="#94a3b8"/>
    <path d="M160,105 C185,105 190,68 205,68 C220,68 225,105 255,105 Z" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="168" y="122" font-size="9" fill="#0d9488" font-family="sans-serif">p_X(x): short, wide (same area)</text>
    <defs><marker id="cv1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#64748b"/></marker></defs>
  </svg>
  <figcaption>Stretching the axis thins the density by the Jacobian factor so total area stays 1.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Change of variables (1-D).</strong> If $X=g(Z)$ with $g$ invertible and differentiable,</p>
  $$p_X(x)=p_Z\big(g^{-1}(x)\big)\left|\frac{d\,g^{-1}(x)}{dx}\right|.$$
  <p>In $d$ dimensions the derivative becomes $|\det J_{g^{-1}}|$, the Jacobian determinant (Track 7.3). The
  <strong>reparameterization trick</strong> is the special case $z=\mu+\sigma\varepsilon$ with $\varepsilon\sim\mathcal N(0,1)$,
  which gives $z\sim\mathcal N(\mu,\sigma^2)$ — but now $\mu,\sigma$ sit <em>outside</em> the randomness.</p>`,

  derivation: String.raw`<p><strong>Part 1 — the change-of-variables formula</strong> (increasing $g$). Match CDFs:
  $F_X(x)=P(g(Z)\le x)=P(Z\le g^{-1}(x))=F_Z(g^{-1}(x)).$ Differentiate both sides in $x$ (chain rule + FTC, Track 7):
  $p_X(x)=p_Z(g^{-1}(x))\cdot(g^{-1})'(x).$ For decreasing $g$ the derivative is negative, so we take absolute value.
  $\blacksquare$</p>
  <hr class="soft">
  <p><strong>Part 2 — why reparameterization lets gradients flow.</strong> We want $\nabla_{\mu,\sigma}\,E_{z\sim\mathcal N(\mu,\sigma^2)}[f(z)]$.</p>
  <p><strong>Step 1 — the problem.</strong> The distribution we average over <em>depends on</em> $\mu,\sigma$, so you
  can't just push the gradient inside the expectation — differentiating a sample $z$ drawn from $\mathcal N(\mu,\sigma^2)$
  w.r.t. $\mu$ is undefined.</p>
  <p><strong>Step 2 — reparameterize.</strong> Write $z=\mu+\sigma\varepsilon$, $\varepsilon\sim\mathcal N(0,1)$. Then
  $E_{z\sim\mathcal N(\mu,\sigma^2)}[f(z)]=E_{\varepsilon\sim\mathcal N(0,1)}[f(\mu+\sigma\varepsilon)].$ Now the
  expectation is over $\varepsilon$, whose distribution does <em>not</em> depend on $\mu,\sigma$.</p>
  <p><strong>Step 3 — move the gradient inside:</strong> $\nabla_{\mu,\sigma}E_\varepsilon[f(\mu+\sigma\varepsilon)]=E_\varepsilon[\nabla_{\mu,\sigma}f(\mu+\sigma\varepsilon)]$,
  with $\partial f/\partial\mu=f'(z)$ and $\partial f/\partial\sigma=f'(z)\,\varepsilon$. $\blacksquare$ Plain English: by
  making the noise the fixed part and $\mu,\sigma$ the adjustable part, sampling becomes an ordinary differentiable
  operation — you can backprop straight through it.</p>`,

  code: [
    { label: "Change of variables, checked numerically", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

# Z ~ Exp(1); X = g(Z) = sqrt(Z). Then p_X(x) = p_Z(x²)·|2x| = e^{-x²}·2x
Z = rng.exponential(1.0, size=2_000_000)
X = np.sqrt(Z)
xs = np.linspace(0.05, 3, 40)
hist, edges = np.histogram(X, bins=xs, density=True)
centers = 0.5*(edges[:-1]+edges[1:])
formula = np.exp(-centers**2) * 2*centers
print("max |empirical - formula| :", round(float(np.max(np.abs(hist-formula))), 3))  # small
` },
    { label: "Reparameterization: gradient flows through sampling", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

# minimize E_{z~N(mu,1)}[ (z-3)² ] over mu.  True optimum: mu=3.
mu = 0.0; lr = 0.1
for step in range(200):
    eps = rng.normal(size=512)          # fixed noise
    z = mu + 1.0*eps                    # reparameterized sample
    # d/dmu of (z-3)² = 2(z-3) * dz/dmu = 2(z-3)*1
    grad = np.mean(2*(z - 3.0))
    mu -= lr*grad
print("learned mu ≈", round(mu, 3), " (should be ~3)")
` }
  ],

  keyPoints: [
    "Transforming a variable rescales its density by the Jacobian factor $|(g^{-1})'|$ (or $|\\det J|$ in $d$-D).",
    "Reason: probability is area and must stay 1, so stretching thins the density.",
    "Reparameterization: $z=\\mu+\\sigma\\varepsilon$, $\\varepsilon\\sim\\mathcal N(0,1)$ makes $z\\sim\\mathcal N(\\mu,\\sigma^2)$.",
    "It moves randomness out of the parameters, so $\\nabla_{\\mu,\\sigma}$ can pass through the sample.",
    "Change of variables + invertible maps = normalizing flows."
  ],

  commonMistakes: [
    { wrong: "Forgetting the Jacobian factor.", why: "$p_X(x)=p_Z(g^{-1}(x))$ alone is wrong — you must multiply by $|(g^{-1})'(x)|$ to conserve probability mass. Missing it is why a 'density' won't integrate to 1." },
    { wrong: "Trying to backprop through a raw sampling op.", why: "$z\\sim\\mathcal N(\\mu,\\sigma^2)$ isn't differentiable in $\\mu,\\sigma$ as written. You must reparameterize (or use the score-function/REINFORCE estimator, Track 7.5) to get gradients." },
    { wrong: "Using reparameterization for discrete variables.", why: "It needs a differentiable transform of continuous noise. Discrete samples aren't differentiable; use a continuous relaxation (Gumbel-softmax) or the score-function estimator instead." }
  ],

  quiz: [
    { q: "Transforming $X=g(Z)$ changes the density by a factor of…", options: ["$|(g^{-1})'(x)|$", "$g'(x)$", "$1$", "$g(x)$"], answer: 0,
      explain: "The change-of-variables formula multiplies by the absolute derivative of the inverse map." },
    { q: "The reparameterization $z=\\mu+\\sigma\\varepsilon$ with $\\varepsilon\\sim\\mathcal N(0,1)$ gives $z\\sim$", options: ["$\\mathcal N(\\mu,\\sigma^2)$", "$\\mathcal N(0,1)$", "$\\mathrm{U}(\\mu,\\sigma)$", "$\\mathcal N(\\sigma,\\mu^2)$"], answer: 0,
      explain: "Shift by $\\mu$, scale by $\\sigma$: a linear map of a standard normal is $\\mathcal N(\\mu,\\sigma^2)$." },
    { q: "Why does reparameterization help training?", options: ["it makes a sampled expectation differentiable in $\\mu,\\sigma$", "it removes the noise", "it lowers the mean", "it discretizes $z$"], answer: 0,
      explain: "With noise fixed as $\\varepsilon$, the gradient passes through $\\mu+\\sigma\\varepsilon$ normally." },
    { q: "In $d$ dimensions the density factor becomes…", options: ["$|\\det J_{g^{-1}}|$", "$\\operatorname{tr}(J)$", "$\\lVert J\\rVert$", "$\\det J^\\top J$"], answer: 0,
      explain: "The Jacobian determinant measures how the map scales volume." },
    { q: "Normalizing flows require each layer to be…", options: ["invertible and differentiable", "linear", "convex", "symmetric"], answer: 0,
      explain: "To evaluate the density you need $g^{-1}$ and its Jacobian determinant, so layers must be invertible and differentiable." }
  ],

  practice: [
    { level: "easy", prompt: "If $\\varepsilon\\sim\\mathcal N(0,1)$, what is the distribution of $z=5+2\\varepsilon$?", solution: "$\\mathcal N(5,4)$: mean $5$, variance $2^2=4$." },
    { level: "med", prompt: "$Z\\sim\\mathrm{U}(0,1)$ and $X=2Z$. Find $p_X(x)$.", solution: "$g^{-1}(x)=x/2$, $(g^{-1})'=1/2$. $p_X(x)=p_Z(x/2)\\cdot\\tfrac12=1\\cdot\\tfrac12=\\tfrac12$ on $[0,2]$ — a $\\mathrm{U}(0,2)$, as expected (stretching halves the density)." },
    { level: "med", prompt: "Why can't you differentiate $z\\sim\\mathcal N(\\mu,\\sigma^2)$ w.r.t. $\\mu$ directly, but you can after writing $z=\\mu+\\sigma\\varepsilon$?", solution: "As written, $z$ is a black-box sample whose dependence on $\\mu$ is through the sampler, not an explicit function — no gradient. Rewriting $z=\\mu+\\sigma\\varepsilon$ makes $z$ an explicit, differentiable function of $\\mu$ (with $\\varepsilon$ a constant for the backward pass), so $\\partial z/\\partial\\mu=1$ and the chain rule applies." },
    { level: "hard", prompt: "AI task: a VAE encoder outputs $\\mu,\\sigma$ and the loss is $E_{z\\sim\\mathcal N(\\mu,\\sigma^2)}[\\text{recon}(z)]+\\mathrm{KL}$. Explain precisely where reparameterization enters and why the KL term is separate.", solution: "The reconstruction term is an expectation over the sampled latent $z$; to get gradients into the encoder's $\\mu,\\sigma$, you draw $\\varepsilon\\sim\\mathcal N(0,1)$ and set $z=\\mu+\\sigma\\varepsilon$, so backprop flows through $z$ into $\\mu,\\sigma$ (usually with a single sample as a Monte-Carlo estimate, Track 7.5). The KL term $\\mathrm{KL}(\\mathcal N(\\mu,\\sigma^2)\\Vert\\mathcal N(0,1))$ has a <em>closed form</em> in $\\mu,\\sigma$ (no sampling needed), so it's differentiated analytically. Thus reparameterization is only required for the intractable expectation (reconstruction), while the KL regularizer is exact — together they form the ELBO (Track 12)." }
  ],

  deepDive: String.raw`<p><strong>Two ways to differentiate through randomness — and why the choice matters.</strong></p>
  <p>To estimate $\nabla_\theta E_{z\sim q_\theta}[f(z)]$ there are two main tools. The <strong>reparameterization
  (pathwise) estimator</strong>, this lesson's trick, rewrites $z=g_\theta(\varepsilon)$ and differentiates through $g$;
  it needs a continuous, reparameterizable $q$ and a differentiable $f$, but it typically has <em>low variance</em>
  because it uses the gradient of $f$ itself. The <strong>score-function (REINFORCE) estimator</strong> (Track 7.5)
  uses $\nabla_\theta\log q_\theta$ and works for <em>any</em> distribution — including discrete ones — but often has
  <em>high variance</em>, needing baselines and many samples to be usable.</p>
  <p>This trade-off shapes model design across ML. VAEs and diffusion use Gaussians precisely so they can reparameterize
  and enjoy low-variance gradients. Discrete latent variables (hard attention, discrete VAEs, some RL policies) can't
  reparameterize directly, so they either relax to a continuous surrogate (Gumbel-softmax, a reparameterizable
  approximation) or accept the score-function estimator's variance and fight it with baselines. Normalizing flows push
  the change-of-variables idea to its limit: stack many invertible layers, track the total Jacobian determinant, and you
  get an exactly-computable density you can both sample from and evaluate — no approximation at all. The one formula
  $p_X=p_Z\,|\det J|$ thus underlies a whole spectrum of generative models, from VAEs to flows to the continuous-time
  transformations of diffusion.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["10.6"] = {
  subtitle: "Averages converge to the truth (LLN), and sums of many things become Gaussian (CLT).",

  aiMoment: String.raw`<p>Two theorems justify a huge amount of ML. The <strong>Law of Large Numbers</strong> is why a
  minibatch average estimates the true expected loss (Track 7.5) — average enough samples and you converge to the truth.
  The <strong>Central Limit Theorem</strong> is why Gaussians are everywhere: sums and averages of many independent
  effects become bell-shaped regardless of the original distribution, which is why measurement noise, aggregated
  features, and initialization sums are modeled as Normal.</p>`,

  plainEnglish: String.raw`<p>The <strong>Law of Large Numbers</strong> says: average more and more independent samples
  and the average homes in on the true mean. The <strong>Central Limit Theorem</strong> says something stronger about the
  <em>shape</em>: the distribution of that average (suitably scaled) becomes a Gaussian bell, no matter what you started
  with.</p>`,

  intuition: String.raw`<p>Take the average of $n$ dice rolls. For small $n$ it's jumpy; as $n$ grows it settles near
  3.5 (LLN). And if you repeat the experiment many times and histogram those averages, the histogram is a bell — even
  though a single die is flat/uniform (CLT).</p>
  <figure class="figure">
  <svg viewBox="0 0 296 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Histogram of sample means forming a bell curve">
    <line x1="20" y1="105" x2="240" y2="105" stroke="#94a3b8"/>
    <g fill="#eef0ff" stroke="#4f46e5">
    <rect x="60" y="92" width="16" height="13"/><rect x="78" y="78" width="16" height="27"/>
    <rect x="96" y="55" width="16" height="50"/><rect x="114" y="38" width="16" height="67"/>
    <rect x="132" y="52" width="16" height="53"/><rect x="150" y="76" width="16" height="29"/>
    <rect x="168" y="90" width="16" height="15"/>
    </g>
    <path d="M52,105 C90,105 100,30 130,30 C160,30 170,105 200,105" fill="none" stroke="#dc2626" stroke-width="2"/>
    <text x="70" y="122" font-size="10" fill="#64748b" font-family="sans-serif">distribution of the sample mean → bell</text>
  </svg>
  <figcaption>Averages of non-Gaussian samples pile up into a Gaussian — the CLT.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Let $X_1,\dots,X_n$ be iid with mean $\mu$, variance $\sigma^2$, and sample mean $\bar X_n=\tfrac1n\sum_i X_i$.
  <strong>Law of Large Numbers:</strong> $\bar X_n\to\mu$ as $n\to\infty$. <strong>Central Limit Theorem:</strong></p>
  $$\frac{\bar X_n-\mu}{\sigma/\sqrt n}\ \xrightarrow{\ d\ }\ \mathcal N(0,1),$$
  <p>i.e. for large $n$, $\bar X_n\approx\mathcal N\!\big(\mu,\ \sigma^2/n\big)$ — regardless of the shape of the original
  distribution (given finite variance).</p>`,

  derivation: String.raw`<p><strong>Part 1 — the LLN, via variance shrinking.</strong> The sample mean has
  $E[\bar X_n]=\mu$ and (independent samples) $\operatorname{Var}(\bar X_n)=\dfrac{\sigma^2}{n}\to0.$ By Chebyshev's
  inequality, $P(|\bar X_n-\mu|>\varepsilon)\le\dfrac{\operatorname{Var}(\bar X_n)}{\varepsilon^2}=\dfrac{\sigma^2}{n\varepsilon^2}\to0.$
  So the average concentrates on $\mu$. $\blacksquare$</p>
  <hr class="soft">
  <p><strong>Part 2 — the CLT, sketch via moment generating functions.</strong> Standardize: $Y_i=(X_i-\mu)/\sigma$ has
  mean 0, variance 1, and MGF $M_Y(t)=1+\tfrac{t^2}{2}+o(t^2)$ (from $E[Y]=0$, $E[Y^2]=1$).</p>
  <p><strong>Step 1.</strong> Let $S_n=\tfrac{1}{\sqrt n}\sum_{i=1}^n Y_i$. By independence, its MGF is
  $M_{S_n}(t)=\big[M_Y(t/\sqrt n)\big]^n$.</p>
  <p><strong>Step 2.</strong> Expand: $M_Y(t/\sqrt n)=1+\dfrac{t^2}{2n}+o(1/n)$, so
  $M_{S_n}(t)=\Big(1+\dfrac{t^2}{2n}+o(1/n)\Big)^n.$</p>
  <p><strong>Step 3.</strong> Take $n\to\infty$ using $(1+a/n)^n\to e^{a}$: $M_{S_n}(t)\to e^{t^2/2}$, which is the MGF of
  $\mathcal N(0,1)$. Since the MGF determines the distribution, $S_n\to\mathcal N(0,1).\;\blacksquare$ Plain English: the
  individual quirks wash out and only the mean and variance survive — leaving a Gaussian.</p>`,

  code: [
    { label: "Law of Large Numbers: the average settles", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

rolls = rng.integers(1, 7, size=100000)
running_mean = np.cumsum(rolls) / np.arange(1, len(rolls)+1)
for n in [10, 100, 1000, 100000]:
    print(f"n={n:6d}: running mean = {running_mean[n-1]:.4f}")   # -> 3.5
` },
    { label: "Central Limit Theorem: averages of uniforms → bell", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

n = 30                                        # samples per average
means = rng.uniform(0, 1, size=(200000, n)).mean(axis=1)
print("mean of sample-means  ≈", round(float(means.mean()), 4), " (true 0.5)")
print("std  of sample-means  ≈", round(float(means.std()), 4),
      " vs σ/√n =", round((1/np.sqrt(12))/np.sqrt(n), 4))
# 'means' is bell-shaped even though a single Uniform(0,1) is flat
` }
  ],

  keyPoints: [
    "LLN: the sample mean $\\bar X_n\\to\\mu$ as $n$ grows (averages converge to the truth).",
    "CLT: $\\bar X_n\\approx\\mathcal N(\\mu,\\sigma^2/n)$ for large $n$, whatever the original shape.",
    "The CLT explains why aggregated noise and features are approximately Gaussian.",
    "Monte-Carlo error shrinks like $\\sigma/\\sqrt n$ — the CLT's scaling (Track 7.5).",
    "Both need finite variance and (roughly) independent samples."
  ],

  commonMistakes: [
    { wrong: "Thinking the CLT makes the raw data Gaussian.", why: "It's the <em>sample mean</em> (or sum) that becomes Gaussian, not the individual $X_i$. A single die stays uniform; only its averages become bell-shaped." },
    { wrong: "Applying the CLT with heavy-tailed data.", why: "It requires finite variance. For distributions like Cauchy (infinite variance), the sample mean does NOT converge to a Gaussian — the CLT fails." },
    { wrong: "Expecting error to drop as $1/n$.", why: "The CLT gives standard error $\\sigma/\\sqrt n$: to halve it you need $4\\times$ the data. This $1/\\sqrt n$ rate limits Monte-Carlo accuracy." }
  ],

  quiz: [
    { q: "As $n\\to\\infty$, the sample mean of iid samples converges to…", options: ["the true mean $\\mu$", "0", "a uniform", "the maximum"], answer: 0,
      explain: "That's the Law of Large Numbers." },
    { q: "The CLT says $\\bar X_n$ is approximately…", options: ["$\\mathcal N(\\mu,\\sigma^2/n)$", "$\\mathcal N(0,1)$", "uniform", "$\\mathcal N(\\mu,\\sigma^2)$"], answer: 0,
      explain: "The sample mean has mean $\\mu$ and variance $\\sigma^2/n$, and becomes Gaussian for large $n$." },
    { q: "The standard error of a mean of $n$ samples scales as…", options: ["$\\sigma/\\sqrt n$", "$\\sigma/n$", "$\\sigma n$", "constant"], answer: 0,
      explain: "$\\operatorname{Var}(\\bar X_n)=\\sigma^2/n$, so the standard deviation is $\\sigma/\\sqrt n$." },
    { q: "For which does the CLT FAIL?", options: ["a distribution with infinite variance (e.g. Cauchy)", "the uniform", "the exponential", "a die"], answer: 0,
      explain: "Finite variance is required; Cauchy has none, so its sample mean isn't Gaussian." },
    { q: "Averaging 100 iid samples with $\\sigma=5$ gives a mean with standard error…", options: ["0.5", "5", "0.05", "50"], answer: 0,
      explain: "$\\sigma/\\sqrt n=5/\\sqrt{100}=5/10=0.5$." }
  ],

  practice: [
    { level: "easy", prompt: "You average 400 iid samples with $\\sigma=8$. Give the standard error of the mean.", solution: "$\\sigma/\\sqrt n=8/\\sqrt{400}=8/20=0.4$." },
    { level: "med", prompt: "Why does a minibatch of size $B$ give an unbiased gradient with standard error $\\propto1/\\sqrt B$?", solution: "The minibatch gradient is a sample mean of per-example gradients: unbiased by linearity of expectation ($E[\\bar g]=\\nabla E[\\ell]$), and by the CLT/LLN its fluctuation around the true gradient has standard deviation $\\sigma_g/\\sqrt B$. So larger batches reduce noise, but only at the $1/\\sqrt B$ rate." },
    { level: "med", prompt: "A single die is uniform on 1–6. What does the CLT predict for the average of many dice?", solution: "The average of $n$ dice is approximately $\\mathcal N(3.5,\\ \\sigma^2/n)$ with $\\sigma^2=35/12\\approx2.92$. For large $n$ it's a tight bell around 3.5 — despite each die being flat." },
    { level: "hard", prompt: "AI task: why are neural network pre-activations often approximately Gaussian at initialization, and what breaks this as training proceeds?", solution: "A pre-activation $z=\\sum_{i=1}^{n_{\\text{in}}} w_i x_i$ is a sum of many independent-ish terms; by the CLT, for wide layers ($n_{\\text{in}}$ large) $z$ is approximately Gaussian regardless of the individual weight/input distributions — which is the basis of signal-propagation and 'neural tangent kernel' analyses that assume Gaussian pre-activations at init. Training breaks the assumptions: weights become correlated and non-independent as they co-adapt, activations develop structure, and nonlinearities skew the distribution — so the clean Gaussian picture holds best at initialization and degrades as the network learns. This is why init schemes (He/Xavier, Lesson 10.2) are derived under the Gaussian/CLT assumption, but training dynamics need the fuller optimization tools of Track 8." }
  ],

  deepDive: String.raw`<p><strong>The CLT is why 'assume Gaussian' is so often justified — and when it isn't.</strong></p>
  <p>The Central Limit Theorem is arguably the reason probabilistic ML leans on Gaussians so heavily. Any quantity that
  is a sum or average of many small, roughly-independent contributions is approximately Gaussian: sensor noise (many
  micro-disturbances), a feature that aggregates many signals, a pre-activation summing many inputs, a bootstrap
  statistic averaging resamples. This is a genuinely deep fact — the Gaussian is an <em>attractor</em>: start with almost
  any finite-variance distribution, add and rescale, and you flow toward the bell. It's why the Gaussian's convenient
  algebra (Lessons 10.2, 10.4) is so often <em>applicable</em>, not just tractable.</p>
  <p>But the caveats are exactly where ML gets bitten. The CLT needs <strong>finite variance</strong>: heavy-tailed
  phenomena (some financial returns, certain gradient distributions, attention over rare tokens) don't average into
  Gaussians, and pretending they do underestimates catastrophic tail events. It needs <strong>enough independence</strong>:
  strongly correlated contributions converge slowly or to a different limit. And it's about the <em>bulk</em>, not the
  tails — the Gaussian approximation is best near the mean and worst exactly where rare, high-impact events live. So the
  CLT both licenses the Gaussian defaults throughout this course and warns you where to distrust them. It closes the
  probability tracks by explaining the shape everything tends toward — and hands off to statistics (Track 11), where we
  turn samples into estimates and quantify how much to trust them.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["10.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 70 minutes.",

  intro: String.raw`<p>This exam spans all of Track 10: densities/CDFs and the uniform, the Gaussian, exponential/Beta/Dirichlet,
  the multivariate Gaussian, change of variables and the reparameterization trick, and the LLN/CLT. <strong>Give yourself
  70 minutes</strong>, produce each answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "Give the mean and variance of $\\mathrm{U}(2,8)$.",
      solution: "Mean $(2+8)/2=5$; variance $(8-2)^2/12=36/12=3$." },
    { level: "easy", prompt: "$X\\sim\\mathcal N(10,16)$. What is $\\sigma$, and roughly what fraction lies in $[2,18]$?",
      solution: "$\\sigma=\\sqrt{16}=4$; $[2,18]=\\mu\\pm2\\sigma$, so about 95%." },
    { level: "med", prompt: "A density $p(x)=cx^2$ on $[0,3]$. Find $c$.",
      solution: "$\\int_0^3 cx^2\\,dx=c\\cdot 9=1\\Rightarrow c=1/9$." },
    { level: "med", prompt: "Start with $\\mathrm{Beta}(2,2)$ and observe 5 heads, 1 tail. Give the posterior and its mean.",
      solution: "$\\mathrm{Beta}(2+5,2+1)=\\mathrm{Beta}(7,3)$; mean $7/10=0.7$." },
    { level: "med", prompt: "$X\\sim\\mathcal N(1,4)$ independent of $Y\\sim\\mathcal N(2,9)$. Distribution of $X+Y$?",
      solution: "Sum of independent Gaussians: $\\mathcal N(1+2,\\,4+9)=\\mathcal N(3,13)$." },
    { level: "med", prompt: "An exponential has mean 5. Find its rate $\\lambda$ and $P(X>10)$.",
      solution: "Mean $1/\\lambda=5\\Rightarrow\\lambda=0.2$. $P(X>10)=e^{-\\lambda\\cdot10}=e^{-2}\\approx0.135$." },
    { level: "hard", prompt: "Reparameterize: write a sample from $\\mathcal N(\\mu,\\sigma^2)$ using $\\varepsilon\\sim\\mathcal N(0,1)$, and give $\\partial z/\\partial\\mu$ and $\\partial z/\\partial\\sigma$.",
      solution: "$z=\\mu+\\sigma\\varepsilon$. Then $\\partial z/\\partial\\mu=1$ and $\\partial z/\\partial\\sigma=\\varepsilon$ (treating $\\varepsilon$ as a constant during backprop). This is why gradients flow into $\\mu,\\sigma$ through the sample." },
    { level: "hard", prompt: "$Z\\sim\\mathrm{U}(0,1)$, $X=-\\ln Z$. Find $p_X(x)$ and name the distribution.",
      solution: "$g(z)=-\\ln z$, $g^{-1}(x)=e^{-x}$, $(g^{-1})'=-e^{-x}$. $p_X(x)=p_Z(e^{-x})\\,|-e^{-x}|=1\\cdot e^{-x}=e^{-x}$ for $x\\ge0$ — the standard $\\mathrm{Exponential}(1)$." },
    { level: "hard", prompt: "You average 625 iid samples with $\\sigma=10$. Give the standard error, and how many samples you'd need to halve it.",
      solution: "Standard error $=\\sigma/\\sqrt n=10/25=0.4$. To halve it (to 0.2) needs $4\\times$ the samples: $n=2500$ (the $1/\\sqrt n$ rate)." },
    { level: "hard", prompt: "AI task: explain, via the CLT, why a wide layer's pre-activations are approximately Gaussian at initialization, and why He init uses variance $2/n_{\\text{in}}$.",
      solution: "A pre-activation $z=\\sum_{i=1}^{n_{\\text{in}}}w_i x_i$ sums many independent-ish terms, so by the CLT it's approximately Gaussian for large $n_{\\text{in}}$, regardless of the weight/input distributions. Its variance is $n_{\\text{in}}\\sigma_w^2\\sigma_x^2$ (independent terms' variances add). To preserve signal variance layer-to-layer you need $n_{\\text{in}}\\sigma_w^2\\approx1$, i.e. $\\sigma_w^2\\approx1/n_{\\text{in}}$; the factor 2 (He) compensates for ReLU zeroing about half the units, so $\\sigma_w^2=2/n_{\\text{in}}$. The CLT gives the Gaussian shape; the variance algebra gives the scale." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Continuous probability is solid. On to Statistics (Track 11) and, when ready, the probability capstone.</li>
    <li><strong>7–8:</strong> Strong. Revisit the reparameterization derivation or the CLT scaling if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive the change-of-variables formula and the CLT sketch; redo Lessons 10.5 and 10.6.</li>
    <li><strong>Below 5:</strong> Rework the track — the Gaussian, reparameterization, and CLT recur throughout generative modeling and statistics.</li>
  </ul>`
};
