/* ============================================================
   TRACK 7 — Calculus II — Integration & Expectation
   7.1 Antiderivatives & the Definite Integral · 7.2 FTC ·
   7.3 Substitution & By Parts · 7.4 The Gaussian Integral ·
   7.5 Integrals as Expectation · 7.E Track Exam
   ============================================================ */
(window.LESSON_CONTENT ||= {})["7.1"] = {
  subtitle: "Area under a curve — the operation that turns a density into a probability.",

  aiMoment: String.raw`<p>A probability density $p(x)$ must integrate to 1 — the total area under it is all the
  probability there is. An <strong>expected value</strong> is an integral, $E[X]=\int x\,p(x)\,dx$. The partition
  function that normalizes an energy-based model is an integral. Before any of that: integration is just adding up area
  under a curve, and the antiderivative is the reverse of the derivative.</p>`,

  plainEnglish: String.raw`<p>The <strong>definite integral</strong> of a function between two points is the area
  trapped under its curve there. An <strong>antiderivative</strong> is a function whose derivative gives you back the
  one you started with — integration undoes differentiation.</p>`,

  intuition: String.raw`<p>Slice the area under a curve into thin rectangles, add their areas, and let the slices get
  thinner. The total approaches the exact area — that limit is the integral.</p>
  <figure class="figure">
  <svg viewBox="0 0 286 160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Riemann rectangles approximating area under a curve">
    <line x1="30" y1="140" x2="240" y2="140" stroke="#94a3b8"/>
    <line x1="30" y1="140" x2="30" y2="20" stroke="#94a3b8"/>
    <g fill="#eef0ff" stroke="#4f46e5">
    <rect x="40" y="120" width="24" height="20"/><rect x="64" y="106" width="24" height="34"/>
    <rect x="88" y="90" width="24" height="50"/><rect x="112" y="72" width="24" height="68"/>
    <rect x="136" y="56" width="24" height="84"/><rect x="160" y="42" width="24" height="98"/>
    </g>
    <path d="M40,128 Q150,10 200,30" fill="none" stroke="#dc2626" stroke-width="2"/>
    <text x="120" y="156" font-size="11" fill="#64748b" font-family="sans-serif">thinner slices → exact area</text>
  </svg>
  <figcaption>The integral is the limit of summed rectangle areas as the slice width shrinks to zero.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>definite integral</strong> is a limit of Riemann sums:</p>
  $$\int_a^b f(x)\,dx=\lim_{n\to\infty}\sum_{i=1}^{n} f(x_i)\,\Delta x,\qquad \Delta x=\tfrac{b-a}{n}.$$
  <p>An <strong>antiderivative</strong> $F$ satisfies $F'=f$; the basic one is $\int x^n\,dx=\dfrac{x^{n+1}}{n+1}+C$
  (for $n\neq-1$), with $\int\tfrac1x\,dx=\ln|x|+C$. The constant $C$ reflects that derivatives forget constants.
  Area below the axis counts as negative (signed area).</p>`,

  derivation: String.raw`<p><strong>Compute $\int_0^1 x^2\,dx$ two ways.</strong></p>
  <p><strong>Way 1 — Riemann sum.</strong> Split $[0,1]$ into $n$ pieces, sample at $x_i=i/n$, width $\Delta x=1/n$:
  $\sum_{i=1}^{n}\left(\tfrac{i}{n}\right)^2\tfrac1n=\tfrac{1}{n^3}\sum_{i=1}^{n} i^2=\tfrac{1}{n^3}\cdot\tfrac{n(n+1)(2n+1)}{6}.$</p>
  <p><strong>Step — take the limit.</strong> The numerator's leading term is $2n^3$, so the expression $\to\tfrac{2n^3}{6n^3}=\tfrac13$ as $n\to\infty$.</p>
  <p><strong>Way 2 — antiderivative.</strong> $F(x)=\tfrac{x^3}{3}$ has $F'=x^2$. So
  $\int_0^1 x^2\,dx=F(1)-F(0)=\tfrac13-0=\tfrac13.\;\blacksquare$ Both agree — and Way 2 is why the Fundamental Theorem
  (next lesson) is such a gift: it replaces an infinite sum with one subtraction.</p>`,

  code: [
    { label: "Numerical integration vs the exact value", src: String.raw`
import numpy as np

f = lambda x: x**2
x = np.linspace(0, 1, 100001)
area_trap = np.trapz(f(x), x)         # trapezoidal rule
print("numerical ∫₀¹ x² dx =", round(area_trap, 6))
print("exact (1/3)         =", round(1/3, 6))

# Riemann sum converging to 1/3
for n in [10, 100, 1000]:
    xi = np.arange(1, n+1)/n
    print(f"n={n:5d}: Riemann sum =", round(float(np.sum(xi**2)/n), 6))
` },
    { label: "A density integrates to 1", src: String.raw`
import numpy as np

# standard normal density should enclose area 1
p = lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi)
x = np.linspace(-10, 10, 200001)
print("∫ p(x) dx =", round(float(np.trapz(p(x), x)), 6))   # ≈ 1.0
` }
  ],

  keyPoints: [
    "The definite integral $\\int_a^b f\\,dx$ is the signed area under $f$, a limit of Riemann sums.",
    "An antiderivative $F$ has $F'=f$; $\\int x^n dx=\\frac{x^{n+1}}{n+1}+C$ ($n\\neq-1$).",
    "Area below the axis is negative (signed area).",
    "A probability density integrates to 1 over its whole range.",
    "Integration accumulates; differentiation measures rate — they are inverses (next lesson)."
  ],

  commonMistakes: [
    { wrong: "Dropping the $+C$ on an indefinite integral.", why: "Antiderivatives are unique only up to a constant; many problems (initial conditions, normalization) need it. Definite integrals cancel it via $F(b)-F(a)$." },
    { wrong: "Forgetting integrals can be negative.", why: "The integral is <em>signed</em> area; a function below the axis contributes negatively. 'Area' in the everyday sense uses $|f|$." },
    { wrong: "Trusting a numerical integral over a too-narrow range.", why: "A density on $(-\\infty,\\infty)$ integrated only over $[-2,2]$ misses the tails and won't reach 1; choose limits wide enough to capture the mass." }
  ],

  quiz: [
    { q: "$\\int_0^2 3x^2\\,dx$ equals…", options: ["8", "12", "6", "4"], answer: 0,
      explain: "Antiderivative $x^3$; $F(2)-F(0)=8-0=8$." },
    { q: "An antiderivative of $\\cos x$ is…", options: ["$\\sin x$", "$-\\sin x$", "$-\\cos x$", "$\\tan x$"], answer: 0,
      explain: "$\\frac{d}{dx}\\sin x=\\cos x$, so $\\int\\cos x\\,dx=\\sin x+C$." },
    { q: "$\\int_a^b f\\,dx$ where $f<0$ on all of $[a,b]$ is…", options: ["negative", "positive", "zero", "undefined"], answer: 0,
      explain: "Signed area below the axis is negative." },
    { q: "$\\int \\frac1x\\,dx$ equals…", options: ["$\\ln|x|+C$", "$x^{-2}+C$", "$-x^{-2}+C$", "$\\frac{x^0}{0}$"], answer: 0,
      explain: "The power rule fails at $n=-1$; the antiderivative of $1/x$ is $\\ln|x|+C$." },
    { q: "Why must a probability density integrate to exactly 1?", options: ["total probability of all outcomes is 1", "densities are always positive", "it makes the mean zero", "to be differentiable"], answer: 0,
      explain: "The area under a density is the probability of 'some outcome happens,' which must be 1." }
  ],

  practice: [
    { level: "easy", prompt: "Evaluate $\\int_1^3 2x\\,dx$.", solution: "Antiderivative $x^2$; $F(3)-F(1)=9-1=8$." },
    { level: "easy", prompt: "Find $\\int (x^3-4)\\,dx$.", solution: "$\\frac{x^4}{4}-4x+C$." },
    { level: "med", prompt: "Compute $\\int_0^1 x^2\\,dx$ from the Riemann-sum limit, showing the algebra.", solution: "$\\sum_{i=1}^n(i/n)^2(1/n)=\\frac{1}{n^3}\\cdot\\frac{n(n+1)(2n+1)}{6}=\\frac{(n+1)(2n+1)}{6n^2}\\to\\frac{2n^2}{6n^2}=\\frac13$ as $n\\to\\infty$." },
    { level: "hard", prompt: "AI task: a model outputs an unnormalized score $\\tilde p(x)=e^{-x^2}$. Write the normalized density and the integral you'd compute, and connect it to the softmax denominator.", solution: "The density is $p(x)=\\tilde p(x)/Z$ with $Z=\\int_{-\\infty}^{\\infty}e^{-x^2}\\,dx$ (the partition function). You must integrate the unnormalized score to get $Z$, then divide. This is the continuous analogue of softmax: softmax's denominator $\\sum_j e^{z_j}$ is the discrete partition function, while here $Z$ is the continuous one. Both ensure the outputs sum/integrate to 1. (We compute this exact $Z=\\sqrt\\pi$ in Lesson 7.4.)" }
  ],

  deepDive: String.raw`<p><strong>Why integration is "harder" than differentiation — and why ML mostly differentiates.</strong></p>
  <p>Differentiation is mechanical: apply a few rules and any elementary function yields a closed-form derivative.
  Integration is not — many perfectly innocent functions, like $e^{-x^2}$, have <em>no</em> elementary antiderivative,
  so you can't write $\int e^{-x^2}dx$ with powers, exponentials, and logs. This asymmetry shapes machine learning:
  training relies on differentiation (gradients always exist and autodiff computes them cheaply), while the integrals
  that appear — partition functions, marginal likelihoods, expectations over latent variables — are usually
  <strong>intractable</strong> and must be approximated.</p>
  <p>That intractability is the reason for a whole toolbox: <strong>Monte Carlo</strong> estimates an integral by
  sampling and averaging (Lesson 7.5); <strong>variational inference</strong> replaces a hard integral with an
  optimization (the ELBO, Track 12); the <strong>reparameterization trick</strong> (Track 10) makes a sampled integral
  differentiable. So the practical rule of thumb holds: in deep learning you differentiate exactly and integrate
  approximately. Knowing which integrals are hard — and why — tells you where the clever approximations have to live.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["7.2"] = {
  subtitle: "Differentiation and integration are inverse operations — the bridge between density and CDF.",

  aiMoment: String.raw`<p>A probability's <strong>CDF</strong> $F(x)=P(X\le x)$ and its <strong>density</strong> $p(x)$
  are linked by the Fundamental Theorem of Calculus: differentiate the CDF to get the density, integrate the density to
  get the CDF. That's why $p(x)=F'(x)$ and $F(b)-F(a)=P(a<X\le b)$. The FTC is the hinge connecting accumulation
  (probability) and rate (density).</p>`,

  plainEnglish: String.raw`<p>The Fundamental Theorem says the two big operations of calculus undo each other: if you
  integrate a function and then differentiate, you're back where you started — and a definite integral can be computed
  by plugging the endpoints into any antiderivative.</p>`,

  intuition: String.raw`<p>Let $A(x)$ be the running area under $f$ up to $x$. Nudge $x$ a little and the area grows by
  a thin sliver of height $f(x)$ and width $dx$ — so the area's rate of growth is exactly $f(x)$. That is
  $A'(x)=f(x)$: integration then differentiation cancel.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Accumulated area and the new sliver of height f(x)">
    <line x1="30" y1="120" x2="240" y2="120" stroke="#94a3b8"/>
    <line x1="30" y1="120" x2="30" y2="20" stroke="#94a3b8"/>
    <path d="M30,100 Q120,30 230,60" fill="none" stroke="#4f46e5" stroke-width="2"/>
    <path d="M30,100 Q120,30 170,46 L170,120 L30,120 Z" fill="#eef0ff" opacity="0.7"/>
    <rect x="170" y="46" width="14" height="74" fill="#fde9cf" stroke="#d97706"/>
    <text x="80" y="100" font-size="11" fill="#4f46e5" font-family="sans-serif">A(x) = area</text>
    <text x="150" y="138" font-size="10" fill="#d97706" font-family="sans-serif">sliver ≈ f(x)·dx</text>
  </svg>
  <figcaption>The area's growth rate at x is the height f(x): A′(x)=f(x).</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>FTC Part 1:</strong> if $A(x)=\int_a^x f(t)\,dt$, then $A'(x)=f(x)$ (differentiating an
  integral returns the integrand). <strong>FTC Part 2:</strong> if $F'=f$, then</p>
  $$\int_a^b f(x)\,dx=F(b)-F(a).$$
  <p>For probability, the CDF is $F(x)=\int_{-\infty}^{x}p(t)\,dt$, so the density is $p(x)=F'(x)$ and
  $P(a<X\le b)=F(b)-F(a)$.</p>`,

  derivation: String.raw`<p><strong>Proof of FTC Part 1.</strong> Let $A(x)=\int_a^x f(t)\,dt$.</p>
  <p><strong>Step 1 — difference quotient.</strong> $A(x+h)-A(x)=\int_x^{x+h} f(t)\,dt$ — the area of a thin strip from
  $x$ to $x+h$.</p>
  <p><strong>Step 2 — bound the strip.</strong> For small $h$, $f$ is nearly constant at $f(x)$ on $[x,x+h]$, so the
  strip's area is approximately $f(x)\cdot h$. (Rigorously, it lies between $h\cdot\min f$ and $h\cdot\max f$ on the
  interval.)</p>
  <p><strong>Step 3 — divide and take the limit.</strong> $\dfrac{A(x+h)-A(x)}{h}\approx f(x)$, and as $h\to0$ the
  bounds squeeze to $f(x)$ (continuity). Hence $A'(x)=f(x).\;\blacksquare$</p>
  <p><strong>Part 2 follows:</strong> $A$ and any antiderivative $F$ differ by a constant, so
  $\int_a^b f=A(b)-A(a)=F(b)-F(a)$ — the constant cancels. Plain English: to add up infinitely many slivers, just
  subtract an antiderivative at the endpoints.</p>`,

  code: [
    { label: "Differentiating an integral returns the integrand", src: String.raw`
import numpy as np

f = lambda t: np.sin(t) + 0.5*t
def A(x, a=0.0):                      # accumulated area via fine trapezoid
    t = np.linspace(a, x, 4000); return np.trapz(f(t), t)

x = 1.2; h = 1e-4
print("A'(x) numeric :", round((A(x+h)-A(x-h))/(2*h), 5))
print("f(x)          :", round(float(f(x)), 5))     # should match -> FTC1
` },
    { label: "CDF and PDF are linked by the FTC", src: String.raw`
import numpy as np
from math import erf

pdf = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)        # standard normal density
cdf = lambda x: 0.5*(1+erf(x/np.sqrt(2)))               # its CDF

x, h = 0.7, 1e-5
print("d/dx CDF :", round((cdf(x+h)-cdf(x-h))/(2*h), 6))
print("pdf(x)   :", round(float(pdf(x)), 6))            # equal -> p(x)=F'(x)
` }
  ],

  keyPoints: [
    "FTC1: $\\frac{d}{dx}\\int_a^x f(t)\\,dt=f(x)$ — differentiation undoes integration.",
    "FTC2: $\\int_a^b f=F(b)-F(a)$ for any antiderivative $F$ — one subtraction, not an infinite sum.",
    "The density is the derivative of the CDF: $p(x)=F'(x)$.",
    "$P(a<X\\le b)=F(b)-F(a)$ — probabilities are CDF differences.",
    "The integration constant cancels in any definite integral."
  ],

  commonMistakes: [
    { wrong: "Confusing the density with the CDF.", why: "$p(x)=F'(x)$ is the rate; $F(x)$ is the accumulated probability. A density can exceed 1; a CDF never does (it's a probability)." },
    { wrong: "Forgetting the lower-limit term.", why: "$\\int_a^b f=F(b)-F(a)$, not just $F(b)$. Omitting $F(a)$ is a frequent slip." },
    { wrong: "Applying FTC to a discontinuous integrand without care.", why: "FTC1 needs $f$ continuous at $x$. At a jump, the accumulated area has a corner and $A'$ isn't $f$ there — relevant for densities with discontinuities." }
  ],

  quiz: [
    { q: "$\\frac{d}{dx}\\int_0^x e^{t^2}\\,dt$ equals…", options: ["$e^{x^2}$", "$2x e^{x^2}$", "$\\int_0^x e^{t^2}dt$", "$e^{x^2}-1$"], answer: 0,
      explain: "FTC1: differentiating the area function returns the integrand evaluated at $x$, namely $e^{x^2}$ — even though the integral has no closed form." },
    { q: "If $F(x)=x^3$ is an antiderivative of $f$, then $\\int_1^2 f\\,dx=$", options: ["7", "8", "1", "9"], answer: 0,
      explain: "$F(2)-F(1)=8-1=7$." },
    { q: "The density $p(x)$ relates to the CDF $F$ by…", options: ["$p=F'$", "$F=p'$", "$p=\\int F$", "$p=1-F$"], answer: 0,
      explain: "Differentiate the CDF to get the density: $p(x)=F'(x)$." },
    { q: "$P(a<X\\le b)$ equals…", options: ["$F(b)-F(a)$", "$F(a)-F(b)$", "$p(b)-p(a)$", "$F(b)$"], answer: 0,
      explain: "A probability over an interval is the difference of the CDF at the endpoints." },
    { q: "Why is the FTC powerful for definite integrals?", options: ["it replaces an infinite sum with one subtraction", "it makes all integrals elementary", "it removes the constant", "it differentiates densities"], answer: 0,
      explain: "Instead of summing infinitely many rectangles, evaluate an antiderivative at two endpoints." }
  ],

  practice: [
    { level: "easy", prompt: "Compute $\\int_0^{\\pi}\\sin x\\,dx$.", solution: "Antiderivative $-\\cos x$: $(-\\cos\\pi)-(-\\cos0)=(1)-(-1)=2$." },
    { level: "easy", prompt: "If $F(x)=\\tfrac12 x^2$ is an antiderivative of $f$, find $\\int_2^4 f\\,dx$.", solution: "$F(4)-F(2)=8-2=6$." },
    { level: "med", prompt: "Find $\\frac{d}{dx}\\int_x^{1} \\cos(t^2)\\,dt$ (note the variable lower limit).", solution: "Flip limits: $\\int_x^1=-\\int_1^x$, so the derivative is $-\\cos(x^2)$ (FTC1 with a sign from the lower limit)." },
    { level: "hard", prompt: "AI task: explain how sampling from a continuous distribution uses the inverse CDF, and why the FTC guarantees it works.", solution: "Inverse-transform sampling: draw $u\\sim\\text{Uniform}(0,1)$, then output $x=F^{-1}(u)$. Because $F(x)=\\int_{-\\infty}^x p$ is a continuous, strictly increasing map from the sample space onto $(0,1)$ (its derivative $p=F'>0$ by FTC1), it has an inverse, and $P(F^{-1}(U)\\le x)=P(U\\le F(x))=F(x)$ — so $x$ has exactly the target distribution. The FTC is what guarantees $F$ is smooth and invertible with $F'=p$, making the construction valid." }
  ],

  deepDive: String.raw`<p><strong>The FTC as the engine of normalizing flows.</strong></p>
  <p>The link $p=F'$ generalizes into one of the most elegant ideas in generative modeling. A <strong>normalizing
  flow</strong> transforms a simple distribution (say a Gaussian) through an invertible map $g$ to model a complex one.
  The change-of-variables formula (Track 10) says the transformed density is $p_X(x)=p_Z(g^{-1}(x))\,|{\det J_{g^{-1}}}|$
  — the Jacobian-determinant factor is the multivariate echo of "the density is the derivative," exactly the $F'$
  relationship seen one dimension at a time.</p>
  <p>This is why flows insist on invertible, differentiable layers (Lesson 1.3): you need both $g^{-1}$ to evaluate the
  density and its Jacobian determinant to rescale it correctly. The FTC's one-dimensional statement — accumulate to get
  $F$, differentiate to get $p$ — becomes, in higher dimensions, the rule that a transformation stretches probability
  mass in proportion to how it stretches volume. From sampling via the inverse CDF to training flows by maximum
  likelihood, the same theorem is doing the bookkeeping between densities and the maps that move them around.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["7.3"] = {
  subtitle: "Two techniques that reverse the chain and product rules — and underlie change-of-variables.",

  aiMoment: String.raw`<p>The <strong>change of variables</strong> in an integral is the engine behind the
  reparameterization trick and normalizing flows: rewrite a hard integral (or a sampled density) in friendlier
  coordinates. <strong>Integration by parts</strong> shows up when manipulating expectations and deriving bounds like
  the ELBO. Both are just the chain rule and product rule run backward.</p>`,

  plainEnglish: String.raw`<p><strong>Substitution</strong> ($u$-substitution) relabels the variable to undo a chain
  rule — you spot an inner function and its derivative and collapse them. <strong>Integration by parts</strong> trades a
  hard integral for an easier one, undoing the product rule.</p>`,

  intuition: String.raw`<p>Substitution stretches or relabels the axis: a chunk of width $dx$ in $x$ becomes width
  $du=g'(x)\,dx$ in $u$, and the integral follows. By parts shifts the derivative from one factor to the other, hoping
  the new integral is simpler.</p>
  <figure class="figure">
  <svg viewBox="0 0 280 110" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Substitution relabels the axis u = g(x)">
    <line x1="20" y1="35" x2="260" y2="35" stroke="#4f46e5" stroke-width="2"/>
    <text x="262" y="39" font-size="12" fill="#4f46e5" font-family="sans-serif">x</text>
    <line x1="20" y1="85" x2="260" y2="85" stroke="#0d9488" stroke-width="2"/>
    <text x="262" y="89" font-size="12" fill="#0d9488" font-family="sans-serif">u</text>
    <g stroke="#94a3b8">
    <line x1="60" y1="35" x2="50" y2="85" marker-end="url(#u1)"/>
    <line x1="120" y1="35" x2="135" y2="85" marker-end="url(#u1)"/>
    <line x1="190" y1="35" x2="225" y2="85" marker-end="url(#u1)"/>
    </g>
    <text x="86" y="105" font-size="10" fill="#64748b" font-family="sans-serif">u = g(x),   du = g′(x) dx  (axis stretches)</text>
    <defs><marker id="u1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>Substitution maps x to u; the spacing changes by the derivative g′(x), which is the du factor.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Substitution:</strong> if $u=g(x)$ then $du=g'(x)\,dx$ and</p>
  $$\int f\big(g(x)\big)\,g'(x)\,dx=\int f(u)\,du.$$
  <p><strong>Integration by parts:</strong> from the product rule,</p>
  $$\int u\,dv=uv-\int v\,du.$$
  <p>Pick $u$ to be the part that gets simpler when differentiated, and $dv$ the part you can integrate. Plain English:
  substitution collapses an inner function; by parts moves the derivative to the other factor.</p>`,

  derivation: String.raw`<p><strong>Both rules come from reversing differentiation rules.</strong></p>
  <p><strong>Substitution = reverse chain rule.</strong> If $F'=f$, then by the chain rule
  $\frac{d}{dx}F(g(x))=f(g(x))\,g'(x)$. Integrating both sides, $\int f(g(x))g'(x)\,dx=F(g(x))=\int f(u)\,du$ with
  $u=g(x)$. $\blacksquare$</p>
  <p><strong>Integration by parts = reverse product rule.</strong> The product rule gives $(uv)'=u'v+uv'$. Integrate
  both sides over $x$: $uv=\int u'v\,dx+\int uv'\,dx$. Rearranged, $\int uv'\,dx=uv-\int u'v\,dx$, i.e.
  $\int u\,dv=uv-\int v\,du$. $\blacksquare$</p>
  <hr class="soft">
  <p><strong>Worked by-parts:</strong> $\int x\,e^{x}\,dx$. Choose $u=x$ (simplifies to $du=dx$) and $dv=e^{x}dx$
  (so $v=e^{x}$). Then $\int x e^{x}dx=xe^{x}-\int e^{x}\,dx=xe^{x}-e^{x}+C=e^{x}(x-1)+C.$ Plain English: differentiating
  $x$ away turned a product into a single exponential.</p>`,

  code: [
    { label: "Substitution, verified numerically", src: String.raw`
import numpy as np

# ∫ 2x cos(x²) dx  with u = x²  ->  ∫ cos(u) du = sin(x²)
F = lambda x: np.sin(x**2)                 # claimed antiderivative
f = lambda x: 2*x*np.cos(x**2)             # integrand
x = np.linspace(0, 1.5, 200001)
print("numeric ∫₀^1.5 :", round(float(np.trapz(f(x), x)), 6))
print("F(1.5)-F(0)    :", round(float(F(1.5)-F(0)), 6))    # match
` },
    { label: "Integration by parts: ∫ x e^x dx", src: String.raw`
import numpy as np

f = lambda x: x*np.exp(x)
F = lambda x: np.exp(x)*(x-1)              # by-parts result
x = np.linspace(0, 2, 200001)
print("numeric ∫₀² x e^x :", round(float(np.trapz(f(x), x)), 5))
print("F(2)-F(0)         :", round(float(F(2)-F(0)), 5))   # match
` }
  ],

  keyPoints: [
    "Substitution reverses the chain rule: $\\int f(g(x))g'(x)\\,dx=\\int f(u)\\,du$, $u=g(x)$.",
    "Integration by parts reverses the product rule: $\\int u\\,dv=uv-\\int v\\,du$.",
    "Choose $u$ to simplify on differentiation, $dv$ to be integrable.",
    "For definite integrals, change the limits too when you substitute.",
    "Change of variables generalizes to densities (the Jacobian factor) in Track 10."
  ],

  commonMistakes: [
    { wrong: "Forgetting the $du=g'(x)\\,dx$ factor.", why: "Substitution only collapses cleanly when the integrand already contains $g'(x)$. Dropping it changes the integral entirely." },
    { wrong: "Not converting the limits in a definite substitution.", why: "After $u=g(x)$, the bounds become $g(a)$ and $g(b)$. Keeping the old $x$-limits with the new $u$-variable is a classic error." },
    { wrong: "Choosing $u$ and $dv$ badly in by-parts.", why: "A poor choice gives a harder integral. Heuristic (LIATE): pick $u$ as the Logs/Inverse-trig/Algebraic part before Trig/Exponential, since those simplify when differentiated." }
  ],

  quiz: [
    { q: "$\\int 2x(x^2+1)^3\\,dx$ — best substitution is…", options: ["$u=x^2+1$", "$u=2x$", "$u=x^3$", "$u=x$"], answer: 0,
      explain: "Then $du=2x\\,dx$ is present, giving $\\int u^3\\,du=\\tfrac{(x^2+1)^4}{4}+C$." },
    { q: "$\\int u\\,dv$ equals…", options: ["$uv-\\int v\\,du$", "$uv+\\int v\\,du$", "$\\int v\\,du-uv$", "$uv$"], answer: 0,
      explain: "Integration by parts: $uv$ minus the integral of $v\\,du$." },
    { q: "$\\int x e^x dx$ equals…", options: ["$e^x(x-1)+C$", "$xe^x+C$", "$e^x(x+1)+C$", "$\\tfrac{x^2}{2}e^x+C$"], answer: 0,
      explain: "By parts with $u=x$, $dv=e^x dx$: $xe^x-\\int e^x dx=e^x(x-1)+C$." },
    { q: "In a definite substitution $u=g(x)$ over $[a,b]$, the new limits are…", options: ["$g(a)$ to $g(b)$", "$a$ to $b$", "$0$ to $1$", "$g'(a)$ to $g'(b)$"], answer: 0,
      explain: "The variable changed, so the bounds must be evaluated under $g$." },
    { q: "Substitution is the integral form of which rule?", options: ["the chain rule", "the product rule", "the quotient rule", "the power rule"], answer: 0,
      explain: "It reverses the chain rule; by parts reverses the product rule." }
  ],

  practice: [
    { level: "easy", prompt: "Evaluate $\\int \\cos(3x)\\,dx$.", solution: "$u=3x$, $du=3dx$: $\\tfrac13\\int\\cos u\\,du=\\tfrac13\\sin(3x)+C$." },
    { level: "med", prompt: "Compute $\\int_0^1 x\\,e^{x^2}\\,dx$.", solution: "$u=x^2$, $du=2x\\,dx$: $\\tfrac12\\int_0^1 e^{u}\\,du=\\tfrac12(e-1)\\approx0.859$." },
    { level: "med", prompt: "Find $\\int \\ln x\\,dx$ by parts.", solution: "$u=\\ln x$, $dv=dx$ → $du=\\tfrac1x dx$, $v=x$. $\\int\\ln x\\,dx=x\\ln x-\\int x\\cdot\\tfrac1x dx=x\\ln x-x+C$." },
    { level: "hard", prompt: "AI task: the reparameterization trick writes $z=\\mu+\\sigma\\varepsilon$ with $\\varepsilon\\sim\\mathcal N(0,1)$. Relate this to $u$-substitution and explain why it helps training.", solution: "It's a change of variables from $\\varepsilon$ to $z$: the map $g(\\varepsilon)=\\mu+\\sigma\\varepsilon$ has constant derivative $\\sigma$, so $dz=\\sigma\\,d\\varepsilon$ — a clean linear substitution. Crucially, the randomness now lives in the fixed $\\varepsilon$, while $\\mu$ and $\\sigma$ are differentiable parameters <em>outside</em> the sampling. So an expectation $E_{z}[f(z)]=E_{\\varepsilon}[f(\\mu+\\sigma\\varepsilon)]$ can be differentiated w.r.t. $\\mu,\\sigma$ by moving the gradient inside — the substitution makes a sampled integral differentiable, which is exactly what lets VAEs backprop through randomness (Track 10)." }
  ],

  deepDive: String.raw`<p><strong>Change of variables, from one integral to whole distributions.</strong></p>
  <p>One-dimensional substitution, $\int f(g(x))g'(x)\,dx=\int f(u)\,du$, hides a deep idea: when you relabel the
  variable, the measure $dx$ must be rescaled by the local stretch $g'(x)$. In higher dimensions that single derivative
  becomes the absolute <strong>Jacobian determinant</strong> $|\det J_g|$, the factor by which $g$ scales volume. This is
  the multivariate change-of-variables formula, and it's the precise rule a normalizing flow uses to turn a simple
  density into a complex one while keeping the total probability equal to 1.</p>
  <p>So the humble $u$-substitution you use to crack $\int 2x\cos(x^2)dx$ is the same mathematics that, scaled up, lets a
  generative model transform Gaussian noise into faces or audio: each invertible layer relabels the space and pays the
  Jacobian-determinant 'toll' to keep densities honest. Integration by parts, meanwhile, is the workhorse behind many
  bounds and identities in variational inference — including manipulations of the ELBO and score-function estimators.
  These two techniques are not just exam tricks; they are how probability gets moved through transformations in modern
  generative models.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["7.4"] = {
  subtitle: "Why ∫e^{−x²}dx = √π — and where the Normal distribution's constant comes from.",

  aiMoment: String.raw`<p>The Normal distribution's density is $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$.
  That $\sqrt{2\pi\sigma^2}$ out front isn't arbitrary — it's the value of the <strong>Gaussian integral</strong>, the
  area under the bell curve, which makes the density integrate to 1. The same integral is the continuous partition
  function in energy-based models, and it underlies the whole Gaussian machinery of weight initialization, VAEs, and
  diffusion. It's a beautiful derivation with a famous trick.</p>`,

  plainEnglish: String.raw`<p>The function $e^{-x^2}$ is the bell curve, and it has <em>no</em> elementary
  antiderivative — yet the total area under it from $-\infty$ to $\infty$ is exactly $\sqrt{\pi}$. You can't get it by
  the usual rules; you get it by a clever trick: square the integral, turn it into a 2-D one, and switch to polar
  coordinates.</p>`,

  intuition: String.raw`<p>One bell curve is hard, but two — one in $x$, one in $y$ — make a round 2-D hump whose
  circular symmetry is perfect for polar coordinates. In polar form the integral becomes elementary, and taking the
  square root gives the answer for the original 1-D curve.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The bell curve with shaded area equal to root pi">
    <line x1="20" y1="120" x2="240" y2="120" stroke="#94a3b8"/>
    <path d="M30,120 C90,120 100,30 130,30 C160,30 170,120 230,120 Z" fill="#eef0ff" stroke="#4f46e5" stroke-width="2"/>
    <text x="100" y="95" font-size="12" fill="#4f46e5" font-family="sans-serif">e^(−x²)</text>
    <text x="92" y="138" font-size="11" fill="#64748b" font-family="sans-serif">total area = √π</text>
  </svg>
  <figcaption>No elementary antiderivative, yet the enclosed area is exactly √π — found via the polar trick.</figcaption>
  </figure>`,

  formalism: String.raw`<p>Define $I=\displaystyle\int_{-\infty}^{\infty} e^{-x^2}\,dx$. The result and its scaled cousin:</p>
  $$\int_{-\infty}^{\infty} e^{-x^2}\,dx=\sqrt{\pi},\qquad \int_{-\infty}^{\infty} e^{-ax^2}\,dx=\sqrt{\tfrac{\pi}{a}}\ (a>0).$$
  <p>Setting $a=\tfrac{1}{2\sigma^2}$ gives $\int e^{-x^2/2\sigma^2}\,dx=\sqrt{2\pi\sigma^2}$, the normalizing constant of
  the Gaussian.</p>`,

  derivation: String.raw`<p><strong>The polar-coordinates trick.</strong></p>
  <p><strong>Step 1 — square it and make it 2-D.</strong> Using a dummy variable $y$ for the second copy,
  $I^2=\Big(\int e^{-x^2}dx\Big)\Big(\int e^{-y^2}dy\Big)=\iint_{\mathbb{R}^2} e^{-(x^2+y^2)}\,dx\,dy.$</p>
  <p><strong>Step 2 — switch to polar coordinates.</strong> With $x^2+y^2=r^2$ and area element $dx\,dy=r\,dr\,d\theta$:
  $I^2=\int_0^{2\pi}\!\!\int_0^{\infty} e^{-r^2}\,r\,dr\,d\theta.$</p>
  <p><strong>Step 3 — the radial integral is elementary.</strong> Substitute $u=r^2$, $du=2r\,dr$:
  $\int_0^{\infty} e^{-r^2}r\,dr=\tfrac12\int_0^{\infty} e^{-u}\,du=\tfrac12.$</p>
  <p><strong>Step 4 — finish.</strong> $I^2=\int_0^{2\pi}\tfrac12\,d\theta=2\pi\cdot\tfrac12=\pi$, so $I=\sqrt{\pi}.\;\blacksquare$
  The 2-D version had the rotational symmetry that the 1-D one lacked, and that's the whole secret.</p>
  <hr class="soft">
  <p><strong>Normalizing the Gaussian.</strong> Substitute $x\to x/(\sigma\sqrt2)$ in $I$ to get
  $\int e^{-x^2/2\sigma^2}dx=\sqrt{2\pi\sigma^2}$. Hence dividing by this makes
  $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-x^2/2\sigma^2}$ integrate to 1 — a valid density.</p>`,

  code: [
    { label: "Numerically confirm ∫e^{−x²} = √π", src: String.raw`
import numpy as np

x = np.linspace(-12, 12, 400001)
I = np.trapz(np.exp(-x**2), x)
print("numeric ∫ e^(-x²) dx =", round(float(I), 6))
print("√π                   =", round(float(np.sqrt(np.pi)), 6))   # match
` },
    { label: "The Gaussian normalizing constant", src: String.raw`
import numpy as np

sigma = 2.0
x = np.linspace(-40, 40, 800001)
Z = np.trapz(np.exp(-x**2/(2*sigma**2)), x)
print("∫ e^(-x²/2σ²) dx     =", round(float(Z), 5))
print("√(2π σ²)             =", round(float(np.sqrt(2*np.pi*sigma**2)), 5))
# so the normalized density integrates to 1:
p = np.exp(-x**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
print("∫ normalized density =", round(float(np.trapz(p, x)), 6))
` }
  ],

  keyPoints: [
    "$\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx=\\sqrt{\\pi}$ — derived via the polar trick, not an antiderivative.",
    "$e^{-x^2}$ has no elementary antiderivative; the definite integral still has a clean value.",
    "$\\int e^{-ax^2}dx=\\sqrt{\\pi/a}$; with $a=1/2\\sigma^2$ this gives $\\sqrt{2\\pi\\sigma^2}$.",
    "That constant is exactly the Gaussian's normalizer $1/\\sqrt{2\\pi\\sigma^2}$.",
    "Squaring to 2-D and using rotational symmetry is the key move."
  ],

  commonMistakes: [
    { wrong: "Trying to find an antiderivative of $e^{-x^2}$.", why: "There isn't one in elementary terms. The related $\\operatorname{erf}$ is <em>defined</em> as this integral; you evaluate the definite version by the polar trick or numerically." },
    { wrong: "Forgetting the $r$ in $dx\\,dy=r\\,dr\\,d\\theta$.", why: "The Jacobian of the polar change of variables contributes the factor $r$ — and it's exactly what makes the radial integral solvable. Dropping it gives the wrong (divergent) answer." },
    { wrong: "Misremembering the constant as $\\sqrt{2\\pi}\\sigma$ vs $\\sqrt{2\\pi\\sigma^2}$.", why: "They're equal: $\\sqrt{2\\pi\\sigma^2}=\\sqrt{2\\pi}\\,\\sigma$. Confusion arises when $\\sigma$ vs $\\sigma^2$ (variance) is meant — always check which one the density uses." }
  ],

  quiz: [
    { q: "$\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx$ equals…", options: ["$\\sqrt{\\pi}$", "$\\pi$", "$1$", "$2\\pi$"], answer: 0,
      explain: "The Gaussian integral evaluates to $\\sqrt{\\pi}$ via the polar trick." },
    { q: "The trick to evaluate it is to…", options: ["square it and use polar coordinates", "integrate by parts", "use the power rule", "differentiate under the integral"], answer: 0,
      explain: "Squaring gives a 2-D rotationally-symmetric integral, elementary in polar form." },
    { q: "$\\int e^{-3x^2}\\,dx$ over $\\mathbb{R}$ equals…", options: ["$\\sqrt{\\pi/3}$", "$\\sqrt{3\\pi}$", "$\\sqrt{\\pi}/3$", "$3\\sqrt{\\pi}$"], answer: 0,
      explain: "$\\int e^{-ax^2}dx=\\sqrt{\\pi/a}$ with $a=3$." },
    { q: "The Gaussian's normalizing constant for variance $\\sigma^2$ is…", options: ["$1/\\sqrt{2\\pi\\sigma^2}$", "$1/(2\\pi\\sigma)$", "$1/\\sigma$", "$1/\\sqrt{\\pi}$"], answer: 0,
      explain: "Because $\\int e^{-x^2/2\\sigma^2}dx=\\sqrt{2\\pi\\sigma^2}$, divide by it to normalize." },
    { q: "In polar coordinates, $dx\\,dy$ becomes…", options: ["$r\\,dr\\,d\\theta$", "$dr\\,d\\theta$", "$r^2\\,dr\\,d\\theta$", "$\\tfrac1r dr\\,d\\theta$"], answer: 0,
      explain: "The Jacobian of the polar transform is $r$, giving the area element $r\\,dr\\,d\\theta$." }
  ],

  practice: [
    { level: "easy", prompt: "What is $\\int_{-\\infty}^{\\infty} e^{-x^2/2}\\,dx$?", solution: "$\\int e^{-ax^2}dx=\\sqrt{\\pi/a}$ with $a=\\tfrac12$: $\\sqrt{2\\pi}\\approx2.5066$ (the standard normal's constant)." },
    { level: "med", prompt: "Carry out the radial integral $\\int_0^\\infty r e^{-r^2}\\,dr$ in full.", solution: "Let $u=r^2$, $du=2r\\,dr$, so $r\\,dr=\\tfrac12 du$. Then $\\int_0^\\infty\\tfrac12 e^{-u}\\,du=\\tfrac12[-e^{-u}]_0^\\infty=\\tfrac12(0-(-1))=\\tfrac12$." },
    { level: "med", prompt: "Use the result to show the standard normal density integrates to 1.", solution: "$\\int\\frac{1}{\\sqrt{2\\pi}}e^{-x^2/2}dx=\\frac{1}{\\sqrt{2\\pi}}\\int e^{-x^2/2}dx=\\frac{1}{\\sqrt{2\\pi}}\\cdot\\sqrt{2\\pi}=1.$ The normalizer is exactly the Gaussian integral's value." },
    { level: "hard", prompt: "AI task: the log-density of a Gaussian is $-\\tfrac{(x-\\mu)^2}{2\\sigma^2}-\\tfrac12\\ln(2\\pi\\sigma^2)$. Identify each term's origin and which matters for maximum-likelihood fitting of $\\mu$.", solution: "The first term is the (negative) squared error scaled by the variance — the 'data fit'. The second, $-\\tfrac12\\ln(2\\pi\\sigma^2)$, is the log of the normalizing constant from the Gaussian integral, ensuring the density integrates to 1. For MLE of $\\mu$ (with $\\sigma$ fixed), only the first term depends on $\\mu$, and maximizing it is minimizing $(x-\\mu)^2$ — i.e. least squares. So Gaussian MLE for the mean <em>is</em> least-squares, and the integral's constant is exactly the piece that drops out when differentiating w.r.t. $\\mu$ (but matters when fitting $\\sigma$). This links Track 7 to MLE in Track 11." }
  ],

  deepDive: String.raw`<p><strong>Why this one integral underwrites so much of ML.</strong></p>
  <p>The Gaussian integral is the load-bearing constant of probabilistic machine learning. Every time a model assumes
  Gaussian noise — linear regression's likelihood, a VAE's prior and posterior, the forward process of a diffusion
  model, Gaussian processes — the density it writes down is only valid because $\sqrt{2\pi\sigma^2}$ makes it integrate
  to 1, and that number is the Gaussian integral. The multivariate version, $\int_{\mathbb{R}^d}e^{-\frac12\mathbf x^\top\Sigma^{-1}\mathbf x}\,d\mathbf x=\sqrt{(2\pi)^d\det\Sigma}$,
  is the same trick generalized (diagonalize $\Sigma$, integrate each axis), and its $\det\Sigma$ is why covariance
  determinants show up in Gaussian log-likelihoods.</p>
  <p>The polar trick also teaches a transferable lesson: a hard problem can become easy in the right coordinate system,
  and choosing coordinates that match a problem's symmetry (here, rotational) is a recurring superpower — the same
  instinct behind diagonalizing in the eigenbasis (Track 5) and whitening data. When you next see $2\pi$ inside a loss or
  a density, you'll know it traces back to this single, elegant evaluation — and that the constant you often ignore is
  doing the quiet work of keeping probabilities probabilities.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["7.5"] = {
  subtitle: "Expectation is a probability-weighted average — an integral we approximate by sampling.",

  aiMoment: String.raw`<p>The thing training actually minimizes is an <strong>expected loss</strong>,
  $E_{x\sim p}[\ell(x)]=\int \ell(x)\,p(x)\,dx$ — an integral over the data distribution. We never compute that integral
  exactly; we estimate it by averaging the loss over a <strong>minibatch</strong>, which is <strong>Monte Carlo</strong>
  integration. Policy-gradient RL, VAE objectives, and dropout-at-inference all rest on the same move: an expectation is
  an integral, and an average of samples estimates it.</p>`,

  plainEnglish: String.raw`<p>The <strong>expected value</strong> of a quantity is its average, weighted by how likely
  each outcome is. For continuous outcomes that weighted average is an integral. Since we usually can't do the integral,
  we draw samples and average — the more samples, the closer the estimate.</p>`,

  intuition: String.raw`<p>Picture the density as a pile of sand on a ruler; the expected value is the balance point —
  where it would teeter on a fulcrum. To find it without weighing the whole pile, grab random handfuls, note where they
  came from, and average — that's the Monte Carlo estimate.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Expected value as the balance point of a density">
    <path d="M30,100 C80,100 90,35 140,35 C190,35 200,100 240,100 Z" fill="#eef0ff" stroke="#4f46e5" stroke-width="2"/>
    <line x1="140" y1="100" x2="140" y2="35" stroke="#dc2626" stroke-dasharray="3 3"/>
    <polygon points="140,104 132,120 148,120" fill="#dc2626"/>
    <text x="118" y="128" font-size="11" fill="#dc2626" font-family="sans-serif">E[X] (balance)</text>
  </svg>
  <figcaption>The expected value is the distribution's balance point — its probability-weighted center.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For a continuous random variable $X$ with density $p$, and any function $g$ (the "law of the
  unconscious statistician"):</p>
  $$E[g(X)]=\int g(x)\,p(x)\,dx,\qquad \operatorname{Var}(X)=E\big[(X-E[X])^2\big]=\int (x-\mu)^2 p(x)\,dx.$$
  <p>The <strong>Monte Carlo estimate</strong> draws $x_1,\dots,x_n\sim p$ and averages:
  $\widehat{E[g]}=\frac1n\sum_{i=1}^{n} g(x_i)$. Plain English: replace the weighted integral with a plain average over
  samples drawn from the right distribution.</p>`,

  derivation: String.raw`<p><strong>Why the sample average works: it's unbiased, and its error shrinks like $1/\sqrt n$.</strong></p>
  <p><strong>Step 1 — unbiasedness.</strong> With $x_i$ drawn iid from $p$,
  $E\big[\widehat{E[g]}\big]=E\Big[\tfrac1n\sum_i g(x_i)\Big]=\tfrac1n\sum_i E[g(x_i)]=\tfrac1n\cdot n\cdot E[g(X)]=E[g(X)].$
  The estimate is centered on the true expectation — no systematic error.</p>
  <p><strong>Step 2 — variance.</strong> Independent samples give
  $\operatorname{Var}\big(\widehat{E[g]}\big)=\tfrac1{n^2}\sum_i\operatorname{Var}(g(x_i))=\dfrac{\operatorname{Var}(g(X))}{n}.$</p>
  <p><strong>Step 3 — error rate.</strong> The standard error is the square root: $\dfrac{\sigma_g}{\sqrt n}$. So to halve
  the error you need $4\times$ the samples. $\blacksquare$ Plain English: averaging more samples zeroes in on the
  integral, and crucially the rate $1/\sqrt n$ does <em>not</em> depend on the dimension — which is why Monte Carlo beats
  grid-based integration in high dimensions.</p>`,

  code: [
    { label: "Monte Carlo: estimate an expectation by sampling", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

# X ~ N(0,1):  E[X^2] = Var(X) = 1
for n in [100, 10_000, 1_000_000]:
    x = rng.normal(size=n)
    print(f"n={n:9d}:  MC estimate of E[X²] =", round(float(np.mean(x**2)), 4))
print("true E[X²] = 1")
` },
    { label: "Minibatch loss is a Monte Carlo estimate", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

data = rng.normal(loc=3.0, size=100_000)        # 'dataset'
loss = lambda x, w: (x - w)**2                  # per-example loss
w = 2.5

full = np.mean(loss(data, w))                   # true expected loss (whole set)
batch = np.mean(loss(rng.choice(data, 64), w))  # minibatch estimate
print("full-data loss :", round(full, 4))
print("minibatch est. :", round(batch, 4), " (noisy but unbiased)")
` }
  ],

  keyPoints: [
    "Expectation is a probability-weighted average: $E[g(X)]=\\int g(x)p(x)\\,dx$.",
    "Variance is the expected squared deviation: $\\operatorname{Var}(X)=E[(X-\\mu)^2]$.",
    "Monte Carlo estimates an expectation by a sample average $\\frac1n\\sum g(x_i)$.",
    "The estimate is unbiased; its error is $\\sigma_g/\\sqrt n$ — independent of dimension.",
    "A minibatch gradient is a Monte Carlo estimate of the full-data gradient."
  ],

  commonMistakes: [
    { wrong: "Sampling from the wrong distribution.", why: "Monte Carlo only estimates $E_{p}[g]$ if the samples come from $p$. Using a different sampler needs an importance-weight correction $g(x)\\,p(x)/q(x)$." },
    { wrong: "Expecting error to fall linearly with $n$.", why: "It falls as $1/\\sqrt n$: 100× the samples gives only 10× less error. This sub-linear rate is why MC estimates stay noisy." },
    { wrong: "Confusing the density value with a probability.", why: "$p(x)$ is a density (can exceed 1); the probability of an interval is its integral $\\int p$. Expectation weights by the density, then integrates." }
  ],

  quiz: [
    { q: "$E[g(X)]$ for continuous $X$ is…", options: ["$\\int g(x)p(x)\\,dx$", "$\\sum g(x)$", "$\\int g(x)\\,dx$", "$g(E[X])$"], answer: 0,
      explain: "Weight $g$ by the density and integrate. ($g(E[X])$ is generally different — Jensen, Track 12.)" },
    { q: "A Monte Carlo estimate's standard error scales as…", options: ["$1/\\sqrt n$", "$1/n$", "$1/n^2$", "constant"], answer: 0,
      explain: "Variance $\\propto1/n$, so the standard error $\\propto1/\\sqrt n$." },
    { q: "For $X\\sim\\mathcal N(0,1)$, $E[X^2]$ equals…", options: ["1", "0", "2", "$\\sqrt{2\\pi}$"], answer: 0,
      explain: "$E[X^2]=\\operatorname{Var}(X)+E[X]^2=1+0=1$." },
    { q: "A minibatch gradient relates to the full gradient as…", options: ["an unbiased Monte Carlo estimate", "a biased lower bound", "exactly equal always", "the variance"], answer: 0,
      explain: "Averaging the gradient over a random subset is an unbiased estimate of the expected (full-data) gradient." },
    { q: "Why does Monte Carlo beat grid integration in high dimensions?", options: ["its $1/\\sqrt n$ rate is dimension-independent", "it is exact", "grids are biased", "it needs no samples"], answer: 0,
      explain: "A grid needs exponentially many points in $d$; Monte Carlo's error rate doesn't depend on $d$, only on $n$." }
  ],

  practice: [
    { level: "easy", prompt: "For a fair die ($X\\in\\{1,\\dots,6\\}$), compute $E[X]$.", solution: "$\\sum_{k=1}^6 k\\cdot\\tfrac16=\\tfrac{21}{6}=3.5$ — the balance point (discrete expectation is a weighted sum)." },
    { level: "med", prompt: "If $E[X]=2$ and $E[X^2]=5$, find $\\operatorname{Var}(X)$.", solution: "$\\operatorname{Var}(X)=E[X^2]-E[X]^2=5-4=1$." },
    { level: "med", prompt: "You estimate $E[g]$ with $n=100$ samples and get standard error $0.2$. How many samples for standard error $0.05$?", solution: "Error scales as $1/\\sqrt n$: to cut it by $4\\times$ ($0.2\\to0.05$) needs $4^2=16\\times$ the samples, so $n=1600$." },
    { level: "hard", prompt: "AI task: REINFORCE estimates $\\nabla_\\theta E_{x\\sim p_\\theta}[r(x)]$. Explain why you can't just push the gradient inside the integral naively, and how the log-derivative trick fixes it.", solution: "The distribution $p_\\theta$ itself depends on $\\theta$, so $\\nabla_\\theta\\int r(x)p_\\theta(x)dx=\\int r(x)\\nabla_\\theta p_\\theta(x)dx$ — but $\\nabla_\\theta p_\\theta$ isn't an expectation you can sample directly. The log-derivative ('score function') trick rewrites $\\nabla_\\theta p_\\theta=p_\\theta\\nabla_\\theta\\log p_\\theta$, giving $\\nabla_\\theta E[r]=E_{x\\sim p_\\theta}[\\,r(x)\\nabla_\\theta\\log p_\\theta(x)\\,]$ — now an expectation you <em>can</em> Monte-Carlo by sampling actions and weighting their log-prob gradients by the reward. (The reparameterization trick, Track 10, is the alternative when $x$ is differentiable in $\\theta$.)" }
  ],

  deepDive: String.raw`<p><strong>Expectation is the through-line from calculus to all of probabilistic ML.</strong></p>
  <p>Almost every training objective is an expectation, and almost every algorithm is a way to estimate one or its
  gradient. Supervised learning minimizes $E_{(x,y)\sim\text{data}}[\ell]$; we estimate it with minibatches.
  Reinforcement learning maximizes $E_{\tau\sim\pi}[\text{return}]$; we estimate its gradient with rollouts. VAEs
  optimize $E_{z\sim q}[\log p(x\mid z)]$ minus a KL; the expectation is estimated with one or a few samples per data
  point. Even Bayesian prediction, $p(y\mid x)=E_{\theta\sim\text{posterior}}[p(y\mid x,\theta)]$, is an expectation over
  parameters. The integral $\int g\,p$ is the object; sampling is how we touch it.</p>
  <p>Two recurring concerns follow directly from this lesson's variance result. First, <strong>variance reduction</strong>
  — control variates, baselines in policy gradients, antithetic sampling — exists because the $1/\sqrt n$ rate is slow,
  and shaving the constant $\sigma_g$ matters enormously. Second, the <strong>bias–variance</strong> character of an
  estimator (Track 11) decides whether more samples or a better estimator helps. So this is where Calculus II hands off
  to probability and statistics: once you see that learning is integration-by-expectation, the rest of the course is
  largely about estimating those expectations well. The Law of Large Numbers (Track 10) is the guarantee that the whole
  enterprise converges.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["7.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 65 minutes.",

  intro: String.raw`<p>This exam spans all of Track 7: the definite integral and antiderivatives, the FTC,
  substitution and integration by parts, the Gaussian integral, and integrals as expectation. <strong>Give yourself 65
  minutes</strong>, produce each answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "Evaluate $\\int_0^2 (3x^2+1)\\,dx$.",
      solution: "Antiderivative $x^3+x$: $(8+2)-(0)=10$." },
    { level: "easy", prompt: "$\\frac{d}{dx}\\int_2^x \\sin(t^3)\\,dt = ?$",
      solution: "By FTC1, it's the integrand at $x$: $\\sin(x^3)$." },
    { level: "med", prompt: "Compute $\\int_0^1 x\\,e^{x^2}\\,dx$ by substitution.",
      solution: "$u=x^2$, $du=2x\\,dx$: $\\tfrac12\\int_0^1 e^{u}\\,du=\\tfrac12(e-1)\\approx0.859$." },
    { level: "med", prompt: "Find $\\int x\\sin x\\,dx$ by parts.",
      solution: "$u=x$, $dv=\\sin x\\,dx$ → $du=dx$, $v=-\\cos x$. $\\int x\\sin x\\,dx=-x\\cos x+\\int\\cos x\\,dx=-x\\cos x+\\sin x+C$." },
    { level: "med", prompt: "State $\\int_{-\\infty}^{\\infty} e^{-2x^2}\\,dx$.",
      solution: "$\\int e^{-ax^2}dx=\\sqrt{\\pi/a}$ with $a=2$: $\\sqrt{\\pi/2}\\approx1.2533$." },
    { level: "med", prompt: "For $X\\sim\\mathcal N(0,\\sigma^2)$, what is $E[X^2]$, and why?",
      solution: "$E[X^2]=\\operatorname{Var}(X)+E[X]^2=\\sigma^2+0=\\sigma^2$ — the second moment of a zero-mean variable is its variance." },
    { level: "hard", prompt: "Show that the standard normal density integrates to 1.",
      solution: "$\\int\\frac{1}{\\sqrt{2\\pi}}e^{-x^2/2}dx=\\frac{1}{\\sqrt{2\\pi}}\\int e^{-x^2/2}dx$. By the Gaussian integral, $\\int e^{-x^2/2}dx=\\sqrt{2\\pi}$, so the product is $\\frac{1}{\\sqrt{2\\pi}}\\cdot\\sqrt{2\\pi}=1$." },
    { level: "hard", prompt: "A Monte Carlo estimate of $E[g]$ with $n=400$ samples has standard error $0.1$. How many samples to reach $0.02$?",
      solution: "Standard error $\\propto1/\\sqrt n$. To shrink it by $5\\times$ ($0.1\\to0.02$) needs $5^2=25\\times$ samples: $n=400\\cdot25=10{,}000$." },
    { level: "hard", prompt: "Why is the sample-mean estimator of an expectation unbiased, and what is its variance for iid samples?",
      solution: "$E[\\tfrac1n\\sum g(x_i)]=\\tfrac1n\\sum E[g(x_i)]=E[g(X)]$ — centered on the truth, hence unbiased. With independent samples, $\\operatorname{Var}(\\tfrac1n\\sum g(x_i))=\\tfrac1{n^2}\\cdot n\\,\\operatorname{Var}(g(X))=\\operatorname{Var}(g(X))/n$, so the standard error is $\\sigma_g/\\sqrt n$." },
    { level: "hard", prompt: "AI task: explain why minibatch SGD works as an optimizer despite using noisy gradients.",
      solution: "The minibatch gradient is an <em>unbiased</em> Monte Carlo estimate of the full-data gradient: $E[\\nabla\\ell_{\\text{batch}}]=\\nabla E[\\ell]$. So on average each step points in the true descent direction, and the noise (variance $\\propto1/\\text{batch size}$) averages out over many steps. The noise even helps — it lets SGD escape sharp minima and saddle points. The $1/\\sqrt{\\text{batch}}$ error rate is why larger batches give smoother but not proportionally better updates, and why learning-rate and batch-size are tuned together." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Integration and expectation are solid — you can read training objectives as integrals. On to Probability (Track 9) and Optimization (Track 8).</li>
    <li><strong>7–8:</strong> Strong. Revisit the Gaussian integral derivation or Monte Carlo variance if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive the polar trick and the unbiasedness of the sample mean; redo Lessons 7.4 and 7.5.</li>
    <li><strong>Below 5:</strong> Rework the track — expectation-as-integral is the backbone of every probabilistic method ahead.</li>
  </ul>`
};
