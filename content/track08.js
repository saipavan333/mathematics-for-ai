/* ============================================================
   TRACK 8 — Optimization Theory
   8.1 Minima/Maxima/Saddle · 8.2 Convexity · 8.3 Gradient Descent ·
   8.4 Momentum & Adam · 8.5 Lagrange & KKT · 8.E Track Exam
   ============================================================ */
(window.LESSON_CONTENT ||= {})["8.1"] = {
  subtitle: "Where the slope is zero — valley, hilltop, or mountain pass — and how the Hessian tells them apart.",

  aiMoment: String.raw`<p>For years people feared deep networks were riddled with bad local minima. The surprising
  empirical truth: in high dimensions, almost every point where training "stalls" is a <strong>saddle</strong> — flat or
  rising in some directions, falling in others — not a true minimum. Knowing the first- and second-order conditions that
  classify a critical point is how you reason about loss landscapes, why momentum helps escape saddles, and what
  Newton's method must guard against.</p>`,

  plainEnglish: String.raw`<p>A <strong>critical point</strong> is where the slope is zero — the gradient vanishes. It
  could be the bottom of a valley (<strong>minimum</strong>), the top of a hill (<strong>maximum</strong>), or a
  mountain pass that goes up one way and down another (<strong>saddle</strong>). The curvature — the Hessian — decides
  which.</p>`,

  intuition: String.raw`<p>At a minimum the surface curves up in every direction; at a maximum, down in every
  direction; at a saddle, up along some axes and down along others. The gradient is zero at all three, so you need
  second-order information (curvature) to tell them apart.</p>
  <figure class="figure">
  <svg viewBox="0 0 330 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Minimum, maximum, and saddle point shapes">
    <path d="M20,30 Q55,100 90,30" fill="none" stroke="#0d9488" stroke-width="2.4"/>
    <circle cx="55" cy="72" r="3" fill="#0d9488"/>
    <text x="36" y="115" font-size="11" fill="#0d9488" font-family="sans-serif">minimum</text>
    <path d="M125,90 Q160,20 195,90" fill="none" stroke="#dc2626" stroke-width="2.4"/>
    <circle cx="160" cy="48" r="3" fill="#dc2626"/>
    <text x="138" y="115" font-size="11" fill="#dc2626" font-family="sans-serif">maximum</text>
    <path d="M230,40 Q265,75 300,40" fill="none" stroke="#7c3aed" stroke-width="2.2"/>
    <path d="M265,108 Q265,75 230,75" fill="none" stroke="#7c3aed" stroke-width="2.2" opacity="0.5"/>
    <path d="M265,108 Q265,75 300,75" fill="none" stroke="#7c3aed" stroke-width="2.2" opacity="0.5"/>
    <circle cx="265" cy="75" r="3" fill="#7c3aed"/>
    <text x="244" y="115" font-size="11" fill="#7c3aed" font-family="sans-serif">saddle</text>
  </svg>
  <figcaption>All three have zero gradient; the Hessian's eigenvalue signs distinguish them.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A <strong>critical point</strong> $\mathbf x^*$ satisfies the first-order condition
  $\nabla f(\mathbf x^*)=\mathbf 0$. The <strong>second-order test</strong> reads the Hessian $H=\nabla^2 f(\mathbf x^*)$:
  $H\succ0$ (positive definite, all $\lambda_i>0$) ⇒ <strong>local minimum</strong>; $H\prec0$ (all $\lambda_i<0$) ⇒
  <strong>local maximum</strong>; $H$ <strong>indefinite</strong> (mixed signs) ⇒ <strong>saddle</strong>. If $H$ is only
  semidefinite (a zero eigenvalue), the test is inconclusive.</p>`,

  derivation: String.raw`<p><strong>Why the Hessian classifies critical points.</strong> Expand to second order around a
  critical point $\mathbf x^*$ (Lesson 6.5).</p>
  <p><strong>Step 1 — the linear term vanishes.</strong> Since $\nabla f(\mathbf x^*)=\mathbf 0$,
  $f(\mathbf x^*+\boldsymbol\Delta)\approx f(\mathbf x^*)+\tfrac12\boldsymbol\Delta^\top H\boldsymbol\Delta.$ The local
  shape is entirely the quadratic form $\boldsymbol\Delta^\top H\boldsymbol\Delta$.</p>
  <p><strong>Step 2 — diagonalize $H$</strong> (symmetric, Track 5.2): in its eigenbasis,
  $\boldsymbol\Delta^\top H\boldsymbol\Delta=\sum_i\lambda_i y_i^2$ where $\mathbf y$ are the rotated coordinates.</p>
  <p><strong>Step 3 — read the signs.</strong> If all $\lambda_i>0$, every term is positive, so $f$ rises in all
  directions → minimum. All $\lambda_i<0$ → falls everywhere → maximum. Mixed signs → rises along positive-$\lambda$
  axes, falls along negative ones → saddle. $\blacksquare$ Plain English: the Hessian's eigenvalues are the curvatures
  along independent axes, and their signs are the whole story.</p>`,

  code: [
    { label: "Classify a critical point by Hessian eigenvalues", src: String.raw`
import numpy as np

# f(x,y) = x² - y²  has a saddle at the origin
H = np.array([[2., 0.],
              [0., -2.]])          # Hessian of x² - y²
vals = np.linalg.eigvalsh(H)
print("eigenvalues:", vals)
kind = ("minimum" if np.all(vals>0) else
        "maximum" if np.all(vals<0) else
        "saddle"  if np.any(vals>0) and np.any(vals<0) else "degenerate")
print("critical point is a:", kind)     # saddle
` },
    { label: "Gradient is zero at all three types", src: String.raw`
import numpy as np
def grad(f, p, h=1e-6):
    p=np.asarray(p,float); g=np.zeros_like(p)
    for i in range(p.size):
        e=np.zeros_like(p); e[i]=h; g[i]=(f(p+e)-f(p-e))/(2*h)
    return g

saddle = lambda p: p[0]**2 - p[1]**2
bowl   = lambda p: p[0]**2 + p[1]**2
print("∇ saddle at origin:", np.round(grad(saddle,[0,0]),6))  # ~0
print("∇ bowl   at origin:", np.round(grad(bowl,[0,0]),6))    # ~0
# same zero gradient -> need the Hessian to tell them apart
` }
  ],

  keyPoints: [
    "Critical point: $\\nabla f=\\mathbf 0$ (first-order condition).",
    "Hessian PD ⇒ minimum; ND ⇒ maximum; indefinite ⇒ saddle.",
    "At a critical point, $f(\\mathbf x^*+\\boldsymbol\\Delta)\\approx f(\\mathbf x^*)+\\tfrac12\\boldsymbol\\Delta^\\top H\\boldsymbol\\Delta$.",
    "A zero Hessian eigenvalue makes the second-order test inconclusive.",
    "In high dimensions, saddles vastly outnumber bad local minima."
  ],

  commonMistakes: [
    { wrong: "Concluding 'minimum' from $\\nabla f=\\mathbf 0$ alone.", why: "Zero gradient only means critical. Maxima and saddles also have it — you must check the Hessian's eigenvalue signs." },
    { wrong: "Thinking deep-net training gets stuck in bad local minima.", why: "Empirically, stalls are usually saddles or flat regions; most minima found are near-equivalent in loss. Momentum and noise help slide off saddles." },
    { wrong: "Ignoring the inconclusive (semidefinite) case.", why: "If an eigenvalue is exactly 0, the quadratic is flat along that axis and higher-order terms decide — the second-order test can't classify it." }
  ],

  quiz: [
    { q: "$\\nabla f=\\mathbf 0$ and Hessian eigenvalues $\\{3,1\\}$. The point is a…", options: ["minimum", "maximum", "saddle", "can't tell"], answer: 0,
      explain: "All positive ⇒ positive definite ⇒ curves up everywhere ⇒ local minimum." },
    { q: "Hessian eigenvalues $\\{4,-1\\}$ at a critical point ⇒…", options: ["saddle", "minimum", "maximum", "degenerate"], answer: 0,
      explain: "Mixed signs ⇒ indefinite ⇒ up one way, down another ⇒ saddle." },
    { q: "For $f(x,y)=x^2-y^2$, the origin is a…", options: ["saddle", "minimum", "maximum", "plateau"], answer: 0,
      explain: "Curves up in $x$ ($+2$), down in $y$ ($-2$): a saddle. Gradient is zero there." },
    { q: "The first-order condition for a critical point is…", options: ["$\\nabla f=\\mathbf 0$", "$H\\succ0$", "$f=0$", "$\\det H=0$"], answer: 0,
      explain: "Zero gradient is necessary for any min/max/saddle; the Hessian then classifies it." },
    { q: "Why are saddles common in high dimensions?", options: ["each eigenvalue's sign is roughly a coin flip; all-same-sign is exponentially unlikely", "Hessians are always indefinite", "gradients never vanish", "minima don't exist"], answer: 0,
      explain: "For a critical point to be a minimum, all $n$ eigenvalues must be positive; with many eigenvalues, having them all one sign is increasingly rare, so mixed-sign saddles dominate." }
  ],

  practice: [
    { level: "easy", prompt: "Find the critical point of $f(x)=x^2-4x+1$ and classify it.", solution: "$f'(x)=2x-4=0\\Rightarrow x=2$. $f''=2>0$ ⇒ minimum." },
    { level: "med", prompt: "Classify the origin for $f(x,y)=x^2+3y^2$.", solution: "$\\nabla f=[2x,6y]=\\mathbf0$ at origin; Hessian $\\mathrm{diag}(2,6)$, both positive ⇒ minimum." },
    { level: "med", prompt: "Classify the origin for $f(x,y)=xy$.", solution: "$\\nabla f=[y,x]=\\mathbf0$ at origin. Hessian $\\begin{bmatrix}0&1\\\\1&0\\end{bmatrix}$ has eigenvalues $\\pm1$ (mixed) ⇒ saddle." },
    { level: "hard", prompt: "AI task: explain why the negative-eigenvalue directions of the Hessian at a saddle are exactly where an optimizer needs to move, and how SGD noise helps.", solution: "At a saddle, $\\nabla f=\\mathbf0$ so plain gradient descent barely moves. The directions with <em>negative</em> curvature are the escape routes — moving along them <em>decreases</em> the loss — but the gradient gives no push there. SGD's stochastic noise injects random components, including along negative-curvature directions, nudging the iterate off the saddle so the negative curvature can then take over and accelerate descent. Momentum similarly accumulates velocity through the flat region. This is the modern view: the challenge isn't bad minima but escaping saddles quickly, and noise/momentum are the tools." }
  ],

  deepDive: String.raw`<p><strong>The geometry of high-dimensional loss surfaces.</strong></p>
  <p>Think about what it takes to be a local minimum in $n$ dimensions: <em>every one</em> of the Hessian's $n$
  eigenvalues must be positive. If eigenvalue signs were independent coin flips, the chance of all-positive is $2^{-n}$ —
  astronomically small for a network with millions of parameters. So the overwhelming majority of critical points are
  saddles, and among the rare minima, theory and experiment suggest most are clustered at similar (low) loss values.
  This reframed deep-learning optimization: the enemy is not getting trapped in a uniquely bad basin, but spending time
  crawling across saddles and flat plateaus.</p>
  <p>It also explains a hierarchy of fixes. Plain gradient descent stalls at saddles (zero gradient). <strong>Momentum</strong>
  (Lesson 8.4) carries velocity through them. <strong>SGD noise</strong> randomly perturbs into escape directions. Genuine
  <strong>second-order</strong> methods can detect negative curvature and move along it deliberately, but the Hessian is
  too big to form (Track 6.4), so practical optimizers approximate. The empirical good news — that found solutions
  generalize well and are roughly interchangeable in loss — is part of why over-parameterized networks train so
  reliably despite a wildly non-convex objective, which is exactly the contrast the next lesson sets up.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["8.2"] = {
  subtitle: "Bowl-shaped problems have one global minimum and no surprises — the gold standard losses aim for.",

  aiMoment: String.raw`<p>Logistic regression, linear regression, and SVMs all have <strong>convex</strong> losses: a
  single global minimum, reachable by gradient descent from anywhere, with no bad local traps. Deep networks are
  <em>non</em>-convex, which is why their optimization is subtle. Recognizing convexity tells you when a problem is
  "easy" (guaranteed global optimum) and frames why the rest of deep learning is hard — and why it works anyway.</p>`,

  plainEnglish: String.raw`<p>A function is <strong>convex</strong> if it's bowl-shaped: the straight line between any two
  points on its graph never dips below the graph. The payoff is huge — for a convex function, any local minimum is
  automatically the <strong>global</strong> minimum, so you can't get stuck.</p>`,

  intuition: String.raw`<p>Draw a chord between two points on the curve. If the chord always lies on or above the curve,
  it's convex — a single valley. A non-convex function wiggles, with multiple valleys where the chord can cut below.</p>
  <figure class="figure">
  <svg viewBox="0 0 354 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Convex bowl with chord above versus non-convex wiggle">
    <polyline points="20,42 38,77 56,98 75,105 94,98 112,77 130,42" fill="none" stroke="#0d9488" stroke-width="2.4"/>
    <line x1="32" y1="67" x2="118" y2="67" stroke="#dc2626" stroke-dasharray="4 3"/>
    <circle cx="32" cy="67" r="3" fill="#dc2626"/><circle cx="118" cy="67" r="3" fill="#dc2626"/>
    <text x="30" y="124" font-size="11" fill="#0d9488" font-family="sans-serif">convex (chord above)</text>
    <path d="M180,70 Q210,20 235,75 Q260,120 290,55" fill="none" stroke="#7c3aed" stroke-width="2.4"/>
    <circle cx="222" cy="48" r="2.5" fill="#94a3b8"/><circle cx="272" cy="78" r="2.5" fill="#94a3b8"/>
    <text x="196" y="126" font-size="11" fill="#7c3aed" font-family="sans-serif">non-convex (many valleys)</text>
  </svg>
  <figcaption>Convex: one valley, chord stays above. Non-convex: several minima to get stuck in.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A function $f$ is <strong>convex</strong> if for all $\mathbf x,\mathbf y$ and $\lambda\in[0,1]$:</p>
  $$f\big(\lambda\mathbf x+(1-\lambda)\mathbf y\big)\le \lambda f(\mathbf x)+(1-\lambda)f(\mathbf y).$$
  <p>For twice-differentiable $f$, this is equivalent to the <strong>Hessian being positive semidefinite everywhere</strong>
  ($H\succeq0$); strictly convex if $H\succ0$. The key consequence: for a convex $f$, <strong>every local minimum is
  global</strong>. Convex examples: $x^2$, $e^x$, $\lVert\mathbf x\rVert$, the logistic loss.</p>`,

  derivation: String.raw`<p><strong>Why a local minimum of a convex function is global.</strong> Let $\mathbf x^*$ be a local
  minimum, and take any other point $\mathbf y$.</p>
  <p><strong>Step 1 — use convexity on the segment</strong> from $\mathbf x^*$ to $\mathbf y$: for $\lambda\in[0,1]$,
  $f\big(\mathbf x^*+\lambda(\mathbf y-\mathbf x^*)\big)\le(1-\lambda)f(\mathbf x^*)+\lambda f(\mathbf y).$</p>
  <p><strong>Step 2 — use 'local minimum'</strong>: for small enough $\lambda$, the left point is in the neighborhood of
  $\mathbf x^*$, so $f(\mathbf x^*)\le f\big(\mathbf x^*+\lambda(\mathbf y-\mathbf x^*)\big)$.</p>
  <p><strong>Step 3 — chain them:</strong> $f(\mathbf x^*)\le(1-\lambda)f(\mathbf x^*)+\lambda f(\mathbf y)$. Subtract
  $(1-\lambda)f(\mathbf x^*)$: $\lambda f(\mathbf x^*)\le\lambda f(\mathbf y)$, and divide by $\lambda>0$:
  $f(\mathbf x^*)\le f(\mathbf y).$</p>
  <p><strong>Step 4 — conclude.</strong> Since $\mathbf y$ was arbitrary, $\mathbf x^*$ is a global minimum. $\blacksquare$
  Plain English: convexity links the local picture to the whole function, so there's nowhere better to hide.</p>`,

  code: [
    { label: "Convexity via the Hessian (PSD everywhere)", src: String.raw`
import numpy as np

# f(x,y) = x² + xy + y²  -> constant Hessian [[2,1],[1,2]]
H = np.array([[2., 1.],
              [1., 2.]])
print("eigenvalues:", np.linalg.eigvalsh(H), "-> all >= 0 ? convex:",
      np.all(np.linalg.eigvalsh(H) >= 0))
` },
    { label: "Check the chord-above-graph definition", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: x**2                         # convex

ok = True
for _ in range(10000):
    x, y = rng.normal(size=2); lam = rng.random()
    lhs = f(lam*x + (1-lam)*y)
    rhs = lam*f(x) + (1-lam)*f(y)
    if lhs > rhs + 1e-9: ok = False
print("f(λx+(1-λ)y) <= λf(x)+(1-λ)f(y) always:", ok)   # True -> convex
` }
  ],

  keyPoints: [
    "Convex = chord never below the graph: $f(\\lambda x+(1-\\lambda)y)\\le\\lambda f(x)+(1-\\lambda)f(y)$.",
    "Twice-differentiable: convex ⇔ Hessian PSD everywhere; strictly convex ⇔ PD.",
    "For convex $f$, every local minimum is the global minimum.",
    "Convex losses: linear/ridge regression, logistic regression, SVM hinge.",
    "Deep networks are non-convex — multiple minima/saddles — yet train well in practice."
  ],

  commonMistakes: [
    { wrong: "Assuming a single minimum means convex.", why: "A function can have one minimum without being convex (e.g. flat-then-steep shapes). Convexity is the stronger chord/Hessian condition, and it's what guarantees no local traps." },
    { wrong: "Thinking non-convex means 'untrainable'.", why: "Deep nets are non-convex yet optimize reliably because over-parameterization, good initialization, and SGD navigate the landscape well — the minima found are typically near-global in loss." },
    { wrong: "Checking convexity at one point only.", why: "The Hessian must be PSD <em>everywhere</em>, not just at the minimum. A function can be locally bowl-shaped but globally non-convex." }
  ],

  quiz: [
    { q: "Which function is convex?", options: ["$x^2$", "$\\sin x$", "$x^3$", "$-x^2$"], answer: 0,
      explain: "$x^2$ has $f''=2>0$ everywhere. $\\sin$ and $x^3$ change curvature sign; $-x^2$ is concave." },
    { q: "For a convex function, a local minimum is…", options: ["always the global minimum", "never global", "a saddle", "a maximum"], answer: 0,
      explain: "Convexity guarantees local = global — the defining benefit." },
    { q: "Twice-differentiable $f$ is convex iff its Hessian is…", options: ["PSD everywhere", "PD at the minimum", "indefinite", "zero"], answer: 0,
      explain: "Positive semidefinite Hessian at every point is equivalent to convexity." },
    { q: "Which loss is NOT convex in the model parameters?", options: ["a deep neural network's loss", "logistic regression loss", "linear regression MSE", "SVM hinge loss"], answer: 0,
      explain: "Composition of nonlinear layers makes a deep net's loss non-convex; the other three are classic convex losses." },
    { q: "If $f$ and $g$ are convex, then $f+g$ is…", options: ["convex", "concave", "non-convex", "linear"], answer: 0,
      explain: "Sums of convex functions are convex (add the chord inequalities) — which is why a convex loss plus L2 (also convex) stays convex." }
  ],

  practice: [
    { level: "easy", prompt: "Is $f(x)=e^x$ convex? Justify.", solution: "$f''(x)=e^x>0$ for all $x$, so yes — strictly convex." },
    { level: "easy", prompt: "Is $f(x)=x^3$ convex on all of $\\mathbb{R}$?", solution: "$f''(x)=6x$, which is negative for $x<0$. So no — it's convex only on $x\\ge0$." },
    { level: "med", prompt: "Show $f(x)=|x|$ is convex.", solution: "By the triangle inequality, $|\\lambda x+(1-\\lambda)y|\\le\\lambda|x|+(1-\\lambda)|y|$ for $\\lambda\\in[0,1]$ — exactly the convexity inequality. (It's convex despite the non-differentiable corner at 0.)" },
    { level: "hard", prompt: "AI task: logistic regression minimizes $\\sum_i\\log(1+e^{-y_i\\mathbf w^\\top\\mathbf x_i})$. Why does convexity in $\\mathbf w$ matter, and what changes for a deep net?", solution: "Each term is convex in $\\mathbf w$ (the log-sum-exp / softplus is convex, composed with a linear function of $\\mathbf w$, which preserves convexity), and sums of convex functions are convex. So the logistic loss is convex in $\\mathbf w$: gradient descent reaches the unique global optimum regardless of initialization, and the solution is reproducible. A deep net applies nonlinear layers <em>before</em> the loss, so the loss is non-convex in the weights — many minima/saddles, sensitivity to initialization and learning rate — which is why deep learning needs careful optimization (this track) even though the final-layer objective alone would be convex." }
  ],

  deepDive: String.raw`<p><strong>Why non-convex deep learning works anyway.</strong></p>
  <p>Classical optimization theory loves convexity because it gives ironclad guarantees: one global minimum, gradient
  descent converges to it, done. Deep networks throw that away — their losses are wildly non-convex — yet they train to
  excellent solutions with simple first-order methods. Several ideas reconcile this. <strong>Over-parameterization</strong>:
  with far more parameters than data, the loss landscape has many global minima and broad connected basins, so descent
  almost always finds one. <strong>Benign geometry</strong>: as Lesson 8.1 argued, the troublesome critical points are
  mostly saddles (escapable), not bad minima. <strong>Implicit regularization</strong>: SGD prefers flat, wide minima
  that generalize, even though the loss alone doesn't ask for them.</p>
  <p>So convexity remains the right mental model for <em>when</em> optimization is trivially safe — and a checklist item
  worth verifying for the convex sub-problems you do encounter (linear/logistic regression, the dual of an SVM, many
  calibration and projection steps). For the non-convex rest, the lesson is humbler: we don't get guarantees, we get
  empirical reliability from architecture, scale, and the noise of SGD. Recognizing which regime you're in tells you
  whether to trust a clean theorem or to reach for the practical toolbox of momentum, adaptivity, and tuning that the
  next lesson begins.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["8.3"] = {
  subtitle: "Roll downhill in steps proportional to the slope — and the learning rate that decides if it works.",

  aiMoment: String.raw`<p>Every model you've trained was optimized by some descendant of <strong>gradient descent</strong>:
  step against the gradient, scaled by the learning rate. <strong>SGD</strong> just estimates that gradient on a
  minibatch (Lesson 7.5). The single most consequential hyperparameter — the learning rate — is exactly the "how far do I
  trust the local slope" knob, and its safe range is dictated by the curvature. This lesson derives the update and its
  convergence from first principles.</p>`,

  plainEnglish: String.raw`<p><strong>Gradient descent</strong> rolls downhill: at each step, move in the direction the
  function decreases fastest (the negative gradient), by an amount set by the <strong>learning rate</strong> $\eta$. Too
  small and you crawl; too big and you overshoot and diverge; just right and you converge.</p>`,

  intuition: String.raw`<p>Picture a ball on the loss surface. The gradient says which way is steepest uphill, so you
  step the opposite way. The step size matters enormously: tiny steps inch along, huge steps bounce across the valley and
  can fly out, and a well-chosen step settles into the bottom.</p>
  <figure class="figure">
  <svg viewBox="0 0 384 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gradient descent trajectories for small, good, and large learning rates">
    <ellipse cx="160" cy="80" rx="130" ry="55" fill="none" stroke="#e2e8f0"/>
    <ellipse cx="160" cy="80" rx="90" ry="38" fill="none" stroke="#e2e8f0"/>
    <ellipse cx="160" cy="80" rx="48" ry="20" fill="none" stroke="#e2e8f0"/>
    <circle cx="160" cy="80" r="3" fill="#94a3b8"/>
    <polyline points="40,30 70,40 95,48 118,55 138,62 152,70" fill="none" stroke="#0d9488" stroke-width="2" marker-end="url(#g3)"/>
    <text x="36" y="24" font-size="10" fill="#0d9488" font-family="sans-serif">good η</text>
    <polyline points="285,135 250,128 220,124 195,120 175,116" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="3 2"/>
    <text x="250" y="148" font-size="10" fill="#2563eb" font-family="sans-serif">η too small (slow)</text>
    <polyline points="40,130 250,40 70,55 230,120" fill="none" stroke="#dc2626" stroke-width="1.8"/>
    <text x="42" y="142" font-size="10" fill="#dc2626" font-family="sans-serif">η too big (overshoots)</text>
    <defs><marker id="g3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#0d9488"/></marker></defs>
  </svg>
  <figcaption>Same bowl, three learning rates: crawl, converge, or bounce out.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The update, from minimizing the first-order Taylor model (Lesson 6.5):</p>
  $$\mathbf w_{t+1}=\mathbf w_t-\eta\,\nabla f(\mathbf w_t).$$
  <p>On a quadratic with Hessian eigenvalues $\lambda_i$, each eigen-coordinate evolves as $z_i\leftarrow(1-\eta\lambda_i)z_i$,
  so descent converges iff $0<\eta<2/\lambda_{\max}$. The slowest direction sets the rate, which depends on the
  <strong>condition number</strong> $\kappa=\lambda_{\max}/\lambda_{\min}$. <strong>SGD</strong> replaces $\nabla f$ with a
  minibatch estimate $\nabla f_{\text{batch}}$.</p>`,

  derivation: String.raw`<p><strong>Convergence rate on a quadratic, and the optimal learning rate.</strong> Work in the
  Hessian's eigenbasis (Track 5.2), so $f=\tfrac12\sum_i\lambda_i z_i^2$.</p>
  <p><strong>Step 1 — per-axis recursion.</strong> The gradient in $z_i$ is $\lambda_i z_i$, so
  $z_i^{(t+1)}=z_i^{(t)}-\eta\lambda_i z_i^{(t)}=(1-\eta\lambda_i)z_i^{(t)}$. After $t$ steps,
  $z_i^{(t)}=(1-\eta\lambda_i)^t z_i^{(0)}$.</p>
  <p><strong>Step 2 — stability.</strong> This decays iff $|1-\eta\lambda_i|<1$, i.e. $0<\eta<2/\lambda_i$ for every axis;
  the binding one is $\lambda_{\max}$, giving $\eta<2/\lambda_{\max}$.</p>
  <p><strong>Step 3 — best worst-case rate.</strong> Convergence speed is $\rho(\eta)=\max_i|1-\eta\lambda_i|$. Minimizing
  this over $\eta$ balances the extremes $\lambda_{\min}$ and $\lambda_{\max}$ at
  $\eta^*=\dfrac{2}{\lambda_{\min}+\lambda_{\max}}$, giving</p>
  $$\rho^*=\frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}=\frac{\kappa-1}{\kappa+1}.$$
  <p><strong>Step 4 — read the consequence.</strong> A large condition number $\kappa$ pushes $\rho^*$ toward 1 — painfully
  slow convergence. $\blacksquare$ Plain English: ill-conditioned bowls (stretched valleys) make plain gradient descent
  crawl, which is exactly the problem momentum and adaptivity fix next.</p>`,

  code: [
    { label: "Learning rate: crawl, converge, diverge", src: String.raw`
import numpy as np

lam = 4.0                                # f = 0.5*lam*x², f'(x)=lam*x
def run(eta, steps=12, x0=1.0):
    x = x0; traj=[x]
    for _ in range(steps): x = x - eta*(lam*x); traj.append(x)
    return np.round(traj, 3)

print("eta=0.05 (small):", run(0.05))    # slow crawl toward 0
print("eta=0.25 (good) :", run(0.25))    # fast to 0
print("eta=0.55 (>2/λ) :", run(0.55))    # |1-eta*lam|>1 -> diverges
print("stability bound 2/λ =", 2/lam)
` },
    { label: "Condition number sets the speed", src: String.raw`
import numpy as np

A = np.diag([1.0, 10.0])                  # eigenvalues 1 and 10 -> kappa=10
b = np.zeros(2); x = np.array([1.0, 1.0])
eta = 2/(1+10)                            # optimal eta = 2/(λmin+λmax)
for t in range(1, 41):
    x = x - eta*(A@x - b)
print("after 40 steps:", np.round(x, 5), " (slow axis lingers)")
print("predicted rate (kappa-1)/(kappa+1) =", round((10-1)/(10+1), 4))
` }
  ],

  keyPoints: [
    "Gradient descent: $\\mathbf w\\leftarrow\\mathbf w-\\eta\\nabla f$, from the first-order Taylor model.",
    "Stability on a quadratic needs $0<\\eta<2/\\lambda_{\\max}$; above it, training diverges.",
    "Best convergence rate is $(\\kappa-1)/(\\kappa+1)$ — ill-conditioning ($\\kappa$ large) means slow.",
    "SGD uses a minibatch gradient: an unbiased, noisy estimate of the full gradient.",
    "The learning rate is the trust radius of a linear approximation to a curved loss."
  ],

  commonMistakes: [
    { wrong: "Cranking the learning rate up for speed.", why: "Past $2/\\lambda_{\\max}$ the steepest direction's iterate grows and the loss explodes to NaN. Speed comes from better directions (momentum/adaptivity), not a bigger $\\eta$." },
    { wrong: "Using one learning rate for all coordinates on an ill-conditioned loss.", why: "A single $\\eta$ must be small enough for the steepest axis, so the flat axes crawl. Per-coordinate scaling (Adam) addresses this." },
    { wrong: "Forgetting the noise in SGD when choosing $\\eta$.", why: "Minibatch gradients have variance $\\propto1/\\text{batch}$; near a minimum, too-large $\\eta$ keeps the iterate bouncing around it. Decaying the learning rate lets it settle." }
  ],

  quiz: [
    { q: "On $f=\\tfrac12\\lambda x^2$ with $\\lambda=4$, gradient descent diverges when $\\eta$ exceeds…", options: ["0.5", "0.25", "1", "4"], answer: 0,
      explain: "Stability needs $\\eta<2/\\lambda=2/4=0.5$." },
    { q: "The optimal learning rate for a quadratic is…", options: ["$2/(\\lambda_{\\min}+\\lambda_{\\max})$", "$1/\\lambda_{\\min}$", "$2/\\lambda_{\\min}$", "$\\lambda_{\\max}$"], answer: 0,
      explain: "It balances the slowest and fastest axes, giving rate $(\\kappa-1)/(\\kappa+1)$." },
    { q: "A condition number $\\kappa=100$ gives best GD rate ≈…", options: ["0.98", "0.5", "0.01", "100"], answer: 0,
      explain: "$(\\kappa-1)/(\\kappa+1)=99/101\\approx0.98$ — very slow (each step shrinks error by only ~2%)." },
    { q: "SGD's minibatch gradient is…", options: ["an unbiased estimate of the full gradient", "always equal to it", "biased upward", "the Hessian"], answer: 0,
      explain: "Averaging the gradient over a random subset is unbiased (Lesson 7.5)." },
    { q: "Doubling a too-small constant learning rate (still stable) roughly…", options: ["halves the steps to converge", "diverges", "has no effect", "changes the minimum"], answer: 0,
      explain: "In the stable regime, larger $\\eta$ shrinks $|1-\\eta\\lambda|$ toward its optimum, speeding convergence — up to the stability limit." }
  ],

  practice: [
    { level: "easy", prompt: "One GD step on $f(x)=x^2$ from $x=3$ with $\\eta=0.1$.", solution: "$f'(3)=6$; $x\\leftarrow3-0.1\\cdot6=2.4$." },
    { level: "med", prompt: "For $f=\\tfrac12\\lambda x^2$ with $\\lambda=10$, give the stability bound on $\\eta$ and the learning rate that converges in one step.", solution: "Stable for $\\eta<2/10=0.2$. One-step convergence needs $1-\\eta\\lambda=0\\Rightarrow\\eta=1/\\lambda=0.1$, which lands exactly at the minimum." },
    { level: "med", prompt: "Explain why a stretched valley ($\\lambda_{\\max}\\gg\\lambda_{\\min}$) makes GD slow.", solution: "A single $\\eta$ must satisfy $\\eta<2/\\lambda_{\\max}$ for stability, but then the flat direction shrinks at rate $1-\\eta\\lambda_{\\min}\\approx1$ — almost no progress. The mismatch between the steep and flat axes (large $\\kappa$) forces slow overall convergence." },
    { level: "hard", prompt: "AI task: warmup-then-decay learning-rate schedules are standard for transformers. Explain each phase in terms of this lesson.", solution: "<em>Warmup</em>: early training has large, noisy gradients and a poorly-conditioned loss; starting with a small $\\eta$ and ramping up avoids the early divergence that a big step would cause before the curvature settles. <em>Decay</em>: as you near a minimum, the minibatch noise (variance $\\propto\\eta^2$ near the optimum) keeps the iterate bouncing; shrinking $\\eta$ reduces that noise floor so the iterate settles, mirroring the $0<\\eta<2/\\lambda_{\\max}$ stability and the need to lower the trust radius as the linear model's region shrinks. Together they respect both the stability bound and the SGD noise that this lesson derives." }
  ],

  deepDive: String.raw`<p><strong>From the convergence rate to every trick that beats it.</strong></p>
  <p>The result $\rho^*=(\kappa-1)/(\kappa+1)$ is the keystone of optimization practice: plain gradient descent is fast on
  well-conditioned problems ($\kappa\approx1$) and miserable on ill-conditioned ones ($\kappa\gg1$). Almost every
  improvement in the field attacks $\kappa$ or sidesteps it. <strong>Momentum</strong> (next lesson) improves the rate to
  roughly $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ — a square-root speedup. <strong>Adam/RMSProp</strong> rescale each coordinate
  to make the effective bowl round ($\kappa\to1$). <strong>Batch/layer normalization</strong> reconditions the loss so
  $\kappa$ shrinks. <strong>Second-order methods</strong> multiply by $H^{-1}$ to remove $\kappa$ entirely (one step on a
  quadratic). Even good <strong>initialization</strong> aims to start with a well-conditioned Jacobian.</p>
  <p>So the learning rate isn't just a number to grid-search; it sits inside a precise theory. The stability ceiling
  $2/\lambda_{\max}$ tells you when you'll diverge; the rate $(\kappa-1)/(\kappa+1)$ tells you how slow you'll be even
  when stable; and the condition number $\kappa$ — a property of the loss surface's curvature, i.e. its Hessian
  eigenvalues (Track 5) — is the quantity everything is really fighting. Read an optimizer's design as "what is it doing
  to $\kappa$?" and the whole zoo in the next lesson snaps into focus.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["8.4"] = {
  subtitle: "A heavy ball that smooths the path, and per-coordinate scaling that rounds the bowl — that's Adam.",

  aiMoment: String.raw`<p><strong>Adam</strong> trains the overwhelming majority of modern networks. It combines two
  ideas: <strong>momentum</strong> (a running average of gradients that powers through noise and saddles) and
  <strong>RMSProp</strong>-style per-coordinate scaling (dividing by a running estimate of each direction's gradient
  magnitude, which makes a stretched loss bowl effectively round). Understanding these two moves — and Adam's bias
  correction — demystifies the default optimizer.</p>`,

  plainEnglish: String.raw`<p><strong>Momentum</strong> treats the parameter like a heavy ball: it builds up speed in
  directions of consistent slope and rolls through small bumps and zig-zags. <strong>Adaptive scaling</strong> gives each
  coordinate its own effective step size, big where gradients are small and small where they're large. <strong>Adam</strong>
  is both together, with a small early-step correction.</p>`,

  intuition: String.raw`<p>In a stretched valley, plain gradient descent bounces wall-to-wall while inching forward.
  Momentum averages out the sideways bounces and accelerates along the valley floor. Adaptive scaling shrinks the steep
  axis and stretches the flat one, so the path goes more directly to the bottom.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 140" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gradient descent zig-zag versus momentum's smoother path">
    <ellipse cx="160" cy="70" rx="135" ry="40" fill="none" stroke="#e2e8f0"/>
    <ellipse cx="160" cy="70" rx="90" ry="26" fill="none" stroke="#e2e8f0"/>
    <ellipse cx="160" cy="70" rx="45" ry="13" fill="none" stroke="#e2e8f0"/>
    <circle cx="160" cy="70" r="3" fill="#94a3b8"/>
    <polyline points="35,40 60,100 85,46 108,94 128,52 145,84 156,62 160,70" fill="none" stroke="#dc2626" stroke-width="1.6"/>
    <text x="30" y="32" font-size="10" fill="#dc2626" font-family="sans-serif">GD (zig-zag)</text>
    <polyline points="35,108 80,92 120,82 155,74 160,70" fill="none" stroke="#0d9488" stroke-width="2.2" marker-end="url(#m3)"/>
    <text x="36" y="124" font-size="10" fill="#0d9488" font-family="sans-serif">momentum (smooth)</text>
    <defs><marker id="m3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#0d9488"/></marker></defs>
  </svg>
  <figcaption>Momentum cancels the perpendicular bouncing and accelerates along the valley.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Momentum:</strong> $\mathbf m_t=\beta\mathbf m_{t-1}+\nabla f_t,\quad \mathbf w_t=\mathbf w_{t-1}-\eta\mathbf m_t.$
  <strong>RMSProp:</strong> $\mathbf v_t=\rho\mathbf v_{t-1}+(1-\rho)\nabla f_t^2,\quad \mathbf w_t=\mathbf w_{t-1}-\eta\,\nabla f_t/(\sqrt{\mathbf v_t}+\varepsilon).$
  <strong>Adam</strong> combines them with bias correction:</p>
  $$\mathbf m_t=\beta_1\mathbf m_{t-1}+(1-\beta_1)\mathbf g_t,\quad \mathbf v_t=\beta_2\mathbf v_{t-1}+(1-\beta_2)\mathbf g_t^2,$$
  $$\hat{\mathbf m}_t=\frac{\mathbf m_t}{1-\beta_1^{t}},\quad \hat{\mathbf v}_t=\frac{\mathbf v_t}{1-\beta_2^{t}},\quad
  \mathbf w_t=\mathbf w_{t-1}-\eta\,\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\varepsilon}.$$
  <p>Defaults: $\beta_1=0.9,\ \beta_2=0.999,\ \varepsilon=10^{-8}$.</p>`,

  derivation: String.raw`<p><strong>Part 1 — momentum is an exponentially-weighted average of gradients.</strong> Unroll
  $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$:
  $\mathbf m_t=\mathbf g_t+\beta\mathbf g_{t-1}+\beta^2\mathbf g_{t-2}+\cdots=\sum_{k\ge0}\beta^k\mathbf g_{t-k}.$
  Recent gradients get weight near 1, older ones decay as $\beta^k$. In a consistent direction the terms reinforce
  (effective step grows toward $\tfrac{1}{1-\beta}\times$); in an oscillating direction they cancel. That's why momentum
  accelerates along the valley and damps the zig-zag. $\blacksquare$</p>
  <hr class="soft">
  <p><strong>Part 2 — why Adam needs bias correction.</strong> Initialize $\mathbf m_0=\mathbf 0$. With $\mathbf g_t$
  roughly stationary, take expectations of $\mathbf m_t=(1-\beta_1)\sum_{k=0}^{t-1}\beta_1^{k}\mathbf g_{t-k}$:</p>
  <p><strong>Step 1.</strong> $E[\mathbf m_t]\approx E[\mathbf g]\,(1-\beta_1)\sum_{k=0}^{t-1}\beta_1^{k}=E[\mathbf g]\,(1-\beta_1^{t}).$</p>
  <p><strong>Step 2.</strong> So $\mathbf m_t$ <em>underestimates</em> $E[\mathbf g]$ by the factor $(1-\beta_1^{t})$ —
  badly at small $t$ (e.g. $t=1$ gives only $1-\beta_1=0.1$ of the gradient).</p>
  <p><strong>Step 3.</strong> Dividing by $(1-\beta_1^{t})$ removes the bias: $\hat{\mathbf m}_t=\mathbf m_t/(1-\beta_1^{t})$
  has $E[\hat{\mathbf m}_t]\approx E[\mathbf g]$. Same for $\hat{\mathbf v}_t$. $\blacksquare$ Plain English: the moving
  averages start at zero and need to be scaled up early so the first steps aren't tiny.</p>`,

  code: [
    { label: "GD vs Momentum vs Adam on an ill-conditioned bowl", src: String.raw`
import numpy as np

A = np.diag([1.0, 100.0])                # kappa = 100: a stretched valley
grad = lambda x: A @ x
def steps_to_converge(update, x0=np.array([1.,1.]), tol=1e-3, cap=2000):
    x=x0.copy(); state=update.init(x)
    for t in range(1,cap+1):
        x,state = update.step(x,grad(x),state,t)
        if np.linalg.norm(x)<tol: return t
    return cap

class GD:
    def __init__(s,eta): s.eta=eta
    def init(s,x): return None
    def step(s,x,g,st,t): return x-s.eta*g, None
class Mom:
    def __init__(s,eta,b=0.9): s.eta,s.b=eta,b
    def init(s,x): return np.zeros_like(x)
    def step(s,x,g,m,t): m=s.b*m+g; return x-s.eta*m, m
class Adam:
    def __init__(s,eta,b1=0.9,b2=0.999,e=1e-8): s.eta,s.b1,s.b2,s.e=eta,b1,b2,e
    def init(s,x): return (np.zeros_like(x),np.zeros_like(x))
    def step(s,x,g,st,t):
        m,v=st; m=s.b1*m+(1-s.b1)*g; v=s.b2*v+(1-s.b2)*g*g
        mh=m/(1-s.b1**t); vh=v/(1-s.b2**t)
        return x-s.eta*mh/(np.sqrt(vh)+s.e), (m,v)

print("GD       steps:", steps_to_converge(GD(0.018)))   # crawls along the flat axis
print("Momentum steps:", steps_to_converge(Mom(0.018)))  # ~√κ speedup over GD
print("Adam     steps:", steps_to_converge(Adam(0.5)))   # per-coordinate scaling wins
` }
  ],

  keyPoints: [
    "Momentum $\\mathbf m=\\beta\\mathbf m+\\mathbf g$ is an EWMA of gradients — accelerates consistent directions, damps oscillation.",
    "RMSProp/Adam divide by $\\sqrt{\\mathbf v}$ (running gradient magnitude) for per-coordinate step sizes.",
    "Adam = momentum + adaptive scaling + bias correction; defaults $\\beta_1{=}0.9,\\beta_2{=}0.999$.",
    "Bias correction $\\hat{\\mathbf m}=\\mathbf m/(1-\\beta_1^t)$ fixes the zero-initialized averages' early underestimate.",
    "Momentum improves the convergence rate from $\\sim\\kappa$ to $\\sim\\sqrt\\kappa$ dependence."
  ],

  commonMistakes: [
    { wrong: "Dropping Adam's bias correction.", why: "Without it, the first steps use averages that are far too small ($\\sim(1-\\beta)$ of the true gradient), so early training stalls or behaves erratically — especially with $\\beta_2=0.999$." },
    { wrong: "Assuming Adam always beats SGD.", why: "Adam converges fast but its adaptive scaling can hurt generalization; well-tuned SGD+momentum often wins final accuracy on vision tasks. The right choice is task-dependent." },
    { wrong: "Setting momentum $\\beta$ too high.", why: "Large $\\beta$ (e.g. 0.99) gives lots of inertia — great for smooth valleys but it can overshoot and oscillate around the minimum if the learning rate isn't lowered to match." }
  ],

  quiz: [
    { q: "Momentum's update $\\mathbf m=\\beta\\mathbf m+\\mathbf g$ is mathematically a…", options: ["exponentially-weighted average of gradients", "second derivative", "projection", "normalization"], answer: 0,
      explain: "Unrolling gives $\\sum_k\\beta^k\\mathbf g_{t-k}$ — recent gradients weighted most." },
    { q: "Adam's per-coordinate scaling divides the step by…", options: ["$\\sqrt{\\hat v_t}+\\varepsilon$ (running gradient magnitude)", "the learning rate", "the Hessian", "the momentum"], answer: 0,
      explain: "$\\hat v_t$ tracks each coordinate's squared-gradient average, giving smaller steps where gradients are large." },
    { q: "Bias correction in Adam exists because…", options: ["the moving averages start at 0 and underestimate early", "gradients are biased", "the learning rate decays", "$\\beta$ is negative"], answer: 0,
      explain: "Zero-initialized $\\mathbf m,\\mathbf v$ are too small at small $t$; dividing by $1-\\beta^t$ removes the bias." },
    { q: "Momentum improves the GD convergence-rate dependence on $\\kappa$ from roughly $\\kappa$ to…", options: ["$\\sqrt\\kappa$", "$\\kappa^2$", "$\\log\\kappa$", "$1$"], answer: 0,
      explain: "Heavy-ball/accelerated methods achieve a $\\sqrt\\kappa$ dependence — a major speedup for ill-conditioned problems." },
    { q: "At $t=1$ with $\\beta_1=0.9$, the uncorrected $\\mathbf m_1$ equals…", options: ["$0.1\\,\\mathbf g_1$ (so correction multiplies by 10)", "$\\mathbf g_1$", "$0.9\\,\\mathbf g_1$", "$\\mathbf 0$"], answer: 0,
      explain: "$\\mathbf m_1=(1-\\beta_1)\\mathbf g_1=0.1\\mathbf g_1$; dividing by $1-\\beta_1^1=0.1$ recovers $\\mathbf g_1$." }
  ],

  practice: [
    { level: "easy", prompt: "With $\\beta=0.9$ and a constant gradient $g$, what does momentum's $\\mathbf m$ converge to?", solution: "The geometric series $\\sum_{k\\ge0}\\beta^k g=\\frac{g}{1-\\beta}=\\frac{g}{0.1}=10g$ — momentum's effective step is $10\\times$ the raw gradient in a consistent direction." },
    { level: "med", prompt: "Why does dividing by $\\sqrt{v}$ help on an ill-conditioned loss?", solution: "Steep coordinates have large gradients, so large $v$, so their step is divided down; flat coordinates have small $v$, so their step is boosted. This equalizes progress across axes — effectively reducing the condition number the optimizer 'sees' toward 1." },
    { level: "med", prompt: "Compute Adam's corrected $\\hat m_2$ if $g_1=g_2=1$, $\\beta_1=0.9$.", solution: "$m_1=0.1$, $m_2=0.9(0.1)+0.1(1)=0.19$. Correction: $\\hat m_2=m_2/(1-0.9^2)=0.19/0.19=1.0$ — exactly the (constant) gradient, as the unbiasing intends." },
    { level: "hard", prompt: "AI task: explain why transformers are almost always trained with Adam (or AdamW) rather than plain SGD.", solution: "Transformer losses are badly conditioned and the gradient scales vary enormously across parameter groups (embeddings vs attention vs LayerNorm), so a single global learning rate (SGD) is a poor fit — some directions need big steps, others tiny. Adam's per-coordinate $1/\\sqrt{v}$ scaling gives each parameter an appropriate effective step automatically, and momentum smooths the noisy, sparse gradients (e.g. rare tokens). AdamW additionally decouples weight decay from the adaptive scaling, fixing a subtle interaction. The result is robust, fast convergence with far less per-layer learning-rate tuning — which matters at the scale and heterogeneity of transformer parameters. (Well-tuned SGD+momentum can still win on some CNNs, but the conditioning of transformers strongly favors Adam.)" }
  ],

  deepDive: String.raw`<p><strong>Every optimizer is approximating the inverse Hessian.</strong></p>
  <p>The ideal step is Newton's, $-H^{-1}\nabla f$, which removes the condition number entirely (one step on a quadratic,
  Lesson 6.5). But $H$ is enormous and indefinite for deep nets, so we approximate the preconditioner $H^{-1}$ cheaply.
  Read through that lens, the optimizer zoo is a hierarchy of approximations: <strong>SGD</strong> uses $H^{-1}\approx\eta I$
  (a scalar). <strong>Momentum</strong> adds memory that mimics the effect of curvature along consistent directions.
  <strong>RMSProp/Adam</strong> use a <em>diagonal</em> approximation $H^{-1}\approx\mathrm{diag}(1/\sqrt{v})$ — per-coordinate
  curvature. <strong>K-FAC</strong> uses a block/Kronecker approximation; <strong>L-BFGS</strong> builds a low-rank inverse
  from past gradients. They differ only in how rich (and how expensive) their model of $H^{-1}$ is.</p>
  <p>This explains the practical trade-offs. Adam's diagonal preconditioner is cheap and robust but ignores
  off-diagonal curvature (interactions between parameters), which is part of why its solutions can generalize slightly
  worse than SGD's flatter minima. The bias correction, the $\varepsilon$ for numerical safety (Track 13), the decoupled
  weight decay of AdamW — each is an engineering patch on the same core idea. So when you pick or tune an optimizer,
  you're really choosing how much curvature information to estimate and trust. The calculus (gradients, Hessians,
  Taylor) and the linear algebra (eigenvalues, conditioning) of the last two tracks are exactly the language this choice
  is made in — and the constrained version of these problems, with explicit constraints, is the final lesson.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["8.5"] = {
  subtitle: "Optimize under constraints: at the best feasible point, the objective pushes straight into the wall.",

  aiMoment: String.raw`<p>Support vector machines maximize a margin <em>subject to</em> classification constraints — a
  constrained optimization solved with <strong>Lagrange multipliers</strong>, whose dual form is what the kernel trick
  acts on. Modern RLHF (TRPO/PPO) maximizes reward <em>subject to</em> a KL constraint that keeps the policy near the
  base model. Whenever "optimize X but keep Y bounded" appears, Lagrange multipliers and the <strong>KKT conditions</strong>
  are the tools.</p>`,

  plainEnglish: String.raw`<p>To optimize a function while obeying a constraint, you look for the best point you can
  reach <em>without breaking the rule</em>. At that point you can't improve the objective without violating the
  constraint — which means the objective's gradient points straight "into" the constraint. The
  <strong>Lagrange multiplier</strong> measures how hard it's pushing.</p>`,

  intuition: String.raw`<p>Picture the objective's contour lines and a constraint curve. As you slide along the
  constraint, you cross contours (improving) until you reach a point where the constraint is <strong>tangent</strong> to a
  contour. There, moving either way along the constraint no longer improves the objective — the gradients of objective and
  constraint are parallel.</p>
  <figure class="figure">
  <svg viewBox="0 0 240 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Objective contours tangent to a constraint line">
    <circle cx="80" cy="110" r="58" fill="none" stroke="#cbd5e1"/>
    <circle cx="80" cy="110" r="38" fill="none" stroke="#cbd5e1"/>
    <circle cx="80" cy="110" r="18" fill="none" stroke="#cbd5e1"/>
    <circle cx="80" cy="110" r="3" fill="#94a3b8"/>
    <line x1="30" y1="170" x2="210" y2="30" stroke="#0d9488" stroke-width="2.2"/>
    <text x="170" y="40" font-size="11" fill="#0d9488" font-family="sans-serif">g(x)=0</text>
    <circle cx="116" cy="74" r="3.5" fill="#dc2626"/>
    <line x1="116" y1="74" x2="142" y2="48" stroke="#4f46e5" stroke-width="2" marker-end="url(#L1)"/>
    <line x1="116" y1="74" x2="138" y2="52" stroke="#dc2626" stroke-width="2" marker-end="url(#L2)" opacity="0.6"/>
    <text x="146" y="46" font-size="10" fill="#4f46e5" font-family="sans-serif">∇f ∥ ∇g</text>
    <defs>
      <marker id="L1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#4f46e5"/></marker>
      <marker id="L2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#dc2626"/></marker>
    </defs>
  </svg>
  <figcaption>The optimum is where a contour is tangent to the constraint — there ∇f and ∇g are parallel.</figcaption>
  </figure>`,

  formalism: String.raw`<p>To minimize $f(\mathbf x)$ subject to $g(\mathbf x)=0$, form the <strong>Lagrangian</strong>
  $\mathcal L(\mathbf x,\lambda)=f(\mathbf x)-\lambda\,g(\mathbf x)$ and set its gradients to zero:</p>
  $$\nabla_{\mathbf x}\mathcal L=\nabla f-\lambda\nabla g=\mathbf 0\ \ (\text{so }\nabla f=\lambda\nabla g),\qquad \nabla_\lambda\mathcal L=-g=0.$$
  <p>For an inequality constraint $g(\mathbf x)\le0$, the optimum satisfies the <strong>KKT conditions</strong>:
  stationarity ($\nabla f+\mu\nabla g=\mathbf0$), primal feasibility ($g\le0$), dual feasibility ($\mu\ge0$), and
  complementary slackness ($\mu\,g=0$ — the constraint is either active or its multiplier is zero).</p>`,

  derivation: String.raw`<p><strong>Why $\nabla f=\lambda\nabla g$ at a constrained optimum.</strong></p>
  <p><strong>Step 1 — allowed directions.</strong> Staying on the constraint $g(\mathbf x)=0$ means moving in directions
  $\mathbf d$ tangent to it; to first order $g$ doesn't change, so $\nabla g^\top\mathbf d=0$. The feasible moves are
  exactly those perpendicular to $\nabla g$.</p>
  <p><strong>Step 2 — optimality.</strong> At a constrained minimum, no feasible move decreases $f$: $\nabla f^\top\mathbf d\ge0$
  for all such $\mathbf d$ (and $\le0$ by taking $-\mathbf d$), so $\nabla f^\top\mathbf d=0$ for every $\mathbf d\perp\nabla g$.</p>
  <p><strong>Step 3 — parallel gradients.</strong> A vector ($\nabla f$) orthogonal to every vector that is orthogonal to
  $\nabla g$ must itself be a multiple of $\nabla g$: $\nabla f=\lambda\nabla g$. $\blacksquare$ Plain English: at the best
  feasible point, the objective only wants to move in the one forbidden direction (straight across the constraint), so its
  gradient lines up with the constraint's. The multiplier $\lambda$ is how strongly it pushes — the "shadow price" of the
  constraint.</p>`,

  code: [
    { label: "Constrained min by Lagrange: min x²+y² s.t. x+y=1", src: String.raw`
import numpy as np

# Stationarity: ∇f = λ∇g  ->  2x=λ, 2y=λ  ->  x=y; constraint x+y=1 -> x=y=0.5
x = y = 0.5
lam = 2*x
print("optimal (x,y):", (x, y), " minimum value:", x**2+y**2)   # 0.5
print("λ (shadow price):", lam)

# verify ∇f is parallel to ∇g at the solution
grad_f = np.array([2*x, 2*y]); grad_g = np.array([1., 1.])
print("∇f =", grad_f, "  λ∇g =", lam*grad_g, "  parallel?",
      np.allclose(grad_f, lam*grad_g))
` },
    { label: "Penalty view: the multiplier as a price", src: String.raw`
import numpy as np
# Solve the SAME problem by minimizing f + (λ/2)·(constraint)² for growing penalty
from numpy import array
def solve(pen):
    # minimize x²+y² + pen*(x+y-1)²  -> closed form via gradient = 0
    # 2x + 2*pen*(x+y-1) = 0 (and symmetric) -> x=y, 2x+2*pen*(2x-1)=0
    x = pen/(1+2*pen); return x
for pen in [1, 10, 100, 1000]:
    x = solve(pen); print(f"penalty={pen:5d}: x≈{x:.4f}  (→ 0.5 as penalty→∞)")
` }
  ],

  keyPoints: [
    "Constrained optimum: $\\nabla f=\\lambda\\nabla g$ — objective and constraint gradients are parallel.",
    "Lagrangian $\\mathcal L=f-\\lambda g$; stationarity in $\\mathbf x$ and $\\lambda$ gives the conditions.",
    "The multiplier $\\lambda$ is the 'shadow price' — how much the optimum improves if the constraint loosens.",
    "Inequalities use KKT: stationarity, primal/dual feasibility, complementary slackness $\\mu g=0$.",
    "Complementary slackness: a constraint is either active ($g=0$) or has multiplier $\\mu=0$."
  ],

  commonMistakes: [
    { wrong: "Forgetting to also impose the constraint $g=0$.", why: "$\\nabla f=\\lambda\\nabla g$ alone is underdetermined; you need the original constraint equation too. It comes from $\\partial\\mathcal L/\\partial\\lambda=0$." },
    { wrong: "Allowing a negative multiplier on an inequality constraint.", why: "KKT requires $\\mu\\ge0$ for $g\\le0$: the constraint can only push 'inward'. A negative $\\mu$ would mean the constraint helps the objective, contradicting that it's binding." },
    { wrong: "Treating an inactive constraint as binding.", why: "If the unconstrained optimum already satisfies $g<0$, complementary slackness sets $\\mu=0$ — the constraint doesn't affect the solution. Check whether it's active first." }
  ],

  quiz: [
    { q: "At a constrained optimum of $f$ subject to $g=0$…", options: ["$\\nabla f=\\lambda\\nabla g$", "$\\nabla f=\\mathbf0$", "$\\nabla g=\\mathbf0$", "$f=g$"], answer: 0,
      explain: "The gradients are parallel; the objective only 'wants' to move across the constraint." },
    { q: "For min $x^2+y^2$ s.t. $x+y=1$, the solution is…", options: ["$(0.5,0.5)$", "$(1,0)$", "$(0,0)$", "$(0.5,0)$"], answer: 0,
      explain: "$2x=2y=\\lambda$ forces $x=y$, and $x+y=1$ gives $x=y=0.5$ — the closest point on the line to the origin." },
    { q: "The Lagrange multiplier $\\lambda$ represents…", options: ["the sensitivity of the optimum to the constraint", "the step size", "the curvature", "the gradient norm"], answer: 0,
      explain: "It's the shadow price: how much the optimal value changes per unit relaxation of the constraint." },
    { q: "Complementary slackness says…", options: ["$\\mu\\,g=0$ (constraint active or multiplier zero)", "$\\mu=g$", "$\\mu+g=0$", "$\\mu g\\ge0$"], answer: 0,
      explain: "Either the inequality constraint is tight ($g=0$) or it isn't binding ($\\mu=0$)." },
    { q: "Which is NOT a KKT condition?", options: ["the Hessian is positive definite", "dual feasibility $\\mu\\ge0$", "primal feasibility $g\\le0$", "stationarity $\\nabla f+\\mu\\nabla g=0$"], answer: 0,
      explain: "KKT are first-order necessary conditions; Hessian definiteness is a separate second-order check." }
  ],

  practice: [
    { level: "easy", prompt: "Write the Lagrangian for minimizing $f(x,y)$ subject to $x+2y=3$.", solution: "$\\mathcal L(x,y,\\lambda)=f(x,y)-\\lambda(x+2y-3)$. Setting $\\nabla_{x,y,\\lambda}\\mathcal L=\\mathbf0$ recovers $\\nabla f=\\lambda[1,2]$ and the constraint." },
    { level: "med", prompt: "Maximize $xy$ subject to $x+y=10$ using Lagrange.", solution: "$\\nabla(xy)=[y,x]=\\lambda[1,1]\\Rightarrow x=y$. Constraint: $2x=10\\Rightarrow x=y=5$, giving max $xy=25$ (AM–GM confirms)." },
    { level: "med", prompt: "Find the point on the line $3x+4y=25$ closest to the origin.", solution: "Minimize $x^2+y^2$ s.t. $3x+4y=25$: $[2x,2y]=\\lambda[3,4]\\Rightarrow x=\\tfrac{3\\lambda}{2},y=2\\lambda$. Sub: $3\\cdot\\tfrac{3\\lambda}{2}+4\\cdot2\\lambda=\\tfrac{9\\lambda}{2}+8\\lambda=\\tfrac{25\\lambda}{2}=25\\Rightarrow\\lambda=2$, so $(x,y)=(3,4)$, distance 5." },
    { level: "hard", prompt: "AI task: PPO maximizes expected reward subject to a KL constraint $\\mathrm{KL}(\\pi\\,\\Vert\\,\\pi_{\\text{old}})\\le\\delta$. Explain the role of the multiplier and how it appears in practice.", solution: "Form the Lagrangian $\\max_\\pi\\,E[\\text{reward}]-\\beta\\,(\\mathrm{KL}-\\delta)$. The multiplier $\\beta\\ge0$ is the price of moving away from the old policy: by KKT, if the KL bound is active ($\\mathrm{KL}=\\delta$) then $\\beta>0$ penalizes further drift; if the unconstrained step already stays within $\\delta$, complementary slackness gives $\\beta=0$ and the constraint is inert. In practice PPO either solves for $\\beta$ to hit the KL target (the Lagrangian/penalty form) or replaces the hard constraint with a clipped surrogate that mimics the same effect — both are ways of enforcing 'improve reward but don't move too far,' exactly the constrained optimization Lagrange multipliers describe." }
  ],

  deepDive: String.raw`<p><strong>Duality: every constrained problem has a shadow problem.</strong></p>
  <p>The Lagrangian $\mathcal L(\mathbf x,\boldsymbol\mu)=f(\mathbf x)+\sum_i\mu_i g_i(\mathbf x)$ defines two problems. The
  <strong>primal</strong> minimizes over $\mathbf x$ the worst-case Lagrangian; the <strong>dual</strong> maximizes over
  $\boldsymbol\mu\ge0$ the best-case one, $\max_{\boldsymbol\mu\ge0}\min_{\mathbf x}\mathcal L$. Weak duality always holds
  (the dual lower-bounds the primal); for convex problems, <strong>strong duality</strong> makes them equal, and the KKT
  conditions are exactly the bridge between the two solutions. This isn't abstract: the SVM you solve in practice is the
  <em>dual</em> problem, which is why support vectors (the active constraints, where $\mu_i>0$ by complementary slackness)
  are the only data points that matter, and why kernels — inner products in the dual — slot in so cleanly.</p>
  <p>The shadow-price reading of multipliers also illuminates regularization. A penalty $f+\lambda R$ (weight decay,
  KL penalty) is the Lagrangian of a constrained problem "minimize $f$ subject to $R\le t$"; the regularization strength
  $\lambda$ is the multiplier, and choosing $\lambda$ is implicitly choosing the budget $t$. So the soft penalties
  everywhere in ML and the hard constraints of classical optimization are two views of one duality — and this connects
  Track 8 forward to MAP estimation in Track 11, where the same $\lambda$ reappears as the strength of a prior. With this,
  the optimization track closes: you can now read training as constrained or unconstrained descent on a curved,
  possibly-non-convex loss, navigated by first- and second-order information.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["8.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 70 minutes.",

  intro: String.raw`<p>This exam spans all of Track 8: critical points and the Hessian test, convexity, gradient
  descent and its convergence, momentum and Adam, and Lagrange/KKT. <strong>Give yourself 70 minutes</strong>, produce
  each answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "Find and classify the critical point of $f(x)=x^2-6x+5$.",
      solution: "$f'(x)=2x-6=0\\Rightarrow x=3$; $f''=2>0\\Rightarrow$ minimum (value $f(3)=-4$)." },
    { level: "easy", prompt: "Is $f(x,y)=x^2+y^2$ convex? Give the Hessian.",
      solution: "$H=\\begin{bmatrix}2&0\\\\0&2\\end{bmatrix}$, PD everywhere ⇒ strictly convex." },
    { level: "med", prompt: "Classify the origin for $f(x,y)=x^2-3y^2$.",
      solution: "$\\nabla f=[2x,-6y]=\\mathbf0$ at origin; Hessian $\\mathrm{diag}(2,-6)$ has mixed signs ⇒ saddle." },
    { level: "med", prompt: "On $f=\\tfrac12\\lambda x^2$ with $\\lambda=8$, give the largest stable learning rate and the one giving one-step convergence.",
      solution: "Stable for $\\eta<2/\\lambda=0.25$. One-step: $\\eta=1/\\lambda=0.125$ (makes $1-\\eta\\lambda=0$)." },
    { level: "med", prompt: "A loss has Hessian eigenvalues $1$ and $100$. What is gradient descent's best convergence rate, and what does it imply?",
      solution: "$\\kappa=100$, rate $(\\kappa-1)/(\\kappa+1)=99/101\\approx0.98$ — each step cuts error by only ~2%, so convergence is very slow; momentum or preconditioning is needed." },
    { level: "med", prompt: "With $\\beta_1=0.9$, compute Adam's bias-corrected $\\hat m_1$ for a first gradient $g_1=2$.",
      solution: "$m_1=(1-0.9)\\cdot2=0.2$; $\\hat m_1=m_1/(1-0.9^1)=0.2/0.1=2$ — the correction recovers the true gradient at step 1." },
    { level: "hard", prompt: "Use Lagrange multipliers to minimize $x^2+y^2$ subject to $2x+y=5$.",
      solution: "$[2x,2y]=\\lambda[2,1]\\Rightarrow x=\\lambda,\\ y=\\lambda/2$. Constraint: $2\\lambda+\\lambda/2=\\tfrac{5\\lambda}{2}=5\\Rightarrow\\lambda=2$, so $(x,y)=(2,1)$, minimum value $5$." },
    { level: "hard", prompt: "Explain why deep-net training rarely gets stuck at bad local minima, in terms of Hessian eigenvalues.",
      solution: "A local minimum requires <em>all</em> $n$ Hessian eigenvalues positive. With many eigenvalues and roughly sign-balanced curvature, the chance of all-positive is exponentially small, so most critical points are saddles (mixed signs), which momentum and SGD noise escape. Empirically the minima that are found cluster at similar low loss, so 'bad' isolated minima are rare." },
    { level: "hard", prompt: "Why does dividing by $\\sqrt{v}$ (Adam/RMSProp) help on ill-conditioned losses, framed via the condition number?",
      solution: "Steep coordinates have large gradients → large $v$ → their steps are scaled down; flat coordinates have small $v$ → steps scaled up. This per-axis rescaling equalizes effective curvature, driving the condition number the optimizer 'sees' toward 1, where gradient descent's rate $(\\kappa-1)/(\\kappa+1)\\to0$ (fast). It's a cheap diagonal approximation to multiplying by $H^{-1}$." },
    { level: "hard", prompt: "AI task: weight decay can be seen as the Lagrangian of a constrained problem. State the constrained form and what the coefficient means.",
      solution: "Minimizing $\\text{loss}+\\lambda\\lVert\\mathbf w\\rVert^2$ is the Lagrangian of 'minimize the loss subject to $\\lVert\\mathbf w\\rVert^2\\le t$'. The coefficient $\\lambda$ is the Lagrange multiplier / shadow price of that budget: larger $\\lambda$ corresponds to a tighter budget $t$ (smaller allowed weight norm). So tuning weight-decay strength is implicitly choosing how large the weights may be — and via duality it equals a Gaussian prior's precision in MAP estimation (Track 11)." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> You can read optimizers and constrained objectives as math. On to Probability (Track 9) and, when ready, the Optimization capstone (GD/momentum/Adam from scratch).</li>
    <li><strong>7–8:</strong> Strong. Revisit the GD convergence rate or the Lagrange tangency argument if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive the Hessian classification and the $(\\kappa-1)/(\\kappa+1)$ rate; redo Lessons 8.1 and 8.3.</li>
    <li><strong>Below 5:</strong> Rework the track — this is where calculus and linear algebra become training, and every later practical choice builds on it.</li>
  </ul>`
};
