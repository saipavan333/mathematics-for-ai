/* ============================================================
   TRACK 6 — Calculus I — Derivatives & Gradients
   Opener: 6.3 Partial Derivatives & the Gradient (the marquee lesson).
   ============================================================ */
(window.LESSON_CONTENT ||= {})["6.3"] = {
  subtitle: "The gradient is the compass training follows — it points straight uphill on the loss.",

  aiMoment: String.raw`<p>Training a neural network is one sentence repeated millions of times:
  <em>$w \leftarrow w-\eta\,\nabla_w\mathcal{L}$</em>. The symbol $\nabla_w\mathcal{L}$ — the gradient of the loss with
  respect to every weight — is a vector that points in the direction the loss <strong>increases</strong> fastest, so we
  step the opposite way. Backpropagation exists for exactly one purpose: to compute this gradient efficiently. If you
  understand the gradient, you understand what every optimizer is steering by.</p>`,

  plainEnglish: String.raw`<p>A <strong>partial derivative</strong> is the slope you feel if you wiggle just one input
  and hold all the others still. The <strong>gradient</strong> collects all those individual slopes into one vector.
  That vector has a magical property: it points in the direction that makes the function climb the steepest, and its
  length says how steep.</p>`,

  intuition: String.raw`<p>Picture the loss as a hilly landscape and yourself standing on it blindfolded. Feel the
  slope under your feet in the east–west direction (one partial), then north–south (another). Combine them and you get
  one arrow pointing straight uphill — the gradient. On a contour map, that arrow is always perpendicular to the
  contour lines.</p>
  <figure class="figure">
  <svg viewBox="0 0 300 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gradient perpendicular to contour ellipses, pointing uphill">
    <ellipse cx="150" cy="125" rx="120" ry="80" fill="none" stroke="#cbd5e1"/>
    <ellipse cx="150" cy="125" rx="86"  ry="56" fill="none" stroke="#cbd5e1"/>
    <ellipse cx="150" cy="125" rx="52"  ry="33" fill="none" stroke="#cbd5e1"/>
    <circle cx="150" cy="125" r="4" fill="#94a3b8"/>
    <text x="156" y="121" font-size="10" fill="#94a3b8" font-family="sans-serif">peak</text>
    <circle cx="64" cy="160" r="4" fill="#4f46e5"/>
    <line x1="64" y1="160" x2="112" y2="142" stroke="#dc2626" stroke-width="2.6" marker-end="url(#g1)"/>
    <text x="30" y="176" font-size="11" fill="#4f46e5" font-family="sans-serif">you</text>
    <text x="92" y="135" font-size="11" fill="#dc2626" font-family="sans-serif">∇f (uphill)</text>
    <defs><marker id="g1" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#dc2626"/></marker></defs>
  </svg>
  <figcaption>Contours are lines of equal height; the gradient crosses them at right angles, pointing to higher ground.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $f:\mathbb{R}^n\to\mathbb{R}$, the <strong>partial derivative</strong> $\dfrac{\partial f}{\partial x_i}$
  is the ordinary derivative in $x_i$ with every other variable held constant. The <strong>gradient</strong> stacks them:</p>
  $$\nabla f(\mathbf x)=\left[\frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \dots,\ \frac{\partial f}{\partial x_n}\right]^{\!\top}.$$
  <p>The <strong>directional derivative</strong> along a unit vector $\mathbf u$ — the slope if you walk in direction
  $\mathbf u$ — is $D_{\mathbf u}f=\nabla f\cdot\mathbf u$. Plain English: $\nabla f$ is the list of one-at-a-time
  slopes, and dotting it with a direction tells you the slope that way.</p>`,

  derivation: String.raw`<p><strong>Claim.</strong> The gradient points in the direction of steepest ascent.
  We'll prove it and get the steepest <em>descent</em> direction for free.</p>
  <p><strong>Step 1 — slope in a chosen direction.</strong> The slope when walking along a unit vector $\mathbf u$ is
  $D_{\mathbf u}f=\nabla f\cdot\mathbf u$.</p>
  <p><strong>Step 2 — write the dot product with the angle.</strong> Using the geometric dot product (Lesson 2.3),
  $\nabla f\cdot\mathbf u=\lVert\nabla f\rVert\,\lVert\mathbf u\rVert\cos\theta=\lVert\nabla f\rVert\cos\theta$, since
  $\lVert\mathbf u\rVert=1$. Here $\theta$ is the angle between $\mathbf u$ and the gradient.</p>
  <p><strong>Step 3 — maximize over direction.</strong> We can only choose $\theta$. The expression
  $\lVert\nabla f\rVert\cos\theta$ is largest when $\cos\theta=1$, i.e. $\theta=0$ — when $\mathbf u$ points the
  <em>same way</em> as $\nabla f$.</p>
  <p><strong>Step 4 — conclude.</strong> Steepest ascent is along $\nabla f$; the maximum slope is exactly
  $\lVert\nabla f\rVert$. The most negative slope is at $\theta=180^\circ$, along $-\nabla f$. That is why gradient
  descent steps in $-\nabla f$. $\blacksquare$</p>
  <hr class="soft">
  <p><strong>Two gradients you must know cold.</strong></p>
  <p><em>Linear:</em> $f(\mathbf x)=\mathbf w\cdot\mathbf x=\sum_i w_ix_i$. Then $\partial f/\partial x_j=w_j$, so
  $\nabla_{\mathbf x} f=\mathbf w$. Plain English: the slope of a linear function is just its weight vector.</p>
  <p><em>Quadratic:</em> $f(\mathbf x)=\tfrac12\mathbf x^\top\mathbf x=\tfrac12\sum_i x_i^2$. Then
  $\partial f/\partial x_j=x_j$, so $\nabla f=\mathbf x$. More generally for symmetric $A$,
  $\nabla\big(\tfrac12\mathbf x^\top A\mathbf x\big)=A\mathbf x$ — the matrix version you'll use for the Hessian and
  for Newton's method.</p>`,

  code: [
    { label: "Gradient by hand, checked by finite differences", src: String.raw`
import numpy as np

# f(x,y) = x^2 + 3xy + y^2     ->  ∇f = [2x+3y, 3x+2y]
def f(p):     x,y = p; return x**2 + 3*x*y + y**2
def grad(p):  x,y = p; return np.array([2*x+3*y, 3*x+2*y])

p = np.array([1.0, 2.0])
analytic = grad(p)

# numerical gradient: nudge each coordinate a tiny bit
eps = 1e-6
num = np.zeros(2)
for i in range(2):
    e = np.zeros(2); e[i] = eps
    num[i] = (f(p+e) - f(p-e)) / (2*eps)

print("analytic ∇f =", analytic)     # [8. 7.]
print("numeric  ∇f =", np.round(num, 5))
print("match?       ", np.allclose(analytic, num))
` },
    { label: "See the gradient field and a descent step (contour + quiver)", src: String.raw`
import numpy as np, matplotlib.pyplot as plt

def f(x,y):  return x**2 + 0.5*y**2          # a bowl
xs = np.linspace(-3,3,30); ys = np.linspace(-3,3,30)
X,Y = np.meshgrid(xs,ys)
gx, gy = 2*X, Y                              # ∇f = [2x, y]

plt.figure(figsize=(5,4))
plt.contour(X, Y, f(X,Y), levels=12, cmap="Blues")
plt.quiver(X, Y, -gx, -gy, color="crimson", alpha=.6)   # NEGATIVE gradient = downhill
p = np.array([2.6, 2.6])
plt.plot(*p, "ko");
plt.title("Red arrows = −∇f point toward the minimum"); plt.tight_layout(); plt.show()
` }
  ],

  keyPoints: [
    "A partial derivative wiggles one variable and freezes the rest; the gradient stacks all partials.",
    "$\\nabla f$ points in the direction of steepest increase; its length is that maximum slope.",
    "Gradient descent steps along $-\\nabla f$ — steepest decrease.",
    "Directional derivative $=\\nabla f\\cdot\\mathbf u$; the gradient is perpendicular to level sets.",
    "$\\nabla(\\mathbf w\\cdot\\mathbf x)=\\mathbf w$ and $\\nabla(\\tfrac12\\mathbf x^\\top\\mathbf x)=\\mathbf x$."
  ],

  commonMistakes: [
    { wrong: "Thinking $-\\nabla f$ points straight at the minimum.", why: "It points to the steepest <em>local</em> drop, which is perpendicular to the contour you're on — not generally toward the bottom. On stretched bowls this causes the zig-zag of plain gradient descent." },
    { wrong: "Taking the gradient with respect to the data instead of the parameters.", why: "Training differentiates the loss w.r.t. weights $w$, not inputs $x$. (Input gradients are a different tool — used for adversarial examples and saliency maps.)" },
    { wrong: "Dropping the minus sign and ascending the loss.", why: "$+\\nabla$ increases the loss. A flipped sign makes training diverge — a classic bug when hand-rolling an optimizer." }
  ],

  quiz: [
    { q: "For $f(x,y)=x^2y$, what is $\\nabla f$ at $(2,3)$?", options: ["$[12,4]$", "$[4,12]$", "$[6,2]$", "$[12,12]$"], answer: 0,
      explain: "$\\partial_x=2xy=12$, $\\partial_y=x^2=4$, so $[12,4]$. Choice $[4,12]$ swaps the partials." },
    { q: "The maximum slope of $f$ at a point equals…", options: ["$\\lVert\\nabla f\\rVert$", "$\\nabla f\\cdot\\nabla f$", "$0$", "the largest partial derivative"], answer: 0,
      explain: "Steepest ascent slope is $\\lVert\\nabla f\\rVert\\cos0=\\lVert\\nabla f\\rVert$. The largest single partial undercounts because it ignores the others." },
    { q: "Walking exactly along a contour line (level set), the directional derivative is…", options: ["0", "$\\lVert\\nabla f\\rVert$", "negative", "undefined"], answer: 0,
      explain: "Along a contour the height doesn't change, so the slope is $0$ — which is why $\\nabla f$ is perpendicular to contours ($\\cos90°=0$)." },
    { q: "$\\nabla(\\mathbf w\\cdot\\mathbf x)$ with respect to $\\mathbf x$ is…", options: ["$\\mathbf w$", "$\\mathbf x$", "$1$", "$\\mathbf w\\cdot\\mathbf x$"], answer: 0,
      explain: "Each $\\partial/\\partial x_j(\\sum_i w_ix_i)=w_j$, so the gradient is $\\mathbf w$ — the weights are the slopes." },
    { q: "Learning rate $\\eta=0.1$, gradient $\\nabla\\mathcal L=[2,-4]$ at $w=[1,1]$. After one step $w\\leftarrow w-\\eta\\nabla\\mathcal L$?", options: ["$[0.8,1.4]$", "$[1.2,0.6]$", "$[0.8,0.6]$", "$[2,-4]$"], answer: 0,
      explain: "$w-0.1[2,-4]=[1-0.2,\\,1+0.4]=[0.8,1.4]$. Choice $[1.2,0.6]$ used $+\\nabla$ (wrong sign)." }
  ],

  practice: [
    { level: "easy", prompt: "Compute the partials of $f(x,y)=3x+y^2$.", solution: "$\\partial_x f=3$ (treat $y$ constant), $\\partial_y f=2y$. So $\\nabla f=[3,2y]$." },
    { level: "easy", prompt: "At which point is $\\nabla f=\\mathbf 0$ for $f(x,y)=x^2+y^2$?", solution: "$\\nabla f=[2x,2y]=\\mathbf 0$ only at the origin $(0,0)$ — the minimum of the bowl." },
    { level: "med", prompt: "Find the direction of steepest ascent of $f(x,y)=x^2+y^2$ at $(3,4)$ and the maximum slope there.", solution: "$\\nabla f=[6,8]$. Direction: the unit vector $[6,8]/10=[0.6,0.8]$. Maximum slope $=\\lVert[6,8]\\rVert=10$." },
    { level: "med", prompt: "For $f(\\mathbf x)=\\tfrac12\\lVert\\mathbf x\\rVert^2$, show $\\nabla f=\\mathbf x$.", solution: "$f=\\tfrac12\\sum_i x_i^2$, so $\\partial f/\\partial x_j=\\tfrac12\\cdot 2x_j=x_j$. Stacking gives $\\nabla f=\\mathbf x$." },
    { level: "hard", prompt: "AI task: a linear model has loss $\\mathcal L(\\mathbf w)=\\tfrac12(\\mathbf w\\cdot\\mathbf x-y)^2$ for one example. Derive $\\nabla_{\\mathbf w}\\mathcal L$ and write the gradient-descent update.", solution: "Let $r=\\mathbf w\\cdot\\mathbf x-y$ (the residual). By the chain rule, $\\nabla_{\\mathbf w}\\mathcal L=r\\cdot\\nabla_{\\mathbf w}(\\mathbf w\\cdot\\mathbf x)=r\\,\\mathbf x=(\\mathbf w\\cdot\\mathbf x-y)\\mathbf x$. Update: $\\mathbf w\\leftarrow\\mathbf w-\\eta(\\mathbf w\\cdot\\mathbf x-y)\\mathbf x$. This is the delta rule — error times input — the seed of all of supervised learning." }
  ],

  deepDive: String.raw`<p><strong>Why steepest descent zig-zags — and why the gradient is perpendicular to contours.</strong></p>
  <p>First, the perpendicularity. Move a tiny step $\mathbf u$ along a contour, where the value $f$ doesn't change.
  The change in $f$ is approximately $\nabla f\cdot\mathbf u=0$ (no height change). A dot product of zero means
  $\nabla f\perp\mathbf u$: the gradient is orthogonal to the contour. So $-\nabla f$ leaves the contour at a right
  angle — the locally fastest way down.</p>
  <p>But "locally fastest" is not "globally toward the minimum." On an elongated valley (contours are stretched
  ellipses), the steepest direction points mostly across the valley, not along it. The next step over-corrects, and
  the path bounces between the walls — the notorious zig-zag of vanilla gradient descent. This single picture
  motivates almost every optimizer upgrade in Track 8: <strong>momentum</strong> averages out the bouncing,
  <strong>RMSProp/Adam</strong> rescale each axis so the bowl looks round, and <strong>Newton's method</strong> uses
  the Hessian (Lesson 6.4) to jump straight to the bottom of a quadratic. The gradient tells you which way is down;
  the curvature tells you how far to trust it.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["6.1"] = {
  subtitle: "The slope at a single instant — the quantity training is built to follow.",

  aiMoment: String.raw`<p>Gradient descent can only work if the loss has a <strong>slope</strong> to follow — a
  derivative. That's why we use smooth activations and smooth losses, and why the kink in ReLU at $0$ needs special
  handling (a sub-gradient). Before gradients of millions of parameters, there's one idea: the derivative, the
  instantaneous rate of change of a function. Everything in this track builds on it.</p>`,

  plainEnglish: String.raw`<p>A <strong>limit</strong> is the value a function heads toward as you approach a point.
  A function is <strong>continuous</strong> if it has no jumps. The <strong>derivative</strong> is the slope of the
  function at a single point — how fast the output changes for a tiny nudge in the input.</p>`,

  intuition: String.raw`<p>Draw a line through two points on a curve (a secant); its slope is the average rate of
  change. Slide the second point toward the first and the secant tips over into the <strong>tangent</strong> — its slope
  is the derivative, the instantaneous rate.</p>
  <figure class="figure">
  <svg viewBox="0 0 280 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Secant lines approaching the tangent">
    <polyline points="40,133 60,109 80,89 100,75 120,65 140,61 150,60 160,61 180,65 200,75 220,89 240,109 260,133" fill="none" stroke="#4f46e5" stroke-width="2.4"/>
    <line x1="100" y1="75" x2="180" y2="65" stroke="#94a3b8" stroke-dasharray="4 3"/>
    <line x1="100" y1="75" x2="140" y2="61" stroke="#94a3b8" stroke-dasharray="4 3"/>
    <line x1="100" y1="75" x2="120" y2="65" stroke="#94a3b8" stroke-dasharray="4 3"/>
    <line x1="55" y1="102" x2="170" y2="33" stroke="#0d9488" stroke-width="2.6"/>
    <circle cx="100" cy="75" r="4" fill="#dc2626"/>
    <text x="82" y="72" font-size="12" fill="#dc2626" font-family="sans-serif">P</text>
    <text x="112" y="26" font-size="11" fill="#0d9488" font-family="sans-serif">tangent (slope = f′)</text>
    <text x="150" y="100" font-size="10" fill="#64748b" font-family="sans-serif">secants</text>
  </svg>
  <figcaption>As the second point slides into P, the secant's slope approaches the tangent's — the derivative.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>derivative</strong> of $f$ at $x$ is the limit of the difference quotient:</p>
  $$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.$$
  <p>It measures the instantaneous rate of change (and the tangent slope). A function is <strong>continuous</strong> at
  $x$ if $\lim_{t\to x}f(t)=f(x)$ (no jump). Differentiable $\Rightarrow$ continuous, but not the reverse — a function
  can be continuous yet have a corner (like $|x|$ at $0$) where no single slope exists.</p>`,

  derivation: String.raw`<p><strong>Derivative of $f(x)=x^2$ from scratch.</strong></p>
  <p><strong>Step 1 — form the quotient:</strong> $\dfrac{(x+h)^2-x^2}{h}$.</p>
  <p><strong>Step 2 — expand the numerator:</strong> $(x^2+2xh+h^2)-x^2=2xh+h^2$.</p>
  <p><strong>Step 3 — divide by $h$:</strong> $\dfrac{2xh+h^2}{h}=2x+h$ (valid for $h\neq0$).</p>
  <p><strong>Step 4 — take the limit</strong> $h\to0$: $f'(x)=2x.\;\blacksquare$ Plain English: near any $x$, bumping the
  input by a hair changes $x^2$ at rate $2x$.</p>
  <hr class="soft">
  <p><strong>Why $|x|$ has no derivative at 0.</strong> From the right ($h\to0^{+}$), $\frac{|0+h|-0}{h}=\frac{h}{h}=1$.
  From the left ($h\to0^{-}$), $\frac{|h|}{h}=\frac{-h}{h}=-1$. The two one-sided limits disagree ($1\neq-1$), so the
  limit — and the derivative — doesn't exist. The corner has no single slope, which is exactly the kink frameworks
  navigate with a sub-gradient for ReLU.</p>`,

  code: [
    { label: "Numerical derivative vs the exact 2x", src: String.raw`
import numpy as np

f = lambda x: x**2
def deriv(f, x, h=1e-6): return (f(x+h) - f(x-h)) / (2*h)   # central difference

for x in [1.0, 2.0, -3.0]:
    print(f"x={x:4}:  numeric f'={deriv(f,x):.5f}   exact 2x={2*x}")
` },
    { label: "Plot f and its derivative", src: String.raw`
import numpy as np, matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
f  = x**2
fp = 2*x                      # exact derivative
plt.figure(figsize=(5,4))
plt.plot(x, f,  label="f(x)=x²")
plt.plot(x, fp, label="f'(x)=2x")
plt.axhline(0, color="k", lw=.5); plt.legend(); plt.title("A function and its slope")
plt.tight_layout(); plt.show()
` }
  ],

  keyPoints: [
    "A limit is the value a function approaches; continuity means no jumps.",
    "The derivative $f'(x)=\\lim_{h\\to0}\\frac{f(x+h)-f(x)}{h}$ is the instantaneous rate / tangent slope.",
    "Differentiable ⇒ continuous, but not vice versa (corners like $|x|$ break differentiability).",
    "$\\frac{d}{dx}x^n=nx^{n-1}$; in particular $(x^2)'=2x$.",
    "ReLU's kink at 0 has no true derivative — frameworks use a sub-gradient (0 or 1)."
  ],

  commonMistakes: [
    { wrong: "Assuming continuous means differentiable.", why: "$|x|$ is continuous everywhere but has no derivative at $0$ — a corner. Smoothness is stronger than continuity." },
    { wrong: "Using a one-sided difference for numerical derivatives.", why: "The central difference $\\frac{f(x+h)-f(x-h)}{2h}$ is far more accurate ($O(h^2)$ error) than the forward difference ($O(h)$) — important for gradient checking (Track 13)." },
    { wrong: "Picking $h$ too small in finite differences.", why: "Too-small $h$ triggers catastrophic cancellation (Track 4): $f(x+h)\\approx f(x)$ in float, so the subtraction loses all digits. There's a sweet spot around $\\sqrt{\\epsilon_{machine}}$." }
  ],

  quiz: [
    { q: "$\\frac{d}{dx}x^3$ at $x=2$ is…", options: ["12", "8", "6", "4"], answer: 0,
      explain: "$(x^3)'=3x^2=3\\cdot4=12$." },
    { q: "Which function is continuous but not differentiable at 0?", options: ["$|x|$", "$x^2$", "$e^x$", "$3x$"], answer: 0,
      explain: "$|x|$ has a corner at 0 (left slope $-1$, right slope $+1$); the others are smooth." },
    { q: "The central difference uses…", options: ["$\\frac{f(x+h)-f(x-h)}{2h}$", "$\\frac{f(x+h)-f(x)}{h}$", "$\\frac{f(x)-f(x-h)}{h}$", "$f'(x)h$"], answer: 0,
      explain: "Symmetric points around $x$ cancel the first-order error, giving $O(h^2)$ accuracy." },
    { q: "If $f'(x)=0$ at a point, the tangent there is…", options: ["horizontal", "vertical", "undefined", "at 45°"], answer: 0,
      explain: "Zero slope means a flat (horizontal) tangent — a candidate max, min, or saddle." },
    { q: "From the limit, $\\frac{d}{dx}(5x)$ is…", options: ["5", "$5x$", "$5h$", "0"], answer: 0,
      explain: "$\\frac{5(x+h)-5x}{h}=\\frac{5h}{h}=5$, independent of $x$ — a line has constant slope." }
  ],

  practice: [
    { level: "easy", prompt: "Differentiate $f(x)=4x^2-3x+7$.", solution: "Term by term: $f'(x)=8x-3$ (the constant 7 differentiates to 0)." },
    { level: "easy", prompt: "What is the slope of $g(x)=x^2$ at $x=-1$?", solution: "$g'(x)=2x$, so $g'(-1)=-2$." },
    { level: "med", prompt: "Use the limit definition to find $f'(x)$ for $f(x)=\\frac1x$.", solution: "$\\frac{1}{h}\\left(\\frac{1}{x+h}-\\frac1x\\right)=\\frac{1}{h}\\cdot\\frac{x-(x+h)}{x(x+h)}=\\frac{-1}{x(x+h)}$. As $h\\to0$: $f'(x)=-\\frac{1}{x^2}$." },
    { level: "hard", prompt: "AI task: explain why ReLU is used despite being non-differentiable at 0, and what value frameworks assign to its derivative there.", solution: "ReLU $=\\max(0,x)$ is differentiable everywhere except the single point $x=0$ (slope 0 for $x<0$, slope 1 for $x>0$). The non-differentiable set is measure-zero, so it almost never matters during training; when an input is exactly 0, frameworks pick a <em>sub-gradient</em> — typically $0$ (PyTorch/TensorFlow) — any value in $[0,1]$ being valid. ReLU's piecewise-linear simplicity gives strong, non-vanishing gradients for $x>0$, which is why it trains deep nets better than the saturating sigmoid despite the kink." }
  ],

  deepDive: String.raw`<p><strong>Differentiability, smoothness, and what "almost everywhere" buys deep learning.</strong></p>
  <p>Classical optimization assumes smooth (differentiable) objectives, yet modern networks are full of kinks: ReLU,
  max-pooling, $L_1$ penalties, hard attention. How does gradient descent survive? The key is that these functions are
  differentiable <em>almost everywhere</em> — the bad set (corners) has measure zero, so a randomly-landed activation is
  smooth with probability 1, and at a corner any sub-gradient is a valid descent direction. Frameworks implement a fixed
  convention (e.g. ReLU'(0)=0) and move on.</p>
  <p>This matters because the alternative — insisting on everywhere-smooth activations like sigmoid/tanh — brings its own
  disease: saturation. Where those curves flatten, the derivative is near zero, so gradients <em>vanish</em> and deep
  stacks stop learning. The field traded a measure-zero non-differentiability (ReLU's kink) for healthy gradients
  everywhere else, and it was a decisive win. The derivative isn't just a slope to compute; its <em>size</em> across the
  input range determines whether learning signal survives many layers — a theme you'll quantify with the chain rule
  next, and with numerical stability in Track 13.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["6.2"] = {
  subtitle: "Differentiate nested functions by multiplying rates — this rule IS backpropagation.",

  aiMoment: String.raw`<p><strong>Backpropagation is the chain rule, applied layer by layer.</strong> A network
  $\mathcal L(f_L(\dots f_1(\mathbf x)))$ is a deep composition, and its gradient is a product of each layer's local
  derivative. Master the chain rule on paper and backprop stops being magic — it's bookkeeping for a product of rates.
  The other rules (power, product, quotient) and the derivatives of $e^x$, $\log$, and sigmoid are the per-layer pieces
  that product is made of.</p>`,

  plainEnglish: String.raw`<p>The <strong>chain rule</strong> says: to find how fast a nested function changes, multiply
  the rates at each stage. If $y$ depends on $u$ and $u$ depends on $x$, then a wiggle in $x$ moves $u$ at rate
  $du/dx$, which moves $y$ at rate $dy/du$ — so $y$ moves at the product $\frac{dy}{du}\cdot\frac{du}{dx}$.</p>`,

  intuition: String.raw`<p>Think of meshed gears. Turn the input gear and the rate passes down the chain, each stage
  multiplying its own gear ratio. The overall ratio from input to output is the product of the individual ratios — that
  product is the chain rule, and reading it backward is backprop.</p>
  <figure class="figure">
  <svg viewBox="0 0 360 96" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Chain rule as multiplying rates along a chain">
    <g font-family="sans-serif" font-size="12">
    <text x="10" y="52" fill="#20242c">x</text>
    <line x1="22" y1="48" x2="58" y2="48" stroke="#94a3b8" marker-end="url(#c1)"/>
    <rect x="60" y="30" width="70" height="36" rx="6" fill="#f7f3ff" stroke="#7c3aed"/>
    <text x="74" y="53" fill="#7c3aed">u = g(x)</text>
    <line x1="130" y1="48" x2="166" y2="48" stroke="#94a3b8" marker-end="url(#c1)"/>
    <text x="132" y="24" font-size="10" fill="#64748b">×g′(x)</text>
    <rect x="168" y="30" width="74" height="36" rx="6" fill="#fff7ed" stroke="#d97706"/>
    <text x="182" y="53" fill="#d97706">y = f(u)</text>
    <line x1="242" y1="48" x2="278" y2="48" stroke="#94a3b8" marker-end="url(#c1)"/>
    <text x="240" y="24" font-size="10" fill="#64748b">×f′(u)</text>
    <text x="286" y="52" fill="#20242c">y</text>
    <text x="86" y="90" fill="#16a34a">dy/dx = f′(u) · g′(x)</text>
    </g>
    <defs><marker id="c1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
  </svg>
  <figcaption>Rates multiply along the chain; backprop reads this product from output back to input.</figcaption>
  </figure>`,

  formalism: String.raw`<p><strong>Core rules.</strong> Power: $(x^n)'=nx^{n-1}$. Product: $(uv)'=u'v+uv'$. Quotient:
  $(u/v)'=\frac{u'v-uv'}{v^2}$. <strong>Chain rule:</strong></p>
  $$(f\circ g)'(x)=f'\big(g(x)\big)\cdot g'(x),\qquad\text{or}\qquad \frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}.$$
  <p>Key derivatives: $\frac{d}{dx}e^x=e^x$, $\frac{d}{dx}\ln x=\frac1x$, and for the sigmoid
  $\sigma(x)=\frac{1}{1+e^{-x}}$, the tidy $\sigma'(x)=\sigma(x)\big(1-\sigma(x)\big)$.</p>`,

  derivation: String.raw`<p><strong>Part 1 — the chain rule.</strong> Let $y=f(u)$, $u=g(x)$.</p>
  <p><strong>Step 1.</strong> The rate is $\dfrac{dy}{dx}=\lim_{h\to0}\dfrac{f(g(x+h))-f(g(x))}{h}$.</p>
  <p><strong>Step 2.</strong> Let $k=g(x+h)-g(x)$, the change in $u$. Multiply and divide by $k$ (when $k\neq0$):
  $\dfrac{f(g(x)+k)-f(g(x))}{k}\cdot\dfrac{k}{h}.$</p>
  <p><strong>Step 3.</strong> As $h\to0$, $k\to0$ too: the first factor $\to f'(g(x))$ and the second $\to g'(x)$.
  Hence $\dfrac{dy}{dx}=f'(g(x))\,g'(x).\;\blacksquare$</p>
  <hr class="soft">
  <p><strong>Part 2 — the sigmoid derivative.</strong> $\sigma(x)=(1+e^{-x})^{-1}$.</p>
  <p><strong>Step 1 — chain rule</strong> with outer $u^{-1}$ and inner $1+e^{-x}$:
  $\sigma'(x)=-(1+e^{-x})^{-2}\cdot(-e^{-x})=\dfrac{e^{-x}}{(1+e^{-x})^2}.$</p>
  <p><strong>Step 2 — rewrite</strong> as a product: $\dfrac{e^{-x}}{(1+e^{-x})^2}=\dfrac{1}{1+e^{-x}}\cdot\dfrac{e^{-x}}{1+e^{-x}}=\sigma(x)\big(1-\sigma(x)\big),$
  using $1-\sigma=\frac{e^{-x}}{1+e^{-x}}$. $\blacksquare$ Plain English: the sigmoid's slope is its output times one
  minus its output — which is why it vanishes when $\sigma$ saturates near 0 or 1.</p>`,

  code: [
    { label: "Chain rule, checked numerically", src: String.raw`
import numpy as np

g  = lambda x: x**2 + 1
f  = lambda u: np.sin(u)
gp = lambda x: 2*x
fp = lambda u: np.cos(u)

x = 1.3
analytic = fp(g(x)) * gp(x)                       # chain rule
numeric  = (f(g(x+1e-6)) - f(g(x-1e-6))) / 2e-6   # finite difference
print("chain rule :", round(analytic, 6))
print("numeric    :", round(numeric, 6), "-> match")
` },
    { label: "Sigmoid derivative = σ(1−σ)", src: String.raw`
import numpy as np
sig = lambda x: 1/(1+np.exp(-x))

x = np.array([-2., 0., 3.])
analytic = sig(x)*(1-sig(x))
numeric  = (sig(x+1e-6) - sig(x-1e-6)) / 2e-6
print("σ(1-σ)  :", np.round(analytic, 6))
print("numeric :", np.round(numeric, 6))
print("max slope at x=0:", round(float(sig(0)*(1-sig(0))), 3), "(saturates -> ~0 far out)")
` }
  ],

  keyPoints: [
    "Chain rule: $(f\\circ g)'=f'(g(x))\\,g'(x)$ — multiply the local rates.",
    "Backpropagation is the chain rule applied across layers (a product of derivatives).",
    "Product rule $(uv)'=u'v+uv'$; quotient $(u/v)'=\\frac{u'v-uv'}{v^2}$.",
    "$\\frac{d}{dx}e^x=e^x$, $\\frac{d}{dx}\\ln x=\\frac1x$, $\\sigma'=\\sigma(1-\\sigma)$.",
    "Saturated sigmoids/tanh give near-zero derivatives → vanishing gradients."
  ],

  commonMistakes: [
    { wrong: "Forgetting the inner derivative in the chain rule.", why: "$\\frac{d}{dx}\\sin(x^2)=\\cos(x^2)\\cdot 2x$, not $\\cos(x^2)$. Dropping $g'(x)$ is the most common differentiation error — and a real backprop bug." },
    { wrong: "Applying the product rule as $(uv)'=u'v'$.", why: "It's $u'v+uv'$. The product of derivatives is not the derivative of the product." },
    { wrong: "Ignoring vanishing gradients from stacked saturating nonlinearities.", why: "Each sigmoid contributes a factor $\\le0.25$ to the product; through many layers the gradient shrinks geometrically toward 0, stalling learning." }
  ],

  quiz: [
    { q: "$\\frac{d}{dx}\\sin(x^2)$ is…", options: ["$2x\\cos(x^2)$", "$\\cos(x^2)$", "$2x\\sin(x^2)$", "$\\cos(2x)$"], answer: 0,
      explain: "Chain rule: outer $\\cos(x^2)$ times inner $2x$." },
    { q: "$\\frac{d}{dx}(x^2 e^x)$ is…", options: ["$2x e^x + x^2 e^x$", "$2x e^x$", "$x^2 e^x$", "$2xe^{x}\\cdot x^2 e^x$"], answer: 0,
      explain: "Product rule: $u'v+uv'=2x\\,e^x+x^2 e^x=e^x(2x+x^2)$." },
    { q: "The maximum value of $\\sigma'(x)=\\sigma(1-\\sigma)$ is…", options: ["0.25 (at $x=0$)", "1", "0.5", "$\\infty$"], answer: 0,
      explain: "$\\sigma(0)=0.5$, so $\\sigma'(0)=0.5\\cdot0.5=0.25$ — the largest the sigmoid's slope ever gets." },
    { q: "If $y=f(u)$, $u=g(v)$, $v=h(x)$, then $\\frac{dy}{dx}$ is…", options: ["$f'(u)g'(v)h'(x)$", "$f'(u)+g'(v)+h'(x)$", "$f'(u)h'(x)$", "$f'g'h'/3$"], answer: 0,
      explain: "Chain rule across three stages multiplies all three local rates — exactly how backprop chains layers." },
    { q: "Ten stacked sigmoids each contribute a slope factor ≤ 0.25. The gradient through all ten is bounded by…", options: ["$0.25^{10}\\approx10^{-6}$", "$0.25\\times10$", "$0.25$", "1"], answer: 0,
      explain: "The factors multiply: $0.25^{10}\\approx9.5\\times10^{-7}$ — vanishing gradients, the reason ReLU/residуals exist." }
  ],

  practice: [
    { level: "easy", prompt: "Differentiate $f(x)=(3x+1)^4$.", solution: "Chain rule: $4(3x+1)^3\\cdot3=12(3x+1)^3$." },
    { level: "easy", prompt: "Differentiate $\\ln(x^2+1)$.", solution: "$\\frac{1}{x^2+1}\\cdot2x=\\frac{2x}{x^2+1}$." },
    { level: "med", prompt: "Differentiate $h(x)=\\frac{e^x}{x}$.", solution: "Quotient rule: $\\frac{e^x\\cdot x-e^x\\cdot1}{x^2}=\\frac{e^x(x-1)}{x^2}$." },
    { level: "hard", prompt: "AI task: a unit computes $L=\\tfrac12(\\sigma(wx+b)-y)^2$. Use the chain rule to find $\\partial L/\\partial w$ and identify each factor as a backprop message.", solution: "Let $z=wx+b$, $a=\\sigma(z)$, so $L=\\tfrac12(a-y)^2$. Chain: $\\frac{\\partial L}{\\partial w}=\\underbrace{(a-y)}_{\\partial L/\\partial a}\\cdot\\underbrace{\\sigma(z)(1-\\sigma(z))}_{\\partial a/\\partial z}\\cdot\\underbrace{x}_{\\partial z/\\partial w}$. Reading right-to-left: the input $x$, the local sigmoid slope $\\sigma(1-\\sigma)$, and the output error $(a-y)$. That product — error × local slope × input — is exactly the gradient backprop computes and propagates; the same three factors reappear at every layer." }
  ],

  deepDive: String.raw`<p><strong>The chain rule as a product of Jacobians — why backprop is cheap.</strong></p>
  <p>For vector-valued layers the chain rule becomes a product of <strong>Jacobian</strong> matrices (Lesson 6.4):
  $\nabla_{\mathbf x}\mathcal L = J_1^\top J_2^\top\cdots J_L^\top\,\nabla_{\mathbf y}\mathcal L$. Two orders are
  possible. Multiply <em>left to right</em> (forward mode) and you build full Jacobian matrices — expensive when there
  are many inputs. Multiply <em>right to left</em> (reverse mode = backprop), starting from the scalar loss, and every
  step is a cheap Jacobian-<em>vector</em> product: a matrix times a vector, never matrix times matrix. Because the loss
  is one number, reverse mode computes the gradient with respect to <em>all</em> parameters in roughly the cost of a
  single forward pass.</p>
  <p>That asymmetry is the whole reason deep learning is feasible. A network has millions of parameters (inputs to the
  loss) and one output (the loss), the exact regime where reverse-mode autodiff wins overwhelmingly. The "magic" of
  backprop is just the chain rule plus the decision to multiply in the cheap order — which you'll build from scratch as a
  tiny autograd engine in Track 13. Every gradient you've ever logged is this product of local derivatives, accumulated
  backward.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["6.4"] = {
  subtitle: "All the first partials (Jacobian) and all the second partials (Hessian) — local slope and curvature.",

  aiMoment: String.raw`<p>Reverse-mode autodiff is, under the hood, a chain of <strong>Jacobian</strong>–vector products
  — each layer contributes its Jacobian, and backprop multiplies them onto the upstream gradient. The
  <strong>Hessian</strong> — the matrix of second derivatives — is the curvature of the loss: its eigenvalues say how
  steep or flat each direction is (Track 5), Newton's method uses it to choose step sizes, and its sign pattern
  distinguishes minima from saddles. Together they are the first- and second-order local pictures of any function.</p>`,

  plainEnglish: String.raw`<p>The <strong>Jacobian</strong> packs every first partial derivative of a function with
  several inputs and outputs into a grid — the best linear approximation of the map near a point. The
  <strong>Hessian</strong> packs every second partial of a scalar function into a grid — its local curvature.</p>`,

  intuition: String.raw`<p>The Jacobian is a table: row $i$, column $j$ is "how much output $i$ changes when input $j$
  wiggles." The Hessian is the table of curvatures: how the slope itself changes. One linearizes a vector map; the other
  bends a scalar surface into a bowl or saddle.</p>
  <figure class="figure">
  <svg viewBox="0 0 340 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Jacobian and Hessian as grids of partial derivatives">
    <g font-family="sans-serif">
    <text x="20" y="20" font-size="11" fill="#4f46e5">Jacobian (m×n)</text>
    <rect x="20" y="28" width="120" height="80" rx="4" fill="#eef0ff" stroke="#4f46e5"/>
    <text x="34" y="56" font-size="12" fill="#20242c">∂f₁/∂x₁  ∂f₁/∂x₂</text>
    <text x="34" y="84" font-size="12" fill="#20242c">∂f₂/∂x₁  ∂f₂/∂x₂</text>
    <text x="196" y="20" font-size="11" fill="#d97706">Hessian (n×n, symmetric)</text>
    <rect x="196" y="28" width="124" height="80" rx="4" fill="#fff7ed" stroke="#d97706"/>
    <text x="210" y="56" font-size="12" fill="#20242c">∂²f/∂x²    ∂²f/∂x∂y</text>
    <text x="210" y="84" font-size="12" fill="#20242c">∂²f/∂y∂x   ∂²f/∂y²</text>
    </g>
  </svg>
  <figcaption>Jacobian = all first partials of a vector map; Hessian = all second partials of a scalar (and it's symmetric).</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $\mathbf f:\mathbb{R}^n\to\mathbb{R}^m$, the <strong>Jacobian</strong> is the $m\times n$
  matrix $J_{ij}=\dfrac{\partial f_i}{\partial x_j}$. It composes by multiplication: $J_{\mathbf f\circ\mathbf g}(\mathbf x)=J_{\mathbf f}(\mathbf g(\mathbf x))\,J_{\mathbf g}(\mathbf x)$.
  For a scalar $f:\mathbb{R}^n\to\mathbb{R}$, the <strong>Hessian</strong> is the $n\times n$ matrix
  $H_{ij}=\dfrac{\partial^2 f}{\partial x_i\,\partial x_j}$ — the Jacobian of the gradient. It is <strong>symmetric</strong>
  (mixed partials commute, Clairaut's theorem), and its definiteness (Track 5.3) classifies critical points.</p>`,

  derivation: String.raw`<p><strong>Two Jacobians/Hessians you must know.</strong></p>
  <p><strong>(A) Jacobian of a linear map $\mathbf f(\mathbf x)=A\mathbf x$.</strong> Component $f_i=\sum_j A_{ij}x_j$, so
  $\dfrac{\partial f_i}{\partial x_k}=A_{ik}$. Hence $J=A$ — a linear map <em>is</em> its own best linear approximation.</p>
  <p><strong>(B) Hessian of the quadratic $f(\mathbf x)=\tfrac12\mathbf x^\top A\mathbf x$ (symmetric $A$).</strong>
  The gradient is $\nabla f=A\mathbf x$ (Lesson 6.3). The Hessian is the Jacobian of the gradient:
  $H=\dfrac{\partial(A\mathbf x)}{\partial\mathbf x}=A.$ So the Hessian of a quadratic is the matrix $A$ itself —
  constant curvature everywhere.</p>
  <hr class="soft">
  <p><strong>The chain rule with Jacobians.</strong> For $\mathbf y=\mathbf f(\mathbf u)$, $\mathbf u=\mathbf g(\mathbf x)$,
  a small input change $\delta\mathbf x$ produces $\delta\mathbf u\approx J_{\mathbf g}\,\delta\mathbf x$, then
  $\delta\mathbf y\approx J_{\mathbf f}\,\delta\mathbf u=J_{\mathbf f}J_{\mathbf g}\,\delta\mathbf x.$ So
  $J_{\mathbf f\circ\mathbf g}=J_{\mathbf f}J_{\mathbf g}$ — composition of maps is multiplication of their local linear
  approximations. $\blacksquare$ This product, taken transpose and right-to-left, is backprop.</p>`,

  code: [
    { label: "Numerical Jacobian; Jacobian of a linear map is A", src: String.raw`
import numpy as np

def jacobian(f, x, h=1e-6):
    x = np.asarray(x, float); fx = f(x); J = np.zeros((fx.size, x.size))
    for j in range(x.size):
        e = np.zeros_like(x); e[j] = h
        J[:, j] = (f(x+e) - f(x-e)) / (2*h)
    return J

A = np.array([[2., 1.], [0., 3.], [1., -1.]])
f = lambda x: A @ x
print("Jacobian of Ax:\n", np.round(jacobian(f, [1., 1.]), 4))   # equals A
print("equals A?", np.allclose(jacobian(f, [1., 1.]), A))
` },
    { label: "Hessian of a quadratic is its matrix", src: String.raw`
import numpy as np

A = np.array([[2., 1.],
              [1., 4.]])                 # symmetric
f = lambda x: 0.5 * x @ A @ x

def hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i]=h; ej = np.zeros(n); ej[j]=h
            H[i,j] = (f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*h*h)
    return H

print("numerical Hessian:\n", np.round(hessian(f, np.array([0.3,-0.5])), 4))
print("equals A?", np.allclose(hessian(f, np.array([0.3,-0.5])), A, atol=1e-3))
` }
  ],

  keyPoints: [
    "Jacobian $J_{ij}=\\partial f_i/\\partial x_j$: the best local linear map of a vector function.",
    "Jacobians compose by multiplication: $J_{f\\circ g}=J_f J_g$ — the matrix form of the chain rule.",
    "Hessian $H_{ij}=\\partial^2 f/\\partial x_i\\partial x_j$ is the Jacobian of the gradient — local curvature.",
    "The Hessian is symmetric (mixed partials commute); its eigenvalues are curvatures (Track 5).",
    "Jacobian of $A\\mathbf x$ is $A$; Hessian of $\\tfrac12\\mathbf x^\\top A\\mathbf x$ is $A$."
  ],

  commonMistakes: [
    { wrong: "Mixing up the Jacobian's orientation ($m\\times n$ vs $n\\times m$).", why: "It's outputs-by-inputs: $m$ rows (one per output $f_i$), $n$ columns (one per input $x_j$). Transposing it breaks the chain-rule multiplication." },
    { wrong: "Forgetting the Hessian is symmetric.", why: "$\\partial^2 f/\\partial x\\partial y=\\partial^2 f/\\partial y\\partial x$ for smooth $f$. Exploiting symmetry halves the work and is required for using the spectral theorem on it." },
    { wrong: "Forming the full Hessian for large models.", why: "It's $n\\times n$ in the parameter count — billions squared is impossible. Practical methods use Hessian-vector products or PSD approximations (Gauss–Newton $J^\\top J$, K-FAC)." }
  ],

  quiz: [
    { q: "For $\\mathbf f:\\mathbb{R}^3\\to\\mathbb{R}^2$, the Jacobian has shape…", options: ["$2\\times3$", "$3\\times2$", "$3\\times3$", "$2\\times2$"], answer: 0,
      explain: "Outputs × inputs $=2\\times3$ (rows = outputs $f_1,f_2$; columns = inputs $x_1,x_2,x_3$)." },
    { q: "The Hessian of $f(x,y)=x^2+3y^2$ is…", options: ["$\\begin{bmatrix}2&0\\\\0&6\\end{bmatrix}$", "$\\begin{bmatrix}2&3\\\\3&6\\end{bmatrix}$", "$\\begin{bmatrix}2x&0\\\\0&6y\\end{bmatrix}$", "$[2,6]$"], answer: 0,
      explain: "$\\partial^2_x=2$, $\\partial^2_y=6$, mixed $=0$: a constant diagonal Hessian." },
    { q: "$J_{f\\circ g}$ equals…", options: ["$J_f(g(x))\\,J_g(x)$", "$J_f+J_g$", "$J_g\\,J_f$", "$J_f(x)\\,J_g(x)$"], answer: 0,
      explain: "Chain rule for maps: multiply the outer Jacobian (evaluated at $g(x)$) by the inner one." },
    { q: "A critical point has Hessian eigenvalues $\\{2,5\\}$. It is a…", options: ["local minimum", "saddle", "local maximum", "flat region"], answer: 0,
      explain: "All positive ⇒ positive definite ⇒ curves up in every direction ⇒ minimum (Track 5.3)." },
    { q: "Why is the Hessian symmetric?", options: ["mixed partials commute (Clairaut)", "it equals its own inverse", "it is always diagonal", "by definition of gradient"], answer: 0,
      explain: "For smooth $f$, $\\partial^2 f/\\partial x_i\\partial x_j=\\partial^2 f/\\partial x_j\\partial x_i$, making $H$ symmetric." }
  ],

  practice: [
    { level: "easy", prompt: "Give the Jacobian of $\\mathbf f(x,y)=[xy,\\;x+y]$.", solution: "$J=\\begin{bmatrix}\\partial(xy)/\\partial x & \\partial(xy)/\\partial y\\\\ \\partial(x+y)/\\partial x & \\partial(x+y)/\\partial y\\end{bmatrix}=\\begin{bmatrix}y & x\\\\ 1 & 1\\end{bmatrix}$." },
    { level: "med", prompt: "Compute the Hessian of $f(x,y)=x^2y$.", solution: "Gradient $[2xy,\\,x^2]$. Then $\\partial^2_x=2y$, $\\partial^2_y=0$, $\\partial_x\\partial_y=2x$. $H=\\begin{bmatrix}2y & 2x\\\\ 2x & 0\\end{bmatrix}$ (symmetric)." },
    { level: "med", prompt: "At $(1,1)$, classify the critical-point type of $f(x,y)=x^2y$ from its Hessian (note: is $(1,1)$ even critical?).", solution: "$\\nabla f=[2xy,x^2]=[2,1]\\neq\\mathbf0$ at $(1,1)$, so it's not a critical point — the curvature classification only applies where $\\nabla f=\\mathbf0$ (e.g. along $x=0$). A good reminder to check the gradient is zero before reading the Hessian." },
    { level: "hard", prompt: "AI task: why do second-order optimizers avoid forming the Hessian, and what do they use instead?", solution: "For a model with $n$ parameters the Hessian is $n\\times n$ — for $n=10^9$ that's $10^{18}$ entries, impossible to store or invert. Methods instead use <em>Hessian–vector products</em> $H\\mathbf v$ (computable in one extra backward pass via the identity $H\\mathbf v=\\nabla(\\nabla f\\cdot\\mathbf v)$, never forming $H$), or cheap PSD approximations: Gauss–Newton/Fisher ($J^\\top J$, always PSD), diagonal approximations (Adam's per-coordinate $v_t$ is a crude diagonal curvature estimate), or block/Kronecker factorizations (K-FAC). They get curvature's benefit without its $O(n^2)$ cost." }
  ],

  deepDive: String.raw`<p><strong>Jacobian-vector vs vector-Jacobian products: the two flavors of autodiff.</strong></p>
  <p>The chain rule $J_{f_L}\cdots J_{f_1}$ can be evaluated against a vector in two directions, and the choice <em>is</em>
  forward- vs reverse-mode autodiff. A <strong>Jacobian-vector product</strong> (JVP), $J\mathbf v$, pushes a direction
  $\mathbf v$ <em>forward</em> through the network — cheap when there are few inputs, giving directional derivatives.
  A <strong>vector-Jacobian product</strong> (VJP), $\mathbf v^\top J$, pulls a cotangent <em>backward</em> — cheap when
  there's one output, which is exactly the loss. Backprop is a sequence of VJPs; each layer only needs to know how to map
  an upstream gradient to a downstream one, never to materialize its full Jacobian.</p>
  <p>This is why a framework asks you to implement, per operation, a <code>backward</code> that takes the output's
  gradient and returns the input's — that function is the VJP. The full Jacobian of a $1000\to1000$ layer is a million
  numbers, but the VJP is just a $1000$-vector in and out. Understanding that the Hessian, too, is reachable as a VJP of
  the gradient ($H\mathbf v=\nabla(\nabla f\cdot\mathbf v)$) is what makes scalable second-order methods possible. The
  Jacobian and Hessian are the objects; JVP and VJP are how we touch them without ever building them — the bridge from
  this lesson to the autograd engine you'll write in Track 13.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["6.5"] = {
  subtitle: "Approximate any function near a point with a line, then a parabola — where GD and Newton come from.",

  aiMoment: String.raw`<p>Gradient descent isn't a heuristic — it falls out of a <strong>first-order Taylor expansion</strong>
  of the loss. Newton's method comes from the <strong>second-order</strong> one. Trust-region methods, learning-rate
  choices, and convergence proofs all reason about how well a local polynomial model matches the true loss. Taylor
  expansion is the lens through which optimization sees the landscape one neighborhood at a time.</p>`,

  plainEnglish: String.raw`<p>A <strong>Taylor expansion</strong> rebuilds a function near a point from its derivatives:
  the value, plus the slope times how far you step, plus half the curvature times the step squared, and so on. Keep one
  term and you get the tangent line; keep two and you get the best-fitting parabola.</p>`,

  intuition: String.raw`<p>Near a point, a smooth curve looks like its tangent line; look a little wider and a parabola
  (using the curvature) hugs it even better. Each extra Taylor term corrects the approximation over a larger
  neighborhood.</p>
  <figure class="figure">
  <svg viewBox="0 0 280 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A curve with its tangent line and quadratic approximation">
    <polyline points="40,110 60,92 80,75 100,62 120,53 140,50 160,53 180,62 200,75 220,92 240,110" fill="none" stroke="#4f46e5" stroke-width="2.4"/>
    <polyline points="55,97 70,83 90,68 110,56 130,49 150,46 170,48" fill="none" stroke="#d97706" stroke-width="1.8"/>
    <line x1="55" y1="91" x2="140" y2="34" stroke="#0d9488" stroke-width="1.8" stroke-dasharray="5 4"/>
    <circle cx="90" cy="68" r="3.5" fill="#dc2626"/>
    <text x="76" y="64" font-size="11" fill="#dc2626" font-family="sans-serif">a</text>
    <text x="120" y="26" font-size="10" fill="#0d9488" font-family="sans-serif">1st order (tangent)</text>
    <text x="90" y="116" font-size="10" fill="#d97706" font-family="sans-serif">2nd order (parabola)</text>
  </svg>
  <figcaption>The tangent matches slope; the parabola also matches curvature, hugging the curve over a wider range.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For a smooth scalar function (1-D), around $x$ with step $h$:</p>
  $$f(x+h)=f(x)+f'(x)\,h+\tfrac12 f''(x)\,h^2+\cdots$$
  <p>For several variables, with step $\boldsymbol\Delta$, gradient $\nabla f$, and Hessian $H$:</p>
  $$f(\mathbf x+\boldsymbol\Delta)\approx f(\mathbf x)+\nabla f(\mathbf x)^\top\boldsymbol\Delta+\tfrac12\boldsymbol\Delta^\top H\,\boldsymbol\Delta.$$
  <p>The first two terms are the linear (tangent) model; adding the third gives the quadratic model. Plain English:
  value + slope·step + ½·curvature·step².</p>`,

  derivation: String.raw`<p><strong>Gradient descent from the first-order model.</strong></p>
  <p><strong>Step 1 — linearize.</strong> Near $\mathbf x$, $f(\mathbf x+\boldsymbol\Delta)\approx f(\mathbf x)+\nabla f^\top\boldsymbol\Delta$.</p>
  <p><strong>Step 2 — pick the most-decreasing direction.</strong> For a fixed step length, $\nabla f^\top\boldsymbol\Delta$
  is most negative when $\boldsymbol\Delta$ points opposite the gradient (dot product minimized at $180°$, Lesson 6.3).</p>
  <p><strong>Step 3 — take a step.</strong> Set $\boldsymbol\Delta=-\eta\,\nabla f$ for a step size $\eta\gt0$:
  $\mathbf x\leftarrow\mathbf x-\eta\nabla f.\;\blacksquare$ That's gradient descent — the learning rate $\eta$ is just how
  far you trust the linear model.</p>
  <hr class="soft">
  <p><strong>Newton's method from the second-order model.</strong> Minimize the quadratic
  $m(\boldsymbol\Delta)=f+\nabla f^\top\boldsymbol\Delta+\tfrac12\boldsymbol\Delta^\top H\boldsymbol\Delta$ exactly. Set
  its gradient to zero: $\nabla_{\boldsymbol\Delta}m=\nabla f+H\boldsymbol\Delta=\mathbf 0\Rightarrow\boldsymbol\Delta=-H^{-1}\nabla f.$
  Plain English: GD guesses a step size; Newton uses the curvature $H$ to compute the exact step that reaches the bottom
  of the local parabola — landing in one shot if the function really is quadratic.</p>`,

  code: [
    { label: "1st- and 2nd-order Taylor approximations", src: String.raw`
import numpy as np, matplotlib.pyplot as plt

f, fp, fpp = np.exp, np.exp, np.exp     # f=e^x, all derivatives e^x
a = 0.0
x = np.linspace(-2, 2, 200)
lin  = f(a) + fp(a)*(x-a)                # tangent line
quad = lin + 0.5*fpp(a)*(x-a)**2         # add curvature

plt.figure(figsize=(5,4))
plt.plot(x, f(x), label="e^x", lw=2)
plt.plot(x, lin,  "--", label="1st order")
plt.plot(x, quad, ":",  label="2nd order")
plt.scatter([a],[f(a)], color="red"); plt.legend(); plt.ylim(-1,7)
plt.title("Taylor at x=0"); plt.tight_layout(); plt.show()
` },
    { label: "GD step vs Newton step on a quadratic", src: String.raw`
import numpy as np

A = np.array([[3., 0.],[0., 1.]]); b = np.array([6., 2.])
f    = lambda x: 0.5*x@A@x - b@x         # minimum at x* = A^{-1} b
grad = lambda x: A@x - b
x0 = np.array([0., 0.])

x_gd = x0 - 0.1*grad(x0)                 # one gradient-descent step (eta=0.1)
x_nt = x0 - np.linalg.solve(A, grad(x0)) # one Newton step = -H^{-1} grad
print("after 1 GD step    :", np.round(x_gd, 3))
print("after 1 Newton step:", np.round(x_nt, 3), " (exact minimum A^-1 b)")
print("true minimum       :", np.round(np.linalg.solve(A, b), 3))
` }
  ],

  keyPoints: [
    "Taylor: $f(x+h)=f(x)+f'(x)h+\\tfrac12 f''(x)h^2+\\cdots$ — value, slope, curvature.",
    "Multivariate: $f(\\mathbf x+\\boldsymbol\\Delta)\\approx f+\\nabla f^\\top\\boldsymbol\\Delta+\\tfrac12\\boldsymbol\\Delta^\\top H\\boldsymbol\\Delta$.",
    "Gradient descent = minimizing the first-order (linear) model: $\\boldsymbol\\Delta=-\\eta\\nabla f$.",
    "Newton's method = minimizing the second-order model: $\\boldsymbol\\Delta=-H^{-1}\\nabla f$.",
    "Newton solves a quadratic in one step; GD's step size $\\eta$ is a trust radius for the linear model."
  ],

  commonMistakes: [
    { wrong: "Expanding around the wrong point.", why: "Taylor is local: $f(x)+f'(x)h$ approximates near $x$, and the error grows with $|h|$. Far from the expansion point the model is meaningless — which is why a too-large learning rate diverges." },
    { wrong: "Trusting Newton's step when $H$ isn't positive definite.", why: "$-H^{-1}\\nabla f$ heads to the parabola's stationary point — a <em>maximum</em> or saddle if $H$ has negative eigenvalues (Track 5.3). Near saddles, raw Newton can climb. Damping or trust regions fix it." },
    { wrong: "Believing more Taylor terms always help globally.", why: "Higher-order terms improve the <em>local</em> fit but can diverge outside the radius of convergence. Optimization uses just first or second order and controls the step instead." }
  ],

  quiz: [
    { q: "The 1st-order Taylor approximation of $f$ at $x$ is the…", options: ["tangent line", "parabola", "constant $f(x)$", "secant"], answer: 0,
      explain: "Value plus slope·step is exactly the tangent line; adding curvature gives the parabola." },
    { q: "Gradient descent's update $\\mathbf x\\leftarrow\\mathbf x-\\eta\\nabla f$ comes from minimizing…", options: ["the first-order model", "the second-order model", "the Hessian", "the exact loss"], answer: 0,
      explain: "Steepest descent on the linear (first-order) Taylor model gives the $-\\eta\\nabla f$ step." },
    { q: "Newton's step is…", options: ["$-H^{-1}\\nabla f$", "$-\\eta\\nabla f$", "$-H\\nabla f$", "$-\\nabla f/\\|\\nabla f\\|$"], answer: 0,
      explain: "Setting the quadratic model's gradient $\\nabla f+H\\boldsymbol\\Delta$ to zero gives $\\boldsymbol\\Delta=-H^{-1}\\nabla f$." },
    { q: "On an exactly quadratic loss, Newton's method reaches the minimum in…", options: ["one step", "two steps", "many steps", "never"], answer: 0,
      explain: "The second-order model equals the true function, so its exact minimizer is the true minimum — one step." },
    { q: "The 2nd-order Taylor of $e^x$ at 0 is…", options: ["$1+x+\\tfrac{x^2}{2}$", "$1+x$", "$x+\\tfrac{x^2}{2}$", "$1+x^2$"], answer: 0,
      explain: "$f(0)=1$, $f'(0)=1$, $f''(0)=1$, so $1+x+\\tfrac12x^2$." }
  ],

  practice: [
    { level: "easy", prompt: "Write the 1st-order Taylor approximation of $\\sin x$ at $x=0$.", solution: "$\\sin0=0$, $\\cos0=1$, so $\\sin x\\approx x$ near 0 (the small-angle approximation)." },
    { level: "med", prompt: "Give the 2nd-order Taylor of $f(x)=\\ln(1+x)$ at $0$.", solution: "$f(0)=0$, $f'(x)=\\frac1{1+x}\\Rightarrow f'(0)=1$, $f''(x)=-\\frac1{(1+x)^2}\\Rightarrow f''(0)=-1$. So $\\ln(1+x)\\approx x-\\tfrac{x^2}{2}$." },
    { level: "med", prompt: "For $f(\\mathbf x)=\\tfrac12\\mathbf x^\\top A\\mathbf x-\\mathbf b^\\top\\mathbf x$, what does one Newton step from any point give?", solution: "$\\nabla f=A\\mathbf x-\\mathbf b$, $H=A$. Newton: $\\boldsymbol\\Delta=-A^{-1}(A\\mathbf x-\\mathbf b)=-\\mathbf x+A^{-1}\\mathbf b$, landing exactly at $\\mathbf x^*=A^{-1}\\mathbf b$ regardless of the start — the quadratic is solved in one step." },
    { level: "hard", prompt: "AI task: relate the learning rate to the second-order Taylor term, and explain why $\\eta\\lt 2/\\lambda_{\\max}(H)$ is needed for stability on a quadratic.", solution: "On a quadratic with Hessian $H$, GD updates each eigen-coordinate as $z_i\\leftarrow(1-\\eta\\lambda_i)z_i$ (Track 5.2). This shrinks (converges) only if $|1-\\eta\\lambda_i|\\lt1$, i.e. $0\\lt\\eta\\lambda_i\\lt2$, for <em>every</em> eigenvalue — the binding constraint is the largest, giving $\\eta\\lt2/\\lambda_{\\max}$. Above it, the steepest direction's update amplifies ($|1-\\eta\\lambda_{\\max}|\\gt1$) and the loss diverges. The first-order model ignores curvature, so $\\eta$ must stay small enough that the neglected $\\tfrac12\\boldsymbol\\Delta^\\top H\\boldsymbol\\Delta$ term doesn't bite — which is exactly the role of the learning rate." }
  ],

  deepDive: String.raw`<p><strong>The whole optimizer zoo, read off the Taylor expansion.</strong></p>
  <p>Every first-order optimizer is a way of using the linear model $f+\nabla f^\top\boldsymbol\Delta$ while guarding
  against its inaccuracy. Plain <strong>gradient descent</strong> steps $-\eta\nabla f$ and trusts a fixed radius $\eta$.
  <strong>Momentum</strong> reuses past gradients to push through the linear model's noise. <strong>Adam</strong> rescales
  each coordinate by a running estimate of curvature ($\sqrt{v_t}$ is a crude diagonal $\sqrt H$), effectively making the
  bowl rounder so a single $\eta$ works across axes. <strong>Newton</strong> and quasi-Newton (L-BFGS) go second-order,
  using $H$ (or an approximation) to set the step exactly. They differ only in how much of the Taylor expansion they
  model and how they trust it.</p>
  <p>This is why the learning rate is the single most important hyperparameter: it's the trust radius of a first-order
  approximation to a curved loss. Too large and you step outside where the linear model holds — the ignored
  $\tfrac12\boldsymbol\Delta^\top H\boldsymbol\Delta$ curvature term flings you uphill and training diverges. Too small
  and you crawl. The whole art of optimization (Track 8) is choosing how much of the local Taylor picture to model and
  how far to trust it — and it all starts from these two or three terms.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["6.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 65 minutes.",

  intro: String.raw`<p>This exam spans all of Track 6: limits and the derivative, the differentiation rules and chain
  rule, the gradient, Jacobian and Hessian, and Taylor expansion (with GD and Newton). <strong>Give yourself 65
  minutes</strong>, produce each answer before checking, and score with the rubric. About half are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "Differentiate $f(x)=5x^3-2x+\\ln x$.",
      solution: "$f'(x)=15x^2-2+\\tfrac1x$ (power rule on each term; $\\ln x\\to1/x$)." },
    { level: "easy", prompt: "Use the chain rule: $\\frac{d}{dx}\\,e^{3x^2}$.",
      solution: "Outer $e^{u}$, inner $u=3x^2$: $e^{3x^2}\\cdot6x=6x\\,e^{3x^2}$." },
    { level: "med", prompt: "For the sigmoid $\\sigma$, compute $\\sigma'(0)$ and explain why deep sigmoid stacks vanish.",
      solution: "$\\sigma'(x)=\\sigma(1-\\sigma)$; $\\sigma(0)=0.5$, so $\\sigma'(0)=0.25$ (its maximum). Each layer multiplies a factor $\\le0.25$ into the backprop product, so through many layers the gradient shrinks like $0.25^{\\text{depth}}\\to0$ — vanishing gradients." },
    { level: "med", prompt: "Find $\\nabla f$ for $f(x,y)=x^2+xy+y^3$ at $(1,2)$.",
      solution: "$\\partial_x=2x+y=4$; $\\partial_y=x+3y^2=1+12=13$. So $\\nabla f(1,2)=[4,13]$." },
    { level: "med", prompt: "Give the Jacobian of $\\mathbf f(x,y)=[x^2y,\\;\\sin x]$.",
      solution: "$J=\\begin{bmatrix}2xy & x^2\\\\ \\cos x & 0\\end{bmatrix}$ (rows are the two outputs, columns the inputs $x,y$)." },
    { level: "med", prompt: "Compute the Hessian of $f(x,y)=x^2+4xy+y^2$ and classify its origin critical point.",
      solution: "$H=\\begin{bmatrix}2&4\\\\4&2\\end{bmatrix}$. Eigenvalues: trace 4, det $4-16=-12$, so $\\lambda=6,-2$ (mixed signs) → indefinite → saddle." },
    { level: "hard", prompt: "Give the 2nd-order Taylor expansion of $f(x)=\\cos x$ at $x=0$, and approximate $\\cos(0.2)$.",
      solution: "$f(0)=1$, $f'(0)=-\\sin0=0$, $f''(0)=-\\cos0=-1$: $\\cos x\\approx1-\\tfrac{x^2}{2}$. So $\\cos(0.2)\\approx1-0.02=0.98$ (true $\\approx0.98007$)." },
    { level: "hard", prompt: "On $f(\\mathbf x)=\\tfrac12\\mathbf x^\\top A\\mathbf x$ with $A=\\mathrm{diag}(4,1)$ from $\\mathbf x_0=[1,1]$, give one GD step ($\\eta=0.1$) and one Newton step.",
      solution: "$\\nabla f=A\\mathbf x_0=[4,1]$. GD: $\\mathbf x_0-0.1[4,1]=[0.6,0.9]$. Newton: $\\mathbf x_0-A^{-1}[4,1]=[1,1]-[1,1]=[0,0]$ — the exact minimum in one step (quadratic)." },
    { level: "hard", prompt: "Explain why backprop multiplies Jacobians right-to-left (from the loss), in cost terms.",
      solution: "The gradient is $J_1^\\top\\cdots J_L^\\top\\nabla_{\\mathbf y}\\mathcal L$. Right-to-left starts from the scalar loss's gradient (a vector) and does vector-Jacobian products — matrix×vector at each step, $O(\\text{layer size})$. Left-to-right would multiply full Jacobian matrices (matrix×matrix), far costlier. With many parameters and one scalar output, reverse mode computes all gradients in about one forward pass's cost." },
    { level: "hard", prompt: "AI task: why must the learning rate satisfy $\\eta<2/\\lambda_{\\max}(H)$ on a quadratic loss?",
      solution: "In the Hessian's eigenbasis, GD updates each coordinate as $z_i\\leftarrow(1-\\eta\\lambda_i)z_i$. Convergence needs $|1-\\eta\\lambda_i|<1$ for all $i$, i.e. $0<\\eta\\lambda_i<2$. The largest eigenvalue binds, giving $\\eta<2/\\lambda_{\\max}$; beyond it the steepest direction's iterate grows and the loss diverges — the curvature term the first-order model ignores, made precise." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Derivatives, the chain rule, and curvature are solid — you can read backprop and optimizers as math. On to Track 7.</li>
    <li><strong>7–8:</strong> Strong. Revisit the Jacobian chain rule or the GD/Newton derivations if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive the chain rule and the GD-from-Taylor step; redo Lessons 6.2 and 6.5.</li>
    <li><strong>Below 5:</strong> Rework the track — calculus is the engine of training; everything in optimization and probability leans on it.</li>
  </ul>`
};
