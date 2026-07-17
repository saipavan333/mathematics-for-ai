/* ============================================================
   TRACK 17 — Neural-Network & Transformer Math
   The math inside modern architectures: nonlinearity, convolution,
   normalization, and attention — assembled into a transformer block.
   Every demo runs and is verified.
   ============================================================ */
window.LESSON_CONTENT ||= {};

/* ------------------------------------------------------------------ 17.1 */
window.LESSON_CONTENT["17.1"] = {
  subtitle: "Why a deep network needs nonlinearities to be more than a line — and how each activation's slope decides whether gradients survive the trip back through the layers.",

  aiMoment: String.raw`Every layer of every neural network ends in a nonlinearity — ReLU in a ResNet, GELU in a transformer, sigmoid in a gate. It's easy to treat these as a detail, but they carry two loads at once: without them a hundred-layer network is algebraically identical to a <em>single</em> linear layer, and <em>with</em> them, the derivative of the activation is what lets — or stops — gradients from flowing back through depth. The vanishing-gradient problem, which stalled deep learning for years, and its fixes (ReLU, careful init, residuals, normalization) are all a story about these slopes.`,

  plainEnglish: String.raw`Stacking straight lines only ever gives you a straight line. To bend and fold space enough to separate real data, you need a <em>kink</em> between layers — that's the activation function. But there's a catch on the way back: during backprop, the gradient gets multiplied by each activation's slope. If those slopes are small, the product shrinks toward zero over many layers (the signal vanishes); if they're around 1, it survives. So the activation you pick shapes both what the network can represent and whether it can be trained.`,

  intuition: String.raw`Compose two linear maps and you get $W_2(W_1x)=(W_2W_1)x$ — one linear map, no matter how many you stack. The nonlinearity between them is the whole point: it's what turns a stack of matrices into a flexible function approximator. Going backward, the chain rule multiplies the gradient by each layer's activation slope. A sigmoid's slope peaks at $0.25$, so through 40 layers the gradient is scaled by $\sim0.25^{40}\approx10^{-24}$ — gone. ReLU's slope is exactly 1 wherever it's active, so it passes the gradient through untouched (though 0 where it's off — a 'dead' unit). That single fact is why ReLU displaced sigmoids in deep nets.`,

  formalism: String.raw`Common activations and their derivatives (the derivative is what multiplies the backward gradient):
$$\operatorname{ReLU}(z)=\max(0,z),\ \operatorname{ReLU}'=\mathbf 1[z>0];\quad \tanh'(z)=1-\tanh^2 z\le1;\quad \sigma'(z)=\sigma(z)(1-\sigma(z))\le\tfrac14.$$
$$\operatorname{GELU}(z)=z\,\Phi(z)\approx\tfrac12 z\big(1+\tanh[\sqrt{2/\pi}(z+0.0447z^3)]\big).$$
Through an $L$-layer stack the input-to-loss gradient carries a factor $\prod_{\ell=1}^{L}\operatorname{diag}(\phi'(a_\ell))\,W_\ell^\top$. If the typical activation slope times weight scale is below 1, the product <strong>vanishes</strong>; above 1, it <strong>explodes</strong>. Keeping it near 1 (via ReLU-like slopes, careful initialization, normalization, and residuals) is what makes depth trainable.`,

  derivation: String.raw`<strong>Two facts, one from the forward pass and one from the backward.</strong>
<ol>
<li><strong>Linearity collapses.</strong> If every layer is $x\mapsto W_\ell x$ (no activation), then $L$ layers give $W_L\cdots W_1\,x=Wx$ for a single matrix $W=W_L\cdots W_1$. Depth adds no expressive power — you must insert a nonlinearity $\phi$ so that $\phi(W_1x)$ can't be absorbed into the next matrix.</li>
<li><strong>Backprop multiplies slopes.</strong> With $a_\ell=W_\ell h_{\ell-1}$ and $h_\ell=\phi(a_\ell)$, the chain rule gives $\dfrac{\partial L}{\partial h_{\ell-1}}=W_\ell^\top\big(\phi'(a_\ell)\odot\dfrac{\partial L}{\partial h_\ell}\big)$. Each step back scales the gradient by $\phi'(a_\ell)$ (and $W_\ell^\top$).</li>
<li><strong>Why sigmoids vanish.</strong> $\sigma'\le\tfrac14$, so $L$ layers multiply the gradient by at most $(\tfrac14)^L$ — exponential decay in depth. Early layers get essentially no learning signal. $\tanh'\le1$ is a bit better but still $<1$ in its saturated tails.</li>
<li><strong>Why ReLU helps.</strong> $\operatorname{ReLU}'\in\{0,1\}$: on active paths it's exactly 1, so it neither shrinks nor grows the gradient — the product doesn't decay from the activation. The cost is that inactive units pass 0 (a 'dead ReLU'); leaky/GELU variants soften that. The code shows the $0.25^L$ collapse next to ReLU's flat 1.</li>
</ol>`,

  code: [
    { label: "Nonlinearity is essential; activation slopes control gradient flow",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# 1) Without a nonlinearity, stacked linear layers collapse to ONE linear map.
W1 = rng.standard_normal((5,4)); W2 = rng.standard_normal((3,5)); x = rng.standard_normal(4)
print("W2(W1 x) == (W2 W1) x :", np.allclose(W2@(W1@x), (W2@W1)@x), " -> depth without nonlinearity buys nothing")

# 2) Activations and their derivatives (the derivative multiplies the backward gradient).
z = np.linspace(-4, 4, 200); sig = 1/(1+np.exp(-z))
gelu = 0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z + 0.044715*z**3)))

# 3) Vanishing gradient: the gradient scales by the PRODUCT of activation slopes over depth.
L = np.arange(1, 41)
sig_factor = 0.25**L        # sigmoid' peaks at 0.25 -> shrinks fast
relu_factor = 1.0**L        # relu' = 1 on active units -> preserved
print(f"gradient factor at depth 40:  sigmoid {sig_factor[-1]:.1e}   ReLU {relu_factor[-1]:.0f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for n,f in [('ReLU',np.maximum(z,0)),('tanh',np.tanh(z)),('sigmoid',sig),('GELU',gelu)]: ax[0].plot(z,f,label=n)
ax[0].set_title('activations'); ax[0].legend(fontsize=8)
for n,f in [("ReLU'",(z>0).astype(float)),("tanh'",1-np.tanh(z)**2),("sigmoid'",sig*(1-sig))]: ax[1].plot(z,f,label=n)
ax[1].set_title("derivatives (gradient flow)"); ax[1].legend(fontsize=8)
ax[2].semilogy(L, sig_factor, label='sigmoid  0.25^L'); ax[2].semilogy(L, relu_factor, label='ReLU  1^L')
ax[2].set_xlabel('depth L'); ax[2].set_ylabel('gradient factor'); ax[2].set_title('why deep sigmoid nets vanish'); ax[2].legend(fontsize=8)
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 200" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="n1a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <text x="180" y="24" text-anchor="middle" font-weight="700" fill="#8f2233">linear only → collapses</text>
  <rect x="20" y="40" width="40" height="34" rx="5" fill="#eef2f7" stroke="#33415c"/><text x="40" y="62" text-anchor="middle">x</text>
  <rect x="92" y="40" width="46" height="34" rx="5" fill="#fbeaed" stroke="#c1121f"/><text x="115" y="62" text-anchor="middle" fill="#8f2233">W₁</text>
  <rect x="170" y="40" width="46" height="34" rx="5" fill="#fbeaed" stroke="#c1121f"/><text x="193" y="62" text-anchor="middle" fill="#8f2233">W₂</text>
  <line x1="60" y1="57" x2="90" y2="57" stroke="#6b7a99" marker-end="url(#n1a)"/><line x1="138" y1="57" x2="168" y2="57" stroke="#6b7a99" marker-end="url(#n1a)"/>
  <text x="180" y="96" text-anchor="middle" fill="#5a6b8c" font-size="10.5">= (W₂W₁)·x  — one linear layer</text>
  <text x="530" y="24" text-anchor="middle" font-weight="700" fill="#245030">with nonlinearity → genuinely deep</text>
  <rect x="352" y="40" width="40" height="34" rx="5" fill="#eef2f7" stroke="#33415c"/><text x="372" y="62" text-anchor="middle">x</text>
  <rect x="418" y="40" width="42" height="34" rx="5" fill="#fbeaed" stroke="#c1121f"/><text x="439" y="62" text-anchor="middle" fill="#8f2233">W₁</text>
  <rect x="486" y="40" width="42" height="34" rx="5" fill="#fff5e6" stroke="#b8860b"/><text x="507" y="62" text-anchor="middle" fill="#7a5b00">σ</text>
  <rect x="554" y="40" width="42" height="34" rx="5" fill="#fbeaed" stroke="#c1121f"/><text x="575" y="62" text-anchor="middle" fill="#8f2233">W₂</text>
  <line x1="392" y1="57" x2="416" y2="57" stroke="#6b7a99" marker-end="url(#n1a)"/><line x1="460" y1="57" x2="484" y2="57" stroke="#6b7a99" marker-end="url(#n1a)"/><line x1="528" y1="57" x2="552" y2="57" stroke="#6b7a99" marker-end="url(#n1a)"/>
  <text x="500" y="96" text-anchor="middle" fill="#5a6b8c" font-size="10.5">the kink can't be absorbed → real depth</text>
  <line x1="30" y1="120" x2="690" y2="120" stroke="#e0e6ef"/>
  <text x="360" y="146" text-anchor="middle" fill="#1f2a44">backward, the gradient is multiplied by the activation's slope φ′(z) at every layer</text>
  <text x="360" y="168" text-anchor="middle" fill="#5a6b8c" font-size="10.5">sigmoid′ ≤ 0.25 → vanishes with depth&#160;&#160;·&#160;&#160;ReLU′ = 1 (active) → preserved</text>
</svg>`,

  keyPoints: [
    String.raw`Stacked linear layers collapse to one linear map ($W_L\cdots W_1$); the <strong>nonlinearity</strong> between layers is what makes depth expressive.`,
    String.raw`Backprop multiplies the gradient by each activation's slope $\phi'(z)$ (and the weights). Slopes $<1$ compound into <strong>vanishing</strong> gradients; $>1$ into <strong>exploding</strong> ones.`,
    String.raw`Sigmoid ($\sigma'\le0.25$) vanishes fast with depth; ReLU ($\text{ReLU}'\in\{0,1\}$) preserves the gradient on active units — the core reason ReLU replaced sigmoids in deep nets.`,
    String.raw`The cost of ReLU is <strong>dead units</strong> (slope 0 for $z<0$); Leaky-ReLU and GELU keep a small/smooth slope there. Transformers favor GELU.`,
    String.raw`Keeping the per-layer gradient factor near 1 — via ReLU-like slopes, good initialization, normalization, and residuals — is what makes very deep networks trainable.`
  ],

  commonMistakes: [
    { wrong: "Using sigmoid/tanh activations throughout a very deep feed-forward network.",
      why: String.raw`Their sub-1 slopes multiply to near-zero over many layers, so early layers barely learn (the vanishing-gradient problem). Use ReLU/GELU for hidden layers; reserve sigmoid/tanh for gates and bounded outputs.` },
    { wrong: "Ignoring dead ReLUs after a bad initialization or too-high learning rate.",
      why: String.raw`A unit whose pre-activation is always negative outputs 0 and has slope 0 forever — it never recovers. Symptoms: a chunk of units permanently at 0. Fixes: careful init, lower LR, or leaky/GELU activations that keep a nonzero slope.` },
    { wrong: "Thinking more layers always means more capacity.",
      why: String.raw`Only true <em>with</em> nonlinearities and healthy gradient flow. Without them, extra linear layers add zero capacity; and even with them, poor gradient flow means deep layers don't train. Depth is useful only when the signal can reach every layer.` }
  ],

  quiz: [
    { q: "Stacking two linear layers W₂(W₁x) with no activation is equivalent to…",
      options: ["A single linear layer (W₂W₁)x", "A quadratic function", "A deeper, more expressive model", "A nonlinear function"], answer: 0,
      explain: String.raw`Matrix products compose: $W_2(W_1x)=(W_2W_1)x$. Depth without nonlinearity adds no expressive power.` },
    { q: "During backprop, the gradient at each layer is multiplied by…",
      options: ["The activation's derivative φ'(z) (and the weights)", "The loss value", "The learning rate", "The input norm"], answer: 0,
      explain: String.raw`The chain rule brings a factor $\phi'(a_\ell)$ (elementwise) and $W_\ell^\top$ at every layer — the origin of vanishing/exploding gradients.` },
    { q: "Why do deep sigmoid networks suffer vanishing gradients?",
      options: ["σ'(z) ≤ 0.25, so the product over L layers shrinks like 0.25^L",
                "σ has no derivative", "σ outputs are too large", "σ is nonlinear"], answer: 0,
      explain: String.raw`The maximum slope is $\tfrac14$, so through depth the gradient is scaled by at most $(\tfrac14)^L$ — exponential decay.` },
    { q: "The 'dead ReLU' problem is that a unit…",
      options: ["Stuck with negative pre-activations outputs 0 with slope 0, so it never updates",
                "Fires too often", "Has slope greater than 1", "Outputs negative values"], answer: 0,
      explain: String.raw`For $z<0$, ReLU is flat (slope 0), so a permanently-negative unit gets no gradient and can't recover — a motivation for leaky/GELU activations.` },
    { q: "Which activation do modern transformers typically use in their MLP blocks?",
      options: ["GELU", "sigmoid", "step function", "linear"], answer: 0,
      explain: String.raw`GELU (a smooth, differentiable-everywhere gate) is standard in transformer feed-forward blocks; it avoids dead units while keeping ReLU-like behavior for large $z$.` }
  ],

  practice: [
    { level: "easy", prompt: "Compute the maximum value of σ'(z) = σ(z)(1−σ(z)) and where it occurs.",
      solution: String.raw`Maximized at $\sigma(z)=\tfrac12$ (i.e. $z=0$), giving $\tfrac12\cdot\tfrac12=\tfrac14$. That $0.25$ ceiling is exactly what makes deep sigmoid stacks vanish.` },
    { level: "easy", prompt: "For a 20-layer network with typical activation slope 0.5 and unit weight scale, estimate the gradient shrinkage from input to output.",
      solution: String.raw`Roughly $0.5^{20}\approx 10^{-6}$ — the input-layer gradient is about a millionth of the output-layer gradient, so early layers learn ~$10^6\times$ slower. This is why activation choice and normalization matter so much at depth.` },
    { level: "med", prompt: "Show that GELU ≈ ReLU for large |z| but is smooth at 0, and explain why smoothness helps.",
      solution: String.raw`As $z\to+\infty$, $\Phi(z)\to1$ so $\operatorname{GELU}(z)\to z$; as $z\to-\infty$, $\Phi(z)\to0$ so $\operatorname{GELU}(z)\to0$ — matching ReLU. Near 0 it curves smoothly (nonzero slope for small negative $z$), so units aren't hard-killed and the gradient is defined everywhere, which stabilizes training.` },
    { level: "med", prompt: "Explain how a residual connection h_out = h_in + F(h_in) changes the backward gradient and fights vanishing.",
      solution: String.raw`$\partial h_{\text{out}}/\partial h_{\text{in}}=I+\partial F/\partial h_{\text{in}}$: the identity term passes the gradient through <em>unattenuated</em>, so even if $F$'s Jacobian is tiny, the gradient still flows. Residuals turn a product of small factors into a sum that includes 1 — the key trick that made 100+ layer nets (ResNets, transformers) trainable.` },
    { level: "hard", prompt: "Derive the condition on weight-initialization variance that keeps activation variance stable across a ReLU layer (He initialization).",
      solution: String.raw`For $a=Wx$ with $x$ having $n$ inputs of variance $v$, $\operatorname{Var}(a)=n\,\sigma_W^2\,v$. ReLU zeros half the mass, halving output variance, so to keep variance constant you need $n\sigma_W^2\cdot\tfrac12=1$, i.e. $\sigma_W^2=2/n$ — He initialization. It's the forward-pass analogue of keeping the backward gradient factor near 1.` },
    { level: "hard", prompt: "Why does gradient explosion (factor > 1) call for a different fix (clipping) than vanishing?",
      solution: String.raw`Exploding gradients blow up in magnitude, causing NaNs and unstable steps; the standard fix is <em>gradient clipping</em> (rescale the gradient if its norm exceeds a threshold), which caps step size without changing direction. Vanishing can't be clipped away — there's nothing to cap — so it needs architectural fixes (ReLU, normalization, residuals) that keep the factor from shrinking in the first place. RNNs, which reuse the same weights over time, hit both and use both remedies.` }
  ],

  deepDive: String.raw`<p><strong>Vanishing gradients shaped modern architecture.</strong> Nearly every structural innovation of the deep-learning era is, in part, a fix for gradient flow: ReLU (slope-1 activations), careful initialization (Xavier/He, keeping variance ~1), batch/layer normalization (rescaling activations mid-network), and residual connections (an identity path the gradient rides through). Transformers use all four at once — which is exactly why they train stably at enormous depth.</p>
<p><strong>The universal approximation caveat.</strong> A single hidden layer with a nonlinearity can approximate any continuous function (Cybenko/Hornik) — but possibly needing exponentially many units. Depth buys the same expressiveness far more parameter-efficiently, <em>provided</em> gradients can reach the deep layers. So universal approximation says nonlinearity makes networks expressive; the gradient-flow story says architecture makes that expressiveness <em>trainable</em>.</p>
<p><strong>Activations are still evolving.</strong> Beyond ReLU/GELU, gated units like SwiGLU (a learned gate times a linear branch) now appear in many large language models' feed-forward blocks, trading a few extra parameters for better performance. The throughline is unchanged: pick nonlinearities whose slopes keep the forward activations and backward gradients well-scaled across depth.</p>`
};

/* ------------------------------------------------------------------ 17.2 */
window.LESSON_CONTENT["17.2"] = {
  subtitle: "A convolution is just a linear layer with two constraints — look only at a local patch, and reuse the same weights everywhere. That's the whole vision inductive bias, and it runs as a matrix multiply.",

  aiMoment: String.raw`Convolutional networks powered the first deep-learning vision revolution (AlexNet, ResNet) and still sit inside the U-Nets that denoise diffusion images. A convolution encodes two beliefs about images: nearby pixels matter together (<em>locality</em>) and a feature is the same wherever it appears (<em>translation equivariance via weight sharing</em>). Those beliefs are what let CNNs learn from far less data than a dense net — and underneath, every convolution is executed as a plain matrix multiply, which is why GPUs run them so fast.`,

  plainEnglish: String.raw`Slide a small filter across the image; at each position, multiply the patch under it by the filter and sum. The <em>same</em> filter is used at every position, so if it learns to detect an edge, it detects that edge anywhere in the image. Compared to a dense layer (where every output pixel connects to every input pixel), a convolution connects each output to only a small neighborhood and shares one tiny set of weights — vastly fewer parameters for the same job.`,

  intuition: String.raw`Think of a dense layer as a giant matrix connecting all inputs to all outputs. A convolution is that matrix with almost all entries forced to zero (each output only sees a local patch) and the surviving entries tied together (the same kernel weights repeat at every position). Those two constraints — <strong>sparsity</strong> and <strong>weight sharing</strong> — are the CNN's inductive bias. And because it's still linear, you can literally rewrite it as one matrix multiply: unfold every patch into a row (<code>im2col</code>), stack them, and multiply by the flattened kernel.`,

  formalism: String.raw`A 2-D valid convolution of image $I$ with a $k\times k$ kernel $K$ is
$$O[i,j]=\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} I[i+u,\ j+v]\,K[u,v].$$
Written as a matrix multiply via <strong>im2col</strong>: form a matrix $P$ whose rows are the flattened $k\times k$ patches, then $\operatorname{vec}(O)=P\,\operatorname{vec}(K)$. Parameter counts tell the story: a conv layer has $k^2\,c_{\text{in}}\,c_{\text{out}}$ weights (independent of image size), whereas a dense layer connecting an $N$-pixel image to $M$ outputs has $N\cdot M$. Convolution is also <strong>translation-equivariant</strong>: shifting the input shifts the output, $O(\text{shift}(I))=\text{shift}(O(I))$.`,

  derivation: String.raw`<strong>Convolution is a constrained linear layer.</strong>
<ol>
<li><strong>It's linear.</strong> Each output $O[i,j]$ is a fixed weighted sum of inputs, so the whole map is $\operatorname{vec}(O)=A\,\operatorname{vec}(I)$ for some matrix $A$ — a linear layer.</li>
<li><strong>Sparse + shared.</strong> $A$'s row for output $(i,j)$ is nonzero only on the $k\times k$ input pixels in that patch (<em>locality</em> → mostly zeros), and every row uses the <em>same</em> $k^2$ values $K$ just shifted to a new position (<em>weight sharing</em> → the free parameters are only $K$, not all of $A$).</li>
<li><strong>im2col makes the matmul explicit.</strong> Instead of building the huge sparse $A$, gather the patches: row $r$ of $P$ is the flattened patch at output position $r$. Then $P\,\operatorname{vec}(K)$ computes every output as a dot product — a dense multiply on a much smaller matrix. This is exactly how cuDNN and friends run convolutions.</li>
<li><strong>Equivariance falls out of sharing.</strong> Because the same $K$ is applied at every location, translating the input by $\Delta$ just relabels which patch feeds which output — the output translates by $\Delta$ too. That built-in symmetry (not learned, imposed by the architecture) is why CNNs generalize across position with far less data. The code verifies $\text{direct conv}=\text{im2col matmul}$ exactly.</li>
</ol>`,

  code: [
    { label: "Convolution == im2col matrix multiply, and an edge detector",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(1)

def conv2d(img, k):                                   # direct convolution (valid)
    kh, kw = k.shape; H, W = img.shape; out = np.zeros((H-kh+1, W-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]): out[i,j] = (img[i:i+kh, j:j+kw]*k).sum()
    return out

def im2col(img, kh, kw):                              # unfold every patch into a row
    H, W = img.shape
    return np.array([img[i:i+kh, j:j+kw].ravel()
                     for i in range(H-kh+1) for j in range(W-kw+1)])

img = rng.standard_normal((8,8))
kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], float)      # Sobel vertical-edge detector
direct = conv2d(img, kernel)
matmul = (im2col(img,3,3) @ kernel.ravel()).reshape(direct.shape)   # convolution AS a matrix multiply
print("convolution == im2col matrix multiply:", np.allclose(direct, matmul))
print(f"parameters: 3x3 conv = 9 (shared across all positions) vs a dense 64->36 layer = {64*36}")

# apply the edge detector to a little shape
im = np.zeros((16,16)); im[4:12, 6:10] = 1.0
plt.figure(figsize=(8,3.4))
plt.subplot(1,2,1); plt.imshow(im, cmap='gray'); plt.title('input'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(np.abs(conv2d(im, kernel)), cmap='magma'); plt.title('vertical edges (conv output)'); plt.axis('off')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 210" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="c2a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <text x="120" y="22" text-anchor="middle" font-weight="700" fill="#1f2a44">slide one shared kernel</text>
  <g stroke="#9fb2c9" fill="#eef4fb">
   <rect x="30" y="36" width="20" height="20"/><rect x="50" y="36" width="20" height="20"/><rect x="70" y="36" width="20" height="20"/><rect x="90" y="36" width="20" height="20"/><rect x="110" y="36" width="20" height="20"/>
   <rect x="30" y="56" width="20" height="20"/><rect x="50" y="56" width="20" height="20"/><rect x="70" y="56" width="20" height="20"/><rect x="90" y="56" width="20" height="20"/><rect x="110" y="56" width="20" height="20"/>
   <rect x="30" y="76" width="20" height="20"/><rect x="50" y="76" width="20" height="20"/><rect x="70" y="76" width="20" height="20"/><rect x="90" y="76" width="20" height="20"/><rect x="110" y="76" width="20" height="20"/>
   <rect x="30" y="96" width="20" height="20"/><rect x="50" y="96" width="20" height="20"/><rect x="70" y="96" width="20" height="20"/><rect x="90" y="96" width="20" height="20"/><rect x="110" y="96" width="20" height="20"/></g>
  <rect x="30" y="36" width="60" height="60" fill="none" stroke="#c1121f" stroke-width="2.5"/>
  <text x="80" y="132" text-anchor="middle" fill="#5a6b8c" font-size="10">3×3 patch → 1 output</text>
  <line x1="150" y1="66" x2="196" y2="66" stroke="#6b7a99" marker-end="url(#c2a)"/>
  <text x="430" y="22" text-anchor="middle" font-weight="700" fill="#245030">= a matrix multiply (im2col)</text>
  <rect x="210" y="40" width="150" height="70" rx="5" fill="#e7f0e8" stroke="#3a7d44"/><text x="285" y="70" text-anchor="middle" fill="#245030">patches P</text><text x="285" y="86" text-anchor="middle" fill="#3a7d44" font-size="10">(each row = a patch)</text>
  <text x="378" y="80" text-anchor="middle" fill="#8f2233" font-size="16">×</text>
  <rect x="396" y="52" width="66" height="46" rx="5" fill="#fff5e6" stroke="#b8860b"/><text x="429" y="79" text-anchor="middle" fill="#7a5b00">vec(K)</text>
  <text x="486" y="80" text-anchor="middle" fill="#6b7a99" font-size="15">=</text>
  <rect x="506" y="52" width="80" height="46" rx="5" fill="#dbe8fb" stroke="#2a6f97"/><text x="546" y="79" text-anchor="middle" fill="#123a5a">output O</text>
  <text x="360" y="150" text-anchor="middle" fill="#1f2a44">locality + weight sharing = a sparse, tied linear layer → few params, translation-equivariant</text>
  <text x="360" y="172" text-anchor="middle" fill="#5a6b8c" font-size="10.5">3×3 conv = 9 shared weights, independent of image size (a dense layer would need thousands)</text>
</svg>`,

  keyPoints: [
    String.raw`A convolution is a linear layer with two constraints: <strong>locality</strong> (each output sees only a small patch) and <strong>weight sharing</strong> (the same kernel is reused at every position).`,
    String.raw`It runs as a matrix multiply: <code>im2col</code> unfolds patches into rows, then $\operatorname{vec}(O)=P\,\operatorname{vec}(K)$ — exactly what GPU conv kernels do.`,
    String.raw`Parameter count is $k^2 c_{\text{in}} c_{\text{out}}$, independent of image size — orders of magnitude fewer than a dense layer, which is the CNN's data efficiency.`,
    String.raw`Weight sharing gives <strong>translation equivariance</strong>: shift the input, the output shifts the same way. That symmetry is built in, not learned.`,
    String.raw`A $1\times1$ convolution is a per-pixel dense layer over channels (pure channel mixing) — a common building block, and the bridge from conv thinking to attention.`
  ],

  commonMistakes: [
    { wrong: "Confusing convolution's parameter count with its compute cost.",
      why: String.raw`A conv has few (shared) <em>parameters</em> but does a lot of <em>work</em> — it applies them at every spatial position, so FLOPs scale with image size $\times k^2 c_{\text{in}} c_{\text{out}}$. Small model, big compute; don't equate the two.` },
    { wrong: "Assuming convolution is translation-invariant.",
      why: String.raw`It's translation-<em>equivariant</em>: shifting the input shifts the output. Invariance (same output regardless of shift) only comes after a pooling/global step that collapses spatial position. The distinction matters for tasks like localization.` },
    { wrong: "Forgetting padding/stride when reasoning about output size and receptive field.",
      why: String.raw`Valid convolution shrinks the map by $k-1$; padding preserves size; stride $s$ downsamples by $\sim s$. The <em>receptive field</em> (how much input an output sees) grows with depth and stride — the quantity that determines how much context a deep conv unit integrates.` }
  ],

  quiz: [
    { q: "A convolution differs from a dense layer because it is…",
      options: ["Local (small patch) and weight-shared (same kernel everywhere)",
                "Nonlinear", "Always larger", "Randomly connected"], answer: 0,
      explain: String.raw`Those two constraints — locality and weight sharing — turn a dense layer into a convolution with far fewer, tied parameters.` },
    { q: "How many weights does a 3×3 conv with 16 input and 32 output channels have?",
      options: ["3·3·16·32 = 4608", "16·32 = 512", "3·3 = 9", "depends on image size"], answer: 0,
      explain: String.raw`$k^2 c_{\text{in}} c_{\text{out}}=9\cdot16\cdot32=4608$ — independent of the image's height and width.` },
    { q: "The im2col trick lets you compute a convolution as…",
      options: ["A matrix multiply of unfolded patches by the flattened kernel",
                "An FFT only", "A sort", "A nonlinear map"], answer: 0,
      explain: String.raw`Stack patches as rows of $P$; then $P\,\operatorname{vec}(K)$ is the convolution — a dense matmul, which is why GPUs do it fast.` },
    { q: "Weight sharing gives a convolution which property?",
      options: ["Translation equivariance (shift in → shift out)", "Rotation invariance",
                "Scale invariance", "Permutation invariance"], answer: 0,
      explain: String.raw`Applying the same kernel everywhere means a shifted input yields a shifted output — equivariance, the CNN's core symmetry.` },
    { q: "A 1×1 convolution is equivalent to…",
      options: ["A dense layer applied independently at each pixel (channel mixing)",
                "Blurring the image", "A pooling operation", "Doing nothing"], answer: 0,
      explain: String.raw`With a $1\times1$ spatial extent, each output pixel is a learned linear combination of the input channels at that pixel — pure channel mixing.` }
  ],

  practice: [
    { level: "easy", prompt: "Apply a horizontal-edge Sobel kernel [[1,2,1],[0,0,0],[-1,-2,-1]] to the shape in the code. How does the output differ from the vertical detector?",
      solution: String.raw`It responds to horizontal edges (top/bottom of the rectangle) instead of vertical ones (left/right). Rotating the kernel rotates which oriented edges light up — a concrete look at what a single conv filter 'detects'.` },
    { level: "easy", prompt: "With 'same' padding, what output size does a k×k convolution produce, and why is padding useful in deep nets?",
      solution: String.raw`Same padding (pad $\lfloor k/2\rfloor$) keeps the output the same $H\times W$ as the input. Without it, each conv shrinks the map by $k-1$, so a deep stack would erode the spatial size to nothing — padding lets you go deep while preserving resolution.` },
    { level: "med", prompt: "Compute the receptive field of a unit after three stacked 3×3 convolutions (stride 1). Why do small kernels stack instead of using one big kernel?",
      solution: String.raw`Each 3×3 adds 2 to the receptive field: $3\to5\to7$. Three 3×3s see a $7\times7$ region using $3\cdot9=27$ weights, versus a single $7\times7$ using $49$ — same receptive field, fewer parameters, and two extra nonlinearities. That's the VGG insight.` },
    { level: "med", prompt: "Express the conv's backward pass: given ∂L/∂O, what are ∂L/∂K and ∂L/∂I in convolution terms?",
      solution: String.raw`$\partial L/\partial K$ is the convolution (correlation) of the input with the upstream gradient (summed over positions — because $K$ is shared, its gradient accumulates over all patches, exactly the '$X^\top\delta$' pattern from 14.4). $\partial L/\partial I$ is a <em>transposed</em>/full convolution of $\partial L/\partial O$ with the flipped kernel. Both are convolutions, so backprop through conv is more convs.` },
    { level: "hard", prompt: "A depthwise-separable convolution splits a k×k conv into a depthwise conv + a 1×1 conv. Compute the parameter saving vs a standard conv.",
      solution: String.raw`Standard: $k^2 c_{\text{in}} c_{\text{out}}$. Separable: depthwise $k^2 c_{\text{in}}$ (one filter per input channel) + pointwise $c_{\text{in}} c_{\text{out}}$. Ratio $\approx \frac{1}{c_{\text{out}}}+\frac{1}{k^2}$ — for $k=3, c_{\text{out}}=256$ about an $8$–$9\times$ reduction. This factorization is what makes MobileNets efficient on phones.` },
    { level: "hard", prompt: "Contrast the inductive biases of convolution and self-attention (next lessons). When does each win?",
      solution: String.raw`Convolution hard-codes locality and translation equivariance — great when those hold (natural images) and when data is limited, since the bias substitutes for data. Self-attention imposes almost no spatial prior; every position can attend to every other, so it models long-range/global relations but needs more data (or added positional bias) to learn what conv assumes for free. Vision transformers work because at scale the data replaces the missing bias; hybrids (conv stem + attention) try to get both.` }
  ],

  deepDive: String.raw`<p><strong>Convolution as prior knowledge.</strong> The reason CNNs beat dense nets on images isn't more capacity — it's <em>less</em>. By hard-wiring locality and translation equivariance, a conv layer refuses to consider the vast space of functions that treat distant pixels as related or a cat in the corner as different from a cat in the center. That refusal is a prior, and priors substitute for data (echoing the Bayesian track): the more correct structure you build in, the less you must learn.</p>
<p><strong>Everything is a matmul.</strong> im2col turning convolution into a dense multiply is one instance of a general truth in deep learning: the heavy operations — dense layers, convolutions, attention — all reduce to matrix multiplies, which is exactly what GPUs/TPUs are built to do. Understanding an architecture often means understanding <em>which</em> matmul it is and what structure (sparsity, sharing, low rank) it imposes on the weight matrix.</p>
<p><strong>From convolution to attention.</strong> A convolution mixes information within a fixed local window with fixed (learned) weights. Self-attention — the rest of this track — mixes information across the <em>whole</em> input with weights <em>computed from the content itself</em>. Seen this way, attention is a convolution whose receptive field is global and whose kernel is data-dependent, which is why it captures long-range structure that a fixed local kernel cannot.</p>`
};

/* ------------------------------------------------------------------ 17.3 */
window.LESSON_CONTENT["17.3"] = {
  subtitle: "As activations flow through layers they drift in scale and destabilize training. Normalization re-centers and re-scales them at every layer — and which axis you normalize over is what separates BatchNorm from the LayerNorm every transformer uses.",

  aiMoment: String.raw`Open any transformer implementation and you'll find a <code>LayerNorm</code> on almost every line — it's applied before (or after) attention and before the MLP in every block. Normalization was the trick that let networks go truly deep: BatchNorm made CNNs train faster and more stably, and LayerNorm did the same for transformers and RNNs. The difference between them is a single choice — which axis you average over — and that choice is why one is used for images and the other for sequences.`,

  plainEnglish: String.raw`Inside a deep network, the numbers coming out of each layer can drift to wildly different scales, which makes the next layer's job unstable and slows learning. Normalization fixes this by, at each layer, shifting the activations to mean 0 and scaling them to variance 1 — then letting the network learn its own preferred scale and shift back (two parameters, $\gamma$ and $\beta$). BatchNorm computes those statistics across the batch; LayerNorm computes them across each example's own features.`,

  intuition: String.raw`Well-conditioned inputs (mean 0, variance 1) keep gradients from exploding or vanishing and smooth the loss surface, so the optimizer can take bigger, safer steps. The catch is <em>which numbers you pool to compute the mean and variance</em>. <strong>BatchNorm</strong> pools down each feature across the whole batch — powerful, but it couples examples together and needs a decent batch size, and behaves differently at train vs. test time. <strong>LayerNorm</strong> pools across each example's own features — completely independent of the batch, so it works with batch size 1, variable-length sequences, and at inference unchanged. That batch-independence is exactly why transformers use it.`,

  formalism: String.raw`Both normalize then apply a learnable affine map:
$$\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}},\qquad y=\gamma\odot\hat x+\beta.$$
The only difference is the axis of $\mu,\sigma^2$. For a batch $X\in\mathbb R^{N\times D}$:
$$\textbf{BatchNorm: }\ \mu_j=\tfrac1N\textstyle\sum_n X_{nj}\ \ (\text{per feature, over the batch}),\qquad \textbf{LayerNorm: }\ \mu_n=\tfrac1D\textstyle\sum_j X_{nj}\ \ (\text{per example, over features}).$$
LayerNorm's input gradient (the normalization couples all $D$ features, so it isn't just elementwise):
$$\frac{\partial L}{\partial x}=\frac{1}{\sqrt{\sigma^2+\varepsilon}}\Big(\,\widehat{g}-\overline{\widehat g}-\hat x\,\overline{\widehat g\odot\hat x}\,\Big),\qquad \widehat g=\frac{\partial L}{\partial y}\odot\gamma,$$
where the bars are means over the feature axis.`,

  derivation: String.raw`<strong>Where the LayerNorm backward formula comes from.</strong>
<ol>
<li><strong>Forward.</strong> Per row: $\mu=\frac1D\sum_j x_j$, $\sigma^2=\frac1D\sum_j(x_j-\mu)^2$, $\hat x=(x-\mu)/\sqrt{\sigma^2+\varepsilon}$, $y=\gamma\hat x+\beta$.</li>
<li><strong>Affine params are easy.</strong> $\partial L/\partial\gamma=\sum_n \frac{\partial L}{\partial y}\odot\hat x$ and $\partial L/\partial\beta=\sum_n\frac{\partial L}{\partial y}$ (sum over the batch, by the shape rule).</li>
<li><strong>Through the normalization.</strong> Let $\widehat g=\frac{\partial L}{\partial y}\odot\gamma=\partial L/\partial\hat x$. Because $\mu$ and $\sigma$ each depend on <em>all</em> features of the row, $x_j$ influences every $\hat x_k$, so you must differentiate through the mean and the variance. Carrying those two dependencies through gives the two correction terms: $\partial L/\partial x=\frac{1}{\sqrt{\sigma^2+\varepsilon}}(\widehat g-\overline{\widehat g}-\hat x\,\overline{\widehat g\odot\hat x})$. The $-\overline{\widehat g}$ term is the mean's contribution; the $-\hat x\,\overline{\widehat g\odot\hat x}$ term is the variance's.</li>
<li><strong>Verify, don't trust.</strong> This is exactly the kind of coupled backward pass that's easy to get wrong — so the code gradient-checks it against finite differences (it matches to $10^{-5}$). Getting normalization's backward right is a rite of passage in implementing a framework.</li>
</ol>`,

  code: [
    { label: "LayerNorm: normalize each row, then a gradient-checked backward pass",
      src: String.raw`import numpy as np
rng = np.random.default_rng(2)

D = 6; eps = 1e-5
x = rng.standard_normal((4, D))*3 + 5           # 4 vectors, off-center and differently scaled
gamma = np.ones(D); beta = np.zeros(D)

def layernorm(x, gamma, beta):                  # normalize each ROW over its D features
    mu = x.mean(-1, keepdims=True); xc = x - mu
    var = (xc**2).mean(-1, keepdims=True); std = np.sqrt(var + eps)
    xn = xc/std
    return gamma*xn + beta, (xn, std)

y, (xn, std) = layernorm(x, gamma, beta)
print("row means  before:", np.round(x.mean(1), 2), " after:", np.round(y.mean(1), 2))
print("row stds   before:", np.round(x.std(1), 2),  " after:", np.round(y.std(1), 2))

# LayerNorm's backward is subtle (mean & variance couple all features) -- verify it.
T = rng.standard_normal((4, D)); dy = 2*(y - T)     # gradient of a squared-error loss
ghat = dy*gamma
dx = (ghat - ghat.mean(-1,keepdims=True) - xn*(ghat*xn).mean(-1,keepdims=True))/std
def loss(x): yy,_ = layernorm(x, gamma, beta); return ((yy - T)**2).sum()
num = np.zeros_like(x); h = 1e-6
for i in np.ndindex(x.shape):
    o=x[i]; x[i]=o+h; lp=loss(x); x[i]=o-h; lm=loss(x); x[i]=o; num[i]=(lp-lm)/(2*h)
print("LayerNorm dx gradient check:", np.allclose(dx, num, atol=1e-5))` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 214" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <text x="175" y="20" text-anchor="middle" font-weight="700" fill="#1f2a44">BatchNorm — down a column</text>
  <g stroke="#c7d0df" fill="#f4f7fb">
   <rect x="70" y="34" width="26" height="22"/><rect x="96" y="34" width="26" height="22"/><rect x="122" y="34" width="26" height="22"/><rect x="148" y="34" width="26" height="22"/><rect x="174" y="34" width="26" height="22"/><rect x="200" y="34" width="26" height="22"/>
   <rect x="70" y="56" width="26" height="22"/><rect x="96" y="56" width="26" height="22"/><rect x="122" y="56" width="26" height="22"/><rect x="148" y="56" width="26" height="22"/><rect x="174" y="56" width="26" height="22"/><rect x="200" y="56" width="26" height="22"/>
   <rect x="70" y="78" width="26" height="22"/><rect x="96" y="78" width="26" height="22"/><rect x="122" y="78" width="26" height="22"/><rect x="148" y="78" width="26" height="22"/><rect x="174" y="78" width="26" height="22"/><rect x="200" y="78" width="26" height="22"/>
   <rect x="70" y="100" width="26" height="22"/><rect x="96" y="100" width="26" height="22"/><rect x="122" y="100" width="26" height="22"/><rect x="148" y="100" width="26" height="22"/><rect x="174" y="100" width="26" height="22"/><rect x="200" y="100" width="26" height="22"/></g>
  <rect x="122" y="34" width="26" height="88" fill="#d1495b" fill-opacity="0.35" stroke="#c1121f" stroke-width="2"/>
  <text x="62" y="80" text-anchor="end" fill="#5a6b8c" font-size="9.5">batch ↓</text>
  <text x="148" y="136" text-anchor="middle" fill="#5a6b8c" font-size="9.5">features →</text>
  <text x="175" y="158" text-anchor="middle" fill="#8f2233" font-size="10">stats per feature, over the batch</text>
  <text x="545" y="20" text-anchor="middle" font-weight="700" fill="#1f2a44">LayerNorm — across a row</text>
  <g stroke="#c7d0df" fill="#f4f7fb">
   <rect x="440" y="34" width="26" height="22"/><rect x="466" y="34" width="26" height="22"/><rect x="492" y="34" width="26" height="22"/><rect x="518" y="34" width="26" height="22"/><rect x="544" y="34" width="26" height="22"/><rect x="570" y="34" width="26" height="22"/>
   <rect x="440" y="56" width="26" height="22"/><rect x="466" y="56" width="26" height="22"/><rect x="492" y="56" width="26" height="22"/><rect x="518" y="56" width="26" height="22"/><rect x="544" y="56" width="26" height="22"/><rect x="570" y="56" width="26" height="22"/>
   <rect x="440" y="78" width="26" height="22"/><rect x="466" y="78" width="26" height="22"/><rect x="492" y="78" width="26" height="22"/><rect x="518" y="78" width="26" height="22"/><rect x="544" y="78" width="26" height="22"/><rect x="570" y="78" width="26" height="22"/>
   <rect x="440" y="100" width="26" height="22"/><rect x="466" y="100" width="26" height="22"/><rect x="492" y="100" width="26" height="22"/><rect x="518" y="100" width="26" height="22"/><rect x="544" y="100" width="26" height="22"/><rect x="570" y="100" width="26" height="22"/></g>
  <rect x="440" y="56" width="156" height="22" fill="#3a7d44" fill-opacity="0.30" stroke="#2f7d4f" stroke-width="2"/>
  <text x="545" y="158" text-anchor="middle" fill="#245030" font-size="10">stats per example, over its features</text>
  <text x="360" y="186" text-anchor="middle" fill="#1f2a44">then rescale: y = γ·x̂ + β  (learnable)</text>
  <text x="360" y="206" text-anchor="middle" fill="#5a6b8c" font-size="10.5">LayerNorm is batch-independent → transformers use it (any batch size / sequence length)</text>
</svg>`,

  keyPoints: [
    String.raw`Normalization sets activations to mean 0, variance 1 ($\hat x=(x-\mu)/\sqrt{\sigma^2+\varepsilon}$), then applies a learnable affine $y=\gamma\hat x+\beta$ so the network keeps control of scale.`,
    String.raw`<strong>BatchNorm</strong> pools per feature over the batch; <strong>LayerNorm</strong> pools per example over its features. Same formula, different axis.`,
    String.raw`LayerNorm is <strong>batch-independent</strong> (identical at train and test, works at batch size 1 and any sequence length), which is why transformers and RNNs use it.`,
    String.raw`BatchNorm couples examples and needs train/eval running statistics; it shines in CNNs with large batches but is awkward for sequences and tiny batches.`,
    String.raw`Normalization's backward pass is <em>coupled</em> (the mean/variance depend on all features), so it's a classic place for gradient bugs — always finite-difference-check it.`
  ],

  commonMistakes: [
    { wrong: "Forgetting that BatchNorm behaves differently in training vs. evaluation.",
      why: String.raw`In training BN uses the current batch's statistics; at eval it must use <em>running</em> (moving-average) statistics collected during training. Leaving the module in train mode at inference (or with a tiny/size-1 batch) gives wrong, unstable outputs — a very common bug. LayerNorm sidesteps this entirely.` },
    { wrong: "Using BatchNorm with very small batches.",
      why: String.raw`Batch statistics from 1–2 examples are noisy estimates of mean/variance, so BN becomes unreliable. Small-batch or per-example settings (transformers, RL, tiny GPUs) favor LayerNorm/GroupNorm, whose statistics don't depend on batch size.` },
    { wrong: "Dropping the learnable γ, β and assuming normalization alone is enough.",
      why: String.raw`Forcing every layer's output to exactly mean 0/variance 1 removes representational freedom. The affine $\gamma,\beta$ let the network undo or reshape the normalization when useful (even recovering the identity), which is why they're part of the layer, not optional.` }
  ],

  quiz: [
    { q: "The difference between BatchNorm and LayerNorm is…",
      options: ["Which axis the mean/variance are computed over", "One is nonlinear",
                "One has no parameters", "LayerNorm is only for images"], answer: 0,
      explain: String.raw`Same normalize-then-affine formula; BN pools over the batch (per feature), LN over the features (per example).` },
    { q: "Why do transformers use LayerNorm rather than BatchNorm?",
      options: ["It's batch-independent — same at train/test, works with any batch size and sequence length",
                "It's faster to compute", "It has no parameters", "It only works on images"], answer: 0,
      explain: String.raw`LN's per-example statistics don't depend on the batch, so it behaves identically regardless of batch size or sequence length — ideal for variable-length sequence models.` },
    { q: "After normalization x̂ = (x−μ)/√(σ²+ε), what do the learnable γ and β do?",
      options: ["Rescale and shift the normalized values so the network controls the output distribution",
                "Compute the mean", "Add noise", "Nothing — they're constants"], answer: 0,
      explain: String.raw`$y=\gamma\hat x+\beta$ lets the layer choose its own scale/shift (even recovering the identity), restoring the representational freedom normalization would otherwise remove.` },
    { q: "For an input batch of shape (N=32, D=512), over how many numbers does LayerNorm average to normalize one token/example?",
      options: ["512 (its own features)", "32 (the batch)", "32×512", "1"], answer: 0,
      explain: String.raw`LayerNorm computes each example's mean/variance over its own $D=512$ features — independent of the other 31 examples.` },
    { q: "BatchNorm at inference time should use…",
      options: ["Running (moving-average) statistics accumulated during training",
                "The single test example's statistics", "Zero mean and unit variance always", "The training set's raw data"], answer: 0,
      explain: String.raw`Eval-mode BN uses the running mean/variance estimated during training, so its output doesn't depend on the (possibly size-1) test batch.` }
  ],

  practice: [
    { level: "easy", prompt: "Verify from the code that after LayerNorm every row has mean ≈ 0 and std ≈ 1 (with γ=1, β=0). What breaks that if γ≠1?",
      solution: String.raw`With $\gamma=1,\beta=0$ the output rows are exactly the normalized $\hat x$ (mean 0, std 1). Setting $\gamma\ne1$ rescales each feature, so the row std is no longer 1 — by design, since $\gamma$ gives the network back its scale control.` },
    { level: "easy", prompt: "Why is the ε inside the square root necessary?",
      solution: String.raw`If a row (or feature) has (near-)zero variance, $1/\sqrt{\sigma^2}$ blows up. Adding a small $\varepsilon$ keeps the denominator bounded and the layer numerically stable — the same guard as in Adam.` },
    { level: "med", prompt: "Contrast pre-LN and post-LN transformer blocks (LN before vs after the sublayer). Why did the field move to pre-LN?",
      solution: String.raw`Post-LN ($\text{LN}(x+\text{Sublayer}(x))$) puts the norm on the residual path, which can shrink the identity signal and makes deep training need careful warmup. Pre-LN ($x+\text{Sublayer}(\text{LN}(x))$) keeps a clean identity residual, so gradients flow through un-normalized — training is far more stable at depth, which is why most modern LLMs use pre-LN (often with RMSNorm).` },
    { level: "med", prompt: "RMSNorm drops the mean-subtraction and divides only by the root-mean-square. Write it and say what it saves.",
      solution: String.raw`$y=\gamma\odot x/\sqrt{\frac1D\sum_j x_j^2+\varepsilon}$ — no $\mu$, no $\beta$. It skips centering and one reduction, is a bit cheaper, and empirically matches LayerNorm; several large LLMs adopt it for that small efficiency win.` },
    { level: "hard", prompt: "The original BatchNorm paper credited 'reducing internal covariate shift,' but later work (Santurkar et al.) disputed that. What's the better-supported explanation?",
      solution: String.raw`Experiments showed BN helps even when covariate shift is artificially reintroduced; the better-supported story is that normalization <em>smooths the loss landscape</em> (improves its Lipschitz/β-smoothness), letting you use larger learning rates and get more stable, faster convergence. The takeaway: BN's benefit is optimization-geometry, not distribution-matching per se.` },
    { level: "hard", prompt: "Derive why BatchNorm couples examples in a batch and what that implies for its gradient.",
      solution: String.raw`BN's $\mu_j,\sigma_j$ are functions of <em>all</em> $N$ examples' feature $j$, so each output $y_{nj}$ depends on every other example's $x_{\cdot j}$. Hence $\partial L/\partial x_{mj}$ has terms flowing through the shared $\mu_j,\sigma_j$ — the backward pass mixes gradients across the batch. That coupling is why BN's effective regularization depends on batch composition, why it interacts badly with tiny batches, and why per-example norms (LN/GroupNorm) are preferred when independence matters.` }
  ],

  deepDive: String.raw`<p><strong>A family of norms, chosen by axis.</strong> BatchNorm (over the batch), LayerNorm (over features), InstanceNorm (over spatial positions per channel), and GroupNorm (over channel groups) are all the same normalize-then-affine operation with a different pooling axis. The right choice depends on what should be treated as 'a distribution to standardize': images with big batches → BN; sequences / small batches → LN; style transfer → IN. Knowing they're one operation with a knob demystifies the zoo.</p>
<p><strong>Normalization made depth practical.</strong> Together with residual connections, normalization is why 100-plus-layer networks train at all: it keeps activation scales bounded layer to layer so gradients neither explode nor vanish, and it flattens the loss surface enough to allow large learning rates. Every architectural fix from the activations lesson (init, residuals, norms) is pulling in the same direction — keep the forward and backward signals well-scaled through depth.</p>
<p><strong>The trend is toward simpler, batch-free norms.</strong> Modern LLMs overwhelmingly use pre-LN placement with RMSNorm — batch-independent, centering-free, cheap, and stable at scale. The arc from BatchNorm (2015, batch-coupled) to RMSNorm (batch-free, minimal) tracks the field's move from vision/CNNs to sequence/transformers, where independence from the batch and rock-solid deep-training stability matter most.</p>`
};

/* ------------------------------------------------------------------ 17.4 */
window.LESSON_CONTENT["17.4"] = {
  subtitle: "The one operation behind every transformer: each token asks a question, matches it against every other token, and gathers a weighted blend of their content. Plus the tiny √d that keeps it from breaking.",

  aiMoment: String.raw`Scaled dot-product attention is the beating heart of the transformer — the architecture behind GPT, BERT, Stable Diffusion's text encoder, and essentially every frontier model. Unlike a convolution's fixed local window, attention lets any token pull information directly from any other token, with weights <em>computed from the content itself</em>. That single mechanism — content-based routing across the whole sequence — is what "Attention Is All You Need" claimed, and it's why understanding this one formula unlocks modern AI.`,

  plainEnglish: String.raw`Give every token three roles. Its <strong>query</strong> is the question it's asking ("what am I looking for?"). Its <strong>key</strong> is the label it advertises ("here's what I am"). Its <strong>value</strong> is the content it offers. To update a token, compare its query against every token's key to get a relevance score, turn those scores into weights that sum to 1, and take the weighted average of everyone's values. A token thus gathers information from wherever it's relevant — a word at the start of a paragraph can directly influence one at the end.`,

  intuition: String.raw`A dot product measures similarity, so $q\cdot k$ scores how well a query matches a key. Softmax turns the row of scores into a probability distribution over keys (weights that sum to 1), and the output is that distribution applied to the values — a soft, content-addressed lookup. There's one landmine: for $d$-dimensional queries and keys, $q\cdot k$ has variance $\sim d$, so for large $d$ the scores get huge, softmax saturates onto a single key, and its gradient goes to zero. Dividing by $\sqrt d$ rescales the variance back to $\sim1$, keeping attention smooth and trainable — that's the whole reason for the mysterious $\sqrt d$.`,

  formalism: String.raw`<strong>Scaled dot-product attention</strong> for queries $Q\in\mathbb R^{n\times d}$, keys $K\in\mathbb R^{m\times d}$, values $V\in\mathbb R^{m\times d_v}$:
$$\operatorname{Attention}(Q,K,V)=\underbrace{\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big)}_{A\ \in\ \mathbb R^{n\times m},\ \text{rows sum to }1}\,V.$$
Each row of the attention matrix $A$ is a distribution over the $m$ keys; the output row is the corresponding weighted average of the values. In <strong>self-attention</strong>, $Q,K,V$ are all linear projections of the same input $X$: $Q=XW_Q,\ K=XW_K,\ V=XW_V$. The $\sqrt d$ appears because if $q_i,k_j$ have unit-variance entries, $\operatorname{Var}(q\cdot k)=d$, and $\operatorname{Var}(q\cdot k/\sqrt d)=1$.`,

  derivation: String.raw`<strong>Building attention, and why the √d is not optional.</strong>
<ol>
<li><strong>Score by similarity.</strong> The relevance of key $j$ to query $i$ is the dot product $s_{ij}=q_i\cdot k_j$ — large when the two vectors point the same way. Stacked, $S=QK^\top$.</li>
<li><strong>Normalize into weights.</strong> Apply softmax along each row: $A_{ij}=\dfrac{e^{s_{ij}}}{\sum_{j'}e^{s_{ij'}}}$, so each query's weights over keys are non-negative and sum to 1 — a convex combination.</li>
<li><strong>Gather values.</strong> The output for query $i$ is $\sum_j A_{ij}v_j$, i.e. $O=AV$: a weighted average of the values, weighted by how much each key matched.</li>
<li><strong>Why divide by √d.</strong> With independent, unit-variance entries, $q\cdot k=\sum_{l=1}^{d} q_l k_l$ has mean 0 and variance $d$ (a sum of $d$ unit-variance terms). So the logits fed to softmax have standard deviation $\sqrt d$ — for $d=64$ that's $\pm8$, enough to push softmax to a near one-hot output whose gradient (recall softmax's Jacobian $\text{diag}(p)-pp^\top$) is $\approx0$. Dividing by $\sqrt d$ restores unit variance, keeping the distribution soft and the gradients alive. The code shows the entropy collapse from $\sim1.3$ to $\sim0.02$ when you drop the scaling.</li>
</ol>`,

  code: [
    { label: "Scaled dot-product attention, and why the √d matters",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e/e.sum(-1, keepdims=True)

def attention(Q, K, V, scale=True):
    d = Q.shape[-1]
    scores = Q @ K.T / (np.sqrt(d) if scale else 1.0)     # (n_query, n_key) similarities
    A = softmax(scores)                                   # each row: a distribution over keys
    return A @ V, A                                       # output = weighted average of values

n, d = 5, 64
Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))
out, A = attention(Q, K, V)
print("output shape:", out.shape, " attention rows sum to 1:", np.allclose(A.sum(1), 1))

# WHY /sqrt(d): the dot products have variance ~ d, which saturates softmax.
print(f"var of QKᵀ entries ~ {np.var(Q@K.T):.0f}  (≈ d = {d});   after /√d ~ {np.var(Q@K.T/np.sqrt(d)):.2f}")
entropy = lambda A: -(A*np.log(A+1e-12)).sum(1).mean()
_, A_unscaled = attention(Q, K, V, scale=False)
print(f"attention entropy:  scaled {entropy(A):.2f}   unscaled {entropy(A_unscaled):.2f}   (unscaled ≈ one-hot → tiny gradients)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a,(M,t) in zip(ax, [(A,'scaled  softmax(QKᵀ/√d)'), (A_unscaled,'unscaled softmax(QKᵀ) — saturates')]):
    im = a.imshow(M, cmap='viridis', vmin=0, vmax=1); a.set_title(t, fontsize=9); a.set_xlabel('key'); a.set_ylabel('query')
fig.colorbar(im, ax=ax, fraction=.025); plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 214" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="a4a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <rect x="14" y="82" width="64" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="46" y="100" text-anchor="middle" fill="#1f2a44">X</text><text x="46" y="114" text-anchor="middle" fill="#5a6b8c" font-size="9">tokens</text>
  <rect x="118" y="34" width="70" height="34" rx="6" fill="#fbeaed" stroke="#c1121f"/><text x="153" y="56" text-anchor="middle" fill="#8f2233">Q = XW_Q</text>
  <rect x="118" y="84" width="70" height="34" rx="6" fill="#e7eff6" stroke="#2a6f97"/><text x="153" y="106" text-anchor="middle" fill="#1c4e70">K = XW_K</text>
  <rect x="118" y="134" width="70" height="34" rx="6" fill="#e7f0e8" stroke="#3a7d44"/><text x="153" y="156" text-anchor="middle" fill="#245030">V = XW_V</text>
  <line x1="78" y1="94" x2="116" y2="55" stroke="#6b7a99" marker-end="url(#a4a)"/><line x1="78" y1="102" x2="116" y2="101" stroke="#6b7a99" marker-end="url(#a4a)"/><line x1="78" y1="110" x2="116" y2="147" stroke="#6b7a99" marker-end="url(#a4a)"/>
  <rect x="222" y="56" width="104" height="44" rx="6" fill="#fff5e6" stroke="#b8860b"/><text x="274" y="74" text-anchor="middle" fill="#7a5b00" font-weight="700">Q·Kᵀ / √d</text><text x="274" y="90" text-anchor="middle" fill="#9a7b20" font-size="9">similarity scores</text>
  <rect x="360" y="56" width="110" height="44" rx="6" fill="#efeaf0" stroke="#6a51a3"/><text x="415" y="74" text-anchor="middle" fill="#4b2e83" font-weight="700">softmax → A</text><text x="415" y="90" text-anchor="middle" fill="#6a51a3" font-size="9">weights, rows sum to 1</text>
  <rect x="510" y="82" width="120" height="44" rx="6" fill="#dbe8fb" stroke="#2a6f97"/><text x="570" y="100" text-anchor="middle" fill="#123a5a" font-weight="700">out = A · V</text><text x="570" y="116" text-anchor="middle" fill="#2a6f97" font-size="9">weighted avg of values</text>
  <line x1="188" y1="51" x2="220" y2="70" stroke="#6b7a99" marker-end="url(#a4a)"/><line x1="188" y1="101" x2="220" y2="86" stroke="#6b7a99" marker-end="url(#a4a)"/>
  <line x1="326" y1="78" x2="358" y2="78" stroke="#6b7a99" marker-end="url(#a4a)"/>
  <line x1="470" y1="82" x2="508" y2="98" stroke="#6b7a99" marker-end="url(#a4a)"/>
  <line x1="188" y1="151" x2="508" y2="112" stroke="#3a7d44" stroke-dasharray="4 3" marker-end="url(#a4a)"/>
  <text x="360" y="194" text-anchor="middle" fill="#1f2a44">each query attends to every key; the output is a content-weighted blend of the values</text>
  <text x="360" y="210" text-anchor="middle" fill="#8f2233" font-size="10.5">√d keeps the scores' variance ≈ 1 so softmax stays soft (doesn't collapse to one key)</text>
</svg>`,

  keyPoints: [
    String.raw`Attention $=\operatorname{softmax}(QK^\top/\sqrt d)\,V$: score each query against all keys, softmax into weights, take the weighted average of the values.`,
    String.raw`It's <strong>content-based routing</strong> with a global receptive field — any token can pull from any other, with data-dependent weights (unlike a conv's fixed local kernel).`,
    String.raw`Each row of the attention matrix is a distribution over keys (sums to 1); <strong>self-attention</strong> makes $Q,K,V$ projections of the same input.`,
    String.raw`The $\sqrt d$ is essential: $\operatorname{Var}(q\cdot k)=d$, so without it the softmax saturates to near one-hot and its gradient vanishes; $/\sqrt d$ restores unit variance.`,
    String.raw`Cost is $O(n^2 d)$ — the $QK^\top$ matrix is $n\times n$ — so attention is quadratic in sequence length, the main scaling bottleneck of transformers.`
  ],

  commonMistakes: [
    { wrong: "Omitting the 1/√d scaling on the scores.",
      why: String.raw`For realistic $d$ (64, 128) the un-scaled logits have std $\sqrt d$, saturating softmax onto one key; its Jacobian $\text{diag}(p)-pp^\top\to0$, so gradients through attention vanish and the model barely learns. The scaling is not decorative.` },
    { wrong: "Forgetting the causal mask in an autoregressive model.",
      why: String.raw`A language model must not let a token attend to <em>future</em> tokens. You add $-\infty$ to the upper-triangular scores before softmax so those weights become 0. Omitting the mask lets the model cheat by peeking ahead, giving great training loss and useless generation.` },
    { wrong: "Treating attention's parameter count as its cost.",
      why: String.raw`The projections $W_Q,W_K,W_V,W_O$ are modest; the expense is the $n\times n$ attention matrix and $O(n^2 d)$ compute/memory. Doubling sequence length quadruples attention cost — which is why long-context work (FlashAttention, sparse/linear attention) targets this term, not the parameters.` }
  ],

  quiz: [
    { q: "Scaled dot-product attention computes…",
      options: ["softmax(QKᵀ/√d) V", "QKᵀV", "softmax(Q)·softmax(K)·V", "Q + K + V"], answer: 0,
      explain: String.raw`Score $QK^\top$, scale by $1/\sqrt d$, softmax into weights, multiply by $V$ — a weighted average of values.` },
    { q: "Why divide the scores by √d?",
      options: ["Var(q·k) = d, so /√d keeps logit variance ~1 and stops softmax from saturating",
                "To normalize the values", "To make it faster", "To add positional info"], answer: 0,
      explain: String.raw`Unit-variance $q,k$ give $\operatorname{Var}(q\cdot k)=d$; without scaling softmax collapses to one-hot with ~zero gradient. $/\sqrt d$ restores variance 1.` },
    { q: "Each row of the attention matrix A = softmax(QKᵀ/√d)…",
      options: ["Sums to 1 (a distribution over keys)", "Sums to 0", "Is one-hot by construction", "Equals the values"], answer: 0,
      explain: String.raw`Softmax normalizes each query's scores into non-negative weights over the keys that sum to 1.` },
    { q: "In SELF-attention, Q, K, and V are…",
      options: ["Three linear projections of the same input X", "Three separate inputs",
                "The same matrix", "Random matrices"], answer: 0,
      explain: String.raw`$Q=XW_Q,\ K=XW_K,\ V=XW_V$ — the sequence attends to itself. (Cross-attention draws K,V from a different source.)` },
    { q: "The computational cost of attention scales with sequence length n as…",
      options: ["O(n²) — the n×n attention matrix", "O(n)", "O(log n)", "O(1)"], answer: 0,
      explain: String.raw`Forming $QK^\top$ and applying it is $O(n^2 d)$ — quadratic in $n$, the core scaling limit of transformers.` }
  ],

  practice: [
    { level: "easy", prompt: "Construct Q, K, V so that query 0's key exactly matches value 2's key, and confirm attention returns value 2 for query 0.",
      solution: String.raw`Set $q_0=k_2$ and make the other keys nearly orthogonal to $q_0$. Then $q_0\cdot k_2$ dominates, softmax puts almost all weight on key 2, and the output for query 0 is $\approx v_2$ — attention as content-addressed lookup.` },
    { level: "easy", prompt: "Add a causal mask to the code (set scores[i,j] = −inf for j > i) and describe the resulting attention matrix.",
      solution: String.raw`After masking, $A$ is lower-triangular: query $i$ only attends to keys $0..i$. That's exactly what a GPT-style decoder needs so a token can't see the future — the mask enforces autoregression.` },
    { level: "med", prompt: "Show that if all keys are identical, attention returns the same output (the average of the values) for every query. Why?",
      solution: String.raw`Identical keys make every score equal within a row, so softmax is uniform ($1/m$ each), and every query outputs the plain mean of the values. Attention degenerates to averaging when keys carry no distinguishing information — a useful sanity limit.` },
    { level: "med", prompt: "Derive Var(q·k) = d for independent, zero-mean, unit-variance entries, and state the resulting logit std.",
      solution: String.raw`$q\cdot k=\sum_{l=1}^d q_l k_l$; each term has mean 0 and variance $\operatorname{Var}(q_l)\operatorname{Var}(k_l)=1$ (independent, zero-mean), and the $d$ terms are independent, so the sum has variance $d$ and std $\sqrt d$. Hence dividing by $\sqrt d$ gives unit-variance logits.` },
    { level: "hard", prompt: "Explain how attention relates to a differentiable dictionary / kernel smoother, and what the softmax temperature controls.",
      solution: String.raw`Attention is a Nadaraya–Watson kernel regression: the output is $\sum_j w_j v_j$ with weights $w_j\propto\exp(q\cdot k_j/\sqrt d)$ — a soft nearest-neighbor lookup keyed by similarity. The effective temperature is $1/\sqrt d$ (or an explicit $\tau$): lower temperature sharpens toward hard retrieval (one key), higher temperature blurs toward uniform averaging. The $\sqrt d$ scaling is precisely a temperature that keeps the smoother in a useful, gradient-friendly regime.` },
    { level: "hard", prompt: "Sketch why attention is permutation-equivariant and what that implies (motivating positional encodings).",
      solution: String.raw`Attention treats its inputs as a <em>set</em>: permuting the tokens permutes $Q,K,V$ rows identically, so the outputs permute the same way and the operation has no inherent notion of order. That's permutation equivariance — great for sets, but language and images have order, so transformers must <em>inject</em> position via positional encodings (sinusoidal, learned, or rotary) added to or mixed into the tokens. Without them, 'dog bites man' and 'man bites dog' look identical to attention.` }
  ],

  deepDive: String.raw`<p><strong>Attention as data-dependent convolution.</strong> A convolution mixes a fixed local neighborhood with fixed learned weights; attention mixes the <em>entire</em> sequence with weights <em>computed from the content</em> ($A=\operatorname{softmax}(QK^\top/\sqrt d)$). That's the trade: attention drops the locality/equivariance prior of convolution in exchange for a global, content-adaptive receptive field. It needs more data (or added positional/locality bias) to compensate for the weaker prior, but at scale it models long-range structure convolutions can't — the crux of why transformers overtook CNNs and RNNs.</p>
<p><strong>The quadratic wall.</strong> The $n\times n$ attention matrix makes cost grow with the square of context length, which is why 'longer context' is expensive and an active research frontier: FlashAttention (an exact, IO-aware kernel that never materializes the full matrix), sparse/local attention, and linear-attention approximations all attack this $O(n^2)$ term. None change the math you just learned — they change how (or how much of) it is computed.</p>
<p><strong>Everything reduces to matmuls again.</strong> Attention is four matrix multiplies ($XW_Q$, $XW_K$, $XW_V$, and $AV$) plus a softmax — the same 'it's all matmuls' story as convolution. What makes it special isn't exotic math; it's that the weight matrix $A$ is computed on the fly from the input, giving the network a way to route information dynamically. The next lesson stacks this into multiple heads and wraps it with the residuals and norms from earlier to form the full transformer block.</p>`
};

/* ------------------------------------------------------------------ 17.5 */
window.LESSON_CONTENT["17.5"] = {
  subtitle: "Run several attentions in parallel, then wrap attention and an MLP each in a residual + LayerNorm. That unit — repeated dozens of times — is the entire transformer.",

  aiMoment: String.raw`Stack this one block 12, 48, or 96 times and you have GPT, BERT, Llama, or the denoiser inside a diffusion model. The transformer block is the single most important architectural unit in modern AI, and it's assembled entirely from pieces you've already built: attention routes information between tokens (17.4), an MLP processes each token (the linear layer of 14.4 with a GELU from 17.1), LayerNorm keeps activations scaled (17.3), and residual connections keep gradients flowing through depth (17.1). This lesson is where the whole course clicks into a working architecture.`,

  plainEnglish: String.raw`A single attention can only track one kind of relationship at a time. <strong>Multi-head</strong> attention runs several attentions in parallel, each in its own smaller subspace, so one head can follow grammatical subject–verb links while another tracks which pronoun refers to whom — then their results are concatenated and combined. The full <strong>block</strong> then does two things in turn, each with a safety rail: mix information across tokens (attention), and process each token on its own (a small MLP), wrapping each step in a normalization and an "add the input back" residual connection.`,

  intuition: String.raw`Split the model dimension into $h$ heads; each head gets its own $Q,K,V$ projections and does attention in a $d/h$-dimensional subspace, letting the model attend to several patterns at once. Concatenate the heads and project back to the model dimension. The block then reads like a checklist of everything in this track: <em>LayerNorm</em> conditions the input, <em>attention</em> lets tokens exchange information, a <em>residual</em> adds the original back so nothing is lost and gradients have a highway, then <em>LayerNorm → MLP → residual</em> gives each token a nonlinear transform. Every ingredient is here for a reason you now know.`,

  formalism: String.raw`<strong>Multi-head attention:</strong>
$$\operatorname{MHA}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\qquad \text{head}_i=\operatorname{Attention}(XW_Q^i,\,XW_K^i,\,XW_V^i),$$
with each head of dimension $d/h$. A <strong>pre-LN transformer block</strong> is two residual sublayers:
$$h=X+\operatorname{MHA}(\operatorname{LN}(X)),\qquad \operatorname{out}=h+\operatorname{MLP}(\operatorname{LN}(h)),\quad \operatorname{MLP}(z)=\operatorname{GELU}(zW_1)\,W_2.$$
Parameter count per block: $\underbrace{4d^2}_{W_Q,W_K,W_V,W_O}+\underbrace{2\,d\,d_{\text{ff}}}_{\text{MLP}}$ (with $d_{\text{ff}}\approx4d$, the MLP holds most of the weights). The block preserves shape ($n\times d\to n\times d$), so blocks stack indefinitely.`,

  derivation: String.raw`<strong>Why each piece is in the block.</strong>
<ol>
<li><strong>Multiple heads.</strong> One softmax-attention produces a single weighting of the values — one 'view'. Splitting into $h$ heads lets the model compute $h$ different attention patterns in parallel (in different learned subspaces) for the same cost as one full-width attention, then $W_O$ mixes them. Empirically, heads specialize (positional, syntactic, rare-token, etc.).</li>
<li><strong>Residual connections.</strong> Writing each sublayer as $x+F(x)$ makes $\partial\,\text{out}/\partial x=I+\partial F/\partial x$ — the identity term is a gradient highway (17.1), so even a 96-layer stack trains. It also means each sublayer only has to learn a <em>refinement</em> of its input, not rebuild it.</li>
<li><strong>LayerNorm.</strong> Attention and MLP both behave badly if their inputs drift in scale; LN (17.3) re-conditions each token's features to mean 0 / variance 1 before each sublayer (pre-LN placement keeps the residual path clean, which is why deep LLMs prefer it).</li>
<li><strong>The MLP.</strong> Attention <em>mixes</em> tokens but is (per value) linear; the position-wise MLP with a GELU adds the per-token nonlinearity that gives the block its representational power — and, at $d_{\text{ff}}\approx4d$, holds ~2/3 of the parameters. Together: attention moves information <em>between</em> tokens, the MLP transforms it <em>within</em> each token. Stacking the block $N$ times is the transformer. The code assembles exactly this and checks the shape is preserved.</li>
</ol>`,

  code: [
    { label: "Multi-head attention + a full transformer block",
      src: String.raw`import numpy as np
rng = np.random.default_rng(0)

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e/e.sum(-1, keepdims=True)
def attention(Q, K, V): d = Q.shape[-1]; A = softmax(Q@K.T/np.sqrt(d)); return A@V

d_model, n_heads, d_ff, seq = 64, 8, 256, 10
dh = d_model // n_heads
Wq, Wk, Wv, Wo = (rng.standard_normal((d_model, d_model))*0.1 for _ in range(4))
W1, W2 = rng.standard_normal((d_model, d_ff))*0.1, rng.standard_normal((d_ff, d_model))*0.1

def multi_head(X):                                    # split into heads, attend, concat, project
    n = X.shape[0]
    Q, K, V = (X@Wq).reshape(n,n_heads,dh), (X@Wk).reshape(n,n_heads,dh), (X@Wv).reshape(n,n_heads,dh)
    heads = [attention(Q[:,h], K[:,h], V[:,h]) for h in range(n_heads)]
    return np.concatenate(heads, 1) @ Wo

def layernorm(x): m = x.mean(-1,keepdims=True); return (x-m)/np.sqrt(((x-m)**2).mean(-1,keepdims=True)+1e-5)
def gelu(z): return 0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))

X = rng.standard_normal((seq, d_model))
h   = X + multi_head(layernorm(X))                    # attention sublayer + residual
out = h + gelu(layernorm(h) @ W1) @ W2                 # MLP sublayer + residual   (a pre-LN block)
print("transformer block:", X.shape, "->", out.shape, " (shape preserved for stacking):", X.shape == out.shape)

params = 4*d_model**2 + 2*d_model*d_ff
print(f"params per block = 4·d² (attention) + 2·d·d_ff (MLP) = {params}   for d={d_model}, d_ff={d_ff}")
print("residuals give dout/dX a +I term (gradient highway), so deep stacks stay trainable")` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 210" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="10.5">
  <defs><marker id="b5a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <rect x="12" y="96" width="46" height="34" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="35" y="117" text-anchor="middle">x</text>
  <rect x="82" y="96" width="42" height="34" rx="6" fill="#efeaf0" stroke="#6a51a3"/><text x="103" y="117" text-anchor="middle" fill="#4b2e83">LN</text>
  <rect x="146" y="92" width="110" height="42" rx="6" fill="#fbeaed" stroke="#c1121f"/><text x="201" y="110" text-anchor="middle" fill="#8f2233" font-weight="700">Multi-Head</text><text x="201" y="124" text-anchor="middle" fill="#a05563">Attention</text>
  <circle cx="292" cy="113" r="14" fill="#e7f0e8" stroke="#3a7d44"/><text x="292" y="117" text-anchor="middle" fill="#245030" font-weight="700">+</text>
  <rect x="326" y="96" width="42" height="34" rx="6" fill="#efeaf0" stroke="#6a51a3"/><text x="347" y="117" text-anchor="middle" fill="#4b2e83">LN</text>
  <rect x="390" y="92" width="104" height="42" rx="6" fill="#e7eff6" stroke="#2a6f97"/><text x="442" y="110" text-anchor="middle" fill="#1c4e70" font-weight="700">MLP</text><text x="442" y="124" text-anchor="middle" fill="#3d6a89">(GELU)</text>
  <circle cx="530" cy="113" r="14" fill="#e7f0e8" stroke="#3a7d44"/><text x="530" y="117" text-anchor="middle" fill="#245030" font-weight="700">+</text>
  <rect x="562" y="96" width="52" height="34" rx="6" fill="#dbe8fb" stroke="#2a6f97"/><text x="588" y="117" text-anchor="middle" fill="#123a5a">out</text>
  <g stroke="#6b7a99">
   <line x1="58" y1="113" x2="80" y2="113" marker-end="url(#b5a)"/><line x1="124" y1="113" x2="144" y2="113" marker-end="url(#b5a)"/>
   <line x1="256" y1="113" x2="277" y2="113" marker-end="url(#b5a)"/><line x1="306" y1="113" x2="324" y2="113" marker-end="url(#b5a)"/>
   <line x1="368" y1="113" x2="388" y2="113" marker-end="url(#b5a)"/><line x1="494" y1="113" x2="515" y2="113" marker-end="url(#b5a)"/>
   <line x1="544" y1="113" x2="560" y2="113" marker-end="url(#b5a)"/></g>
  <path d="M35,96 L35,64 L292,64 L292,99" fill="none" stroke="#3a7d44" stroke-dasharray="4 3" marker-end="url(#b5a)"/>
  <path d="M292,127 L292,150 L530,150 L530,127" fill="none" stroke="#3a7d44" stroke-dasharray="4 3" marker-end="url(#b5a)"/>
  <text x="200" y="58" text-anchor="middle" fill="#245030" font-size="9.5">residual</text>
  <text x="411" y="164" text-anchor="middle" fill="#245030" font-size="9.5">residual</text>
  <text x="360" y="190" text-anchor="middle" fill="#1f2a44">attention mixes tokens · MLP transforms each token · LN scales · residual carries gradients</text>
  <text x="360" y="206" text-anchor="middle" fill="#5a6b8c" font-size="10">stack N of these (with positional info added to x) = a transformer</text>
</svg>`,

  keyPoints: [
    String.raw`<strong>Multi-head</strong> attention runs $h$ attentions in parallel subspaces ($d/h$ each), concatenates, and projects with $W_O$ — several relationship patterns for the price of one.`,
    String.raw`A transformer <strong>block</strong> = attention sublayer + MLP sublayer, each wrapped in LayerNorm and a residual: $h=x+\text{MHA}(\text{LN}(x))$, $\text{out}=h+\text{MLP}(\text{LN}(h))$.`,
    String.raw`Attention moves information <em>between</em> tokens; the position-wise MLP (with GELU) transforms it <em>within</em> each token. Both are needed.`,
    String.raw`Residuals are the gradient highway that makes deep stacks trainable; LayerNorm keeps scales sane; together they let blocks stack dozens deep.`,
    String.raw`Params per block $=4d^2+2d\,d_{\text{ff}}$; with $d_{\text{ff}}\approx4d$ the MLP holds ~2/3 of the weights. The block preserves shape, so $N$ copies stack into GPT/BERT.`
  ],

  commonMistakes: [
    { wrong: "Thinking multi-head attention costs more than single-head.",
      why: String.raw`Each head works in a $d/h$ subspace, so the total work of $h$ heads equals one full-width attention — you get multiple attention patterns at (roughly) the same cost, not $h\times$. The heads are a reshaping of the same computation.` },
    { wrong: "Omitting positional information and expecting the model to understand order.",
      why: String.raw`Attention is permutation-equivariant — it treats tokens as a set. Without positional encodings (sinusoidal, learned, or rotary) added to the inputs, the block can't tell 'dog bites man' from 'man bites dog'. Position must be injected explicitly.` },
    { wrong: "Removing the residual (or the MLP) 'to simplify'.",
      why: String.raw`Drop the residual and deep stacks stop training (gradients vanish — 17.1). Drop the MLP and the block is almost entirely linear in each token (attention averages values), losing most of its expressive power. Each piece is load-bearing.` }
  ],

  quiz: [
    { q: "Multi-head attention combines the heads by…",
      options: ["Concatenating them and applying an output projection W_O", "Averaging them",
                "Taking the max", "Adding them to the input"], answer: 0,
      explain: String.raw`$\operatorname{MHA}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)W_O$ — concatenate the per-head outputs, then project back to the model dimension.` },
    { q: "A pre-LN transformer block's attention sublayer computes…",
      options: ["x + MHA(LN(x))", "LN(x + MHA(x))", "MHA(x)", "x · MHA(x)"], answer: 0,
      explain: String.raw`Pre-LN applies LayerNorm to the input, runs attention, and adds the residual: $x+\operatorname{MHA}(\operatorname{LN}(x))$.` },
    { q: "Within a transformer block, what does the position-wise MLP contribute that attention does not?",
      options: ["A per-token nonlinear transformation (attention only mixes/averages values)",
                "Positional information", "Normalization", "The residual connection"], answer: 0,
      explain: String.raw`Attention forms weighted averages of values (linear in $V$); the MLP with GELU applies a nonlinear transform to each token independently — the block's main nonlinearity and most of its parameters.` },
    { q: "With d_model = 512 and d_ff = 2048, roughly what fraction of a block's parameters are in the MLP?",
      options: ["About 2/3", "About 1/10", "Exactly 1/2", "Almost none"], answer: 0,
      explain: String.raw`Attention: $4d^2=4\cdot512^2\approx1.05$M. MLP: $2\,d\,d_{\text{ff}}=2\cdot512\cdot2048\approx2.1$M. So the MLP is ~$2/3$ of the ~$3.1$M total.` },
    { q: "Why can transformer blocks be stacked to great depth and still train?",
      options: ["Residual connections pass gradients through unattenuated (an identity highway)",
                "Attention has no parameters", "LayerNorm removes all gradients", "The blocks are linear"], answer: 0,
      explain: String.raw`Each sublayer $x+F(x)$ contributes an $+I$ term to the Jacobian, so gradients flow to the earliest layers regardless of depth — the ResNet insight, reused in every transformer.` }
  ],

  practice: [
    { level: "easy", prompt: "Change n_heads to 1 and to 16 (keeping d_model = 64). What must d/h satisfy, and does the parameter count change?",
      solution: String.raw`$d/h$ must be an integer head dimension (64/1=64, 64/16=4). The projection matrices are still $d\times d$ regardless of $h$, so the parameter count is unchanged — only how the same $d$ dimensions are partitioned into parallel attentions changes.` },
    { level: "easy", prompt: "Remove the two residual connections from the code and (conceptually) stack 50 blocks. What would happen to training?",
      solution: String.raw`Without residuals, the input-to-loss gradient becomes a product of 50 sublayer Jacobians and vanishes (17.1), so early blocks get almost no signal and the deep network fails to train. Residuals convert that product into a sum containing the identity, which is why they're mandatory at depth.` },
    { level: "med", prompt: "Add sinusoidal positional encodings to X before the block and explain how they break permutation equivariance.",
      solution: String.raw`Add $\text{PE}[pos,2i]=\sin(pos/10000^{2i/d})$, $\text{PE}[pos,2i+1]=\cos(\cdot)$ to each token by position. Now two tokens with the same content but different positions have different inputs, so permuting tokens changes the outputs — order becomes visible to attention. Learned and rotary (RoPE) encodings achieve the same, RoPE by rotating $Q,K$ by a position-dependent angle.` },
    { level: "med", prompt: "Compare an encoder block (bidirectional attention) and a decoder block (causal-masked attention). When is each used?",
      solution: String.raw`Encoder attention is unmasked — every token sees the whole sequence — good for understanding tasks (BERT, classification, the text encoder in diffusion). Decoder attention is causally masked so a token only sees the past — required for autoregressive generation (GPT). Same block math; the only change is the attention mask.` },
    { level: "hard", prompt: "Estimate the FLOPs of a block for sequence length n and model dim d, and identify when attention vs. the MLP dominates.",
      solution: String.raw`Projections + MLP are $O(n\,d^2)$ (and $O(n\,d\,d_{\text{ff}})$); the attention matrix $QK^\top$ and $AV$ are $O(n^2 d)$. Attention dominates when $n\gtrsim d$ (long sequences); the MLP/projections dominate when $n\lesssim d$ (short sequences, big models). This crossover is why long-context efficiency work targets the $n^2$ term while parameter-scaling work targets the $d^2$ term.` },
    { level: "hard", prompt: "Argue that a transformer block is a general-purpose 'mix then transform' unit and why that generality made it dominant across modalities.",
      solution: String.raw`Attention is a data-dependent mixing operator over a set of tokens, and the MLP is a shared per-token transform; together they impose almost no modality-specific structure. Feed it patches (vision), tokens (language), audio frames, or graph nodes and the same block applies — you only change the tokenization and positional scheme. That minimal, uniform inductive bias, plus residual+norm stability that scales to enormous depth and data, is why one architecture now spans text, images, audio, and multimodal models.` }
  ],

  deepDive: String.raw`<p><strong>This block is the whole course, assembled.</strong> Trace it: the MLP is the linear layer whose gradients you derived in 14.4, with the GELU and gradient-flow reasoning of 17.1; LayerNorm is 17.3; attention with its softmax and $\sqrt d$ is 17.4; residuals come from the vanishing-gradient fix; and if you trained it, the loss would be the softmax cross-entropy of 14.5 optimized by the Adam of the optimization track. The transformer isn't a new kind of math — it's a particularly effective arrangement of everything you've already learned.</p>
<p><strong>Why one architecture ate everything.</strong> A transformer block makes almost no assumptions about its input beyond 'it's a set of vectors', so it applies unchanged to language tokens, image patches (ViT), audio, protein sequences, and actions. That generality, combined with residual+norm stability that scales to hundreds of layers and internet-scale data, is why a single design now underlies LLMs, vision models, and the denoisers of diffusion. Weaker inductive bias than a CNN, but at scale the data supplies what the bias would have.</p>
<p><strong>Where you are now.</strong> With this track you can read a transformer implementation line by line and know why each operation is there — and you've built, verified, and understood every component from arithmetic through matrix calculus, probability, diffusion, and architecture. The natural next steps are scale (mixture-of-experts, efficient attention, quantization) and the training recipes (learning-rate schedules, RLHF) that turn this block into a useful model — but the mathematical core is now yours.</p>`
};

/* ------------------------------------------------------------------ 17.E */
window.LESSON_CONTENT["17.E"] = {
  exam: true,
  intro: String.raw`Ten problems across nonlinearity, convolution, normalization, and attention — ending at the full transformer block. Budget about <strong>75 minutes</strong>. Reason on paper; a REPL is for checking (a param count, an attention row, a LayerNorm output). The payoff of this track: you should be able to read a transformer implementation and justify every line. Solutions and a rubric follow.`,
  problems: [
    { level: "easy", prompt: "Nonlinearity. (a) What is W₂(W₁x) with no activation equal to? (b) Why does that make nonlinearities essential to depth?",
      solution: String.raw`(a) $(W_2W_1)x$ — a single linear map. (b) Without a nonlinearity between layers, any depth collapses to one linear layer, so stacking adds no expressive power; the nonlinearity is what prevents the matrices from being absorbed into one.` },
    { level: "easy", prompt: "Convolution. (a) How many parameters does a 3×3 conv with cᵢₙ=16, cₒᵤₜ=32 have? (b) Name the two constraints that make a conv a special linear layer.",
      solution: String.raw`(a) $k^2 c_{\text{in}} c_{\text{out}}=9\cdot16\cdot32=4608$, independent of image size. (b) Locality (each output sees only a local patch) and weight sharing (same kernel at every position → translation equivariance).` },
    { level: "med", prompt: "Normalization. (a) Over which axis does LayerNorm compute mean/variance vs. BatchNorm? (b) Why do transformers use LayerNorm?",
      solution: String.raw`(a) LayerNorm: per example over its features. BatchNorm: per feature over the batch. (b) LayerNorm is batch-independent — identical at train and test, works at batch size 1 and any sequence length — so it suits variable-length sequence models.` },
    { level: "med", prompt: "Attention. (a) Write scaled dot-product attention. (b) Why the 1/√d?",
      solution: String.raw`(a) $\operatorname{softmax}(QK^\top/\sqrt d)\,V$. (b) With unit-variance $q,k$, $\operatorname{Var}(q\cdot k)=d$; without scaling the logits have std $\sqrt d$, saturating softmax to near one-hot with ~zero gradient. Dividing by $\sqrt d$ restores unit variance.` },
    { level: "med", prompt: "Attention shapes/cost. (a) What does each row of A = softmax(QKᵀ/√d) sum to? (b) What is the compute cost in sequence length n, and why?",
      solution: String.raw`(a) 1 — each row is a distribution over keys. (b) $O(n^2 d)$: forming the $n\times n$ score matrix $QK^\top$ and applying it to $V$ are both quadratic in $n$ — the transformer's core scaling bottleneck.` },
    { level: "hard", prompt: "Derive Var(q·k) = d for independent zero-mean unit-variance entries, and state the resulting logit standard deviation.",
      solution: String.raw`$q\cdot k=\sum_{l=1}^d q_l k_l$. Each term has mean 0 and variance $\operatorname{Var}(q_l)\operatorname{Var}(k_l)=1$; the $d$ terms are independent, so the sum has variance $d$ and standard deviation $\sqrt d$. Hence the $1/\sqrt d$ scaling produces unit-variance logits.` },
    { level: "hard", prompt: "Transformer block. Write the two residual sublayers of a pre-LN block and state what each of the four ingredients (LN, attention, MLP, residual) contributes.",
      solution: String.raw`$h=x+\operatorname{MHA}(\operatorname{LN}(x))$; $\operatorname{out}=h+\operatorname{MLP}(\operatorname{LN}(h))$. LN keeps activations well-scaled; attention mixes information <em>between</em> tokens; the MLP (GELU) applies a nonlinear transform <em>within</em> each token; the residual $x+F(x)$ is a gradient highway (adds $+I$ to the Jacobian) so deep stacks train.` },
    { level: "med", prompt: "Multi-head. (a) How are the heads combined? (b) Why doesn't using h heads cost h× a single attention?",
      solution: String.raw`(a) Concatenate the per-head outputs and apply $W_O$: $\operatorname{Concat}(\text{head}_1..\text{head}_h)W_O$. (b) Each head works in a $d/h$ subspace, so the total work of $h$ heads equals one full-width attention — it's a partition of the same computation, not $h$ copies.` },
    { level: "med", prompt: "Gradient flow. Why do deep sigmoid nets vanish, and how do ReLU and residual connections each help?",
      solution: String.raw`Backprop multiplies by each activation's slope; $\sigma'\le0.25$ compounds to $\sim0.25^L$ (vanishes). ReLU's slope is 1 on active units, so it doesn't shrink the gradient; a residual $x+F(x)$ contributes $\partial/\partial x=I+\partial F/\partial x$, passing gradient through unattenuated regardless of $F$. Both keep the per-layer factor near 1.` },
    { level: "hard", prompt: "Big picture. Trace how a transformer block reuses earlier parts of this course, and give one reason the architecture generalized across text, images, and audio.",
      solution: String.raw`The MLP is the linear layer + backprop of matrix calculus (14.4) with a GELU (17.1); LayerNorm is 17.3; attention's softmax and $\sqrt d$ are 17.4; residuals fix vanishing gradients (17.1); training would use softmax cross-entropy (14.5) and Adam (optimization). It generalizes because a block only assumes 'a set of vectors' — feed it language tokens, image patches, or audio frames and the same math applies, with residual+norm stability that scales to great depth and data.` }
  ],
  rubric: String.raw`<ul>
<li><strong>9–10:</strong> You can derive the √d scaling, write a transformer block from memory, and justify every component. You can read and reason about any modern architecture — you've reached the goal of this course.</li>
<li><strong>7–8:</strong> Strong. Revisit whichever slipped: the LayerNorm axis (17.3), the √d variance argument (17.4), or the block's residual structure (17.5).</li>
<li><strong>5–6:</strong> The pieces are landing but not yet fluent. Re-run the attention (17.4) and block (17.5) code, changing d, n_heads, and the √d scaling to see each effect.</li>
<li><strong>Below 5:</strong> Rework the track in order. It builds one layer at a time — nonlinearity → convolution → normalization → attention → the block — so anchor on each demo before moving up.</li>
</ul>`
};

