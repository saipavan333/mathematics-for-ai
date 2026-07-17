/* ============================================================
   TRACK 16 — Diffusion & Generative Models
   The math of modern generative AI: VAEs and GANs, then the
   forward/reverse diffusion process, score matching, and
   Langevin sampling. Every demo runs and is verified.
   ============================================================ */
window.LESSON_CONTENT ||= {};

/* ------------------------------------------------------------------ 16.1 */
window.LESSON_CONTENT["16.1"] = {
  subtitle: "Turn the ELBO into a neural network: an encoder that compresses data to a latent code and a decoder that rebuilds it — trained so you can generate new data by sampling the latent.",

  aiMoment: String.raw`A <strong>variational autoencoder</strong> is the ELBO from the last track wearing two neural networks. It's how you learn a smooth <em>latent space</em> you can sample from, interpolate in, and decode into new data — and it's the compression front-end of Stable Diffusion (the 'VAE' that maps images to the latent space the diffusion model actually operates in). Understanding the VAE is the cleanest on-ramp to every deep generative model, because it makes the two moving parts explicit: reconstruct the data, and keep the latent close to a distribution you can sample.`,

  plainEnglish: String.raw`Squeeze each data point through a bottleneck into a small <em>latent code</em>, then expand it back out and check you rebuilt the original (reconstruction). At the same time, force the cloud of latent codes to look like a plain bell curve $\mathcal N(0,I)$. Once training is done, you throw away the encoder: draw a random code from that bell curve, run it through the decoder, and out comes a brand-new, data-like sample.`,

  intuition: String.raw`The encoder maps each $x$ not to a single code but to a little Gaussian (a mean $\mu$ and spread $\sigma$) in latent space; you sample a code $z$ from it. The decoder turns $z$ back into data. Two forces balance: the <strong>reconstruction</strong> term wants codes that rebuild $x$ faithfully, and the <strong>KL</strong> term pulls every encoding Gaussian toward the shared prior $\mathcal N(0,I)$ so the codes tile a single, gap-free region. That second force is what makes a randomly drawn $z$ decode into something sensible — without it you'd have an ordinary autoencoder with a latent space full of holes.`,

  formalism: String.raw`A VAE maximizes the ELBO (per datapoint) over encoder $q_\phi$ and decoder $p_\theta$:
$$\mathcal L(\theta,\phi)=\underbrace{\mathbb E_{q_\phi(z\mid x)}\!\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}}-\underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)}_{\text{latent regularizer}},\qquad p(z)=\mathcal N(0,I).$$
With a Gaussian encoder $q_\phi(z\mid x)=\mathcal N(\mu,\mathrm{diag}\,\sigma^2)$ and the <strong>reparameterization</strong> $z=\mu+\sigma\odot\varepsilon,\ \varepsilon\sim\mathcal N(0,I)$, the KL has a closed form
$$\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\Vert\,\mathcal N(0,I)\big)=\tfrac12\sum_j\big(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\big),$$
and a Gaussian decoder makes the reconstruction term a squared error. In practice we minimize $\text{recon}+\beta\,\text{KL}$ (a $\beta$ knob trades sharpness against a well-behaved latent).`,

  derivation: String.raw`<strong>From the ELBO to a trainable loss.</strong>
<ol>
<li><strong>Start from the ELBO</strong> (15.4): $\log p(x)\ge\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$. Maximizing the right side over $\theta,\phi$ trains the model; the encoder $q_\phi$ is a network that outputs $\mu,\log\sigma^2$ for any $x$ ('amortized' inference).</li>
<li><strong>Reconstruction as squared error.</strong> With $p_\theta(x\mid z)=\mathcal N(\hat x_\theta(z),I)$, $\log p_\theta(x\mid z)=-\tfrac12\lVert x-\hat x_\theta(z)\rVert^2+\text{const}$ — so maximizing it is minimizing reconstruction MSE.</li>
<li><strong>KL in closed form.</strong> The KL between two Gaussians is analytic; for $\mathcal N(\mu,\sigma^2)$ vs $\mathcal N(0,I)$ it's $\tfrac12\sum_j(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2)$ — pushing $\mu\to0,\sigma\to1$.</li>
<li><strong>Reparameterize so gradients flow.</strong> Sampling $z\sim q_\phi$ would block $\nabla_\phi$; writing $z=\mu+\sigma\odot\varepsilon$ moves the randomness to $\varepsilon$ and makes $z$ a differentiable function of $\mu,\sigma$. Now the whole ELBO is one differentiable expression and ordinary backprop (Track 14) trains both networks end-to-end. The code below implements exactly this — its analytic gradients pass a finite-difference check.</li>
</ol>`,

  code: [
    { label: "A tiny VAE on 2-D data (gradient-checked backprop), trained from scratch",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(3)

# data: a mixture of three 2-D Gaussian blobs
mus = np.array([[-1.5,-1.0],[1.5,-0.5],[0.0,1.6]]); wts = np.array([.4,.35,.25])
def sample_data(n):
    k = rng.choice(3, n, p=wts); return mus[k] + 0.28*rng.standard_normal((n,2))

H = 32
def init():
    r = lambda a,b: rng.standard_normal((a,b))*0.5
    return dict(W1=r(H,2), b1=np.zeros(H), Wmu=r(2,H), bmu=np.zeros(2), Wlv=r(2,H), blv=np.zeros(2),
                W3=r(H,2), b3=np.zeros(H), W4=r(2,H), b4=np.zeros(2))

def step(P, x, eps, beta):                          # forward + backward for one minibatch
    N = len(x)
    h1 = np.tanh(x@P['W1'].T + P['b1'])
    mu = h1@P['Wmu'].T + P['bmu']
    lv = np.clip(h1@P['Wlv'].T + P['blv'], -6, 6)   # log-variance
    std = np.exp(0.5*lv); z = mu + std*eps          # reparameterize: z = mu + std * eps
    h3 = np.tanh(z@P['W3'].T + P['b3']); xh = h3@P['W4'].T + P['b4']
    recon = ((xh-x)**2).sum(1).mean()
    kl = (0.5*(mu**2 + np.exp(lv) - 1 - lv).sum(1)).mean()
    loss = recon + beta*kl                          # = -ELBO (reconstruction + beta * KL)
    # --- backprop (linear-layer + reparam + KL gradients, all from Tracks 14-15) ---
    dxh = 2*(xh-x)/N
    dW4 = dxh.T@h3; db4 = dxh.sum(0); da3 = (dxh@P['W4'])*(1-h3**2)
    dW3 = da3.T@z; db3 = da3.sum(0); dz = da3@P['W3']
    dmu = dz + beta*mu/N
    dlv = (dz*eps)*0.5*std + beta*0.5*(np.exp(lv)-1)/N
    dWmu, dbmu = dmu.T@h1, dmu.sum(0); dWlv, dblv = dlv.T@h1, dlv.sum(0)
    da1 = (dmu@P['Wmu'] + dlv@P['Wlv'])*(1-h1**2); dW1, db1 = da1.T@x, da1.sum(0)
    g = dict(W1=dW1,b1=db1,Wmu=dWmu,bmu=dbmu,Wlv=dWlv,blv=dblv,W3=dW3,b3=db3,W4=dW4,b4=db4)
    return loss, g

P = init(); m = {k:np.zeros_like(v) for k,v in P.items()}; v = {k:np.zeros_like(u) for k,u in P.items()}
for t in range(1, 1501):                            # Adam
    xb = sample_data(128); eps = rng.standard_normal((128,2))
    loss, g = step(P, xb, eps, beta=0.12)
    for k in P:
        m[k]=0.9*m[k]+0.1*g[k]; v[k]=0.999*v[k]+0.001*g[k]**2
        P[k]-=0.008*(m[k]/(1-0.9**t))/(np.sqrt(v[k]/(1-0.999**t))+1e-8)
print(f"final loss (reconstruction + beta*KL) = {loss:.3f}")

data = sample_data(600)
h1 = np.tanh(data@P['W1'].T + P['b1']); recon = np.tanh((h1@P['Wmu'].T+P['bmu'])@P['W3'].T + P['b3'])@P['W4'].T + P['b4']
gen = np.tanh(rng.standard_normal((600,2))@P['W3'].T + P['b3'])@P['W4'].T + P['b4']   # decode N(0,I) codes
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a,(X,ttl,c) in zip(ax,[(data,'data','#2a6f97'),(recon,'reconstructions','#b8860b'),(gen,'samples from N(0,I) prior','#3a7d44')]):
    a.scatter(X[:,0], X[:,1], s=7, color=c, alpha=.5); a.set_title(ttl); a.set_aspect('equal'); a.set_xlim(-3,3); a.set_ylim(-3,3)
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 190" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="v1a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <rect x="12" y="52" width="70" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="47" y="76" text-anchor="middle" fill="#1f2a44">x</text>
  <rect x="110" y="48" width="104" height="48" rx="6" fill="#dbe8fb" stroke="#2a6f97"/><text x="162" y="68" text-anchor="middle" fill="#123a5a" font-weight="700">encoder</text><text x="162" y="84" text-anchor="middle" fill="#2a6f97" font-size="10">→ μ, log σ²</text>
  <rect x="244" y="52" width="116" height="40" rx="6" fill="#fff5e6" stroke="#b8860b"/><text x="302" y="70" text-anchor="middle" fill="#7a5b00" font-weight="700">z = μ + σ⊙ε</text><text x="302" y="85" text-anchor="middle" fill="#9a7b20" font-size="10">ε ~ N(0, I)</text>
  <rect x="390" y="48" width="104" height="48" rx="6" fill="#e7f0e8" stroke="#3a7d44"/><text x="442" y="72" text-anchor="middle" fill="#245030" font-weight="700">decoder</text><text x="442" y="87" text-anchor="middle" fill="#3a7d44" font-size="10">→ x̂</text>
  <rect x="524" y="52" width="70" height="40" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="559" y="76" text-anchor="middle" fill="#1f2a44">x̂</text>
  <line x1="82" y1="72" x2="108" y2="72" stroke="#6b7a99" marker-end="url(#v1a)"/>
  <line x1="214" y1="72" x2="242" y2="72" stroke="#6b7a99" marker-end="url(#v1a)"/>
  <line x1="360" y1="72" x2="388" y2="72" stroke="#6b7a99" marker-end="url(#v1a)"/>
  <line x1="494" y1="72" x2="522" y2="72" stroke="#6b7a99" marker-end="url(#v1a)"/>
  <text x="620" y="60" text-anchor="middle" fill="#8f2233" font-weight="700" font-size="10.5">to generate:</text>
  <text x="620" y="76" text-anchor="middle" fill="#5a6b8c" font-size="10">draw z ~ N(0,I),</text>
  <text x="620" y="90" text-anchor="middle" fill="#5a6b8c" font-size="10">run the decoder</text>
  <text x="300" y="132" text-anchor="middle" fill="#1f2a44">loss = reconstruction ‖x − x̂‖²  +  KL( q(z|x) ‖ N(0, I) )</text>
  <text x="300" y="154" text-anchor="middle" fill="#5a6b8c" font-size="10.5">rebuild the data, and keep the latent codes shaped like a bell curve you can sample from</text>
</svg>`,

  keyPoints: [
    String.raw`A VAE = encoder ($x\to\mu,\sigma$) + reparameterized sample ($z=\mu+\sigma\odot\varepsilon$) + decoder ($z\to\hat x$), trained to maximize the ELBO.`,
    String.raw`The loss is <strong>reconstruction + KL</strong>: rebuild $x$, and pull every encoding Gaussian toward the prior $\mathcal N(0,I)$. The KL is what makes the latent space samplable.`,
    String.raw`<strong>Generate by decoding prior samples:</strong> throw away the encoder, draw $z\sim\mathcal N(0,I)$, decode. No encoder needed at generation time.`,
    String.raw`The KL between diagonal Gaussians is closed-form, and the reparameterization trick makes the whole ELBO differentiable — so a VAE trains with plain backprop.`,
    String.raw`VAEs give smooth, structured latent spaces but tend to produce <em>blurry</em> samples and to <em>bridge</em> between separated modes — limitations that motivate GANs and diffusion.`
  ],

  commonMistakes: [
    { wrong: "Building a plain autoencoder (no KL term) and expecting to generate by sampling z ~ N(0, I).",
      why: String.raw`Without the KL regularizer the latent codes sprawl into an arbitrary region with gaps; a prior sample lands in a hole and decodes to garbage. The KL is precisely what packs the codes into the samplable $\mathcal N(0,I)$ region.` },
    { wrong: "Sampling z directly from the encoder's distribution instead of via z = μ + σ⊙ε.",
      why: String.raw`Direct sampling blocks the gradient to the encoder. Reparameterization ($z=\mu+\sigma\odot\varepsilon$) keeps $z$ differentiable in $\mu,\sigma$ so backprop can train the encoder — the same trick as 15.4.` },
    { wrong: "Cranking β up for a 'cleaner' latent and then getting posterior collapse.",
      why: String.raw`Too-strong KL drives $q(z\mid x)\to\mathcal N(0,I)$ regardless of $x$ — the latent stops carrying information and reconstructions collapse to the data mean. Balance $\beta$ (or anneal it) so both terms stay alive.` }
  ],

  quiz: [
    { q: "The two terms of a VAE's loss are…",
      options: ["reconstruction error and KL(q(z|x) ‖ N(0,I))", "generator loss and discriminator loss",
                "forward and reverse KL", "bias and variance"], answer: 0,
      explain: String.raw`Maximizing the ELBO = minimizing reconstruction error plus $\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$ with $p(z)=\mathcal N(0,I)$.` },
    { q: "What is the KL term's job?",
      options: ["Pull the encoding distributions toward N(0,I) so the latent space is samplable",
                "Sharpen the reconstructions", "Speed up training", "Remove the decoder"], answer: 0,
      explain: String.raw`It regularizes the aggregate posterior toward the prior, packing codes into a gap-free region you can later sample from to generate.` },
    { q: "To generate a new sample from a trained VAE you…",
      options: ["Draw z ~ N(0, I) and run the decoder", "Run the encoder on noise",
                "Maximize the discriminator", "Average the training set"], answer: 0,
      explain: String.raw`The decoder maps prior samples to data; the encoder is only needed during training.` },
    { q: "Why does a VAE reparameterize as z = μ + σ⊙ε rather than sampling z ~ N(μ,σ²) directly?",
      options: ["So gradients can flow to μ and σ through the sample", "To make z discrete",
                "To remove the KL term", "To avoid using a decoder"], answer: 0,
      explain: String.raw`With $\varepsilon$ fixed, $z$ is a differentiable function of $\mu,\sigma$, so backprop reaches the encoder — otherwise the sampling op stops the gradient.` },
    { q: "A common visible weakness of VAE samples is…",
      options: ["Blurriness / bridging between separated modes", "Perfect sharpness",
                "Inability to reconstruct", "Requiring labels"], answer: 0,
      explain: String.raw`The Gaussian decoder + continuous prior tend to average and to connect modes, giving smooth-but-blurry samples — a key reason GANs and diffusion were developed.` }
  ],

  practice: [
    { level: "easy", prompt: "Set β = 0 (drop the KL) and regenerate from N(0, I). Why do the samples look wrong even though reconstructions look fine?",
      solution: String.raw`With no KL the encoder is free to spread codes anywhere, so reconstructions are great but the latent no longer matches $\mathcal N(0,I)$ — prior samples land in unused regions and decode to nonsense. It's the concrete failure the KL prevents.` },
    { level: "easy", prompt: "Raise β to 2.0 and describe posterior collapse in the reconstructions.",
      solution: String.raw`The reconstructions blur toward the overall data mean: a strong KL forces $q(z\mid x)\approx\mathcal N(0,I)$ for every $x$, so $z$ carries little information and the decoder can't tell inputs apart. Balancing $\beta$ is essential.` },
    { level: "med", prompt: "Interpolate in latent space: encode two data points to their means and decode points along the line between them. What do you observe?",
      solution: String.raw`Decoding $z(t)=(1-t)\mu_1+t\mu_2$ produces a smooth morph from one data point to the other — evidence the KL organized the latent into a continuous, meaningful space (the property that makes latent-space editing possible).` },
    { level: "med", prompt: "Write the closed-form KL between N(μ, σ²) and N(0, 1) for a single dimension and confirm it's zero only at μ=0, σ=1.",
      solution: String.raw`$\mathrm{KL}=\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)$. Its gradient vanishes at $\mu=0$ and $\sigma^2=1$, where $\mathrm{KL}=\tfrac12(0+1-1-0)=0$ — the prior itself, the only zero.` },
    { level: "hard", prompt: "Explain why a Gaussian-decoder VAE produces blurry samples in terms of the reconstruction likelihood.",
      solution: String.raw`A Gaussian likelihood penalizes squared error, whose optimal single prediction for a multimodal conditional $p(x\mid z)$ is the <em>mean</em> of the modes — a blur. The model has no incentive to commit to one sharp mode, so it hedges. GANs (adversarial loss) and diffusion (iterative refinement) avoid this mean-seeking behavior.` },
    { level: "hard", prompt: "Relate the VAE objective to rate–distortion / the information bottleneck.",
      solution: String.raw`Reconstruction is 'distortion' and the KL upper-bounds the 'rate' (bits the latent carries about $x$). $\beta$ traces the rate–distortion frontier: small $\beta$ = high rate, sharp reconstructions; large $\beta$ = low rate, compressed but blurry. A VAE is an information bottleneck with a learned code, which is why $\beta$-VAEs are studied for disentanglement.` }
  ],

  deepDive: String.raw`<p><strong>Amortized inference is the key trick.</strong> Classical variational inference (15.4) optimizes a separate $q$ per datapoint; a VAE trains one encoder network that <em>predicts</em> the variational parameters for any input in a single pass. That amortization is what makes VI scale to millions of images — and it's the same move behind 'amortized' everything (learned optimizers, hypernetworks).</p>
<p><strong>Latents power modern systems.</strong> Stable Diffusion doesn't diffuse pixels; it uses a VAE to compress images ~8× per side into a latent, runs the diffusion model there (far cheaper), then decodes. So the humble VAE is a load-bearing component of state-of-the-art image generation — the diffusion happens in the space this lesson builds.</p>
<p><strong>Why we keep going.</strong> The VAE's blur and mode-bridging come from a single-shot Gaussian decoder trained on a likelihood bound. The next two ideas attack that from opposite ends: GANs replace the likelihood with an adversarial critic that punishes blur, and diffusion replaces the single shot with hundreds of small denoising steps. Both are in this track — the VAE is where the generative-modeling story starts.</p>`
};

/* ------------------------------------------------------------------ 16.2 */
window.LESSON_CONTENT["16.2"] = {
  subtitle: "Two networks play a game: a generator forges fake data and a discriminator tries to catch it. At equilibrium the fakes are indistinguishable — and the math says the generator is minimizing a Jensen–Shannon divergence.",

  aiMoment: String.raw`For years the sharpest generative images came from <strong>GANs</strong> — StyleGAN's photorealistic faces, image-to-image translation, super-resolution. The idea is a genuine conceptual leap: instead of writing down a likelihood to maximize, you train a second network to <em>be the loss</em>. A discriminator learns to spot fakes, and the generator improves precisely by defeating it. Understanding why that adversarial game has the data distribution as its equilibrium — and why it's so twitchy to train — is essential context for the whole generative-modeling landscape.`,

  plainEnglish: String.raw`Picture a forger and a detective. The <strong>generator</strong> (forger) turns random noise into fake data. The <strong>discriminator</strong> (detective) looks at a sample and guesses "real or fake?". They train against each other: the detective gets better at spotting fakes, which forces the forger to make better fakes, and round it goes. When the forger wins completely — the detective can only guess 50/50 — the fakes match the real data.`,

  intuition: String.raw`The discriminator $D(x)$ outputs its probability that $x$ is real. For a <em>fixed</em> generator, the best possible detective compares the two densities: $D^\*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_{\text{gen}}(x)}$ — near 1 where real data dominates, near 0 where fakes dominate, exactly $\tfrac12$ where they overlap. Substitute that ideal detective back in and the generator's objective becomes the <strong>Jensen–Shannon divergence</strong> between real and fake distributions. Driving it to zero means $p_{\text{gen}}=p_{\text{data}}$: the game's whole purpose is to match the data distribution, with the discriminator as a self-sharpening ruler.`,

  formalism: String.raw`The GAN objective is a min–max game over generator $G$ and discriminator $D$:
$$\min_G\max_D\ V(D,G)=\mathbb E_{x\sim p_{\text{data}}}[\log D(x)]+\mathbb E_{z\sim p_z}[\log(1-D(G(z)))].$$
For fixed $G$, the optimal discriminator is
$$D^\*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_{\text{gen}}(x)}.$$
Substituting $D^\*$ collapses the value function to
$$V(G,D^\*)=2\,\mathrm{JS}\big(p_{\text{data}}\,\Vert\,p_{\text{gen}}\big)-2\log 2,$$
so minimizing over $G$ minimizes the (non-negative) Jensen–Shannon divergence, with the unique optimum $p_{\text{gen}}=p_{\text{data}}$.`,

  derivation: String.raw`<strong>Why the game targets the data distribution.</strong>
<ol>
<li><strong>Optimal discriminator, pointwise.</strong> For fixed $G$, at each $x$ the objective is $p_{\text{data}}(x)\log D+p_{\text{gen}}(x)\log(1-D)$. Differentiate in $D$ and set to zero: $\frac{p_{\text{data}}}{D}-\frac{p_{\text{gen}}}{1-D}=0\Rightarrow D^\*=\frac{p_{\text{data}}}{p_{\text{data}}+p_{\text{gen}}}.$</li>
<li><strong>Plug it back.</strong> $V(G,D^\*)=\mathbb E_{p_{\text{data}}}\!\big[\log\tfrac{p_{\text{data}}}{p_{\text{data}}+p_{\text{gen}}}\big]+\mathbb E_{p_{\text{gen}}}\!\big[\log\tfrac{p_{\text{gen}}}{p_{\text{data}}+p_{\text{gen}}}\big].$</li>
<li><strong>Recognize the divergence.</strong> Writing $p_{\text{data}}+p_{\text{gen}}=2\cdot\frac{p_{\text{data}}+p_{\text{gen}}}{2}$ turns this into $-2\log 2+\mathrm{KL}\!\big(p_{\text{data}}\Vert\tfrac{p_{\text{data}}+p_{\text{gen}}}{2}\big)+\mathrm{KL}\!\big(p_{\text{gen}}\Vert\tfrac{p_{\text{data}}+p_{\text{gen}}}{2}\big)=-2\log 2+2\,\mathrm{JS}(p_{\text{data}}\Vert p_{\text{gen}}).$</li>
<li><strong>Read the optimum.</strong> JS $\ge 0$ with equality iff the two distributions are equal, so the generator's best response is $p_{\text{gen}}=p_{\text{data}}$, at which $D^\*\equiv\tfrac12$ (the detective is reduced to a coin flip). The code below traces this JS curve and shows it bottoming out exactly when the generator matches the data.</li>
</ol>`,

  code: [
    { label: "The optimal discriminator, and the Jensen–Shannon divergence it induces",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt

# Two 1-D Gaussians: real data ~ N(2, 1); generator output ~ N(mu_g, 1).
def gauss(x, m, s=1.0): return np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))
xs = np.linspace(-6, 8, 2000)
trap = lambda f: np.sum((f[:-1] + f[1:]) * 0.5 * np.diff(xs))     # simple integral, no scipy

# For a FIXED generator, the best discriminator is D*(x) = p_data/(p_data + p_gen).
# Plugging D* back in, the generator's objective is 2*JS(p_data ‖ p_gen) - 2log2,
# so training the generator MINIMIZES the Jensen-Shannon divergence.
def JS(mu_g):
    pd, pg = gauss(xs, 2.0), gauss(xs, mu_g); mix = 0.5*(pd + pg)
    kl = lambda p, q: trap(np.where(p > 0, p*np.log(p/q), 0.0))
    return 0.5*kl(pd, mix) + 0.5*kl(pg, mix)

mgs = np.linspace(-3, 7, 60); js = [JS(m)/np.log(2) for m in mgs]   # in bits (0..1)
best = mgs[int(np.argmin(js))]
print(f"JS minimized at generator mean = {best:.2f} (data mean = 2.0); JS there = {min(js):.3f} bits")

Dstar = gauss(xs, 2.0) / (gauss(xs, 2.0) + gauss(xs, 0.0) + 1e-12)   # D* while the generator is still at 0
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(xs, gauss(xs, 2.0), color='#2a6f97', label='p_data'); ax[0].plot(xs, gauss(xs, 0.0), color='#3a7d44', label='p_gen')
ax[0].plot(xs, Dstar, color='#d1495b', ls='--', label='optimal D*(x)'); ax[0].set_xlim(-6, 8); ax[0].legend(fontsize=8)
ax[0].set_title('the optimal discriminator separates real from fake')
ax[1].plot(mgs, js, color='#c1121f'); ax[1].axvline(2.0, ls=':', color='#2a6f97', label='data mean')
ax[1].set_xlabel('generator mean'); ax[1].set_ylabel('JS(p_data ‖ p_gen)  [bits]'); ax[1].legend(fontsize=8)
ax[1].set_title('generator training minimizes JS → matches the data')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 196" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="g2a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <rect x="12" y="30" width="86" height="38" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="55" y="53" text-anchor="middle" fill="#1f2a44">noise z</text>
  <rect x="128" y="26" width="110" height="46" rx="6" fill="#e7f0e8" stroke="#3a7d44"/><text x="183" y="46" text-anchor="middle" fill="#245030" font-weight="700">Generator G</text><text x="183" y="62" text-anchor="middle" fill="#3a7d44" font-size="10">noise → fake</text>
  <rect x="270" y="30" width="96" height="38" rx="6" fill="#eef2f7" stroke="#3a7d44"/><text x="318" y="53" text-anchor="middle" fill="#245030">fake G(z)</text>
  <rect x="270" y="112" width="96" height="38" rx="6" fill="#eef2f7" stroke="#2a6f97"/><text x="318" y="135" text-anchor="middle" fill="#123a5a">real data</text>
  <rect x="404" y="66" width="120" height="48" rx="6" fill="#fbeaed" stroke="#c1121f"/><text x="464" y="86" text-anchor="middle" fill="#8f2233" font-weight="700">Discriminator D</text><text x="464" y="102" text-anchor="middle" fill="#a05563" font-size="10">real or fake?</text>
  <rect x="560" y="66" width="146" height="48" rx="6" fill="#eef2f7" stroke="#33415c"/><text x="633" y="86" text-anchor="middle" fill="#1f2a44">D(x) = P(real)</text><text x="633" y="102" text-anchor="middle" fill="#5a6b8c" font-size="10">D* = p_data/(p_data+p_gen)</text>
  <line x1="98" y1="49" x2="126" y2="49" stroke="#6b7a99" marker-end="url(#g2a)"/>
  <line x1="238" y1="49" x2="268" y2="49" stroke="#6b7a99" marker-end="url(#g2a)"/>
  <line x1="366" y1="49" x2="402" y2="78" stroke="#6b7a99" marker-end="url(#g2a)"/>
  <line x1="366" y1="131" x2="402" y2="102" stroke="#6b7a99" marker-end="url(#g2a)"/>
  <line x1="524" y1="90" x2="558" y2="90" stroke="#6b7a99" marker-end="url(#g2a)"/>
  <text x="360" y="176" text-anchor="middle" fill="#1f2a44">min_G max_D  E[log D(real)] + E[log(1 − D(fake))]   →   solved when p_gen = p_data</text>
</svg>`,

  keyPoints: [
    String.raw`A GAN is a min–max game: the discriminator maximizes its ability to tell real from fake; the generator minimizes it. No explicit likelihood — the critic <em>is</em> the loss.`,
    String.raw`For a fixed generator, the optimal discriminator is $D^\*=\frac{p_{\text{data}}}{p_{\text{data}}+p_{\text{gen}}}$ — and it equals $\tfrac12$ everywhere exactly when the fakes match the real data.`,
    String.raw`With $D^\*$ plugged in, the generator minimizes $\mathrm{JS}(p_{\text{data}}\Vert p_{\text{gen}})$; its global optimum is $p_{\text{gen}}=p_{\text{data}}$.`,
    String.raw`Adversarial training avoids the likelihood's mean-seeking blur, which is why classic GANs produced <strong>sharp</strong> samples — but the game is delicate and prone to instability.`,
    String.raw`The signature failure is <strong>mode collapse</strong>: the generator dumps all its mass on a few outputs that fool the current $D$, ignoring the rest of the data distribution.`
  ],

  commonMistakes: [
    { wrong: "Using the saturating generator loss log(1 − D(G(z))) as-is early in training.",
      why: String.raw`When fakes are obviously bad, $D(G(z))\approx0$ and that loss has a vanishing gradient — the generator can't learn. The standard fix is the <em>non-saturating</em> loss: maximize $\log D(G(z))$ instead, which gives strong gradients exactly when the generator is losing.` },
    { wrong: "Training the discriminator to optimality every step and expecting stable learning.",
      why: String.raw`A perfect $D$ gives the generator near-zero gradient (it's already fully caught), stalling learning. GAN training is a moving-target balance; you interleave modest $D$ and $G$ updates rather than fully solving either.` },
    { wrong: "Reading low discriminator loss as 'the generator is great'.",
      why: String.raw`The two losses are adversarial and non-stationary — neither is a progress meter. A generator can achieve low $D$-loss via mode collapse while ignoring most of the data. Judge GANs by sample quality/diversity metrics (e.g. FID), not the training loss.` }
  ],

  quiz: [
    { q: "For a fixed generator, the optimal discriminator D*(x) equals…",
      options: ["p_data/(p_data + p_gen)", "p_gen/(p_data + p_gen)", "½ everywhere", "p_data − p_gen"], answer: 0,
      explain: String.raw`Maximizing $p_{\text{data}}\log D+p_{\text{gen}}\log(1-D)$ pointwise gives $D^\*=\frac{p_{\text{data}}}{p_{\text{data}}+p_{\text{gen}}}$.` },
    { q: "Substituting D* into the value function, the generator ends up minimizing…",
      options: ["the Jensen–Shannon divergence between p_data and p_gen", "the reconstruction error",
                "the entropy of the data", "the KL from N(0,I)"], answer: 0,
      explain: String.raw`$V(G,D^\*)=2\,\mathrm{JS}(p_{\text{data}}\Vert p_{\text{gen}})-2\log 2$, so minimizing over $G$ minimizes JS.` },
    { q: "At the global optimum (p_gen = p_data), what does the discriminator output?",
      options: ["½ everywhere (it can only guess)", "1 for real, 0 for fake", "0 everywhere", "the data density"], answer: 0,
      explain: String.raw`When $p_{\text{gen}}=p_{\text{data}}$, $D^\*=\frac{p_{\text{data}}}{2p_{\text{data}}}=\tfrac12$ — the detective is reduced to a coin flip.` },
    { q: "Mode collapse is when…",
      options: ["The generator produces only a few outputs, ignoring much of the data distribution",
                "The discriminator wins permanently", "The KL term dominates", "The learning rate is too small"], answer: 0,
      explain: String.raw`The generator finds a handful of samples that fool the current $D$ and piles mass there, sacrificing diversity — a hallmark GAN failure.` },
    { q: "Why did GANs historically give sharper samples than VAEs?",
      options: ["The adversarial loss punishes blur, unlike a likelihood's mean-seeking behavior",
                "GANs have more parameters", "GANs use the reparameterization trick", "GANs maximize entropy"], answer: 0,
      explain: String.raw`A pixel-wise likelihood optimally predicts the mean of plausible outputs (a blur); a discriminator flags averaged/blurry samples as fake, pushing the generator to commit to sharp, realistic ones.` }
  ],

  practice: [
    { level: "easy", prompt: "Confirm from the code that JS(p_data ‖ p_gen) reaches 0 exactly when the generator mean equals the data mean.",
      solution: String.raw`The printed minimum sits at generator mean ≈ 2.0 (the data mean), with JS ≈ 0 bits. JS is non-negative and zero only when the distributions coincide, so the game's optimum is $p_{\text{gen}}=p_{\text{data}}$.` },
    { level: "easy", prompt: "Plot D*(x) when the generator already matches the data (both N(2,1)). What do you get and why?",
      solution: String.raw`$D^\*(x)=\frac{p_{\text{data}}}{p_{\text{data}}+p_{\text{gen}}}=\tfrac12$ for all $x$ — a flat line at 0.5. The discriminator has no signal to exploit because real and fake are identical.` },
    { level: "med", prompt: "Explain, using gradients, why the non-saturating loss −log D(G(z)) trains better than log(1 − D(G(z))) early on.",
      solution: String.raw`When $D(G(z))\approx0$ (bad fakes), $\frac{d}{dG}\log(1-D)\to0$ (saturates, no signal), whereas $\frac{d}{dG}\log D$ is large — so the generator gets a strong push exactly when it most needs one. Same fixed point, far better early gradients.` },
    { level: "med", prompt: "The JS divergence saturates to log2 when supports don't overlap. Why is that a problem for training a generator far from the data?",
      solution: String.raw`If $p_{\text{gen}}$ and $p_{\text{data}}$ have disjoint support, JS is constant ($\log 2$), so its gradient w.r.t. the generator is ~0 — no learning signal to pull the fakes toward the data. This 'no overlap ⇒ no gradient' pathology motivated Wasserstein GANs, which use an Earth-Mover distance that still has a gradient across a gap.` },
    { level: "hard", prompt: "Sketch how a Wasserstein GAN changes the objective and why it stabilizes training.",
      solution: String.raw`WGAN replaces JS with the Wasserstein-1 (Earth-Mover) distance: $\max_{\lVert f\rVert_L\le1}\mathbb E_{p_{\text{data}}}[f]-\mathbb E_{p_{\text{gen}}}[f]$, with a 1-Lipschitz 'critic' $f$ (enforced by weight clipping or a gradient penalty). Because Wasserstein varies smoothly even when supports don't overlap, the critic provides usable gradients everywhere, reducing mode collapse and giving a loss that actually correlates with sample quality.` },
    { level: "hard", prompt: "Compare VAEs, GANs, and (previewing) diffusion on the axes: likelihood, sample sharpness, training stability, mode coverage.",
      solution: String.raw`VAE: has a likelihood (ELBO), stable to train, but blurry and sometimes mode-bridging. GAN: no explicit likelihood, sharp samples, unstable and prone to mode collapse (poor coverage). Diffusion (next lessons): likelihood-based (an ELBO), stable, sharp, and excellent mode coverage — at the cost of slow, many-step sampling. Diffusion essentially buys GAN-level sharpness with VAE-level stability by trading generation speed.` }
  ],

  deepDive: String.raw`<p><strong>Why GANs are twitchy.</strong> The min–max game has no single loss that decreases monotonically — the generator and discriminator chase a moving target, and the JS divergence gives no gradient when their supports don't overlap (common early on). That combination causes oscillation, mode collapse, and sensitivity to architecture and learning rates. A decade of fixes — non-saturating loss, Wasserstein distance, gradient penalties, spectral normalization, two-timescale updates — is essentially engineering around this instability.</p>
<p><strong>Adversarial ideas outlived pure GANs.</strong> Even as diffusion overtook GANs for raw image quality, the adversarial <em>loss</em> lives on: as a perceptual/sharpening term in autoencoders and super-resolution, in the discriminators that critique diffusion decoders, and in domain-adaptation and representation learning. "Train a network to be your loss function" is a durable idea independent of the specific GAN recipe.</p>
<p><strong>The through-line to diffusion.</strong> All three models minimize a divergence to the data distribution — VAEs a KL bound, GANs the JS divergence, diffusion a (very cleverly decomposed) KL/ELBO. The next three lessons build diffusion, which keeps the VAE's stable likelihood training and the GAN's sharpness by replacing one hard generation step with a long chain of easy denoising steps.</p>`
};

/* ------------------------------------------------------------------ 16.3 */
window.LESSON_CONTENT["16.3"] = {
  subtitle: "Destroy data by slowly stirring in Gaussian noise until it's pure static. The process is fixed and needs no learning — and it has a closed form that jumps to any noise level in one step.",

  aiMoment: String.raw`Every diffusion model (DDPM, Stable Diffusion, Imagen) is built on a deliberately dumb idea: take a clean image and gradually add noise until it becomes indistinguishable from TV static. That's the <strong>forward process</strong>. It has no parameters — you never train it. Its entire job is to define a smooth bridge from 'data' to 'easy-to-sample noise' that a network will later learn to walk <em>backwards</em>. The one piece of math that makes training practical is the forward process's closed form, which lets you skip straight to any point on that bridge.`,

  plainEnglish: String.raw`Start with a real data point. At each step, shrink it a touch and add a little Gaussian noise. Repeat a few hundred times and, whatever you started from, you end up at standard Gaussian noise. Nothing here is learned — it's a fixed recipe for turning data into noise. The clever bit: you don't have to simulate all the steps to see the sample at step $t$; a single formula gives it directly.`,

  intuition: String.raw`Think of the signal fading while noise fades in. Each step multiplies the current point by a number slightly less than 1 and adds fresh noise. Because a Gaussian plus a Gaussian is a Gaussian, chaining many steps is <em>still</em> just one Gaussian — so the sample at any step $t$ is the original scaled down by $\sqrt{\bar\alpha_t}$ plus noise scaled by $\sqrt{1-\bar\alpha_t}$. Early on ($\bar\alpha_t\approx1$) you mostly see the data; late ($\bar\alpha_t\approx0$) you see pure $\mathcal N(0,I)$. That single closed form is why diffusion can be trained one random timestep at a time.`,

  formalism: String.raw`Fix a small variance schedule $\beta_1,\dots,\beta_T$. Each forward step is Gaussian:
$$q(x_t\mid x_{t-1})=\mathcal N\!\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big).$$
Let $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$. Composing the steps gives the <strong>closed form</strong> (the 'nice property'):
$$q(x_t\mid x_0)=\mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)\quad\Longleftrightarrow\quad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ \ \varepsilon\sim\mathcal N(0,I).$$
As $t\to T$, $\bar\alpha_t\to0$, so $x_T\approx\mathcal N(0,I)$ — a distribution you can sample trivially. (This 'variance-preserving' schedule keeps the total scale near constant.)`,

  derivation: String.raw`<strong>Why many small Gaussian steps collapse into one.</strong>
<ol>
<li><strong>One step, reparameterized.</strong> $x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\varepsilon_{t-1}$ with $\varepsilon_{t-1}\sim\mathcal N(0,I)$.</li>
<li><strong>Two steps.</strong> Substitute $x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\varepsilon_{t-2}$: $\;x_t=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\big(\sqrt{\alpha_t(1-\alpha_{t-1})}\,\varepsilon_{t-2}+\sqrt{1-\alpha_t}\,\varepsilon_{t-1}\big).$</li>
<li><strong>Add the two independent Gaussians.</strong> Their variances add: $\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}$. So $x_t=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}\,\bar\varepsilon$ — the same form, one step wider.</li>
<li><strong>Induct.</strong> Repeating down to $x_0$ gives $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ with $\bar\alpha_t=\prod_s\alpha_s$. The signal coefficient and the noise coefficient always square-sum to 1, which is what keeps the scale controlled and sends $x_T\to\mathcal N(0,I)$. The code applies exactly this formula to watch data dissolve into noise.</li>
</ol>`,

  code: [
    { label: "The forward process: watch data dissolve into Gaussian noise",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def sample_data(n):                                   # a noisy ring
    a = rng.uniform(0, 2*np.pi, n); r = 1 + 0.06*rng.standard_normal(n)
    return np.c_[r*np.cos(a), r*np.sin(a)]
x0 = sample_data(700)

T = 40; betas = np.linspace(1e-4, 0.06, T)            # the (fixed) variance schedule
abar = np.cumprod(1 - betas)                          # ᾱ_t = ∏ (1 - β_s)

# 'Nice property': jump straight to any step t with one formula.
def q_sample(x0, t):                                  # x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε
    return np.sqrt(abar[t])*x0 + np.sqrt(1 - abar[t])*rng.standard_normal(x0.shape)

print(f"per-coordinate variance:  t=0 -> {x0.var():.2f},   t={T-1} -> {q_sample(x0, T-1).var():.2f}   (→ 1.0)")
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
for a, t in zip(ax, [0, 10, 25, 39]):
    xt = q_sample(x0, t); a.scatter(xt[:,0], xt[:,1], s=5, color='#2a6f97', alpha=.5)
    a.set_title(f't = {t}   (ᾱ = {abar[t]:.2f})'); a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect('equal')
plt.suptitle('Forward diffusion: data → Gaussian noise'); plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 168" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="f3a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7a99"/></marker></defs>
  <rect x="14" y="44" width="86" height="44" rx="6" fill="#dbe8fb" stroke="#2a6f97"/><text x="57" y="64" text-anchor="middle" fill="#123a5a" font-weight="700">x₀</text><text x="57" y="80" text-anchor="middle" fill="#2a6f97" font-size="10">data</text>
  <rect x="160" y="44" width="70" height="44" rx="6" fill="#e5ecf3" stroke="#5a7a94"/><text x="195" y="70" text-anchor="middle" fill="#274b63">x₁</text>
  <rect x="290" y="44" width="70" height="44" rx="6" fill="#eceef1" stroke="#7a8aa0"/><text x="325" y="70" text-anchor="middle" fill="#3a4a63">x₂</text>
  <text x="415" y="70" text-anchor="middle" fill="#6b7a99" font-size="16">· · ·</text>
  <rect x="470" y="44" width="70" height="44" rx="6" fill="#f0eef2" stroke="#9a8aa8"/><text x="505" y="70" text-anchor="middle" fill="#4b3a63">x_t</text>
  <rect x="600" y="44" width="104" height="44" rx="6" fill="#efeaf0" stroke="#8a7a99"/><text x="652" y="64" text-anchor="middle" fill="#3a2347" font-weight="700">x_T</text><text x="652" y="80" text-anchor="middle" fill="#6a51a3" font-size="10">≈ N(0, I)</text>
  <g stroke="#6b7a99">
   <line x1="100" y1="66" x2="158" y2="66" marker-end="url(#f3a)"/><line x1="230" y1="66" x2="288" y2="66" marker-end="url(#f3a)"/>
   <line x1="360" y1="66" x2="398" y2="66" marker-end="url(#f3a)"/><line x1="432" y1="66" x2="468" y2="66" marker-end="url(#f3a)"/>
   <line x1="540" y1="66" x2="598" y2="66" marker-end="url(#f3a)"/></g>
  <text x="129" y="36" text-anchor="middle" fill="#8f2233" font-size="9.5">+ noise</text>
  <text x="360" y="112" text-anchor="middle" fill="#1f2a44">closed form:  x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε      (jump to any t in one step)</text>
  <text x="360" y="134" text-anchor="middle" fill="#5a6b8c" font-size="10.5">fixed process, no learning — it just defines the bridge from data to noise the model will reverse</text>
</svg>`,

  keyPoints: [
    String.raw`The forward process is a <strong>fixed</strong> (unlearned) chain of tiny Gaussian noising steps that turns any data point into $\mathcal N(0,I)$ noise.`,
    String.raw`Closed form: $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ with $\bar\alpha_t=\prod_s(1-\beta_s)$ — jump to <em>any</em> noise level in one step, no simulation.`,
    String.raw`That closed form is what makes training cheap: pick a random $t$, noise the data once, and ask the network to undo it.`,
    String.raw`Signal and noise coefficients satisfy $(\sqrt{\bar\alpha_t})^2+(\sqrt{1-\bar\alpha_t})^2=1$ — a variance-preserving schedule that keeps the scale controlled and drives $x_T\to\mathcal N(0,I)$.`,
    String.raw`The schedule $\beta_t$ (how fast noise is added) is a design choice; the endpoint being $\mathcal N(0,I)$ is what lets sampling start from pure noise (next lessons).`
  ],

  commonMistakes: [
    { wrong: "Thinking the forward process is learned or has parameters.",
      why: String.raw`It's completely fixed by the schedule $\beta_t$ — no network, no training. All the learning is in <em>reversing</em> it. Confusing the two is the most common conceptual slip about diffusion.` },
    { wrong: "Simulating all t steps to get a noisy sample during training.",
      why: String.raw`Unnecessary and slow — the closed form $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ produces $x_t$ in one shot. Training samples a random $t$ per example and uses this directly.` },
    { wrong: "Assuming any noise schedule works equally well.",
      why: String.raw`The schedule controls how signal and noise trade off over time; a poor schedule spends most steps where little changes and starves the interesting middle. Cosine and other schedules were introduced specifically to allocate steps better, improving sample quality.` }
  ],

  quiz: [
    { q: "The closed-form forward marginal q(x_t | x₀) is Gaussian with mean and variance…",
      options: ["mean √ᾱ_t·x₀, variance (1−ᾱ_t)I", "mean x₀, variance β_t I", "mean 0, variance I", "mean ᾱ_t·x₀, variance ᾱ_t I"], answer: 0,
      explain: String.raw`$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$ — the composition of all the small steps.` },
    { q: "As t → T (ᾱ_t → 0), the distribution of x_t approaches…",
      options: ["N(0, I)", "the data distribution", "a point mass at x₀", "a uniform distribution"], answer: 0,
      explain: String.raw`With $\bar\alpha_t\to0$ the signal vanishes and the variance $\to1$, so $x_T\approx\mathcal N(0,I)$ — the easy-to-sample endpoint.` },
    { q: "Why can many small Gaussian noising steps be written as a single Gaussian?",
      options: ["A sum of independent Gaussians is Gaussian, and their variances add",
                "Because β_t is small", "Because the data is Gaussian", "They can't — it's an approximation"], answer: 0,
      explain: String.raw`Composing linear-Gaussian steps stays Gaussian; the derivation shows the variances combine to $1-\bar\alpha_t$.` },
    { q: "Is the forward process learned?",
      options: ["No — it's a fixed, parameter-free noising schedule", "Yes — it's trained with the decoder",
                "Only the mean is learned", "It's learned adversarially"], answer: 0,
      explain: String.raw`The forward process has no parameters; all learning happens in the reverse (denoising) direction.` },
    { q: "With ᾱ_t = 0.36, the signal is scaled by roughly… and the noise by roughly…",
      options: ["0.6 and 0.8", "0.36 and 0.64", "0.8 and 0.6", "1 and 0"], answer: 0,
      explain: String.raw`Signal $\sqrt{\bar\alpha_t}=\sqrt{0.36}=0.6$; noise $\sqrt{1-\bar\alpha_t}=\sqrt{0.64}=0.8$. They square-sum to 1.` }
  ],

  practice: [
    { level: "easy", prompt: "Change the schedule to betas = linspace(1e-4, 0.02, 40) (gentler). How does the t=39 panel change?",
      solution: String.raw`Gentler $\beta$ means larger $\bar\alpha_T$ (less total noise), so at $t=39$ the ring is still partly visible rather than full static. The schedule directly controls how far along the data-to-noise bridge you've traveled by the final step.` },
    { level: "easy", prompt: "Verify numerically that (√ᾱ_t)² + (√(1−ᾱ_t))² = 1 for several t.",
      solution: String.raw`It's identically 1 by construction: $\bar\alpha_t+(1-\bar\alpha_t)=1$. This 'variance preservation' is what keeps $x_t$ from blowing up or vanishing in scale as $t$ grows.` },
    { level: "med", prompt: "Given a noised x_t and the true ε used to make it, recover x₀. Which quantity would a denoiser predict, and why is that equivalent?",
      solution: String.raw`From $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$, $x_0=(x_t-\sqrt{1-\bar\alpha_t}\varepsilon)/\sqrt{\bar\alpha_t}$. So predicting $\varepsilon$ is equivalent to predicting $x_0$ (each determines the other given $x_t,t$). DDPM trains a network to predict $\varepsilon$ — the subject of the next lesson.` },
    { level: "med", prompt: "Show that q(x_t | x₀) for image pixels means training can sample a different random t for each image in a batch. Why is that efficient?",
      solution: String.raw`Because $x_t$ is available in closed form for any $t$, each example can be noised to its own random timestep in one operation — no need to run a chain. One forward pass then trains the denoiser across all noise levels simultaneously, which is why diffusion training is simple and parallel.` },
    { level: "hard", prompt: "The forward process is a discretization of an SDE dx = −½β(t)x dt + √β(t) dW. Identify which term shrinks the signal and which injects noise.",
      solution: String.raw`The drift $-\tfrac12\beta(t)x\,dt$ pulls $x$ toward 0 (signal decay), and the diffusion $\sqrt{\beta(t)}\,dW$ injects Brownian noise. This 'variance-preserving' SDE is the continuous-time limit of the DDPM schedule; its marginals are exactly the Gaussian $q(x_t\mid x_0)$ above, connecting discrete diffusion to the score-SDE framework.` },
    { level: "hard", prompt: "Contrast variance-preserving (VP) with variance-exploding (VE) schedules and what each implies for the endpoint.",
      solution: String.raw`VP (used here) keeps total variance ≈1 by scaling the signal down as noise comes in, so $x_T\sim\mathcal N(0,I)$. VE instead keeps the signal at full scale and adds ever-larger noise $\sigma_t$, so $x_T\sim\mathcal N(x_0,\sigma_{\max}^2I)$ with huge $\sigma_{\max}$; you then sample from $\mathcal N(0,\sigma_{\max}^2I)$. Both define valid data↔noise bridges; they differ in normalization and which noise levels the network sees.` }
  ],

  deepDive: String.raw`<p><strong>Diffusion is a deep latent-variable model.</strong> Read $x_1,\dots,x_T$ as latents and the forward process as a fixed inference chain; then a diffusion model is a hierarchical VAE with hundreds of latent layers and a <em>frozen</em> encoder. That's not an analogy — the training objective is literally an ELBO over this chain, which is why Track 15's variational machinery transfers directly. The one thing that's learned is the reverse process.</p>
<p><strong>The schedule is a real design axis.</strong> Linear schedules (as here) are simple but spend many steps near $t=0$ where the image barely changes; cosine schedules and learned/continuous schedules reallocate steps to where the signal actually degrades, improving both likelihood and sample quality. Sampling-time step count (and skipping steps, as in DDIM) is a separate knob that trades quality for speed.</p>
<p><strong>Why this fixed process is the enabling trick.</strong> By defining a smooth, closed-form path from data to $\mathcal N(0,I)$, the forward process converts an intractable "generate data from scratch" problem into a sequence of easy "remove a little noise" problems. The next lesson shows that learning to remove that noise is the same as learning the <em>score</em> of the data — the gradient field that points toward where data lives.</p>`
};

/* ------------------------------------------------------------------ 16.4 */
window.LESSON_CONTENT["16.4"] = {
  subtitle: "To reverse the noise you teach a network to denoise — and, astonishingly, learning to denoise is the same as learning the score ∇log p(x), the vector field that points toward where data lives.",

  aiMoment: String.raw`The entire training loss of a diffusion model is: add known noise to a data point, then have a network predict that noise — a plain mean-squared error. That's it. The deep reason this works is one of the most beautiful results in generative modeling: <strong>the optimal denoiser is the score</strong> $\nabla\log p(x)$, the gradient field pointing toward higher data density. So 'predict the noise' and 'learn the direction toward the data' are the same objective, which is why DDPM (denoising) and score-based models (score matching) turned out to be the same thing.`,

  plainEnglish: String.raw`Hand the network a noisy point and ask: which way was the clean data? The answer is a little arrow pointing back toward the data — and the field of all those arrows is the <strong>score</strong>. You train it with the easiest possible target: you know exactly what noise you added, so you just ask the network to predict it. Predicting the added noise <em>is</em> pointing back toward the data.`,

  intuition: String.raw`The score $\nabla\log p(x)$ points uphill on the data density — toward the modes, away from empty regions. A noisy sample sits off the data; the direction that most increases its log-density is exactly the direction back toward clean data, i.e. the negative of the noise you added (rescaled). So a denoiser trained to predict that noise learns the score field for free. Because a single noise level only reveals the score of the <em>blurred</em> data, diffusion trains the denoiser at <em>all</em> noise levels at once — sharp scores up close, coarse scores far away.`,

  formalism: String.raw`The <strong>score</strong> of a distribution is $s(x)=\nabla_x\log p(x)$. <strong>Denoising score matching</strong>: corrupt $x_0$ to $\tilde x=x_0+\sigma\varepsilon$ and fit $s_\theta$ to the score of the noised point,
$$\min_\theta\ \mathbb E_{x_0,\varepsilon}\Big[\big\lVert s_\theta(\tilde x)-\nabla_{\tilde x}\log q(\tilde x\mid x_0)\big\rVert^2\Big],\qquad \nabla_{\tilde x}\log q(\tilde x\mid x_0)=-\frac{\varepsilon}{\sigma}.$$
In DDPM form, with $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$, the network $\varepsilon_\theta$ is trained by the simple objective
$$\mathcal L=\mathbb E_{t,x_0,\varepsilon}\big[\lVert\varepsilon-\varepsilon_\theta(x_t,t)\rVert^2\big],\qquad\text{and}\qquad s_\theta(x_t,t)=\nabla\log q(x_t)=-\frac{\varepsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}.$$
Predicting the noise $\varepsilon$ and estimating the score are the same, up to the fixed scale $-\sqrt{1-\bar\alpha_t}$.`,

  derivation: String.raw`<strong>Why the optimal denoiser is the score (Tweedie / DSM).</strong>
<ol>
<li><strong>Score of a single noised pair.</strong> For $\tilde x=x_0+\sigma\varepsilon$, $q(\tilde x\mid x_0)=\mathcal N(x_0,\sigma^2I)$, so $\nabla_{\tilde x}\log q(\tilde x\mid x_0)=\dfrac{x_0-\tilde x}{\sigma^2}=-\dfrac{\varepsilon}{\sigma}$ — the arrow from the noisy point back to the clean one.</li>
<li><strong>Averaging gives the marginal score.</strong> A known identity (Vincent, 2011) says the minimizer of the DSM objective satisfies $s_\theta(\tilde x)=\mathbb E\big[\nabla_{\tilde x}\log q(\tilde x\mid x_0)\,\big|\,\tilde x\big]=\nabla_{\tilde x}\log q(\tilde x)$ — matching each noisy point's conditional score in expectation recovers the score of the whole noised distribution $q(\tilde x)$.</li>
<li><strong>Tweedie ties it to denoising.</strong> For Gaussian noise, $\nabla\log q(\tilde x)=\dfrac{\mathbb E[x_0\mid\tilde x]-\tilde x}{\sigma^2}$: the score is (posterior-mean denoised estimate − current point), rescaled. So 'best guess of the clean $x_0$' and 'score' carry the same information.</li>
<li><strong>Hence the simple loss.</strong> Since the DSM target is $-\varepsilon/\sigma$, training a network to predict $\varepsilon$ (an MSE) is equivalent to learning the score, up to the known factor. The code trains exactly this denoiser and confirms the learned field lines up with the analytic score (cosine ≈ 0.94).</li>
</ol>`,

  code: [
    { label: "Denoising score matching: a tiny denoiser learns the score field",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(5)

mus = np.array([[-1.5,-1.0],[1.5,-0.5],[0.0,1.6]]); wts = np.array([.4,.35,.25])
def sample_data(n): k = rng.choice(3, n, p=wts); return mus[k] + 0.28*rng.standard_normal((n,2))
sigma = 0.35

# A tiny MLP score network s_theta: R^2 -> R^2.  Denoising-score-matching target = -eps/sigma.
H = 64
p = {'W1': rng.standard_normal((H,2))*.5, 'b1': np.zeros(H), 'W2': rng.standard_normal((2,H))*.1, 'b2': np.zeros(2)}
opt = {k:[np.zeros_like(v), np.zeros_like(v)] for k,v in p.items()}
for t in range(1, 2001):
    x0 = sample_data(256); eps = rng.standard_normal((256,2)); xt = x0 + sigma*eps
    target = -eps/sigma                               # points the noisy sample back toward the clean one
    h = np.tanh(xt@p['W1'].T + p['b1']); s = h@p['W2'].T + p['b2']
    d = s - target; loss = (d**2).sum(1).mean(); dS = 2*d/len(xt)
    g = {'W2': dS.T@h, 'b2': dS.sum(0)}
    dh = (dS@p['W2'])*(1-h**2); g['W1'] = dh.T@xt; g['b1'] = dh.sum(0)
    for k in p:                                        # Adam
        m, v = opt[k]; m[:] = 0.9*m + 0.1*g[k]; v[:] = 0.999*v + 0.001*g[k]**2
        p[k] -= 0.005*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)

# compare the learned score field to the ANALYTIC score of the noised mixture
s2 = 0.28**2 + sigma**2
def true_score(x):
    d = ((x[:,None,:]-mus[None])**2).sum(-1); lw = np.log(wts) - 0.5*d/s2
    lw -= lw.max(1,keepdims=True); r = np.exp(lw); r /= r.sum(1,keepdims=True)
    return np.einsum('nk,nkd->nd', r, (mus[None]-x[:,None,:])/s2)
gx, gy = np.meshgrid(np.linspace(-3,3,18), np.linspace(-3,3,18)); G = np.c_[gx.ravel(), gy.ravel()]
learned = np.tanh(G@p['W1'].T + p['b1'])@p['W2'].T + p['b2']; true = true_score(G)
cos = ((learned*true).sum(1)/(np.linalg.norm(learned,axis=1)*np.linalg.norm(true,axis=1)+1e-9)).mean()
print(f"mean cosine(learned score, true score) = {cos:.3f}   (1.0 = perfect alignment)")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a,(S,ttl,c) in zip(ax,[(true,'true score  ∇log p(x)','#6a51a3'),(learned,'learned score (denoiser)','#3a7d44')]):
    a.quiver(gx, gy, S[:,0].reshape(gx.shape), S[:,1].reshape(gx.shape), color=c, alpha=.85)
    a.scatter(mus[:,0], mus[:,1], color='#d1495b', s=50, marker='*'); a.set_title(ttl); a.set_aspect('equal')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 200" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="s4r" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#c1121f"/></marker>
        <marker id="s4g" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#8a97b3"/></marker></defs>
  <ellipse cx="150" cy="104" rx="96" ry="58" fill="#dbe8fb" stroke="#2a6f97"/>
  <text x="150" y="150" text-anchor="middle" fill="#123a5a" font-size="10.5">data region  p(x)</text>
  <circle cx="150" cy="100" r="4.5" fill="#2a6f97"/><text x="150" y="90" text-anchor="middle" fill="#123a5a" font-size="10">x₀</text>
  <circle cx="300" cy="46" r="4.5" fill="#3a2347"/><text x="316" y="44" fill="#3a2347" font-size="10">x̃ = x₀ + σε</text>
  <line x1="154" y1="97" x2="296" y2="48" stroke="#8a97b3" stroke-dasharray="4 3" marker-end="url(#s4g)"/>
  <text x="238" y="62" fill="#5a6b8c" font-size="9.5">forward: add noise ε</text>
  <line x1="296" y1="52" x2="160" y2="97" stroke="#c1121f" stroke-width="1.6" marker-end="url(#s4r)"/>
  <text x="250" y="92" fill="#8f2233" font-size="9.5">score points back to data</text>
  <text x="500" y="66" text-anchor="middle" font-weight="700" fill="#1f2a44">the training objective</text>
  <text x="500" y="94" text-anchor="middle" fill="#1f2a44">min  ‖ ε_θ(x̃) − ε ‖²</text>
  <text x="500" y="120" text-anchor="middle" fill="#245030">⇔   s_θ(x) ≈ ∇log p(x)</text>
  <text x="500" y="148" text-anchor="middle" fill="#5a6b8c" font-size="10.5">learning to denoise = learning the score</text>
</svg>`,

  keyPoints: [
    String.raw`The <strong>score</strong> $s(x)=\nabla\log p(x)$ is a vector field pointing toward higher data density (toward the modes).`,
    String.raw`<strong>The optimal denoiser is the score.</strong> Predicting the noise you added ($-\varepsilon/\sigma$) is, in expectation, exactly the score of the noised distribution (Vincent's DSM identity; Tweedie's formula).`,
    String.raw`So diffusion's whole training loss is a mean-squared error: $\lVert\varepsilon-\varepsilon_\theta(x_t,t)\rVert^2$. No adversary, no likelihood evaluation — just denoising.`,
    String.raw`The network is trained at <em>all</em> noise levels $t$ (noise-conditioned), because each level exposes the score of the data blurred to that degree.`,
    String.raw`$\varepsilon$-prediction and score are interchangeable: $s_\theta(x_t,t)=-\varepsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$. That score is what drives sampling in the next lesson.`
  ],

  commonMistakes: [
    { wrong: "Training a single-noise-level denoiser and expecting to generate from pure noise.",
      why: String.raw`One $\sigma$ only teaches the score of the data blurred by that $\sigma$. Sampling starts from huge noise and ends at clean data, so you need the score across the whole range of noise levels — hence the time/noise conditioning $\varepsilon_\theta(x_t,t)$.` },
    { wrong: "Confusing the score ∇log p(x) with the gradient of the loss ∇θ L.",
      why: String.raw`The score is the gradient of $\log p$ <em>with respect to the input $x$</em> (a vector field over data space), not the parameter gradient used for SGD. They're different objects; the network's <em>output</em> approximates the score.` },
    { wrong: "Assuming you need p(x) or its normalizer to learn the score.",
      why: String.raw`The score $\nabla\log p$ drops the normalizing constant ($\nabla\log Z=0$), and denoising score matching never evaluates $p$ at all — it only uses noised samples. That's exactly why score-based generative modeling sidesteps the intractable partition function (echoing the MCMC lesson).` }
  ],

  quiz: [
    { q: "The 'score' in score-based generative models refers to…",
      options: ["∇_x log p(x), a vector field over data space", "the loss value", "the discriminator output", "the log-likelihood number"], answer: 0,
      explain: String.raw`The score is the gradient of the log-density with respect to the input, pointing toward higher-probability regions.` },
    { q: "For a noised pair x̃ = x₀ + σε, the score of q(x̃|x₀) is…",
      options: ["(x₀ − x̃)/σ² = −ε/σ", "x̃ − x₀", "σ·ε", "0"], answer: 0,
      explain: String.raw`$q(\tilde x\mid x_0)=\mathcal N(x_0,\sigma^2 I)$, so $\nabla_{\tilde x}\log q=\frac{x_0-\tilde x}{\sigma^2}=-\varepsilon/\sigma$ — the arrow back to the clean point.` },
    { q: "The DDPM training loss ‖ε − ε_θ(x_t, t)‖² is equivalent to…",
      options: ["Learning the score of the noised data (up to a known scale)",
                "Maximizing a discriminator", "Computing the exact likelihood", "Minimizing reconstruction of x_t"], answer: 0,
      explain: String.raw`Predicting $\varepsilon$ recovers the score via $s_\theta=-\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ — denoising and score matching are the same objective.` },
    { q: "Why is the denoiser conditioned on the timestep/noise level t?",
      options: ["Different noise levels have different score fields; sampling needs them all",
                "To speed up training", "To make it a classifier", "It isn't — t is ignored"], answer: 0,
      explain: String.raw`Each $t$ corresponds to data blurred by a different amount, with its own score. Sampling traverses all levels, so the network must know which level it's at.` },
    { q: "Why don't score-based models need the normalizing constant Z of p(x)?",
      options: ["∇log p = ∇log p̃ since ∇log Z = 0 — the score is normalizer-free",
                "Because Z = 1", "Because the data is Gaussian", "They do need Z"], answer: 0,
      explain: String.raw`Taking the gradient of $\log p=\log\tilde p-\log Z$ kills the constant $Z$, so the score ignores it — the same reason MCMC only needs the unnormalized density.` }
  ],

  practice: [
    { level: "easy", prompt: "For a 1-D Gaussian p(x) = N(μ, σ²), compute the score ∇log p(x). Which way does it point?",
      solution: String.raw`$\log p=-\frac{(x-\mu)^2}{2\sigma^2}+\text{const}$, so $\nabla\log p=\frac{\mu-x}{\sigma^2}$ — it points from $x$ back toward the mean $\mu$, with strength growing the farther you are. That's the 1-D version of 'toward the data'.` },
    { level: "easy", prompt: "Increase σ in the code to 1.0 and re-check the cosine similarity. What changes and why?",
      solution: String.raw`With larger $\sigma$ the noised distribution is smoother, so its score is coarser and easier to match — the learned field aligns broadly but loses fine structure near the modes. Small $\sigma$ gives sharp but harder-to-learn scores. This is why diffusion uses a <em>range</em> of $\sigma$.` },
    { level: "med", prompt: "Show that predicting x₀ and predicting ε are equivalent parameterizations of the same denoiser.",
      solution: String.raw`Given $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$, either target determines the other: $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\hat\varepsilon)/\sqrt{\bar\alpha_t}$. So an $x_0$-predictor and an $\varepsilon$-predictor encode the same information; DDPM found $\varepsilon$-prediction trains more stably, but they're the same model reparameterized.` },
    { level: "med", prompt: "Use Tweedie's formula to write the score in terms of the optimal denoiser E[x₀|x_t]. Why is 'best denoise' literally the score?",
      solution: String.raw`Tweedie: $\nabla\log q(x_t)=\frac{\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]-x_t}{1-\bar\alpha_t}$. The score is the (rescaled) difference between the posterior-mean clean estimate and the current noisy point — i.e. the denoising direction. Estimating $\mathbb E[x_0\mid x_t]$ and estimating the score are the same task.` },
    { level: "hard", prompt: "Explicit score matching minimizes E‖s_θ − ∇log p‖², which needs the unknown ∇log p. Explain how denoising score matching removes that dependence.",
      solution: String.raw`Vincent (2011) showed $\mathbb E_{x}\lVert s_\theta(x)-\nabla\log q_\sigma(x)\rVert^2$ equals $\mathbb E_{x_0,\tilde x}\lVert s_\theta(\tilde x)-\nabla\log q_\sigma(\tilde x\mid x_0)\rVert^2$ plus a constant. The conditional score $\nabla\log q_\sigma(\tilde x\mid x_0)=-\varepsilon/\sigma$ is <em>known</em>, so DSM turns an objective involving the intractable $\nabla\log q_\sigma$ into a simple regression on the added noise — tractable and exactly what diffusion trains.` },
    { level: "hard", prompt: "Connect the score to energy-based models: if p(x) ∝ exp(−E(x)), what is the score, and what does that say about score models vs. EBMs?",
      solution: String.raw`$\nabla\log p=-\nabla E(x)$ — the score is the negative gradient of the energy, i.e. the 'force' pulling samples toward low energy. Score models learn this force field directly (no normalizer), while EBMs learn $E$ and pay the intractable $Z$. Score-based diffusion is, in effect, a normalizer-free way to fit and sample an energy landscape — which is why its sampler (next lesson) is Langevin dynamics, the classic EBM sampler.` }
  ],

  deepDive: String.raw`<p><strong>Two research lines, one algorithm.</strong> DDPM arrived from the variational/denoising side ('predict the noise, minimize an ELBO') and NCSN/score-SDE from the score-matching side ('estimate $\nabla\log p$ at many noise scales, then sample with Langevin'). Song &amp; Ho's work showed they're the same model: $\varepsilon$-prediction and noise-conditioned score estimation differ only by a known scaling. That unification is why 'diffusion model' and 'score-based model' are used interchangeably.</p>
<p><strong>Why noise-conditioning is essential.</strong> The clean data's score is undefined off its (often low-dimensional) support — a network can't learn a gradient toward a set it never samples near. Adding noise smears the data over all of space, giving a well-defined score everywhere; using a <em>ladder</em> of noise levels means the sampler can start in the smooth, high-noise regime and anneal down to the sharp, low-noise data. The time embedding $t$ is what lets one network hold this whole ladder.</p>
<p><strong>The payoff is the next lesson.</strong> Once you have $s_\theta(x,t)\approx\nabla\log q(x_t)$, you have the gradient field pointing toward data at every noise level — everything needed to walk noise back into data. Sampling is just following that field downhill (with a dash of randomness): Langevin dynamics and the reverse diffusion process, which also closes the loop back to the MCMC lesson.</p>`
};

/* ------------------------------------------------------------------ 16.5 */
window.LESSON_CONTENT["16.5"] = {
  subtitle: "Generate by running the forward process backwards: start from pure noise and repeatedly step toward data along the score, with a dash of randomness. It's Langevin dynamics — MCMC with the learned score.",

  aiMoment: String.raw`This is where an image appears out of static. Diffusion sampling starts from a draw of pure Gaussian noise and, guided by the learned score, denoises it step by step until a clean sample emerges. The update rule is <strong>Langevin dynamics</strong> — the exact MCMC idea from Track 15, now powered by the neural score instead of a hand-written density. So the sampler that paints a Stable Diffusion image is the same 'follow the log-density gradient, add a little noise' loop you used to sample a coin's posterior. Understanding it demystifies the whole generation process — and every speed/quality trick (DDIM, guidance) is a modification of it.`,

  plainEnglish: String.raw`To create something new, begin with random static. Look at where the score says 'data is this way' and take a small step in that direction, then jiggle slightly. Repeat. Early steps use the coarse, high-noise score (find roughly where data is); later steps use the sharp, low-noise score (snap onto the fine details). After enough steps the static has been reorganized into a fresh, realistic sample.`,

  intuition: String.raw`Following the score alone is gradient ascent on log-density — it marches every point to the nearest peak, so all samples would collapse onto a few modes. Adding calibrated noise at each step (Langevin dynamics) turns that march into genuine <em>sampling</em>: you visit each region in proportion to its probability instead of piling up on the peaks. Doing it while <strong>annealing</strong> the noise from high to low is what lets samples first spread out to find all the modes, then settle precisely onto the data — which is why diffusion covers modes so well where a GAN might collapse.`,

  formalism: String.raw`<strong>Langevin dynamics</strong> samples $p$ using only its score:
$$x\leftarrow x+\tfrac{\eta}{2}\,\nabla\log p(x)+\sqrt{\eta}\,z,\qquad z\sim\mathcal N(0,I),$$
whose stationary distribution is $p$ as $\eta\to0$. <strong>Annealed</strong> Langevin sweeps noise levels high$\to$low, using $s_\theta(x,\sigma)$ at each. The <strong>reverse diffusion</strong> (DDPM) step is the special structured version:
$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\Big)+\sigma_t z.$$
In continuous time it's Anderson's reverse-time SDE, $dx=\big[-\tfrac12\beta x-\beta\,\nabla\log q_t(x)\big]dt+\sqrt\beta\,d\bar w$ — the forward SDE run backward, steered by the score.`,

  derivation: String.raw`<strong>Why score + noise = sampling, and why we anneal.</strong>
<ol>
<li><strong>Ascent alone collapses.</strong> $x\leftarrow x+\tfrac{\eta}{2}\nabla\log p(x)$ is gradient ascent on $\log p$; it converges to a mode (a MAP point). Every run from anywhere in a basin lands on the same peak — that's optimization, not sampling.</li>
<li><strong>Add the right noise.</strong> Injecting $\sqrt{\eta}\,z$ makes the update a discretization of the Langevin SDE $dx=\tfrac12\nabla\log p\,dt+dw$, whose stationary distribution is exactly $p$. The precise noise scale $\sqrt{\eta}$ (matched to the $\tfrac{\eta}{2}$ drift) is what balances 'climb toward data' against 'explore', so long-run visits match $p$. This is the score-powered cousin of Metropolis/HMC from 15.3.</li>
<li><strong>Anneal to avoid traps.</strong> At low noise the score is near-zero in the empty gaps between modes, so a cold Langevin chain started from noise can't find the data. Starting at <em>high</em> noise (where the smoothed score points broadly toward the data mass) and lowering the level gradually guides samples into the right regions first, then sharpens — Song &amp; Ermon's annealed Langevin.</li>
<li><strong>Reverse diffusion is the structured case.</strong> Undoing the forward chain one step needs the posterior $q(x_{t-1}\mid x_t)$, which for small steps is Gaussian with a mean fixed by the score; substituting $s_\theta=-\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ gives the DDPM update above. The code runs annealed Langevin with the (analytic) score and turns noise into the three data modes.</li>
</ol>`,

  code: [
    { label: "Annealed Langevin: turn pure noise into samples using the score",
      src: String.raw`import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# Use the (analytic) score of a 3-mode Gaussian mixture to sample it from noise.
# (In a real diffusion model this score is the trained network s_theta(x, t).)
mus = np.array([[-1.5,-1.0],[1.5,-0.5],[0.0,1.6]]); wts = np.array([.4,.35,.25]); s2 = 0.25
def score(x):                                          # ∇log p(x) for the mixture
    d = ((x[:,None,:]-mus[None])**2).sum(-1); lw = np.log(wts) - 0.5*d/s2
    lw -= lw.max(1,keepdims=True); r = np.exp(lw); r /= r.sum(1,keepdims=True)
    return np.einsum('nk,nkd->nd', r, (mus[None]-x[:,None,:])/s2)

x = rng.standard_normal((500,2))*2.0                   # start from pure noise
for sig in np.linspace(1.2, 0.05, 20):                 # anneal noise level high -> low
    step = 0.15*sig**2
    for _ in range(15):
        x = x + step*score(x) + np.sqrt(2*step)*rng.standard_normal(x.shape)   # Langevin update
print("sample mean:", np.round(x.mean(0), 2), "  (data mean:", np.round((wts[:,None]*mus).sum(0), 2), ")")

gx, gy = np.meshgrid(np.linspace(-3,3,20), np.linspace(-3,3,20)); G = np.c_[gx.ravel(), gy.ravel()]; S = score(G)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].quiver(gx, gy, S[:,0].reshape(gx.shape), S[:,1].reshape(gx.shape), color='#6a51a3', alpha=.7)
ax[0].scatter(mus[:,0], mus[:,1], color='#d1495b', s=60, marker='*'); ax[0].set_title('the score field (the "denoiser")'); ax[0].set_aspect('equal')
ax[1].scatter(x[:,0], x[:,1], s=6, color='#3a7d44', alpha=.5); ax[1].scatter(mus[:,0], mus[:,1], color='#d1495b', s=60, marker='*')
ax[1].set_title('annealed Langevin: noise → samples'); ax[1].set_xlim(-3,3); ax[1].set_ylim(-3,3); ax[1].set_aspect('equal')
plt.tight_layout()` }
  ],

  diagram: String.raw`<svg viewBox="0 0 720 168" width="100%" style="max-width:720px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" font-size="11">
  <defs><marker id="r5a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#3a7d44"/></marker></defs>
  <rect x="14" y="44" width="104" height="44" rx="6" fill="#efeaf0" stroke="#8a7a99"/><text x="66" y="64" text-anchor="middle" fill="#3a2347" font-weight="700">x_T</text><text x="66" y="80" text-anchor="middle" fill="#6a51a3" font-size="10">noise ≈ N(0,I)</text>
  <rect x="178" y="44" width="70" height="44" rx="6" fill="#f0eef2" stroke="#9a8aa8"/><text x="213" y="70" text-anchor="middle" fill="#4b3a63">x_t</text>
  <text x="300" y="70" text-anchor="middle" fill="#6b7a99" font-size="16">· · ·</text>
  <rect x="352" y="44" width="70" height="44" rx="6" fill="#eceef1" stroke="#7a8aa0"/><text x="387" y="70" text-anchor="middle" fill="#3a4a63">x₁</text>
  <rect x="470" y="44" width="104" height="44" rx="6" fill="#dbe8fb" stroke="#2a6f97"/><text x="522" y="64" text-anchor="middle" fill="#123a5a" font-weight="700">x₀</text><text x="522" y="80" text-anchor="middle" fill="#2a6f97" font-size="10">data</text>
  <g stroke="#3a7d44">
   <line x1="118" y1="66" x2="176" y2="66" marker-end="url(#r5a)"/><line x1="248" y1="66" x2="286" y2="66" marker-end="url(#r5a)"/>
   <line x1="314" y1="66" x2="350" y2="66" marker-end="url(#r5a)"/><line x1="422" y1="66" x2="468" y2="66" marker-end="url(#r5a)"/></g>
  <text x="147" y="36" text-anchor="middle" fill="#245030" font-size="9.5">denoise</text>
  <text x="360" y="112" text-anchor="middle" fill="#1f2a44">each step:  x ← x + (η/2)·∇log p(x) + √η·z      (follow the score, add a little noise)</text>
  <text x="360" y="134" text-anchor="middle" fill="#5a6b8c" font-size="10.5">the reverse of forward diffusion — Langevin dynamics driven by the learned score (this is MCMC)</text>
</svg>`,

  keyPoints: [
    String.raw`Generate by walking noise back to data: repeatedly step along the score and add calibrated noise — the <strong>Langevin</strong> update $x\leftarrow x+\tfrac{\eta}{2}\nabla\log p+\sqrt{\eta}\,z$.`,
    String.raw`Score-following alone is gradient ascent and collapses to modes; the added noise turns optimization into <em>sampling</em> (visiting regions in proportion to probability).`,
    String.raw`<strong>Annealing</strong> the noise high$\to$low lets samples find all the modes first, then sharpen — the reason diffusion covers modes far better than a GAN.`,
    String.raw`Reverse diffusion (DDPM) is the structured version: $x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\big)+\sigma_t z$ — one denoise step of the learned chain.`,
    String.raw`It's exactly the MCMC of Track 15 with a <em>learned</em> score: 'sample an unnormalized distribution via its log-gradient' now generates images.`
  ],

  commonMistakes: [
    { wrong: "Dropping the noise term to 'clean up' the sample (pure gradient ascent).",
      why: String.raw`Without the $\sqrt{\eta}z$ term you do MAP optimization, not sampling — every run collapses to a mode and you lose all diversity. The calibrated noise is what makes the long-run distribution equal $p$ rather than a peak.` },
    { wrong: "Running Langevin at a single (low) noise level from a noise start.",
      why: String.raw`At low noise the score is ~0 in the empty gaps, so a chain starting far from data never gets a useful gradient and stalls. You must anneal from high noise (broad, informative score) down to low — the whole point of noise-conditioned scores.` },
    { wrong: "Expecting few-step sampling to look like many-step sampling for free.",
      why: String.raw`The Langevin/reverse-SDE guarantees hold as the step size $\to0$ (many steps). Taking big steps to save time introduces discretization error (blur, artifacts). Fast samplers (DDIM, higher-order/ODE solvers) are careful schemes to cut steps while controlling that error — not just 'use a bigger step'.` }
  ],

  quiz: [
    { q: "The Langevin sampling update is…",
      options: ["x ← x + (η/2)∇log p(x) + √η·z", "x ← x − η∇log p(x)", "x ← ∇log p(x)", "x ← x + η·z"], answer: 0,
      explain: String.raw`Score drift $\tfrac{\eta}{2}\nabla\log p$ plus noise $\sqrt{\eta}\,z$; its stationary distribution is $p$ as $\eta\to0$.` },
    { q: "Why add noise instead of pure gradient ascent on log p?",
      options: ["Ascent collapses to a mode (MAP); noise makes it sample the whole distribution",
                "Noise speeds convergence", "To match the discriminator", "Ascent diverges"], answer: 0,
      explain: String.raw`Deterministic ascent finds one peak; the calibrated noise makes visitation proportional to probability — genuine sampling, not optimization.` },
    { q: "What is the purpose of annealing the noise level from high to low during sampling?",
      options: ["Broad high-noise scores locate the modes first; low-noise scores then sharpen onto data",
                "To reduce compute", "To normalize the density", "To train the network"], answer: 0,
      explain: String.raw`Low-noise scores vanish between modes, so a cold start can't find data; annealing uses the smooth high-noise score to get close, then refines.` },
    { q: "Diffusion sampling is essentially which classical algorithm, with a learned score?",
      options: ["Langevin MCMC", "k-means", "the EM algorithm", "gradient descent on weights"], answer: 0,
      explain: String.raw`It's Langevin dynamics (an MCMC method) driven by $s_\theta\approx\nabla\log p$ — the same 'sample via the log-gradient' idea as Track 15's MCMC.` },
    { q: "In the DDPM reverse step, what plays the role of ∇log p(x)?",
      options: ["The trained noise predictor ε_θ(x_t, t) (since s_θ = −ε_θ/√(1−ᾱ_t))",
                "The discriminator", "The data mean", "The KL term"], answer: 0,
      explain: String.raw`The network's $\varepsilon$-prediction is the score up to a known scale, so the reverse step is Langevin/ancestral sampling steered by $\varepsilon_\theta$.` }
  ],

  practice: [
    { level: "easy", prompt: "Remove the noise term (set it to zero) in the sampling loop. What happens to the 500 samples?",
      solution: String.raw`They collapse onto the three mode centers (and mostly the nearest one) — pure gradient ascent gives MAP points, not samples. Diversity within each mode disappears, demonstrating why the Langevin noise is essential.` },
    { level: "easy", prompt: "Skip the annealing: run Langevin at a single small σ from the noise start. Describe the failure.",
      solution: String.raw`Many samples get stranded in the low-density gaps where the score is ~0, never reaching the modes; the sample cloud looks diffuse and wrong. It shows why noise-conditioned, annealed sampling is required.` },
    { level: "med", prompt: "Relate the annealed-Langevin loop to the reverse diffusion (DDPM) loop. What corresponds to what?",
      solution: String.raw`Each noise level $\sigma$ (or timestep $t$) is one rung; the score $s_\theta(x,\sigma)$ ↔ $-\varepsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$; the Langevin drift+noise ↔ the DDPM posterior-mean step + $\sigma_t z$. Reverse diffusion is annealed Langevin with the schedule and step sizes fixed by the forward process.` },
    { level: "med", prompt: "Explain how classifier-free guidance biases sampling toward a text prompt by modifying the score.",
      solution: String.raw`Guidance replaces the score with $\nabla\log p(x)+w\,[\nabla\log p(x\mid c)-\nabla\log p(x)]$ — pushing samples in the direction that increases the conditional (prompt-matching) log-density more than the unconditional. Larger $w$ = stronger prompt adherence at some cost to diversity/fidelity. It's a score edit, which is why it slots directly into this sampler.` },
    { level: "hard", prompt: "State the probability-flow ODE and why it enables deterministic, fast sampling and exact likelihoods.",
      solution: String.raw`Every diffusion SDE has a companion ODE, $dx=[-\tfrac12\beta x-\tfrac12\beta\nabla\log q_t(x)]dt$, whose trajectories have the <em>same</em> marginals $q_t$ but no noise. Integrating it deterministically (with a fast ODE solver) samples in few steps and, via the instantaneous change-of-variables, yields exact log-likelihoods — the basis of DDIM and modern fast samplers.` },
    { level: "hard", prompt: "Derive the reverse-time SDE intuition: why does running the forward SDE backward require the score?",
      solution: String.raw`Anderson's theorem: if the forward process is $dx=f\,dt+g\,dw$, the time-reversal is $dx=[f-g^2\nabla\log q_t(x)]\,dt+g\,d\bar w$. The extra $-g^2\nabla\log q_t$ term is the correction that turns 'spread out into noise' into 'concentrate back onto data' — and it's exactly the score. So reversing diffusion is impossible without knowing $\nabla\log q_t$, which is precisely what the denoiser learned in 16.4.` }
  ],

  deepDive: String.raw`<p><strong>The loop finally closes.</strong> Track 15's MCMC lesson ended by noting that 'sample from an unnormalized distribution using its log-gradient' is exactly what a diffusion model does at generation time. Here it is literally: the sampler is Langevin dynamics, and the log-gradient is the neural score. Diffusion is the marriage of two threads of this course — variational/score learning (how to fit the model) and MCMC (how to sample it) — with a neural network supplying the one ingredient classical MCMC lacked: a good score for complex, high-dimensional data.</p>
<p><strong>Speed is the active frontier.</strong> Vanilla DDPM needs hundreds to thousands of denoising steps because the Langevin/SDE guarantees are asymptotic in step size. DDIM (deterministic, non-Markovian), the probability-flow ODE with high-order solvers, and distillation methods (progressive/consistency models) cut this to tens or even one step. Every one of them is a smarter way to integrate the same score field — the physics is fixed, the numerics improve.</p>
<p><strong>Control lives in the score.</strong> Because generation is 'follow the score', you steer it by editing the score: conditioning on a class or text (classifier-free guidance), inpainting (fix known pixels, denoise the rest), and image editing (start from a partially-noised real image) are all score/initialization edits on this same loop. That composability — one trained score, many controllable samplers — is a big part of why diffusion took over generative modeling, and it's the natural bridge into the final track on the neural architectures (like the U-Net and transformer denoisers) that produce these scores at scale.</p>`
};

/* ------------------------------------------------------------------ 16.E */
window.LESSON_CONTENT["16.E"] = {
  exam: true,
  intro: String.raw`Ten problems across VAEs, GANs, and diffusion — roughly easy → hard. Budget about <strong>75 minutes</strong>. Reason on paper; use a REPL only to <em>check</em> (a JS curve, a score field, a sampling loop). Keep sight of the unifying idea: every model here minimizes a divergence to the data distribution, and diffusion does it by turning one hard step into many easy denoising steps. Solutions and a rubric follow.`,
  problems: [
    { level: "easy", prompt: "VAE basics. (a) What are the two terms of the VAE loss? (b) How do you generate a new sample from a trained VAE?",
      solution: String.raw`(a) Reconstruction error $+\ \mathrm{KL}(q_\phi(z\mid x)\Vert\mathcal N(0,I))$ — i.e. the negative ELBO. (b) Draw $z\sim\mathcal N(0,I)$ and run the decoder; the encoder is only used in training.` },
    { level: "easy", prompt: "Reparameterization. Why write z = μ + σ⊙ε instead of sampling z ~ N(μ,σ²) directly?",
      solution: String.raw`So gradients reach $\mu,\sigma$: with $\varepsilon\sim\mathcal N(0,I)$ fixed, $z$ is a differentiable function of the encoder outputs, and backprop can train the encoder. Direct sampling blocks the gradient.` },
    { level: "med", prompt: "GAN theory. For a fixed generator, (a) what is the optimal discriminator D*(x)? (b) What divergence does the generator minimize once D* is substituted, and where is its minimum?",
      solution: String.raw`(a) $D^\*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_{\text{gen}}(x)}$. (b) $V(G,D^\*)=2\,\mathrm{JS}(p_{\text{data}}\Vert p_{\text{gen}})-2\log2$, minimized (JS$=0$) exactly at $p_{\text{gen}}=p_{\text{data}}$, where $D^\*\equiv\tfrac12$.` },
    { level: "med", prompt: "Forward diffusion. (a) Give the closed-form q(x_t | x₀). (b) What distribution does x_T approach and why does that matter?",
      solution: String.raw`(a) $\mathcal N(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$, i.e. $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$. (b) $x_T\to\mathcal N(0,I)$ (since $\bar\alpha_t\to0$), which is trivially samplable — sampling starts there.` },
    { level: "med", prompt: "Score. (a) Define the score of p(x). (b) Compute it for a 1-D Gaussian N(μ, σ²).",
      solution: String.raw`(a) $s(x)=\nabla_x\log p(x)$, a vector field pointing toward higher density. (b) $\nabla\log\mathcal N(x;\mu,\sigma^2)=\frac{\mu-x}{\sigma^2}$ — toward the mean, strength growing with distance.` },
    { level: "med", prompt: "Denoising = score. For x̃ = x₀ + σε, what is ∇log q(x̃|x₀), and why does the DDPM loss ‖ε − ε_θ‖² amount to learning the score?",
      solution: String.raw`$\nabla_{\tilde x}\log q(\tilde x\mid x_0)=\frac{x_0-\tilde x}{\sigma^2}=-\varepsilon/\sigma$. Fitting $s_\theta$ to this (denoising score matching) recovers $\nabla\log q(\tilde x)$ in expectation; equivalently predicting $\varepsilon$ gives the score via $s_\theta=-\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$. Denoising and score matching are one objective.` },
    { level: "hard", prompt: "Sampling. (a) Write the Langevin update. (b) Why is the noise term essential? (c) Why anneal the noise level from high to low?",
      solution: String.raw`(a) $x\leftarrow x+\tfrac{\eta}{2}\nabla\log p(x)+\sqrt{\eta}\,z$, $z\sim\mathcal N(0,I)$. (b) Without noise it's gradient ascent → collapses to a mode (MAP), no diversity; the calibrated noise makes the stationary distribution $p$. (c) Low-noise scores vanish in the gaps between modes, so a cold start can't find data; high→low annealing locates modes with the smooth score first, then sharpens.` },
    { level: "hard", prompt: "GAN derivation. Derive the optimal discriminator and show V(G, D*) = 2·JS(p_data ‖ p_gen) − 2log2.",
      solution: String.raw`Maximize $p_{\text{data}}\log D+p_{\text{gen}}\log(1-D)$ pointwise: $D^\*=\frac{p_{\text{data}}}{p_{\text{data}}+p_{\text{gen}}}$. Substitute and factor $p_{\text{data}}+p_{\text{gen}}=2\cdot\frac{p_{\text{data}}+p_{\text{gen}}}{2}$: the expression becomes $-2\log2+\mathrm{KL}(p_{\text{data}}\Vert m)+\mathrm{KL}(p_{\text{gen}}\Vert m)$ with $m=\tfrac{p_{\text{data}}+p_{\text{gen}}}{2}$, and the two KLs sum to $2\,\mathrm{JS}$.` },
    { level: "med", prompt: "Efficiency. Why is the forward process not learned, and how does its closed form make training cheap?",
      solution: String.raw`The forward process is a fixed noising schedule with no parameters — all learning is in reversing it. Its closed form $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ lets you jump to any timestep in one operation, so training just samples a random $t$ per example, noises it once, and regresses the denoiser — no chain simulation.` },
    { level: "hard", prompt: "Big picture. Compare VAE, GAN, and diffusion on likelihood, sample sharpness, training stability, mode coverage, and sampling speed — and name the classical algorithm diffusion sampling reduces to.",
      solution: String.raw`VAE: likelihood-based (ELBO), stable, but blurry / mode-bridging. GAN: no explicit likelihood, sharp, unstable and prone to mode collapse. Diffusion: likelihood-based (an ELBO over the chain), stable, sharp, excellent mode coverage — but slow (many denoising steps). Diffusion sampling is <strong>Langevin MCMC</strong> driven by the learned score, so it inherits Track 15's 'sample an unnormalized distribution via its log-gradient' — now generating images.` }
  ],
  rubric: String.raw`<ul>
<li><strong>9–10:</strong> You can derive the GAN divergence, the diffusion forward marginal, and the denoising=score identity, and you understand why sampling is Langevin MCMC. You can read a modern generative-models paper's math.</li>
<li><strong>7–8:</strong> Strong. Revisit whichever slipped: the ELBO=recon−KL framing (16.1), the optimal-D derivation (16.2), or Tweedie/DSM (16.4).</li>
<li><strong>5–6:</strong> The ideas are landing but not fluent. Re-run the forward-diffusion (16.3) and annealed-Langevin (16.5) code, changing one knob at a time until the behavior is predictable.</li>
<li><strong>Below 5:</strong> Rework the track in order. The spine is simple — turn data into noise (fixed), learn to denoise (= the score), then follow the score back — anchor there and rebuild.</li>
</ul>`
};

