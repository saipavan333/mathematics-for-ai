/* ============================================================
   "In one breath" — the single simplest sentence for every lesson.
   The gentlest on-ramp: what a mentor would say before any math.
   Rendered as a callout at the top of each lesson by app.js.
   ============================================================ */
window.NUTSHELLS = {
  /* Track 1 — Arithmetic & Algebra */
  "1.1": "A logarithm just asks 'what power do I raise the base to?' — and its magic trick is turning multiplication into addition, which is the only reason a model can multiply a thousand tiny probabilities without the answer collapsing to zero.",
  "1.2": "Σ and Π are shorthand for 'add up all of these' and 'multiply all of these' — and since every loss you'll ever minimize is a Σ over your training examples, reading them fluently is like reading music before you play.",
  "1.3": "A function is a machine: put a number in, get a number out. A neural network is thousands of these machines bolted together, so getting comfortable with 'input → rule → output' now is getting comfortable with the whole field.",
  "1.4": "Big-O ignores the small stuff and asks one thing: as the input grows, how fast does the work grow? It's why doubling a sentence's length quadruples attention's cost — the single fact behind 'why is long context so expensive?'",

  /* Track 2 — Linear Algebra I */
  "2.1": "A vector is just a list of numbers you can picture as an arrow in space. Every word, image, and user in an AI system becomes one of these lists — so a vector is literally how a machine holds a meaning.",
  "2.2": "Adding vectors is walking one arrow then the next; scaling one stretches or shrinks it. That's all — and the 'residual connection' that lets networks be 1000 layers deep is nothing more than adding two vectors: x + f(x).",
  "2.3": "The dot product is one number saying how much two arrows point the same way. That's exactly how attention decides which words are relevant to each other — 'similar' simply means 'big dot product.'",
  "2.4": "A matrix is a machine that takes a vector in and spits a transformed vector out — rotating, stretching, or squishing space. Every layer of a neural net is one such transformation, so a 'forward pass' is just matrix × vector, over and over.",
  "2.5": "These are the everyday tools of array math: flipping a matrix (transpose), the do-nothing matrix (identity), multiplying element-by-element (Hadamard), and the auto-stretching that lets you add one bias to a whole batch at once (broadcasting).",

  /* Track 3 — Linear Algebra II */
  "3.1": "Solving Ax = b means finding the inputs that produce a wanted output, and Gaussian elimination is the tidy bookkeeping that cracks it — the same kind of problem you solve when fitting a line to data.",
  "3.2": "Rank counts how many genuinely independent directions a matrix carries — how much real information is inside it. It's why LoRA can fine-tune a giant model cheaply: the change it needs is low-rank, only a few directions.",
  "3.3": "A basis is a minimal set of directions you can build everything else from — like primary colors for space. Embedding dimensions are a learned basis for meaning: a few hundred directions that mix to represent any word.",
  "3.4": "Two vectors are orthogonal (a right angle) when they share nothing — their dot product is zero. 'Projecting' finds the closest point in a subspace, which is exactly what least-squares does: the best fit is the shadow of the data on the model.",
  "3.5": "QR splits a matrix into a clean rotation (Q) and a simple triangle (R). It's the numerically trustworthy way to fit linear regression — the same answer as the textbook formula, but without the round-off errors that formula quietly builds up.",

  /* Track 4 — Linear Algebra III */
  "4.1": "A norm is a ruler for how 'big' a vector is — and different rulers care about different things. The L2 ruler shrinks big values; the L1 ruler pushes values to exactly zero, which is why L1 gives you sparse, mostly-zero weights.",
  "4.2": "A matrix norm measures how much a transformation can amplify a vector — its maximum gain. Keeping that gain in check (spectral normalization) is how you stop a network's signals from exploding, and it's key to stable GAN training.",
  "4.3": "The inverse 'undoes' a matrix, and the pseudo-inverse does its best when a true undo is impossible. The practical lesson: you almost never actually compute an inverse — you 'solve,' which is faster and far more stable.",
  "4.4": "LU factorization turns one hard solve into two easy triangular ones. The deeper idea is the 'condition number': some problems magnify tiny errors enormously, and knowing when you're in that danger zone separates working code from code that quietly returns garbage.",

  /* Track 5 — Linear Algebra IV */
  "5.1": "An eigenvector is a special direction a matrix only stretches (never rotates), and its eigenvalue is by how much. They reveal a matrix's natural axes — and the eigenvalues of the loss's curvature tell you which directions are steep and which are flat.",
  "5.2": "Diagonalizing means finding a matrix's natural axes so it becomes pure stretching, no rotation. Symmetric matrices — like every covariance matrix — always have perfectly perpendicular axes, the mathematical gift that makes PCA work.",
  "5.3": "A positive-definite matrix curves upward like a bowl in every direction. That's the exact condition for a real minimum (not a saddle), and it's what makes covariance matrices and kernels so well-behaved.",
  "5.4": "SVD says any matrix — any transformation at all — is just a rotation, then a stretch along clean axes, then another rotation. It's the Swiss-army knife of linear algebra: compression, recommendations, and low-rank adapters all fall out of it.",
  "5.5": "PCA finds the few directions your data actually spreads along and throws the rest away — keeping the shape while using far fewer numbers. It's lossy compression with a proof that it's the best compression of its size.",

  /* Track 6 — Calculus I */
  "6.1": "A derivative is just the slope — how fast something changes at a point. Learning needs slopes: a model feels which way is downhill on its error and steps that way, so 'smooth enough to have a slope' is a requirement, not a nicety.",
  "6.2": "The chain rule gives the slope of nested functions by multiplying the slopes of each layer. That's the whole engine of deep learning — backpropagation is literally the chain rule applied from the output back to every weight.",
  "6.3": "When a function has many inputs, the gradient is the arrow pointing in the steepest-uphill direction. Training walks the opposite way, downhill, one small step at a time. Every model that ever learned anything did it by following this arrow.",
  "6.4": "The Jacobian is the gradient's big brother for functions with many outputs — a whole table of slopes — and the Hessian is the curvature, how the slope itself bends. Backprop multiplies Jacobians; curvature tells you how big a step is safe.",
  "6.5": "A Taylor expansion approximates any curvy function nearby with a simple line or parabola — zoom in far enough and everything looks straight. Gradient descent is exactly what you get by trusting that straight-line view for one small step.",

  /* Track 7 — Calculus II */
  "7.1": "An integral adds up infinitely many tiny slices to get a total — usually the area under a curve. In probability it's how a 'density' becomes an actual probability, since the whole area under a distribution must add to 1.",
  "7.2": "This is the theorem that integration and differentiation are opposites — each undoes the other. It's why, if you know how fast probability piles up (the density), you can integrate to get the running total (the CDF), and vice versa.",
  "7.3": "These are the two moves for harder integrals: cleverly rename a variable (substitution) or trade one integral for an easier one (by parts). That 'change of variables' idea is the same one behind the reparameterization trick and normalizing flows.",
  "7.4": "There's one famous bell-curve integral with a beautiful answer (√π), found by a slick polar-coordinate trick. It's the reason the Normal distribution has that 1/√(2π) out front — it's just the number that makes its total area equal 1.",
  "7.5": "An 'expectation' is a weighted average, written as an integral — and it's quietly the most important idea in ML: the loss you truly want to minimize is an average over all data, which training approximates by averaging over minibatches.",

  /* Track 8 — Optimization */
  "8.1": "At the bottom of a valley the slope is zero — but so is the top of a hill and a 'saddle' (up one way, down another). In the millions of dimensions of a neural net, the places training stalls are almost always saddles, not bad valleys.",
  "8.2": "A convex function is a single smooth bowl: wherever you start, downhill leads to the one true bottom. When your loss is convex (like logistic regression), optimization is stress-free — there are no bad local minima to fear.",
  "8.3": "Gradient descent isn't a guess — it falls out of the math: approximate the loss by a line, and 'step downhill' is provably the best move. This one update, with the gradient estimated on a minibatch, is how essentially every model trains.",
  "8.4": "Plain gradient descent zig-zags; momentum gives it a heavy ball's inertia so it rolls smoothly, and Adam adds a custom step size for every parameter. Adam is the optimizer training most of the models you've heard of.",
  "8.5": "When you must optimize under a constraint ('do your best, but stay on this line'), Lagrange multipliers turn it back into an ordinary problem. It's the machinery behind support vector machines and the KL 'leash' that keeps RLHF models in line.",

  /* Track 9 — Probability I */
  "9.1": "Probability starts by listing everything that could happen (the sample space) and giving each outcome a share that all add to 1. Three simple rules are the whole foundation — everything else is built from them.",
  "9.2": "Conditional probability is 'given that this happened, how likely is that?' — updating your belief on new information. A language model is built from exactly this: it predicts each next word given all the words so far.",
  "9.3": "Bayes' theorem is the formula for changing your mind with evidence: start with a prior belief, see data, get an updated belief. It's one line of algebra, and it underlies spam filters, medical tests, and Bayesian deep learning alike.",
  "9.4": "A random variable is a number whose value is uncertain. Its 'expectation' is the long-run average you'd get, and its 'variance' is how much it jumps around. Training a model is, at heart, minimizing an expected value.",
  "9.5": "Covariance measures whether two quantities move together. Pack all the pairwise covariances into a matrix and you get the object at the heart of PCA, whitening, and every Gaussian model — a compact summary of how your features relate.",

  /* Track 10 — Probability II */
  "10.1": "For continuous things, probability lives in a 'density' — the height of a curve — and you get real probabilities by measuring area under it. The humble uniform (equally-likely) distribution is the raw randomness that seeds dropout and shuffling.",
  "10.2": "The bell curve is the default shape of randomness in nature and in ML. Just two numbers — its center and its spread — describe it completely, and carefully-scaled Gaussians are how a network's weights are first born.",
  "10.3": "A small zoo of handy distributions: the exponential for waiting times, the Beta for a probability that is itself uncertain, and the Dirichlet for a whole set of probabilities that must sum to 1 (like a mix of topics).",
  "10.4": "The bell curve in many dimensions — a fuzzy ellipsoidal cloud described by a center and a covariance matrix. The latent space of a VAE and the whole idea of a Gaussian process live inside this one distribution.",
  "10.5": "When you transform a random variable, its density stretches and squishes by the Jacobian. The reparameterization trick uses this to write randomness as 'mean + noise × spread' — the clever move that lets a VAE learn through a random step.",
  "10.6": "The Law of Large Numbers says averages settle down as you gather more samples; the Central Limit Theorem says those averages turn bell-shaped. Together they're why a noisy minibatch gradient is still a trustworthy estimate of the true one.",

  /* Track 11 — Statistics */
  "11.1": "An estimator is a recipe for guessing an unknown number from data. 'Bias' is how wrong it is on average; 'variance' is how much it wobbles from run to run. A model's error splits neatly into exactly these pieces, plus unavoidable noise.",
  "11.2": "Maximum likelihood picks the parameters that make your observed data look most likely — 'whatever explains what I saw best.' The punchline: the cross-entropy loss you train classifiers with is exactly this, in disguise.",
  "11.3": "MAP is maximum likelihood with an opinion added — a prior belief about the answer before you see data. A 'weights should be small' prior gives you L2 regularization for free; a 'weights should be sparse' prior gives L1.",
  "11.4": "Too simple a model misses the pattern (bias); too complex a model memorizes the noise (variance). The whole art of ML is balancing the two — and this one curve is why model size and regularization matter so much.",
  "11.5": "This is the toolkit for asking 'is this real, or just luck?' — p-values, confidence intervals, and the bootstrap (re-sampling your data to feel out uncertainty). Comparing two models' scores honestly is exactly this kind of test.",

  /* Track 12 — Information Theory */
  "12.1": "Entropy measures surprise — how uncertain an outcome is, in bits. A fair coin is 1 bit; a loaded one is less. A language model's 'perplexity' is just its entropy dressed up: how many equally-likely choices it feels it's guessing among.",
  "12.2": "Cross-entropy is how many extra bits you waste when your predicted distribution is wrong; KL divergence is that waste itself — the gap between two distributions. Training a classifier is literally shrinking this gap to the true labels.",
  "12.3": "Mutual information measures how much knowing one thing tells you about another — shared information, in bits. It's zero for unrelated things, and maximizing it is the quiet goal behind contrastive self-supervised learning.",
  "12.4": "Jensen's inequality is a simple fact about curves that lets you swap an impossible calculation for a solvable lower bound. That bound — the ELBO — is the exact objective a VAE trains on, so this small inequality powers modern generative models.",

  /* Track 13 — Numerical Methods */
  "13.1": "Computers can't store most numbers exactly — they round. Knowing where those tiny errors hide (and why bfloat16 trades precision for range) is what lets you train huge models without them silently blowing up or grinding to zero.",
  "13.2": "Some correct formulas give wrong answers on a computer because they overflow or cancel catastrophically. The log-sum-exp trick — subtract the max first — is the classic fix, and it's why every softmax you've ever used quietly subtracts a constant.",
  "13.3": "LU, Cholesky, and QR are three ways to break a matrix apart to solve problems fast and stably. This is the practical 'which tool when' guide — the judgment that keeps linear-algebra code both quick and correct.",
  "13.4": "Autodiff is how frameworks get exact gradients automatically, with no hand calculus. The key insight: computing all the gradients backward costs about the same as one forward pass — the entire reason training deep nets is affordable.",
  "13.5": "Gradient checking is the cheap sanity test — compare your gradient to a nudge-and-measure estimate — that catches backprop bugs instantly. Mixed precision is doing the heavy math in 16 bits to train faster without losing accuracy.",

  /* Track 14 — Matrix Calculus & Backprop */
  "14.1": "When you take derivatives with vectors and matrices, the only real trick is bookkeeping: the gradient always has the same shape as the thing you differentiated. Nail that one rule and matrix calculus stops being scary.",
  "14.2": "A small handful of patterns (like 'the derivative of aᵀx is a') covers almost every gradient in ML. Learn them the way you learned 'the derivative of x² is 2x' — by recognizing the shape, not re-deriving each time.",
  "14.3": "The chain rule for vectors is just multiplying each layer's sensitivity table (its Jacobian) together. Do that multiplication starting from the output and you get backprop — which is why one backward pass produces every gradient at once.",
  "14.4": "Deriving the gradients of a single dense layer by hand is the moment backprop clicks: a weight's gradient is simply its input times the incoming error. See it once here and you can read the backward pass of any network.",
  "14.5": "Softmax and cross-entropy look messy alone, but together their gradient collapses to the most elegant result in ML: predicted minus target, 'p − y.' It's why the two are always paired, and why every classifier's output layer is so clean.",

  /* Track 15 — Bayesian Inference */
  "15.1": "Bayesian updating means holding a belief as a whole distribution and sharpening it with each new data point. In the lucky 'conjugate' case, updating is pure arithmetic — you just add your heads and tails to the prior's counts.",
  "15.2": "Instead of one best-fit line, keep every plausible line and see how much they disagree. The payoff is honest error bars that grow wide exactly where you have no data — a model that knows what it doesn't know.",
  "15.3": "When the answer is a distribution too complicated to write down, MCMC draws samples from it anyway — using only its shape, never its impossible-to-compute total. It's a guided random walk that visits each region as often as it's probable.",
  "15.4": "Instead of chasing an impossible distribution exactly, pick a simple stand-in and tune it to fit as closely as possible. The quantity you maximize to do that — the ELBO — is exactly the loss that trains a VAE.",
  "15.5": "A Gaussian process puts a probability distribution over whole functions, not just numbers. Feed it a few points and it returns a smooth best-guess curve plus a shaded band of uncertainty that swells wherever data is sparse.",

  /* Track 16 — Diffusion & Generative Models */
  "16.1": "A VAE squeezes data down to a small 'latent code' and learns to rebuild it — while keeping those codes shaped like a simple bell curve. That last part is the magic: afterward you can draw a random code and decode brand-new data.",
  "16.2": "A GAN is a forger and a detective locked in competition: the generator makes fakes, the discriminator tries to catch them, and each pushes the other to improve. When the detective can only guess, the fakes have become indistinguishable from real.",
  "16.3": "Take a clean image and slowly stir in noise until it's pure static — that's the (unlearned) forward process. One tidy formula lets you jump straight to any noise level, which is what makes training a diffusion model practical.",
  "16.4": "To reverse the noise, a network learns to look at a fuzzy point and say 'the clean data was that way.' Astonishingly, learning to denoise is the same as learning the 'score' — the field of arrows pointing toward where real data lives.",
  "16.5": "Generation is running the noise backward: start from static and repeatedly nudge toward data along the learned arrows, with a dash of randomness. It's the exact same 'follow the gradient' idea as MCMC — now painting images.",

  /* Track 17 — Neural-Net & Transformer Math */
  "17.1": "Stack straight lines and you still get a straight line, so the 'kinks' between layers (activations) are what make depth powerful. But those same kinks decide whether the learning signal survives the trip back — the whole vanishing-gradient story.",
  "17.2": "A convolution is a dense layer with two rules baked in: only look at a small patch, and reuse the same weights everywhere. Those two constraints are the entire 'vision' inductive bias — and they run as a plain matrix multiply.",
  "17.3": "As numbers flow through a deep net they drift in scale and destabilize training. Normalization re-centers them at each layer; the only difference between the two flavors is which direction you average over, and LayerNorm's choice is why transformers use it.",
  "17.4": "Attention lets every word ask a question and gather answers from every other word, weighted by relevance — a soft, content-based lookup across the whole sentence. It's the single mechanism at the heart of every transformer.",
  "17.5": "Run several attentions in parallel (each watching for a different kind of relationship), then wrap attention and a small network in the 'add-the-input-back' trick and normalization. Stack that block dozens of times and you have GPT.",

  /* Capstones */
  "C.1": "Build PCA with nothing but NumPy and watch it squeeze 64-pixel digits down to a handful of numbers — then rebuild them. Real compression, from the eigenvectors up, running live in your browser.",
  "C.2": "Code all three optimizers by hand and race them across a tricky valley, seeing exactly why momentum and Adam beat plain gradient descent. After this, your optimizer is no longer a black box.",
  "C.3": "Build a classifier that learns what each class 'looks like' and uses Bayes' rule to decide — then watch a good prior rescue it when data is scarce. The whole generative-vs-discriminative story in one runnable project.",
  "C.4": "Build a tiny language model and measure it with the very quantities that train GPT — cross-entropy, KL, perplexity, mutual information — proving the famous identity CE = H + KL with real numbers.",
  "C.5": "Build the core of PyTorch — a reverse-mode autograd engine — in about 40 lines, check it against finite differences, then train a real 2-layer network with it. After this, loss.backward() is never magic again."
};
