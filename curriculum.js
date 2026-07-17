/* ============================================================
   Curriculum map — 13 tracks + capstones.
   Titles, per-lesson syllabus ("covers"), and AI anchor.
   Authored lesson bodies live in content/trackNN.js.
   ============================================================ */
window.CURRICULUM = [
  {
    id: "1", short: "Arithmetic & Algebra", title: "Arithmetic & Algebra Refresher",
    lessons: [
      { id: "1.1", title: "Fractions, Exponents & Logarithms",
        covers: ["Fractions & the rules that survive into ML", "Exponent laws", "Logarithms as the inverse of exponentials", "log-of-a-product → sum-of-logs", "Natural log and base change"],
        anchor: "Log-probabilities turn a product of tiny probabilities into a stable sum." },
      { id: "1.2", title: "Summation (Σ) & Product (Π) Notation",
        covers: ["Reading Σ and Π", "Index ranges and dummy variables", "Linearity of sums", "Double sums", "Turning Π into Σ with logs"],
        anchor: "Every loss is a sum over examples; softmax is a sum in the denominator." },
      { id: "1.3", title: "Equations, Inequalities & Functions",
        covers: ["Solving linear & quadratic equations", "Inequalities and their direction rules", "Functions: domain, range", "Composition f(g(x))", "Inverse functions"],
        anchor: "A neural network is a giant function composition; activations have domains/ranges." },
      { id: "1.4", title: "Big-O Notation for ML",
        covers: ["Growth rates, not exact counts", "O, Θ, Ω", "Common classes: O(n), O(n log n), O(n²)", "Why attention is O(n²)", "Memory vs compute complexity"],
        anchor: "Self-attention costs O(n²) in sequence length — the reason long context is hard." },
      { id: "1.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "2", short: "Linear Algebra I", title: "Linear Algebra I — Vectors & Matrices",
    lessons: [
      { id: "2.1", title: "Scalars, Vectors & Matrices",
        covers: ["Scalars, vectors, matrices, tensors", "Shapes and indexing", "NumPy arrays & dtypes", "Row vs column vectors", "Vectors as points and as arrows"],
        anchor: "A token embedding is a vector; a weight layer is a matrix; a batch is a tensor." },
      { id: "2.2", title: "Vector Addition & Scalar Multiplication",
        covers: ["Tip-to-tail addition", "Scaling and direction", "Linear combinations", "The span of vectors", "Zero vector & additive inverse"],
        anchor: "A residual connection literally adds two vectors: x + f(x)." },
      { id: "2.3", title: "The Dot Product",
        covers: ["Algebraic definition", "Geometric definition with cosθ", "Cosine similarity", "Orthogonality (dot = 0)", "Projection of one vector on another"],
        anchor: "An attention score is a dot product between a query and a key vector." },
      { id: "2.4", title: "Matrix Multiplication — Two Views",
        covers: ["Row·column rule", "Shapes that conform", "Matrix as a linear transformation of space", "Composition of transforms", "Cost: O(n³) for n×n"],
        anchor: "A forward pass through a dense layer is exactly a matrix–vector product." },
      { id: "2.5", title: "Transpose, Identity, Hadamard & Broadcasting",
        covers: ["Transpose and (AB)ᵀ=BᵀAᵀ", "Identity & zero matrices", "Element-wise (Hadamard) product", "NumPy broadcasting rules", "Bias add as broadcasting"],
        anchor: "Gating (LSTM/GLU) is a Hadamard product; bias-add relies on broadcasting." },
      { id: "2.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "3", short: "Linear Algebra II", title: "Linear Algebra II — Systems, Spaces & Rank",
    lessons: [
      { id: "3.1", title: "Systems of Equations & Gaussian Elimination",
        covers: ["Linear systems as Ax=b", "Row operations", "Row echelon form", "Back-substitution", "Consistent vs inconsistent systems"],
        anchor: "Least-squares regression solves an over-determined linear system." },
      { id: "3.2", title: "Rank, Column Space & Null Space",
        covers: ["Column space = reachable outputs", "Null space = directions that vanish", "Rank as number of independent directions", "Rank–nullity theorem", "Full vs low rank"],
        anchor: "LoRA assumes weight updates are low-rank — a small rank captures the change." },
      { id: "3.3", title: "Linear Independence, Basis & Dimension",
        covers: ["Independence", "Basis = minimal spanning set", "Dimension", "Coordinates in a basis", "Change of basis"],
        anchor: "Embedding dimensions are a learned basis for meaning." },
      { id: "3.4", title: "Orthogonality, Projections & Gram–Schmidt",
        covers: ["Orthogonal & orthonormal sets", "Projection onto a subspace", "Least-squares as projection", "Gram–Schmidt process", "Orthogonal matrices preserve length"],
        anchor: "Orthogonal initialization keeps signal norm stable through deep nets." },
      { id: "3.5", title: "QR Decomposition",
        covers: ["A = QR", "Q orthonormal, R upper-triangular", "Solving least-squares with QR", "Numerical advantage over normal equations", "Connection to Gram–Schmidt"],
        anchor: "QR is the numerically stable way to fit linear regression." },
      { id: "3.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "4", short: "Linear Algebra III", title: "Linear Algebra III — Norms, Distances & Inverses",
    lessons: [
      { id: "4.1", title: "Vector Norms (L1, L2, L∞, Lp)",
        covers: ["What a norm measures", "L2 (Euclidean), L1 (taxicab), L∞ (max)", "Unit balls of each norm", "Sparsity of L1", "Lp family"],
        anchor: "L1 regularization yields sparse weights; L2 yields small weights (weight decay)." },
      { id: "4.2", title: "Matrix Norms (Frobenius, Spectral)",
        covers: ["Frobenius norm", "Spectral (operator) norm", "Norms as bounds on amplification", "Lipschitz constant of a layer", "Spectral normalization"],
        anchor: "Spectral norm controls how much a layer can blow up activations (GAN stability)." },
      { id: "4.3", title: "Matrix Inverse & Pseudo-inverse",
        covers: ["When an inverse exists", "Singular vs invertible", "Moore–Penrose pseudo-inverse", "Why we rarely form inverses", "Solve, don't invert"],
        anchor: "The normal-equation solution uses a pseudo-inverse, but solvers avoid it." },
      { id: "4.4", title: "Solving Ax=b, LU & Numerical Stability",
        covers: ["LU decomposition", "Forward/back substitution", "Condition number", "Catastrophic cancellation", "Why stability matters in training"],
        anchor: "Ill-conditioned matrices make optimization and inversion fragile." },
      { id: "4.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "5", short: "Linear Algebra IV", title: "Linear Algebra IV — Eigenvalues, PCA & SVD",
    lessons: [
      { id: "5.1", title: "Eigenvalues & Eigenvectors",
        covers: ["Directions a matrix only stretches", "Characteristic equation det(A−λI)=0", "Computing eigenpairs", "Geometric meaning", "Repeated & complex eigenvalues"],
        anchor: "Hessian eigenvalues describe the curvature of the loss surface." },
      { id: "5.2", title: "Diagonalization & Symmetric Matrices",
        covers: ["A = PDP⁻¹", "Powers of a matrix", "Symmetric ⇒ real eigenvalues & orthogonal eigenvectors", "Spectral theorem", "Orthogonal diagonalization"],
        anchor: "Covariance matrices are symmetric — PCA diagonalizes them." },
      { id: "5.3", title: "Positive Definite Matrices",
        covers: ["xᵀAx > 0", "Tests for positive definiteness", "Cholesky factor", "Connection to convexity", "PSD covariance & kernels"],
        anchor: "A positive-definite Hessian means a local minimum, not a saddle." },
      { id: "5.4", title: "Singular Value Decomposition (SVD)",
        covers: ["A = UΣVᵀ for any matrix", "Full derivation from AᵀA", "Singular values vs eigenvalues", "Geometric picture: rotate–stretch–rotate", "Rank from singular values"],
        anchor: "SVD powers recommendation systems and is the math behind low-rank adapters." },
      { id: "5.5", title: "PCA from SVD & Low-Rank Approximation",
        covers: ["Centering data", "Principal components = top singular directions", "Variance explained", "Eckart–Young low-rank theorem", "Reconstruction error"],
        anchor: "PCA compresses embeddings; low-rank approximation is model compression." },
      { id: "5.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "6", short: "Calculus I", title: "Calculus I — Derivatives & Gradients",
    lessons: [
      { id: "6.1", title: "Limits, Continuity & the Derivative",
        covers: ["Limits as 'getting arbitrarily close'", "Continuity intuitively", "Derivative as slope and as rate", "The difference quotient", "Differentiable ⇒ continuous"],
        anchor: "Training needs smooth losses so a slope exists to follow downhill." },
      { id: "6.2", title: "Differentiation Rules & the Chain Rule",
        covers: ["Power, product, quotient rules", "The chain rule", "Derivatives of exp, log, sigmoid", "Composing rules", "Why the chain rule is backprop"],
        anchor: "Backpropagation is the chain rule applied layer by layer." },
      { id: "6.3", title: "Partial Derivatives & the Gradient",
        covers: ["Partial derivatives", "The gradient vector", "Why the gradient points uphill", "Directional derivatives", "Gradient of a dot product / quadratic"],
        anchor: "The gradient is the direction of steepest loss increase — we step the other way." },
      { id: "6.4", title: "Jacobian & Hessian",
        covers: ["Jacobian of a vector function", "Chain rule with Jacobians", "Second derivatives", "The Hessian matrix", "Curvature and definiteness"],
        anchor: "Reverse-mode autodiff multiplies Jacobians; the Hessian sets the best step size." },
      { id: "6.5", title: "Taylor Expansion (1st & 2nd order)",
        covers: ["Linear approximation", "Quadratic approximation", "Remainder term", "Multivariate Taylor", "Approximating the loss locally"],
        anchor: "Gradient descent comes straight out of a first-order Taylor expansion of the loss." },
      { id: "6.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "7", short: "Calculus II", title: "Calculus II — Integration & Expectation",
    lessons: [
      { id: "7.1", title: "Antiderivatives & the Definite Integral",
        covers: ["Integral as accumulated area", "Riemann sums", "Antiderivatives", "Notation and bounds", "Signed area"],
        anchor: "A probability density integrates to total probability 1." },
      { id: "7.2", title: "Fundamental Theorem of Calculus",
        covers: ["Differentiation and integration are inverses", "FTC part I & II", "Evaluating definite integrals", "Why it matters", "Worked examples"],
        anchor: "Relates a CDF to its density: differentiate one to get the other." },
      { id: "7.3", title: "Substitution & Integration by Parts",
        covers: ["u-substitution", "Integration by parts", "Choosing u and dv", "Definite-integral substitution", "Common ML integrals"],
        anchor: "Change of variables underlies normalizing flows and the reparameterization trick." },
      { id: "7.4", title: "The Gaussian Integral",
        covers: ["∫e^{−x²}dx = √π via the polar trick", "Normalizing the Normal distribution", "Completing the square", "Moments of the Gaussian", "Multivariate normalization"],
        anchor: "This integral is why the Normal distribution's constant is 1/√(2πσ²)." },
      { id: "7.5", title: "Integrals as Expectation",
        covers: ["E[X] as an integral", "Law of the unconscious statistician", "Expectation of functions", "Variance as an integral", "Monte-Carlo estimation"],
        anchor: "Every expected loss is an integral we approximate by averaging minibatches." },
      { id: "7.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "8", short: "Optimization", title: "Optimization Theory",
    lessons: [
      { id: "8.1", title: "Minima, Maxima & Saddle Points",
        covers: ["Critical points", "First-order condition ∇f=0", "Second-order test", "Saddle points in high dimensions", "Why deep nets have many saddles"],
        anchor: "Most stuck-looking points in deep learning are saddles, not bad minima." },
      { id: "8.2", title: "Convexity",
        covers: ["Convex sets & functions", "The chord-above-graph test", "Second-derivative test for convexity", "Global vs local minima", "Convex losses (logistic, linear)"],
        anchor: "Logistic regression has a convex loss — one global minimum, no surprises." },
      { id: "8.3", title: "Gradient Descent from Taylor Expansion",
        covers: ["Derive the update from a 1st-order model", "Learning-rate intuition", "Steepest descent direction", "Convergence on convex functions", "Step-size pitfalls"],
        anchor: "SGD is this update with the gradient estimated on a minibatch." },
      { id: "8.4", title: "Momentum & Adam",
        covers: ["Momentum as a heavy ball", "RMSProp scaling", "Adam = momentum + adaptive scale", "Bias correction", "Why Adam is the default"],
        anchor: "Adam is the optimizer training most modern networks." },
      { id: "8.5", title: "Lagrange Multipliers & KKT",
        covers: ["Constrained optimization", "Lagrange multipliers", "Equality vs inequality constraints", "KKT conditions", "Duality (intro)"],
        anchor: "Constrained objectives appear in SVMs and in KL-constrained RLHF." },
      { id: "8.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "9", short: "Probability I", title: "Probability I — Foundations",
    lessons: [
      { id: "9.1", title: "Sample Spaces, Events & Axioms",
        covers: ["Sample space & events", "Kolmogorov axioms", "Union, intersection, complement", "Inclusion–exclusion", "Equally-likely outcomes"],
        anchor: "Dropout flips a coin per neuron — a sample space in action." },
      { id: "9.2", title: "Conditional Probability & Independence",
        covers: ["P(A|B) definition", "Multiplication rule", "Independence", "Conditional independence", "Chain rule of probability"],
        anchor: "A language model factorizes text via the chain rule of probability." },
      { id: "9.3", title: "Bayes' Theorem",
        covers: ["Prior, likelihood, posterior", "Full derivation", "Law of total probability", "Base-rate intuition", "Naive Bayes"],
        anchor: "Naive Bayes and Bayesian deep learning both start from Bayes' theorem." },
      { id: "9.4", title: "Random Variables, Expectation & Variance",
        covers: ["Discrete random variables", "PMFs", "Expected value", "Variance & standard deviation", "Linearity of expectation"],
        anchor: "The expected loss is what training actually minimizes." },
      { id: "9.5", title: "Covariance, Correlation & Joint Distributions",
        covers: ["Joint, marginal, conditional", "Covariance", "Correlation", "Independence vs uncorrelated", "Covariance matrix"],
        anchor: "Covariance matrices feed PCA, whitening, and Gaussian models." },
      { id: "9.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "10", short: "Probability II", title: "Probability II — Continuous Distributions & Theorems",
    lessons: [
      { id: "10.1", title: "PDFs, CDFs & the Uniform",
        covers: ["Density vs probability", "PDF integrates to 1", "CDF and quantiles", "Uniform distribution", "Sampling via inverse CDF"],
        anchor: "Uniform noise seeds dropout masks and data shuffling." },
      { id: "10.2", title: "The Gaussian (Normal) Distribution",
        covers: ["The bell curve", "Mean & variance", "Standardization (z-scores)", "68–95–99.7 rule", "Why Gaussians are everywhere (CLT preview)"],
        anchor: "Weight initialization samples from a carefully scaled Gaussian." },
      { id: "10.3", title: "Exponential, Beta & Dirichlet",
        covers: ["Exponential (waiting times)", "Beta on [0,1]", "Dirichlet over simplices", "Conjugacy", "Use in topic models"],
        anchor: "Dirichlet priors generate topic distributions in LDA." },
      { id: "10.4", title: "The Multivariate Gaussian",
        covers: ["Mean vector & covariance matrix", "Contours as ellipses", "Whitening", "Marginals & conditionals", "Log-density"],
        anchor: "Gaussian latent spaces (VAEs) and Gaussian processes live here." },
      { id: "10.5", title: "Change of Variables & the Reparameterization Trick",
        covers: ["Density under a transform", "Jacobian determinant", "z = μ + σ·ε", "Why it enables backprop through sampling", "Normalizing flows (preview)"],
        anchor: "The reparameterization trick lets VAEs backprop through randomness." },
      { id: "10.6", title: "Law of Large Numbers & CLT",
        covers: ["Sample mean converges", "LLN statement", "Central Limit Theorem", "Proof sketch via MGFs", "Simulation"],
        anchor: "Minibatch gradients are unbiased estimates that concentrate by the LLN." },
      { id: "10.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "11", short: "Statistics", title: "Statistics & Estimation",
    lessons: [
      { id: "11.1", title: "Point Estimation, Bias & Variance",
        covers: ["Estimators", "Bias of an estimator", "Variance of an estimator", "Mean squared error", "Consistency"],
        anchor: "Model error decomposes into bias², variance, and noise." },
      { id: "11.2", title: "Maximum Likelihood Estimation (MLE)",
        covers: ["Likelihood vs probability", "Log-likelihood", "MLE for Bernoulli, Gaussian, Categorical", "Why we maximize log", "MLE = minimizing cross-entropy"],
        anchor: "Cross-entropy loss IS the negative log-likelihood of the data." },
      { id: "11.3", title: "MAP Estimation & Priors",
        covers: ["Adding a prior", "Posterior ∝ likelihood × prior", "MAP derivation", "Gaussian prior ⇒ L2", "Laplace prior ⇒ L1"],
        anchor: "L2 regularization is exactly MAP with a Gaussian prior on weights." },
      { id: "11.4", title: "The Bias–Variance Tradeoff",
        covers: ["Decomposing expected error", "Underfitting vs overfitting", "Model capacity", "Formal derivation", "Double descent (note)"],
        anchor: "The classic curve behind choosing model size and regularization." },
      { id: "11.5", title: "Hypothesis Testing, CIs & Bootstrap",
        covers: ["Null & alternative", "p-values (correctly)", "Confidence intervals", "The bootstrap", "A/B testing models"],
        anchor: "Comparing two models' metrics is a hypothesis test." },
      { id: "11.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "12", short: "Information Theory", title: "Information Theory",
    lessons: [
      { id: "12.1", title: "Entropy",
        covers: ["Surprise = −log p", "Entropy as average surprise", "Derivation & units (bits/nats)", "Max-entropy distributions", "Entropy of Bernoulli"],
        anchor: "Language-model perplexity is two-to-the-entropy of its predictions." },
      { id: "12.2", title: "Cross-Entropy & KL Divergence",
        covers: ["Cross-entropy H(p,q)", "KL divergence", "Why KL ≥ 0 (Gibbs)", "Why KL is asymmetric", "Cross-entropy = entropy + KL"],
        anchor: "Classification training minimizes cross-entropy = entropy + KL to the labels." },
      { id: "12.3", title: "Mutual Information",
        covers: ["Information shared by two variables", "MI = H(X)−H(X|Y)", "Non-negativity", "Estimation difficulties", "InfoNCE (preview)"],
        anchor: "Contrastive learning maximizes a bound on mutual information." },
      { id: "12.4", title: "Jensen's Inequality & the ELBO",
        covers: ["Convexity and expectations", "Jensen's inequality", "Bits vs nats recap", "Deriving the ELBO", "Information gain in trees"],
        anchor: "The VAE's training objective (ELBO) comes from Jensen's inequality." },
      { id: "12.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "13", short: "Numerical Methods", title: "Numerical Methods & Computational Math",
    lessons: [
      { id: "13.1", title: "Floating Point & Rounding Error",
        covers: ["IEEE 754 floats", "float32 vs float16 vs bfloat16", "Machine epsilon", "Rounding error accumulation", "Representable range"],
        anchor: "bfloat16 trades precision for range to train large models stably." },
      { id: "13.2", title: "Numerical Stability & Log-Sum-Exp",
        covers: ["Catastrophic cancellation", "Overflow/underflow in softmax", "The log-sum-exp trick", "Stable cross-entropy", "Subtracting the max"],
        anchor: "Every framework's softmax subtracts the max for stability — here's why." },
      { id: "13.3", title: "Matrix Factorizations in Practice",
        covers: ["LU vs Cholesky vs QR", "When to use which", "Cost comparison", "Conditioning", "Solvers under the hood"],
        anchor: "Choosing the right factorization keeps linear solves fast and stable." },
      { id: "13.4", title: "Automatic Differentiation (Forward vs Reverse)",
        covers: ["Dual numbers (forward mode)", "The computational graph", "Reverse mode = backprop", "Why reverse is O(1)× the forward cost", "VJPs"],
        anchor: "Reverse-mode autodiff is exactly how every deep-learning framework gets gradients." },
      { id: "13.5", title: "Gradient Checking & Mixed Precision",
        covers: ["Finite-difference gradients", "Checking analytic gradients", "Relative error", "Mixed-precision training", "Loss scaling"],
        anchor: "Gradient checking catches backprop bugs; mixed precision speeds training." },
      { id: "13.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "14", short: "Matrix Calculus", title: "Matrix Calculus & Backpropagation",
    lessons: [
      { id: "14.1", title: "Gradients, Jacobians & the Shape Rule",
        covers: ["Scalar vs vector vs matrix derivatives", "The gradient and the Jacobian", "The shape rule", "Finite-difference gradient checks"],
        anchor: "Why W.grad has exactly the same shape as W." },
      { id: "14.2", title: "Differentiating Vector & Matrix Expressions",
        covers: ["The core identities (aᵀx, xᵀAx, …)", "Index-notation derivations", "Least squares & the normal equations", "Trace and Frobenius derivatives"],
        anchor: "Deriving the gradient of a linear layer with squared loss." },
      { id: "14.3", title: "The Vector Chain Rule & Jacobians",
        covers: ["Composing Jacobians", "Vector–Jacobian products (VJPs)", "Reverse vs forward mode", "Why backprop costs about one forward pass"],
        anchor: "Backpropagation is the chain rule, bracketed right-to-left." },
      { id: "14.4", title: "Backprop Through a Linear Layer",
        covers: ["Deriving dW, db, dX", "The input-times-error pattern", "Summing the bias over the batch", "Stacking layers into a network"],
        anchor: "What nn.Linear.backward() actually computes." },
      { id: "14.5", title: "The Softmax + Cross-Entropy Gradient",
        covers: ["The softmax Jacobian", "The clean p − y result", "Numerical stability (log-sum-exp)", "Link to logistic regression"],
        anchor: "The gradient into the logits of every classifier." },
      { id: "14.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "15", short: "Bayesian Inference", title: "Bayesian Inference & Probabilistic ML",
    lessons: [
      { id: "15.1", title: "Conjugate Priors & Bayesian Updating",
        covers: ["Bayes' rule for parameters", "Beta–Bernoulli conjugacy", "Posterior = prior + data", "Credible intervals & pseudo-counts"],
        anchor: "A/B testing and Thompson sampling with full uncertainty." },
      { id: "15.2", title: "Bayesian Linear Regression",
        covers: ["Gaussian posterior over weights", "Precision = prior + data", "Predictive uncertainty (noise + weights)", "Ridge regression as MAP"],
        anchor: "Predictions with error bars that grow away from data." },
      { id: "15.3", title: "MCMC: Metropolis–Hastings",
        covers: ["Sampling unnormalized posteriors", "The acceptance ratio", "Detailed balance", "Burn-in & mixing"],
        anchor: "How PyMC/Stan sample any posterior — and how diffusion sampling works." },
      { id: "15.4", title: "Variational Inference & the ELBO",
        covers: ["Inference as optimization", "log p(x) = ELBO + KL", "The reparameterization trick", "The VAE objective"],
        anchor: "The ELBO is exactly the loss that trains a VAE." },
      { id: "15.5", title: "Gaussian Processes",
        covers: ["A prior over functions", "The kernel", "GP regression posterior", "Uncertainty that grows away from data"],
        anchor: "Bayesian optimization and infinitely-wide neural nets." },
      { id: "15.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "16", short: "Diffusion & GenAI", title: "Diffusion & Generative Models",
    lessons: [
      { id: "16.1", title: "Variational Autoencoders (VAEs)",
        covers: ["The ELBO as encoder + decoder", "Reconstruction + KL", "The reparameterization trick", "Latent-space generation"],
        anchor: "The compression front-end of Stable Diffusion." },
      { id: "16.2", title: "GANs: Generation as a Game",
        covers: ["The min–max objective", "Optimal discriminator", "Jensen–Shannon divergence", "Mode collapse & instability"],
        anchor: "How StyleGAN makes sharp images — and why GANs are twitchy." },
      { id: "16.3", title: "The Forward Diffusion Process",
        covers: ["Fixed Gaussian noising", "The closed form q(x_t|x₀)", "Noise schedules", "Data → N(0,I)"],
        anchor: "The fixed process every diffusion model reverses." },
      { id: "16.4", title: "Score Matching & Denoising",
        covers: ["The score ∇log p(x)", "Denoising = score matching", "The DDPM ε-prediction loss", "Tweedie's formula"],
        anchor: "Why the whole training loss is 'predict the noise'." },
      { id: "16.5", title: "The Reverse Process & Sampling",
        covers: ["Langevin dynamics", "Annealed sampling", "Reverse diffusion (DDPM)", "The MCMC connection"],
        anchor: "How an image appears out of pure noise." },
      { id: "16.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "17", short: "NN & Transformers", title: "Neural-Network & Transformer Math",
    lessons: [
      { id: "17.1", title: "Depth, Nonlinearity & Gradient Flow",
        covers: ["Why nonlinearity matters", "Activation derivatives", "Vanishing/exploding gradients", "ReLU, GELU, dead units"],
        anchor: "Why deep nets need ReLU/GELU and residuals to train." },
      { id: "17.2", title: "Convolution as Matrix Multiply",
        covers: ["Locality & weight sharing", "im2col", "Parameter count vs dense", "Translation equivariance"],
        anchor: "The CNN vision inductive bias, run as a matmul." },
      { id: "17.3", title: "Normalization: BatchNorm & LayerNorm",
        covers: ["Normalize then affine (γ, β)", "Batch vs feature axis", "Why LayerNorm for transformers", "The backward pass"],
        anchor: "The LayerNorm in every transformer block." },
      { id: "17.4", title: "Attention: The QKV Mechanism",
        covers: ["Query, key, value", "softmax(QKᵀ/√d)·V", "Why the √d scaling", "Self-attention"],
        anchor: "The core of every transformer." },
      { id: "17.5", title: "Multi-Head Attention & the Transformer Block",
        covers: ["Multiple heads", "Residual + LayerNorm + MLP", "The full block", "Parameter & FLOP counts"],
        anchor: "The GPT/BERT block — the whole course assembled." },
      { id: "17.E", title: "Track Exam", exam: true }
    ]
  },
  {
    id: "C", short: "Capstones", title: "Capstones — Projects That Run",
    lessons: [
      { id: "C.1", title: "PCA from Scratch on MNIST",
        covers: ["Implement PCA with only NumPy", "Apply to MNIST digits", "Visualize the top-16 principal components", "Variance explained curve", "Reconstruct digits from k components"],
        anchor: "Dimensionality reduction and embedding compression." },
      { id: "C.2", title: "Gradient Descent, Momentum & Adam",
        covers: ["Implement GD, momentum, Adam from scratch", "Non-convex 2-D loss surface", "Compare convergence", "Animate the trajectories", "Effect of learning rate"],
        anchor: "Understand what your optimizer is actually doing." },
      { id: "C.3", title: "A Bayesian Classifier from Scratch",
        covers: ["Gaussian Naive Bayes", "Derive the decision boundary", "MLE vs MAP as data grows", "Posterior probabilities", "Compare to logistic regression"],
        anchor: "Generative classification and the role of priors." },
      { id: "C.4", title: "Cross-Entropy, KL & Mutual Information",
        covers: ["Compute CE, KL, MI for a tiny LM", "Verify CE = H + KL numerically", "Perplexity", "MI between tokens", "Bits vs nats in practice"],
        anchor: "The information theory behind language-model training." },
      { id: "C.5", title: "A Tiny Autograd Engine",
        covers: ["Reverse-mode AD from scratch", "Scalar Value class with backward()", "Verify against finite differences", "Train a 2-layer MLP", "Visualize the loss curve"],
        anchor: "Build the core of PyTorch in ~100 lines." }
    ]
  }
];
