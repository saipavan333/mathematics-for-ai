/* ============================================================
   TRACK 9 — Probability I — Foundations
   Opener: 9.3 Bayes' Theorem.
   ============================================================ */
(window.LESSON_CONTENT ||= {})["9.3"] = {
  subtitle: "Flip a conditional: turn 'evidence given cause' into 'cause given evidence'.",

  aiMoment: String.raw`<p>A spam filter knows $P(\text{words}\mid\text{spam})$ from training, but what it actually needs
  at inference is $P(\text{spam}\mid\text{words})$. <strong>Bayes' theorem</strong> is the rule that flips one into the
  other. It's the engine of Naive Bayes classifiers, the way Bayesian neural networks turn a prior over weights into a
  posterior after seeing data, and the reason a "99% accurate" medical test can still be wrong most of the time when
  the disease is rare.</p>`,

  plainEnglish: String.raw`<p>You often know the chance of the <em>evidence</em> assuming some <em>cause</em> is true,
  but you want the reverse — the chance of the cause given the evidence you saw. Bayes' theorem swaps them. It starts
  from a <strong>prior</strong> belief and updates it into a <strong>posterior</strong> using how well the cause
  explains the evidence.</p>`,

  intuition: String.raw`<p>Split the world by cause, then by evidence. Most "positive" results can come from the
  huge healthy group (false positives) rather than the tiny sick group — so a positive test shifts your belief, but
  the rare prior keeps it in check.</p>
  <figure class="figure">
  <svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bayes tree: prior splits into likelihoods">
    <g font-family="sans-serif" font-size="11">
    <circle cx="40" cy="100" r="5" fill="#4f46e5"/>
    <line x1="45" y1="98" x2="130" y2="55" stroke="#94a3b8"/>
    <line x1="45" y1="102" x2="130" y2="150" stroke="#94a3b8"/>
    <text x="70" y="64" fill="#64748b">P(D)=.01</text>
    <text x="66" y="150" fill="#64748b">P(¬D)=.99</text>
    <circle cx="138" cy="55" r="5" fill="#0d9488"/>
    <circle cx="138" cy="150" r="5" fill="#0d9488"/>
    <line x1="143" y1="53" x2="225" y2="38" stroke="#cbd5e1"/>
    <line x1="143" y1="150" x2="225" y2="120" stroke="#cbd5e1"/>
    <text x="158" y="38" fill="#16a34a">+ : .99</text>
    <text x="150" y="120" fill="#dc2626">+ : .05</text>
    <text x="232" y="40" fill="#20242c">true +  (.0099)</text>
    <text x="232" y="122" fill="#20242c">false + (.0495)</text>
    <text x="90" y="186" fill="#94a3b8">most positives are false → posterior ≈17%</text>
    </g>
  </svg>
  <figcaption>Rare cause + imperfect test ⇒ a positive result is usually a false alarm.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For events $A$ (cause) and $B$ (evidence) with $P(B)\gt0$:</p>
  $$P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}=\frac{\overbrace{P(B\mid A)}^{\text{likelihood}}\,\overbrace{P(A)}^{\text{prior}}}{\underbrace{P(B)}_{\text{evidence}}}.$$
  <p>The denominator expands by the <strong>law of total probability</strong> over a partition $\{A_i\}$:
  $P(B)=\sum_i P(B\mid A_i)P(A_i)$. The result $P(A\mid B)$ is the <strong>posterior</strong>. In words: posterior $\propto$
  likelihood $\times$ prior, with the evidence as the normalizer that makes everything sum to 1.</p>`,

  derivation: String.raw`<p><strong>Bayes' theorem in three lines.</strong> It falls straight out of the definition of
  conditional probability.</p>
  <p><strong>Step 1 — two ways to write the joint.</strong> By definition,
  $P(A\mid B)=\dfrac{P(A\cap B)}{P(B)}$ and $P(B\mid A)=\dfrac{P(A\cap B)}{P(A)}$. Both share the joint $P(A\cap B)$.</p>
  <p><strong>Step 2 — solve the second for the joint:</strong> $P(A\cap B)=P(B\mid A)\,P(A)$.</p>
  <p><strong>Step 3 — substitute into the first:</strong>
  $P(A\mid B)=\dfrac{P(B\mid A)\,P(A)}{P(B)}.\;\blacksquare$</p>
  <p><strong>Step 4 — make the denominator computable.</strong> Split $B$ across the cause being true or false:
  $P(B)=P(B\mid A)P(A)+P(B\mid\lnot A)P(\lnot A)$. This is the law of total probability, and it's what turns Bayes
  into an arithmetic you can actually evaluate. Plain English: weigh how well each hypothesis predicts the evidence,
  then normalize.</p>`,

  code: [
    { label: "The rare-disease test, by Bayes", src: String.raw`
# Prevalence 1%, sensitivity 99%, false-positive rate 5%.
P_D   = 0.01                 # prior P(disease)
P_pos_given_D  = 0.99        # sensitivity
P_pos_given_nD = 0.05        # 1 - specificity

# Law of total probability for the evidence P(+)
P_pos = P_pos_given_D*P_D + P_pos_given_nD*(1-P_D)
posterior = P_pos_given_D*P_D / P_pos

print("P(+)            =", round(P_pos, 4))
print("P(disease | +)  =", round(posterior, 4))   # ~0.1667 -> only ~17%!
` },
    { label: "A tiny Naive Bayes decision", src: String.raw`
import numpy as np
# Two classes, prior, and likelihood of a feature 'word appears'
prior = {"spam": 0.3, "ham": 0.7}
like  = {"spam": 0.8, "ham": 0.1}    # P(word | class)

post = {}
for c in prior:
    post[c] = like[c] * prior[c]
Z = sum(post.values())               # the evidence P(word)
for c in post: post[c] /= Z
print("posterior:", {c: round(v,3) for c,v in post.items()})
print("decision :", max(post, key=post.get))
` }
  ],

  keyPoints: [
    "Bayes flips a conditional: $P(A\\mid B)=P(B\\mid A)P(A)/P(B)$.",
    "Posterior $\\propto$ likelihood $\\times$ prior; the evidence $P(B)$ just normalizes.",
    "The evidence expands by total probability: $P(B)=\\sum_i P(B\\mid A_i)P(A_i)$.",
    "Low base rates keep posteriors low even after strong evidence (base-rate effect).",
    "$P(A\\mid B)$ and $P(B\\mid A)$ are different numbers — never swap them."
  ],

  commonMistakes: [
    { wrong: "Confusing $P(A\\mid B)$ with $P(B\\mid A)$ (the 'prosecutor's fallacy').", why: "$P(\\text{evidence}\\mid\\text{cause})$ can be huge while $P(\\text{cause}\\mid\\text{evidence})$ is tiny — exactly the rare-disease case. They're linked only through the prior." },
    { wrong: "Ignoring the base rate (prior).", why: "Dropping $P(A)$ overstates the posterior. A 99%-accurate test on a 1-in-10,000 condition still yields mostly false positives." },
    { wrong: "Forgetting to normalize by $P(B)$.", why: "Likelihood × prior gives an <em>unnormalized</em> score. Without dividing by the evidence, your 'probabilities' won't sum to 1." }
  ],

  quiz: [
    { q: "Disease prevalence 1%, test sensitivity 99%, false-positive rate 5%. $P(\\text{disease}\\mid +)$?", options: ["≈17%", "≈99%", "≈95%", "≈50%"], answer: 0,
      explain: "$\\frac{.99\\cdot.01}{.99\\cdot.01+.05\\cdot.99}=\\frac{.0099}{.0594}\\approx.167$. The 99% is $P(+\\mid D)$, not $P(D\\mid +)$ — base rate dominates." },
    { q: "Posterior is proportional to…", options: ["likelihood × prior", "likelihood ÷ prior", "prior ÷ evidence", "likelihood × evidence"], answer: 0,
      explain: "$P(A\\mid B)\\propto P(B\\mid A)P(A)$; the evidence is the constant that normalizes it." },
    { q: "Prior $P(A)=0.5$, $P(B\\mid A)=0.8$, $P(B\\mid\\lnot A)=0.2$. Find $P(A\\mid B)$.", options: ["0.8", "0.5", "0.64", "0.2"], answer: 0,
      explain: "$P(B)=.8(.5)+.2(.5)=.5$; $P(A\\mid B)=.8(.5)/.5=.8$." },
    { q: "Which quantity is the 'evidence' in Bayes?", options: ["$P(B)$", "$P(A)$", "$P(B\\mid A)$", "$P(A\\mid B)$"], answer: 0,
      explain: "$P(B)$ — the marginal probability of the observation — sits in the denominator and normalizes the posterior." },
    { q: "Doubling the prior $P(A)$ (others fixed, $P(B)$ recomputed) makes the posterior…", options: ["larger", "smaller", "unchanged", "negative"], answer: 0,
      explain: "Posterior rises with the prior (more plausible cause → higher updated belief), though not exactly double because $P(B)$ also shifts." }
  ],

  practice: [
    { level: "easy", prompt: "State Bayes' theorem and label prior, likelihood, posterior, evidence.", solution: "$P(A\\mid B)=\\frac{P(B\\mid A)P(A)}{P(B)}$: $P(A)$ prior, $P(B\\mid A)$ likelihood, $P(A\\mid B)$ posterior, $P(B)$ evidence." },
    { level: "easy", prompt: "If $P(B\\mid A)=P(B\\mid\\lnot A)$, what does observing $B$ do to your belief in $A$?", solution: "Nothing: the evidence is equally likely either way, so the posterior equals the prior — $B$ is uninformative about $A$." },
    { level: "med", prompt: "A bag is 'fair' (prior 0.9) or 'biased' giving heads 0.9 (prior 0.1). You flip heads. Posterior that it's biased?", solution: "$P(H\\mid\\text{fair})=.5$, $P(H\\mid\\text{biased})=.9$. Evidence $=.5(.9)+.9(.1)=.54$. Posterior(biased)$=.9(.1)/.54=.09/.54\\approx.167$. One heads barely moves the strong prior." },
    { level: "hard", prompt: "AI task: explain why a spam model uses log-posteriors $\\log P(c)+\\sum_i\\log P(w_i\\mid c)$ and why it's called 'naive'.", solution: "Taking logs turns the product of many word-likelihoods into a sum (Lesson 1.2), avoiding underflow and making the decision a fast dot-product-like score. It's 'naive' because it assumes words are conditionally independent given the class — $P(w_1,w_2\\mid c)=P(w_1\\mid c)P(w_2\\mid c)$ — which is false (words correlate) but works remarkably well and keeps the math tractable." }
  ],

  deepDive: String.raw`<p><strong>Bayes as sequential updating — today's posterior is tomorrow's prior.</strong></p>
  <p>Bayes isn't a one-shot formula; it's an update rule you can iterate. After seeing evidence $B_1$, your belief
  becomes $P(A\mid B_1)$. When new evidence $B_2$ arrives (conditionally independent given $A$), you feed the previous
  posterior back in as the prior:
  $P(A\mid B_1,B_2)\propto P(B_2\mid A)\,P(A\mid B_1)$. Belief accumulates, evidence by evidence.</p>
  <p>This is the conceptual root of the whole next two tracks. <strong>Maximum likelihood</strong> (Track 11) is
  Bayes with a flat prior — you trust the data alone. <strong>MAP estimation</strong> keeps the prior, which is why
  $L_2$ regularization turns out to be a Gaussian prior on the weights. And <strong>Bayesian deep learning</strong>
  carries a full posterior over weights instead of a single point, so the model can say "I don't know" in regions with
  little data. The humble three-line theorem scales all the way up to uncertainty-aware networks.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["9.1"] = {
  subtitle: "The rulebook of chance: outcomes, events, and three axioms everything else follows from.",

  aiMoment: String.raw`<p><strong>Dropout</strong> flips an independent coin for every neuron each forward pass;
  <strong>data augmentation</strong> draws a random transform per image; a stochastic policy <strong>samples</strong> an
  action. All of these are random experiments, and to reason about them you need the basic grammar of probability:
  the set of possible outcomes, the events you care about, and the axioms that make probabilities consistent.</p>`,

  plainEnglish: String.raw`<p>The <strong>sample space</strong> is the set of all possible outcomes of an experiment.
  An <strong>event</strong> is any collection of outcomes you might ask about. A <strong>probability</strong> assigns
  each event a number between 0 and 1 saying how likely it is — obeying a few common-sense rules.</p>`,

  intuition: String.raw`<p>Draw the sample space as a box and events as regions inside it. "A or B" is the union of
  the regions, "A and B" the overlap, "not A" everything outside A. Probability is the fraction of the box each region
  covers.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Venn diagram of two events in a sample space">
    <rect x="12" y="12" width="236" height="126" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="20" y="28" font-size="12" fill="#64748b" font-family="sans-serif">Ω</text>
    <circle cx="100" cy="78" r="48" fill="#4f46e5" fill-opacity="0.22" stroke="#4f46e5"/>
    <circle cx="160" cy="78" r="48" fill="#0d9488" fill-opacity="0.22" stroke="#0d9488"/>
    <path d="M130,42 A48,48 0 0,1 130,114 A48,48 0 0,1 130,42" fill="#7c3aed" fill-opacity="0.32"/>
    <text x="74" y="82" font-size="13" fill="#4f46e5" font-family="sans-serif">A</text>
    <text x="178" y="82" font-size="13" fill="#0d9488" font-family="sans-serif">B</text>
    <text x="118" y="135" font-size="10" fill="#7c3aed" font-family="sans-serif">A∩B</text>
  </svg>
  <figcaption>The box is the sample space Ω; events A and B are regions; their overlap is A∩B.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A <strong>sample space</strong> $\Omega$ holds all outcomes; an <strong>event</strong> $A\subseteq\Omega$.
  A probability $P$ obeys the <strong>Kolmogorov axioms</strong>: (1) $P(A)\ge0$; (2) $P(\Omega)=1$; (3) for disjoint
  events, $P(A\cup B)=P(A)+P(B)$. From these: the complement $P(A^c)=1-P(A)$, and for any (possibly overlapping) events
  the <strong>inclusion–exclusion</strong> rule</p>
  $$P(A\cup B)=P(A)+P(B)-P(A\cap B).$$`,

  derivation: String.raw`<p><strong>Derive inclusion–exclusion from the axioms.</strong></p>
  <p><strong>Step 1 — split the union into disjoint pieces.</strong> $A\cup B=A\cup(B\cap A^c)$, and $A$ and $B\cap A^c$
  don't overlap. By axiom 3, $P(A\cup B)=P(A)+P(B\cap A^c)$.</p>
  <p><strong>Step 2 — split $B$ similarly.</strong> $B=(B\cap A)\cup(B\cap A^c)$, disjoint, so
  $P(B)=P(B\cap A)+P(B\cap A^c)$, giving $P(B\cap A^c)=P(B)-P(A\cap B)$.</p>
  <p><strong>Step 3 — substitute:</strong> $P(A\cup B)=P(A)+P(B)-P(A\cap B).\;\blacksquare$ Plain English: adding
  $P(A)$ and $P(B)$ double-counts the overlap, so subtract it once.</p>
  <p><strong>Complement:</strong> $A$ and $A^c$ are disjoint and fill $\Omega$, so $P(A)+P(A^c)=P(\Omega)=1$, hence
  $P(A^c)=1-P(A)$ — often the easy way to compute "at least one."</p>`,

  code: [
    { label: "Simulate, and check inclusion–exclusion", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

rolls = rng.integers(1, 7, size=1_000_000)        # a fair die, many times
A = rolls % 2 == 0          # 'even'
B = rolls > 3               # 'greater than 3'
pA, pB, pAB = A.mean(), B.mean(), (A & B).mean()
print("P(A)=", round(pA,3), " P(B)=", round(pB,3), " P(A∩B)=", round(pAB,3))
print("P(A∪B) measured :", round((A | B).mean(), 3))
print("P(A)+P(B)-P(A∩B):", round(pA+pB-pAB, 3))    # inclusion-exclusion matches
` },
    { label: "Complement: P(at least one) the easy way", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

# P(at least one six in 4 rolls) = 1 - P(no six)
trials = rng.integers(1, 7, size=(1_000_000, 4))
at_least_one_six = (trials == 6).any(axis=1).mean()
print("simulated P(≥1 six) :", round(float(at_least_one_six), 4))
print("1 - (5/6)^4         :", round(1 - (5/6)**4, 4))     # complement rule
` }
  ],

  keyPoints: [
    "Sample space $\\Omega$ = all outcomes; an event is a subset $A\\subseteq\\Omega$.",
    "Axioms: $P(A)\\ge0$, $P(\\Omega)=1$, and additivity for disjoint events.",
    "Complement: $P(A^c)=1-P(A)$ — the trick for 'at least one'.",
    "Inclusion–exclusion: $P(A\\cup B)=P(A)+P(B)-P(A\\cap B)$.",
    "Probabilities are fractions of the sample space; everything else builds on these rules."
  ],

  commonMistakes: [
    { wrong: "Adding probabilities of overlapping events.", why: "$P(A\\cup B)=P(A)+P(B)$ only when $A,B$ are disjoint. Otherwise you double-count $P(A\\cap B)$ and can exceed 1." },
    { wrong: "Computing 'at least one' directly.", why: "It's usually far easier via the complement: $P(\\text{at least one})=1-P(\\text{none})$, especially for independent repeats." },
    { wrong: "Confusing mutually exclusive with independent.", why: "Disjoint events <em>can't</em> co-occur ($P(A\\cap B)=0$), which makes them strongly <em>dependent</em>, not independent. The two ideas are unrelated (see Lesson 9.2)." }
  ],

  quiz: [
    { q: "A fair die: $P(\\text{even or} >3)$? (even=$\\{2,4,6\\}$, $>3=\\{4,5,6\\}$)", options: ["$2/3$", "$1$", "$1/2$", "$5/6$"], answer: 0,
      explain: "$P(A)=3/6$, $P(B)=3/6$, $P(A\\cap B)=|\\{4,6\\}|/6=2/6$. Union $=3/6+3/6-2/6=4/6=2/3$." },
    { q: "If $P(A)=0.7$, then $P(A^c)$ is…", options: ["0.3", "0.7", "1.7", "0"], answer: 0,
      explain: "$P(A^c)=1-P(A)=0.3$." },
    { q: "For disjoint events, $P(A\\cup B)=$", options: ["$P(A)+P(B)$", "$P(A)P(B)$", "$P(A)+P(B)-P(A\\cap B)$ with $P(A\\cap B)\\neq0$", "1"], answer: 0,
      explain: "Disjoint means $P(A\\cap B)=0$, so the union is just the sum." },
    { q: "$P(\\text{at least one six in 3 rolls})$ of a fair die is…", options: ["$1-(5/6)^3\\approx0.42$", "$3/6$", "$(1/6)^3$", "$1/2$"], answer: 0,
      explain: "Complement: $1-P(\\text{no six})=1-(5/6)^3\\approx0.421$." },
    { q: "Which CANNOT be a valid probability?", options: ["1.2", "0", "0.5", "1"], answer: 0,
      explain: "Axiom 1 and 2 force $0\\le P\\le1$; $1.2$ is impossible." }
  ],

  practice: [
    { level: "easy", prompt: "Two coins are flipped. What is the sample space and $P(\\text{at least one head})$?", solution: "$\\Omega=\\{HH,HT,TH,TT\\}$, 4 equally likely. At least one head excludes $TT$: $3/4$. (Or $1-P(TT)=1-1/4$.)" },
    { level: "med", prompt: "$P(A)=0.5$, $P(B)=0.4$, $P(A\\cap B)=0.2$. Find $P(A\\cup B)$ and $P(A^c\\cap B^c)$.", solution: "$P(A\\cup B)=0.5+0.4-0.2=0.7$. By De Morgan, $P(A^c\\cap B^c)=P((A\\cup B)^c)=1-0.7=0.3$." },
    { level: "med", prompt: "A card is drawn. $P(\\text{king or heart})$?", solution: "$P(\\text{king})=4/52$, $P(\\text{heart})=13/52$, overlap (king of hearts) $=1/52$. Union $=4/52+13/52-1/52=16/52=4/13$." },
    { level: "hard", prompt: "AI task: with dropout rate $p=0.5$ on a layer of 100 neurons, what's the probability a specific neuron is kept on all of 3 forward passes, and why does dropout act as ensembling?", solution: "Each pass keeps the neuron with probability $1-p=0.5$ independently, so all-3-kept is $0.5^3=0.125$. Across passes, each forward uses a random subset of neurons — effectively sampling a different sub-network from $2^{100}$ possibilities. Training averages gradients over these sampled sub-networks, and at test time (with weights scaled) the full network approximates the average of that exponential ensemble — which is why dropout reduces overfitting: it's an implicit ensemble built from the sample space of neuron masks." }
  ],

  deepDive: String.raw`<p><strong>Why the axioms matter: consistency and the frequentist vs Bayesian readings.</strong></p>
  <p>Kolmogorov's three axioms are deliberately minimal, yet they're exactly strong enough to forbid incoherent
  beliefs. The famous <strong>Dutch book</strong> argument shows that if your probabilities violate the axioms, a
  cunning bookmaker can construct a set of bets you'll accept that guarantees you lose money no matter the outcome.
  Coherence — not philosophy — is what pins probabilities to these rules. Everything downstream (Bayes, expectation,
  entire distributions) is a consequence of this small core plus the definition of conditional probability.</p>
  <p>The axioms are also neutral between interpretations. A <strong>frequentist</strong> reads $P(A)$ as the long-run
  fraction of times $A$ happens over many repetitions — the view behind the simulations in this lesson and the Law of
  Large Numbers (Track 10). A <strong>Bayesian</strong> reads $P(A)$ as a degree of belief updated by evidence (Lesson
  9.3). Both obey the same axioms, which is why the math is shared even when the philosophy differs. In ML you'll switch
  freely: dropout and data sampling are naturally frequentist, while model uncertainty and priors are naturally
  Bayesian. The rulebook is the same; only the story you tell about the numbers changes.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["9.2"] = {
  subtitle: "Updating probability when you know something — and the chain rule a language model is built on.",

  aiMoment: String.raw`<p>A language model assigns a probability to text by the <strong>chain rule of probability</strong>:
  $P(x_1,\dots,x_n)=\prod_t P(x_t\mid x_{<t})$ — each token conditioned on all the previous ones. That factorization
  <em>is</em> autoregressive generation. <strong>Conditional probability</strong> and <strong>independence</strong> are
  the machinery behind it, behind Naive Bayes, and behind every "given the evidence" calculation in ML.</p>`,

  plainEnglish: String.raw`<p><strong>Conditional probability</strong> is the probability of one thing given that you
  already know another — it shrinks the world to the outcomes consistent with what you know. Two events are
  <strong>independent</strong> if knowing one tells you nothing about the other.</p>`,

  intuition: String.raw`<p>Conditioning on $B$ throws away every outcome outside $B$ and re-normalizes: among the
  $B$-world, what fraction is also $A$? Independence means that fraction is the same as $A$'s share of the whole world —
  $B$ didn't change anything.</p>
  <figure class="figure">
  <svg viewBox="0 0 296 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Conditioning restricts the sample space to B">
    <rect x="12" y="12" width="236" height="126" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <circle cx="150" cy="78" r="54" fill="#0d9488" fill-opacity="0.18" stroke="#0d9488" stroke-width="2"/>
    <circle cx="100" cy="78" r="44" fill="#4f46e5" fill-opacity="0.18" stroke="#4f46e5" stroke-dasharray="3 3"/>
    <path d="M133,44 A44,44 0 0,1 133,112 A54,54 0 0,1 133,44" fill="#7c3aed" fill-opacity="0.4"/>
    <text x="186" y="50" font-size="12" fill="#0d9488" font-family="sans-serif">B (new world)</text>
    <text x="118" y="80" font-size="11" fill="#7c3aed" font-family="sans-serif">A∩B</text>
    <text x="20" y="130" font-size="10" fill="#64748b" font-family="sans-serif">P(A|B) = shaded / area of B</text>
  </svg>
  <figcaption>Given B, only B's outcomes remain; P(A|B) is the overlap as a fraction of B.</figcaption>
  </figure>`,

  formalism: String.raw`<p>For $P(B)>0$, the <strong>conditional probability</strong> is</p>
  $$P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad\text{so}\qquad P(A\cap B)=P(A\mid B)\,P(B)\ \ (\text{multiplication rule}).$$
  <p>Events $A,B$ are <strong>independent</strong> iff $P(A\cap B)=P(A)P(B)$, equivalently $P(A\mid B)=P(A)$. The
  <strong>chain rule</strong> generalizes the multiplication rule:
  $P(x_1,\dots,x_n)=P(x_1)\,P(x_2\mid x_1)\cdots P(x_n\mid x_1,\dots,x_{n-1}).$</p>`,

  derivation: String.raw`<p><strong>The chain rule of probability,</strong> the backbone of autoregressive models.</p>
  <p><strong>Step 1 — apply the multiplication rule once:</strong> $P(x_1,x_2)=P(x_2\mid x_1)P(x_1)$.</p>
  <p><strong>Step 2 — peel off one more variable,</strong> treating $(x_1,x_2)$ as the "known" part:
  $P(x_1,x_2,x_3)=P(x_3\mid x_1,x_2)\,P(x_1,x_2)=P(x_3\mid x_1,x_2)P(x_2\mid x_1)P(x_1)$.</p>
  <p><strong>Step 3 — induct:</strong> repeating $n-1$ times,
  $P(x_1,\dots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_1,\dots,x_{t-1}).\;\blacksquare$</p>
  <p>Plain English: the joint probability of a whole sequence factors into a product of next-token conditionals — which
  is exactly what a language model learns and what it multiplies (in log-space, Lesson 1.2) to score or generate text.</p>`,

  code: [
    { label: "Conditional probability from a joint table", src: String.raw`
import numpy as np
# joint P(weather, umbrella): rows = {sunny, rainy}, cols = {no umbrella, umbrella}
J = np.array([[0.45, 0.05],     # sunny
              [0.10, 0.40]])    # rainy
P_rain = J[1].sum()
P_umb_given_rain = J[1,1] / P_rain         # P(umbrella | rainy)
print("P(rain)               =", round(P_rain, 3))
print("P(umbrella | rain)    =", round(P_umb_given_rain, 3))   # 0.8
print("independent? P(A∩B)==P(A)P(B):",
      np.isclose(J[1,1], P_rain * J[:,1].sum()))               # False -> dependent
` },
    { label: "Chain rule scores a sequence", src: String.raw`
import numpy as np
# toy next-token probabilities for "the cat sat"
p = {"the": 0.10,
     "cat|the": 0.04,
     "sat|the cat": 0.20}
joint = p["the"] * p["cat|the"] * p["sat|the cat"]
print("P(sequence)      =", joint)
print("log P (stable)   =", round(float(np.log(p["the"])+np.log(p["cat|the"])+np.log(p["sat|the cat"])), 4))
` }
  ],

  keyPoints: [
    "$P(A\\mid B)=P(A\\cap B)/P(B)$: condition by restricting to $B$ and re-normalizing.",
    "Multiplication rule: $P(A\\cap B)=P(A\\mid B)P(B)$.",
    "Independent ⇔ $P(A\\cap B)=P(A)P(B)$ ⇔ $P(A\\mid B)=P(A)$.",
    "Chain rule: $P(x_1,\\dots,x_n)=\\prod_t P(x_t\\mid x_{<t})$ — the basis of language models.",
    "Conditional independence (given a third variable) underlies Naive Bayes."
  ],

  commonMistakes: [
    { wrong: "Swapping $P(A\\mid B)$ and $P(B\\mid A)$.", why: "They have different denominators ($P(B)$ vs $P(A)$) and can differ wildly — the prosecutor's fallacy (Lesson 9.3). Bayes relates them." },
    { wrong: "Assuming independence to multiply probabilities.", why: "$P(A\\cap B)=P(A)P(B)$ only if independent. For dependent events use $P(A\\mid B)P(B)$; assuming independence (e.g. Naive Bayes) is a modeling choice, not a fact." },
    { wrong: "Confusing independence with mutual exclusivity.", why: "Mutually exclusive events ($P(A\\cap B)=0$) are dependent (if one happens the other can't). Independence means $P(A\\cap B)=P(A)P(B)$, generally nonzero." }
  ],

  quiz: [
    { q: "From the joint $P(A\\cap B)=0.2$, $P(B)=0.5$, find $P(A\\mid B)$.", options: ["0.4", "0.1", "0.7", "0.25"], answer: 0,
      explain: "$P(A\\mid B)=0.2/0.5=0.4$." },
    { q: "$A,B$ independent with $P(A)=0.3$, $P(B)=0.5$. $P(A\\cap B)=$", options: ["0.15", "0.8", "0.2", "0.5"], answer: 0,
      explain: "Independence: $P(A)P(B)=0.3\\times0.5=0.15$." },
    { q: "A language model's $P(x_1,\\dots,x_n)$ factorizes as…", options: ["$\\prod_t P(x_t\\mid x_{<t})$", "$\\sum_t P(x_t)$", "$\\prod_t P(x_t)$", "$P(x_1)P(x_n)$"], answer: 0,
      explain: "Chain rule: each token conditioned on all previous ones — autoregressive factorization." },
    { q: "If $P(A\\mid B)=P(A)$, then $A$ and $B$ are…", options: ["independent", "mutually exclusive", "equal", "complementary"], answer: 0,
      explain: "Knowing $B$ doesn't change $P(A)$ — the definition of independence." },
    { q: "Two mutually exclusive events with positive probability are…", options: ["dependent", "independent", "identical", "always equally likely"], answer: 0,
      explain: "If one occurs the other can't, so knowing one changes the other's probability to 0 — strongly dependent." }
  ],

  practice: [
    { level: "easy", prompt: "A die shows an even number. What is $P(\\text{it's a }6\\mid\\text{even})$?", solution: "Among evens $\\{2,4,6\\}$, one is 6: $P=1/3$. (Or $\\frac{P(6)}{P(\\text{even})}=\\frac{1/6}{1/2}=1/3$.)" },
    { level: "med", prompt: "Show $P(A\\mid B)=P(A)$ implies $P(B\\mid A)=P(B)$.", solution: "$P(A\\mid B)=P(A)$ means $P(A\\cap B)=P(A)P(B)$ (symmetric in $A,B$). Then $P(B\\mid A)=P(A\\cap B)/P(A)=P(A)P(B)/P(A)=P(B)$. Independence is mutual." },
    { level: "med", prompt: "Cards: draw two without replacement. $P(\\text{both aces})$?", solution: "$P(\\text{first ace})=4/52$; given that, $P(\\text{second ace})=3/51$. Multiplication rule: $\\frac{4}{52}\\cdot\\frac{3}{51}=\\frac{12}{2652}=\\frac{1}{221}$. (Dependent — the first draw changes the second.)" },
    { level: "hard", prompt: "AI task: Naive Bayes assumes features are conditionally independent given the class. Write the resulting classifier score and explain what breaks if features are actually correlated.", solution: "Under the assumption, $P(c\\mid x_1,\\dots,x_d)\\propto P(c)\\prod_i P(x_i\\mid c)$, so the log-score is $\\log P(c)+\\sum_i\\log P(x_i\\mid c)$ — a sum of per-feature evidence. If features are correlated (e.g. duplicated words), the product double-counts their shared evidence, making the classifier <em>overconfident</em>: probabilities pushed toward 0 or 1. The ranking of classes is often still correct (which is why Naive Bayes works in practice), but the calibrated probabilities are unreliable. The fix is to model dependencies (full covariance, or a less naive model) at the cost of more parameters." }
  ],

  deepDive: String.raw`<p><strong>Conditional independence: the assumption that makes big models tractable.</strong></p>
  <p>Two variables are <em>conditionally independent given $Z$</em> if $P(A\cap B\mid Z)=P(A\mid Z)P(B\mid Z)$ — once you
  know $Z$, $A$ and $B$ carry no extra information about each other. This is the single most useful simplifying
  assumption in probabilistic ML, because it slashes the number of parameters you must estimate. A full joint over $d$
  binary features needs $2^d-1$ numbers; assume conditional independence given the class and you need only about $2d$.
  That's the whole reason <strong>Naive Bayes</strong> scales, and it's the structure encoded by <strong>graphical
  models</strong> (Bayesian networks): edges say what depends on what, and missing edges are conditional-independence
  assumptions.</p>
  <p>The same idea, relaxed, defines modern sequence models. A Markov assumption — $x_t$ depends only on a fixed window —
  is conditional independence of $x_t$ from the distant past given the recent past. A transformer drops that window
  (every token can attend to every previous one), keeping the exact chain-rule factorization $\prod_t P(x_t\mid x_{<t})$
  with <em>no</em> independence assumption, which is precisely why it models long-range dependencies that Markov models
  miss — at the $O(n^2)$ cost of Lesson 1.4. So 'what do we assume is independent?' is not a footnote; it's the dial
  that trades parameters and compute against modeling power across the whole field.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["9.4"] = {
  subtitle: "Turn outcomes into numbers, then summarize them: the average and the spread.",

  aiMoment: String.raw`<p>Training minimizes an <strong>expected</strong> loss; the quantity you actually report is
  an average over data. The <strong>variance</strong> of a minibatch gradient controls how noisy each SGD step is, and
  <strong>linearity of expectation</strong> is the quiet workhorse behind nearly every derivation in ML theory. Random
  variables and their two key summaries — mean and variance — are the vocabulary for all of it.</p>`,

  plainEnglish: String.raw`<p>A <strong>random variable</strong> attaches a number to each random outcome (the value on a
  die, the loss on a random example). The <strong>expected value</strong> is its long-run average, weighted by
  probability. The <strong>variance</strong> measures how far the values typically spread from that average.</p>`,

  intuition: String.raw`<p>Draw the distribution as bars whose heights are probabilities. The expected value is the
  balance point of those bars; the variance is how wide they're spread around it — narrow bars mean a predictable
  variable, wide bars a volatile one.</p>
  <figure class="figure">
  <svg viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A PMF as bars with the mean as a balance point">
    <line x1="20" y1="115" x2="240" y2="115" stroke="#94a3b8"/>
    <g fill="#eef0ff" stroke="#4f46e5">
    <rect x="40" y="85" width="26" height="30"/><rect x="80" y="55" width="26" height="60"/>
    <rect x="120" y="40" width="26" height="75"/><rect x="160" y="70" width="26" height="45"/>
    <rect x="200" y="95" width="26" height="20"/>
    </g>
    <line x1="126" y1="115" x2="126" y2="30" stroke="#dc2626" stroke-dasharray="3 3"/>
    <polygon points="126,119 118,134 134,134" fill="#dc2626"/>
    <text x="104" y="147" font-size="11" fill="#dc2626" font-family="sans-serif">E[X] (balance)</text>
  </svg>
  <figcaption>Bar heights are probabilities; the mean balances them, the variance is their spread.</figcaption>
  </figure>`,

  formalism: String.raw`<p>A discrete random variable $X$ has a <strong>probability mass function</strong> $p(x)=P(X=x)$
  with $\sum_x p(x)=1$. Its <strong>expected value</strong> and <strong>variance</strong>:</p>
  $$E[X]=\sum_x x\,p(x),\qquad \operatorname{Var}(X)=E\big[(X-\mu)^2\big]=E[X^2]-\mu^2,\quad \mu=E[X].$$
  <p>The standard deviation is $\sigma=\sqrt{\operatorname{Var}(X)}$. Expectation is <strong>linear</strong>:
  $E[aX+bY]=aE[X]+bE[Y]$ for any $a,b$ — and this holds <em>even when $X,Y$ are dependent</em>.</p>`,

  derivation: String.raw`<p><strong>The computational formula $\operatorname{Var}(X)=E[X^2]-\mu^2$.</strong></p>
  <p><strong>Step 1 — expand the square:</strong> $\operatorname{Var}(X)=E[(X-\mu)^2]=E[X^2-2\mu X+\mu^2].$</p>
  <p><strong>Step 2 — use linearity of expectation:</strong> $=E[X^2]-2\mu\,E[X]+\mu^2$.</p>
  <p><strong>Step 3 — substitute $E[X]=\mu$:</strong> $=E[X^2]-2\mu^2+\mu^2=E[X^2]-\mu^2.\;\blacksquare$ Plain English:
  variance is "the mean of the squares minus the square of the mean."</p>
  <hr class="soft">
  <p><strong>Linearity of expectation, even for dependent variables.</strong> $E[aX+bY]=\sum_{x,y}(ax+by)p(x,y)
  =a\sum_{x,y}x\,p(x,y)+b\sum_{x,y}y\,p(x,y)=aE[X]+bE[Y].$ No independence was used — the joint $p(x,y)$ marginalizes
  away. This is why linearity is so powerful: it never needs an independence assumption (variance, by contrast, does —
  Lesson 9.5).</p>`,

  code: [
    { label: "Expectation and variance from a PMF", src: String.raw`
import numpy as np

x = np.array([0, 1, 2, 3])
p = np.array([0.1, 0.4, 0.3, 0.2])          # PMF, sums to 1
mu  = np.sum(x * p)
var = np.sum((x - mu)**2 * p)
print("E[X]          =", round(float(mu), 3))
print("Var(X)        =", round(float(var), 3))
print("E[X²] - μ²     =", round(float(np.sum(x**2 * p) - mu**2), 3))   # same
` },
    { label: "Linearity of expectation (dependent is fine)", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

X = rng.integers(1, 7, size=1_000_000)      # a die
Y = X                                       # perfectly DEPENDENT on X
print("E[3X + 2Y] measured :", round(float(np.mean(3*X + 2*Y)), 3))
print("3E[X] + 2E[Y]       :", round(float(3*np.mean(X) + 2*np.mean(Y)), 3))
# equal despite X,Y dependent -> linearity needs no independence
` }
  ],

  keyPoints: [
    "A random variable maps outcomes to numbers; its PMF gives $P(X=x)$.",
    "$E[X]=\\sum_x x\\,p(x)$ is the probability-weighted average (balance point).",
    "$\\operatorname{Var}(X)=E[X^2]-\\mu^2$ measures spread; $\\sigma=\\sqrt{\\operatorname{Var}}$.",
    "Expectation is linear: $E[aX+bY]=aE[X]+bE[Y]$ — always, even for dependent variables.",
    "Variance is NOT linear in general; the variance of a sum needs covariance (Lesson 9.5)."
  ],

  commonMistakes: [
    { wrong: "Assuming $\\operatorname{Var}(X+Y)=\\operatorname{Var}(X)+\\operatorname{Var}(Y)$ always.", why: "That holds only when $X,Y$ are uncorrelated. In general there's a $+2\\operatorname{Cov}(X,Y)$ term (Lesson 9.5)." },
    { wrong: "Computing $E[g(X)]$ as $g(E[X])$.", why: "Only for linear $g$. For nonlinear $g$ they differ (Jensen's inequality, Track 12): e.g. $E[X^2]\\neq(E[X])^2$ — their difference is exactly the variance." },
    { wrong: "Forgetting expectation's linearity works without independence.", why: "Many proofs get overcomplicated by assuming independence unnecessarily. $E[X+Y]=E[X]+E[Y]$ needs nothing — use it freely." }
  ],

  quiz: [
    { q: "A fair die's $E[X]$ is…", options: ["3.5", "3", "6", "1"], answer: 0,
      explain: "$\\sum_{k=1}^6 k/6=21/6=3.5$." },
    { q: "$E[X]=2$, $E[X^2]=6$. Then $\\operatorname{Var}(X)=$", options: ["2", "4", "6", "8"], answer: 0,
      explain: "$\\operatorname{Var}=E[X^2]-E[X]^2=6-4=2$." },
    { q: "$E[3X+5]$ where $E[X]=4$ is…", options: ["17", "12", "15", "9"], answer: 0,
      explain: "Linearity (and $E[\\text{const}]=\\text{const}$): $3\\cdot4+5=17$." },
    { q: "For a fair coin ($X=1$ head, $0$ tail), $\\operatorname{Var}(X)=$", options: ["0.25", "0.5", "1", "0"], answer: 0,
      explain: "$E[X]=0.5$, $E[X^2]=0.5$, so $\\operatorname{Var}=0.5-0.25=0.25$ (the max for a Bernoulli, at $p=0.5$)." },
    { q: "$E[X^2]$ versus $(E[X])^2$…", options: ["$E[X^2]\\ge(E[X])^2$ always", "always equal", "$E[X^2]\\le(E[X])^2$", "unrelated"], answer: 0,
      explain: "Their difference is $\\operatorname{Var}(X)\\ge0$, so $E[X^2]\\ge(E[X])^2$ (a case of Jensen)." }
  ],

  practice: [
    { level: "easy", prompt: "$X$ takes value 10 w.p. 0.3 and 0 w.p. 0.7. Find $E[X]$.", solution: "$10\\cdot0.3+0\\cdot0.7=3$." },
    { level: "med", prompt: "A Bernoulli$(p)$ variable ($1$ w.p. $p$). Show $E[X]=p$ and $\\operatorname{Var}(X)=p(1-p)$.", solution: "$E[X]=1\\cdot p+0\\cdot(1-p)=p$. $E[X^2]=1^2\\cdot p=p$, so $\\operatorname{Var}=E[X^2]-E[X]^2=p-p^2=p(1-p)$." },
    { level: "med", prompt: "Two dice; let $S$ be their sum. Find $E[S]$ using linearity.", solution: "$S=X_1+X_2$, each a die with $E=3.5$. Linearity: $E[S]=3.5+3.5=7$ — no need to enumerate the 36 outcomes." },
    { level: "hard", prompt: "AI task: a minibatch gradient averages $B$ per-example gradients. If each has variance $\\sigma^2$ (and they're roughly independent), what is the variance of the minibatch gradient, and what does this say about batch size?", solution: "$\\operatorname{Var}\\!\\big(\\tfrac1B\\sum_{i=1}^B g_i\\big)=\\tfrac1{B^2}\\sum_i\\operatorname{Var}(g_i)=\\tfrac{\\sigma^2}{B}$ (using variance of a sum of independent terms, Lesson 9.5). So gradient noise falls as $1/B$ and its standard deviation as $1/\\sqrt B$ — doubling the batch only cuts noise by $\\sqrt2$. This is why very large batches give diminishing returns on gradient quality (and why learning rate is often scaled with batch size to compensate)." }
  ],

  deepDive: String.raw`<p><strong>Linearity of expectation: the trick that skips the hard part.</strong></p>
  <p>Linearity of expectation is deceptively powerful because it lets you compute the average of a complicated sum
  <em>without</em> ever finding the distribution of that sum. The classic example: how many fixed points does a random
  permutation have on average? Directly, the distribution is a mess. But write the count as $\sum_i \mathbf 1[\text{item }i\text{ stays}]$,
  and by linearity the expected count is $\sum_i P(\text{item }i\text{ stays})=n\cdot\tfrac1n=1$ — one, regardless of $n$,
  in a single line. The indicator-variable trick ("count by summing 0/1 events, then take expectations term by term")
  solves an astonishing range of problems this way.</p>
  <p>In ML this shows up constantly. The <strong>expected loss</strong> over a dataset is a sum of per-example losses, so
  its expectation and gradient decompose example-by-example — the reason minibatching gives an unbiased estimate
  (Track 7). Analysis of <strong>dropout</strong>, <strong>bagging</strong>, and <strong>random features</strong> leans on
  summing indicator or per-component expectations. And crucially, none of it requires independence — a freedom variance
  does not share. That asymmetry (means add freely, variances only add when uncorrelated) is exactly what the next
  lesson formalizes, and it's why covariance is the missing piece for reasoning about spread.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["9.5"] = {
  subtitle: "Do two variables move together? Covariance measures it — and builds the matrix behind PCA.",

  aiMoment: String.raw`<p>The <strong>covariance matrix</strong> is everywhere: PCA diagonalizes it (Track 5), whitening
  uses its inverse square root, and it's the shape parameter of a multivariate Gaussian (Track 10). Correlated features
  cause the collinearity that destabilizes regression (Track 4). And "correlation is not causation" is the caveat behind
  every data-driven claim. Covariance and correlation are how we quantify relationships between variables.</p>`,

  plainEnglish: String.raw`<p><strong>Covariance</strong> asks whether two variables tend to move together: positive if
  they rise and fall together, negative if one rises as the other falls, zero if there's no linear trend.
  <strong>Correlation</strong> is covariance rescaled to a clean $[-1,1]$, so magnitudes are comparable.</p>`,

  intuition: String.raw`<p>Plot pairs of values as a scatter. An up-sloping cloud is positive covariance, a down-sloping
  one negative, and a shapeless blob near zero. Correlation is how tightly the cloud hugs a straight line.</p>
  <figure class="figure">
  <svg viewBox="0 0 330 130" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Scatter plots showing positive, negative, and zero correlation">
    <g fill="#4f46e5"><circle cx="22" cy="95" r="2.5"/><circle cx="35" cy="88" r="2.5"/><circle cx="48" cy="82" r="2.5"/><circle cx="60" cy="70" r="2.5"/><circle cx="72" cy="62" r="2.5"/><circle cx="85" cy="50" r="2.5"/><circle cx="55" cy="76" r="2.5"/><circle cx="40" cy="80" r="2.5"/></g>
    <text x="26" y="120" font-size="10" fill="#4f46e5" font-family="sans-serif">positive</text>
    <g fill="#0d9488"><circle cx="128" cy="50" r="2.5"/><circle cx="142" cy="60" r="2.5"/><circle cx="156" cy="66" r="2.5"/><circle cx="168" cy="76" r="2.5"/><circle cx="182" cy="84" r="2.5"/><circle cx="195" cy="94" r="2.5"/><circle cx="150" cy="70" r="2.5"/><circle cx="175" cy="80" r="2.5"/></g>
    <text x="140" y="120" font-size="10" fill="#0d9488" font-family="sans-serif">negative</text>
    <g fill="#7c3aed"><circle cx="238" cy="60" r="2.5"/><circle cx="252" cy="88" r="2.5"/><circle cx="266" cy="52" r="2.5"/><circle cx="280" cy="80" r="2.5"/><circle cx="294" cy="64" r="2.5"/><circle cx="258" cy="72" r="2.5"/><circle cx="284" cy="56" r="2.5"/><circle cx="244" cy="82" r="2.5"/></g>
    <text x="248" y="120" font-size="10" fill="#7c3aed" font-family="sans-serif">≈ zero</text>
  </svg>
  <figcaption>Up-slope = positive, down-slope = negative, blob = zero covariance.</figcaption>
  </figure>`,

  formalism: String.raw`<p>The <strong>covariance</strong> and <strong>correlation</strong>:</p>
  $$\operatorname{Cov}(X,Y)=E\big[(X-\mu_X)(Y-\mu_Y)\big]=E[XY]-E[X]E[Y],\qquad \rho=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}\in[-1,1].$$
  <p>For a vector, the <strong>covariance matrix</strong> $\Sigma_{ij}=\operatorname{Cov}(X_i,X_j)$ is symmetric PSD.
  Variance of a sum: $\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y)$.
  <strong>Independent ⇒ uncorrelated</strong> ($\operatorname{Cov}=0$), but the converse is false.</p>`,

  derivation: String.raw`<p><strong>Variance of a sum.</strong></p>
  <p><strong>Step 1 — center and expand:</strong> $\operatorname{Var}(X+Y)=E\big[((X-\mu_X)+(Y-\mu_Y))^2\big].$</p>
  <p><strong>Step 2 — square out:</strong> $=E[(X-\mu_X)^2]+E[(Y-\mu_Y)^2]+2E[(X-\mu_X)(Y-\mu_Y)].$</p>
  <p><strong>Step 3 — name the pieces:</strong> $=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y).\;\blacksquare$
  The cross term is why variances only add when the covariance is zero.</p>
  <hr class="soft">
  <p><strong>Independent ⇒ uncorrelated (converse false).</strong> If $X,Y$ independent, $E[XY]=E[X]E[Y]$, so
  $\operatorname{Cov}=E[XY]-E[X]E[Y]=0$. But zero covariance does <em>not</em> imply independence: let $X$ be symmetric
  about 0 and $Y=X^2$. Then $E[XY]=E[X^3]=0$ and $E[X]=0$, so $\operatorname{Cov}(X,Y)=0$ — yet $Y$ is a deterministic
  function of $X$, about as dependent as possible. Covariance only sees <em>linear</em> association.</p>`,

  code: [
    { label: "Covariance matrix and correlation", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

X = rng.normal(size=1000)
Y = 2*X + rng.normal(scale=0.5, size=1000)   # strongly related to X
data = np.vstack([X, Y])
cov = np.cov(data)                            # 2x2 covariance matrix
corr = np.corrcoef(data)
print("covariance matrix:\n", np.round(cov, 3))
print("correlation ρ(X,Y):", round(float(corr[0,1]), 3))   # near +1
` },
    { label: "Uncorrelated but NOT independent", src: String.raw`
import numpy as np
rng = np.random.default_rng(0)

X = rng.normal(size=1_000_000)
Y = X**2                                      # deterministic function of X
print("Cov(X, X²) ≈", round(float(np.cov(X, Y)[0,1]), 4))   # ≈ 0
print("but Y is fully determined by X -> dependent!")
# covariance sees only linear relationships
` }
  ],

  keyPoints: [
    "$\\operatorname{Cov}(X,Y)=E[XY]-E[X]E[Y]$: positive/negative/zero linear co-movement.",
    "Correlation $\\rho=\\operatorname{Cov}/(\\sigma_X\\sigma_Y)\\in[-1,1]$ is scale-free.",
    "$\\operatorname{Var}(X+Y)=\\operatorname{Var}(X)+\\operatorname{Var}(Y)+2\\operatorname{Cov}(X,Y)$.",
    "Independent ⇒ uncorrelated, but uncorrelated does NOT imply independent.",
    "The covariance matrix $\\Sigma$ is symmetric PSD — its eigenvectors are the PCA axes (Track 5)."
  ],

  commonMistakes: [
    { wrong: "Reading zero correlation as independence.", why: "Correlation only captures <em>linear</em> association; $Y=X^2$ has zero correlation yet total dependence. Use it as a linear-relationship measure, not an independence test." },
    { wrong: "Inferring causation from correlation.", why: "A strong $\\rho$ can come from a hidden common cause or coincidence. Correlation is symmetric and directionless; causation needs experiments or causal assumptions." },
    { wrong: "Adding variances of correlated variables.", why: "$\\operatorname{Var}(X+Y)$ needs the $+2\\operatorname{Cov}$ term. Ignoring positive covariance underestimates the true spread (e.g. correlated errors)." }
  ],

  quiz: [
    { q: "If $X,Y$ are independent, $\\operatorname{Cov}(X,Y)$ is…", options: ["0", "1", "$\\sigma_X\\sigma_Y$", "undefined"], answer: 0,
      explain: "Independence gives $E[XY]=E[X]E[Y]$, so covariance is 0." },
    { q: "Correlation always lies in…", options: ["$[-1,1]$", "$[0,1]$", "$[0,\\infty)$", "$(-\\infty,\\infty)$"], answer: 0,
      explain: "Dividing covariance by $\\sigma_X\\sigma_Y$ bounds it to $[-1,1]$ (Cauchy–Schwarz)." },
    { q: "$\\operatorname{Var}(X)=4$, $\\operatorname{Var}(Y)=9$, $\\operatorname{Cov}(X,Y)=2$. $\\operatorname{Var}(X+Y)=$", options: ["17", "13", "15", "21"], answer: 0,
      explain: "$4+9+2\\cdot2=17$." },
    { q: "Zero correlation implies independence…", options: ["only sometimes (e.g. jointly Gaussian)", "always", "never", "only for discrete variables"], answer: 0,
      explain: "In general no ($Y=X^2$). For jointly Gaussian variables, uncorrelated does imply independent — a special, important case." },
    { q: "The covariance matrix of a random vector is always…", options: ["symmetric and positive semidefinite", "diagonal", "invertible", "orthogonal"], answer: 0,
      explain: "$\\Sigma=E[(\\mathbf x-\\boldsymbol\\mu)(\\mathbf x-\\boldsymbol\\mu)^\\top]$ is symmetric PSD; it can be singular (rank-deficient) if features are collinear." }
  ],

  practice: [
    { level: "easy", prompt: "If $\\operatorname{Cov}(X,Y)=6$, $\\sigma_X=2$, $\\sigma_Y=5$, find $\\rho$.", solution: "$\\rho=6/(2\\cdot5)=0.6$." },
    { level: "med", prompt: "Show $\\operatorname{Var}(X-Y)=\\operatorname{Var}(X)+\\operatorname{Var}(Y)-2\\operatorname{Cov}(X,Y)$.", solution: "Apply the sum formula to $X+(-Y)$: $\\operatorname{Var}(-Y)=\\operatorname{Var}(Y)$ and $\\operatorname{Cov}(X,-Y)=-\\operatorname{Cov}(X,Y)$, giving the minus sign on the cross term." },
    { level: "med", prompt: "Give an example of two uncorrelated but dependent variables.", solution: "Let $X\\in\\{-1,0,1\\}$ symmetric and $Y=X^2$. Then $E[X]=0$, $E[XY]=E[X^3]=0$, so $\\operatorname{Cov}=0$ (uncorrelated), yet $Y$ is fully determined by $X$ (dependent)." },
    { level: "hard", prompt: "AI task: whitening transforms features to have identity covariance. Given the covariance matrix $\\Sigma=Q\\Lambda Q^\\top$, write the whitening transform and explain why it helps optimization.", solution: "Whitening applies $\\mathbf z=\\Sigma^{-1/2}(\\mathbf x-\\boldsymbol\\mu)$ where $\\Sigma^{-1/2}=Q\\Lambda^{-1/2}Q^\\top$ (from the spectral decomposition, Track 5). Then $\\operatorname{Cov}(\\mathbf z)=\\Sigma^{-1/2}\\Sigma\\Sigma^{-1/2}=I$: features become uncorrelated with unit variance. This removes the correlation/scale differences that make the loss's Hessian ill-conditioned, shrinking the condition number toward 1 so gradient descent converges fast (Track 8). It's the batch-norm / PCA-whitening intuition: a rounder bowl is easier to roll down." }
  ],

  deepDive: String.raw`<p><strong>The covariance matrix is the bridge to half this course.</strong></p>
  <p>Once you stack variables into a vector, the covariance matrix $\Sigma$ becomes the central object, and every earlier
  track reappears through it. It is symmetric PSD, so the <strong>spectral theorem</strong> (Track 5) gives orthogonal
  eigenvectors — and those eigenvectors are exactly the <strong>principal components</strong>, with eigenvalues equal to
  the variance along each. Its <strong>condition number</strong> (Track 4) predicts how ill-conditioned a Gaussian model
  or a regression will be. Its inverse (the <strong>precision matrix</strong>) encodes conditional independence structure,
  and appears in the exponent of the multivariate Gaussian (Track 10). Whitening uses $\Sigma^{-1/2}$; Mahalanobis
  distance uses $\Sigma^{-1}$ to measure "how many standard deviations away" in correlated space.</p>
  <p>This is why so much of practical ML is really covariance manipulation in disguise. Batch normalization approximately
  whitens activations to improve conditioning. PCA and its cousins compress by keeping the top-variance directions of
  $\Sigma$. Gaussian processes are defined entirely by a covariance (kernel) function. Even attention can be read as a
  data-dependent covariance over tokens. The humble question "do these two move together?" scales up into the matrix that
  organizes probability, geometry, and optimization all at once — and it hands off directly to the continuous
  distributions of Track 10, where $\Sigma$ becomes the shape of the multivariate Gaussian.</p>`
};

/* ------------------------------------------------------------ */
(window.LESSON_CONTENT ||= {})["9.E"] = {
  exam: true,
  subtitle: "Ten mixed problems at interview difficulty. Target: 65 minutes.",

  intro: String.raw`<p>This exam spans all of Track 9: sample spaces and axioms, conditional probability and
  independence, Bayes' theorem, random variables (expectation and variance), and covariance/correlation.
  <strong>Give yourself 65 minutes</strong>, produce each answer before checking, and score with the rubric. About half
  are calculation.</p>`,

  problems: [
    { level: "easy", prompt: "$P(A)=0.6$, $P(B)=0.5$, $P(A\\cap B)=0.3$. Find $P(A\\cup B)$.",
      solution: "Inclusion–exclusion: $0.6+0.5-0.3=0.8$." },
    { level: "easy", prompt: "$P(\\text{at least one head in 3 fair coin flips})$?",
      solution: "Complement: $1-P(\\text{all tails})=1-(1/2)^3=7/8=0.875$." },
    { level: "med", prompt: "From a joint with $P(A\\cap B)=0.24$ and $P(B)=0.4$, find $P(A\\mid B)$; are $A,B$ independent if $P(A)=0.6$?",
      solution: "$P(A\\mid B)=0.24/0.4=0.6=P(A)$, so yes — independent (knowing $B$ doesn't change $P(A)$)." },
    { level: "med", prompt: "Disease prevalence 2%, test sensitivity 90%, false-positive rate 10%. Find $P(\\text{disease}\\mid +)$.",
      solution: "$\\frac{.9\\cdot.02}{.9\\cdot.02+.1\\cdot.98}=\\frac{.018}{.018+.098}=\\frac{.018}{.116}\\approx0.155$ (~15.5%)." },
    { level: "med", prompt: "$X$ has PMF $P(0)=0.5,P(1)=0.3,P(2)=0.2$. Compute $E[X]$ and $\\operatorname{Var}(X)$.",
      solution: "$E[X]=0(.5)+1(.3)+2(.2)=0.7$. $E[X^2]=0+.3+4(.2)=1.1$. $\\operatorname{Var}=1.1-0.49=0.61$." },
    { level: "med", prompt: "$E[X]=5$. Find $E[2X-3]$ and, if $\\operatorname{Var}(X)=4$, $\\operatorname{Var}(2X-3)$.",
      solution: "$E[2X-3]=2\\cdot5-3=7$. $\\operatorname{Var}(2X-3)=2^2\\operatorname{Var}(X)=4\\cdot4=16$ (additive constants don't affect variance; scale squares)." },
    { level: "hard", prompt: "$\\operatorname{Var}(X)=9$, $\\operatorname{Var}(Y)=16$, $\\rho(X,Y)=0.5$. Find $\\operatorname{Var}(X+Y)$.",
      solution: "$\\operatorname{Cov}=\\rho\\sigma_X\\sigma_Y=0.5\\cdot3\\cdot4=6$. $\\operatorname{Var}(X+Y)=9+16+2\\cdot6=37$." },
    { level: "hard", prompt: "Give an example of two variables that are uncorrelated but dependent, and explain what covariance misses.",
      solution: "Let $X$ be symmetric about 0 and $Y=X^2$. $E[X]=0$ and $E[XY]=E[X^3]=0$, so $\\operatorname{Cov}(X,Y)=0$ — uncorrelated. Yet $Y$ is fully determined by $X$. Covariance measures only <em>linear</em> co-movement, so it's blind to the nonlinear (parabolic) dependence." },
    { level: "hard", prompt: "Two dice; let $S$ be the sum. Use linearity to find $E[S]$ and explain why enumerating 36 outcomes is unnecessary.",
      solution: "$S=X_1+X_2$ with $E[X_i]=3.5$. Linearity gives $E[S]=3.5+3.5=7$ regardless of dependence between the dice — expectation of a sum is the sum of expectations, no joint distribution needed." },
    { level: "hard", prompt: "AI task: a language model factorizes $P(x_1,\\dots,x_n)=\\prod_t P(x_t\\mid x_{<t})$. Explain why this is exact (not an approximation) and what a Markov model assumes instead.",
      solution: "The chain rule of probability holds for <em>any</em> joint distribution — repeatedly applying $P(A\\cap B)=P(A\\mid B)P(B)$ gives the product of conditionals with no assumptions, so a transformer that conditions each token on the full history models the joint exactly. A Markov model adds a conditional-independence assumption: $P(x_t\\mid x_{<t})=P(x_t\\mid x_{t-k:t-1})$ (only the last $k$ tokens matter), which is an approximation that discards long-range dependence in exchange for far fewer parameters and $O(n)$ instead of $O(n^2)$ cost." }
  ],

  rubric: String.raw`<p>Count problems solved correctly before checking.</p>
  <ul>
    <li><strong>9–10:</strong> Probability foundations are solid. On to Track 10 (Continuous Distributions) and, when ready, the Bayesian-classifier capstone.</li>
    <li><strong>7–8:</strong> Strong. Revisit Bayes (base rates) or the variance-of-a-sum formula if either slipped.</li>
    <li><strong>5–6:</strong> Re-derive $\\operatorname{Var}=E[X^2]-\\mu^2$ and the covariance sum rule; redo Lessons 9.4 and 9.5.</li>
    <li><strong>Below 5:</strong> Rework the track — expectation, variance, and covariance underpin every statistical and probabilistic method ahead.</li>
  </ul>`
};
