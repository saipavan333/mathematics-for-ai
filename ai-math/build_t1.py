lessons_t1 = r"""
{t:1,n:1,title:"Fractions & Exponents",
ai:"The softmax function in every classifier computes \\(e^{z_i}/\\sum_j e^{z_j}\\)—an exponent divided by a fraction. Temperature scaling writes it as \\(e^{z_i/T}\\). These two primitives appear in virtually every ML formula.",
pe:"A <strong>fraction</strong> \\(a/b\\) is division. An <strong>exponent</strong> \\(b^n\\) multiplies \\(b\\) by itself \\(n\\) times. Together they build softmax, learning rate schedules, and every log-probability calculation.",
int:"Think of exponentiation as compounding: $1 doubling daily is \\(2^n\\) dollars after \\(n\\) days. A fraction is a ratio: 3 cups flour per 4 cookies. Softmax exponentiates each logit to make it positive, then divides by the total to create a valid probability.",
form:`<div class="fb">
<div class="fr"><span class="fl">Fraction</span>\\(a/b,\\;b\\neq0\\)</div>
<div class="fr"><span class="fl">Exponent</span>\\(b^n=\\underbrace{b\\cdot b\\cdots b}_{n}\\)</div>
<div class="fr"><span class="fl">Product rule</span>\\(a^m\\cdot a^n=a^{m+n}\\)</div>
<div class="fr"><span class="fl">Power rule</span>\\((a^m)^n=a^{mn}\\)</div>
<div class="fr"><span class="fl">Zero power</span>\\(a^0=1\\;(a\\neq0)\\)</div>
<div class="fr"><span class="fl">Negative exp</span>\\(a^{-n}=1/a^n\\)</div>
<div class="fr"><span class="fl">Frac. exp</span>\\(a^{p/q}=\\sqrt[q]{a^p}\\)</div>
</div>`,
der:[
["Why \\(a^0=1\\)","Quotient rule: \\(a^n/a^n=a^{n-n}=a^0\\). Any nonzero number over itself equals 1, so \\(a^0=1\\)."],
["Why \\(a^{-n}=1/a^n\\)","\\(a^0/a^n=a^{0-n}=a^{-n}\\). Since \\(a^0=1\\), we get \\(a^{-n}=1/a^n\\)."],
["Product rule","\\(a^m\\cdot a^n\\): \\(m\\) copies of \\(a\\) times \\(n\\) copies equals \\(m+n\\) copies: \\(a^{m+n}\\)."],
["Power rule","\\((a^m)^n\\) means repeat the block \\(a^m\\) exactly \\(n\\) times: total \\(mn\\) copies of \\(a\\), so \\(a^{mn}\\)."],
["Fractional exponent","\\((a^{1/n})^n=a^{n/n}=a\\), so \\(a^{1/n}\\) must be the \\(n\\)th root of \\(a\\). Then \\(a^{p/q}=(a^{1/q})^p\\)."]
],
code:`import numpy as np

# Exponent rules
print(2**3 * 2**4)   # 128 = 2^7  (product rule)
print((2**3)**2)     # 64  = 2^6  (power rule)
print(2**0)          # 1          (zero power)
print(2**-3)         # 0.125      (negative exponent)
print(8**(1/3))      # 2.0        (cube root)

# Softmax: exponents + fractions
def softmax(z, T=1.0):
    e = np.exp(z / T)
    return e / e.sum()

z = np.array([2.0, 1.0, 0.1])
print(softmax(z))          # [0.659 0.242 0.099] sums to 1
print(softmax(z, T=0.1))   # sharper (low T)
print(softmax(z, T=10.0))  # flatter (high T)`,
diag:`<svg viewBox="0 0 420 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;border-radius:7px"><rect width="420" height="170" fill="#0d1117"/><text x="12" y="18" fill="#8b949e" font-size="11">y = 2ˣ  (exponential growth)</text><line x1="38" y1="150" x2="280" y2="150" stroke="#30363d"/><line x1="38" y1="18" x2="38" y2="150" stroke="#30363d"/><polyline fill="none" stroke="#58a6ff" stroke-width="2.5" points="38,148 80,138 120,122 160,100 200,72 240,38 265,18"/><text x="267" y="17" fill="#58a6ff" font-size="10">2ˣ</text><text x="30" y="155" fill="#8b949e" font-size="9">0</text><text x="118" y="155" fill="#8b949e" font-size="9">2</text><text x="196" y="155" fill="#8b949e" font-size="9">4</text><rect x="300" y="28" width="100" height="62" fill="#161b22" rx="5" stroke="#30363d"/><text x="350" y="54" fill="#f59e0b" font-size="20" text-anchor="middle" font-weight="bold">3</text><line x1="310" y1="62" x2="390" y2="62" stroke="#f59e0b" stroke-width="1.5"/><text x="350" y="82" fill="#f59e0b" font-size="20" text-anchor="middle" font-weight="bold">4</text><text x="350" y="105" fill="#8b949e" font-size="9" text-anchor="middle">= 0.75</text><text x="350" y="18" fill="#8b949e" font-size="9" text-anchor="middle">Fraction</text></svg>`,
kp:["\\(a^m\\cdot a^n=a^{m+n}\\): add exponents when multiplying the same base.",
"\\(a^0=1\\) and \\(a^{-n}=1/a^n\\) follow from the quotient rule.",
"Softmax = exponentiate every logit, then divide by their sum.",
"Temperature \\(T\\) divides the exponent: low \\(T\\)→peaked, high \\(T\\)→uniform.",
"\\((a+b)^2=a^2+2ab+b^2\\): never forget the cross-term."],
err:[["\\((a+b)^2=a^2+b^2\\)","Missing the cross-term \\(2ab\\). Full expansion: \\((a+b)^2=a^2+2ab+b^2\\)."],
["\\(a^{m+n}=a^m+a^n\\)","The rule is \\(a^m\\cdot a^n=a^{m+n}\\) — for multiplication, not addition."],
["\\(\\log(x+y)=\\log x+\\log y\\)","Only the product rule holds: \\(\\log(xy)=\\log x+\\log y\\). No rule for sums inside a log."]],
quiz:[
{q:"\\((2^3\\times2^4)/2^5=?\\)",o:["4","128","64","1"],a:0,e:"\\(2^3\\cdot2^4=2^7\\), then \\(2^7/2^5=2^2=4\\)."},
{q:"\\(\\sqrt[3]{8}\\) equals…",o:["\\(8^{1/3}\\)","\\(8^3\\)","\\(3^8\\)","\\(8/3\\)"],a:0,e:"Fractional exponent: \\(8^{1/3}=2\\) since \\(2^3=8\\)."},
{q:"Softmax with \\(T\\to0\\) produces…",o:["One-hot on argmax","Uniform","All zeros","Random"],a:0,e:"Tiny \\(T\\) makes \\(e^{z/T}\\) of the largest \\(z\\) dominate overwhelmingly."},
{q:"\\(5^{-2}=?\\)",o:["\\(1/25\\)","\\(-25\\)","\\(25\\)","\\(-1/25\\)"],a:0,e:"\\(5^{-2}=1/5^2=1/25\\)."},
{q:"\\((x^2)^3/x^4\\) simplifies to…",o:["\\(x^2\\)","\\(x^{10}\\)","\\(x^3\\)","\\(1\\)"],a:0,e:"\\((x^2)^3=x^6\\); \\(x^6/x^4=x^2\\)."}],
prac:[
{p:"Simplify \\((x^3\\cdot x^{-1})/x^2\\).",s:"Numerator: \\(x^3\\cdot x^{-1}=x^2\\). Then \\(x^2/x^2=x^0=1\\).",d:"easy"},
{p:"A learning rate starts at 0.1 and is multiplied by 0.9 each epoch. Find when it drops below 0.01.",s:"Rate at epoch \\(n\\): \\(0.1\\cdot0.9^n<0.01\\Rightarrow 0.9^n<0.1\\). Taking logs: \\(n>\\ln0.1/\\ln0.9\\approx21.9\\). So after epoch 22.",d:"medium"},
{p:"Prove \\((a+b)^3=a^3+3a^2b+3ab^2+b^3\\).",s:"\\((a+b)^3=(a+b)(a+b)^2=(a+b)(a^2+2ab+b^2)=a^3+2a^2b+ab^2+a^2b+2ab^2+b^3=a^3+3a^2b+3ab^2+b^3\\). ∎",d:"medium"},
{p:"Show \\(\\text{softmax}(z+c)=\\text{softmax}(z)\\) for any scalar \\(c\\).",s:"\\(e^{z_i+c}/\\sum_j e^{z_j+c}=e^c e^{z_i}/(e^c\\sum_j e^{z_j})=e^{z_i}/\\sum_j e^{z_j}\\). The \\(e^c\\) cancels. This is why subtracting the max for numerical stability doesn't change the output.",d:"ai"}],
dd:`<div class="ddw"><h3>Why Base \\(e\\) and Not 2 or 10?</h3><p>The exponential \\(e^x\\) is the unique function that is its own derivative: \\((e^x)'=e^x\\). This makes \\(d/dx\\,\\ln x=1/x\\) — the simplest possible form. Using base 10 would give \\(d/dx\\,\\log_{10}x=1/(x\\ln10)\\), cluttering every gradient with \\(\\ln10\\approx2.303\\). The choice of \\(e\\) is not convention — it's the only base where calculus works out cleanly. Every PyTorch <code>torch.log</code> and NumPy <code>np.log</code> is the natural log. When an ML paper writes \\(\\log\\) without a base, it means \\(\\ln\\).</p></div>`
},
{t:1,n:2,title:"Logarithms",
ai:"Language models multiply thousands of small probabilities: \\(P(w_1)P(w_2|w_1)\\cdots\\). Multiplying many numbers smaller than 1 underflows to zero in float32, so practitioners work in log-space, converting the product into a sum: \\(\\log P(w_1)+\\log P(w_2|w_1)+\\cdots\\). The cross-entropy loss \\(-\\log p_\\text{correct}\\) is a negative log-likelihood.",
pe:"A <strong>logarithm</strong> \\(\\log_b x\\) answers: <em>what power of \\(b\\) gives \\(x\\)?</em> The natural log \\(\\ln\\) uses base \\(e\\approx2.718\\) and has the cleanest derivative: \\((\\ln x)'=1/x\\).",
int:"Imagine a number line where tick marks represent 1, 10, 100, 1000 — equal spacing on a <em>log scale</em>. The distance from 1 to 1000 is only 3 units (because \\(\\log_{10}1000=3\\)). Logs compress enormous ranges into manageable ones, turning multiplication into addition.",
form:`<div class="fb">
<div class="fr"><span class="fl">Definition</span>\\(\\log_b x=y\\iff b^y=x\\quad(b>0,b\\neq1,x>0)\\)</div>
<div class="fr"><span class="fl">Product</span>\\(\\log(xy)=\\log x+\\log y\\)</div>
<div class="fr"><span class="fl">Quotient</span>\\(\\log(x/y)=\\log x-\\log y\\)</div>
<div class="fr"><span class="fl">Power</span>\\(\\log(x^n)=n\\log x\\)</div>
<div class="fr"><span class="fl">Change of base</span>\\(\\log_b x=\\ln x/\\ln b\\)</div>
<div class="fr"><span class="fl">Key values</span>\\(\\ln e=1,\\;\\ln1=0,\\;\\ln x\\to-\\infty\\text{ as }x\\to0^+\\)</div>
</div>`,
der:[
["Product rule","Let \\(\\log_b x=m\\) and \\(\\log_b y=n\\), so \\(b^m=x,b^n=y\\). Then \\(xy=b^m\\cdot b^n=b^{m+n}\\). Taking \\(\\log_b\\): \\(\\log_b(xy)=m+n=\\log_b x+\\log_b y\\). ∎"],
["Power rule","\\(x^n=x\\cdot x\\cdots x\\) (\\(n\\) times). Apply product rule \\(n\\) times: \\(\\log(x^n)=n\\log x\\)."],
["Change of base","Let \\(y=\\log_b x\\), so \\(b^y=x\\). Take \\(\\ln\\): \\(y\\ln b=\\ln x\\). Solve: \\(y=\\ln x/\\ln b\\)."],
["Log-sum-exp stability","Naive \\(\\log\\sum_i e^{z_i}\\) overflows for large \\(z\\). Let \\(m=\\max_i z_i\\). Then \\(\\log\\sum_i e^{z_i}=m+\\log\\sum_i e^{z_i-m}\\). The shifted values \\(z_i-m\\le0\\) so \\(e^{z_i-m}\\le1\\): no overflow."],
["Cross-entropy from NLL","Minimize the negative log-likelihood: \\(-\\sum_i\\log P(y_i|x_i)\\). For a categorical model outputting \\(\\hat p\\), this is \\(-\\log\\hat p_{y_i}\\) averaged over samples. That's cross-entropy loss."]
],
code:`import numpy as np

# Log rules
print(np.log(6))             # ln(2×3)
print(np.log(2)+np.log(3))  # same: ln2 + ln3

# Log-space multiplication avoids underflow
probs = [0.9, 0.8, 0.7, 0.6, 0.5]
print(np.prod(probs))                     # may underflow for long seqs
print(np.exp(sum(np.log(p) for p in probs)))  # identical, safe

# Numerically stable log-sum-exp
def lse(z):
    m = z.max()
    return m + np.log(np.sum(np.exp(z - m)))

z = np.array([1000., 1001., 1002.])
# np.log(np.sum(np.exp(z)))  → would overflow!
print(lse(z))   # 1002.407... (correct)

# Cross-entropy loss = -log(correct class probability)
def ce_loss(y_true, probs):
    return -np.log(probs[y_true] + 1e-9)

probs = np.array([0.7, 0.2, 0.1])
print(ce_loss(0, probs))   # -log(0.7) ≈ 0.357`,
diag:`<svg viewBox="0 0 420 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;border-radius:7px"><rect width="420" height="160" fill="#0d1117"/><text x="12" y="18" fill="#8b949e" font-size="10">ln(x) and eˣ are inverses — each undoes the other</text><line x1="38" y1="140" x2="390" y2="140" stroke="#30363d"/><line x1="200" y1="10" x2="200" y2="148" stroke="#30363d"/><polyline fill="none" stroke="#58a6ff" stroke-width="2" points="42,138 80,128 120,113 160,95 200,75 240,52 270,32 290,18"/><text x="292" y="17" fill="#58a6ff" font-size="10">eˣ</text><polyline fill="none" stroke="#f59e0b" stroke-width="2" points="45,145 60,132 80,122 120,107 160,96 200,75 250,62 310,52 380,44"/><text x="382" y="43" fill="#f59e0b" font-size="10">ln(x)</text><line x1="38" y1="75" x2="390" y2="75" stroke="#30363d" stroke-dasharray="3,3"/><text x="4" y="79" fill="#8b949e" font-size="8">y=0</text><circle cx="200" cy="75" r="3.5" fill="#3fb950"/><text x="205" y="68" fill="#3fb950" font-size="9">(0,1)↔(1,0)</text></svg>`,
kp:["\\(\\log(xy)=\\log x+\\log y\\): multiplication becomes addition in log-space.",
"Cross-entropy loss \\(-\\log p_\\text{correct}\\) penalises low confidence.",
"Log-sum-exp trick: subtract max before exponentiating to prevent overflow.",
"\\(\\ln\\) and \\(\\exp\\) are inverses: \\(e^{\\ln x}=x\\) and \\(\\ln(e^x)=x\\).",
"\\(\\ln x\\) is defined only for \\(x>0\\): always apply softmax before taking log."],
err:[["\\(\\log(x+y)=\\log x+\\log y\\)","Only the product rule holds. There is no log rule for sums."],
["\\(\\log(xy)=\\log x\\cdot\\log y\\)","Multiplication inside the log becomes addition outside: \\(\\log(xy)=\\log x+\\log y\\)."],
["\\(\\ln\\) is defined for \\(x\\le0\\)","\\(\\ln x\\) requires \\(x>0\\). A negative model output passed directly to \\(\\log\\) gives NaN."]],
quiz:[
{q:"\\(\\ln(e^3)=?\\)",o:["3","\\(e^3\\)","\\(1/3\\)","\\(\\ln3\\)"],a:0,e:"\\(\\ln\\) and \\(\\exp\\) are inverses."},
{q:"CE loss for correct class predicted at 0.5?",o:["\\(\\ln2\\approx0.693\\)","0.5","2","\\(-0.5\\)"],a:0,e:"\\(-\\ln0.5=\\ln2\\approx0.693\\)."},
{q:"Why log-probabilities instead of raw probabilities in LMs?",o:["Avoid underflow when multiplying many small numbers","Logs are faster","Probabilities can't be multiplied","Logs are positive"],a:0,e:"Products of thousands of sub-1 probabilities flush to 0 in float32. Summing logs avoids this."},
{q:"\\(\\log_2 64=?\\)",o:["6","8","32","4"],a:0,e:"\\(2^6=64\\)."},
{q:"If \\(\\ln x=5\\), then \\(x=?\\)",o:["\\(e^5\\)","5","\\(e/5\\)","\\(\\ln5\\)"],a:0,e:"Definition: \\(\\ln x=5\\Rightarrow x=e^5\\)."}],
prac:[
{p:"Prove \\(\\log_b x=\\ln x/\\ln b\\) from first principles.",s:"Let \\(y=\\log_b x\\Rightarrow b^y=x\\). Take \\(\\ln\\) both sides: \\(y\\ln b=\\ln x\\). Divide: \\(y=\\ln x/\\ln b\\). ∎",d:"easy"},
{p:"A model assigns probabilities 0.9, 0.85, 0.95, 0.7 to four consecutive correct tokens. Compute the NLL.",s:"NLL \\(=-(\\ln0.9+\\ln0.85+\\ln0.95+\\ln0.7)=-(-0.105-0.163-0.051-0.357)=0.676\\).",d:"medium"},
{p:"Implement the log-sum-exp trick and verify it matches \\(\\ln\\sum e^{z_i}\\) for small inputs.",s:"<code>def lse(z): m=z.max(); return m+np.log(np.sum(np.exp(z-m)))\nz=np.array([1.,2.,3.])\nassert abs(np.log(np.sum(np.exp(z)))-lse(z))<1e-10</code>",d:"medium"},
{p:"Show algebraically that \\(\\text{softmax}\\) output is unchanged if you replace \\(z_i\\) with \\(z_i+\\ln c\\) for any \\(c>0\\).",s:"\\(e^{z_i+\\ln c}/\\sum_j e^{z_j+\\ln c}=ce^{z_i}/(c\\sum_j e^{z_j})=e^{z_i}/\\sum_j e^{z_j}\\). Adding \\(\\ln c\\) scales all exponentials by \\(c\\), which cancels in the fraction.",d:"ai"}],
dd:`<div class="ddw"><h3>Natural Log as Area Under \\(1/x\\)</h3><p>There is a beautiful geometric definition: \\(\\ln x = \\int_1^x \\frac{1}{t}\\,dt\\). This is the area under the curve \\(y=1/t\\) from 1 to \\(x\\). When \\(x>1\\) the area is positive; when \\(x<1\\) it is negative (integral from right to left). The product rule follows immediately: \\(\\ln(xy)=\\int_1^{xy}1/t\\,dt=\\int_1^x1/t\\,dt+\\int_x^{xy}1/t\\,dt\\). In the second integral, substitute \\(u=t/x\\): \\(\\int_1^y1/u\\,du=\\ln y\\). So \\(\\ln(xy)=\\ln x+\\ln y\\) without needing the exponential at all.</p></div>`
},
{t:1,n:3,title:"Summation & Product Notation",
ai:"The MSE loss is \\(\\frac{1}{N}\\sum_{i=1}^N(y_i-\\hat y_i)^2\\). Batch norm computes \\(\\mu=\\frac{1}{B}\\sum_{i=1}^B x_i\\). The attention score matrix \\(A_{ij}=\\sum_k Q_{ik}K_{jk}/\\sqrt{d}\\) is a sum over the head dimension. These symbols compress enormous loops into single expressions.",
pe:"\\(\\sum\\) means \"add all these up.\" \\(\\prod\\) means \"multiply all these together.\" The running index (usually \\(i\\) or \\(k\\)) is just a counter that steps through the range.",
int:"Think of \\(\\sum_{i=1}^N a_i\\) as instructions on a factory assembly line: \"pick up boxes labelled 1 through \\(N\\) and stack them.\" \\(\\prod\\) says to chain them together multiplicatively. The index letter is irrelevant—\\(\\sum_i a_i=\\sum_k a_k\\).",
form:`<div class="fb">
<div class="fr"><span class="fl">Sum</span>\\(\\displaystyle\\sum_{i=m}^n a_i=a_m+a_{m+1}+\\cdots+a_n\\)</div>
<div class="fr"><span class="fl">Product</span>\\(\\displaystyle\\prod_{i=m}^n a_i=a_m\\cdot a_{m+1}\\cdots a_n\\)</div>
<div class="fr"><span class="fl">Linearity</span>\\(\\sum_i(ca_i+b_i)=c\\sum_i a_i+\\sum_i b_i\\)</div>
<div class="fr"><span class="fl">Double sum</span>\\(\\sum_i\\sum_j a_{ij}=\\sum_j\\sum_i a_{ij}\\) (swappable)</div>
<div class="fr"><span class="fl">Geometric</span>\\(\\sum_{k=0}^{n-1}r^k=(1-r^n)/(1-r)\\quad r\\neq1\\)</div>
</div>`,
der:[
["Constants factor out","\\(\\sum_i ca_i=ca_1+ca_2+\\cdots=c(a_1+\\cdots)=c\\sum_i a_i\\). Plain distribution law."],
["Sums split","\\(\\sum_i(a_i+b_i)=(a_1+b_1)+\\cdots=(a_1+\\cdots)+(b_1+\\cdots)=\\sum_i a_i+\\sum_i b_i\\)."],
["Double sum swaps","For finite sums, addition is commutative and associative: we can add in any order."],
["Geometric series","Let \\(S=1+r+\\cdots+r^{n-1}\\). Multiply by \\(r\\): \\(rS=r+\\cdots+r^n\\). Subtract: \\(S(1-r)=1-r^n\\), so \\(S=(1-r^n)/(1-r)\\). ∎"],
["Key non-rule","\\(\\sum_i(a_i b_i)\\neq(\\sum_i a_i)(\\sum_i b_i)\\). A dot product cannot be factored this way—this is why attention is expensive."]
],
code:`import numpy as np

a = np.array([1.,4.,9.,16.,25.])
print(np.sum(a))         # 55  (vectorized)
print(sum(a))            # 55  (Python loop — same, but slower)

# Matrix: double sum
A = np.array([[1,2,3],[4,5,6]])
print(np.sum(A))         # 21  (sum all elements)
print(np.sum(A, axis=0)) # [5 7 9]  (sum over rows)
print(np.sum(A, axis=1)) # [6 15]   (sum over cols)

# MSE
y     = np.array([3., -0.5, 2., 7.])
y_hat = np.array([2.5, 0.,  2., 8.])
mse   = np.mean((y - y_hat)**2)
print(f"MSE = {mse:.4f}")  # 0.3750

# Product via log-sum for numerical safety
log_probs = np.log([0.9, 0.8, 0.7])
joint = np.exp(np.sum(log_probs))
print(f"Joint prob = {joint:.4f}")`,
diag:`<svg viewBox="0 0 420 150" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;border-radius:7px"><rect width="420" height="150" fill="#0d1117"/><text x="12" y="18" fill="#8b949e" font-size="10">∑ stacks (adds); ∏ chains (multiplies)</text><rect x="12" y="28" width="38" height="26" fill="rgba(88,166,255,.12)" stroke="#58a6ff" rx="3"/><text x="31" y="46" fill="#58a6ff" font-size="12" text-anchor="middle">a₁</text><rect x="55" y="28" width="38" height="26" fill="rgba(88,166,255,.12)" stroke="#58a6ff" rx="3"/><text x="74" y="46" fill="#58a6ff" font-size="12" text-anchor="middle">a₂</text><rect x="98" y="28" width="38" height="26" fill="rgba(88,166,255,.12)" stroke="#58a6ff" rx="3"/><text x="117" y="46" fill="#58a6ff" font-size="12" text-anchor="middle">a₃</text><text x="140" y="46" fill="#8b949e" font-size="10">+…</text><rect x="12" y="63" width="170" height="26" fill="rgba(63,185,80,.1)" stroke="#3fb950" rx="3"/><text x="97" y="81" fill="#3fb950" font-size="11" text-anchor="middle">Σ aᵢ  (total)</text><line x1="205" y1="28" x2="205" y2="110" stroke="#30363d"/><text x="218" y="44" fill="#f59e0b" font-size="10">∏: a₁ × a₂ × a₃ ×…</text><text x="218" y="62" fill="#8b949e" font-size="10">= exp(Σ ln aᵢ)</text><text x="218" y="80" fill="#8b949e" font-size="10">Always work in log-space!</text><rect x="216" y="90" width="194" height="24" fill="rgba(240,136,62,.08)" stroke="#f0883e" rx="3"/><text x="313" y="106" fill="#f0883e" font-size="10" text-anchor="middle">Σ log aᵢ  (sum of logs)</text></svg>`,
kp:["\\(\\sum_i(ca_i+b_i)=c\\sum_i a_i+\\sum_i b_i\\): sums are linear.",
"\\(\\sum_i a_i b_i\\neq(\\sum_i a_i)(\\sum_i b_i)\\): dot products can't be factored.",
"\\(\\prod_i a_i=\\exp(\\sum_i\\ln a_i)\\): always compute products via log-sums.",
"Double sums \\(\\sum_i\\sum_j\\) can swap order when both ranges are finite.",
"Vectorized \\(\\texttt{np.sum}\\) replaces Python loops; prefer it always."],
err:[["\\(\\sum_i(a_ib_i)=(\\sum_ia_i)(\\sum_ib_i)\\)","Test with \\(a=[1,2],b=[3,4]\\): LHS=11, RHS=21. The factoring is wrong."],
["The dummy index letter matters","\\(\\sum_i a_i=\\sum_k a_k\\). The letter is irrelevant; range and expression are what matter."],
["Empty sum is undefined","By convention, an empty sum (upper limit below lower) equals 0; an empty product equals 1."]],
quiz:[
{q:"\\(\\sum_{i=1}^4 i^2=?\\)",o:["30","20","14","24"],a:0,e:"\\(1+4+9+16=30\\)."},
{q:"Which correctly factors a constant?",o:["\\(\\sum_i ca_i=c\\sum_i a_i\\)","\\(\\sum_i ca_i=c^n\\sum_i a_i\\)","\\(\\sum_i(a_i+c)=c\\sum_ia_i\\)","\\(c\\sum_ia_i=\\sum_ia_i^c\\)"],a:0,e:"Plain linearity of addition."},
{q:"\\(\\sum_{k=0}^5 2^k=?\\)",o:["63","64","32","31"],a:0,e:"Geometric: \\((1-2^6)/(1-2)=63\\). Or: 1+2+4+8+16+32=63."},
{q:"The MSE \\(\\frac{1}{N}\\sum_i(y_i-\\hat y_i)^2\\). What is its value for \\(y=[1,3],\\hat y=[2,2]\\)?",o:["1","2","0.5","4"],a:0,e:"Errors: \\((-1)^2+(1)^2=2\\). Mean: \\(2/2=1\\)."},
{q:"\\(\\sum_{i=1}^N\\sum_{j=1}^N A_{ij}B_{ij}\\) equals…",o:["Frobenius inner product \\(\\langle A,B\\rangle_F\\)","Matrix product \\(AB\\)","Trace of \\(AB\\)","\\(\\|A+B\\|^2\\)"],a:0,e:"Summing element-wise products over all \\(i,j\\) is the Frobenius inner product."}],
prac:[
{p:"Compute \\(\\sum_{k=1}^5 k\\) and \\(\\prod_{k=1}^5 k\\) by hand, then verify with NumPy.",s:"Sum: 15. Product: 120 (= 5!). <code>import numpy as np; k=np.arange(1,6); print(k.sum(), k.prod())</code>",d:"easy"},
{p:"Show \\(\\sum_{i=1}^N(y_i-\\bar y)=0\\) where \\(\\bar y=\\frac1N\\sum_i y_i\\).",s:"\\(\\sum_i(y_i-\\bar y)=\\sum_iy_i-N\\bar y=N\\bar y-N\\bar y=0\\). The mean balances all deviations.",d:"medium"},
{p:"Prove the geometric series formula \\(\\sum_{k=0}^{n-1}r^k=(1-r^n)/(1-r)\\).",s:"Let \\(S=1+r+\\cdots+r^{n-1}\\). Multiply by \\(r\\): \\(rS=r+\\cdots+r^n\\). Subtract: \\(S(1-r)=1-r^n\\Rightarrow S=(1-r^n)/(1-r)\\). ∎",d:"medium"},
{p:"Implement batch-norm mean and variance using only \\(\\sum\\) (no np.mean/np.var).",s:"<code>x=np.array([2.,4.,6.,8.,10.])\nN=len(x)\nmu=np.sum(x)/N\nvar=np.sum((x-mu)**2)/N\nprint(mu,var) # 6.0, 8.0</code>",d:"ai"}],
dd:`<div class="ddw"><h3>Einstein Summation (einsum)</h3><p>In ML papers and PyTorch, repeated indices signal summation: \\(C_{ik}=A_{ij}B_{jk}\\) means sum over \\(j\\). NumPy and PyTorch expose this as <code>np.einsum('ij,jk->ik', A, B)</code>. This unifies dot products (<code>'i,i->'</code>), matrix-vector products (<code>'ij,j->i'</code>), outer products (<code>'i,j->ij'</code>), and the attention score tensor (<code>'bid,bjd->bij'</code>). Mastering einsum lets you write and read complex tensor operations in one line without introducing intermediate variables.</p></div>`
},
{t:1,n:4,title:"Functions: Domain, Range & Composition",
ai:"Every neural-network layer is a function: it maps an input tensor to an output tensor. The forward pass is a composition \\(f^{(L)}\\circ\\cdots\\circ f^{(1)}\\). Backpropagation applies the chain rule to that composition. Knowing whether a function is invertible matters for flow-based generative models and normalising flows.",
pe:"A <strong>function</strong> \\(f:A\\to B\\) maps each input in \\(A\\) to exactly one output in \\(B\\). The <strong>domain</strong> is the set of valid inputs; the <strong>range</strong> is the set of outputs actually produced.",
int:"A vending machine is a function: one coin in → exactly one snack out. Two buttons may give the same snack (many-to-one is fine), but one button must never give two different snacks. \"Composition\" \\(g\\circ f\\) means run \\(f\\) first, then feed the result into \\(g\\).",
form:`<div class="fb">
<div class="fr"><span class="fl">Function</span>\\(f:A\\to B\\) — exactly one output per input</div>
<div class="fr"><span class="fl">Domain</span>\\(\\text{dom}(f)=A\\)</div>
<div class="fr"><span class="fl">Range</span>\\(\\text{Im}(f)=\\{f(x):x\\in A\\}\\subseteq B\\)</div>
<div class="fr"><span class="fl">Composition</span>\\((g\\circ f)(x)=g(f(x))\\)</div>
<div class="fr"><span class="fl">Inverse</span>\\(f^{-1}\\) exists iff \\(f\\) is bijective (1-to-1 and onto)</div>
<div class="fr"><span class="fl">Identity</span>\\(f\\circ f^{-1}=f^{-1}\\circ f=\\text{id}\\)</div>
</div>`,
der:[
["Domain of \\(\\ln x\\)","\\(\\ln x\\) requires \\(x>0\\). Passing negative values gives NaN. Softmax must precede log in cross-entropy to guarantee positive inputs."],
["Composition is not commutative","\\(f(x)=x^2\\), \\(g(x)=x+1\\): \\((f\\circ g)(2)=f(3)=9\\) but \\((g\\circ f)(2)=g(4)=5\\). Layer order in a network is not interchangeable."],
["Inverse of a composition","\\((g\\circ f)^{-1}=f^{-1}\\circ g^{-1}\\): reverse order, like taking off shoes then socks in reverse."],
["Bijection condition","For \\(f^{-1}\\) to exist: (a) injective — different inputs → different outputs; (b) surjective — every element of \\(B\\) is reached. ReLU fails (a): both \\(-3\\) and \\(-5\\) map to 0."],
["NN as composition","A network with \\(L\\) layers is \\(F=f^{(L)}\\circ\\cdots\\circ f^{(1)}\\). Backprop computes \\(\\partial\\ell/\\partial\\theta^{(k)}\\) by applying the chain rule \\(L-k\\) times from the output."]
],
code:`import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))
relu    = lambda x: np.maximum(0, x)
tanh    = np.tanh

x = np.linspace(-3, 3, 7)
print("sigmoid range (0,1):", sigmoid(x).round(3))
print("relu range [0,∞):   ", relu(x).round(3))
print("tanh range (-1,1):  ", tanh(x).round(3))

# Composition = forward pass
np.random.seed(0)
W1, b1 = np.random.randn(4,3), np.zeros(4)
W2, b2 = np.random.randn(2,4), np.zeros(2)

def layer(W, b, act, x): return act(W @ x + b)

x_in = np.array([1., 2., 3.])
h = layer(W1, b1, relu, x_in)         # f^(1)
y = layer(W2, b2, sigmoid, h)         # f^(2) ∘ f^(1)
print("output:", y.round(4))

# Verify: log is the inverse of exp
a = 5.0
print(np.exp(np.log(a)))  # 5.0  (composition = identity)`,
diag:`<svg viewBox="0 0 420 140" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;border-radius:7px"><rect width="420" height="140" fill="#0d1117"/><text x="12" y="16" fill="#8b949e" font-size="10">Forward = compose left→right  |  Backward = chain rule right→left</text><rect x="10" y="28" width="55" height="26" fill="#21262d" rx="4" stroke="#30363d"/><text x="37" y="45" fill="#8b949e" font-size="10" text-anchor="middle">x ∈ ℝⁿ</text><rect x="90" y="28" width="75" height="26" fill="rgba(88,166,255,.12)" rx="4" stroke="#58a6ff"/><text x="127" y="45" fill="#58a6ff" font-size="10" text-anchor="middle">f⁽¹⁾=relu(W₁x+b₁)</text><rect x="190" y="28" width="75" height="26" fill="rgba(240,136,62,.12)" rx="4" stroke="#f0883e"/><text x="227" y="45" fill="#f0883e" font-size="10" text-anchor="middle">f⁽²⁾=relu(W₂h+b₂)</text><rect x="290" y="28" width="120" height="26" fill="rgba(163,113,247,.12)" rx="4" stroke="#a371f7"/><text x="350" y="45" fill="#a371f7" font-size="10" text-anchor="middle">f⁽³⁾=softmax(W₃z+b₃)</text><defs><marker id="ar" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0L5,2.5L0,5Z" fill="#3fb950"/></marker></defs><line x1="65" y1="41" x2="89" y2="41" stroke="#3fb950" stroke-width="1.3" marker-end="url(#ar)"/><line x1="165" y1="41" x2="189" y2="41" stroke="#3fb950" stroke-width="1.3" marker-end="url(#ar)"/><line x1="265" y1="41" x2="289" y2="41" stroke="#3fb950" stroke-width="1.3" marker-end="url(#ar)"/><text x="210" y="90" fill="#8b949e" font-size="10" text-anchor="middle">F = f⁽³⁾ ∘ f⁽²⁾ ∘ f⁽¹⁾</text><text x="210" y="110" fill="#8b949e" font-size="10" text-anchor="middle">∂ℓ/∂x = (∂f⁽³⁾/∂z)(∂f⁽²⁾/∂h)(∂f⁽¹⁾/∂x)  ← chain rule</text></svg>`,
kp:["A function maps every input to exactly one output — no ambiguity.",
"Domain = valid inputs; range = actual outputs (may be smaller than codomain).",
"Composition \\(g\\circ f\\): apply \\(f\\) first, then \\(g\\). Order always matters.",
"Inverse exists iff the function is bijective (injective + surjective).",
"The forward pass is function composition; backprop is the chain rule on it."],
err:[["Confusing range with codomain","Codomain is the declared output type (e.g., \\(\\mathbb{R}\\)). Range is the subset actually produced. Sigmoid has codomain \\(\\mathbb{R}\\) but range \\((0,1)\\)."],
["\\(f\\circ g=g\\circ f\\)","Composition is generally not commutative. Swapping layer order changes the network."],
["ReLU has an inverse","ReLU maps all \\(x\\le0\\) to 0, so it's not injective. No inverse exists."]],
quiz:[
{q:"Range of sigmoid \\(\\sigma(x)=1/(1+e^{-x})\\)?",o:["\\((0,1)\\)","\\([0,1]\\)","\\(\\mathbb{R}\\)","\\((-1,1)\\)"],a:0,e:"Approaches 0 and 1 but never reaches them; range is open interval \\((0,1)\\)."},
{q:"\\((f\\circ g)(3)\\) with \\(f(x)=2x\\), \\(g(x)=x+1\\)?",o:["8","7","9","6"],a:0,e:"\\(g(3)=4\\), then \\(f(4)=8\\)."},
{q:"Which has the same value as its own derivative?",o:["\\(e^x\\)","\\(\\sigma(x)\\)","\\(\\text{ReLU}(x)\\)","\\(\\tanh(x)\\)"],a:0,e:"\\((e^x)'=e^x\\) — the defining property of the exponential."},
{q:"Why doesn't ReLU have an inverse?",o:["Not injective — −1 and −2 both map to 0","Range isn't ℝ","Not continuous","Grows too fast"],a:0,e:"Non-injectivity prevents inversion: can't recover the original value from the output 0."},
{q:"\\((g\\circ f)^{-1}=?\\)",o:["\\(f^{-1}\\circ g^{-1}\\)","\\(g^{-1}\\circ f^{-1}\\)","\\(f^{-1}\\cdot g^{-1}\\)","\\((f\\circ g)\\)"],a:0,e:"Reverse order: undo \\(g\\) first, then \\(f\\)."}],
prac:[
{p:"Find domain and range of \\(f(x)=\\sqrt{\\ln x}\\).",s:"Need \\(\\ln x\\ge0\\Rightarrow x\\ge1\\). Domain: \\([1,\\infty)\\). Range: \\([0,\\infty)\\).",d:"easy"},
{p:"Show \\((g\\circ f)^{-1}=f^{-1}\\circ g^{-1}\\) for \\(f(x)=2x,g(x)=x+3\\).",s:"\\(g\\circ f\\)(x)=2x+3\\). Inverse: \\((y-3)/2\\). Check \\(f^{-1}(g^{-1}(y))=f^{-1}(y-3)=(y-3)/2\\). ✓",d:"medium"},
{p:"Prove that if \\(f\\) and \\(g\\) are both injective, then \\(g\\circ f\\) is injective.",s:"Suppose \\(g(f(x_1))=g(f(x_2))\\). Since \\(g\\) injective: \\(f(x_1)=f(x_2)\\). Since \\(f\\) injective: \\(x_1=x_2\\). ∎",d:"hard"},
{p:"Implement a 2-layer ReLU network as function composition.",s:"<code>f1=lambda x:np.maximum(0,W1@x+b1)\nf2=lambda h:1/(1+np.exp(-(W2@h+b2)))\noutput=(f2 ∘ f1)(x)  # = f2(f1(x))</code>",d:"ai"}],
dd:`<div class="ddw"><h3>Bijections in Flow-Based Models</h3><p>Normalising flows (RealNVP, Glow, NICE) build a generative model as a composition of bijective transformations \\(f_1,\\ldots,f_K\\) so that \\(p_X(x)=p_Z(f_K(\\cdots f_1(x)\\cdots))\\cdot|\\det J_{F}(x)|\\), where \\(J_F\\) is the Jacobian of the full composition. Because each \\(f_k\\) is invertible, you can both generate samples (run the inverse \\(f_k^{-1}\\) from \\(z\\to x\\)) and evaluate exact likelihoods (run forward \\(x\\to z\\)). Non-invertible activations like ReLU are forbidden in these architectures for exactly this reason. This is the real-world consequence of the theory of function inverses.</p></div>`
},
{t:1,n:5,title:"Big-O Notation in ML",
ai:"Self-attention in the Transformer paper (Vaswani et al. 2017) is \\(O(n^2 d)\\) in sequence length \\(n\\). For \\(n=128{,}000\\) tokens this is the dominant cost, motivating FlashAttention, linear attention, and state-space models. Matrix multiplication is \\(O(n^3)\\) for square matrices and dominates the FLOP count of large language models.",
pe:"<strong>Big-O notation</strong> describes how runtime or memory <em>scales</em> with input size—it captures the dominant growth rate, ignoring constant factors and lower-order terms.",
int:"If sorting 1000 items takes 1 second with an \\(O(n^2)\\) algorithm, sorting 10000 items takes \\(\\approx100\\) seconds (10× data → 100× time). An \\(O(n\\log n)\\) algorithm takes only \\(\\approx13\\) seconds. The gap compounds at scale.",
form:`<div class="fb">
<div class="fr"><span class="fl">Definition</span>\\(f(n)=O(g(n))\\) iff \\(\\exists C>0,n_0\\) s.t. \\(f(n)\\le Cg(n)\\forall n\\ge n_0\\)</div>
<div class="fr"><span class="fl">Drop constants</span>\\(O(5n^2)=O(n^2)\\)</div>
<div class="fr"><span class="fl">Drop lower terms</span>\\(O(n^2+n)=O(n^2)\\)</div>
<div class="fr"><span class="fl">Ordering</span>\\(O(1){<}O(\\log n){<}O(n){<}O(n\\log n){<}O(n^2){<}O(n^3){<}O(2^n)\\)</div>
<div class="fr"><span class="fl">ML examples</span>Attention: \\(O(n^2d)\\); matmul \\((m{\\times}k)(k{\\times}p)\\): \\(O(mkp)\\); embedding lookup: \\(O(1)\\)</div>
</div>`,
der:[
["Drop constants","If \\(f(n)=5n\\) and \\(g(n)=n\\): \\(f(n)\\le5g(n)\\) for all \\(n\\), so \\(C=5\\) works. \\(5n=O(n)\\)."],
["Drop lower-order terms","\\(n^2+n\\le2n^2\\) for \\(n\\ge1\\) (since \\(n\\le n^2\\)). So \\(n^2+n=O(n^2)\\) with \\(C=2\\)."],
["Self-attention complexity","For each of \\(n\\) queries: compute dot products with all \\(n\\) keys of dimension \\(d\\). That's \\(n\\cdot nd=n^2d\\) multiplications. Doubling sequence length quadruples cost."],
["Matrix multiply","\\((m\\times k)\\cdot(k\\times p)\\): output has \\(mp\\) entries; each requires \\(k\\) multiply-adds. Total: \\(O(mkp)\\). Square matrices: \\(O(n^3)\\)."],
["Why constants matter in practice","Flash Attention has the same \\(O(n^2d)\\) complexity as standard attention but is 3–10× faster due to memory-hierarchy tiling. Big-O predicts scaling; profiling reveals constants."]
],
code:`import numpy as np, time

# O(n^3) matmul scaling
for n in [128, 256, 512]:
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    t = time.perf_counter()
    for _ in range(10): A @ B
    t = (time.perf_counter()-t)/10
    print(f"n={n}: {t*1000:.2f}ms  ratio≈{t/(n**3)*1e9:.2f}ns/op")

# Attention: O(n^2 * d)
def attn(Q, K):  # (n,d)
    return Q @ K.T / Q.shape[1]**0.5   # (n,n)

# Same complexity as naive double loop, but 100× faster constant
n, d = 64, 128
Q = np.random.randn(n, d)
K = np.random.randn(n, d)
print("Attention output shape:", attn(Q,K).shape)  # (64, 64)`,
diag:`<svg viewBox="0 0 420 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;border-radius:7px"><rect width="420" height="170" fill="#0d1117"/><text x="12" y="16" fill="#8b949e" font-size="10">Growth rates: n doubles → runtime multiplies by…</text><line x1="38" y1="148" x2="390" y2="148" stroke="#30363d"/><line x1="38" y1="18" x2="38" y2="148" stroke="#30363d"/><text x="33" y="153" fill="#8b949e" font-size="8">n</text><text x="133" y="153" fill="#8b949e" font-size="8">2n</text><text x="213" y="153" fill="#8b949e" font-size="8">4n</text><text x="293" y="153" fill="#8b949e" font-size="8">8n</text><polyline fill="none" stroke="#3fb950" stroke-width="2" points="38,133 138,127 218,121 298,115 378,109"/><text x="381" y="108" fill="#3fb950" font-size="9">O(log n)</text><polyline fill="none" stroke="#58a6ff" stroke-width="2" points="38,133 138,118 218,103 298,88 378,73"/><text x="381" y="72" fill="#58a6ff" font-size="9">O(n)</text><polyline fill="none" stroke="#f59e0b" stroke-width="2" points="38,133 138,93 218,53 298,18"/><text x="300" y="16" fill="#f59e0b" font-size="9">O(n²)</text><text x="210" y="163" fill="#8b949e" font-size="9" text-anchor="middle">Attention is O(n²d): 2× seq len = 4× cost</text></svg>`,
kp:["Big-O ignores constants and lower-order terms; it captures the scaling shape.",
"Self-attention is \\(O(n^2d)\\): doubling context length quadruples computation.",
"Matrix multiply \\((m\\times k)(k\\times p)\\) is \\(O(mkp)\\).",
"Same Big-O but very different constants: FlashAttention vs standard attention.",
"\\(O(2n)=O(n)\\): adding more copies of \\(n\\) doesn't change the class."],
err:[["\\(O(2n)\\neq O(n)\\)","Wrong—\\(O(2n)=O(n)\\). Constants are absorbed. Big-O only cares about the shape."],
["Lower complexity always wins","A GPU-optimised \\(O(n^2)\\) can beat a serial \\(O(n\\log n)\\) for practical \\(n\\) if the constant is small enough. Profile before optimising."],
["Big-O describes exact runtime","It describes worst-case <em>growth</em>. Actual runtime depends on constants, memory, hardware, and implementation."]],
quiz:[
{q:"Attention on seq 4096 vs 1024 takes approximately…",o:["16× longer","4× longer","2× longer","same"],a:0,e:"\\(O(n^2)\\): \\((4096/1024)^2=16\\)."},
{q:"\\(O(n^3+n^2)=?\\)",o:["\\(O(n^3)\\)","\\(O(n^5)\\)","\\(O(n^2)\\)","\\(O(2n^3)\\)"],a:0,e:"Drop the lower-order term \\(n^2\\)."},
{q:"Dense layer: in=768, out=3072, batch=32. Complexity?",o:["\\(O(B\\cdot d_{in}\\cdot d_{out})\\)","\\(O(d_{in}+d_{out})\\)","\\(O(B^2)\\)","\\(O(d_{out})\\)"],a:0,e:"Matrix multiply: \\(B\\times d_{in}\\times d_{out}\\)."},
{q:"Show \\(n^2+100n+500=O(n^2)\\). What are \\(C,n_0\\)?",o:["\\(C=3,n_0=100\\)","\\(C=1,n_0=1\\)","\\(C=100,n_0=500\\)","\\(C=3,n_0=500\\)"],a:0,e:"For \\(n\\ge100\\): \\(100n\\le n^2\\) and \\(500\\le n^2\\), so \\(n^2+100n+500\\le3n^2\\). Take \\(C=3,n_0=100\\)."},
{q:"Why is FlashAttention faster despite same \\(O(n^2d)\\)?",o:["Tiles computation to fit in fast SRAM, reducing memory bandwidth","Fewer attention heads","Uses \\(O(n\\log n)\\) algorithm","Smaller model dimension"],a:0,e:"Same FLOPs, but IO-bound operations are dramatically reduced by keeping data in L1/L2 cache."}],
prac:[
{p:"Count multiply-adds for \\((3\\times4)\\times(4\\times2)\\) and express in Big-O.",s:"Output: \\(3\\times2=6\\) entries. Each needs 4 multiply-adds. Total: 24. Big-O: \\(O(m\\cdot k\\cdot p)=O(3\\cdot4\\cdot2)\\).",d:"easy"},
{p:"An \\(O(n^2)\\) algorithm takes 4s for \\(n=100\\). Estimate for \\(n=1000\\).",s:"Ratio: \\((1000/100)^2=100\\). Estimate: \\(100\\times4=400\\)s.",d:"easy"},
{p:"Show \\(n^2+100n+500=O(n^2)\\) by exhibiting \\(C\\) and \\(n_0\\).",s:"For \\(n\\ge100\\): \\(100n\\le n^2\\) and \\(500\\le n^2\\). So \\(n^2+100n+500\\le3n^2\\). \\(C=3, n_0=100\\). ✓",d:"medium"},
{p:"Time matmul for \\(n\\in\\{128,256,512\\}\\) and check that the ratio between consecutive times is \\(\\approx8\\) (matching \\(O(n^3)\\)).",s:"<code>for n in [128,256,512]:\n  A=np.random.randn(n,n);B=np.random.randn(n,n)\n  t0=time.time();A@B;print(n,time.time()-t0)\n# ratios should be ~8x each doubling</code>",d:"ai"}],
dd:`<div class="ddw"><h3>FLOP Counting in LLMs</h3><p>For a GPT-style model: one transformer layer with hidden dim \\(d\\) and sequence \\(n\\) costs roughly \\(24nd^2\\) FLOPs (attention projections + FFN) plus \\(4n^2d\\) for attention scores. With 96 layers and \\(d=12288\\) (GPT-3 scale): \\(96\\times(24\\times2048\\times12288^2)\\approx350\\) trillion FLOPs per forward pass. Chinchilla scaling (Hoffmann et al. 2022) finds that the optimal number of training tokens \\(\\propto\\sqrt{C}\\) where \\(C\\) is the compute budget — a direct application of Big-O reasoning to training economics.</p></div>`
},
"""

with open('/sessions/peaceful-serene-bardeen/mnt/Mathematics/ai-math/index.html', 'a') as f:
    f.write(lessons_t1)
print("Track 1 appended:", len(lessons_t1), "bytes")
