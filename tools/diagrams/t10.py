import numpy as np, math as m
from math import gamma
@reg(10)
def d_10_1():
    cv=Canvas(620,282)
    cv.title(310,22,"A density (PDF) and its running total (CDF)")
    F1=Frame(cv,42,54,232,176,-0.3,3.3,-0.05,0.78); F1.axes(ylabel="density")
    cv.polygon([F1.P(0.5,0),F1.P(0.5,0.5),F1.P(2.5,0.5),F1.P(2.5,0)],fill=PAL["indigoF"],opacity=0.7)
    cv.poly([F1.P(-0.3,0),F1.P(0.5,0),F1.P(0.5,0.5),F1.P(2.5,0.5),F1.P(2.5,0),F1.P(3.3,0)],stroke=PAL["indigo"],w=2.3)
    cv.text(F1.X(1.5),F1.Y(0.58),"PDF — uniform",fill=PAL["indigo"],size=10.5,weight="600")
    cv.text(F1.X(1.5),F1.Y(0.24),"area = 1",fill=PAL["ink3"],size=10)
    F2=Frame(cv,354,54,232,176,-0.3,3.3,-0.05,1.18); F2.axes(ylabel="P(X ≤ x)")
    cv.poly([F2.P(-0.3,0),F2.P(0.5,0),F2.P(2.5,1.0),F2.P(3.3,1.0)],stroke=PAL["green"],w=2.3)
    cv.line(F2.X(-0.3),F2.Y(1.0),F2.X(3.3),F2.Y(1.0),stroke=PAL["ink3"],w=0.8,dash="3 3")
    cv.text(F2.X(0.55),F2.Y(0.92),"CDF — climbs 0→1",fill=PAL["green"],size=10.5,weight="600",anchor="start")
    caption(cv,"The CDF at x is the area under the PDF up to x; its slope is the density, and it always ends at 1.")
    return "10.1",cv.svg()
@reg(10)
def d_10_2():
    cv=Canvas(560,300)
    cv.title(280,22,"The Normal: 68 – 95 – 99.7 % of the mass within 1, 2, 3 σ")
    F=Frame(cv,58,52,440,200,-3.7,3.7,-0.02,0.45)
    F.axes()
    pdf=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi); xs=np.linspace(-3.6,3.6,180)
    for lo,hi,col in [(-3,3,PAL["blueF"]),(-2,2,"#bcd2f2"),(-1,1,PAL["blue"])]:
        seg=[x for x in xs if lo<=x<=hi]
        cv.polygon([F.P(seg[0],0)]+[F.P(x,pdf(x)) for x in seg]+[F.P(seg[-1],0)],fill=col,opacity=0.7)
    cv.poly([F.P(x,pdf(x)) for x in xs],stroke=PAL["indigo"],w=2.3)
    for s in [-3,-2,-1,0,1,2,3]:
        cv.text(F.X(s),F.Y(0)+15,("μ" if s==0 else f"{s:+d}σ"),fill=PAL["ink3"],size=9.5)
    cv.text(F.X(0),F.Y(0.30),"68%",fill="#fff",size=11,weight="700")
    cv.text(F.X(0),F.Y(0.075),"95%",fill=PAL["ink"],size=10.5,weight="700")
    cv.text(F.X(2.5),F.Y(0.055),"99.7%",fill=PAL["ink2"],size=9.5,weight="600")
    caption(cv,"Distance from the mean is measured in σ; almost everything (99.7%) lands within three of them.")
    return "10.2",cv.svg()
@reg(10)
def d_10_3():
    cv=Canvas(560,300)
    cv.title(280,22,"The Beta family bends to fit any belief about a probability")
    F=Frame(cv,58,50,440,204,-0.03,1.03,-0.05,2.9)
    F.axes(xlabel="p")
    def beta(a,b): return lambda x:(x**(a-1)*(1-x)**(b-1))*gamma(a+b)/(gamma(a)*gamma(b)) if 0<x<1 else 0
    xs=np.linspace(0.002,0.998,180)
    for (a,b,col,lab,lx) in [(2,5,PAL["blue"],"Beta(2,5)",0.2),(2,2,PAL["amber"],"Beta(2,2)",0.5),(5,2,PAL["green"],"Beta(5,2)",0.8)]:
        f=beta(a,b); cv.poly([F.P(x,min(f(x),2.85)) for x in xs],stroke=col,w=2.2)
        cv.text(F.X(lx),F.Y(min(f(lx),2.85))-8,lab,fill=col,size=10,weight="600")
    caption(cv,"Two 'pseudo-count' knobs (a,b) slide the peak left, center, or right — the go-to prior for a probability.")
    return "10.3",cv.svg()
@reg(10)
def d_10_4():
    cv=Canvas(560,300)
    cv.title(280,22,"A 2-D Gaussian: the covariance sets the tilt and spread")
    F=Frame(cv,60,44,300,240,-3.5,3.5,-3.5,3.5)
    F.grid(range(-3,4),range(-3,4)); F.axes()
    Sig=np.array([[1.7,0.95],[0.95,0.9]]); w,V=np.linalg.eigh(Sig)
    for k,col in [(1.0,PAL["blue"]),(2.0,PAL["indigo"]),(3.0,PAL["blueF"])]:
        th=np.linspace(0,2*np.pi,140); pts=[]
        for t in th:
            u=np.array([np.cos(t)*np.sqrt(w[0]),np.sin(t)*np.sqrt(w[1])])*k
            x=V@u; pts.append(F.P(x[0],x[1]))
        cv.poly(pts,stroke=col,w=1.8)
    O=F.P(0,0)
    for i,col in zip([1,0],[PAL["red"],PAL["green"]]):
        vec=V[:,i]*np.sqrt(w[i])*2.0
        if vec[0]<0: vec=-vec
        cv.arrow(*O,*F.P(vec[0],vec[1]),stroke=col,w=2.0)
    caption(cv,"Contours are ellipses; their axes are Σ's eigenvectors, lengths ∝ √eigenvalue (the spread each way).")
    return "10.4",cv.svg()
@reg(10)
def d_10_5():
    cv=Canvas(560,300)
    cv.title(280,22,"Reparameterization: z = μ + σ·ε reshapes fixed noise ε")
    F=Frame(cv,58,52,440,196,-4.5,5.6,-0.02,0.45)
    F.axes()
    n=lambda x,mu,s:np.exp(-((x-mu)/s)**2/2)/(s*np.sqrt(2*np.pi)); xs=np.linspace(-4.5,5.6,220)
    cv.poly([F.P(x,n(x,0,1)) for x in xs],stroke=PAL["ink3"],w=2.0,dash="5 3")
    cv.poly([F.P(x,n(x,2.2,1.5)) for x in xs],stroke=PAL["indigo"],w=2.4)
    cv.text(F.X(-1.5),F.Y(0.34),"ε ~ N(0,1)",fill=PAL["ink3"],size=10.5,weight="600",anchor="start")
    cv.text(F.X(2.2),F.Y(n(2.2,2.2,1.5))-8,"z = μ+σε ~ N(μ,σ²)",fill=PAL["indigo"],size=10.5,weight="600")
    cv.arrow(F.X(0.5),F.Y(0.36),F.X(1.7),F.Y(0.29),stroke=PAL["amber"],w=1.8,head=6)
    cv.text(F.X(1.1),F.Y(0.40),"shift μ, scale σ",fill=PAL["amber"],size=9.5)
    caption(cv,"Sample ε once, then compute z = μ+σε — gradients flow through μ and σ, making sampling trainable.")
    return "10.5",cv.svg()
@reg(10)
def d_10_6():
    cv=Canvas(560,300)
    cv.title(280,22,"Central Limit Theorem: averages turn Normal as n grows")
    F=Frame(cv,58,50,440,204,-0.05,1.05,-0.15,6.2)
    F.axes(xlabel="sample mean of n uniforms")
    cv.poly([F.P(0,0),F.P(0,1),F.P(1,1),F.P(1,0)],stroke=PAL["ink3"],w=1.8)
    cv.poly([F.P(0,0),F.P(0.5,2),F.P(1,0)],stroke=PAL["blue"],w=1.9)
    for n,col in [(5,PAL["amber"]),(30,PAL["red"])]:
        s=1/np.sqrt(12*n); xs=np.linspace(0.02,0.98,140)
        g=lambda x:np.exp(-((x-0.5)/s)**2/2)/(s*np.sqrt(2*np.pi))
        cv.poly([F.P(x,min(g(x),6.1)) for x in xs],stroke=col,w=1.9)
    cv.text(F.X(0.02),F.Y(1.15),"n=1 (flat)",fill=PAL["ink3"],size=9.5,anchor="start")
    cv.text(F.X(0.5),F.Y(2.15),"n=2",fill=PAL["blue"],size=9.5)
    cv.text(F.X(0.66),F.Y(3.0),"n=5",fill=PAL["amber"],size=9.5,anchor="start")
    cv.text(F.X(0.5),F.Y(6.0),"n=30",fill=PAL["red"],size=9.5)
    caption(cv,"Whatever the starting shape (here: flat), the average of n draws tightens into a bell around the true mean.")
    return "10.6",cv.svg()
