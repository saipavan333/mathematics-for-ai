import numpy as np, math as m
@reg(8)
def d_8_1():
    cv=Canvas(560,300)
    cv.title(280,22,"Critical points: where the slope is zero (min, max, saddle-like)")
    F=Frame(cv,58,46,440,214,-2.8,2.8,-2.0,1.7)
    F.grid([-2,-1,0,1,2],[-1,0,1]); F.axes()
    f=lambda x:0.2*x**4-1.1*x**2+0.1*x
    fpp=lambda x:2.4*x**2-2.2
    cv.poly(F.curve(f,-2.8,2.8,clipy=(-2.0,1.7)),stroke=PAL["indigo"],w=2.4)
    roots=[r.real for r in np.roots([0.8,0,-2.2,0.1]) if abs(r.imag)<1e-6]
    mins=[(r,f(r)) for r in roots if fpp(r)>0]
    gmin=min(mins,key=lambda t:t[1])[0]
    for r in sorted(roots):
        y=f(r); cv.dot(*F.P(r,y),fill=PAL["red"])
        cv.line(F.X(r-0.45),F.Y(y),F.X(r+0.45),F.Y(y),stroke=PAL["amber"],w=1.5)
        if fpp(r)<0: lab="local max"; dy=-12
        elif abs(r-gmin)<1e-6: lab="global min"; dy=20
        else: lab="local min"; dy=20
        cv.text(F.X(r),F.Y(y)+dy,lab,fill=PAL["ink2"],size=10,weight="600")
    caption(cv,"Setting f′(x)=0 finds every flat spot; the second derivative says which are valleys and which are peaks.")
    return "8.1",cv.svg()
@reg(8)
def d_8_2():
    cv=Canvas(560,300)
    cv.title(280,22,"Convex = a bowl: every chord lies on or above the curve")
    F=Frame(cv,58,46,440,220,-2.6,2.6,-0.3,4.3)
    F.grid([-2,-1,0,1,2],[0,1,2,3,4]); F.axes()
    f=lambda x:0.55*x*x+0.2
    cv.poly(F.curve(f,-2.6,2.6,clipy=(-0.3,4.3)),stroke=PAL["indigo"],w=2.4)
    A=(-1.8,f(-1.8)); B=(2.2,f(2.2)); chord=lambda x:A[1]+(B[1]-A[1])*(x-A[0])/(B[0]-A[0])
    xs=np.linspace(A[0],B[0],40)
    cv.polygon([F.P(x,chord(x)) for x in xs]+[F.P(x,f(x)) for x in reversed(xs)],fill=PAL["amberF"],opacity=0.65)
    cv.line(*F.P(*A),*F.P(*B),stroke=PAL["amber"],w=2.0)
    cv.dot(*F.P(*A),fill=PAL["ink"]); cv.dot(*F.P(*B),fill=PAL["ink"])
    cv.text(F.X(A[0])-6,F.Y(A[1]),"A",fill=PAL["ink"],size=11,weight="700",anchor="end")
    cv.text(F.X(B[0])+6,F.Y(B[1]),"B",fill=PAL["ink"],size=11,weight="700",anchor="start")
    cv.text(F.X(0.35),F.Y(chord(0.35))-8,"chord stays above",fill=PAL["amber"],size=10.5,weight="600")
    caption(cv,"Convex = one bowl, one global minimum — so logistic and linear losses train without local traps.")
    return "8.2",cv.svg()
@reg(8)
def d_8_3():
    cv=Canvas(560,300)
    cv.title(280,22,"Gradient descent: step downhill by −η · f′(x)")
    F=Frame(cv,58,46,440,220,-2.7,2.7,-0.3,4.2)
    F.grid([-2,-1,0,1,2],[0,1,2,3,4]); F.axes()
    f=lambda x:0.6*x*x
    cv.poly(F.curve(f,-2.65,2.65,clipy=(-0.3,4.2)),stroke=PAL["indigo"],w=2.4)
    x=2.3; eta=0.35; pts=[x]
    for _ in range(6): x=x-eta*(1.2*x); pts.append(x)
    for i in range(len(pts)-1):
        x0,x1=pts[i],pts[i+1]
        cv.arrow(F.X(x0),F.Y(f(x0)),F.X(x1),F.Y(f(x1)),stroke=PAL["amber"],w=1.6,head=5)
        cv.dot(*F.P(x0,f(x0)),r=3,fill=PAL["red"])
    cv.dot(*F.P(pts[-1],f(pts[-1])),r=3.4,fill=PAL["green"])
    cv.text(F.X(2.3),F.Y(f(2.3))+2,"start",fill=PAL["red"],size=10,anchor="start",weight="600")
    cv.text(F.X(0),F.Y(0)+16,"min",fill=PAL["green"],size=10,weight="600")
    caption(cv,"Each step moves opposite the slope; as the slope flattens the steps shrink and settle at the minimum.")
    return "8.3",cv.svg()
@reg(8)
def d_8_4():
    cv=Canvas(560,300)
    cv.title(280,22,"Momentum rolls along the valley; plain GD zig-zags across it")
    F=Frame(cv,58,46,440,224,-5.0,5.0,-2.6,2.6)
    F.grid(range(-4,5),range(-2,3)); F.axes()
    a,b=1.0,9.0
    for c,col in [(1.2,PAL["indigo"]),(4.5,PAL["blue"]),(10.0,PAL["blueF"])]:
        th=np.linspace(0,2*np.pi,140)
        cv.poly([F.P(np.sqrt(2*c/a)*np.cos(t),np.sqrt(2*c/b)*np.sin(t)) for t in th],stroke=col,w=1.6)
    def run(mom,eta,beta,steps):
        x=np.array([4.6,1.35]); v=np.zeros(2); path=[x.copy()]
        for _ in range(steps):
            g=np.array([a*x[0],b*x[1]])
            if mom: v=beta*v-eta*g; x=x+v
            else: x=x-eta*g
            path.append(x.copy())
        return np.array(path)
    def draw(path,col):
        cv.poly([F.P(p[0],p[1]) for p in path if abs(p[0])<5 and abs(p[1])<2.6],stroke=col,w=1.8)
        for p in path:
            if abs(p[0])<5 and abs(p[1])<2.6: cv.dot(F.X(p[0]),F.Y(p[1]),r=2.0,fill=col)
    draw(run(False,0.19,0,16),PAL["red"]); draw(run(True,0.05,0.86,24),PAL["green"])
    cv.text(F.X(-4.8),F.Y(2.32),"— plain GD (zig-zag)",fill=PAL["red"],size=10,anchor="start",weight="600")
    cv.text(F.X(-4.8),F.Y(1.98),"— momentum (smooth)",fill=PAL["green"],size=10,anchor="start",weight="600")
    caption(cv,"In a stretched valley GD bounces off the steep walls; momentum builds speed along the shallow floor.")
    return "8.4",cv.svg()
@reg(8)
def d_8_5():
    cv=Canvas(560,300)
    cv.title(280,22,"Lagrange: the optimum is where a contour just kisses the constraint")
    F=Frame(cv,60,44,340,224,-1.7,1.7,-1.7,1.7)
    F.grid([-1,0,1],[-1,0,1]); F.axes()
    th=np.linspace(0,2*np.pi,160)
    cv.poly([F.P(np.cos(t),np.sin(t)) for t in th],stroke=PAL["blue"],w=2.3)
    cv.text(F.X(np.cos(2.5)),F.Y(np.sin(2.5)),"g = 0",fill=PAL["blue"],size=10.5,weight="600",anchor="end")
    for c in [-1.0,0.0,1.0]:
        cv.poly([F.P(-1.7,c+1.7),F.P(1.7,c-1.7)],stroke=PAL["ink3"],w=1.3,dash="4 3")
    cv.poly([F.P(0.0,1.414),F.P(1.414,0.0)],stroke=PAL["indigo"],w=1.9)
    pt=(0.7071,0.7071); d=np.array([1.0,1.0])/np.sqrt(2)
    cv.arrow(F.X(pt[0]),F.Y(pt[1]),F.X(pt[0]+d[0]*0.92),F.Y(pt[1]+d[1]*0.92),stroke=PAL["green"],w=2.0)
    cv.arrow(F.X(pt[0]),F.Y(pt[1]),F.X(pt[0]+d[0]*0.52),F.Y(pt[1]+d[1]*0.52),stroke=PAL["red"],w=2.8)
    cv.dot(*F.P(*pt),fill=PAL["red"])
    cv.text(F.X(pt[0]+d[0]*0.92)+5,F.Y(pt[1]+d[1]*0.92),"∇g",fill=PAL["green"],size=11,weight="700",anchor="start")
    cv.text(F.X(pt[0]+d[0]*0.30)+9,F.Y(pt[1]+d[1]*0.30)+3,"∇f",fill=PAL["red"],size=11,weight="700",anchor="start")
    cv.text(F.X(-1.05),F.Y(-1.4),"dashed = contours of f = x + y",fill=PAL["ink3"],size=9.5,anchor="start")
    caption(cv,"At the best feasible point ∇f ∥ ∇g — you can't increase f without stepping off the constraint.")
    return "8.5",cv.svg()
