import numpy as np, math as m
def _seg(F,fn,x0,x1,ylo,yhi,n=140,xlo=None,xhi=None):
    pts=[]
    for i in range(n+1):
        x=x0+(x1-x0)*i/n
        try: y=fn(x)
        except Exception: continue
        if y<ylo or y>yhi: continue
        if xlo is not None and (x<xlo or x>xhi): continue
        pts.append(F.P(x,y))
    return pts
@reg(6)
def d_6_1():
    cv=Canvas(560,300)
    cv.title(280,22,"The derivative is the slope the secants zoom into")
    F=Frame(cv,58,46,340,220,-0.3,4.3,-0.3,3.7)
    F.grid(range(0,5),range(0,4)); F.axes()
    f=lambda x:0.25*x*x
    cv.poly(_seg(F,f,-0.3,3.75,-0.3,3.7),stroke=PAL["indigo"],w=2.4)
    x0=2; sl=1.0; tan=lambda x:f(x0)+sl*(x-x0)
    cv.poly(_seg(F,tan,0.9,3.2,-0.3,3.7),stroke=PAL["red"],w=2.2)
    for h,col in [(1.5,PAL["ink3"]),(1.0,PAL["amber"]),(0.5,PAL["green"])]:
        x1=x0+h; cv.line(*F.P(x0,f(x0)),*F.P(x1,f(x1)),stroke=col,w=1.4,dash="4 3"); cv.dot(*F.P(x1,f(x1)),r=2.6,fill=col)
    cv.dot(*F.P(x0,f(x0)),fill=PAL["red"])
    cv.text(F.X(x0)-8,F.Y(f(x0))+4,"P",fill=PAL["red"],size=12,weight="700",anchor="end")
    cv.text(F.X(3.5)+2,F.Y(f(3.5))-2,"secants",fill=PAL["amber"],size=10,anchor="start")
    cv.text(F.X(3.2)+3,F.Y(tan(3.2))+2,"tangent",fill=PAL["red"],size=10.5,anchor="start",weight="600")
    caption(cv,"As the far point slides toward P, the secant slope → the tangent slope f′(2) = 1.")
    return "6.1",cv.svg()
@reg(6)
def d_6_2():
    cv=Canvas(600,250)
    cv.title(300,22,"The chain rule multiplies the slopes along the path")
    y=115
    def box(cx,lab,fill,stroke):
        cv.rect(cx-42,y-28,84,56,fill=fill,stroke=stroke,sw=1.6,rx=10)
        cv.text(cx,y+5,lab,fill=stroke,size=12.5,weight="700")
    cv.text(66,y+4,"x",fill=PAL["ink"],size=15,weight="700")
    cv.arrow(80,y,150,y,stroke=PAL["axis"],w=2); cv.text(115,y-10,"g′(x)=2",fill=PAL["blue"],size=10)
    box(196,"u = g(x)",PAL["blueF"],PAL["blue"])
    cv.arrow(240,y,310,y,stroke=PAL["axis"],w=2); cv.text(275,y-10,"f′(u)=3",fill=PAL["green"],size=10)
    box(356,"y = f(u)",PAL["greenF"],PAL["green"])
    cv.arrow(400,y,470,y,stroke=PAL["axis"],w=2)
    cv.text(506,y+4,"y",fill=PAL["ink"],size=15,weight="700")
    cv.text(300,192,"dy/dx = f′(u) · g′(x) = 3 · 2 = 6",fill=PAL["indigo"],size=15,weight="700")
    caption(cv,"Backprop is exactly this: multiply the local slopes from the output back to the input.")
    return "6.2",cv.svg()
@reg(6)
def d_6_3():
    cv=Canvas(560,300)
    cv.title(280,22,"The gradient points straight uphill — perpendicular to the contours")
    F=Frame(cv,60,44,320,240,-2.6,2.6,-2.6,2.6)
    F.grid(range(-2,3),range(-2,3)); F.axes()
    for c,col in [(0.6,PAL["blueF"]),(1.6,PAL["blue"]),(3.2,PAL["indigo"])]:
        th=np.linspace(0,2*np.pi,140)
        cv.poly([F.P(np.sqrt(c)*np.cos(t),np.sqrt(c/3)*np.sin(t)) for t in th],stroke=col,w=1.9)
    px,py=1.0,np.sqrt((1.6-1.0)/3)
    g=np.array([2*px,6*py]); g=g/np.linalg.norm(g)*1.35
    cv.arrow(F.X(px),F.Y(py),F.X(px+g[0]),F.Y(py+g[1]),stroke=PAL["red"],w=2.6)
    cv.dot(F.X(px),F.Y(py),fill=PAL["red"])
    cv.text(F.X(px+g[0])+4,F.Y(py+g[1])-4,"∇f",fill=PAL["red"],size=13,weight="700",anchor="start")
    cv.text(F.X(-1.7),F.Y(2.2),"f = x² + 3y²",fill=PAL["ink3"],size=10.5,anchor="start")
    caption(cv,"∇f is perpendicular to the level curve and aims at the steepest rise; −∇f is the downhill step.")
    return "6.3",cv.svg()
@reg(6)
def d_6_4():
    cv=Canvas(620,300)
    cv.title(310,22,"The Hessian is curvature: a bowl (minimum) vs a saddle")
    def bowl(px):
        F=Frame(cv,px,54,232,204,-2.4,2.4,-2.4,2.4); F.grid(range(-2,3),range(-2,3)); F.axes()
        for r,col in [(0.8,PAL["blueF"]),(1.5,PAL["blue"]),(2.1,PAL["indigo"])]:
            th=np.linspace(0,2*np.pi,120); cv.poly([F.P(r*np.cos(t),r*np.sin(t)) for t in th],stroke=col,w=1.9)
        cv.dot(*F.P(0,0),fill=PAL["red"]); cv.text(px+116,46,"bowl: both ways curve up",fill=PAL["indigo"],size=10.5,weight="600")
    def saddle(px):
        F=Frame(cv,px,54,232,204,-2.4,2.4,-2.4,2.4); F.grid(range(-2,3),range(-2,3)); F.axes()
        def seg(lst): return [F.P(x,y) for (x,y) in lst if -2.35<=x<=2.35 and -2.35<=y<=2.35]
        ys=np.linspace(-2.4,2.4,90); xs=np.linspace(-2.4,2.4,90)
        for c in [1.0,2.4]:
            cv.poly(seg([(np.sqrt(c+y*y),y) for y in ys]),stroke=PAL["blue"],w=1.7)
            cv.poly(seg([(-np.sqrt(c+y*y),y) for y in ys]),stroke=PAL["blue"],w=1.7)
            cv.poly(seg([(x,np.sqrt(c+x*x)) for x in xs]),stroke=PAL["amber"],w=1.7)
            cv.poly(seg([(x,-np.sqrt(c+x*x)) for x in xs]),stroke=PAL["amber"],w=1.7)
        cv.poly([F.P(-2.35,-2.35),F.P(2.35,2.35)],stroke=PAL["ink3"],w=1,dash="3 3")
        cv.poly([F.P(-2.35,2.35),F.P(2.35,-2.35)],stroke=PAL["ink3"],w=1,dash="3 3")
        cv.dot(*F.P(0,0),fill=PAL["red"]); cv.text(px+116,46,"saddle: up one way, down the other",fill=PAL["red"],size=10.5,weight="600")
    bowl(36); saddle(352)
    caption(cv,"Positive-definite Hessian ⇒ bowl (a true minimum). Mixed signs ⇒ saddle — flat gradient, but no minimum.")
    return "6.4",cv.svg()
@reg(6)
def d_6_5():
    cv=Canvas(560,300)
    cv.title(280,22,"Taylor: match the value, then the slope, then the curvature")
    F=Frame(cv,58,48,440,214,-0.3,6.4,-1.7,1.7)
    F.grid(range(0,7),[-1,0,1]); F.axes()
    a=0.9
    cv.poly(_seg(F,np.cos,-0.3,6.4,-1.7,1.7),stroke=PAL["indigo"],w=2.4)
    t1=lambda x:np.cos(a)-np.sin(a)*(x-a)
    cv.poly(_seg(F,t1,-0.3,3.2,-1.7,1.7),stroke=PAL["green"],w=2.0)
    t2=lambda x:np.cos(a)-np.sin(a)*(x-a)-0.5*np.cos(a)*(x-a)**2
    cv.poly(_seg(F,t2,-0.9,3.2,-1.7,1.7),stroke=PAL["red"],w=2.0)
    cv.dot(*F.P(a,np.cos(a)),fill=PAL["ink"]); cv.text(F.X(a)-2,F.Y(np.cos(a))+16,"a",fill=PAL["ink"],size=11,weight="700",anchor="end")
    lx=F.X(3.35); ly=F.Y(1.5)
    cv.text(lx,ly,"— cos x",fill=PAL["indigo"],size=10.5,weight="600",anchor="start")
    cv.text(lx,ly+16,"— 1st order (tangent)",fill=PAL["green"],size=10.5,weight="600",anchor="start")
    cv.text(lx,ly+32,"— 2nd order (parabola)",fill=PAL["red"],size=10.5,weight="600",anchor="start")
    caption(cv,"Near a, the tangent matches the slope; the parabola matches curvature too — a tighter fit.")
    return "6.5",cv.svg()
