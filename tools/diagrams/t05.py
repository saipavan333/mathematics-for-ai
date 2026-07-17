import numpy as np, math as m
@reg(5)
def d_5_1():
    cv=Canvas(560,300)
    cv.title(280,22,"An eigenvector keeps its direction — the matrix only scales it")
    F=Frame(cv,58,46,320,224,-0.4,3.7,-0.4,3.7)
    F.grid(range(0,4),range(0,4)); F.axes(); O=F.P(0,0)
    cv.arrow(*O,*F.P(3,3),stroke=PAL["blue"],w=2.0)
    cv.arrow(*O,*F.P(1,1),stroke=PAL["indigo"],w=3.2)
    cv.text(F.X(1)-8,F.Y(1)+14,"v",fill=PAL["indigo"],size=13,weight="700",anchor="end")
    cv.text(F.X(3)-4,F.Y(3)+2,"Av = 3v",fill=PAL["blue"],size=11,weight="700",anchor="end")
    cv.arrow(*O,*F.P(1.4,0),stroke=PAL["green"],w=2.6)
    cv.arrow(*O,*F.P(2.4,0.5) if False else O,stroke=PAL["green"],w=1)
    A=np.array([[2.0,1.0],[1.0,2.0]]); w=A@np.array([1.4,0.0])
    cv.arrow(*O,*F.P(w[0],w[1]),stroke=PAL["amber"],w=2.6)
    cv.text(F.X(1.4),F.Y(0)-8,"w",fill=PAL["green"],size=12,weight="700")
    cv.text(F.X(w[0])+6,F.Y(w[1]),"Aw (turned!)",fill=PAL["amber"],size=10.5,weight="600",anchor="start")
    tx=430
    cv.text(tx,140,"A = [2 1]",anchor="start",fill=PAL["ink"],size=10.5,weight="600")
    cv.text(tx,155,"       [1 2]",anchor="start",fill=PAL["ink"],size=10.5,weight="600")
    cv.text(tx,180,"v stays on its ray",anchor="start",fill=PAL["indigo"],size=10)
    cv.text(tx,194,"→ eigenvector, λ=3",anchor="start",fill=PAL["indigo"],size=10)
    cv.text(tx,214,"w rotates → not one",anchor="start",fill=PAL["amber"],size=10)
    return "5.1",cv.svg()
@reg(5)
def d_5_2():
    cv=Canvas(560,300)
    cv.title(280,22,"A symmetric matrix has perpendicular eigen-axes")
    F=Frame(cv,60,44,240,240,-3.4,3.4,-3.4,3.4)
    F.grid(range(-3,4),range(-3,4)); F.axes()
    A=np.array([[2.0,1.0],[1.0,2.0]])
    th=np.linspace(0,2*np.pi,160); circ=np.array([np.cos(th),np.sin(th)]); ell=A@circ
    cv.poly([F.P(x,y) for x,y in circ.T],stroke=PAL["ink3"],w=1.5,dash="4 3")
    cv.poly([F.P(x,y) for x,y in ell.T],stroke=PAL["indigo"],w=2.4)
    w,V=np.linalg.eigh(A); O=F.P(0,0)
    labs={1:("λ=3",PAL["blue"]),0:("λ=1",PAL["green"])}
    dirs=[]
    for i in [1,0]:
        vec=V[:,i]*w[i]
        if vec[0]<0 or (abs(vec[0])<1e-9 and vec[1]<0): vec=-vec
        dirs.append(vec/np.linalg.norm(vec))
        cv.arrow(*O,*F.P(vec[0],vec[1]),stroke=labs[i][1],w=2.6)
        cv.text(F.X(vec[0]),F.Y(vec[1])+(-8 if vec[1]>0 else 16),labs[i][0],fill=labs[i][1],size=11,weight="700")
    cv.right_angle(F.X(0),F.Y(0),dirs[0][0],-dirs[0][1],dirs[1][0],-dirs[1][1],s=12,stroke=PAL["axis"])
    caption(cv,"Circle → ellipse whose axes lie exactly along the eigenvectors; symmetric ⇒ they're orthogonal.")
    return "5.2",cv.svg()
@reg(5)
def d_5_3():
    cv=Canvas(560,300)
    cv.title(280,22,"Positive definite = a bowl: xᵀAx > 0 in every direction")
    F=Frame(cv,60,44,240,240,-2.7,2.7,-2.7,2.7)
    F.grid(range(-2,3),range(-2,3)); F.axes()
    A=np.array([[2.0,0.6],[0.6,1.2]]); w,V=np.linalg.eigh(A)
    th=np.linspace(0,2*np.pi,120)
    for c,col in [(0.7,PAL["blueF"]),(1.8,PAL["blue"]),(3.4,PAL["indigo"])]:
        pts=[]
        for t in th:
            u=np.array([np.cos(t)/np.sqrt(w[0]),np.sin(t)/np.sqrt(w[1])])*np.sqrt(c)
            x=V@u; pts.append(F.P(x[0],x[1]))
        cv.poly(pts,stroke=col,w=2.0)
    cv.dot(*F.P(0,0),fill=PAL["red"]); cv.text(F.X(0)+8,F.Y(0)+14,"min",fill=PAL["red"],size=10,anchor="start",weight="600")
    caption(cv,"Every slice through the bottom curves upward, so the contours are closed ellipses around one minimum.")
    return "5.3",cv.svg()
@reg(5)
def d_5_4():
    cv=Canvas(620,300)
    cv.title(310,22,"SVD = rotate, stretch, rotate: a circle becomes an ellipse")
    A=np.array([[1.4,0.7],[0.2,1.1]]); U,S,Vt=np.linalg.svd(A)
    th=np.linspace(0,2*np.pi,160)
    def panel(px,pts,axes,labels,title_txt):
        F=Frame(cv,px,54,232,200,-2.2,2.2,-2.2,2.2)
        F.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); F.axes(); O=F.P(0,0)
        cv.poly([F.P(x,y) for x,y in pts],stroke=PAL["indigo"],w=2.2)
        for vec,col,lab in zip(axes,[PAL["red"],PAL["green"]],labels):
            v=vec.copy()
            if v[0]<0: v=-v
            cv.arrow(*O,*F.P(v[0],v[1]),stroke=col,w=2.2)
            cv.text(F.X(v[0]),F.Y(v[1])+(-7 if v[1]>=0 else 15),lab,fill=col,size=10.5,weight="700")
        cv.text(px+116,46,title_txt,fill=PAL["ink2"],size=10.5,weight="600")
    circ=[(np.cos(t),np.sin(t)) for t in th]
    ell=[tuple(A@np.array([np.cos(t),np.sin(t)])) for t in th]
    panel(36,circ,[Vt[0],Vt[1]],["v₁","v₂"],"input: unit circle")
    panel(352,ell,[S[0]*U[:,0],S[1]*U[:,1]],["σ₁u₁","σ₂u₂"],"output: stretched ellipse")
    caption(cv,"Vᵀ lines the axes up, Σ stretches them by σ₁ and σ₂, then U rotates the ellipse into place.")
    return "5.4",cv.svg()
@reg(5)
def d_5_5():
    cv=Canvas(560,300)
    cv.title(280,22,"PCA finds the directions the data actually varies along")
    F=Frame(cv,60,44,330,240,-3.6,3.6,-3.6,3.6)
    F.grid(range(-3,4),range(-3,4)); F.axes()
    rng=np.random.default_rng(3); L=np.linalg.cholesky(np.array([[1.7,1.05],[1.05,0.85]]))
    X=(L@rng.standard_normal((2,170))).T
    for p in X:
        if abs(p[0])<3.5 and abs(p[1])<3.5: cv.circle(F.X(p[0]),F.Y(p[1]),2.0,fill=PAL["blue"])
    cov=np.cov((X-X.mean(0)).T); w,V=np.linalg.eigh(cov); O=F.P(0,0)
    for i,(col,lab) in zip([1,0],[(PAL["red"],"PC1"),(PAL["green"],"PC2")]):
        vec=V[:,i]*np.sqrt(w[i])*2.3
        if vec[0]<0: vec=-vec
        cv.arrow(*O,*F.P(vec[0],vec[1]),stroke=col,w=2.8)
        cv.text(F.X(vec[0]),F.Y(vec[1])+(-8 if vec[1]>=0 else 16),lab,fill=col,size=11,weight="700")
    caption(cv,"PC1 is the long axis of the cloud (most variance); PC2 ⊥ PC1 catches what's left.")
    return "5.5",cv.svg()
