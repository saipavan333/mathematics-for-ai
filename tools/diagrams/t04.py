import math as m
import numpy as np
@reg(4)
def d_4_1():
    cv=Canvas(620,300)
    cv.title(310,22,"Unit balls: each norm draws its own “circle of radius 1”")
    F=Frame(cv,194,44,232,232,-1.55,1.55,-1.55,1.55)
    F.grid([-1,0,1],[-1,0,1]); F.axes()
    cv.polygon([F.P(1,1),F.P(1,-1),F.P(-1,-1),F.P(-1,1)],fill="none",stroke=PAL["amber"],w=2.2)
    cx,cy=F.P(0,0); r=F.X(1)-F.X(0)
    cv.circle(cx,cy,r,fill="none",stroke=PAL["indigo"],sw=2.2)
    cv.polygon([F.P(1,0),F.P(0,1),F.P(-1,0),F.P(0,-1)],fill="none",stroke=PAL["green"],w=2.2)
    cv.text(F.X(1.04),F.Y(1.16),"L∞",fill=PAL["amber"],size=12,weight="700",anchor="start")
    cv.text(F.X(0.74),F.Y(0.74)+2,"L2",fill=PAL["indigo"],size=12,weight="700")
    cv.text(F.X(0.42),F.Y(0.42)+2,"L1",fill=PAL["green"],size=12,weight="700")
    caption(cv,"L1's corners sit on the axes — that's why an L1 penalty drives weights to exactly zero (sparsity).")
    return "4.1",cv.svg()
@reg(4)
def d_4_2():
    cv=Canvas(560,300)
    cv.title(280,22,"A matrix stretches the unit circle into an ellipse")
    F=Frame(cv,60,44,236,236,-2.4,2.4,-2.4,2.4)
    F.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); F.axes()
    A=np.array([[1.6,0.6],[0.3,1.2]])
    th=np.linspace(0,2*np.pi,160); circ=np.array([np.cos(th),np.sin(th)]); ell=A@circ
    cv.poly([F.P(x,y) for x,y in circ.T],stroke=PAL["ink3"],w=1.6,dash="4 3")
    cv.poly([F.P(x,y) for x,y in ell.T],stroke=PAL["indigo"],w=2.4)
    U,S,Vt=np.linalg.svd(A); O=F.P(0,0)
    a1=S[0]*U[:,0]; a2=S[1]*U[:,1]
    if a1[0]<0: a1=-a1
    if a2[1]<0: a2=-a2
    cv.arrow(*O,*F.P(a1[0],a1[1]),stroke=PAL["red"],w=2.4)
    cv.arrow(*O,*F.P(a2[0],a2[1]),stroke=PAL["green"],w=2.2)
    cv.text(F.X(a1[0])+4,F.Y(a1[1])-6,"σ₁ = %.2f"%S[0],fill=PAL["red"],size=11,weight="700",anchor="start")
    cv.text(F.X(a2[0])-4,F.Y(a2[1])-6,"σ₂ = %.2f"%S[1],fill=PAL["green"],size=10.5,weight="600",anchor="end")
    tx=410
    cv.text(tx,130,"circle (radius 1)",anchor="start",fill=PAL["ink3"],size=10.5)
    cv.text(tx,150,"→ ellipse",anchor="start",fill=PAL["indigo"],size=11,weight="600")
    cv.text(tx,178,"spectral norm ‖A‖₂",anchor="start",fill=PAL["ink2"],size=10.5)
    cv.text(tx,194,"= σ₁ = biggest stretch",anchor="start",fill=PAL["red"],size=10.5,weight="600")
    return "4.2",cv.svg()
@reg(4)
def d_4_3():
    cv=Canvas(620,292)
    cv.title(310,22,"Invertible reshapes; singular collapses (and can't be undone)")
    def panel(px,A,ok,label):
        F=Frame(cv,px,54,232,196,-0.5,3.2,-0.6,2.8)
        F.grid([0,1,2,3],[0,1,2]); F.axes()
        sq=np.array([[0,1,1,0,0],[0,0,1,1,0]]); img=A@sq
        cv.poly([F.P(0,0),F.P(1,0),F.P(1,1),F.P(0,1),F.P(0,0)],stroke=PAL["ink3"],w=1.3)
        col=PAL["indigo"] if ok else PAL["red"]
        cv.poly([F.P(x,y) for x,y in img.T],stroke=col,w=2.4)
        cv.text(px+116,46,label,fill=col,size=10.5,weight="600")
    panel(36,np.array([[1.5,0.5],[0.4,1.3]]),True,"invertible: det = 1.75 ≠ 0")
    panel(352,np.array([[1.0,2.0],[0.5,1.0]]),False,"singular: det = 0 → flat line")
    caption(cv,"A singular matrix squashes the square onto a line — two inputs share one output, so nothing can undo it.")
    return "4.3",cv.svg()
@reg(4)
def d_4_4():
    cv=Canvas(620,300)
    cv.title(310,22,"Conditioning: near-parallel lines make the answer fragile")
    def xy(m_,b,F,x0,x1): return [F.P(x0,m_*x0+b),F.P(x1,m_*x1+b)]
    def inter(m1,b1,m2,b2):
        x=(b2-b1)/(m1-m2); return (x,m1*x+b1)
    def panel(px,L,shiftline,label,ok):
        F=Frame(cv,px,54,232,196,-0.5,4.5,-0.5,4.5)
        F.grid(range(0,5),range(0,5)); F.axes()
        (m1,b1),(m2,b2)=L
        cv.poly(xy(m1,b1,F,-0.5,4.5),stroke=PAL["blue"],w=2.0)
        cv.poly(xy(m2,b2,F,-0.5,4.5),stroke=PAL["green"],w=2.0)
        p0=inter(m1,b1,m2,b2)
        si,sb=shiftline  # which line index (0/1) and new b
        if si==0: p1=inter(m1,sb,m2,b2); cv.poly(xy(m1,sb,F,-0.5,4.5),stroke=PAL["blue"],w=1.4,dash="5 3")
        else: p1=inter(m1,b1,m2,sb); cv.poly(xy(m2,sb,F,-0.5,4.5),stroke=PAL["green"],w=1.4,dash="5 3")
        cv.dot(*F.P(*p0),fill=PAL["ink"])
        cv.dot(*F.P(*p1),fill=PAL["red"])
        if m.hypot(p1[0]-p0[0],p1[1]-p0[1])>0.25:
            cv.arrow(*F.P(*p0),*F.P(*p1),stroke=PAL["red"],w=1.8,head=6)
        cv.text(px+116,46,label,fill=PAL["indigo"] if ok else PAL["red"],size=10.5,weight="600")
    panel(36,[(1.0,0.0),(-1.0,3.0)],(1,3.5),"well-conditioned: solution barely moves",True)
    panel(352,[(0.8,0.3),(0.95,0.0)],(0,0.6),"ill-conditioned: solution jumps far",False)
    caption(cv,"Same size nudge to a line: perpendicular lines shrug it off; near-parallel lines swing the answer wildly.")
    return "4.4",cv.svg()
