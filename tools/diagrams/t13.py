import numpy as np, math as m
@reg(13)
def d_13_1():
    cv=Canvas(560,236)
    cv.title(280,22,"Floating point: values crowd near zero, thin out far away")
    x0,x1=60,512; y=138
    F=Frame(cv,x0,0,x1-x0,1,0,16,0,1)
    cv.line(x0,y,x1,y,stroke=PAL["ink2"],w=1.6)
    for e in range(-1,4):
        base=2**e
        for k in range(4):
            v=base*(1+k/4); cv.line(F.X(v),y-7,F.X(v),y+7,stroke=PAL["blue"],w=1.3)
    cv.line(F.X(16),y-7,F.X(16),y+7,stroke=PAL["blue"],w=1.3)
    for v in [0.5,1,2,4,8,16]:
        cv.text(F.X(v),y+22,str(v),fill=PAL["ink3"],size=9.5)
    cv.text(F.X(0.75),y-16,"tiny gaps",fill=PAL["green"],size=9.5)
    cv.text(F.X(11),y-16,"big gaps",fill=PAL["red"],size=9.5)
    caption(cv,"Each power-of-two band holds the same count of values, so the spacing doubles every octave — precision fades with size.")
    return "13.1",cv.svg()
@reg(13)
def d_13_2():
    cv=Canvas(560,290)
    cv.title(280,22,"Log-sum-exp: subtract the max so exp never overflows")
    def chip(x,y,txt,fill,stroke,w=74):
        cv.rect(x,y,w,26,fill=fill,stroke=stroke,sw=1.2,rx=5); cv.text(x+w/2,y+17,txt,fill=stroke,size=10.5,weight="600")
    cv.text(66,74,"raw logits",fill=PAL["ink2"],size=10.5,weight="600",anchor="start")
    for i,v in enumerate([88,89,90]): chip(66+i*70,86,str(v),PAL["blueF"],PAL["blue"],w=60)
    cv.text(292,105,"exp →",fill=PAL["ink3"],size=11,anchor="start")
    chip(352,86,"1.6e38",PAL["redF"],PAL["red"]); chip(440,86,"∞ overflow ✗",PAL["redF"],PAL["red"],w=94)
    cv.text(66,152,"minus max (90)",fill=PAL["ink2"],size=10.5,weight="600",anchor="start")
    for i,v in enumerate(["−2","−1","0"]): chip(66+i*70,164,v,PAL["greenF"],PAL["green"],w=60)
    cv.text(292,183,"exp →",fill=PAL["ink3"],size=11,anchor="start")
    chip(352,164,"0.14",PAL["greenF"],PAL["green"]); chip(440,164,"1.0  ✓ safe",PAL["greenF"],PAL["green"],w=94)
    caption(cv,"Softmax is unchanged if you shift every logit by a constant — so subtract the max and each exp lands in (0, 1].")
    return "13.2",cv.svg()
@reg(13)
def d_13_3():
    cv=Canvas(560,262)
    cv.title(280,22,"Factorizations rewrite A as a product of simpler triangles")
    def mat(x0,label,kind):
        s=92; y0=64
        cv.rect(x0,y0,s,s,fill="#fff",stroke=PAL["ink3"],sw=1.3)
        if kind=="full": cv.rect(x0+2,y0+2,s-4,s-4,fill=PAL["indigoF"])
        if kind=="lower": cv.polygon([(x0,y0),(x0,y0+s),(x0+s,y0+s)],fill=PAL["blueF"])
        if kind=="upper": cv.polygon([(x0,y0),(x0+s,y0),(x0+s,y0+s)],fill=PAL["greenF"])
        cv.rect(x0,y0,s,s,fill="none",stroke=PAL["ink3"],sw=1.3)
        cv.text(x0+s/2,y0+s+22,label,fill=PAL["ink2"],size=10.5,weight="600")
    mat(78,"A (any square)","full")
    cv.text(190,112,"=",fill=PAL["ink"],size=22,weight="700")
    mat(220,"L (lower △)","lower")
    cv.text(332,112,"·",fill=PAL["ink"],size=22,weight="700")
    mat(360,"U (upper △)","upper")
    caption(cv,"LU = lower × upper triangular, so Ax=b solves by easy substitution. For symmetric-PD A, Cholesky A=LLᵀ halves the work.")
    return "13.3",cv.svg()
@reg(13)
def d_13_4():
    cv=Canvas(600,264)
    cv.title(300,22,"Reverse-mode autodiff: values go forward, gradients come back")
    y=118
    def node(cx,lab,fill,stroke):
        cv.raw(f'<circle cx="{cx}" cy="{y}" r="23" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
        cv.text(cx,y+5,lab,fill=stroke,size=12,weight="700")
    xs=[72,192,312,432,540]
    specs=[("x",PAL["neutralF"],PAL["ink2"]),("× w",PAL["blueF"],PAL["blue"]),("+ b",PAL["blueF"],PAL["blue"]),("σ",PAL["greenF"],PAL["green"]),("L",PAL["indigoF"],PAL["indigo"])]
    for cx,(lab,f,s) in zip(xs,specs): node(cx,lab,f,s)
    for i in range(4): cv.arrow(xs[i]+23,y-7,xs[i+1]-23,y-7,stroke=PAL["ink3"],w=1.8,head=6)
    for i in range(4): cv.arrow(xs[i+1]-23,y+11,xs[i]+23,y+11,stroke=PAL["red"],w=1.6,head=6,dash="4 3")
    cv.text(300,y-30,"forward: compute the output",fill=PAL["ink3"],size=10)
    cv.text(300,y+40,"backward: ∂L/∂· by the chain rule",fill=PAL["red"],size=10)
    caption(cv,"One forward sweep gives the loss; one backward sweep multiplies local derivatives into every gradient — that's backprop.")
    return "13.4",cv.svg()
@reg(13)
def d_13_5():
    cv=Canvas(560,300)
    cv.title(280,22,"Gradient check: a central difference should match your formula")
    F=Frame(cv,58,46,440,220,-0.3,4.2,-0.3,3.6)
    F.grid([0,1,2,3,4],[0,1,2,3]); F.axes()
    f=lambda x:0.22*x*x+0.3
    cv.poly(F.curve(f,-0.3,3.82,clipy=(-0.3,3.6)),stroke=PAL["indigo"],w=2.4)
    x0=2.2; h=1.1; sl=0.44*x0
    cv.poly([F.P(x0-1.4,f(x0)-sl*1.4),F.P(x0+1.4,f(x0)+sl*1.4)],stroke=PAL["red"],w=2.0)
    cv.line(*F.P(x0-h,f(x0-h)),*F.P(x0+h,f(x0+h)),stroke=PAL["green"],w=1.8,dash="5 3")
    for xx in [x0-h,x0+h]: cv.dot(*F.P(xx,f(xx)),r=3,fill=PAL["green"])
    cv.dot(*F.P(x0,f(x0)),fill=PAL["red"])
    cv.text(F.X(x0-h),F.Y(f(x0-h))-10,"x−h",fill=PAL["green"],size=9.5)
    cv.text(F.X(x0+h)+4,F.Y(f(x0+h))-8,"x+h",fill=PAL["green"],size=9.5,anchor="start")
    cv.text(F.X(0.1),F.Y(3.35),"— analytic slope  f′(x)",fill=PAL["red"],size=10,anchor="start",weight="600")
    cv.text(F.X(0.1),F.Y(3.0),"— central difference",fill=PAL["green"],size=10,anchor="start",weight="600")
    caption(cv,"The symmetric slope (f(x+h)−f(x−h))/2h matches f′(x) to O(h²) — three evaluations catch almost any gradient bug.")
    return "13.5",cv.svg()
