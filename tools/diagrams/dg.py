# -*- coding: utf-8 -*-
import math
PAL = dict(
    ink="#1f2a44", ink2="#4a5878", ink3="#6b7891",
    axis="#9aa6bd", grid="#e7ecf4",
    blue="#2a6f97", blueF="#dbe8fb",
    green="#3a7d44", greenF="#e3efe5",
    indigo="#4f46e5", indigoF="#e5e7ff",
    amber="#b9722a", amberF="#f7e7d2",
    red="#c0392b", redF="#f7e0dc",
    violet="#7c4dff", violetF="#ece6ff",
    neutralF="#eef2f7", plate="#f8fafc",
)
def esc(s):
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
class Canvas:
    def __init__(self, w, h, fs=11.5):
        self.w=w; self.h=h; self.fs=fs; self.el=[]
    def raw(self,s): self.el.append(s)
    def text(self,x,y,s,anchor="middle",fill=None,size=None,weight=None,style=None):
        fill=fill or PAL["ink"]; s=esc(s)
        a=f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}"'
        if size: a+=f' font-size="{size}"'
        if weight: a+=f' font-weight="{weight}"'
        if style: a+=f' font-style="{style}"'
        self.el.append(f'<text {a}>{s}</text>')
    def title(self,x,y,s): self.text(x,y,s,weight="700",fill=PAL["ink"],size=self.fs+1.5)
    def line(self,x1,y1,x2,y2,stroke=None,w=1.0,dash=None,cap="round"):
        stroke=stroke or PAL["axis"]; d=f' stroke-dasharray="{dash}"' if dash else ''
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d} stroke-linecap="{cap}"/>')
    def poly(self,pts,stroke=None,w=2.0,fill="none",dash=None):
        stroke=stroke or PAL["ink"]; d=f' stroke-dasharray="{dash}"' if dash else ''
        p=" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
        self.el.append(f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"{d} stroke-linejoin="round" stroke-linecap="round"/>')
    def polygon(self,pts,fill=None,stroke="none",w=1.0,opacity=None):
        fill=fill or PAL["neutralF"]; o=f' opacity="{opacity}"' if opacity is not None else ''
        p=" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
        self.el.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"{o}/>')
    def rect(self,x,y,w,h,fill=None,stroke="none",sw=1.0,rx=0,opacity=None):
        fill=fill or PAL["neutralF"]; o=f' opacity="{opacity}"' if opacity is not None else ''
        self.el.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{o}/>')
    def circle(self,x,y,r,fill=None,stroke="none",sw=1.0):
        fill=fill or PAL["ink"]
        self.el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def ellipse(self,cx,cy,rx,ry,rot=0,fill="none",stroke=None,w=1.5,dash=None):
        stroke=stroke or PAL["ink"]; d=f' stroke-dasharray="{dash}"' if dash else ''
        tr=f' transform="rotate({rot:.2f} {cx:.1f} {cy:.1f})"' if rot else ''
        self.el.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"{d}{tr}/>')
    def dot(self,x,y,r=3.2,fill=None): self.circle(x,y,r,fill=fill or PAL["ink"],stroke="#fff",sw=1.2)
    def arrow(self,x1,y1,x2,y2,stroke=None,w=2.0,head=7.0,dash=None):
        stroke=stroke or PAL["ink"]; ang=math.atan2(y2-y1,x2-x1)
        bx=x2-head*0.9*math.cos(ang); by=y2-head*0.9*math.sin(ang)
        self.line(x1,y1,bx,by,stroke=stroke,w=w,dash=dash)
        left=(x2-head*math.cos(ang-0.42), y2-head*math.sin(ang-0.42))
        right=(x2-head*math.cos(ang+0.42), y2-head*math.sin(ang+0.42))
        self.polygon([(x2,y2),left,right],fill=stroke)
    def right_angle(self,cx,cy,ux,uy,vx,vy,s=10,stroke=None):
        stroke=stroke or PAL["axis"]
        p1=(cx+ux*s,cy+uy*s); p2=(cx+ux*s+vx*s,cy+uy*s+vy*s); p3=(cx+vx*s,cy+vy*s)
        self.poly([p1,p2,p3],stroke=stroke,w=1.2)
    def svg(self):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" style="max-width:{self.w}px" '
                f'xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,Arial" '
                f'font-size="{self.fs}">' + "".join(self.el) + '</svg>')
class Frame:
    def __init__(self,cv,px,py,pw,ph,xmin,xmax,ymin,ymax):
        self.cv=cv; self.px=px; self.py=py; self.pw=pw; self.ph=ph
        self.xmin=xmin; self.xmax=xmax; self.ymin=ymin; self.ymax=ymax
    def X(self,x): return self.px+(x-self.xmin)/(self.xmax-self.xmin)*self.pw
    def Y(self,y): return self.py+self.ph-(y-self.ymin)/(self.ymax-self.ymin)*self.ph
    def P(self,x,y): return (self.X(x),self.Y(y))
    def curve(self,f,x0,x1,n=140,clipy=None):
        pts=[]
        for i in range(n+1):
            x=x0+(x1-x0)*i/n
            try: y=f(x)
            except Exception: continue
            if clipy is not None and (y<clipy[0] or y>clipy[1]):
                if y>clipy[1]: y=clipy[1]+(self.ymax-self.ymin)*0.02
                else: continue
            pts.append(self.P(x,y))
        return pts
    def axes(self,xlabel=None,ylabel=None):
        cv=self.cv
        cv.line(self.px,self.py+self.ph,self.px+self.pw,self.py+self.ph,stroke=PAL["axis"],w=1.3)
        cv.line(self.px,self.py,self.px,self.py+self.ph,stroke=PAL["axis"],w=1.3)
        if xlabel: cv.text(self.px+self.pw,self.py+self.ph+16,xlabel,anchor="end",fill=PAL["ink3"],size=10.5)
        if ylabel: cv.text(self.px-4,self.py-6,ylabel,anchor="middle",fill=PAL["ink3"],size=10.5)
def caption(cv,s): cv.text(cv.w/2,cv.h-8,s,fill=PAL["ink3"],size=10.5,style="italic")
DIAGRAMS={}
def reg(track):
    def deco(fn):
        DIAGRAMS.setdefault(track,[]).append(fn); return fn
    return deco
def _fgrid(self, xs, ys, col=None):
    col = col or PAL["grid"]
    for x in xs: self.cv.line(self.X(x), self.py, self.X(x), self.py+self.ph, stroke=col, w=1)
    for y in ys: self.cv.line(self.px, self.Y(y), self.px+self.pw, self.Y(y), stroke=col, w=1)
Frame.grid = _fgrid
def caption(cv,s):
    size=10.5; maxw=cv.w-22; est=len(s)*size*0.512
    if est>maxw: size=max(8.6,size*maxw/est)
    cv.text(cv.w/2,cv.h-8,s,fill=PAL["ink3"],size=round(size,1),style="italic")
def _curve2(self,f,x0,x1,n=170,clipy=None):
    pts=[]
    for i in range(n+1):
        x=x0+(x1-x0)*i/n
        try: y=f(x)
        except Exception: continue
        if clipy is not None and (y<clipy[0] or y>clipy[1]): continue
        pts.append(self.P(x,y))
    return pts
Frame.curve=_curve2
