import io, cairosvg
from PIL import Image, ImageDraw
G={}; exec(open('/tmp/dg.py').read(),G)
for t in range(1,14):
    exec(open(f'/tmp/t{t:02d}.py').read(),G)
D=G['DIAGRAMS']
tiles=[]
for t in range(1,14):
    for fn in D.get(t,[]):
        lid,svg=fn()
        png=cairosvg.svg2png(bytestring=svg.encode('utf-8'),output_width=300,background_color="#f8fafc")
        im=Image.open(io.BytesIO(png)).convert("RGB")
        c=Image.new("RGB",(300,im.height+18),"#ffffff"); c.paste(im,(0,18))
        d=ImageDraw.Draw(c); d.rectangle([0,0,299,17],fill="#e9edf3"); d.text((5,4),lid,fill="#222")
        tiles.append(c)
cols=4; pad=8; cw=300; 
rows=(len(tiles)+cols-1)//cols
rowh=[max(tiles[r*cols+c].height for c in range(cols) if r*cols+c<len(tiles)) for r in range(rows)]
W=cols*cw+(cols+1)*pad; H=sum(rowh)+(rows+1)*pad
sheet=Image.new("RGB",(W,H),"#cdd3de")
y=pad
for r in range(rows):
    x=pad
    for c in range(cols):
        idx=r*cols+c
        if idx<len(tiles): sheet.paste(tiles[idx],(x,y))
        x+=cw+pad
    y+=rowh[r]+pad
# split into 2 halves vertically for readability
half=H//2
top=sheet.crop((0,0,W,half+120)); bot=sheet.crop((0,half-120,W,H))
top.save("/sessions/compassionate-fervent-hopper/mnt/outputs/montage_A.png")
bot.save("/sessions/compassionate-fervent-hopper/mnt/outputs/montage_B.png")
print("tiles:",len(tiles),"sheet:",sheet.size)
