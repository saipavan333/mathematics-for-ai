import sys, io, cairosvg
from PIL import Image, ImageDraw
G={}
exec(open('/tmp/dg.py').read(), G)
OUT="/sessions/compassionate-fervent-hopper/mnt/outputs"
tracks=[int(a) for a in sys.argv[1:]] or list(range(1,14))
for t in tracks:
    try: exec(open(f'/tmp/t{t:02d}.py').read(), G)
    except FileNotFoundError: pass
DIAGRAMS=G['DIAGRAMS']
for t in tracks:
    imgs=[]
    for fn in DIAGRAMS.get(t,[]):
        lid,svg=fn()
        png=cairosvg.svg2png(bytestring=svg.encode('utf-8'),scale=2.0,background_color="#f8fafc")
        im=Image.open(io.BytesIO(png)).convert("RGB")
        strip=Image.new("RGB",(im.width,24),"#e9edf3"); d=ImageDraw.Draw(strip); d.text((8,7),f"Lesson {lid}",fill="#222")
        c=Image.new("RGB",(im.width,im.height+strip.height),"#fff"); c.paste(strip,(0,0)); c.paste(im,(0,strip.height))
        imgs.append(c)
    if not imgs: continue
    W=max(i.width for i in imgs); H=sum(i.height for i in imgs)+12*(len(imgs)+1)
    sheet=Image.new("RGB",(W+24,H),"#d7dce6"); yy=12
    for i in imgs: sheet.paste(i,(12,yy)); yy+=i.height+12
    p=f"{OUT}/sheet_t{t:02d}.png"; sheet.save(p); print("wrote",p,sheet.size)
