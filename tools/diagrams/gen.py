G={}; exec(open('/tmp/dg.py').read(),G)
for t in range(1,14): exec(open(f'/tmp/t{t:02d}.py').read(),G)
D=G['DIAGRAMS']
out=['/* auto-generated — verified geometric diagrams for tracks 1-13 (one per lesson) */',
     ';(function(){var L=window.LESSON_CONTENT||(window.LESSON_CONTENT={});',
     'function setd(id,svg){if(L[id])L[id].diagram=svg;}']
ids=[]
for t in range(1,14):
    for fn in D.get(t,[]):
        lid,svg=fn()
        assert '`' not in svg, lid+" has backtick"
        assert '${' not in svg, lid+" has dollar-brace"
        assert svg.startswith('<svg'), lid
        out.append('setd("%s", String.raw`%s`);'%(lid,svg))
        ids.append(lid)
out.append('})();')
open('/sessions/compassionate-fervent-hopper/mnt/Mathematics/content/diagrams.js','w',encoding='utf-8').write('\n'.join(out))
print("wrote diagrams.js with",len(ids),"diagrams")
print("ids:",",".join(ids))
