from pathlib import Path
import json,urllib.request,re,html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'articulo_boletin_geologia_uis';OUT.mkdir(exist_ok=True)
SOURCE=OUT/'fuentes';SOURCE.mkdir(exist_ok=True)
url='https://revistas.uis.edu.co/index.php/revistaboletindegeologia/about/submissions'
try:
    content=urllib.request.urlopen(url,timeout=35).read().decode('utf-8');(SOURCE/'normas_enlace.html').write_text(content,encoding='utf-8')
    links=[]
    for match in re.finditer(r'<a\b[^>]*href=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</a>',content,re.S):
        text=html.unescape(re.sub('<[^>]+>',' ',match[2])).strip()
        if any(w in text.lower() for w in ['normas','plantilla','chequeo']):links.append(dict(text=text,url=match[1]))
    print(json.dumps(links,ensure_ascii=False,indent=2));(SOURCE/'normas_enlaces.json').write_text(json.dumps(links,ensure_ascii=False,indent=2),encoding='utf-8')
except Exception as exc:print('Submission page',type(exc).__name__,str(exc))
records=[]
for name in ['metadata_doi_crossref.json','metadata_complemento_20261010.json']:
    for x in json.loads((ROOT/'referencias'/name).read_text(encoding='utf-8')):
        m=x.get('metadata',{});records.append(m)
bydoi={m.get('DOI'):m for m in records if m.get('DOI')}
(SOURCE/'bibliografia_verificada.json').write_text(json.dumps(list(bydoi.values()),ensure_ascii=False,indent=2),encoding='utf-8')
for doi,m in bydoi.items():
    print(doi,':',m.get('title'),';',m.get('author',[{}])[0].get('family'),';',m.get('published',m.get('issued')))
