import re, os, collections
tex=open('template_revised.tex',encoding='utf-8').read()
# remove commented lines for active checks
active='\n'.join([ln for ln in tex.splitlines() if not ln.lstrip().startswith('%')])
labels=re.findall(r'\\label\{([^}]+)\}', active)
refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', active)
cites=[]
for m in re.finditer(r'\\cite\{([^}]+)\}', active):
    cites += [x.strip() for x in m.group(1).split(',') if x.strip()]
bibs=re.findall(r'^\\bibitem\{([^}]+)\}', active, flags=re.M)
figs=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', active)
sections=re.findall(r'\\(section|subsection|subsubsection)\{([^}]*)\}(?:\\label\{([^}]+)\})?', active)
print('SECTIONS')
for k,t,l in sections: print(f'{k}: {t} [{l}]')
print('\nCOUNTS')
print('labels',len(labels),'unique',len(set(labels)))
print('refs',len(refs),'undefined',sorted(set(refs)-set(labels)))
print('duplicate labels',[x for x,c in collections.Counter(labels).items() if c>1])
print('cites',len(cites),'unique',len(set(cites)),'bibitems',len(bibs))
print('missing cites',sorted(set(cites)-set(bibs)))
print('uncited bibs',sorted(set(bibs)-set(cites)))
print('figures included',len(figs))
for f in figs:
    if f.startswith('Definitions/'): continue
    print(f, 'OK' if os.path.exists(f) else 'MISSING')
print('\nRISK PHRASES')
phrases=['only method','decisive','state of the art','competitive or better','exceeds classical','superior','outperforms','novel','first']
for p in phrases:
    hits=[(i+1,ln.strip()) for i,ln in enumerate(active.splitlines()) if p.lower() in ln.lower()]
    print('---',p,len(hits))
    for i,ln in hits[:8]: print(i,ln[:220])
print('\nBIB ACTIVE ENV')
for i,ln in enumerate(tex.splitlines(),1):
    if any(x in ln for x in ['\\begin{thebibliography}','\\end{thebibliography}','\\bibliography{','\\bibliographystyle']):
        if not ln.lstrip().startswith('%'): print(i,ln)
