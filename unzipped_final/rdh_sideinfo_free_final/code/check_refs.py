import re,io
tex=io.open('template_revised.tex',encoding='utf-8').read()
bib=io.open('step4j_refs.bib',encoding='utf-8').read()

# cited keys (order of first appearance)
cited=[]
for m in re.finditer(r'\\cite\{([^}]*)\}',tex):
    for k in m.group(1).split(','):
        k=k.strip()
        if k and k not in cited:
            cited.append(k)

# bib keys
bibkeys=re.findall(r'@\w+\{([^,]+),',bib)

citedset=set(cited); bibset=set(bibkeys)
print('cited count:',len(cited))
print('bib count  :',len(bibkeys))
print('duplicated bib keys:',[k for k in bibkeys if bibkeys.count(k)>1])
missing=[k for k in cited if k not in bibset]
uncited=[k for k in bibkeys if k not in citedset]
print('CITED-but-missing-in-bib:',missing)
print('IN-bib-but-uncited      :',uncited)
print('first-appearance order  :')
for i,k in enumerate(cited,1):
    print('  %2d %s'%(i,k))
print('INTEGRITY OK' if not missing and not uncited else 'INTEGRITY FAIL')
