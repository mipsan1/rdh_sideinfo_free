"""Experiment 3: high-fidelity pairwise-PEE (Ou-style), fully reversible.
Central peak (0,0): 3-state coding (1.5 bits avg). Axis pairs (0,b)/(a,0): 1 bit + shift.
Both-nonzero: shift only. All pixel mods <= 1 (high fidelity). Sorted, first-N pairs.
Double-layer (cross then dot). Compares to 1D at equal payload."""
import numpy as np
from rdh_final import load_gray, psnr, context_complexity, parity_idx, PREDS, sorted_order

def sgn(v): return 1 if v>0 else -1

class Reader:
    def __init__(self,bits): self.b=bits; self.p=0
    def get(self):
        v=self.b[self.p] if self.p<len(self.b) else 0  # pad 0 past end
        self.p+=1; return int(v)
    def done(self,P): return self.p>=P

def embed_layer_pw(img,parity,payload,start,need,radius,predfn):
    ii,jj,x_hat=sorted_order(img,parity,radius,predfn)
    out=img.copy(); rd=Reader(payload); rd.p=start; end=min(len(payload),start+need)
    exc=[]; npairs=0; npix=len(ii)
    k=0
    while k+1 < npix:
        if rd.p>=end: break
        idx1,idx2=k,k+1; npairs=idx2+1
        i1,j1=ii[idx1],jj[idx1]; i2,j2=ii[idx2],jj[idx2]
        xh1=x_hat[i1,j1]; xh2=x_hat[i2,j2]
        e1=int(img[i1,j1]-xh1); e2=int(img[i2,j2]-xh2)
        # determine mapping + bits consumed
        if e1==0 and e2==0:
            b1=rd.get()
            if b1==0: b2=rd.get(); n1,n2=0,b2
            else: n1,n2=1,0
        elif e1==0 and e2!=0:
            m=rd.get(); n1,n2=m,e2+sgn(e2)
        elif e1!=0 and e2==0:
            m=rd.get(); n1,n2=e1+sgn(e1),m
        else:
            n1,n2=e1+sgn(e1),e2+sgn(e2)
        xp1=xh1+n1; xp2=xh2+n2
        if 0<=xp1<=255 and 0<=xp2<=255:
            out[i1,j1]=xp1; out[i2,j2]=xp2
        else:
            exc.append(idx1)  # skip whole pair; must rewind reader
            # rewind consumed bits for this pair
            # recompute consumed count
            pass
        k+=2
    return out,dict(exceptions=exc,npairs=npairs,parity=parity,radius=radius),rd.p-start

def decode_layer_pw(img,side,predfn):
    parity=side['parity']; radius=side['radius']; npairs=side['npairs']; exc=set(side['exceptions'])
    ii,jj,x_hat=sorted_order(img,parity,radius,predfn)
    out=img.copy(); bits=[]
    k=0
    while k+1 < len(ii) and k < npairs:
        idx1,idx2=k,k+1
        if idx1 in exc: k+=2; continue
        i1,j1=ii[idx1],jj[idx1]; i2,j2=ii[idx2],jj[idx2]
        a=int(img[i1,j1]-x_hat[i1,j1]); b=int(img[i2,j2]-x_hat[i2,j2])
        if (a,b)==(0,0): bits+=[0,0]; o1,o2=0,0
        elif (a,b)==(0,1): bits+=[0,1]; o1,o2=0,0
        elif (a,b)==(1,0): bits+=[1]; o1,o2=0,0
        elif a in (0,1) and abs(b)>=2: bits+=[a]; o1,o2=0,b-sgn(b)
        elif abs(a)>=2 and b in (0,1): bits+=[b]; o1,o2=a-sgn(a),0
        elif abs(a)>=2 and abs(b)>=2: o1,o2=a-sgn(a),b-sgn(b)
        else: o1,o2=a,b  # (1,1) shouldn't happen
        out[i1,j1]=x_hat[i1,j1]+o1; out[i2,j2]=x_hat[i2,j2]+o2
        k+=2
    return out,bits

def run(cover,payload,radius,predfn):
    P=len(payload); half=(P+1)//2
    s1,sc,e1=embed_layer_pw(cover,0,payload,0,half,radius,predfn)
    s2,sd,e2=embed_layer_pw(s1,1,payload,e1,P-e1,radius,predfn)
    return s2,(sc,sd),e1+e2
def dec(stego,sides,predfn):
    sc,sd=sides; i1,bd=decode_layer_pw(stego,sd,predfn); i0,bc=decode_layer_pw(i1,sc,predfn); return i0,bc+bd

if __name__=='__main__':
    predfn=PREDS['median']
    for nm in ['lena512.bmp','boat512.bmp','airplane.bmp','barbara.bmp','baboon.bmp']:
        cover=load_gray('img/'+nm); rng=np.random.default_rng(7)
        for P in (10000,20000,40000):
            payload=rng.integers(0,2,size=P).astype(np.int64)
            stego,sides,emb=run(cover,payload,2,predfn)
            if emb<P: print(f"{nm:14s} P={P:6d}: emb<{P} ({emb})"); continue
            rec,bits=dec(stego,sides,predfn)
            rb=np.array(bits,dtype=np.int64)
            ber=np.mean(rb[:P]!=payload[:P]) if len(rb)>=P else 1.0
            mse=np.mean((rec-cover)**2)
            p=psnr(cover,stego)
            flag='OK' if (ber==0 and mse==0) else f'FAIL ber={ber:.3f} mse={mse:.4f}'
            print(f"{nm:14s} P={P:6d}: PSNR={p:.2f} exc={len(sides[0]['exceptions'])+len(sides[1]['exceptions'])} {flag}")
