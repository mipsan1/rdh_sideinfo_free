"""Final configurable map-free pipeline: selectable predictor, best-of(single,double) + T-sweep."""
import numpy as np, math
from rdh_v4 import load_gray, psnr, context_complexity, parity_idx

def _nbrs(img):
    H,W=img.shape; f=img.astype(np.float64); NAN=np.nan
    up=np.full((H,W),NAN); up[1:,:]=f[:-1,:]
    dn=np.full((H,W),NAN); dn[:-1,:]=f[1:,:]
    lf=np.full((H,W),NAN); lf[:,1:]=f[:,:-1]
    rt=np.full((H,W),NAN); rt[:,:-1]=f[:,1:]
    return up,dn,lf,rt

def pred_mean(img,parity):
    up,dn,lf,rt=_nbrs(img)
    with np.errstate(invalid='ignore'): m=np.nanmean(np.stack([up,dn,lf,rt]),0)
    return np.floor(m+0.5).astype(np.int64)

def pred_median(img,parity):
    up,dn,lf,rt=_nbrs(img)
    with np.errstate(invalid='ignore'): m=np.nanmedian(np.stack([up,dn,lf,rt]),0)
    return np.floor(m+0.5).astype(np.int64)

def pred_invgrad(img,parity):
    up,dn,lf,rt=_nbrs(img)
    with np.errstate(invalid='ignore'):
        dv=np.abs(up-dn); dh=np.abs(lf-rt)
        vert=(up+dn)/2; horz=(lf+rt)/2
        wv=1.0/(dv+1.0); wh=1.0/(dh+1.0)
        m=(wv*vert+wh*horz)/(wv+wh)
        # fallback to nanmean where an axis is fully nan
        mm=np.nanmean(np.stack([up,dn,lf,rt]),0)
        m=np.where(np.isnan(m),mm,m)
    return np.floor(m+0.5).astype(np.int64)

PREDS={'mean':pred_mean,'median':pred_median,'invgrad':pred_invgrad}

def sorted_order(img,parity,radius,predfn):
    x_hat=predfn(img,parity)
    comp=context_complexity(img,parity,radius)
    ii,jj=parity_idx(*img.shape,parity)
    c=comp[ii,jj]; order=np.lexsort((jj,ii,c))
    return ii[order],jj[order],x_hat

def embed_layer(img,parity,T,payload,start,need,radius,predfn):
    ii,jj,x_hat=sorted_order(img,parity,radius,predfn)
    out=img.copy(); bp=start; end=min(len(payload),start+need); exc=[]; N=0
    for idx in range(len(ii)):
        if bp>=end: break
        i,j=ii[idx],jj[idx]; xh=x_hat[i,j]; e=int(img[i,j]-xh); N=idx+1
        if abs(e)<=T:
            b=int(payload[bp]); xp=xh+2*e+b
            if 0<=xp<=255: out[i,j]=xp; bp+=1
            else: exc.append(idx)
        else:
            ep=e+(T+1) if e>T else e-(T+1); xp=xh+ep
            if 0<=xp<=255: out[i,j]=xp
            else: exc.append(idx)
    return out,dict(T=T,N=N,exceptions=exc,parity=parity,radius=radius),bp-start

def decode_layer(img,side,predfn):
    T=side['T'];N=side['N'];parity=side['parity'];radius=side['radius'];exc=set(side['exceptions'])
    ii,jj,x_hat=sorted_order(img,parity,radius,predfn)
    out=img.copy(); bits=[]
    for idx in range(N):
        if idx in exc: continue
        i,j=ii[idx],jj[idx]; xh=x_hat[i,j]; ep=int(img[i,j]-xh)
        if -2*T<=ep<=2*T+1: e=ep>>1; b=ep-2*e; bits.append(b); out[i,j]=xh+e
        elif ep>2*T+1: out[i,j]=xh+(ep-(T+1))
        else: out[i,j]=xh+(ep+(T+1))
    return out,bits

def run_double(cover,T,payload,radius,predfn):
    P=len(payload); half=(P+1)//2
    s1,sc,e1=embed_layer(cover,0,T,payload,0,half,radius,predfn)
    s2,sd,e2=embed_layer(s1,1,T,payload,e1,P-e1,radius,predfn)
    return s2,(sc,sd),e1+e2
def dec_double(stego,sides,predfn):
    sc,sd=sides; i1,bd=decode_layer(stego,sd,predfn); i0,bc=decode_layer(i1,sc,predfn); return i0,bc+bd

def run_single(cover,T,payload,radius,predfn):
    s,side,emb=embed_layer(cover,0,T,payload,0,len(payload),radius,predfn); return s,side,emb
def dec_single(stego,side,predfn):
    _,bits=decode_layer(stego,side,predfn); return bits

def best_psnr(cover,P,pred='median',radius=2,Ts=range(0,9),seed=7):
    predfn=PREDS[pred]; rng=np.random.default_rng(seed); best=None
    payload=rng.integers(0,2,size=P).astype(np.int64)
    for mode in ('single','double'):
        for T in Ts:
            if mode=='single':
                stego,side,emb=run_single(cover,T,payload,radius,predfn)
                if emb<P: continue
                bits=dec_single(stego,side,predfn); rec,_=decode_layer(stego,side,predfn)
            else:
                stego,sides,emb=run_double(cover,T,payload,radius,predfn)
                if emb<P: continue
                rec,bits=dec_double(stego,sides,predfn)
            rb=np.array(bits,dtype=np.int64)
            ber=np.mean(rb[:P]!=payload[:P]) if len(rb)>=P else 1.0
            mse=np.mean((rec-cover)**2)
            if ber==0 and mse==0:
                p=psnr(cover,stego)
                if best is None or p>best[0]: best=(p,mode,T)
    return best

if __name__=='__main__':
    import sys
    names=['lena512.bmp','boat512.bmp','airplane.bmp','barbara.bmp','baboon.bmp']
    preds=sys.argv[1:] or ['median','invgrad']
    print(f"{'image':14s} "+" ".join(f"{p:>16s}" for p in preds))
    for nm in names:
        cover=load_gray('img/'+nm); cells=[]
        for p in preds:
            b=best_psnr(cover,10000,pred=p)
            cells.append(f"{b[0]:.2f}({b[1][0]}T{b[2]})" if b else 'none')
        print(f"{nm:14s} "+" ".join(f"{c:>16s}" for c in cells))
