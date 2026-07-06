"""
Decisive test: does block-wise multi-predictor selection add value in the map-free setting?
Compare, at a fixed NET payload P:
  (1) single rhombus (no overhead, embeds P)
  (2) block multi-predictor with signaling overhead (embeds P + overhead)
  (3) block multi-predictor ORACLE (no overhead, upper bound)
Candidates per block: rhombus / vertical / horizontal. Selection minimizes sum|e| (paper's D).
"""
import numpy as np, math
from rdh_mapfree_v2 import load_gray, psnr, context_complexity, parity_idx

def pred_components(img):
    H,W=img.shape; f=img.astype(np.float64); NAN=np.nan
    up=np.full((H,W),NAN); up[1:,:]=f[:-1,:]
    dn=np.full((H,W),NAN); dn[:-1,:]=f[1:,:]
    lf=np.full((H,W),NAN); lf[:,1:]=f[:,:-1]
    rt=np.full((H,W),NAN); rt[:,:-1]=f[:,1:]
    with np.errstate(invalid='ignore'):
        rhom=np.nanmean(np.stack([up,dn,lf,rt]),axis=0)
        vert=np.nanmean(np.stack([up,dn]),axis=0)
        horz=np.nanmean(np.stack([lf,rt]),axis=0)
    def fix(a): return np.floor(np.where(np.isnan(a),0.0,a)+0.5).astype(np.int64)
    return fix(rhom),fix(vert),fix(horz)

def select_pred_map(img,parity,block):
    """per-block choice in {0,1,2} minimizing sum|e| over that block's parity pixels."""
    H,W=img.shape
    rhom,vert,horz=pred_components(img)
    grid=(np.add.outer(np.arange(H),np.arange(W))%2)
    mask=(grid==parity)
    preds=[rhom,vert,horz]
    nby,nbx=H//block,W//block
    pmap=np.zeros((nby,nbx),dtype=np.uint8)
    for by in range(nby):
        for bx in range(nbx):
            y0,y1=by*block,(by+1)*block; x0,x1=bx*block,(bx+1)*block
            m=mask[y0:y1,x0:x1]
            x=img[y0:y1,x0:x1][m]
            best=None
            for k,P in enumerate(preds):
                e=x-P[y0:y1,x0:x1][m]
                d=np.sum(np.abs(e))
                if best is None or d<best[1]: best=(k,d)
            pmap[by,bx]=best[0]
    return pmap

def xhat_from_map(img,pmap,block):
    H,W=img.shape
    rhom,vert,horz=pred_components(img)
    preds=[rhom,vert,horz]
    xh=np.zeros((H,W),dtype=np.int64)
    for by in range(pmap.shape[0]):
        for bx in range(pmap.shape[1]):
            y0,y1=by*block,(by+1)*block; x0,x1=bx*block,(bx+1)*block
            xh[y0:y1,x0:x1]=preds[pmap[by,bx]][y0:y1,x0:x1]
    return xh

def sorted_order_xh(img,parity,radius,xh):
    comp=context_complexity(img,parity,radius)
    ii,jj=parity_idx(*img.shape,parity)
    c=comp[ii,jj]; order=np.lexsort((jj,ii,c))
    return ii[order],jj[order]

def embed_layer(img,parity,T,payload,start,need,radius,xh):
    ii,jj=sorted_order_xh(img,parity,radius,xh)
    out=img.copy(); bp=start; end=min(len(payload),start+need); exc=[]; N=0; emb=0
    for idx in range(len(ii)):
        if bp>=end: break
        i,j=ii[idx],jj[idx]; h=xh[i,j]; e=int(img[i,j]-h); N=idx+1
        if abs(e)<=T:
            b=int(payload[bp]); ep=2*e+b; xp=h+ep
            if 0<=xp<=255: out[i,j]=xp; bp+=1; emb+=1
            else: exc.append(idx)
        else:
            ep=e+(T+1) if e>T else e-(T+1); xp=h+ep
            if 0<=xp<=255: out[i,j]=xp
            else: exc.append(idx)
    return out,dict(T=T,N=N,exceptions=exc,parity=parity,radius=radius),emb

def decode_layer(img,side,xh):
    T=side['T'];N=side['N'];exc=set(side['exceptions']);parity=side['parity'];radius=side['radius']
    ii,jj=sorted_order_xh(img,parity,radius,xh)
    out=img.copy(); bits=[]
    for idx in range(N):
        if idx in exc: continue
        i,j=ii[idx],jj[idx]; h=xh[i,j]; ep=int(img[i,j]-h)
        if -2*T<=ep<=2*T+1:
            e=ep>>1; b=ep-2*e; bits.append(b); out[i,j]=h+e
        elif ep>2*T+1: out[i,j]=h+(ep-(T+1))
        else: out[i,j]=h+(ep+(T+1))
    return out,bits

def run_single(cover,T,payload,radius):
    rhom,_,_=pred_components(cover)
    P=len(payload)
    s1,sc,e1=embed_layer(cover,0,T,payload,0,P,radius,rhom)
    rhom2,_,_=pred_components(s1)
    s2,sd,e2=embed_layer(s1,1,T,payload,e1,P-e1,radius,rhom2)
    return s2,e1+e2

def run_multi(cover,T,payload,radius,block,oracle=False):
    P=len(payload)
    pm0=select_pred_map(cover,0,block); xh0=xhat_from_map(cover,pm0,block)
    s1,sc,e1=embed_layer(cover,0,T,payload,0,P,radius,xh0)
    pm1=select_pred_map(s1,1,block); xh1=xhat_from_map(s1,pm1,block)
    s2,sd,e2=embed_layer(s1,1,T,payload,e1,P-e1,radius,xh1)
    overhead=0 if oracle else 2*pm0.size+2*pm1.size
    return s2,e1+e2,overhead

def best_single(cover,P,radius=2,Ts=range(0,8)):
    rng=np.random.default_rng(1); best=None
    for T in Ts:
        payload=rng.integers(0,2,size=P).astype(np.int64)
        stego,total=run_single(cover,T,payload,radius)
        if total<P: continue
        p=psnr(cover,stego)
        if best is None or p>best[1]: best=(T,p)
    return best

def best_multi(cover,P_net,radius=2,block=16,oracle=False,Ts=range(0,8)):
    rng=np.random.default_rng(1); best=None
    for T in Ts:
        # first find overhead (indep of payload content) via a probe
        probe=rng.integers(0,2,size=P_net+5000).astype(np.int64)
        _,_,overhead=run_multi(cover,T,probe,radius,block,oracle)
        total_need=P_net+overhead
        payload=rng.integers(0,2,size=total_need).astype(np.int64)
        stego,total,oh=run_multi(cover,T,payload,radius,block,oracle)
        if total<total_need: continue
        p=psnr(cover,stego)
        if best is None or p>best[1]: best=(T,p,overhead)
    return best

if __name__=='__main__':
    import sys
    imgs=[('img/lena512.bmp','Lena'),('img/baboon.bmp','Baboon'),('img/barbara.bmp','Barbara')]
    for path,name in imgs:
        cover=load_gray(path)
        print(f"\n===== {name} =====")
        for P in (10000,20000,40000):
            s=best_single(cover,P)
            m16=best_multi(cover,P,block=16)
            o16=best_multi(cover,P,block=16,oracle=True)
            print(f"P={P:6d}: single={s[1]:.2f}  multi16(sig)={m16[1]:.2f}(oh={m16[2]})  "
                  f"multiORACLE={o16[1]:.2f}  gain_oracle={o16[1]-s[1]:+.2f}  gain_real={m16[1]-s[1]:+.2f}")
