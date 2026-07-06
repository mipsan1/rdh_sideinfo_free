"""Experiment 4: predictor exploration. All predictors recoverable (dot-only real neighbors).
Fast single-layer T=0 estimate of PSNR + e0 fraction, across 5 images."""
import numpy as np
from rdh_v4 import load_gray, context_complexity, parity_idx

def neighbors(img):
    H,W=img.shape; f=img.astype(np.float64); NAN=np.nan
    up=np.full((H,W),NAN); up[1:,:]=f[:-1,:]
    dn=np.full((H,W),NAN); dn[:-1,:]=f[1:,:]
    lf=np.full((H,W),NAN); lf[:,1:]=f[:,:-1]
    rt=np.full((H,W),NAN); rt[:,:-1]=f[:,1:]
    return up,dn,lf,rt

def preds(img):
    up,dn,lf,rt=neighbors(img)
    st=np.stack([up,dn,lf,rt])
    with np.errstate(invalid='ignore'):
        mean=np.nanmean(st,0); med=np.nanmedian(st,0)
        dv=np.abs(up-dn); dh=np.abs(lf-rt)
        # edge-directed: choose axis with smaller gradient
        vert=(up+dn)/2; horz=(lf+rt)/2
        edge=np.where(dv<dh,vert,np.where(dh<dv,horz,mean))
        # inverse-gradient weighted average of the two axes
        wv=1.0/(dv+1.0); wh=1.0/(dh+1.0)
        wgt=(wv*vert+wh*horz)/(wv+wh)
        avg_mm=(mean+med)/2
    R=lambda a: np.floor(a+0.5)
    return {'mean4':R(mean),'median4':R(med),'edge':R(edge),'invgrad':R(wgt),'mean+med':R(avg_mm)}

def estimate(img,parity,pred):
    H,W=img.shape; ii,jj=parity_idx(H,W,parity)
    valid=~np.isnan(pred[ii,jj])
    e=np.where(valid,np.round(img[ii,jj]-np.nan_to_num(pred[ii,jj])),0).astype(int)
    e0=100*np.mean(e==0); std=np.std(e.astype(float))
    c=context_complexity(img,parity,2)[ii,jj]
    o=np.lexsort((jj,ii,c)); es=e[o]
    cum=np.cumsum((es==0).astype(int)); N=np.searchsorted(cum,10000)+1
    sq=0.5*10000+(N-10000); psnr=10*np.log10(255**2/(sq/(H*W)))
    return e0,std,psnr

names=['lena512.bmp','boat512.bmp','airplane.bmp','barbara.bmp','baboon.bmp']
pred_names=['mean4','median4','edge','invgrad','mean+med']
print(f"{'image':14s} "+" ".join(f"{p:>16s}" for p in pred_names))
for nm in names:
    img=load_gray('img/'+nm); pr=preds(img)
    cells=[]
    for p in pred_names:
        e0,std,psnr=estimate(img,0,pr[p])
        cells.append(f"{e0:4.1f}%/{psnr:5.1f}")
    print(f"{nm:14s} "+" ".join(f"{c:>16s}" for c in cells))
print("(cell = e0% / est.PSNR@10k dB, single-layer T=0 estimate)")
