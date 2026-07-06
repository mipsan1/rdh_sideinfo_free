"""
Step 2a: improved complexity (windowed variance over context set) + half-split across layers.
Still map-free, still shifting PEE, still fully reversible.
"""
import numpy as np
from PIL import Image
import math

def load_gray(path):
    im = Image.open(path)
    if im.mode != 'L':
        im = im.convert('L')
    return np.array(im, dtype=np.int64)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float('inf') if mse == 0 else 10*math.log10(255.0*255.0/mse)

def integral(a):
    return np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1,0),(1,0)))

def box_sum(ii, r):
    """box sum with radius r using integral image ii (shape H+1,W+1). returns HxW."""
    H1, W1 = ii.shape
    H, W = H1-1, W1-1
    y0 = np.clip(np.arange(H)-r, 0, H); y1 = np.clip(np.arange(H)+r+1, 0, H)
    x0 = np.clip(np.arange(W)-r, 0, W); x1 = np.clip(np.arange(W)+r+1, 0, W)
    Y0, X0 = np.meshgrid(y0, x0, indexing='ij'); Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
    return ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]

def context_complexity(img, parity, radius=2):
    """variance of OTHER-parity pixels within a (2r+1)^2 window. recoverable by decoder."""
    H, W = img.shape
    grid = (np.add.outer(np.arange(H), np.arange(W)) % 2)
    other = (grid != parity).astype(np.float64)  # context set = other parity
    val = img.astype(np.float64) * other
    val2 = (img.astype(np.float64)**2) * other
    ii_cnt = integral(other); ii_val = integral(val); ii_val2 = integral(val2)
    cnt = box_sum(ii_cnt, radius); s1 = box_sum(ii_val, radius); s2 = box_sum(ii_val2, radius)
    cnt = np.where(cnt==0, 1, cnt)
    mean = s1/cnt
    var = s2/cnt - mean*mean
    return np.maximum(var, 0.0)

def rhombus_pred(img, parity):
    H, W = img.shape
    f = img.astype(np.float64); NAN=np.nan
    up=np.full((H,W),NAN); up[1:,:]=f[:-1,:]
    dn=np.full((H,W),NAN); dn[:-1,:]=f[1:,:]
    lf=np.full((H,W),NAN); lf[:,1:]=f[:,:-1]
    rt=np.full((H,W),NAN); rt[:,:-1]=f[:,1:]
    with np.errstate(invalid='ignore'):
        mean=np.nanmean(np.stack([up,dn,lf,rt]),axis=0)
    return np.floor(mean+0.5).astype(np.int64)

def parity_idx(H, W, parity):
    return np.where((np.add.outer(np.arange(H),np.arange(W))%2)==parity)

def sorted_order(img, parity, radius):
    x_hat = rhombus_pred(img, parity)
    comp = context_complexity(img, parity, radius)
    ii, jj = parity_idx(*img.shape, parity)
    c = comp[ii,jj]
    order = np.lexsort((jj, ii, c))
    return ii[order], jj[order], x_hat

def embed_layer(img, parity, T, payload_bits, start_bit, need, radius):
    ii, jj, x_hat = sorted_order(img, parity, radius)
    out = img.copy()
    bit_ptr = start_bit
    end_bit = min(len(payload_bits), start_bit+need)
    exceptions=[]; N=0; embedded=0
    for idx in range(len(ii)):
        if bit_ptr >= end_bit: break
        i,j = ii[idx], jj[idx]; xh=x_hat[i,j]; e=int(img[i,j]-xh); N=idx+1
        if abs(e)<=T:
            b=int(payload_bits[bit_ptr]); ep=2*e+b; xp=xh+ep
            if 0<=xp<=255: out[i,j]=xp; bit_ptr+=1; embedded+=1
            else: exceptions.append(idx)
        else:
            ep=e+(T+1) if e>T else e-(T+1); xp=xh+ep
            if 0<=xp<=255: out[i,j]=xp
            else: exceptions.append(idx)
    return out, dict(T=T,N=N,exceptions=exceptions,parity=parity,radius=radius), embedded

def decode_layer(img, side):
    T=side['T']; N=side['N']; parity=side['parity']; radius=side['radius']; exc=set(side['exceptions'])
    ii, jj, x_hat = sorted_order(img, parity, radius)
    out=img.copy(); bits=[]
    for idx in range(N):
        if idx in exc: continue
        i,j=ii[idx],jj[idx]; xh=x_hat[i,j]; ep=int(img[i,j]-xh)
        if -2*T<=ep<=2*T+1:
            e=ep>>1; b=ep-2*e; bits.append(b); out[i,j]=xh+e
        elif ep>2*T+1:
            out[i,j]=xh+(ep-(T+1))
        else:
            out[i,j]=xh+(ep+(T+1))
    return out, bits

def run(cover, T, payload_bits, radius=2):
    P=len(payload_bits); half=(P+1)//2
    s1, side_c, e1 = embed_layer(cover, 0, T, payload_bits, 0, half, radius)
    s2, side_d, e2 = embed_layer(s1, 1, T, payload_bits, e1, P-e1, radius)
    return s2,(side_c,side_d),e1+e2

def decode(stego, sides):
    sc,sd=sides
    i1,bd=decode_layer(stego,sd)
    i0,bc=decode_layer(i1,sc)
    return i0, bc+bd

def experiment(path,name,payloads=(10000,20000,40000),Ts=(0,1,2,3,4,5),radius=2):
    cover=load_gray(path)
    rng=np.random.default_rng(12345)
    print(f"\n===== {name} (radius={radius}) =====")
    for P in payloads:
        best=None
        for T in Ts:
            payload=rng.integers(0,2,size=P).astype(np.int64)
            stego,sides,total=run(cover,T,payload,radius)
            if total<P: continue
            rec,bits=decode(stego,sides)
            rec_bits=np.array(bits,dtype=np.int64)
            ber=np.mean(rec_bits[:P]!=payload[:P]) if len(rec_bits)>=P else 1.0
            mse=np.mean((rec-cover)**2)
            if ber==0 and mse==0:
                p=psnr(cover,stego)
                nexc=len(sides[0]['exceptions'])+len(sides[1]['exceptions'])
                if best is None or p>best[1]:
                    best=(T,p,nexc,sides[0]['N'],sides[1]['N'])
        if best:
            print(f"P={P:6d}: T={best[0]} PSNR={best[1]:.2f} dB exc={best[2]} N(c/d)={best[3]}/{best[4]}")
        else:
            print(f"P={P:6d}: no reversible config")

if __name__=='__main__':
    import sys
    imgs=[('img/lena512.bmp','Lena')]
    if len(sys.argv)>1 and sys.argv[1]=='all':
        imgs=[('img/lena512.bmp','Lena'),('img/baboon.bmp','Baboon'),('img/boat512.bmp','Boat'),
              ('img/airplane.bmp','Airplane'),('img/barbara.bmp','Barbara')]
    for path,name in imgs:
        experiment(path,name)
