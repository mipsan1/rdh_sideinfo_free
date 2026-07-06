"""Experiment 1: net-bpp comparison. Original multi-predictor method needs a full
embpos map (512x512 bool) + pred_idx (2 bit/block). Map-free needs only T,N,exceptions.
We estimate REAL side-info cost by generating a realistic (spatially-clustered) embpos map
from the smoothest-N sorted positions and compressing it (bz2/zlib), plus the entropy bound.
"""
import numpy as np, bz2, zlib, math
from rdh_v4 import load_gray, sorted_order

def Hbits(n_ones, total):
    p=n_ones/total
    if p<=0 or p>=1: return 0.0
    return total*(-p*math.log2(p)-(1-p)*math.log2(1-p))

def embpos_sideinfo(img, P):
    """Realistic embpos map: mark the smoothest positions used to embed P bits (both parities)."""
    H,W=img.shape; total=H*W
    emb=np.zeros((H,W),dtype=np.uint8)
    # split across two parities like the real method; mark first N smoothest per parity
    per=P//2
    for parity in (0,1):
        ii,jj,_=sorted_order(img,parity,2)
        # assume ~ (per / e0frac) positions marked; here mark exactly the first 'per' smoothest as embedded
        sel=slice(0,per)
        emb[ii[sel],jj[sel]]=1
    n_ones=int(emb.sum())
    raw=np.packbits(emb.flatten()).tobytes()
    bz=len(bz2.compress(raw,9))*8
    zl=len(zlib.compress(raw,9))*8
    ent=Hbits(n_ones,total)
    return n_ones, ent, bz, zl

names=['lena512.bmp','boat512.bmp','airplane.bmp','barbara.bmp','baboon.bmp']
pred_idx_bits=4096*2  # 8192, before compression (blocks choose among <=4 predictors)
print(f"{'image':12s} {'P':>6s} {'ones':>7s} {'H-bound':>9s} {'bz2':>8s} {'zlib':>8s} {'orig_side':>10s} {'net_orig':>9s} {'net_free':>9s}")
for nm in names:
    img=load_gray('img/'+nm)
    for P in (10000,20000,40000):
        n1,ent,bz,zl=embpos_sideinfo(img,P)
        side_orig=min(bz,zl)+pred_idx_bits  # best-case compressed map + pred idx
        net_orig=P-side_orig
        net_free=P-80  # map-free: ~80 bits total side info
        print(f"{nm:12s} {P:6d} {n1:7d} {ent:9.0f} {bz:8d} {zl:8d} {side_orig:10d} {net_orig:9d} {net_free:9d}")
print("\n(bits. net = gross P - side info. Original must self-embed the embpos map.)")
