"""Experiment 2: capacity-PSNR curves. median predictor, best-of(single,double)+T-sweep."""
import numpy as np
from rdh_final import load_gray, best_psnr

names=['lena512.bmp','boat512.bmp','airplane.bmp','barbara.bmp','baboon.bmp']
payloads=[5000,10000,15000,20000,30000,40000,50000]
print("payload "+" ".join(f"{p:>12s}" for p in [n.split('.')[0][:8] for n in names]))
res={n:{} for n in names}
for P in payloads:
    row=[]
    for nm in names:
        cover=load_gray('img/'+nm)
        b=best_psnr(cover,P,pred='median',radius=2,Ts=range(0,12))
        if b: res[nm][P]=b; row.append(f"{b[0]:5.2f}({b[1][0]}T{b[2]})")
        else: res[nm][P]=None; row.append('   none    ')
    print(f"{P:7d} "+" ".join(f"{c:>12s}" for c in row))
print("\n(cell=PSNR dB (mode s/d, T). median predictor, map-free, fully reversible.)")
