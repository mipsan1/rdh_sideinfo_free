import sys, numpy as np
from rdh_final import load_gray, best_psnr
nm=sys.argv[1]
payloads=[5000,10000,20000,30000,40000,50000]
cover=load_gray('img/'+nm)
for P in payloads:
    Ts=range(0,8) if P<=20000 else range(0,12)
    b=best_psnr(cover,P,pred='median',radius=2,Ts=Ts)
    print(f"{nm:14s} P={P:6d}: "+(f"{b[0]:5.2f} dB (mode={b[1]} T={b[2]})" if b else 'none'),flush=True)
