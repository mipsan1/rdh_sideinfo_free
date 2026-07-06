"""SSIM + runtime at 10k best reversible operating point (map-free median pipeline)."""
import numpy as np, time, cv2
from rdh_v4 import load_gray, psnr
from rdh_final import best_psnr, run_single, dec_single, run_double, dec_double, decode_layer, PREDS

def ssim(a,b):
    a=a.astype(np.float64); b=b.astype(np.float64)
    C1=(0.01*255)**2; C2=(0.03*255)**2
    k=(11,11); s=1.5
    mu_a=cv2.GaussianBlur(a,k,s); mu_b=cv2.GaussianBlur(b,k,s)
    mu_a2=mu_a*mu_a; mu_b2=mu_b*mu_b; mu_ab=mu_a*mu_b
    sa=cv2.GaussianBlur(a*a,k,s)-mu_a2
    sb=cv2.GaussianBlur(b*b,k,s)-mu_b2
    sab=cv2.GaussianBlur(a*b,k,s)-mu_ab
    m=((2*mu_ab+C1)*(2*sab+C2))/((mu_a2+mu_b2+C1)*(sa+sb+C2))
    return float(m.mean())

names=['lena512.bmp','boat512.bmp','airplane.bmp','barbara.bmp','baboon.bmp']
labels=['Lena','Boat','Airplane','Barbara','Baboon']
P=10000; predfn=PREDS['median']; rng=np.random.default_rng(7)
payload=rng.integers(0,2,size=P).astype(np.int64)

print(f"{'Image':10s} {'mode':7s} {'PSNR':>7s} {'SSIM':>8s} {'enc_ms':>8s} {'dec_ms':>8s} {'BER':>4s} {'MSE':>4s}")
for nm,lab in zip(names,labels):
    cover=load_gray('img/'+nm)
    b=best_psnr(cover,P,pred='median')
    if b is None:
        print(f'{lab:10s} none'); continue
    _,mode,T=b
    t0=time.perf_counter()
    if mode=='single':
        stego,side,emb=run_single(cover,T,payload,2,predfn)
    else:
        stego,sides,emb=run_double(cover,T,payload,2,predfn)
    t1=time.perf_counter()
    if mode=='single':
        rec,bits=decode_layer(stego,side,predfn)
    else:
        rec,bits=dec_double(stego,sides,predfn)
    t2=time.perf_counter()
    rb=np.array(bits,dtype=np.int64)
    ber=float(np.mean(rb[:P]!=payload[:P])) if len(rb)>=P else 1.0
    mse=float(np.mean((rec-cover)**2))
    print(f"{lab:10s} {mode:7s} {psnr(cover,stego):7.2f} {ssim(cover,stego):8.5f} {1000*(t1-t0):8.1f} {1000*(t2-t1):8.1f} {ber:4.1f} {mse:4.1f}")
