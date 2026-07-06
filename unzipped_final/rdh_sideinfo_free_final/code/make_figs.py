import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':11,'figure.dpi':150,'savefig.dpi':300,'font.family':'DejaVu Sans'})

images=['Lena','Boat','Airplane','Barbara','Baboon']
payk=[5,10,20,30,40,50]  # thousands of bits
curves={
 'Lena':[60.39,56.80,53.31,51.28,49.77,48.55],
 'Boat':[60.07,56.89,53.68,51.69,50.16,48.77],
 'Airplane':[65.56,61.87,57.97,55.66,53.93,52.52],
 'Barbara':[62.86,58.93,54.80,51.88,49.67,47.93],
 'Baboon':[54.61,50.99,46.86,44.24,42.22,40.58],
}
markers=['o','s','^','D','v']

# Fig 1: capacity-PSNR curves
plt.figure(figsize=(6.2,4.4))
for im,mk in zip(images,markers):
    plt.plot(payk,curves[im],marker=mk,label=im,linewidth=1.6,markersize=6)
plt.xlabel('Payload (x1000 bits)')
plt.ylabel('PSNR (dB)')
plt.title('Capacity--PSNR of the proposed side-information-free scheme')
plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fig1_capacity_psnr.png')
plt.close()

# Fig 2: net-bpp comparison at 10k bits
net_map={'Lena':-54952,'Boat':-54560,'Airplane':-32320,'Barbara':-32776,'Baboon':-44912}
net_free={im:9920 for im in images}
x=np.arange(len(images)); w=0.38
plt.figure(figsize=(6.6,4.4))
b1=plt.bar(x-w/2,[net_map[i] for i in images],w,label='Map-based',color='#d1495b')
b2=plt.bar(x+w/2,[net_free[i] for i in images],w,label='Proposed (map-free)',color='#2e86ab')
plt.axhline(0,color='k',linewidth=0.8)
plt.xticks(x,images)
plt.ylabel('Net capacity (bits) at 10,000-bit payload')
plt.title('Net embedding capacity: map-based vs. side-information-free')
plt.legend()
plt.grid(True,axis='y',alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_netbpp.png')
plt.close()

# Fig 3: predictor comparison at 10k
pred={'Median-of-4':[56.80,56.89,61.87,58.93,50.99],
      'Inverse-gradient':[56.77,56.81,61.79,58.88,51.07],
      'Mean rhombus':[56.78,56.84,61.80,58.96,51.32]}
x=np.arange(len(images)); w=0.26
cols=['#2e86ab','#e9c46a','#8ac926']
plt.figure(figsize=(7.0,4.4))
for k,(name,vals) in enumerate(pred.items()):
    plt.bar(x+(k-1)*w,vals,w,label=name,color=cols[k])
plt.xticks(x,images)
plt.ylabel('PSNR (dB) at 10,000 bits')
plt.ylim(48,64)
plt.title('Predictor comparison (differences < 0.3 dB)')
plt.legend()
plt.grid(True,axis='y',alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_predictors.png')
plt.close()

# Fig 4: pairwise vs 1D at 10k
pw10=[55.58,55.32,58.97,55.96,49.10]
od10=[56.80,56.89,61.87,58.93,50.99]
x=np.arange(len(images)); w=0.38
plt.figure(figsize=(6.6,4.4))
plt.bar(x-w/2,pw10,w,label='Pairwise PEE (high-fidelity)',color='#9d4edd')
plt.bar(x+w/2,od10,w,label='Proposed 1D shifting PEE',color='#2e86ab')
plt.xticks(x,images)
plt.ylabel('PSNR (dB) at 10,000 bits')
plt.ylim(46,64)
plt.title('Faithful pairwise PEE vs. proposed 1D PEE')
plt.legend()
plt.grid(True,axis='y',alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_pairwise.png')
plt.close()

# Fig 5: SOTA comparison on Lena @10k
methods=['Sachnev\n2009','Ou\n2013','Kim\n2019','Ou\n2019','Zhang\n2022','Li\n2011','Li\n2015','Proposed']
vals=[58.45,59.75,60.12,60.71,60.94,60.97,61.20,56.80]
cols=['#adb5bd']*7+['#2e86ab']
plt.figure(figsize=(7.2,4.4))
plt.bar(range(len(methods)),vals,color=cols)
plt.xticks(range(len(methods)),methods,fontsize=9)
plt.ylabel('PSNR (dB), Lena @ 10,000 bits')
plt.ylim(54,63)
plt.title('Reported PSNR vs. proposed (side info ~80 bits)')
plt.grid(True,axis='y',alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_sota.png')
plt.close()

print('done: fig1..fig5 saved')
