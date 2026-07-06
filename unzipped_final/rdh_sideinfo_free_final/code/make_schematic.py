import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans'})
fig=plt.figure(figsize=(8.2,10.2))

# Panel (a): checkerboard + median-of-4 prediction context
ax=fig.add_subplot(2,1,1)
n=6
cross='#bcd4e6'; dot='#f6d7b0'
for i in range(n):
    for j in range(n):
        parity=(i+j)%2
        col=cross if parity==0 else dot
        ax.add_patch(mp.Rectangle((j,n-1-i),1,1,facecolor=col,edgecolor='white',lw=1.5))
        lab='C' if parity==0 else 'D'
        ax.text(j+0.5,n-1-i+0.5,lab,ha='center',va='center',color='#555',fontsize=9)
# center cross cell to be predicted
ci,cj=2,2
ax.add_patch(mp.Rectangle((cj,n-1-ci),1,1,facecolor='#3a7ca5',edgecolor='k',lw=2))
ax.text(cj+0.5,n-1-ci+0.5,'x',ha='center',va='center',color='white',fontsize=12,fontweight='bold')
# 4 orthogonal dot neighbors with arrows
for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
    ni,nj=ci+di,cj+dj
    ax.add_patch(mp.Rectangle((nj,n-1-ni),1,1,facecolor='#e9a44c',edgecolor='k',lw=2))
    ax.annotate('',xy=(cj+0.5-0.30*dj,n-1-ci+0.5+0.30*di),
                xytext=(nj+0.5,n-1-ni+0.5),
                arrowprops=dict(arrowstyle='->',color='k',lw=1.6))
ax.set_xlim(-0.2,n+0.2); ax.set_ylim(-0.2,n+0.2); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(a) Checkerboard parities and median-of-four prediction')
ax.text(0.5,-0.6,'C = cross parity, D = dot parity;  x is predicted from its 4 opposite-parity (D) neighbors',fontsize=8.5,ha='left')
ax.legend(handles=[mp.Patch(color=cross,label='Cross (C)'),mp.Patch(color=dot,label='Dot (D)'),
                   mp.Patch(color='#3a7ca5',label='Target pixel'),mp.Patch(color='#e9a44c',label='Prediction context')],
          loc='upper center',bbox_to_anchor=(0.5,-0.12),ncol=2,fontsize=8,frameon=False)

# Panel (b): two-layer embedding / extraction pipeline
ax2=fig.add_subplot(2,1,2); ax2.axis('off'); ax2.set_xlim(0,10); ax2.set_ylim(0,10)
def box(x,y,w,h,txt,fc):
    ax2.add_patch(mp.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=fc,ec='k',lw=1.4))
    ax2.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8.7)
def arrow(x0,y0,x1,y1):
    ax2.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='->',lw=1.6,color='k'))
# Embedding (top row)
ax2.text(0.2,9.4,'Embedding',fontsize=10,fontweight='bold')
box(0.2,7.4,2.0,1.4,'Cover\nimage','#eeeeee')
box(2.9,7.4,3.0,1.4,'Layer 1:\nembed C using D',cross)
box(6.6,7.4,3.0,1.4,'Layer 2:\nembed D using C\'',dot)
arrow(2.2,8.1,2.9,8.1); arrow(5.9,8.1,6.6,8.1)
# Stego
box(6.6,5.2,3.0,1.2,'Stego image','#d7ecd9')
arrow(8.1,7.4,8.1,6.4)
# Extraction (bottom)
ax2.text(0.2,4.3,'Extraction (reverse order)',fontsize=10,fontweight='bold')
box(6.6,2.3,3.0,1.4,'Invert Layer 2:\nrestore D (uses C\')',dot)
box(2.9,2.3,3.0,1.4,'Invert Layer 1:\nrestore C (uses D)',cross)
box(0.2,2.3,2.0,1.4,'Recovered\ncover + bits','#eeeeee')
arrow(6.6,5.2,8.1,3.7); arrow(6.6,3.0,5.9,3.0); arrow(2.9,3.0,2.2,3.0)
ax2.text(0.2,0.9,'Order/complexity computed only from unmodified opposite parity\n=> no position map, no predictor-index map (side info ~ tens of bits).',fontsize=8.3)
ax2.set_title('(b) Side-information-free double-layer pipeline')

plt.tight_layout()
plt.savefig('fig_method.png',dpi=600,bbox_inches='tight')
print('saved fig_method.png')
