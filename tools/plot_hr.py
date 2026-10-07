"""Plot recorded evaluation data; never resimulate or alter evaluation states."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--episode',default='artifacts/hr/demo/episode_000.npz')
p.add_argument('--out',default='artifacts/hr/verification.png')
args=p.parse_args()
z=np.load(args.episode)['trajectory'];t=z[:,0]
angles=np.rad2deg(np.arctan2(np.sin(np.cumsum(z[:,2:5],axis=1)),np.cos(np.cumsum(z[:,2:5],axis=1))))
fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True,layout='constrained')
fig.suptitle('Triple pendulum: one learned policy, continuous 35-second evaluation',fontsize=15)
for i,color in enumerate(['#009d9a','#8254cb','#e45482']):
    axes[0].plot(t,angles[:,i],label=f'Link {i+1}',color=color,lw=1.2)
axes[0].axhspan(-15,15,color='#3dbe85',alpha=.18)
axes[0].set_ylabel('Absolute angle (deg)');axes[0].legend(loc='upper right',ncol=3)
axes[1].plot(t,z[:,1],color='#3767c0');axes[1].axhline(.95,color='gray',ls='--');axes[1].axhline(-.95,color='gray',ls='--')
axes[1].set_ylabel('Cart x (m)');axes[1].set_ylim(-1.05,1.05)
axes[2].plot(t,500*z[:,9],color='#b66f19');axes[2].set_ylabel('Cart force (N)')
axes[3].plot(t,z[:,11],color='#208853',lw=2);axes[3].axhline(10,color='gray',ls='--',label='Required: 10 seconds');axes[3].legend()
axes[3].set_ylabel('Continuous hold (s)');axes[3].set_xlabel('Simulation time (s)')
for ax in axes:
    ax.axvspan(18,18.2,color='#e75545',alpha=.4)
    ax.grid(alpha=.2);ax.set_xlim(0,t[-1])
axes[0].annotate('Top-link push: 5 N for 0.2 s',xy=(18.1,0),xytext=(20,100),arrowprops={'arrowstyle':'->'})
fig.savefig(args.out,dpi=160)
print(Path(args.out).resolve())
