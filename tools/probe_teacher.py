import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from algorithms.gps.trajectory import *

if __name__=='__main__':
    d=np.load(sys.argv[1] if len(sys.argv)>1 else 'runs/gps/search_0.npz')
    p=Plant(); rng=np.random.default_rng(8)
    if '--tracking' in sys.argv:
        d=dict(d)
        a,b=p.derivatives(d['x'],d['u'])
        q=np.repeat(np.diag([2.,8,8,8,.2,.2,.2,.2])[None],len(d['x']),axis=0)
        q[-1]=d['terminal_p']
        d['k'],_=backward(a,b,np.zeros((len(q),8)),q,np.zeros_like(d['u']),.1,np.zeros_like(d['u']),1e-6)
        np.savez('runs/gps/tracking.npz',**d)
    for width in [0,.03,.10,.15]:
        counts=[]; carts=[]
        for j in range(40):
            x=d['x'][0].copy(); x[:4]+=rng.uniform(-width,width,4); x[4:]+=rng.normal(0,width,4)
            longest=cur=0; cart=0
            for t in range(400):
                if t<len(d['u']): a=d['u'][t]+d['k'][t]@(x-d['x'][t])
                else: a=d['terminal_k']@(x-d['goal'])
                x=p.step(x,np.clip(a,-1,1))
                angles=np.arctan2(np.sin(np.cumsum(x[1:4])),np.cos(np.cumsum(x[1:4])))
                cur=cur+1 if np.max(abs(angles))<.2 else 0
                longest=max(longest,cur); cart=max(cart,abs(x[0]))
            counts.append(longest*.05); carts.append(cart)
        print('width',width,'teacher success',sum(np.array(counts)>=10),'hold mean',np.mean(counts),'cartmax',max(carts),flush=True)
