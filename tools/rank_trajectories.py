"""Training-only comparison of trajectory initializations on fixed training seeds."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from algorithms.gps.trajectory import Plant
from algorithms.gps.refine_policy import score

plant=Plant(threads=4);rng=np.random.default_rng(314)
initial=np.tile(np.array([0,np.pi,0,0,0,0,0,0.]),(64,1))
initial[:,:4]+=rng.uniform(-.05,.05,(64,4))
initial[:,4:]+=rng.normal(0,.05,(64,4))
for f in sorted(Path('runs/gps').glob('search_*.npz')):
    d=np.load(f)
    ref=np.concatenate([d['x'][:-1],d['goal'][None]])
    weights=np.c_[np.concatenate([d['k'][:,0,:],d['terminal_k']]),np.r_[d['u'][:,0],0.]]
    value,hold=score(plant,weights[None],ref,initial,steps=400)
    print(f.name,'pass',int((hold>=10).sum()),'value',value[0],flush=True)
