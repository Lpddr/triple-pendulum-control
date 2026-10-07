"""Training experiment: search a dynamically feasible swing-up trajectory."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from algorithms.gps.trajectory import *

torch.set_num_threads(1)
plant=Plant()
p,kt=terminal_matrix(plant)
rng=np.random.default_rng(42)
x0=np.array([0,np.pi,0,0,0,0,0,0.])
for trial in range(12):
    n=[80,100,120,160][trial%4]
    goal=np.zeros(8)
    u=np.zeros((n,1))
    if trial>=4:
        u=.1*np.sin(np.arange(n)[:,None]*rng.uniform(.1,.8)+rng.uniform(0,6.28))
    cost=Cost(goal,p,n)
    cost.q=np.diag([1.,.01,.01,.01,.001,.001,.001,.001])
    cost.terminal=np.diag([100,1000,1000,1000,10,10,10,10])
    print('TRIAL',trial,'n',n,flush=True)
    x,u,k,v=improve(plant,x0,u,cost,200)
    if np.linalg.norm(x[-1]-goal)<1.:
        cost.terminal=p*10
        x,u,k,v=improve(plant,x0,u,cost,100)
    out=Path(f'runs/gps/search_{trial}.npz');out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,x=x,u=u,k=k,goal=goal,terminal_k=kt,terminal_p=p,cost=v,dt=plant.dt)
    print('RESULT',trial,np.round(x[-1],3),flush=True)
    if np.linalg.norm(x[-1])<.05:
        break
