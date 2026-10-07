import numpy as np
import torch
from tools.evaluate_gps import is_upright
from algorithms.gps.policy import PhasePolicy
from algorithms.gps.trajectory import Plant,Cost


def test_vertical_means_every_absolute_link_angle():
    assert is_upright(np.zeros(4),np.zeros(4))
    # Each relative angle is only 0.1 rad, but top link leans 0.3 rad.
    assert not is_upright(np.array([0,.1,.1,.1]),np.zeros(4))
    # A high-speed pass through the top is not stable upright.
    assert not is_upright(np.zeros(4),np.array([0,.5,.5,.5]))
    assert not is_upright(np.array([.96,0,0,0]),np.zeros(4))


def test_phase_is_only_network_input_and_saturates():
    actor=PhasePolicy(2)
    with torch.no_grad():
        actor.embedding.weight.zero_()
        actor.embedding.weight[:,0]=torch.tensor([1.,2.])
    x=torch.zeros((1,8),dtype=torch.float64);x[0,0]=.1
    assert actor.action(x,0).item()==.1
    assert actor.action(x,1).item()==.2
    assert actor.action(x,100000).item()==.2
    x[0,0]=2
    assert actor.action(x,100000).item()==1.


def test_model_linearization_predicts_actual_rk4_transition():
    p=Plant(threads=1)
    x=np.array([.1,.2,-.1,.3,.2,-.3,.1,.4])
    u=np.array([[.1]])
    nxt=p.step(x,u[0])
    a,b=p.derivatives(np.stack([x,nxt]),u)
    dx=np.array([1,-2,1,2,1,-1,2,-2])*1e-5
    du=np.array([2e-5])
    actual=p.step(x+dx,u[0]+du)
    np.testing.assert_allclose(actual,nxt+a[0]@dx+b[0]@du,atol=2e-7,rtol=0)
