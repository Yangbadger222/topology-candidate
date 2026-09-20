"""Checkpoint failures must not silently ignore model weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,pytest
from road_model import canonical,road_loss,freeze,gradient_audit,category


def test_prefix_and_collision():
    assert set(canonical({'model.module.net.x':torch.ones(1)}))=={'x'}
    with pytest.raises(ValueError): canonical({'model.x':torch.ones(1),'x':torch.ones(1)})


def test_road_only_loss_gradient():
    logits=torch.randn(1,2,4,4,requires_grad=True)
    loss,_,_=road_loss(logits[:,1],torch.ones(1,4,4),torch.ones(1,4,4,dtype=torch.bool)); loss.backward()
    assert logits.grad[:,0].count_nonzero()==0 and logits.grad[:,1].count_nonzero()>0
