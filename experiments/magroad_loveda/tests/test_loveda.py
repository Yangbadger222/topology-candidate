import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from dataset import LoveDARoad,road_target,inner_boundary,hard_negative_targets
from utils import load_config
from planning.astar_eval import astar_eval


def test_loading_and_native_shape(tmp_path):
    for split in ('Train','Val'):
        for domain in ('Urban','Rural'):
            root=tmp_path/split/domain; (root/'images_png').mkdir(parents=True); (root/'masks_png').mkdir()
            Image.fromarray(np.zeros((1024,1024,3),dtype=np.uint8)).save(root/'images_png/1.png')
            label=np.ones((1024,1024),dtype=np.uint8); label[0]=3; label[1]=0
            Image.fromarray(label).save(root/'masks_png/1.png')
    ds=LoveDARoad(tmp_path,'train','all'); assert len(ds)==2
    x=ds[0]; assert x['rgb'].shape==(1024,1024,3) and x['rgb'].dtype==torch.float32
    assert x['road_mask'].shape==(1024,1024) and x['road_mask'][0].all() and not x['valid'][1].any()
    assert len(LoveDARoad(tmp_path,'val','urban',True))==1


def test_class_extraction():
    target,valid=road_target(np.array([[0,1,2,3,4,5,6,7,255]],dtype=np.uint8))
    assert target.tolist()==[[0.,0.,0.,1.,0.,0.,0.,0.,0.]]
    assert valid.tolist()==[[False,True,True,True,True,True,True,True,False]]


def test_inner_boundary_is_inside_only():
    mask=np.ones((4,8),dtype=bool)
    got=inner_boundary(mask,1).numpy()
    expected=np.ones((4,8),dtype=bool); expected[1:3,1:7]=False
    assert np.array_equal(got,expected)


def test_hard_negative_priority_and_weights():
    label=np.array([[2,2,2,2,2,3,3,3],[2,2,2,2,2,3,3,3],[2,2,2,2,2,3,3,3]],dtype=np.uint8)
    got=hard_negative_targets(label,boundary_width=1,building_weight=2,boundary_weight=4)
    assert np.all(got['loss_weight'].numpy()[:,5:]==1)
    assert not np.any((got['building_boundary'] & got['road_mask'].bool()).numpy())
    assert set(np.unique(got['loss_weight'].numpy()[:,:5]))=={2.,4.}


def test_hard_negative_v2_config_and_priority():
    config=load_config(Path(__file__).resolve().parents[1]/'config/loveda/vitb_1024_loveda_lora_hard_negative_v2.yaml')
    label=np.ones((20,24),dtype=np.uint8); label[:,0]=0; label[2:18,3:19]=2; label[:,23]=3
    got=hard_negative_targets(label,boundary_width=config.BUILDING_BOUNDARY_WIDTH,building_weight=config.BUILDING_NEGATIVE_WEIGHT,boundary_weight=config.BUILDING_BOUNDARY_WEIGHT)
    weight=got['loss_weight'].numpy()
    assert config.BUILDING_BOUNDARY_MODE=='inner'
    assert np.all(weight[:,0]==0)
    assert np.all(weight[:,1]==1)
    assert np.any(weight[2:18,3:19]==2) and np.any(weight[2:18,3:19]==3)
    assert np.all(weight[:,23]==1)
    assert not np.any((got['building_boundary'] & got['road_mask'].bool()).numpy())


def test_continuous():
    p=np.zeros((7,12)); p[3,:]=1; r=astar_eval(p,(0,3),(11,3)); assert r['route_found'] and r['route_length']==11


def test_gap():
    p=np.zeros((7,12)); p[3,:]=1; p[3,6]=0; assert not astar_eval(p,(0,3),(11,3))['route_found']


def test_alternative():
    p=np.zeros((9,12)); p[4,:]=1; p[4,5]=0; p[2,3:9]=1; p[2:5,3]=1; p[2:5,8]=1
    r=astar_eval(p,(0,4),(11,4)); assert r['route_found'] and r['route_length']>11


def test_false_shortcut():
    p=np.zeros((9,12)); p[4,:]=.2; p[2,:]=1; p[2:5,0]=1; p[2:5,11]=1
    r=astar_eval(p,(0,4),(11,4)); assert r['route_found']; assert all(p[y,x]>=.5 for x,y in r['route_coordinates'])


def test_corner_blocked():
    assert not astar_eval(np.eye(2),(0,0),(1,1))['route_found']


def test_ignored_loss():
    from road_model import road_loss
    x=torch.zeros((1,2,2),requires_grad=True); target=torch.ones_like(x); valid=torch.zeros_like(x,dtype=torch.bool)
    loss,_,_=road_loss(x,target,valid); loss.backward(); assert loss==0 and not x.grad.any()
