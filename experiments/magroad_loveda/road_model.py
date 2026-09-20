import sys, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from model import MaGRoad


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def is_lora(name):
    return name.startswith('image_encoder.') and '.qkv.linear_' in name


def category(name):
    if is_lora(name): return 'lora'
    if name.startswith('image_encoder.'): return 'encoder_base'
    if name.startswith('map_decoder.'): return 'map_decoder'
    if name.startswith('topo_net.'): return 'toponet'
    return 'other'


def canonical(state):
    result={}
    for key,value in state.items():
        while key.startswith(('module.','model.','net.')): key=key.split('.',1)[1]
        if key in result: raise ValueError(f'Checkpoint key collision {key}')
        result[key]=value
    return result


def load_base(model,path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    state=canonical(ckpt.get('state_dict',ckpt))
    # Official WildRoad loss buffer is not model weights; new road loss uses configured weight.
    ignored_loss_buffers={k:state.pop(k) for k in ('mask_criterion.pos_weight',) if k in state and k not in model.state_dict()}
    if ignored_loss_buffers: print('Explicitly excluded training-loss buffers:',list(ignored_loss_buffers))
    result=model.load_state_dict(state,strict=False)
    missing=list(result.missing_keys); unexpected=list(result.unexpected_keys)
    print(json.dumps({'loaded_keys':len(state)-len(unexpected),'missing_keys':missing,'unexpected_keys':unexpected},indent=2))
    if any(not is_lora(k) for k in missing) or unexpected:
        raise ValueError('Base checkpoint must load completely, except new zero-initialized LoRA')
    if any(is_lora(k) for k in state): raise ValueError('Expected non-adapted WildRoad base')
    if tuple(model.image_encoder.pos_embed.shape)!=(1,64,64,768): raise ValueError('1024 ViT-B position embedding required')
    return {'path':str(Path(path).resolve()),'sha256':sha256(path),'loaded_keys':len(state),'missing_keys':missing,'unexpected_keys':unexpected}


def freeze(model,config):
    mode=config['TRAIN_MODE']; decoder=mode=='loveda_lora_encoder_decoder'
    if mode not in ('loveda_lora_encoder','loveda_lora_encoder_decoder'): raise ValueError(mode)
    expected={'TRAIN_ENCODER_BASE':False,'TRAIN_ENCODER_LORA':True,'TRAIN_MAP_DECODER':decoder,'TRAIN_TOPONET':False}
    for k,v in expected.items():
        if config.get(k,v)!=v: raise ValueError(f'{mode}: incompatible {k}')
    stats={g:{'total':0,'trainable':0} for g in ('encoder_base','lora','map_decoder','toponet','other')}
    for n,p in model.named_parameters():
        g=category(n); p.requires_grad_(g=='lora' or (g=='map_decoder' and decoder)); stats[g]['total']+=p.numel(); stats[g]['trainable']+=p.numel() if p.requires_grad else 0
    print('====== Trainable Parameters ======'); print(json.dumps(stats,indent=2))
    assert stats['lora']['trainable']>0 and stats['encoder_base']['trainable']==stats['toponet']['trainable']==0
    return stats


def gradient_audit(model):
    result={g:{'grad_parameters':0,'squared_norm':0.} for g in ('encoder_base','lora','map_decoder','toponet','other')}
    for n,p in model.named_parameters():
        if p.grad is not None:
            g=category(n); result[g]['grad_parameters']+=1; result[g]['squared_norm']+=float(p.grad.detach().float().square().sum())
    for v in result.values(): v['norm']=v.pop('squared_norm')**.5
    print('Gradient audit:',json.dumps(result))
    assert result['lora']['norm']>0 and all(result[g]['grad_parameters']==0 for g in ('encoder_base','toponet','other'))
    if model.config.TRAIN_MODE=='loveda_lora_encoder': assert result['map_decoder']['grad_parameters']==0
    else: assert result['map_decoder']['norm']>0
    return result


def road_loss(logits,target,valid,pos_weight=1.,dice_weight=1.):
    logits=logits.float(); target=target.float(); valid=valid.float()
    # Ignore applies to both BCE and Dice, including no-data/invalid classes.
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=logits.new_tensor(pos_weight))*valid).sum()/valid.sum().clamp_min(1)
    probability=logits.sigmoid()*valid; target=target*valid
    dims=(1,2)
    dice=(1-(2*(probability*target).sum(dims)+1)/(probability.sum(dims)+target.sum(dims)+1)).mean()
    return bce+dice_weight*dice,bce,dice


def hard_negative_road_loss(logits,target,valid,loss_weight,building,boundary,pos_weight=1.,dice_weight=1.):
    logits=logits.float(); target=target.float(); valid=valid.bool()
    pixel_bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=logits.new_tensor(pos_weight))
    weighted_bce=(pixel_bce*loss_weight.float()*valid.float()).sum()/valid.sum().clamp_min(1)
    probability=logits.sigmoid()*valid.float(); valid_target=target*valid.float(); dims=(1,2)
    dice=(1-(2*(probability*valid_target).sum(dims)+1)/(probability.sum(dims)+valid_target.sum(dims)+1)).mean()
    regions={
        'road_positive_bce': valid & (target>.5),
        'normal_background_bce': valid & (target<=.5) & ~building.bool(),
        'building_bce': valid & building.bool(),
        'building_boundary_bce': valid & boundary.bool(),
    }
    diagnostics={}
    for name,mask in regions.items():
        diagnostics[name]=(pixel_bce[mask].mean() if mask.any() else pixel_bce.new_zeros(()))
    return weighted_bce+dice_weight*dice,weighted_bce,dice,diagnostics


class RoadOnlyMaGRoad(MaGRoad):
    def __init__(self,config):
        if config.PATCH_SIZE!=1024 or config.SAM_VERSION!='vit_b' or config.USE_SAM_DECODER or config.NO_SAM: raise ValueError('Requires native 1024 SAM ViT-B naive map decoder')
        config.MAGROAD_INIT_ONLY=True
        super().__init__(config)
        self.base_reference=load_base(self,config.PRETRAINED_MAGROAD_CKPT)
        self.parameter_statistics=freeze(self,config)
        self.initial_checkpoint_reference=None
        if config.get('INIT_CHECKPOINT'):
            self.initial_checkpoint_reference=self.load_initial_checkpoint(config.INIT_CHECKPOINT)
        self.topo_net.register_forward_pre_hook(self._topology_forbidden)

    def load_initial_checkpoint(self,path):
        raw=torch.load(path,map_location='cpu',weights_only=False)
        state=canonical(raw.get('state_dict',raw))
        result=self.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys: raise ValueError('Initial checkpoint strict load failed')
        experiment=raw.get('experiment_config',{})
        history=raw.get('validation_history',[])
        best_epoch=max(history,key=lambda row:row.get('road_iou',-1)).get('epoch') if history else None
        if raw.get('epoch')!=8 or best_epoch!=8 or experiment.get('TRAIN_MODE')!='loveda_lora_encoder' or experiment.get('LORA_RANK')!=4:
            raise ValueError('INIT_CHECKPOINT is not LoveDA Encoder-LoRA A best epoch 8 rank 4')
        reference={'path':str(Path(path).resolve()),'sha256':sha256(path),'epoch':raw.get('epoch'),'global_step':raw.get('global_step'),'best_road_iou_epoch':best_epoch,'train_mode':experiment.get('TRAIN_MODE'),'lora_rank':experiment.get('LORA_RANK')}
        print('Initial checkpoint:',json.dumps(reference,indent=2))
        return reference

    @staticmethod
    def _topology_forbidden(*args): raise RuntimeError('Topology forward forbidden in road-only adaptation')

    def train(self,mode=True):
        super().train(mode)
        self.topo_net.eval()
        if self.config.TRAIN_MODE=='loveda_lora_encoder': self.map_decoder.eval()
        return self

    def forward_road_only(self,rgb):
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(1024,1024,3): raise ValueError('Expected float RGB BHWC 0..255, native 1024')
        x=(rgb.permute(0,3,1,2)-self.pixel_mean)/self.pixel_std
        if self.config.get('GRADIENT_CHECKPOINTING', False) and self.training and torch.is_grad_enabled():
            # Non-reentrant checkpointing supports frozen inputs with trainable LoRA inside.
            x=self.image_encoder.patch_embed(x)
            if self.image_encoder.pos_embed is not None: x=x+self.image_encoder.pos_embed
            for block in self.image_encoder.blocks: x=checkpoint(block,x,use_reentrant=False)
            embeddings=self.image_encoder.neck(x.permute(0,3,1,2))
        else:
            embeddings=self.image_encoder(x)
        # Native map channel 0=keypoint, 1=road; never supervise keypoint.
        return self.map_decoder(embeddings)[:,1]

    def optimizer_groups(self):
        groups=[]
        for group,lr in [('lora',self.config.LORA_LR),('map_decoder',self.config.DECODER_LR)]:
            params=[p for n,p in self.named_parameters() if category(n)==group and p.requires_grad]
            if params: groups.append({'params':params,'lr':lr,'name':group})
        included={id(p) for g in groups for p in g['params']}; expected={id(p) for p in self.parameters() if p.requires_grad}
        assert included==expected
        return groups

    def adapter_state(self):
        return {'adapter_state':{n:p.detach().cpu() for n,p in self.state_dict().items() if is_lora(n) or (n.startswith('map_decoder.') and self.config.TRAIN_MAP_DECODER)},'base_reference':self.base_reference,'config':self.config.to_dict(),'parameter_statistics':self.parameter_statistics}

    def load_adapter(self,path):
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['base_reference']['sha256']!=self.base_reference['sha256']: raise ValueError('Adapter base SHA mismatch')
        state=data['adapter_state']; expected=set(self.adapter_state()['adapter_state'])
        if set(state)!=expected: raise ValueError('Adapter keys do not match experiment mode')
        self.load_state_dict(state,strict=False)
