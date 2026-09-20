import sys,json,time,argparse,random,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from utils import load_config
from dataset import LoveDARoad,LOVEDA_CLASSES,ROAD_CLASS_ID,BUILDING_CLASS_ID,IGNORE_CLASS_ID
from road_model import RoadOnlyMaGRoad,road_loss,hard_negative_road_loss,gradient_audit
from visuals import save_visual,target_inference


def gpu_stats():
    return {'gpu':torch.cuda.get_device_name(),'allocated_mib':torch.cuda.memory_allocated()/2**20,'reserved_mib':torch.cuda.memory_reserved()/2**20,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}


def metrics(c):
    tp,fp,fn,tn=map(float,c)
    return {'road_iou':tp/max(tp+fp+fn,1),'road_precision':tp/max(tp+fp,1),'road_recall':tp/max(tp+fn,1),'road_f1':2*tp/max(2*tp+fp+fn,1)}


def hn_dataset_config(config):
    if not config.get('USE_HARD_NEGATIVE',False): return None
    return {
        'boundary_width':int(config.BUILDING_BOUNDARY_WIDTH),
        'normal_weight':float(config.NORMAL_NEGATIVE_WEIGHT),
        'building_weight':float(config.BUILDING_NEGATIVE_WEIGHT),
        'boundary_weight':float(config.BUILDING_BOUNDARY_WEIGHT),
        'positive_weight':1.,
    }


class HardNegativeEpochCheckpoint(pl.Callback):
    """Save auditable epoch checkpoints and a Recall-eligible candidate."""
    def __init__(self,output):
        self.output=Path(output)/'checkpoints'; self.output.mkdir(parents=True,exist_ok=True)
        self.best_score=None

    def on_validation_end(self,trainer,pl_module):
        if trainer.sanity_checking or not pl_module.validation_history: return
        result=pl_module.validation_history[-1]; epoch=int(result['epoch'])
        epoch_path=self.output/f'epoch_{epoch}.ckpt'
        trainer.save_checkpoint(epoch_path)
        shutil.copy2(epoch_path,self.output/'last.ckpt')
        if not result['recall_guardrail_passed']: return
        score=(float(result['boundary_fp']),float(result['building_fp']),-float(result['road_iou']))
        if self.best_score is None or score<self.best_score:
            self.best_score=score; shutil.copy2(epoch_path,self.output/'best_valid_candidate.ckpt')
            (self.output/'best_valid_candidate.json').write_text(json.dumps({'epoch':epoch,'selection_priority':['boundary_fp','building_fp','road_iou'],'score':score,'metrics':result},indent=2))


class TrainerModel(RoadOnlyMaGRoad):
    def __init__(self,config,audit=False):
        super().__init__(config); self.audit=audit; self.audit_rows=[]; self.started=time.time()
        self.out=Path(config.OUTPUT_DIR); self.out.mkdir(parents=True,exist_ok=True)
        self.validation_history=[]; self.training_losses=[]

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.optimizer_groups(),lr=self.config.BASE_LR,weight_decay=self.config.WEIGHT_DECAY)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[max(1,int(self.config.TRAIN_EPOCHS*.8))],gamma=.1)
        return {'optimizer':opt,'lr_scheduler':scheduler}

    def transfer_batch_to_device(self,batch,device,dataloader_idx):
        return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

    def training_step(self,batch,batch_idx):
        logits=self.forward_road_only(batch['rgb'])
        if self.config.get('USE_HARD_NEGATIVE',False):
            loss,bce,dice,diagnostics=hard_negative_road_loss(logits,batch['road_mask'],batch['valid'],batch['loss_weight'],batch['building_mask'],batch['building_boundary'],self.config.BCE_POS_WEIGHT,self.config.DICE_WEIGHT)
        else:
            loss,bce,dice=road_loss(logits,batch['road_mask'],batch['valid'],self.config.BCE_POS_WEIGHT,self.config.DICE_WEIGHT); diagnostics={}
        if not torch.isfinite(loss): raise RuntimeError('Non-finite road loss')
        for name,value in [('loss_total',loss),('loss_bce',bce),('loss_dice',dice),*diagnostics.items()]: self.log('train/'+name,value,on_step=True,on_epoch=True,batch_size=len(logits))
        self.training_losses.append({'epoch':self.current_epoch,'batch':batch_idx,'global_step':self.global_step,'loss_total':float(loss.detach()),'loss_bce':float(bce.detach()),'loss_dice':float(dice.detach()),**{k:float(v.detach()) for k,v in diagnostics.items()}})
        if batch_idx%100==0: (self.out/'training_losses.json').write_text(json.dumps(self.training_losses,indent=2))
        counts=self.confusion(logits.float().sigmoid(),batch['road_mask'],batch['valid'],.5)
        for name,value in metrics(counts).items(): self.log('train/'+name,value,on_step=True,on_epoch=False,batch_size=len(logits))
        return loss

    def on_train_epoch_end(self):
        # Road-only train metrics already logged; do not invoke unused inherited topology metrics.
        (self.out/'training_losses.json').write_text(json.dumps(self.training_losses,indent=2))

    def on_after_backward(self):
        # FP16 grads remain scaled here; norm audit checks presence/nonzero, not magnitude comparisons.
        if self.audit:
            row=gradient_audit(self); row['global_step']=self.global_step; self.audit_rows.append(row)
            (self.out/'gradient_audit.json').write_text(json.dumps(self.audit_rows,indent=2))
        if self.global_step==0:
            print('First backward VRAM:',json.dumps(gpu_stats())); (self.out/'first_iteration_vram.json').write_text(json.dumps(gpu_stats(),indent=2))

    @staticmethod
    def confusion(prob,gt,valid,threshold):
        pred=prob>=threshold; gt=gt>.5; valid=valid.bool()
        return torch.stack([(pred&gt&valid).sum(),(pred&~gt&valid).sum(),(~pred&gt&valid).sum(),(~pred&~gt&valid).sum()]).detach().cpu()

    def on_validation_epoch_start(self):
        self.counts={round(t,2):torch.zeros(4,dtype=torch.int64) for t in np.arange(.05,1.,.05)}; self.loss_sum=np.zeros(3); self.val_images=0
        self.hn_counts={'building_pixels':0,'building_predicted_road':0,'boundary_pixels':0,'boundary_predicted_road':0,'building_probability_sum':0.,'boundary_probability_sum':0.}
        self.hn_hist={'building':np.zeros(1001,dtype=np.int64),'boundary':np.zeros(1001,dtype=np.int64)}

    def validation_step(self,batch,batch_idx):
        logits=self.forward_road_only(batch['rgb'])
        if self.config.get('USE_HARD_NEGATIVE',False):
            loss,bce,dice,_=hard_negative_road_loss(logits,batch['road_mask'],batch['valid'],batch['loss_weight'],batch['building_mask'],batch['building_boundary'],self.config.BCE_POS_WEIGHT,self.config.DICE_WEIGHT)
        else: loss,bce,dice=road_loss(logits,batch['road_mask'],batch['valid'],self.config.BCE_POS_WEIGHT,self.config.DICE_WEIGHT)
        prob=logits.float().sigmoid()
        for t in self.counts: self.counts[t]+=self.confusion(prob,batch['road_mask'],batch['valid'],t)
        if 'building_mask' in batch:
            for key,mask in [('building',batch['building_mask'].bool()),('boundary',batch['building_boundary'].bool())]:
                values=prob[mask].detach().cpu().numpy()
                self.hn_counts[key+'_pixels']+=int(values.size); self.hn_counts[key+'_predicted_road']+=int(np.count_nonzero(values>=.5)); self.hn_counts[key+'_probability_sum']+=float(values.sum(dtype=np.float64))
                self.hn_hist[key]+=np.bincount(np.minimum((values*1000).astype(np.int64),1000),minlength=1001)
        self.loss_sum+=np.array([float(loss),float(bce),float(dice)])*len(logits); self.val_images+=len(logits)
        if batch_idx<self.config.VAL_VISUAL_COUNT and not self.trainer.sanity_checking:
            for i,identity in enumerate(batch['id']): save_visual(batch['rgb'][i].cpu().numpy(),prob[i].cpu().numpy(),self.out/'validation'/f'epoch_{self.current_epoch:03d}'/identity,.5,batch['road_mask'][i].cpu().numpy(),batch['valid'][i].cpu().numpy())

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking: return
        curve={str(t):metrics(c) for t,c in self.counts.items()}; best=max(self.counts,key=lambda t:(metrics(self.counts[t])['road_f1'],-abs(t-.5)))
        def percentile(hist,q):
            if hist.sum()==0:return 0.
            return float(np.searchsorted(np.cumsum(hist),q*hist.sum(),side='left')/1000.)
        hn={}
        for key in ('building','boundary'):
            count=max(self.hn_counts[key+'_pixels'],1)
            hn[key+'_fp']=self.hn_counts[key+'_predicted_road']/count
            hn['mean_'+key+'_probability']=self.hn_counts[key+'_probability_sum']/count
            hn['p50_'+key+'_probability']=percentile(self.hn_hist[key],.50); hn['p90_'+key+'_probability']=percentile(self.hn_hist[key],.90); hn['p95_'+key+'_probability']=percentile(self.hn_hist[key],.95)
        result={'epoch':self.current_epoch,'images':self.val_images,**metrics(self.counts[.5]),**hn,'loss':float(self.loss_sum[0]/max(self.val_images,1)),'bce_loss':float(self.loss_sum[1]/max(self.val_images,1)),'dice_loss':float(self.loss_sum[2]/max(self.val_images,1)),'road_threshold':best,'threshold_curve':curve,'seconds_since_start':time.time()-self.started,'vram':gpu_stats()}
        result['road_recall_guardrail']=float(self.config.get('ROAD_RECALL_BASELINE',.677946)-self.config.get('ROAD_RECALL_TOLERANCE',.03)); result['recall_guardrail_passed']=result['road_recall']>=result['road_recall_guardrail']
        result['selection_boundary_fp']=result['boundary_fp'] if result['recall_guardrail_passed'] else 10.+result['boundary_fp']
        for name in ('road_iou','road_precision','road_recall','road_f1','loss','bce_loss','dice_loss'): self.log('val/'+name,result[name],prog_bar=True)
        for name in ('building_fp','boundary_fp','mean_building_probability','mean_boundary_probability','selection_boundary_fp'): self.log('val/'+name,result[name],prog_bar=name in ('building_fp','boundary_fp'))
        self.validation_history.append(result); (self.out/'validation_history.json').write_text(json.dumps(self.validation_history,indent=2)); (self.out/'road_threshold.json').write_text(json.dumps({'road_threshold':best,'source':'LoveDA validation F1','epoch':self.current_epoch},indent=2)); print('Validation:',json.dumps(result))
        adapter=self.adapter_state(); adapter.update(epoch=self.current_epoch,metrics=result); torch.save(adapter,self.out/f'adapter_epoch_{self.current_epoch:03d}.pt')
        if self.config.TARGET_DOMAIN_INFER_DIR: target_inference(self,self.config.TARGET_DOMAIN_INFER_DIR,self.out/'target_domain'/f'epoch_{self.current_epoch:03d}',.5,best)
        (self.out/'run_status.json').write_text(json.dumps({'status':'validation_finished','epoch':self.current_epoch,'time_seconds':time.time()-self.started,'vram':gpu_stats()},indent=2))
        if not result['recall_guardrail_passed']:
            print('WARNING: ROAD RECALL REGRESSION',result['road_recall'],'<',result['road_recall_guardrail'])
            previous=self.validation_history[-2:] if len(self.validation_history)>=2 else []
            if len(previous)==2 and all(not row.get('recall_guardrail_passed',True) for row in previous):
                print('Early stop: recall guardrail failed for two consecutive epochs'); self.trainer.should_stop=True

    def on_save_checkpoint(self,checkpoint):
        checkpoint['experiment_config']=self.config.to_dict(); checkpoint['base_reference']=self.base_reference; checkpoint['parameter_statistics']=self.parameter_statistics; checkpoint['validation_history']=self.validation_history; checkpoint['training_losses']=self.training_losses
        checkpoint['initial_checkpoint_reference']=self.initial_checkpoint_reference

    def on_load_checkpoint(self,checkpoint):
        if checkpoint.get('base_reference',{}).get('sha256')!=self.base_reference['sha256']: raise ValueError('Resume base checkpoint SHA mismatch')
        if checkpoint['experiment_config']['TRAIN_MODE']!=self.config.TRAIN_MODE: raise ValueError('Resume train mode mismatch')
        self.validation_history=checkpoint.get('validation_history',[]); self.training_losses=checkpoint.get('training_losses',[])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--resume'); p.add_argument('--gradient-audit',action='store_true'); p.add_argument('--max-steps',type=int,default=-1); p.add_argument('--max-epochs',type=int); p.add_argument('--check-val-every-n-epoch',type=int,default=1); p.add_argument('--limit-train',type=int); p.add_argument('--limit-val',type=int); p.add_argument('--eval-only',action='store_true'); p.add_argument('--eval_target_domain',action='store_true'); p.add_argument('--output-dir'); a=p.parse_args()
    cfg=load_config(a.config)
    if a.output_dir: cfg.OUTPUT_DIR=a.output_dir
    pl.seed_everything(cfg.SEED,workers=True); torch.set_float32_matmul_precision('high'); torch.cuda.reset_peak_memory_stats(); print('Startup VRAM:',json.dumps(gpu_stats()))
    model=TrainerModel(cfg,a.gradient_audit)
    (model.out/'config_used.json').write_text(json.dumps(cfg.to_dict(),indent=2)); (model.out/'parameter_statistics.json').write_text(json.dumps(model.parameter_statistics,indent=2)); (model.out/'base_reference.json').write_text(json.dumps(model.base_reference,indent=2))
    if a.eval_target_domain:
        if a.resume:
            state=torch.load(a.resume,map_location='cpu',weights_only=False); model.load_state_dict(state['state_dict'],strict=True)
        model.cuda(); target_inference(model,cfg.TARGET_DOMAIN_INFER_DIR,model.out/'target_domain'/'baseline'); return
    print('LoveDA class mapping:',json.dumps(LOVEDA_CLASSES)); print('Class IDs:',json.dumps({'IGNORE_CLASS_ID':IGNORE_CLASS_ID,'BUILDING_CLASS_ID':BUILDING_CLASS_ID,'ROAD_CLASS_ID':ROAD_CLASS_ID}))
    hn=hn_dataset_config(cfg)
    train=LoveDARoad(cfg.LOVEDA_ROOT,'train',cfg.LOVEDA_DOMAIN,cfg.LOVEDA_AUGMENTATION,a.limit_train,cfg.get('LOVEDA_TRAIN_RGB_ROOT'),hn); val=LoveDARoad(cfg.LOVEDA_ROOT,'val',cfg.LOVEDA_DOMAIN,False,a.limit_val,None,hn)
    workers=cfg.DATA_WORKER_NUM
    kwargs={'batch_size':cfg.BATCH_SIZE,'num_workers':workers,'pin_memory':True,'persistent_workers':workers>0}
    dl=DataLoader(train,shuffle=True,**kwargs); vl=DataLoader(val,shuffle=False,**kwargs)
    callbacks=[ModelCheckpoint(dirpath=model.out/'checkpoints',filename='best_road_iou',monitor='val/road_iou',mode='max',save_last=True,enable_version_counter=False),ModelCheckpoint(dirpath=model.out/'checkpoints',filename='best_road_recall',monitor='val/road_recall',mode='max',enable_version_counter=False),LearningRateMonitor(logging_interval='step')]
    if cfg.get('USE_HARD_NEGATIVE',False): callbacks += [ModelCheckpoint(dirpath=model.out/'checkpoints',filename='best_boundary_fp',monitor='val/boundary_fp',mode='min',enable_version_counter=False),HardNegativeEpochCheckpoint(model.out)]
    if cfg.EARLY_STOPPING_PATIENCE: callbacks.append(EarlyStopping(monitor='val/road_iou',mode='max',patience=cfg.EARLY_STOPPING_PATIENCE))
    trainer=pl.Trainer(accelerator='gpu',devices=1,strategy='auto',max_epochs=a.max_epochs or cfg.TRAIN_EPOCHS,max_steps=a.max_steps,check_val_every_n_epoch=a.check_val_every_n_epoch,precision=cfg.PRECISION,accumulate_grad_batches=cfg.ACCUMULATE_GRAD_BATCHES,gradient_clip_val=1.,callbacks=callbacks,logger=TensorBoardLogger(str(model.out/'tensorboard'),name='road_only'),log_every_n_steps=1,num_sanity_val_steps=0,enable_progress_bar=False)
    try:
        if a.eval_only:
            # Lightning 2.5 defaults to weights_only=True when restoring a
            # checkpoint.  Our auditable checkpoints contain numpy metadata,
            # so load the trusted state dict explicitly and validate without
            # asking Lightning to unpickle the full training state.
            if not a.resume: raise ValueError('--resume is required with --eval-only')
            state=torch.load(a.resume,map_location='cpu',weights_only=False)
            model.load_state_dict(state['state_dict'],strict=True)
            trainer.validate(model,vl,ckpt_path=None)
        else: trainer.fit(model,dl,vl,ckpt_path=a.resume)
    except torch.cuda.OutOfMemoryError:
        stats=gpu_stats(); (model.out/'oom.json').write_text(json.dumps(stats,indent=2)); print('OOM VRAM:',json.dumps(stats)); raise
    (model.out/'run_status.json').write_text(json.dumps({'status':'complete','time_seconds':time.time()-model.started,'vram':gpu_stats()},indent=2))
    torch.save({**model.adapter_state(),'epoch':model.validation_history[-1]['epoch'] if model.validation_history else model.current_epoch},model.out/'adapter_last.pt')

if __name__=='__main__': main()
