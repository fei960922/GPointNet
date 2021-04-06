# 2021.03.01 

# Logistic
import tqdm 
import argparse 
import traceback 
import datetime
import json
import os 
import sys
import time
sys.path.append('.')
sys.path.append('./src')

# Main
import torch 
import pytorch_lightning as pl 
from utils import data_util
from utils import util_torch
import numpy as np
import src.network_torch as network_torch
torch.multiprocessing.set_sharing_strategy('file_system')

class GPointNet(pl.LightningModule):

    def __init__(self, config, energy_net="default"):

        super().__init__() 
        if type(config) is dict: 
            self.C = argparse.ArgumentParser()
            for k, v in config.items():
                setattr(self.C, k, v)
        else:
            self.C = config 
        self.energy_net = getattr(network_torch, "energy_point_" + self.C.net_type)(self.C) if energy_net == "default" else energy_net
        self.i_epoch = 0
        self.eval_process = []
        self.save_hyperparameters(config)
        self.evaluation = util_torch.async_evaluation(self.C) if self.C.do_evaluation else None
        self.syn_buffer = {}
        self.global_fault = 0
        self.langevin_noise_decay = 1

    # Langevin dynamic for point input 
    # inpainting supported 
    def langevin_dynamic_point(self, pcs_init, use_noise=True, inpaint_observe_pnt=-1, reconstruct=False):

        if self.C.activate_eval:
            self.energy_net.eval()
        with torch.enable_grad():
            u = pcs_init if reconstruct else torch.autograd.Variable(pcs_init.detach(), requires_grad=True)
            for i in range(self.C.sample_step):
                energy = self.energy_net(u)
                grad = torch.autograd.grad(energy.sum(), [u], retain_graph=True)[0]
                # grad[torch.isnan(grad)] = 0
                du = 0.5 * self.C.step_size * self.C.step_size * grad
                if self.C.ref_sigma != 0: 
                    du -= 0.5 * self.C.step_size * self.C.step_size * u / self.C.ref_sigma / self.C.ref_sigma
                if use_noise: 
                    noise_decay = max(0, (self.C.sample_step - i - 5) / (self.C.sample_step - 5)) if self.C.noise_decay else 1
                    du += noise_decay * self.langevin_noise_decay *self.C.step_size * torch.randn(u.shape, device=u.device)
                if inpaint_observe_pnt > 0:
                    du = torch.cat([torch.zeros((u.shape[0], inpaint_observe_pnt, u.shape[2]), device=u.device), du[:, inpaint_observe_pnt:]], 1) 
                u = torch.clamp(u + du, -self.C.langevin_clip, self.C.langevin_clip)   
        if self.C.activate_eval:      
            self.energy_net.train()
        return u

    def _reconstruction(self, reconstruct_pcs):

        u_pcs = torch.randn(reconstruct_pcs.shape, device=reconstruct_pcs.device, 
                    requires_grad=True) * self.C.ref_sigma
        for i in range(self.C.rec_step): 

            syn_pcs = self.langevin_dynamic_point(u_pcs, reconstruct=True, use_noise=False)
            error = (syn_pcs - reconstruct_pcs)**2
            grad = torch.autograd.grad(error.mean(), [u_pcs], retain_graph=True)[0]
            ratio = 0.9**i #(self.C.rec_step - i)**2 / self.C.rec_step**2
            u_pcs = u_pcs - self.C.rec_step_size * ratio * torch.sign(grad)
            # if i%20==0:
            #     print("%s-%d: %.4f" % (self.C.category, i, error.mean().cpu().data.numpy()*1000))
        
        return u_pcs, self.langevin_dynamic_point(u_pcs), error.mean().cpu().data.numpy()*1000

    
    def forward(self, pcs_init=None, reconstruct_pcs=None, batch_size=-1):

        if reconstruct_pcs is not None: 
            return self._reconstruction(reconstruct_pcs)

        if pcs_init is None: 
            shape = (self.C.batch_size if batch_size == -1 else batch_size, self.C.point_dim, self.C.num_point)
            pcs_init = torch.rand(shape, device=self.device) * 2 - 1 if self.C.ref_sigma == 0 \
            else torch.randn(shape, device=self.device) * self.C.ref_sigma
        return self.langevin_dynamic_point(pcs_init)

    def training_step(self, batch, batch_idx):

        if self.C.warm_start and batch_idx in self.syn_buffer: 
            pcs_init = self.syn_buffer[batch_idx]
        else:
            pcs_init = torch.rand(batch.shape, device=batch.device) * 2 - 1 if self.C.ref_sigma == 0 \
            else torch.randn(batch.shape, device=batch.device) * self.C.ref_sigma
        syn_pcs = self(pcs_init) 
        obs_res, syn_res = torch.mean(self.energy_net(batch)), torch.mean(self.energy_net(syn_pcs))
        train_loss = syn_res - obs_res
        var = syn_pcs.var()
        self.log('training/lr', self.scheduler.get_last_lr()[0])
        self.log('training/noise_decay', self.langevin_noise_decay)
        self.log('training/obs_res', obs_res)
        self.log('training/syn_res', syn_res)
        self.log('training/loss', train_loss)
        self.log('syn/var', var)
        self.log('syn/scale_max', syn_pcs.max())
        self.log('syn/scale_min', syn_pcs.min())

        if self.C.stable_check and var.cpu().data.numpy() > self.C.ref_sigma**2 + 0.1:

        # Activate stable check if var is bigger than threshold. 
        # Resample 10 times, if every one is bigger than threshold, load previous checkpoint. 
            print("Stable check activated : var = %.4f" % var.cpu().data.numpy())
            for i in range(10):
                if self.C.warm_start and batch_idx in self.syn_buffer: 
                    pcs_init = self.syn_buffer[batch_idx]
                else:
                    pcs_init = torch.rand(batch.shape, device=batch.device) * 2 - 1 if self.C.ref_sigma == 0 \
                    else torch.randn(batch.shape, device=batch.device) * self.C.ref_sigma
                syn_pcs = self(pcs_init) 
                var = syn_pcs.var().cpu().data.numpy()
                if var < self.C.ref_sigma**2 + 0.1:
                    print("Pass check with %d-th rerun, now var = %.4f" % (i+1, var))
                    obs_res, syn_res = torch.mean(self.energy_net(batch)), torch.mean(self.energy_net(syn_pcs))
                    train_loss = syn_res - obs_res
                    break 
            if i == 9: 
                last_checkpoint = self.current_step - (self.current_step % self.C.eval_step)
                self.global_fault += 1 
                if self.global_fault > 10: 
                    raise ValueError("[Iteration] Restore happened more than 10 times.")
                print("Failed after 10 rerun. Restore checkpoint %d." % last_checkpoint)
                self.load_state_dict(torch.load(self.C.output_dir + "/checkpoint_%d.ckpt" % last_checkpoint)["state_dict"])
                return {'loss': torch.zeros((1), device=batch.device, requires_grad=True), 'syn_pcs': syn_pcs}
        if self.C.warm_start:
            self.syn_buffer[batch_idx] = syn_pcs
        if abs(train_loss.cpu().data.numpy()) > 1e7:
            raise ValueError("[Iteration] train_loss larger than 1e7 found.")
        return {'loss': train_loss, 'syn_pcs': syn_pcs} 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.C.lr, betas=(self.C.beta1_des, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, self.C.lr_decay)
        return [optimizer], [{'scheduler':self.scheduler, 'interval': 'step'} ]

    def last_step(self):

        if self.evaluation == 2:
            for dic in self.evaluation.last_wait():
                for k, v in dic.items(): 
                    for name, scalar in v.items():
                        self.logger.experiment.add_scalar('syn/%s' % name, scalar, k)

class CheckpointEveryNSteps(pl.Callback):
    
    def __init__(self, every_n_step, output_dir):
        self.every_n_step = every_n_step
        self.output_dir = output_dir

    def on_batch_end(self, trainer, model):
        # Update langevin noise decay 
        current_step = trainer.global_step + 1
        if model.C.langevin_decay:
            model.langevin_noise_decay = max(0, (int(model.C.num_steps)*0.9 - current_step) / (int(model.C.num_steps)*0.9))
        if current_step % self.every_n_step == 0:
            samples = model().cpu().data.numpy()
            samples = np.swapaxes(samples, 1, 2) if samples.shape[1] < 10 else samples
            np.save('%s/des_syn_%d.npy' % (self.output_dir, current_step), samples)
            model.logger.experiment.add_figure("syn/synthesis", util_torch.visualize(samples, num_rows=5, num_cols=4, mode=0), current_step)
            model.logger.experiment.add_image("syn/black_syn", util_torch.visualize(samples, num_rows=5, num_cols=4, mode=1, output_filename='%s/des_syn_%d.png' % (self.output_dir, current_step)), current_step, dataformats='HW')
            if model.C.do_evaluation == 1:
                test_data = np.load("%s/%s_test.npy" %(model.C.data_path, model.C.category))[:model.C.test_size]
                if model.C.normalize == "per_shape": 
                    samples = samples[:model.C.test_size] + test_data.mean(axis=1, keepdims=True).repeat(samples.shape[1], axis=1)
                    samples = samples / np.abs(test_data).max(axis=1, keepdims=True).repeat(samples.shape[1], axis=1)
                dic = util_torch.quantitative_analysis(samples, test_data, model.C.batch_size, full=False)
                for name, scalar in dic.items():
                    model.logger.experiment.add_scalar('syn/%s' % name, scalar, current_step)
            elif model.C.do_evaluation == 2:
                dic = model.evaluation.add_evaluation(current_step)
                for k, v in dic.items(): 
                    for name, scalar in v.items():
                        model.logger.experiment.add_scalar('syn/%s' % name, scalar, k)
            trainer.save_checkpoint(self.output_dir + "/checkpoint_%d.ckpt" % current_step)

def parse_config():

    parser = argparse.ArgumentParser()

    # Point cloud related
    parser.add_argument('-hidden_size', type=list, default=[[64,128,256,512,1024], [512,256,64]])
    parser.add_argument('-net_type', type=str, default="default_medium")
    parser.add_argument('-num_point', type=int, default=2048)
    parser.add_argument('-point_dim', type=int, default=3)
    parser.add_argument('-argment_mode', type=int, default=0)
    parser.add_argument('-argment_noise', type=float, default=0.01)
    parser.add_argument('-random_sample', type=int, default=1)
    parser.add_argument('-visualize_mode', type=int, default=0)
    parser.add_argument('-learning_mode', type=int, default=0)
    parser.add_argument('-normalize', type=str, default="ebp")
    parser.add_argument('-batch_norm', type=str, default="ln", help='BatchNorm(bn) / LayerNorm(ln) / InstanceNorm(in) / None')
    parser.add_argument('-activate_eval', type=int, default=0)
     
    # EBM related
    parser.add_argument('-batch_size', type=int, default=128, help='')
    parser.add_argument('-lr', type=float, default=0.0005, help='')
    parser.add_argument('-lr_decay', type=float, default=0.998, help='')
    parser.add_argument('-beta1_des', type=float, default=0.9, help='')
    parser.add_argument('-sample_step', type=int, default=64, help='')
    parser.add_argument('-activation', type=str, default="ReLU", help='')
    parser.add_argument('-step_size', type=float, default=0.01, help='')
    parser.add_argument('-noise_decay', type=int, default=0, help='')
    parser.add_argument('-langevin_decay', type=int, default=0, help='')
    parser.add_argument('-ref_sigma', type=float, default=0.3, help='')
    parser.add_argument('-num_chain', type=float, default=1, help='')
    parser.add_argument('-langevin_clip', type=float, default=1, help='')
    parser.add_argument('-warm_start', type=int, default=0)

    # Logistic related
    parser.add_argument('-num_steps', type=int, default=2000)
    parser.add_argument('-stable_check', type=int, default=1)
    parser.add_argument('-do_evaluation', type=int, default=1)
    parser.add_argument('-seed', type=int, default=666, help='')
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('-eval_step', type=int, default=50)
    parser.add_argument('-drop_last', action='store_true')
    parser.add_argument('-mode', type=str, default="train", help='')
    parser.add_argument('-data_size', type=int, default=10000)
    parser.add_argument('-test_size', type=int, default=16)
    parser.add_argument('-debug', type=int, default=99, help='')
    parser.add_argument('-cuda', type=str, default="-1", help='')
    parser.add_argument('-data_path', type=str, default="data")
    parser.add_argument('-checkpoint_path', type=str, default="")
    parser.add_argument('-category', type=str, default="chair")
    parser.add_argument('-output_dir', type=str, default="default")
    parser.add_argument('-fp16', type=str, default="None", help='/O1/O2')
    
    return parser.parse_args()

def main(opt):

    if opt.category.split("_")[0] == "Flow":
        train_data = data_util.ShapeNet15kPointClouds(
            categories=opt.category.split("_")[1:], split='train',
            tr_sample_size=opt.num_point,
            normalize_per_shape=1,
            normalize_std_per_axis=1,
            root_dir=opt.data_path, 
            random_subsample=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, 
        shuffle=opt.shuffle, num_workers=torch.cuda.device_count() * 4)
    else: 
        train_data = util_torch.PointCloudDataSet(opt)
        data_collator = util_torch.PointCloudDataCollator(opt)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, drop_last=opt.drop_last, 
        shuffle=opt.shuffle, collate_fn = data_collator, num_workers=torch.cuda.device_count() * 4)
    util_torch.save_evaluation_ref(train_loader, opt.category)
    if opt.cuda == "-1":
        opt.cuda = 1
    elif opt.cuda == "-2":
        opt.cuda = None
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    model = GPointNet(opt)
    print(sum(p.numel() for p in model.parameters()))
    if opt.mode == "train":
        trainer = pl.Trainer(gpus=opt.cuda, 
                            resume_from_checkpoint=opt.checkpoint_path,
                            max_steps=opt.num_steps, 
                            default_root_dir=opt.output_dir,
                            amp_level=opt.fp16, precision=32 if opt.fp16 == "None" else 16,
                            log_every_n_steps=10,
                            flush_logs_every_n_steps=10,
                            # accelerator="ddp",
                            callbacks=[CheckpointEveryNSteps(opt.eval_step, opt.output_dir)]
                            )
        if trainer.fit(model, train_loader) == 1:
            model.last_step()
    elif opt.mode == "synthesis":
        assert len(opt.checkpoint_path) > 0 
        model.load_state_dict(torch.load(opt.checkpoint_path))
        syn_pcs = model()
    elif opt.mode == "representation":
        model.load_state_dict(torch.load(opt.checkpoint_path))
        train_dataset = torch.utils.data.TensorDataset(np.load("data/%s_train.npy" % opt.category))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
        train_feature = [model.energy_net(batch, out_local=True) for batch in train_loader]
        train_feature = torch.cat(train_feature).cpu().data.numpy() 
        np.save(opt.output_dir + "train_feature_mean.npy", np.mean(train_feature, -1))
        np.save(opt.output_dir + "train_feature_max.npy", np.max(train_feature, -1))

        test_dataset = torch.utils.data.TensorDataset(np.load("data/%s_test.npy" % opt.category))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
        test_feature = [model.energy_net(batch, out_local=True) for batch in test_loader]
        test_feature = torch.cat(test_feature).cpu().data.numpy() 
        np.save(opt.output_dir + "test_feature_mean.npy", np.mean(test_feature, -1))
        np.save(opt.output_dir + "test_feature_max.npy", np.max(test_feature, -1))

    elif opt.mode == "test_reconstruction":
        model.test_reconstruction(train_data)

def pre_process(opt):

    if opt.output_dir == "default": 
        opt.output_dir = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    opt.output_dir = os.path.join('output', "pytorch", "_".join([opt.category, opt.net_type, opt.output_dir]))
    if opt.net_type == "default_residual":
        opt.net_type = "default"
        opt.hidden_size = [[64,64,64,128,128,128,256,256,256,512,1024], [512,256,256,256,64,64,64]]
    elif opt.net_type == "default_big":
        opt.net_type = "default"
        opt.hidden_size = [[64,128,256,512,1024,2048], [1024,512,256,64]]
    elif opt.net_type == "default_small":
        opt.net_type = "default"
        opt.hidden_size = [[64,64,64,1024], [512,256,64]]
    elif opt.net_type == "default_medium":
        opt.net_type = "default"
        opt.hidden_size = [[64,64,128,256,1024], [512,256,128,64]]
    opt.swap_axis = True
    if len(opt.checkpoint_path) == 0:
        opt.checkpoint_path = None 
    opt.device = "cuda:%s" % opt.cuda if opt.cuda!="" else "cpu"
    opt.shuffle = not opt.warm_start
    return opt

if __name__ == '__main__':

    opt = parse_config()
    opt = pre_process(opt)
    try: 
        main(opt)
        data_util.PushNotification().post_text("[GPointNet] %s finished!" % opt.output_dir)
    except Exception as e:
        error = traceback.format_exc().split('\n')
        print(traceback.format_exc()) 
        data_util.PushNotification().post_text("[GPointNet] FAILED in %s! Error message: %s" % (opt.output_dir,"\n".join(error[-2:])))
