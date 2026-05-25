
import os
import time
import copy
import numpy as np
import torch
import torchaudio
from glob import glob
import re
import hydra
import wandb
import omegaconf
from tqdm import tqdm

import signal
import sys
import json

from utils.torch_utils import training_stats
from utils.torch_utils import misc
import utils.log as utils_logging
import utils.training_utils as t_utils

#----------------------------------------------------------------------------

class Trainer():
    def __init__(
        self,
        args=None,
        dset=None,
        network=None,
        diff_params=None,
        tester=None,
        device='cpu',
        is_main=True,
        ddp=False,
        train_sampler=None,
    ):
        assert args is not None, "args dictionary is None"
        self.args=args

        self.ckpt_dir = os.path.join(self.args.model_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.last_time_ckpt = time.time()
        self.time_ckpt_interval =  4 * 60 * 60  # 4 hours
        self.received_sigterm = False

        signal.signal(signal.SIGTERM, self.handle_sigterm)

        self.is_main = is_main
        self.ddp = ddp
        self.train_sampler = train_sampler


        assert dset is not None, "dset is None"
        self.dset=dset
        self.train_iter = iter(self.dset)

        assert network is not None, "network is None"
        self.network=network

        assert diff_params is not None, "diff_params is None"
        self.diff_params=diff_params

        assert device is not None, "device is None"
        self.device=device

        # self.tester = tester
        # self.tester.use_wandb = False # We do not want to interfere with the training wandb, as we do the logging in Trainer() and not in Tester()
        self.tester = tester

        if self.tester is not None:
            self.tester.use_wandb = False
            
        self.optimizer = hydra.utils.instantiate(args.exp.optimizer, params=network.parameters())
        
        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)
        # Torch settings
        torch.manual_seed(self.args.exp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        self.total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print("total_params: ",self.total_params/1e6, "M")
        
        # Checkpoint Resuming
        self.latest_checkpoint = None
        resuming = False
        if self.args.exp.resume:
            if self.args.exp.resume_checkpoint != "None":
                resuming = self.resume_from_checkpoint(checkpoint_path=self.args.exp.resume_checkpoint)
            else:
                resuming = self.resume_latest_if_exists()
            if not resuming:
                print("Could not resume from checkpoint")
                print("training from scratch")
            else:
                print("Resuming from iteration {}".format(self.it))
        if not resuming:
            self.it = 0
            self.latest_checkpoint = None
            if tester is not None:
                self.tester.it = 0

        # Model Summary
        if self.args.logging.print_model_summary:
            with torch.no_grad():
                audio = torch.zeros([args.exp.batch_size,args.exp.audio_len], device=device).unsqueeze(1)
                sigma = torch.ones([args.exp.batch_size], device=device)
                misc.print_module_summary(self.network, [audio, sigma ], max_nesting=2)

        # Logger Setup
        if self.args.logging.log and self.is_main:
            self.setup_wandb()
            self.setup_logging_variables()

        # Profiler
        self.profiler, self.profile, self.profile_total_steps = t_utils.profile(self.args.logging)


    def handle_sigterm(self, signum, frame):
        print("SIGTERM received, saving checkpoint...", flush=True)
        self.received_sigterm = True

        if self.is_main:
            self.save_checkpoint(tag="preempt")

        if self.ddp:
            torch.distributed.destroy_process_group()

        sys.exit(0)

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        config["total_params"]=self.total_params
        self.wandb_run=wandb.init(project=self.args.logging.wandb.project, config=config, dir=self.args.model_dir)
        wandb.watch(self.network, log="all", log_freq=self.args.logging.heavy_log_interval) #wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name = os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id #adding the experiment number to the run name, bery important, I hope this does not crash

    def setup_logging_variables(self):
        self.sigma_bins = np.logspace(np.log10(self.args.diff_params.sde_hp.sigma_min), np.log10(self.args.diff_params.sde_hp.sigma_max), num=self.args.logging.num_sigma_bins, base=10)

    def load_state_dict(self, state_dict):
        return t_utils.load_state_dict(state_dict, network=self.network, ema=self.ema, optimizer=self.optimizer)

    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        # Resume training from latest checkpoint available in the output director
        if checkpoint_path is not None:
            try:
                checkpoint=torch.load(checkpoint_path, map_location=self.device,weights_only=False)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=157007 #large number to mean that we loaded somethin, but it is arbitrary
                return self.load_state_dict(checkpoint)
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it=0

            try:
                checkpoint=torch.load(os.path.join(self.args.model_dir,checkpoint_path), map_location=self.device)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=157007 #large number to mean that we loaded somethin, but it is arbitrary
                self.network.load_state_dict(checkpoint['ema_model'])
                return True
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it=0
                return False
        else:
            try:
                print("trying to load a project checkpoint")
                print("checkpoint_id", checkpoint_id)
                print("model_dir", self.args.model_dir)
                print("exp_name", self.args.exp.exp_name)
                if checkpoint_id is None:
                    # find latest checkpoint_id
                    save_basename = f"{self.args.exp.exp_name}-*.pt"
                    save_name = f"{self.args.model_dir}/{save_basename}"
                    list_weights = glob(save_name)
                    id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
                    list_ids = [int(id_regex.search(weight_path).groups()[0])
                                for weight_path in list_weights]
                    checkpoint_id = max(list_ids)
    
                checkpoint = torch.load(
                    f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=159000 #large number to mean that we loaded somethin, but it is arbitrary
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                print(e)
                return False

    def state_dict(self):
        return {
            'it': self.it,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'args': self.args,
        }

    # def save_checkpoint(self):
    #     save_basename = f"{self.args.exp.exp_name}-{self.it}.pt"
    #     save_name = f"{self.args.model_dir}/{save_basename}"
    #     torch.save(self.state_dict(), save_name)
    #     print("saving",save_name)
    #     if self.args.logging.remove_old_checkpoints:
    #         try:
    #             os.remove(self.latest_checkpoint)
    #             print("removed last checkpoint", self.latest_checkpoint)
    #         except:
    #             print("could not remove last checkpoint", self.latest_checkpoint)
    #     self.latest_checkpoint=save_name
    def save_checkpoint(self, tag=None):
        if not self.is_main:
            return

        tag = tag or str(self.it)

        net = self.network.module if hasattr(self.network, "module") else self.network

        ckpt = {
            "it": self.it,
            "network": net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "args": self.args,
        }

        ckpt_name = f"{self.args.exp.exp_name}-{tag}-{self.it}.pt"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        tmp_path = ckpt_path + ".tmp"

        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)

        latest = {
            "checkpoint": ckpt_path,
            "iteration": self.it,
            "time": time.time(),
        }

        latest_path = os.path.join(self.ckpt_dir, "latest.json")
        latest_tmp = latest_path + ".tmp"

        with open(latest_tmp, "w") as f:
            json.dump(latest, f, indent=2)

        os.replace(latest_tmp, latest_path)

        self.latest_checkpoint = ckpt_path
        print(f"Saved checkpoint: {ckpt_path}", flush=True)
        
    def resume_latest_if_exists(self):
        latest_path = os.path.join(self.ckpt_dir, "latest.json")

        if not os.path.exists(latest_path):
            return False

        with open(latest_path, "r") as f:
            latest = json.load(f)

        ckpt_path = latest["checkpoint"]
        print(f"Resuming from latest checkpoint: {ckpt_path}", flush=True)

        return self.resume_from_checkpoint(checkpoint_path=ckpt_path)
    
    def process_loss_for_logging(self, error: torch.Tensor, sigma: torch.Tensor):
        """
        This function is used to process the loss for logging. It is used to group the losses by the values of sigma and report them using training_stats.
        args:
            error: the error tensor with shape [batch, audio_len]
            sigma: the sigma tensor with shape [batch]
        """
        #sigma values are ranged between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max. We need to quantize the values of sigma into 10 logarithmically spaced bins between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max
        torch.nan_to_num(error) #not tested might crash
        error = error.detach().cpu().numpy()
        training_stats.report('loss', error.mean())

        for i in range(len(self.sigma_bins)):
            if i == 0:
                mask = sigma <= self.sigma_bins[i]
            elif i == len(self.sigma_bins)-1:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])

            else:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])
            mask = mask.squeeze(-1).cpu()
            if mask.sum() > 0:
                # find the index of the first element of the mask
                idx = np.where(mask==True)[0][0]
                training_stats.report('error_sigma_'+str(self.sigma_bins[i]),error[idx].mean())

    # def get_batch(self):
    #     ''' Get an audio example from dset and apply the transform (spectrogram + compression)'''
    #     sample = next(self.dset).to(self.device)
    #     return sample
    
    def get_batch(self):
        try:
            sample,target = next(self.train_iter)
        except StopIteration:
            if getattr(self, "train_sampler", None) is not None:
                self.train_sampler.set_epoch(self.it)
            self.train_iter = iter(self.dset)
            sample,target = next(self.train_iter)

        return sample.to(self.device, non_blocking=True),target.to(self.device, non_blocking=True)
    
    def train_step(self,debug=False):
        '''Training step'''
        self.optimizer.zero_grad()

        sample,target = self.get_batch()
        if target.ndim == 2:
            target = target.unsqueeze(1)
        noise = None

        error, sigma = self.diff_params.loss_fn(self.network, target, n=noise,cond=sample)
        loss = error.mean()
        loss.backward()
        
        if self.args.exp.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

        # Update weights.
        self.optimizer.step()

        if self.args.logging.log and self.is_main:
            self.process_loss_for_logging(error, sigma)
        if debug:
            B = target.shape[0]
            t = torch.ones(B, device=target.device) * 0.01
            noise = torch.randn_like(target)

            x_t = target + t[:, None, None] * noise

            with torch.no_grad():
                x0_hat = self.diff_params.denoiser(
                    xn=x_t,
                    net=self.network,
                    t=t,
                    cond=sample,
                )

            print("sample", sample.shape)
            print("target", target.shape)
            print("x_t", x_t.shape)
            print("x0_hat", x0_hat.shape)
            print("MSE noisy-target:", ((x_t - target) ** 2).mean().item())
            print("MSE denoised-target:", ((x0_hat - target) ** 2).mean().item())
            idx = 0

            sample_cpu = sample[idx].detach().cpu()
            target_cpu = target[idx].detach().cpu()
            import soundfile as sf
            # binaural condition
            sf.write(
                "debug_audio/sample_binaural.wav",
                sample_cpu.T,
                16000
            )

            # dry mono target
            sf.write(
                "debug_audio/target_mono.wav",
                target_cpu.T,
                16000
            )

            # quick mono downmix for alignment comparison
            sample_mono = sample_cpu.mean(dim=0, keepdim=True)

            sf.write(
                "debug_audio/sample_downmix.wav",
                sample_mono.T,
                16000
            )

            print("Saved debug audio.")

    def update_ema(self):
        """Update exponential moving average of self.network weights."""

        ema_rampup = self.args.exp.ema_rampup  #ema_rampup should be set to 10000 in the config file
        ema_rate = self.args.exp.ema_rate #ema_rate should be set to 0.9999 in the config file
        t = self.it * self.args.exp.batch_size
        with torch.no_grad():
            if t < ema_rampup:
                s = np.clip(t / ema_rampup, 0.0, ema_rate)
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * s + src * (1-s))
            else:
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * ema_rate + src * (1-ema_rate))

    def easy_logging(self):
        """
         Do the simplest logging here. This will be called every 1000 iterations or so
        I will use the training_stats.report function for this, and aim to report the means and stds of the losses in wandb
        """
        training_stats.default_collector.update()
        loss_mean = training_stats.default_collector.mean('loss')
        self.wandb_run.log({'loss': loss_mean}, step=self.it)

        #Fancy plot for error with respect to sigma
        sigma_means = []
        sigma_stds = []
        for i in range(len(self.sigma_bins)):
            a = training_stats.default_collector.mean('error_sigma_'+str(self.sigma_bins[i]))
            sigma_means.append(a)
            a = training_stats.default_collector.std('error_sigma_'+str(self.sigma_bins[i]))
            sigma_stds.append(a)

        figure = utils_logging.plot_loss_by_sigma(sigma_means,sigma_stds, self.sigma_bins)
        wandb.log({"loss_dependent_on_sigma": figure}, step=self.it, commit=True)

    def heavy_logging(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """
        if self.tester is not None:
            if self.latest_checkpoint is not None:
                self.tester.load_checkpoint(self.latest_checkpoint)
            audio = self.tester.do_test(it=self.it)

            for i, x in enumerate(audio):
                self.log_audio(x, f"sample_{i}")

    def log_audio(self, x, name):
        string = name+"_"+self.args.tester.name

        audio_path = utils_logging.write_audio_file(x,self.args.exp.sample_rate, string,path=self.args.model_dir, normalize=True)
        self.wandb_run.log({"audio_"+str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},step=self.it)

        if self.args.logging.log_spectrograms:
            spec_sample = utils_logging.plot_spectrogram_from_raw_audio(x, self.args.logging.stft)
            self.wandb_run.log({"spec_"+str(string): spec_sample}, step=self.it)

    def training_loop(self):
        pbar = tqdm(total=self.args.exp.max_iters, desc="Training")
        pbar.update(self.it)
        while True:
            self.train_step()
            self.update_ema()
            
            if self.is_main and self.profile and self.args.logging.log and wandb.run is not None:
                if self.it < self.profile_total_steps:
                    self.profiler.step()
                elif self.it == self.profile_total_steps +1:
                    #log trace as an artifact in wandb
                    profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
                    profile_art.add_file(glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
                    wandb.log_artifact(profile_art)
                    print("profiling done")
                elif self.it > self.profile_total_steps +1:
                    self.profile = False

            if self.is_main and self.it>0 and self.it%self.args.logging.save_interval==0 and self.args.logging.save_model:
                self.save_checkpoint()

            if self.is_main and self.it>0 and self.it%self.args.logging.heavy_log_interval==0 and self.args.logging.log:
                self.heavy_logging()

            if self.is_main and self.it>0 and self.it%self.args.logging.log_interval==0 and self.args.logging.log:
                self.easy_logging()
            now = time.time()

            if self.is_main and (now - self.last_time_ckpt) >= self.time_ckpt_interval:
                self.save_checkpoint(tag="time")
                self.last_time_ckpt = now

            pbar.update(1)
            # Update state.
            self.it += 1
            try:
                if self.it > self.args.exp.max_iters:
                    break
            except:
                pass

        pbar.close()
    #----------------------------------------------------------------------------
