from datetime import date
import re
import torch
import os
import numpy as np
import copy
from glob import glob
from tqdm import tqdm
import omegaconf
import hydra
import soundfile as sf

from testing.operators.subband_filtering import BlindSubbandFiltering
from testing.operators.reverb import RIROperator

import utils.log as utils_logging
import utils.training_utils as tr_utils
from collections import OrderedDict
def strip_module_prefix(sd):
    return OrderedDict(
        (k.removeprefix("module."), v)
        for k, v in sd.items()
    )
    
class Tester():
    def __init__(
        self, args, network, diff_params, test_set=None, device=None, in_training=False,
    ):
        self.args=args
        self.network=network
        self.diff_params=copy.copy(diff_params)
        self.device=device
        self.test_set = test_set
        self.in_training = in_training

        self.sampler=hydra.utils.instantiate(args.tester.sampler, self.network, self.diff_params, self.args)
        # sigma_data = self.estimate_sigma_data(self.test_set)
        # print("sigma_data =", sigma_data)


    def estimate_sigma_data(self,dataset, num_examples=None):
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        N = len(dataset) if num_examples is None else min(num_examples, len(dataset))

        for i in tqdm(range(N)):
            target = dataset[i][0]  # clean mono

            if not torch.is_tensor(target):
                target = torch.tensor(target)

            target = target.float()

            total_sum += target.sum().item()
            total_sq_sum += (target ** 2).sum().item()
            total_count += target.numel()

        mean = total_sum / total_count
        var = total_sq_sum / total_count - mean ** 2

        return var ** 0.5


    def _prepare_binaural_time_for_cues(self, signal, channels=2):
        signal = torch.as_tensor(signal, device=self.device, dtype=torch.float32)
        while signal.ndim > 3 and 1 in signal.shape:
            signal = signal.squeeze(0)

        if signal.ndim == 1:
            raise ValueError(f"Expected binaural signal, got shape {signal.shape}")
        if signal.ndim == 2:
            if signal.shape[0] == channels:
                signal = signal.unsqueeze(0)
            elif signal.shape[1] == channels:
                signal = signal.T.unsqueeze(0)
            else:
                raise ValueError(f"Cannot infer channel axis for signal shape {signal.shape}")
        elif signal.ndim == 3:
            if signal.shape[1] == channels:
                pass
            elif signal.shape[2] == channels:
                signal = signal.transpose(1, 2)
            else:
                raise ValueError(f"Cannot infer channel axis for signal shape {signal.shape}")
        else:
            raise ValueError(f"Expected signal with 2 or 3 dims, got {signal.shape}")

        return signal

    def estimate_ild_gains_for_cues(self, signal, channels=2):
        signal = self._prepare_binaural_time_for_cues(signal, channels=channels)
        channel_rms = signal.pow(2).mean(dim=-1).sqrt().clamp_min(1e-8)
        mean_rms = channel_rms.mean(dim=1, keepdim=True).clamp_min(1e-8)
        return channel_rms / mean_rms, channel_rms

    def estimate_itd_samples_gcc_phat_for_cues(self, signal, max_delay_seconds=1e-3, channels=2):
        signal = self._prepare_binaural_time_for_cues(signal, channels=channels)
        B, C, T = signal.shape
        sample_rate = getattr(self.args.exp, "sample_rate", 16000)
        max_lag = max(1, int(round(max_delay_seconds * sample_rate)))
        max_lag = min(max_lag, T - 1)

        n_fft = 1 << int((2 * T - 1).bit_length())
        ref_fft = torch.fft.rfft(signal[:, 0, :], n=n_fft)

        delays = torch.zeros(B, C, device=signal.device, dtype=signal.dtype)
        for c in range(1, C):
            sig_fft = torch.fft.rfft(signal[:, c, :], n=n_fft)
            cross = sig_fft * ref_fft.conj()
            cross = cross / cross.abs().clamp_min(1e-8)
            corr = torch.fft.irfft(cross, n=n_fft)
            corr = torch.cat([corr[:, -max_lag:], corr[:, : max_lag + 1]], dim=-1)
            lag_idx = corr.abs().argmax(dim=-1)
            delays[:, c] = lag_idx.to(dtype=signal.dtype) - max_lag

        return delays

    def estimate_binaural_cues_for_diagnostics(self, signal, max_delay_seconds=1e-3, channels=2):
        gains, rms = self.estimate_ild_gains_for_cues(signal, channels=channels)
        delays = self.estimate_itd_samples_gcc_phat_for_cues(
            signal,
            max_delay_seconds=max_delay_seconds,
            channels=channels,
        )
        sample_rate = getattr(self.args.exp, "sample_rate", 16000)
        left = rms[:, 0].clamp_min(1e-8)
        right = rms[:, 1].clamp_min(1e-8) if rms.shape[1] > 1 else left

        return {
            "rms": rms.detach(),
            "ild_gains": gains.detach(),
            "ild_right_minus_left_db": (20.0 * torch.log10(right / left)).detach(),
            "itd_samples": delays.detach(),
            "itd_us": (delays / float(sample_rate) * 1e6).detach(),
        }

    def format_binaural_cue_comparison(self, name_to_signal):
        cues = {
            name: self.estimate_binaural_cues_for_diagnostics(signal)
            for name, signal in name_to_signal.items()
        }
        lines = []
        for name, cue in cues.items():
            gains = cue["ild_gains"][0].detach().cpu().tolist()
            delays = cue["itd_samples"][0].detach().cpu().tolist()
            delays_us = cue["itd_us"][0].detach().cpu().tolist()
            ild_db = cue["ild_right_minus_left_db"][0].detach().cpu().item()
            lines.append(
                f"{name}: ILD R-L={ild_db:.2f} dB, "
                f"gains={['%.3f' % g for g in gains]}, "
                f"ITD samples={['%.2f' % d for d in delays]}, "
                f"ITD us={['%.1f' % d for d in delays_us]}"
            )

        names = list(cues.keys())
        if len(names) >= 2:
            ref_name = names[0]
            ref = cues[ref_name]
            for name in names[1:]:
                cue = cues[name]
                ild_delta = (cue["ild_right_minus_left_db"] - ref["ild_right_minus_left_db"])[0].detach().cpu().item()
                itd_delta = (cue["itd_samples"] - ref["itd_samples"])[0].detach().cpu().tolist()
                lines.append(
                    f"{name} - {ref_name}: delta ILD R-L={ild_delta:.2f} dB, "
                    f"delta ITD samples={['%.2f' % d for d in itd_delta]}"
                )

        return "\n".join(lines)

    def load_latest_checkpoint(self):
        #load the latest checkpoint from self.args.model_dir
        try:
            # find latest checkpoint_id
            save_basename = f"{self.args.exp.exp_name}-*.pt"
            save_name = f"{self.args.model_dir}/{save_basename}"
            list_weights = glob(save_name)
            id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
            list_ids = [int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights]
            checkpoint_id = max(list_ids)

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
            try:
                self.network.load_state_dict(state_dict['ema'])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict['model'], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    # def load_checkpoint(self, path):
    #     state_dict = torch.load(path, map_location=self.device,weights_only=False)
    #     try:
    #         self.it=state_dict['it']
    #     except:
    #         self.it=0
    #     print("loading checkpoint")
    #     return tr_utils.load_state_dict(state_dict, ema=self.network)


    def load_checkpoint(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=False)

            self.it = state_dict.get("it", 0)

            print("loading checkpoint")

            if "network" in state_dict:
                state_dict["network"] = strip_module_prefix(state_dict["network"])

            if "ema" in state_dict:
                state_dict["ema"] = strip_module_prefix(state_dict["ema"])

            return tr_utils.load_state_dict(state_dict, ema=self.network)
        
        except Exception as e:
            print(f"Could not load checkpoint: {type(e).__name__}: {e}")

            print("Using random model weights.")
            self.it = 0
            return self.network


    def load_checkpoint_legacy(self, path):
        state_dict = torch.load(path, map_location=self.device)

        try:
            print("load try 1")
            self.network.load_state_dict(state_dict['ema'])
        except:
            #self.network.load_state_dict(state_dict['model'])
            try:
                print("load try 2")
                dic_ema = {}
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['ema_weights']):
                    dic_ema[key] = tensor
                self.network.load_state_dict(dic_ema)
            except:
                print("load try 3")
                dic_ema = {}
                i=0
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['model'].values()):
                    if tensor.requires_grad:
                        dic_ema[key]=state_dict['ema_weights'][i]
                        i=i+1
                    else:
                        dic_ema[key]=tensor     
                self.network.load_state_dict(dic_ema)
        try:
            self.it=state_dict['it']
        except:
            self.it=0


    ##############################
    ### UNCONDITIONAL SAMPLING ###
    ##############################

    def sample_unconditional(self, mode):
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len
        shape = [self.args.tester.unconditional.num_samples, audio_len]
        preds = self.sampler.predict_unconditional(shape, self.device)

        if not self.in_training:
            for i in range(len(preds)):
                path_generated = utils_logging.write_audio_file(preds[i], self.args.exp.sample_rate, f"unconditional_{i}", path=self.paths["unconditional"])

        return preds





    #######################
    ### DEREVERBERATION ###
    #######################

    def test_dereverberation(self, mode, blind=False):

        if self.test_set is None:
            print("No test set specified")
            return
        if len(self.test_set) == 0:
            print("No samples found in test set")
            return
        
        for i, (original, rir,  filename) in enumerate(tqdm(self.test_set)):

            seg = torch.from_numpy(original).float().to(self.device)
            seg = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * seg / seg.std() #Normalize the input to match sigma_data of dataset

            #read and prepare the RIR
            RIR=torch.Tensor(rir).to(self.device)

            with torch.no_grad():

                # Forward pass with true RIR
                operator_ref = RIROperator(self.args.tester.informed_dereverberation.op_hp, time_kernel_size=RIR.shape[-1], sample_rate=self.args.exp.sample_rate)
                operator_ref.update_params(RIR)
                y = operator_ref.degradation(seg.unsqueeze(0))

                if blind: # Initialize operator
                    assert self.args.tester.blind_dereverberation.operator == "subband_filtering"
                    operator_blind = BlindSubbandFiltering(self.args.tester.informed_dereverberation.op_hp, sample_rate=self.args.exp.sample_rate)
                    with torch.no_grad():
                        operator_blind.update_H(use_noise=True)

            pred = self.sampler.predict_conditional(y, operator_blind if blind else operator_ref, shape=(1,seg.shape[-1]), blind=blind)

            path_original=utils_logging.write_audio_file(seg, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"original"])
            path_degraded=utils_logging.write_audio_file(y, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"])
            path_reconstructed=utils_logging.write_audio_file(pred, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"reconstructed"])
            
            utils_logging.write_audio_file(RIR.detach().cpu(), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"true_rir"])
            if blind:
                utils_logging.write_audio_file(self.sampler.operator.get_time_RIR().detach().cpu(), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"estimated_rir"])
            
            print(path_reconstructed)

    def test_binaural_dereverberation(self, mode, blind=False):

        if self.test_set is None:
            print("No test set specified")
            return
        if len(self.test_set) == 0:
            print("No samples found in test set")
            return
        
        for i, (original, rir, filename,h_orig,hrtf) in enumerate(tqdm(self.test_set)):
            if self.args.tester.sampling_params.lambda_sisdr ==0 and self.args.tester.sampling_params.lambda_stft == 0:
                break
            seg_raw = torch.from_numpy(original).float().to(self.device)

            #read and prepare the RIR
            y_raw=torch.Tensor(rir).to(self.device)

            input_scaling = self.args.tester.get("input_scaling", {})
            target_sigma = input_scaling.get(
                "target_sigma",
                self.args.tester.posterior_sampling.warm_initialization.scaling_factor,
            )
            cond_to_target_std = input_scaling.get("cond_to_target_std", 0.3843)

            # Deployable input scaling: only the observed binaural signal is used
            # for sampler input. The clean target scaling below is for reference
            # audio/debug denoising in this paired evaluation path.
            y = y_raw * (target_sigma * cond_to_target_std) / (y_raw.std() + 1e-8)
            seg = seg_raw * target_sigma / (seg_raw.std() + 1e-8)

            # x0 = seg.unsqueeze(0).unsqueeze(0)
            # sample=y.unsqueeze(0)
            # sigma = torch.ones(1, device=x0.device) * 1e-5
            # noise = torch.randn_like(x0)
            # x_t = x0 + sigma[:, None, None] * noise

            # x0_hat = self.sampler.diff_params.denoiser(
            #     xn=x_t,
            #     net=self.sampler.model,
            #     t=sigma,
            #     cond=sample,
            # )
            # path_reconstructed=utils_logging.write_audio_file(x0.detach().cpu(), self.args.exp.sample_rate,  os.path.basename(filename)[: -4]+"debug_x0", path=self.paths[mode+"reconstructed"],stereo=False)
            # path_reconstructed=utils_logging.write_audio_file(x0_hat.detach().cpu(), self.args.exp.sample_rate,  os.path.basename(filename)[: -4]+"debug_x0_hat", path=self.paths[mode+"reconstructed"],stereo=False)
            # break

            h_orig = torch.Tensor(h_orig).to(self.device)

            f_name_new = (
                os.path.basename(filename)[: -4]
                + f'_zeta_{self.args.tester.posterior_sampling.zeta}'
                + '_h_orig'
                + f'_lambda_sisdr_{self.args.tester.sampling_params.lambda_sisdr}'
                + f'_lambda_stft_{self.args.tester.sampling_params.lambda_stft}'
                + f'_T_{self.args.tester.sampling_params.T}'
                + f'_eta{self.args.tester.sampling_params.eta}'
                + f'_b{self.args.tester.sampling_params.beta}'
                + f'_wu{self.args.tester.sampling_params.warmup_steps}'
                + f'_hwu{self.args.tester.sampling_params.h_orig_warmup_mode}'
                + f'_hf{self.args.tester.sampling_params.h_first_init}'
                + f'_init_{self.args.tester.posterior_sampling.warm_initialization.mode}'
            )
            h_debug_dir = os.path.join(self.paths[mode], "h_tilde_vs_h_orig")
            pred = self.sampler.predict_conditional(
                y,
                h_orig=h_orig,
                h_debug_dir=h_debug_dir,
                h_debug_name=f_name_new,
                h_orig_ctf_x_ref=seg.unsqueeze(0).unsqueeze(0),
                h_debug_x0=seg.unsqueeze(0).unsqueeze(0),
            ) #, operator_blind if blind else operator_ref, shape=(1,seg.shape[-1]), blind=blind)
            path_original=utils_logging.write_audio_file(seg, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"original"],stereo=False)
            path_degraded=utils_logging.write_audio_file(y.unsqueeze(0), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"],stereo=True)
            # y_hat = self.sampler.conv_h(seg.unsqueeze(0).unsqueeze(0),h_orig)
            # path_degraded=utils_logging.write_audio_file(y_hat, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"],stereo=True)

            path_reconstructed=utils_logging.write_audio_file(pred, self.args.exp.sample_rate, f_name_new, path=self.paths[mode+"reconstructed"],stereo=True)
            # path_h=utils_logging.write_audio_file(h.unsqueeze(0), self.args.exp.sample_rate, f_name_new, path=self.paths[mode+"true_rir"],stereo=True)

            # Force Garbage Collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(path_reconstructed)

    def test_blind_binaural_dereverberation(self, mode):

        if self.test_set is None:
            print("No test set specified")
            return
        if len(self.test_set) == 0:
            print("No samples found in test set")
            return
        
        for i, (original, rir, filename,h_orig,hrtf) in enumerate(tqdm(self.test_set)):
            # if self.args.tester.sampling_params.lambda_sisdr ==0 and self.args.tester.sampling_params.lambda_stft == 0:
            #     break
            seg_raw = torch.from_numpy(original).float().to(self.device)

            #read and prepare the RIR
            y_raw=torch.Tensor(rir).to(self.device)

            input_scaling = self.args.tester.get("input_scaling", {})
            target_sigma = input_scaling.get(
                "target_sigma",
                self.args.tester.posterior_sampling.warm_initialization.scaling_factor,
            )
            cond_to_target_std = input_scaling.get("cond_to_target_std", 0.3843)

            y = y_raw * (target_sigma * cond_to_target_std) / (y_raw.std() + 1e-8)
            seg = seg_raw * target_sigma / (seg_raw.std() + 1e-8)


            hrtf = torch.Tensor(hrtf).to(self.device)
            h_orig_t = torch.Tensor(h_orig).to(self.device)

            cue_report = self.format_binaural_cue_comparison({
                "real_brir": h_orig_t,
                "observed_y": y,
                "anechoic_hrtf": hrtf,
            })
            cue_dir = os.path.join(self.paths[mode], "binaural_cue_diagnostics")
            os.makedirs(cue_dir, exist_ok=True)
            cue_path = os.path.join(cue_dir, os.path.basename(filename)[: -4] + "_ild_itd.txt")
            with open(cue_path, "w") as f:
                f.write(cue_report + "\n")
            # print("\n" + cue_report)

            run_tag = self.args.tester.sampling_params.get("run_tag", "blind")
            f_name_new = (
                os.path.basename(filename)[: -4]
                + f'_r_{run_tag}'
                + f'_z{self.args.tester.posterior_sampling.zeta}'
                + f'_a{self.args.tester.sampling_params.alpha}'
                + f'_e{self.args.tester.sampling_params.eta}'
                + f'_b{self.args.tester.sampling_params.beta}'
                + f'_i{self.args.tester.posterior_sampling.warm_initialization.mode}'
                + f'_T{self.args.tester.sampling_params.T}'
                + f'_wu{self.args.tester.sampling_params.warmup_steps}'
                + f'_bs{self.args.tester.sampling_params.blind_schedule}'
                + f'_lsisdr{self.args.tester.sampling_params.lambda_sisdr}'
                + f'_lstft{self.args.tester.sampling_params.lambda_stft}'
                + f'_hf{self.args.tester.sampling_params.h_first_init}'
                + f'_td{self.args.tester.sampling_params.reverb_tail_delay}'
            )
            h_debug_dir = os.path.join(self.paths[mode], "h_tilde_0")
            oracle_h_init = self.args.tester.sampling_params.h_first_init in (
                "h_orig",
                "h_orig_direct_decay",
            )
            sampler_kwargs = {
                "h_debug_dir": h_debug_dir,
                "h_debug_name": f_name_new,
                "h_debug_x0": seg.unsqueeze(0).unsqueeze(0),
            }
            if oracle_h_init:
                sampler_kwargs.update({
                    "h_orig": h_orig_t,
                    "h_orig_ctf_x_ref": seg.unsqueeze(0).unsqueeze(0),
                })

            pred = self.sampler.predict_unconditional(
                y,
                **sampler_kwargs,
            ) #, operator_blind if blind else operator_ref, shape=(1,seg.shape[-1]), blind=blind)
            path_original=utils_logging.write_audio_file(seg, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"original"],stereo=False)
            path_degraded=utils_logging.write_audio_file(y.unsqueeze(0), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"],stereo=True)
            # y_hat = self.sampler.conv_h(seg.unsqueeze(0).unsqueeze(0),h_orig)
            # path_degraded=utils_logging.write_audio_file(y_hat, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"],stereo=True)

            path_reconstructed=utils_logging.write_audio_file(pred, self.args.exp.sample_rate, f_name_new, path=self.paths[mode+"reconstructed"],stereo=True)
            # path_h=utils_logging.write_audio_file(h.unsqueeze(0), self.args.exp.sample_rate, f_name_new, path=self.paths[mode+"true_rir"],stereo=True)

            # Force Garbage Collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(path_reconstructed)

    def test_monaural_dereverberation(self, mode, blind=False):

        if self.test_set is None:
            print("No test set specified")
            return
        if len(self.test_set) == 0:
            print("No samples found in test set")
            return
        
        for i, (original, rir,  filename,h_orig,hrtf) in enumerate(tqdm(self.test_set)):
            #binaural to mono:
            idx = 0
            rir = rir[idx,:]
            seg = torch.from_numpy(original).float().to(self.device)
            seg = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * seg / seg.std() #Normalize the input to match sigma_data of dataset

            #read and prepare the RIR
            y=torch.Tensor(rir).to(self.device).unsqueeze(0)
            h_orig = torch.tensor(h_orig.T[idx,:],dtype=y.dtype).to(self.device).unsqueeze(0)
            hrtf = torch.tensor(hrtf.T[idx,:],dtype=y.dtype).to(self.device).unsqueeze(0)
            
            pred,h,y_ = self.sampler.predict_conditional(y,seg) #, operator_blind if blind else operator_ref, shape=(1,seg.shape[-1]), blind=blind)
            f_name_new = os.path.basename(filename)[: -4]+f'_alpha_{self.args.tester.sampling_params.alpha}' +f'_zeta_{self.args.tester.posterior_sampling.zeta}'+f'_M_{self.args.tester.sampling_params.M}' +f'_w_steps_{self.args.tester.sampling_params.warmup_steps}'+f'_lambda_sisdr_{self.args.tester.sampling_params.lambda_sisdr}'+f'_lambda_stft_{self.args.tester.sampling_params.lambda_stft}'+f'_T_{self.args.tester.sampling_params.T}'+f'_eta_{self.args.tester.sampling_params.eta}'+f'_sigma_max_{self.args.tester.sampling_params.sde_hp.sigma_max}'
            path_original=utils_logging.write_audio_file(seg, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"original"],stereo=False)
            path_degraded=utils_logging.write_audio_file(y.unsqueeze(0), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"],stereo=False)
            path_reconstructed=utils_logging.write_audio_file(pred.unsqueeze(0), self.args.exp.sample_rate, f_name_new, path=self.paths[mode+"reconstructed"],stereo=False)
            path_h=utils_logging.write_audio_file(h.unsqueeze(0), self.args.exp.sample_rate, 'h_hat_'+f_name_new, path=self.paths[mode+"true_rir"],stereo=False)
            path_y=utils_logging.write_audio_file(y_.unsqueeze(0), self.args.exp.sample_rate, 'y_hat_'+f_name_new, path=self.paths[mode+"true_rir"],stereo=False)
            path_h_orig=utils_logging.write_audio_file(h_orig.unsqueeze(0), self.args.exp.sample_rate, 'h_'+f_name_new, path=self.paths[mode+"true_rir"],stereo=False)

            # Force Garbage Collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(path_reconstructed)

    def prepare_directories(self, mode, unconditional=False, blind=False):
            
            today=date.today() 
            self.paths={}

            if "overriden_name" in self.args.tester.keys() and self.args.tester.overriden_name is not None:
                self.path_sampling = os.path.join(self.args.model_dir, self.args.tester.overriden_name)
            else:
                self.path_sampling = os.path.join(self.args.model_dir,'test'+today.strftime("%d_%m_%Y"))
            if not os.path.exists(self.path_sampling):
                os.makedirs(self.path_sampling)

            self.paths[mode]=os.path.join(self.path_sampling,mode,self.args.exp.exp_name)

            if not os.path.exists(self.paths[mode]):
                os.makedirs(self.paths[mode])

            if not unconditional:
                self.paths[mode+"original"]=os.path.join(self.paths[mode],"original")
                if not os.path.exists(self.paths[mode+"original"]):
                    os.makedirs(self.paths[mode+"original"])
                self.paths[mode+"degraded"]=os.path.join(self.paths[mode],"degraded")
                if not os.path.exists(self.paths[mode+"degraded"]):
                    os.makedirs(self.paths[mode+"degraded"])
                self.paths[mode+"reconstructed"]=os.path.join(self.paths[mode],"reconstructed")
                if not os.path.exists(self.paths[mode+"reconstructed"]):
                    os.makedirs(self.paths[mode+"reconstructed"])
                    
                if "dereverberation" in mode:
                    self.paths[mode+"true_rir"]=os.path.join(self.paths[mode],"true_rir")
                    if not os.path.exists(self.paths[mode+"true_rir"]):
                        os.makedirs(self.paths[mode+"true_rir"])

                    if mode == "blind_dereverberation":
                        self.paths[mode+"estimated_rir"]=os.path.join(self.paths[mode],"estimated_rir")
                        if not os.path.exists(self.paths[mode+"estimated_rir"]):
                            os.makedirs(self.paths[mode+"estimated_rir"])

    def save_experiment_args(self, mode):
        with open(os.path.join(self.paths[mode], ".argv"), 'w') as f: #Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        for m in self.args.tester.modes:

            if m == "unconditional":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                return self.sample_unconditional(m)
            
            elif m == "informed_dereverberation":
                print("testing informed dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_dereverberation(m)

            elif m == "blind_dereverberation":
                print("testing blind dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_dereverberation(m, blind=True)

            elif m == "binaural_dereverberation":
                print("testing binaural dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_binaural_dereverberation(m, blind=True)

            elif m == "blind_binaural_dereverberation":
                print("testing blind binaural dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_blind_binaural_dereverberation(m)


            elif m == "monaural_dereverberation":
                print("testing binaural dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_monaural_dereverberation(m, blind=True)

            else:
                print("Warning: unknown mode: ", m)
