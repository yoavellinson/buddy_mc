from tqdm import tqdm
import utils.log as utils_logging
import torch
import torchaudio
import numpy as np
from nara_wpe.wpe import wpe
# from nara_wpe.utils import stft, istft
import wandb
import os
import utils.reverb_utils as reverb_utils
from utils.losses import get_loss
import torch.nn as nn
from testing.EulerHeunSampler import EulerHeunSampler
import torchaudio.functional as F_audio
import torch.nn.functional as F

class BinauralToMonoEulerHeunSamplerDPS(EulerHeunSampler):
    """
        Euler Heun sampler for DPS 
        inverse problem solver
    """

    def __init__(self, model, diff_params, args):
        super().__init__(model, diff_params, args)
        self.zeta = self.args.tester.posterior_sampling.zeta
        self.stft_options = dict(size=510, shift=128)
        self.rec_loss_sisdr = SiSDRLoss() 
        self.rec_loss_stft = CompressedSTFTLoss()
        self.warmup_steps = self.args.tester.sampling_params.warmup_steps
        self.M = self.args.tester.sampling_params.M
        self.alpha = self.args.tester.sampling_params.alpha
        # self.beta_min = self.args.tester.sampling_params.beta_min
        # self.beta_max = self.args.tester.sampling_params.beta_max
        self.lambda_stft = self.args.tester.sampling_params.lambda_stft
        self.lambda_sisdr = self.args.tester.sampling_params.lambda_sisdr

    def initialize_x(self, shape, device, schedule):
        """
        Initialize mono latent x for binaural-to-mono conditional sampling.

        self.y: [B, 2, T] binaural observation
        shape:  [B, 1, T] mono latent shape
        """

        # Store STFT of binaural observation for likelihood losses
        self.Y = self.stft(self.y)  # [B, 2, F, TT]

        mode = self.args.tester.posterior_sampling.warm_initialization.mode
        sigma_max = schedule[0]

        if mode == "none":
            x = sigma_max * torch.randn(shape, device=device)

        elif mode == "reverb_scaled":
            # Use a single binaural channel as warm start.
            # Averaging channels creates comb-filter / double-speech artifacts
            # because of interaural delays.
            mono_init = self.y[:, :1, :]  # [B,1,T]

            scale = self.args.tester.posterior_sampling.warm_initialization.scaling_factor

            mono_init = scale * mono_init / (
                mono_init.std(dim=-1, keepdim=True) + 1e-8
            )

            x = mono_init + sigma_max * torch.randn(
                shape,
                device=device,
                dtype=mono_init.dtype,
            )

        elif mode == "wpe_scaled":
            print("Processing WPE")

            delay = self.args.tester.posterior_sampling.warm_initialization.wpe.delay
            iterations = self.args.tester.posterior_sampling.warm_initialization.wpe.iterations
            taps = self.args.tester.posterior_sampling.warm_initialization.wpe.taps

            x_pred_list = []

            for b in range(self.y.shape[0]):
                Y_b = self.Y[b]  # [2,F,TT]
                Y_np = Y_b.detach().cpu().numpy().transpose(2, 0, 1)  # [TT,2,F]

                Z = wpe(
                    Y_np,
                    taps=taps,
                    delay=delay,
                    iterations=iterations,
                    statistics_mode="full",
                )  # [TT,2,F]

                Z = torch.from_numpy(Z.transpose(1, 2, 0)).to(
                    device=self.y.device,
                    dtype=self.Y.dtype,
                )  # [2,F,TT]

                y_wpe = self.istft(Z.unsqueeze(0), length=self.y.shape[-1])  # [1,2,T]

                # Use one channel, not mean, to avoid comb-filter/double-speech artifacts
                mono_wpe = y_wpe[:, :1, :]  # left channel: [1,1,T]
                # mono_wpe = y_wpe[:, 1:2, :]  # right channel alternative

                x_pred_list.append(mono_wpe)

            x_pred = torch.cat(x_pred_list, dim=0).to(device=device, dtype=self.y.dtype)

            scale = self.args.tester.posterior_sampling.warm_initialization.scaling_factor
            x_pred = scale * x_pred / (x_pred.std(dim=-1, keepdim=True) + 1e-8)

            x = x_pred + sigma_max * torch.randn(
                shape,
                device=device,
                dtype=x_pred.dtype,
            )

        else:
            raise NotImplementedError(f"Unknown warm initialization mode: {mode}")

        assert x.shape == shape, f"initialize_x produced {x.shape}, expected {shape}"

        return x
    

    def get_Tweedie_estimate(self, x, t_i):
        """
        Compute denoised mono estimate x0_hat.

        x:      [B, 1, T] noisy mono latent
        self.y: [B, 2, T] binaural condition
        t_i:    scalar or [B] noise level / time
        """

        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B,T] -> [B,1,T]

        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B,1,T], got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected mono latent x with C=1, got {x.shape}")

        if not hasattr(self, "y"):
            raise RuntimeError("self.y is not set. Call predict_conditional(y=...) first.")

        cond = self.y

        if cond.ndim == 2:
            cond = cond.unsqueeze(0)

        if cond.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has batch {x.shape[0]}, cond has batch {cond.shape[0]}"
            )

        if cond.shape[1] != 2:
            raise ValueError(f"Expected binaural condition [B,2,T], got {cond.shape}")

        x_hat = self.diff_params.denoiser(
            xn=x,
            net=self.model,
            t=t_i,
            cond=cond,
        )

        return x_hat

    def reset_buffers(self):
        """Call this at the very beginning of predict() for every new file"""
        if hasattr(self, 'running_Rxx'):
            del self.running_Rxx
        if hasattr(self, 'running_rxy'):
            del self.running_rxy
        # Clear the CUDA cache to defragment memory
        torch.cuda.empty_cache()

    def stft(self, time_signal):
        """
        Args:
            time_signal: [J, Time_Samples] - Binaural waveform (e.g., [2, 131840])
        Returns:
            stft_signal: [J, F, T] - Complex STFT (e.g., [2, 257, 1031])
        """
        # 1. Extract Options
        B,C,T = time_signal.shape

        time_signal = time_signal.reshape(B*C,T)
        n_fft = self.stft_options['size']
        hop_length = self.stft_options['shift']
        win_length = n_fft # Standard for perfect reconstruction
        
        device = time_signal.device
        window = torch.hann_window(win_length, periodic=False).to(device)

        # 2. Compute STFT
        # return_complex=True is mandatory for your CTF estimation grad
        stft_signal = torch.stft(
            time_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,          # Matches 'fading' logic
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        _,F,T = stft_signal.shape
        return stft_signal.reshape(B,C,F,T)
    
    def istft(self, stft_signal, length=None):
        n_fft = self.stft_options["size"]
        hop_length = self.stft_options["shift"]
        win_length = n_fft

        device = stft_signal.device
        window = torch.hann_window(win_length, periodic=False).to(device)

        if stft_signal.ndim == 4:
            B, C, Freq, Frames = stft_signal.shape
            stft_signal = stft_signal.reshape(B * C, Freq, Frames)

            time_signal = torch.istft(
                stft_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=length,
                return_complex=False,
            )

            return time_signal.reshape(B, C, -1)

        elif stft_signal.ndim == 3:
            return torch.istft(
                stft_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=length,
                return_complex=False,
            )

        else:
            raise ValueError(f"Expected STFT shape [B,C,F,T] or [C,F,T], got {stft_signal.shape}")
    
    def conv_h(self, wav, h):

        # wav: [B,1,T]
        # h: [2,L]
        B, _, T = wav.shape
        h = h.squeeze()
        if h.shape[0] != 2:
            h = h.T
        h = h.to(wav.device, wav.dtype)

        # duplicate mono speech to both ears
        wav = wav.repeat(1, 2, 1)  # [B,2,T]
        # grouped conv
        h = h.flip(-1)[:, None, :]  # [2,1,L]
        pad = h.shape[-1] - 1
        out = F.conv1d(
            F.pad(wav, (pad, pad)),
            h,
            groups=2
        )

        return out[..., :T]
       
    def get_likelihood_score_h_orig(self, x_den, x, h_orig,i):
        y_hat = self.conv_h(x_den,h_orig)
        Y_hat = self.stft(y_hat)
        rec = self.lambda_sisdr*self.rec_loss_sisdr(y_hat,self.y)
        rec += self.lambda_stft*self.rec_loss_stft(Y_hat,self.Y)
        rec_grads = torch.autograd.grad(outputs=rec, inputs=x)[0]
        normguide = torch.norm(rec_grads)/(self.args.exp.audio_len**0.5)
        return self.zeta / (normguide+1e-8) * rec_grads, rec

    def stepEM_h_orig(self, x_i, t_i, t_iplus1, gamma_i,i,h_orig, eps=1e-5):
        x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i)
        x_hat = x_hat.detach().requires_grad_(True)
        
        #E step - Denoise using posterior sampleing
        x_den = self.get_Tweedie_estimate(x_hat, t_hat) #\hat{x_0}
        lh_score, rec_loss_value = self.get_likelihood_score_h_orig(x_den, x_hat, h_orig,i)
        x_hat_ng = x_hat.detach()
        score = self.Tweedie2score(x_den, x_hat_ng, t_hat)

        if self.args.tester.posterior_sampling.constraint_speech_magnitude.use:
            s_scale = self.args.tester.posterior_sampling.constraint_speech_magnitude.speech_scaling
            x_den = x_den * (s_scale / (x_den.detach().std() + 1e-8))
                

        ode_integrand = self.diff_params._ode_integrand(x_hat_ng, t_hat, score) + lh_score
        dt = t_iplus1 - t_hat

        x_iplus1 = x_hat_ng + dt * ode_integrand

        return x_iplus1.detach_(), x_den.detach(),rec_loss_value

    def predict_h_orig(
        self,
        shape, 
        device,
        h_orig
        ):
        self.reset_buffers()
        # get the noise schedule
        t = self.create_schedule().to(device)
        # sample prior
        x = self.initialize_x(shape,device, t)

        gamma = self.get_gamma(t).to(device)
        pbar = tqdm(range(0, self.T, 1))

        for i in pbar:
            self.step_counter=i
            x, x_den,rec_loss_value = self.stepEM_h_orig(x, t[i] , t[i+1], gamma[i],i,h_orig)
            pbar.set_postfix({
               "rec": f"{rec_loss_value.item():.4f}"})
        return x_den

    def predict_unconditional(self, *args, **kwargs):
        raise ValueError("DPS not made for unconditional sampling")

    def predict_conditional(
        self,
        y,  #observations 
        h_orig=None,
        shape=None,
        blind=False,
        **kwargs
    ):  
        if len(y.shape)==2:
            y = y.unsqueeze(0)
        self.y = y
        if shape is None:
            # shape = y.shape
            shape = (y.shape[0], 1, y.shape[-1])

        x_den = self.predict_h_orig(shape, y.device, h_orig)

        return x_den


# class CompressedSTFTLoss(nn.Module):
#     """
#     Exact implementation of Eq. 6 from the provided paper.
#     Uses power-law compression (2/3) on the magnitude while preserving phase.
#     """
#     def __init__(self, compression_factor=2/3):
#         super(CompressedSTFTLoss, self).__init__()
#         self.alpha = compression_factor

#     def forward(self, Y_hat, Y):
#         """
#         Args:
#             Y_hat: [J, F, T] - Estimated complex STFT (h * x_hat_0)
#             Y: [J, F, T] - Observed complex STFT
#         """
#         # 1. Apply Compression S_comp to both signals
#         # S_comp = |Y|^alpha * exp(j * phase(Y))
        
#         # Process Y_hat
#         mag_hat = torch.abs(Y_hat)
#         phase_hat = torch.angle(Y_hat)
#         S_hat = (mag_hat + 1e-8).pow(self.alpha) * torch.exp(1j * phase_hat)
        
#         # Process Y (Observed)
#         mag_y = torch.abs(Y)
#         phase_y = torch.angle(Y)
#         S_y = (mag_y + 1e-8).pow(self.alpha) * torch.exp(1j * phase_y)
        
#         # 2. Compute the Complex L2 distance (Eq. 6)
#         # Using view_as_real to handle complex squared distance
#         diff = S_y - S_hat
#         squared_diff = torch.view_as_real(diff).pow(2).sum(dim=-1) # [J, F, T]
        
#         # 3. Mean over all dimensions (M time frames, K freq bins, J channels)
#         # The 1/M in Eq. 6 suggests a mean over the time dimension
#         loss = torch.mean(squared_diff)
        
#         return loss
    
class CompressedSTFTLoss(nn.Module):
    def __init__(self, compression_factor=2/3, weight=512, summean=True):
        super().__init__()
        self.alpha = compression_factor
        self.weight = weight
        self.summean = summean

    def forward(self, Y_hat, Y):
        # Y_hat, Y: [..., F, T], complex

        Y_hat_comp = (Y_hat.abs() + 1e-8).pow(self.alpha) * torch.exp(1j * Y_hat.angle())
        Y_comp = (Y.abs() + 1e-8).pow(self.alpha) * torch.exp(1j * Y.angle())

        diff2 = (Y_comp - Y_hat_comp).abs().pow(2)  # [..., F, T]

        if self.summean:
            loss = torch.mean(torch.sum(diff2, dim=-2))  # sum F, mean rest
        else:
            loss = torch.mean(diff2)

        return self.weight * loss
    
def _sisdr_time_safe(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8, return_db: bool = True):
    """
    est, ref: [B, C, L] real tensors (time-domain)
    SI-SDR = 10*log10( ||s_target||^2 / ||e_noise||^2 ), with zero-mean per sample.
    Numerically safe (no div by zero, no log of zero).
    Returns [B, C] (dB if return_db).
    """
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    ref_energy = (ref**2).sum(dim=-1, keepdim=True).clamp_min(eps)  # [B,C,1]
    scale = (est * ref).sum(dim=-1, keepdim=True) / ref_energy      # [B,C,1]
    s_target = scale * ref                                          # [B,C,L]

    e_noise = est - s_target

    num = (s_target**2).sum(dim=-1).clamp_min(eps)  # [B,C]
    den = (e_noise**2).sum(dim=-1).clamp_min(eps)   # [B,C]
    ratio = (num / den) #.clamp(1e-4, 1e4)
    if return_db:
        return 10.0 * torch.log10(ratio)
    else:
        return ratio

class SiSDRLoss(nn.Module):
    """
    SI-SDR loss for batched 2-channel audio given complex STFTs.
    est_stft, ref_stft: [B, 2, F, T], complex dtype
    Returns scalar loss = -mean(SI-SDR_dB over batch & channels).
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, est_time: torch.Tensor, ref_time: torch.Tensor) -> torch.Tensor:
   
        B, C, _ = ref_time.shape
        target_len = ref_time.size(-1)
        # Reshape back to [B, C, L] and keep in float32
        ref_time = ref_time.view(B, C, -1).to(torch.float32)
        est_time = est_time.view(B, C, -1).to(torch.float32)

        # Extra guard: if any sample is entirely zero, add a tiny dither (prevents 0/0)
        silent_ref = (ref_time.abs().sum(dim=-1, keepdim=True) == 0)
        if silent_ref.any():
            ref_time = ref_time + silent_ref * (self.eps * torch.randn_like(ref_time))

        sisdr_bc = _sisdr_time_safe(est_time, ref_time, eps=self.eps, return_db=True)  # [B, C]
        loss_per = -sisdr_bc  # maximize SI-SDR

        if self.reduction == 'mean':
            return loss_per.mean()
        elif self.reduction == 'sum':
            return loss_per.sum()
        else:
            return loss_per