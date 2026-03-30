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

class BinauralEulerHeunSamplerDPS(EulerHeunSampler):
    """
        Euler Heun sampler for DPS 
        inverse problem solver
    """

    def __init__(self, model, diff_params, args):
        super().__init__(model, diff_params, args)
        self.zeta = self.args.tester.posterior_sampling.zeta
        self.stft_options = dict(size=510, shift=128)
        self.rec_loss = CompressedSTFTLoss(compression_factor=1/2)#SiSDRLoss() #L2ComplexSTFTSumMean()
        self.beta = 0.9
        self.warmup_steps = 25

    def initialize_x(self, shape, device, schedule):
        Y = self.stft(self.y)
        self.Y = Y.permute(2,1,0)
        if self.args.tester.posterior_sampling.warm_initialization.mode == "none":
            x = schedule[0]*torch.randn(shape).to(device)

        elif self.args.tester.posterior_sampling.warm_initialization.mode == "reverb_scaled":
            x = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * self.y.clone() / self.y.std() + schedule[0] * torch.randn(shape).to(device)
        
        elif self.args.tester.posterior_sampling.warm_initialization.mode == "wpe_scaled":
            print("Processing WPE")

            delay = self.args.tester.posterior_sampling.warm_initialization.wpe.delay
            iterations = self.args.tester.posterior_sampling.warm_initialization.wpe.iterations
            taps = self.args.tester.posterior_sampling.warm_initialization.wpe.taps
            
            Y = Y.cpu().numpy().transpose(2, 0, 1)
            Z = np.empty_like(Y)

            Z_right_1 = wpe(
                Y,
                taps=taps,
                delay=delay,
                iterations=iterations,
                statistics_mode='full'
            )
            Z_left_1 = wpe(
                Y[:, [1, 0], :],
                taps=taps,
                delay=delay,
                iterations=iterations,
                statistics_mode='full'
            )
            Z[:,0,:] = Z_left_1[:,1,:]
            Z[:,1,:] = Z_right_1[:,1,:]
            Z = Z.transpose(1, 2, 0)

            x_pred = self.istft(torch.from_numpy(Z)).to(self.y.device).type(self.y.dtype)
            if x_pred.shape[-1] > self.y.shape[-1]:
                x_pred = x_pred[..., :self.y.shape[-1]]

            x_pred = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * x_pred / x_pred.std()
            x = x_pred + schedule[0] * torch.randn(shape).to(device)

        else:
            raise NotImplementedError
        
        return x
    
    def stft(self, time_signal):
        """
        Args:
            time_signal: [J, Time_Samples] - Binaural waveform (e.g., [2, 131840])
        Returns:
            stft_signal: [J, F, T] - Complex STFT (e.g., [2, 257, 1031])
        """
        # 1. Extract Options
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
        
        return stft_signal


    def istft(self, stft_signal, length=None):
        """
        Args:
            stft_signal: [J, F, T] - Complex STFT
            length: Optional original length to trim padding exactly
        Returns:
            time_signal: [J, Time_Samples] - Reconstructed waveform
        """
        # 1. Extract Options
        n_fft = self.stft_options['size']
        hop_length = self.stft_options['shift']
        win_length = n_fft
        
        device = stft_signal.device
        window = torch.hann_window(win_length, periodic=False).to(device)

        # 2. Compute iSTFT
        # PyTorch natively handles the batch (J) dimension
        time_signal = torch.istft(
            stft_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=length, # Trims the 'fading' padding automatically
            return_complex=False
        )
        
        return time_signal
    
    def apply_ctf(self, S_binaural, h_tilde):
        """
        Args:
            S_binaural: [F, T, J] - Anechoic binaural signal (e.g., [257, 515, 2])
            h_tilde: [J, F, M] - Estimated binaural CTF (e.g., [2, 257, 12])
        Returns:
            Y_reverb: [J, F, T] - Generated reverberant STFT
        """
        # 1. Setup dimensions
        F, T, J = S_binaural.shape
        J_h, F_h, M = h_tilde.shape
        device = S_binaural.device
        h_tilde = h_tilde.to(device)
        # Ensure channels match
        assert J == J_h, f"Channel mismatch: Signal has {J}, but Filter has {J_h}"

        # 2. Reshape S to [J, F, T] for easier processing
        # Move channel to front: [2, 257, 515]
        S = S_binaural.permute(2, 0, 1).to(device)

        # 3. Pad the start of each channel
        # Padding M-1 frames on the time dimension (dim=2)
        S_padded = torch.nn.functional.pad(S, (M - 1, 0)) # [2, 257, 515 + M - 1]

        # 4. Create History Matrix for both channels
        # S_history shape: [2, 257, 515, M]
        # .unfold(2, M, 1) slides across the Time dimension
        S_history = S_padded.unfold(2, M, 1).flip(-1)

        # 5. Parallel Convolution (MIMO/Direct)
        # j=channels, f=freqs, t=time, m=taps
        # We multiply each channel's signal by its specific RIR
        # 'jfm' (filter) * 'jftm' (history) -> 'jft' (output)
        Y_reverb = torch.einsum('jfm,jftm->jft', h_tilde, S_history)

        return Y_reverb
    

    def generate_early_binaural_output(self, x_den, h_tilde, early_taps=2, cutoff_freq=100):
        """
        Implements: y_hat_B = [W * h_early] * x_den_clean + HPF
        
        Args:
            x_den: [2, Time] - Clean speech estimate from Tweedie
            h_tilde: [2, F, M] - Full estimated binaural RIR
            early_taps: Number of taps to keep (e.g., 2)
            cutoff_freq: Cutoff for the HPF in Hz (Default 100Hz)
        """
        # 1. Standardize x_den to STFT [J, F, T]
        X_den_stft = self.stft(x_den) 
        
        # 2. Truncate to Early Reflections only
        h_early = h_tilde[:, :, :early_taps]
        
        # 3. Apply CTF (Convolution in STFT domain)
        # permute X to [F, T, J] for your apply_ctf logic
        Y_hat_early = self.apply_ctf(X_den_stft.permute(1, 2, 0), h_early)
        
        # 4. Back to Time Domain [2, Time]
        y_binaural = self.istft(Y_hat_early, length=self.y.shape[-1])
        
        # 5. Apply High-Pass Filter (HPF)
        # This removes low-frequency artifacts and 'DC-drift'
        sample_rate = self.args.exp.sample_rate # 16000
        y_binaural_filtered = F_audio.highpass_biquad(
            y_binaural, 
            sample_rate=sample_rate, 
            cutoff_freq=cutoff_freq, 
            Q=0.707 # Standard Butterworth Q-factor
        )
        
        return y_binaural_filtered
    
    def get_likelihood_score(self, X_den, x, h_tilde):

        Y_hat = self.apply_ctf(X_den.permute(1,0,2),h_tilde)
        # y_hat = self.istft(Y_hat)
        # rec = self.rec_loss(y_hat,self.y)
        rec = self.rec_loss(Y_hat,self.Y.permute(2,1,0))
        rec_grads = torch.autograd.grad(outputs=rec, inputs=x)[0]

        # Normalize weighting parameter zeta
        normguide = torch.norm(rec_grads)/(self.args.exp.audio_len**0.5)
        return self.zeta / (normguide+1e-8) * rec_grads, rec
    
    def update_statistics(self, Xj, Yj, i, j, M=12):
            """
            Args:
                j: The channel index (0 for Left, 1 for Right)
            """
            # 1. Initialization of buffers on first call
            if not hasattr(self, 'running_Rxx'):
                F = Xj.shape[0]
                dev = Xj.device
                self.running_Rxx = torch.zeros((2, F, M, M), dtype=Xj.dtype, device=dev)
                self.running_rxy = torch.zeros((2, F, 1, M), dtype=Xj.dtype, device=dev)

            # 2. Compute current Snapshot
            Xj_padded = torch.nn.functional.pad(Xj, (M - 1, 0))
            X_mat = Xj_padded.unfold(1, M, 1).flip(-1) 
            
            Rxx_snapshot = torch.einsum('fti,ftj->fij', X_mat, X_mat.conj())
            rxy_snapshot = torch.einsum('ft,ftm->fm', Yj, X_mat.conj()).unsqueeze(1)

            # 3. Recursive Update (per channel j)
            if i < self.warmup_steps:
                # During warmup, we take the snapshot directly to initialize the buffer
                self.running_Rxx[j] = Rxx_snapshot
                self.running_rxy[j] = rxy_snapshot
            else:
                # Apply forgetting factor to the specific channel
                self.running_Rxx[j] = self.beta * self.running_Rxx[j] + (1 - self.beta) * Rxx_snapshot
                self.running_rxy[j] = self.beta * self.running_rxy[j] + (1 - self.beta) * rxy_snapshot



    def stepEM(self, x_i, t_i, t_iplus1, gamma_i,i, M=12, eps=1e-5):
        x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i)
        x_hat = x_hat.detach().requires_grad_(True)
        
        #E step - Denoise using posterior sampleing
        x_den = self.get_Tweedie_estimate(x_hat, t_hat) #\hat{x_0}
        X_den = self.stft(x_den).permute(2, 1, 0)
        # X_den = torch.tensor(X_den_np)
        h_tilde = torch.zeros((2,X_den.shape[1],M),dtype=X_den.dtype)
        #M step
        eye = torch.eye(M, device=x_den.device).unsqueeze(0) # [1, M, M]
        for j in range(2):
            Xj = X_den[:,:,j].T
            Yj = self.Y[:,:,j].T
            self.update_statistics(Xj=Xj,Yj=Yj,i=i,j=j)
            Rxx_stable = self.running_Rxx[j] + eps * eye
            h_column = torch.linalg.solve(Rxx_stable, self.running_rxy[j].conj().transpose(1, 2))
            h_tilde[j] = h_column.conj().transpose(1, 2).squeeze()

   
        if self.args.tester.posterior_sampling.constraint_speech_magnitude.use:
            x_den = self.args.tester.posterior_sampling.constraint_speech_magnitude.speech_scaling / x_den.detach().std() * x_den #Match the sigma_data of dataset

        lh_score, rec_loss_value = self.get_likelihood_score(X_den, x_hat,h_tilde)
        x_hat.detach_()

      
        score = self.Tweedie2score(x_den, x_hat, t_hat)
    
        ode_integrand = self.diff_params._ode_integrand(x_hat, t_hat, score) + lh_score
        dt = t_iplus1 - t_hat


        x_iplus1 = x_hat + dt * ode_integrand

        return x_iplus1.detach_(), x_den.detach(),h_tilde.detach()



    def predict(
        self,
        shape, 
        device,
        blind=False
    ):
        # get the noise schedule
        t = self.create_schedule().to(device)

        # sample prior
        x = self.initialize_x(shape,device, t)

        # parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma = self.get_gamma(t).to(device)

        for i in tqdm(range(0, self.T, 1)):
            self.step_counter=i
            x, x_den,h_tilde = self.stepEM(x, t[i] , t[i+1], gamma[i],i)

        # return  x_den.detach()
        y_final = self.generate_early_binaural_output(x_den, h_tilde)

        # Global normalization (preserves the L/R ratio)
        y_final = y_final / (torch.max(torch.abs(y_final)) + 1e-8)
        return y_final
    
    def predict_unconditional(self, *args, **kwargs):
        raise ValueError("DPS not made for unconditional sampling")

    def predict_conditional(
        self,
        y,  #observations 
        shape=None,
        blind=False,
        **kwargs
    ):

        self.y = y

        if shape is None:
            shape = y.shape
        return self.predict(shape, y.device, blind)


class CompressedSTFTLoss(nn.Module):
    """
    Exact implementation of Eq. 6 from the provided paper.
    Uses power-law compression (2/3) on the magnitude while preserving phase.
    """
    def __init__(self, compression_factor=2/3):
        super(CompressedSTFTLoss, self).__init__()
        self.alpha = compression_factor

    def forward(self, Y_hat, Y):
        """
        Args:
            Y_hat: [J, F, T] - Estimated complex STFT (h * x_hat_0)
            Y: [J, F, T] - Observed complex STFT
        """
        # 1. Apply Compression S_comp to both signals
        # S_comp = |Y|^alpha * exp(j * phase(Y))
        
        # Process Y_hat
        mag_hat = torch.abs(Y_hat)
        phase_hat = torch.angle(Y_hat)
        S_hat = (mag_hat + 1e-8).pow(self.alpha) * torch.exp(1j * phase_hat)
        
        # Process Y (Observed)
        mag_y = torch.abs(Y)
        phase_y = torch.angle(Y)
        S_y = (mag_y + 1e-8).pow(self.alpha) * torch.exp(1j * phase_y)
        
        # 2. Compute the Complex L2 distance (Eq. 6)
        # Using view_as_real to handle complex squared distance
        diff = S_y - S_hat
        squared_diff = torch.view_as_real(diff).pow(2).sum(dim=-1) # [J, F, T]
        
        # 3. Mean over all dimensions (M time frames, K freq bins, J channels)
        # The 1/M in Eq. 6 suggests a mean over the time dimension
        loss = torch.mean(squared_diff)
        
        return loss
    

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
   
        B=1
        C,_ =ref_time.shape
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