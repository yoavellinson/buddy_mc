import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.nn.functional import pad
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def chirp_log_torch(t, f0, f1, t1, device=None):
    """
    PyTorch implementation of scipy.signal.chirp(method='logarithmic')
    
    Args:
        t (Tensor): Time vector (seconds)
        f0 (float): Frequency at t=0 (Hz)
        f1 (float): Frequency at t=t1 (Hz)
        t1 (float): Time at which f1 is reached (seconds)
    """
    if device is None:
        device = t.device
        
    # k = (f1/f0)^(1/t1)
    k = (f1 / f0) ** (1 / t1)
    
    # Phase = 2 * pi * f0 * (k^t - 1) / ln(k)
    # This is the integral of f(t) = f0 * k^t
    phi = 2 * torch.pi * f0 * (k**t - 1) / torch.log(torch.tensor(k, device=device))
    
    return torch.sin(phi)

class MStep():
    """
        Euler Heun sampler for DPS 
        inverse problem solver
    """

    def __init__(self):
        super().__init__()
        self.stft_options = dict(size=512, shift=128)
        self.M = 256
        self.alpha = 0.0
        self.lambda_h = 1e-2



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
            return_complex=False
        )
        
        return time_signal
    
        
    def generate_sweep_pair(self,sr=16000, duration=4.0, f1=62.5):
        """
        Generates the synchronized Sine Sweep and Inverse Filter pair.
        """
        f2 = (sr / 2 )
        num_sample = int(duration * sr)
        taxis = np.arange(0, num_sample, 1) / (num_sample - 1)

        w1 = 2 * np.pi * f1 / sr
        w2 = 2 * np.pi * f2 / sr
        lw = np.log(w2 / w1)

        # Generate Sweep
        sweep = 0.3*np.sin(w1 * (num_sample - 1) / lw * (np.exp(taxis * lw) - 1))

        # Generate Inverse Filter
        envelope = (w2 / w1) ** (-taxis)
        inv = np.flip(sweep) * envelope
        scaling = np.pi * num_sample * (w1 / w2 - 1) / (2 * (w2 - w1) * np.log(w1 / w2))
        inv = inv / scaling

        # Convert to Torch and Pad
        sweep_t = pad(torch.from_numpy(sweep).float(), (512, 512))
        inv_t = pad(torch.from_numpy(inv).float(), (512, 512))
   
        return sweep_t, inv_t
    
    def apply_ctf_stft(self, X, H):
        """
        X: [F, T]
        H: [F, L]
        """
        F, T = X.shape
        L = H.shape[-1]
        
        X_padded = pad(X, (L - 1, 0)) 
        
        X_history = X_padded.unfold(1, L, 1) 
        X_history = X_history.flip(-1) 

        Y = torch.einsum('ftl,fl->ft', X_history, H.conj())
        
        return Y
    
    def get_rir_from_ctf(self,ctf, sr=16000):
        """
        Standalone function to extract a time-domain RIR from CTF coefficients.
        
        Args:
            ctf (Tensor): [F, L] complex tensor (Estimated CTF coefficients)
            sinesweep (Tensor): [Time] pre-generated sine sweep signal
            invfilter (Tensor): [Time] pre-generated inverse filter signal
            TF: Transformation object (must have .stft and .istft methods)
            sr (int): Sampling rate (default 16000)
            
        Returns:
            Tensor: [Time] The estimated time-domain Room Impulse Response
        """

        device = ctf.device
        L = ctf.shape[-1]
        
        # 1. Prepare Probe Signal
        # Convert sine sweep to STFT domain
        sinesweep,invfilter = self.generate_sweep_pair()
        sinesweep = sinesweep.to(device)
        sinesweep_spec = self.stft(sinesweep) # [F, T]

        # 2. Convolve CTF with Sine Sweep in STFT Domain
        ir_spec = self.apply_ctf_stft(sinesweep_spec,ctf)
        
        # 3. Transform back to Time Domain
        ir_time = self.istft(ir_spec) # [Time]
        sf.write('/home/workspace/yoavellinson/buddy_mc/test_m_step/sinesweep.wav',sinesweep,sr)
        sf.write('/home/workspace/yoavellinson/buddy_mc/test_m_step/ir_time.wav',ir_time,sr)

        # 4. Deconvolution via Inverse Filtering
        # Convolution with inverse filter collapses the sine sweep into an impulse
        invfilter = invfilter.to(device)
        rir = torchaudio.functional.convolve (ir_time,invfilter, mode="same")

        # 5. Alignment and Post-processing
        # Find peak (direct path) and trim leading zeros
        peak_idx = torch.argmax(rir.abs())
        start_offset = int(sr * 0.0025)
        
        rir = rir[max(0, peak_idx - start_offset) :]
        
        max_val = rir.abs().max()
        # if max_val > 1.0:
        rir = rir / max_val
        max_val = ir_time.abs().max()
        ir_time = ir_time/max_val
        return rir,ir_time
    
    def compute_simple_correlations(self, Xj):
        """
        Xj: [T, F, 1]
        self.Y: [T, F, 1]

        Returns:
            Rxx: [F, M, M]
            rxy: [F, 1, M]
        """
        # [T, F, 1] -> [F, T]
        X_ft = Xj.squeeze(-1).transpose(0, 1)
        Y_ft = self.Y.squeeze(-1).transpose(0, 1)

        F_bins, T = X_ft.shape
        M = self.M

        # Build causal history: [F, T, M]
        X_pad = torch.nn.functional.pad(X_ft, (M - 1, 0))
        X_hist = X_pad.unfold(1, M, 1).flip(-1)

        # Rxx = sum_t x_t x_t^H
        Rxx = torch.einsum('ftm,ftn->fmn', X_hist, X_hist.conj())

        # rxy = sum_t y_t x_t^H
        rxy = torch.einsum('ft,ftm->fm', Y_ft, X_hist.conj()).unsqueeze(1)

        return Rxx, rxy
    
    def stepEM(self, x_i, t_i, t_iplus1, gamma_i,i, eps=1e-5):
        x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i)
        x_hat = x_hat.detach().requires_grad_(True)
        
        #E step - Denoise using posterior sampleing
        x_den = self.get_Tweedie_estimate(x_hat, t_hat) #\hat{x_0}
        if self.args.tester.posterior_sampling.constraint_speech_magnitude.use:
            s_scale = self.args.tester.posterior_sampling.constraint_speech_magnitude.speech_scaling
            x_den = x_den * (s_scale / (x_den.detach().std() + 1e-8))
        X_den = self.stft(x_den).permute(2, 1, 0)

        if i < self.warmup_steps:
            h_tilde = torch.zeros((X_den.shape[1], self.M), dtype=X_den.dtype, device=X_den.device)
            h_tilde[:, 0] = 1.0

        else:
            eye = torch.eye(self.M, device=X_den.device, dtype=X_den.dtype).unsqueeze(0)

            Rxx_snapshot, rxy_snapshot = self.compute_simple_correlations(X_den)

            # reg_weight = (t_i / 0.99) * 10.0 
            # Rxx_stable = Rxx_snapshot + (eps  + reg_weight) * eye
            Rxx_stable = Rxx_snapshot + (eps + self.lambda_h) * eye

            h_column = torch.linalg.solve(Rxx_stable, rxy_snapshot.conj().transpose(1, 2))
            h_new = h_column.transpose(1, 2).squeeze(1)
            
            if not hasattr(self, 'prev_h_tilde'):
                h_tilde = h_new
            else:
                h_tilde = self.eta * self.prev_h_tilde + (1.0 - self.eta) * h_new
            self.prev_h_tilde = h_tilde.detach()
   
        decay = torch.exp(-self.alpha * torch.arange(self.M).to(h_tilde.device)).view(1, self.M)
        h_tilde = (h_tilde*decay ).to(device=X_den.device)
        
        lh_score, rec_loss_value = self.get_likelihood_score(x_den, x_hat, h_tilde,t_hat,i)

        x_hat_ng = x_hat.detach()
        score = self.Tweedie2score(x_den, x_hat_ng, t_hat)

        ode_integrand = self.diff_params._ode_integrand(x_hat_ng, t_hat, score) + lh_score
        dt = t_iplus1 - t_hat

        x_iplus1 = x_hat_ng + dt * ode_integrand

        return x_iplus1.detach_(), x_den.detach(),rec_loss_value,h_tilde.detach()

    
    def conv_h(self, wav, h):
        """
        Convolve mono waveform tensor with mono impulse response tensor.

        Args:
            wav : torch.Tensor
                Shape [T] or [1, T]
            h : torch.Tensor
                Shape [L] or [1, L]

        Returns:
            torch.Tensor
                Shape [1, T + L - 1]
        """

        # Flatten to 1D
        wav = wav.squeeze()
        T = wav.shape[0]
        h = h.squeeze()

        # Add batch/channel dims for conv1d
        wav = wav[None, None, :]   # [1,1,T]
        h = h.flip(0)[None, None, :]  # flip for true convolution

        # Full convolution padding
        pad = h.shape[-1] - 1

        out = F.conv1d(F.pad(wav, (pad, pad)), h)

        return out.squeeze(0)[:,:T].squeeze() #back to original shape
    
    def test_rir_est(self,h):
        sinesweep,invfilter = self.generate_sweep_pair()
        sinesweep_rev = self.conv_h(sinesweep,h)
        rir = torchaudio.functional.convolve(invfilter, sinesweep_rev, mode="full")

        peak_idx = torch.argmax(rir.abs())
        start_offset = int(sr * 0.0025)
        
        rir = rir[max(0, peak_idx - start_offset) :]
        
        max_val = rir.abs().max()
        rir = rir / max_val
        return rir
    
    def test_chirp(self,h):
        x,invfilter = self.generate_sweep_pair()
        y = self.conv_h(x,h)

        x_den = x
        X_den = self.stft(x_den).permute(1, 0)
        self.Y = self.stft(y).permute(1, 0)
        eye = torch.eye(self.M, device=X_den.device, dtype=X_den.dtype).unsqueeze(0)

        Rxx_snapshot, rxy_snapshot = self.compute_simple_correlations(X_den)
        Rxx_stable = Rxx_snapshot + (self.lambda_h) * eye

        h_column = torch.linalg.solve(Rxx_stable, rxy_snapshot.conj().transpose(1, 2))
        H_tilde = h_column.transpose(1, 2).squeeze(1)
        h_hat,y = self.get_rir_from_ctf(H_tilde)
        Y_hat = self.apply_ctf_stft(X_den.T,H_tilde)
        y_hat=self.istft(Y_hat)

        return h_hat,y_hat
    
 
    
    def __call__(self,h):
        x,sr = sf.read('/home/workspace/yoavellinson/buddy_mc/experiments/monaural_testing_gridsearch/test23_04_2026/monaural_dereverberation/VCTK_16k_monaural_h_orig/original/p226_312_mic2.wav')
        x = torch.tensor(x,dtype=torch.float32)
        y = self.conv_h(x,h)

        x_den = x
        X_den = self.stft(x_den).permute(1, 0)
        self.Y = self.stft(y).permute(1, 0)
        Rxx_snapshot, rxy_snapshot = self.compute_simple_correlations(X_den)
        Rxx_stable = Rxx_snapshot #+ (self.lambda_h) * eye

        h_column = torch.linalg.solve(Rxx_stable, rxy_snapshot.conj().transpose(1, 2))
        H_tilde = h_column.transpose(1, 2).squeeze(1)
        h_hat,_ = self.get_rir_from_ctf(H_tilde)
        Y_hat = self.apply_ctf_stft(X_den.T,H_tilde)
        y_hat=self.istft(Y_hat)

        return h_hat,y_hat,y
    
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

def plot_rirs_and_fft(
    rir_est,
    rir_gt,
    fs=16000,
    labels=("Estimated RIR", "Ground Truth RIR"),
    title="RIR Comparison",
    normalize_time=True,
    normalize_fft=True,
    db_fft=True,
    match_est_to_gt=True,
):
    """
    Plot:
    1. Estimated RIR time domain
    2. GT RIR time domain
    3. Estimated FFT
    4. GT FFT

    If match_est_to_gt=True:
        rir_est is cropped or zero-padded to match len(rir_gt)
    """

    rir_est = np.asarray(rir_est).squeeze()
    rir_gt = np.asarray(rir_gt).squeeze()

    if rir_est.ndim != 1 or rir_gt.ndim != 1:
        raise ValueError("Both inputs must be 1D single-channel signals.")

    # -------------------------------------------------
    # Match estimated RIR length to GT
    # -------------------------------------------------
    if match_est_to_gt:
        L_gt = len(rir_gt)
        L_est = len(rir_est)

        if L_est > L_gt:
            rir_est = rir_est[:L_gt]          # cut
        elif L_est < L_gt:
            pad_len = L_gt - L_est
            rir_est = np.pad(rir_est, (0, pad_len))   # zero pad

    # -------------------------------------------------
    # Normalize time domain
    # -------------------------------------------------
    if normalize_time:
        rir_est = rir_est / (np.max(np.abs(rir_est)) + 1e-12)
        rir_gt = rir_gt / (np.max(np.abs(rir_gt)) + 1e-12)

    # Time axes
    t_est = np.arange(len(rir_est)) / fs
    t_gt = np.arange(len(rir_gt)) / fs

    # -------------------------------------------------
    # FFT
    # -------------------------------------------------
    fft_est = np.fft.rfft(rir_est)
    fft_gt = np.fft.rfft(rir_gt)

    mag_est = np.abs(fft_est)
    mag_gt = np.abs(fft_gt)

    if normalize_fft:
        mag_est /= (np.max(mag_est) + 1e-12)
        mag_gt /= (np.max(mag_gt) + 1e-12)

    if db_fft:
        mag_est = 20 * np.log10(mag_est + 1e-12)
        mag_gt = 20 * np.log10(mag_gt + 1e-12)
        fft_ylabel = "Magnitude [dB]"
    else:
        fft_ylabel = "Magnitude"

    f_est = np.fft.rfftfreq(len(rir_est), d=1 / fs)
    f_gt = np.fft.rfftfreq(len(rir_gt), d=1 / fs)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title)

    axs[0, 0].plot(t_est, rir_est)
    axs[0, 0].set_title(f"{labels[0]} - Time")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t_gt, rir_gt)
    axs[0, 1].set_title(f"{labels[1]} - Time")
    axs[0, 1].grid(True)

    axs[1, 0].plot(f_est, mag_est)
    axs[1, 0].set_title(f"{labels[0]} - FFT")
    axs[1, 0].grid(True)

    axs[1, 1].plot(f_gt, mag_gt)
    axs[1, 1].set_title(f"{labels[1]} - FFT")
    axs[1, 1].grid(True)

    for ax in axs[0]:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

    for ax in axs[1]:
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(fft_ylabel)

    plt.tight_layout()
    plt.savefig('/home/workspace/yoavellinson/buddy_mc/test_m_step/hs.png')

if __name__=="__main__":
    h_wav = '/home/workspace/yoavellinson/buddy_mc/experiments/monaural_testing_gridsearch/test23_04_2026/monaural_dereverberation/VCTK_16k_monaural_h_orig/true_rir/h_hat_p226_312_mic2_alpha_0.05_zeta_0.35_M_128_warmup_steps_0_lambda_sisdr_0.0_lambda_stft_1.0_lambda_h_0.001_eta_0.9.wav'
    h,sr = sf.read(h_wav)
    h = torch.tensor(h,dtype=torch.float32)
    m_step =MStep()
    h_hat,y_hat,y = m_step(h)
    h_hat = h_hat[:h.shape[0]]
    # print(h_hat.shape)
    plot_rirs_and_fft(h_hat,h,fs=16000)
    sisdr = _sisdr_time_safe(y_hat,y)
    print(f'SI-SDR:{sisdr:.3f}dB')
    sf.write('/home/workspace/yoavellinson/buddy_mc/test_m_step/y_hat.wav',y_hat,16000)
    sf.write('/home/workspace/yoavellinson/buddy_mc/test_m_step/y.wav',y,16000)
    sf.write('/home/workspace/yoavellinson/buddy_mc/test_m_step/h.wav',h,16000)
    sf.write('/home/workspace/yoavellinson/buddy_mc/test_m_step/h_hat.wav',h_hat,16000)
    


