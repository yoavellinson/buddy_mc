import json
import os
import time

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import utils.setup as setup


class CTFTransform:
    def __init__(self, n_fft=510, hop_length=128, M=64, normalize=True, ridge=1e-4):
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.M = int(M)
        self.normalize = bool(normalize)
        self.ridge = float(ridge)

    def _window(self, device, dtype):
        return torch.hann_window(self.n_fft, periodic=False, device=device, dtype=dtype)

    def stft(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected audio [B,C,T] or [B,T], got {x.shape}")

        B, C, T = x.shape
        spec = torch.stft(
            x.reshape(B * C, T),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self._window(x.device, x.dtype),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return spec.reshape(B, C, spec.shape[-2], spec.shape[-1])

    def estimate_from_pair(self, x_ref, y_ref, return_specs=False):
        X = self.stft(x_ref)
        Y = self.stft(y_ref)
        if X.shape[1] == 1 and Y.shape[1] > 1:
            X = X.expand(-1, Y.shape[1], -1, -1)
        if X.shape[:3] != Y.shape[:3]:
            raise ValueError(f"Cannot estimate CTF from X={X.shape}, Y={Y.shape}")

        X_pad = F.pad(X, (self.M - 1, 0))
        X_hist = X_pad.unfold(-1, self.M, 1).flip(-1)
        Rxx = torch.einsum("bcftm,bcftn->bcfmn", X_hist.conj(), X_hist)
        rxy = torch.einsum("bcftm,bcft->bcfm", X_hist.conj(), Y).unsqueeze(-1)

        eye = torch.eye(self.M, device=Rxx.device, dtype=Rxx.dtype)
        diag_power = Rxx.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-10)
        Rxx = Rxx + self.ridge * diag_power[..., None, None] * eye
        h = torch.linalg.solve(Rxx, rxy).squeeze(-1).conj()
        if return_specs:
            return h, X, Y
        return h

    def apply_ctf_stft(self, X, h):
        if X.shape[1] == 1 and h.shape[1] > 1:
            X = X.expand(-1, h.shape[1], -1, -1)
        if X.shape[:3] != h.shape[:3]:
            raise ValueError(f"Cannot apply CTF with X={X.shape}, h={h.shape}")
        X_pad = F.pad(X, (h.shape[-1] - 1, 0))
        X_hist = X_pad.unfold(-1, h.shape[-1], 1).flip(-1)
        return torch.einsum("bcftm,bcfm->bcft", X_hist, h.conj())

    def istft(self, X, length):
        if X.ndim != 4:
            raise ValueError(f"Expected STFT [B,C,F,T], got {X.shape}")
        B, C, Freq, Frames = X.shape
        y = torch.istft(
            X.reshape(B * C, Freq, Frames),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self._window(X.device, X.real.dtype),
            center=True,
            normalized=False,
            onesided=True,
            length=length,
            return_complex=False,
        )
        return y.reshape(B, C, length)

    @staticmethod
    def sisdr_db(est, ref, eps=1e-8):
        est = est - est.mean(dim=-1, keepdim=True)
        ref = ref - ref.mean(dim=-1, keepdim=True)
        scale = (est * ref).sum(dim=-1, keepdim=True) / ref.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
        target = scale * ref
        noise = est - target
        ratio = target.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
        return 10.0 * torch.log10(ratio.clamp_min(eps))

    def reconstruction_metrics(self, x_ref, y_ref, h, X=None, Y=None):
        if X is None:
            X = self.stft(x_ref)
        if Y is None:
            Y = self.stft(y_ref)
        Y_hat = self.apply_ctf_stft(X, h)
        stft_rel_mse = (Y_hat - Y).abs().pow(2).mean() / Y.abs().pow(2).mean().clamp_min(1e-10)
        y_hat = self.istft(Y_hat, length=y_ref.shape[-1])
        time_rel_mse = (y_hat - y_ref).pow(2).mean() / y_ref.pow(2).mean().clamp_min(1e-10)
        return {
            "ctf_stft_rel_mse": float(stft_rel_mse.detach().cpu()),
            "ctf_time_rel_mse": float(time_rel_mse.detach().cpu()),
            "ctf_time_sisdr_db": float(self.sisdr_db(y_hat, y_ref).mean().detach().cpu()),
        }

    def complex_to_channels(self, h):
        h = torch.view_as_real(h.resolve_conj().contiguous())
        h = h.permute(0, 1, 4, 2, 3).reshape(h.shape[0], h.shape[1] * 2, h.shape[2], h.shape[3])
        if self.normalize:
            scale = h.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-8)
            h = h / scale[:, None, None, None]
        return h

    def batch_to_ctf(self, batch, device, return_metrics=False):
        metrics = {}
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            y, x = batch[:2]
            y = y.to(device, non_blocking=True).float()
            x = x.to(device, non_blocking=True).float()
            h, X, Y = self.estimate_from_pair(x, y, return_specs=True)
            if return_metrics:
                metrics = self.reconstruction_metrics(x, y, h, X=X, Y=Y)
            clean = self.complex_to_channels(h)
            return (clean, metrics) if return_metrics else clean

        h = batch.to(device, non_blocking=True)
        if torch.is_complex(h):
            h = self.complex_to_channels(h)
        else:
            h = h.float()
        return (h, metrics) if return_metrics else h


class CTFGaussianDenoiserTrainer:
    def __init__(self, args, loader, network, device, is_main=True, ddp=False, train_sampler=None):
        self.args = args
        self.loader = loader
        self.train_iter = iter(loader)
        self.network = network
        self.device = device
        self.is_main = is_main
        self.ddp = ddp
        self.train_sampler = train_sampler
        self.it = 0

        ctf_cfg = OmegaConf.to_container(args.ctf, resolve=True)
        ctf_transform_keys = {"n_fft", "hop_length", "M", "normalize", "ridge"}
        self.ctf = CTFTransform(
            **{k: v for k, v in ctf_cfg.items() if k in ctf_transform_keys}
        )
        self.optimizer = hydra.utils.instantiate(args.exp.optimizer, params=network.parameters())
        self.ckpt_dir = os.path.join(args.model_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if bool(args.exp.resume):
            self.resume_latest_if_exists()

    def sample_sigma(self, batch_size):
        sigma_min = float(self.args.denoiser.sigma_min)
        sigma_max = float(self.args.denoiser.sigma_max)
        if self.args.denoiser.sigma_sampling == "log":
            u = torch.rand(batch_size, device=self.device)
            sigma = sigma_min * (sigma_max / sigma_min) ** u
        else:
            sigma = torch.empty(batch_size, device=self.device).uniform_(sigma_min, sigma_max)
        return sigma

    def next_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.it)
            self.train_iter = iter(self.loader)
            return next(self.train_iter)

    def train_step(self):
        clean, metrics = self.ctf.batch_to_ctf(
            self.next_batch(),
            self.device,
            return_metrics=bool(self.args.ctf.get("log_reconstruction", True)),
        )
        sigma = self.sample_sigma(clean.shape[0]).to(clean.dtype)
        noise = torch.randn_like(clean)
        noisy = clean + sigma[:, None, None, None] * noise

        pred_noise = self.network(noisy, sigma=sigma)
        loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if bool(self.args.exp.use_grad_clip):
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.args.exp.max_grad_norm))
        self.optimizer.step()

        with torch.no_grad():
            denoised = noisy - sigma[:, None, None, None] * pred_noise
            denoised_mse = F.mse_loss(denoised, clean)
            noisy_mse = F.mse_loss(noisy, clean)

        stats = {
            "loss": float(loss.detach().cpu()),
            "denoised_mse": float(denoised_mse.detach().cpu()),
            "noisy_mse": float(noisy_mse.detach().cpu()),
            "sigma": float(sigma.mean().detach().cpu()),
        }
        stats.update(metrics)
        return stats

    def state_dict(self):
        net = self.network.module if hasattr(self.network, "module") else self.network
        return {
            "it": self.it,
            "network": net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": self.args,
        }

    def save_checkpoint(self, tag=None):
        if not self.is_main:
            return
        tag = tag or str(self.it)
        ckpt_path = os.path.join(self.ckpt_dir, f"{self.args.exp.exp_name}-{tag}-{self.it}.pt")
        tmp_path = ckpt_path + ".tmp"
        torch.save(self.state_dict(), tmp_path)
        os.replace(tmp_path, ckpt_path)
        latest = {"checkpoint": ckpt_path, "iteration": self.it, "time": time.time()}
        latest_path = os.path.join(self.ckpt_dir, "latest.json")
        with open(latest_path + ".tmp", "w") as f:
            json.dump(latest, f, indent=2)
        os.replace(latest_path + ".tmp", latest_path)
        print(f"Saved checkpoint: {ckpt_path}", flush=True)

    def resume_latest_if_exists(self):
        latest_path = os.path.join(self.ckpt_dir, "latest.json")
        if not os.path.exists(latest_path):
            return False
        with open(latest_path, "r") as f:
            latest = json.load(f)
        ckpt = torch.load(latest["checkpoint"], map_location=self.device, weights_only=False)
        net = self.network.module if hasattr(self.network, "module") else self.network
        net.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.it = int(ckpt["it"])
        print(f"Resumed CTF denoiser from iteration {self.it}", flush=True)
        return True

    def training_loop(self):
        pbar = tqdm(total=int(self.args.exp.max_iters), initial=self.it, desc="CTF denoiser")
        while self.it <= int(self.args.exp.max_iters):
            stats = self.train_step()
            if self.is_main and self.it % int(self.args.denoiser.log_interval) == 0:
                print(
                    f"it={self.it} loss={stats['loss']:.6g} "
                    f"denoised_mse={stats['denoised_mse']:.6g} noisy_mse={stats['noisy_mse']:.6g} "
                    f"sigma={stats['sigma']:.4g} "
                    f"ctf_stft_rel_mse={stats.get('ctf_stft_rel_mse', float('nan')):.3g} "
                    f"ctf_time_sisdr={stats.get('ctf_time_sisdr_db', float('nan')):.2f}dB",
                    flush=True,
                )
            if self.is_main and self.it > 0 and self.it % int(self.args.denoiser.save_interval) == 0:
                self.save_checkpoint()
            self.it += 1
            pbar.update(1)
        pbar.close()
        self.save_checkpoint(tag="final")


def _main(args):
    ddp_env_keys = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    ddp = all(k in os.environ for k in ddp_env_keys)

    if not ddp and any(k in os.environ for k in ddp_env_keys):
        print(
            "Partial DDP environment detected; running single-process. "
            f"Found: {[k for k in ddp_env_keys if k in os.environ]}",
            flush=True,
        )

    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    os.makedirs(args.model_dir, exist_ok=True)
    args.exp.model_dir = args.model_dir

    train_set = hydra.utils.instantiate(args.dset.train)
    train_sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=int(args.exp.batch_size),
        num_workers=int(args.exp.num_workers),
        pin_memory=True,
        worker_init_fn=setup.worker_init_fn,
        timeout=0,
        prefetch_factor=20 if int(args.exp.num_workers) > 0 else None,
    )

    network = hydra.utils.instantiate(args.network).to(device)
    if ddp:
        network = DDP(network, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if rank == 0:
        print()
        print("CTF Gaussian denoiser training options:")
        print(f"Output directory:     {args.model_dir}")
        print(f"Network architecture: {args.network._target_}")
        print(f"Dataset:              {args.dset.train._target_}")
        print(f"CTF n_fft/hop/M:      {args.ctf.n_fft}/{args.ctf.hop_length}/{args.ctf.M}")
        print(f"CTF ridge:            {args.ctf.ridge}")
        print(f"Gaussian sigma:       {args.denoiser.sigma_min} - {args.denoiser.sigma_max}")
        print(f"Batch size:           {args.exp.batch_size}")
        print()

    try:
        trainer = CTFGaussianDenoiserTrainer(
            args=args,
            loader=loader,
            network=network,
            device=device,
            is_main=rank == 0,
            ddp=ddp,
            train_sampler=train_sampler,
        )
        trainer.training_loop()
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


@hydra.main(config_path="conf", config_name="conf_ctf_denoiser", version_base=None)
def main(args):
    _main(args)


if __name__ == "__main__":
    main()
