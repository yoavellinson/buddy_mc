import os
import hydra
import torch
import utils.setup as setup
import urllib
from omegaconf import open_dict
from hydra.core.hydra_config import HydraConfig

from testing.tester import Tester

def _main(args):

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")

    args.exp.model_dir = args.model_dir

    #################
    ## diff params ##
    #################

    diff_params = hydra.utils.instantiate(args.diff_params)

    #############
    ## Network ##
    #############

    network = hydra.utils.instantiate(args.network)
    network = network.to(device)

    ########################################
    ## diff params of the Operator if any ##
    ########################################

    diff_params_op = hydra.utils.instantiate(args.diff_params_op) if "diff_params_op" in args.keys() else None

    ####################################
    ## Network of the Operator if any ##
    ####################################

    network_op = hydra.utils.instantiate(args.network_op) if "network_op" in args.keys() else None
    network_op = network_op.to(device) if network_op is not None else None

    ##############
    ## test set ##
    ##############
    
    test_set = hydra.utils.instantiate(args.dset.test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1,  num_workers=args.exp.num_workers, pin_memory=True, worker_init_fn=setup.worker_init_fn)

    #############
    ## Tester  ##
    #############

    tester = Tester(args=args, network=network, diff_params=diff_params, test_set=test_set, device=device)

    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Network architecture:    {args.network._target_}')
    print(f'Diffusion parameterization:  {args.diff_params._target_}')
    print(f'Experiment:                  {args.exp.exp_name}')
    print(f'Tester:                  {args.tester.tester._target_}')
    print(f'Sampler:                  {args.tester.sampler._target_}')
    print(f'Checkpoint:                  {args.tester.checkpoint}')
    print(f'sample rate:                  {args.exp.sample_rate}')
    audio_len = args.exp.audio_len if not "audio_len" in args.tester.unconditional.keys() else args.tester.unconditional.audio_len
    print(f'audio len:                  {audio_len}')
    print()


    if args.tester.checkpoint != 'None':
        ckpt_path=os.path.join(dirname, args.tester.checkpoint)
        #leave the option of downloading the ckpt for later
        #if not os.path.exists(ckpt_path):
        #    print("downloading checkpoint from huggingface")
        #    urllib.request.urlretrieve("http://google.com/index.html", filename="local/index.html")
        #    HF_path="https://huggingface.co/Eloimoliner/babe/resolve/main/"+os.path.basename(args.tester.checkpoint)
        #    urllib.request.urlretrieve(HF_path, filename=ckpt_path)
           
        try:
            #relative path
            ckpt_path=os.path.join(dirname, args.tester.checkpoint)
            tester.load_checkpoint(ckpt_path) 
        except:
            #absolute path
            tester.load_checkpoint(os.path.join(args.model_dir,args.tester.checkpoint)) 
    else:
        print("trying to load latest checkpoint")
        tester.load_latest_checkpoint()

    tester.do_test()

@hydra.main(config_path="conf", config_name="conf_VCTK_binaural", version_base=None)
def main(args):
    gpu = args.get("gpu", 0)

    print(f"Running on GPU {gpu}")
    torch.cuda.set_device(gpu)

    _main(args)

if __name__ == "__main__":
    import sys

    overrides = [
        "-m",
        "--config-name=conf_VCTK_binaural_to_mono.yaml",
        "hydra/launcher=basic",
        "tester=blind_dereverberation_binaural",
        "tester.checkpoint=/home/workspace/yoavellinson/buddy_mc/experiments_binaural_to_mono/checkpoints/VCTK_16k_binaural_DPS-250000-250000.pt",
        "model_dir=experiments_binaural_to_mono",
        "dset=vctk_16k_4s_binaural",
        "+gpu=1",
        "dset.test.num_examples=1",

        # Choose one:
        "tester.modes=[blind_binaural_dereverberation]",
        # "tester.modes=[binaural_dereverberation]",

        "exp.exp_name=VCTK_16k_binaural_DPS_250k_blind_observed_y_decay_no_decay",
        "+tester.sampling_params.run_tag='blind_observed_y_decay_no_decay'",

        # Sampling core.
        "tester.posterior_sampling.zeta=5",
        "tester.sampling_params.T=800",
        "tester.sampling_params.Schurn=10",
        "tester.sampling_params.Snoise=1",
        "tester.sampling_params.warmup_steps=300",
        "tester.sampling_params.blind_schedule='standard'",  # 'standard' or 'learn_freeze'

        # Losses.
        "tester.sampling_params.lambda_sisdr=1.0",
        "tester.sampling_params.lambda_stft=1.0",
        "tester.posterior_sampling.constraint_speech_magnitude.use=false",

        # X/H initialization and smoothing.
        "tester.posterior_sampling.warm_initialization.mode='none','reverb_scaled'",  # informed winner favors none; keep reverb_scaled as control
        "tester.sampling_params.h_first_init='observed_y_direct_decay'",
        "tester.sampling_params.eta=0.05,0.1,0.15,0.2",
        "tester.sampling_params.beta=0.6,0.7,0.8,0.9",

        # Diagnostics.
        "tester.sampling_params.alpha=0.0",
        "tester.sampling_params.reverb_tail_delay=3",
    ]

    def _split_csv_override(key):
        prefix = key + "="
        for item in overrides:
            if item.startswith(prefix):
                return [x.strip() for x in item[len(prefix):].split(",") if x.strip()]
        return []

    def _without_overrides(keys):
        prefixes = tuple(key + "=" for key in keys)
        return [item for item in overrides if not item.startswith(prefixes)]

    def _split_chunks(items, n):
        chunks = [[] for _ in range(n)]
        for i, item in enumerate(items):
            chunks[i % n].append(item)
        return chunks

    user_args = sys.argv[1:]
    auto_4gpu = False
    single_run = False
    worker_jobs = None
    worker_gpu = None
    gpus = [0, 1, 2, 3]
    passthrough = []
    i = 0
    while i < len(user_args):
        arg = user_args[i]
        if arg == "--auto-4gpu":
            auto_4gpu = True
            i += 1
        elif arg == "--single-run":
            single_run = True
            i += 1
        elif arg == "--worker-jobs":
            worker_jobs = user_args[i + 1]
            i += 2
        elif arg.startswith("--worker-jobs="):
            worker_jobs = arg.split("=", 1)[1]
            i += 1
        elif arg == "--worker-gpu":
            worker_gpu = int(user_args[i + 1])
            i += 2
        elif arg.startswith("--worker-gpu="):
            worker_gpu = int(arg.split("=", 1)[1])
            i += 1
        elif arg == "--sweep-gpus":
            gpus = [int(x) for x in user_args[i + 1].split(",") if x]
            i += 2
        elif arg.startswith("--sweep-gpus="):
            gpus = [int(x) for x in arg.split("=", 1)[1].split(",") if x]
            i += 1
        else:
            passthrough.append(arg)
            i += 1

    if worker_jobs is not None:
        import subprocess

        if worker_gpu is None:
            raise RuntimeError("--worker-jobs requires --worker-gpu")
        jobs = []
        for spec in worker_jobs.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            eta, beta = spec.split(":", 1)
            jobs.append((eta, beta))

        for eta, beta in jobs:
            cmd = [
                sys.executable,
                __file__,
                "--single-run",
                f"+gpu={worker_gpu}",
                f"tester.sampling_params.eta={eta}",
                f"tester.sampling_params.beta={beta}",
                *passthrough,
            ]
            print(f"GPU {worker_gpu}: running eta={eta} beta={beta}")
            subprocess.run(cmd, check=True)
    elif auto_4gpu:
        import subprocess

        etas = _split_csv_override("tester.sampling_params.eta")
        betas = _split_csv_override("tester.sampling_params.beta")
        if not etas or not betas:
            raise RuntimeError("--auto-4gpu requires eta and beta overrides in the bottom overrides list")

        jobs = [(eta, beta) for eta in etas for beta in betas]
        chunks = _split_chunks(jobs, len(gpus))
        procs = []
        for gpu_id, chunk in zip(gpus, chunks):
            if not chunk:
                continue
            job_spec = ";".join(f"{eta}:{beta}" for eta, beta in chunk)
            print(f"GPU {gpu_id}: queued {job_spec}")
            cmd = [
                sys.executable,
                __file__,
                "--worker-gpu",
                str(gpu_id),
                "--worker-jobs",
                job_spec,
                *passthrough,
            ]
            procs.append(subprocess.Popen(cmd))

        failures = []
        for gpu_id, proc in zip([g for g, c in zip(gpus, chunks) if c], procs):
            ret = proc.wait()
            if ret != 0:
                failures.append((gpu_id, ret))
        if failures:
            raise SystemExit(f"auto-GPU worker failures: {failures}")
    else:
        if single_run:
            run_overrides = _without_overrides([
                "+gpu",
                "tester.sampling_params.eta",
                "tester.sampling_params.beta",
            ])
        else:
            run_overrides = overrides
        sys.argv = [sys.argv[0], *run_overrides, *passthrough]
        main()
