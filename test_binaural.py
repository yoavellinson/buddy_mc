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

# @hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
# def main(args):
#     torch.cuda.set_device(args.gpu)
#     _main(args)
@hydra.main(config_path="conf", config_name="conf_VCTK_binaural", version_base=str(hydra.__version__))
def main(args):
    # 1. Get the job index from the multirun sweep
    job_num = HydraConfig.get().job.num
    
    # 2. Unlock the config to add the 'gpu' key if it doesn't exist
    with open_dict(args):
        # This cycles 0, 1, 2, 3 based on your 4 GPUs
        args.gpu = args.get("gpu", job_num % 4)
    
    # 3. Set the device globally for this process
    print(f"Running Job {job_num} on GPU {args.gpu}")
    torch.cuda.set_device(args.gpu)
    
    _main(args)

if __name__=="__main__":
    main()

# if __name__ == "__main__":
#     import os
#     import sys
#     for zeta in [0.001,0.005,0.01,0.05]:
#         # --- DEBUGGING SHIM START ---
#         # Only run this if we are in a debugging environment or running manually
#         if len(sys.argv) == 1:  # If no arguments are passed, use these defaults
#             sys.path.append(os.getcwd())
#             os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
#             os.environ["HYDRA_FULL_ERROR"] = "1"
#             os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#             # Define your paths
#             ckpt = "/home/workspace/yoavellinson/buddy_mc/ckpt/VCTK_16k_4s_time-190000.pt"
#             name = "buddy_wpe-init_noise-prior_N-201_rir-aligned_1exp"
#             path_experiment = f"experiments/{name}"
            
#             # Ensure the directory exists
#             os.makedirs(path_experiment, exist_ok=True)

#             # Inject arguments into sys.argv
#             sys.argv.extend([
#                 "--config-name=conf_VCTK_binaural.yaml",
#                 "tester=blind_dereverberation_binaural",
#                 f"tester.checkpoint={ckpt}",
#                 f"tester.posterior_sampling.zeta={zeta}",
#                 f"model_dir={path_experiment}",
#                 "+gpu=0",
#                 "dset=vctk_16k_4s_binaural",
#                 "dset.test.num_examples=1",
#             ])
#         print(zeta)
#         main()


'''

python test_binaural.py -m \
    --config-name=conf_VCTK_binaural.yaml \
    tester=blind_dereverberation_binaural \
    tester.checkpoint=/home/workspace/yoavellinson/buddy_mc/ckpt/VCTK_16k_4s_time-190000.pt \
    tester.posterior_sampling.zeta=0.22,0.3,0.5,0.05,0.005 \
    tester.posterior_sampling.rec_loss.name='sisdr','stft_comp' \
    tester.sampling_params.alpha=0.32,0.35 \
    tester.sampling_params.warmup_steps=5,10,15,20 \
    tester.sampling_params.beta_min=0.4,0.6 \
    model_dir=experiments/buddy_grid_search \
    dset=vctk_16k_4s_binaural \
    dset.test.num_examples=20

'''

'''

python test_binaural.py -m \
    --config-name=conf_VCTK_binaural.yaml \
    tester=blind_dereverberation_binaural \
    tester.checkpoint=/home/workspace/yoavellinson/buddy_mc/ckpt/VCTK_16k_4s_time-190000.pt \
    model_dir=experiments/final_params \
    dset=vctk_16k_4s_binaural \
    tester.posterior_sampling.warm_initialization='none', \
    dset.test.num_examples=-1
'''