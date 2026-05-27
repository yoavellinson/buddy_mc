import os
import numpy as np
import torch
import random
import glob
import soundfile as sf
import pandas as pd
from pathlib import Path
from scipy import signal
import torch.distributed as dist
from torch.utils.data import get_worker_info

class VCTKTrain(torch.utils.data.IterableDataset):
    def __init__(self,
        fs=16000,
        segment_length=65536,
        path="", #path to the dataset
        speakers_discard=[], #list of speakers to discard
        speakers_test=[], #list of speakers to use for testing, discarded here
        normalize=False,  #to normalize or not. I don't normalize by default
        seed=0
        ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.train_samples=[]
        #iterate over speakers directories
        speakers=os.listdir(path)
        for s in speakers:
            if s in speakers_discard:
                continue
            elif s in speakers_test:
                continue
            else:
                self.train_samples.extend(glob.glob(os.path.join(path,s,"*.wav")))

        assert len(self.train_samples)>0 , "error in dataloading: empty or nonexistent folder"

        self.segment_length=int(segment_length)
        self.fs=fs

        self.normalize=normalize
        if self.normalize:
            raise NotImplementedError("normalization not implemented yet")

    def __iter__(self):

        while True:
            num=random.randint(0,len(self.train_samples)-1)
            file=self.train_samples[num]
            data, samplerate = sf.read(file)
            assert samplerate==self.fs, "wrong sampling rate"
            segment=data
            #Stereo to mono
            if len(data.shape)>1 :
                segment=np.mean(segment,axis=1)

            L=len(segment)

            #crop or pad to get to the right length
            if L>self.segment_length:
                #get random segment
                idx=np.random.randint(0,L-self.segment_length)
                segment=segment[idx:idx+self.segment_length]
            elif L<=self.segment_length:
                #pad with zeros to get to the right length randomly
                idx=np.random.randint(0,self.segment_length-L)
                #copy segment to get to the right length
                segment=np.pad(segment,(idx,self.segment_length-L-idx),'wrap')

            yield  segment


class VCTKTest(torch.utils.data.Dataset):
    def __init__(self,
        fs=16000,
        segment_length=65536,
        path="", #path to the dataset
        speakers_discard=[], #list of speakers to discard
        speakers_test=[], #list of speakers to use for testing, discarded here
        normalize=False,  #to normalize or not. I don't normalize by default
        seed=0,
        num_examples=8,
        shuffle=True,
        ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.test_samples=[]
        #iterate over speakers directories
        speakers=os.listdir(path)
        for s in speakers:
            if s in speakers_discard:
                continue
            elif s in speakers_test:
                self.test_samples.extend(glob.glob(os.path.join(path,s,"*.wav")))
            else:
                continue

        self.test_samples = sorted(self.test_samples)
        assert len(self.test_samples)>=num_examples , "error in dataloading: not enough examples"

        if num_examples > 0:
            if shuffle:
                self.test_samples=random.sample(self.test_samples,num_examples)
            else:
                self.test_samples=self.test_samples[:num_examples]

        self.segment_length=int(segment_length)
        self.fs=fs

        self.normalize=normalize
        if self.normalize:
            raise NotImplementedError("normalization not implemented yet")

        self.test_audio=[]
        self.filenames=[]
        self._fs=[]
        for file in self.test_samples:
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            assert samplerate==self.fs, "wrong sampling rate"
            assert len(data.shape)==1, "wrong number of channels"

            L=len(data)

            if self.segment_length > 0:
                #crop or pad to get to the right length
                if L>self.segment_length:
                    #get random segment
                    idx=np.random.randint(0,L-self.segment_length)
                    segment=data[idx:idx+self.segment_length]
                elif L<=self.segment_length:
                    #pad with zeros to get to the right length randomly
                    idx=np.random.randint(0,self.segment_length-L)
                    #copy segment to get to the right length
                    segment=np.pad(data,(idx,self.segment_length-L-idx),'wrap')
            else:
                segment = data

            self.test_audio.append(segment) #use only 50s

    def __getitem__(self, idx):
        return self.test_audio[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)

class VCTKTestPaired(torch.utils.data.Dataset):
    def __init__(self,
        fs=16000,
        segment_length=65536,
        path="", #path to the dataset
        speakers_discard=[], #list of speakers to discard
        speakers_test=[], #list of speakers to use for testing, discarded here
        normalize=False,  #to normalize or not. I don't normalize by default
        seed=0,
        num_examples=8,
        shuffle=True,
        ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.test_samples=[]
        self.rir_samples=[]
        #iterate over speakers directories
        speakers=os.listdir(os.path.join(path, "clean"))
        for s in speakers:
            if s in speakers_discard:
                continue
            elif s in speakers_test:
                new_samples=glob.glob(os.path.join(path,"clean",s,"*.wav"))
                self.test_samples.extend(new_samples)
                for file in new_samples:
                    id=os.path.splitext(os.path.basename(file))[0]
                    self.rir_samples.append(os.path.join(path,"rir",s,id+".wav"))
            else:
                continue
            

        #self.test_samples = sorted(self.test_samples)
        assert len(self.test_samples)>=num_examples , "error in dataloading: not enough examples"
        assert len(self.test_samples)==len(self.rir_samples), "error in dataloading: the rir files are not paired"

        if num_examples > 0:
            self.test_samples=self.test_samples[:num_examples]

        self.segment_length=int(segment_length)
        self.fs=fs

        self.normalize=normalize
        if self.normalize:
            raise NotImplementedError("normalization not implemented yet")

        self.test_audio=[]
        self.test_rir=[]
        self.filenames=[]
        self._fs=[]
        for i in range(len(self.test_samples)):
            file=self.test_samples[i]
            file_rir=self.rir_samples[i]

            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data_rir, samplerate_rir = sf.read(file_rir)
            assert samplerate==self.fs, "wrong sampling rate"
            assert samplerate_rir==self.fs, "wrong sampling rate"
            assert len(data.shape)==1, "wrong number of channels"
            assert len(data_rir.shape)==1, "wrong number of channels"

            direct_path=np.argmax(np.abs(data_rir))
            data_rir=data_rir[direct_path:]

            data_rir /= np.abs(data_rir).max() 

            L=len(data)
            segment = data
            self.test_audio.append(segment) 
            self.test_rir.append(data_rir) 


    def __getitem__(self, idx):
        return self.test_audio[idx], self.test_rir[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)



class BinauralVCTKTestPaired(torch.utils.data.Dataset):
    def __init__(self,
        fs=16000,
        segment_length=65536,
        path="", #path to the dataset
        speakers_discard=[], #list of speakers to discard
        speakers_test=[], #list of speakers to use for testing, discarded here
        normalize=False,  #to normalize or not. I don't normalize by default
        seed=0,
        num_examples=8,
        shuffle=True,
        rir_df_path ='/shared/cycle1_biu_gannot_prj/datsets/hrtf_db/csvs/HRTF_train_VAE_wsj0_10k_mp_clean.csv',
        sofa_root_path ='/shared/cycle1_biu_gannot_prj/datsets/hrtf_db/hrtf_10k_mp'
        ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.rir_df = pd.read_csv(rir_df_path)
        self.rir_df = self.rir_df[self.rir_df['rt_60'].between(1.2, 1.9)] #high rev for testing only
        self.test_samples=[]
        self.rir_samples=[]
        self.hrtf_samples=[]
        self.sofa_root_path = sofa_root_path

        #iterate over speakers directories
        speakers=os.listdir(path)
        for s in speakers:
            if s in speakers_discard:
                continue
            elif s in speakers_test:
                new_samples=glob.glob(os.path.join(path,s,"*.wav"))
                self.test_samples.extend(new_samples)
                for file in new_samples:
                    rir_exists= False
                    while not rir_exists:
                        line = self.rir_df.sample(n=1, replace=True)
                        try:
                            rir_path = self.get_path(line['hrir_rev_1_path'].item())
                            hrtf_path = self.get_path(line['hrir_zero_1_path'].item())
                        except:
                            rir_path = line['hrir_rev_1_path'].item()
                            hrtf_path = line['hrir_zero_1_path'].item()
                        rir_exists = Path(rir_path).exists() and Path(hrtf_path).exists()
                    self.hrtf_samples.append(hrtf_path)
                    self.rir_samples.append(rir_path)
            else:
                continue
            

        #self.test_samples = sorted(self.test_samples)
        assert len(self.test_samples)>=num_examples , "error in dataloading: not enough examples"
        assert len(self.test_samples)==len(self.rir_samples), "error in dataloading: the rir files are not paired"

        if num_examples > 0:
            self.test_samples=self.test_samples[:num_examples]

        self.segment_length=int(segment_length)
        self.fs=fs

        self.normalize=normalize
        if self.normalize:
            raise NotImplementedError("normalization not implemented yet")

        self.test_audio=[]
        self.test_rir=[]
        self.test_h = []
        self.test_hrtf = []
        self.filenames=[]
        self._fs=[]
        for i in range(len(self.test_samples)):
            file=self.test_samples[i]
            file_rir=self.rir_samples[i]
            file_hrtf = self.hrtf_samples[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            assert samplerate==self.fs, "wrong sampling rate"
            assert len(data.shape)==1, "wrong number of channels"


            data_rir = self.conv_h(file,file_rir)
            L_rir = data_rir.shape[-1]
            L_clean = data.shape[0]

            if L_rir > L_clean:
                data = np.pad(data, (0, L_rir - L_clean), mode="constant")
            elif L_clean > L_rir:
                data = data[:L_rir]
            assert data.shape[0] == data_rir.shape[-1]
            segment_rir, shared_idx = self.fix_length_2d(data_rir)
            segment, _ = self.fix_length(data, shared_idx)
            self.test_rir.append(segment_rir)
            self.test_audio.append(segment) 
            h, samplerate = sf.read(file_rir)
            self.test_h.append(h)
            hrtf, samplerate = sf.read(file_hrtf)
            self.test_hrtf.append(hrtf)
            
    def get_path(self, org_path):
        org_path = Path(org_path)
        new_root = Path(self.sofa_root_path)
        try:
            idx = org_path.parts.index("hrtf_10k_mp")
            relative = Path(*org_path.parts[idx:])  # sofas/...
        except ValueError:
            raise ValueError(f"'sofas' not found in path: {org_path}")

        new_path = new_root.parent / relative
        if new_path.exists():
            return new_path

        else:
            return None

    def fix_length_2d(self, segment, idx=None):
        # segment: [C, T]
        C, L = segment.shape
        
        if L > self.segment_length:
            # If no index provided, pick one. Use this same idx for the target!
            if idx is None:
                idx = np.random.randint(0, L - self.segment_length)
            segment = segment[:, idx : idx + self.segment_length]
        
        elif L < self.segment_length:
            # Force static zero padding at the end for both
            pad_width = self.segment_length - L
            segment = np.pad(segment, ((0, 0), (0, pad_width)), mode='constant')
            idx = 0 # Offset is zero in padding mode
            
        return segment, idx

    def fix_length(self, segment, idx=None):
        # segment: [T]
        L = len(segment)
        
        if L > self.segment_length:
            if idx is None:
                idx = np.random.randint(0, L - self.segment_length)
            segment = segment[idx : idx + self.segment_length]
            
        elif L < self.segment_length:
            # Use constant zero padding to match the 2D version
            pad_width = self.segment_length - L
            segment = np.pad(segment, (0, pad_width), mode='constant')
            idx = 0
            
        return segment, idx
    
    def conv_h(self,wav_path,h_path):
        wav,fs = sf.read(wav_path)
        h,fs = sf.read(h_path)
        rend_L = signal.fftconvolve(wav,h[:,0])
        rend_R = signal.fftconvolve(wav,h[:,1])
        stereo_audio_h = np.concatenate((rend_L[:,np.newaxis],rend_R[:,np.newaxis]),axis=1)
        return stereo_audio_h.T

    def __getitem__(self, idx):
        return self.test_audio[idx], self.test_rir[idx], self.filenames[idx],self.test_h[idx],self.test_hrtf[idx]

    def __len__(self):
        return len(self.test_samples)
    
class BinauralVCTKTrain(torch.utils.data.IterableDataset):
    def __init__(self,
        fs=16000,
        segment_length=65536,
        path="", #path to the dataset
        speakers_discard=[], #list of speakers to discard
        speakers_test=[], #list of speakers to use for testing, discarded here
        normalize=False,  #to normalize or not. I don't normalize by default
        seed=0,
        shuffle=True,
        rir_df_path ='/shared/cycle1_biu_gannot_prj/datsets/hrtf_db/csvs/HRTF_train_VAE_wsj0_10k_mp_clean.csv',
        sofa_root_path ='/shared/cycle1_biu_gannot_prj/datsets/hrtf_db/hrtf_10k_mp'
        ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.rir_df = pd.read_csv(rir_df_path)
        self.rir_df = self.rir_df[self.rir_df['rt_60'].between(1.2, 1.9)] #high rev for testing only
        self.train_samples=[]
        self.hrtf_samples=[]
        self.sofa_root_path = sofa_root_path

        #iterate over speakers directories
        speakers=os.listdir(path)
        for s in speakers:
            if s in speakers_discard:
                continue
            new_samples=glob.glob(os.path.join(path,s,"*.wav"))
            self.train_samples.extend(new_samples)
            for file in new_samples:
                rir_exists= False
                while not rir_exists:
                    line = self.rir_df.sample(n=1, replace=True)
                    try:
                        hrtf_path = self.get_path(line['hrir_zero_1_path'].item())
                    except:
                        hrtf_path = line['hrir_zero_1_path'].item()
                    rir_exists = Path(hrtf_path).exists()
                self.hrtf_samples.append(hrtf_path)
            else:
                continue
            
        self.segment_length=int(segment_length)
        self.fs=fs

        self.normalize=normalize
        if self.normalize:
            raise NotImplementedError("normalization not implemented yet")
            
    def get_path(self, org_path):
        org_path = Path(org_path)
        new_root = Path(self.sofa_root_path)
        try:
            idx = org_path.parts.index("hrtf_10k_mp")
            relative = Path(*org_path.parts[idx:])  # sofas/...
        except ValueError:
            raise ValueError(f"'sofas' not found in path: {org_path}")

        new_path = new_root.parent / relative
        if new_path.exists():
            return new_path

        else:
            return None

    def fix_length_2d(self, segment, idx=None):
        # segment: [C, T]
        C, L = segment.shape
        
        if L > self.segment_length:
            # If no index provided, pick one. Use this same idx for the target!
            if idx is None:
                idx = np.random.randint(0, L - self.segment_length)
            segment = segment[:, idx : idx + self.segment_length]
        
        elif L < self.segment_length:
            # Force static zero padding at the end for both
            pad_width = self.segment_length - L
            segment = np.pad(segment, ((0, 0), (0, pad_width)), mode='constant')
            idx = 0 # Offset is zero in padding mode
            
        return segment, idx

    def fix_length(self, segment, idx=None):
        # segment: [T]
        L = len(segment)
        
        if L > self.segment_length:
            if idx is None:
                idx = np.random.randint(0, L - self.segment_length)
            segment = segment[idx : idx + self.segment_length]
            
        elif L < self.segment_length:
            # Use constant zero padding to match the 2D version
            pad_width = self.segment_length - L
            segment = np.pad(segment, (0, pad_width), mode='constant')
            idx = 0
            
        return segment, idx
    
    def conv_h(self,wav_path,h_path):
        wav,fs = sf.read(wav_path)
        h,fs = sf.read(h_path)
        rend_L = signal.fftconvolve(wav,h[:,0])
        rend_R = signal.fftconvolve(wav,h[:,1])
        stereo_audio_h = np.concatenate((rend_L[:,np.newaxis],rend_R[:,np.newaxis]),axis=1)
        return stereo_audio_h.T

    def __len__(self):
        return len(self.train_samples)
    

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        base_seed = torch.initial_seed() % (2**32)
        seed = base_seed + 1000 * rank + worker_id

        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        while True:
            num = py_rng.randint(0, len(self.train_samples) - 1)

            file = self.train_samples[num]
            hrtf = self.hrtf_samples[num]

            segment = self.conv_h(file, hrtf)  # (C, L)

            C, L = segment.shape

            if L > self.segment_length:
                idx = np_rng.integers(0, L - self.segment_length + 1)
                segment = segment[:, idx:idx + self.segment_length]

            elif L < self.segment_length:
                pad_total = self.segment_length - L
                idx = np_rng.integers(0, pad_total + 1)

                segment = np.pad(
                    segment,
                    pad_width=((0, 0), (idx, pad_total - idx)),
                    mode="constant",
                    constant_values=0,
                )

            yield torch.from_numpy(segment).float()

class BinauralToMonoVCTKTrain(BinauralVCTKTrain):
    def __init__(self, fs=16000, segment_length=65536, path="", speakers_discard=[], speakers_test=[], normalize=False, seed=0, shuffle=True, rir_df_path='/shared/cycle1_biu_gannot_prj/datsets/hrtf_db/csvs/HRTF_train_VAE_wsj0_10k_mp_clean.csv', sofa_root_path='/shared/cycle1_biu_gannot_prj/datsets/hrtf_db/hrtf_10k_mp'):
        super().__init__(fs, segment_length, path, speakers_discard, speakers_test, normalize, seed, shuffle, rir_df_path, sofa_root_path)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        base_seed = torch.initial_seed() % (2**32)
        seed = base_seed + 1000 * rank + worker_id

        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        while True:
            num = py_rng.randint(0, len(self.train_samples) - 1)

            file = self.train_samples[num]
            hrtf = self.hrtf_samples[num]

            segment = self.conv_h(file, hrtf)  # (C, L)
            mono,_ = sf.read(file)

            C, L = segment.shape
            L_mono = mono.shape[0]
            if L > L_mono:
                mono = np.pad(mono, (0, L - L_mono), mode="constant")
            elif L_mono > L:
                mono = mono[:L]

            L = segment.shape[-1]
            assert mono.shape[0] == L

            if L > self.segment_length:
                idx = np_rng.integers(0, L - self.segment_length + 1)
                segment = segment[:, idx:idx + self.segment_length]
                mono = mono[idx:idx + self.segment_length]

            elif L < self.segment_length:
                pad_total = self.segment_length - L
                idx = np_rng.integers(0, pad_total + 1)

                segment = np.pad(segment, ((0, 0), (idx, pad_total - idx)), mode="constant")
                mono = np.pad(mono, (idx, pad_total - idx), mode="constant")

            yield torch.from_numpy(segment).float(), torch.from_numpy(mono).float().unsqueeze(0)

if __name__ =="__main__":
    segment_length= 65536
    fs= 16000
    path='/shared/cycle1_biu_gannot_prj/datsets/vctk/VCTK-Corpus-0.92/wav16k'
    speakers_discard= ["p280", "p315"]
    speakers_test= ["p226", "p287"]
    normalize= False
    num_examples= 16
    ds = BinauralVCTKTrain(fs=fs,segment_length=segment_length,path=path,speakers_discard=speakers_discard,speakers_test=speakers_test,normalize=normalize)
    for d in ds:
        print(d.shape) #C,L (2,65536)
        break

'''

srun --partition=sandbox \
     --container-image=docker://cr.me-west1.nebius.cloud#i00d60vg8wj9bggce3:diffusion_train \
     --gres=gpu:1 --qos=sandbox_owner_90 --time=02:00:00 nvidia-smi
    
     

srun --partition=sandbox \
  --container-image=docker://cr.me-west1.nebius.cloud#i00d60vg8wj9bggce3/diffusion_train:v1 \
  --gres=gpu:1 \
  --qos=sandbox_owner_90 \
  --time=02:00:00 \
  --container-mounts=/home/users/$USER:/home/users/$USER,/shared/cycle1_biu_gannot_prj:/shared/cycle1_biu_gannot_prj \
  --pty /bin/bash

'''