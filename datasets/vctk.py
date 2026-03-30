import os
import numpy as np
import torch
import random
import glob
import soundfile as sf
import pandas as pd
from pathlib import Path
from scipy import signal

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
        rir_df_path ='/home/workspace/yoavellinson/binaural_TSE_Gen/csvs/HRTF_test_VAE_wsj0_1k_mp.csv',
        ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.rir_df = pd.read_csv(rir_df_path)
        self.test_samples=[]
        self.rir_samples=[]
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
                        rir_path = line['hrir_rev_1_path'].item()
                        rir_exists = Path(rir_path).exists()
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
        self.filenames=[]
        self._fs=[]
        for i in range(len(self.test_samples)):
            file=self.test_samples[i]
            file_rir=self.rir_samples[i]

            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            assert samplerate==self.fs, "wrong sampling rate"
            assert len(data.shape)==1, "wrong number of channels"


            segment = self.fix_length(data)
            self.test_audio.append(segment) 

            data_rir = self.conv_h(file,file_rir)
            segment_rir = self.fix_length_2d(data_rir)
            self.test_rir.append(segment_rir) 
            
    def fix_length_2d(self,segment):
        #segment: C,T
        L=len(segment[0,:])
        #crop or pad to get to the right length
        if L>self.segment_length:
            #get random segment
            idx=np.random.randint(0,L-self.segment_length)
            segment=segment[:,idx:idx+self.segment_length]
        elif L<=self.segment_length:
            pad=np.zeros((2,self.segment_length-L))            
            segment = np.concatenate((segment,pad),axis=1)
        return segment

    def fix_length(self,segment):
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
        return segment
    
    def conv_h(self,wav_path,h_path):
        wav,fs = sf.read(wav_path)
        h,fs = sf.read(h_path)
        rend_L = signal.fftconvolve(wav,h[:,0])
        rend_R = signal.fftconvolve(wav,h[:,1])
        stereo_audio_h = np.concatenate((rend_L[:,np.newaxis],rend_R[:,np.newaxis]),axis=1)
        return stereo_audio_h.T

    def __getitem__(self, idx):
        return self.test_audio[idx], self.test_rir[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)
    

if __name__ =="__main__":
    segment_length= 65536
    fs= 16000
    path='/dsi/gannot-lab/gannot-lab1/datasets/VCTK/DS_10283_3443/VCTK-Corpus-0.92/wav16k'
    speakers_discard= ["p280", "p315"]
    speakers_test= ["p226", "p287"]
    normalize= False
    num_examples= 16
    ds = BinauralVCTKTestPaired(fs=fs,segment_length=segment_length,path=path,speakers_discard=speakers_discard,speakers_test=speakers_test,normalize=normalize,num_examples=num_examples)
