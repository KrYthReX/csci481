import os
import torch
import torchaudio
import random

class AudioPreprocessor: #set audio length to be 5 seconds
    def __init__(self, input_dir, output_dir, sample_rate=16000, clip_length=5.0):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        os.makedirs(self.output_dir, exist_ok=True)


    def _fix_length(self, waveform):
        num_samples = int(self.sample_rate * self.clip_length)  
        curr_length = waveform.shape[1]  

        if curr_length > num_samples:
            # waveform = waveform[:, :num_samples]  
            start = random.randint(0, curr_length - num_samples)
            waveform = waveform[:, start:start + num_samples]
            
        elif curr_length < num_samples:
            # repeat_amount = num_samples // curr_length
            # remainder = num_samples % curr_length
            # waveform = waveform.repeat(1, repeat_amount)
            # if remainder > 0:
            #     waveform = torch.cat((waveform, waveform[:, :remainder]), dim=1)

            while waveform.shape[1] < num_samples:
                remaining_duration = num_samples - waveform.shape[1]
                snippet = min(curr_length, remaining_duration)
                start = random.randint(0, curr_length - snippet)
                waveform = torch.cat((waveform, waveform[:, start:start + snippet]), dim=1)
            
        return waveform


    def _normalize(self, waveform):
        return (waveform - waveform.mean()) / (waveform.std() + 1e-6)


    def process_and_save(self):
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.input_dir, file_name)
                waveform, sr = torchaudio.load(file_path)

                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

                waveform = self._fix_length(waveform)
                waveform = self._normalize(waveform)

                # save_path = os.path.join(self.output_dir, file_name.replace(".wav", ".pt"))
                # torch.save(waveform, save_path)
                # print(f"Processed and saved: {save_path}")
                
                save_path = os.path.join(self.output_dir, file_name)
                torchaudio.save(save_path, waveform, self.sample_rate)
                print(f"saved at: {save_path}")
                
                
                
input_dir = "/home/downina3/CSCI481/fin_project/task1/dev"
output_dir = "/home/downina3/CSCI481/fin_project/task1/dev"
processor = AudioPreprocessor(input_dir, output_dir)
processor.process_and_save()
