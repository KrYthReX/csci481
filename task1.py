import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuration
config = {
    "sample_rate": 16000,
    "n_mels": 40,
    "batch_size": 64,
    # "batch_size": {"values": [16, 32, 64]},
    #batch size sweep doesnt work automatically, manual info
    #batch 16 - best 85.76% at epoch 9 w/ .001 LR, optim adam
    #batch 32 - best 83.10% .001 LR, sgd, epcoh 10
    #batch 64 best 86.47% .001 adam, epoch 14
    #batch 128 best 85.34 .001 adam epoch 16
    #batch 256 best 85.12 .001 adam epoch 19
    #batch 512 best 85.32 .001
    "num_epochs": 80,
    "learning_rate": 0.0001,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

#location to save models
save_dir = "./saved_models"

# Audio preprocessing
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=config["sample_rate"],
    n_fft=400, #400/16000 = 25ms
    hop_length=160, #160/16000 = 10ms
    n_mels=config["n_mels"] #40 mels
)

class SpeakerDataset(Dataset):
    def __init__(self, wav_dir, max_files=None):
        self.data = []
        self.speakers = {}
        
        #load in the speaker file and the speaker id
        for fname in os.listdir(wav_dir):
            if fname.endswith(".wav"):
                spk_file = fname.replace(".wav", ".spkid.txt")
                with open(os.path.join(wav_dir, spk_file)) as f:
                    speaker = f.read().strip()
                if speaker not in self.speakers:
                    self.speakers[speaker] = []
                self.data.append((os.path.join(wav_dir, fname), speaker))
                self.speakers[speaker].append(len(self.data)-1)

        # Filter speakers with <2 files
        self.valid_speakers = [s for s in self.speakers if len(self.speakers[s]) >= 2]
        if max_files:
            self.data = self.data[:max_files]

    def __len__(self):
        return 10000  # Fixed number of pairs per epoch
        # return self.__len__()

    def __getitem__(self, idx):
        #even idx: two audio samples are chosen from the same speaker
        #odd: two different speakers are chosen so model learns how diff speakers look
        
        if idx % 2 == 0:  #same speaker
            speaker = random.choice(self.valid_speakers)
            i1, i2 = random.sample(self.speakers[speaker], 2)
        else:  #diff speaker
            s1, s2 = random.sample(self.valid_speakers, 2)
            i1 = random.choice(self.speakers[s1])
            i2 = random.choice(self.speakers[s2])
            
        file1, _ = self.data[i1]
        file2, _ = self.data[i2]
        
        #load two given audio files
        wave1, samp1 = torchaudio.load(file1) #waveform, sample rate
        wave2, samp2 = torchaudio.load(file2)
        
        #convert the files into mel spectrograms
        spec1 = mel_transform(wave1).squeeze().T  #time, n_mels
        spec2 = mel_transform(wave2).squeeze().T
        
        #1 is the same spealker, 0 if different
        label = torch.tensor(1.0 if idx % 2 == 0 else 0.0)
        
        return spec1, spec2, label


def collate_fn(batch):
    specs1, specs2, labels = zip(*batch)
    
    #finds max length of all spectrograms and pads it to same length
    max_len = max(s.shape[0] for s in specs1 + specs2)
    
    def pad_spectrogram(spec):
        pad_size = max_len - spec.shape[0]
        return torch.nn.functional.pad(spec, (0, 0, 0, pad_size))
    
    specs1 = torch.stack([pad_spectrogram(s) for s in specs1])
    specs2 = torch.stack([pad_spectrogram(s) for s in specs2])
    labels = torch.stack(labels)
    
    return specs1, specs2, labels

class SpeakerVerificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(config["n_mels"], 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(0.2), 

            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(0.2), 

            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
            
            
        )
        
        #classification head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.2), #dropout before linear 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        #input shape: [batch, time, n_mels]
        x1 = x1.permute(0, 2, 1)  # [batch, n_mels, time]
        x2 = x2.permute(0, 2, 1)
        
        emb1 = self.encoder(x1).squeeze(-1)
        emb2 = self.encoder(x2).squeeze(-1)
        
        #absolute difference between embeddings
        diff = torch.abs(emb1 - emb2)
        return self.head(diff).squeeze()


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    
    for specs1, specs2, labels in tqdm(train_loader):
        specs1 = specs1.to(config["device"])
        specs2 = specs2.to(config["device"])
        labels = labels.to(config["device"])
        
        optimizer.zero_grad()
        outputs = model(specs1, specs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float() #same speaker
        correct += (preds == labels).sum().item()
    
    acc = correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc



def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for specs1, specs2, labels in loader:
            specs1 = specs1.to(config["device"])
            specs2 = specs2.to(config["device"])
            labels = labels.to(config["device"])
            
            outputs = model(specs1, specs2)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
    
    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

if __name__ == "__main__":
    
    best_val = 0.8467 #arbitrary best acc (84.76%)
    # best_val = 0.0

    train_dataset = SpeakerDataset("/home/downina3/CSCI481/fin_project/task1/train_fx")
    dev_dataset = SpeakerDataset("/home/downina3/CSCI481/fin_project/task1/dev_fx")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size"], collate_fn=collate_fn, num_workers=4)

    model = SpeakerVerificationModel().to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()

    #training loop
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, dev_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}\n")
        # print(f"val acc 2:{val_acc:.2f} ")
        print(f"val accuracy ", val_acc) 
        # 
        if val_acc > best_val:
            
            best_model_path = f"{save_dir}/best_mod_val_acc{val_acc:.4f}.pth"
            best_val = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"model saved with {val_acc:.4f} val acc.")
