#this class produces training data objects, turning tokens into ID's, and builds overlapping windows for training. 
import os
import torch
from torch.utils.data import Dataset
class TokenDatasetGen(Dataset):
    
    #root directory will be c:\Users\marcg\OneDrive\Documents\RNNPopGenerator\POP909 (for me)
    def __init__(self, root_directory, token_to_id, seq_len = 64):
        self.root_directory = root_directory
        self.token_to_id = token_to_id
        self.seq_len = seq_len
        self.samples = []
        self.samples_gen()
        
    def samples_gen(self):

        folders = [f for f in os.listdir(self.root_directory) 
              if os.path.isdir(os.path.join(self.root_directory, f)) 
              and f.isdigit()]  
    
        folders.sort(key=int)  # Sort numerically

    # Use the filtered folders list in the loop instead of os.listdir
        for folder_name in folders:  # Changed this line to use folders instead of os.listdir
            print(folder_name)
            folder_path = os.path.join(self.root_directory, folder_name)
            print(folder_path)
            tokens_file = os.path.join(folder_path, "tokens.txt")
            print(tokens_file)

            with open(tokens_file) as f:
                tokens = [line.strip() for line in f if line.strip()]
            
            ids = [self.token_to_id[t] for t in tokens if t in self.token_to_id]

            for i in range(len(ids) - self.seq_len):
                x_seq = ids[i : i + self.seq_len]
                y_seq = ids[i + 1 : i + self.seq_len + 1]
                self.samples.append((x_seq, y_seq))

    def __len__(self):
        return len(self.samples)

    #this is gonna be used by PyTorch's DataLoader 
    def __getitem__(self, idx):
        x_seq, y_seq = self.samples[idx]
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y_seq, dtype=torch.long)




