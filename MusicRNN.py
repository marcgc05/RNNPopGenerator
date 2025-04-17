import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from vocabTools import token_sequence_to_ids, ids_to_token_sequence
import os

class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
        super(MusicRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer to convert token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length)
        
        # Convert tokens to embeddings
        embeds = self.embedding(x)  # shape: (batch_size, sequence_length, embedding_dim)
        
        # Pass through RNN
        rnn_out, hidden = self.rnn(embeds, hidden)
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Pass through final linear layer
        output = self.fc(rnn_out)
        
        return output, hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

def train_model(model, train_dataset, valid_dataset=None, epochs=50, batch_size=32, 
                learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if valid_dataset:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    checkpoint_path = 'model_and_checkpoints/checkpoint.pth'

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    model = model.to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            hidden = model.init_hidden(x.size(0), device)

            optimizer.zero_grad()
            output, hidden = model(x, hidden)
            output = output.view(-1, output.size(-1))
            y = y.view(-1)
            loss = criterion(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}')

        # Validation
        if valid_dataset:
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)
                    hidden = model.init_hidden(x.size(0), device)
                    output, hidden = model(x, hidden)
                    output = output.view(-1, output.size(-1))
                    y = y.view(-1)
                    valid_loss += criterion(output, y).item()

            avg_valid_loss = valid_loss / len(valid_loader)
            print(f'Validation Loss: {avg_valid_loss:.4f}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

    return model

def generate_music(model, token_to_id, id_to_token, seed_tokens=['START'], max_length=500, temperature=0.7):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set to evaluation mode

    checkpoint = torch.load("model_and_checkpoints/trained_model.pth", map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict']) #If checkpoint contains extra data
    except KeyError:
        model.load_state_dict(checkpoint) #If checkpoint only contains model state_dict
    
    # Convert seed tokens to ids using helper function
    current_sequence = token_sequence_to_ids(seed_tokens, token_to_id)
    generated_ids = current_sequence.copy()  # Store all generated IDs

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input: take last 64 tokens (pad if necessary)
            input_seq = torch.tensor([current_sequence[-64:]]).long().to(device)
            if input_seq.size(1) < 64:
                padding = torch.zeros(1, 64 - input_seq.size(1)).long().to(device)
                input_seq = torch.cat([padding, input_seq], dim=1)
            
            # Run model for next token probabilities
            output, _ = model(input_seq)
            next_token_logits = output[0, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=0)
            next_token_id = torch.multinomial(next_token_probs, 1).item()
            
            generated_ids.append(next_token_id)
            current_sequence.append(next_token_id)
            
            # Use get() to safely access the generated token from id_to_token
            next_token = id_to_token.get(next_token_id)
            if next_token is None:
                print(f"Warning: Generated token id {next_token_id} not found in token mapping.")
                break
            # Stop if we've generated an "END" token.
            if next_token == 'END':
                break

    # Convert generated IDs back to tokens using helper function
    generated_music = ids_to_token_sequence(generated_ids, id_to_token)
    return generated_music