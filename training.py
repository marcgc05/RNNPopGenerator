import midiGen, tokenGen, vocabTools, MusicRNN, torch
import os
from vocabTools import load_vocab
from TokenDatasetGen import TokenDatasetGen
from MusicRNN import MusicRNN
from MusicRNN import train_model


def trainAndSaveModel():

    vocab, token_to_id, id_to_token = load_vocab()
    
    dataset = TokenDatasetGen("POP909", token_to_id)


    model = MusicRNN(
    vocab_size=len(vocab),
    embedding_dim=128, 
    hidden_dim=256,    
    num_layers=2,
    dropout=0.5
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Train the model
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=1,  
        batch_size=64,
        learning_rate=0.001
    )

    torch.save(model.state_dict(), 'trained_model.pth')


