import midiGen, tokenGen, vocabTools, MusicRNN, torch
import os
from vocabTools import load_vocab
from TokenDatasetGen import TokenDatasetGen
from MusicRNN import MusicRNN
from MusicRNN import generate_music
from MusicRNN import train_model
import tokenReformat
import shutil

# After training your model:

def generate_and_save_musicxml():
    
    answer = input("Do you want to update the model to it's last saved checkpoint? y/n")

    if (answer == "y"):
        if os.path.exists("checkpoint.pth"):
            shutil.copyfile("checkpoint.pth", "trained_model.pth")
            print("Model updated: checkpoint.pth → trained_model.pth")
        else:
            print("checkpoint.pth not found. No update performed.")
    
    vocab, token_to_id, id_to_token = load_vocab()
    
    dataset = TokenDatasetGen("C:/Users/natha/OneDrive/Documents/RNNPopGen/RNNPopGenerator/POP909", token_to_id)

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

    print("Generating music...")
    
    generated_tokens = generate_music(
        model,
        token_to_id,
        id_to_token,
        seed_tokens=['START'],
        max_length=500,
        temperature=0.7
    )
    
    # Save tokens to a text file
    token_file = 'generated_piece.txt'
    with open(token_file, 'w') as f:
        for token in generated_tokens:
            f.write(token + '\n')

    # Convert the saved tokens to MusicXML using the function from tokenReformat
    xml_file = 'generated_piece.musicxml'
    tokenReformat.tokens_to_musicxml(token_file, xml_file)
    print(f"Music saved as {xml_file}")