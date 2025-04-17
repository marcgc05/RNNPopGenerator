import midiGen, tokenGen, vocabTools, MusicRNN, torch
import os
from vocabTools import load_vocab
from TokenDatasetGen import TokenDatasetGen
from MusicRNN import MusicRNN
from MusicRNN import generate_music
from MusicRNN import train_model
import tokenReformat

# After training your model:

def generate_and_save_musicxml():

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

    print("Generating music...")
    
    generated_tokens = generate_music(
        model,
        token_to_id,
        id_to_token,
        seed_tokens=['START'],
        max_length=500,
        temperature=0.7
    )
    

    #find the highest generated piece numerically
    generated_pieces = os.listdir("generated_songs")
    max_index = 0

    for piece in generated_pieces:
        if piece.startswith("generated_piece") and piece.endswith(".musicxml"):
            try:
                index = int(piece[len("generated_piece"):piece.rfind(".")])
                if index > max_index:
                    max_index = index
            except ValueError:
                continue

    #name them accordingly
    nextindex = max_index + 1
    token_file = f'generated_songs/generated_piece{nextindex}.txt'
    xml_file = f'generated_songs/generated_piece{nextindex}.musicxml'

    with open(token_file, 'w') as f:
        for token in generated_tokens:
            f.write(token + '\n')

    # Convert the saved tokens to MusicXML using the function from tokenReformat
    tokenReformat.tokens_to_musicxml(token_file, xml_file)
    print(f"Music saved as {xml_file}")