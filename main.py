import midiGen, tokenGen, vocabTools, MusicRNN, torch
import os
from vocabTools import load_vocab
from TokenDatasetGen import TokenDatasetGen
from MusicRNN import MusicRNN
from MusicRNN import train_model
import training
import generation
import shutil

#Run this to generate midi then tokens:
#ensure that the POP909 data is in the repo folder.
answer1 = input("Would you like to regenerate your midi, tokens, and vocab? y/n\n")

if (answer1 == "y"):
    midiGen.GenerateMidiTxt()
    tokenGen.GenerateTokensTxt()
    #Generates a vocab file in the POP909 directory
    vocabTools.GenerateVocabTxt()

#generation.generate_and_save_musicxml()

answer = input("Do you want to update the model to it's last saved checkpoint? y/n")

if (answer == "y"):
    if os.path.exists("model_and_checkpoints/checkpoint.pth"):
        shutil.copyfile("model_and_checkpoints/checkpoint.pth", "model_and_checkpoints/trained_model.pth")
        print("Model updated: checkpoint.pth → trained_model.pth")
    else:
        print("checkpoint.pth not found. No update performed.")

#generating 5 pieces
for i in range(5):
    generation.generate_and_save_musicxml()