import midiGen, tokenGen, vocabTools, MusicRNN, torch
import os
from vocabTools import load_vocab
from TokenDatasetGen import TokenDatasetGen
from MusicRNN import MusicRNN
from MusicRNN import train_model
import training
import generation

#Run this to generate midi then tokens:
#ensure that the POP909 data is in the repo folder.
answer1 = input("Would you like to regenerate your midi, tokens, and vocab? y/n")

if (answer1 == "y"):
    midiGen.GenerateMidiTxt()
    tokenGen.GenerateTokensTxt()
    #Generates a vocab file in the POP909 directory
    vocabTools.GenerateVocabTxt()

#generation.generate_and_save_musicxml()

generation.generate_and_save_musicxml()