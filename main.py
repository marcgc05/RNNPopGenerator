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


""""
midiGen.GenerateMidiTxt()
tokenGen.GenerateTokensTxt()
#Generates a vocab file in the POP909 directory
vocabTools.GenerateVocabTxt()
"""

#generation.generate_and_save_musicxml()

#training.trainAndSaveModel()

#generating 5 pieces
for i in range(20):
    generation.generate_and_save_musicxml()
