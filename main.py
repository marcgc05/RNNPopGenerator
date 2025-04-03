import midiGen, tokenGen

#Run this to generate midi then tokens:
#ensure that the POP909 data is in the repo folder.

midiGen.GenerateMidiTxt()
tokenGen.GenerateTokensTxt()
#Generates a vocab file in the POP909 directory
vocabTools.GenerateVocabTxt()