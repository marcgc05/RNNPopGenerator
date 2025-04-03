import midiGen, tokenGen
from tokenReformat import tokens_to_musicxml

#Run this to generate midi then tokens:
#ensure that the POP909 data is in the repo folder.

midiGen.GenerateMidiTxt()
tokenGen.GenerateTokensTxt()
