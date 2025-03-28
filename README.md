Timeline 

Primary idea: Gather pop music and create each segment separately (verse, refrain, etc.), since pop music is super formulaic. We could then take this further and create variations in each segment, much like no pop music is the exact same structure wise.  

 

Download MusicXML files from sites like MuseScore 

Pop909 database 

https://github.com/music-x-lab/POP909-Dataset 

Feed songs through librosa for individual segments 

Tokenize songs and time stamp them 

Use an RNN, and train it on the MusicXML data 

PyTorch library for RNN (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) 

Use music21 to read MusicXML files 

Use the read data to create tokens 

It should learn to predict the next note based on probabilities 

It can also learn more specific concepts like Rythm, style, etc. 

Also, will be able to retain more information than a Markov chain 

An RNN (Recurrent Neural Network) is a type of neural network designed to process sequences of data, where the order and context of previous elements matter. 

Unlike traditional neural networks that treat each input independently, RNNs have a "memory" of what they've seen before. As they process each element in a sequence (like a musical note), they use both the current input and what they remember from earlier in the sequence to make predictions. 
