import json
import numpy as np
from tensorflow import keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    
    def __init__(self, model_path="music_model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
            
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
        
               
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        
        # create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        
        # map seed to integer
        seed = [self._mappings[symbol] for symbol in seed]
        
        
        for step in range(num_steps):
            
            # limit the seed to the max_sequence_length
            seed = seed[-max_sequence_length:]
            
            # one-hot encoding the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]
            
            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.5] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # update seed
            seed.append(output_int)
            
            # map integer to encoding
            output_symbol = [k for k, value in self._mappings.items() if value==output_int][0]
            
            # check whether we are at the end of a melody
            if output_symbol== "/":
                break
            
            # update the melody
            melody.append(output_symbol)
            
        return melody           
            
      
    
    def _sample_with_temperature(self, probabilities, temperature):
        # temperature -> infinity   , this means more random(more unpredictable)
        # temperature -> 0   ,this means more deterministic
        # temperature = 1
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))     #[0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilities)
        
        return index    
    
    
    
    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        
        # create a musci21 stream
        stream = m21.stream.Stream()
        
        # parse all the symbols in the melody and create note/rest objects
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1
        
        for i, symbol in enumerate(melody):
            
            # handle case in which we habe a node/rest
            if symbol != "_" or i+1==len(melody):
                
                #ensure we are dealing with the note/rest beyond the first one
                if start_symbol is not None:
                    
                    quarter_length_duration = step_duration * step_counter      # 0.25 * 4 = 1
                    
                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength = quarter_length_duration)
                    
                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)
                    
                    stream.append(m21_event) 
                    
                    # reset the step counter
                    step_counter = 1   
                    
                start_symbol = symbol
                    
            # handle case in which we have a prolongation sigh "_"
            else:
                step_counter+=1
        
        # write the m21 stream to a midi file
        stream.write(format, file_name)
    
        
        
          
if __name__ == "__main__":
    mg = MelodyGenerator()     
    seed = "72 _ _ _ 72 _ _ _ 79 _ _ _ 79 _ _ _ 81 _ _ _ 81 _ _ _ 79 _ _ _ 79 _ _ _ 77 _ _ _ 77 _ _ _ 76 _ _ _ 76 _ _ _ 74 _ _ _ 74 _ _ 76 72 _ _ _ _ _ _ _"      # example of 9th song of dataset folder
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.6)
    print(melody)
    mg.save_melody(melody)
        
            