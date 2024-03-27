import os
import json
import music21 as m21
from music21 import *
import numpy as np
from tensorflow import keras

KERN_DATASET_PATH = "Germany/test_songs"
SAVE_DIR = "test_dataset"
SINGLE_FILE_DATASET = "file_test_dataset"
MAPPING_PATH = "_test_mapping.json"
SEQUENCE_LENGTH = 64

# Duration are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25,   # 16th note
    0.5,   # 8th note
    0.75,
    1,   # quarter note
    1.5,
    2,   # half note
    3,
    4    # whole note
]

def load_songs_in_kern(dataset_path):
    
    songs = []
    
    # go through all the files and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file [-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    
    # estimate key using Music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    # get interval for transposition. e.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song


def encode_song(song, time_step=0.25):
    # p = 60, d=1.0 -> [60, "_","_","_"]

    encoded_song = []

    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi   #60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol="r"
        
        # convert the note/rest into time series notation    
        steps = int(event.duration.quarterLength / time_step)
        
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
                
    # cast the encoded song to a string
    encoded_song = " ".join(map(str,encoded_song))
    
    return encoded_song


def preprocess(dataset_path):

    # Load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    
    for i, song in enumerate(songs):
        
        # Filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        
        
        # Transpose songs to Cmaj or Amin
        song = transpose(song)
        
        # encode song with music time series representation
        encoded_song = encode_song(song)
        
        # Save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    
    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    
    songs = songs[:-1]  
      
    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
        
    return songs



def create_mapping(songs, mapping_path):
    mappings = {}
    
    # Identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # Create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    # Save vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings,fp, indent=4)


def convert_song_to_int(songs):
    
    int_songs = []
    
    # load the mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    
    # cast songs string to list
    songs = songs.split()
    
    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
        
    return int_songs


def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14, ...] -> i: [11, 12], t: 13; i: [12, 13], t: 14
    
    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_song_to_int(songs)
    
    # generate the training sequences
    # 100 symbols, 64 seq_len, 100-64 = 36 sequences
    inputs = []
    targets = []
    
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    
    # one-hot encode the sequences 
    # inputs: (no. of sequences, sequence_length, vocabulary size)
    # [[0,1,2], [1,1,2]] -> [[[1,0,0], [0,1,0], [0,0,1]], []]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets,dtype='uint8')
    
    print(f"There are {len(inputs)} sequences.")
    
    return inputs, targets
    

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR,SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    print(len(inputs))
    print(len(targets))


if __name__ == "__main__":
    main()
    
    # load songs
    # songs = load_songs_in_kern(KERN_DATASET_PATH)
    # print(f"Loaded {len(songs)} songs.")
    # song = songs[0]
    
    # preprocess(KERN_DATASET_PATH)
    # print(f"Has acceptable duration? {has_acceptable_durations(song, ACCEPTABLE_DURATIONS)}")
    
    # transposed_song = transpose(song)
    
    # song.show()
    # transposed_song.show()
    
    songs = create_single_file_dataset(SAVE_DIR,SINGLE_FILE_DATASET, SEQUENCE_LENGTH)


