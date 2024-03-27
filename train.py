from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from tensorflow import keras
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import numpy as np
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow import keras

OUTPUT_UNITS = 40
NUM_UNITS = [256]
LOSS= "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 55     # 40 to 60
BATCH_SIZE = 64
SAVE_MODEL_PATH = "music_model.h5"

def build_model(output_units, num_units, loss, learning_rate):
    
    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    
    output = keras.layers.Dense(output_units, activation="softmax")(x)
    
    model = keras.Model(input, output)
    
    # compile model
    model.compile(loss=loss, 
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=["accuracy"])
    
    model.summary()
    
    return model

def train_test_split_data():
    # generate the data
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    
    # split the data into train and test sets
    input_train, input_test, target_train, target_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
    
    return input_train, input_test, target_train, target_test

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    
    # get train and test data
    input_train, input_test, target_train, target_test = train_test_split_data()
    
    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)
    
    # train the model
    my_model = model.fit(input_train, target_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(input_test, target_test))
    
    # save the model
    model.save(SAVE_MODEL_PATH)
    
    plot_model(my_model,

           to_file='My_model_plot.png',

           show_shapes=True,

           show_layer_names=True)
    
    
    # Get training and test loss histories
    training_loss = my_model.history['loss']
    test_loss = my_model.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Get training and test accuracy 
    training_accuracy = my_model.history['accuracy']
    test_accuracy = my_model.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_accuracy) + 1)

    # Visualize accuracy -
    plt.plot(epoch_count, training_accuracy, 'r--')
    plt.plot(epoch_count, test_accuracy, 'b-')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    train()