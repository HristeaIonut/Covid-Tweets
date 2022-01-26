import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras.models
import pickle as pkl
import numpy as np

model = keras.models.load_model('models/model.h5')
tweet = input("Enter tweet: ")

result = np.argmax(model.predict([tweet]))
if result == 0:
    print("Classified \"" + tweet + "\" as negative")
elif result == 1:
    print("Classified \"" + tweet + "\" as neutral")
else:
    print("Classified \"" + tweet + "\" as positive")