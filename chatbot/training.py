import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
import tensorflow as tf
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']
data_file = open('E:\MY Work\AI Track\Sync\ChatBot\intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])



# shuffle our features and turn into np.array
random.shuffle(training)
max_length = max(max(len(sub_list) for sub_list in sublist) for sublist in training)

# Pad the sub-lists with zeros to match the maximum length
padded_list = [[sub_list + [0] * (max_length - len(sub_list)) for sub_list in sublist] for sublist in training]

training = np.array(padded_list)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

#In TensorFlow's Keras API, tf.keras.layers.Dropout(0.5) represents a layer that randomly sets a fraction of input units to 0 at each update during training time. The 0.5 parameter specifies the probability of dropping out each input unit, which in this case is 50%.
#The purpose of using dropout is to prevent overfitting, which occurs when a model becomes too specialized to the training data and performs poorly on new, unseen data. By randomly dropping out units, dropout helps to reduce the complex co-adaptations between neurons, forcing the network to learn more robust features and improving its generalization ability.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')
