#!/usr/bin/env python
# coding: utf-8

# **SPEECH EMOTION RECOGNITION**

# In[ ]:


import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the path to the main folder containing subfolders with audio files
data_folder = 'path_to_main_folder'

# Define a dictionary to map actor subfolder names to emotion labels
# You may need to create this mapping based on your dataset
emotion_labels = {
    'Actor_1': 'happy',
    'Actor_2': 'sad',
    'Actor_3': 'angry',
    # Add more actors/emotions as needed
}

# Initialize empty lists for data and labels
data = []
labels = []

# Loop through subfolders and load audio files
for actor_folder, emotion in emotion_labels.items():
    actor_path = os.path.join(data_folder, actor_folder)
    for audio_file in os.listdir(actor_path):
        if audio_file.endswith('.wav'):  # Adjust file extension if needed
            audio_path = os.path.join(actor_path, audio_file)

            # Load audio and extract features (MFCCs for this example)
            audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfccs_processed = np.mean(mfccs.T, axis=0)

            # Append data and label to the lists
            data.append(mfccs_processed)
            labels.append(emotion)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode emotion labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Define a simple deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(emotion_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions on new data
# Replace 'new_audio_path' with the path to the audio file you want to classify
new_audio_path = r'C:\Users\jakku\Downloads\audio\Actor_21\03-01-03-01-02-02-21.wav'
new_audio, sample_rate = librosa.load(new_audio_path, res_type='kaiser_fast')
new_mfccs = librosa.feature.mfcc(y=new_audio, sr=sample_rate, n_mfcc=13)
new_mfccs_processed = np.mean(new_mfccs.T, axis=0)
new_features = np.array([new_mfccs_processed])
prediction = model.predict(new_features)
predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])
print(f'Predicted emotion: {predicted_emotion[0]}')
print("audio data stored")


# In[1]:


import os

# Absolute directory path
folder_path = r'C:\Users\jakku\Downloads\audio'

# Initialize a variable to count audio files
total_audio_files = 0

# Recursively search for audio files in subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.wav'):
            total_audio_files += 1

print(f'Total audio files in the folder and its subfolders: {total_audio_files}')

