import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Directory where the dataset is located
data_dir = 'data'  # Replace with the path to the 'data' directory

# Create a mapping of genres to integer labels
genre_labels = os.listdir(os.path.join(data_dir, 'genres_original'))
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(genre_labels)

# Parameters for the network
input_shape = (224, 224, 3)
num_classes = len(genre_labels)
batch_size = 32

# Load the VGG19 model pre-trained on ImageNet data
base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
# Freeze the layers except the last 4 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Create the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,  # Use 20% of the data for validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'images_original'),  # This should be the path to your spectrogram images
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'images_original'),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10
)

# Save the model
model.save('models/vgg19_genre_classifier.h5')

# Save the label encoder
np.save('models/label_classes.npy', label_encoder.classes_)
