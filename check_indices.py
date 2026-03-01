import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = 'Railway Track fault Detection Updated/Train'
if os.path.exists(train_dir):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(train_dir, target_size=(300, 300), class_mode='binary')
    print("Class Indices:", gen.class_indices)
else:
    print("Directory not found")
