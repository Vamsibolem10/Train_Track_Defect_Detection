import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam

# 1. Dataset Configuration
# Get the root directory (one level up from this script's directory)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_dir = os.path.join(base_path, 'Railway Track fault Detection Updated', 'Train')
validation_dir = os.path.join(base_path, 'Railway Track fault Detection Updated', 'Validation')

def check_dirs():
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        print(f"Error: Dataset directories not found at {train_dir} or {validation_dir}")
        return False
    return True

if __name__ == "__main__":
    if check_dirs():
        # 2. Data Preprocessing and Augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(300, 300),
            batch_size=20,
            class_mode='binary'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(300, 300),
            batch_size=20,
            class_mode='binary'
        )

        # 3. Robust Model Selection
        # We'll use MobileNetV2 for real-time inference speed and robustness
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        def build_robust_model(model_name='MobileNetV2'):
            print(f"Building {model_name} for robust real-time performance...")
            if model_name == 'MobileNetV2':
                base_model = tf.keras.applications.MobileNetV2(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
            else:
                base_model = InceptionV3(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
            
            base_model.trainable = False
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
            return model

        model = build_robust_model('MobileNetV2')

        # 4. Model Compilation
        model.compile(optimizer=Adam(learning_rate=0.0001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

        # 5. Callbacks for "Run Once" Robustness
        # EarlyStopping prevents wasting time, ModelCheckpoint saves the "Best"
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_save_path = os.path.join(os.path.dirname(__file__), 'mymodel.h5')
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')

        # 5. Training
        print("Starting training with EarlyStopping for maximum robustness...")
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=10, # Increased epochs but with early stopping so it stops when "best"
            callbacks=[early_stop, checkpoint],
            verbose=2
        )

        # 6. Save the Model
        model.save(model_save_path)
        print(f'Model training complete and saved as {model_save_path}')

        # 7. Performance Visualization
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
    else:
        print("Please ensure the dataset is downloaded and extracted correctly.")
