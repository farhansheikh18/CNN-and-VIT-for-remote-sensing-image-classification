
# Define your dataset and labels
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Set up variables
image_size = 256
batch_size = 32
num_classes = 2
num_epochs = 1
learning_rate = 0.0001

# Define your dataset and labels
data_dir = 'C:\\Users\\sa\\Desktop\\farhan\\WHU-RS19' # Update with your own path
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

# Collect class labels from subdirectories in the training set
class_labels = [label for label in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, label))]
# Get the list of image filenames in the training directory
image_filenames = [os.path.join(train_dir, label, img) for label in class_labels for img in os.listdir(os.path.join(train_dir, label))]

# Create indices corresponding to the images
indices = np.arange(len(image_filenames))
# Encode class labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(class_labels)
# Split indices into training and test sets
index_label_pairs = list(zip(indices, encoded_labels))

# Split indices into training and test sets
index_pairs_train, index_pairs_test= train_test_split(index_label_pairs, test_size=0.2, random_state=42)

# Extract indices and labels from the pairs
indices_train, class_train = zip(*index_pairs_train)
indices_test, class_test = zip(*index_pairs_test)

# Convert to numpy arrays
indices_train = np.array(indices_train)
indices_test = np.array(indices_test)
class_train = np.array(class_train)
class_test = np.array(class_test)
print("Length of indices_train:", len(indices_train))
print("Length of class_train:", len(class_train))
print("Length of indices_test:", len(indices_test))
print("Length of class_test:", len(class_test))

# Split your class labels into training and test sets
# class_train, class_test = train_test_split(encoded_labels, test_size=0.2, random_state=42)

# Create a function to preprocess your images
def preprocess_image(image_path):
    # print(image_path)
    # image = cv2.imread(image_path)
    image = image_path

    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

# Create CNN model
def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(class_labels), activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create ViT model
def create_vit_model():
    import tensorflow_addons as tfa
    
    model = tfa.vision.models.vit.Vit(
        image_size=(image_size, image_size),
        patch_size=16,
        num_layers=12,
        num_classes=len(class_labels),
        d_model=256,
        ff_dim=512,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
        name='vit_model'
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train and evaluate CNN model
cnn_model = create_cnn_model()
cnn_train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=preprocess_image
)

cnn_train_generator = cnn_train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='sparse',
    classes=class_labels,
    subset='training'
)

cnn_val_generator = cnn_train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='sparse',
    classes=class_labels,
    subset='validation'
)

cnn_model.fit(cnn_train_generator, epochs=num_epochs, validation_data=cnn_val_generator)

# Evaluate CNN model on the test set
cnn_test_generator = cnn_train_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='sparse',
    classes=class_labels
)
# Print class labels and directory structure for verification
print("Class Labels:", class_labels)

# Print number of images in the training and validation directories
print("Number of training images:", sum(len(files) for _, dirs, files in os.walk(train_dir)))
print("Number of validation images:", sum(len(files) for _, dirs, files in os.walk(validation_dir)))


cnn_predictions = cnn_model.predict(cnn_test_generator)
cnn_features_train = cnn_model.predict(cnn_train_generator)
cnn_features_val =cnn_model.predict(cnn_val_generator)
# cnn_predicted_labels = tf.argmax(cnn_predictions, axis=1)
# cnn_accuracy = accuracy_score(cnn_test_generator.labels, cnn_predicted_labels.numpy())
# print(f"CNN Test Accuracy: {cnn_accuracy}")



# Define ViT model for classification
def create_vit_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Replace this with the actual ViT model implementation
    vit_output = tf.keras.layers.Flatten()(inputs)  # Placeholder, replace with actual ViT layers
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(vit_output)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# Create ViT model
vit_model = create_vit_model(input_shape=cnn_features_train.shape[1:], num_classes=num_classes)

# Compile and train ViT model
# print(len(cnn_features_train), len(class_train))
# print(len(cnn_features_val), len(class_test))

# print(np.array(cnn_features_train).shape, np.array(class_train).shape)
# print(np.array(cnn_features_val).shape, np.array(class_test).shape)

# print(np.array(cnn_features_train), np.array(class_train))
# print(np.array(cnn_features_val), np.array(class_test))
print("Length of cnn_features_train:", len(cnn_features_train))
print("Length of class_train:", len(class_train))
print("Length of cnn_features_val:", len(cnn_features_val))
print("Length of class_test:", len(class_test))
print("class_train:", class_train[:5])
print("class_test:", class_test[:5])
print("Shape of cnn_features_train:", cnn_features_train.shape)
print("Shape of cnn_features_val:", cnn_features_val.shape)


vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
vit_model.fit(cnn_features_train, class_train, epochs=num_epochs, batch_size=batch_size, validation_data=(cnn_features_val, class_test))


