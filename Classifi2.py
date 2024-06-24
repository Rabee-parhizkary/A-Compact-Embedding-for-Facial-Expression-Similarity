import os
from PIL import Image
import numpy as np
from tensorflow.python.keras.utils import np_utils
from keras.api.models import Model
from keras.api.layers import Dense, Dropout , Flatten
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.optimizers import Adam
import tensorflow.python.keras.backend as K
import FEC


height = 224
width = 224
channels = 3
batch_size = 32
nb_classes = 7
batch_num = 16

def convert_path(path):
    return path.replace("\\", "/")

train_path = convert_path(os.path.join('E:/', 'vision', 'project', 'FECNet-master (1)', 'FECNet-master', 'data', 'train'))
test_path = convert_path(os.path.join('E:/', 'vision', 'project', 'FECNet-master (1)', 'FECNet-master', 'data', 'test'))

def readData(train_path, test_path):
    def get_image_paths_and_labels(base_path):
        image_paths = []
        labels = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png')):
                    image_path = convert_path(os.path.join(convert_path(root), convert_path(file)))
                    label = convert_path(os.path.basename(convert_path(root)))  # Assuming the folder name is the class label
                    image_paths.append(image_path)
                    labels.append(label)  # Assuming the folder name is the class index
        return image_paths, labels

    train_image_paths, train_labels = get_image_paths_and_labels(train_path)
    test_image_paths, test_labels = get_image_paths_and_labels(test_path)

    def load_and_preprocess_images(image_paths):
        images = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')  # Convert image to RGB
                image = image.resize((height, width))
                img_array = np.array(image)
                if img_array.shape == (height, width, channels):
                    images.append(img_array)
                else:
                    print(f"Skipping {image_path} due to incorrect shape: {img_array.shape}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return np.array(images)

    X_train = load_and_preprocess_images(train_image_paths)
    X_test = load_and_preprocess_images(test_image_paths)

    Y_train = np.array(train_labels)
    Y_test = np.array(test_labels)

    return X_train, Y_train, X_test, Y_test

x, img_input = FEC.create_model()
X_train, Y_train, X_test, Y_test = readData(train_path, test_path)

# Create a label to index mapping
unique_labels = np.unique(np.concatenate((Y_train, Y_test)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Convert string labels to integers
Y_train = np.array([label_to_index[label] for label in Y_train])
Y_test = np.array([label_to_index[label] for label in Y_test])

# One-hot encode the labels
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

x = Flatten()(x)  # Flatten the output tensor
x = Dense(1024, activation='relu', name='final_dense1')(x)
x = Dropout(0.5, name='final_drop')(x)

predictions = Dense(nb_classes, activation='softmax', name='final_classifi')(x)

reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_lr=0.00001, verbose=1)
callbacks = [reduce_learning_rate]

model = Model(inputs=img_input, outputs=predictions)
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['acc'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=100, verbose=1, callbacks=callbacks)

score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print("score:", score)
