import os
import cv2
import pickle
import imutils
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout


def load_data():
    img_paths = []

    directories = ['./Dataset/mask_weared_incorrect/', './Dataset/with_mask/', './Dataset/without_mask/']
    for folder in directories:
        for img_path in os.listdir(folder):
            if folder == './Dataset/mask_weared_incorrect/':
                label = 'mask_weared_incorrect'
            elif folder == './Dataset/with_mask/':
                label = 'with_mask'
            else:
                label = 'without_mask'

            img_paths.append([folder + img_path, label])

    X = np.array([cv2.imread(img_path[0]) for img_path in img_paths]) / 255.0
    y = np.array([label[1] for label in img_paths])

    encoded_y = []
    for label in y:
        # labels will be encoded as such: 0=mask_weared_incorrect, 1=with_mask, 2=without_mask
        if label == 'mask_weared_incorrect':
            encoded_y.append(0)
        elif label == 'with_mask':
            encoded_y.append(1)
        else:
            encoded_y.append(2)

    encoded_y = np.array(encoded_y)
    categorical_y = to_categorical(y=encoded_y, num_classes=3)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Encoded y shape: {encoded_y.shape}")
    print(f"Categorical labels shape: {categorical_y.shape}")

    return X, categorical_y


def split_dataset(data, labels):
    x_train, x_, y_train, y_ = train_test_split(data, labels, test_size=0.3, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=0.333, random_state=42, shuffle=True)

    print(f"\nTraining data: {x_train.shape},  labels: {y_train.shape}")
    print(f"Validation data: {x_val.shape},  labels: {y_val.shape}")
    print(f"Testing data: {x_test.shape},  labels: {y_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test


def display_random_set(data, labels, classes):
    # Function will display 6 random images
    for i in range(6):
        plt.subplot(2, 3, (i+1))
        random_val = np.random.randint(low=0, high=len(data))
        img = data[random_val]
        plt.imshow(img)
        plt.axis(False)
        plt.title(classes[np.argmax(labels[random_val])])
    plt.show()


def build_model(input_dim, num_classes):
    input_ = Input(shape=input_dim)

    x = Conv2D(16, kernel_size=(2, 2), padding='same', activation='relu')(input_)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    '''x = Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)'''

    x = Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(num_classes, activation='softmax')(x)

    model = Model(input_, x)

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])
    print(model.summary())

    return model


def save_history_file(filename_to_save_as, history):
    pickle_out = open(filename_to_save_as, 'wb')
    pickle.dump(history.history, pickle_out)
    pickle_out.close()


def load_history_file(history_filename):
    pickle_in = open(history_filename, 'rb')
    saved_history = pickle.load(pickle_in)
    return saved_history


def plot_curves(history):
    plt.figure(figsize=(10, 5))
    sns.set_style(style='dark')
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Training & Validation Loss')
    plt.legend(['Train loss', 'Validation loss'])

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy & Validation Accuracy')
    plt.legend(['Accuracy', 'Validation Accuracy'])

    plt.show()


def main():
    X, y = load_data()

    # Labels arranged based on index encoding.

    classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(data=X, labels=y)
    display_random_set(data=x_train, labels=y_train, classes=classes)

    epochs = 150
    checkpoint = ModelCheckpoint(filepath='./mask_status.h5', monitor='val_loss', verbose=1, save_best_only=True)
    model = build_model(input_dim=x_train[0].shape, num_classes=len(classes))
    '''history = model.fit(x=x_train, y=y_train, batch_size=512, epochs=epochs, verbose=1, callbacks=[checkpoint],
                     validation_data=(x_val, y_val))
    save_history_file(filename_to_save_as='./mask_status.pkl', history=history)'''

    history_loaded = load_history_file(history_filename='./mask_status.pkl')
    print(history_loaded.keys())
    classifier = load_model(filepath='./mask_status.h5')
    print(classifier.evaluate(x_test, y_test))

    plot_curves(history=history_loaded)

    # prediction on 6 random images
    for i in range(6):
        random_val = np.random.randint(low=0, high=len(x_test))
        img = x_test[random_val]
        actual_label = np.argmax(y_test[random_val])
        prediction = np.argmax(classifier.predict(np.expand_dims(img, axis=0)))
        plt.subplot(2, 3, (i+1))
        plt.imshow(img)
        plt.axis(False)
        plt.title(f"Actual: {actual_label}, pred: {prediction}")
    plt.show()


if __name__ == '__main__':
    main()
