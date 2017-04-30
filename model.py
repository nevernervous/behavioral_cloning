import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import cv2

rows, cols, ch = 64, 64, 3

TARGET_SIZE = (64, 64)


def augment_brightness_camera_images(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:, :, 2] = image1[:, :, 2] * random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def resize_to_target_size(image):
    return cv2.resize(image, TARGET_SIZE)


def crop_and_resize(image):
    '''
    :param image: The input image of dimensions 160x320x3
    :return: Output image of size 64x64x3
    '''

    #  crop 55 pixels from the top and 25 pixels from the bottom
    cropped_image = image[55:135, :, :]

    # resize the image to size 64x64x3
    processed_image = resize_to_target_size(cropped_image)
    return processed_image


def preprocess_image(image):
    image = crop_and_resize(image)
    image = image.astype(np.float32)

    # Normalize image
    image = image/255.0 - 0.5
    return image


def get_augmented_row(row):
    steering = row['steering']

    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    # image = load_img(row[camera].strip())
    image = load_img('data/' + row[camera].strip())
    image = img_to_array(image)

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -steering
        image = cv2.flip(image, 1)

    # Apply brightness augmentation
    image = augment_brightness_camera_images(image)

    # Crop, resize and normalize the image
    image = preprocess_image(image)
    return image, steering


def get_data_generator(data_frame, batch_size=32):
    batches_per_epoch = data_frame.shape[0] // batch_size

    i = 0
    while True:
        start = i*batch_size
        end = start+batch_size - 1

        x_train = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_train = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        for index, row in data_frame.loc[start:end].iterrows():
            x_train[j], y_train[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index to loop the data_frame again.
            i = 0
        yield x_train, y_train


if __name__ == "__main__":
    BATCH_SIZE = 32  # The lower the better
    NB_EPOCH = 3  # The higher the better

    # Read the data from csv file
    # df = pd.read_csv('data/driving_log.csv', header=None, usecols=[0, 1, 2, 3], names=["center", "left", "right", "steering"])
    df = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])

    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    num_rows_training = int(df.shape[0] * 0.8)

    training_data = df.loc[0:num_rows_training-1]
    validation_data = df.loc[num_rows_training:]

    training_generator = get_data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = get_data_generator(validation_data, batch_size=BATCH_SIZE)

    # create Sequential model.
    model = Sequential()

    # layer 1 output shape is 32x32x32
    model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # layer 2 output shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 output shape is 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    # layer 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

    # layer 5
    model.add(Dense(512))
    model.add(ELU())

    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    samples_per_epoch = (20000 // BATCH_SIZE) * BATCH_SIZE

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=NB_EPOCH, nb_val_samples=3000)

    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())