import tensorflow as tf
from utils import *
import numpy as np
import im_capture as im
import argparse
from pprint import pprint
import os

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ap = argparse.ArgumentParser()
ap.add_argument(
    '-i',
    '--img',
    metavar='IMG',
    type=str,
    default='tmp',
    help='the filepath to save screenshots to'
)
args = vars(ap.parse_args())

class_names = ["Blastoise", "Bulbasaur", "Charizard", "Charmander", "Charmeleon", "Ivysaur", "Pikachu", "Squirtle", "Venusaur", "Wartortle"]

base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers:
    layer.trainable = False

def make_MobileNet_custom_model(base_model):
    # get base model output
    x = base_model.output
    # add GlobalAveragePooling2D layer
    x = tf.keras.layers.GlobalAveragePooling2D(name='CustomLayer_1')(x)
    # add Dense layer of 512 units
    x = tf.keras.layers.Dense(name='CustomLayer_2', units=512, activation='relu')(x)
    # add output Dense layer with 10 units and softmax activation function
    predictions = tf.keras.layers.Dense(name='OutputLayer', units=10, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

def build_model():
    model = make_MobileNet_custom_model(base_model)
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model

def train(model):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

    batch_size = 8

    train_generator = train_datagen.flow_from_directory('data/Pokemon10/train', batch_size=batch_size, target_size=(128, 128), shuffle=True, class_mode='categorical')
    # class_dictionary = train_generator.class_indices
    # print(class_dictionary)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    test_generator = test_datagen.flow_from_directory('data/Pokemon10/test', batch_size=batch_size, target_size=(128, 128), class_mode='categorical')

    time_summary = TimeSummary()

    summary = model.fit_generator(
        train_generator,
        epochs=30,
        steps_per_epoch=20,
        verbose=1,
        validation_data=test_generator,
        validation_steps=20,
        callbacks=[time_summary]
    )

def run_im_capture():
    im.run(filepath=args['img'])

def probabilityToPercentage(x):
    return (x/1)*100

def print_results(y_probs):
    percentages = list(map(probabilityToPercentage, y_probs))
    for i in range(len(class_names)):
        print("The probability of {} is {:.2f}%".format(class_names[i], percentages[i]))

def predict(model):
    # Manual Input
    # img_path = 'images/pikachu.jpeg'
    # img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))

    # Input via image grabber
    img = tf.keras.preprocessing.image.load_img(args['img'] + '.png', target_size=(128, 128))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    y_probs = model.predict(x)[0]
    os.remove(args['img'] + '.png')
    print_results(y_probs)

def save_model(model):
    save_model = str(input("Do you want to save the model? y/N "))
    if (save_model == "y"):
        model.save('my_model.h5')
        print("Model has been saved.")
    elif (save_model == "N"):
        print("Model wasn't saved.")
    else:
        print("Input y/N only!")

def run():
    use_pretrained_model = str(input("Do you want to use pretrained model? y/N "))
    if (use_pretrained_model == "y"):
        model = load_model()
        run_im_capture()
        predict(model)
    elif (use_pretrained_model == "N"):
        model = build_model()
        train(model)
        run_im_capture()
        predict(model)
        save_model(model)
    else:
        print("Input y/N only!")

if __name__ == '__main__':
    run()
