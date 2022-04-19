import json
import os
import time
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from models.cnn1 import build_cnn1_model
from models.fcnn1 import build_fcnn1_model
from models.fcnn2 import build_fcnn2_model
from utils.Datasets import datapreparation, read_prepared_data, datapreparation_cnn1, prepare_datasets_cnn1
from utils.params import Params
import sklearn.preprocessing as skp

from utils.plotting import plot_history

seed = 12
np.random.seed(seed)


start_time = time.strftime("%d%m%y_%H%M%S")


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    type=str,
    help="Pass name of model as defined in hparams.yaml."
    )
parser.add_argument(
    "--write_data",
    required=False,
    default=False,
    help="Set to true to write_data."
    )
args = parser.parse_args()
# Parse our YAML file which has our model parameters.
params = Params("hparams.yaml", args.model_name)

if not os.path.exists(params.log_dir):
    os.makedirs(params.log_dir)
if not os.path.exists(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
if not os.path.exists("figs"):
    os.makedirs("figs")


# CNN model(since different models use different types of data)
if params.model_name == "cnn1":

    if args.write_data:
        datapreparation_cnn1()

    X_train, X_valid, X_test, y_train, y_valid, y_test, z = prepare_datasets_cnn1(0.25, 0.2)

    # use gpu if it's available
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as session:

        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_vis_dev

        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        # build the cnn model
        model = build_cnn1_model(input_shape)

        optimiser = keras.optimizers.Adam(learning_rate=params.lr)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()
        # train this model based on the parameters from YMAL file
        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=params.batch_size,
                            epochs=params.num_epochs)

        # draw the graph about training and validation's loss and accuracy
        fig = plot_history(history)
        fig.savefig(os.path.join("figs", "{}_training_vis".format(params.model_name)))

        # save model
        model.save(os.path.join(params.checkpoint_dir, params.model_name))

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print('\nTest accuracy:', test_acc)
        train_losses = [float(i) for i in history.history["loss"]]
        train_accuracy = [float(i) for i in history.history["accuracy"]]
        val_losses = [float(i) for i in history.history["val_loss"]]
        val_accuracy = [float(i) for i in history.history["val_accuracy"]]

        # Some log information to help you keep track of your model information.
        logs = {
            "model": str(params.model_name),
            "train_losses": train_losses,
            "train_accs": train_accuracy,
            "val_losses": val_losses,
            "val_accs": val_accuracy,
            "best_val_epoch": int(np.argmax(history.history["val_accuracy"]) + 1),
            "lr": str(params.lr),
            "batch_size": str(params.batch_size)
        }

        with open(os.path.join(params.log_dir, "{}_{}.json".format(args.model_name, start_time)), 'w') as f:
            json.dump(logs, f)

else:

    df = pd.read_csv('Data/features_3_sec.csv')
    if args.write_data:
        datapreparation(df, params)

    # split the data into 70% train data, 20% valid data and 10% test data
    X_train, y_train = read_prepared_data(params.data_dir, "train.csv")
    X_valid, y_valid = read_prepared_data(params.data_dir, "valid.csv")
    X_test, y_test = read_prepared_data(params.data_dir, "test.csv")

    # Scale the Features
    scaler = skp.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    # use gpu if it's available
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as session:

        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_vis_dev

        input_shape = (X_train.shape[1], )

        # build fcnn model,there are two fcnn models to choose
        if params.model_name == "fcnn1":
            model = build_fcnn1_model(input_shape)
        elif params.model_name == "fcnn2":
            model = build_fcnn2_model(input_shape)

        model.compile(optimizer=params.optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        # train this model based on the parameters from YMAL file
        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=params.batch_size,
                            epochs=params.num_epochs)

        # draw the graph about training and validation's loss and accuracy
        fig = plot_history(history)
        fig.savefig(os.path.join("figs", "{}_training_vis".format(params.model_name)))

        # save model
        model.save(os.path.join(params.checkpoint_dir, params.model_name))

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

        print('\nTest accuracy:', test_acc)
        train_losses = [float(i) for i in history.history["loss"]]
        train_accuracy = [float(i) for i in history.history["accuracy"]]
        val_losses = [float(i) for i in history.history["val_loss"]]
        val_accuracy = [float(i) for i in history.history["val_accuracy"]]

        # Some log information to help you keep track of your model information.
        logs = {
            "model": str(params.model_name),
            "train_losses": train_losses,
            "train_accs": train_accuracy,
            "val_losses": val_losses,
            "val_accs": val_accuracy,
            "best_val_epoch": int(np.argmax(history.history["val_accuracy"]) + 1),
            "batch_size": str(params.batch_size)
        }

        with open(os.path.join(params.log_dir, "{}_{}.json".format(args.model_name, start_time)), 'w') as f:
            json.dump(logs, f)









