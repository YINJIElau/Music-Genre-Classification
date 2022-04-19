from __future__ import print_function
import os
import json
import argparse
import numpy as np
from utils import Datasets
from utils.Datasets import prepare_datasets_cnn1, read_prepared_data
from utils.params import Params
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from models.cnn1 import build_cnn1_model
from models.fcnn1 import build_fcnn1_model
from models.fcnn2 import build_fcnn2_model
import sklearn.preprocessing as skp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "val_json", type=str, help="Directory of validation json file which indictates the best epoch.")
    parser.add_argument(
        "eval_iter", type=int, default=5, help="Number of times to train and evaluate model")
    args = parser.parse_args()

    with open(args.val_json) as f:  
        model_params = json.load(f)

    params = Params("hparams.yaml", model_params["model"].upper())

    log_dir = os.path.join(params.log_dir, "eval_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    
    acc_scores = []

    if params.model_name == "cnn1":

        X_train, X_valid, X_test, y_train, y_valid, y_test, z = prepare_datasets_cnn1(0.25, 0.2)

        for iter_i in range(args.eval_iter):
            print("Training model for iteration {}...".format(iter_i))

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.compat.v1.Session(config=config) as session:

                os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_vis_dev

                input_shape = (X_train.shape[1], X_train.shape[2], 1)
                model = build_cnn1_model(input_shape)

                optimiser = keras.optimizers.Adam(learning_rate=params.lr)
                model.compile(optimizer=optimiser,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=params.batch_size,
                                    epochs=model_params["best_val_epoch"])

                # Just save the last epoch of each iteration.

                model.save(os.path.join(params.checkpoint_dir,"checkpoint_{}_epoch_{}_iter_{}".format(model_params["model"],
                                                                                                      model_params["best_val_epoch"],
                                                                                                      iter_i)))
                print("Evaluating model for iteration {}...".format(iter_i))

                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
                print("Accuracy for iteration {}\t {}".format(iter_i, test_acc))

                acc_scores.append(float(test_acc))

        logs = {
            "model": model_params["model"],
            "num_epochs": model_params["best_val_epoch"],
            "lr": model_params['lr'],
            "batch_size": model_params["batch_size"],
            "eval_iterations": args.eval_iter,
            "acc_scores": acc_scores,
            "mean_acc": float(np.mean(acc_scores)),
            "var_acc": float(np.var(acc_scores)),
        }

        with open(
                os.path.join(log_dir, "{}_{}.json".format(model_params["model"], args.eval_iter)), 'w') as f:
            json.dump(logs, f)

    else:

        X_train, y_train = read_prepared_data(params.data_dir, "train.csv")
        X_valid, y_valid = read_prepared_data(params.data_dir, "valid.csv")
        X_test, y_test = read_prepared_data(params.data_dir, "test.csv")

        scaler = skp.StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

        for iter_i in range(args.eval_iter):
            print("Training model for iteration {}...".format(iter_i))

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.compat.v1.Session(config=config) as session:

                os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_vis_dev

                input_shape = (X_train.shape[1],)

                if params.model_name == "fcnn1":
                    model = build_fcnn1_model(input_shape)
                elif params.model_name == "fcnn2":
                    model = build_fcnn2_model(input_shape)

                model.compile(optimizer=params.optimizer,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=params.batch_size,
                                    epochs=model_params["best_val_epoch"])

                # Just save the last epoch of each iteration.

                model.save(os.path.join(params.checkpoint_dir,"checkpoint_{}_epoch_{}_iter_{}".format(model_params["model"],
                                                                                                      model_params["best_val_epoch"],
                                                                                                      iter_i)))

                print("Evaluating model for iteration {}...".format(iter_i))

                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
                print("Accuracy for iteration {}\t {}".format(iter_i, test_acc))

                acc_scores.append(float(test_acc))

        logs = {
            "model": model_params["model"],
            "num_epochs": model_params["best_val_epoch"],
            "batch_size": model_params["batch_size"],
            "eval_iterations": args.eval_iter,
            "acc_scores": acc_scores,
            "mean_acc": float(np.mean(acc_scores)),
            "var_acc": float(np.var(acc_scores)),
        }

        with open(
                os.path.join(log_dir, "{}_{}.json".format(model_params["model"], args.eval_iter)), 'w') as f:
            json.dump(logs, f)




if __name__ == '__main__':

    main()