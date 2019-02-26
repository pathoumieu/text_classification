import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Concatenate
from sklearn.metrics import accuracy_score, f1_score
from keras.optimizers import Adam


def auc(y_true, y_pred):
    """
    Define tensorflow AUC metric for Keras early stopping.
    """
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def f1_score_K(y_true, y_pred):
    """
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def max_accuracy(y_true, y_pred):
    """
    Compute max accuracy for all values of y_pred as threshold.

    Parameters
    ----------
    y_true: array of bool

    y_pred: array of float


    Returns
    -------
    max_accuracy, max_threshold: tuple of (float, float)
        `max_accuracy` is the maximum accuracy obtained for this prediction.
        `max_threshold` is threshold for which accuracy is maximum.
    """
    accuracies = []
    thresholds = sorted(np.unique(y_pred))
    for a in thresholds:
        pred = y_pred >= a
        accuracies.append(accuracy_score(y_true, pred))
    idx = np.argmax(accuracies)
    if idx > 0:
        threshold = (thresholds[idx] + thresholds[idx-1]) / 2
    else:
        threshold = thresholds[idx]
    return accuracies[idx], threshold


def max_f1(y_true, y_pred):
    """
    Compute max F1-score for all values of y_pred as threshold.

    Parameters
    ----------
    y_true: array of bool

    y_pred: array of float


    Returns
    -------
    max_accuracy, max_threshold: tuple of (float, float)
        `max_f1` is the maximum F1-score obtained for this prediction.
        `max_threshold` is threshold for which F1-score is maximum.
    """
    f1s = []
    thresholds = sorted(np.unique(y_pred))
    for a in thresholds:
        pred = y_pred >= a
        f1s.append(f1_score(y_true, pred))
    idx = np.argmax(f1s)
    if idx > 0:
        threshold = (thresholds[idx] + thresholds[idx-1]) / 2
    else:
        threshold = thresholds[idx]
    return f1s[idx], threshold


class Hybrid_RNN(object):
    """
    Class representing Keras Hybrid Recurrent Neural Network model.
    """
    def __init__(self, max_words, max_len, embedding_size=100,
                 dropout_rate=0.5, lstm_size=128, dense_size_features=256,
                 dense_size_concat=128, add_dense=None,
                 learning_rate=0.001):
        """
        Initialize architecture and hyperparameters.

        Parameters
        ----------
        max_words: int
            Maximum number of words in corpus (Train + Test).
        max_len: int
            Maximum length of sequence for LSTM.
        embedding_size: int
            Dimension of word embeddings.
        dense_size_features: int
            Size of first dense layer for extracted features.
        dense_size_concat: int
            Size of dense layer for processing concatenation of feature layer
            and LSTM layer.
        add_dense: int or None
            If int, size of dense layer added to first dense extracted feature
            representation.
            If None, no additional dense layer is added to extracted features
            dense representation.
        dropout_rate: float
            Dropout rate for each layer.
        lstm_size: int
            Number of LSTM units.
        learning_rate: float
            Learning rate for ADAM optimizer.
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.dense_size_features = dense_size_features
        self.dense_size_concat = dense_size_concat
        self.add_dense = add_dense
        self.dropout_rate = dropout_rate
        self.lstm_size = lstm_size
        self.learning_rate = learning_rate

    def init_hyperparams(self, *args, **kwargs):
        """
        Update network hyperparameters.
        """
        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])

    def init_network(self, features_cols):
        """
        Define architecture for Keras model.
        """
        # Define dense layers for extracted features.
        features = Input(name='feature_inputs', shape=[len(features_cols)])
        feature_layer = Dense(self.dense_size_features, name='FC2')(features)
        feature_layer = Activation('relu')(feature_layer)
        feature_layer = Dropout(self.dropout_rate)(feature_layer)

        if self.add_dense is not None:
            feature_layer = Dense(self.add_dense, name='FC3')(feature_layer)
            feature_layer = Activation('relu')(feature_layer)
            feature_layer = Dropout(self.dropout_rate)(feature_layer)

        # Define word embeddings and LSTM layer for sequences
        inputs = Input(name='sequence_inputs', shape=[self.max_len])
        sequence_layer = Embedding(self.max_words, self.embedding_size,
                                   input_length=self.max_len)(inputs)
        sequence_layer = LSTM(self.lstm_size)(sequence_layer)

        # Concatenate LSTM representation and extracted features representation
        layer = Concatenate()([sequence_layer, feature_layer])

        # Add dense layer
        layer = Dense(self.dense_size_concat, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(self.dropout_rate)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)

        self.model = Model(inputs=[inputs, features], outputs=layer)

    def train(self, train_matrix, train_features, train_labels,
              valid_matrix, valid_features, valid_labels,
              class_weight={0: 1.0, 1: 1.0},
              epochs=10, patience=2):
        """
        Compile model, train on defined train dataset, early stop on valid dataset.

        Parameters
        ----------
        class_weight: dict
            Dictionnary of weights for each class in binary cross entropy loss function.
        epochs: int
            Maximum number of epochs.
        patience: int
            Number of epochs for which early stopping metric is allowed not to improve.
        """
        optimizer = Adam(lr=self.learning_rate)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=[f1_score_K])

        self.model.fit([train_matrix, train_features], train_labels,
                       validation_data=([valid_matrix, valid_features], valid_labels),
                       batch_size=64,
                       epochs=epochs,
                       class_weight=class_weight,
                       callbacks=[EarlyStopping(monitor='val_f1_score_K',
                                                patience=patience,
                                                mode='max',
                                                min_delta=0.0001,
                                                verbose=1,
                                                restore_best_weights=True)])
