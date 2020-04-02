from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Conv2D, Input, Dense
from tensorflow.python.keras.regularizers import l2


class DQFDNeuralNet:
    """Standard DQN model architecture + l2 regularization to prevent
    overfitting on small demo sets."""

    def __new__(cls, input_shape, n_actions: int, window_length):
        input_shape = (input_shape[0], input_shape[1], window_length)
        frame = Input(shape=input_shape)
        cv1 = Conv2D(
            32, kernel_size=(8, 8), strides=4, activation="relu", kernel_regularizer=l2(1e-4),
        )(frame)
        cv2 = Conv2D(
            64, kernel_size=(4, 4), strides=2, activation="relu", kernel_regularizer=l2(1e-4),
        )(cv1)
        cv3 = Conv2D(
            64, kernel_size=(3, 3), strides=1, activation="relu", kernel_regularizer=l2(1e-4),
        )(cv2)
        dense = Flatten()(cv3)
        dense = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(dense)
        buttons = Dense(n_actions, activation="linear", kernel_regularizer=l2(1e-4))(dense)
        model = Model(inputs=frame, outputs=buttons)
        model.summary()
        return model
