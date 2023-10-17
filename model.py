from keras.models import Model
from keras.layers import Input, Lambda, Reshape, Bidirectional, LSTM, Dense, Dropout
from customBlock import Residual

import numpy as np
import typing
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer


def DefinedModel(input_dim, output_dim, activation='leaky_relu', drop_out=0.2):
    inputs = Input(shape=input_dim, name='input')

    input = Lambda(lambda x: x / 255)(inputs)

    x1 = Residual(32, activation=activation,
                  skip_conv=True, strides=1, drop_out=drop_out)(input)

    x2 = Residual(32, activation=activation,
                  skip_conv=True, strides=2, drop_out=drop_out)(x1)
    x3 = Residual(32, activation=activation,
                  skip_conv=False, strides=1, drop_out=drop_out)(x2)

    x4 = Residual(64, activation=activation,
                  skip_conv=True, strides=2, drop_out=drop_out)(x3)
    x5 = Residual(64, activation=activation,
                  skip_conv=False, strides=1, drop_out=drop_out)(x4)

    x6 = Residual(128, activation=activation,
                  skip_conv=True, strides=2, drop_out=drop_out)(x5)
    x7 = Residual(128, activation=activation,
                  skip_conv=True, strides=1, drop_out=drop_out)(x6)

    x8 = Residual(128, activation=activation,
                  skip_conv=True, strides=2, drop_out=drop_out)(x7)
    x9 = Residual(128, activation=activation,
                  skip_conv=False, strides=1, drop_out=drop_out)(x8)

    squeezed = Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = Bidirectional(LSTM(256, return_sequences=True))(squeezed)
    blstm = Dropout(drop_out)(blstm)

    blstm = Bidirectional(LSTM(64, return_sequences=True))(blstm)
    blstm = Dropout(drop_out)(blstm)

    output = Dense(output_dim + 1, activation='softmax', name='output')(blstm)

    model = Model(inputs=inputs, outputs=output)

    return model


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(
            image, *self.input_shape[:2][::-1]
        )

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text
