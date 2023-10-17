import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Add, ReLU, LeakyReLU, Dropout
from keras import Model


class Residual(Model):
    def __init__(self,
                 filter_num: int,
                 strides=2,
                 kernel_size=3,
                 skip_conv=True,
                 padding='same',
                 kernel_initializer='he_uniform',
                 activation='relu',
                 drop_out=0.2):
        super(Residual, self).__init__()
        self._conv1 = Conv2D(filter_num, kernel_size, padding=padding,
                             strides=strides, kernel_initializer=kernel_initializer)
        self._conv2 = Conv2D(filter_num, kernel_size, padding=padding,
                             kernel_initializer=kernel_initializer)
        self._conv3 = Conv2D(filter_num, 1, padding=padding,
                             strides=strides, kernel_initializer=kernel_initializer)
        self._batch_norm1 = BatchNormalization()
        self._batch_norm2 = BatchNormalization()
        self._add = Add()

        # Activation
        def actv(layer, alpha=0.1) -> tf.Tensor:
            if activation == 'relu':
                layer = ReLU()(layer)
            elif activation == 'leaky_relu':
                layer = LeakyReLU(alpha=alpha)(layer)
            return layer

        # Drop out
        def dr(layer) -> tf.Tensor:
            if drop_out:
                layer = Dropout(drop_out)(layer)
            return layer

        # Skip convolution
        def skcv(layer) -> tf.Tensor:
            if skip_conv:
                layer = self._conv3(layer)
            return layer

        self._activation = actv
        self._drop_out = dr
        self._skip_conv = skcv

    def call(self, inputs):
        x_skip = inputs

        x = self._conv1(inputs)
        x = self._batch_norm1(x)
        x = self._activation(x)

        x = self._conv2(x)
        x = self._batch_norm2(x)

        # Skip convolution
        x_skip = self._skip_conv(x_skip)

        # Add skip connection
        x = self._add([x, x_skip])
        x = self._activation(x)

        x = self._drop_out(x)

        return x
