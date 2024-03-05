from keras.layers import Layer
from keras import backend as K


class SaltAndPepperNoise(Layer):

    def __init__(self, ratio, **kwargs):
        super(SaltAndPepperNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.ratio = ratio

    @override
    def call(self, inputs):
        def noised():
            shp = K.shape(inputs)[1:]
            mask_select = K.random_bernoulli(shape=shp, p=self.ratio)
            mask_noise = K.random_bernoulli(shape=shp, p=0.5)
            out = inputs * (1 - mask_select) + mask_noise * mask_select
            return out

        return K.in_train_phase(noised(), inputs)

    def get_config(self):
        config = {'ratio': self.ratio}
        base_config = super(SaltAndPepperNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
