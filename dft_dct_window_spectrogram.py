"""
The MIT License

Copyright (c) 2010-2018 Google, Inc. http://angularjs.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Author: A. Pimenov
e-mail: pimenov@idrnd.net, i7p9h9@gmail.com
"""


import warnings
import numpy as np
from keras.layers import Layer
from keras.layers import Input
from keras.layers import ZeroPadding1D
from keras.models import Model

import keras
import keras.backend as K

from scipy.signal import get_window

class DftSpectrogram(Layer):
    def __init__(self,
                 length=200,
                 shift=150,
                 window = None,
                 nfft=256,
                 fts="dft",
                 mode="abs",
                 normalize_feature=False,
                 normalize_signal=False,
                 trainable=False,
                 **kwargs):
        """
        Requirements
        ------------
        input shape must meet the conditions: mod((input.shape[0] - length), shift) == 0
        nfft >= length

        Parameters
        ------------
        :param length: Length of each segment.
        :param shift: Number of points to step for segments
        :param nfft: number of dft points, if None => nfft === length
        :param window: window vector/None if rect./Str or tuple for get_window() function 
        :param normalize_feature: zero mean, and unit std for 2d features, doesn't work for "complex" mode
        :param normalize_spectrogram: zero mean, and unit std for 1d input signal
        :param fts: "dft" of "dct" 
        :param mode: 
        For DFT: "abs" - amplitude spectrum; "real" - only real part, "imag" - only imag part,
        "complex" - concatenate real and imag part, "log" - log10 of magnitude spectrogram
        For DCT: 1/2/3/4 - type of tranform 
        :param kwargs: unuse

        Input
        -----
        input mut have shape: [n_batch, signal_length, 1]

        Returns
        -------
        A keras model that has output shape of
        (None, nfft / 2, n_time) (if type == "abs" || "real" || "imag") or
        (None, nfft / 2, n_frame, 2) (if type = "abs" & `img_dim_ordering() == 'tf').
        (None, 1, nfft / 2, n_frame) (if type = "abs" & `img_dim_ordering() == 'th').
        (None, nfft / 2, n_frame, 2) (if type = "complex" & `img_dim_ordering() == 'tf').
        (None, 2, nfft / 2, n_frame) (if type = "complex" & `img_dim_ordering() == 'th').
        If fts=="dct" nfft instead of nfft/2
        number of time point of output spectrogram: n_time = (input.shape[0] - length) / shift + 1
        """
        super(DftSpectrogram, self).__init__(**kwargs)
        self.fts = fts
        self.trainable = trainable
        self.length = length
        self.shift = shift
        self.mode = mode
        self.normalize_feature = normalize_feature
        self.normalize_signal = normalize_signal
        self.window = window
        if nfft is None:
            self.nfft = length
        else:
            self.nfft = nfft

        assert self.nfft >= length

    def build(self, input_shape):
        nfft = self.nfft
        length = self.length
        assert len(input_shape) >= 2
        assert nfft >= length
        if self.window==None:
            self.window=np.ones((length,))
        elif type(self.window)==np.ndarray:
            assert self.window.shape[0]==length
        else:
            self.window = get_window(self.window,length)
        if nfft>length:
            self.window = np.concatenate([self.window,np.zeros((nfft-length))])
        if self.fts=='dft':
            self.__real_kernel = np.asarray([self.window[n]*np.cos(2 * np.pi * np.arange(0, nfft) * n / nfft)
                                                  for n in range(nfft)])
            self.__imag_kernel = -np.asarray([self.window[n]*np.sin(2 * np.pi * np.arange(0, nfft) * n / nfft)
                                                  for n in range(nfft)])
        else:
            self.__real_kernel = np.asarray([self.window[n]*np.ones(nfft) if n==0 and self.mode in [1,3] else
                                             self.window[n]*(-1)**np.arange(0,nfft) if n==nfft-1 and self.mode==1 else
                                             2*self.window[n]*np.cos(np.pi * (np.arange(0, nfft) + (0.5 if self.mode in [3,4] else 0))
                                                    * (n+(0.5 if self.mode in [2,4] else 0))
                                                    / (nfft-(1 if self.mode==1 else 0)))
                                             for n in range(nfft)])
            self.__imag_kernel = np.zeros((nfft,nfft))
        if input_shape[-1] > 1:
            self.__real_kernel = np.stack([self.__real_kernel] * input_shape[-1], axis=-2)
            self.__imag_kernel = np.stack([self.__imag_kernel] * input_shape[-1], axis=-2)
        else:
            self.__real_kernel = self.__real_kernel[:, np.newaxis, :]
            self.__imag_kernel = self.__imag_kernel[:, np.newaxis, :]

        if self.length < self.nfft:
            self.__real_kernel[length - nfft:, :, :] = 0.0
            self.__imag_kernel[length - nfft:, :, :] = 0.0

        self.real_kernel = K.variable(self.__real_kernel, dtype=K.floatx(), name="real_kernel")
        self.imag_kernel = K.variable(self.__imag_kernel, dtype=K.floatx(), name="imag_kernel")

        self.real_kernel.values = self.__real_kernel
        self.imag_kernel.values = self.__imag_kernel

        if self.trainable:
            self.trainable_weights.append(self.real_kernel)
            self.trainable_weights.append(self.imag_kernel)
        else:
            self.non_trainable_weights.append(self.real_kernel)
            self.non_trainable_weights.append(self.imag_kernel)

        self.built = True

    def call(self, inputs, **kwargs):
        if self.normalize_signal:
            inputs = (inputs - K.mean(inputs, axis=(1, 2), keepdims=True)) /\
                     (K.std(inputs, axis=(1, 2), keepdims=True) + K.epsilon())

        if self.length < self.nfft:
            inputs = ZeroPadding1D(padding=(0, self.nfft - self.length))(inputs)

        real_part = K.conv1d(inputs, kernel=self.real_kernel, strides=self.shift, padding="valid")
        imag_part = K.conv1d(inputs, kernel=self.imag_kernel, strides=self.shift, padding="valid")

        real_part = K.expand_dims(real_part)
        imag_part = K.expand_dims(imag_part)
        if self.mode == "abs":
            fft = K.sqrt(K.square(real_part) + K.square(imag_part))
        elif self.mode == "real":
            fft = real_part
        elif self.mode == "imag":
            fft = imag_part
        elif self.mode == "complex":
            fft = K.concatenate((real_part, imag_part), axis=-1)
        elif self.mode == "log":
            fft = K.clip(K.sqrt(K.square(real_part) + K.square(imag_part)), K.epsilon(), None)
            fft = K.log(fft) / np.log(10)
        if self.fts == 'dft':
            fft = K.permute_dimensions(fft, (0, 2, 1, 3))[:, :self.nfft // 2, :, :]
        else:
            fft = real_part
            fft = K.permute_dimensions(fft, (0, 2, 1, 3))
        if self.normalize_feature:
            if self.mode == "complex":
                warnings.warn("spectrum normalization will not applied with mode == \"complex\"")
            else:
                fft = (fft - K.mean(fft, axis=(1, 2), keepdims=True)) / (
                        K.std(fft, axis=(1, 2), keepdims=True) + K.epsilon())

        if K.image_dim_ordering() is 'th':
            fft = K.permute_dimensions(fft, (0, 3, 1, 2))

        return fft

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]

        if input_shape[1] is None:
            times = None
        else:
            times = ((input_shape[1] - self.length) + self.shift) // self.shift
        if self.fts == "dct":
            self.nfft*=2
        if K.image_dim_ordering() is 'th':
            output_shape = [1, input_shape[0], self.nfft // 2, times]
        else:
            output_shape = [input_shape[0], self.nfft // 2, times, 1]
        print("spectrogram shape: {}".format(output_shape[1:]))
        if self.fts == "dct":
            self.nfft/=2
        return tuple(output_shape)

    def get_config(self):
        config = {
            'length': self.length,
            'shift': self.shift,
            'nfft': self.nfft,
            'window': self.window,
            'fts': self.fts,
            'mode': self.mode,
            'trainable': self.trainable,
            'normalize_spectrogram': self.normalize_feature,
            'normalize_signal': self.normalize_signal
        }
        base_config = super(DftSpectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from itertools import product
    from scipy.fftpack import fft, dct
    class TestLayer(object):
        def __init__(self, prop: dict, verbose=False, input_shape: tuple=None):
            self.__prop = prop
            self.__verbose = verbose
            self.__input_shape = input_shape
            self.model = self.__get_model(input_shape=input_shape)

            self.keras_features = None
            self.numpy_features = None

        def np_spectrogram(self, s: np.ndarray) -> np.ndarray:
            """
            :param s: numpy signal
            :return:
            """
            nfft = self.__prop["nfft"]
            length = self.__prop["length"]
            shift = self.__prop["shift"]
            self.window = self.__prop["window"]
            if self.window == None:
                self.window = np.ones((length,))
            elif type(self.window) == np.ndarray:
                assert self.window.shape[0] == length
            else:
                self.window = get_window(self.window, length)
            s = s.astype(np.float32)
            if self.__prop["fts"]=='dft':
                if nfft is None:
                    S = [np.fft.fft(self.window*s[n:n + length]) for n in range(0, N - length + shift, shift)]
                    nfft = length
                else:
                    S = [np.fft.fft(self.window*s[n:n + length], n=nfft) for n in range(0, N - length + shift, shift)]
                S = np.asarray(S).T[:nfft // 2, :].astype(np.complex64)
            else:
                if nfft is None:
                    S = [dct(self.window*s[n:n + length],self.__prop["mode"],length) for n in range(0, N - length + shift, shift)]
                    nfft = length
                else:
                    S = [dct(self.window*s[n:n + length],self.__prop["mode"],nfft) for n in range(0, N - length + shift, shift)]
                S = np.asarray(S).T[:, :].astype(np.float64)

            return S

        def __get_model(self, input_shape=None) -> keras.models.Model:
            if input_shape is None:
                model_input = Input(shape=(None, 1))
            else:
                model_input = Input(shape=input_shape)
            x = DftSpectrogram(**self.__prop)(model_input)

            model = Model(inputs=model_input, outputs=x)

            return model

        def plot_features(self):
            if self.__prop["mode"] is "complex":
                plt.figure()
                plt.subplot(2, 2, 1).set_title("keras real features")
                plt.pcolormesh(self.keras_feature[:, :, 0].squeeze())
                plt.subplot(2, 2, 2).set_title("numpy imag features")
                plt.pcolormesh(self.numpy_features[:, :, 0].squeeze())

                plt.subplot(2, 2, 3).set_title("keras imag features")
                plt.pcolormesh(self.keras_feature[:, :, 1].squeeze())
                plt.subplot(2, 2, 4).set_title("numpy imag features")
                plt.pcolormesh(self.numpy_features[:, :, 1].squeeze())
            else:
                plt.figure()
                plt.subplot(1, 2, 1).set_title("keras features")
                plt.pcolormesh(self.keras_feature.squeeze())
                plt.subplot(1, 2, 2).set_title("numpy features")
                plt.pcolormesh(self.numpy_features.squeeze())

        @staticmethod
        def normalize(x: np.ndarray):
            return (x - np.mean(x)) / np.std(x)

        def compare(self, signal):
            self.keras_feature = self.model.predict(signal[None, :, None])

            if self.__prop["normalize_signal"]:
                signal = self.normalize(signal)
            numpy_features = self.np_spectrogram(signal)
            if self.__prop["mode"] == "complex":
                self.numpy_features = np.stack([np.real(numpy_features), np.imag(numpy_features)], axis=-1)
            elif self.__prop["mode"] == "abs":
                self.numpy_features = np.abs(numpy_features)
            elif self.__prop["mode"] == "real":
                self.numpy_features = np.real(numpy_features)
            elif self.__prop["mode"] == "imag":
                self.numpy_features = np.imag(numpy_features)
            elif self.__prop["mode"] == "log":
                self.numpy_features = np.log10(np.clip(np.abs(numpy_features), 1e-7, None))
            else:
                self.numpy_features = numpy_features
            if self.__prop["normalize_feature"] and self.__prop["mode"] is not "complex":
                self.numpy_features = self.normalize(self.numpy_features)
            if self.__verbose:
                self.plot_features()
                plt.show(block=True)
            rmse = np.sqrt(np.mean((self.numpy_features.squeeze() - self.keras_feature.squeeze()) ** 2))
            print("\nRMSE = {} for properties: \n{}".format(rmse, self.__prop))


            assert rmse < 1e-5
            return rmse


    modes = [1,2,3,4]
    windows = [None,"hann","hamming",'triang',('kaiser', 4.0),('tukey', 0.25)]
    normalize_features = [True, False]
    normalize_signals = [True, False]
    nffts = [None, 64]
    length = 50
    shift = 25
    print('DCT test')
    for mode, win, norm_f, norm_s, nfft in product(modes,windows, normalize_features, normalize_signals, nffts):
        prop = {"length": length,
                "shift": shift,
                "fts": 'dct',
                "window": win,
                "nfft": None,
                "mode": mode,
                "normalize_feature": norm_f,
                "normalize_signal": norm_s,
                "trainable": False
                }
        N = 250 + np.random.randint(0, 10) * shift
        s = np.sin(2 * np.pi * 25 * np.linspace(0, N / 100, N)).astype(np.float32)
        test = TestLayer(prop=prop, verbose=False)
        rmse = test.compare(s)

        assert rmse < 1e-5

    print("all DCT + window functions  tests successful complete")
    modes = ["abs", "real", "imag", "complex", "log"]
    print('DFT test')
    for mode, win, norm_f, norm_s, nfft in product(modes,windows, normalize_features, normalize_signals, nffts):
        prop = {"length": length,
                "shift": shift,
                "fts": 'dft',
                "window": win,
                "nfft": None,
                "mode": mode,
                "normalize_feature": norm_f,
                "normalize_signal": norm_s,
                "trainable": False
                }
        N = 250 + np.random.randint(0, 10) * shift
        s = np.sin(2 * np.pi * 25 * np.linspace(0, N / 100, N)).astype(np.float32)
        test = TestLayer(prop=prop, verbose=False)
        rmse = test.compare(s)

        assert rmse < 1e-5

    print("all DFT + window functions tests successful complete")
