# Copyright 2020 ByteDance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy
import tensorflow as tf
import yaml
from absl import logging

from neurst.utils.compat import is_tf_tensor


class SpecAugment(object):
    """ Implements SpecAugment, https://arxiv.org/abs/1904.08779 """

    _PREDEF_SETTINGS = {
        # LibriSpeech basic
        "LB": {
            "time_wrap_w": 80,
            "freq_mask_n": 1,
            "freq_mask_f": 27,
            "time_mask_n": 1,
            "time_mask_t": 100,
            "time_mask_p": 1.
        },
        # LibriSpeech double
        "LD": {
            "time_wrap_w": 80,
            "freq_mask_n": 2,
            "freq_mask_f": 27,
            "time_mask_n": 2,
            "time_mask_t": 100,
            "time_mask_p": 1.
        },
        # Switchboard mild
        "SM": {
            "time_wrap_w": 40,
            "freq_mask_n": 2,
            "freq_mask_f": 15,
            "time_mask_n": 2,
            "time_mask_t": 70,
            "time_mask_p": 0.2
        },
        # Switchboard strong
        "SS": {
            "time_wrap_w": 40,
            "freq_mask_n": 2,
            "freq_mask_f": 27,
            "time_mask_n": 2,
            "time_mask_t": 70,
            "time_mask_p": 0.2
        }
    }

    def __init__(self,
                 time_wrap_w: int,
                 freq_mask_n: int,
                 freq_mask_f: int,
                 time_mask_n: int,
                 time_mask_t: int,
                 time_mask_p: float,
                 mask_value=None):
        """ Initializes spec augment.

        Args:
            time_wrap_w: The time wrapping parameter.
            freq_mask_n: The number of times to apply frequency masking.
            freq_mask_f: The maximum consecutive frequency to be masked.
            time_mask_n: The number of times to apply time masking.
            time_mask_t: The maximum consecutive time steps to be masked.
            time_mask_p: The upper bound ratio of consecutive time steps of time masking.
            mask_value: The mask value, use the mean of spectrogram if not provided.
        """
        self._time_wrap_w = time_wrap_w
        self._freq_mask_n = freq_mask_n
        self._freq_mask_f = freq_mask_f
        self._time_mask_n = time_mask_n
        self._time_mask_t = time_mask_t
        self._time_mask_p = time_mask_p
        self._mask_value = mask_value
        assert self._time_mask_t > 0
        assert self._freq_mask_f > 0
        if self._time_wrap_w > 0:
            logging.info("WARNING(specaug): here the time wrapping is not implemented "
                         "because it is not a major factor in improving the performance "
                         "according to the original paper, and slows down the training speed.")

    @classmethod
    def build(cls, setting):
        if setting is None:
            return None
        setting = yaml.load(setting, Loader=yaml.FullLoader)
        if isinstance(setting, str):
            setting = cls._PREDEF_SETTINGS.get(setting, None)
        if setting is None:
            return None
        assert isinstance(setting, dict), f"Unknown type of setting: {setting}"
        return cls(**setting)

    @staticmethod
    def freq_or_time_masking_numpy(spectrogram, n, F,
                                   mask_value, axis, p=None):
        """ Mask the spectrogram.

        Repeat n times:
            f ~ [0, F)
            f0 ~ [0, nu - f)
            The [f0, f0 + f) positions of `spectrogram` on `axis` are overwritten by `mask_value`.

        Args:
            spectrogram: numpy.ndarray, the audio feature map, of shape [nframes, ndim].
            n: The number of repeated times.
            F: The maximum mask range.
            mask_value: The value to overwrite the masked positions.
            axis: The masked axis of `spectrogram`.
            p: The upper bound ratio of the mask range if provided.

        Returns:
            The augmented spectrogram.
        """
        num_frames_or_freqs = spectrogram.shape[axis]
        if num_frames_or_freqs < F:
            return spectrogram
        if p:
            F = min(F, math.floor(num_frames_or_freqs * p))
        # sample mask range
        f = numpy.random.randint(0, F, size=n)
        f0 = numpy.random.randint(0, num_frames_or_freqs - f)
        for i in range(n):
            if f0[i] == 0:
                continue
            if axis == 0:
                spectrogram[f0[i]: f0[i] + f[i], :] = mask_value
            else:
                assert axis == 1
                spectrogram[:, f0[i]: f0[i] + f[i]] = mask_value
        return spectrogram

    @staticmethod
    def freq_or_time_mask_tf(spectrogram, n, F, axis, p=None):
        """ Mask the spectrogram.

        Repeat n times:
            f ~ [0, F)
            f0 ~ [0, nu - f)
            The [f0, f0 + f) positions of `spectrogram` on `axis` are overwritten by `mask_value`.

        Args:
            spectrogram: numpy.ndarray, the audio feature map, of shape [nframes, ndim].
            n: The number of repeated times.
            F: The maximum mask range.
            axis: The masked axis of `spectrogram`.
            p: The upper bound ratio of the mask range if provided.

        Returns:
            The mask vector on `axis`.
        """
        num_frames_or_freqs = tf.shape(spectrogram)[axis]

        def _apply(n, F, p):
            # F = tf.cond(tf.less(num_frames_or_freqs, F * 2 + 1), lambda: num_frames_or_freqs // n, lambda: F)
            if p:
                F = tf.minimum(F, tf.math.floor(num_frames_or_freqs * p))
            masks = []
            # sample mask range
            for _ in range(n):
                f = tf.random.uniform(shape=[], minval=0, maxval=F, dtype=tf.int32)
                f0 = tf.random.uniform(shape=[], minval=0, maxval=num_frames_or_freqs - f, dtype=tf.int32)
                mask = tf.pad(tf.ones([f, ], dtype=tf.bool), paddings=[[f0, num_frames_or_freqs - f - f0]],
                              mode='CONSTANT', constant_values=False)
                masks.append(mask)
            final_mask = tf.reduce_any(tf.convert_to_tensor(masks, dtype=tf.bool), axis=0)
            return tf.cast(final_mask, spectrogram.dtype)

        return tf.cond(tf.less(num_frames_or_freqs, F),
                       lambda: tf.zeros([num_frames_or_freqs, ], dtype=spectrogram.dtype),
                       lambda: _apply(n, F, p))

    def _call_tf(self, spectrogram, true_length=None):
        """ Applies specaug based on TensorFlow operations.

        Args:
            spectrogram: A tf.Tensor of shape [nframes, nfeatures] or [batch, nframes, nfeatures].
            true_length: A tf.Tensor of shape [batch, ] if `spectrogram` is a tensor of shape
                [batch, nframes, nfeatures], else None.

        Returns:
            A tuple (augmented spectrogram, new true_length) if ndims of `spectrogram` is 3,
            else the augmented spectrogram.
        """
        _ = true_length
        ndim = spectrogram.get_shape().ndims
        if ndim == 2:
            time_mask = SpecAugment.freq_or_time_mask_tf(
                spectrogram, self._time_mask_n, self._time_mask_t, axis=0)
            freq_mask = SpecAugment.freq_or_time_mask_tf(
                spectrogram, self._freq_mask_n, self._freq_mask_f, axis=1)
            demask = tf.einsum("M,N->MN", 1 - time_mask, 1 - freq_mask)
            mask_value = self._mask_value
            if mask_value is None:
                mask_value = tf.reduce_mean(spectrogram)
            return spectrogram * demask + (1. - demask) * mask_value  # mask_value
        assert ndim == 3  # [batch, frames, dim]
        raise NotImplementedError

    def _call_numpy(self, spectrogram):
        """ Applies specaug based on numpy operations.

        Args:
            spectrogram: A numpy.ndarray of shape [nframes, nfeatures]

        Returns:
            The augmented spectrogram.
        """
        ndim = spectrogram.ndim
        if ndim > 2:
            raise ValueError("batch specaug is not implemented for numpy array.")
        mask_value = self._mask_value
        if mask_value is None:
            mask_value = spectrogram.mean()
        if self._freq_mask_n > 0:
            spectrogram = self.freq_or_time_masking_numpy(
                spectrogram, n=self._freq_mask_n, F=self._freq_mask_f,
                mask_value=mask_value, axis=1)
        if self._time_mask_n > 0:
            spectrogram = self.freq_or_time_masking_numpy(
                spectrogram, n=self._time_mask_n, F=self._time_mask_t,
                mask_value=mask_value, axis=0, p=self._time_mask_p)
        return spectrogram

    def __call__(self, spectrogram, true_length=None):
        """ Applies specaug.

        Args:
            spectrogram: A numpy.ndarray of shape [nframes, nfeatures]
                or a tf.Tensor of shape [nframes, nfeatures] or [batch, nframes, nfeatures].
            true_length: A tf.Tensor of shape [batch, ] if `spectrogram` is a tensor of shape
                [batch, nframes, nfeatures], else None.

        Returns:
            A tuple (augmented spectrogram, new true_length) if ndims of `spectrogram` is 3,
            else the augmented spectrogram.
        """
        if is_tf_tensor(spectrogram):
            return self._call_tf(spectrogram)
        return self._call_numpy(spectrogram)
