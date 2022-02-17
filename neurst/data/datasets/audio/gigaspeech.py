"""
This script defines how to process the GigaSpeech dataset.
"""
import json
import os

import tensorflow as tf
from absl import logging
from scipy.io import wavfile

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag


@register_dataset
class GigaSpeech(RawAudioDataset):
    """
    The GigaSpeech class reads {audio file path: list of (begin time (s), end time (s), transcript)}
    from GigaSpeech.json, and provides (audio, transcript) for creating TFRecords.
    """
    SUBSET_CHOICES = ["XL", "L", "M", "S", "XS", "DEV", "TEST"]

    def __init__(self, args):
        super(GigaSpeech, self).__init__(args)
        if not args["subset"] in GigaSpeech.SUBSET_CHOICES:
            raise ValueError(f"Subset {args['subset']} for GigaSpeech dataset "
                             f"must be in one of {', '.join(GigaSpeech.SUBSET_CHOICES)}")
        self._subset = "{" + args["subset"] + "}"
        logging.info(f"input subset = {self._subset}")
        self._aud_transc_transla_dict = None
        self._num_samples = None

    @property
    def num_samples(self):
        if self._num_samples is not None:
            return self._num_samples

        if self._aud_transc_transla_dict is None:
            self.load_transcripts()
        num = 0
        for wav_path in self._aud_transc_transla_dict:
            num += len(self._aud_transc_transla_dict[wav_path])
        self._num_samples = num
        logging.info(f"num of samples is {self._num_samples}.")
        return self._num_samples

    @property
    def status(self):
        return {
            "audio": DataStatus.PROJECTED,
            "transcript": DataStatus.RAW,
        }

    @staticmethod
    def class_or_method_args():
        this_args = super(GigaSpeech, GigaSpeech).class_or_method_args()
        this_args.extend([
            Flag("subset", dtype=Flag.TYPE.STRING, default="XL",
                 choices=GigaSpeech.SUBSET_CHOICES,
                 help="The dataset portion to be extracted, i.e. XL, DEV, TEST.")
        ])
        return this_args

    def load_transcripts(self):
        """
        Loads transcripts (and translations if exists).
        Storage format:
            {audio file path: list of (begin time (s), end time (s), transcript)}
        """
        if self._aud_transc_transla_dict is not None:
            return
        self._aud_transc_transla_dict = {}

        meta_file_path = os.path.join(self._input_tarball, "GigaSpeech.json")
        if not tf.io.gfile.exists(meta_file_path):
            raise ValueError(f"Not found GigaSpeech.json at {self._input_tarball}")

        logging.info(f"Loading transcriptions from file: {meta_file_path}")

        repl_marks = [["<QUESTIONMARK>", "?"], ["<EXCLAMATIONPOINT>", "!"],
                      ["<PERIOD>", "."], ["<COMMA>", ","],
                      [" ?", "?"], [" !", "!"], [" .", "."], [" ,", ","]]

        with tf.io.gfile.GFile(meta_file_path, mode="r") as meta_file:
            meta = json.load(meta_file)
            audios = meta["audios"]
            for audio in audios:
                audio_subset = audio["subsets"]
                path = os.path.join(self._input_tarball, audio["path"])
                if self._subset not in audio_subset:
                    continue

                segments = audio["segments"]
                for segment in segments:
                    seg_subset = segment["subsets"]
                    if self._subset not in seg_subset:
                        continue

                    transcript = segment["text_tn"]
                    if "<SIL>" in transcript or "<NOISE>" in transcript \
                            or "<MUSIC>" in transcript or "<OTHER>" in transcript:
                        continue
                    for ori, rpl in repl_marks:
                        transcript = transcript.replace(ori, rpl)
                    transcript = transcript.lower()
                    # (begin time (s), end time (s), src)
                    if path not in self._aud_transc_transla_dict:
                        self._aud_transc_transla_dict[path] = []
                    self._aud_transc_transla_dict[path].append((float(segment["begin_time"]),
                                                                float(segment["end_time"]),
                                                                transcript))

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
            shard_id: current shard
            total_shards: num of shards in total
        """
        if total_shards > 1:
            total_samples = self.num_samples
            samples_per_part = total_samples // total_shards
            range_begin = samples_per_part * shard_id
            if shard_id == total_shards - 1:
                range_end = total_samples + 1
                logging.info(f"Iterate on dataset from {range_begin} "
                             f"to the end (total {total_samples}).")
            else:
                range_end = range_begin + samples_per_part
                logging.info(f"Iterate on dataset from {range_begin} "
                             f"to {range_end} (total {total_samples}).")

        def gen_audio_text():
            """ Returns (audio, transcript)."""
            if self._aud_transc_transla_dict is None:
                self.load_transcripts()

            current_sample = 0
            hit_end = False
            pid = os.getpid()

            for opus_path in self._aud_transc_transla_dict:
                origin_wav = None
                sample_rate = 16000
                opus_file_name = os.path.basename(opus_path.strip('.opus'))
                local_tmp_opus = os.path.join(os.path.dirname(__file__),
                                              f"{pid}-{opus_file_name}.opus")
                local_tmp_wav = os.path.join(os.path.dirname(__file__),
                                             f"{pid}-{opus_file_name}.wav")
                for start_time, end_time, transcript in self._aud_transc_transla_dict[opus_path]:
                    current_sample += 1
                    if total_shards > 1:
                        if current_sample < range_begin:
                            continue
                        if current_sample >= range_end:
                            hit_end = True
                            break
                    # extract to wav only when iterate on a new one
                    if origin_wav is None:
                        local_opus = opus_path
                        if opus_path.startswith("hdfs://"):
                            tf.io.gfile.copy(opus_path, local_tmp_opus, overwrite=True)
                            local_opus = local_tmp_opus
                        os.system(f"ffmpeg -y -i {local_opus} -ar 16000 -ac 1 {local_tmp_wav}")
                        sample_rate, origin_wav = wavfile.read(local_tmp_wav)

                    start = int(start_time * sample_rate)
                    end = int(end_time * sample_rate) + 1
                    data_sample = self._pack_example_as_dict(
                        audio=self.extract_audio_feature(sig=origin_wav[start:end],
                                                         rate=sample_rate),
                        transcript=transcript,
                        src_lang=self.LANGUAGES.EN
                    )
                    if map_func is None:
                        yield data_sample
                    else:
                        yield map_func(data_sample)
                if os.path.exists(local_tmp_wav):
                    os.remove(local_tmp_wav)
                if os.path.exists(local_tmp_opus):
                    os.remove(local_tmp_opus)
                if hit_end:
                    break

        return gen_audio_text
