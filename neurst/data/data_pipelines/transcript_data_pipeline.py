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
import re

from sacremoses.normalize import MosesPunctNormalizer

from neurst.data.data_pipelines import register_data_pipeline
from neurst.data.data_pipelines.text_data_pipeline import TextDataPipeline

PUNC_PATTERN = re.compile(r"[,\.\!\(\);:、\?\-\+=\"/><《》\[\]，。：；「」【】{}`@#\$%\^&\*]")


@register_data_pipeline("transcript")
class TranscriptDataPipeline(TextDataPipeline):
    PUNC_NORMERS = dict()

    def __init__(self, *args, **kwargs):
        """ Initializes the data pipeline for transcription(audio2text) data.

        Args:
            remove_punctuation: Whether to remove punctuation.
            lowercase: Whether to lowercase the text.
        """
        self._remove_punc = kwargs.get("remove_punctuation", False)
        self._lc = kwargs.get("lowercase", "True")
        super(TranscriptDataPipeline, self).__init__(*args, **kwargs)

    @staticmethod
    def cleanup_transcript(language, transcript, lowercase=True, remove_punctuation=True):
        if lowercase:
            transcript = transcript.lower()
        if language not in ["zh", "ja"]:
            if language not in TranscriptDataPipeline.PUNC_NORMERS:
                TranscriptDataPipeline.PUNC_NORMERS[language] = MosesPunctNormalizer(lang=language)
            transcript = TranscriptDataPipeline.PUNC_NORMERS[language].normalize(transcript)
            transcript = transcript.replace("' s ", "'s ").replace(
                "' ve ", "'ve ").replace("' m ", "'m ").replace("' t ", "'t ").replace("' re ", "'re ")
        if remove_punctuation:
            transcript = PUNC_PATTERN.sub(" ", transcript)
        transcript = " ".join(transcript.strip().split())
        return transcript

    def process(self, input, is_processed=False):
        if not is_processed:
            input = self.cleanup_transcript(self._language, input, self._lc, self._remove_punc)
        return super(TranscriptDataPipeline, self).process(input, is_processed)
