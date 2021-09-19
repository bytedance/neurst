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
import tensorflow as tf

from neurst.tasks import build_task
from neurst.utils.checkpoints import restore_checkpoint_if_possible_v2
from neurst.utils.hparams_sets import get_hyper_parameters
from neurst.utils.misc import assert_equal_numpy


def test_openai_gpt2():
    from transformers import GPT2Model, GPT2Tokenizer

    input_text = "Here is some text to encode"
    pt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    pt_model = GPT2Model.from_pretrained("gpt2", return_dict=True)
    pt_outputs = pt_model(**pt_tokenizer([input_text], return_tensors="pt"))

    task = build_task({
        "class": "lm",
        "params": {
            "data_pipeline.class": "GPT2DataPipeline",
            "max_len": 50,
            "begin_of_sentence": "eos"
        }
    })

    model_cfgs = get_hyper_parameters("gpt2_117m")
    model = task.build_model(model_cfgs)
    restore_checkpoint_if_possible_v2(model, "117M", model_name="OpenAIGPT2")
    input_ids = task._data_pipeline.process(input_text)
    tf_inputs = {
        "trg_input": tf.convert_to_tensor([input_ids], tf.int64),
        "trg_length": tf.convert_to_tensor([len(input_ids)], tf.int64)
    }
    _, gen_init = model.get_symbols_to_logits_fn(tf_inputs, is_training=False, is_inference=False)
    tf_outputs = model.get_decoder_output(gen_init["decoder_input"],
                                          cache=gen_init["decoder_internal_cache"],
                                          is_training=False)
    assert_equal_numpy(pt_outputs.last_hidden_state.detach().numpy(), tf_outputs[:, :-1].numpy(), 5e-4)


if __name__ == "__main__":
    test_openai_gpt2()
