# Simultaneous Translation

This README contains instructions for training and evaluating a wait-k based simultaneous translation system with [SimulEval](https://github.com/facebookresearch/SimulEval). For more example models, see [iwslt21-simul-trans](/examples/iwslt21/SIMUL_TRANS.md).

## Requirements
**SimulEval**
```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval/
pip install -e .
```
It is worth noting that there are some conflicts between python multiprocessing and CUDA initialization of tensorflow, so we make some changes to `SimulEval/simuleval/cli.py` and actually use [`neurst/cli/simuleval_cli.py`](/neurst/cli/simuleval_cli.py) instead.

The changes are as follow:
```python
# add init method to import tensorflow and restrict memory usage
def init():
    global tf
    import tensorflow as tf
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices('GPU')[0], 
        True
    )

# set init method as the initializer of multiprocessing.Pool
# with Pool(args.num_processes) as p:
with Pool(args.num_processes, initializer=init) as p:
```

## Wait-k Training
Following [examples/translation](/examples/translation/README.md), we can train a wait-k based transformer model with extra options:
```bash
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/training_args.yml,wmt14_en_de/translation_bpe.yml \
    --hparams_set waitk_transformer_base \
    --model_dir /wmt14_en_de/waitk_benchmark_base \
    --task WaitkTranslation \  # overwrite the task.class in wmt14_en_de/translation_bpe.yml
    --wait_k 3 # the wait k lagging
```

- The self-attention in the encoder is monotonic.
- To enable multi-path training (different `k` for different training batch), we can set `--wait_k '[3,5,7]'`.
- For validation inner the training process, it will only pick the first `k` for evaluation. For a single validation process, one can overwrite the `k` for evaluation with `--eval_task_args "{'wait_k':5}"`

For more details, see [waitk_transformer.py](/neurst/models/waitk_transformer.py) and [waitk_translation.py](/neurst/tasks/waitk_translation.py).

## Evaluating Latency with SimulEval
As mentioned above, the original SimulEval has conflict on multiprocessing with TensorFlow. So here we use an upgraded version. 
```bash
python3 -m neurst.cli.simuleval_cli \
    --agent simul_trans_text_agent \
    --data-type text \
    --source path_to_src_file \
    --target path_to_trg_file \
    --model-dir path_to_model_dir \
    --num-processes 12 \
    --wait-k 7 \
    --output temp
```

- Here [simul_trans_text_agent](/neurst/utils/simuleval_agents/simul_trans_text_agent.py) is the standard implementation for wait-k transformer.
- We can increase `--num-processes` to speed up decoding, which will result in an exponential increase in GPU memory.
- We can also change `--wait-k` to balance BLEU and Average Latency.


