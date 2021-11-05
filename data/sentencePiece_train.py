# https://github.com/google/sentencepiece/blob/master/python/README.md
# https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
import sentencepiece as spm
import argparse
import os

def main(args):
    india_lang_set = "<hi>,<en>,<bn>,<ml>,<ta>,<te>,<gu>,<ne>,<mr>,<kn>,<sa>,<ur>,<or>,<pa>,<as>,<bh>,"
    all_lang_set = "<eo>,<ku>,<sq>,<ja>,<hy>,<ko>,<th>,<en>,<kk>,<es>,<da>,<vi>,<fi>,<el>,<ms>,<nb>,<be>,<hi>,<mt>," \
                   "<gl>,<lt>,<ka>,<pt>,<uk>,<ta>,<de>,<sv>,<sr>,<sl>,<cs>,<fa>,<ru>,<mk>,<fr>,<hr>,<id>,<bg>,<sk>," \
                   "<tr>,<ro>,<so>,<et>,<ur>,<ar>,<ca>,<mr>,<pl>,<nl>,<zh>,<my>,<he>,<it>,<mn>,<sw>,<hu>,<gu>,<lv>," \
                   "<az>,<bs>,<af>,<bn>,<no>,<eu>"
    spm.SentencePieceTrainer.train(
        input=args.input_file,
        model_prefix=os.path.join(args.output_path, "bpe"),
        vocab_size=args.vocab_size,
        model_type="bpe",
        input_sentence_size=args.input_sentence_size,
        user_defined_symbols=all_lang_set,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  type=str, required=True,
                        help="input file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="output src file")
    parser.add_argument("--input_sentence_size", type=int, required=False, default=1000000,
                        help="input_sentence_size")
    parser.add_argument("--vocab_size", type=int, required=False, default=32000,
                        help="vocab_size")
    args = parser.parse_args()
    main(args)
