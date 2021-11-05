import argparse
import os


def main(args):
    if args.output_path is None:
        args.output_path = args.input_path+"_clean"
    if not os.path.isdir(args.output_path):
        print("mkdir -p %s"%args.output_path)
    file_list = os.listdir(args.input_path)
    if args.suffix is not None:
        file_list = list(filter(lambda x: x.endswith(args.suffix), file_list))

    normalize = os.path.join(args.moses_path, "scripts/tokenizer/normalize-punctuation.perl")
    rmnprint = os.path.join(args.moses_path, "scripts/tokenizer/remove-non-printing-char.perl")
    unescape = os.path.join(args.moses_path, "scripts/tokenizer/deescape-special-chars.perl")

    for file in file_list:
        mode, language, inout  = file.split(".")
        if inout == "in":
            language=language.split("2")[0]
        elif inout == "out":
            language = language.split("2")[-1]
        input_file = os.path.join(args.input_path, file)
        output_file = os.path.join(args.output_path, file)
        print("perl %s -l %s < %s | "%(normalize, language, input_file),
              "perl %s -l %s | "%(rmnprint, language),
              "perl %s -l %s > %s" % (unescape, language, output_file),
              )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--moses_path", type=str,
                        default="",
                        help="path for Moses")
    parser.add_argument("--input_path", type=str,
                        default="",
                        help="input file path")
    parser.add_argument("--output_path", type=str,
                        default=None,
                        help="output file path")
    parser.add_argument("--suffix", type=str,
                        default=None,
                        help="filter files with given suffix (for test set process)")
    args = parser.parse_args()
    main(args)
