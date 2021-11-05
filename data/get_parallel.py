import argparse
import os
import random



def find_lang_code(sentence, langcode_dict):
    for key in langcode_dict.keys():
        val = langcode_dict[key]
        if sentence.find(val) != -1:
            return val
    return None


def file_len(fname):
    with open(fname) as f:
        size = len([0 for _ in f])
        return size

def write_to_trainfile(src_in, src_out, tgt_in, tgt_out, code_in, code_out,
                       select_pr=1.0, ):
    with open(src_in, "r") as src_infile, \
            open(src_out, "r") as src_outfile, \
            open(tgt_in, "w+") as train_in, \
            open(tgt_out, "w+") as train_out:
        for line_srcin, line_srcout in zip(src_infile, src_outfile):
            if select_pr < 1.0 and random.random() > select_pr:
                continue
            line_srcin = "<%s> " % code_out + line_srcin
            train_in.write(line_srcin.strip() + "\n")
            train_out.write(line_srcout.strip() + "\n")


def process_mono(mono_dir, args):
    file_list = os.listdir(mono_dir)
    file_list = list(filter(lambda x: args.mode in x, file_list))
    if len(file_list) != 1:
        raise IOError("more than one file in mono dir")
    mono_code = mono_dir.split(os.sep)[-1]
    train_input_name = args.output_path + os.sep + "%s.%s2%s.in" % (args.mode, mono_code, mono_code)
    train_output_name = args.output_path + os.sep + "%s.%s2%s.out" % (args.mode, mono_code, mono_code)

    write_to_trainfile(
        src_in=mono_dir + os.sep + file_list[0],
        src_out=mono_dir + os.sep + file_list[0],
        tgt_in=train_input_name,
        tgt_out=train_output_name,
        code_in=mono_code,
        code_out=mono_code,
    )


def process_para(para_dir, args):
    file_list = os.listdir(para_dir)
    file_list = list(filter(lambda x: args.mode in x, file_list))
    if len(file_list) != 2:
        raise IOError("not two files in para dir")
    src_code, tgt_code = para_dir.split(os.sep)[-1].split("_")
    if src_code > tgt_code:
        src_code, tgt_code = tgt_code, src_code
    train_input_name = args.output_path + os.sep + "%s.%s2%s.in" % (args.mode, src_code, tgt_code)
    train_output_name = args.output_path + os.sep + "%s.%s2%s.out" % (args.mode, src_code, tgt_code)
    src_filename = file_list[0].split(".")[0]

    write_to_trainfile(
        src_in=para_dir + os.sep + src_filename + "." + src_code,
        src_out=para_dir + os.sep + src_filename + "." + tgt_code,
        tgt_in=train_input_name,
        tgt_out=train_output_name,
        code_in=src_code,
        code_out=tgt_code,
    )
    train_input_name = args.output_path + os.sep + "%s.%s2%s.in" % (args.mode, tgt_code, src_code)
    train_output_name = args.output_path + os.sep + "%s.%s2%s.out" % (args.mode, tgt_code, src_code)
    write_to_trainfile(
        src_in=para_dir + os.sep + src_filename + "." + tgt_code,
        src_out=para_dir + os.sep + src_filename + "." + src_code,
        tgt_in=train_input_name,
        tgt_out=train_output_name,
        code_in=tgt_code,
        code_out=src_code,
    )


def main(args):
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)
    sub_dir_list = os.listdir(args.input_path)
    sub_dir_list = filter(lambda x: os.path.isdir(args.input_path + os.sep + x), sub_dir_list)
    sub_dir_list = list(sub_dir_list)
    for sub_dir in sub_dir_list:
        # mono case
        if "_" not in sub_dir:
            process_mono(args.input_path + os.sep + sub_dir, args)
        else:
            process_para(args.input_path + os.sep + sub_dir, args)


if __name__ == '__main__':
    # example input = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="pipetest",
                        # required=True,
                        help="input file")
    parser.add_argument("--output_path", type=str,
                        default="pipetest_out",
                        # required=True,
                        help="output src file")
    parser.add_argument("--mode", type=str,
                        default="train",
                        # required=True,
                        help="the file mode (train/test/dev)")
    args = parser.parse_args()
    main(args)
