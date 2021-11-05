import argparse
import os


def get_file_len(fname):
    with open(fname) as f:
        size = len([0 for _ in f])
        return size


def normalize_length(lang_length_dict, total_samples=10000000, power=0.2):
    sum_len = 0
    sum_len_power = 0
    min_len = float("inf")
    for language in lang_length_dict.keys():
        sum_len += lang_length_dict[language]
        sum_len_power += lang_length_dict[language] ** power
        min_len = min(min_len, lang_length_dict[language])
    
    for language in lang_length_dict.keys():
        lang_length_dict[language] = int((lang_length_dict[language] ** power / sum_len_power) * total_samples)
    return lang_length_dict


def main(args):
    file_list = os.listdir(args.input_path)
    lang_length_dict = {}
    lang_file_dict = {}
    for file_name in file_list:
        mode, language, inout = file_name.split(".")
        if inout == "in":
            language = language.split("2")[0]
        elif inout == "out":
            language = language.split("2")[-1]
        input_file = os.path.join(args.input_path, file_name)
        file_len = get_file_len(input_file)
        if language not in lang_length_dict.keys():
            lang_length_dict[language] = file_len
            lang_file_dict[language] = [input_file]
        else:
            lang_length_dict[language] += file_len
            lang_file_dict[language] += [input_file]
    lang_length_dict = normalize_length(lang_length_dict, total_samples=args.total_samples)
    for language in lang_length_dict.keys():
        print("cat %s | shuf -n %s > tmp_%s" % (" ".join(lang_file_dict[language]), lang_length_dict[language], language))
    print("cat tmp_* > samples")
    for language in lang_length_dict.keys():
        print("rm tmp_%s"%language)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="",
                        help="input file path")
    parser.add_argument("--total_samples", type=int,
                        default=10000000,
    )

    args = parser.parse_args()
    main(args)
