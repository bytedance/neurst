import json
import copy
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gigaspeech_file", type=str, required=True,
                        help="The GigaSpeech.json file.")
    parser.add_argument("--gigast_file", type=str, required=True,
                        help="The GigaST data file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="The output file.")
    args = parser.parse_args()

    aid_sid_seg = {}
    with open(args.gigast_file) as fp:
        gigast = json.load(fp)
    for audio in gigast.pop("audios"):
        for segment in audio["segments"]:
            sid = segment["sid"]
            aid = sid.split("_")[0]
            if aid not in aid_sid_seg:
                aid_sid_seg[aid] = {}
            aid_sid_seg[aid][sid] = segment

    data = copy.deepcopy(gigast)
    data["audios"] = []

    with open(args.gigaspeech_file) as fp:
        gigaspeech = json.load(fp)

    for audio in gigaspeech["audios"]:
        aid = audio["aid"]
        if aid not in aid_sid_seg:
            continue
        segments = audio.pop("segments")
        this_audio = copy.deepcopy(audio)
        this_audio["segments"] = []
        for segment in segments:
            sid = segment["sid"]
            matched_elems = aid_sid_seg.get(aid, None)
            if matched_elems is None:
                continue
            this_segment = copy.deepcopy(segment)
            this_segment["text_raw"] = ""
            this_segment["text_tn"] = matched_elems[sid]["text_raw"]
            this_segment["extra"] = matched_elems[sid]["extra"]
            this_audio["segments"].append(this_segment)
        data["audios"].append(this_audio)
        break
    with open(args.output_file, "w") as fw:
        json.dump(data, fw, indent=4)
