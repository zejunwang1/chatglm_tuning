# coding=utf-8
# email: wangzejunscut@126.com

import argparse
import json

def main(args):
    f = open(args.output_file, mode="w")
    with open(args.input_file, mode="r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line.strip())
            json_dict = {}
            json_dict["instruction"] = data["content"]
            json_dict["output"] = data["summary"]
            f.write(json.dumps(json_dict, ensure_ascii=False))
            f.write("\n")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)

