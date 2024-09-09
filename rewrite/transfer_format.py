import json
import os


def single_transfer(input_file, output_dir):
    res = []
    source = input_file.replace("InternVL2-Llama3-76B_all_data_new-", "").replace(".json_lmm_exp_response_1.jsonl", "")
    with open(input_file, "r") as fi:
        for line in fi.readlines():
            temp = json.loads(line)
            ori_id = temp["id"]
            ori_conversations = temp["conversations"]
            image = temp["image"]
            # source = temp["source"]
            response = temp["response"].replace("Reviced Answer: ", "").replace("<response: ", "")\
                .replace("Reviced Answer:", "").replace("<response:", "")
            if response.endswith(">"):
                response = response[:-1]
            gpt_res = ori_conversations[1]
            gpt_res["value"] = response
            conversations = [ori_conversations[0], gpt_res]
            curr = {"id": ori_id, "image": image, "conversations": conversations, "source": source}
            res.append(curr)
    file_name = source + "_new.json"
    with open(os.path.join(output_dir, file_name), "w") as fo:
        fo.write(json.dumps(res))


def transfer(input_dir, output_dir):
    for each in os.listdir(input_dir):
        if not each.endswith("jsonl"):
            continue
        single_transfer(each, output_dir)
    return


if __name__ == '__main__':
    transfer("rewritten_data", "format_data")
