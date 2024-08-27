from datasets import load_dataset
import json

ds = load_dataset("lmms-lab/egoschema", "Subset")
id_list = []
sample_id = 0
res = []

for each in ds["test"]:
    question_idx = each["question_idx"]
    video = each["video_idx"]
    q_id = video
    if q_id in id_list:
        print("exist q_id")
    else:
        id_list.append(q_id)
    inputs = "Based on the video content you've just observed, answer the following " \
             "multiple-choice question. Carefully consider the question and options provided, " \
             "then select the most appropriate answer."
    inputs += "\nQuestion: " + each["question"]
    inputs += "\nOptions:\n" + "\n".join(each["option"])
    outputs = "The answer is: " + each["option"][int(each["answer"])]
    curr = {"video": video,
            "conversations": [{"from": "human", "value": inputs},
                              {"from": "gpt", "value": outputs}],
            "id": q_id, "sample_id": sample_id,
            "metadata": {
                "dataset": "egoschema",
                "split": "test",
                "task_instruction": "",
                "num_sample": 500,
                "question_type": "MCQ"
            }}
    sample_id += 1
    res.append(curr)

with open("../mm_instruct_data/egoschema_instruct_data.json", "w") as fo:
    fo.write(json.dumps(res, indent=2))




