import json
import os
import tarfile
import tempfile
from tqdm import tqdm
from huggingface_hub import HfApi


def create_tar_file(image_paths, output_path, base_dir, max_size=5 * 1024 * 1024 * 1024):  # 5GB
    current_size = 0
    current_tar = None
    file_count = 0

    for img_path in tqdm(image_paths, desc="Creating tar files"):
        file_size = os.path.getsize(img_path)
        if current_tar is None or current_size + file_size > max_size:
            if current_tar:
                current_tar.close()
            file_count += 1
            current_tar_path = f"{output_path}_{file_count}.tar.gz"
            current_tar = tarfile.open(current_tar_path, "w:gz")
            current_size = 0

        arcname = os.path.relpath(img_path, base_dir)
        current_tar.add(img_path, arcname=arcname)
        current_size += file_size

    if current_tar:
        current_tar.close()


def process_and_upload_data(json_path, short_name, repo_id):
    with open(json_path, "r") as f:
        data = json.load(f) if not json_path.endswith(".jsonl") else [json.loads(line) for line in f]

    image_paths = set()
    processed_data = []
    base_dir = "/mnt/bn/vl-research/data/llava_data"

    for idx, item in enumerate(tqdm(data, desc="Processing data")):
        try:
            if "image" in item:
                image_path = os.path.join(base_dir, item['image'])
                if os.path.exists(image_path):
                    image_paths.add(image_path)
                else:
                    print(f"Image not found: {image_path}")
                    continue

            item_id = item.get("id", f"{idx:06d}")
            processed_item = {"id": item_id, "conversations": item["conversations"], "data_source": short_name}
            if "image" in item:
                processed_item["image"] = item["image"]
            processed_data.append(processed_item)

        except Exception as e:
            print(f"Error processing item {idx}: {e}")

    # Save processed data to JSON
    output_json = f"{short_name}_processed.json"
    with open(output_json, "w") as f:
        json.dump(processed_data, f)

    # Create tar files
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_base_path = os.path.join(tmpdir, f"{short_name}_images")
        create_tar_file(image_paths, tar_base_path, base_dir)

        # Upload files to Hugging Face Hub
        api = HfApi()
        api.upload_file(
            path_or_fileobj=output_json,
            path_in_repo=f"{short_name}/{output_json}",
            repo_id=repo_id,
            repo_type="dataset",
        )

        for tar_file in os.listdir(tmpdir):
            if tar_file.startswith(f"{short_name}_images") and tar_file.endswith(".tar.gz"):
                api.upload_file(
                    path_or_fileobj=os.path.join(tmpdir, tar_file),
                    path_in_repo=f"{short_name}/{tar_file}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

    print(f"Uploaded {short_name} data and images to {repo_id}")


# Example usage
json_paths = [
    "/mnt/bn/vl-research/data/llava_instruct/ureader_tr_sft.json",
    # "/mnt/bn/vl-research/data/llava_instruct/allava/Evol-Instruct-GPT4-Turbo-143K.json",
    "/mnt/bn/vl-research/data/llava_instruct/synthdog_zh/synthdog_zh_100k.json",
    "/mnt/bn/vl-research/data/llava_instruct/synthdog_en/synthdog_en_100k.json",
]

short_names = [
    "ureader_tr",
    # "evol_instruct",
    "synthdog_zh",
    "synthdog_en",
]

repo_id = "lmms-lab/LLaVA-OneVision-Mid-Data"

for json_path, short_name in zip(json_paths, short_names):
    process_and_upload_data(json_path, short_name, repo_id)