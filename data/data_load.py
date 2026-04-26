from datasets import load_dataset
import json
import os

# save inside Adaptive-RAG-Conflict-Resolution/data/
BASE_DIR = os.path.dirname(__file__)

# dataset = load_dataset("hotpot_qa", "distractor")

# dev_data = dataset["validation"].to_list()
# train_data = dataset["train"].to_list()

# with open(os.path.join(BASE_DIR, "hotpot_dev.json"), "w", encoding="utf-8") as f:
#     json.dump(dev_data, f, ensure_ascii=False, indent=2)

# with open(os.path.join(BASE_DIR, "hotpot_train.json"), "w", encoding="utf-8") as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=2)

# print("Saved:")
# print(os.path.join(BASE_DIR, "hotpot_dev.json"))
# print(os.path.join(BASE_DIR, "hotpot_train.json"))

dataset_ramdocs = load_dataset("HanNight/RAMDocs")

# usually RAMDocs only has test
ramdocs_test = dataset_ramdocs["test"].to_list()

with open(os.path.join(BASE_DIR, "ramdocs_test.json"), "w", encoding="utf-8") as f:
    json.dump(ramdocs_test, f, ensure_ascii=False, indent=2)

print("Saved RAMDocs test to:", os.path.join(BASE_DIR, "ramdocs_test.json"))