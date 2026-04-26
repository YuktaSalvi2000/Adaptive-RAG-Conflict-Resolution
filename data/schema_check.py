import json

def inspect_schema(path, name, num_samples=3):
    print(f"\n{'='*20} {name} {'='*20}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    for i, sample in enumerate(data[:num_samples]):
        print(f"\n--- Sample {i+1} ---")

        print("\nTop-level schema:")
        for k, v in sample.items():
            print(f"{k}: {type(v).__name__}")

        # HotpotQA context
        if "context" in sample:
            ctx = sample["context"]
            print("\nContext structure:")
            print("Type:", type(ctx).__name__)

            if isinstance(ctx, dict):
                print("Context keys:", list(ctx.keys()))
                for ck, cv in ctx.items():
                    print(f"  {ck}: {type(cv).__name__}")
                    if isinstance(cv, list) and len(cv) > 0:
                        print(f"  first {ck} item:", cv[0])
            elif isinstance(ctx, list) and len(ctx) > 0:
                print("First item type:", type(ctx[0]).__name__)
                print("Example:", ctx[0])

        # RAMDocs documents
        if "documents" in sample:
            docs = sample["documents"]
            print("\nDocuments structure:")
            print("Type:", type(docs).__name__)

            if isinstance(docs, list) and len(docs) > 0:
                print("First document type:", type(docs[0]).__name__)
                if isinstance(docs[0], dict):
                    print("First document keys:", list(docs[0].keys()))
                print("Example:", docs[0])

        if "supporting_facts" in sample:
            sf = sample["supporting_facts"]
            print("\nSupporting facts structure:")
            print("Type:", type(sf).__name__)
            print("Example:", sf)


inspect_schema("data/hotpot_dev.json", "Hotpot DEV")
inspect_schema("data/hotpot_train.json", "Hotpot TRAIN")
inspect_schema("data/ramdocs_test.json", "RAMDocs")