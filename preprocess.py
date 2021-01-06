
import json

with open("data/category_id.txt", 'r') as w:
    valid_catg = [int(s.split(" ")[0]) for s in w]


def preprocess(filepath, storepath):
    data = []

    with open(filepath, 'r') as f:
        raw = json.load(f)
    for style in raw:
        style["items"] = [item for item in style["items"]
                          if (item["categoryid"] in valid_catg)]
        if len(style["items"]) <= 1 or len(style["items"]) > 6:
            continue
        data.append(style)
    print(len(data))
    with open(storepath, 'w') as w:
        json.dump(data, w, indent=4)


if __name__ == "__main__":
    print("train")
    preprocess("data/train_no_dup.json", "data/train_filtered.json")
    print("test")
    preprocess("data/test_no_dup.json", "data/test_filtered.json")
    label = {}
    with open("data/category_id.txt", 'r') as w:
        for idx, s in enumerate(w):
            l = s.split(" ")
            label[int(l[0])] = idx
