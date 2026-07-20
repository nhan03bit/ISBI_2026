import os
import io
import ast
import math
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
base_dir = "."
train_csv_path = os.path.join(base_dir, "train_fold_1.csv")
val_csv_path = os.path.join(base_dir, "val_fold_1.csv")
train_output_dir = os.path.join(base_dir, "wds_shards_train_raw")
val_output_dir = os.path.join(base_dir, "wds_shards_val_raw")

# All PadChest PNGs now live (flat) in a single folder after extract_padchest.sh.
# Set this to wherever your images are. If they are ever nested in sub-folders,
# the recursive index below still finds them.
IMAGE_ROOT = os.path.join(base_dir, "padchest")

shard_size = 10000


# ---------------------------------------------------------------------------
# Build a filename -> full path index ONCE (fast O(1) lookups afterwards)
# ---------------------------------------------------------------------------
def build_image_index(root):
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # last one wins if a name repeats; PadChest names are unique anyway
            index[fn] = os.path.join(dirpath, fn)
    return index


print(f"Indexing images under: {IMAGE_ROOT}")
image_index = build_image_index(IMAGE_ROOT)
print(f"Found {len(image_index)} image files")


def find_image(filename):
    return image_index.get(filename)


# ---------------------------------------------------------------------------
# Shard writer (label alignment unchanged: labels[sample_idx] tracks df order)
# ---------------------------------------------------------------------------
def write_shards(df, labels, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(df)
    num_shards = math.ceil(num_samples / shard_size)
    print(f"{split_name}: {num_samples} samples, {num_shards} shards")

    missing = 0
    sample_idx = 0
    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{split_name}_{shard_idx:04d}.tar")
        done_marker = shard_path + ".done"
        shard_samples = min(shard_size, num_samples - sample_idx)

        if os.path.exists(done_marker):
            print(f"Skipping {shard_path} (already done)")
            sample_idx += shard_samples
            continue

        if os.path.exists(shard_path):
            # No .done marker means a previous run was killed mid-write (e.g. hit
            # the Slurm time limit) and left a truncated tar behind; redo it.
            print(f"Removing incomplete shard {shard_path} (no .done marker)")
            os.remove(shard_path)

        with wds.TarWriter(shard_path) as dst:
            for _ in tqdm(range(shard_size), desc=f"{split_name} shard {shard_idx + 1}/{num_shards}"):
                if sample_idx >= num_samples:
                    break

                row = df.iloc[sample_idx]
                filename = row["filename"]
                image_path = find_image(filename)

                if image_path is None:
                    missing += 1
                    sample_idx += 1
                    continue

                try:
                    with open(image_path, "rb") as f:
                        img_bytes = f.read()

                    label_vector = torch.tensor(labels[sample_idx], dtype=torch.float32)
                    buffer_y = io.BytesIO()
                    torch.save(label_vector, buffer_y)

                    sample = {
                        "__key__": f"{split_name}_{sample_idx:08d}",
                        "img": img_bytes,
                        "cls": buffer_y.getvalue(),
                    }
                    dst.write(sample)

                except Exception as e:
                    print(f"Lỗi ảnh {filename}: {e}")

                sample_idx += 1

        open(done_marker, "w").close()
        print(f"Saved {shard_path}")

    if missing:
        print(f"[{split_name}] WARNING: {missing}/{num_samples} images not found and skipped")


# ---------------------------------------------------------------------------
# Load CSVs, binarize labels, write shards
# ---------------------------------------------------------------------------
df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)

df_train["label_list"] = df_train["label"].apply(ast.literal_eval)
df_val["label_list"] = df_val["label"].apply(ast.literal_eval)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(df_train["label_list"])
y_val = mlb.transform(df_val["label_list"])

os.makedirs(train_output_dir, exist_ok=True)
torch.save({"class_names": mlb.classes_}, os.path.join(train_output_dir, "label_info.pt"))

print(f"Tổng số lớp: {len(mlb.classes_)}")

write_shards(df_train, y_train, train_output_dir, "train")
write_shards(df_val, y_val, val_output_dir, "val")

print("Done Create WebDataset shards!")