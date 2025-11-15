import os
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk


class GPTDataset(Dataset):
    def __init__(
        self, token_ids, tokenizer, max_length, stride, mmap_dir, split, train_size
    ):
        print(f"Creating dataset '{split}'...")

        if split == "train":
            start_idx = 0
            end_idx = train_size
        else:  # validation
            start_idx = train_size
            end_idx = len(token_ids)

        input_file = os.path.join(mmap_dir, "input_ids.npy")
        target_file = os.path.join(mmap_dir, "target_ids.npy")

        if os.path.exists(input_file) and os.path.exists(target_file):
            print(f"Loading pre-processed mmap dataset '{split}' from disk...")
            self.input_ids = np.memmap(
                input_file,
                dtype=np.int32,
                mode="r",
                shape=(end_idx - start_idx, max_length),
            )
            self.target_ids = np.memmap(
                target_file,
                dtype=np.int32,
                mode="r",
                shape=(end_idx - start_idx, max_length),
            )
        else:
            os.makedirs(mmap_dir, exist_ok=True)

            self.input_ids = np.memmap(
                input_file,
                dtype=np.int32,
                mode="w+",
                shape=(end_idx - start_idx, max_length),
            )
            self.target_ids = np.memmap(
                target_file,
                dtype=np.int32,
                mode="w+",
                shape=(end_idx - start_idx, max_length),
            )

            # Process the dataset and write to memory-mapped files.
            idx = 0
            for i, ids in tqdm(
                enumerate(token_ids),
                desc=f"Processing {split} dataset",
                total=end_idx - start_idx,
            ):
                if split == "train" and i >= train_size:
                    break

                if split == "val" and i < train_size:
                    continue

                for j in range(0, len(ids) - max_length, stride):
                    input_chunk = ids[j : j + max_length]
                    target_chunk = ids[j + 1 : j + max_length + 1]

                    self.input_ids[idx] = input_chunk
                    self.target_ids[idx] = target_chunk

                    idx += 1

        print(f"Dataset '{split}' created")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(
            self.target_ids[idx], dtype=torch.long
        )


def prepare_data_loaders(cfg, tokenizer, cache_dir="dataset/openwebtext2_tokenized"):
    if os.path.exists(cache_dir):
        dataset = load_from_disk(cache_dir)
    else:
        dataset = load_dataset("Geralt-Targaryen/openwebtext2", split="train")

        def tokenize(example):
            tokens = tokenizer.encode(
                example["text"], allowed_special={"<|endoftext|>"}
            )
            return {"token_ids": tokens}

        dataset = dataset.map(tokenize, remove_columns=["text"], num_proc=cfg.n_workers)

        dataset.save_to_disk(cache_dir)

    token_ids = dataset["token_ids"]

    # 90/10 dataset split.
    train_size = int(0.9 * len(token_ids))

    train_dataset = GPTDataset(
        token_ids,
        tokenizer,
        max_length=cfg.n_ctx,
        stride=cfg.n_ctx,
        mmap_dir="dataset/train_mmap",
        split="train",
        train_size=train_size,
    )

    val_dataset = GPTDataset(
        token_ids,
        tokenizer,
        max_length=cfg.n_ctx,
        stride=cfg.n_ctx,
        mmap_dir="dataset/val_mmap",
        split="val",
        train_size=train_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.n_batch,
        drop_last=True,
        shuffle=True,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.n_batch,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.n_val_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
