import os
import json
import numpy as np
import torch

from functools import partial
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


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, data_source):
        self.data = data
        self.encoded_texts = []

        # Pretokenize all entries.
        for entry in data:
            # Prepare LIMA dataset separately. May contain multi-turn
            # conversations.
            if data_source == "lima":
                conversations = entry["conversations"]
                for i in range(0, len(conversations) - 1, 2):
                    instruction = conversations[i]
                    response = conversations[i + 1]
                    formatted_text = format_instruction_response(instruction, response)

                    self.encoded_texts.append(
                        tokenizer.encode(
                            formatted_text, allowed_special={"<|endoftext|>"}
                        )
                    )
            elif data_source == "concat":
                if "conversations" in entry:
                    conversations = entry["conversations"]
                    for i in range(0, len(conversations) - 1, 2):
                        instruction = conversations[i]
                        response = conversations[i + 1]
                        formatted_text = format_instruction_response(
                            instruction, response
                        )

                        self.encoded_texts.append(
                            tokenizer.encode(
                                formatted_text, allowed_special={"<|endoftext|>"}
                            )
                        )
                else:
                    self.encoded_texts.append(
                        tokenizer.encode(
                            entry["text"], allowed_special={"<|endoftext|>"}
                        )
                    )
            else:
                # Alpaca GPT-4 dataset includes "text" key which is already
                # formatted correctly.
                self.encoded_texts.append(
                    tokenizer.encode(entry["text"], allowed_special={"<|endoftext|>"})
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]


def instruct_collate_fn(
    batch, pad_token_id, device, ignore_idx=-100, allowed_max_len=None
):
    packed_sequences = []

    current_pack = []
    current_len = 0

    for seq in batch:
        seq_len = len(seq)

        if seq_len + current_len <= allowed_max_len:
            current_pack.extend(seq)
            current_len += seq_len
        else:
            if current_pack:
                packed_sequences.append(current_pack)

            if seq_len > allowed_max_len:
                for i in range(0, seq_len, allowed_max_len):
                    packed_sequences.append(seq[i : i + allowed_max_len])

                current_pack = []
                current_len = 0
            else:
                current_pack = list(seq)
                current_len = seq_len

    if current_pack:
        packed_sequences.append(current_pack)

    batch_max_len = max(len(item) + 1 for item in packed_sequences)
    inputs, targets = [], []

    for item in packed_sequences:
        new_item = list(item)
        new_item.append(pad_token_id)

        padded = new_item + [pad_token_id] * (batch_max_len - len(new_item))

        input = torch.tensor(padded[:-1])
        target = torch.tensor(padded[1:])

        mask = target == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        # Mask padding token IDs so training loss is not affected.
        if indices.numel() > 1:
            target[indices[1:]] = ignore_idx

        inputs.append(input)
        targets.append(target)

    inputs_tensor = torch.stack(inputs).to(device)
    targets_tensor = torch.stack(targets).to(device)

    return inputs_tensor, targets_tensor


def prepare_instruct_data_loaders(
    cfg,
    tokenizer,
    file_path,
    eos_id,
    device="cpu",
    allowed_max_len=1024,
    data_source="",
):
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line.strip()) for line in f]

    print(f"Loaded {len(dataset)} samples from '{file_path}'")

    init_collate_fn = partial(
        instruct_collate_fn,
        pad_token_id=eos_id,
        device=device,
        allowed_max_len=allowed_max_len,
    )

    train_size = int(len(dataset) * 0.90)

    train_data = dataset[:train_size]
    val_data = dataset[train_size:]

    train_dataset = InstructionDataset(train_data, tokenizer, data_source)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.n_batch,
        drop_last=True,
        shuffle=True,
        num_workers=cfg.n_workers,
        collate_fn=init_collate_fn,
    )

    val_dataset = InstructionDataset(val_data, tokenizer, data_source)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.n_batch,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.n_val_workers,
        collate_fn=init_collate_fn,
    )

    return train_loader, val_loader


def format_instruction_response(instruction, response):
    formatted = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
        f"\n\n### Response:\n{response}"
    )
    return formatted
