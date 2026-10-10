import io
import polars as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from core.config import MENU, ITEM_TO_ID, NUM_ITEMS, VALID_PAIR_SET

def process_dynamic_csv(file_buffer, batch_size=32, seed=42):
    rng = np.random.default_rng(seed)
    content = file_buffer.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    df_raw = pl.read_csv(io.BytesIO(content))
    file_buffer.seek(0)
    
    df_clean = (
        df_raw.select(["Order ID", "Item Name"])
        .filter(pl.col("Item Name").is_in(MENU))
        .with_columns(pl.col("Item Name").replace(ITEM_TO_ID).cast(pl.Int64).alias("item_id"))
    )
    
    pairs = (
        df_clean.join(df_clean, on="Order ID", suffix="_target")
        .filter(pl.col("item_id") != pl.col("item_id_target"))
        .select(["item_id", "item_id_target"])
    )
    
    pos_anchors = pairs["item_id"].to_numpy()
    pos_targets = pairs["item_id_target"].to_numpy()
    pos_labels = np.ones(len(pos_anchors), dtype=np.float32)

    neg_anchors, neg_targets = [], []
    while len(neg_anchors) < len(pos_anchors):
        a = int(rng.integers(0, NUM_ITEMS))
        b = int(rng.integers(0, NUM_ITEMS))
        if a != b and (a, b) not in VALID_PAIR_SET:
            neg_anchors.append(a)
            neg_targets.append(b)

    all_a = np.concatenate([pos_anchors, np.array(neg_anchors)])
    all_t = np.concatenate([pos_targets, np.array(neg_targets)])
    all_y = np.concatenate([pos_labels, np.zeros(len(neg_anchors), dtype=np.float32)])

    ds = TensorDataset(torch.tensor(all_a, dtype=torch.long), torch.tensor(all_t, dtype=torch.long), torch.tensor(all_y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def build_global_market_testloader(batch_size=64):
    pos_a, pos_t = [], []
    for (a, b) in VALID_PAIR_SET:
        pos_a.extend([a] * 10)
        pos_t.extend([b] * 10)
    
    rng = np.random.default_rng(999)
    neg_a, neg_t = [], []
    while len(neg_a) < len(pos_a):
        a, b = int(rng.integers(0, NUM_ITEMS)), int(rng.integers(0, NUM_ITEMS))
        if a != b and (a, b) not in VALID_PAIR_SET:
            neg_a.append(a)
            neg_t.append(b)
            
    all_a = np.concatenate([np.array(pos_a), np.array(neg_a)])
    all_t = np.concatenate([np.array(pos_t), np.array(neg_t)])
    all_y = np.concatenate([np.ones(len(pos_a), dtype=np.float32), np.zeros(len(neg_a), dtype=np.float32)])
    
    ds = TensorDataset(torch.tensor(all_a, dtype=torch.long), torch.tensor(all_t, dtype=torch.long), torch.tensor(all_y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)