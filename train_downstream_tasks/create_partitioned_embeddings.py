#!/usr/bin/env python3
import os
import re
import glob
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from PartitionedEmbeddingModel import PartitionedEmbeddingModel

CRAFTED_FEATURES = 34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "ae", "net", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # or it's already a raw state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format (no state_dict found).")


def _infer_dims_from_state(sd):
    input_dim = None
    hidden_dim = None
    for dec_key in ("decoder.0.weight", "decoder.fc.weight", "decoder.weight"):
        if dec_key in sd:
            hidden_dim, input_dim = sd[dec_key].shape
            break
    if input_dim is None or hidden_dim is None:
        w = sd.get("encoder_unknown.0.weight", None)
        if w is None:
            raise ValueError("Could not infer dims from state_dict (no decoder/encoder keys found)")
        hidden_dim, input_dim = w.shape

    adv_hidden_dim = sd.get("discriminator.0.weight", torch.empty(0, 0)).shape[0] or max(512, hidden_dim // 2)

    known_dim = sd.get("encoder_known.2.weight", torch.empty(CRAFTED_FEATURES, hidden_dim)).shape[0]

    adv_output_dim = sd.get("discriminator.2.weight", torch.empty(CRAFTED_FEATURES, adv_hidden_dim)).shape[0]

    return dict(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        adv_hidden_dim=int(adv_hidden_dim),
        known_dim=int(known_dim),
        adv_output_dim=int(adv_output_dim),
    )


def latest_ckpt_for(model_dir: str, lam: str = "1.0") -> str:
    candidates: List[str] = []
    for lam_dir in (f"lambda_{lam}", f"lambda_{lam}.0"):
        pat = os.path.join("../train_plmx_stream/trained_models", model_dir, lam_dir, "epoch_*.pt")
        candidates.extend(glob.glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints under trained_models/{model_dir}/lambda_{lam}*")

    def _ep_num(p: str) -> int:
        m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=_ep_num)[-1]


class ESM2VAEModel(nn.Module):

    def __init__(self, esm2_model_name: str, vae_weights_path: str, lambda_str: str = "1.0"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(esm2_model_name)
        self.esm2 = AutoModel.from_pretrained(esm2_model_name).to(device)
        self.esm2.eval()

        for p in self.esm2.parameters():
            p.requires_grad = False

        ckpt_obj = torch.load(vae_weights_path, map_location="cpu")
        sd = _extract_state_dict(ckpt_obj)
        dims = _infer_dims_from_state(sd)

        self.hidden_size = int(self.esm2.config.hidden_size)
        if dims["input_dim"] != self.hidden_size:
            raise ValueError(
                f"Checkpoint expects input_dim={dims['input_dim']}, but {esm2_model_name} has hidden_size={self.hidden_size}.\n"
                f"Point to the matching weights for this backbone."
            )

        try:
            self.ae = PartitionedEmbeddingModel(
                input_dim=dims["input_dim"],
                known_dim=dims["known_dim"],
                hidden_dim=dims["hidden_dim"],
                adv_output_dim=dims["adv_output_dim"],
                adv_hidden_dim=dims["adv_hidden_dim"],  
            ).to(device)
        except TypeError:
            self.ae = PartitionedEmbeddingModel(
                input_dim=dims["input_dim"],
                known_dim=dims["known_dim"],
                hidden_dim=dims["hidden_dim"],
                adv_output_dim=dims["adv_output_dim"],
            ).to(device)

        self.ae.load_state_dict(sd, strict=True)
        self.ae.eval()

        print(
            f"[AE configured] input={dims['input_dim']}, hidden={dims['hidden_dim']}, "
            f"adv_hidden={dims['adv_hidden_dim']}, known={dims['known_dim']}, adv_out={dims['adv_output_dim']}"
        )

    @torch.no_grad()
    def _postprocess_one(self, embeddings_i: torch.Tensor, attn_i: torch.Tensor):
        """
        embeddings_i: (L_max, H) on device
        attn_i:       (L_max,) on device
        Returns CPU tensors:
            valid_embeddings: (L_valid, H)
            z:               (L_valid, H)
            z_trunc:         (L_valid, 34)
            z_binary_trunc:  (L_valid, 34)
        """
        valid = embeddings_i[attn_i == 1]
        if valid.shape[0] >= 2:
            valid = valid[1:-1]
        else:
            valid = torch.zeros((0, self.hidden_size), device=device)

        if valid.shape[0] == 0:
            z = torch.zeros_like(valid)
        else:
            _, _, _, z, _ = self.ae(valid)

        binary = z.clone()
        for i in range(valid.size(0)):
            row = z[i]
            ss8_idx = torch.argmax(row[23:31]).item()
            binary[i, 23:31] = 0
            if 0 <= ss8_idx < 8:
                binary[i, 23 + ss8_idx] = 1
            ss3_idx = torch.argmax(row[20:23]).item()
            binary[i, 20:23] = 0
            if 0 <= ss3_idx < 3:
                binary[i, 20 + ss3_idx] = 1
            aa_idx = torch.argmax(row[:20]).item()
            binary[i, :20] = 0
            if 0 <= aa_idx < 20:
                binary[i, aa_idx] = 1

        return (
            valid.detach().cpu(),
            z.detach().cpu(),
            z[:, :CRAFTED_FEATURES].detach().cpu(),
            binary[:, :CRAFTED_FEATURES].detach().cpu(),
        )

    @torch.no_grad()
    def forward_one(self, sequence: str):
        tokenized = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True).to(device)
        out = self.esm2(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
        embeddings = out.last_hidden_state.squeeze(0)
        attn = tokenized["attention_mask"].squeeze(0)
        return self._postprocess_one(embeddings, attn)

    @torch.no_grad()
    def forward_batch(self, sequences: List[str], max_length: int | None = None):
        tok = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True).to(device)
        if max_length is not None and tok["input_ids"].shape[1] > max_length:
            tok = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

        out = self.esm2(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        embeddings = out.last_hidden_state  # (B, L_max, H)
        attn = tok["attention_mask"]       # (B, L_max)

        valid_list, latent_list, trunc_list, binary_list = [], [], [], []
        B = embeddings.size(0)
        for i in range(B):
            v, z, zt, zb = self._postprocess_one(embeddings[i], attn[i])
            valid_list.append(v)
            latent_list.append(z)
            trunc_list.append(zt)
            binary_list.append(zb)
        return valid_list, latent_list, trunc_list, binary_list


def pad_all(arr_list, max_length, feature_dim):
    n = len(arr_list)
    out = np.zeros((n, max_length, feature_dim), dtype=np.float32)
    for i, a in enumerate(arr_list):
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        L = min(a.shape[0], max_length)
        if L > 0:
            out[i, :L, :a.shape[1]] = a[:L]
    return out


def ensure_dir_for_prefix(prefix: str):
    d = os.path.dirname(os.path.abspath(f"{prefix}_placeholder.txt"))
    if d:
        os.makedirs(d, exist_ok=True)


def generate_and_save_embeddings_fast(model: ESM2VAEModel, sequences: List[str], output_prefix: str,
                                      max_length=256, embedding_dim: int | None = None, batch_size=16, esm_max_len=None):
    ensure_dir_for_prefix(output_prefix)

    if embedding_dim is None:
        embedding_dim = model.hidden_size

    valid_list: List[torch.Tensor] = []
    latent_list: List[torch.Tensor] = []
    trunc_list:  List[torch.Tensor] = []
    binary_list: List[torch.Tensor] = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding (batched)"):
        batch = sequences[i:i + batch_size]
        v_list, z_list, t_list, b_list = model.forward_batch(batch, max_length=esm_max_len)
        valid_list.extend(v_list)
        latent_list.extend(z_list)
        trunc_list.extend(t_list)
        binary_list.extend(b_list)

    valid_np  = [v.numpy() for v in valid_list]
    latent_np = [z.numpy() for z in latent_list]
    trunc_np  = [t.numpy() for t in trunc_list]
    binary_np = [b.numpy() for b in binary_list]

    valid_means  = np.stack([v.mean(axis=0) if v.shape[0] > 0 else np.zeros((embedding_dim,), dtype=np.float32) for v in valid_np])
    latent_means = np.stack([z.mean(axis=0) if z.shape[0] > 0 else np.zeros((embedding_dim,), dtype=np.float32) for z in latent_np])
    trunc_means  = np.stack([t.mean(axis=0) if t.shape[0] > 0 else np.zeros((CRAFTED_FEATURES,), dtype=np.float32) for t in trunc_np])
    binary_means = np.stack([b.mean(axis=0) if b.shape[0] > 0 else np.zeros((CRAFTED_FEATURES,), dtype=np.float32) for b in binary_np])

    valids   = pad_all(valid_np,  max_length=max_length, feature_dim=embedding_dim)
    latents  = pad_all(latent_np, max_length=max_length, feature_dim=embedding_dim)
    truncs   = pad_all(trunc_np,  max_length=max_length, feature_dim=CRAFTED_FEATURES)
    binaries = pad_all(binary_np, max_length=max_length, feature_dim=CRAFTED_FEATURES)

    pd.DataFrame(valid_means).to_csv(f"{output_prefix}_valid_embeddings.csv", index=False)
    pd.DataFrame(latent_means).to_csv(f"{output_prefix}_latent_embeddings.csv", index=False)
    pd.DataFrame(trunc_means).to_csv(f"{output_prefix}_truncated_latent_embeddings.csv", index=False)
    pd.DataFrame(binary_means).to_csv(f"{output_prefix}_binary_embeddings.csv", index=False)

    np.save(f"{output_prefix}_valid_embeddings_full.npy", valids)
    np.save(f"{output_prefix}_latent_embeddings_full.npy", latents)
    np.save(f"{output_prefix}_truncated_latent_embeddings_full.npy", truncs)
    np.save(f"{output_prefix}_binary_embeddings_full.npy", binaries)


if __name__ == "__main__":
    max_length = 1024
    batch_size = 16
    esm_max_len = 1024
    lambda_str = "1.0"

    model_list = [
        "facebook/esm2_t33_650M_UR50D",
        "Rostlab/prot_bert_bfd",
        "facebook/esm2_t12_35M_UR50D",
    ]

    # Load datasets (adjust paths to your environment)
    train_agg_df = pd.read_csv("downstream_task_data/train_sequences_AGG.csv")
    test_agg_df  = pd.read_csv("downstream_task_data/test_sequences_AGG.csv")
    train_ev_df  = pd.read_csv("downstream_task_data/train_sequences_EV.csv")
    test_ev_df   = pd.read_csv("downstream_task_data/test_sequences_EV.csv")
    train_tm_df  = pd.read_csv("downstream_task_data/train_sequences_TM.csv")
    test_tm_df   = pd.read_csv("downstream_task_data/test_sequences_TM.csv")
    leptin_df   = pd.read_csv("downstream_task_data/leptin_sequence.csv")

    datasets = {
        "train_AGG": train_agg_df["Sequence"].tolist(),
        "test_AGG":  test_agg_df["Sequence"].tolist(),
        "train_EV":  train_ev_df["Sequence"].tolist(),
        "test_EV":   test_ev_df["Sequence"].tolist(),
        "train_TM":  train_tm_df["Sequence"].tolist(),
        "test_TM":   test_tm_df["Sequence"].tolist(),
        "leptin":   leptin_df["Sequence"].tolist(),
    }

    for hf_name in model_list:
        short_name = hf_name.split("/")[-1]
        print(f"\n=== Processing model: {hf_name} ===")

        vae_weights_path = latest_ckpt_for(short_name, lam=lambda_str)
        print(f"Using checkpoint: {vae_weights_path}")

        esm2_vae_model = ESM2VAEModel(
            esm2_model_name=hf_name,
            vae_weights_path=vae_weights_path,
            lambda_str=lambda_str,
        ).to(device)

        model_output_root = f"partitioned_embeddings/{short_name}"
        os.makedirs(model_output_root, exist_ok=True)

        for split_name, seqs in datasets.items():
            print(f"Generating embeddings for {split_name} ...")

            if "prot_bert" in hf_name.lower():
                seqs = [" ".join(list(s)) for s in seqs]

            output_prefix = os.path.join(model_output_root, split_name)
            generate_and_save_embeddings_fast(
                esm2_vae_model,
                seqs,
                output_prefix,
                max_length=max_length,
                embedding_dim=esm2_vae_model.hidden_size,
                batch_size=batch_size,
                esm_max_len=esm_max_len,
            )

    print("\n[Done] All models processed.")
