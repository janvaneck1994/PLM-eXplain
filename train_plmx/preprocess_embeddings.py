#!/usr/bin/env python3
import os
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import random
from typing import List, Tuple

PLDDT_THRESHOLD = 70.0

TRAIN_FRAC = 0.90
VAL_FRAC = 0.025
TEST_FRAC = 0.025
TRAIN_HOLDOUT_FRAC = 0.05

def load_esm2_model(model_name: str, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl

def load_protbert_model(model_name: str, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl

def generate_esm2_embedding(sequence: str, tokenizer, model, device: str = "cpu") -> np.ndarray:
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.squeeze(0).cpu().numpy()

def generate_protbert_embedding(sequence: str, tokenizer, model, device: str = "cpu") -> np.ndarray:
    spaced = " ".join(list(sequence.strip().upper()))
    inputs = tokenizer(
        spaced,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    reps = out.last_hidden_state.squeeze(0).cpu().numpy()
    if reps.shape[0] < 3:
        return np.zeros((0, reps.shape[-1] if reps.ndim == 2 else 0), dtype=np.float32)
    reps = reps[1:-1, :]
    return reps

def process_protein_file(
    input_file: str,
    model_type: str,
    tokenizer_or_collater,
    model,
    device: str = "cpu"
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_parquet(input_file)

    required_cols = ["Chain", "AminoAcid", "PLDDT"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing} in file {input_file}")

    protein_sequences = df.groupby("Chain")["AminoAcid"].apply(lambda x: "".join(x)).to_dict()
    feature_columns = list(df.columns)

    all_embeddings: List[np.ndarray] = []
    all_values: List[pd.DataFrame] = []
    all_trace: List[pd.DataFrame] = []

    src = os.path.basename(input_file)

    for chain_id, seq in protein_sequences.items():
        if model_type.lower().startswith("rostlab/prot_bert"):
            embedding = generate_protbert_embedding(seq, tokenizer_or_collater, model, device)
        else:
            embedding = generate_esm2_embedding(seq, tokenizer_or_collater, model, device)

        chain_df = df[df["Chain"] == chain_id].reset_index(drop=True)

        L_embed = embedding.shape[0]
        L_rows  = len(chain_df)
        L = min(L_embed, L_rows)
        if L_embed != L_rows:
            embedding = embedding[:L, :]
            chain_df = chain_df.iloc[:L, :].copy()

        trace_df = pd.DataFrame({
            "SourceFile": src,
            "Chain": chain_id,
            "ChainPosition": np.arange(L, dtype=int),
            "AminoAcid": chain_df["AminoAcid"].values[:L],
        })

        mask = chain_df["PLDDT"].values[:L] >= PLDDT_THRESHOLD
        if np.any(mask):
            all_embeddings.append(embedding[mask])
            all_values.append(chain_df.iloc[:L, :].loc[mask, feature_columns].copy())
            all_trace.append(trace_df.loc[mask].copy())

    if not all_embeddings:
        return (
            np.zeros((0, 0), dtype=np.float32),
            pd.DataFrame(columns=feature_columns),
            pd.DataFrame(columns=["SourceFile", "Chain", "ChainPosition", "AminoAcid"]),
        )

    combined_embeddings = np.vstack(all_embeddings)
    combined_values = pd.concat(all_values, ignore_index=True)
    combined_trace = pd.concat(all_trace, ignore_index=True)
    return combined_embeddings, combined_values, combined_trace

def process_files_serial(files: List[str], model_type: str, tokenizer_or_collater, model, device: str = "cpu"):
    results = []
    for f in tqdm(files, desc=f"Processing files (sequential) [{model_type}]"):
        try:
            emb, vals, trace = process_protein_file(f, model_type, tokenizer_or_collater, model, device)
            results.append((emb, vals, trace))
        except Exception as e:
            print(f"Error processing {f}: {e}")
    return results

def save_split_results(results, out_npz_path: str, out_csv_path: str, out_trace_path: str):
    valid = [
        (e, v, t) for (e, v, t) in results
        if isinstance(e, np.ndarray) and e.size > 0 and not v.empty and not t.empty
    ]
    if not valid:
        print(f"No valid residue rows after filtering. Skipping save for {out_npz_path}.")
        return

    all_embeddings = np.vstack([emb for emb, _, _ in valid])
    all_values = pd.concat([vals for _, vals, _ in valid], ignore_index=True)
    all_trace = pd.concat([tr for _, _, tr in valid], ignore_index=True)

    Path(out_npz_path).parent.mkdir(parents=True, exist_ok=True)

    print("all_embeddings: ", all_embeddings.shape)
    np.savez(out_npz_path, embeddings=all_embeddings)
    all_values.to_csv(out_csv_path, index=False)
    all_trace.to_csv(out_trace_path, index=False)

    print(f"Saved embeddings: {out_npz_path}")
    print(f"Saved values    : {out_csv_path}")
    print(f"Saved trace     : {out_trace_path}")

def split_file_list_four(
    files: List[str],
    seed: int = 42,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    holdout_frac: float = TRAIN_HOLDOUT_FRAC,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    assert train_frac >= 0 and val_frac >= 0 and test_frac >= 0 and holdout_frac >= 0
    total = train_frac + val_frac + test_frac + holdout_frac
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        print(f"[Warn] Fractions sum to {total:.6f} (expected 1.0). Continuing with rounding strategy.")

    files = sorted(files)
    rnd = random.Random(seed)
    rnd.shuffle(files)

    n = len(files)
    n_train  = int(math.floor(n * train_frac))
    n_val    = int(math.floor(n * val_frac))
    n_test   = int(math.floor(n * test_frac))
    n_hold   = int(math.floor(n * holdout_frac))

    remainder = n - (n_train + n_val + n_test + n_hold)
    n_train += remainder

    i0 = 0
    train_files         = files[i0:i0 + n_train];   i0 += n_train
    val_files           = files[i0:i0 + n_val];     i0 += n_val
    test_files          = files[i0:i0 + n_test];    i0 += n_test
    train_holdout_files = files[i0:i0 + n_hold];    i0 += n_hold

    assert len(set(train_files) & set(val_files)) == 0
    assert len(set(train_files) & set(test_files)) == 0
    assert len(set(train_files) & set(train_holdout_files)) == 0
    assert len(set(val_files) & set(test_files)) == 0
    assert len(set(val_files) & set(train_holdout_files)) == 0
    assert len(set(test_files) & set(train_holdout_files)) == 0
    assert i0 == n

    return train_files, val_files, test_files, train_holdout_files

def collect_input_files(input_folder: str):
    exts = (".csv", ".parquet")
    return [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(exts)
    ]

def run_for_model(
    input_folder: str,
    output_base_dir: str,
    model_name: str,
    device: str = "cpu",
    seed: int = 42,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    holdout_frac: float = TRAIN_HOLDOUT_FRAC,
    max_files: int = None,  # set naar int voor snelle sanity checks
):
    files = collect_input_files(input_folder)
    if not files:
        print(f"No CSV/Parquet files found in {input_folder}.")
        return
    if max_files:
        files = files[:max_files]

    train_files, val_files, test_files, train_holdout_files = split_file_list_four(
        files,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        holdout_frac=holdout_frac,
    )

    if model_name.lower().startswith("rostlab/prot_bert"):
        tokenizer_or_collater, model = load_protbert_model(model_name, device)
        model_stub = model_name.split("/")[-1]
    else:
        tokenizer_or_collater, model = load_esm2_model(model_name, device)
        model_stub = model_name.split("/")[-1]

    out_dir = Path(output_base_dir) / model_stub
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_stub} | Split: test | Files: {len(test_files)}")
    test_results = process_files_serial(test_files, model_name, tokenizer_or_collater, model, device=device)
    save_split_results(
        test_results,
        out_npz_path=str(out_dir / "embeddings_test.npz"),
        out_csv_path=str(out_dir / "values_test.csv"),
        out_trace_path=str(out_dir / "trace_test.csv"),
    )

    print(f"Model: {model_stub} | Split: val | Files: {len(val_files)}")
    val_results = process_files_serial(val_files, model_name, tokenizer_or_collater, model, device=device)
    save_split_results(
        val_results,
        out_npz_path=str(out_dir / "embeddings_val.npz"),
        out_csv_path=str(out_dir / "values_val.csv"),
        out_trace_path=str(out_dir / "trace_val.csv"),
    )

    print(f"Model: {model_stub} | Split: train_holdout | Files: {len(train_holdout_files)}")
    holdout_results = process_files_serial(train_holdout_files, model_name, tokenizer_or_collater, model, device=device)
    save_split_results(
        holdout_results,
        out_npz_path=str(out_dir / "embeddings_train_holdout.npz"),
        out_csv_path=str(out_dir / "values_train_holdout.csv"),
        out_trace_path=str(out_dir / "trace_train_holdout.csv"),
    )

    print(f"Model: {model_stub} | Split: train | Files: {len(train_files)}")
    train_results = process_files_serial(train_files, model_name, tokenizer_or_collater, model, device=device)
    save_split_results(
        train_results,
        out_npz_path=str(out_dir / "embeddings_train.npz"),
        out_csv_path=str(out_dir / "values_train.csv"),
        out_trace_path=str(out_dir / "trace_train.csv"),
    )

if __name__ == "__main__":
    INPUT_FOLDER = "data/residue"
    OUTPUT_BASE_DIR = "data/output/embeddings/"
    DEVICE = "cuda"
    SEED = 42

    MODELS_TO_RUN = [
        "facebook/esm2_t12_35M_UR50D",
        "Rostlab/prot_bert_bfd",
        "facebook/esm2_t33_650M_UR50D",
    ]
    # -----------------------------------------

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    for model_name in MODELS_TO_RUN:
        run_for_model(
            input_folder=INPUT_FOLDER,
            output_base_dir=OUTPUT_BASE_DIR,
            model_name=model_name,
            device=DEVICE,
            seed=SEED,
            train_frac=TRAIN_FRAC,
            val_frac=VAL_FRAC,
            test_frac=TEST_FRAC,
            holdout_frac=TRAIN_HOLDOUT_FRAC,
            max_files=None,
        )
