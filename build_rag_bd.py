# build_rag_db.py
import os
from os.path import join
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import faiss  # for write_index

# Your existing RAG code
from rag.rag_appplication import EmbeddingModelContainer, VectorStore


def _safe_name(s: str) -> str:
    return (s or "unknown").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _pick_doc_ref(meta: dict) -> str:
    return (
        meta.get("document_reference")
        or meta.get("source")
        or meta.get("document_id")
        or "unknown"
    )


def _pick_page(meta: dict):
    return meta.get("page_number", meta.get("page", "unknown"))


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def build(args: DictConfig):
    # if not getattr(args, "use_rag", False):
    #     raise ValueError("This script builds the RAG DB only. Set use_rag=True in your config/CLI.")

    # -------- paths (robust under Hydra chdir) --------
    base_models = to_absolute_path(args.base_models)
    base_rag_documents = to_absolute_path(args.base_rag_documents)
    local_logging_dir = to_absolute_path(args.local_logging_dir)

    # -------- device selection (same policy as your main run) --------
    device = (
        torch.device("cuda:1") if torch.cuda.device_count() > 1
        else torch.device("cuda:0") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # -------- run/output dir --------
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    emb_name = _safe_name(getattr(args, "rag_embedding_model", "unknown_embedder"))
    rag_name = _safe_name(getattr(args, "rag_name", "rag"))
    pathology = _safe_name(getattr(args, "pathology", "NA"))

    run_name = f"RAGDB_{pathology}_{rag_name}_{emb_name}_{dt}"
    out_dir = join(local_logging_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save args snapshot (handy for provenance)
    with open(join(out_dir, "args_snapshot.yaml"), "w", encoding="utf-8") as f:
        f.write(str(args))

    # -------- embedding model (document embedder) --------
    embed_max_len = args.rag_embed_max_length if hasattr(args, "rag_embed_max_length") else 512
    embedding_model_container = EmbeddingModelContainer(
        model_name_or_path=args.rag_embedding_model,
        device=device,
        embed_max_length=embed_max_len,
    )
    embedding_model_container.load_model(base_models)

    # -------- document paths --------
    document_paths = [
        join(base_rag_documents, doc.lstrip("/"))
        for doc in args.rag_documents
    ]

    # Optional: pre-embedded vectors
    pre_embed_path = None
    if hasattr(args, "rag_pre_embed_path") and args.rag_pre_embed_path:
        pre_embed_path = join(base_rag_documents, args.rag_pre_embed_path.lstrip("/"))
        if not os.path.exists(pre_embed_path):
            raise FileNotFoundError(f"rag_pre_embed_path not found: {pre_embed_path}")

    # -------- build vector store (this is the “DB build”) --------
    vector_store = VectorStore(
        document_paths=document_paths,
        embedding_model_container=embedding_model_container,
        chunk_size=args.rag_chunk_size,
        chunk_overlap=args.rag_chunk_overlap,
        smart_chunking=args.rag_smart_chunking,
        pre_embed_path=pre_embed_path,
        use_bm25=args.rag_vector_store_use_bm25 if hasattr(args, "rag_vector_store_use_bm25") else False,
    )

    # -------- persist FAISS index --------
    faiss_index_path = join(out_dir, "faiss.index")
    faiss.write_index(vector_store.index, faiss_index_path)

    # -------- per-chunk table --------
    tok = embedding_model_container.tokenizer
    chunk_rows = []
    for row_id, doc in enumerate(vector_store.doc_chunks):
        text = doc.metadata.get("embed_text", doc.page_content)

        token_size = doc.metadata.get("token_size", None)
        if token_size is None and tok is not None:
            try:
                token_size = len(tok.encode(text, add_special_tokens=False))
            except Exception:
                token_size = None

        chunk_rows.append({
            "row_id": row_id,  # FAISS row id (ordering)
            "chunk_id": doc.metadata.get("chunk_id", None),
            "document_reference": _pick_doc_ref(doc.metadata),
            "source": doc.metadata.get("source", None),
            "document_id": doc.metadata.get("document_id", None),
            "format": doc.metadata.get("format", None),
            "page_number": _pick_page(doc.metadata),
            "order_in_document": doc.metadata.get("order_in_document", None),
            "token_size": token_size,
            "char_len": len(text),
            "word_count": len(text.split()),
        })

    df_chunks = pd.DataFrame(chunk_rows)
    chunks_csv = join(out_dir, "rag_db_chunks.csv")
    df_chunks.to_csv(chunks_csv, index=False)

    # -------- per-document summary --------
    def _uniq_join(x):
        vals = sorted({str(v) for v in x if v is not None})
        return ",".join(vals)

    g = df_chunks.groupby("document_reference", dropna=False)

    df_summary = g.agg(
        num_chunks=("row_id", "count"),
        formats=("format", _uniq_join),
        avg_token_size=("token_size", "mean"),
        std_token_size=("token_size", "std"),
        min_token_size=("token_size", "min"),
        max_token_size=("token_size", "max"),
        avg_char_len=("char_len", "mean"),
        avg_word_count=("word_count", "mean"),
        min_word_count=("word_count", "min"),
        max_word_count=("word_count", "max"),
    ).reset_index()

    total_chunks = len(df_chunks)
    df_summary["pct_of_db"] = df_summary["num_chunks"] / max(total_chunks, 1) * 100.0
    df_summary = df_summary.sort_values(["num_chunks", "document_reference"], ascending=[False, True])

    # Add new rows, mean and sum
    mean_row = {
        "document_reference": "ALL_DOCUMENTS_MEAN",
        "num_chunks": df_summary["num_chunks"].mean(),
        "formats": "ALL",
        "avg_token_size": df_summary["avg_token_size"].mean(),
        "std_token_size": df_summary["std_token_size"].mean(),
        "min_token_size": df_summary["min_token_size"].mean(),
        "max_token_size": df_summary["max_token_size"].mean(),
        "avg_char_len": df_summary["avg_char_len"].mean(),
        "avg_word_count": df_summary["avg_word_count"].mean(),
        "min_word_count": df_summary["min_word_count"].mean(),
        "max_word_count": df_summary["max_word_count"].mean(),
        "pct_of_db": 100.0,
    }
    sum_row = {
        "document_reference": "ALL_DOCUMENTS_SUM",
        "num_chunks": df_summary["num_chunks"].sum(),
        "formats": "ALL",
        "avg_token_size": np.nan,
        "std_token_size": np.nan,
        "min_token_size": np.nan,
        "max_token_size": np.nan,
        "avg_char_len": np.nan,
        "avg_word_count": np.nan,
        "min_word_count": df_summary["min_word_count"].min(),
        "max_word_count": df_summary["max_word_count"].max(),
        "pct_of_db": 100.0,
    }
    # Add rows for sum per pathology
    # 1. Tag each document_reference with pathology
    pathologies = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "other"]
    for pathology in pathologies:
        if pathology != "other":
            pathology_mask = df_summary["document_reference"].str.contains(pathology, case=False, na=False)
            if pathology_mask.any():
                pathology_sum_row = {
                    "document_reference": f"SUM_{pathology.upper()}",
                    "num_chunks": df_summary.loc[pathology_mask, "num_chunks"].sum(),
                    "formats": "ALL",
                    "avg_token_size": np.nan,
                    "std_token_size": np.nan,
                    "min_token_size": np.nan,
                    "max_token_size": np.nan,
                    "avg_char_len": np.nan,
                    "avg_word_count": np.nan,
                    "min_word_count": df_summary.loc[pathology_mask, "min_word_count"].min(),
                    "max_word_count": df_summary.loc[pathology_mask, "max_word_count"].max(),
                    "pct_of_db": (df_summary.loc[pathology_mask, "num_chunks"].sum() / total_chunks) * 100.0,
                }
                df_summary = pd.concat([df_summary, pd.DataFrame([pathology_sum_row])], ignore_index=True)
        else:
            pathology_mask = ~df_summary["document_reference"].str.contains(
                "|".join(pathologies[:-1]), case=False, na=False
            )
            if pathology_mask.any():
                pathology_sum_row = {
                    "document_reference": "SUM_OTHER",
                    "num_chunks": df_summary.loc[pathology_mask, "num_chunks"].sum(),
                    "formats": "ALL",
                    "avg_token_size": np.nan,
                    "std_token_size": np.nan,
                    "min_token_size": np.nan,
                    "max_token_size": np.nan,
                    "avg_char_len": np.nan,
                    "avg_word_count": np.nan,
                    "min_word_count": df_summary.loc[pathology_mask, "min_word_count"].min(),
                    "max_word_count": df_summary.loc[pathology_mask, "max_word_count"].max(),
                    "pct_of_db": (df_summary.loc[pathology_mask, "num_chunks"].sum() / total_chunks) * 100.0,
                }
                df_summary = pd.concat([df_summary, pd.DataFrame([pathology_sum_row])], ignore_index=True)
    

    df_summary = pd.concat([df_summary, pd.DataFrame([mean_row, sum_row])], ignore_index=True)

    summary_csv = join(out_dir, "rag_db_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    # -------- overall metadata --------
    overall = {
        "out_dir": out_dir,
        "timestamp": dt,
        "rag_name": getattr(args, "rag_name", None),
        "pathology": getattr(args, "pathology", None),
        "embedding_model": getattr(args, "rag_embedding_model", None),
        "embed_max_length": embed_max_len,
        "chunk_size": getattr(args, "rag_chunk_size", None),
        "chunk_overlap": getattr(args, "rag_chunk_overlap", None),
        "smart_chunking": getattr(args, "rag_smart_chunking", None),
        "pre_embed_path": pre_embed_path,
        "total_chunks": int(total_chunks),
        "unique_documents": int(df_chunks["document_reference"].nunique(dropna=False)),
        "faiss_index_path": faiss_index_path,
        "faiss_index_type": type(vector_store.index).__name__,
        "faiss_dimension": int(getattr(vector_store.index, "d", -1)),
        "use_bm25": bool(getattr(vector_store, "use_bm25", False)),
    }

    # Optional BM25 corpus stats (lengths based on BM25 tokenizer)
    if getattr(vector_store, "use_bm25", False) and vector_store.bm25_corpus_tokens is not None:
        lens = [len(toks) for toks in vector_store.bm25_corpus_tokens]
        overall.update({
            "bm25_num_docs": int(len(lens)),
            "bm25_avg_len": float(np.mean(lens)) if lens else None,
            "bm25_min_len": int(np.min(lens)) if lens else None,
            "bm25_max_len": int(np.max(lens)) if lens else None,
        })

    with open(join(out_dir, "rag_db_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n=== RAG DB build complete ===")
    print(f"Output dir: {out_dir}")
    print(f"- FAISS index:      {faiss_index_path}")
    print(f"- Chunk table:      {chunks_csv}")
    print(f"- Doc summary:      {summary_csv}")
    print(f"- Overall metadata: {join(out_dir, 'rag_db_overall.json')}")

if __name__ == "__main__":
    build()
