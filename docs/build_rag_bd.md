# Build the RAG corpus index

`build_rag_bd.py` builds a FAISS index and corpus statistics using
`configs/config.yaml`, `configs/paths/`, and the selected `configs/rag/` profile.

```bash
python build_rag_bd.py rag=Qwen3-Embedding-0.6B_skipstep \
  base_models=/path/to/models base_rag_documents=/path/to/guidelines \
  local_logging_dir=/path/to/output
```

| Argument | Purpose |
| --- | --- |
| `paths`, `rag` | Path and retrieval YAML profile names. |
| `base_models` | Local embedding-model directory. |
| `base_rag_documents` | Root directory for corpus files. |
| `rag_documents` | File paths relative to that root. |
| `rag_embedding_model`, `rag_embed_max_length` | Document embedder and input token limit. |
| `rag_chunk_size`, `rag_chunk_overlap` | Chunk size and overlap. |
| `rag_pre_embed_path` | Optional precomputed embeddings path, relative to the corpus root. |
| `rag_vector_store_use_bm25` | Build the BM25 corpus alongside the vector index. |
| `local_logging_dir` | Destination for timestamped `RAGDB_*` folders. |
| `pathology`, `rag_name` | Output run labels; `rag_documents` determines the corpus. |

Outputs: `faiss.index`, `rag_db_chunks.csv`, `rag_db_summary.csv`,
`rag_db_overall.json`, and `args_snapshot.yaml`. This is optional preparation;
[inference](run_rag.md) builds its own vector store rather than loading this index.

Copy `configs/config.yaml` to `configs/config_local.yaml`, keep its defaults,
and run `python build_rag_bd.py --config-name config_local`. CLI values override
YAML; `--cfg job --resolve` prints the composed settings.
