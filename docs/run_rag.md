# Run model inference

Main loop of the framework. `run_rag.py` runs a diagnostic agent with optional guideline retrieval. It uses `configs/config.yaml` and profiles under `configs/paths/`, `configs/model/`, and `configs/rag/`.

```bash
python run_rag.py model=Gemma2Instruct9B rag=Qwen3-Embedding-0.6B_febstep \
  use_rag=true pathology=appendicitis base_seed=2023 seed=123
```

| Argument | Purpose |
| --- | --- |
| `paths`, `model`, `rag` | YAML profile names for paths, language model, and retrieval. |
| `base_mimic`, `base_models`, `base_rag_documents` | Patient data, local models, and guideline roots. |
| `lab_test_mapping_path`, `local_logging_dir` | Laboratory mapping file and run-output folder. |
| `pathology` | `appendicitis`, `cholecystitis`, `diverticulitis`, or `pancreatitis`. |
| `use_rag` | Enable retrieval; `false` runs the baseline agent. |
| `base_seed`, `seed` | Unsuffixed run seed and current random seed; differing values add `_SEED=<seed>`. |
| `max_context_length`, `summarize` | Context budget and progress summarization. |
| `include_ref_range`, `bin_lab_results` | Laboratory reference ranges and result binning. |
| `include_tool_use_examples`, `provide_diagnostic_criteria` | Additional prompt content. |
| `rag_documents`, `rag_chunk_size`, `rag_chunk_overlap` | Corpus files and chunking. |
| `rag_embedding_model`, `rag_query_embedding_model` | Document and optional separate query embedder. |
| `rag_top_k`, `rag_re_rank`, `rag_re_rank_model`, `rag_top_k_rerank` | Retrieval candidates and reranking. |
| `rag_vector_store_use_bm25`, `rag_retrieval_hybrid_bm25` | BM25 indexing and hybrid retrieval. |
| `rag_requery`, `rag_requery_only_last_report` | Query rewriting; optionally use only the latest report. |
| `rag_u_ordering`, `rag_skip_first_retrieval` | Retrieved-context ordering and skipping the initial retrieval. |
| `first_patient`, `run_descr` | Start at an admission ID and append a run-name suffix. |

The current script limits input to the first 50 admissions before applying
`first_patient`. It writes trajectories, logs, and argument snapshots under
`local_logging_dir`. Keep `base_seed` consistent with [evaluation](evaluation.md)
and [inspection](inspection.md).

Copy `configs/config.yaml` to `configs/config_local.yaml`, keep its defaults,
and run `python run_rag.py --config-name config_local`. CLI values override YAML;
`--cfg job --resolve` prints settings without running inference.
