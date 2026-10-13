from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import numpy as np
import os
import json
from os.path import join

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
import tiktoken
        
from sklearn.metrics.pairwise import cosine_similarity

from agents.prompts import RERANKER_PROMPT

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk, ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )

# --- Embedding Model Container ---

class EmbeddingModelContainer:
    def __init__(
        self, 
        model_name_or_path='dunzhang/stella_en_400M_v5', 
        device='cpu', 
        trust_remote_code=True, 
        embed_max_length=512
    ):
        """
        Initialize the embedding model container with an embedding model.
        :param model_name_or_path: The name or path of the embedding model.
        :param device: The device to run the model on ('cpu' or 'cuda').
        :param trust_remote_code: Whether to trust remote code for custom models.
        """
        self.model_name = model_name_or_path
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.embed_max_length = embed_max_length

        # Only type flags we actually need for loading behavior
        name_l = self.model_name.lower()
        self.is_nv_embed = "nv-embed" in name_l
        self.is_stella = "stella" in name_l
        self.is_medcpt = "medcpt" in name_l
        self.is_qwen3 = ("qwen3" in name_l) and ("embedding" in name_l)

    def load_model(self, base_models: str) -> None:
        model_path = join(base_models, self.model_name)

        if self.is_stella:
            # Stella uses SentenceTransformer with prompt support
            self.embedding_model = SentenceTransformer(
                model_path,
                trust_remote_code=self.trust_remote_code,
                device=self.device
            )
            self.embedding_model.max_seq_length = self.embed_max_length
            self.tokenizer = self.embedding_model.tokenizer

        elif self.is_qwen3:
            # Qwen3-Embedding (SentenceTransformer backend)
            # Example HF id: "Qwen/Qwen3-Embedding-0.6B"
            self.embedding_model = SentenceTransformer(
                model_path,
                trust_remote_code=self.trust_remote_code,
                device=self.device,
            )
            self.embedding_model.max_seq_length = self.embed_max_length
            self.tokenizer = self.embedding_model.tokenizer

        elif self.is_nv_embed:
            # NV-Embed style model with custom .encode(instruction=...) via remote code
            self.embedding_model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float16
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=self.trust_remote_code,
            )

        elif self.is_medcpt:
            # MedCPT encoder
            self.embedding_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
            )

        else:
            # Generic SentenceTransformer model (includes MiniLM, etc.)
            self.embedding_model = SentenceTransformer(
                model_path,
                device=self.device
            )
            self.embedding_model.max_seq_length = self.embed_max_length
            self.tokenizer = self.embedding_model.tokenizer

    def add_eos(self, texts):
        """
        Add EOS token to the end of each text.
        """
        return [text + self.tokenizer.eos_token for text in texts]

    def embed(self, texts):
        """
        Embed a list of texts.
        """
        if self.is_nv_embed:
            # NV-Embed: encode with instruction prefix, then normalize
            passage_prefix = ""
            passages = texts
            max_length = self.embed_max_length
            batch_size = 4
            passage_embeddings = []
            
            for i in range(0, len(passages), batch_size):
                batch = passages[i:i + batch_size]
                batch_emb = self.embedding_model.encode(
                    batch, 
                    instruction=passage_prefix, 
                    max_length=max_length
                )
                batch_emb_np = batch_emb.clone().detach().cpu().numpy()
                del batch_emb
                passage_embeddings.append(torch.from_numpy(batch_emb_np))
            
            passage_embeddings = torch.cat(passage_embeddings, dim=0)
            passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
            embeddings = passage_embeddings

        elif self.is_medcpt:
            # MedCPT – [CLS] pooling
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts, 
                    truncation=True, 
                    padding=True, 
                    return_tensors='pt', 
                    max_length=self.embed_max_length,
                ).to(self.device)
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        elif self.is_stella:
            # Stella SentenceTransformer style, we can normalize here
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_tensor=False
            )

        elif self.is_qwen3:
            # Qwen3-Embedding – SentenceTransformer encode with normalization
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_tensor=False,
            )

        else:
            # Generic SentenceTransformer (incl. MiniLM)
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_tensor=False
            )

        return embeddings

    def embed_query(self, query, prompt_name='s2p_query'):
        """
        Embed a single query string.
        """
        if self.is_nv_embed:
            task_name_to_instruct = {
                "example": "Given the patient's information, retrieve relevant passages to treat them",
                "requery": "Retrieve medical guidelines passages to answer this question.",
            }
            query_prefix = "Instruct: " + task_name_to_instruct["requery"] + "\nQuestion: "
            queries = [query]
            max_length = self.embed_max_length

            query_embeddings = self.embedding_model.encode(
                queries, 
                instruction=query_prefix, 
                max_length=max_length
            )
            query_embeddings = query_embeddings.clone().detach().cpu()
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            embedding = query_embeddings

        elif self.is_stella:
            # Use prompt_name from caller (e.g. "s2p_query")
            embedding = self.embedding_model.encode([query], prompt_name=prompt_name)

        elif self.is_qwen3:
            # Qwen3-Embedding: use the built-in "query" prompt
            # (recommended usage from Qwen docs)
            embedding = self.embedding_model.encode(
                [query],
                prompt_name="query",
                normalize_embeddings=True,
            )

        elif self.is_medcpt:
            query_text = query
            with torch.no_grad():
                inputs = self.tokenizer(
                    [query_text], 
                    truncation=True, 
                    padding=True, 
                    return_tensors='pt', 
                    max_length=self.embed_max_length,
                ).to(self.device)
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        else:
            # Generic SentenceTransformer query
            embedding = self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            )

        return embedding[0]
    
    def free_memory(self):
        """
        Free up memory by deleting the embedding model.
        """
        del self.tokenizer
        del self.embedding_model

        
# --- Vector Store ---
            
class VectorStore:
    def __init__(
        self,
        document_paths,
        embedding_model_container,
        chunk_size=250,
        chunk_overlap=0,
        pre_embed_path=None,
        use_bm25: bool = False,
        bm25_tokenizer=None
    ):
        """
        Initialize the vector store with documents and embeddings.
        :param document_paths: List of document file paths.
        :param embedding_model_container: An instance of EmbeddingModelContainer.
        :param chunk_size: Size of text chunks for splitting documents.
        :param chunk_overlap: Overlap size between chunks.
        :param pre_embed_path: Optional path to precomputed embeddings (.npy).
        :param use_bm25: Whether to build a BM25 index for hybrid retrieval.
        :param bm25_tokenizer: Optional callable(text) -> List[str] for BM25 tokenization.
        """
        self.embedding_model = embedding_model_container
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # BM25-related fields
        self.use_bm25 = use_bm25
        self.bm25_tokenizer = bm25_tokenizer or self._default_bm25_tokenizer
        self.bm25 = None
        self.bm25_corpus_tokens = None

        # Docling fields
        # --- Docling converter + HybridChunker for markdown files ---
        # Reuse the *same* tokenizer as the embedding model so that
        # token-aware chunking matches your embedding context window.
        self.docling_converter = DocumentConverter()

        hf_tok = getattr(self.embedding_model, "tokenizer", None)
        self.docling_tokenizer = None
        self.docling_chunker = None

        self.docling_chunker = HybridChunker(
            tokenizer=self.embedding_model.tokenizer,
            serializer_provider=MDTableSerializerProvider(),
            merge_peers=True,
            # max_tokens=getattr(self.embedding_model, "embed_max_length", 512),
            max_tokens=chunk_size                
        )

        # --- End Docling setup ---

        # 1) Load & chunk docs
        self.docs = self.load_documents(document_paths)
        self.doc_chunks = self.split_documents(self.docs)

        # 2) Dense vector store (FAISS)
        self.index, self.doc_chunks = self.create_vector_store(
            self.doc_chunks, pre_embed_path=pre_embed_path
        )

        # 3) Optional BM25 index
        if self.use_bm25:
            self._build_bm25_index(self.doc_chunks)

    # ---------- BM25 helpers ----------

    def _default_bm25_tokenizer(self, text: str):
        # Tiny, fast tokenizer: lowercase + whitespace split.
        # Swap for nltk / spaCy if you want smarter tokenization.
        return text.lower().split()

    def _build_bm25_index(self, doc_chunks):
        """
        Build a BM25Okapi index over page_content of each chunk.
        """
        corpus_tokens = [
            self.bm25_tokenizer(doc.page_content)
            for doc in doc_chunks
        ]
        self.bm25_corpus_tokens = corpus_tokens
        self.bm25 = BM25Okapi(corpus_tokens)

    def search_bm25(self, query: str, top_k: int):
        """
        Run BM25 search over chunks.
        Returns: list of (doc_index, score) sorted descending.
        """
        if not self.use_bm25 or self.bm25 is None:
            raise ValueError("BM25 index not built. Set use_bm25=True when creating VectorStore.")

        tokenized_query = self.bm25_tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)  # array of shape (num_docs,) :contentReference[oaicite:2]{index=2}  

        # Get top_k indices by score
        scores = np.array(scores)
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = [(int(i), float(scores[i])) for i in top_indices]
        return results

    # ---------- existing methods unchanged below ----------

    def load_documents(self, document_paths):
        docs = []
        for path in document_paths:
            if "chunkr" in path.lower():
                docs.extend(self.load_chunkr_file(path))
            elif "medcpt" in path.lower():
                docs.extend(self.load_json_file(path))
            elif path.lower().endswith('.pdf'):
                loader = PyPDFLoader(path)
                print(f"Loading PDF file: {path}")
                docs.extend(loader.load())
            elif path.lower().endswith('.md'):
                docs.extend(self.load_markdown_file(path))
            else:
                raise ValueError(f"Unsupported file format for: {path}")
        return docs

    def load_markdown_file(self, path):
        """
        Load a Markdown file via Docling, then chunk it with HybridChunker
        and return contextualized chunks as LangChain Documents.
        """
        doc_name = os.path.basename(path)

        # If we didn't manage to set up a docling_chunker, fall back
        # to the old "single big markdown document" behaviour.
        if self.docling_chunker is None or self.docling_converter is None:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = Document(
                page_content=content,
                metadata={
                    "source": doc_name,
                    "format": "markdown",
                },
            )
            return [doc]

        # --- 1) Parse Markdown into a DoclingDocument ---
        conv_res = self.docling_converter.convert(str(path))
        dl_doc = conv_res.document

        # --- 2) Run HybridChunker on the DoclingDocument ---
        chunks = list(self.docling_chunker.chunk(dl_doc=dl_doc))

        lc_docs = []
        hf_tokenizer = getattr(self.embedding_model, "tokenizer", None)

        def _get_headings(meta):
            # works across docling versions (meta can be pydantic model or dict)
            if meta is None:
                return []
            if isinstance(meta, dict):
                return meta.get("headings") or []
            return getattr(meta, "headings", None) or []

        for i, ch in enumerate(chunks):
            print(i, getattr(ch.meta, "headings", None))
            doc_chunk = DocChunk.model_validate(ch)   # robust: normalizes meta/doc_items fields
            heading_path = _get_headings(doc_chunk.meta)

            # --- 3) Contextualize the chunk (adds heading context, etc.) ---            
            chunk_text = ch.text
            embed_text = self.docling_chunker.contextualize(chunk=ch)
            
            # Optional but recommended: compute token length with your embed tokenizer
            token_size = None
            if hf_tokenizer is not None:
                tokens = hf_tokenizer.encode(chunk_text, add_special_tokens=False)
                token_size = len(tokens)

            MAX_TITLE_LENGTH = 100
            # Join titles but trim each subheading to avoid overly long titles that bloat token count without adding much signal.
            heading_path_titles = [
                h[:MAX_TITLE_LENGTH] + ("..." if len(h) > MAX_TITLE_LENGTH else "")
                for h in heading_path
            ]

            # Keep the longest title for the last title
            heading_path_titles.pop(-1)
            heading_path_titles.append(heading_path[-1])  # keep full last title for specificity
        
            title = heading_path_titles[0] if heading_path_titles else "unknown"
            # Remove first level title (always too long and redundant)
            heading_path_titles = heading_path_titles[1:]

            heading_path_str = ", ".join(heading_path_titles)

            # Basic metadata – you can add more by inspecting ch.meta.doc_items
            meta = {
                "source": doc_name,
                "format": "markdown_docling",       # <- important for split_documents
                "chunk_id": f"{doc_name}_chunk_{i}",
                "document_reference": doc_name,
                "page_number": "unknown",
                "order_in_document": i,
                "embed_text": embed_text,
                "heading_path": heading_path,                 # list[str]
                "heading_path_str": heading_path_str, # convenient for filtering/debug
                "title": title,
            }
            if token_size is not None:
                meta["token_size"] = token_size

            lc_doc = Document(
                page_content=chunk_text, # <- Change 17.02.2026: keep original chunk text for page_content cause the retrieval was lowkey better this way
                # page_content=embed_text,   # <- Change 07.01.2026: embed the contextualized text
                metadata=meta,
            )
            lc_docs.append(lc_doc)

        return lc_docs
    
    def load_json_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        docs = []
        for key, chunk in data.items():
            content = chunk.get("a", "")
            metadata = {
                "chunk_id": key,
                "document_id": chunk.get("d", "unknown"),
                "source": chunk.get("t", "unknown"),
                "format": "medcpt",
                "tags": chunk.get("m", ""),
            }
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
        return docs

    def load_chunkr_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            chunk_data = json.load(file)

        doc_name = os.path.basename(path)
        docs = []
        for chunk in chunk_data:
            for segment in chunk["segments"]:
                content = segment["content"]
                metadata = {
                    "segment_id": segment["segment_id"],
                    "bbox": segment["bbox"],
                    "page": segment["page_number"],
                    "page_width": segment["page_width"],
                    "page_height": segment["page_height"],
                    "segment_type": segment["segment_type"],
                    "image": segment.get("image"),
                    "html": segment.get("html"),
                    "markdown": segment.get("markdown"),
                    "chunk_length": chunk["chunk_length"],
                    "source": doc_name,
                }
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
        return docs

    def split_documents(self, documents):
        """
        Split documents into chunks and add metadata.
        :param documents: List of Document objects.
        :return: List of Document chunks.
        """
        tokenizer = self.embedding_model.tokenizer
        max_length = self.chunk_size
        doc_chunks = []

        for doc in documents:
            # ---------- MEDCPT or Docling-based markdown (already chunked) ----------
            if doc.metadata.get('format') in ('medcpt', 'markdown_docling'):
                doc_chunks.append(doc)

            # ---------- ALL OTHER FORMATS (PDF, plain text, etc.) ----------
            else:
                text = doc.page_content
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_chunks = [
                    tokens[i:i + max_length]
                    for i in range(0, len(tokens), max_length - self.chunk_overlap)
                ]
                
                for i, token_chunk in enumerate(token_chunks):
                    chunk_text = tokenizer.decode(token_chunk)
                    chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                    
                    chunk_id = f"chunk_{len(doc_chunks)}"
                    chunk.metadata['chunk_id'] = chunk.metadata.get('segment_id', chunk_id)
                    chunk.metadata['token_size'] = len(token_chunk)
                    chunk.metadata['document_reference'] = chunk.metadata.get('source', 'unknown')
                    chunk.metadata['page_number'] = chunk.metadata.get('page', 'unknown')
                    chunk.metadata['order_in_document'] = i
                    
                    doc_chunks.append(chunk)

        return doc_chunks


    def create_vector_store(self, doc_chunks, pre_embed_path=None):
        if pre_embed_path is None:
            # Prefer embed_text if present, otherwise fallback to page_content
            if self.embedding_model.is_medcpt:
                texts = [
                    [
                        doc.metadata.get("heading_path_str", doc.metadata.get("embed_text", doc.page_content)),
                        doc.metadata.get("embed_text", doc.page_content),
                    ]
                    for doc in doc_chunks
                ]
            else:
                texts = [
                    doc.metadata.get("embed_text", doc.page_content)
                    for doc in doc_chunks
                ]

            embeddings = self.embedding_model.embed(texts)
            embeddings = np.array(embeddings).astype('float32')

            for doc, emb in zip(doc_chunks, embeddings):
                doc.metadata['embedding'] = emb

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            return index, doc_chunks

        else:
            embeds = np.load(pre_embed_path)
            print(f"Pre-computed embeddings shape: {embeds.shape}")
            print(f"Document chunks size: {len(doc_chunks)}")

            dimension = embeds.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeds.astype('float32'))
            return index, doc_chunks

    def get_vector_store(self):
        return self.index, self.doc_chunks



# --- Reranker Container (unchanged) ---

class RerankerContainer:
    """
    Holds a CrossEncoder reranker and provides a model-agnostic rerank() API.
    """
    def __init__(
            self,
            model_name_or_path: str,
            device: str = "cpu",
            instruction: str | None = None,  #optional task/instruction for some rerankers
        ):
        self.model_name = model_name_or_path
        self.device = device

        name_l = self.model_name.lower()
        self.is_qwen3_reranker = ("qwen3" in name_l) and ("reranker" in name_l)

        # Default instruction from Qwen3 docs (can be overridden)
        self.instruction = (
            instruction
            or RERANKER_PROMPT
        )

    def load_model(self, base_models: str) -> None:
        model_path = join(base_models, self.model_name)
        max_length = 0
        if ("medcpt" in self.model_name.lower()):
            max_length = 512

        if (max_length > 0):
            self.cross_encoder = CrossEncoder(
                model_path,
                device=self.device,
                max_length=max_length  # token limit for some models
            )
        else:
            self.cross_encoder = CrossEncoder(
                model_path,
                device=self.device
            )

        # -------------------- FIX: ensure padding token is defined --------------------
        tok = self.cross_encoder.tokenizer
        model = self.cross_encoder.model

        # CrossEncoder.predict tokenizes with padding=True; batched seq-cls models
        # error out if config.pad_token_id is missing. :contentReference[oaicite:1]{index=1}
        if tok.pad_token is None:
            # Common and safe inference fallback for decoder-ish tokenizers:
            tok.pad_token = tok.eos_token  # e.g., GPT/Qwen families :contentReference[oaicite:2]{index=2}

        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.pad_token_id  # :contentReference[oaicite:3]{index=3}

        # Optional (usually harmless): keep padding consistent
        tok.padding_side = "right"
        # ---------------------------------------------------------------------------      

    # --- Qwen3 reranker formatting helpers ---
    def _format_queries_qwen3(self, query: str, instruction: str | None = None) -> str:
        """
        Format the query for Qwen3-Reranker as a chat-style prompt.
        Mirrors the official example.
        """
        # https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        )
        if instruction is None:
            instruction = self.instruction

        return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"

    def _format_document_qwen3(self, document: str) -> str:
        """
        Format the document for Qwen3-Reranker as a chat-style completion.
        """
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return f"<Document>: {document}{suffix}"
    # --- Qwen3 END ---

    def rerank(self, query: str, documents, instruction: str | None = None):
        instr = instruction or self.instruction

        # Helper: pick the text we want the reranker to see
        def _get_rerank_text(doc):
            return doc.metadata.get("embed_text", doc.page_content)

        if self.is_qwen3_reranker:
            # Qwen3: apply the special formatting to both sides of the pair
            pairs = [
                [
                    self._format_queries_qwen3(query, instr),
                    self._format_document_qwen3(_get_rerank_text(doc)),
                ]
                for doc in documents
            ]
        else:
            # Generic CrossEncoder: vanilla [query, doc] pairs
            pairs = [
                [query, _get_rerank_text(doc)]
                for doc in documents
            ]

        scores = self.cross_encoder.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        for doc, score in ranked:
            try:
                doc.metadata['rerank_score'] = float(score)
            except Exception:
                pass

        return [doc for doc, _ in ranked]


# --- Retriever with optional hybrid BM25 + dense --- 

class Retriever:
    def __init__(
        self, 
        vector_store, 
        embedding_model_container, 
        top_k_retrieval=20,      # # of candidates from fusion/dense
        top_k_rerank=5,          # # kept after cross-encoder reranking
        re_rank=False, 
        rerank_container: "RerankerContainer | None" = None,
        prompt_name='s2p_query',
        hybrid: bool = False,
        hybrid_alpha: float = 0.5,
        bm25_k: int | None = None
    ):
        """
        :param hybrid: If True, combine dense and BM25 scores.
        :param hybrid_alpha: Weight for dense scores in fusion:
                             combined = alpha * dense + (1-alpha) * bm25.
        :param bm25_k: Number of top docs to fetch from BM25 before fusion.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model_container
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.re_rank = re_rank
        self.rerank_container = rerank_container
        self.prompt_name = prompt_name

        self.hybrid = hybrid
        self.hybrid_alpha = hybrid_alpha
        self.bm25_k = bm25_k or top_k_retrieval

        if self.re_rank and self.rerank_container is None:
            raise ValueError("re_rank=True but no RerankedContainer was provided.")

        if self.hybrid and not getattr(self.vector_store, "use_bm25", False):
            raise ValueError("hybrid=True but VectorStore was created with use_bm25=False.")

    def retrieve(self, query):
        # ----- 1) Dense query embedding -----
        query_embedding = self.embedding_model.embed_query(query, prompt_name=self.prompt_name)
        query_embedding = np.array([query_embedding]).astype('float32')

        # Dense retrieval from FAISS
        D, I = self.vector_store.index.search(query_embedding, self.top_k_retrieval)
        dense_indices = I[0]
        dense_distances = D[0]   # L2 distances (smaller = better)
        dense_scores = -dense_distances  # convert to similarity

        dense_score_dict = {int(idx): float(score) for idx, score in zip(dense_indices, dense_scores)}

        bm25_score_dict = {}
        fusion_scores = {}

        if self.hybrid:
            # ----- 2) BM25 retrieval -----
            bm25_results = self.vector_store.search_bm25(query, top_k=self.bm25_k)
            bm25_score_dict = {doc_idx: score for doc_idx, score in bm25_results}

            # ----- 3) Score fusion -----
            all_indices = set(dense_score_dict.keys()) | set(bm25_score_dict.keys())

            dense_vals = np.array(list(dense_score_dict.values()))
            if len(dense_vals) > 0 and dense_vals.max() > dense_vals.min():
                dense_min, dense_max = dense_vals.min(), dense_vals.max()
                def norm_dense(x): return (x - dense_min) / (dense_max - dense_min)
            else:
                def norm_dense(x): return 1.0

            bm25_vals = np.array(list(bm25_score_dict.values()))
            if len(bm25_vals) > 0 and bm25_vals.max() > bm25_vals.min():
                bm25_min, bm25_max = bm25_vals.min(), bm25_vals.max()
                def norm_bm25(x): return (x - bm25_min) / (bm25_max - bm25_min)
            else:
                def norm_bm25(x): return 1.0

            alpha = self.hybrid_alpha

            for idx in all_indices:
                d = dense_score_dict.get(idx, None)
                s = bm25_score_dict.get(idx, None)

                d_norm = norm_dense(d) if d is not None else 0.0
                s_norm = norm_bm25(s) if s is not None else 0.0

                combined = alpha * d_norm + (1.0 - alpha) * s_norm
                fusion_scores[idx] = combined

            fused_sorted = sorted(
                fusion_scores.items(),
                key=lambda kv: kv[1],
                reverse=True
            )

            final_indices = [idx for idx, _ in fused_sorted[:self.top_k_retrieval]]

        else:
            # Dense-only path
            final_indices = list(dense_indices)

        # ----- 3.5) Attach retrieval provenance metadata (and dedupe in one pass) -----
        doc_chunks   = self.vector_store.doc_chunks
        dense_scores = dense_score_dict
        bm25_scores  = bm25_score_dict
        hybrid       = self.hybrid

        sentinel = object()
        seen = set()
        seen_add = seen.add

        retrieved_docs = []
        append = retrieved_docs.append

        for idx in final_indices:
            doc = doc_chunks[idx]
            md = doc.metadata

            # IMPORTANT: if chunk_id might be missing/None, don't collapse everything to one doc
            chunk_id = md.get("chunk_id", idx)

            if chunk_id in seen:
                continue
            seen_add(chunk_id)

            # Single hash lookup per dict (avoids `if idx in d` then `d[idx]`)
            dense_score = dense_scores.get(idx, sentinel)
            bm25_score  = bm25_scores.get(idx, sentinel)

            dense_hit = dense_score is not sentinel
            bm25_hit  = bm25_score  is not sentinel

            md["dense_hit"] = dense_hit
            md["bm25_hit"]  = bm25_hit

            if dense_hit:
                md["dense_score"] = float(dense_score)
            if bm25_hit:
                md["bm25_score"] = float(bm25_score)

            if hybrid:
                md["retrieval_source"] = (
                    "hybrid"     if (dense_hit and bm25_hit) else
                    "dense_only" if dense_hit else
                    "bm25_only"  if bm25_hit else
                    "unknown"
                )
            else:
                md["retrieval_source"] = "dense_only"

            append(doc)

        # ----- 4) Optional cross-encoder reranking -----
        if self.re_rank and self.rerank_container is not None and len(retrieved_docs) > 0:
            reranked_docs = self.rerank_container.rerank(query, retrieved_docs)
            retrieved_docs = reranked_docs[:self.top_k_rerank]
        else:
            if self.top_k_rerank is not None:
                retrieved_docs = retrieved_docs[:self.top_k_rerank]

        return self._format_results(retrieved_docs)

    def _format_results(self, documents):
        results = []
        for doc in documents:
            format = doc.metadata.get('format', 'unknown')
            rerank_score = doc.metadata.get('rerank_score', 0.0)

            base = {
                'chunk_id': doc.metadata.get('chunk_id'),
                'score': rerank_score,
                'format': format,
                # provenance:
                'retrieval_source': doc.metadata.get('retrieval_source', None),
                'dense_hit': doc.metadata.get('dense_hit', False),
                'bm25_hit': doc.metadata.get('bm25_hit', False),
                'dense_score': doc.metadata.get('dense_score', None),
                'bm25_score': doc.metadata.get('bm25_score', None),
            }

            if format == 'medcpt':
                base.update({
                    'document_reference': doc.metadata.get('source'),
                    'document_id': doc.metadata.get('document_id'),
                    'tags': doc.metadata.get('tags'),
                    'content': doc.page_content,
                })
            else:
                base.update({
                    'document_reference': doc.metadata.get('document_reference'),
                    'page_number': doc.metadata.get('page_number'),
                    'token_size': doc.metadata.get('token_size'),
                    'order_in_document': doc.metadata.get('order_in_document'),
                    'content': doc.page_content,
                })
            #if docling add titles
            if format == 'markdown_docling':
                base.update({
                    'title': doc.metadata.get('title'),
                    'heading_path': doc.metadata.get('heading_path'),
                    'heading_path_str': doc.metadata.get('heading_path_str'),
                })

            results.append(base)

        return results
