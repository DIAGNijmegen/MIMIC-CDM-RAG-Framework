import os
from os.path import join
import random
from datetime import datetime
import time
from tqdm import tqdm

import numpy as np
import torch

import hydra
from omegaconf import DictConfig
from loguru import logger
import langchain

from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from models.models import CustomLLM
from agents.agent import build_agent_executor_ZeroShot

# --- RAG Imports ---
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from rag.rag_appplication import EmbeddingModelContainer, VectorStore, Retriever, RerankerContainer
# --- End RAG Imports ---

# --- NLTK imports ---
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
# --- End NLTK imports ---

def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run(args: DictConfig):
    if type(args.base_seed) is not int or type(args.seed) is not int:
        raise ValueError("base_seed and seed must be integers.")
    if not args.self_consistency:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load patient data
    hadm_info_clean = load_hadm_from_file(
        f"{args.pathology}_hadm_info_first_diag", base_mimic=args.base_mimic
    )

    tags = {
        "system_tag_start": args.system_tag_start,
        "user_tag_start": args.user_tag_start,
        "ai_tag_start": args.ai_tag_start,
        "system_tag_end": args.system_tag_end,
        "user_tag_end": args.user_tag_end,
        "ai_tag_end": args.ai_tag_end,
    }

    # Load desired model
    llm = CustomLLM(
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        tags=tags,
        max_context_length=args.max_context_length,
        exllama=args.exllama,
        seed=args.seed,
        self_consistency=args.self_consistency,
    )
    llm.load_model(args.base_models)

    # --- RAG Initialization ---
    if args.use_rag:
        # Determine the device for RAG components
        device = (
            torch.device("cuda:1") if torch.cuda.device_count() > 1
            else torch.device("cuda:0") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Initialize the embedding model
        embedding_model_container = EmbeddingModelContainer(
            model_name_or_path=args.rag_embedding_model,
            device=device,
            embed_max_length=args.rag_embed_max_length if hasattr(args, 'rag_embed_max_length') else 512,
        )
        embedding_model_container.load_model(args.base_models)

        # Prepare document paths
        document_paths = [
            os.path.join(args.base_rag_documents, doc.lstrip('/'))
            for doc in args.rag_documents
        ]

        # Initialize the vector store
        vector_store = VectorStore(
            document_paths=document_paths,
            embedding_model_container=embedding_model_container,
            chunk_size=args.rag_chunk_size,
            chunk_overlap=args.rag_chunk_overlap,
            pre_embed_path=os.path.join(args.base_rag_documents, args.rag_pre_embed_path.lstrip('/')) if hasattr(args, 'rag_pre_embed_path') else None,
            use_bm25=args.rag_vector_store_use_bm25 if hasattr(args, 'rag_vector_store_use_bm25') else False,
        )

        # --- IF USING MEDCPT, or other: DOCUMENTS EMBEDDER DIFFERENT FROM QUERY EMBEDDER ---
        if hasattr(args, 'rag_query_embedding_model'):
            if args.rag_query_embedding_model != None and args.rag_query_embedding_model != '':
                vector_store.embedding_model.free_memory() #Free Memory of the documents embedder, we don't need it anymore
                embedding_model_container = EmbeddingModelContainer( #Initialize the query embedding model over the previous one
                    model_name_or_path=args.rag_query_embedding_model,
                    device=device,
                    embed_max_length=args.rag_embed_max_length if hasattr(args, 'rag_embed_max_length') else 512,
                )
                embedding_model_container.load_model(args.base_models)
        
        # 3) Cross-encoder reranker
        if (args.rag_re_rank if hasattr(args, 'rag_re_rank') else False):
            rerank_container = RerankerContainer(
                model_name_or_path=args.rag_re_rank_model,
                device=device,
            )
            rerank_container.load_model(args.base_models)

        # Initialize the retriever
        retriever = Retriever(
            vector_store=vector_store,
            embedding_model_container=embedding_model_container,
            top_k_retrieval=args.rag_top_k,
            top_k_rerank=args.rag_top_k_rerank if hasattr(args, 'rag_top_k_rerank') else args.rag_top_k,
            re_rank=args.rag_re_rank,
            rerank_container=rerank_container if (args.rag_re_rank if hasattr(args, 'rag_re_rank') else False) else None,
            prompt_name=args.rag_prompt_name,
            hybrid=args.rag_retrieval_hybrid_bm25 if hasattr(args, 'rag_retrieval_hybrid_bm25') else False,
            hybrid_alpha=args.rag_retrieval_hybrid_bm25_alpha if hasattr(args, 'rag_retrieval_hybrid_bm25_alpha') else 1.0,
            bm25_k=args.rag_retrieval_hybrid_bm25_top_k if hasattr(args, 'rag_retrieval_hybrid_bm25_top_k') else 0,
        )
    # --- End RAG Initialization ---

    date_time = datetime.fromtimestamp(time.time())
    str_date = date_time.strftime("%d-%m-%Y_%H-%M-%S")
    args.model_name = args.model_name.replace("/", "_")
    run_name = f"{args.pathology}_{args.agent}_{args.model_name}"

    if args.use_rag:
        run_name += f"_{args.rag_name}"

    if args.seed != args.base_seed:
        run_name += f"_SEED={args.seed}"

    run_name += f"_{str_date}"

    if args.fewshot:
        run_name += "_FEWSHOT"
    if args.include_ref_range:
        if args.bin_lab_results:
            raise ValueError(
                "Binning and printing reference ranges concurrently is not supported."
            )
        run_name += "_REFRANGE"
    if args.bin_lab_results:
        run_name += "_BIN"
    if args.include_tool_use_examples:
        run_name += "_TOOLEXAMPLES"
    if args.provide_diagnostic_criteria:
        run_name += "_DIAGCRIT"
    if not args.summarize:
        run_name += "_NOSUMMARY"
    # if args.use_rag:
    #     run_name += "_RAG"
    if args.run_descr:
        run_name += str(args.run_descr)
    run_dir = join(args.local_logging_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)
    # Apply path to utils.nlp
    os.environ["LOG_PATH"] = run_dir

    # Setup logfile and logpickle
    results_log_path = join(run_dir, f"{run_name}_results.pkl")
    eval_log_path = join(run_dir, f"{run_name}_eval.pkl")
    log_path = join(run_dir, f"{run_name}.log")
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
    langchain.debug = True

    # Save all the arguments/hydra parameters in a single file in the logs folder
    args_path = join(run_dir, f"{run_name}_args.yaml")
    with open(args_path, "w") as f:
        f.write(str(args))

    # Set LangSmith project name (optional)
    # os.environ["LANGCHAIN_PROJECT"] = run_name

    #DEBUG: run only for 50 patients
    # hadm_info_clean = dict(list(hadm_info_clean.items())[:50])

    # Predict for all patients
    first_patient_seen = False
    # for _id in hadm_info_clean.keys():
    for _id in tqdm(hadm_info_clean.keys(), desc="Processing "+args.pathology+ " patients", total=len(list(hadm_info_clean.keys()))):
        if args.first_patient and not first_patient_seen:
            if _id == args.first_patient:
                first_patient_seen = True
            else:
                continue

        logger.info(f"Processing patient: {_id}")

        # Build the agent executor, passing retrieved documents if RAG is used
        agent_executor = build_agent_executor_ZeroShot(
            patient=hadm_info_clean[_id],
            llm=llm,
            lab_test_mapping_path=args.lab_test_mapping_path,
            logfile=log_path,
            max_context_length=args.max_context_length,
            tags=tags,
            include_ref_range=args.include_ref_range,
            bin_lab_results=args.bin_lab_results,
            include_tool_use_examples=args.include_tool_use_examples,
            provide_diagnostic_criteria=args.provide_diagnostic_criteria,
            summarize=args.summarize,
            model_stop_words=args.stop_words,
            rag_retriever_agent=retriever if args.use_rag else None,  # RAG
            rag_requery=args.rag_requery if hasattr(args, 'rag_requery') else False,  # RAG
            rag_requery_max_length=args.rag_requery_max_length if (hasattr(args, 'rag_requery_max_length')) else 2048,  # RAG
            rag_requery_only_last_report=args.rag_requery_only_last_report if (hasattr(args, 'rag_requery_only_last_report')) else False,  # RAG
            rag_u_ordering=args.rag_u_ordering if (hasattr(args, 'rag_u_ordering')) else False, #RAG Lost-in-the-middle prevention
            rag_skip_first_retrieval=args.rag_skip_first_retrieval if (hasattr(args, 'rag_skip_first_retrieval')) else False, #RAG option to skip the first retrieval step
        )

        # Prepare the input for the agent executor
        input_data = {"input": hadm_info_clean[_id]["Patient History"].strip()}
        if args.use_rag:
            input_data["documents"] = ''

        # Run the agent executor
        result = agent_executor(input_data)

        # Access the step data from the agent
        step_data = agent_executor.agent.step_data

        # Include the step_data in the results
        result['step_data'] = step_data

        # RAG: Access the retrieved documents per step
        retrieved_docs_per_step = agent_executor.agent.retrieved_docs_per_step

        # Save results, including retrieved documents and step data if RAG is used
        if args.use_rag:
            result["retrieval"] = retrieved_docs_per_step

        append_to_pickle_file(results_log_path, {_id: result})

    logger.info("Processing completed.")

if __name__ == "__main__":
    run()
