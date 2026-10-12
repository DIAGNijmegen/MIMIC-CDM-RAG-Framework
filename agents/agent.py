import pickle
from typing import List, Tuple, Union, Dict, Any, Optional
from hashlib import sha256
import pandas as pd

from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.schema.messages import BaseMessage
# from langchain.schema import AgentAction
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks import FileCallbackHandler

from pydantic import Field  # Added for proper initialization

from agents.prompts import (
    CHAT_TEMPLATE,
    SUMMARIZE_OBSERVATION_TEMPLATE,
    DIAG_CRIT_TOOL_DESCR,
    TOOL_USE_EXAMPLES,
    TOOL_USE_EXAMPLES_RAG, #RAG
    DIAG_CRIT_TOOL_USE_EXAMPLE,
    CHAT_TEMPLATE_RAG, #RAG
    REQUERY_PROMPT, #RAG
    REQUERY_PROMPT_SINGLE_REPORT, #RAG
)
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
from tools.Tools import (
    RunLaboratoryTests,
    RunImaging,
    DoPhysicalExamination,
    ReadDiagnosticCriteria,
)
from tools.utils import action_input_pretty_printer
from utils.nlp import calculate_num_tokens, truncate_text

STOP_WORDS = ["Observation:", "Observations:", "observation:", "observations:"]
REQUERY_EOS = "<END_QUESTION>"

class TextSummaryCache:
    def __init__(self):
        self.cache = {}

    def hash_text(self, text):
        return sha256(text.encode()).hexdigest()

    def add_summary(self, text, summary):
        text_hash = self.hash_text(text)
        if text_hash in self.cache:
            return
        self.cache[text_hash] = summary

    def get_summary(self, text):
        text_hash = self.hash_text(text)
        return self.cache.get(text_hash, None)


class CustomZeroShotAgent(ZeroShotAgent):
    lab_test_mapping_df: pd.DataFrame = None
    observation_summary_cache: TextSummaryCache = TextSummaryCache()
    stop: List[str]
    max_context_length: int
    tags: Dict[str, str]
    summarize: bool
    # --- RAG ---
    rag_retriever_agent: Any = None #RAG
    rag_requery: bool = False #RAG
    rag_requery_max_length: int = 2048 #RAG
    rag_requery_only_last_report: bool = False #RAG
    rag_u_ordering: bool = False #RAG: Mitigate lost-in-the-middle issue by using a U-shaped ordering of retrieved documents (best relevance at start and end of the list, worst in the middle)
    rag_skip_first_retrieval: bool = False #RAG option to skip the first retrieval step
    retrieved_docs_per_step: Dict[int, List[Dict]] = {}
    requery_chain: Optional[LLMChain] = None
    
    # step_counter: int = 0
    # --- End RAG ---
    # Added to store step-by-step data
    step_data: List[Dict[str, Any]] = Field(default_factory=list)


    class Config:
        arbitrary_types_allowed = True

    # Initialize step_data in __init__
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_data = []

        if self.rag_requery and self.rag_retriever_agent is not None:
            # Create a separate prompt for requery/refinement
            prompt_requery = create_prompt_requery(self.tags, single_report=self.rag_requery_only_last_report)

            # IMPORTANT CHANGE:
            # Do NOT wrap or .bind() the LLM here, because LLMChain expects a BaseLanguageModel,
            # and .bind() returns a Runnable in your LangChain version.
            # Instead, reuse the same llm object and control "shortness" via post-processing.
            self.requery_chain = LLMChain(
                llm=self.llm_chain.llm,
                prompt=prompt_requery,
                verbose=True,  # keep if you want logs
            )
            
    # Allow for multiple stop criteria instead of just taking the observation prefix string
    @property
    def _stop(self) -> List[str]:
        return self.stop

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any
    ) -> Union[AgentFinish, AgentAction]:
        """
        Overrides the default plan method to include Retrieval-Augmented Generation (RAG) retrieval when available.

        Args:
            intermediate_steps: List of tuples containing previous agent actions and their outputs.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentAction or AgentFinish: The next action for the agent to perform.
        """
        # Calculate the current step number
        current_step = len(intermediate_steps)

        question = ""
        # Extract current information to form the question
        if self.rag_retriever_agent is not None:
            if not self.rag_requery_only_last_report:
                question = self.extract_current_information(intermediate_steps, kwargs)
            else:
                # Only use the last report/input for the question
                question = intermediate_steps[-1][1] if intermediate_steps else kwargs.get("input", "")

            #Print the question
            print(f"RAG Query (length): {len(question)}")
            print(f"RAG Query: {question}")

        # Initialize retrieved documents content
        retrieved_docs_content = ""

        # Perform RAG retrieval if a retriever agent is available
        # Start retrieving only after the step counter reaches a certain point to allow the agent to gather some initial information first. This can be tuned based on the specific use case.
        if self.rag_retriever_agent is not None and (not self.rag_skip_first_retrieval or current_step > 0):
            if self.rag_requery:
                # Call the separate requery chain; its LLM already has stop=[REQUERY_EOS, ai_tag_end]
                refined = self.requery_chain.predict(
                    original_text=question,
                    stop=[REQUERY_EOS, self.tags["ai_tag_end"]] + self._stop
                )

                # C. Make post-processing ultra-simple and robust

                # 1) Strip any hidden reasoning if the model emits </think> blocks
                refined = refined.split("</think>")[-1]
                refined = refined.strip()

                # 2) If <END_QUESTION> ever appears, cut at it
                refined = refined.split(REQUERY_EOS)[0].strip()

                # 3) Choose the actual query line more intelligently

                # Split into non-empty lines
                raw_lines = [ln.strip() for ln in refined.splitlines() if ln.strip()]

                def is_garbage_line(s: str) -> bool:
                    # Lines that are only punctuation/quotes/braces/colons are useless
                    stripped = s.strip()
                    if not stripped:
                        return True
                    # All chars are from this small punctuation set → treat as garbage
                    punctuation_only = set(':",{}')
                    return all(ch in punctuation_only for ch in stripped)

                # Keep only lines that are not "garbage"
                candidate_lines = [ln for ln in raw_lines if not is_garbage_line(ln)]

                line = ""
                #Prefer the first or longest line, but only if it contains at least 10 words (to avoid picking short garbage lines)
                if candidate_lines:
                    valid_lines = [ln for ln in candidate_lines if len(ln.split()) >= 10]
                    if valid_lines:
                        # line = max(valid_lines, key=len)
                        line = valid_lines[0]
                    else:
                        # line = max(candidate_lines, key=len)
                        line = candidate_lines[0]
                else:
                    line = refined

                # 4) Drop the "Rewritten search question:" prefix if the model echoes it
                prefix = "Rewritten search question:"
                if line.lower().startswith(prefix.lower()):
                    line = line[len(prefix):].strip()

                # 5) Strip leading/trailing quotes
                line = line.strip(" \"'")

                # 6) Hard length limit to prevent any residual looping from harming retrieval
                max_len_chars = getattr(self, "rag_requery_max_length", 2048)
                line = line[:max_len_chars].strip()

                # # 7) Ensure it looks like a question
                # if line and "?" not in line:
                #     line = line.rstrip(".") + "?"

                line = line.replace("▁", " ")
                line = line.replace("  ", " ")

                # If line starts with "query:" or "question:", remove that prefix to avoid confusion for the retriever
                line_lower = line.lower()
                if "query" in line_lower:
                    line = line.split("query", 1)[-1].strip()
                elif "question" in line_lower:
                    line = line.split("question", 1)[-1].strip()
                if line.startswith(":"):
                    line = line[1:].strip()

                line = line.strip()

                #Uppercase the first letter to make it look nicer for the retriever, but only if it's not already uppercase
                if line and not line[0].isupper():
                    line = line[0].upper() + line[1:]

                question = line.strip()
                print(f"Refined question for RAG retrieval: {question}")

            retrieved_docs = self.rag_retriever_agent.retrieve(question)

            if self.rag_u_ordering:
                # --- U-shaped ordering: high relevance at the start and end, low in the middle ---
                # Assumes retrieved_docs is already sorted by relevance (best -> worst).
                front, back = [], []
                ranks = []
                for rank, doc in enumerate(retrieved_docs):
                    (front if rank % 2 == 0 else back).append((rank,doc))
                u_docs = front + back[::-1]  # e.g., [d1, d3, d5, d6, d4, d2]

                # base_gid = 1 + current_step * len(retrieved_docs)
                # base_gid = 1 + (current_step-1) * len(retrieved_docs)

                if self.rag_skip_first_retrieval:
                    base_gid = 1 + (current_step-1) * len(retrieved_docs)
                else:                    
                    base_gid = 1 + current_step * len(retrieved_docs)

                retrieved_docs_content = "\n".join(
                    f"[G{base_gid + rank}] Source: {doc.get('title') or 'Unknown source'}\n"
                    f"Section: {doc.get('heading_path_str', 'N/A')}\n"
                    f"Snippet: {(doc.get('content') or '').strip()}\n"
                    for rank, doc in u_docs
                )

                # Store retrieved documents using current_step as the key
                self.retrieved_docs_per_step[current_step] = retrieved_docs
                kwargs["documents"] = retrieved_docs_content  # Add retrieved documents to kwargs

                retrieved_docs = [] # Clear retrieved docs to avoid duplication in prompt since we're embedding them into the scratchpad now
            else:
                # --- Original ordering (best relevance first), linear ordering best first worst last --*

                if self.rag_skip_first_retrieval:
                    base_gid = 1 + (current_step-1) * len(retrieved_docs)
                else:
                    base_gid = 1 + current_step * len(retrieved_docs)

                retrieved_docs_content = "\n".join(
                    f"[G{base_gid + i}] Source: {doc.get('title') or 'Unknown source'}\n"
                    f"Section: {doc.get('heading_path_str', 'N/A')}\n"
                    f"Snippet: {(doc.get('content') or '').strip()}\n"
                    for i, doc in enumerate(retrieved_docs)
                )
                # Store retrieved documents using current_step as the key
                self.retrieved_docs_per_step[current_step] = retrieved_docs
                kwargs["documents"] = retrieved_docs_content  # Add retrieved documents to kwargs

                retrieved_docs = [] # Clear retrieved docs to avoid duplication in prompt since we're embedding them into the scratchpad now
        else:
            kwargs["documents"] = ""  # Ensure 'documents' is in kwargs even if no retrieval was done

        # Construct the scratchpad and update kwargs
        thoughts, updated_kwargs = self._construct_scratchpad(intermediate_steps, **kwargs)
        updated_kwargs.update({
            "agent_scratchpad": thoughts,
            "stop": self._stop,
            "documents": kwargs.get("documents", "")  # Ensure 'documents' is in updated_kwargs even if no retrieval was done
        })

        # Prepare inputs for the prompt
        prompt_variables = self.llm_chain.prompt.input_variables
        inputs_for_prompt = {k: v for k, v in updated_kwargs.items() if k in prompt_variables}

        # Format the prompt
        prompt_text = self.llm_chain.prompt.format(**inputs_for_prompt)

        # Calculate the total number of tokens in the prompt, including retrieved documents
        total_tokens = calculate_num_tokens(self.llm_chain.llm.tokenizer, [prompt_text])

        # # Manage token limits by summarizing documents if necessary
        # if total_tokens >= self.max_context_length - 100 and retrieved_docs_content:
        #     # Summarize the retrieved documents to reduce token count
        #     summarized_content = self.summarize_docs(retrieved_docs_content)
        #     kwargs["documents"] = summarized_content
        #     inputs_for_prompt["documents"] = summarized_content
        #     # Reformat the prompt with summarized documents
        #     prompt_text = self.llm_chain.prompt.format(**inputs_for_prompt)

        # --- Added code to record intermediate data ---
        # Store the prompt and other relevant data
        self.step_data.append({
            'step': current_step,
            'inputs_for_prompt': inputs_for_prompt,
            'prompt_text': prompt_text,
            'retrieved_docs_content': retrieved_docs_content,
            'rag_query': question if self.rag_requery is not None else "Patient reports",
            # You can add more fields here if needed
        })
        # --- End of added code ---

        # Prepare inputs for the LLM chain
        llm_chain_input_keys = self.llm_chain.input_keys
        inputs_for_llm_chain = {k: v for k, v in updated_kwargs.items() if k in llm_chain_input_keys}

        # Generate the output using the LLM chain
        full_output = self.llm_chain.predict(**inputs_for_llm_chain, stop=self._stop)

        # --- Added code to record the output ---
        # Update the last entry in step_data with the full output
        self.step_data[-1]['full_output'] = full_output
        # --- End of added code ---

        # Parse and return the LLM output
        return self.output_parser.parse(full_output)

    def extract_current_information(
        self, intermediate_steps: List[Tuple[AgentAction, str]], kwargs: Any
    ) -> str:
        # Concatenate all observations and the current input
        # observations = " \n".join([observation for action, observation in intermediate_steps])
        # Concatenate all valid observations (from allowed tools) and the current input
        # observations = ""
        # for action, observation in intermediate_steps:
        #     if action.tool in self.allowed_tools:
        #         observations += f"{observation}\n"
        observations = "\n".join(
            [
                f"{observation}"
                for action, observation in intermediate_steps
                if action.tool in self.allowed_tools
            ]
        )
        current_input = kwargs.get("input", "")
        current_information = f"{current_input}\n\n{observations}"
        return current_information.strip()

    def summarize_docs(self, docs_content: str) -> str:
        # Implement summarization logic
        # For simplicity, truncate the content to fit within token limits
        max_tokens = self.max_context_length - 500  # Adjust as needed
        tokens = self.llm_chain.llm.tokenizer.encode(docs_content)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return self.llm_chain.llm.tokenizer.decode(tokens)

    # Need to override to pass input so that we can calculate the number of tokes
    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts, kwargs = self._construct_scratchpad(intermediate_steps, **kwargs)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}

        # --- RAG ---
        # Ensure 'documents' is included in inputs with a default value
        if 'documents' not in kwargs and self.rag_retriever_agent is not None:
            new_inputs['documents'] = ''
        # --- End RAG ---

        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += (
                f"\n{self.observation_prefix}{observation.strip()}\n{self.llm_prefix} "
            )
        
        # Ensure 'documents' is in kwargs with a default value
        if 'documents' not in kwargs and self.rag_retriever_agent is not None:
            kwargs['documents'] = ''
        
        if self.rag_retriever_agent is not None: #RAG
            # When formatting the prompt, include 'documents'
            prompt_text = self.llm_chain.prompt.format(
                input=kwargs["input"],
                agent_scratchpad=thoughts,
                documents=kwargs["documents"],
            )
        else:
            # When formatting the prompt, exclude 'documents'
            prompt_text = self.llm_chain.prompt.format(
                input=kwargs["input"],
                agent_scratchpad=thoughts,
            )
        

        # Calculate token count with 'documents' included
        if (
            calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [prompt_text],
            )
            >= self.max_context_length - 100
        ) and self.summarize:
            thoughts = self._summarize_steps(intermediate_steps)
        
        if self.rag_retriever_agent is not None: #RAG
            # Repeat the process with updated thoughts
            prompt_text = self.llm_chain.prompt.format(
                input=kwargs["input"],
                agent_scratchpad=thoughts,
                documents=kwargs["documents"],
            )
        else:
            # Repeat the process with updated thoughts
            prompt_text = self.llm_chain.prompt.format(
                input=kwargs["input"],
                agent_scratchpad=thoughts,
            )
        
        # Worst-case scenario handling
        if (
            calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [prompt_text],
            )
            >= self.max_context_length - 100
        ):
            prompt_and_input_tokens = calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [
                    self.llm_chain.prompt.format( # RAG
                        input=kwargs["input"],
                        agent_scratchpad="",
                        documents=kwargs["documents"],
                    ) if self.rag_retriever_agent is not None else 
                    self.llm_chain.prompt.format(
                        input=kwargs["input"],
                        agent_scratchpad="",
                    )
                ],
            )
            if prompt_and_input_tokens > self.max_context_length - 100:
                prompt_tokens = calculate_num_tokens(
                    self.llm_chain.llm.tokenizer,
                    [
                        self.llm_chain.prompt.format( # RAG
                            input="",
                            agent_scratchpad="",
                            documents=kwargs["documents"],
                        ) if self.rag_retriever_agent is not None else
                        self.llm_chain.prompt.format(
                            input="",
                            agent_scratchpad="",
                        )
                    ],
                )
                kwargs["input"] = truncate_text(
                    self.llm_chain.llm.tokenizer,
                    kwargs["input"],
                    self.max_context_length - prompt_tokens - 200,
                )
                thoughts = ""
            else:
                thoughts = truncate_text(
                    self.llm_chain.llm.tokenizer,
                    thoughts,
                    self.max_context_length - prompt_and_input_tokens - 100,
                )
            thoughts += (
                f'{self.tags["ai_tag_end"]}{self.tags["user_tag_start"]}'
                f'Provide a Final Diagnosis and Treatment.'
                f'{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}{self.llm_prefix}'
            )
        
        # Return the thoughts and updated kwargs
        return " " + thoughts.strip(), kwargs

    # Takes all tool requests and observations and summarizes them one-by-one
    def _summarize_steps(self, intermediate_steps):
        prompt = PromptTemplate(
            template=SUMMARIZE_OBSERVATION_TEMPLATE,
            input_variables=["observation"],
            partial_variables={
                "system_tag_start": self.tags["system_tag_start"],
                "system_tag_end": self.tags["system_tag_end"],
                "user_tag_start": self.tags["user_tag_start"],
                "user_tag_end": self.tags["user_tag_end"],
                "ai_tag_start": self.tags["ai_tag_start"],
            },
        )
        chain = LLMChain(llm=self.llm_chain.llm, prompt=prompt)
        summaries = []
        summaries.append("A summary of information I know thus far:")
        for indx, (action, observation) in enumerate(intermediate_steps):
            # Only summarize valid actions
            if action.tool in self.allowed_tools:
                # Keep format as in instruction to re-enforce schema
                summaries.append("Action: " + action.tool)
                if action.tool in [
                    "Laboratory Tests",
                    "Imaging",
                    "Diagnostic Criteria",
                ]:
                    summaries.append(
                        "Action Input: "
                        + action_input_pretty_printer(
                            action.tool_input["action_input"], self.lab_test_mapping_df
                        )
                    )
                # Check cache to not re-summarize same observation
                summary = self.observation_summary_cache.get_summary(observation)
                if not summary:
                    # Summary of each step should be minimal and should not exceed max_context_length
                    prompt_tokens = calculate_num_tokens(
                        self.llm_chain.llm.tokenizer,
                        [
                            prompt.format(observation=""),
                        ],
                    )

                    observation = truncate_text(
                        self.llm_chain.llm.tokenizer,
                        observation,
                        self.max_context_length
                        - prompt_tokens
                        - 100,  # Gives a max of 100 tokens to generate for the summary if we are near context length limit. Usually only used when model does really weird infinite generations of action inputs and doesnt hit a stop token so shouldnt be much actual info to summarize anyway
                    )
                    summary = chain.predict(observation=observation, stop=[])
                    # Add to cache
                    self.observation_summary_cache.add_summary(observation, summary)
                summaries.append("Observation: " + summary)
            else:
                # Include invalid requests in summary to not run into infinite loop of same invalid tool being ordered
                invalid_request = action.log
                # Condense invalid request to the action and everything afterwards. Can remove thinking
                if "action:" in action.log.lower():
                    invalid_request = action.log[action.log.lower().index("action:") :]
                summaries.append(
                    f"I tried '{invalid_request}', but it was an invalid request."
                )
                # If invalid tool was final request, remind of valid tools and diagnosis option. Add string to last summary because we dont want to force newlines that the prompt templates maybe do not want
                if indx == len(intermediate_steps) - 1:
                    summaries[-1] = summaries[-1] + (
                        f'{self.tags["ai_tag_end"]}{self.tags["user_tag_start"]}Please choose a valid tool from {self.allowed_tools} or provide a Final Diagnosis and Treatment.{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}{self.llm_prefix}'
                    )
                    return "\n".join(summaries)
        summaries.append(self.llm_prefix)
        return "\n".join(summaries)


def create_prompt(
    tags, tool_names, add_tool_descr, tool_use_examples
) -> PromptTemplate:
    template = PromptTemplate(
        template=CHAT_TEMPLATE,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": action_input_pretty_printer(tool_names, None),
            "add_tool_descr": add_tool_descr,
            "examples": tool_use_examples,
            "system_tag_start": tags["system_tag_start"],
            "user_tag_start": tags["user_tag_start"],
            "ai_tag_start": tags["ai_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_end": tags["user_tag_end"],
        },
    )
    return template

def create_prompt_rag(
    tags, tool_names, add_tool_descr, tool_use_examples
) -> PromptTemplate:
    template = PromptTemplate(
        template=CHAT_TEMPLATE_RAG,
        input_variables=["input", "agent_scratchpad", "documents"],
        partial_variables={
            "tool_names": action_input_pretty_printer(tool_names, None),
            "add_tool_descr": add_tool_descr,
            "examples": tool_use_examples,
            "system_tag_start": tags["system_tag_start"],
            "user_tag_start": tags["user_tag_start"],
            "ai_tag_start": tags["ai_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_end": tags["user_tag_end"],
        },
    )
    return template

def create_prompt_requery(
    tags,
    single_report: bool = False,
) -> PromptTemplate:
    template = PromptTemplate(
        template=REQUERY_PROMPT if not single_report else REQUERY_PROMPT_SINGLE_REPORT,
        input_variables=["original_text"],
        partial_variables={
            "system_tag_start": tags["system_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_start": tags["user_tag_start"],
            "user_tag_end": tags["user_tag_end"],
            "ai_tag_start": tags["ai_tag_start"],
            # B. Note: we no longer mention requery_eos in the prompt text
            # but we still use REQUERY_EOS as a stop token in the LLM config.
        },
    )
    return template


def build_agent_executor_ZeroShot(
    patient,
    llm,
    lab_test_mapping_path,
    logfile,
    max_context_length,
    tags,
    include_ref_range,
    bin_lab_results,
    include_tool_use_examples,
    provide_diagnostic_criteria,
    summarize,
    model_stop_words,
    rag_retriever_agent=None, #RAG
    rag_requery=False, #RAG
    rag_requery_max_length = 2048, #RAG
    rag_requery_only_last_report=False, #RAG
    rag_u_ordering=False, #RAG
    rag_skip_first_retrieval=False, #RAG
):
    with open(lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)

    # Define which tools the agent can use to answer user queries
    tools = [
        DoPhysicalExamination(action_results=patient),
        RunLaboratoryTests(
            action_results=patient,
            lab_test_mapping_df=lab_test_mapping_df,
            include_ref_range=include_ref_range,
            bin_lab_results=bin_lab_results,
        ),
        RunImaging(action_results=patient),
    ]

    # Go through options and see if we want to add any extra tools.
    add_tool_use_examples = ""
    add_tool_descr = ""
    if provide_diagnostic_criteria:
        tools.append(ReadDiagnosticCriteria())
        add_tool_descr += DIAG_CRIT_TOOL_DESCR
        add_tool_use_examples += DIAG_CRIT_TOOL_USE_EXAMPLE

    tool_names = [tool.name for tool in tools]

    # Create prompt
    tool_use_examples = ""
    if include_tool_use_examples:
        if rag_retriever_agent is None:
            tool_use_examples = TOOL_USE_EXAMPLES.format(
                add_tool_use_examples=add_tool_use_examples
            )
        else:
            tool_use_examples = TOOL_USE_EXAMPLES_RAG.format(
                add_tool_use_examples=add_tool_use_examples
            )

    if rag_retriever_agent is None:
        prompt = create_prompt(tags, tool_names, add_tool_descr, tool_use_examples)
    else:
        prompt = create_prompt_rag(tags, tool_names, add_tool_descr, tool_use_examples)

    # Create output parser
    output_parser = DiagnosisWorkflowParser(lab_test_mapping_df=lab_test_mapping_df)

    # Initialize logging callback if file provided
    handler = None
    if logfile:
        handler = [FileCallbackHandler(logfile)]

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=handler)

    # Create agent
    agent = CustomZeroShotAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=list(STOP_WORDS + model_stop_words),
        allowed_tools=tool_names,
        verbose=True,
        return_intermediate_steps=True,
        max_context_length=max_context_length,
        tags=tags,
        lab_test_mapping_df=lab_test_mapping_df,
        summarize=summarize,
        rag_retriever_agent=rag_retriever_agent, #RAG
        rag_requery=rag_requery, #RAG
        rag_requery_max_length=rag_requery_max_length, #RAG
        rag_requery_only_last_report=rag_requery_only_last_report, #RAG
        rag_u_ordering=rag_u_ordering, #RAG
        rag_skip_first_retrieval=rag_skip_first_retrieval, #RAG
    )

    # Init agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        return_intermediate_steps=True,
        callbacks=handler,
    )

    return agent_executor
