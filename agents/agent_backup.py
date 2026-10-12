import pickle
from typing import List, Tuple, Union, Dict, Any
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


from agents.prompts import (
    CHAT_TEMPLATE,
    SUMMARIZE_OBSERVATION_TEMPLATE,
    DIAG_CRIT_TOOL_DESCR,
    TOOL_USE_EXAMPLES,
    DIAG_CRIT_TOOL_USE_EXAMPLE,
    CHAT_TEMPLATE_RAG, #RAG
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
    retrieved_docs_per_step: Dict[int, List[Dict]] = {}
    # step_counter: int = 0
    # --- End RAG ---

    class Config:
        arbitrary_types_allowed = True

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

        # Extract current information to form the question
        question = self.extract_current_information(intermediate_steps, kwargs)

        # Initialize retrieved documents content
        retrieved_docs_content = ""

        # Perform RAG retrieval if a retriever agent is available
        if self.rag_retriever_agent is not None:
            retrieved_docs = self.rag_retriever_agent.retrieve(question)
            retrieved_docs_content = "\n".join(doc["content"] for doc in retrieved_docs)
            # Store retrieved documents using current_step as the key
            self.retrieved_docs_per_step[current_step] = retrieved_docs
            kwargs["documents"] = retrieved_docs_content  # Add retrieved documents to kwargs

        # Construct the scratchpad and update kwargs
        thoughts, updated_kwargs = self._construct_scratchpad(intermediate_steps, **kwargs)
        updated_kwargs.update({
            "agent_scratchpad": thoughts,
            "stop": self._stop
        })

        # Prepare inputs for the prompt
        prompt_variables = self.llm_chain.prompt.input_variables
        inputs_for_prompt = {k: v for k, v in updated_kwargs.items() if k in prompt_variables}

        # Format the prompt
        prompt_text = self.llm_chain.prompt.format(**inputs_for_prompt)

        # Calculate the total number of tokens in the prompt, including retrieved documents
        total_tokens = calculate_num_tokens(self.llm_chain.llm.tokenizer, [prompt_text])

        # Manage token limits by summarizing documents if necessary
        if total_tokens >= self.max_context_length - 100 and retrieved_docs_content:
            # Summarize the retrieved documents to reduce token count
            summarized_content = self.summarize_docs(retrieved_docs_content)
            kwargs["documents"] = summarized_content
            inputs_for_prompt["documents"] = summarized_content
            # Reformat the prompt with summarized documents
            prompt_text = self.llm_chain.prompt.format(**inputs_for_prompt)

        # Prepare inputs for the LLM chain
        llm_chain_input_keys = self.llm_chain.input_keys
        inputs_for_llm_chain = {k: v for k, v in updated_kwargs.items() if k in llm_chain_input_keys}

        # Generate the output using the LLM chain
        full_output = self.llm_chain.predict(**inputs_for_llm_chain, stop=self._stop)

        # Parse and return the LLM output
        return self.output_parser.parse(full_output)

    def extract_current_information(
        self, intermediate_steps: List[Tuple[AgentAction, str]], kwargs: Any
    ) -> str:
        # Concatenate all observations and the current input
        observations = " ".join([observation for action, observation in intermediate_steps])
        current_input = kwargs.get("input", "")
        current_information = f"{observations} {current_input}"
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
                f'{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}Final'
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
        tool_use_examples = TOOL_USE_EXAMPLES.format(
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
