"""Utility functions for working with agents."""

import sys
import time
import uuid
from typing import Optional, Any

import signal
import threading
import time
from typing import Optional

from langgraph.prebuilt import create_react_agent
from ra_aid.agents.ciayn_agent import CiaynAgent
from ra_aid.agents.ciayn_agent import CiaynAgent
from ra_aid.console.formatting import print_stage_header, print_error
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from typing import List, Any
from ra_aid.console.output import print_agent_output
from ra_aid.logging_config import get_logger
from ra_aid.exceptions import AgentInterrupt
from ra_aid.tool_configs import (
    get_implementation_tools,
    get_research_tools,
    get_planning_tools,
    get_web_research_tools
)
from ra_aid.prompts import (
    IMPLEMENTATION_PROMPT,
    EXPERT_PROMPT_SECTION_IMPLEMENTATION,
    HUMAN_PROMPT_SECTION_IMPLEMENTATION,
    EXPERT_PROMPT_SECTION_RESEARCH,
    WEB_RESEARCH_PROMPT_SECTION_RESEARCH,
    WEB_RESEARCH_PROMPT_SECTION_CHAT,
    WEB_RESEARCH_PROMPT_SECTION_PLANNING,
    RESEARCH_PROMPT,
    RESEARCH_ONLY_PROMPT,
    HUMAN_PROMPT_SECTION_RESEARCH,
    PLANNING_PROMPT,
    EXPERT_PROMPT_SECTION_PLANNING,
    WEB_RESEARCH_PROMPT_SECTION_PLANNING,
    HUMAN_PROMPT_SECTION_PLANNING,
    WEB_RESEARCH_PROMPT,
)
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage
from anthropic import APIError, APITimeoutError, RateLimitError, InternalServerError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.tools.memory import (
    _global_memory,
    get_memory_value,
    get_related_files,
)
from ra_aid.tool_configs import get_research_tools
from ra_aid.prompts import (
    RESEARCH_PROMPT,
    RESEARCH_ONLY_PROMPT,
    EXPERT_PROMPT_SECTION_RESEARCH,
    HUMAN_PROMPT_SECTION_RESEARCH
)


console = Console()

logger = get_logger(__name__)

@tool
def output_markdown_message(message: str) -> str:
    """Outputs a message to the user, optionally prompting for input."""
    console.print(Panel(Markdown(message.strip()), title="🤖 Assistant"))
    return "Message output."

def create_agent(
    model: BaseChatModel,
    tools: List[Any],
    *,
    checkpointer: Any = None
) -> Any:
    """Create a react agent with the given configuration.
    
    Args:
        model: The LLM model to use
        tools: List of tools to provide to the agent
        checkpointer: Optional memory checkpointer
        
    Returns:
        The created agent instance
    """
    try:
        # Get model name if available
        provider = _global_memory.get('config', {}).get('provider')
        model_name = _global_memory.get('config', {}).get('model')
        
        # Use REACT agent for Anthropic Claude models, otherwise use CIAYN
        if provider == 'anthropic' and 'claude' in model_name:
            logger.debug("Using create_react_agent to instantiate agent.")
            return create_react_agent(model, tools, checkpointer=checkpointer)
        else:
            logger.debug("Using CiaynAgent agent instance.")
            return CiaynAgent(model, tools)
            
    except Exception as e:
        # Default to REACT agent if provider/model detection fails
        logger.warning(f"Failed to detect model type: {e}. Defaulting to REACT agent.")
        return create_react_agent(model, tools, checkpointer=checkpointer)

def run_research_agent(
    base_task_or_query: str,
    model,
    *,
    expert_enabled: bool = False,
    research_only: bool = False,
    hil: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None
) -> Optional[str]:
    """Run a research agent with the given configuration.

    Args:
        base_task_or_query: The main task or query for research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        research_only: Whether this is a research-only task
        hil: Whether human-in-the-loop mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running

    Returns:
        Optional[str]: The completion message if task completed successfully

    Example:
        result = run_research_agent(
            "Research Python async patterns",
            model,
            expert_enabled=True,
            research_only=True
        )
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting research agent with thread_id=%s", thread_id)
    logger.debug("Research configuration: expert=%s, research_only=%s, hil=%s, web=%s",
                expert_enabled, research_only, hil, web_research_enabled)

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_research_tools(
        research_only=research_only,
        expert_enabled=expert_enabled,
        human_interaction=hil,
        web_research_enabled=config.get('web_research_enabled', False)
    )

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""
    web_research_section = WEB_RESEARCH_PROMPT_SECTION_RESEARCH if config.get('web_research_enabled') else ""

    # Get research context from memory
    key_facts = _global_memory.get("key_facts", "")
    code_snippets = _global_memory.get("code_snippets", "")
    related_files = _global_memory.get("related_files", "")

    # Build prompt
    prompt = (RESEARCH_ONLY_PROMPT if research_only else RESEARCH_PROMPT).format(
        base_task=base_task_or_query,
        research_only_note='' if research_only else ' Only request implementation if the user explicitly asked for changes to be made.',
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        key_facts=key_facts,
        code_snippets=code_snippets,
        related_files=related_files
    )

    # Set up configuration
    run_config = {
         "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    if config:
        run_config.update(config)

    try:
        # Display console message if provided
        if console_message:
            console.print(Panel(Markdown(console_message), title="🔬 Looking into it..."))

        # Run agent with retry logic if available
        if agent is not None:
            logger.debug("Research agent completed successfully")
            return run_agent_with_retry(agent, prompt, run_config)
        else:
            # Just run web research tools directly if no agent
            logger.debug("No model provided, running web research tools directly")
            return run_web_research_agent(
                base_task_or_query,
                model=None,
                expert_enabled=expert_enabled,
                hil=hil,
                web_research_enabled=web_research_enabled,
                memory=memory,
                config=config,
                thread_id=thread_id,
                console_message=console_message
            )
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Research agent failed: %s", str(e), exc_info=True)
        raise

def run_web_research_agent(
    query: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None
) -> Optional[str]:
    """Run a web research agent with the given configuration.

    Args:
        query: The mainquery for web research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running

    Returns:
        Optional[str]: The completion message if task completed successfully

    Example:
        result = run_web_research_agent(
            "Research latest Python async patterns",
            model,
            expert_enabled=True
        )
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting web research agent with thread_id=%s", thread_id)
    logger.debug("Web research configuration: expert=%s, hil=%s, web=%s",
                expert_enabled, hil, web_research_enabled)

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools using restricted web research toolset
    tools = get_web_research_tools(expert_enabled=expert_enabled)

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""

    # Get research context from memory
    key_facts = _global_memory.get("key_facts", "")
    code_snippets = _global_memory.get("code_snippets", "")
    related_files = _global_memory.get("related_files", "")

    # Build prompt
    prompt = WEB_RESEARCH_PROMPT.format(
        web_research_query=query,
        expert_section=expert_section,
        human_section=human_section,
        key_facts=key_facts,
        code_snippets=code_snippets,
        related_files=related_files
    )

    # Set up configuration
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    if config:
        run_config.update(config)

    try:
        # Display console message if provided
        if console_message:
            console.print(Panel(Markdown(console_message), title="🔬 Researching..."))

        logger.debug("Web research agent completed successfully")
        return run_agent_with_retry(agent, prompt, run_config)

    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Web research agent failed: %s", str(e), exc_info=True)
        raise

def run_planning_agent(
    base_task: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None
) -> Optional[str]:
    """Run a planning agent to create implementation plans.

    Args:
        base_task: The main task to plan implementation for
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)

    Returns:
        Optional[str]: The completion message if planning completed successfully
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting planning agent with thread_id=%s", thread_id)
    logger.debug("Planning configuration: expert=%s, hil=%s", expert_enabled, hil)

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_planning_tools(expert_enabled=expert_enabled, web_research_enabled=config.get('web_research_enabled', False))

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_PLANNING if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_PLANNING if hil else ""
    web_research_section = WEB_RESEARCH_PROMPT_SECTION_PLANNING if config.get('web_research_enabled') else ""

    # Build prompt
    planning_prompt = PLANNING_PROMPT.format(
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        base_task=base_task,
        research_notes=get_memory_value('research_notes'),
        related_files="\n".join(get_related_files()),
        key_facts=get_memory_value('key_facts'),
        key_snippets=get_memory_value('key_snippets'),
        research_only_note='' if config.get('research_only') else ' Only request implementation if the user explicitly asked for changes to be made.'
    )

    # Set up configuration
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    if config:
        run_config.update(config)

    try:
        print_stage_header("Planning Stage")
        logger.debug("Planning agent completed successfully")
        return run_agent_with_retry(agent, planning_prompt, run_config)
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Planning agent failed: %s", str(e), exc_info=True)
        raise

def run_task_implementation_agent(
    base_task: str,
    tasks: list,
    task: str,
    plan: str,
    related_files: list,
    model,
    *,
    expert_enabled: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None
) -> Optional[str]:
    """Run an implementation agent for a specific task.

    Args:
        base_task: The main task being implemented
        tasks: List of tasks to implement
        plan: The implementation plan
        related_files: List of related files
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)

    Returns:
        Optional[str]: The completion message if task completed successfully
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting implementation agent with thread_id=%s", thread_id)
    logger.debug("Implementation configuration: expert=%s, web=%s", expert_enabled, web_research_enabled)
    logger.debug("Task details: base_task=%s, current_task=%s", base_task, task)
    logger.debug("Related files: %s", related_files)

    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_implementation_tools(expert_enabled=expert_enabled, web_research_enabled=config.get('web_research_enabled', False))

    # Create agent
    agent = create_agent(model, tools, checkpointer=memory)

    # Build prompt
    prompt = IMPLEMENTATION_PROMPT.format(
        base_task=base_task,
        task=task,
        tasks=tasks,
        plan=plan,
        related_files=related_files,
        key_facts=get_memory_value('key_facts'),
        key_snippets=get_memory_value('key_snippets'),
        research_notes=get_memory_value('research_notes'),
        work_log=get_memory_value('work_log'),
        expert_section=EXPERT_PROMPT_SECTION_IMPLEMENTATION if expert_enabled else "",
        human_section=HUMAN_PROMPT_SECTION_IMPLEMENTATION if _global_memory.get('config', {}).get('hil', False) else "",
        web_research_section=WEB_RESEARCH_PROMPT_SECTION_CHAT if config.get('web_research_enabled') else ""
    )

    # Set up configuration
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    if config:
        run_config.update(config)

    try:
        logger.debug("Implementation agent completed successfully")
        return run_agent_with_retry(agent, prompt, run_config)
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Implementation agent failed: %s", str(e), exc_info=True)
        raise

_CONTEXT_STACK = []
_INTERRUPT_CONTEXT = None
_FEEDBACK_MODE = False

def _request_interrupt(signum, frame):
    global _INTERRUPT_CONTEXT
    if _CONTEXT_STACK:
        _INTERRUPT_CONTEXT = _CONTEXT_STACK[-1]

    if _FEEDBACK_MODE:
        print()
        print(" 👋 Bye!")
        print()
        sys.exit(0)

class InterruptibleSection:
    def __enter__(self):
        _CONTEXT_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _CONTEXT_STACK.remove(self)

def check_interrupt():
    if _CONTEXT_STACK and _INTERRUPT_CONTEXT is _CONTEXT_STACK[-1]:
        raise AgentInterrupt("Interrupt requested")

def run_agent_with_retry(agent, prompt: str, config: dict) -> Optional[str]:
    """
    Run an agent with robust retry logic for handling API errors and interruptions.
    
    This function manages agent execution with comprehensive error handling, retry mechanisms, 
    and interrupt support. It streams agent outputs, tracks execution depth, and implements 
    exponential backoff for transient API errors.
    
    Parameters:
        agent (Any): The agent to be executed
        prompt (str): The input prompt for the agent
        config (dict): Configuration settings for agent execution
    
    Returns:
        Optional[str]: A success message if the agent completes execution, otherwise None
    
    Raises:
        RuntimeError: If maximum retry attempts are exhausted
        AgentInterrupt: If the agent execution is manually interrupted
        KeyboardInterrupt: If a keyboard interrupt is detected
    
    Notes:
        - Supports nested agent execution tracking via agent_depth
        - Handles various API errors with exponential backoff
        - Provides interrupt handling and signal management
        - Logs debug information and error details
        - Resets global memory flags after task/plan completion
    """
    logger.debug("Running agent with prompt length: %d", len(prompt))
    original_handler = None
    if threading.current_thread() is threading.main_thread():
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _request_interrupt)

    max_retries = 20
    base_delay = 1

    with InterruptibleSection():
        try:
            # Track agent execution depth
            current_depth = _global_memory.get('agent_depth', 0)
            _global_memory['agent_depth'] = current_depth + 1

            for attempt in range(max_retries):
                logger.debug("Attempt %d/%d", attempt + 1, max_retries)
                check_interrupt()
                try:
                    for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, config):
                        logger.debug("Agent output: %s", chunk)
                        check_interrupt()
                        print_agent_output(chunk)
                        if _global_memory['plan_completed']:
                            _global_memory['plan_completed'] = False
                            _global_memory['task_completed'] = False
                            _global_memory['completion_message'] = ''
                            break
                        if _global_memory['task_completed'] or _global_memory['plan_completed']:
                            _global_memory['task_completed'] = False
                            _global_memory['completion_message'] = ''
                            break
                    logger.debug("Agent run completed successfully")
                    return "Agent run completed successfully"
                except (KeyboardInterrupt, AgentInterrupt):
                    raise
                except (InternalServerError, APITimeoutError, RateLimitError, APIError) as e:
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached, failing: %s", str(e))
                        raise RuntimeError(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                    logger.warning("API error (attempt %d/%d): %s", attempt + 1, max_retries, str(e))
                    delay = base_delay * (2 ** attempt)
                    print_error(f"Encountered {e.__class__.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    start = time.monotonic()
                    while time.monotonic() - start < delay:
                        check_interrupt()
                        time.sleep(0.1)
        finally:
            # Reset depth tracking
            _global_memory['agent_depth'] = _global_memory.get('agent_depth', 1) - 1

            if original_handler and threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, original_handler)

class AgentSupervisor:
    def __init__(self, model=None, max_turns=10):
        """
        Initialize an AgentSupervisor instance for managing task checklists.
        
        Parameters:
            model (Optional[BaseChatModel]): The language model to be used. 
                If not provided, retrieves the default model from global configuration.
            max_turns (int, optional): Maximum number of turns allowed for task execution. 
                Defaults to 10.
        
        Attributes:
            model (BaseChatModel): The language model for agent interactions.
            max_turns (int): Maximum execution turns limit.
            checklist (List): An empty list to store task checklist items.
        """
        self.model = model or _global_memory.get('config', {}).get('model')
        self.max_turns = max_turns
        self.checklist = []

    def load_checklist(self, checklist_path: str):
        """
        Load a checklist from a specified file path.
        
        Parameters:
            checklist_path (str): The file path to the checklist text file.
        
        Side Effects:
            Populates the instance's `checklist` attribute with lines read from the file.
        
        Raises:
            FileNotFoundError: If the specified file cannot be found.
            IOError: If there are issues reading the file.
        """
        with open(checklist_path, 'r') as file:
            self.checklist = file.readlines()

    def save_checklist(self, checklist_path: str):
        """
        Save the current checklist to a specified file path.
        
        Parameters:
            checklist_path (str): The file path where the checklist will be saved.
        
        Raises:
            IOError: If there is an issue writing to the specified file path.
        """
        with open(checklist_path, 'w') as file:
            file.writelines(self.checklist)

    def validate_checklist(self) -> bool:
        # Implement validation logic for the checklist
        """
        Validates the checklist by ensuring all items are non-empty.
        
        Checks that each item in the checklist is a non-empty string after stripping whitespace.
        
        Returns:
            bool: True if all checklist items are non-empty, False otherwise.
        """
        return all(item.strip() for item in self.checklist)

    def authorize_task_completion(self, task_output: str) -> bool:
        # Implement authorization logic based on the checklist
        """
        Determines whether a task can be considered complete based on checklist validation and task output.
        
        Parameters:
            task_output (str): The output text from the task being evaluated for completion.
        
        Returns:
            bool: True if the checklist is valid and the task output contains the word "complete", False otherwise.
        
        Notes:
            - Checks the validity of the current checklist using `validate_checklist()`
            - Performs a case-insensitive search for the word "complete" in the task output
        """
        return self.validate_checklist() and "complete" in task_output.lower()
