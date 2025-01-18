"""Tools for spawning and managing sub-agents."""

from langchain_core.tools import tool
from typing import Dict, Any, Union, List
from typing_extensions import TypeAlias
from ..agent_utils import AgentInterrupt, AgentSupervisor
from ra_aid.exceptions import AgentInterrupt
ResearchResult = Dict[str, Union[str, bool, Dict[int, Any], List[Any], None]]
from rich.console import Console
from ra_aid.tools.memory import _global_memory
from ra_aid.console.formatting import print_error
from .memory import get_memory_value, get_related_files, get_work_log, reset_work_log
from .human import ask_human
from ..llm import initialize_llm
from ..console import print_task_header

CANCELLED_BY_USER_REASON = "The operation was explicitly cancelled by the user. This typically is an indication that the action requested was not aligned with the user request."

RESEARCH_AGENT_RECURSION_LIMIT = 2

console = Console()

@tool("request_research")
def request_research(query: str) -> ResearchResult:
    """Spawn a research-only agent to investigate the given query.

    This function creates a new research agent to investigate the given query. It includes 
    recursion depth limiting to prevent infinite recursive research calls.

    Args:
        query: The research question or project description
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    # Check recursion depth
    current_depth = _global_memory.get('agent_depth', 0)
    if current_depth >= RESEARCH_AGENT_RECURSION_LIMIT:
        print_error("Maximum research recursion depth reached")
        return {
            "completion_message": "Research stopped - maximum recursion depth reached",
            "key_facts": get_memory_value("key_facts"),
            "related_files": get_related_files(),
            "research_notes": get_memory_value("research_notes"),
            "key_snippets": get_memory_value("key_snippets"),
            "success": False,
            "reason": "max_depth_exceeded"
        }

    success = True
    reason = None
    
    try:
        # Run research agent
        from ..agent_utils import run_research_agent
        result = run_research_agent(
            query,
            model,
            expert_enabled=True,
            research_only=True,
            hil=config.get('hil', False),
            console_message=query,
            config=config
        )
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print_error(f"Error during research: {str(e)}")
        success = False
        reason = f"error: {str(e)}"
    finally:
        # Get completion message if available
        completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
        
        # Get and reset work log if at root depth
        work_log = get_work_log() if current_depth == 1 else None
        if current_depth == 1:
            reset_work_log()
            
        # Clear completion state from global memory
        _global_memory['completion_message'] = ''
        _global_memory['task_completed'] = False
        
    return {
        "work_log": work_log,
        "completion_message": completion_message,
        "key_facts": get_memory_value("key_facts"),
        "related_files": get_related_files(),
        "research_notes": get_memory_value("research_notes"),
        "key_snippets": get_memory_value("key_snippets"),
        "success": success,
        "reason": reason
    }

@tool("request_web_research")
def request_web_research(query: str) -> ResearchResult:
    """Spawn a web research agent to investigate the given query using web search.
    
    Args:
        query: The research question or project description
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    success = True
    reason = None
    
    try:
        # Run web research agent
        from ..agent_utils import run_web_research_agent
        result = run_web_research_agent(
            query,
            model,
            expert_enabled=True,
            hil=config.get('hil', False),
            console_message=query,
            config=config
        )
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print_error(f"Error during web research: {str(e)}")
        success = False
        reason = f"error: {str(e)}"
    finally:
        # Get completion message if available
        completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
        
        # Get and reset work log if at root depth
        current_depth = _global_memory.get('agent_depth', 0)
        work_log = get_work_log() if current_depth == 1 else None
        if current_depth == 1:
            reset_work_log()
            
        # Clear completion state from global memory
        _global_memory['completion_message'] = ''
        _global_memory['task_completed'] = False
        
    return {
        "work_log": work_log,
        "completion_message": completion_message,
        "key_snippets": get_memory_value("key_snippets"),
        "research_notes": get_memory_value("research_notes"),
        "success": success,
        "reason": reason
    }

@tool("request_research_and_implementation")
def request_research_and_implementation(query: str) -> Dict[str, Any]:
    """Spawn a research agent to investigate and implement the given query.
    
    Args:
        query: The research question or project description
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    try:
        # Run research agent
        from ..agent_utils import run_research_agent
        result = run_research_agent(
            query,
            model,
            expert_enabled=True,
            research_only=False,
            hil=config.get('hil', False),
            console_message=query,
            config=config
        )
        
        success = True
        reason = None
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        console.print(f"\n[red]Error during research: {str(e)}[/red]")
        success = False
        reason = f"error: {str(e)}"
        
    # Get completion message if available
    completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
    
    # Get and reset work log if at root depth
    current_depth = _global_memory.get('agent_depth', 0)
    work_log = get_work_log() if current_depth == 1 else None
    if current_depth == 1:
        reset_work_log()
    
    # Clear completion state from global memory
    _global_memory['completion_message'] = ''
    _global_memory['task_completed'] = False
    _global_memory['plan_completed'] = False

    return {
        "work_log": work_log,
        "completion_message": completion_message,
        "key_facts": get_memory_value("key_facts"),
        "related_files": get_related_files(),
        "research_notes": get_memory_value("research_notes"),
        "key_snippets": get_memory_value("key_snippets"),
        "success": success,
        "reason": reason
    }

@tool("request_task_implementation")
def request_task_implementation(task_spec: str) -> Dict[str, Any]:
    """Spawn an implementation agent to execute the given task.
    
    Args:
        task_spec: The full task specification
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    # Get required parameters
    tasks = [_global_memory['tasks'][task_id] for task_id in sorted(_global_memory['tasks'])]
    plan = _global_memory.get('plan', '')
    related_files = list(_global_memory['related_files'].values())
    
    try:
        print_task_header(task_spec)
        # Run implementation agent
        from ..agent_utils import run_task_implementation_agent
        result = run_task_implementation_agent(
            base_task=_global_memory.get('base_task', ''),
            tasks=tasks,
            task=task_spec,
            plan=plan, 
            related_files=related_files,
            model=model,
            expert_enabled=True,
            config=config
        )
        
        success = True
        reason = None
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print_error(f"Error during task implementation: {str(e)}")
        success = False
        reason = f"error: {str(e)}"
        
    # Get completion message if available
    completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
    
    # Get and reset work log if at root depth
    current_depth = _global_memory.get('agent_depth', 0)
    work_log = get_work_log() if current_depth == 1 else None
    if current_depth == 1:
        reset_work_log()
    
    # Clear completion state from global memory
    _global_memory['completion_message'] = ''
    _global_memory['task_completed'] = False
        
    return {
        "work_log": work_log,
        "key_facts": get_memory_value("key_facts"),
        "related_files": get_related_files(),
        "key_snippets": get_memory_value("key_snippets"),
        "completion_message": completion_message,
        "success": success,
        "reason": reason
    }

@tool("request_implementation")
def request_implementation(task_spec: str) -> Dict[str, Any]:
    """
    Spawn a planning agent to create an implementation plan for a given task specification.
    
    This function initializes a language model, runs a planning agent to generate an implementation strategy, and handles various execution scenarios including successful completion, user interruption, and unexpected errors.
    
    Parameters:
        task_spec (str): A detailed specification of the task requiring an implementation plan.
    
    Returns:
        Dict[str, Any]: A comprehensive result dictionary containing:
            - work_log (Optional[List]): Detailed log of work performed during task planning
            - completion_message (Optional[str]): Message describing task completion status
            - key_facts (List): Important facts extracted during the planning process
            - related_files (List): Files potentially relevant to the implementation
            - key_snippets (List): Code or text snippets of significance
            - success (bool): Indicates whether the planning process completed successfully
            - reason (Optional[str]): Explanation for task failure or interruption
    
    Raises:
        KeyboardInterrupt: If the user forcefully terminates the planning process
        Exception: For any unexpected errors during planning
    
    Example:
        result = request_implementation("Create a web scraper for news articles")
        if result['success']:
            print(result['completion_message'])
            print("Related files:", result['related_files'])
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    try:
        # Run planning agent
        from ..agent_utils import run_planning_agent
        result = run_planning_agent(
            task_spec,
            model,
            config=config,
            expert_enabled=True,
            hil=config.get('hil', False)
        )
        
        success = True
        reason = None
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print_error(f"Error during planning: {str(e)}")
        success = False
        reason = f"error: {str(e)}"
        
    # Get completion message if available
    completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
    
    # Get and reset work log if at root depth
    current_depth = _global_memory.get('agent_depth', 0)
    work_log = get_work_log() if current_depth == 1 else None
    if current_depth == 1:
        reset_work_log()
    
    # Clear completion state from global memory
    _global_memory['completion_message'] = ''
    _global_memory['task_completed'] = False
    _global_memory['plan_completed'] = False
        
    return {
        "work_log": work_log,
        "completion_message": completion_message,
        "key_facts": get_memory_value("key_facts"),
        "related_files": get_related_files(),
        "key_snippets": get_memory_value("key_snippets"),
        "success": success,
        "reason": reason
    }

@tool("authorize_task_completion")
def authorize_task_completion(task_output: str) -> bool:
    """
    Authorize task completion using an AgentSupervisor.
    
    Validates the output of a task against predefined criteria to determine if the task can be considered complete.
    
    Args:
        task_output (str): The output or result of the task requiring authorization.
    
    Returns:
        bool: True if the task meets completion criteria, False otherwise.
    
    Raises:
        ValueError: If configuration or checklist loading fails.
        RuntimeError: If supervisor authorization process encounters an error.
    
    Notes:
        - Uses global configuration to initialize supervisor model
        - Loads checklist from specified path
        - Relies on AgentSupervisor's authorization mechanism
    """
    # Initialize supervisor
    config = _global_memory.get('config', {})
    supervisor_model = initialize_llm(config.get('supervisor_provider', 'anthropic'), config.get('supervisor_model', 'claude-3-5-sonnet-20241022'))
    supervisor = AgentSupervisor(model=supervisor_model, max_turns=config.get('supervisor_max_turns', 10))
    
    # Load checklist
    checklist_path = config.get('checklist_path', 'checklist.txt')
    supervisor.load_checklist(checklist_path)
    
    # Authorize task completion
    return supervisor.authorize_task_completion(task_output)
