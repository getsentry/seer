import textwrap

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import GeminiProvider
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.prompts import format_repo_prompt
from seer.automation.autofix.tools.tools import SemanticSearchTools
from seer.automation.codegen.codegen_context import CodegenContext


def semantic_search(query: str, context: AutofixContext | CodegenContext) -> str:
    """
    Uses an LLM agent to search the codebase and return relevant files and insights.

    Args:
        query: The search query to find relevant code
        context: The context object (either AutofixContext or CodegenContext)

    Returns:
        A string containing the agent's response with relevant files and insights
    """
    tools = SemanticSearchTools(context=context)

    agent_config = AgentConfig(interactive=False)
    agent = LlmAgent(
        config=agent_config,
        tools=tools.get_tools(can_access_repos=True),
        name="SemanticSearchAgent",
    )

    state = context.state.get()
    readable_repos = state.readable_repos
    repo_str = format_repo_prompt(
        readable_repos=readable_repos, unreadable_repos=state.unreadable_repos
    )

    initial_tree_str = "No directory information available."
    tree_outputs = []
    for repo in readable_repos:
        tree_output = tools.tree(
            path="", repo_name=repo.full_name
        )  # get a top level tree for the repo
        tree_outputs.append(f"Tree for repository '{repo.full_name}':\n{tree_output}")
    if tree_outputs:
        initial_tree_str = "\n\n".join(tree_outputs)

    system_prompt = textwrap.dedent(
        """\
        You are an exceptional codebase search agent. Your goal is to find the most relevant files in the codebase that match the user's query.

        Use the available tools to explore the codebase tree structure, and when necessary, to read the contents of specific files.

        {repo_str}

        Here is the initial directory structure for the accessible repositories:
        {initial_tree_str}

        In your final response, list the relevant files and their full paths, and explain why they are relevant to the query."""
    ).format(repo_str=repo_str, initial_tree_str=initial_tree_str)

    run_config = RunConfig(
        system_prompt=system_prompt,
        prompt=query,
        model=GeminiProvider(model_name="gemini-2.0-flash-001"),
        temperature=0.0,
        max_iterations=32,
    )

    return agent.run(run_config)
