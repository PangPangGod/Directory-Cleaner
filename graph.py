from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import os
from pathlib import Path

# 1. 디렉토리 받아서 tool로 디렉토리 접근하는거 (tool)
# 2. 디렉토리 분석해서 plan 짜는거 (tool)
# 3. python code 생성하는거 (tool)

def _traverse_directory(directory: str | Path, max_depth: int, current_depth: int) -> str:
    """Internal helper function: Recursively traverses a directory and returns a string representation of its tree structure."""
    lines = []
    indent = " " * 4 * (current_depth - 1)
    lines.append(f"{indent}{os.path.basename(directory)}/")
    
    try:
        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                if current_depth >= max_depth:
                    lines.append(f"{indent}    {entry}/")
                else:
                    lines.extend(
                        _traverse_directory(path, max_depth, current_depth + 1).splitlines()
                    )
            else:
                lines.append(f"{indent}    {entry}")
    except PermissionError:
        lines.append(f"{indent}    [접근 권한 없음]")
    
    return "\n".join(lines)

@tool
def traverse_directory(directory: str | Path, max_depth: int = 2) -> str:
    """Recursively traverses a specified directory and returns a structured text output representing the directory tree.
    This function starts at the given directory and explores its contents.
    The resulting output is a neatly indented string that visually represents the folder and file hierarchy.
    """
    # 기본적으로 @tool decorator 이용할 시 recursive하게 call 하는거 막아둠. 그래서 우회해서 helper function 만들고 사용.
    return _traverse_directory(directory, max_depth, current_depth=1)

@tool
def planner(message: str) -> str:
    """
    Directory Planner Tool

    This function takes an input string containing detailed descriptions of directories (e.g., directory names and their contents)
    and generates a plan on how to organize and use these directories. It categorizes the directories into different groups, such as
    "Projects" and "Data/Reports", and returns a summary of the planned organization.

    Parameters:
    -----------
    message : str
        A detailed description of directories and files. This message can include names and descriptions of various folders.

    Returns:
    --------
    str
        A summary of the directory organization plan. 
        Example output: "Projects: aura-activity-review, auto_talk, ... / Data: db, done, ref, ..."
    """

    prompts = ChatPromptTemplate([
        (
            "system", 
            "You are an expert in organizing file systems and automating directory restructuring. Your task is to analyze a detailed description of current desktop directories and files, and then produce a comprehensive, structured organization plan. The plan should clearly map each folder to a specific category, provide a precise new directory structure (including categories and subcategories such as 'Projects', 'Research & Data', 'Personal Documents', etc.), and list the specific actions needed (like creating new directories and moving existing ones). Your output should be detailed enough to directly inform an automated tool for directory creation and file movement, and include any recommendations for further improvements (such as archiving or cleanup scripts)."
        ),
        (
            "human", 
            "Below is the detailed description of the current desktop directories and files:\n\n{message}\n\nBased on this description, generate a detailed organization plan that clearly specifies:\n- The proposed new directory structure with clear categories and subcategories\n- The mapping of each existing folder to its designated new location\n- A list of actions required to create the new folders and move the existing directories accordingly\n- Any additional recommendations for improving the overall organization and maintenance of the desktop files\nEnsure that your plan is structured, clear, and comprehensive, so it can be used as a direct guide for automating the directory reorganization process."
        )
    ])
    response = model.invoke(prompts.format(message=message))
    return response.content

@tool
def code_executor(directory: str, plans: str) -> str:
    """
    Generate a code solution

    Args:
        directory (str): Directory That will be executed.
        plans (str): plans that are generated.
    Returns:
        Code str
    """

    print("---GENERATING CODE SOLUTION---")

    thinking_model = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        max_tokens=10000,
        thinking={"type": "enabled", "budget_tokens": 5000}
    )

    prompts = ChatPromptTemplate([
        (
            "system", 
            "You are an expert in generating code for automating directory restructuring tasks. Your job is to generate a complete, executable Python script that, given a base directory and a detailed organization plan, creates the new directory structure and moves the specified directories/files into their respective locations. The script must include all necessary imports, error handling, and clear comments explaining each step."
        ),
        (
            "human",
            "Below is the base directory and the detailed organization plan:\n\nDirectory: {directory}\n\nOrganization Plan:\n{plans}\n\nPlease generate a complete Python script that automates the creation of the new directories and moves existing folders/files as specified in the plan. Ensure the script is ready to run and handles potential errors gracefully."
        )
    ])
    response = thinking_model.invoke(prompts.format(directory=directory, plans=plans))
    return response.content

model = ChatAnthropic(model="claude-3-5-haiku-latest")
tools = [traverse_directory, planner, code_executor]
tool_node = ToolNode(tools)
llm_with_tools = model.bind_tools(tools)


def call_llm(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        print("[DEBUG] THIS IS TOOL CALL")
        print(f"[DEBUG] {last_message}")

        return Command(
            goto="tools",
        )

    return Command(goto=END)


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_llm)
workflow.add_node("route", should_continue)
workflow.add_node("tools", tool_node)

## route 거치되 tools 분리하던지 아니면 command를 새로 짜던지.. 근데 복잡한 구조 줄이려면 route...
workflow.add_edge(START, "agent")
workflow.add_edge("agent", "route")
workflow.add_edge("tools", "agent")

app = workflow.compile()

if __name__ == "__main__":
    for chunk in app.stream(
        {
            "messages": [
                ("human", r"()<- 해당 디렉토리 정리하고 싶어!")
            ]
        },
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
