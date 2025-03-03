from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage

import os
from pathlib import Path

from pydantic import BaseModel, Field
from typing import List, Annotated

# 1. 디렉토리 받아서 tool로 디렉토리 접근하는거 (tool)
# 2. 디렉토리 분석해서 plan 짜는거 (tool)
# 3. python code 생성하는거 (tool)


# Helper function
def _traverse_directory(
    directory: str | Path, max_depth: int, current_depth: int
) -> str:
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
                        _traverse_directory(
                            path, max_depth, current_depth + 1
                        ).splitlines()
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
def planner(
    messages: Annotated[list, InjectedState("messages")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Directory Planner Tool

    This function takes an input messages containing detailed descriptions of directories (e.g., directory names and their contents)
    and generates a plan on how to organize and use these directories. It categorizes the directories into different groups, such as
    "Projects" and "Data/Reports", and returns a summary of the planned organization.

    Returns:
    --------
    str
        A summary of the directory organization plan.
        Example output: "Projects: aura-activity-review, auto_talk, ... / Data: db, done, ref, ..."
    """

    print("--[DEBUG] Messages Test--")
    prettify_message = prettify_messages(messages)
    print(prettify_message)

    prompts = ChatPromptTemplate(
        [
            (
                "system",
                """You are an expert in file system organization and directory automation.  
Your task is to analyze the structure of a desktop directory and create a **practical reorganization plan**.  

Your output should focus on:  
- A **clear, logical directory structure** that improves accessibility.  
- A **general mapping** of existing files and folders to their new locations.  
- A **set of essential actions** needed to implement the new organization.  

Ensure the plan is **straightforward, automation-friendly, and avoids unnecessary complexity**.  
Do not over-categorize or introduce deep hierarchies unless strictly necessary.""",
            ),
            (
                "human",
                """Below is the previous conversations of current desktop directory structure and file list:
[previous conversations]
{message}

[Instructions]
Based on previous conversations, generate an **efficient reorganization plan** that includes:
- A **simplified directory structure** with logical groupings.
- A **concise mapping** of existing files and folders to their new locations.
- A **list of necessary actions** to create directories and move files.

Keep the plan **clear and practical**. Avoid unnecessary complexity or excessive categorization.""",
            ),
        ]
    )
    response = model.invoke(prompts.format(message=prettify_message))

    # Update Messages & save plans on current State
    return Command(
        update={
            "messages": [
                ToolMessage(content=response.content, tool_call_id=tool_call_id)
            ],
            "plans": response.content,
        }
    )


def prettify_messages(messages: List[BaseMessage]) -> str:
    """Messages 전달할 때 Token 전달용으로 Parse하는 Helper Function."""
    pretty_printed = []

    for message in messages:
        text = (
            message.content[0].get("text")
            if hasattr(message, "tool_calls")
            else message.content
        )
        pretty_printed.append(text)
        pretty_printed.append("=====")
    return "\n".join(pretty_printed)


## 실제 코드 만들고 실행까지 해야 함.
@tool
def code_executor(directory: str, plans: Annotated[str, InjectedState("plans")]) -> str:
    """
    Generate Python code solution based on plans.

    Args:
        directory (str): Directory That will be executed.
        plans (str): plans generated.
    Returns:
        Code (str)
    """

    print("---GENERATING CODE SOLUTION---")

    thinking_model = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        max_tokens=10000,
        thinking={"type": "enabled", "budget_tokens": 5000},
    )

    prompts = ChatPromptTemplate(
        [
            (
                "system",
                "You are an expert in generating code for automating directory restructuring tasks. Your job is to generate a complete, executable Python script that, given a base directory and a detailed organization plan, creates the new directory structure and moves the specified directories/files into their respective locations. The script must include all necessary imports, error handling, and clear comments explaining each step.",
            ),
            (
                "human",
                "Below is the base directory and the detailed organization plan:\n\nDirectory: {directory}\n\nOrganization Plan:\n{plans}\n\nPlease generate a complete Python script that automates the creation of the new directories and moves existing folders/files as specified in the plan. Ensure the script is ready to run and handles potential errors gracefully.",
            ),
        ]
    )
    response = thinking_model.invoke(prompts.format(directory=directory, plans=plans))
    return response.content


model = ChatAnthropic(model="claude-3-5-haiku-latest")
tools = [traverse_directory, planner, code_executor]
tool_node = ToolNode(tools)
llm_with_tools = model.bind_tools(tools)


## Define States
class CleanDirectoryState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    plans: str = Field(default=None, description="Genarated Directory Cleaning Plan.")


def call_llm(state: CleanDirectoryState):
    response = llm_with_tools.invoke(state.messages)
    return {"messages": [response]}


def should_continue(state: CleanDirectoryState):
    messages = state.messages
    last_message = messages[-1]

    if last_message.tool_calls:
        print("[DEBUG] THIS IS TOOL CALL")
        print(f"[DEBUG] {last_message}")

        return Command(
            goto="tools",
        )

    return Command(goto=END)


workflow = StateGraph(CleanDirectoryState)

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
                (
                    "human",
                    r"()<- 해당 디렉토리 정리하고 싶은데 확인해서 계획 세우고 코드까지 만들어줘.",
                )
            ]
        },
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
