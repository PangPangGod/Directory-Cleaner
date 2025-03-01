from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic

# 1. 디렉토리 받아서 tool로 디렉토리 접근하는거 (tool)
# 2. 디렉토리 분석해서 plan 짜는거 (tool)
# 3. python code 생성하는거 (tool)

from langchain_core.tools import tool
import os
from pathlib import Path


@tool
def traverse_directory(
    directory: str | Path, max_depth: int = 1, current_depth: int = 1
) -> str:
    """Search Directory recursivly and return output."""
    lines = []
    indent = " " * 4 * (current_depth - 1)
    # 현재 디렉토리 이름 추가
    lines.append(f"{indent}{os.path.basename(directory)}/")

    try:
        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                # 현재 깊이가 max_depth 이상이면, 재귀 호출 없이 디렉토리 이름만 추가
                if current_depth >= max_depth:
                    lines.append(f"{indent}    {entry}/")
                else:
                    # 재귀적으로 하위 디렉토리 탐색, 결과를 리스트에 extend
                    lines.extend(
                        traverse_directory(
                            path, max_depth, current_depth + 1
                        ).splitlines()
                    )
            else:
                lines.append(f"{indent}    {entry}")
    except PermissionError:
        lines.append(f"{indent}    [접근 권한 없음]")

    return "\n".join(lines)


model = ChatAnthropic(model="claude-3-5-haiku-latest")
tools = [traverse_directory]
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
                ("human", r"<Directory Name> -> 이 디렉토리에 뭐 있는지 보여줘.")
            ]
        },
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
