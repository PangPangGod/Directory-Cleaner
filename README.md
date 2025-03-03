# Directory Cleaner with Agent

for testing LLM Tools / Agent

25-03-02: Initial Commit
---

25-03-03:
- Update MessageState to Pydantic Format CleanDirectoryState.
- Edit @tool planner args to use Command with InjectedState and InjectedToolCallID.
- Edit @tool code_executor args plans(str) -> plans(Annotated[str, InjectedState("plans")])
(WIP) -> Need to edit @tool code_executor to write code -> test -> execute with HumanInterrupt