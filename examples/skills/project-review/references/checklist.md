# Project review checklist

- What starts the application, and who owns each long-lived resource?
- Does each session keep its own working directory and state?
- Can a task be cancelled while waiting for a tool or user approval?
- Do errors reach the caller and remain visible to the user?
- Are declared parameters used by the execution path?
- Do permissions apply to subagents and external tools?
- Which behavior has automated coverage?
