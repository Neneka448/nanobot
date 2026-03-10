---
name: memory
description: Workspace memory files for durable knowledge and searchable history.
always: true
---

# Memory

## Structure

Follow the memory file locations listed in the system prompt for the current workspace.

Depending on the active memory implementation, the workspace may contain:

- Long-term memory files that are loaded into context
- Short-term memory files used for staged consolidation
- History logs that are searchable but not automatically loaded into context

## Search Past Events

Choose the search method based on file size and the history file used by the workspace:

- Small history files: use `read_file`, then search in-memory
- Large or long-lived history files: use the `exec` tool for targeted search

Use the specific history file path from the system prompt when constructing commands.

Prefer targeted command-line search for large history files.

## How to Use Memory Files

- Write stable, reusable knowledge only to long-term memory files.
- Treat history files as searchable archives for prior events and decisions.
- If the workspace has a short-term memory file, it is usually maintained automatically by the consolidation pipeline unless the user explicitly asks you to edit it.

## Auto-consolidation

Older conversations may be consolidated into the workspace memory files automatically. Use the concrete file locations exposed in the system prompt for the active memory implementation.
