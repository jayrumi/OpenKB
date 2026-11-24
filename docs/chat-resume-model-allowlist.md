# Deployment-specific resumed-chat model allowlist

A saved chat retains its original model ID. Changing `query_model` does not
migrate that chat or its history. During a provider migration, an operator can
opt in to refusing resumes using unapproved model IDs:

```sh
export OPENKB_CHAT_ALLOWED_MODELS='["openai/responses/gpt-6-sol"]'
```

Set this environment variable in the API deployment and, if used, the CLI
process. It is a JSON array of exact provider/model strings, not prefixes or
wildcards. Values are not trimmed, rewritten, or silently upgraded. The code
contains no hardcoded model-family policy.

- **Unset:** existing resume behavior is retained, including legacy model IDs.
- **Empty array `[]`:** all resumes are refused.
- **Malformed JSON, wrong shape, or blank/non-string entries:** fail closed
  with a configuration error naming the variable.
- **Disallowed saved model:** REST `/api/v1/chat` returns HTTP 409 for both JSON
  and streaming requests, before resolving credentials, constructing run config,
  creating an agent, appending logs, or starting SSE/tool/model work. CLI
  `openkb chat --resume` reports an error before key setup or entering the REPL.

Refusal tells the user to start a new chat. Saved chat JSON is unchanged and
history remains available through session list/load. No automatic migration or
replay occurs. New chats continue to use the effective `query_model` (falling
back to `model`); configure that stage separately to the approved migration
model. This guard does not change indexing, compilation, or one-off queries,
and is not a substitute for provider-side admission controls.

This development change does not set the variable, edit KB/global settings,
restart services, or deploy anything.
