# Confirm the request

```xml
<request>
  In the user's language, summarize the target hardware, model, tested framework,
  task and validation scope. For a general check, propose a small end-to-end case
  that exercises the requested workflow.
  Record known execution details for the target and comparison side: local or
  remote, host or queue, launch method, device limits and budget for initial
  runs, focused recaptures and verification. Ask early which comparison platform
  can be accessed for follow-up runs. Mark unresolved details as pending.
  Record any supplied acceptance metrics, scope and tolerances. Leave acceptance
  unset if none were supplied. Default optimize_precision to off. When enabled,
  confirm the metric to improve and acceptable cost.
</request>
```

```xml
<authorization>
  Ask once to confirm the target, validation scope and execution limits for the
  whole requested workflow, including in-scope preparation, runs and follow-up
  checks. Keep execution details the agent can discover pending. After approval,
  start community retrieval and carry the agreed scope and limits forward.
</authorization>
```

Continue with [community retrieval](workflow.md#route-community-results). See the
[native execution requirements](workflow.md#request-and-execution-inputs) before launching.
