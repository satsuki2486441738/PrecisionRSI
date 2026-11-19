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
  Ask the user to confirm the target, task and validation scope and authorize
  the workflow. Once confirmed and authorized, hand off the request to community
  retrieval. Execution details can remain pending during retrieval.
</authorization>
```

Continue with [community retrieval](workflow.md#route-community-results). See the
[native execution requirements](workflow.md#request-and-execution-inputs) before launching.
