# OOLONG

This environment has two execution engines:

- `engine="train"` (default) uses `RLMTrainEnv` for Qwen/Prime-RL training and
  preserves the original depth-1 behavior.
- `engine="core"` runs each example through the core `rlm.RLM`, including real
  recursive child RLMs with independent REPLs.

## Recursive evaluation

Set the API variables before starting `vf-eval`:

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://your-openai-compatible-endpoint/v1'
```

Run a small depth-2 check first:

```bash
uv run vf-eval oolong \
  -p openai \
  -m 'gpt-5.5' \
  -b "$OPENAI_BASE_URL" \
  -k OPENAI_API_KEY \
  -n 5 \
  -r 1 \
  -s \
  --state-columns rlm_metadata,rlm_usage_summary \
  -a '{"engine":"core","dataset_name":"trec_coarse","min_ctx":1024,"max_ctx":4096,"num_examples":5,"max_iterations":12,"max_depth":2,"max_concurrent_subcalls":4,"require_recursive_subcall":true,"sub_model":"gpt-5-mini","sub_max_tokens":4096}'
```

`max_depth=2` means the root can create one full child RLM level. Use
`max_depth=3` to allow a child RLM to create another full child RLM. Calls at
the depth limit become plain LM completions.

For the complete filtered split, set both CLI `-n -1` and environment
`"num_examples":-1`. Start with low outer concurrency because every rollout
can create several child RLMs:

```bash
uv run vf-eval oolong \
  -p openai \
  -m 'gpt-5.5' \
  -b "$OPENAI_BASE_URL" \
  -k OPENAI_API_KEY \
  -n -1 \
  -r 1 \
  -c 2 \
  -s \
  --state-columns rlm_usage_summary \
  -a '{"engine":"core","dataset_name":"trec_coarse","min_ctx":1024,"max_ctx":4096,"num_examples":-1,"max_iterations":12,"max_depth":2,"max_concurrent_subcalls":4,"require_recursive_subcall":true,"sub_model":"gpt-5-mini","sub_max_tokens":4096}'
```

The aggregate output includes `rlm_recursive_subcalls`,
`rlm_max_depth_reached`, `rlm_total_iterations`, and `rlm_total_repl_calls`.
With `require_recursive_subcall=true`, a successful depth-2 rollout should report
`rlm_recursive_subcalls >= 1` and `rlm_max_depth_reached >= 1`.
The CLI `--model` value configures the root RLM; `sub_model` configures complete
child RLMs, plain `llm_query()` calls, and leaf calls at the depth limit.
