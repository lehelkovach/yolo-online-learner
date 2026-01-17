## Debugging & refactoring guide (agent-friendly)

### Debugging order (fastest signal first)

1) **Reproducibility**
   - fix the seed
   - log the config
   - ensure runs are comparable

2) **Data validity**
   - BBP boxes in-bounds
   - confidences in [0,1]
   - frame timestamps monotonic

3) **Bottleneck correctness**
   - ensure attention truly selects a tiny set (ideally 1)
   - confirm inhibition-of-return prevents fixation lock

4) **Stability**
   - prototype count bounded
   - graph edge count bounded
   - learning rates gated by novelty/error

### What to log every stage (minimum)

- **frame**: frame_idx, timestamp
- **attention**: selected BBP/prototype id
- **gaze/fovea**: center, jitter, fovea bbox
- **prediction**: predicted embedding + error
- **novelty/gain**: surprise, habituation state
- **memory**: WM contents + evictions
- **graph**: node/edge counts + prune events

### When to refactor

Refactor only when a stage is stable and tested:

- **Interface refactor**: when 2+ modules share a pattern (e.g., `step()` methods)
- **Performance refactor**: when logs show dropped frames or runaway memory
- **Correctness refactor**: when a metric is ambiguous or non-identifiable

### Where to refactor (safe seams)

- `perception/`: keep pure I/O (frames, detections, embeddings extraction)
- `attention/`: keep selection + foveation + WM (no learning rules here)
- `features/` and `objects/`: learning rules and prototypes (pure state updates)
- `graph/`: memory substrate + decay/prune
- `experiments/`: logging, config, metrics (no learning logic)

### How to use an agent safely

Ask the agent to make changes in “one PR = one mechanism” units:

- Add module + tests + metrics logging in the same PR
- Provide a minimal example run command
- Avoid multi-stage “big bang” PRs

