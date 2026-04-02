<h1 align='center'>myautoresearch</h1>

My take on autoresearch. You shouldn't use it yet because I need to like produce some examples and make readme better etc...

## How it works

Use AI agents to search for algorithms or solutions to other measurable coding problems.

The agent runs `mar start` which outputs a summary of previously submitted solutions and instructions on how to evaluate and submit new solutions. The agent will create a Python file and define a solution in it. Then it will run:

```shell
mar evaluate --file filename.py --object object_in_file --name run_name --description "run description"
```

The agent will see the metrics and a leaderboard comparing the solution to other solutions. It will iterate on the solution until it can't improve it anymore. It will then submit the best solution:

```shell
mar submit --name run_name --result "description and results of the experiments"
```

Only submitted solutions are shown in `mar start`. This prevents spamming context with similar versions of the same algorithm. `mar start` also moves all unsubmitted solutions from the previous session to discarded, this means the agent only sees its attempts and all submitted runs in the leaderboard.

## How to use

### 1. Initialize a project

First you need to prepare a folder for the experiments. In it, run `mar init` - this will set up the project structure:

```text
project/
├── workdir/              # Working directory
├── runs/
│   ├── unsubmitted/      # Runs not yet submitted
│   ├── submitted/        # Submitted runs (appear in summary and leaderboard)
│   └── discarded/        # Discarded runs (kept for reference)
├── templates/
│   ├── workdir/          # Template files copied to working directory on `mar start`
│   └── eval/             # Template files copied to evaluation run directories
├── scripts/
│   ├── initialize.py     # Empty initialization script (runs on `mar start`)
│   └── evaluate.py       # Evaluation script template
├── task.md               # Problem description for the AI agent
└── config.yaml           # Configuration file
```

Then do `cd workdir`. All other `mar` commands must be run from the working directory.

### 2. Configure the project

Edit `config.yaml` to set your preferences:

```yaml
# Name of the working directory. Default is "workdir".
# This is configurable since it impacts model context.
work_dir: "workdir"

# Optionally specify name of the model, it will be shown in summaries.
author: "Qwen3.5" # default: null

# Maximum runtime in seconds for a feasible run
max_time: 60 # default: null

# Timeout for evaluation (hard limit). Setting to twice the max_time is good
# to allow the agent to see the metrics even if run is infeasible.
timeout: 120 # default: null

# Number of top runs to show in leaderboard.
top_k: 10 # default: 10

# If a run is not in top_k runs, it will show it at the bottom of the leaderboard
# along with this many neighboring runs by rank
n_neighbors: 2 # default: 2

# if True, if any per-step metrics are logged, the logs will be copied to working directory
# and a short instruction will be displayed on how to inspect them after each evaluation.
copy_logger: false # default: false
```

### 3. Define evaluation script

The `scripts/evaluate.py` file defines how algorithms are evaluated. Edit this file to match your problem.

First, decide on what the agent should submit as the solution. Usually it's going to be a class, an instantiated class, or a function, although it may be something else like a string or a numpy array depending on the problem. If applicable, it may be a good idea to write down a stub for it.

Then you can define the `run` method on the evaluator, where the object submitted by the AI agent will be in the `object` attribute:

```python
from click import echo # use instead of print
from myautoresearch import Evaluator, run

class MyEvaluator(Evaluator):
    def evaluate(self):
        # self.object is the object imported from the submitted file
        algorithm = self.object

        ... # evaluate it

        # Log step-by-step metrics (optional)
        for step, value in enumerate(history):
            self.log_step(step, "train loss", value)

        # Log final metrics
        self.log_final(
            metric="accuracy",
            value=accuracy,
            maximize=True, # Whether higher is better
        )

        # Mark as infeasible if needed
        if violates_constraints:
            self.set_infeasible("Constraint violation reason")

if __name__ == "__main__": # don't modify this part
    run(MyEvaluator())
```

The evaluator has three main methods useful inside `run`:

```python
class Evaluator:
    def log_step(self, step: int, metric: str, value: Any) -> None:
        """Log an intermediate numeric metric, like train loss. If any are logged, and
        ``copy_logger`` configuration optin is enabled, the logger will be saved
        to working directory, and a short instruction for the agent will be displayed
        after evaluating a run on how to inspect it."""

    def log_final(
        self,
        metric: str,  # name of the metric
        value: Any,  # value
        maximize: bool | None,
        is_main: bool = True,
        display_value: bool = True,
        display_rank: bool = True,
        display_leaderboard: bool = True,
        display_summary: bool = True,
        weight: float = 1.0,
    ) -> None:
        """Log a final metric. At least one main metric must be logged so that solutions can be compared.

        Args:
            metric: Name of the metric.
            value: Value of the metric.
            maximize: `True` if higher is better, `False` if lower is better, `None` if not numeric.
            is_main: At least one metric should be main to rank the runs.
                If there are multiple main metrics, an average rank is computed from their ranks.
                But it is usually a good idea to manually design a formula to compute a final score,
                and use that score as the only main metric. Defaults to True.
            display_value: Show this metric after a run is evaluated.
                In most cases it is fine to set this to True on all metrics. Defaults to True.
            display_rank: Show this metric rank and name of best run by this metric after a run is evaluated.
                Defaults to True.
            display_leaderboard: Show this metric for all other runs in the leaderboard after a run is evaluated.
                Keep the number of metrics in the leaderboard under 4 to make it more readable. Defaults to True.
            display_summary: Show this metric for all submited runs in the summary shown when agent runs `mar start`.
                Keep the number of metrics in the summary under 10 to avoid filling the context when number of submissions is large.
            weight: This metric's weight for computing average rank from main metrics. Defaults to 1.0.
        """

    def set_infeasible(self, reason: str) -> None:
        """Mark this run as infeasible and specify a reason that the AI agent will see."""
```

If you need any extra files for the evaluation, put them in the `templates/eval` directory. They will be copied next to the evaluation script when evaluating.

Tip: If you modify the evaluation script, you can run `mar reevaluate` to rerun all submitted runs.

### 4. Write task.md

`task.md` defines the description of the problem, and its contents will be shown to the AI agent when it runs `mar start`. It is recommended to include at least three sections: Task, API and Evaluation. Make sure you specify what the agent should submit exactly. If applicable, put a stub in the API section so that it knows the exact interface it should follow.

### 5. Test your setup

Run `mar start` to check what the agent is going to see when it runs that command. This also displays an instruction on how to evaluate and submit runs, which you can use to submit some baseline runs.

### 6. Submit baseline runs

It is a good idea to submit a few baseline runs before running the AI agent. Pass `--baseline` flag to `mar evaluate` to mark a submission as baseline (note that this flag is hidden from --help to hide it from AI agent). While doing that, you can also check what the agent is going to see when it runs `mar evaluate`, and possibly tune your evaluation script to improve the display, or to make the scoring better. Don't forget to run `mar reevaluate` if you make any changes to evaluation script.

### 7. Run the agent

You can use this prompt template:

```text
Your goal will be to develop <describe the task>. Run `mar start` in the shell and follow the instructions.
```

This prompt is usually sufficient as `mar start` outputs detailed instructions on how to do everything else.

If running in a loop, you can use modifiers:

- **explore**: The goal of this session is exploration. Instead of incremental modifications to existing solutions from the leaderboard, you must try approaches that are not explored or under-explored in the leaderboard.
- **exploit**: The goal of this session is exploitation - you must analyze the leaderboard and focus on the most promising approaches.
- **novel**: The goal of this session is to explore novel approaches. Instead of trying known solutions, you must design your own algorithm from scratch. It should be new, not a modification of an existing solution.
- **analyse**: The goal of this session is to perform a deep analysis of the problem. You must analyze the problem thoroughly before designing a solution, find new approaches that could be missed by tackling the problem head-on.
- **unusual**: The goal of this session is to explore wild and unusual solutions. You must design and evaluate the most unconventional solutions you can come up with.
- **research**: The goal of this session is research. You must use your web tools to search for latest and most efficient algorithms - look up studies, benchmarks, implementations, and evaluate the most promising approaches.
- **inspect**: The goal of this session is inspection. You must focus on manually evaluating and debugging solutions  through comprehensive testing using custom evaluation scripts in order to identify and bottlenecks and areas of improvement. Analyze best solutions in the leaderboard as well as your ideas. Compare your results with `mar evaluate` to see how your evaluation methodology maps onto the final score. Design better solutions based on your comprehensive analysis.

Put the name of the modifier after `mar start` in the prompt, for example `mar start explore`. Alternating between modifiers can improve the diversity of the solutions.

## Other info

### Additional Commands

- `mar reevaluate`: Rerun all submitted runs with the current evaluation script. Use after modifying `scripts/evaluate.py`.
- `mar load <name>`: Allows agent to load the source code of a previous run into the working directory.
- `mar list <status>`: List run names by status (`unsubmitted`, `submitted`, `discarded`, or `all`).
- `mar leaderboard <status>`: Display the leaderboard for runs with specified status.
- `mar discard <name>`: Move a run to discarded.
- `mar rename <old> <new>`: Rename a run.
- `mar summary`: Show summaries of submitted runs with their descriptions.
- `mar config --<option> <value>`: Update configuration options from CLI.
- `mar --help`: Show help for any command (e.g., `mar evaluate --help`).

### Error Handling

- **Exception**: Exceptions are handled automatically. If the evaluation script raises an exception, the run is deleted, and the agent sees the stack trace.
- **Timeout**: If runtime exceeds `timeout` (from config), the run is deleted.
- **Max time exceeded**:  If runtime exceeds `max_time` (from config), the run is marked infeasible but kept.

Infeasible runs are still useful to submit as they document what approaches don't work.

### Working Directory Requirement

All `mar` commands (except `mar init`) must be run from the working directory. If you run a command from the wrong directory, you'll get an error message indicating that you need to change to the working directory.

## General tips

- Make sure evaluation is deterministic - use same random seed, etc.
- If timing algorithms, perform a few warmup iterations first.
