from typing import Literal

MAR_INSTRUCTION = """# Instructions

## Evaluating a run

To evaluate a run, use the `mar evaluate` shell command. Pass the following flags:

--file TEXT: name of the python file to submit for evaluation, e.g. "algorithm.py".
--object TEXT: name of the item (class, object, variable) that will be imported from specified file and evaluated. The object is imported as `from <file> import <object>`.
--name TEXT: unique descriptive name for this run.
--description TEXT: describe your algorithm. The description should be concise but detailed, it needs to contain all information necessary to recreate the run.

Useful optional flags for `mar evaluate`:
--extra-files file1.py file2.py: Include additional files needed for evaluation (e.g., helper modules, data files).
--overwrite: Overwrite an existing unsubmitted run with the same name.

After running the command, the results of evaluation will be displayed in the terminal. You can also display results manually by running `mar leaderboard`. Keep improving your algorithm and try to beat the current best run, and pave the way for future sessions to get even better results.

## Submitting a run

Finally, submit one run using `mar submit` shell command. Sumbit the run that will be the most useful for future sessions. This can be your best attempt, or the most promising approach you've tried. If all your runs failed, you should still submit your best attempt to document what doesn't work. Pass the following flags:

--name TEXT: the same name as you passed to `mar evaluate`. You can list all names via `mar list unsubmitted`.
--result TEXT: describe results of your experiments - what did you try, what worked, what didn't work, did your best attempt beat current leader, can it be improved. Include any new information that will help future sessions. The summary already shows all metric values, don't duplicate them here.

Only submit one run per session. If you've changed your mind and want to submit a different run, use `mar discard <name>` to discard the previously submitted one.
"""

AFTER_SUMMARY_RUNS = "If you'd like to see submitted runs again, you can use `mar summary` command. You can also load the source code of any run using `mar load <name>`, but use it only if necessary."

ModifierLiteral = Literal["explore", "exploit", "novel", "analyse", "unusual", "research", "inspect"]

MODIFIER_INSTRUCTION = "This session was started with `{modifier_name}` instruction. You must follow this instruction very closely as it will help future sessions:\n{modifier}"

MODIFIERS: dict[ModifierLiteral, str] = {
    "explore": "The goal of this session is exploration. Instead of incremental modifications to existing solutions from the leaderboard, you must try approaches that have not been explored yet. Your main goal is not to beat the leader, but to thorougly explore the problem's search space, try many diverse approaches and submit the most promising one.",
    "exploit": "The goal of this session is exploitation - you must analyze the leaderboard and focus on the most promising approaches.",
    "novel": "The goal of this session is to explore novel approaches. Instead of trying known solutions, you must design your own algorithm from scratch. It should be new, not a modification of an existing solution.",
    "analyse": "The goal of this session is to perform a deep analysis of the problem. You must analyze the problem thoroughly before designing a solution, find new approaches that could be missed by tackling the problem head-on.",
    "unusual": "The goal of this session is to explore wild and unusual solutions. You must design and evaluate the most unconventional solutions you can come up with.",
    "research": "The goal of this session is research. You must use your web tools to search for latest and most efficient algorithms - look up studies, benchmarks, implementations, and evaluate the most promising approaches.",
    "inspect": "The goal of this session is inspection. You must focus on manually evaluating and debugging solutions  through comprehensive testing using custom evaluation scripts in order to identify and bottlenecks and areas of improvement. Analyze best solutions in the leaderboard as well as your ideas. Compare your results with `mar evaluate` to see how your evaluation methodology maps onto the final score. Design better solutions based on your comprehensive analysis. Note: be mindful of time within your custom evaluation scripts, use appropirate timeouts."
}

EVALUATE_TEMPLATE = """from click import echo # use instead of print
from myautoresearch import Evaluator, run

class MyEvaluator(Evaluator):
    def evaluate(self):
        # the evaluated object is `self.object`
        pass


if __name__ == "__main__":
    run(MyEvaluator())
"""

README = """# README

This file is not shown to the AI agent. You can fill in the prompt template below and copy it.

## Prompt

Your goal will be to develop {TASK}. Run `mar start` in the shell and follow the instructions.
"""