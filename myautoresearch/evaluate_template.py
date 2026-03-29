EVALUATE_TEMPLATE = """from click import echo # use instead of print
from myautoresearch import Evaluator, run

class MyEvaluator(Evaluator):
    def evaluate(self):
        # the evaluated object is `self.object`
        pass


if __name__ == "__main__":
    run(MyEvaluator())
"""