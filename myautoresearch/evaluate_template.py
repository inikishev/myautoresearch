EVALUATE_TEMPLATE = """from click import echo # use instead of print
from myautoresearch.evaluator import Evaluator

class MyEvaluator(Evaluator):
    def run(self):
        # the evaluated object is `self.object`
        pass


if __name__ == "__main__":
    evaluator = MyEvaluator()
    evaluator.run()
    evaluator.save()
"""