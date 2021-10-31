from bayes_opt import BayesianOptimization


class BOMed(BayesianOptimization):
    def __init__(self, f, pbounds, random_state = 1):
        super(BOMed, self).__init__(f, pbounds, random_state)
        self.f = f
        self.pbounds = pbounds
        self.random_state = random_state
