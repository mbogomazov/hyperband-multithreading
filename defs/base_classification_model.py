from pandas import DataFrame


class BaseClassificationModel:

    space = {}
    data = DataFrame()

    def __init__(self, data):
        self.data = data

    def get_params():
        pass

    def try_params(n_iterations, params):
        pass
