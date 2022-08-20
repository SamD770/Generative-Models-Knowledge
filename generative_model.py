class GenerativeModel:
    """Provides a standard interface for analysis code to interact with all types of models."""
    def eval_nll(self, x):
        raise NotImplementedError()

    def generate_sample(self, batch_size):
        raise NotImplementedError()

    @staticmethod
    def load_serialised(save_dir, save_file, **params):
        raise NotImplementedError()

