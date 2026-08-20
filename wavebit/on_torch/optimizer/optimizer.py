from tqdm import tqdm

class Grad:
    def __init__(self):
        pass



class OptimizerTorch(Grad):

    def __init__(self, *args, **kwargs):

        super().__init__()

    def grad(self, parameters, forward, loss_fn):
        [setattr(getattr(self, name), 'requires_grad', True) for name in parameters]
        result = forward()  # Forward Prop.
        loss = loss_fn(result)  # Loss
        loss.backward()  # Back Prop.
        grad = {name: getattr(self, name).grad for name in parameters}  # gradient

        return grad

    def _build_optimizer(self, _parameters, _opt, *args, **kwargs):
        _parameters_to_fit = [(getattr(self, name)) for name in _parameters]
        res = _opt(_parameters_to_fit, *args, **kwargs)
        return res

    def fit(self, parameters, forward, loss_fn, optimizer, opt_options, iteration=1):
        optimizer = self._build_optimizer(parameters, optimizer, **opt_options)
        [setattr(getattr(self, name), 'requires_grad', True) for name in parameters]

        for _ in tqdm(range(iteration)):
            optimizer.zero_grad()
            result = forward()  # Forward Prop.
            loss_value = loss_fn(result)  # Loss

            loss_value.backward()  # Back Prop.
            optimizer.step()

        return [getattr(self, name) for name in parameters]