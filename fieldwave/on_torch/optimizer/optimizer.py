from tqdm import tqdm

class Grad:
    def __init__(self):
        pass



class OptimizerTorch(Grad):

    def __init__(self, *args, **kwargs):

        super().__init__()

    def grad(self, pois, forward, loss_fn):
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]
        result = forward()  # Forward Prop.
        loss = loss_fn(result)  # Loss
        loss.backward()  # Back Prop.
        grad = {poi: getattr(self, poi).grad for poi in pois}  # gradient

        return grad

    def _build_optimizer(self, _pois, _opt, *args, **kwargs):
        _parameters_to_fit = [(getattr(self, poi)) for poi in _pois]
        res = _opt(_parameters_to_fit, *args, **kwargs)
        return res

    def fit(self, pois, forward, loss_fn, optimizer, opt_options, iteration=1):
        optimizer = self._build_optimizer(pois, optimizer, **opt_options)
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]

        for _ in tqdm(range(iteration)):
            optimizer.zero_grad()
            result = forward()  # Forward Prop.
            loss_value = loss_fn(result)  # Loss

            loss_value.backward()  # Back Prop.
            optimizer.step()

        return [getattr(self, poi) for poi in pois]