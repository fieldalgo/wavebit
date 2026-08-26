"""One module per backend, each holding what that framework does differently.

Empty on purpose -- importing this package must not import NumPy's, JAX's or PyTorch's
namespace on behalf of a caller who wants one of the other two. `main.rcwa` imports the single
module it needs, and for JAX that import is also what lifts the 32-bit cap.
"""
