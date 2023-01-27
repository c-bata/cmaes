try:
    import scipy as _
except ImportError as e:
    raise ImportError(
        "Tried to import 'scipy' but failed. Please make sure that the package is" 
        f"installed correctly to use this feature: {e}"
    ) from e

from cmaes.scipy._minimize import minimize  # NOQA
