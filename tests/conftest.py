import pytest
from three_rpr_sympy_jax import create_model


@pytest.fixture(scope="module")
def model():
    return create_model()
