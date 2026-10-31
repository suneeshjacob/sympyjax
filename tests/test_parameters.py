import json
import pytest
from three_rpr_sympy_jax.parameters import (
    RobotParams, parameters_from_json, parameters_to_dict, validate_parameters
)


def test_default_parameters_valid():
    validate_parameters(RobotParams())


def test_roundtrip_json(tmp_path):
    p = tmp_path / "params.json"
    p.write_text(json.dumps(parameters_to_dict(RobotParams())))
    assert parameters_from_json(p) == RobotParams()


def test_triangle_rejected():
    with pytest.raises(ValueError, match="triangle"):
        validate_parameters(RobotParams(c12=100))


def test_unknown_parameter_rejected(tmp_path):
    p = tmp_path / "params.json"
    p.write_text('{"unknown_param": 7}')
    with pytest.raises(ValueError, match="Unknown"):
        parameters_from_json(p)
