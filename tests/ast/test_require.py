import pytest
import ast
import scenic.ast as s
import scenic.parser as p

def test_require_basic():
    m = p.parse_string("require True", "exec")
    r = m.body[0]
    assert isinstance(r, s.Require)
    assert isinstance(r.value, ast.Constant)
    assert isinstance(r.prob, ast.Constant)
    assert r.value.value
    assert r.prob.value == 1


def test_require_named():
    m = p.parse_string("require True as name_of_req", "exec")
    r = m.body[0]
    assert isinstance(r, s.Require)
    assert r.name == "name_of_req"


def test_require_soft():
    m = p.parse_string("require[0.2] x", "exec")
    r = m.body[0]
    assert isinstance(r, s.Require)
    assert isinstance(r.prob, ast.Constant)
    assert r.prob.value == 0.2


def test_require_named_soft():
    m = p.parse_string("require[0.2] x as maybe", "exec")
    r = m.body[0]
    assert isinstance(r, s.Require)
    assert isinstance(r.prob, ast.Constant)
    assert r.prob.value == 0.2
    assert r.name == "maybe"


def test_require_soft_no_prob():
    with pytest.raises(SyntaxError):
        p.parse_string("require[] x as maybe", "exec")
