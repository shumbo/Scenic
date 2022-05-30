import ast
import scenic.ast as s
import scenic.parser as p

def test_param_basic():
    m = p.parse_string("param k1=0", "exec")
    r = m.body[0]
    assert isinstance(r, s.Param)
    assert len(r.elts) == 1
    assert r.elts[0][0] == "k1"
    assert isinstance(r.elts[0][1], ast.Constant)
    assert r.elts[0][1].value == 0


def test_param_multiple():
    m = p.parse_string("param k1=0, k2=1", "exec")
    r = m.body[0]
    assert isinstance(r, s.Param)
    assert len(r.elts) == 2

    # first param
    assert r.elts[0][0] == "k1"
    assert isinstance(r.elts[0][1], ast.Constant)
    assert r.elts[0][1].value == 0

    # second param
    assert r.elts[1][0] == "k2"
    assert isinstance(r.elts[1][1], ast.Constant)
    assert r.elts[1][1].value == 1
