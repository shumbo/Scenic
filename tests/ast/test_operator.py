import ast
import scenic.ast as s
import scenic.parser as p

def test_deg():
    m = p.parse_string("10 deg", "exec")
    deg = m.body[0].value
    assert isinstance(deg, s.Deg)
    assert isinstance(deg.value, ast.Constant)
    assert deg.value.value == 10

def test_deg_precedence_1():
    m = p.parse_string("10 + 20 deg", "exec")
    plus = m.body[0].value
    assert isinstance(plus, ast.BinOp)
    assert isinstance(plus.op, ast.Add)
    assert isinstance(plus.right, s.Deg)
    assert plus.right.value.value == 20

def test_deg_precedence_2():
    m = p.parse_string("(10 + 20) deg", "exec")
    deg = m.body[0].value
    assert isinstance(deg, s.Deg)
    assert isinstance(deg.value, ast.BinOp)

def test_field_at():
    m = p.parse_string("vf at v", "exec")
    f = m.body[0].value
    assert isinstance(f, s.FieldAt)
    
def test_vector_constructor():
    m = p.parse_string("1 @ 2", "exec")
    f = m.body[0].value
    assert isinstance(f, s.Vector)
