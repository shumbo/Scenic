import ast
import scenic.ast as s
import scenic.parser as p

def test_assign_ego():
    m = p.parse_string("ego = new Car", "exec")
    a = m.body[0]
    assert isinstance(a, s.EgoAssign)
    assert isinstance(a.value, s.New)
