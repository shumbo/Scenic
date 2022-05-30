import ast
import scenic.ast as s
import scenic.parser as p


def test_instance_basic():
    m = p.parse_string("new Car", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    n = r.value
    assert n.className == "Car"
    assert len(n.specifiers) == 0


def test_instance_with_single():
    m = p.parse_string("new Car with foo 30", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    n = r.value
    assert n.className == "Car"
    s0 = n.specifiers[0]
    assert isinstance(s0, s.With)
    assert s0.name == "foo"
    assert s0.expr.value == 30


def test_instance_with_multiple():
    m = p.parse_string("new Car with foo 30, with bar 40", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    n = r.value
    assert n.className == "Car"
    s1 = n.specifiers[1]
    assert isinstance(s1, s.With)
    assert s1.name == "bar"
    assert s1.expr.value == 40


def test_instance_at():
    m = p.parse_string("new Cat at (0, 0)", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.At)


def test_instance_offset_along_01():
    m = p.parse_string("new Cat offset by (10, 20)", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Offset)
    assert s1.along is None
    assert isinstance(s1.amount, ast.Tuple)
    assert s1.amount.elts[0].value == 10
    assert s1.amount.elts[1].value == 20


def test_instance_offset_along_02():
    m = p.parse_string("new Cat offset along vf by (10, 20)", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Offset)
    assert isinstance(s1.along, ast.Name)
    assert s1.along.id == "vf"
    assert isinstance(s1.amount, ast.Tuple)
    assert s1.amount.elts[0].value == 10
    assert s1.amount.elts[1].value == 20


def test_instance_left_of_01():
    m = p.parse_string("new Cat left of v", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "left"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert s1.distance is None


def test_instance_left_of_02():
    m = p.parse_string("new Cat left of v by 3", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "left"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert isinstance(s1.distance, ast.Constant)
    assert s1.distance.value == 3


def test_instance_right_of_01():
    m = p.parse_string("new Cat right of v", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "right"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert s1.distance is None


def test_instance_right_of_02():
    m = p.parse_string("new Cat right of v by 3", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "right"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert isinstance(s1.distance, ast.Constant)
    assert s1.distance.value == 3


def test_instance_ahead_of_01():
    m = p.parse_string("new Cat ahead of v", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "ahead"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert s1.distance is None


def test_instance_ahead_of_02():
    m = p.parse_string("new Cat ahead of v by 3", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "ahead"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert isinstance(s1.distance, ast.Constant)
    assert s1.distance.value == 3


def test_instance_behind_01():
    m = p.parse_string("new Cat behind v", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "behind"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert s1.distance is None


def test_instance_behind_02():
    m = p.parse_string("new Cat behind v by 3", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Position)
    # assert direction
    assert s1.direction == "behind"
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v"
    # assert distance
    assert isinstance(s1.distance, ast.Constant)
    assert s1.distance.value == 3


def test_instance_beyond_01():
    m = p.parse_string("new Cat beyond v1 by v2", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Beyond)
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v1"
    # assert offset
    assert isinstance(s1.offset, ast.Name)
    assert s1.offset.id == "v2"
    # assert distance
    assert s1.fromPosition is None


def test_instance_beyond_02():
    m = p.parse_string("new Cat beyond v1 by v2 from pos", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Beyond)
    # assert position
    assert isinstance(s1.position, ast.Name)
    assert s1.position.id == "v1"
    # assert offset
    assert isinstance(s1.offset, ast.Name)
    assert s1.offset.id == "v2"
    # assert distance
    assert isinstance(s1.fromPosition, ast.Name)
    assert s1.fromPosition.id == "pos"


def test_instance_visible_01():
    m = p.parse_string("new Cat visible", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Visible)
    assert s1.region is None


def test_instance_visible_02():
    m = p.parse_string("new Cat visible from point", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Visible)
    assert isinstance(s1.region, ast.Name)
    assert s1.region.id == "point"


def test_instance_not_visible_01():
    m = p.parse_string("new Cat not visible", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.NotVisible)
    assert s1.region is None


def test_instance_visible_02():
    m = p.parse_string("new Cat not visible from point", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.NotVisible)
    assert isinstance(s1.region, ast.Name)
    assert s1.region.id == "point"


def test_instance_in_01():
    m = p.parse_string("new Cat in region", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.In)
    assert isinstance(s1.region, ast.Name)
    assert s1.region.id == "region"


def test_instance_in_02():
    m = p.parse_string("new Cat on another_region", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.In)
    assert isinstance(s1.region, ast.Name)
    assert s1.region.id == "another_region"


def test_instance_following_01():
    m = p.parse_string("new Cat following field for 10", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Following)
    assert isinstance(s1.field, ast.Name)
    assert s1.field.id == "field"
    assert s1.fromPoint is None
    assert isinstance(s1.distance, ast.Constant)
    assert s1.distance.value == 10


def test_instance_following_02():
    m = p.parse_string("new Cat following field from point for 10", "exec")
    r = m.body[0]
    assert isinstance(r, ast.Expr)
    assert isinstance(r.value, s.New)
    s1 = r.value.specifiers[0]
    assert isinstance(s1, s.Following)
    assert isinstance(s1.field, ast.Name)
    assert s1.field.id == "field"
    assert isinstance(s1.fromPoint, ast.Name)
    assert s1.fromPoint.id == "point"
    assert isinstance(s1.distance, ast.Constant)
    assert s1.distance.value == 10
