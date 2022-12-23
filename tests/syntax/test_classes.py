
import math
import pytest

from scenic.core.errors import TokenParseError, ASTParseError, RuntimeParseError
from tests.utils import compileScenic, sampleScene, sampleEgoFrom, sampleEgoActions


def test_old_constructor_statement():
    with pytest.raises(SyntaxError):
        compileScenic("""
            constructor Foo:
                blah: (19, -3)
            ego = new Foo with blah 12
        """)

def test_python_class():
    scenario = compileScenic("""
        class Foo(object):
            def __init__(self, x):
                 self.x = x
        ego = new Object with width Foo(4).x
    """)
    scene = sampleScene(scenario, maxIterations=1)
    ego = scene.egoObject
    assert ego.width == 4

def test_invalid_attribute():
    with pytest.raises(SyntaxError):
        compileScenic("""
            class Foo:\n
                blah[baloney_attr]: 4
        """)

def test_property_simple():
    scenario = compileScenic("""
        class Foo:
            position: (3, 9, 0)
            flubber: -12
        ego = new Foo
    """)
    scene = sampleScene(scenario, maxIterations=1)
    ego = scene.egoObject
    assert type(ego).__name__ == 'Foo'
    assert tuple(ego.position) == (3, 9, 0)
    assert ego.flubber == -12

def test_property_inheritance():
    scenario = compileScenic("""
        class Foo:
            flubber: -12
        class Bar(Foo):
            flubber: 7
        ego = new Bar
    """)
    scene = sampleScene(scenario, maxIterations=1)
    ego = scene.egoObject
    assert type(ego).__name__ == 'Bar'
    assert ego.flubber == 7

def test_attribute_additive():
    """Additive properties"""
    ego = sampleEgoFrom("""
        class Parent:
            foo[additive]: 1
        class Child(Parent):
            foo[additive]: 2
        ego = new Child
    """)
    assert ego.foo == (2, 1)

def test_attribute_dynamic():
    """
    Test dynamic properties
    Since there isn't a easy way to test dynamic properties, assert that the internal flag is set to True
    """
    ego = sampleEgoFrom("""
        class CLS:
            foo[dynamic]: 1
        ego = new CLS
    """)
    # FIXME(shun): Write a proper test to check that the property is dynamic
    assert ego._scenic_properties["foo"].isDynamic

def test_attribute_final_override():
    """Properties marked as `final` cannot be overwritten"""
    with pytest.raises(RuntimeParseError) as excinfo:
        compileScenic(
            """
                class Parent():
                    one[final]: 1
                class Child(Parent):
                    one: 2
                ego = new Object at (1,1,1)
            """
        )
    assert "property cannot be overridden" in str(excinfo.value)

def test_attribute_final_specifier():
    """Properties marked as `final` cannot be specified"""
    with pytest.raises(RuntimeParseError) as excinfo:
        compileScenic(
            """
                class MyObject():
                    one[final]: 1
                ego = new MyObject with one 2
            """
        )
    assert "cannot be directly specified" in str(excinfo.value)

def test_isinstance_issubclass():
    scenario = compileScenic("""
        class Foo: pass
        ego = new Foo
        if isinstance(ego, Foo):
            other = new Object at (10, 0)
        if not isinstance(other, Foo):
            new Object at (20, 0)
        if issubclass(Foo, Point):
            new Object at (30, 0)
    """)
    scene = sampleScene(scenario)
    assert len(scene.objects) == 4
