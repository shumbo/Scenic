from tests.utils import compileScenic, sampleEgoFrom, sampleParamP, sampleParamPFrom

## MeshShape tests

def test_mesh_shape():
    p = sampleParamPFrom("""
        ego = Object facing 30 deg
        other = Object facing 65 deg, at 10@10
        param p = relative heading of other
    """)
