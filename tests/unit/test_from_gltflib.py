def test_load_accessor():
    # One for each _TRIMESH_DTYPES
    # One for each _SHAPE_LOOKUP
    # bytestride
    # sparse
    pass


def test_load_bufferviews():
    # when uri is specified, it can load the source and get the resource
    # it correctly deals with byteoffset, bytelength
    pass


def test_load_attribute():
    # position, normal, tangennt, color_0, joints, and weights_0 all point to a 0 accessor
    # all can be optional
    # texcoords Ys are flipped
    pass


def test_load_image():
    # given an image + resource, load the bytes
    # given an image + bufferview, load the bytes
    pass


def test_load_primitive():
    # if no accessors provided, load from gltf
    # Targets -> target collection or None
    # load_attribute is called for all primitive attributes and all targets
    pass


def test_material_from_gltflib():
    # normal, occlusion, and emmisve are textures from image source object
    # the rest just get passed in
    pass


def test_mesh_from_gltflib():
    # material is singular
    # material is a list
    # material is specified
    pass


def test_scene_from_gltflib():
    # Test load bufferview is called
    # test load accessors is called
    # test load images is called
    # test load materials is called
    # test node children
    pass

