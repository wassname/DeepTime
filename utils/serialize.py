from json_tricks import dump, dumps, load, loads, strip_comments

def torch_encode(obj, primitives=False):
    from torch import Tensor
    if isinstance(obj, Tensor):
        if primitives:
            return obj.numpy().tolist()
        raise NotImplemented()
    return obj

def serialize(o):
    s = dumps(o, extra_obj_encoders=[torch_encode], primitives=True)
    return loads(s)
