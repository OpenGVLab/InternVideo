import os

# backbone (e.g., conv / transformer)
backbones = {}
def register_backbone(name):
    def decorator(cls):
        backbones[name] = cls
        return cls
    return decorator

# neck (e.g., FPN)
necks = {}
def register_neck(name):
    def decorator(cls):
        necks[name] = cls
        return cls
    return decorator

# location generator (point, segment, etc)
generators = {}
def register_generator(name):
    def decorator(cls):
        generators[name] = cls
        return cls
    return decorator

# meta arch (the actual implementation of each model)
meta_archs = {}
def register_meta_arch(name):
    def decorator(cls):
        meta_archs[name] = cls
        return cls
    return decorator

# builder functions
def make_backbone(name, **kwargs):
    backbone = backbones[name](**kwargs)
    return backbone

def make_neck(name, **kwargs):
    neck = necks[name](**kwargs)
    return neck

def make_meta_arch(name, **kwargs):
    meta_arch = meta_archs[name](**kwargs)
    return meta_arch

def make_generator(name, **kwargs):
    generator = generators[name](**kwargs)
    return generator
