from alphaction.modeling import registry
from . import slowfast, i3d

@registry.BACKBONES.register("Slowfast-Resnet50")
@registry.BACKBONES.register("Slowfast-Resnet101")
def build_slowfast_resnet_backbone(cfg):
    model = slowfast.SlowFast(cfg)
    return model

@registry.BACKBONES.register("I3D-Resnet50")
@registry.BACKBONES.register("I3D-Resnet101")
@registry.BACKBONES.register("I3D-Resnet50-Sparse")
@registry.BACKBONES.register("I3D-Resnet101-Sparse")
def build_i3d_resnet_backbone(cfg):
    model = i3d.I3D(cfg)
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)