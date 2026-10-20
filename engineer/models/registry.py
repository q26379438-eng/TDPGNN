from engineer.registry import Registry

BACKBONES = Registry('backbone')
HEADS = Registry('head')
GENERATOR = Registry('pose_generator')
T_GENERATOR = Registry('T_pose_generator')
