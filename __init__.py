from .nodes import LoadFloatModels, FloatProcess
 
NODE_CLASS_MAPPINGS = { 
    "LoadFloatModels" : LoadFloatModels,
    "FloatProcess" : FloatProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "LoadFloatModels" : "Load Float Models",
     "FloatProcess" : "Float Process",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']