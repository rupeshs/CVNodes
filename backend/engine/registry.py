"""
Node registry: every processing node registers itself here via the
@register_node decorator. The frontend has zero built-in knowledge of any
node - it fetches this registry (as JSON) from /api/nodes and builds its
UI dynamically. To add a new OpenCV operation, drop a class in
backend/nodes/ and decorate it; nothing else needs to change.
"""

REGISTRY = {}


def register_node(type_name):
    """Class decorator that registers a node under a unique type id,
    e.g. "cv/GaussianBlur"."""

    def wrapper(cls):
        if type_name in REGISTRY:
            raise ValueError(f"Node type '{type_name}' is already registered")
        cls.TYPE = type_name
        cls.INPUTS = getattr(cls, "INPUTS", [])
        cls.OUTPUTS = getattr(cls, "OUTPUTS", [])
        cls.WIDGETS = getattr(cls, "WIDGETS", [])
        cls.CATEGORY = getattr(cls, "CATEGORY", "Uncategorized")
        cls.NAME = getattr(cls, "NAME", type_name)
        cls.IS_OUTPUT_NODE = getattr(cls, "IS_OUTPUT_NODE", False)
        REGISTRY[type_name] = cls
        return cls

    return wrapper


def get_node_definitions():
    """Serializable description of every registered node, for the frontend."""
    defs = []
    for type_name, cls in REGISTRY.items():
        defs.append(
            {
                "type": type_name,
                "name": cls.NAME,
                "category": cls.CATEGORY,
                "inputs": cls.INPUTS,
                "outputs": cls.OUTPUTS,
                "widgets": cls.WIDGETS,
                "is_output_node": cls.IS_OUTPUT_NODE,
            }
        )
    return defs
