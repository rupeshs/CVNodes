# Writing a custom node

Drop a `.py` file in this folder with a `@register_node`-decorated class,
restart the server, and it shows up in the UI automatically - the frontend
has zero hardcoded knowledge of any node. It fetches `/api/nodes` on page
load and builds a LiteGraph node class from whatever it gets back, so
adding a node is purely a backend change.

## Minimal example

```python
from engine.registry import register_node

@register_node("cv/MyNode")
class MyNode:
    NAME = "My Node"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [{"name": "amount", "type": "FLOAT", "default": 1.0, "min": 0, "max": 10, "step": 0.1}]

    def run(self, image, amount=1.0):
        ...
        return {"image": result}
```

That's the whole contract. Everything below explains each piece in detail.

## The class attributes

| Attribute | Required | Meaning |
|---|---|---|
| `@register_node("cv/MyNode")` | yes | The node's unique type id. Must be globally unique across every node (built-in and custom) - registering a duplicate raises `ValueError` at import time. Convention: `"cv/PascalCaseName"`. |
| `NAME` | no (defaults to the type id) | Display name shown as the node's title in the UI. |
| `CATEGORY` | no (defaults to `"Uncategorized"`) | A free-text label like `"OpenCV/Filters"`. Controls where the node appears in the right-click "Add Node" menu - use `/` for nested submenus (`"OpenCV/Filters"` becomes an `OpenCV` submenu containing a `Filters` submenu). This is independent of the type id in `@register_node(...)`, so it's safe to reorganize categories without breaking saved workflows. |
| `INPUTS` | no (defaults to `[]`) | List of `{"name": ..., "type": "IMAGE"}` - one entry per input slot, in order. |
| `OUTPUTS` | no (defaults to `[]`) | Same shape as `INPUTS` - one entry per output slot, in order. |
| `WIDGETS` | no (defaults to `[]`) | List of widget definitions (see below) - these become the node's on-canvas controls. |
| `IS_OUTPUT_NODE` | no (defaults to `False`) | Set `True` if this node's result should be returned to the frontend after a run (see [Output nodes](#output-nodes) below). |
| `run(self, **kwargs)` | yes | The actual processing function. See [How `run()` gets called](#how-run-gets-called). |

## Categories

There's no central list of categories to register anywhere - a category is
just whatever string you put in `CATEGORY`. The right-click "Add Node" menu
builds its submenu tree dynamically from whatever's currently in use across
every registered node (built-in and custom).

Current categories: `IO`, `OpenCV/Color`, `OpenCV/Compositing`,
`OpenCV/Detection`, `OpenCV/Filters`, `OpenCV/Threshold`, `OpenCV/Transform`.

To add a new one, just use a string that doesn't exist yet:

```python
CATEGORY = "OpenCV/Geometry"   # didn't exist before - that's fine
```

Restart the server and it appears as a new submenu automatically. A few
things worth knowing:

- **Nesting** - `/` creates submenus. `"OpenCV/Geometry"` becomes an
  `OpenCV` submenu containing a `Geometry` submenu; you can nest deeper,
  e.g. `"OpenCV/Geometry/Warp"`.
- **Renaming is safe** - the category is independent of the type id in
  `@register_node(...)`, so changing `CATEGORY` on an existing node just
  moves it in the menu. It doesn't affect saved workflows or execution.
- **Typos create near-duplicate categories silently** - `"OpenCV/Fitlers"`
  won't error, it'll just show up as its own near-empty submenu next to
  `OpenCV/Filters`. Double-check spelling against the list above.

## How `run()` gets called

For every node in the graph, in dependency order:

1. **Inputs** - for each entry in `INPUTS`, if that slot is wired to an upstream node, its value (whatever that upstream node's `run()` returned under the matching output name) is passed as a keyword argument named after `INPUTS[i]["name"]`.
2. **Widgets** - for each entry in `WIDGETS`, the current value from the UI is passed as a keyword argument named after `WIDGETS[i]["name"]`. If the graph JSON doesn't have a value for it (e.g. a node added programmatically), the widget's `"default"` is used instead.
3. `instance = cls()` (no constructor arguments - don't require any) then `instance.run(**kwargs)` is called.
4. The return value must be a `dict` mapping each name in `OUTPUTS` to its value. Extra keys are ignored; missing ones become `None` for anything downstream.

So your `run()` signature should accept every input and widget name as a keyword argument, e.g. for the example above: `def run(self, image, amount=1.0)`. Give widget parameters a Python default too - matches the pattern the built-in nodes use and keeps `run()` callable on its own (handy for testing, see below).

Multiple inputs/outputs work exactly the same way - `cv/Blend` (in `backend/nodes/compositing_nodes.py`) takes two images and returns one:

```python
INPUTS = [{"name": "image_a", "type": "IMAGE"}, {"name": "image_b", "type": "IMAGE"}]
OUTPUTS = [{"name": "image", "type": "IMAGE"}]

def run(self, image_a, image_b, alpha=0.5):
    ...
    return {"image": blended}
```

## Image data format

An `"IMAGE"` value is a `numpy.ndarray`, `uint8`, in OpenCV's native layout:

- Color images: shape `(H, W, 3)`, channel order **BGR** (not RGB).
- Grayscale (e.g. after `cv/Grayscale`, `cv/Threshold`, `cv/CannyEdge`): shape `(H, W)`, 2D.
- Images with transparency (e.g. after `cv/RemoveBackground`): shape `(H, W, 4)`, **BGRA**.

If your node only handles 3-channel color input, guard for the 2D/4-channel
cases the way the built-ins do, e.g.:

```python
if image.ndim == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
```

## Widget types

| `type` | Extra keys | UI control | Value passed to `run()` |
|---|---|---|---|
| `INT` | `default`, `min`, `max`, `step` | Draggable number, rounds to whole numbers | `int` |
| `FLOAT` | `default`, `min`, `max`, `step` | Draggable number, 2 decimal places | `float` |
| `BOOL` | `default` | Toggle switch | `bool` |
| `STRING` | `default` | `label: value` box, click to edit | `str` |
| `COMBO` | `default`, `options` (list of strings) | Dropdown | `str` (one of `options`) |
| `IMAGE_UPLOAD` | `default` (usually `""`) | "Click to choose image…" button, uploads via `/api/upload`, shows an inline thumbnail once picked | `str` (the uploaded filename, resolved against `backend/uploads/` on the backend side - see `cv/LoadImage` in `backend/nodes/io_nodes.py`) |

Example widgets, straight from the built-in nodes:

```python
WIDGETS = [
    {"name": "thresh", "type": "INT", "default": 127, "min": 0, "max": 255, "step": 1},
    {
        "name": "mode",
        "type": "COMBO",
        "default": "BINARY",
        "options": ["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV", "OTSU"],
    },
]
```

## Output nodes

Set `IS_OUTPUT_NODE = True` on a node whose result should come back from a
run - like `cv/PreviewImage` and `cv/SaveImage`. Only output nodes:

- Have their `run()` result returned in the `/api/execute` response (any
  numpy image value gets base64-PNG-encoded automatically by the server).
- Get an inline thumbnail drawn on the node itself in the UI after a run
  (same mechanism `cv/LoadImage` uses to preview the picked file).

A processing node in the middle of a chain (e.g. `cv/GaussianBlur`) doesn't
need this - its output only matters to whatever it's wired into.

## Errors

Let exceptions happen naturally - don't wrap `run()` in a broad `try/except`
that swallows errors. Anything raised inside `run()` is caught by the graph
executor, tagged with your node's `NAME` and id, and shown to the user as a
red error status in the UI (`"<Node Name>: <your exception message>"`). One
failing node doesn't crash the server or the rest of the graph run.

## Adding a dependency

Give your node its own subfolder instead of a lone `.py` file, with the
node code as `__init__.py` and a `requirements.txt` next to it:

```
custom_nodes/
  your_node/
    __init__.py        # the @register_node class(es), same as any node
    requirements.txt    # just this node's extra packages
```

That's the whole setup - nothing to run by hand. On the next server
startup, `import_all()` (in `engine/discovery.py`) sees the subfolder has a
`requirements.txt` and runs `pip install -r` on it, **using the same venv
the server itself is running in**, before importing that node's code. See
`custom_nodes/removebg/` for a real example (needs `rembg[cpu]`).

A few things worth knowing:

- **It's cached.** A hash of `requirements.txt` is stored in a
  `.requirements_installed` marker file next to it once the install
  succeeds. Unchanged on the next restart → pip is skipped entirely, so you
  don't pay the install cost every time you restart the server during
  development. Edit the requirements file → the hash changes → it
  reinstalls automatically.
- **A failed install doesn't crash the app.** If `pip install` fails (no
  internet, a bad pin, whatever), it's logged and that node's subsequent
  import fails too - caught the same way any other broken module is, with
  a `[cvnodes] Skipped node module '...'` warning. Every other node still
  works.
- **This does not give the node its own environment.** Everything still
  installs into **one shared Python environment** - this automates *when*
  and *how* `pip install` runs, not *where* it installs to. Two custom
  nodes wanting incompatible versions of the same package can still
  conflict with each other (this is the same shared-environment model
  ComfyUI uses today, and a well-known source of custom-node conflicts
  there - their own answer, "Nodes v3", is moving toward per-node process
  isolation, which is a much bigger change than this project takes on).
- **A node with no extra dependencies doesn't need a subfolder** - a plain
  `.py` file straight in `custom_nodes/` (like `nop.py`) is simpler and
  works exactly the same way.

## Testing your node without the UI

Since `run()` takes plain keyword arguments, you can call it directly in a
Python shell (from `backend/`, with the venv active):

```python
import cv2
from nodes.filter_nodes import GaussianBlurNode  # or your custom_nodes module

img = cv2.imread("uploads/sample.jpg")
result = GaussianBlurNode().run(img, ksize=9, sigma=2.0)["image"]
cv2.imwrite("test_out.png", result)
```

Or exercise it through the real HTTP API with curl, once the server's
running - useful for checking the full graph-execution path:

```bash
curl -s http://127.0.0.1:8010/api/nodes | python -m json.tool   # confirm it's registered
```

## Checklist

- [ ] `@register_node("cv/YourNode")` with a type id nothing else uses
- [ ] `NAME`, `CATEGORY` set (category controls the Add Node menu grouping)
- [ ] `INPUTS` / `OUTPUTS` / `WIDGETS` match your `run()` signature's parameter names exactly
- [ ] `run()` handles 2D/4-channel images gracefully if it assumes 3-channel BGR
- [ ] Node with an extra dependency lives in its own subfolder with a `requirements.txt` (not a lone `.py` file)
- [ ] Restarted the server and confirmed the node shows up (title bar text, widgets rendering correctly, no console errors)
