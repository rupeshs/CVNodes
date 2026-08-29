"""Your extension nodes go here. Every module in this package is
auto-imported at startup, same as backend/nodes/ - drop in a .py file with
a @register_node-decorated class and restart the server. See README.md in
this folder."""

from engine.discovery import import_all

import_all(__name__, __path__)
