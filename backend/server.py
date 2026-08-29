"""Entry point. Run from the backend/ directory:

    pip install -r requirements.txt
    uvicorn server:app --reload

Then open http://127.0.0.1:8000 in a browser.
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, OUTPUT_DIR, UPLOAD_DIR
from engine.graph import GraphExecutionError, execute_graph
from engine.image_io import numpy_to_base64_png
from engine.registry import get_node_definitions
import nodes  # noqa: F401 - triggers auto-discovery/registration of built-in nodes
import custom_nodes  # noqa: F401 - triggers auto-discovery/registration of extension nodes

FRONTEND_DIR = (BASE_DIR.parent / "frontend").resolve()

app = FastAPI(title="CVNodes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/nodes")
def list_nodes():
    return get_node_definitions()


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    suffix = Path(file.filename or "image.png").suffix or ".png"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest = UPLOAD_DIR / safe_name
    contents = await file.read()
    dest.write_bytes(contents)
    return {"filename": safe_name, "url": f"/uploads/{safe_name}"}


@app.post("/api/execute")
async def execute(graph: dict):
    try:
        raw_results = execute_graph(graph)
    except GraphExecutionError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"node_id": exc.node_id, "message": exc.message}},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    encoded_results = {}
    for node_id, payload in raw_results.items():
        values = {}
        for key, value in payload["values"].items():
            if hasattr(value, "shape"):  # numpy image array
                values[key] = {"kind": "image", "data": numpy_to_base64_png(value)}
            else:
                values[key] = {"kind": "raw", "data": value}
        encoded_results[str(node_id)] = {
            "type": payload["type"],
            "name": payload["name"],
            "values": values,
        }

    return {"results": encoded_results}
