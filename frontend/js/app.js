/*
 * CVNodes frontend.
 *
 * This file has NO built-in knowledge of any OpenCV node. On load it fetches
 * node definitions from /api/nodes and builds a LiteGraph node class for
 * each one dynamically. Adding a new node type is purely a backend change
 * (drop a file in backend/nodes/) - this file never needs to be touched.
 */

const statusEl = document.getElementById("status");
const statusTextEl = statusEl.querySelector(".status-text");

let idleStatusMessage = "";
let statusFadeTimer = null;

// state: "idle" | "running" | "success" | "error"
function setStatus(message, state = "idle") {
  clearTimeout(statusFadeTimer);
  statusTextEl.textContent = message;
  statusEl.dataset.state = state;

  if (state === "idle") {
    idleStatusMessage = message;
  } else if (state === "success") {
    // Don't leave a stale "Done" sitting there forever - fade back to the
    // last idle message after a moment, like a toast.
    statusFadeTimer = setTimeout(() => {
      statusTextEl.textContent = idleStatusMessage;
      statusEl.dataset.state = "idle";
    }, 2500);
  }
}

function triggerFileUpload(onDone) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.style.display = "none";
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) {
      document.body.removeChild(input);
      return;
    }
    const form = new FormData();
    form.append("file", file);
    setStatus("Uploading…", "running");
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      onDone(json.filename);
      setStatus(`Uploaded ${json.filename}`, "success");
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`, "error");
    }
    document.body.removeChild(input);
  });
  document.body.appendChild(input);
  input.click();
}

// Custom LiteGraph widget: renders as a button, opens a file picker, and
// stores the uploaded filename as its value (so it serializes into
// widgets_values just like any other widget).
function addImageUploadWidget(node, def) {
  const widget = {
    name: def.name,
    type: "image_upload",
    value: def.default || "",
    options: {},
    computeSize(width) {
      return [width, 28];
    },
    draw(ctx, gnode, widgetWidth, y, H) {
      const margin = 6;
      ctx.fillStyle = "#3a3a44";
      ctx.strokeStyle = "#55555f";
      ctx.beginPath();
      ctx.roundRect(margin, y, widgetWidth - margin * 2, H, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#dddde5";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const label = widget.value ? widget.value : "Click to choose image...";
      ctx.fillText(label, widgetWidth / 2, y + H / 2 + 1);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    },
    mouse(event, pos, gnode) {
      if (event.type === "pointerdown" || event.type === "mousedown") {
        triggerFileUpload((filename) => {
          widget.value = filename;
          if (typeof gnode.setPreviewImage === "function") {
            gnode.setPreviewImage(`/uploads/${filename}`);
          }
          gnode.setDirtyCanvas(true, true);
        });
        return true;
      }
      return false;
    },
  };
  node.widgets = node.widgets || [];
  node.widgets.push(widget);
  return widget;
}

// Custom LiteGraph widget for STRING params. LiteGraph's built-in "text"
// widget draws the label and value without checking whether they fit,
// so a long value (e.g. a filename) overlaps/crushes into the label on a
// narrow node. This draws "label: value", ellipsizing the value to fit,
// and reuses the canvas's own prompt() dialog for editing (the same one
// the built-in text widget uses under the hood).
function addStringWidget(node, def) {
  const widget = {
    name: def.name,
    type: "cv_text",
    value: def.default ?? "",
    options: {},
    computeSize(width) {
      return [width, 26];
    },
    draw(ctx, gnode, widgetWidth, y, H) {
      const margin = 6;
      ctx.fillStyle = "#2a2a32";
      ctx.strokeStyle = "#454550";
      ctx.beginPath();
      ctx.roundRect(margin, y, widgetWidth - margin * 2, H, 4);
      ctx.fill();
      ctx.stroke();

      ctx.font = "11px sans-serif";
      ctx.textBaseline = "middle";

      const labelText = `${widget.name}:`;
      ctx.fillStyle = "#9a9aa5";
      ctx.textAlign = "left";
      ctx.fillText(labelText, margin + 8, y + H / 2 + 1);
      const labelWidth = ctx.measureText(labelText).width;

      let valueText = String(widget.value ?? "");
      const availableWidth = widgetWidth - margin * 2 - 16 - labelWidth - 8;
      ctx.fillStyle = "#e6e6ec";
      if (ctx.measureText(valueText).width > availableWidth) {
        while (valueText.length > 1 && ctx.measureText(valueText + "…").width > availableWidth) {
          valueText = valueText.slice(0, -1);
        }
        valueText += "…";
      }
      ctx.fillText(valueText, margin + 8 + labelWidth + 6, y + H / 2 + 1);

      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    },
    mouse(event, pos, gnode) {
      if (event.type === "pointerdown" || event.type === "mousedown") {
        // A real mousedown on the canvas sets LGraphCanvas.active_canvas
        // before dispatching to us - prompt() (a LiteGraph builtin) reads
        // that global itself, same as the built-in "text" widget does.
        const canvas = LGraphCanvas.active_canvas;
        if (canvas && typeof canvas.prompt === "function") {
          canvas.prompt("Value", widget.value, (v) => {
            widget.value = v;
            gnode.setDirtyCanvas(true, true);
          }, event);
        }
        return true;
      }
      return false;
    },
  };
  node.widgets = node.widgets || [];
  node.widgets.push(widget);
  return widget;
}

function addWidgetForDef(node, def) {
  switch (def.type) {
    case "INT":
      node.addWidget(
        "number",
        def.name,
        def.default ?? 0,
        function (v) {
          this.value = Math.round(v);
        },
        { min: def.min, max: def.max, step: def.step || 1, precision: 0 }
      );
      break;
    case "FLOAT":
      node.addWidget("number", def.name, def.default ?? 0, null, {
        min: def.min,
        max: def.max,
        step: def.step || 0.1,
        precision: 2,
      });
      break;
    case "BOOL":
      node.addWidget("toggle", def.name, !!def.default, null);
      break;
    case "COMBO":
      node.addWidget("combo", def.name, def.default, null, { values: def.options || [] });
      break;
    case "STRING":
      addStringWidget(node, def);
      break;
    case "IMAGE_UPLOAD":
      addImageUploadWidget(node, def);
      break;
    default:
      addStringWidget(node, def);
  }
}

// LiteGraph's computeSize() sizes a node from its slots/widgets only - it
// never checks whether the title text itself fits, so a long name (e.g.
// "Remove Background") overflows the rounded title bar on a narrow node.
const titleMeasureCtx = document.createElement("canvas").getContext("2d");
titleMeasureCtx.font = "bold 14px Arial";

const PREVIEW_MAX_WIDTH = 260;
const PREVIEW_MAX_HEIGHT = 200;
const PREVIEW_MARGIN = 8;
const PREVIEW_TEXT_LINE_HEIGHT = 20;
const PREVIEW_TEXT_PAD_V = 10;

function registerDynamicNode(def) {
  // Output nodes (Preview/Save Image, Preview Text) show their result once
  // the graph runs; LoadImage shows the picked image immediately after
  // upload - all three share the same inline-preview mechanism below.
  const hasPreview =
    def.is_output_node || (def.widgets || []).some((w) => w.type === "IMAGE_UPLOAD");

  function DynamicNode() {
    (def.inputs || []).forEach((inp) => this.addInput(inp.name, inp.type));
    (def.outputs || []).forEach((out) => this.addOutput(out.name, out.type));
    (def.widgets || []).forEach((w) => addWidgetForDef(this, w));
    // Without this, LiteGraph omits widgets_values from serialize(), and the
    // backend has no way to read widget values (node ids/params) out of the graph.
    this.serialize_widgets = true;
    this.size = this.computeSize();
    const titleWidth = titleMeasureCtx.measureText(def.name).width;
    this.size[0] = Math.max(this.size[0], titleWidth + 40);
    if (hasPreview) {
      this.baseSize = [this.size[0], this.size[1]];
      this.previewImg = null;
      this.previewText = null;
    }
  }
  DynamicNode.title = def.name;
  DynamicNode.desc = `${def.category} - ${def.type}`;
  DynamicNode.prototype.onExecute = function () {}; // execution happens server-side

  if (hasPreview) {
    // Draws the last result (image or text) inline under the node's widgets,
    // like ComfyUI's preview/save nodes - onDrawForeground runs after
    // widgets, in a coordinate space local to the node's content area
    // (below the title).
    DynamicNode.prototype.onDrawForeground = function (ctx) {
      if (this.flags.collapsed) return;
      const top = this.baseSize[1] + PREVIEW_MARGIN;

      if (this.previewImg) {
        const availW = this.size[0] - PREVIEW_MARGIN * 2;
        const availH = this.size[1] - top - PREVIEW_MARGIN;
        if (availW <= 0 || availH <= 0) return;
        const img = this.previewImg;
        const scale = Math.min(availW / img.width, availH / img.height);
        const w = img.width * scale;
        const h = img.height * scale;
        ctx.drawImage(img, PREVIEW_MARGIN + (availW - w) / 2, top, w, h);
        return;
      }

      if (this.previewText != null) {
        const lines = this.previewText === "" ? ["(empty)"] : this.previewText.split("\n");
        const boxW = this.size[0] - PREVIEW_MARGIN * 2;
        const boxH = lines.length * PREVIEW_TEXT_LINE_HEIGHT + PREVIEW_TEXT_PAD_V * 2;
        ctx.fillStyle = "#2a2a32";
        ctx.strokeStyle = "#454550";
        ctx.beginPath();
        ctx.roundRect(PREVIEW_MARGIN, top, boxW, boxH, 4);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#e6e6ec";
        ctx.font = lines.length > 1 ? "14px monospace" : "bold 16px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        lines.forEach((line, i) => {
          ctx.fillText(line, this.size[0] / 2, top + PREVIEW_TEXT_PAD_V + i * PREVIEW_TEXT_LINE_HEIGHT + PREVIEW_TEXT_LINE_HEIGHT / 2);
        });
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
      }
    };

    DynamicNode.prototype.setPreviewImage = function (dataUrl) {
      const img = new Image();
      img.onload = () => {
        this.previewText = null;
        this.previewImg = img;
        const scale = Math.min(PREVIEW_MAX_WIDTH / img.width, PREVIEW_MAX_HEIGHT / img.height, 1);
        const previewW = img.width * scale;
        const previewH = img.height * scale;
        this.size[0] = Math.max(this.baseSize[0], previewW + PREVIEW_MARGIN * 2);
        this.size[1] = this.baseSize[1] + previewH + PREVIEW_MARGIN * 2;
        this.setDirtyCanvas(true, true);
      };
      img.src = dataUrl;
    };

    DynamicNode.prototype.setPreviewText = function (text) {
      this.previewImg = null;
      this.previewText = text;
      const lineCount = text === "" ? 1 : text.split("\n").length;
      const boxH = lineCount * PREVIEW_TEXT_LINE_HEIGHT + PREVIEW_TEXT_PAD_V * 2;
      this.size[0] = this.baseSize[0];
      this.size[1] = this.baseSize[1] + boxH + PREVIEW_MARGIN * 2;
      this.setDirtyCanvas(true, true);
    };
  }

  LiteGraph.registerNodeType(def.type, DynamicNode);
  // registerNodeType() derives .category from the type id's "/" prefix and
  // overwrites whatever was set beforehand, so this has to happen after -
  // it groups nodes in the right-click "Add Node" menu (splitting on "/"
  // for nested submenus, e.g. "OpenCV/Filters") without touching the type
  // id itself, which has to stay identical to the backend's registry key.
  DynamicNode.category = def.category;
}

// After loading a saved workflow, LoadImage nodes already have a filename
// widget value (from a previous upload) but no thumbnail yet - fetch it back.
function restoreLoadImagePreviews(graph) {
  for (const node of graph._nodes) {
    if (typeof node.setPreviewImage !== "function") continue;
    const uploadWidget = (node.widgets || []).find((w) => w.type === "image_upload");
    if (uploadWidget && uploadWidget.value) {
      node.setPreviewImage(`/uploads/${uploadWidget.value}`);
    }
  }
}

// Pushes each output node's result image onto the node itself, so it shows
// inline on the canvas (see DynamicNode.prototype.setPreviewImage above).
function applyNodePreviews(graph, results) {
  for (const [nodeId, payload] of Object.entries(results || {})) {
    const node = graph.getNodeById(Number(nodeId));
    if (!node) continue;
    const values = Object.values(payload.values);

    const imageValue = values.find((v) => v.kind === "image");
    if (imageValue && typeof node.setPreviewImage === "function") {
      node.setPreviewImage(`data:image/png;base64,${imageValue.data}`);
      continue;
    }

    const rawValue = values.find((v) => v.kind === "raw");
    if (rawValue && typeof node.setPreviewText === "function") {
      node.setPreviewText(String(rawValue.data));
    }
  }
}

async function runGraph(graph) {
  setStatus("Running…", "running");
  const data = graph.serialize();
  try {
    const res = await fetch("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const json = await res.json();
    if (!res.ok) {
      const msg = json.error?.message || "Execution failed";
      setStatus(`Error: ${msg}`, "error");
      return;
    }
    applyNodePreviews(graph, json.results);
    setStatus("Done", "success");
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  }
}

function resizeCanvas(canvasEl, lgCanvas) {
  const wrap = document.getElementById("canvas-wrap");
  canvasEl.width = wrap.clientWidth;
  canvasEl.height = wrap.clientHeight;
  lgCanvas.resize();
}

async function main() {
  const canvasEl = document.getElementById("graph-canvas");
  const graph = new LGraph();
  const lgCanvas = new LGraphCanvas(canvasEl, graph);

  window.addEventListener("resize", () => resizeCanvas(canvasEl, lgCanvas));
  resizeCanvas(canvasEl, lgCanvas);

  try {
    const res = await fetch("/api/nodes");
    const defs = await res.json();
    defs.forEach(registerDynamicNode);
    setStatus(`Loaded ${defs.length} node types`, "idle");
  } catch (err) {
    setStatus(`Failed to load node definitions: ${err.message}`, "error");
  }

  graph.start();

  document.getElementById("run-btn").addEventListener("click", () => runGraph(graph));

  document.getElementById("save-btn").addEventListener("click", () => {
    const data = JSON.stringify(graph.serialize(), null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workflow.json";
    a.click();
    URL.revokeObjectURL(url);
  });

  const loadInput = document.getElementById("load-input");
  document.getElementById("load-btn").addEventListener("click", () => loadInput.click());
  loadInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    graph.configure(JSON.parse(text));
    restoreLoadImagePreviews(graph);
    e.target.value = "";
  });
}

main();
