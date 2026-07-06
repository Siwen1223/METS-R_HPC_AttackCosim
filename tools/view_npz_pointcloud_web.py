#!/usr/bin/env python3
"""
Open one NPZ LiDAR point-cloud frame in a browser-based WebGL viewer.

Edit the hyperparameters below, then run:

    python tools/view_npz_pointcloud_web.py

The script writes a self-contained HTML file and optionally opens it in the
default browser. No command-line arguments are used.
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import webbrowser
from pathlib import Path

import numpy as np


# ----------------------------- Hyperparameters -----------------------------

NPZ_PATH = "V2V-Attack-Dataset/runs/run_000020/sensors/vehicle_1/lidar/00000120.npz"
POINT_ARRAY_KEY = "lidar"  # Use None to automatically select the first array.
OUTPUT_HTML = "tools/npz_pointcloud_viewer.html"

MAX_POINTS = 80000
COLOR_BY = "distance"  # Options: "distance", "height", "intensity"
POINT_SIZE = 2.0
BACKGROUND_COLOR = "#05070a"
AUTO_OPEN_BROWSER = True

# CARLA LiDAR is often easier to inspect with y flipped to match map plots.
FLIP_Y = False


# ------------------------------- Data loading -------------------------------

def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def load_npz_points(npz_path: Path, key: str | None) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    selected_key = key
    if selected_key is None:
        if not data.files:
            raise ValueError(f"No arrays found in NPZ file: {npz_path}")
        selected_key = data.files[0]
    if selected_key not in data.files:
        raise KeyError(f"Key '{selected_key}' not found. Available keys: {data.files}")

    arr = np.asarray(data[selected_key])
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected an Nx3 or Nx4 array, got shape {arr.shape} for key '{selected_key}'")

    points = arr[:, :4].astype(np.float32, copy=True) if arr.shape[1] >= 4 else arr[:, :3].astype(np.float32, copy=True)
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    if points.size == 0:
        raise ValueError("Point cloud is empty after filtering invalid coordinates.")

    if FLIP_Y:
        points[:, 1] *= -1.0

    if len(points) > MAX_POINTS:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), size=MAX_POINTS, replace=False)
        points = points[np.sort(indices)]

    return points


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def make_colors(points: np.ndarray, mode: str) -> np.ndarray:
    xyz = points[:, :3]
    if mode == "height":
        t = normalize(xyz[:, 2])
    elif mode == "intensity" and points.shape[1] >= 4:
        t = normalize(points[:, 3])
    else:
        center = xyz.mean(axis=0)
        t = normalize(np.linalg.norm(xyz - center, axis=1))

    colors = np.zeros((len(points), 3), dtype=np.float32)
    colors[:, 0] = np.clip(1.6 - 2.2 * t, 0.05, 1.0)
    colors[:, 1] = np.clip(1.5 - np.abs(t - 0.48) * 2.6, 0.10, 1.0)
    colors[:, 2] = np.clip(2.2 * t - 0.2, 0.15, 1.0)
    return colors


def pack_float32_base64(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values.astype(np.float32, copy=False))
    return base64.b64encode(values.tobytes()).decode("ascii")


def pointcloud_metadata(points: np.ndarray, npz_path: Path) -> dict:
    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = xyz.mean(axis=0)
    span = maxs - mins
    return {
        "source": str(npz_path),
        "point_count": int(len(points)),
        "color_by": COLOR_BY,
        "point_size": float(POINT_SIZE),
        "background": BACKGROUND_COLOR,
        "center": [float(v) for v in center],
        "mins": [float(v) for v in mins],
        "maxs": [float(v) for v in maxs],
        "span": [float(v) for v in span],
        "camera_distance": float(max(np.linalg.norm(span), 1.0) * 1.4),
    }


# ------------------------------ HTML generation -----------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NPZ Point Cloud Viewer</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: {background};
      color: #e8eef7;
      font-family: Arial, sans-serif;
    }}
    #viewer {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    #panel {{
      position: fixed;
      top: 14px;
      left: 14px;
      max-width: 520px;
      padding: 10px 12px;
      background: rgba(5, 7, 10, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      line-height: 1.45;
      font-size: 13px;
      user-select: none;
    }}
    #panel b {{ color: #ffffff; }}
    #hint {{
      position: fixed;
      right: 14px;
      bottom: 12px;
      padding: 8px 10px;
      background: rgba(5, 7, 10, 0.68);
      border-radius: 6px;
      color: #cfd8e3;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <canvas id="viewer"></canvas>
  <div id="panel"></div>
  <div id="hint">Drag: rotate | Wheel: zoom | Double click: reset</div>
  <script>
    const metadata = {metadata_json};
    const pointBase64 = "{point_base64}";
    const colorBase64 = "{color_base64}";

    function decodeFloat32(base64) {{
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return new Float32Array(bytes.buffer);
    }}

    const points = decodeFloat32(pointBase64);
    const colors = decodeFloat32(colorBase64);
    const canvas = document.getElementById("viewer");
    const gl = canvas.getContext("webgl", {{ antialias: true }});
    if (!gl) {{
      document.body.innerHTML = "<p style='padding:20px'>WebGL is not available in this browser.</p>";
      throw new Error("WebGL unavailable");
    }}

    document.getElementById("panel").innerHTML =
      "<b>NPZ Point Cloud Viewer</b><br>" +
      "File: " + metadata.source + "<br>" +
      "Points: " + metadata.point_count.toLocaleString() + "<br>" +
      "Color: " + metadata.color_by + "<br>" +
      "XYZ min: [" + metadata.mins.map(v => v.toFixed(2)).join(", ") + "]<br>" +
      "XYZ max: [" + metadata.maxs.map(v => v.toFixed(2)).join(", ") + "]";

    const vsSource = `
      attribute vec3 aPosition;
      attribute vec3 aColor;
      uniform mat4 uMvp;
      uniform float uPointSize;
      varying vec3 vColor;
      void main() {{
        gl_Position = uMvp * vec4(aPosition, 1.0);
        gl_PointSize = uPointSize;
        vColor = aColor;
      }}
    `;
    const fsSource = `
      precision mediump float;
      varying vec3 vColor;
      void main() {{
        vec2 d = gl_PointCoord - vec2(0.5);
        if (dot(d, d) > 0.25) discard;
        gl_FragColor = vec4(vColor, 1.0);
      }}
    `;

    function compileShader(type, source) {{
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
        throw new Error(gl.getShaderInfoLog(shader));
      }}
      return shader;
    }}

    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vsSource));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fsSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
      throw new Error(gl.getProgramInfoLog(program));
    }}
    gl.useProgram(program);

    function bindAttribute(name, data, size) {{
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(program, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    }}

    bindAttribute("aPosition", points, 3);
    bindAttribute("aColor", colors, 3);

    const uMvp = gl.getUniformLocation(program, "uMvp");
    const uPointSize = gl.getUniformLocation(program, "uPointSize");

    let yaw = -0.75;
    let pitch = 0.55;
    let distance = metadata.camera_distance;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener("mousedown", e => {{
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    }});
    window.addEventListener("mouseup", () => dragging = false);
    window.addEventListener("mousemove", e => {{
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.006;
      pitch += (e.clientY - lastY) * 0.006;
      pitch = Math.max(-1.45, Math.min(1.45, pitch));
      lastX = e.clientX;
      lastY = e.clientY;
      draw();
    }});
    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      distance *= Math.exp(e.deltaY * 0.001);
      distance = Math.max(0.1, distance);
      draw();
    }}, {{ passive: false }});
    canvas.addEventListener("dblclick", () => {{
      yaw = -0.75;
      pitch = 0.55;
      distance = metadata.camera_distance;
      draw();
    }});

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      gl.viewport(0, 0, canvas.width, canvas.height);
      draw();
    }}
    window.addEventListener("resize", resize);

    function perspective(fovy, aspect, near, far) {{
      const f = 1.0 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, (2 * far * near) * nf, 0
      ];
    }}

    function multiply(a, b) {{
      const out = new Array(16).fill(0);
      for (let r = 0; r < 4; r++) {{
        for (let c = 0; c < 4; c++) {{
          for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
        }}
      }}
      return out;
    }}

    function translation(x, y, z) {{
      return [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1];
    }}

    function rotationX(a) {{
      const c = Math.cos(a), s = Math.sin(a);
      return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1];
    }}

    function rotationZ(a) {{
      const c = Math.cos(a), s = Math.sin(a);
      return [c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1];
    }}

    function draw() {{
      gl.clearColor(...hexToRgb(metadata.background), 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);

      const aspect = canvas.width / Math.max(1, canvas.height);
      const proj = perspective(Math.PI / 4, aspect, 0.05, Math.max(5000, distance * 20));
      const center = metadata.center;
      let view = translation(0, 0, -distance);
      view = multiply(view, rotationX(pitch));
      view = multiply(view, rotationZ(yaw));
      view = multiply(view, translation(-center[0], -center[1], -center[2]));
      const mvp = multiply(proj, view);

      gl.uniformMatrix4fv(uMvp, false, new Float32Array(mvp));
      gl.uniform1f(uPointSize, metadata.point_size * (window.devicePixelRatio || 1));
      gl.drawArrays(gl.POINTS, 0, metadata.point_count);
    }}

    function hexToRgb(hex) {{
      const clean = hex.replace("#", "");
      const n = parseInt(clean, 16);
      return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    }}

    resize();
  </script>
</body>
</html>
"""


def write_html(npz_path: Path, points: np.ndarray, output_path: Path) -> None:
    colors = make_colors(points, COLOR_BY)
    html = HTML_TEMPLATE.format(
        background=BACKGROUND_COLOR,
        metadata_json=json.dumps(pointcloud_metadata(points, npz_path)),
        point_base64=pack_float32_base64(points[:, :3]),
        color_base64=pack_float32_base64(colors),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    npz_path = resolve_repo_path(NPZ_PATH)
    output_path = resolve_repo_path(OUTPUT_HTML)
    points = load_npz_points(npz_path, POINT_ARRAY_KEY)
    write_html(npz_path, points, output_path)
    print(f"Loaded {len(points):,} points from {npz_path}")
    print(f"Viewer written to {output_path}")
    if AUTO_OPEN_BROWSER:
        webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main()
