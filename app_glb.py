# app.py
# Streamlit glb Inspector: upload a .glb and see rich stats
# Requirements: streamlit, pygltflib, numpy, pillow

import io
import base64
import json
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import streamlit as st
from pygltflib import GLTF2, BufferView, Accessor, Image as GLTFImage

try:
    import pandas as pd
except ImportError:
    pd = None

st.set_page_config(page_title="glb Inspector", layout="wide")

# ---------------------------
# Helpers: safe get / formats
# ---------------------------
MODE_MAP = {
    0: "POINTS", 1: "LINES", 2: "LINE_LOOP", 3: "LINE_STRIP",
    4: "TRIANGLES", 5: "TRIANGLE_STRIP", 6: "TRIANGLE_FAN"
}
COMPONENT_TYPE_MAP = {
    5120: "BYTE", 5121: "UNSIGNED_BYTE", 5122: "SHORT",
    5123: "UNSIGNED_SHORT", 5125: "UNSIGNED_INT", 5126: "FLOAT"
}
TYPE_NUM_COMPONENTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16
}

def sizeof_fmt(num: int, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

# ---------------------------
# Buffer access utils
# ---------------------------
def get_binary_blob(gltf: GLTF2) -> bytes:
    # For GLB, data is in a single binary blob
    try:
        return gltf.binary_blob()
    except Exception:
        return b""

def slice_from_bufferview(bb: bytes, bv: BufferView) -> bytes:
    offset = bv.byteOffset or 0
    length = bv.byteLength or 0
    return bb[offset: offset + length]

def get_accessor_count(gltf: GLTF2, acc_idx: Optional[int]) -> int:
    if acc_idx is None or acc_idx < 0:
        return 0
    try:
        return gltf.accessors[acc_idx].count or 0
    except Exception:
        return 0

def triangles_from_primitive(gltf: GLTF2, prim) -> int:
    mode = prim.mode if prim.mode is not None else 4  # default TRIANGLES
    idx_count = get_accessor_count(gltf, prim.indices)
    pos_count = 0
    if prim.attributes and hasattr(prim.attributes, "POSITION") and prim.attributes.POSITION is not None:
        pos_count = get_accessor_count(gltf, prim.attributes.POSITION)

    if mode == 4:  # TRIANGLES
        if idx_count > 0:
            return idx_count // 3
        return pos_count // 3
    elif mode == 5:  # TRIANGLE_STRIP
        n = idx_count if idx_count > 0 else pos_count
        return max(0, n - 2)
    elif mode == 6:  # TRIANGLE_FAN
        n = idx_count if idx_count > 0 else pos_count
        return max(0, n - 2)
    else:
        return 0

def vertices_from_primitive(gltf: GLTF2, prim) -> int:
    if prim.attributes and hasattr(prim.attributes, "POSITION") and prim.attributes.POSITION is not None:
        return get_accessor_count(gltf, prim.attributes.POSITION)
    return 0

# ---------------------------
# Image extraction & size
# ---------------------------
def image_bytes_from_gltf(gltf: GLTF2, img: GLTFImage, bb: bytes) -> Optional[bytes]:
    # Case 1: bufferView
    if img.bufferView is not None and img.bufferView >= 0:
        try:
            bv = gltf.bufferViews[img.bufferView]
            return slice_from_bufferview(bb, bv)
        except Exception:
            pass
    # Case 2: data URI
    if img.uri and img.uri.startswith("data:"):
        try:
            header, b64 = img.uri.split(",", 1)
            return base64.b64decode(b64)
        except Exception:
            return None
    # External URI is uncommon in .glb; not handled here.
    return None

def image_size_from_bytes(b: bytes) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(b)) as im:
            return im.width, im.height
    except Exception:
        return None

# ---------------------------
# Scene graph pretty-print
# ---------------------------
def node_name(gltf: GLTF2, idx: int) -> str:
    try:
        n = gltf.nodes[idx]
        return n.name or f"Node_{idx}"
    except Exception:
        return f"Node_{idx}"

def build_tree(gltf: GLTF2) -> List[Dict]:
    trees = []
    def node_dict(i: int) -> Dict:
        nd = gltf.nodes[i]
        d = {
            "id": i,
            "name": nd.name or f"Node_{i}",
            "mesh": getattr(nd, "mesh", None),
            "skin": getattr(nd, "skin", None),
            "camera": getattr(nd, "camera", None),
            "children": []
        }
        if getattr(nd, "matrix", None):
            d["matrix"] = nd.matrix
        else:
            if nd.translation: d["translation"] = nd.translation
            if nd.rotation: d["rotation"] = nd.rotation
            if nd.scale: d["scale"] = nd.scale
        if nd.children:
            for c in nd.children:
                d["children"].append(node_dict(c))
        return d

    if not gltf.scenes or len(gltf.scenes) == 0:
        return trees
    for s_idx, scene in enumerate(gltf.scenes):
        root = {"sceneIndex": s_idx, "name": scene.name or f"Scene_{s_idx}", "roots": []}
        if scene.nodes:
            for n in scene.nodes:
                root["roots"].append(node_dict(n))
        trees.append(root)
    return trees

# ---------------------------
# Helper: Load GLB and extract stats
# ---------------------------
def load_glb_stats(glb_bytes: bytes, filename: str) -> Dict:
    """Load GLB file and extract all statistics"""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as tmp_file:
        tmp_file.write(glb_bytes)
        tmp_path = tmp_file.name
    
    try:
        gltf = GLTF2().load(tmp_path)
        bb = get_binary_blob(gltf)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # Calculate stats
    num_meshes = len(gltf.meshes or [])
    num_nodes = len(gltf.nodes or [])
    num_materials = len(gltf.materials or [])
    num_images = len(gltf.images or [])
    num_textures = len(gltf.textures or [])
    
    total_primitives = 0
    total_vertices = 0
    total_triangles = 0
    has_normal = False
    has_texcoord = False
    has_color = False
    
    for mesh in (gltf.meshes or []):
        for prim in (mesh.primitives or []):
            total_primitives += 1
            total_vertices += vertices_from_primitive(gltf, prim)
            total_triangles += triangles_from_primitive(gltf, prim)
            
            if prim.attributes:
                if hasattr(prim.attributes, "NORMAL") and prim.attributes.NORMAL is not None:
                    has_normal = True
                if hasattr(prim.attributes, "TEXCOORD_0") and prim.attributes.TEXCOORD_0 is not None:
                    has_texcoord = True
                if hasattr(prim.attributes, "COLOR_0") and prim.attributes.COLOR_0 is not None:
                    has_color = True
    
    # Texture info
    max_texture_size = 0
    total_texture_pixels = 0
    texture_sizes = []
    
    for img in (gltf.images or []):
        by = image_bytes_from_gltf(gltf, img, bb)
        wh = image_size_from_bytes(by) if by else None
        if wh:
            texture_sizes.append(wh)
            max_texture_size = max(max_texture_size, max(wh))
            total_texture_pixels += wh[0] * wh[1]
    
    # Material complexity
    num_pbr_textures = 0
    has_normal_map = False
    has_occlusion_map = False
    has_emissive_map = False
    
    for m in (gltf.materials or []):
        pbr = getattr(m, "pbrMetallicRoughness", None)
        if pbr:
            if getattr(pbr, "baseColorTexture", None):
                num_pbr_textures += 1
            if getattr(pbr, "metallicRoughnessTexture", None):
                num_pbr_textures += 1
        if hasattr(m, "normalTexture") and m.normalTexture:
            has_normal_map = True
            num_pbr_textures += 1
        if hasattr(m, "occlusionTexture") and m.occlusionTexture:
            has_occlusion_map = True
            num_pbr_textures += 1
        if hasattr(m, "emissiveTexture") and m.emissiveTexture:
            has_emissive_map = True
            num_pbr_textures += 1
    
    # Check compression
    extensions_used = gltf.extensionsUsed or []
    has_draco = "KHR_draco_mesh_compression" in extensions_used
    
    # Vertices to triangles ratio (optimization indicator)
    vtx_tri_ratio = total_vertices / total_triangles if total_triangles > 0 else 0
    
    return {
        "filename": filename,
        "gltf": gltf,
        "bb": bb,
        "file_size": len(glb_bytes),
        "num_meshes": num_meshes,
        "num_nodes": num_nodes,
        "num_materials": num_materials,
        "num_images": num_images,
        "num_textures": num_textures,
        "total_primitives": total_primitives,
        "total_vertices": total_vertices,
        "total_triangles": total_triangles,
        "draw_calls": total_primitives,
        "has_normal": has_normal,
        "has_texcoord": has_texcoord,
        "has_color": has_color,
        "max_texture_size": max_texture_size,
        "total_texture_pixels": total_texture_pixels,
        "texture_sizes": texture_sizes,
        "num_pbr_textures": num_pbr_textures,
        "has_normal_map": has_normal_map,
        "has_occlusion_map": has_occlusion_map,
        "has_emissive_map": has_emissive_map,
        "has_draco": has_draco,
        "vtx_tri_ratio": vtx_tri_ratio,
        "extensions_used": extensions_used,
    }

# ---------------------------
# UI
# ---------------------------
st.title("🧭 GLB Inspector & Comparator")

# Info box with comparison guide
with st.expander("📊 3D 모델 비교 가이드", expanded=False):
    st.markdown("""
    ### 🎯 두 모델 비교 시 핵심 지표
    
    #### 1️⃣ **기하학적 복잡도 (Geometry Complexity)**
    - **삼각형 수 (Triangles)**: 더 많을수록 디테일↑, 성능↓
    - **정점 수 (Vertices)**: 실제 데이터량
    - **Vertices/Triangles 비율**: ~1.0에 가까우면 인덱싱 최적화 잘됨
    - **프리미티브 수**: 적을수록 드로우 콜 감소 = 성능↑
    
    #### 2️⃣ **텍스처 퀄리티**
    - **텍스처 해상도**: 높을수록 선명하지만 메모리↑ (2K=2048px, 4K=4096px)
    - **텍스처 개수**: 많을수록 메모리↑
    - **총 텍스처 픽셀**: 실제 메모리 사용량 지표
    - **텍스처 포맷**: PNG(손실없음) vs JPEG(압축)
    
    #### 3️⃣ **머티리얼 복잡도**
    - **PBR 텍스처**: BaseColor, Normal, Metallic/Roughness, Occlusion, Emissive
    - **Normal맵**: 있으면 표면 디테일↑
    - **Occlusion맵**: 있으면 그림자 현실감↑
    - **Emissive맵**: 발광 효과
    
    #### 4️⃣ **파일 효율성**
    - **파일 크기**: 네트워크 전송/로딩 시간
    - **압축 여부**: Draco 압축 사용 시 크기↓
    - **파일 크기 / 삼각형 비율**: 효율성 지표
    
    #### 5️⃣ **최적화 수준**
    - **드로우 콜 수**: 적을수록 CPU 효율↑ (200개 이하 권장)
    - **Attribute 종류**: NORMAL, TEXCOORD_0 (UV), COLOR_0 유무
    - **인덱싱 사용**: 메모리 효율성
    
    ### 💡 성능 권장 기준
    - **모바일**: 삼각형 ~10만개, 텍스처 2K 이하, 드로우콜 50개 이하
    - **웹/PC**: 삼각형 ~100만개, 텍스처 4K 이하, 드로우콜 200개 이하
    - **최적 Vtx/Tri 비율**: 0.5~1.5 (인덱싱 효율 좋음)
    """)

st.markdown("---")

# File uploaders - support up to 2 files for comparison
col1, col2 = st.columns(2)

with col1:
    uploaded1 = st.file_uploader("`.glb` 파일 1 (또는 단일 파일)", type=["glb"], key="file1")

with col2:
    uploaded2 = st.file_uploader("`.glb` 파일 2 (비교용, 선택사항)", type=["glb"], key="file2")

if not uploaded1:
    st.info("좌측에 `.glb` 파일을 업로드하세요. 두 번째 파일을 추가하면 비교 분석이 가능합니다.")
    st.stop()

# Load file(s)
stats1 = load_glb_stats(uploaded1.read(), uploaded1.name)
stats2 = load_glb_stats(uploaded2.read(), uploaded2.name) if uploaded2 else None

# ---------------------------
# Comparison mode vs Single mode
# ---------------------------
if stats2:
    st.header("📊 모델 비교 (Comparison)")
    
    # Comparison table
    def format_diff(val1, val2, is_larger_better=False, unit=""):
        if val1 == 0:
            pct_diff = 0
        else:
            pct_diff = ((val2 - val1) / val1) * 100
        
        if abs(pct_diff) < 0.1:
            diff_str = "동일"
            color = "🟢"
        elif (pct_diff > 0 and is_larger_better) or (pct_diff < 0 and not is_larger_better):
            diff_str = f"{abs(pct_diff):+.1f}% 👍"
            color = "🟢"
        else:
            diff_str = f"{pct_diff:+.1f}%"
            color = "🔴" if abs(pct_diff) > 50 else "🟡"
        
        return f"{color} {diff_str}"
    
    def format_value(val, unit=""):
        if isinstance(val, float):
            return f"{val:,.2f}{unit}"
        elif isinstance(val, int):
            return f"{val:,}{unit}"
        elif isinstance(val, bool):
            return "✅" if val else "❌"
        else:
            return str(val)
    
    # Create comparison dataframe
    comparison_data = {
        "항목": [],
        stats1["filename"]: [],
        stats2["filename"]: [],
        "차이": []
    }
    
    # File size
    comparison_data["항목"].append("📦 파일 크기")
    comparison_data[stats1["filename"]].append(sizeof_fmt(stats1["file_size"]))
    comparison_data[stats2["filename"]].append(sizeof_fmt(stats2["file_size"]))
    comparison_data["차이"].append(format_diff(stats1["file_size"], stats2["file_size"], is_larger_better=False))
    
    # Triangles
    comparison_data["항목"].append("🔺 삼각형 수")
    comparison_data[stats1["filename"]].append(format_value(stats1["total_triangles"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["total_triangles"]))
    comparison_data["차이"].append(format_diff(stats1["total_triangles"], stats2["total_triangles"], is_larger_better=True))
    
    # Vertices
    comparison_data["항목"].append("📍 정점 수")
    comparison_data[stats1["filename"]].append(format_value(stats1["total_vertices"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["total_vertices"]))
    comparison_data["차이"].append(format_diff(stats1["total_vertices"], stats2["total_vertices"], is_larger_better=True))
    
    # Vtx/Tri ratio
    comparison_data["항목"].append("📊 정점/삼각형 비율")
    comparison_data[stats1["filename"]].append(format_value(stats1["vtx_tri_ratio"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["vtx_tri_ratio"]))
    comparison_data["차이"].append("optimal: 0.5~1.5")
    
    # Draw calls
    comparison_data["항목"].append("🎨 드로우 콜")
    comparison_data[stats1["filename"]].append(format_value(stats1["draw_calls"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["draw_calls"]))
    comparison_data["차이"].append(format_diff(stats1["draw_calls"], stats2["draw_calls"], is_larger_better=False))
    
    # Textures
    comparison_data["항목"].append("🖼️ 텍스처 개수")
    comparison_data[stats1["filename"]].append(format_value(stats1["num_images"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["num_images"]))
    comparison_data["차이"].append(format_diff(stats1["num_images"], stats2["num_images"], is_larger_better=True))
    
    # Max texture size
    comparison_data["항목"].append("📐 최대 텍스처 해상도")
    comparison_data[stats1["filename"]].append(f"{stats1['max_texture_size']}px" if stats1['max_texture_size'] > 0 else "없음")
    comparison_data[stats2["filename"]].append(f"{stats2['max_texture_size']}px" if stats2['max_texture_size'] > 0 else "없음")
    comparison_data["차이"].append(format_diff(stats1["max_texture_size"], stats2["max_texture_size"], is_larger_better=True))
    
    # Total texture pixels
    comparison_data["항목"].append("💾 총 텍스처 픽셀")
    comparison_data[stats1["filename"]].append(format_value(stats1["total_texture_pixels"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["total_texture_pixels"]))
    comparison_data["차이"].append(format_diff(stats1["total_texture_pixels"], stats2["total_texture_pixels"], is_larger_better=True))
    
    # Materials
    comparison_data["항목"].append("🎨 머티리얼 수")
    comparison_data[stats1["filename"]].append(format_value(stats1["num_materials"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["num_materials"]))
    comparison_data["차이"].append(format_diff(stats1["num_materials"], stats2["num_materials"], is_larger_better=False))
    
    # PBR Textures
    comparison_data["항목"].append("✨ PBR 텍스처 수")
    comparison_data[stats1["filename"]].append(format_value(stats1["num_pbr_textures"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["num_pbr_textures"]))
    comparison_data["차이"].append(format_diff(stats1["num_pbr_textures"], stats2["num_pbr_textures"], is_larger_better=True))
    
    # Attributes
    comparison_data["항목"].append("📝 NORMAL 속성")
    comparison_data[stats1["filename"]].append(format_value(stats1["has_normal"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["has_normal"]))
    comparison_data["차이"].append("있으면 좋음")
    
    comparison_data["항목"].append("🗺️ TEXCOORD (UV)")
    comparison_data[stats1["filename"]].append(format_value(stats1["has_texcoord"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["has_texcoord"]))
    comparison_data["차이"].append("텍스처 매핑 필수")
    
    # Normal map
    comparison_data["항목"].append("🗻 Normal 맵")
    comparison_data[stats1["filename"]].append(format_value(stats1["has_normal_map"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["has_normal_map"]))
    comparison_data["차이"].append("디테일↑")
    
    # Occlusion map
    comparison_data["항목"].append("🌑 Occlusion 맵")
    comparison_data[stats1["filename"]].append(format_value(stats1["has_occlusion_map"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["has_occlusion_map"]))
    comparison_data["차이"].append("현실감↑")
    
    # Draco compression
    comparison_data["항목"].append("🗜️ Draco 압축")
    comparison_data[stats1["filename"]].append(format_value(stats1["has_draco"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["has_draco"]))
    comparison_data["차이"].append("크기↓")
    
    # Meshes
    comparison_data["항목"].append("🧱 메시 수")
    comparison_data[stats1["filename"]].append(format_value(stats1["num_meshes"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["num_meshes"]))
    comparison_data["차이"].append(format_diff(stats1["num_meshes"], stats2["num_meshes"], is_larger_better=False))
    
    # Nodes
    comparison_data["항목"].append("🌳 노드 수")
    comparison_data[stats1["filename"]].append(format_value(stats1["num_nodes"]))
    comparison_data[stats2["filename"]].append(format_value(stats2["num_nodes"]))
    comparison_data["차이"].append(format_diff(stats1["num_nodes"], stats2["num_nodes"], is_larger_better=False))
    
    # Display comparison table
    if pd is not None:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        # Fallback if pandas not available
        for i in range(len(comparison_data["항목"])):
            st.write(f"**{comparison_data['항목'][i]}**")
            col1, col2, col3 = st.columns(3)
            col1.write(comparison_data[stats1["filename"]][i])
            col2.write(comparison_data[stats2["filename"]][i])
            col3.write(comparison_data["차이"][i])
    
    
    st.markdown("---")
    
    # Select which model to inspect in detail
    selected_model = st.radio("상세 정보를 볼 모델 선택:", [stats1["filename"], stats2["filename"]])
    stats = stats1 if selected_model == stats1["filename"] else stats2
    gltf = stats["gltf"]
    bb = stats["bb"]
    
else:
    # Single file mode
    st.header(f"📄 {stats1['filename']}")
    stats = stats1
    gltf = stats["gltf"]
    bb = stats["bb"]

# ---------------------------
# Display detailed stats (works for both modes)
# ---------------------------

# Extract stats for display
size = stats["file_size"]
num_meshes = stats["num_meshes"]
num_nodes = stats["num_nodes"]
num_materials = stats["num_materials"]
num_images = stats["num_images"]
num_textures = stats["num_textures"]
total_primitives = stats["total_primitives"]
total_vertices = stats["total_vertices"]
total_triangles = stats["total_triangles"]
draw_calls = stats["draw_calls"]
num_animations = len(gltf.animations or [])
num_skins = len(gltf.skins or [])
num_cameras = len(gltf.cameras or [])
extensions_used = stats["extensions_used"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("파일 크기", sizeof_fmt(size))
c2.metric("메시(프리미티브)", f"{num_meshes} meshes / {total_primitives} prims")
c3.metric("정점/삼각형 수(합계)", f"{total_vertices:,} / {total_triangles:,}")
c4.metric("드로우 콜(추정)", f"{draw_calls}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("노드", f"{num_nodes}")
c6.metric("머티리얼", f"{num_materials}")
c7.metric("텍스처 / 이미지", f"{num_textures} / {num_images}")
c8.metric("애니메이션 / 스킨 / 카메라", f"{num_animations} / {num_skins} / {num_cameras}")

# ---------------------------
# Meshes & primitives
# ---------------------------
with st.expander("🧱 Meshes & Primitives (정점/인덱스/모드/머티리얼)", expanded=True):
    st.info("""
    **📖 설명**: Mesh는 3D 모델의 기하학적 형태를 정의합니다. 각 Mesh는 하나 이상의 Primitive로 구성됩니다.
    - **Primitive**: 실제 렌더링 단위. 각 Primitive는 하나의 드로우 콜에 해당합니다.
    - **Mode**: 렌더링 방식 (TRIANGLES, TRIANGLE_STRIP 등)
    - **Vertices**: 정점 개수 (3D 공간상의 점)
    - **Indices**: 인덱스 개수 (정점을 어떻게 연결할지 정의)
    - **Attributes**: POSITION(위치), NORMAL(법선), TEXCOORD(UV좌표), COLOR(색상) 등
    """)
    for mi, mesh in enumerate(gltf.meshes or []):
        st.markdown(f"**Mesh {mi} — {mesh.name or 'unnamed'}**")
        for pi, prim in enumerate(mesh.primitives or []):
            mode = MODE_MAP.get(prim.mode if prim.mode is not None else 4, str(prim.mode))
            mat_idx = prim.material if prim.material is not None else -1
            pos_acc = getattr(prim.attributes, "POSITION", None) if prim.attributes else None
            nrm_acc = getattr(prim.attributes, "NORMAL", None) if prim.attributes else None
            uv_acc = getattr(prim.attributes, "TEXCOORD_0", None) if prim.attributes else None
            col_acc = getattr(prim.attributes, "COLOR_0", None) if prim.attributes else None
            idx_count = get_accessor_count(gltf, prim.indices)
            vtx_count = get_accessor_count(gltf, pos_acc)

            st.write(f"- Primitive {pi}: mode={mode}, vertices={vtx_count:,}, indices={idx_count:,}, material={mat_idx}")
            attrs = []
            if nrm_acc is not None: attrs.append("NORMAL")
            if uv_acc is not None: attrs.append("TEXCOORD_0")
            if col_acc is not None: attrs.append("COLOR_0")
            if attrs:
                st.caption("  · Attributes: " + ", ".join(["POSITION"] + attrs))
        st.divider()

# ---------------------------
# Materials (PBR, normal/occlusion/emissive)
# ---------------------------
with st.expander("🎨 Materials (PBR)", expanded=False):
    st.info("""
    **📖 설명**: Material은 표면의 시각적 속성을 정의합니다.
    - **PBR (Physically Based Rendering)**: 물리 기반 렌더링 방식
    - **BaseColorFactor**: 기본 색상 (RGBA, 0~1 범위)
    - **MetallicFactor**: 금속성 (0=비금속, 1=금속)
    - **RoughnessFactor**: 거칠기 (0=매끄러움, 1=거침)
    - **NormalTexture**: 표면 디테일을 위한 노멀맵
    - **OcclusionTexture**: 주변광 차폐 맵 (그림자 효과)
    - **EmissiveTexture/Factor**: 자체 발광 효과
    """)
    if not gltf.materials:
        st.write("Materials not present.")
    else:
        for i, m in enumerate(gltf.materials):
            st.markdown(f"**Material {i} — {m.name or 'unnamed'}**")
            pbr = getattr(m, "pbrMetallicRoughness", None)
            if pbr:
                bc = getattr(pbr, "baseColorFactor", None)
                mr = getattr(pbr, "metallicFactor", None)
                rr = getattr(pbr, "roughnessFactor", None)
                bct = getattr(pbr, "baseColorTexture", None)
                mrt = getattr(pbr, "metallicRoughnessTexture", None)
                st.write(f"- BaseColorFactor: {bc}")
                st.write(f"- Metallic/Roughness: {mr}, {rr}")
                if bct and getattr(bct, 'index', None) is not None:
                    st.write(f"- BaseColorTexture index: {bct.index}")
                if mrt and getattr(mrt, 'index', None) is not None:
                    st.write(f"- MetallicRoughnessTexture index: {mrt.index}")
            if m.normalTexture and m.normalTexture.index is not None:
                st.write(f"- NormalTexture index: {m.normalTexture.index}")
            if m.occlusionTexture and m.occlusionTexture.index is not None:
                st.write(f"- OcclusionTexture index: {m.occlusionTexture.index}")
            if m.emissiveTexture and m.emissiveTexture.index is not None:
                st.write(f"- EmissiveTexture index: {m.emissiveTexture.index}")
            if m.emissiveFactor:
                st.write(f"- EmissiveFactor: {m.emissiveFactor}")
            st.divider()

# ---------------------------
# Textures & Images (dimensions)
# ---------------------------
with st.expander("🖼️ Textures & Images (해상도, MIME)", expanded=False):
    st.info("""
    **📖 설명**: 3D 모델의 표면에 입히는 이미지들입니다.
    - **Image**: 실제 이미지 데이터 (PNG, JPEG 등)
    - **Texture**: Image를 참조하고 샘플링 방법을 정의
    - **해상도**: 큰 텍스처는 메모리를 많이 사용하지만 더 선명합니다
    - **MIME Type**: image/png, image/jpeg 등
    - **최적화 팁**: 모바일/웹용은 2048×2048 이하 권장
    """)
    if gltf.images:
        for i, img in enumerate(gltf.images):
            by = image_bytes_from_gltf(gltf, img, bb)
            wh = image_size_from_bytes(by) if by else None
            mime = img.mimeType or "unknown"
            if wh:
                st.write(f"**Image {i}** — {wh[0]}×{wh[1]} px · MIME: {mime}")
            else:
                st.write(f"**Image {i}** — (size unknown) · MIME: {mime}")
            # Optionally preview (small)
            if by and wh:
                try:
                    st.image(by, caption=f"Image {i}", use_container_width=False)
                except Exception:
                    pass
    else:
        st.write("No images.")
    if gltf.textures:
        st.caption(f"Textures: {len(gltf.textures)} (texture objects referencing images/samplers)")

# ---------------------------
# Nodes, Scenes (hierarchy)
# ---------------------------
with st.expander("🌳 Scenes & Node Graph", expanded=False):
    st.info("""
    **📖 설명**: Scene과 Node는 3D 모델의 계층 구조를 정의합니다.
    - **Scene**: 렌더링할 전체 장면 (여러 개 가능)
    - **Node**: 씬 그래프의 노드. Mesh, Camera, Light 등을 포함
    - **Transform**: 각 Node의 위치(translation), 회전(rotation), 크기(scale)
    - **Hierarchy**: 부모-자식 관계로 변환이 상속됨 (예: 손→팔→몸통)
    - **Matrix**: 변환을 4×4 행렬로 직접 표현할 수도 있음
    """)
    trees = build_tree(gltf)
    if not trees:
        st.write("No scenes defined.")
    else:
        for scene in trees:
            st.markdown(f"**Scene {scene['sceneIndex']} — {scene['name']}**")
            st.json(scene)

# ---------------------------
# Animations & Skins
# ---------------------------
# 지금은 필요하지 않으므로 주석 처리
# with st.expander("🕺 Animations & 🦴 Skins", expanded=False):
#     st.info("""
#     **📖 설명**: Animation과 Skin은 모델의 움직임을 정의합니다.
    
#     **Animation**:
#     - **Channels**: 어떤 노드의 어떤 속성(위치/회전/크기)을 애니메이션할지 정의
#     - **Samplers**: 시간에 따른 값의 변화(키프레임 데이터)
#     - **Path**: translation(이동), rotation(회전), scale(크기), weights(모프타겟)
    
#     **Skin (골격 애니메이션)**:
#     - **Joints**: 뼈대(bone) 노드들의 리스트
#     - **Skeleton**: 루트 뼈대 노드
#     - **InverseBindMatrices**: 각 뼈의 초기 변환 역행렬
#     - 캐릭터 애니메이션에 주로 사용
#     """)
#     if gltf.animations:
#         for i, anim in enumerate(gltf.animations):
#             st.markdown(f"**Animation {i} — {anim.name or 'unnamed'}**")
#             ch_cnt = len(anim.channels or [])
#             sm_cnt = len(anim.samplers or [])
#             st.write(f"- Channels: {ch_cnt}, Samplers: {sm_cnt}")
#             # list targets
#             for ci, ch in enumerate(anim.channels or []):
#                 tgt = ch.target
#                 st.write(f"  · Channel {ci}: node={tgt.node}, path={tgt.path}")
#             st.divider()
#     else:
#         st.write("No animations.")

#     if gltf.skins:
#         for i, sk in enumerate(gltf.skins):
#             st.markdown(f"**Skin {i} — {sk.name or 'unnamed'}**")
#             st.write(f"- Joints: {len(sk.joints or [])}")
#             st.write(f"- Skeleton root: {sk.skeleton}")
#             st.divider()
#     else:
#         st.caption("No skins.")

# ---------------------------
# Accessors & BufferViews
# ---------------------------
with st.expander("📦 Accessors & BufferViews (타입/컴포넌트/카운트/min/max)", expanded=False):
    st.info("""
    **📖 설명**: GLB 파일의 데이터 접근 메커니즘입니다 (저수준 정보).
    
    **Buffer**: 실제 바이너리 데이터 덩어리 (정점, 인덱스, 애니메이션 데이터 등)
    
    **BufferView**: Buffer의 특정 구간을 가리킴
    - byteOffset: 시작 위치
    - byteLength: 데이터 크기
    - byteStride: 데이터 간격 (인터리빙용)
    - target: 용도 (34962=ARRAY_BUFFER, 34963=ELEMENT_ARRAY_BUFFER)
    
    **Accessor**: BufferView의 데이터를 해석하는 방법
    - type: SCALAR(1개), VEC2(2개), VEC3(3개), VEC4(4개), MAT4(16개) 등
    - componentType: FLOAT, UNSIGNED_SHORT, UNSIGNED_INT 등
    - count: 데이터 개수
    - min/max: 바운딩 박스 계산에 사용
    """)
    if gltf.accessors:
        st.write(f"**Accessors: {len(gltf.accessors)}**")
        for i, acc in enumerate(gltf.accessors):
            t = acc.type
            ct = COMPONENT_TYPE_MAP.get(acc.componentType, str(acc.componentType))
            st.write(
                f"- Acc {i}: type={t}, component={ct}, count={acc.count}, "
                f"normalized={bool(acc.normalized)}, min={getattr(acc, 'min', None)}, max={getattr(acc, 'max', None)}, "
                f"bufferView={acc.bufferView}"
            )
    else:
        st.write("No accessors.")
    st.divider()
    if gltf.bufferViews:
        st.write(f"**BufferViews: {len(gltf.bufferViews)}**")
        for i, bv in enumerate(gltf.bufferViews):
            st.write(
                f"- BV {i}: buffer={bv.buffer}, byteOffset={bv.byteOffset or 0}, byteLength={bv.byteLength}, "
                f"byteStride={bv.byteStride}, target={bv.target}"
            )
    else:
        st.write("No bufferViews.")
    if gltf.buffers:
        total = sum([(b.byteLength or 0) for b in gltf.buffers])
        st.caption(f"Buffers: {len(gltf.buffers)}, total size ~ {sizeof_fmt(total)}")

# ---------------------------
# Cameras & Lights
# ---------------------------
# 지금은 필요하지 않으므로 주석 처리
# with st.expander("📷 Cameras & 💡 Lights", expanded=False):
#     st.info("""
#     **📖 설명**: 씬의 카메라와 조명 설정입니다.
    
#     **Camera**:
#     - **Perspective**: 원근 카메라 (일반적인 3D 뷰)
#       - yfov: 수직 시야각 (라디안)
#       - znear/zfar: 렌더링 범위
#       - aspectRatio: 가로/세로 비율
#     - **Orthographic**: 직교 카메라 (2D, CAD 뷰)
#       - xmag/ymag: 뷰포트 크기
    
#     **Lights (KHR_lights_punctual 확장)**:
#     - **directional**: 태양광 같은 방향성 조명
#     - **point**: 전구 같은 점 조명
#     - **spot**: 손전등 같은 스포트라이트
#     - intensity: 밝기
#     - color: RGB 색상
#     - range: 조명 범위
#     """)
#     if gltf.cameras:
#         for i, cam in enumerate(gltf.cameras):
#             st.markdown(f"**Camera {i} — {cam.type}**")
#             if cam.perspective:
#                 p = cam.perspective
#                 st.write(f"- Perspective: yfov={p.yfov}, znear={p.znear}, zfar={p.zfar}, aspectRatio={p.aspectRatio}")
#             if cam.orthographic:
#                 o = cam.orthographic
#                 st.write(f"- Orthographic: xmag={o.xmag}, ymag={o.ymag}, znear={o.znear}, zfar={o.zfar}")
#     else:
#         st.write("No cameras.")

#     # KHR_lights_punctual
#     lights = None
#     try:
#         if gltf.extensions and "KHR_lights_punctual" in gltf.extensions:
#             lights = gltf.extensions["KHR_lights_punctual"].get("lights", None)
#     except Exception:
#         lights = None
#     if lights:
#         st.markdown("**KHR_lights_punctual:**")
#         for i, l in enumerate(lights):
#             st.write(f"- Light {i}: type={l.get('type')}, color={l.get('color')}, intensity={l.get('intensity')}, range={l.get('range')}")
#     else:
#         st.caption("No punctual lights (KHR_lights_punctual).")

# ---------------------------
# Extensions & Extras
# ---------------------------
# 지금은 필요하지 않으므로 주석 처리
# with st.expander("🧩 Extensions / Extras / Asset", expanded=False):
#     st.info("""
#     **📖 설명**: glTF 2.0의 확장 기능과 메타데이터입니다.
    
#     **Extensions**:
#     - glTF 표준을 확장하는 추가 기능들
#     - 예: KHR_lights_punctual (조명), KHR_materials_unlit (발광 재질), 
#           KHR_draco_mesh_compression (압축), KHR_texture_basisu (고효율 텍스처)
#     - extensionsUsed: 사용된 확장 목록
#     - extensionsRequired: 필수 확장 (없으면 렌더링 불가)
    
#     **Asset**:
#     - version: glTF 버전 (보통 "2.0")
#     - generator: 파일을 만든 도구 (Blender, Maya 등)
#     - copyright: 저작권 정보
    
#     **Extras**: 커스텀 메타데이터 (애플리케이션별 추가 정보)
#     """)
#     st.write(f"extensionsUsed: {extensions_used}")
#     st.write(f"extensionsRequired: {extensions_req}")
#     if gltf.extensions:
#         st.write("Top-level extensions object:")
#         try:
#             # Convert non-serializable objects to str
#             st.json(json.loads(json.dumps(gltf.extensions, default=str)))
#         except Exception:
#             st.write(str(gltf.extensions))
#     asset = gltf.asset
#     if asset:
#         st.write(f"Asset: version={asset.version}, generator={asset.generator}, copyright={asset.extras if hasattr(asset, 'extras') else None}")

# ---------------------------
# Quality quick checks
# ---------------------------
with st.expander("✅ Quick Checks (간단 점검)", expanded=False):
    st.info("""
    **📖 설명**: 3D 모델의 최적화 상태를 간단히 체크합니다.
    
    **체크 항목**:
    - **삼각형 수**: 너무 많으면 성능 저하 (모바일: ~10만, PC: ~100만 권장)
    - **텍스처 크기**: 4K 이상은 메모리 부담 (모바일: 2K 이하 권장)
    - **드로우 콜**: 많을수록 CPU 부담 증가 (배칭/머징으로 줄이기)
    - **최적화 팁**:
      - LOD (Level of Detail) 사용
      - 텍스처 아틀라스로 드로우 콜 감소
      - 불필요한 정점/폴리곤 제거
      - 압축 확장 사용 (Draco, KTX2)
    """)
    hints = []
    if total_triangles > 1_000_000:
        hints.append("삼각형 수가 100만 개 이상입니다. 실시간 렌더링에 무거울 수 있어요.")
    if num_images > 0:
        # try to detect very large textures
        large_tex = []
        for i, img in enumerate(gltf.images or []):
            by = image_bytes_from_gltf(gltf, img, bb)
            wh = image_size_from_bytes(by) if by else None
            if wh and max(wh) >= 4096:
                large_tex.append((i, wh))
        if large_tex:
            hints.append(f"아주 큰 텍스처가 있습니다: {', '.join([f'Image {i} {w}x{h}' for i,(w,h) in large_tex])}")
    if draw_calls > 200:
        hints.append("프리미티브(드로우콜)가 많습니다. 머지/배칭을 고려해보세요.")
    if not hints:
        st.write("특별히 문제로 보이는 항목이 없습니다 👍")
    else:
        for h in hints:
            st.write("- " + h)

st.success("분석 완료. 상단/아코디언에서 세부 항목을 확인하세요.")
