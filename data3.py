import datetime
import os
import tempfile

import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from matplotlib.figure import Figure


FONT_PATH = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")

try:
    fm.fontManager.addfont(FONT_PATH)
    mpl.rc("font", family=fm.FontProperties(fname=FONT_PATH).get_name())
    mpl.rc("axes", unicode_minus=False)
except Exception:
    pass


DEFAULT_CHARACTER = "마리오"
PALETTE = {
    0: (245, 248, 255),
    1: (210, 60, 60),
    2: (255, 220, 180),
    3: (120, 80, 40),
    4: (70, 110, 220),
    5: (245, 215, 70),
    6: (35, 35, 35),
    7: (244, 143, 177),
    8: (255, 255, 255),
    9: (156, 108, 196),
    10: (255, 167, 38),
    11: (129, 199, 132),
    12: (79, 195, 247),
}

CHARACTERS = {
    "마리오": {
        "note": "기본 캐릭터입니다. 모자, 얼굴, 옷 색이 뚜렷해서 RGB 읽기 좋습니다.",
        "pattern": np.array([[0,0,1,1,1,1,0,0],[0,1,1,1,1,1,1,0],[0,0,3,2,2,3,0,0],[0,3,2,2,2,2,3,0],[0,0,1,4,4,1,0,0],[0,1,4,5,5,4,1,0],[0,3,4,4,4,4,3,0],[0,0,3,0,0,3,0,0]], dtype=int),
    },
    "커비": {
        "note": "둥근 실루엣이 분명해서 8x8에서도 가장 안정적으로 보이는 캐릭터입니다.",
        "pattern": np.array([[0,0,7,7,7,7,0,0],[0,7,7,7,7,7,7,0],[7,7,7,7,7,7,7,7],[7,7,6,7,7,6,7,7],[7,7,7,1,1,7,7,7],[7,7,7,7,7,7,7,7],[0,1,7,7,7,7,1,0],[0,0,1,0,0,1,0,0]], dtype=int),
    },
    "팩맨 유령": {
        "note": "실루엣과 눈이 단순해서 필터로 선과 경계를 찾는 활동에 특히 잘 어울립니다.",
        "pattern": np.array([[0,0,12,12,12,12,0,0],[0,12,12,12,12,12,12,0],[12,12,8,6,8,6,12,12],[12,12,8,6,8,6,12,12],[12,12,12,12,12,12,12,12],[12,12,12,12,12,12,12,12],[12,12,12,12,12,12,12,12],[12,0,12,0,12,0,12,0]], dtype=int),
    },
    "피카츄": {
        "note": "노란색 중심 캐릭터라 밝기 조절과 필터 반응을 직관적으로 볼 수 있습니다.",
        "pattern": np.array([[6,5,0,0,0,0,5,6],[6,5,5,0,0,5,5,6],[0,5,5,5,5,5,5,0],[5,5,6,5,5,6,5,5],[5,5,5,1,1,5,5,5],[0,5,5,5,5,5,5,0],[0,3,5,5,5,5,3,0],[0,0,3,0,0,3,0,0]], dtype=int),
    },
}

BINARY_PATTERNS = {
    "계단": np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
    "체스판": np.array(
        [[(row + col) % 2 for col in range(6)] for row in range(6)],
        dtype=int,
    ),
    "원": np.array(
        [
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
}

FILTERS = {
    "세로선 찾기": (np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=float), "세로선"),
    "가로선 찾기": (np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=float), "가로선"),
    "경계 찾기": (np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float), "경계나 모서리"),
}

PORT_URLS = {"1": "https://padlet.com/ps0andd/p_1", "2": "https://padlet.com/ps0andd/p_2", "5": "https://padlet.com/ps0andd/p_5", "6": "https://padlet.com/ps0andd/p_6"}
QA_URLS = {"1": "https://padlet.com/ps0andd/q_1", "2": "https://padlet.com/ps0andd/q_2", "5": "https://padlet.com/ps0andd/q_5", "6": "https://padlet.com/ps0andd/q_6"}

RGB_PALETTE_20 = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (255, 105, 180),
    (139, 69, 19),
    (255, 220, 180),
]

ART_GRID_SIZE = 5


def rgb_from_pattern(pattern):
    image = np.zeros((pattern.shape[0], pattern.shape[1], 3), dtype=np.uint8)
    for index, color in PALETTE.items():
        image[pattern == index] = color
    return image


def current_character():
    selected = st.session_state.get("i3_character", DEFAULT_CHARACTER)
    if selected not in CHARACTERS:
        st.session_state["i3_character"] = DEFAULT_CHARACTER
        return DEFAULT_CHARACTER
    return selected


def base_image(name=None):
    return rgb_from_pattern(CHARACTERS[name or current_character()]["pattern"])


def base_channels(name=None):
    image = base_image(name)
    return image[:, :, 0].astype(int), image[:, :, 1].astype(int), image[:, :, 2].astype(int)


def df_from(array):
    return pd.DataFrame(array.astype(int), index=range(1, array.shape[0] + 1), columns=range(1, array.shape[1] + 1))


def clean_df(frame):
    df = pd.DataFrame(frame).apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.clip(0, 255).round().astype(int)
    df.index = range(1, df.shape[0] + 1)
    df.columns = range(1, df.shape[1] + 1)
    return df


def reset_character(name):
    r, g, b = base_channels(name)
    st.session_state["i3_r_df"] = df_from(r)
    st.session_state["i3_g_df"] = df_from(g)
    st.session_state["i3_b_df"] = df_from(b)
    st.session_state["i3_scale"] = 1.0
    st.session_state["i3_add_r"] = 0
    st.session_state["i3_add_g"] = 0
    st.session_state["i3_add_b"] = 0
    st.session_state["i3_character_applied"] = name


def ensure_state():
    st.session_state.setdefault("i3_character", DEFAULT_CHARACTER)
    if (
        "i3_r_df" not in st.session_state
        or "i3_g_df" not in st.session_state
        or "i3_b_df" not in st.session_state
        or st.session_state.get("i3_character_applied") != current_character()
    ):
        reset_character(current_character())
    st.session_state.setdefault("i3_binary_shape", "계단")
    if (
        "i3_binary_grid" not in st.session_state
        or st.session_state.get("i3_binary_shape_applied") != st.session_state.get("i3_binary_shape", "계단")
    ):
        st.session_state["i3_binary_grid"] = BINARY_PATTERNS[st.session_state.get("i3_binary_shape", "계단")].copy()
        st.session_state["i3_binary_shape_applied"] = st.session_state.get("i3_binary_shape", "계단")
    st.session_state.setdefault("i3_filter_name", "세로선 찾기")
    st.session_state.setdefault("i3_filter_source", "기본 픽셀 아트")
    st.session_state.setdefault("i3_hypothesis_checked", False)
    st.session_state.setdefault("i3_dl_hidden1", 3)
    st.session_state.setdefault("i3_dl_hidden2", 3)
    st.session_state["i3_dl_hidden1"] = int(np.clip(st.session_state.get("i3_dl_hidden1", 3), 2, 5))
    st.session_state["i3_dl_hidden2"] = int(np.clip(st.session_state.get("i3_dl_hidden2", 3), 2, 5))
    st.session_state.setdefault("i3_art_selected_color", 1)
    if int(st.session_state.get("i3_art_selected_color", 1)) >= len(RGB_PALETTE_20):
        st.session_state["i3_art_selected_color"] = 1
    if (
        "i3_art_grid" not in st.session_state
        or np.array(st.session_state["i3_art_grid"]).shape != (ART_GRID_SIZE, ART_GRID_SIZE)
    ):
        st.session_state["i3_art_grid"] = np.zeros((ART_GRID_SIZE, ART_GRID_SIZE), dtype=int)
    else:
        st.session_state["i3_art_grid"] = np.clip(
            np.array(st.session_state["i3_art_grid"], dtype=int),
            0,
            len(RGB_PALETTE_20) - 1,
        )
    st.session_state.setdefault("i3_show_rgb_matrices", False)
    for idx in range(1, 5):
        st.session_state.setdefault(f"i3_saved_{idx}", "")
        st.session_state.setdefault(f"i3_saved_time_{idx}", "")
        st.session_state.setdefault(f"i3_saved_detail_{idx}", {})


def edited_image():
    st.session_state["i3_r_df"] = clean_df(st.session_state["i3_r_df"])
    st.session_state["i3_g_df"] = clean_df(st.session_state["i3_g_df"])
    st.session_state["i3_b_df"] = clean_df(st.session_state["i3_b_df"])
    return np.stack([st.session_state["i3_r_df"].values, st.session_state["i3_g_df"].values, st.session_state["i3_b_df"].values], axis=2).clip(0, 255).astype(np.uint8)


def transformed_image():
    image = edited_image().astype(float)
    added = np.zeros_like(image)
    added[:, :, 0] = int(st.session_state.get("i3_add_r", 0))
    added[:, :, 1] = int(st.session_state.get("i3_add_g", 0))
    added[:, :, 2] = int(st.session_state.get("i3_add_b", 0))
    return np.clip(float(st.session_state.get("i3_scale", 1.0)) * image + added, 0, 255).astype(np.uint8)


def to_gray(image):
    return np.round(0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(float)


def convolve_same(image, kernel):
    h, w = kernel.shape
    pad = np.pad(image, ((h // 2, h // 2), (w // 2, w // 2)), mode="constant")
    out = np.zeros_like(image, dtype=float)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            out[row, col] = np.sum(pad[row : row + h, col : col + w] * kernel)
    return out


def filter_result():
    source_name = st.session_state.get("i3_filter_source", "기본 픽셀 아트")
    image = {"기본 픽셀 아트": base_image(), "현재 편집한 이미지": edited_image(), "변환 결과 이미지": transformed_image()}[source_name]
    kernel, feature = FILTERS[st.session_state.get("i3_filter_name", "세로선 찾기")]
    gray = to_gray(image)
    response = np.abs(convolve_same(gray, kernel))
    display = np.zeros_like(response) if float(response.max()) == 0 else response / float(response.max()) * 255
    peak = np.unravel_index(int(np.argmax(response)), response.shape)
    return image, gray, kernel, feature, response, display, int(peak[0] + 1), int(peak[1] + 1)


def draw_image(image, title, cmap=None, show_values=False):
    fig = Figure(figsize=(4.0, 4.0))
    ax = fig.subplots()
    if cmap:
        ax.imshow(image, cmap=cmap, interpolation="nearest", vmin=0, vmax=255)
    else:
        ax.imshow(image, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(image.shape[1]))
    ax.set_yticks(range(image.shape[0]))
    ax.set_xticklabels(range(1, image.shape[1] + 1))
    ax.set_yticklabels(range(1, image.shape[0] + 1))
    ax.set_xticks(np.arange(-0.5, image.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, image.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    if show_values:
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                val = int(image[row, col])
                ax.text(col, row, str(val), ha="center", va="center", fontsize=6, color="white" if val > 145 else "black")
    fig.tight_layout()
    return fig


def upscaled_display_image(image, cmap=None, pixel_size=28):
    arr = np.array(image)
    if arr.ndim == 2:
        clipped = np.clip(arr.astype(float), 0, 255) / 255.0
        if cmap:
            rgb = (mpl.colormaps[cmap](clipped)[:, :, :3] * 255).astype(np.uint8)
        else:
            gray = np.clip(arr, 0, 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=2)
    else:
        rgb = np.clip(arr, 0, 255).astype(np.uint8)

    enlarged = np.repeat(np.repeat(rgb, pixel_size, axis=0), pixel_size, axis=1)
    line_color = np.array([235, 235, 235], dtype=np.uint8)
    enlarged[::pixel_size, :, :] = line_color
    enlarged[:, ::pixel_size, :] = line_color
    return enlarged


def layer_positions(count, top=0.82, bottom=0.18):
    if int(count) <= 1:
        return [0.5]
    return np.linspace(top, bottom, int(count)).tolist()


def draw_deep_learning_structure(hidden1=3, hidden2=3):
    input_nodes = 3
    output_nodes = 1
    fig = Figure(figsize=(7.6, 3.6))
    ax = fig.subplots()
    ax.axis("off")

    layers = [
        {"x": 0.12, "ys": layer_positions(input_nodes), "label": "입력층", "color": "#bbdefb", "edge": "#1565c0"},
        {"x": 0.36, "ys": layer_positions(hidden1), "label": f"1층\n{hidden1}개", "color": "#ffe082", "edge": "#ef6c00"},
        {"x": 0.60, "ys": layer_positions(hidden2), "label": f"2층\n{hidden2}개", "color": "#c8e6c9", "edge": "#2e7d32"},
        {"x": 0.84, "ys": layer_positions(output_nodes), "label": "출력층", "color": "#f8bbd0", "edge": "#c2185b"},
    ]

    total_connections = 0
    for left, right in zip(layers, layers[1:]):
        total_connections += len(left["ys"]) * len(right["ys"])
        for y1 in left["ys"]:
            for y2 in right["ys"]:
                ax.plot([left["x"], right["x"]], [y1, y2], color="#cfd8dc", linewidth=0.8, alpha=0.8, zorder=1)

    for layer in layers:
        node_size = max(180, min(420, 1500 / max(len(layer["ys"]), 1)))
        for y in layer["ys"]:
            ax.scatter(
                layer["x"],
                y,
                s=node_size,
                color=layer["color"],
                edgecolors=layer["edge"],
                linewidths=1.2,
                zorder=3,
                clip_on=False,
            )
        ax.text(layer["x"], 0.06, layer["label"], ha="center", va="center", fontsize=10, fontweight="bold")

    ax.text(0.12, 0.92, "RGB 숫자", ha="center", fontsize=10, color="#1565c0")
    ax.text(0.36, 0.92, "선과 경계", ha="center", fontsize=10, color="#ef6c00")
    ax.text(0.60, 0.92, "모양 조각", ha="center", fontsize=10, color="#2e7d32")
    ax.text(0.84, 0.92, "전체 판단", ha="center", fontsize=10, color="#c2185b")
    ax.set_xlim(0.02, 0.94)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("딥러닝의 다층 구조", fontsize=13, fontweight="bold", pad=10)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.10)
    return fig, total_connections


def render_stage_cards(big_concept, essential_question, output_text):
    cards = [
        ("개념", big_concept, "#e3f2fd", "#1565c0"),
        ("본질 질문", essential_question, "#fff8e1", "#ef6c00"),
        ("오늘의 산출물", output_text, "#e8f5e9", "#2e7d32"),
    ]
    cols = st.columns(3)
    for col, (title, body, bg, border) in zip(cols, cards):
        col.markdown(
            f"""
            <div style="height:100%; padding:14px 16px; border-radius:14px; background:{bg}; border:1px solid {border};">
                <div style="font-weight:700; color:{border}; margin-bottom:6px;">{title}</div>
                <div style="line-height:1.6; font-size:0.96rem; color:#263238;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def apply_local_style():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        div[data-baseweb="tab"] {
            background: #f4f8fc;
            border-radius: 0.8rem;
            padding: 0.45rem 0.9rem;
            border: 1px solid #dbe7f3;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #e8f3ff;
            border-color: #90caf9;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #e5eef7;
            border-radius: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pretty_title(text, color1, color2):
    return f"""
    <div style='
        background: linear-gradient(90deg, {color1} 0%, {color2} 100%);
        border-radius: 18px;
        box-shadow: 0 2px 8px 0 rgba(33,150,243,0.06);
        padding: 4px 18px 0px 18px;
        margin-bottom: 10px;'>
        <h4 style='margin-top:0;'><b>{text}</b></h4>
    </div>
    """


def page_banner(title, description):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #e3f2fd 0%, #d1c4e9 100%);
            border-radius: 22px;
            padding: 22px 24px;
            box-shadow: 0 8px 20px rgba(33, 150, 243, 0.10);
            border: 1px solid #dbe7f3;
            margin-bottom: 14px;
        ">
            <div style="font-size:0.9rem; font-weight:700; color:#5e35b1; margin-bottom:8px;">F.U.T.U.R.E. 프로젝트 3DAY</div>
            <div style="font-size:1.9rem; font-weight:800; color:#1f2937; margin-bottom:8px;">{title}</div>
            <div style="font-size:1rem; line-height:1.7; color:#37474f;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_intro(title, description, question, color1="#e8f5e9", color2="#c8e6c9"):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
            border-radius: 18px;
            padding: 18px 20px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.06);
            margin-bottom: 12px;
        ">
            <div style="font-size:1.05rem; font-weight:800; color:#1f2937; margin-bottom:8px;">{title}</div>
            <div style="font-size:0.97rem; line-height:1.7; color:#37474f; margin-bottom:12px;">{description}</div>
            <div style="
                background: rgba(255,255,255,0.72);
                border-radius: 12px;
                padding: 10px 12px;
                border: 1px solid rgba(255,255,255,0.85);
                color:#37474f;
                line-height:1.6;
            ">
                <b>핵심 탐구 질문</b><br>{question}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_value_card(item):
    title = item.get("title", "")
    value = item.get("value", "")
    detail = item.get("detail", "")
    bg = item.get("bg", "#ffffff")
    border = item.get("border", "#dbe7f3")
    st.markdown(
        f"""
        <div style="
            height:100%;
            padding:14px 16px;
            border-radius:16px;
            background:{bg};
            border:1px solid {border};
            box-shadow:0 2px 8px rgba(33, 150, 243, 0.08);
            margin-bottom:8px;
        ">
            <div style="font-size:0.92rem; color:#546e7a; margin-bottom:6px; font-weight:600;">{title}</div>
            <div style="font-size:1.25rem; color:#263238; font-weight:700; margin-bottom:4px;">{value}</div>
            <div style="font-size:0.86rem; color:#607d8b; line-height:1.5;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_value_cards(items, columns=1):
    if columns <= 1:
        for item in items:
            _render_value_card(item)
        return

    for start in range(0, len(items), columns):
        row_items = items[start:start + columns]
        row_cols = st.columns(columns)
        for col, item in zip(row_cols, row_items):
            with col:
                _render_value_card(item)
        for col in row_cols[len(row_items):]:
            with col:
                st.empty()


def average_pool(image, block_size=2):
    arr = np.array(image, dtype=float)
    pooled_h = arr.shape[0] // block_size
    pooled_w = arr.shape[1] // block_size
    trimmed = arr[: pooled_h * block_size, : pooled_w * block_size]
    return trimmed.reshape(pooled_h, block_size, pooled_w, block_size).mean(axis=(1, 3))


def toggle_binary_cell(row, col):
    grid = st.session_state["i3_binary_grid"].copy()
    grid[row, col] = 1 - int(grid[row, col])
    st.session_state["i3_binary_grid"] = grid


def binary_grid_text(grid):
    return " / ".join("".join(str(int(value)) for value in row) for row in grid)


def matrix_text(grid):
    return " / ".join(",".join(str(int(value)) for value in row) for row in grid)


def binary_matrix_frame(grid):
    return pd.DataFrame(
        grid.astype(int),
        index=range(1, grid.shape[0] + 1),
        columns=range(1, grid.shape[1] + 1),
    )


def character_gray_matrix(name):
    return np.clip(255 - to_gray(base_image(name)), 0, 255).astype(int)


def combine_gray_matrices(name_a, name_b, k_value):
    matrix_a = character_gray_matrix(name_a).astype(float)
    matrix_b = character_gray_matrix(name_b).astype(float)
    result = k_value * matrix_a + (1 - k_value) * matrix_b
    return matrix_a.astype(int), matrix_b.astype(int), np.clip(result, 0, 255).round().astype(int)


def art_image():
    index_grid = np.array(st.session_state["i3_art_grid"], dtype=int)
    image = np.zeros((index_grid.shape[0], index_grid.shape[1], 3), dtype=np.uint8)
    for idx, rgb in enumerate(RGB_PALETTE_20):
        image[index_grid == idx] = rgb
    return image


def select_art_color(index):
    st.session_state["i3_art_selected_color"] = int(index)


def art_cell_label(index):
    labels = {
        0: "⬜",
        1: "⬛",
        2: "🟥",
        3: "🟩",
        4: "🟦",
        5: "🟨",
        6: "🟪",
        7: "🟦",
        8: "🟧",
        9: "🟪",
        10: "🟫",
        11: "🟨",
    }
    return labels.get(int(index), "⬜")


def paint_art_cell(row, col):
    grid = np.array(st.session_state["i3_art_grid"], dtype=int)
    grid[row, col] = int(st.session_state.get("i3_art_selected_color", 1))
    st.session_state["i3_art_grid"] = grid
    st.session_state["i3_show_rgb_matrices"] = False


def clear_art_grid():
    st.session_state["i3_art_grid"] = np.zeros((ART_GRID_SIZE, ART_GRID_SIZE), dtype=int)
    st.session_state["i3_show_rgb_matrices"] = False


def art_channel_frames():
    image = art_image()
    row_index = range(1, image.shape[0] + 1)
    col_index = range(1, image.shape[1] + 1)
    return (
        pd.DataFrame(image[:, :, 0].astype(int), index=row_index, columns=col_index),
        pd.DataFrame(image[:, :, 1].astype(int), index=row_index, columns=col_index),
        pd.DataFrame(image[:, :, 2].astype(int), index=row_index, columns=col_index),
    )


def render_clickable_binary_grid():
    grid = st.session_state["i3_binary_grid"]
    for row in range(grid.shape[0]):
        cols = st.columns(grid.shape[1], gap="small")
        for col in range(grid.shape[1]):
            label = "⬛" if int(grid[row, col]) == 1 else "⬜"
            cols[col].button(
                label,
                key=f"i3_binary_btn_{row}_{col}",
                on_click=toggle_binary_cell,
                args=(row, col),
                use_container_width=True,
            )


def save_activity_result(index, summary, details=None):
    st.session_state[f"i3_saved_{index}"] = summary
    st.session_state[f"i3_saved_time_{index}"] = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state[f"i3_saved_detail_{index}"] = details or {}


def saved_status_text(index):
    saved_time = st.session_state.get(f"i3_saved_time_{index}", "")
    return f"저장 완료: {saved_time}" if saved_time else "아직 저장하지 않았습니다."


def normalize_pdf_output(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("latin1")
    return bytes(value)


def matrix_to_pdf_text(title, matrix):
    arr = np.array(matrix).astype(int)
    lines = [", ".join(str(int(value)) for value in row) for row in arr]
    return f"{title}\n" + "\n".join(lines)


def student_text_or_default(text, default="작성 내용 없음"):
    value = str(text).strip() if text is not None else ""
    return value if value else default


def add_text_box_to_pdf(pdf, title, text, fill_color=(245, 245, 245)):
    pdf.set_fill_color(*fill_color)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 7, title, ln=1, fill=True)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, student_text_or_default(text))
    pdf.ln(1)


def add_array_image_to_pdf(pdf, title, image, cmap=None):
    tmp_path = None
    fig = draw_image(np.array(image), title, cmap)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = tmp.name
        fig.savefig(tmp_path, format="png", dpi=180, bbox_inches="tight")
        display_w = 85
        display_h = 85
        if pdf.get_y() + display_h > pdf.h - 20:
            pdf.add_page()
        y = pdf.get_y()
        x = (pdf.w - display_w) / 2
        pdf.image(tmp_path, x=x, y=y, w=display_w)
        pdf.set_y(y + display_h + 3)
        pdf.set_x(pdf.l_margin)
    finally:
        fig.clear()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(25, 118, 210)
        self.rect(0, 0, self.w, 20, "F")
        self.set_xy(10, 5)
        self.set_text_color(255, 255, 255)
        self.set_font("Nanum", "", 16)
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 3차시 포트폴리오", ln=1, align="C")
        self.set_text_color(33, 33, 33)
        self.ln(10)


def create_pdf(student, rows, reflection_entries, social_meaning, action_plan, group_question):
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Nanum", "", FONT_PATH, uni=True)
    pdf.set_font("Nanum", "", 11)
    pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 7, f"모둠명: {student['group']}\n학번: {student['id']}\n이름: {student['name']}\n캐릭터: {student['character']}\n작성일: {datetime.datetime.now():%Y-%m-%d}")
    pdf.ln(2)
    for row in rows:
        title = row.get("title", "")
        body = row.get("body", "")
        details = row.get("details", {})
        pdf.set_fill_color(227, 242, 253)
        pdf.cell(0, 8, title, ln=1, fill=True)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, body)
        for writing_title, writing_value in details.get("writings", []):
            add_text_box_to_pdf(pdf, writing_title, writing_value)
        for matrix_title, matrix_value in details.get("matrices", []):
            pdf.set_font("Nanum", "", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, matrix_to_pdf_text(matrix_title, matrix_value))
            pdf.ln(1)
            pdf.set_font("Nanum", "", 11)
        for image_title, image_value, cmap in details.get("images", []):
            add_array_image_to_pdf(pdf, image_title, image_value, cmap)
        pdf.ln(1)
    if any(str(text).strip() for _, text in reflection_entries):
        pdf.set_fill_color(227, 242, 253)
        pdf.cell(0, 8, "문제 4. 결과 해석에서 내가 쓴 글", ln=1, fill=True)
        for title, text in reflection_entries:
            add_text_box_to_pdf(pdf, title, text)
    pdf.cell(0, 8, "모둠의 생각 1. 사회에 주는 시사점", ln=1, fill=True)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, student_text_or_default(social_meaning))
    pdf.ln(1)
    pdf.cell(0, 8, "모둠의 생각 2. 우리의 실천 제안", ln=1, fill=True)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, student_text_or_default(action_plan))
    pdf.ln(1)
    pdf.cell(0, 8, "모둠 심화 질문", ln=1, fill=True)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, student_text_or_default(group_question))
    return normalize_pdf_output(pdf.output(dest="S"))


def practice_rows():
    titles = [
        "문제 1. 이진 행렬로 이미지 인식 이해",
        "문제 2. 행렬의 합, 차, 실수배",
        "문제 3. RGB 행렬과 나의 픽셀 아트",
    ]
    return [
        {
            "title": title,
            "body": st.session_state.get(f"i3_saved_{idx}", "") or "아직 결과 저장 버튼을 누르지 않았습니다.",
            "details": st.session_state.get(f"i3_saved_detail_{idx}", {}),
        }
        for idx, title in enumerate(titles, start=1)
    ]


def reflection_entries():
    return [
        ("이번 활동의 핵심", st.session_state.get("i3_reflect_core", "")),
        ("결과를 그대로 믿기 어려운 이유", st.session_state.get("i3_reflect_limit", "")),
        ("더 알고 싶은 점", st.session_state.get("i3_reflect_next", "")),
    ]


def reflection_status_text():
    return "입력 완료" if any(str(text).strip() for _, text in reflection_entries()) else "아직 작성하지 않았습니다."


def run():
    apply_local_style()
    ensure_state()
    page_banner(
        "이미지를 행렬로 보는 인공지능",
        "픽셀 아트를 고르고, RGB 행렬을 읽고 바꾸고 변환해 보며 인공지능이 이미지를 숫자로 이해하는 방식을 익힙니다.",
    )
    st.markdown("<hr style='border:2px solid #2196F3;'>", unsafe_allow_html=True)

    tabs = st.tabs(["1️⃣ [F.U] 문제 발견", "2️⃣ [T] 수학의 언어", "3️⃣ [U] AI 활용", "4️⃣ [R] 결과 해석", "5️⃣ [E] 세상과 연결"])

    with tabs[0]:
        stage_intro(
            "문제 인식 및 숨겨진 데이터 찾기",
            "실생활에서 인공지능이 사진을 읽어 판단하는 상황을 떠올리며, 그림이 어떻게 숫자 데이터로 바뀌는지 탐색하는 과정입니다.",
            "인공지능은 그림을 어떤 숫자 배열로 바꾸어 읽고 있을까?",
            "#e3f2fd",
            "#bbdefb",
        )
        st.markdown(pretty_title("문제 제기: 얼굴 인식 출입문은 사진을 어떻게 구별할까?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write(
            "학교나 휴대폰의 얼굴 인식 기능은 사람의 얼굴 사진을 그냥 '그림'으로만 보는 것이 아니라, "
            "작은 칸들로 나누어 숫자 데이터처럼 읽어들입니다. 즉, 밝고 어두운 정도와 색의 차이를 "
            "숫자로 바꾼 뒤 그 패턴을 비교해서 같은 사람인지 판단합니다."
        )
        st.info(
            "탐구 질문: 인공지능은 사진을 이루는 많은 칸을 어떤 숫자 배열로 바꾸고, 그 숫자 패턴을 어떻게 읽어낼까요?"
        )
        with st.expander("행렬의 정의와 성분 표현 보기", expanded=False):
            st.markdown(
                """
                **행렬의 정의**

                행렬은 숫자를 가로와 세로로 줄 맞추어 놓은 직사각형 배열입니다.
                """
            )
            matrix_left, matrix_right = st.columns([1, 1])
            with matrix_left:
                matrix_example = pd.DataFrame(
                    [[79, 41, 58], [57, 22, 37]],
                    index=["제1행", "제2행"],
                    columns=["제1열", "제2열", "제3열"],
                )
                st.table(matrix_example)
            with matrix_right:
                st.latex(r"A=\begin{bmatrix}79 & 41 & 58 \\ 57 & 22 & 37\end{bmatrix}")
            st.write("위처럼 숫자를 줄 맞추어 놓은 것이 하나의 행렬입니다.행렬 안의 한 칸에 들어 있는 숫자를 **성분**이라고 합니다.예를 들어 2행 3열의 성분은 37이므로, 이를 `a₂₃` 또는 `(2,3)`로 나타낼 수 있습니다.")
            st.latex(r"a_{2,3}=37")
            st.caption("일반적으로 행렬은 A, B, C 같은 대문자로 나타내고, 성분은 aᵢⱼ처럼 행과 열 번호를 붙여 표현합니다.")

        hypothesis = st.text_area(
            "나의 가설 쓰기",
            key="i3_hypothesis",
            height=90,
            placeholder="예: 인공지능은 사진을 작은 칸으로 나누고, 각 칸의 밝기나 색을 숫자로 바꾼 뒤 그 배열을 비교할 것 같다.",
        )
        fu_question = st.text_area(
            "학생 질문 만들기",
            key="i3_fu_question",
            height=80,
            placeholder="예: 얼굴의 어떤 부분이 숫자 패턴으로 더 중요하게 읽힐까?",
        )
        if st.button("가설 확인하기", key="i3_hypothesis_btn"):
            if hypothesis.strip():
                st.session_state["i3_hypothesis_checked"] = True
            else:
                st.warning("가설을 먼저 한 줄이라도 적어 보면 더 좋습니다.")

        if st.session_state.get("i3_hypothesis_checked", False):
            st.success("가설 확인: 인공지능은 사진을 바로 눈으로 보는 대신, 칸마다 숫자를 기록한 행렬로 바꾸어 모양의 특징을 읽을 수 있습니다.")
            st.markdown(pretty_title("작은 그림을 숫자 표로 바꾸기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            st.write(
                "먼저 아주 단순한 흑백 그림으로 시작해 봅시다. 채워진 칸은 1, 비어 있는 칸은 0으로 두면, "
                "작은 그림도 숫자 행렬로 바꿀 수 있습니다. 아래 패턴판을 눌러 보며 그림과 행렬이 함께 바뀌는지 확인해 보세요."
            )

            shape_name = st.selectbox("기본 모양 선택", list(BINARY_PATTERNS.keys()), key="i3_binary_shape")
            binary = st.session_state["i3_binary_grid"]
            filled_count = int(binary.sum())
            render_value_cards(
                [
                    {
                        "title": "현재 기본 모양",
                        "value": shape_name,
                        "detail": "기본 모양을 바꾼 뒤 칸을 직접 눌러 원하는 패턴으로 다시 만들 수 있습니다.",
                        "bg": "#f4f9ff",
                        "border": "#90caf9",
                    },
                    {
                        "title": "행렬 크기",
                        "value": "6×6",
                        "detail": "채워진 칸은 1, 비어 있는 칸은 0으로 읽는 이진 행렬 활동입니다.",
                        "bg": "#fff8e1",
                        "border": "#ffcc80",
                    },
                ],
                columns=2,
            )

            left, right = st.columns([1, 1])
            with left:
                st.markdown(pretty_title(f"클릭하며 만드는 6×6 패턴: {shape_name}", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
                render_clickable_binary_grid()
                st.caption("업로드한 예시처럼 각 칸을 직접 눌러 원하는 패턴으로 바꿔 보세요.")
            with right:
                st.markdown(pretty_title("패턴이 바뀌면 행렬도 함께 바뀝니다", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
                st.table(binary_matrix_frame(binary))

            if st.button("문제 발견 결과 저장", key="i3_save_1"):
                hypothesis_text = hypothesis.strip() if hypothesis.strip() else "가설 미작성"
                save_activity_result(
                    1,
                    f"탐구 질문은 '얼굴 인식 출입문은 사진을 어떻게 구별할까?'였고, "
                    f"내가 세운 가설은 '{hypothesis_text}'입니다. "
                    f"내가 추가로 만든 질문은 '{fu_question.strip() or '질문 미작성'}'입니다. "
                    f"{shape_name} 기본 모양을 바탕으로 직접 6×6 패턴을 만들었고, 최종 행렬은 {binary_grid_text(binary)} 입니다. "
                    f"채워진 칸은 {filled_count}칸, 비어 있는 칸은 {binary.size - filled_count}칸입니다. "
                    f"이 활동을 통해 그림도 숫자 배열로 바꾸면 인공지능이 읽을 수 있다는 점을 확인했습니다.",
                    details={
                        "writings": [
                            ("나의 가설 쓰기", hypothesis_text),
                            ("학생 질문 만들기", fu_question.strip()),
                        ],
                        "matrices": [("6×6 이진 행렬", binary.copy())],
                        "images": [("문제 발견 패턴", (binary * 255).astype(np.uint8), "gray_r")],
                    },
                )
        else:
            st.info("가설 확인하기를 누르면 아래 실습 활동이 이어집니다.")
        st.caption(saved_status_text(1))

    with tabs[1]:
        stage_intro(
            "현상을 수학의 언어로 바꾸기",
            "그림을 행렬로 바꾸고, 행렬의 덧셈·뺄셈·실수배가 이미지 합성과 제거에 어떻게 쓰이는지 수학의 언어로 이해하는 과정입니다.",
            "행렬의 덧셈과 실수배는 그림의 변화와 어떻게 연결될까?",
            "#fff8e1",
            "#ffecb3",
        )
        st.markdown(pretty_title("행렬의 덧셈, 뺄셈, 실수배", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write("행렬의 덧셈과 뺄셈은 **같은 위치에 있는 수끼리** 계산합니다. 즉, 왼쪽 위는 왼쪽 위끼리, 오른쪽 아래는 오른쪽 아래끼리 계산합니다.")
        with st.expander("간단한 예시 보기", expanded=False):
                st.write("먼저 두 행렬을 이렇게 정해 봅시다.")
                st.latex(r"A=\begin{bmatrix}10 & 30 \\ 50 & 20\end{bmatrix},\quad B=\begin{bmatrix}20 & 10 \\ 10 & 40\end{bmatrix}")
                st.write("덧셈은 같은 칸끼리 더합니다. 예를 들어 왼쪽 위는 `10+20`, 오른쪽 위는 `30+10`입니다.")
                st.latex(r"A+B=\begin{bmatrix}10+20 & 30+10 \\ 50+10 & 20+40\end{bmatrix}=\begin{bmatrix}30 & 40 \\ 60 & 60\end{bmatrix}")
                st.write("뺄셈도 같은 규칙입니다. 예를 들어 왼쪽 위는 `10-20`, 오른쪽 아래는 `20-40`입니다.")
                st.latex(r"A-B=\begin{bmatrix}10-20 & 30-10 \\ 50-10 & 20-40\end{bmatrix}=\begin{bmatrix}-10 & 20 \\ 40 & -20\end{bmatrix}")
                st.write("실수배는 행렬 안의 모든 수에 같은 수를 곱하는 것입니다. 밝기를 절반으로 줄이는 것과 비슷합니다.")
                st.latex(r"0.5A=\begin{bmatrix}0.5\times10 & 0.5\times30 \\ 0.5\times50 & 0.5\times20\end{bmatrix}=\begin{bmatrix}5 & 15 \\ 25 & 10\end{bmatrix}")

        st.markdown(pretty_title("캐릭터를 겹치거나 일부 제거해 보기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write("이번에는 두 캐릭터를 섞는 비율을 `kA + (1-k)B` 형태로 살펴봅니다. 같은 자리 성분끼리 계산하되, `k`가 커질수록 A의 영향이 커지고 `k`가 작아질수록 B의 영향이 커집니다.")
        st.write("아래 4가지 캐릭터를 8×8 그레이스케일 행렬로 바꾸어 놓았습니다. 두 캐릭터를 고르고 `k` 슬라이더를 움직이며 두 이미지가 어떻게 섞이는지 관찰해 보세요.")

        choose_left, choose_right = st.columns(2)
        with choose_left:
            char_a = st.selectbox("행렬 A 캐릭터", list(CHARACTERS.keys()), key="i3_gray_char_a")
        with choose_right:
            char_b = st.selectbox("행렬 B 캐릭터", list(CHARACTERS.keys()), index=1, key="i3_gray_char_b")
        k_value = st.slider("k 값 (0 ≤ k ≤ 1)", 0.0, 1.0, 0.5, 0.1, key="i3_gray_k")
        t_question = st.text_input(
            "수학의 언어로 다시 묻기",
            key="i3_t_question",
            placeholder="예: 두 캐릭터를 섞을 때 어떤 자리의 숫자가 더 크게 달라질까?",
        )
        render_value_cards(
            [
                {
                    "title": "행렬 A",
                    "value": char_a,
                    "detail": "k값이 커질수록 A의 모습이 더 강하게 남습니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "행렬 B",
                    "value": char_b,
                    "detail": "k값이 작아질수록 B의 모습이 더 강하게 남습니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "현재 k 값",
                    "value": f"{k_value:.1f}",
                    "detail": "두 행렬을 어떤 비율로 섞을지 조절하는 값입니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
            ],
            columns=3,
        )

        matrix_a, matrix_b, matrix_result = combine_gray_matrices(char_a, char_b, k_value)

        if "i3_gray_show_matrix" not in st.session_state:
            st.session_state["i3_gray_show_matrix"] = False
        toggle_label = "행렬 숨기기" if st.session_state["i3_gray_show_matrix"] else "행렬 보기"

        formula_text = f"합성 행렬 = {k_value:.1f}A + ({1 - k_value:.1f})B"
        result_cols = st.columns([3.3, 1])
        with result_cols[0]:
            st.markdown(pretty_title("합성 행렬", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            st.info(formula_text)
        with result_cols[1]:
            st.markdown(pretty_title("행렬 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            if st.button(toggle_label, key="i3_gray_toggle_matrix", use_container_width=True):
                st.session_state["i3_gray_show_matrix"] = not st.session_state["i3_gray_show_matrix"]
            if not st.session_state["i3_gray_show_matrix"]:
                st.caption("버튼을 누르면 8×8 행렬이 보입니다.")

        preview_cols = st.columns(3)
        with preview_cols[0]:
            st.pyplot(draw_image(matrix_a, f"{char_a} → 행렬 A", "gray_r"), use_container_width=True)
            if st.session_state["i3_gray_show_matrix"]:
                st.table(df_from(matrix_a))
        with preview_cols[1]:
            st.pyplot(draw_image(matrix_b, f"{char_b} → 행렬 B", "gray_r"), use_container_width=True)
            if st.session_state["i3_gray_show_matrix"]:
                st.table(df_from(matrix_b))
        with preview_cols[2]:
            st.pyplot(draw_image(matrix_result, "가중 합성 결과", "gray_r"), use_container_width=True)
            if st.session_state["i3_gray_show_matrix"]:
                st.table(df_from(matrix_result))

        if st.button("합성된 이미지 및 행렬 저장", key="i3_save_2"):
            save_activity_result(
                2,
                f"{char_a}를 행렬 A, {char_b}를 행렬 B로 두고 가중 합성 실습을 했다. "
                f"사용한 식은 {formula_text} 이다. "
                f"활동 중 내가 만든 질문은 '{t_question.strip() or '질문 미작성'}'이다. "
                f"최종 결과 행렬은 {matrix_text(matrix_result)} 이다.",
                details={
                    "writings": [("수학의 언어로 다시 묻기", t_question.strip())],
                    "matrices": [
                        (f"{char_a} 행렬 A", matrix_a.copy()),
                        (f"{char_b} 행렬 B", matrix_b.copy()),
                        ("가중 합성 결과 행렬", matrix_result.copy()),
                    ],
                    "images": [("가중 합성 결과 이미지", matrix_result.copy(), "gray_r")],
                },
            )
        st.caption(saved_status_text(2))

    with tabs[2]:
        stage_intro(
            "AI 도구로 시뮬레이션하기",
            "색 이미지를 R, G, B 세 행렬로 나누어 읽는 방식을 직접 실험하며, AI가 색 이미지를 데이터로 처리하는 과정을 체험하는 단계입니다.",
            "AI는 한 장의 그림을 왜 R, G, B 세 행렬로 나누어 볼까?",
            "#e8f5e9",
            "#c8e6c9",
        )
        st.markdown(pretty_title("RGB 행렬이란?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write("한 칸의 색은 `(R, G, B)` 세 숫자로 표현합니다. 따라서 하나의 색 이미지는 빨강 행렬, 초록 행렬, 파랑 행렬 세 장으로 나누어 볼 수 있습니다.")
        rgb_info_left, rgb_info_right = st.columns([1, 1])
        with rgb_info_left:
            rgb_info_df = pd.DataFrame(
                [
                    {"기호": "R", "색상 의미": "빨강의 세기", "값의 범위": "0 ~ 255"},
                    {"기호": "G", "색상 의미": "초록의 세기", "값의 범위": "0 ~ 255"},
                    {"기호": "B", "색상 의미": "파랑의 세기", "값의 범위": "0 ~ 255"},
                ]
            )
            st.dataframe(rgb_info_df, use_container_width=True, hide_index=True, height=141)
        with rgb_info_right:
            st.latex(r"\text{한 픽셀의 색}=(R,G,B)")
            st.write("예를 들어 `(255, 0, 0)`은 빨강, `(255, 255, 0)`은 노랑, `(0, 0, 0)`은 검정에 가깝습니다.")

        st.markdown(pretty_title("나의 픽셀 아트 만들기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        selected_color_idx = int(st.session_state.get("i3_art_selected_color", 1))
        selected_rgb = RGB_PALETTE_20[selected_color_idx]
        st.write("색상 팔레트에서 하나를 고른 뒤, 아래 5×5 캔버스의 칸을 눌러 나만의 픽셀 아트를 만들어 보세요.")
        render_value_cards(
            [
                {
                    "title": "현재 선택한 색상",
                    "value": str(selected_rgb),
                    "detail": "팔레트에서 고른 색이 다음에 누르는 칸에 바로 적용됩니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "캔버스 크기",
                    "value": "5×5",
                    "detail": "간단한 픽셀 아트를 만든 뒤 RGB 행렬로 바로 바꿔 볼 수 있습니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
            ],
            columns=2,
        )
        u_question = st.text_input(
            "AI 활용 질문 만들기",
            key="i3_u_question",
            placeholder="예: 같은 그림이라도 R, G, B 중 어떤 행렬이 더 중요한 정보를 줄까?",
        )

        st.markdown(pretty_title("12가지 색상 팔레트", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        for row_start in range(0, len(RGB_PALETTE_20), 4):
            palette_cols = st.columns(4)
            for offset, palette_idx in enumerate(range(row_start, min(row_start + 4, len(RGB_PALETTE_20)))):
                rgb = RGB_PALETTE_20[palette_idx]
                with palette_cols[offset]:
                    st.markdown(
                        f"<div style='height:22px; border-radius:6px; border:1px solid #cfd8dc; background: rgb{rgb}; margin-bottom:6px;'></div>",
                        unsafe_allow_html=True,
                    )
                    palette_cols[offset].button(
                        f"{rgb}",
                        key=f"i3_palette_{palette_idx}",
                        on_click=select_art_color,
                        args=(palette_idx,),
                        use_container_width=True,
                    )

        art_preview_col, art_editor_col = st.columns([0.95, 1.05])
        with art_preview_col:
            st.markdown(pretty_title("나의 픽셀 아트 미리 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.image(upscaled_display_image(art_image(), pixel_size=32), caption="나의 픽셀 아트", use_container_width=True)
        with art_editor_col:
            st.markdown(pretty_title("5×5 픽셀 캔버스", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            art_grid = np.array(st.session_state["i3_art_grid"], dtype=int)
            for row in range(art_grid.shape[0]):
                row_cols = st.columns(art_grid.shape[1], gap="small")
                for col in range(art_grid.shape[1]):
                    row_cols[col].button(
                        art_cell_label(art_grid[row, col]),
                        key=f"i3_art_btn_{row}_{col}",
                        on_click=paint_art_cell,
                        args=(row, col),
                        use_container_width=True,
                    )
            st.caption("캔버스의 칸을 누르면 현재 선택한 색으로 바로 칠해집니다. RGB 행렬은 아래 버튼을 눌렀을 때만 펼쳐집니다.")

        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("캔버스 지우기", key="i3_clear_art", use_container_width=True):
                clear_art_grid()
        with action_cols[1]:
            if st.button("RGB행렬 보기", key="i3_show_rgb_btn", use_container_width=True):
                st.session_state["i3_show_rgb_matrices"] = True
        with action_cols[2]:
            save_now = st.button("현재 결과 저장", key="i3_save_3", use_container_width=True)

        if st.session_state.get("i3_show_rgb_matrices", False):
            r_frame, g_frame, b_frame = art_channel_frames()
            rgb_cols = st.columns(3)
            with rgb_cols[0]:
                st.markdown("##### R 행렬")
                st.table(r_frame)
            with rgb_cols[1]:
                st.markdown("##### G 행렬")
                st.table(g_frame)
            with rgb_cols[2]:
                st.markdown("##### B 행렬")
                st.table(b_frame)

        if save_now:
            used_colors = sorted({tuple(RGB_PALETTE_20[idx]) for idx in np.array(st.session_state["i3_art_grid"]).flatten()})
            r_frame, g_frame, b_frame = art_channel_frames()
            save_activity_result(
                3,
                f"5×5 픽셀 아트를 만들고 RGB 행렬을 확인했다. 현재 선택 색상은 {selected_rgb}였고, "
                f"활동 중 내가 만든 질문은 '{u_question.strip() or '질문 미작성'}'이었다. "
                f"작품에 사용한 색은 {used_colors}이다. "
                f"픽셀 아트의 팔레트 번호 행렬은 {matrix_text(np.array(st.session_state['i3_art_grid']))} 이다.",
                details={
                    "writings": [("AI 활용 질문 만들기", u_question.strip())],
                    "matrices": [
                        ("R 행렬", r_frame.values.copy()),
                        ("G 행렬", g_frame.values.copy()),
                        ("B 행렬", b_frame.values.copy()),
                    ],
                    "images": [("나의 픽셀 아트", art_image().copy(), None)],
                },
            )
        st.caption(saved_status_text(3))

    with tabs[3]:
        example_name = current_character()
        example_image = base_image(example_name)
        gray_image = to_gray(example_image)
        edge_kernel = FILTERS["경계 찾기"][0]
        edge_response = np.abs(convolve_same(gray_image, edge_kernel))
        edge_display = np.zeros_like(edge_response) if float(edge_response.max()) == 0 else edge_response / float(edge_response.max()) * 255
        grouped_display = average_pool(edge_display, block_size=2)

        st.session_state["i3_saved_4"] = (
            f"딥러닝은 이미지를 한 번에 이해하지 않고, {example_name} 그림을 예시로 "
            f"1층에서 선과 경계, 2층에서 모양 조각, 3층에서 전체 대상을 차례로 이해한다는 설명을 확인했다."
        )
        st.session_state["i3_saved_detail_4"] = {
            "matrices": [
                ("1층 반응 행렬", edge_display.round().astype(int)),
                ("2층 묶음 행렬", grouped_display.round().astype(int)),
            ],
            "images": [
                (f"입력 이미지: {example_name}", example_image.copy(), None),
                ("1층: 선과 경계 찾기", edge_display.copy(), "magma"),
                ("2층: 특징 묶기", grouped_display.copy(), "magma"),
            ],
        }

        stage_intro(
            "결과의 의미와 한계 고민하기",
            "딥러닝이 이미지를 한 번에 읽지 않고 여러 층을 거치며 특징을 모아 가는 과정을 해석하는 단계입니다.",
            "작은 픽셀 정보는 여러 층을 지나며 어떻게 의미 있는 특징으로 바뀔까?",
            "#f3e5f5",
            "#e1bee7",
        )
        st.info("이미지 한 장도 인공지능에게는 숫자들의 모임입니다. 딥러닝은 이 숫자를 바로 정답으로 바꾸지 않고, 작은 특징부터 큰 의미까지 차례로 생각합니다.")
        with st.expander("딥러닝이란 무엇일까?", expanded=False):
            st.write("딥러닝은 인공지능이 여러 층(layer)을 거치며 데이터를 조금씩 분석해서 의미를 찾아가는 학습 방법입니다.")
            st.info("쉽게 말하면, 한 번에 정답을 맞히는 것이 아니라 `작은 특징 찾기 → 부분 모양 이해하기 → 전체 판단하기` 순서로 생각하는 방법입니다.")
        st.markdown(pretty_title("딥러닝의 다층 구조 그림", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        structure_slot = st.empty()
        caption_slot = st.empty()
        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            st.slider("1층 뉴런 수", min_value=2, max_value=5, key="i3_dl_hidden1")
        with slider_col2:
            st.slider("2층 뉴런 수", min_value=2, max_value=5, key="i3_dl_hidden2")

        structure_fig, total_connections = draw_deep_learning_structure(
            int(st.session_state["i3_dl_hidden1"]),
            int(st.session_state["i3_dl_hidden2"]),
        )
        structure_slot.pyplot(structure_fig, use_container_width=True)
        caption_slot.caption("입력층의 RGB 숫자가 여러 층을 지나며 선, 부분 모양, 전체 판단으로 이어지는 흐름을 그림으로 나타낸 것입니다.")

        render_value_cards(
            [
                {
                    "title": "1층 뉴런 수",
                    "value": str(int(st.session_state["i3_dl_hidden1"])),
                    "detail": "선, 경계, 밝은 부분처럼 기본 특징을 먼저 찾습니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "2층 뉴런 수",
                    "value": str(int(st.session_state["i3_dl_hidden2"])),
                    "detail": "기본 특징을 묶어 부분 모양을 만듭니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "총 연결선 수",
                    "value": str(int(total_connections)),
                    "detail": "연결이 많아질수록 더 많은 관계를 조합하며 생각할 수 있습니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
            ],
            columns=3,
        )

        if total_connections <= 20:
            thinking_text = "연결선이 적으면 비교하는 길이 단순해서, 아주 기본적인 특징을 중심으로 생각한다고 볼 수 있습니다."
        elif total_connections <= 35:
            thinking_text = "연결선이 늘어나면 여러 특징을 함께 비교하고 묶을 수 있어, 조금 더 깊게 생각하는 구조가 됩니다."
        else:
            thinking_text = "연결선이 많아질수록 한 번에 살피는 관계가 많아져 더 복잡한 특징까지 조합하며 깊게 생각할 수 있습니다."

        st.info(
            f"현재 총 연결선 수는 `3×{int(st.session_state['i3_dl_hidden1'])} + "
            f"{int(st.session_state['i3_dl_hidden1'])}×{int(st.session_state['i3_dl_hidden2'])} + "
            f"{int(st.session_state['i3_dl_hidden2'])}×1 = {int(total_connections)}`개입니다. {thinking_text}"
        )

        st.markdown(pretty_title("딥러닝은 층마다 어떻게 볼까?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        layer_df = pd.DataFrame(
            [
                {"단계": "입력", "하는 일": "RGB 숫자와 픽셀 위치를 받음", "학생용 설명": "이미지를 숫자판으로 받아들인다."},
                {"단계": "1층", "하는 일": f"뉴런 {int(st.session_state['i3_dl_hidden1'])}개가 선, 경계, 밝은 부분을 나누어 찾음", "학생용 설명": "어디가 튀는지 먼저 살핀다."},
                {"단계": "2층", "하는 일": f"뉴런 {int(st.session_state['i3_dl_hidden2'])}개가 특징들을 묶어 부분 모양을 만듦", "학생용 설명": "작은 특징들을 묶어 부분 모양을 만든다."},
                {"단계": "3층", "하는 일": "캐릭터 전체나 이미지 종류를 판단함", "학생용 설명": "여러 조각을 합쳐 전체 대상을 알아본다."},
            ]
        )
        st.dataframe(layer_df, use_container_width=True, hide_index=True, height=176)

        st.markdown(pretty_title("그림으로 이해해 보기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        flow_cols = st.columns(3)
        with flow_cols[0]:
            st.image(upscaled_display_image(example_image, pixel_size=26), caption=f"입력 이미지: {example_name}", use_container_width=True)
            st.caption("인공지능은 먼저 이 그림을 RGB 숫자들의 모음으로 받습니다.")
        with flow_cols[1]:
            st.image(upscaled_display_image(edge_display, cmap="magma", pixel_size=26), caption="1층: 선과 경계 찾기", use_container_width=True)
            st.caption("눈에 띄는 경계와 변화가 큰 곳이 먼저 강조됩니다.")
        with flow_cols[2]:
            st.image(upscaled_display_image(grouped_display, cmap="magma", pixel_size=52), caption="2층: 특징 묶기", use_container_width=True)
            st.caption("가까운 특징들을 묶어 눈, 귀, 몸통 같은 더 큰 단서로 생각합니다.")

        final_left, final_right = st.columns([1.1, 0.9])
        with final_left:
            st.markdown(pretty_title("마지막에는 어떻게 판단할까?", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            st.write(
                f"앞의 층에서 모은 정보를 바탕으로 마지막 층은 "
                f"`이 그림은 {example_name}와 비슷하다`처럼 전체 대상을 판단합니다. "
                "즉, 딥러닝은 작은 특징을 차곡차곡 쌓아 가며 이미지를 이해합니다."
            )
        with final_right:
            st.markdown(
                """
                <div style="padding:16px;border-radius:14px;background:linear-gradient(180deg,#eef6ff 0%,#f8fbff 100%);border:1px solid #d7e8fb;">
                    <div style="font-weight:700;color:#1565c0;margin-bottom:8px;">딥러닝의 생각 흐름</div>
                    <div style="font-size:0.98rem;line-height:1.7;">
                        입력 이미지<br>
                        ↓<br>
                        선과 경계 찾기<br>
                        ↓<br>
                        모양 조각 묶기<br>
                        ↓<br>
                        전체 대상 판단
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(pretty_title("짧게 성찰해 보기", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
        reflect_cols = st.columns(3)
        with reflect_cols[0]:
            st.text_area(
                "이번 활동의 핵심",
                key="i3_reflect_core",
                height=90,
                placeholder="예: 그림은 곧 숫자 배열이고, 층이 올라갈수록 더 큰 의미를 본다.",
            )
        with reflect_cols[1]:
            st.text_area(
                "결과를 그대로 믿기 어려운 이유",
                key="i3_reflect_limit",
                height=90,
                placeholder="예: 일부 특징만 강조되면 실제와 다르게 판단할 수 있다.",
            )
        with reflect_cols[2]:
            st.text_area(
                "더 알고 싶은 점",
                key="i3_reflect_next",
                height=90,
                placeholder="예: 층이 더 많아지면 어떤 특징을 더 배우게 될까?",
            )
        st.caption("설명 그림은 화면에서 확인하고, 아래에 적은 성찰 글은 포트폴리오 PDF에 함께 반영됩니다.")

    with tabs[4]:
        stage_intro(
            "우리의 삶과 사회로 연결하기",
            "오늘 활동을 정리해 포트폴리오로 저장하고, 이미지 인공지능이 실제 사회에서 쓰일 때의 의미를 생각해 보는 과정입니다.",
            "AI가 이미지를 읽어 판단할 때 우리는 무엇을 확인해야 할까?",
            "#fff3e0",
            "#ffe0b2",
        )
        st.markdown(pretty_title("학생 정보 입력 및 포트폴리오 저장", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        render_value_cards(
            [
                {"title": "문제 1", "value": saved_status_text(1), "detail": "문제 발견 저장 상태", "bg": "#f4f9ff", "border": "#90caf9"},
                {"title": "문제 2", "value": saved_status_text(2), "detail": "수학의 언어 저장 상태", "bg": "#f1f8e9", "border": "#aed581"},
                {"title": "문제 3", "value": saved_status_text(3), "detail": "AI 활용 저장 상태", "bg": "#fff8e1", "border": "#ffcc80"},
                {"title": "문제 4", "value": reflection_status_text(), "detail": "결과 해석 글 작성 상태", "bg": "#fce4ec", "border": "#f48fb1"},
            ],
            columns=2,
        )
        st.caption("문제 1~4에서 학생이 작성한 글과 저장한 결과가 PDF에 반영됩니다.")
        c1, c2, c3 = st.columns(3)
        with c1:
            group_name = st.text_input("모둠 이름", key="i3_group")
        with c2:
            stu_id = st.text_input("학번", max_chars=5, key="i3_id")
        with c3:
            stu_name = st.text_input("이름", key="i3_name")
        if group_name and stu_id and stu_name:
            if len(stu_id) >= 3 and stu_id[2] in PORT_URLS:
                pdf = create_pdf(
                    {"group": group_name, "id": stu_id, "name": stu_name, "character": current_character()},
                    practice_rows(),
                    reflection_entries(),
                    st.session_state.get("i3_social_meaning", ""),
                    st.session_state.get("i3_action_plan", ""),
                    st.session_state.get("i3_q_deep", ""),
                )
                p1, p2 = st.columns(2)
                with p1:
                    st.download_button("📄 이미지 탐구 포트폴리오 PDF 다운로드", pdf, f"{stu_id}_{stu_name}_3차시_이미지포트폴리오.pdf", "application/pdf", use_container_width=True)
                with p2:
                    st.markdown(f"""<a href="{PORT_URLS[stu_id[2]]}" target="_blank" style="display:block;padding:10px;background:linear-gradient(90deg,#43a047 0%,#66bb6a 100%);color:white;text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;">{stu_id[2]}반 포트폴리오 패들렛 바로가기</a>""", unsafe_allow_html=True)
            else:
                st.warning("학번의 세 번째 숫자가 1, 2, 5, 6인지 확인해 주세요.")
        else:
            st.warning("모둠, 학번, 이름을 입력하면 포트폴리오를 바로 받을 수 있습니다.")
        st.markdown("---")
        st.markdown(pretty_title("교사 논쟁적 질문을 읽고 모둠의 생각 정리하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        st.info("만약 학교 출입문 얼굴 인식 AI가 어떤 학생을 '등록되지 않은 사람'으로 잘못 분류했다면, 학교는 '대부분 맞으니 그대로 사용해도 된다'고 할 수 있을까요? 아니면 한 번의 오분류가 있더라도 반드시 사람이 다시 확인해야 할까요? 왜 그렇게 생각하나요?")
        social_cols = st.columns(2)
        with social_cols[0]:
            st.markdown(pretty_title("모둠의 생각 1", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
            social_meaning = st.text_area(
                "모둠의 생각 1. 사회에 주는 시사점",
                height=110,
                key="i3_social_meaning",
                placeholder="예: AI는 편리하지만 오분류가 생기면 학생의 안전과 권리에 영향을 줄 수 있다.",
            )
        with social_cols[1]:
            st.markdown(pretty_title("모둠의 생각 2", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
            action_plan = st.text_area(
                "모둠의 생각 2. 우리의 실천 제안",
                height=110,
                key="i3_action_plan",
                placeholder="예: 중요한 판단에는 사람이 한 번 더 확인하는 절차를 두어야 한다.",
            )
        st.markdown("---")
        st.markdown(pretty_title("모둠 심화 질문 만들기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write("모둠원과 함께 오늘 활동을 돌아보며 심화 질문을 하나 만들고, 앞서 작성한 답변과 함께 패들렛에 공유해 봅시다.")
        st.info(
            "🔥 [우리의 딥(Deep) 퀘스천] (윤리와 철학)\n"
            "배운 지식이나 기술이 실제 사회에 적용될 때 생길 수 있는 부작용이나 윤리적 딜레마를 다루며, "
            "정답 없이 서로의 가치관을 깊이 있게 나눌 수 있는 토론형 질문입니다.\n"
            "👉 예: 만약 인공지능이 복잡한 조건문만으로 회사 면접의 합격자를 결정한다면, "
            "우리는 그 알고리즘의 기준이 인간보다 공정하다고 믿을 수 있을까?"
        )
        question = st.text_area(
            "🔥 [우리의 딥(Deep) 퀘스천]",
            height=100,
            key="i3_q_deep",
            placeholder="예: 얼굴 인식 AI가 빠르고 편리하더라도, 오분류된 한 사람의 불이익까지 감수하며 계속 사용해도 공정하다고 할 수 있을까?",
        )
        if group_name and stu_id and social_meaning and action_plan and question and len(stu_id) >= 3 and stu_id[2] in QA_URLS:
            st.success("✅ 성찰 질문 작성이 완료되었습니다! 텍스트를 복사하여 패들렛에 업로드하세요.")
            report = f"""[F.U.T.U.R.E. 프로젝트 3DAY 성찰 일지]\n모둠명: {group_name}\n\n🔥 [우리가 만든 딥(Deep) 퀘스천]\n{question}\n\n💡 [교사의 심화 질문에 대한 우리의 생각]\n(사회에 주는 시사점) {social_meaning}\n(우리의 실천 제안) {action_plan}\n"""
            st.code(report, language="markdown")
            st.markdown(f"""<a href="{QA_URLS[stu_id[2]]}" target="_blank" style="display:inline-block;padding:10px 20px;background:linear-gradient(90deg,#1976d2 0%,#42a5f5 100%);color:white;text-decoration:none;border-radius:8px;font-weight:bold;">{stu_id[2]}반 질문 패들렛 바로가기</a>""", unsafe_allow_html=True)
        else:
            st.warning("사회에 주는 시사점, 우리의 실천 제안, 딥(Deep) 퀘스천을 모두 작성하면 공유용 글이 나타납니다.")
    st.markdown("<hr style='border:2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
