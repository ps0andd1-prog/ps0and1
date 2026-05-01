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
from PIL import Image


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

FACE_PHOTO_PATH = os.path.join(os.path.dirname(__file__), "image", "face_grid_source.png")
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

PORT_URLS = {"1": "https://padlet.com/ps0andd/p_1", "2": "https://padlet.com/ps0andd/p_2", "5": "https://padlet.com/ps0andd/p_5", "6": "https://padlet.com/ps0andd/p_6"}
QA_URLS = {"1": "https://padlet.com/ps0andd/q_1", "2": "https://padlet.com/ps0andd/q_2", "5": "https://padlet.com/ps0andd/q_5", "6": "https://padlet.com/ps0andd/q_6"}


def rgb_from_pattern(pattern):
    image = np.zeros((pattern.shape[0], pattern.shape[1], 3), dtype=np.uint8)
    for index, color in PALETTE.items():
        image[pattern == index] = color
    return image


def face_grid_image(size=20):
    if os.path.exists(FACE_PHOTO_PATH):
        image = Image.open(FACE_PHOTO_PATH).convert("RGB")
        image = image.resize((size, size), RESAMPLE_BILINEAR)
        return np.array(image, dtype=np.uint8)

    fallback = np.full((size, size, 3), 240, dtype=np.uint8)
    fallback[size // 4 : size - size // 4, size // 4 : size - size // 4] = (245, 210, 190)
    return fallback


def current_character():
    selected = st.session_state.get("i3_character", DEFAULT_CHARACTER)
    if selected not in CHARACTERS:
        st.session_state["i3_character"] = DEFAULT_CHARACTER
        return DEFAULT_CHARACTER
    return selected


def base_image(name=None):
    return rgb_from_pattern(CHARACTERS[name or current_character()]["pattern"])


def df_from(array):
    return pd.DataFrame(array.astype(int), index=range(1, array.shape[0] + 1), columns=range(1, array.shape[1] + 1))


def ensure_state():
    st.session_state.setdefault("i3_character", DEFAULT_CHARACTER)
    current_character()
    st.session_state.setdefault("i3_binary_shape", "계단")
    st.session_state.setdefault("i3_binary_editor_version", 0)
    st.session_state.setdefault("i3_binary_show_values", False)
    if (
        "i3_binary_grid" not in st.session_state
        or st.session_state.get("i3_binary_shape_applied") != st.session_state.get("i3_binary_shape", "계단")
    ):
        set_binary_grid(BINARY_PATTERNS[st.session_state.get("i3_binary_shape", "계단")].copy())
        st.session_state["i3_binary_shape_applied"] = st.session_state.get("i3_binary_shape", "계단")
    st.session_state.setdefault("i3_gray_show_matrix", False)
    st.session_state.setdefault("i3_social_question_prompt", "")
    st.session_state.setdefault("i3_social_image_thought", "")
    st.session_state.setdefault("i3_social_image_symbol", "")
    for idx in range(1, 5):
        st.session_state.setdefault(f"i3_saved_{idx}", "")
        st.session_state.setdefault(f"i3_saved_time_{idx}", "")
        st.session_state.setdefault(f"i3_saved_detail_{idx}", {})


def to_gray(image):
    return np.round(0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(float)


def draw_image(image, title, cmap=None, show_values=False, value_range=None, fig_size=(4.0, 4.0), value_fontsize=8):
    fig = Figure(figsize=fig_size)
    ax = fig.subplots()
    plot_min, plot_max = (0, 255) if value_range is None else value_range
    if cmap:
        ax.imshow(image, cmap=cmap, interpolation="nearest", vmin=plot_min, vmax=plot_max)
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
                if plot_max == plot_min:
                    brightness = 0
                else:
                    brightness = (float(image[row, col]) - plot_min) / (plot_max - plot_min) * 255
                ax.text(col, row, str(val), ha="center", va="center", fontsize=value_fontsize, color="white" if brightness > 145 else "black")
    fig.tight_layout()
    return fig




def render_deep_learning_reading_material():
    example_name = current_character()
    example_image = base_image(example_name)
    gray_image = to_gray(example_image)
    edge_kernel = FILTERS["경계 찾기"][0]
    edge_response = np.abs(convolve_same(gray_image, edge_kernel))
    edge_display = np.zeros_like(edge_response) if float(edge_response.max()) == 0 else edge_response / float(edge_response.max()) * 255
    grouped_display = average_pool(edge_display, block_size=2)

    st.markdown(pretty_title("심화 자료: 딥러닝은 이미지를 어떻게 이해할까?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
    st.info(
        "이 부분은 3차시 핵심 활동을 마친 뒤, 더 알고 싶은 학생들이 가볍게 보는 심화 자료입니다. 딥러닝은 이미지를 바로 정답으로 바꾸는 것이 아니라, "
        "선과 경계 찾기 → 특징 묶기 → 전체 판단하기의 순서로 단계적으로 읽어 갑니다."
    )
    st.info("딥러닝은 이미지를 한 번에 이해하지 않고, `선과 경계 찾기 → 부분 모양 묶기 → 전체 대상 판단하기` 순서로 해석합니다.")

    slider_col1, slider_col2 = st.columns(2)
    with slider_col1:
        st.slider("1층 뉴런 수", min_value=2, max_value=5, key="i3_dl_hidden1")
    with slider_col2:
        st.slider("2층 뉴런 수", min_value=2, max_value=5, key="i3_dl_hidden2")

    structure_fig, total_connections = draw_deep_learning_structure(
        int(st.session_state["i3_dl_hidden1"]),
        int(st.session_state["i3_dl_hidden2"]),
    )
    st.pyplot(structure_fig, use_container_width=True)

    layer_df = pd.DataFrame(
        [
            {"단계": "입력", "하는 일": "RGB 숫자와 픽셀 위치를 받음", "학생용 설명": "이미지를 숫자판으로 받아들인다."},
            {"단계": "1층", "하는 일": f"뉴런 {int(st.session_state['i3_dl_hidden1'])}개가 선, 경계, 밝은 부분을 찾음", "학생용 설명": "어디가 튀는지 먼저 살핀다."},
            {"단계": "2층", "하는 일": f"뉴런 {int(st.session_state['i3_dl_hidden2'])}개가 특징을 묶어 부분 모양을 만듦", "학생용 설명": "작은 특징들을 묶어 부분 모양을 만든다."},
            {"단계": "출력", "하는 일": "전체 이미지의 의미나 종류를 판단함", "학생용 설명": "여러 조각을 합쳐 전체 대상을 알아본다."},
        ]
    )
    st.dataframe(layer_df, use_container_width=True, hide_index=True, height=176)

    flow_cols = st.columns(3)
    with flow_cols[0]:
        st.image(upscaled_display_image(example_image, pixel_size=26), caption=f"입력 이미지: {example_name}", use_container_width=True)
    with flow_cols[1]:
        st.image(upscaled_display_image(edge_display, cmap="magma", pixel_size=26), caption="1층: 선과 경계 찾기", use_container_width=True)
    with flow_cols[2]:
        st.image(upscaled_display_image(grouped_display, cmap="magma", pixel_size=52), caption="2층: 특징 묶기", use_container_width=True)

    st.caption(
        f"현재 총 연결선 수는 `3×{int(st.session_state['i3_dl_hidden1'])} + "
        f"{int(st.session_state['i3_dl_hidden1'])}×{int(st.session_state['i3_dl_hidden2'])} + "
        f"{int(st.session_state['i3_dl_hidden2'])}×1 = {int(total_connections)}`개입니다."
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


def sanitize_binary_frame(frame):
    df = pd.DataFrame(frame).apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.clip(0, 1).round().astype(int)
    df.index = range(1, df.shape[0] + 1)
    df.columns = range(1, df.shape[1] + 1)
    return df


def set_binary_grid(grid, refresh_editor=False):
    arr = np.array(grid, dtype=int)
    arr = np.clip(arr, 0, 1)
    st.session_state["i3_binary_grid"] = arr
    if refresh_editor:
        st.session_state["i3_binary_editor_version"] = int(st.session_state.get("i3_binary_editor_version", 0)) + 1


def character_gray_matrix(name):
    return np.clip(255 - to_gray(base_image(name)), 0, 255).astype(int)


def combine_gray_matrices(name_a, name_b, k_value):
    matrix_a = character_gray_matrix(name_a).astype(float)
    matrix_b = character_gray_matrix(name_b).astype(float)
    result = k_value * matrix_a + (1 - k_value) * matrix_b
    return matrix_a.astype(int), matrix_b.astype(int), np.clip(result, 0, 255).round().astype(int)


def art_cell_label(index):
    labels = {
        0: "⬜",
        1: "⬛",
        2: "🟥",
        3: "🟧",
        4: "🟨",
        5: "🟩",
        6: "🟦",
        7: "🟪",
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


def single_channel_image(image, channel_idx):
    channel_image = np.zeros_like(image)
    channel_image[:, :, channel_idx] = image[:, :, channel_idx]
    return channel_image


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


def class_key_from_ids(*student_ids):
    for student_id in student_ids:
        value = str(student_id).strip()
        if len(value) >= 3 and value[2] in PORT_URLS:
            return value[2]
    return ""


def create_pdf(student, rows):
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Nanum", "", FONT_PATH, uni=True)
    pdf.set_font("Nanum", "", 11)
    pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        0,
        7,
        (
            f"모둠명: {student['group']}\n"
            f"학번: {student['id_1']}\n"
            f"이름: {student['name_1']}\n"
            f"활동 캐릭터: {student['character']}\n"
            f"작성일: {datetime.datetime.now():%Y-%m-%d}"
        ),
    )
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
    if any(str(text).strip() for _, text in social_image_prompt_entries()):
        pdf.set_fill_color(237, 231, 246)
        pdf.cell(0, 8, "문제 4. 우리의 픽셀 아트 프롬프트", ln=1, fill=True)
        for title, text in social_image_prompt_entries():
            add_text_box_to_pdf(pdf, title, text)
        add_text_box_to_pdf(pdf, "GPT 이미지 생성 프롬프트", build_social_image_prompt())
    return normalize_pdf_output(pdf.output(dest="S"))


def practice_rows():
    titles = [
        "문제 1. 얼굴 인식과 변수 발견",
        "문제 2. 이미지를 행렬로 표현하기",
        "문제 3. 가중 합성과 이미지 변화",
    ]
    return [
        {
            "title": title,
            "body": st.session_state.get(f"i3_saved_{idx}", "") or "아직 결과 저장 버튼을 누르지 않았습니다.",
            "details": st.session_state.get(f"i3_saved_detail_{idx}", {}),
        }
        for idx, title in enumerate(titles, start=1)
    ]


def social_image_prompt_entries():
    return [
        ("사회적 질문", st.session_state.get("i3_social_question_prompt", "")),
        ("이미지를 통해 표현하려는 생각", st.session_state.get("i3_social_image_thought", "")),
        ("이미지에 넣고 싶은 핵심 장면/상징", st.session_state.get("i3_social_image_symbol", "")),
    ]


def build_social_image_prompt():
    question = student_text_or_default(
        st.session_state.get("i3_social_question_prompt", ""),
        "사회적 질문을 먼저 적어 주세요.",
    )
    thought = student_text_or_default(
        st.session_state.get("i3_social_image_thought", ""),
        "이미지를 통해 표현하려는 생각을 적어 주세요.",
    )
    symbol = student_text_or_default(
        st.session_state.get("i3_social_image_symbol", ""),
        "넣고 싶은 장면이나 상징을 적어 주세요.",
    )
    return (
        "다음 조건에 맞는 우리의 픽셀 아트를 만들어 줘.\n\n"
        "1. 이 이미지는 환경과 관련된 사회적 메시지를 담아 줘.\n"
        f"2. 이 이미지는 다음 사회적 질문에서 시작해 줘: \"{question}\"\n"
        f"3. 이미지를 통해 표현하려는 생각은 다음과 같아: \"{thought}\"\n"
        f"4. 이미지에 넣고 싶은 핵심 장면이나 상징은 다음과 같아: \"{symbol}\"\n"
        "5. 결과물은 이해하기 쉬운 단순하고 선명한 15×15 회색조 픽셀 이미지로 만들어 줘.\n"
        "6. 15×15 크기에는 많은 내용을 담기 어려우므로, 하나의 핵심 메시지만 보이도록 매우 단순하게 구성해 줘.\n"
        "7. 장면은 1개만 중심에 두고, 상징도 1~2개개 정도만 사용해 줘.\n"
        "8. 회색조 이미지이므로 색은 흰색부터 검은색까지의 밝기 차이로만 표현해 줘.\n"
        "9. 작은 글씨나 복잡한 배경, 세부 묘사는 넣지 말고, 멀리서 봐도 바로 이해되는 단순한 도형과 상징 중심으로 표현해 줘.\n"
        "10. 완성된 이미지를 보여 준 뒤, 그 이미지를 R행렬, G행렬, B행렬의 세 개 15×15 행렬로 분해해서 하나의 TXT 파일로 저장해서 다운로드할 수 있는 형태의 텍스트로도 정리해 줘.\n"
        "11. 회색조 이미지이므로 각 위치의 R, G, B 값은 서로 같게 맞춰 줘.\n"
        "12. 각 행렬 값은 0부터 255 사이의 정수로 표현해 줘.\n"
        "13. 완성된 이미지는 PNG처럼 바로 저장하거나 다운로드할 수 있는 형태로도 제시해 줘.\n"
        "14. 마지막에는 이 이미지가 사회적 질문과 어떻게 연결되는지 2~3문장으로 설명해 줘."
    )


def social_prompt_status_text():
    has_prompt = any(str(text).strip() for _, text in social_image_prompt_entries())
    return "입력 완료" if has_prompt else "아직 작성하지 않았습니다."


def run():
    apply_local_style()
    ensure_state()
    page_banner(
        "이미지를 행렬로 보는 인공지능",
        "그림은 숫자 배열이고, 색 이미지는 RGB 세 행렬이며, 행렬 연산으로 그림이 변하고, 인공지능은 그 숫자 패턴을 읽는다는 핵심 흐름을 실습으로 익힙니다.",
    )
    st.markdown("<hr style='border:2px solid #2196F3;'>", unsafe_allow_html=True)

    tabs = st.tabs(["1️⃣ [F.U] 문제 발견", "2️⃣ [T] 수학의 언어", "3️⃣ [U] AI 이해", "4️⃣ [R.E] 세상과 연결"])

    with tabs[0]:
        stage_intro(
            "문제 발견",
            "실생활 얼굴 인식 상황을 떠올리며, AI가 얼굴을 구별할 때 어떤 정보를 변수로 삼을지 짧게 가설을 세우는 도입 단계입니다.",
            "AI는 얼굴을 어떻게 인식할까?",
            "#e3f2fd",
            "#bbdefb",
        )
        st.markdown(pretty_title("문제제기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write(
            "휴대폰 얼굴 인식은 사람의 얼굴을 읽어 같은 사람인지 구별합니다. "
            "그런데 컴퓨터는 얼굴을 볼 때 무엇을 정보로 삼고 있을까요?"
        )
        with st.expander("행렬의 정의와 성분 표현 보기", expanded=False):
            st.markdown(
                """
                행렬은 숫자를 가로와 세로로 줄 맞추어 놓은 직사각형 배열입니다.
                한 칸에 들어 있는 숫자를 **성분**이라고 하고, `a(2,3)`처럼 행과 열의 위치로 나타낼 수 있습니다.
                """
            )
            st.latex(r"A=\begin{bmatrix}79 & 41 \\ 57 & 22\end{bmatrix}, \quad a_{2,1}=57")

        fu_face_image = face_grid_image(25)
        fu_left, fu_right = st.columns([1, 1])
        with fu_left:
            st.pyplot(draw_image(fu_face_image, "25×25 격자 실제 얼굴 이미지", None), use_container_width=True)
        with fu_right:
            st.markdown("**이 얼굴을 작은 칸으로 나누면 무엇이 보일까?**")
            st.markdown("**AI는 어떤 정보를 변수로 삼아 얼굴을 구별할까?**")
            st.caption("생각나는 정보를 체크해 보고, 아래에 한 줄 가설을 적어 보세요.")

            feature_cols = st.columns(3)
            feature_options = [
                ("위치", "i3_fu_feature_position"),
                ("밝기", "i3_fu_feature_brightness"),
                ("색", "i3_fu_feature_color"),
                ("경계", "i3_fu_feature_edge"),
                ("얼굴 부분의 차이", "i3_fu_feature_part"),
            ]
            selected_features = []
            for idx, (label, key) in enumerate(feature_options):
                with feature_cols[idx % 3]:
                    if st.checkbox(label, key=key):
                        selected_features.append(label)

        hypothesis = st.text_input(
            "짧은 가설 쓰기",
            key="i3_hypothesis",
            placeholder="예: AI는 얼굴의 위치와 밝기 차이를 보고 사람을 구별할 것 같다.",
        )

        if st.button("문제 1 결과 저장", key="i3_save_1"):
            selected_text = ", ".join(selected_features) if selected_features else "위치, 밝기, 색, 경계, 얼굴 부분의 차이"
            hypothesis_text = hypothesis.strip() or "가설 미작성"
            save_activity_result(
                1,
                "탐구 질문은 “AI는 얼굴을 어떻게 인식할까?”였다. "
                "나는 얼굴을 구별하는 데 필요한 정보와 변수를 생각해 보았다. "
                f"AI가 이미지를 읽을 때 {selected_text}와 같은 정보를 사용할 수 있다고 가설을 세웠다. "
                f"한 줄 가설은 “{hypothesis_text}”이다.",
                details={
                    "writings": [
                        ("체크한 변수 후보", selected_text),
                        ("짧은 가설 쓰기", hypothesis_text),
                    ],
                    "images": [("25×25 격자 실제 얼굴 이미지", fu_face_image, None)],
                },
            )
        st.caption(saved_status_text(1))
        st.caption("다음 [T] 단계에서는 작은 그림을 숫자 표로 바꾸며 이 생각을 더 구체적으로 확인합니다.")

    with tabs[1]:
        stage_intro(
            "수학의 언어",
            "현실의 대상을 수학의 언어로 표현하는 단계입니다. 이미지를 행과 열을 가진 수의 배열, 즉 행렬로 나타내며 그림과 숫자 표현의 대응을 살펴봅니다.",
            "격자로 표현된 이미지를 어떻게 행렬로 나타낼까?",
            "#fff8e1",
            "#ffecb3",
        )
        
        st.markdown(pretty_title("격자로 표현된 이미지를 어떻게 행렬로 나타낼까?", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write(
            "이번에는 `계단`과 `체스판` 무늬 중 하나를 선택해, 왼쪽 행렬과 오른쪽 이미지를 함께 바꾸어 봅니다. "
            "행렬의 한 칸은 이미지의 한 칸과 정확히 대응하므로, 숫자를 바꾸면 그림이 바뀌고 그림의 칸을 누르면 행렬도 함께 바뀝니다."
        )
        with st.expander("행렬의 연산 간단히 보기", expanded=False):
            st.write(
                "행렬의 합은 **같은 위치에 있는 수끼리** 더합니다. "
                "예를 들어 왼쪽 위는 왼쪽 위끼리, 오른쪽 아래는 오른쪽 아래끼리 계산합니다. "
                "실수배는 행렬의 **모든 칸에 같은 수를 곱하는 것**입니다."
            )
            st.latex(r"A=\begin{bmatrix}1 & 0 \\ 1 & 1\end{bmatrix},\quad B=\begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}")
            st.latex(r"A+B=\begin{bmatrix}1+0 & 0+1 \\ 1+1 & 1+0\end{bmatrix}=\begin{bmatrix}1 & 1 \\ 2 & 1\end{bmatrix}")
            st.latex(r"2A=\begin{bmatrix}2\times1 & 2\times0 \\ 2\times1 & 2\times1\end{bmatrix}=\begin{bmatrix}2 & 0 \\ 2 & 2\end{bmatrix}")
            st.caption("즉, 행렬은 자리 바꾸어 계산하는 것이 아니라 같은 자리의 성분끼리 계산합니다.")

        binary_shape = st.radio(
            "기본 무늬 선택",
            ["계단", "체스판"],
            horizontal=True,
            key="i3_binary_shape",
        )
        if st.session_state.get("i3_binary_shape_applied") != binary_shape:
            set_binary_grid(BINARY_PATTERNS[binary_shape].copy(), refresh_editor=True)
            st.session_state["i3_binary_shape_applied"] = binary_shape

        current_grid = np.array(st.session_state["i3_binary_grid"], dtype=int)
        editor_key = f"i3_t_binary_editor_{int(st.session_state.get('i3_binary_editor_version', 0))}"

        activity_left, activity_right = st.columns([1.1, 0.9], gap="large")
        with activity_left:
            st.markdown(pretty_title("행렬에 0 또는 1 입력하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            input_matrix = st.data_editor(
                binary_matrix_frame(current_grid),
                key=editor_key,
                hide_index=True,
                use_container_width=True,
                height=260,
                num_rows="fixed",
                column_config={
                    col: st.column_config.NumberColumn(
                        str(col),
                        min_value=0,
                        max_value=1,
                        step=1,
                        format="%d",
                        width=42,
                    )
                    for col in range(1, current_grid.shape[1] + 1)
                },
            )
            answer_matrix = sanitize_binary_frame(input_matrix)
            if not np.array_equal(answer_matrix.values, current_grid):
                set_binary_grid(answer_matrix.values.copy())
            current_grid = np.array(st.session_state["i3_binary_grid"], dtype=int)

        with activity_right:
            st.markdown(pretty_title("오른쪽 표현 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            toggle_label = "행렬 숨기기" if st.session_state.get("i3_binary_show_values", False) else "행렬 보기"
            if st.button(toggle_label, key="i3_binary_toggle_values", use_container_width=True):
                st.session_state["i3_binary_show_values"] = not st.session_state.get("i3_binary_show_values", False)

            st.pyplot(
                draw_image(
                    current_grid,
                    f"{binary_shape} 무늬 이미지",
                    "gray_r",
                    show_values=st.session_state.get("i3_binary_show_values", False),
                    value_range=(0, 1),
                    fig_size=(5.0, 5.0),
                    value_fontsize=15,
                ),
                use_container_width=True,
            )
            if st.session_state.get("i3_binary_show_values", False):
                st.caption("이미지 위에 현재 행렬의 0,1 값이 함께 표시되어 같은 위치 성분을 바로 비교할 수 있습니다.")
            else:
                st.caption("지금은 무늬 이미지만 보입니다. `행렬 보기` 버튼을 누르면 이미지 위에 0,1 값이 나타납니다.")

        base_pattern = BINARY_PATTERNS[binary_shape]
        if np.array_equal(current_grid, base_pattern):
            st.success(f"현재 행렬은 기본 `{binary_shape}` 무늬와 같습니다.")
        else:
            st.info("행렬의 값을 바꾸면 오른쪽 표현도 바로 함께 바뀝니다. 지금은 기본 무늬를 직접 수정한 상태입니다.")

        st.info(
            "- 이미지는 행과 열을 가진 값의 배열로 표현될 수 있습니다.\n"
            "- 따라서 이미지는 행렬로 나타낼 수 있습니다.\n"
            "- AI도 이미지를 이런 수학적 표현으로 읽을 수 있습니다."
        )

        if st.button("문제 2 결과 저장", key="i3_save_2"):
            save_activity_result(
                2,
                f"`{binary_shape}` 무늬를 0과 1로 이루어진 6×6 행렬로 표현하였다. 최종 행렬은 {binary_grid_text(current_grid)} 이다. "
                "행렬의 값을 바꾸면 오른쪽 이미지와 행렬 표현도 함께 바뀌는 것을 확인하였다. "
                "즉, 이미지는 행렬로 표현될 수 있음을 이해하였다.",
                details={
                    "matrices": [(f"{binary_shape} 6×6 이진 행렬", current_grid.copy())],
                    "images": [(f"{binary_shape} 6×6 이미지", (current_grid * 255).astype(np.uint8), "gray_r")],
                },
            )
        st.caption(saved_status_text(2))

    with tabs[2]:
        stage_intro(
            "AI의 이해: 행렬 합성으로 이미지 변화 해석하기",
            "두 이미지를 행렬 A, B로 두고 가중 합성 모델 C = kA + (1-k)B를 시뮬레이션하며, 이미지 변화가 행렬식과 어떻게 연결되는지 해석하는 단계입니다.",
            "AI는 이미지를 어떻게 합성할까?",
            "#e8f5e9",
            "#c8e6c9",
        )
        st.markdown(pretty_title("행렬 합성으로 이미지 변화 살펴보기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write(
            "이번 활동은 AI의 이미지 처리 원리를 단순화한 행렬 합성 시뮬레이션입니다. "
            "두 이미지를 행렬 A, 행렬 B로 두고 `C = kA + (1-k)B`를 적용해, k 값에 따라 결과 행렬 C와 결과 이미지가 어떻게 달라지는지 살펴봅니다."
        )

        choose_left, choose_right = st.columns(2)
        with choose_left:
            char_a = st.selectbox("행렬 A 캐릭터", list(CHARACTERS.keys()), key="i3_gray_char_a")
        with choose_right:
            char_b = st.selectbox("행렬 B 캐릭터", list(CHARACTERS.keys()), index=1, key="i3_gray_char_b")
        blend_left, blend_right = st.columns([1, 1])
        with blend_left:
            st.markdown(pretty_title("k 값을 움직이며 합성해 보기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            k_value = st.slider("k 값 (0 ≤ k ≤ 1)", 0.0, 1.0, 0.5, 0.1, key="i3_gray_k")
        with blend_right:
            st.markdown(pretty_title("합성 계산식", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.latex(fr"C = {k_value:.1f}A + ({1-k_value:.1f})B")
        matrix_a, matrix_b, matrix_result = combine_gray_matrices(char_a, char_b, k_value)
        if "i3_gray_show_matrix" not in st.session_state:
            st.session_state["i3_gray_show_matrix"] = False
        toggle_label = "행렬 숨기기" if st.session_state["i3_gray_show_matrix"] else "행렬 보기"
        formula_text = f"C = {k_value:.1f}A + ({1 - k_value:.1f})B"
        with blend_right:
            if st.button(toggle_label, key="i3_gray_toggle_matrix", use_container_width=True):
                st.session_state["i3_gray_show_matrix"] = not st.session_state["i3_gray_show_matrix"]
            if not st.session_state["i3_gray_show_matrix"]:
                st.caption("버튼을 누르면 아래에 행렬 A, 행렬 B, 결과 행렬 C가 함께 보입니다.")

        preview_cols = st.columns(3)
        with preview_cols[0]:
            st.pyplot(draw_image(matrix_a, f"{char_a} → 행렬 A", "gray_r"), use_container_width=True)
            if st.session_state["i3_gray_show_matrix"]:
                st.dataframe(df_from(matrix_a), use_container_width=True, height=320)
        with preview_cols[1]:
            st.pyplot(draw_image(matrix_b, f"{char_b} → 행렬 B", "gray_r"), use_container_width=True)
            if st.session_state["i3_gray_show_matrix"]:
                st.dataframe(df_from(matrix_b), use_container_width=True, height=320)
        with preview_cols[2]:
            st.pyplot(draw_image(matrix_result, "결과 행렬 C", "gray_r"), use_container_width=True)
            if st.session_state["i3_gray_show_matrix"]:
                st.dataframe(df_from(matrix_result), use_container_width=True, height=320)

        st.markdown(pretty_title(" 1️⃣ 모둠활동: k = 1일 때 어떤 캐릭터가 남을까?", "#f1f8e9", "#c5e1a5"), unsafe_allow_html=True)
        st.markdown("**k = 1일 때 결과 이미지는 어떤 행렬과 같아지는지, 식 `C = kA + (1-k)B`를 이용해 설명해 보자.**")
        blank_answer = st.text_input("k = 1일 때 남는 이미지 또는 행렬 정리", key="i3_gray_blank_answer", placeholder=f"예: 마리오")
        math_principle = st.text_area(
            "수학적 설명 쓰기",
            key="i3_gray_math_principle",
            height=100,
            placeholder="C = kA + (1-k)B",
        )
        if st.button("모둠 해석 확인하기", key="i3_gray_blank_check", use_container_width=True):
            cleaned_answer = str(blank_answer).strip().replace(" ", "")
            cleaned_corrects = [str(char_a).strip().replace(" ", ""), "행렬A", "A"]
            if cleaned_answer in cleaned_corrects:
                st.success(f"좋아요. k = 1이면 `C = 1A + 0B = A` 이므로 결과는 행렬 A와 같고, `{char_a}` 이미지만 남습니다.")
            else:
                st.warning("핵심은 `C = 1A + 0B = A` 입니다. 이 식을 이용해 왜 결과가 행렬 A와 같아지는지 다시 설명해 보세요.")
                st.info(f"정리 예시: `{char_a}` 이미지, 즉 행렬 A")

        if st.button("합성된 이미지 및 행렬 저장", key="i3_save_3"):
            save_activity_result(
                3,
                f"두 이미지를 행렬 A, B로 두고 가중 합성 모델을 시뮬레이션하였다. "
                f"사용한 식은 {formula_text} 이다. "
                f"k 값에 따라 결과 이미지와 결과 행렬 C가 달라짐을 확인하였다. "
                f"k = 1일 때 C = 1A + 0B = A가 됨을 '{math_principle.strip() or '설명 미작성'}'으로 설명하였다. "
                f"이를 통해 이미지 변화가 행렬식과 대응됨을 이해하였다. "
                f"모둠이 정리한 남는 이미지 또는 행렬은 '{blank_answer.strip() or '답 미작성'}'이다. "
                f"최종 결과 행렬 C는 {matrix_text(matrix_result)} 이다.",
                details={
                    "writings": [
                        ("k = 1일 때 남는 이미지 또는 행렬", blank_answer.strip()),
                        ("수학적 설명 쓰기", math_principle.strip()),
                    ],
                    "matrices": [
                        (f"{char_a} 행렬 A", matrix_a.copy()),
                        (f"{char_b} 행렬 B", matrix_b.copy()),
                        ("결과 행렬 C", matrix_result.copy()),
                    ],
                    "images": [("결과 이미지 C", matrix_result.copy(), "gray_r")],
                },
            )
        st.caption(saved_status_text(3))
    with tabs[3]:
        stage_intro(
            "R.E: 우리의 삶과 사회로 연결하기",
            "오늘 활동을 정리해 포트폴리오로 저장하고, 환경과 관련된 사회적 메시지를 담은 우리의 픽셀 아트 프롬프트를 완성하는 과정입니다.",
            "환경과 관련된 질문을 15×15 회색조 픽셀 이미지로 바꾸려면 어떤 장면과 메시지를 담아야 할까?",
            "#fff3e0",
            "#ffe0b2",
        )
        st.markdown(pretty_title("2️⃣ 모둠활동: 우리의 픽셀 아트 만들기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.info(
            "이제 모둠별로 1개의 픽셀 아트를 기획해 봅시다. 환경과 관련된 사회적 질문에서 시작해, 이미지를 통해 전하고 싶은 생각을 정리하면 "
            "GPT에 바로 넣을 수 있는 프롬프트가 자동으로 완성됩니다. 15×15 회색조는 아주 작은 크기이므로, 많은 내용을 넣기보다 하나의 핵심 메시지가 바로 보이도록 단순하게 표현하는 것이 중요합니다."
        )
        render_value_cards(
            [
                {
                    "title": "출발점",
                    "value": "사회적 질문",
                    "detail": "이미지는 반드시 하나의 환경 질문에서 시작합니다. 예: 길가에 쓰레기가 많아지면 우리 동네는 어떤 불편을 겪게 될까?",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "고정 조건",
                    "value": "15×15 회색조 + 한 메시지",
                    "detail": "완성 이미지는 15×15 회색조로 만들고, 복잡한 장면 대신 하나의 메시지가 바로 보이게 단순하게 표현합니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
            ],
            columns=2,
        )
        prompt_cols = st.columns([1, 1])
        with prompt_cols[0]:
            st.text_area(
                "사회적 질문",
                key="i3_social_question_prompt",
                height=110,
                placeholder="예: 길가에 쓰레기가 많아지면 우리 동네는 어떤 불편을 겪게 될까?",
            )
            st.text_area(
                "이미지를 통해 표현하려는 생각",
                key="i3_social_image_thought",
                height=110,
                placeholder="예: 깨끗한 환경을 지키려면 모두가 작은 실천을 해야 한다는 한 가지 메시지를 보여 주고 싶다.",
            )
        with prompt_cols[1]:
            st.text_area(
                "이미지에 넣고 싶은 핵심 장면/상징",
                key="i3_social_image_symbol",
                height=110,
                placeholder="예: 쓰레기통 1개, 떨어진 플라스틱 컵 1개",
            )
            st.caption("15×15 픽셀에는 많은 내용을 담기 어렵기 때문에, 장면과 상징은 1~2개만 정하는 것이 좋습니다.")

        st.markdown(pretty_title("완성된 GPT 이미지 생성 프롬프트", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.code(build_social_image_prompt(), language="markdown")
        st.caption("이 프롬프트를 GPT에 입력하면 환경 메시지가 담긴 15×15 회색조 이미지, 이미지 파일 다운로드 형태, R·G·B 행렬 분해 결과, 그리고 각 행렬을 TXT로 저장하거나 다운로드할 수 있는 텍스트 형식까지 함께 요청할 수 있습니다. 작성 내용은 포트폴리오 PDF에도 반영됩니다.")

        st.session_state["i3_saved_4"] = (
            "환경과 관련된 사회적 질문에서 시작해 15×15 회색조 우리의 픽셀 아트를 만들기 위한 GPT 프롬프트를 구성했다."
        )
        st.session_state["i3_saved_detail_4"] = {
            "writings": social_image_prompt_entries() + [("GPT 이미지 생성 프롬프트", build_social_image_prompt())]
        }

        st.markdown("---")
        st.markdown(pretty_title("학생 정보 입력 및 포트폴리오 저장", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        render_value_cards(
            [
                {"title": "문제 1", "value": saved_status_text(1), "detail": "문제 발견 저장 상태", "bg": "#f4f9ff", "border": "#90caf9"},
                {"title": "문제 2", "value": saved_status_text(2), "detail": "수학의 언어 저장 상태", "bg": "#f1f8e9", "border": "#aed581"},
                {"title": "문제 3", "value": saved_status_text(3), "detail": "AI 활용 저장 상태", "bg": "#fff8e1", "border": "#ffcc80"},
                {"title": "문제 4", "value": social_prompt_status_text(), "detail": "R.E 연결 프롬프트 작성 상태", "bg": "#fce4ec", "border": "#f48fb1"},
            ],
            columns=2,
        )
        st.caption("문제 1~4에서 학생이 작성한 글과 저장한 결과가 PDF에 반영됩니다.")
        group_name = st.text_input("모둠 이름", key="i3_group")
        st.markdown(pretty_title("학생 정보", "#f4f9ff", "#dbeafe"), unsafe_allow_html=True)
        info_cols = st.columns(2)
        with info_cols[0]:
            stu_id_1 = st.text_input("학번", max_chars=5, key="i3_id_1")
        with info_cols[1]:
            stu_name_1 = st.text_input("이름", key="i3_name_1")

        if group_name and stu_id_1 and stu_name_1:
            pdf = create_pdf(
                {
                    "group": group_name,
                    "id_1": stu_id_1,
                    "name_1": stu_name_1,
                    "character": current_character(),
                },
                practice_rows(),
            )
            p1, p2 = st.columns(2)
            with p1:
                st.download_button(
                    "📄 이미지 탐구 포트폴리오 PDF 다운로드",
                    pdf,
                    f"{group_name}_{stu_name_1}_3차시_이미지포트폴리오.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
            with p2:
                class_key = class_key_from_ids(stu_id_1)
                port_url = PORT_URLS.get(class_key)
                if port_url:
                    st.markdown(f"""<a href="{port_url}" target="_blank" style="display:block;padding:10px;background:linear-gradient(90deg,#43a047 0%,#66bb6a 100%);color:white;text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;">{class_key}반 포트폴리오 패들렛 바로가기</a>""", unsafe_allow_html=True)
        else:
            st.warning("모둠 이름과 학번, 이름을 입력하면 포트폴리오를 바로 받을 수 있습니다.")
    st.markdown("<hr style='border:2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
