import datetime
import os
import tempfile

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
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
}

FACE_PHOTO_PATH = os.path.join(os.path.dirname(__file__), "image", "face_grid_source.png")
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

PORT_URLS = {"1": "https://padlet.com/ps0andd/p_1", "2": "https://padlet.com/ps0andd/p_2", "5": "https://padlet.com/ps0andd/p_5", "6": "https://padlet.com/ps0andd/p_6"}
QA_URLS = {"1": "https://padlet.com/ps0andd/q_1", "2": "https://padlet.com/ps0andd/q_2", "5": "https://padlet.com/ps0andd/q_5", "6": "https://padlet.com/ps0andd/q_6"}
GALLERY_URLS = {"1": "https://padlet.com/ps0andd/g_1", "2": "https://padlet.com/ps0andd/g_2", "5": "https://padlet.com/ps0andd/g_5", "6": "https://padlet.com/ps0andd/g_6"}
GPT_URL = "https://chatgpt.com"
SOCIAL_TOPIC_SUPPORT = {
    "환경과 기후": {
        "guide": "예: 쓰레기 문제, 기후 위기, 물과 숲을 지키는 행동",
        "questions": [
            "길가에 쓰레기가 많아지면 우리 동네 사람들은 어떤 불편을 겪게 될까?",
            "기후가 더워지면 우리 학교와 마을의 생활은 어떻게 달라질까?",
            "깨끗한 물과 숲을 지키기 위해 우리가 먼저 바꿀 수 있는 행동은 무엇일까?",
        ],
        "symbols": ["버려진 일회용 컵", "뜨거운 태양 아래 시든 나무", "물방울을 감싼 손"],
    },
    "안전과 기술": {
        "guide": "예: AI 오분류, 개인정보 보호, 기술 사용에서의 사람 확인",
        "questions": [
            "AI가 사람을 잘못 판단하면 누가 어떤 피해를 입을 수 있을까?",
            "편리한 기술을 사용할 때 개인정보는 어디까지 맡겨도 괜찮을까?",
            "자동화된 판단을 그대로 믿기 전에 사람이 확인해야 할 것은 무엇일까?",
        ],
        "symbols": ["자물쇠가 걸린 스마트폰", "경고 표시가 뜬 얼굴 인식 화면", "사람 손과 로봇 손 사이의 확인 표시"],
    },
    "다양성과 존중": {
        "guide": "예: 차별 반대, 서로의 다름 존중, 함께 살아가기",
        "questions": [
            "다르다는 이유로 배제되는 사람이 생기면 공동체는 무엇을 잃게 될까?",
            "서로 다른 생각과 모습을 가진 사람들이 함께 지내려면 어떤 태도가 필요할까?",
            "존중받는 교실을 만들기 위해 우리가 멈춰야 할 말과 행동은 무엇일까?",
        ],
        "symbols": ["서로 다른 색의 손이 맞잡은 모습", "원 안에 함께 서 있는 사람들", "깨진 선을 잇는 하트"],
    },
}


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
    if st.session_state.get("i3_binary_shape") not in BINARY_PATTERNS:
        st.session_state["i3_binary_shape"] = "계단"
    st.session_state.setdefault("i3_binary_editor_version", 0)
    st.session_state.setdefault("i3_binary_show_values", False)
    if (
        "i3_binary_grid" not in st.session_state
        or st.session_state.get("i3_binary_shape_applied") != st.session_state.get("i3_binary_shape", "계단")
    ):
        set_binary_grid(BINARY_PATTERNS[st.session_state.get("i3_binary_shape", "계단")].copy())
        st.session_state["i3_binary_shape_applied"] = st.session_state.get("i3_binary_shape", "계단")
    st.session_state.setdefault("i3_gray_show_matrix", False)
    st.session_state.setdefault("i3_gray_math_principle", "C = kA + (1-k)B")
    st.session_state.setdefault("i3_social_topic", "환경과 기후")
    st.session_state.setdefault("i3_social_question_prompt", "")
    st.session_state.setdefault("i3_social_image_symbol_1", "")
    st.session_state.setdefault("i3_generated_prompt", "")
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
                ax.text(
                    col,
                    row,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=value_fontsize,
                    color="#1565c0",
                    path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
                )
    fig.tight_layout()
    return fig
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
    frame = pd.DataFrame(
        grid.astype(int),
        index=[str(row) for row in range(1, grid.shape[0] + 1)],
        columns=[str(col) for col in range(1, grid.shape[1] + 1)],
    )
    frame.index.name = "행/열"
    return frame


def sanitize_binary_frame(frame):
    df = pd.DataFrame(frame).copy()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.clip(0, 1).round().astype(int)
    df.index = range(1, df.shape[0] + 1)
    df.columns = range(1, df.shape[1] + 1)
    return df


def binary_matrix_editor(grid, key_prefix):
    edited_frame = st.data_editor(
        binary_matrix_frame(grid),
        key=key_prefix,
        hide_index=False,
        use_container_width=True,
        height=260,
        num_rows="fixed",
        column_config={
            str(col): st.column_config.NumberColumn(
                str(col),
                min_value=0,
                max_value=1,
                step=1,
                format="%d",
                width=42,
            )
            for col in range(1, grid.shape[1] + 1)
        },
    )
    return sanitize_binary_frame(edited_frame).values


def set_binary_grid(grid, refresh_editor=False):
    arr = np.array(grid, dtype=int)
    arr = np.clip(arr, 0, 1)
    st.session_state["i3_binary_grid"] = arr
    if refresh_editor:
        st.session_state["i3_binary_editor_version"] = int(st.session_state.get("i3_binary_editor_version", 0)) + 1


def character_gray_matrix(name):
    return np.clip(to_gray(base_image(name)), 0, 255).astype(int)


def combine_gray_matrices(name_a, name_b, k_value):
    matrix_a = character_gray_matrix(name_a).astype(float)
    matrix_b = character_gray_matrix(name_b).astype(float)
    result = k_value * matrix_a + (1 - k_value) * matrix_b
    return matrix_a.astype(int), matrix_b.astype(int), np.clip(result, 0, 255).round().astype(int)


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


def pdf_body_text(title, body):
    text = str(body)
    if title == "문제 3. 가중 합성과 이미지 변화":
        marker = " 최종 결과 행렬 C는 "
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    return text


def pdf_detail_payload(title, details):
    payload = {
        "writings": list(details.get("writings", [])),
        "matrices": list(details.get("matrices", [])),
        "images": list(details.get("images", [])),
    }

    payload["images"] = [
        item for item in payload["images"] if item[0] != "25×25 격자 실제 얼굴 이미지"
    ]

    if title == "문제 3. 가중 합성과 이미지 변화":
        payload["matrices"] = []

    return payload


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(25, 118, 210)
        self.rect(0, 0, self.w, 20, "F")
        self.set_xy(10, 5)
        self.set_text_color(255, 255, 255)
        self.set_font("Nanum", "", 16)
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 2-1 포트폴리오", ln=1, align="C")
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
        body = pdf_body_text(title, row.get("body", ""))
        details = pdf_detail_payload(title, row.get("details", {}))
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
    if any(str(text).strip() for _, text in social_image_prompt_entries()) or str(st.session_state.get("i3_generated_prompt", "")).strip():
        pdf.set_fill_color(237, 231, 246)
        pdf.cell(0, 8, "문제 4. 우리의 픽셀 아트 프롬프트", ln=1, fill=True)
        for title, text in social_image_prompt_entries():
            add_text_box_to_pdf(pdf, title, text)
        add_text_box_to_pdf(pdf, "학생이 작성한 프롬프트 만들기 결과", st.session_state.get("i3_generated_prompt", ""))
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
        ("선택한 주제", st.session_state.get("i3_social_topic", "")),
        ("생각해 볼 질문", st.session_state.get("i3_social_question_prompt", "")),
        ("상징 1", st.session_state.get("i3_social_image_symbol_1", "")),
    ]


def build_social_image_prompt():
    topic = student_text_or_default(
        st.session_state.get("i3_social_topic", ""),
        "사회적 주제를 먼저 선택해 주세요.",
    )
    question = student_text_or_default(
        st.session_state.get("i3_social_question_prompt", ""),
        "생각해 볼 질문을 먼저 적어 주세요.",
    )
    symbol_1 = student_text_or_default(
        st.session_state.get("i3_social_image_symbol_1", ""),
        "상징을 적어 주세요.",
    )
    return (
    "20×20 회색조 픽셀아트를 만들어 주세요.\n\n"

    "[핵심 조건]\n"
    "- 이미지는 반드시 20행 20열의 정사각형 픽셀 격자로 구성합니다.\n"
    "- 한 칸에는 하나의 회색조 값만 사용합니다.\n"
    "- 각 픽셀 칸은 크게 확대한 정사각형 블록으로 보여 주세요.\n"
    "- 모든 픽셀 칸에는 격자선이 보이게 해 주세요.\n"
    "- 각 픽셀 칸의 중앙에 해당 픽셀의 회색조 숫자를 직접 적어 주세요.\n"
    "- 숫자는 픽셀 색 위에 잘 보이도록 표시해 주세요.\n"
    "- 어두운 픽셀에는 흰색 숫자\n"
    "- 밝은 픽셀에는 검은색 숫자\n"
    "- 작은 세부 묘사는 넣지 말고 큰 사각형 픽셀 블록만 사용합니다.\n"
    "- 글자 설명, 제목, 배경 장식은 넣지 않습니다.\n"
    "- 단, 픽셀 칸 안의 회색조 숫자는 반드시 표시합니다.\n"
    "- 핵심 상징이 바로 보이도록 단순하게 구성합니다.\n\n"

    "[내용 조건]\n"
    f"- 주제: {topic}\n"
    f"- 생각해 볼 질문: {question}\n"
    f"- 핵심 상징: {symbol_1}\n\n"

    "[색상 조건]\n"
    "- 사용할 수 있는 색상 값은 다음 13개뿐입니다.\n"
    "- 0, 30, 60, 90, 120, 140, 160, 180, 200, 210, 220, 230, 255\n"
    "- 행렬의 모든 값은 반드시 위 숫자 중 하나여야 합니다.\n"
    "- 픽셀 칸 안에 표시되는 숫자도 반드시 해당 칸의 회색조 값과 일치해야 합니다.\n\n"

    "[출력 순서]\n"
    "1. 픽셀아트 이미지: 20×20 격자 형태로 출력하고, 각 픽셀 위에 회색조 숫자를 표시\n"
    f"2. 생각해 볼 질문: {question}\n"
)


def social_prompt_status_text():
    prompt = str(st.session_state.get("i3_generated_prompt", "")).strip()
    return "프롬프트 생성 완료" if prompt else "아직 작성하지 않았습니다."


def run():
    apply_local_style()
    ensure_state()
    page_banner(
        "이미지를 행렬로 보는 인공지능",
        "그림은 숫자 배열이고, 행렬 연산으로 그림이 변하고, 인공지능은 그 숫자 패턴을 읽는다는 핵심 흐름을 실습으로 익힙니다.",
    )
    st.markdown("<hr style='border:2px solid #2196F3;'>", unsafe_allow_html=True)

    tabs = st.tabs(["1️⃣ [F.U] 문제 발견", "2️⃣ [T] 수학의 언어", "3️⃣ [U] AI 이해", "4️⃣ [R.E] 세상과 연결"])

    with tabs[0]:
        stage_intro(
            "문제 발견",
            "사람이 얼굴을 인식하는 상황을 떠올리며, AI가 얼굴을 구별할때 어떤 정보를 변수로 삼을지 짧게 가설을 세우는 도입 단계입니다.",
            "AI는 얼굴을 어떻게 인식할까?",
            "#e3f2fd",
            "#bbdefb",
        )
        st.markdown(pretty_title("❔문제제기: AI는 얼굴을 어떤 정보로 구별할까?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.markdown(
            "휴대폰 **얼굴 인식**은 사람의 얼굴을 읽어 휴대폰 소유자와 같은 사람인지 구별합니다. "
            "그런데 **AI는 얼굴을 볼 때 무엇을 정보로 삼고 있을까요?**"
        )
        
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
            placeholder="예: AI는 얼굴의 ___과___을 보고 사람을 구별할 것 같다.",
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
        with st.expander("행렬의 정의와 성분 표현 보기", expanded=False):
            st.markdown(
                """
                여러 개의 수를 직사각형 모양으로 배열하여 괄호로 묶어 나타낸것을 **행렬**이라고 합니다.
                행렬을 구성하고 있는 각각의 수를 **성분**이라고 하고, `(2,1)`처럼 행과 열의 위치로 나타낼 수 있습니다. 일반적으로 m개의 행과 n개의 열로 이루어진 행렬을 `m×n 행렬`이라고 부릅니다.
                """
            )
            st.latex(r"A=\begin{bmatrix}79 & 41 \\ 57 & 22\end{bmatrix}, \quad a_{21}=57")
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
        st.markdown(
            "이번에는 `계단`과 `체스판` 무늬 중 하나를 선택해, 왼쪽 행렬과 오른쪽 이미지를 함께 바꾸어 봅니다. "
            "**행렬의 한 칸**은 **이미지의 한 칸**과 정확히 대응하므로, 숫자를 바꾸면 그림이 바뀌고 그림의 칸을 누르면 행렬도 함께 바뀝니다."
        )
        
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

        activity_left, activity_right = st.columns([1.35, 0.85], gap="medium")
        with activity_left:
            st.markdown(pretty_title("행렬에 0 또는 1 입력하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            answer_matrix = binary_matrix_editor(current_grid, editor_key)
            if not np.array_equal(answer_matrix, current_grid):
                set_binary_grid(answer_matrix.copy())
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

        st.markdown(
            "- **이미지**는 행과 열을 가진 **값의 배열**로 표현될 수 있습니다.\n"
            "- 따라서 이미지는 **행렬**로 나타낼 수 있습니다.\n"
            "- **AI**도 이미지를 이런 **수학적 표현**으로 읽을 수 있습니다."
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
        with st.expander("행렬의 연산 간단히 보기", expanded=False):
            st.write(
                "행렬의 합은 **같은 위치에 있는 수끼리** 더합니다. "
                "실수배는 행렬의 **모든 칸에 같은 수를 곱하는 것**입니다."
            )
            st.latex(r"A=\begin{bmatrix}1 & 0 \\ 1 & 1\end{bmatrix},\quad B=\begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}")
            st.latex(r"A+B=\begin{bmatrix}1+0 & 0+1 \\ 1+1 & 1+0\end{bmatrix}=\begin{bmatrix}1 & 1 \\ 2 & 1\end{bmatrix}")
            st.latex(r"2A=\begin{bmatrix}2\times1 & 2\times0 \\ 2\times1 & 2\times1\end{bmatrix}=\begin{bmatrix}2 & 0 \\ 2 & 2\end{bmatrix}")
            st.caption("즉, 행렬은 자리 바꾸어 계산하는 것이 아니라 같은 자리의 성분끼리 계산합니다.")

        st.markdown(
            "이번 활동은 **AI의 이미지 처리 원리**를 단순화한 **행렬 합성 시뮬레이션**입니다. "
            "두 이미지를 **행렬 A**, **행렬 B**로 두고 `C = kA + (1-k)B`를 적용해, **k 값**에 따라 **결과 행렬 C**와 결과 이미지가 어떻게 달라지는지 살펴봅니다."
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
            st.latex(fr"C = kA + (1-k)B")
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
                st.caption("버튼을 누르면 이미지 위에 0~255 회색조 행렬값이 함께 보입니다.")
            else:
                st.caption("이미지 위 숫자는 각 픽셀의 회색조 행렬값입니다. 값의 범위는 0~255입니다.")

        preview_cols = st.columns(3)
        with preview_cols[0]:
            st.pyplot(
                draw_image(
                    matrix_a,
                    f"{char_a} → 행렬 A",
                    "gray",
                    show_values=st.session_state["i3_gray_show_matrix"],
                    value_range=(0, 255),
                    fig_size=(4.6, 4.6),
                    value_fontsize=8,
                ),
                use_container_width=True,
            )
        with preview_cols[1]:
            st.pyplot(
                draw_image(
                    matrix_b,
                    f"{char_b} → 행렬 B",
                    "gray",
                    show_values=st.session_state["i3_gray_show_matrix"],
                    value_range=(0, 255),
                    fig_size=(4.6, 4.6),
                    value_fontsize=8,
                ),
                use_container_width=True,
            )
        with preview_cols[2]:
            st.pyplot(
                draw_image(
                    matrix_result,
                    "결과 행렬 C",
                    "gray",
                    show_values=st.session_state["i3_gray_show_matrix"],
                    value_range=(0, 255),
                    fig_size=(4.6, 4.6),
                    value_fontsize=8,
                ),
                use_container_width=True,
            )

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
                    "images": [("결과 이미지 C", matrix_result.copy(), "gray")],
                },
            )
        st.caption(saved_status_text(3))
    with tabs[3]:
        stage_intro(
            "R.E: 우리의 삶과 사회로 연결하기",
            "앞 단계에서 배운 행렬 표현을 바탕으로, 사회적 주제를 담은 픽셀아트 프롬프트를 직접 만드는 단계입니다.",
            "사회적 주제를 담은 이미지를 행렬 기반 픽셀아트로 어떻게 표현할 수 있을까?",
            "#fff3e0",
            "#ffe0b2",
        )
        st.markdown(pretty_title("2️⃣ 모둠활동: 사회적 주제 픽셀아트 프롬프트 만들기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.markdown(
            "이제 모둠별로 1개의 **픽셀아트 프롬프트**를 기획해 봅시다. 먼저 **주제**를 고르고, 그 주제 안에서 **깊은 질문**을 정한 뒤, "
            "이미지에 꼭 들어갈 **상징 1개**를 정리하면 프롬프트 만들기 버튼으로 바로 사용할 문장을 만들 수 있습니다. "
            "픽셀아트는 많은 내용을 넣기보다 하나의 **핵심 장면**이 선명하게 보이도록 단순하게 표현하는 것이 중요합니다."
        )

        st.markdown(pretty_title("1단계:주제 선택", "#f4f9ff", "#dbeafe"), unsafe_allow_html=True)
        topic_choice = st.radio(
            "우리 모둠이 표현할 사회적 주제",
            ["환경과 기후", "안전과 기술", "다양성과 존중"],
            key="i3_social_topic",
            horizontal=True,
        )
        topic_support = SOCIAL_TOPIC_SUPPORT[topic_choice]
        st.caption(f"주제 안내: {topic_support['guide']}")

        st.markdown(pretty_title("2단계: 질문과 상징 정리", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        st.text_area(
            "깊은 질문(D.E.E.P Question)",
            key="i3_social_question_prompt",
            height=110,
            placeholder=f"예: {topic_support['questions'][0]}",
        )
        st.text_input(
            "상징 1",
            key="i3_social_image_symbol_1",
            placeholder=f"예: {topic_support['symbols'][0]}",
        )
        st.markdown("**상징은 1개**만 넣는 것이 좋습니다. 픽셀아트에서는 복잡한 배경보다 **핵심 장면**이 먼저 보여야 합니다.")

        st.markdown(pretty_title("3단계:프롬프트 만들기", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
        if st.button("프롬프트 만들기", key="i3_make_social_prompt", use_container_width=True):
            st.session_state["i3_generated_prompt"] = build_social_image_prompt()
        generated_prompt = st.session_state.get("i3_generated_prompt", "")
        if generated_prompt:
            st.code(generated_prompt, language="markdown")
            st.caption("입력 내용을 바꾸었다면 버튼을 다시 눌러 새 프롬프트를 만들어 주세요.")
            class_key = class_key_from_ids(st.session_state.get("i3_id_1", ""))
            gallery_url = GALLERY_URLS.get(class_key)
            if gallery_url:
                link_cols = st.columns(2)
                with link_cols[0]:
                    st.markdown(
                        f"""<a href="{GPT_URL}" target="_blank" style="display:block;padding:11px;background:linear-gradient(90deg,#7e57c2 0%,#42a5f5 100%);color:white;text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;margin-top:8px;">GPT 바로가기</a>""",
                        unsafe_allow_html=True,
                    )
                with link_cols[1]:
                    st.markdown(
                        f"""<a href="{gallery_url}" target="_blank" style="display:block;padding:11px;background:linear-gradient(90deg,#7e57c2 0%,#42a5f5 100%);color:white;text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;margin-top:8px;">{class_key}반 갤러리 패들렛 바로가기</a>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("학생 정보의 학번을 입력하면 우리 반 갤러리 패들렛 바로가기 버튼이 나타납니다.")
        else:
            st.info("주제, 질문, 상징 1개를 정리한 뒤 프롬프트 만들기 버튼을 눌러 보세요.")

        st.session_state["i3_saved_4"] = (
            f"주제는 '{topic_choice}'였고, 사회적 주제를 담은 20×20 픽셀아트 프롬프트를 구상했다."
        )
        st.session_state["i3_saved_detail_4"] = {
            "writings": social_image_prompt_entries() + [("학생이 작성한 프롬프트 만들기 결과", generated_prompt)]
        }

        st.markdown("---")
        st.markdown(pretty_title("💾 학생 정보 입력 및 포트폴리오 저장", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.info("모둠 이름과 학생 정보를 입력하면, 앞 단계에서 저장한 활동 결과와 위에서 만든 프롬프트를 PDF 포트폴리오로 받을 수 있습니다.")
        group_name = st.text_input("모둠 이름", key="i3_group")

        info_cols = st.columns(2)
        with info_cols[0]:
            stu_id_1 = st.text_input("학번", max_chars=5, key="i3_id_1")
        with info_cols[1]:
            stu_name_1 = st.text_input("이름", key="i3_name_1")

        class_key = class_key_from_ids(stu_id_1)
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
                    f"{group_name}_{stu_name_1}_2-1_이미지포트폴리오.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
                st.warning("⚠️ 모둠원들이 동시에 PDF 다운로드 버튼을 누르면 오류가 날 수 있습니다. 한 명씩 차례대로 눌러 주세요.")
            with p2:
                port_url = PORT_URLS.get(class_key)
                if port_url:
                    st.markdown(f"""<a href="{port_url}" target="_blank" style="display:block;padding:10px;background:linear-gradient(90deg,#43a047 0%,#66bb6a 100%);color:white;text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;">{class_key}반 포트폴리오 패들렛 바로가기</a>""", unsafe_allow_html=True)
                else:
                    st.info("학번의 세 번째 숫자가 1, 2, 5, 6 중 하나이면 반별 패들렛 버튼이 나타납니다.")
        else:
            st.warning("모둠 이름과 학번, 이름을 입력하면 포트폴리오를 바로 받을 수 있습니다.")
    st.markdown("<hr style='border:2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
