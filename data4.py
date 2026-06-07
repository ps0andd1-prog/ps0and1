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


font_path = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")

try:
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc("font", family=font_name)
    mpl.rc("axes", unicode_minus=False)
except Exception:
    pass


LOSS_X = np.array([1, 2, 3, 4], dtype=float)
LOSS_Y = np.array([55, 62, 70, 76], dtype=float)
LOSS_INTERCEPT = 49.0

COMPARE_DATASETS = {
    "기온과 아이스크림 판매량": {
        "x": np.array([20, 22, 24, 26, 28, 30, 32], dtype=float),
        "y": np.array([78, 94, 109, 131, 143, 166, 179], dtype=float),
        "story": "하루 평균 기온이 높아질수록 아이스크림 판매량은 대체로 증가하지만, 날씨, 행사, 요일 차이 때문에 실제 판매량은 직선 주변에 조금씩 흩어지는 상황입니다.",
        "x_label": "하루 평균 기온",
        "x_unit": "℃",
        "y_label": "아이스크림 판매량",
        "y_unit": "개",
        "best": "직선",
        "quad_compare_coeffs": np.array([0.15, 0.8, 3.0], dtype=float),
    },
    "던진 공의 시간과 높이": {
        "x": np.array([0, 1, 2, 3, 4, 5, 6], dtype=float),
        "y": np.array([1.0, 4.6, 7.6, 8.8, 7.5, 4.4, 1.2], dtype=float),
        "story": "공을 던지면 처음에는 높이가 올라가지만, 어느 순간 가장 높아진 뒤 다시 내려오는 상황입니다.",
        "x_label": "시간",
        "x_unit": "초",
        "y_label": "높이",
        "y_unit": "m",
        "best": "곡선",
    },
    "수면 시간과 집중도": {
        "x": np.array([4, 5, 6, 7, 8, 9, 10], dtype=float),
        "y": np.array([45, 58, 72, 84, 90, 88, 85], dtype=float),
        "story": "잠이 부족하면 집중도가 낮고, 8~9시간에서는 높게 유지되며, 10시간에서는 약간 낮아지는 상황입니다.",
        "x_label": "수면 시간",
        "x_unit": "시간",
        "y_label": "집중도",
        "y_unit": "점",
        "best": "곡선",
    },
}

PORT_URLS = {
    "1": "https://padlet.com/ps0andd/p_1",
    "2": "https://padlet.com/ps0andd/p_2",
    "5": "https://padlet.com/ps0andd/p_5",
    "6": "https://padlet.com/ps0andd/p_6",
}
GALLERY_URLS = {
    "1": "https://padlet.com/ps0andd/g_1",
    "2": "https://padlet.com/ps0andd/g_2",
    "5": "https://padlet.com/ps0andd/g_5",
    "6": "https://padlet.com/ps0andd/g_6",
}
CANVA_AI_URL = "https://www.canva.com/ai"

class ThemedPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=15)
        self._font_family = "Nanum"
        self.footer_left = ""
        self.c_primary = (25, 118, 210)
        self.c_primary_lt = (227, 242, 253)
        self.c_border = (200, 200, 200)
        self.c_text_muted = (120, 120, 120)

    def header(self):
        self.set_fill_color(*self.c_primary)
        self.rect(0, 0, self.w, 22, "F")
        self.set_xy(10, 6)
        self.set_text_color(255, 255, 255)
        self.set_font(self._font_family, "", 19)
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 2-2 탐구 포트폴리오", ln=1, align="C")
        self.set_text_color(33, 33, 33)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.set_y(-12)
        self.set_font(self._font_family, "", 9)
        self.set_text_color(*self.c_text_muted)
        if self.footer_left:
            self.cell(0, 8, self.footer_left, 0, 0, "L")
        self.cell(0, 8, f"{self.page_no()} / {{nb}}", 0, 0, "R")

    def h2(self, text):
        self.set_fill_color(*self.c_primary_lt)
        self.set_text_color(21, 101, 192)
        self.set_font(self._font_family, "", 12)
        self.cell(0, 9, text, ln=1, fill=True)
        self.ln(2)
        self.set_text_color(33, 33, 33)

    def p(self, text, size=11, lh=6):
        self.set_font(self._font_family, "", size)
        self.multi_cell(0, lh, text)
        self.ln(2)

    def kv_card(self, title, kv_pairs):
        self.h2(title)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.3)
        self.set_font(self._font_family, "", 11)
        col_w = (self.w - 20) / 2
        cell_h = 8
        x0 = 10
        for i, (k, v) in enumerate(kv_pairs):
            x = x0 + (i % 2) * col_w
            if i % 2 == 0 and i > 0:
                self.ln(cell_h)
            self.set_x(x)
            self.set_text_color(120, 120, 120)
            self.cell(col_w * 0.35, cell_h, str(k), border=1)
            self.set_text_color(33, 33, 33)
            self.cell(col_w * 0.65, cell_h, str(v), border=1)
        if len(kv_pairs) % 2 == 1:
            self.set_x(x0 + col_w)
            self.cell(col_w * 0.35, cell_h, "", border=1)
            self.cell(col_w * 0.65, cell_h, "", border=1)
        self.ln(cell_h + 3)


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
        [data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #e5eef7;
            border-radius: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_text(value, default="작성한 내용이 없습니다."):
    text = str(value).strip() if value is not None else ""
    return text if text else default


def save_stage_result(answer_key, content):
    st.session_state[answer_key] = content
    st.session_state[f"{answer_key}_time"] = datetime.datetime.now().strftime("%H:%M:%S")


def saved_status_text(answer_key):
    saved_time = st.session_state.get(f"{answer_key}_time", "")
    return f"저장 완료: {saved_time}" if saved_time else "아직 저장하지 않았습니다."


def normalize_pdf_output(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("latin1")
    return bytes(value)


def add_figure_to_pdf(pdf, title, fig):
    if fig is None:
        return
    tmp_path = None
    try:
        pdf.h2(title)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = tmp.name
        fig.savefig(tmp_path, format="png", dpi=180, bbox_inches="tight")
        if pdf.get_y() > 210:
            pdf.add_page()
        pdf.image(tmp_path, x=12, w=pdf.w - 24)
        pdf.ln(4)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def add_text_box_to_pdf(pdf, title, text, fill_color=(245, 245, 245)):
    pdf.set_font(pdf._font_family, "", 11)
    pdf.set_text_color(21, 101, 192)
    pdf.cell(0, 8, title, ln=1)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font(pdf._font_family, "", 10)
    pdf.set_fill_color(*fill_color)
    pdf.multi_cell(0, 6, clean_text(text), border=1, fill=True)
    pdf.ln(3)


def class_key_from_ids(*student_ids):
    valid_classes = {"1", "2", "5", "6"}
    for student_id in student_ids:
        value = str(student_id).strip()
        if len(value) >= 3 and value[2] in valid_classes:
            return value[2]
    return ""


AI_ETHICS_TOPIC_GUIDES = {
    "AI 공정성": "예: AI가 성별, 지역, 성적, 배경에 따라 누군가를 불리하게 판단하지 않도록 하는 문제",
    "개인정보 보호": "예: 얼굴, 위치, 성적, 검색 기록 같은 개인 데이터를 AI가 어디까지 사용해도 되는가의 문제",
    "인간의 책임": "예: AI 예측이 틀렸을 때 사람이 어떻게 확인하고 책임져야 하는가의 문제",
}


AI_ETHICS_TOPIC_EXAMPLES = {
    "AI 공정성": {
        "question": "예: AI가 편리하다고 해서 학생의 가능성을 자동으로 판단해도 괜찮을까?",
        "message": "예: AI의 판단이 모두에게 공정한지 사람이 계속 점검해야 한다.",
        "symbol_1": "예: AI 로봇과 다양한 학생들",
        "symbol_2": "예: 공정함을 뜻하는 저울",
    },
    "개인정보 보호": {
        "question": "예: 더 정확한 예측을 위해서라면 나의 얼굴과 위치 정보까지 AI에 맡겨도 괜찮을까?",
        "message": "예: 편리함보다 먼저, 내 정보가 어디에 쓰이는지 알고 선택할 권리가 필요하다.",
        "symbol_1": "예: 자물쇠가 걸린 스마트폰",
        "symbol_2": "예: 얼굴 데이터와 보호막",
    },
    "인간의 책임": {
        "question": "예: AI가 틀린 판단을 했을 때 그 책임은 AI에게 있을까, 사람에게 있을까?",
        "message": "예: AI는 도구일 뿐이며, 최종 결정과 책임은 사람이 확인해야 한다.",
        "symbol_1": "예: AI 화면을 확인하는 사람",
        "symbol_2": "예: 책임을 뜻하는 체크리스트",
    },
}


def prompt_text(value, default):
    text = str(value).strip() if value is not None else ""
    return text if text else default


def poster_text_or_default(value, default):
    text = str(value).strip() if value is not None else ""
    return text if text else default


def build_canva_ethics_prompt():
    topic = prompt_text(st.session_state.get("d3_ethics_topic", ""), "AI 공정성")
    question = poster_text_or_default(
        st.session_state.get("d3_ethics_question", ""),
        "깊은 질문을 먼저 적어 주세요.",
    )
    message = poster_text_or_default(
        st.session_state.get("d3_ethics_message", ""),
        "포스터로 전하고 싶은 메시지를 먼저 적어 주세요.",
    )
    symbol_1 = poster_text_or_default(
        st.session_state.get("d3_ethics_symbol_1", ""),
        "상징 1을 먼저 적어 주세요.",
    )
    symbol_2 = poster_text_or_default(
        st.session_state.get("d3_ethics_symbol_2", ""),
        "상징 2를 먼저 적어 주세요.",
    )
    style = prompt_text(st.session_state.get("d3_poster_style", ""), "깔끔한 공익광고")
    color_guides = {
        "깔끔한 공익광고": "흰색과 밝은 회색을 바탕으로 신뢰감을 주는 파랑, 차분한 남색, 포인트 노랑을 사용해 깨끗하고 공적인 느낌을 줍니다.",
        "강한 경고 메시지": "검정 또는 짙은 회색 배경에 빨강, 주황, 노랑을 포인트로 사용해 위험성과 경각심이 즉시 느껴지게 합니다.",
        "따뜻한 교육 캠페인": "부드러운 크림색, 따뜻한 초록, 하늘색, 연한 주황을 사용해 공감과 배려가 느껴지는 분위기를 만듭니다.",
        "미래적인 디지털 스타일": "짙은 남색 또는 어두운 배경에 청록, 보라, 전기 파랑을 사용하고 디지털 그리드나 빛나는 선 느낌을 더합니다.",
    }
    color_guide = color_guides.get(style, color_guides["깔끔한 공익광고"])

    return (
        "인공지능 윤리 포스터를 만들어 주세요.\n\n"

        "[핵심 조건]\n"
        "- 인공지능을 더 책임 있게 사용하자는 윤리적 메시지를 사회적으로 전달하는 공익 포스터로 구성합니다.\n"
        "- 대상은 학생만이 아니라 모든 사람입니다.\n"
        "- 나이와 직업이 달라도 한눈에 이해하고 공감할 수 있게 구성합니다.\n"
        "- 공공장소, 학교 게시판, 온라인 캠페인에서 함께 볼 수 있는 포스터처럼 만듭니다.\n"
        "- 글자는 너무 많지 않게 합니다.\n"
        "- AI의 편리함만 강조하지 않습니다.\n"
        "- 선택한 윤리 주제의 방향이 분명히 드러나게 합니다.\n"
        "- 상징들은 단순한 장식이 아니라 인공지능 윤리 문제를 이해하게 만드는 중심 이미지가 되도록 배치합니다.\n\n"

        "[내용 조건]\n"
        f"- 주제: {topic}\n"
        f"- 깊은 질문(D.E.E.P Question): {question}\n"
        f"- 핵심 메시지: {message}\n"
        f"- 상징 1: {symbol_1}\n"
        f"- 상징 2: {symbol_2}\n\n"

        "[디자인 조건]\n"
        f"- 디자인 분위기: {style}\n"
        f"- 색감: {color_guide}\n"
        "- 깊은 질문은 포스터 전체에서 가장 큰 글씨와 가장 강한 대비로 배치합니다.\n"
        "- 제목, 핵심 메시지, 보조 문장은 모두 깊은 질문보다 작고 덜 강조되게 합니다.\n"
        "- 깊은 질문은 포스터를 보는 사람이 가장 먼저 읽는 문장이 되게 합니다.\n"
        "- 핵심 메시지는 깊은 질문 아래쪽에 배치하고, 질문에 대한 답이나 행동 방향처럼 보이게 합니다.\n\n"

        "[출력 순서]\n"
        "1. 인공지능 윤리 포스터 이미지\n"
        f"2. 깊은 질문: {question}\n"
        f"3. 핵심 메시지: {message}\n"
    )


def poster_prompt_entries():
    return [
        ("선택한 윤리 주제", st.session_state.get("d3_ethics_topic", "")),
        ("깊은 질문(D.E.E.P Question)", st.session_state.get("d3_ethics_question", "")),
        ("포스터로 전하고 싶은 메시지", st.session_state.get("d3_ethics_message", "")),
        ("상징 1", st.session_state.get("d3_ethics_symbol_1", "")),
        ("상징 2", st.session_state.get("d3_ethics_symbol_2", "")),
        ("포스터 분위기", st.session_state.get("d3_poster_style", "")),
        ("Canva 포스터 제작 프롬프트", st.session_state.get("d3_canva_prompt", "")),
    ]


def create_portfolio_pdf(
    student_info,
    principle_data,
    mission_rows,
    poster_entries,
    figure_items,
):
    pdf = ThemedPDF()
    pdf.add_font("Nanum", "", font_path, uni=True)
    pdf.set_font("Nanum", "", 12)
    pdf._font_family = "Nanum"
    pdf.footer_left = f"{student_info.get('group', '')} - {student_info.get('name_1', '')}"
    pdf.add_page()

    kvs = [
        ("모둠명", student_info.get("group", "")),
        ("학번", student_info.get("id_1", "")),
        ("이름", student_info.get("name_1", "")),
        ("작성일", datetime.datetime.now().strftime("%Y-%m-%d")),
    ]
    pdf.kv_card("학생 정보", kvs)

    intro_title, intro_answer = principle_data[0]
    pdf.h2("탐구 출발 질문")
    add_text_box_to_pdf(pdf, intro_title, intro_answer)

    linked_reflections = [
        principle_data[1][1] if len(principle_data) > 1 else "",
        principle_data[2][1] if len(principle_data) > 2 else "",
    ]

    for idx, ((mission_title, mission_text), (figure_title, figure)) in enumerate(zip(mission_rows, figure_items)):
        pdf.add_page()
        pdf.h2(mission_title)
        add_text_box_to_pdf(pdf, "실험 결과 요약", mission_text, fill_color=(250, 250, 250))
        reflection_text = linked_reflections[idx] if idx < len(linked_reflections) else ""
        if reflection_text.strip():
            add_text_box_to_pdf(pdf, "나의 해석", reflection_text, fill_color=(245, 245, 245))
        add_figure_to_pdf(pdf, figure_title, figure)

    pdf.add_page()
    pdf.h2("문제 4. 모둠활동 2. AI 윤리 포스터 프롬프트")
    for title, answer_text in poster_entries:
        add_text_box_to_pdf(pdf, title, answer_text, fill_color=(250, 250, 250))

    return normalize_pdf_output(pdf.output(dest="S"))


def principle_box(problem_number, title, question, key, model_answer):
    st.markdown(pretty_title(f"문제 {problem_number}. {title}", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
    st.info(f"탐구 질문: {question}")
    st.text_area(
        "나의 설명 작성하기",
        height=110,
        key=key,
        placeholder="그래프와 시뮬레이션을 관찰한 뒤, 내 말로 정리해 보세요.",
    )
    with st.expander("막혔을 때만 모범 답안 보기"):
        st.success(model_answer)
    st.text_input(
        "학생 질문 만들기(더 탐구해 보고 싶은 질문이 있다면 적어 보세요)",
        key=f"{key}_student_q",
        placeholder="이 활동을 하며 더 탐구해 보고 싶은 질문을 적어 보세요.",
    )



def render_stage_cards(concept_text, essential_question, output_text):
    cards = [
        ("개념", concept_text, "#e3f2fd", "#1565c0"),
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


def show_pretty_table(df, height=170):
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


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
            <div style="font-size:0.9rem; font-weight:700; color:#5e35b1; margin-bottom:8px;">F.U.T.U.R.E. 프로젝트 4DAY</div>
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


def quadratic_curve(x, a, b, c):
    return a * x**2 + b * x + c


def quadratic_vertex(a, b, c):
    x_v = -b / (2 * a)
    y_v = quadratic_curve(x_v, a, b, c)
    return x_v, y_v


def sse(y_true, y_pred):
    return float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


@st.cache_data(show_spinner=False)
def cached_loss_curve(intercept):
    slopes = np.linspace(3.0, 10.5, 180)
    losses = [sse(LOSS_Y, slope * LOSS_X + intercept) for slope in slopes]
    return slopes, np.array(losses)


@st.cache_data(show_spinner=False)
def cached_polyfit(x_values, y_values, degree):
    coeffs = np.polyfit(np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), degree)
    return coeffs.tolist()


def fit_degree(x_values, y_values, degree):
    return np.array(cached_polyfit(tuple(x_values), tuple(y_values), int(degree)), dtype=float)


def poly_to_latex(coeffs, show_zero_terms=False, force_degree=None):
    degree = int(force_degree) if force_degree is not None else len(coeffs) - 1
    parts = []
    for idx, coef in enumerate(coeffs):
        power = degree - idx
        if abs(coef) < 1e-9 and not show_zero_terms:
            continue
        sign = "-" if coef < 0 else "+"
        magnitude = abs(coef)
        if power == 0:
            body = f"{magnitude:.2f}"
        elif power == 1:
            body = f"{magnitude:.2f}x"
        else:
            body = f"{magnitude:.2f}x^{{{power}}}"
        parts.append((sign, body))
    if not parts:
        return "y = 0"
    first_sign, first_body = parts[0]
    text = f"- {first_body}" if first_sign == "-" else first_body
    for sign, body in parts[1:]:
        text += f" {sign} {body}"
    return f"y = {text}"


def poly_to_text(coeffs, show_zero_terms=False, force_degree=None):
    return poly_to_latex(coeffs, show_zero_terms=show_zero_terms, force_degree=force_degree).replace("^", "**").replace("{", "").replace("}", "")


def hill_loss(x):
    return (x - 3.0) ** 2 + 2.0


def hill_gradient(x):
    return 2 * (x - 3.0)


def reset_walk(start_x):
    st.session_state["d3_walk_anchor"] = float(start_x)
    st.session_state["d3_walk_history"] = [float(start_x)]


def step_walk(learning_rate, count=1):
    history = st.session_state.get("d3_walk_history", [-8.0])
    x_val = float(history[-1])
    for _ in range(count):
        x_val = x_val - learning_rate * hill_gradient(x_val)
        history.append(float(x_val))
    st.session_state["d3_walk_history"] = history


def make_quadratic_figure(a, b, c):
    fig = Figure(figsize=(5.4, 3.4))
    ax = fig.subplots()
    x_v, y_v = quadratic_vertex(a, b, c)
    xs = np.linspace(x_v - 6, x_v + 6, 400)
    ys = quadratic_curve(xs, a, b, c)
    ax.plot(xs, ys, color="#1976d2", linewidth=2.2)
    ax.scatter([x_v], [y_v], color="#d32f2f", s=70, zorder=3)
    ax.set_title("이차함수와 꼭짓점")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.45)
    return fig


def make_prediction_line_figure(slope, intercept=LOSS_INTERCEPT):
    fig = Figure(figsize=(5.5, 3.5))
    ax = fig.subplots()
    y_pred = slope * LOSS_X + intercept
    ax.scatter(LOSS_X, LOSS_Y, color="#1565c0", s=70, label="실제 데이터", zorder=3)
    xs = np.linspace(0.5, 4.5, 100)
    ax.plot(xs, slope * xs + intercept, color="#ef6c00", linewidth=2, label="AI 예측선")
    first_error = True
    for x_val, y_real, y_hat in zip(LOSS_X, LOSS_Y, y_pred):
        ax.vlines(
            x_val,
            y_real,
            y_hat,
            color="#d32f2f",
            linestyle="--",
            linewidth=2.6,
            alpha=0.95,
            label="오차" if first_error else None,
            zorder=2,
        )
        ax.text(
            x_val + 0.05,
            (y_real + y_hat) / 2,
            "오차",
            color="#c62828",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffebee", edgecolor="#ef9a9a", alpha=0.92),
        )
        first_error = False
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(48, 82)
    ax.set_xlabel("공부 시간")
    ax.set_ylabel("시험 점수")
    ax.set_title("예측선과 오차")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    return fig


def make_loss_surface_figure(slope, intercept=LOSS_INTERCEPT):
    fig = Figure(figsize=(4.8, 3.5))
    ax = fig.subplots()
    slopes, losses = cached_loss_curve(intercept)
    current_loss = sse(LOSS_Y, slope * LOSS_X + intercept)
    best_idx = int(np.argmin(losses))
    ax.plot(slopes, losses, color="#7b1fa2", linewidth=2)
    ax.scatter([slopes[best_idx]], [losses[best_idx]], color="#2e7d32", s=70, label="가장 낮은 지점")
    ax.scatter(
        [slope],
        [current_loss],
        color="#d32f2f",
        edgecolors="white",
        linewidths=1.8,
        s=170,
        zorder=4,
        label="현재 위치",
    )
    ax.set_xlabel("기울기")
    ax.set_ylabel("손실")
    ax.set_title("손실함수(이차함수)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    return fig


def get_descent_step_preview(slope, intercept=LOSS_INTERCEPT, step_size=0.3):
    current_loss = sse(LOSS_Y, slope * LOSS_X + intercept)
    left_slope = max(3.0, round(float(slope) - step_size, 2))
    right_slope = min(10.5, round(float(slope) + step_size, 2))

    candidates = []
    if left_slope != slope:
        candidates.append(
            {
                "next_slope": left_slope,
                "next_loss": sse(LOSS_Y, left_slope * LOSS_X + intercept),
                "direction": "왼쪽으로 한 걸음: 기울기를 조금 줄이는 방향",
            }
        )
    if right_slope != slope:
        candidates.append(
            {
                "next_slope": right_slope,
                "next_loss": sse(LOSS_Y, right_slope * LOSS_X + intercept),
                "direction": "오른쪽으로 한 걸음: 기울기를 조금 늘리는 방향",
            }
        )

    if not candidates:
        return {
            "current_slope": float(slope),
            "current_loss": current_loss,
            "next_slope": float(slope),
            "next_loss": current_loss,
            "direction": "지금은 더 이동할 수 없는 범위 끝에 있습니다.",
            "loss_change": 0.0,
        }

    best_candidate = min(candidates, key=lambda item: item["next_loss"])
    if best_candidate["next_loss"] < current_loss:
        next_slope = best_candidate["next_slope"]
        next_loss = best_candidate["next_loss"]
        direction = best_candidate["direction"]
    else:
        next_slope = float(slope)
        next_loss = current_loss
        direction = "지금은 거의 가장 낮은 지점에 가까워서, 더 내려갈 방향이 뚜렷하지 않습니다."

    return {
        "current_slope": float(slope),
        "current_loss": current_loss,
        "next_slope": float(next_slope),
        "next_loss": float(next_loss),
        "direction": direction,
        "loss_change": float(current_loss - next_loss),
    }


def make_loss_step_figure(current_slope, next_slope, intercept=LOSS_INTERCEPT):
    fig = Figure(figsize=(5.1, 3.4))
    ax = fig.subplots()
    slopes, losses = cached_loss_curve(intercept)
    current_loss = sse(LOSS_Y, current_slope * LOSS_X + intercept)
    next_loss = sse(LOSS_Y, next_slope * LOSS_X + intercept)

    ax.plot(slopes, losses, color="#7b1fa2", linewidth=2)
    ax.scatter([current_slope], [current_loss], color="#d32f2f", s=85, zorder=3, label="현재 위치")
    ax.scatter([next_slope], [next_loss], color="#2e7d32", s=85, zorder=4, label="다음 위치")
    if abs(next_slope - current_slope) > 1e-9:
        ax.annotate(
            "",
            xy=(next_slope, next_loss),
            xytext=(current_slope, current_loss),
            arrowprops=dict(arrowstyle="->", color="#ef6c00", lw=2),
        )
    ax.set_xlabel("기울기")
    ax.set_ylabel("손실")
    ax.set_title("내려가는 방향으로 한 걸음 이동")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    return fig


def make_loss_bundle_figure(slope, intercept=LOSS_INTERCEPT):
    fig = Figure(figsize=(9.5, 3.6))
    ax1, ax2 = fig.subplots(1, 2)
    y_pred = slope * LOSS_X + intercept
    ax1.scatter(LOSS_X, LOSS_Y, color="#1565c0", s=65, label="실제 데이터", zorder=3)
    xs = np.linspace(0.5, 4.5, 100)
    ax1.plot(xs, slope * xs + intercept, color="#ef6c00", linewidth=2, label="AI 예측선")
    for x_val, y_real, y_hat in zip(LOSS_X, LOSS_Y, y_pred):
        ax1.vlines(x_val, y_real, y_hat, color="#90a4ae", linestyle="--", alpha=0.9)
    ax1.set_xlim(0.5, 4.5)
    ax1.set_ylim(48, 82)
    ax1.set_xlabel("공부 시간")
    ax1.set_ylabel("시험 점수")
    ax1.set_title("예측선과 오차")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left")

    slopes, losses = cached_loss_curve(intercept)
    current_loss = sse(LOSS_Y, slope * LOSS_X + intercept)
    best_idx = int(np.argmin(losses))
    ax2.plot(slopes, losses, color="#7b1fa2", linewidth=2)
    ax2.scatter([slopes[best_idx]], [losses[best_idx]], color="#2e7d32", s=70, label="가장 낮은 지점")
    ax2.scatter([slope], [current_loss], color="#d32f2f", s=80, zorder=3, label="현재 위치")
    ax2.set_xlabel("기울기")
    ax2.set_ylabel("손실")
    ax2.set_title("손실함수(이차함수)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return fig


def make_intro_error_figure():
    months = ["1월", "3월", "5월", "7월", "9월", "11월"]
    positions = np.arange(len(months), dtype=float)
    actual_values = np.array([1, 7, 16, 27, 21, 9], dtype=float)
    ai_predictions = np.array([5, 12, 9, 20, 28, 16], dtype=float)

    fig = Figure(figsize=(5.5, 3.4))
    ax = fig.subplots()
    ax.plot(positions, actual_values, color="#90caf9", linewidth=1.8, alpha=0.9)
    ax.scatter(
        positions,
        actual_values,
        color="#1565c0",
        edgecolors="white",
        linewidths=1.2,
        s=85,
        label="실제값",
        zorder=3,
    )
    ax.plot(positions, ai_predictions, color="#ffb74d", linewidth=1.6, linestyle="--", alpha=0.95)
    ax.scatter(
        positions,
        ai_predictions,
        color="#ef6c00",
        marker="*",
        s=170,
        label="AI 예측값",
        zorder=4,
    )

    first_error = True
    for x_pos, y_real, y_hat in zip(positions, actual_values, ai_predictions):
        ax.vlines(
            x_pos,
            min(y_real, y_hat),
            max(y_real, y_hat),
            colors="#d32f2f",
            linestyles="-",
            linewidth=3.4,
            alpha=0.98,
            label="오차" if first_error else None,
            zorder=2,
        )
        ax.text(
            x_pos + 0.08,
            (y_real + y_hat) / 2,
            "오차",
            color="#c62828",
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#fff5f5", edgecolor="#ef9a9a", alpha=0.98),
        )
        first_error = False

    ax.set_xticks(positions)
    ax.set_xticklabels(months)
    ax.set_ylim(0, 30)
    ax.set_xlabel("홀수 월")
    ax.set_ylabel("평균기온(℃)")
    ax.set_title("홀수 월 실제값과 AI 예측값")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, months, actual_values, ai_predictions


def make_walk_figure(history):
    fig = Figure(figsize=(5.8, 3.5))
    ax = fig.subplots()
    xs = np.linspace(-10, 12, 300)
    ys = hill_loss(xs)
    ys_hist = [hill_loss(x) for x in history]
    ax.plot(xs, ys, color="#455a64", linewidth=2.3, label="오차 언덕")
    ax.plot(history, ys_hist, color="#e53935", linewidth=1.8, marker="o", markersize=6, label="AI의 이동 경로")
    ax.scatter([3], [2], color="#1b5e20", marker="*", s=230, label="가장 낮은 지점", zorder=4)
    ax.set_xlim(-10, 12)
    ax.set_ylim(0, 60)
    ax.set_xlabel("현재 위치 x")
    ax.set_ylabel("손실")
    ax.set_title("조금씩 내려가며 최솟값 찾기")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    return fig


def make_walk_history_figure(history):
    fig = Figure(figsize=(5.8, 2.9))
    ax = fig.subplots()
    steps = list(range(len(history)))
    losses = [hill_loss(x) for x in history]
    ax.plot(steps, losses, color="#2e7d32", linewidth=2, marker="o")
    ax.set_xlabel("이동 횟수")
    ax.set_ylabel("손실")
    ax.set_title("이동할수록 손실이 어떻게 바뀌는가")
    ax.grid(True, linestyle="--", alpha=0.35)
    return fig


def make_learning_rate_preview(start_x, learning_rate):
    history = [float(start_x)]
    x_val = float(start_x)
    for _ in range(6):
        x_val = x_val - learning_rate * hill_gradient(x_val)
        history.append(float(x_val))
    fig = Figure(figsize=(3.4, 2.5))
    ax = fig.subplots()
    xs = np.linspace(-10, 12, 250)
    ax.plot(xs, hill_loss(xs), color="#cfd8dc", linewidth=1.8)
    ax.plot(history, [hill_loss(x) for x in history], color="#ef5350", marker="o", linewidth=1.5)
    ax.scatter([3], [2], color="#2e7d32", marker="*", s=120, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"보폭 {learning_rate:.2f}", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.25)
    return fig


def make_walk_bundle_figure(history):
    fig = Figure(figsize=(9.2, 3.4))
    ax1, ax2 = fig.subplots(1, 2)

    xs = np.linspace(-10, 12, 300)
    ys = hill_loss(xs)
    ys_hist = [hill_loss(x) for x in history]
    ax1.plot(xs, ys, color="#455a64", linewidth=2.1, label="오차 언덕")
    ax1.plot(history, ys_hist, color="#e53935", linewidth=1.7, marker="o", markersize=5, label="이동 경로")
    ax1.scatter([3], [2], color="#1b5e20", marker="*", s=210, zorder=4, label="가장 낮은 지점")
    ax1.set_xlabel("현재 위치 x")
    ax1.set_ylabel("손실")
    ax1.set_title("오차 언덕 내려가기")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right")

    steps = list(range(len(history)))
    losses = [hill_loss(x) for x in history]
    ax2.plot(steps, losses, color="#2e7d32", linewidth=2, marker="o")
    ax2.set_xlabel("이동 횟수")
    ax2.set_ylabel("손실")
    ax2.set_title("이동할수록 손실 변화")
    ax2.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def get_regression_compare_metrics(dataset_name):
    dataset = COMPARE_DATASETS[dataset_name]
    x_values = dataset["x"]
    y_values = dataset["y"]
    coeffs_linear = fit_degree(x_values, y_values, 1)
    coeffs_quad = np.array(dataset.get("quad_compare_coeffs", fit_degree(x_values, y_values, 2)), dtype=float)
    linear_pred = np.polyval(coeffs_linear, x_values)
    quad_pred = np.polyval(coeffs_quad, x_values)
    return x_values, y_values, coeffs_linear, coeffs_quad, sse(y_values, linear_pred), sse(y_values, quad_pred)


def build_compare_dataset_values_df(dataset_name):
    dataset = COMPARE_DATASETS[dataset_name]
    x_column = f"{dataset['x_label']} ({dataset['x_unit']})"
    y_column = f"{dataset['y_label']} ({dataset['y_unit']})"
    return pd.DataFrame({x_column: dataset["x"], y_column: dataset["y"]})


def make_regression_compare_figure(dataset_name, show_linear=True, show_quad=True, selected_model=None):
    dataset = COMPARE_DATASETS[dataset_name]
    x_values, y_values, coeffs_linear, coeffs_quad, linear_loss, quad_loss = get_regression_compare_metrics(dataset_name)
    fig = Figure(figsize=(5.8, 3.5))
    ax = fig.subplots()
    x_smooth = np.linspace(x_values.min(), x_values.max(), 220)
    ax.scatter(x_values, y_values, color="#263238", s=65, label="실제 데이터", zorder=3)
    linear_selected = selected_model in (None, "직선(1차)")
    quad_selected = selected_model in (None, "곡선(2차)")
    if show_linear:
        ax.plot(
            x_smooth,
            np.polyval(coeffs_linear, x_smooth),
            color="#1e88e5",
            linewidth=3 if linear_selected else 1.6,
            linestyle="--",
            alpha=1.0 if linear_selected else 0.38,
            label="직선 모델(1차)",
        )
    if show_quad:
        ax.plot(
            x_smooth,
            np.polyval(coeffs_quad, x_smooth),
            color="#ef6c00",
            linewidth=3 if quad_selected else 1.6,
            alpha=1.0 if quad_selected else 0.38,
            label="곡선 모델(2차)",
        )
    ax.set_xlabel(f"{dataset['x_label']} ({dataset['x_unit']})")
    ax.set_ylabel(f"{dataset['y_label']} ({dataset['y_unit']})")
    ax.set_title(f"{dataset_name} 자료 비교")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    return fig, linear_loss, quad_loss, coeffs_linear, coeffs_quad


def loss_feedback(loss_value):
    if loss_value <= 6:
        return "success", "✨미션 성공! 거의 가장 낮은 오차에 도착했습니다."
    if loss_value <= 20:
        return "info", "✨꽤 잘 맞췄어요. 기울기를 조금만 더 조절해 보세요."
    return "warning", "✨오차가 아직 큽니다. 선이 데이터에 더 가까워지게 조절해 보세요."


def walk_feedback(loss_value, steps):
    if loss_value <= 2.2 and steps <= 8:
        return "success", "✨성공! 적절한 보폭으로 빠르게 계곡 근처에 도착했습니다."
    if loss_value <= 4.5:
        return "info", "✨거의 다 왔습니다. 보폭을 조금 다듬으면 더 좋아집니다."
    return "warning", "✨아직 언덕 위에 있습니다. 보폭이 너무 크거나 너무 작은지 확인해 보세요."


def build_mission_rows():
    slope = float(st.session_state.get("d3_loss_slope", 7.0))
    current_loss = sse(LOSS_Y, slope * LOSS_X + LOSS_INTERCEPT)

    dataset_name = st.session_state.get("d3_compare_dataset", "기온과 아이스크림 판매량")
    if dataset_name not in COMPARE_DATASETS:
        dataset_name = "기온과 아이스크림 판매량"
    _, _, linear_coeffs, quad_coeffs, linear_loss, quad_loss = get_regression_compare_metrics(dataset_name)

    return [
        (
            "문제 2. 오차가 가장 작은 예측선 찾기",
            (
                f"선택한 기울기: {slope:.2f}\n"
                f"절편은 {LOSS_INTERCEPT:.0f}으로 고정\n"
                f"현재 손실: {current_loss:.2f}"
            ),
        ),
        (
            "문제 3. 모둠활동 1. 직선과 곡선 모델 비교",
            (
                f"선택한 데이터셋: {dataset_name}\n"
                f"직선 모델의 오차: {linear_loss:.2f}\n"
                f"곡선 모델의 오차: {quad_loss:.2f}\n"
                f"직선 모델 식: {poly_to_text(linear_coeffs)}\n"
                f"곡선 모델 식: {poly_to_text(quad_coeffs, show_zero_terms=True, force_degree=2)}"
            ),
        ),
    ]


def build_figure_items():
    slope = float(st.session_state.get("d3_loss_slope", 7.0))
    dataset_name = st.session_state.get("d3_compare_dataset", "기온과 아이스크림 판매량")
    if dataset_name not in COMPARE_DATASETS:
        dataset_name = "기온과 아이스크림 판매량"

    compare_fig, _, _, _, _ = make_regression_compare_figure(
        dataset_name,
        True,
        True,
        st.session_state.get("d3_u_model_choice", "직선(1차)"),
    )

    return [
        ("손실함수(이차함수)와 예측선", make_loss_bundle_figure(slope)),
        ("직선과 곡선 모델 비교", compare_fig),
    ]


def run():
    apply_local_style()

    page_banner(
        "오차를 줄이며 예측하는 AI",
        "코딩 없이 그래프와 슬라이더를 직접 움직이며, AI가 실제값과 예측값의 차이를 줄이면서 "
        "더 알맞은 예측선을 찾는 과정을 쉽게 이해합니다.",
    )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)

    tabs = st.tabs(
        [
            "1️⃣ [F.U] 문제 발견",
            "2️⃣ [T] 수학의 언어",
            "3️⃣ [U] AI 이해",
            "4️⃣ [R.E] 세상과 연결",
        ]
    )

    with tabs[0]:
        stage_intro(
            "문제제기",
            "AI가 게임에서 여러 번 시도하고 고치는 장면을 보며, 예측이 틀렸을 때 생기는 "
            "오차가 무엇인지 확인하는 도입 단계입니다.",
            "AI는 왜 오차를 줄이려고 할까?",
            "#e3f2fd",
            "#bbdefb",
        )

        st.markdown(pretty_title("❔ 문제제기: AI는 왜 오차를 줄이려고 할까?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.markdown(
            "아래 영상은 OpenAI의 AI들이 숨바꼭질 게임을 반복하며 **더 나은 행동**을 찾아가는 예시입니다. "
            "처음부터 완벽하게 행동하는 것이 아니라, **결과를 보며 계속 고쳐 간다는 점**에 주목해 보세요."
        )
        st.video("https://www.youtube.com/watch?v=kopoLzvh5jY")

        video_q1 = st.text_input(
            "AI가 계속 고치려고 하는 것은 무엇일까?",
            key="d3_fu_video_q1",
            placeholder="힌트: 영상 속 AI가 다음 판에서 바꾸는 행동을 떠올려 보세요.",
        )
        video_q2 = st.text_input(
            "AI는 오차를 이용하여 학습합니다. 오차는 어떤 두 값을 비교할 때 생길까?",
            key="d3_fu_video_q2",
            placeholder="힌트: ____로 나온 결과와 미리 ___한 값을 비교합니다.",
        )
        if st.button("정답 보기", key="d3_fu_video_answer_btn", use_container_width=True):
            st.session_state["d3_fu_show_video_answer"] = True
        if st.session_state.get("d3_fu_show_video_answer", False):
            answer_col1, answer_col2 = st.columns(2)
            with answer_col1:
                st.markdown(pretty_title("변수의 뜻", "#f1f8e9", "#c8e6c9"), unsafe_allow_html=True)
                st.markdown(
                    """
                    - **실제값**: 게임이나 문제에서 실제로 나타난 결과입니다.
                    - **예측값**: AI가 미리 예상한 결과입니다.
                    - **오차**: 실제값과 예측값 사이의 차이입니다.
                    """
                )
                st.latex(r"\text{오차}=|\text{실제값}-\text{예측값}|")
            with answer_col2:
                st.markdown(pretty_title("AI 학습의 핵심", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
                st.markdown(
                    "인공지능은 처음부터 정답을 아는 것이 아니라, **실제값**과 **예측값**을 비교합니다. "
                    "그리고 **오차**, 즉 `|실제값 - 예측값|`이 작아지도록 자신의 전략이나 예측 방법을 계속 고쳐 가며 학습합니다."
                )

        if st.button("문제 1 결과 저장", key="d3_save_1", use_container_width=True):
            video_q1_text = video_q1.strip() if video_q1.strip() else "아직 작성하지 않았습니다."
            video_q2_text = video_q2.strip() if video_q2.strip() else "아직 작성하지 않았습니다."
            save_stage_result(
                "d3_answer_1",
                "[영상 관찰]\n"
                f"AI가 계속 고치려고 하는 것: {video_q1_text}\n"
                f"오차가 생기는 비교: {video_q2_text}\n\n"
                "[핵심 이해]\n"
                "인공지능은 실제값과 예측값을 비교하고, 그 차이인 오차가 작아지도록 학습한다.",
            )
        st.caption(saved_status_text("d3_answer_1"))
    with tabs[1]:
        stage_intro(
            "수학의 언어: 오차를 손실함수로 구조화하기",
            "예측선의 기울기를 바꾸어 보며 실제 데이터에 가장 잘 맞는 선을 찾아보는 단계입니다.",
            "어떤 기울기에서 손실이 가장 작아질까?",
            "#fff8e1",
            "#ffecb3",
        )

        st.markdown(pretty_title("1. 오차를 손실함수로 표현하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown(
                "AI가 만든 **예측**이 실제 데이터와 얼마나 가까운지 보려면, 여러 **오차**를 하나로 모아야 합니다. "
                "이 활동에서는 **오차를 제곱해서 모두 더한 값**을 **손실**이라고 부릅니다."
            )

        with col_m2:
            st.latex(r"\boxed{\text{오차}=\text{실제값}-\text{예측값}}")
            st.latex(r"\boxed{\text{손실함수(이차함수)}=\text{오차}^2+\cdots}")


        st.markdown(pretty_title("2. 데이터를 잘 설명하는 기울기 찾기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.markdown(
            "슬라이더로 **기울기 m**을 바꾸며, 데이터 점들과 가장 가까워지는 **예측선**을 찾아봅니다. "
            "**손실이 가장 작을 때의 m**이 데이터를 가장 잘 설명하는 기울기입니다."
        )
        slope = st.slider("기울기 m", 3.0, 10.5, 4.0, 0.1, key="d3_loss_slope")
        current_loss = sse(LOSS_Y, slope * LOSS_X + LOSS_INTERCEPT)
        best_m = float(np.sum(LOSS_X * (LOSS_Y - LOSS_INTERCEPT)) / np.sum(LOSS_X**2))

        col_m1, col_m2 = st.columns([1.2, 1])
        with col_m1:
            st.markdown(pretty_title("선택한 기울기(m)의 예측선", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.pyplot(make_prediction_line_figure(slope))
        with col_m2:
            st.markdown(pretty_title("기울기에 따른 손실함수", "#f3e5f5", "#e1bee7"), unsafe_allow_html=True)
            st.pyplot(make_loss_surface_figure(slope))

        render_value_cards(
            [
                {
                    "title": "현재 m",
                    "value": f"{slope:.1f}",
                    "detail": "슬라이더를 움직이면 예측선과 손실이 함께 바뀝니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "현재 손실",
                    "value": f"{current_loss:.2f}",
                    "detail": "오차제곱의 합입니다. 작을수록 예측선이 실제 데이터에 더 가깝습니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },

            ],
            columns=2,
        )

        st.markdown(pretty_title("3. 손실함수가 최소가 되는 기울기 m 찾기", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
        st.markdown(
            "**손실**은 예측선이 실제 데이터와 얼마나 멀리 떨어져 있는지를 나타내는 값입니다. "
            "따라서 **손실이 가장 작아지는 기울기 m**을 찾으면, **데이터에 가장 잘 맞는 예측선**을 찾을 수 있습니다."
        )
        group_m_answer = st.text_input(
            "문항 1: 손실함수가 최소가 되는 m의 값은 무엇인가?",
            key="d3_t_group_m",
            placeholder="예: 표와 그래프를 보고 m 값을 적어 보세요.",
        )
        if st.button("정답 확인", key="d3_t_group_answer_btn", use_container_width=True):
            st.session_state["d3_t_show_answer"] = True
        if st.session_state.get("d3_t_show_answer", False):
            st.success(
                f"정답 예시: 손실함수가 최소가 되는 기울기는 m ≈ {best_m:.1f}입니다. "
                "이때 손실함수 그래프에서 가장 낮은 점에 가장 가깝습니다."
            )

        if st.button("문제 2 결과 저장", key="d3_save_2", use_container_width=True):
            save_stage_result(
                "d3_answer_2",
                "[손실함수 구조화]\n"
                "FU에서 인식한 오차를 오차제곱의 합인 손실함수로 표현하였다.\n"
                "여러 기울기 m을 비교하며 손실함수가 어떻게 달라지는지 확인하였다.\n"
                "손실함수는 기울기 m에 대한 이차함수처럼 U자 모양이 되며, 가장 낮은 점이 최솟값임을 확인하였다.\n\n"
                f"문항 1 답: {group_m_answer.strip() if group_m_answer.strip() else '아직 작성하지 않았습니다.'}\n"
                "\n"
                "[핵심 정리]\n"
                f"손실함수가 최소가 되는 m은 약 {best_m:.1f}이며, 그 이유는 오차제곱의 합이 가장 작기 때문이다. "
                "이를 통해 인공지능은 손실함수라는 이차함수의 최솟값을 찾는 과정으로 학습한다고 이해하였다.",
            )
        st.caption(saved_status_text("d3_answer_2"))

    with tabs[2]:
        stage_intro(
            "AI 이해: 손실을 기준으로 모델 비교하기",
            "T 단계에서 배운 손실을 줄이는 기준으로 여러 모델을 비교합니다. "
            "직선 모델과 곡선 모델 중 어떤 것이 데이터에 더 적절한지 디지털 도구로 실험하는 단계입니다.",
            "직선과 곡선 모델 중 어떤 모델이 데이터를 더 잘 설명할까?",
            "#e8f5e9",
            "#c8e6c9",
        )

        st.markdown(pretty_title("1. 머신러닝이란?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.markdown(
            "**머신러닝**은 사람이 규칙을 하나하나 정해 주는 것이 아니라, 컴퓨터가 **데이터의 패턴**을 살펴보며 "
            "**예측값이 실제값에 가까워지는 식**을 찾아가는 방법입니다. "
            "즉, 여러 모델을 비교하면서 **오차가 더 작은 예측 방법**을 선택하는 과정입니다."
        )
        st.markdown(pretty_title("2. 머신러닝으로 데이터 예측하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.markdown(
            "**비교할 데이터셋**을 고르고 **직선 모델** 또는 **곡선 모델**을 선택해 봅니다. "
            "선택한 모델의 **식과 그래프**를 보며, 어떤 모델이 데이터 점들에 더 가깝게 맞는지 비교합니다."
        )

        model_left, model_right = st.columns([1, 1.25])
        with model_left:
            st.markdown(pretty_title("비교할 데이터셋 선택", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            if st.session_state.get("d3_compare_dataset") not in COMPARE_DATASETS:
                st.session_state["d3_compare_dataset"] = "기온과 아이스크림 판매량"
            dataset_name = st.selectbox(
                "비교할 데이터셋",
                list(COMPARE_DATASETS.keys()),
                key="d3_compare_dataset",
            )
            dataset = COMPARE_DATASETS[dataset_name]
            model_choice = st.selectbox(
                "머신러닝 모델 선택",
                ["직선(1차)", "곡선(2차)"],
                key="d3_u_model_choice",
                help="직선은 한 방향으로 늘거나 줄어드는 데이터에, 곡선은 휘어진 변화가 있는 데이터에 잘 맞는 경우가 많습니다.",
            )

        compare_fig, linear_loss, quad_loss, linear_coeffs, quad_coeffs = make_regression_compare_figure(
            dataset_name, True, True, model_choice
        )
        selected_coeffs = linear_coeffs if model_choice == "직선(1차)" else quad_coeffs
        recommended_model = "직선(1차)" if dataset["best"] == "직선" else "곡선(2차)"
        with model_left:
            st.markdown(f"**{model_choice} 모델 식**")
            st.latex(
                poly_to_latex(
                    selected_coeffs,
                    show_zero_terms=(model_choice == "곡선(2차)"),
                    force_degree=2 if model_choice == "곡선(2차)" else None,
                )
            )
        with model_right:
            st.markdown(pretty_title("그래프 개형 비교하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            st.pyplot(compare_fig)

        st.markdown(pretty_title("직선 모델과 곡선 모델 비교", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        render_value_cards(
            [
                {
                    "title": "직선 모델 손실",
                    "value": f"{linear_loss:.2f}",
                    "detail": "값이 작을수록 직선이 데이터에 더 잘 맞습니다.",
                    "bg": "#e3f2fd",
                    "border": "#64b5f6",
                },
                {
                    "title": "곡선 모델 손실",
                    "value": f"{quad_loss:.2f}",
                    "detail": "값이 작을수록 곡선이 데이터에 더 잘 맞습니다.",
                    "bg": "#ffebee",
                    "border": "#ef9a9a",
                },
            ],
            columns=2,
        )

        st.markdown(pretty_title("1️⃣ 모둠활동: 어떤 모델이 데이터에 더 잘 맞을까?", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
        group_model_answer = st.selectbox(
            f"문항 1: {dataset_name} 데이터는 어떤 모델이 더 잘 맞는가?",
            ["직선(1차)", "곡선(2차)"],
            key="d3_u_group_model_answer",
        )
        group_model_reason = st.text_area(
            "문항 2: 왜 그 모델이 더 적절한지 오차나 데이터의 모양을 바탕으로 설명하시오.",
            key="d3_u_group_model_reason",
            height=110,
            placeholder="오차, 손실, 데이터의 모양을 바탕으로 설명하시오.",
        )
        if st.button("모둠활동 정답 확인", key="d3_u_group_answer_btn", use_container_width=True):
            st.session_state["d3_u_show_answer"] = True
        if st.session_state.get("d3_u_show_answer", False):
            if recommended_model == "직선(1차)":
                st.success(
                    f"정답 예시: {recommended_model} 모델이 더 적절합니다. "
                    "데이터가 거의 한 방향으로 일정하게 변하고, 직선만으로도 오차를 충분히 작게 설명할 수 있기 때문입니다. "
                    "2차 모델이 손실을 조금 더 줄이더라도 불필요하게 휘어진 식이 될 수 있습니다."
                )
            else:
                st.success(
                    f"정답 예시: {recommended_model} 모델이 더 적절합니다. "
                    f"데이터가 휘어진 모양이며, 곡선 모델의 손실({quad_loss:.2f})이 직선 모델의 손실({linear_loss:.2f})보다 작기 때문입니다."
                )

        if st.button("문제 3 결과 저장", key="d3_save_3", use_container_width=True):
            save_stage_result(
                "d3_answer_3",
                "[모둠활동 1. 모델 비교와 해석]\n"
                f"선택한 데이터셋: {dataset_name}\n"
                "데이터의 모양을 보고 직선 모델과 곡선 모델을 비교하였다.\n"
                f"머신러닝이 찾은 직선 모델 손실: {linear_loss:.2f}\n"
                f"머신러닝이 찾은 곡선 모델 손실: {quad_loss:.2f}\n"
                f"더 적절하다고 선택한 모델: {group_model_answer}\n"
                f"선택 이유: {group_model_reason.strip()}\n\n"
                "[핵심 정리]\n"
                "직선 모델과 곡선 모델 중 더 적절한 모델을 선택하고, 손실 또는 오차의 크기를 근거로 이유를 설명하였다.",
            )
        st.caption(saved_status_text("d3_answer_3"))

    with tabs[3]:
        stage_intro(
            "세상과 연결: 인공지능 윤리 포스터 프롬프트 만들기",
            "앞 단계에서 배운 오차, 손실, 모델 비교를 바탕으로 AI 예측을 사회에서 사용할 때 필요한 "
            "윤리적 메시지를 정리하고, Canva 포스터 제작 프롬프트로 표현하는 단계입니다.",
            "AI를 어떻게 써야 사람과 사회에 도움이 될까?",
            "#fff3e0",
            "#ffe0b2",
        )

        class_num = class_key_from_ids(st.session_state.get("d3_id_1", ""))
        st.markdown(pretty_title("2️⃣모둠활동: AI 윤리 포스터 프롬프트 만들기", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        st.markdown(
            "인공지능 윤리는 AI를 만들고 사용할 때 **사람에게 해를 주지 않고, 공정하고 책임 있게 쓰이도록 고민하는 기준**입니다. "
            "AI가 **오차를 줄여 좋은 예측**을 하더라도, 그 예측이 누군가에게 불리하거나 **개인정보**를 함부로 쓰거나 "
            "**책임을 피하는 방식**으로 사용된다면 문제가 될 수 있습니다."
        )
        st.markdown(
            "이번 활동에서는 ‘AI가 잘 맞히는가?’를 넘어, 모든 사람이 보고 공감할 수 있는 "
            "**인공지능 윤리 포스터 메시지**를 만들어 봅니다."
        )

        st.markdown(pretty_title("1. 윤리 주제 선택하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        ethics_topic = st.radio(
            "포스터로 다룰 인공지능 윤리 주제",
            list(AI_ETHICS_TOPIC_GUIDES.keys()),
            horizontal=True,
            key="d3_ethics_topic",
        )
        st.info(AI_ETHICS_TOPIC_GUIDES[ethics_topic])
        topic_examples = AI_ETHICS_TOPIC_EXAMPLES.get(ethics_topic, AI_ETHICS_TOPIC_EXAMPLES["AI 공정성"])
        st.markdown(pretty_title("2. 질문과 메시지 정리하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        prompt_cols = st.columns(2)
        with prompt_cols[0]:
            st.text_area(
                "깊은 질문(D.E.E.P Question)",
                height=110,
                key="d3_ethics_question",
                placeholder=topic_examples["question"],
            )
        with prompt_cols[1]:
            st.text_area(
                "포스터로 전하고 싶은 메시지",
                height=110,
                key="d3_ethics_message",
                placeholder=topic_examples["message"],
            )
        symbol_cols = st.columns(2)
        with symbol_cols[0]:
            st.text_input(
                "이미지에 넣을 상징 1",
                key="d3_ethics_symbol_1",
                placeholder=topic_examples["symbol_1"],
            )
        with symbol_cols[1]:
            st.text_input(
                "이미지에 넣을 상징 2",
                key="d3_ethics_symbol_2",
                placeholder=topic_examples["symbol_2"],
            )
        st.selectbox(
            "포스터 분위기",
            ["깔끔한 공익광고", "강한 경고 메시지", "따뜻한 교육 캠페인", "미래적인 디지털 스타일"],
            key="d3_poster_style",
        )
        st.caption("선택한 분위기에 따라 Canva 프롬프트의 색감과 시각 방향이 자동으로 달라집니다.")

        st.markdown(pretty_title("3. Canva 포스터 프롬프트 만들기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        if st.button("Canva 포스터 프롬프트 만들기", key="d3_make_canva_prompt", use_container_width=True):
            st.session_state["d3_canva_prompt"] = build_canva_ethics_prompt()
            save_stage_result(
                "d3_answer_4",
                "[모둠활동 2. AI 윤리 포스터 프롬프트]\n"
                + "\n".join(f"{title}: {clean_text(value)}" for title, value in poster_prompt_entries()),
            )
        if st.session_state.get("d3_canva_prompt"):
            st.success("프롬프트가 생성되었습니다. Canva에서 포스터를 만들 때 아래 내용을 복사해 활용하세요.")
            st.code(st.session_state["d3_canva_prompt"], language="markdown")
            gallery_url = GALLERY_URLS.get(class_num)
            if gallery_url:
                link_cols = st.columns(2)
                with link_cols[0]:
                    st.markdown(
                        f"""<a href="{CANVA_AI_URL}" target="_blank"
                           style="display: block; padding: 11px; background: linear-gradient(90deg, #00c4cc 0%, #7d2ae8 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 8px;">
                           Canva AI 바로가기
                        </a>""",
                        unsafe_allow_html=True,
                    )
                with link_cols[1]:
                    st.markdown(
                        f"""<a href="{gallery_url}" target="_blank"
                           style="display: block; padding: 11px; background: linear-gradient(90deg, #7e57c2 0%, #42a5f5 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 8px;">
                           {class_num}반 갤러리 패들렛 바로가기
                        </a>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("학생 정보의 학번을 입력하면 우리 반 갤러리 패들렛 바로가기 버튼이 나타납니다.")
        else:
            st.info("주제, 질문, 메시지, 상징을 정리한 뒤 버튼을 누르면 Canva 포스터 제작 프롬프트가 생성됩니다.")

        st.caption(saved_status_text("d3_answer_4"))

        st.markdown("---")
        st.markdown(pretty_title("💾  학생 정보 입력 및 탐구 포트폴리오 저장", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.info("모둠활동 내용을 정리한 뒤 모둠 이름과 학생 정보를 입력하면, 활동 결과가 포함된 PDF 포트폴리오를 받을 수 있습니다.")
        group_name = st.text_input("모둠 이름 (예: 1모둠)", key="d3_group")
        info_cols = st.columns(2)
        with info_cols[0]:
            stu_id_1 = st.text_input("학번 (예: 10101)", max_chars=5, key="d3_id_1")
        with info_cols[1]:
            stu_name_1 = st.text_input("이름 (예: 홍길동)", key="d3_name_1")

        class_num = class_key_from_ids(stu_id_1)
        if group_name and stu_id_1 and stu_name_1:
            if class_num in PORT_URLS:
                student_info = {
                    "group": group_name,
                    "id_1": stu_id_1,
                    "name_1": stu_name_1,
                }
                principle_data = [
                    (
                        "문제 1. 오차란 무엇인가",
                        clean_text(st.session_state.get("d3_answer_1", ""))
                        + "\n\n[학생 질문]\n"
                        + clean_text(st.session_state.get("d3_answer_1_student_q", "")),
                    ),
                    (
                        "문제 2. 손실함수(이차함수)와 최솟값",
                        clean_text(st.session_state.get("d3_answer_2", ""))
                        + "\n\n[학생 질문]\n"
                        + clean_text(st.session_state.get("d3_answer_2_student_q", "")),
                    ),
                    (
                        "문제 3. 모둠활동 1. 데이터에 맞는 모델 고르기",
                        clean_text(st.session_state.get("d3_answer_3", ""))
                        + "\n\n[학생 질문]\n"
                        + clean_text(st.session_state.get("d3_answer_3_student_q", "")),
                    ),
                ]
                pdf_bytes = create_portfolio_pdf(
                    student_info,
                    principle_data,
                    build_mission_rows(),
                    poster_prompt_entries(),
                    build_figure_items(),
                )

                save_cols = st.columns(2)
                with save_cols[0]:
                    st.download_button(
                        label="탐구 결과 PDF 다운로드",
                        data=pdf_bytes,
                        file_name=f"{group_name}_{stu_name_1}_2-2_탐구포트폴리오.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.warning("⚠️ 모둠원들이 동시에 PDF 다운로드 버튼을 누르면 오류가 날 수 있습니다. 한 명씩 차례대로 눌러 주세요.")
                with save_cols[1]:
                    port_url = PORT_URLS.get(class_num)
                    st.markdown(
                        f"""<a href="{port_url}" target="_blank"
                           style="display: block; padding: 10px; background: linear-gradient(90deg, #43a047 0%, #66bb6a 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                           {class_num}반 포트폴리오 패들렛 열기
                        </a>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.error("학번의 세 번째 숫자가 해당 학급(1, 2, 5, 6반)인지 다시 확인해 주세요.")
        else:
            st.warning("모둠 이름과 학번, 이름을 입력하면 탐구 포트폴리오를 바로 받을 수 있습니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
