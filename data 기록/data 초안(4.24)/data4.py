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
    "생수의 양과 무게": {
        "x": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "y": np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1], dtype=float),
        "story": "생수의 양이 늘어날수록 전체 무게도 거의 일정한 비율로 커지는 상황입니다.",
        "x_label": "생수의 양",
        "x_unit": "L",
        "y_label": "전체 무게",
        "y_unit": "kg",
        "best": "직선",
        "quad_compare_coeffs": np.array([0.16, 0.10, 0.85], dtype=float),
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
}

BASE_YEAR = 2000.0
TEMP_YEARS = np.array(
    [
        2000, 2001, 2002, 2003, 2004,
        2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014,
        2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023, 2024,
    ],
    dtype=float,
)
TEMP_VALUES = np.array(
    [
        12.2, 12.4, 12.4, 12.2, 12.9,
        12.1, 12.6, 13.0, 12.7, 12.7,
        12.4, 12.1, 12.1, 12.6, 12.8,
        13.1, 13.4, 12.8, 12.8, 13.3,
        13.0, 13.3, 12.9, 13.7, 14.5,
    ],
    dtype=float,
)
FUN_X = TEMP_YEARS[:-1] - BASE_YEAR
FUN_Y = TEMP_VALUES[:-1]
FUN_X_DISPLAY = TEMP_YEARS[:-1]
FUTURE_YEAR = TEMP_YEARS[-1]
FUTURE_X = float(FUTURE_YEAR - BASE_YEAR)
FUTURE_TRUE_Y = float(TEMP_VALUES[-1])


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
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 4차시 탐구 포트폴리오", ln=1, align="C")
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


def create_portfolio_pdf(
    student_info,
    principle_data,
    mission_rows,
    social_meaning,
    action_plan,
    deep_question,
    figure_items,
):
    pdf = ThemedPDF()
    pdf.add_font("Nanum", "", font_path, uni=True)
    pdf.set_font("Nanum", "", 12)
    pdf._font_family = "Nanum"
    pdf.footer_left = f"{student_info.get('group', '')} - {student_info.get('name_1', '')}, {student_info.get('name_2', '')}"
    pdf.add_page()

    kvs = [
        ("모둠명", student_info.get("group", "")),
        ("학생 1 학번", student_info.get("id_1", "")),
        ("학생 1 이름", student_info.get("name_1", "")),
        ("학생 2 학번", student_info.get("id_2", "")),
        ("학생 2 이름", student_info.get("name_2", "")),
        ("작성일", datetime.datetime.now().strftime("%Y-%m-%d")),
    ]
    pdf.kv_card("학생 정보", kvs)

    intro_title, intro_answer = principle_data[0]
    pdf.h2("탐구 출발 질문")
    add_text_box_to_pdf(pdf, intro_title, intro_answer)

    pdf.h2("세상과 연결한 생각")
    pdf.p(f"모둠의 생각 1. 사회에 주는 시사점\n{clean_text(social_meaning)}", size=10)
    pdf.p(f"모둠의 생각 2. 우리의 실천 제안\n{clean_text(action_plan)}", size=10)
    pdf.p(f"모둠 심화 질문\n{clean_text(deep_question)}", size=10)

    linked_reflections = [
        principle_data[1][1] if len(principle_data) > 1 else "",
        principle_data[2][1] if len(principle_data) > 2 else "",
        principle_data[3][1] if len(principle_data) > 3 else "",
    ]

    for idx, ((mission_title, mission_text), (figure_title, figure)) in enumerate(zip(mission_rows, figure_items)):
        pdf.add_page()
        pdf.h2(mission_title)
        add_text_box_to_pdf(pdf, "실험 결과 요약", mission_text, fill_color=(250, 250, 250))
        reflection_text = linked_reflections[idx] if idx < len(linked_reflections) else ""
        if reflection_text.strip():
            add_text_box_to_pdf(pdf, "나의 해석", reflection_text, fill_color=(245, 245, 245))
        add_figure_to_pdf(pdf, figure_title, figure)

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
    for x_val, y_real, y_hat in zip(LOSS_X, LOSS_Y, y_pred):
        ax.vlines(x_val, y_real, y_hat, color="#90a4ae", linestyle="--", alpha=0.9)
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
    ax.scatter([slope], [current_loss], color="#d32f2f", s=80, zorder=3, label="현재 위치")
    ax.set_xlabel("기울기")
    ax.set_ylabel("오차를 모아 본 값")
    ax.set_title("오차를 모아 본 그래프")
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
    ax.set_ylabel("오차를 모아 본 값")
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
    ax2.set_ylabel("오차를 모아 본 값")
    ax2.set_title("오차를 모아 본 그래프")
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


def make_regression_compare_figure(dataset_name, show_linear=True, show_quad=True):
    dataset = COMPARE_DATASETS[dataset_name]
    x_values, y_values, coeffs_linear, coeffs_quad, linear_loss, quad_loss = get_regression_compare_metrics(dataset_name)
    fig = Figure(figsize=(5.8, 3.5))
    ax = fig.subplots()
    x_smooth = np.linspace(x_values.min(), x_values.max(), 220)
    ax.scatter(x_values, y_values, color="#263238", s=65, label="실제 데이터", zorder=3)
    if show_linear:
        ax.plot(
            x_smooth,
            np.polyval(coeffs_linear, x_smooth),
            color="#1e88e5",
            linewidth=2,
            linestyle="--",
            label="직선 모델",
        )
    if show_quad:
        ax.plot(
            x_smooth,
            np.polyval(coeffs_quad, x_smooth),
            color="#ef6c00",
            linewidth=2,
            label="곡선 모델",
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


def manual_feedback(loss_value):
    if loss_value <= 12:
        return "success", "미션 성공! 연도별 평균기온 데이터에 모델을 아주 잘 맞췄습니다."
    if loss_value <= 28:
        return "info", "거의 맞췄어요. 계수를 조금만 더 손보면 됩니다."
    return "warning", "아직 오차가 큽니다. 기울기나 곡선의 모양을 다시 조절해 보세요."


def get_manual_vs_ml_metrics(model_type, coef1, coef2, coef3, degree):
    if model_type == "직선":
        manual_coeffs = np.array([coef1, coef2], dtype=float)
    else:
        manual_coeffs = np.array([coef1, coef2, coef3], dtype=float)
    ml_coeffs = fit_degree(FUN_X, FUN_Y, degree)
    manual_pred = np.polyval(manual_coeffs, FUN_X)
    ml_pred = np.polyval(ml_coeffs, FUN_X)
    ml_future_pred = float(np.polyval(ml_coeffs, FUTURE_X))
    return manual_coeffs, ml_coeffs, sse(FUN_Y, manual_pred), sse(FUN_Y, ml_pred), ml_future_pred


def make_manual_vs_ml_figure(model_type, coef1, coef2, coef3, degree, show_prediction=False, user_guess=None):
    manual_coeffs, ml_coeffs, manual_loss, ml_loss, ml_future_pred = get_manual_vs_ml_metrics(
        model_type, coef1, coef2, coef3, degree
    )
    fig = Figure(figsize=(5.8, 3.5))
    ax = fig.subplots()
    x_smooth = np.linspace(FUN_X.min(), FUTURE_X, 220)
    x_display_smooth = BASE_YEAR + x_smooth
    ax.scatter(FUN_X_DISPLAY, FUN_Y, color="#263238", s=65, label="실제 평균기온 데이터", zorder=3)
    ax.plot(x_display_smooth, np.polyval(manual_coeffs, x_smooth), color="#8e24aa", linewidth=2, label="내가 맞춘 모델")
    ax.plot(
        x_display_smooth,
        np.polyval(ml_coeffs, x_smooth),
        color="#2e7d32",
        linewidth=2,
        linestyle="--",
        label=f"머신러닝 {degree}차 모델",
    )
    ax.axvline(FUTURE_YEAR, color="#90a4ae", linestyle=":", linewidth=1.3)
    if show_prediction:
        ax.scatter([FUTURE_YEAR], [FUTURE_TRUE_Y], color="#263238", s=80, marker="D", zorder=6, label="실제값")
        if user_guess is not None:
            user_x = FUTURE_YEAR - 0.18
            user_error = abs(float(user_guess) - FUTURE_TRUE_Y)
            ax.scatter([user_x], [user_guess], color="#1565c0", s=85, zorder=4, label="내 예측값")
            ax.vlines(
                user_x,
                min(float(user_guess), FUTURE_TRUE_Y),
                max(float(user_guess), FUTURE_TRUE_Y),
                colors="#1565c0",
                linewidth=2.2,
                alpha=0.9,
            )
            ax.text(
                user_x - 0.04,
                (float(user_guess) + FUTURE_TRUE_Y) / 2,
                f"오차 {user_error:.1f}",
                color="#1565c0",
                fontsize=9,
                ha="right",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bbdefb", alpha=0.9),
            )
        ai_x = FUTURE_YEAR + 0.18
        ai_error = abs(ml_future_pred - FUTURE_TRUE_Y)
        ax.scatter([ai_x], [ml_future_pred], color="#d32f2f", s=95, marker="*", zorder=5, label="AI 예측값")
        ax.vlines(
            ai_x,
            min(ml_future_pred, FUTURE_TRUE_Y),
            max(ml_future_pred, FUTURE_TRUE_Y),
            colors="#d32f2f",
            linewidth=2.2,
            alpha=0.9,
        )
        ax.text(
            ai_x + 0.04,
            (ml_future_pred + FUTURE_TRUE_Y) / 2,
            f"오차 {ai_error:.1f}",
            color="#d32f2f",
            fontsize=9,
            ha="left",
            va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ffcdd2", alpha=0.9),
            )
    ax.set_xlim(FUN_X_DISPLAY.min() - 0.7, FUTURE_YEAR + 0.7)
    ax.set_xlabel("연도")
    ax.set_ylabel("평균기온(℃)")
    ax.set_title("연도별 평균기온: 직접 맞추기 vs 머신러닝")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    return fig, manual_loss, ml_loss, manual_coeffs, ml_coeffs, ml_future_pred


def build_mission_rows():
    slope = float(st.session_state.get("d3_loss_slope", 7.0))
    current_loss = sse(LOSS_Y, slope * LOSS_X + LOSS_INTERCEPT)

    dataset_name = st.session_state.get("d3_compare_dataset", "생수의 양과 무게")
    _, _, linear_coeffs, quad_coeffs, linear_loss, quad_loss = get_regression_compare_metrics(dataset_name)

    manual_model_type = st.session_state.get("d3_manual_model_type", "곡선")
    manual_a = float(st.session_state.get("d3_manual_a", 0.03))
    manual_b = float(st.session_state.get("d3_manual_b", 12.2))
    manual_c = float(st.session_state.get("d3_manual_c", 12.2))
    degree = int(st.session_state.get("d3_ml_degree", 2))
    manual_coeffs, ml_coeffs, manual_loss, ml_loss, future_pred = get_manual_vs_ml_metrics(
        manual_model_type, manual_a, manual_b, manual_c, degree
    )
    future_guess = float(st.session_state.get("d3_future_guess", 13.5))
    user_future_error = abs(future_guess - FUTURE_TRUE_Y)
    ai_future_error = abs(future_pred - FUTURE_TRUE_Y)

    return [
        (
            "문제 2. 오차가 가장 작은 예측선 찾기",
            (
                f"선택한 기울기: {slope:.2f}\n"
                f"절편은 {LOSS_INTERCEPT:.0f}으로 고정\n"
                f"현재 오차를 모아 본 값: {current_loss:.2f}"
            ),
        ),
        (
            "문제 3. 직선과 곡선 모델 비교",
            (
                f"선택한 데이터셋: {dataset_name}\n"
                f"직선 모델의 오차: {linear_loss:.2f}\n"
                f"곡선 모델의 오차: {quad_loss:.2f}\n"
                f"직선 모델 식: {poly_to_text(linear_coeffs)}\n"
                f"곡선 모델 식: {poly_to_text(quad_coeffs, show_zero_terms=True, force_degree=2)}"
            ),
        ),
        (
            "문제 4. 직접 맞추기와 머신러닝 모델 선택",
            (
                f"내가 선택한 모델: {manual_model_type}\n"
                f"직접 조절한 식: {poly_to_text(manual_coeffs)}\n"
                f"내 식의 오차: {manual_loss:.2f}\n"
                f"선택한 머신러닝 모델: {degree}차\n"
                f"머신러닝 식의 오차: {ml_loss:.2f}\n"
                f"{FUTURE_YEAR:.0f}년 실제 평균기온: {FUTURE_TRUE_Y:.2f}\n"
                f"{FUTURE_YEAR:.0f}년 내 예측: {future_guess:.2f}\n"
                f"{FUTURE_YEAR:.0f}년 AI 예측: {future_pred:.2f}\n"
                f"내 오차 거리: {user_future_error:.2f}\n"
                f"AI 오차 거리: {ai_future_error:.2f}"
            ),
        ),
    ]


def build_figure_items():
    slope = float(st.session_state.get("d3_loss_slope", 7.0))
    dataset_name = st.session_state.get("d3_compare_dataset", "생수의 양과 무게")
    manual_model_type = st.session_state.get("d3_manual_model_type", "곡선")
    manual_a = float(st.session_state.get("d3_manual_a", 0.03))
    manual_b = float(st.session_state.get("d3_manual_b", 12.2))
    manual_c = float(st.session_state.get("d3_manual_c", 12.2))
    degree = int(st.session_state.get("d3_ml_degree", 2))

    compare_fig = None
    if dataset_name != "생수의 양과 무게":
        compare_fig, _, _, _, _ = make_regression_compare_figure(dataset_name)
    manual_fig, _, _, _, _, _ = make_manual_vs_ml_figure(
        manual_model_type,
        manual_a,
        manual_b,
        manual_c,
        degree,
        show_prediction=st.session_state.get("d3_show_prediction", False),
        user_guess=float(st.session_state.get("d3_future_guess", 13.5)),
    )

    return [
        ("오차를 모아 본 그래프와 예측선", make_loss_bundle_figure(slope)),
        ("직선과 곡선 모델 비교", compare_fig),
        ("직접 맞춘 식과 머신러닝 비교", manual_fig),
    ]


def build_pdf_signature(student_info, social_meaning, action_plan, deep_question):
    snapshot = {
        "student_info": student_info,
        "social_meaning": social_meaning,
        "action_plan": action_plan,
        "deep_question": deep_question,
        "answers": {
            "a1": st.session_state.get("d3_answer_1", ""),
            "a2": st.session_state.get("d3_answer_2", ""),
            "a3": st.session_state.get("d3_answer_3", ""),
            "a4": st.session_state.get("d3_answer_4", ""),
        },
        "controls": {
            "loss_slope": st.session_state.get("d3_loss_slope", 7.0),
            "dataset": st.session_state.get("d3_compare_dataset", "생수의 양과 무게"),
            "manual_model_type": st.session_state.get("d3_manual_model_type", "곡선"),
            "manual_a": st.session_state.get("d3_manual_a", 0.03),
            "manual_b": st.session_state.get("d3_manual_b", 12.2),
            "manual_c": st.session_state.get("d3_manual_c", 12.2),
            "ml_degree": st.session_state.get("d3_ml_degree", 2),
            "future_guess": st.session_state.get("d3_future_guess", 13.5),
            "show_prediction": st.session_state.get("d3_show_prediction", False),
        },
    }
    return repr(snapshot)
    return repr(snapshot)

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
            "3️⃣ [U] AI 활용",
            "4️⃣ [R] 결과 해석",
            "5️⃣ [E] 세상과 연결",
        ]
    )

    with tabs[0]:
        stage_intro(
            "문제 인식 및 숨겨진 데이터 찾기",
            "실생활 그래프에서 실제값과 AI 예측값을 비교하며, 컴퓨터가 오차를 줄이는 방향으로 "
            "어떻게 생각을 고쳐 가는지 가볍게 체험하는 과정입니다.",
            "컴퓨터는 시간에 따라 변하는 데이터를 어떤 규칙으로 이해하고, 예측의 오차를 어떻게 줄여 갈까?",
            "#e3f2fd",
            "#bbdefb",
        )

        intro_fig, months, actual_values, ai_predictions = make_intro_error_figure()
        error_values = np.abs(ai_predictions - actual_values)
        max_error_idx = int(np.argmax(error_values))
        st.markdown(pretty_title("문제 제기: 그래프를 보고 AI가 예측을 고쳐 보자", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write(
                "냉난방 준비, 옷차림, 에너지 사용량처럼 우리 생활에는 `앞으로 얼마나 더워질지`를 미리 짐작해야 하는 일이 많습니다. "
                "아래 그래프는 홀수 월 평균기온의 실제값과 AI가 만든 예측값을 함께 보여 줍니다. "
                "이제 우리는 빨간 선으로 표시된 오차를 보고, AI가 다음에는 어느 방향으로 예측을 바꾸어야 할지 생각해 볼 수 있습니다."
            )
        col1, col2 = st.columns([1, 1])
        with col1:
            render_value_cards(
                [
                    {
                        "title": "오차가 가장 큰 달",
                        "value": f"{months[max_error_idx]}월",
                        "detail": "이 달은 실제값과 AI 예측값의 차이가 가장 크게 나타납니다.",
                        "bg": "#fff8e1",
                        "border": "#ffcc80",
                    },
                    {
                        "title": "가장 큰 오차 거리",
                        "value": f"{error_values[max_error_idx]:.1f}",
                        "detail": "빨간 세로선이 길수록 실제값과 더 멀리 떨어져 있습니다.",
                        "bg": "#ffebee",
                        "border": "#ef9a9a",
                    },
                ]
            )
            st.info(
                "오차는 `실제값과 예측값 사이의 거리`입니다. "
                "그래프에서는 파란 점(실제값)과 주황 별(AI 예측값)을 잇는 빨간 세로선 길이로 볼 수 있습니다."
            )
            st.caption("이제 아래 문제에서 `AI는 오차를 보고 다음 예측을 어떻게 고쳐 가는가`를 한 문장으로 직접 정리해 보세요.")
        with col2:
            st.markdown(pretty_title("실제값과 AI 예측값 그래프", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            st.pyplot(intro_fig)

        st.markdown(pretty_title("문제 1. 그래프에서 오차를 읽고 예측 고치기", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
        sentence_cols = st.columns([4.8, 1.8, 1.2])
        with sentence_cols[0]:
            st.markdown("**AI의 학습은 정답을 한 번에 맞히는 것이 아니라, 오차를**")
        with sentence_cols[1]:
            answer_blank = st.text_input(
                "빈칸",
                key="d3_answer_1_blank",
                label_visibility="collapsed",
            )
        with sentence_cols[2]:
            st.markdown("**과정이다.**")
        if answer_blank.strip():
            st.session_state["d3_answer_1"] = (
                f"AI의 학습은 정답을 한 번에 맞히는 것이 아니라, 오차를 {answer_blank.strip()} 과정이다."
            )
            st.success("정답 확인: AI의 학습은 정답을 한 번에 맞히는 것이 아니라, 오차를 줄여 가는 과정이다.")
        else:
            st.session_state["d3_answer_1"] = ""
            st.caption("문장 속 빈칸에 들어갈 말을 직접 적어 보세요.")
    with tabs[1]:
        stage_intro(
            "현상을 수학의 언어로 바꾸기",
            "예측이 얼마나 잘 맞는지를 한눈에 보기 위해 오차를 모두 모아 본 값(손실함수)을 만들고, "
            "그 값이 가장 작은 지점을 찾는 과정입니다.",
            "왜 가장 좋은 예측은 오차를 모아 본 값이 가장 작은 곳에 있을까?",
            "#fff8e1",
            "#ffecb3",
        )

        st.markdown(pretty_title("오차가 가장 작은 예측선을 찾아라", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write(
            "이번에는 절편을 고정한 채 기울기만 바꾸며 오차를 줄여 보세요. "
            "기울기를 잘 맞추면 오른쪽 그래프의 빨간 점이 가장 아래쪽에 가까워집니다. "
            "이 그래프(손실함수)는 '오차를 모두 모아 본 값'을 나타내며, 가장 낮은 지점이 가장 잘 맞는 예측선입니다."
        )
        st.info("손실함수: 오차를 하나로 모아 본 값이라고 생각하면됩니다.")
        st.info("경사하강법은 오차를 모아 본 값이 더 작아지는 방향으로 조금씩 내려가며 더 좋은 예측선을 찾는 생각 방식입니다.")

        slope = st.slider("예측선의 기울기", 3.0, 10.5, 7.0, 0.1, key="d3_loss_slope")
        current_loss = sse(LOSS_Y, slope * LOSS_X + LOSS_INTERCEPT)

        col_m1, col_m2 = st.columns([1.2, 1])
        with col_m1:
            st.markdown(pretty_title("예측선 조절 그래프", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            st.pyplot(make_prediction_line_figure(slope))
        with col_m2:
            st.markdown(pretty_title("손실함수 그래프", "#f3e5f5", "#e1bee7"), unsafe_allow_html=True)
            st.pyplot(make_loss_surface_figure(slope))

        render_value_cards(
            [
                {
                    "title": "현재 기울기",
                    "value": f"{slope:.1f}",
                    "detail": "슬라이더를 움직이면 예측선의 기울기가 함께 바뀝니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "오차를 모아 본 값",
                    "value": f"{current_loss:.2f}",
                    "detail": "이 값이 작을수록 현재 예측선이 실제 데이터에 더 잘 맞습니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
            ],
            columns=2,
        )
        st.markdown(pretty_title("경사하강법 미니 시뮬레이션", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        st.caption("손실이 더 작은 쪽을 살펴보고, 현재 위치에서 한 걸음만 이동해 보는 작은 시뮬레이션입니다.")
        if "d3_show_descent_step" not in st.session_state:
            st.session_state["d3_show_descent_step"] = False
        if st.button("한 걸음 이동해 보기", key="d3_loss_step_btn", use_container_width=True):
            st.session_state["d3_show_descent_step"] = True
        if st.session_state.get("d3_show_descent_step", False):
            step_preview = get_descent_step_preview(slope)
            step_cols = st.columns([1.05, 1.2])
            with step_cols[0]:
                render_value_cards(
                    [
                        {
                            "title": "현재 위치",
                            "value": f"기울기 {step_preview['current_slope']:.1f}",
                            "detail": f"현재 손실은 {step_preview['current_loss']:.2f}입니다.",
                            "bg": "#f4f9ff",
                            "border": "#90caf9",
                        },
                        {
                            "title": "손실이 줄어드는 방향",
                            "value": step_preview["direction"],
                            "detail": "양옆을 비교해 더 낮아지는 쪽으로 조금 이동합니다.",
                            "bg": "#fff8e1",
                            "border": "#ffcc80",
                        },
                        {
                            "title": "한 걸음 이동한 뒤",
                            "value": f"기울기 {step_preview['next_slope']:.1f}",
                            "detail": f"손실이 {step_preview['next_loss']:.2f}로 바뀝니다.",
                            "bg": "#f1f8e9",
                            "border": "#aed581",
                        },
                        {
                            "title": "손실 변화량",
                            "value": f"{step_preview['loss_change']:.2f}",
                            "detail": "양수이면 그만큼 손실이 줄었다는 뜻입니다.",
                            "bg": "#ede7f6",
                            "border": "#b39ddb",
                        },
                    ],
                    columns=2,
                )
            with step_cols[1]:
                st.pyplot(make_loss_step_figure(step_preview["current_slope"], step_preview["next_slope"]))
        st.caption(
            "관찰 포인트: 오차를 제곱해서 더하면 전체 모양이 U자 형태가 되고, "
            "가장 낮은 지점이 가장 잘 맞는 예측선이 됩니다."
        )

        message_type, message_text = loss_feedback(current_loss)
        getattr(st, message_type)(f"현재 오차를 모아 본 값: {current_loss:.2f} {message_text}")

        principle_box(
            "2",
            "오차를 모아 본 그래프와 최솟값",
            "오차를 모아 본 그래프가 U자 모양일 때, 가장 좋은 예측선은 수학적으로 어떤 지점에 있다고 설명할 수 있을까요?",
            "d3_answer_2",
            "오차를 모아 본 그래프는 이차함수처럼 가장 낮은 지점을 가지며, 그 지점이 바로 오차가 가장 작은 최솟값입니다. 즉, AI가 가장 잘 예측하는 상태는 이 그래프의 꼭짓점 근처입니다.",
        )

    with tabs[2]:
        stage_intro(
            "AI 도구로 시뮬레이션하기",
            "여러 데이터의 모양을 직접 비교하며, AI가 어떤 상황에서 직선 모델을 쓰고 "
            "어떤 상황에서 곡선 모델을 써야 하는지 실험하는 단계입니다.",
            "어떤 데이터에는 직선이, 어떤 데이터에는 곡선이 더 잘 맞을까?",
            "#e8f5e9",
            "#c8e6c9",
        )

        st.markdown(pretty_title("직선 모델이 좋을까, 곡선 모델이 좋을까?", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write(
            "모든 데이터가 직선으로 설명되는 것은 아닙니다. 먼저 데이터의 모양을 보고, "
            "직선 모델과 곡선 모델 중 어떤 것이 더 자연스럽게 맞는지 비교해 봅시다."
        )
        st.info("AI는 오차를 줄이는 식을 찾을 뿐 아니라, 데이터 모양에 더 잘 맞는 모델 자체를 선택해야 합니다.")
        st.info("선형회귀는 직선 모델, 다항회귀(2차)는 곡선 모델이라고 생각하면 훨씬 쉽습니다.")
        if st.session_state.get("d3_compare_dataset") not in COMPARE_DATASETS:
            st.session_state["d3_compare_dataset"] = "생수의 양과 무게"
        dataset_name = st.selectbox(
            "비교할 데이터셋을 선택하세요.",
            list(COMPARE_DATASETS.keys()),
            key="d3_compare_dataset",
        )
        dataset = COMPARE_DATASETS[dataset_name]
        st.info(dataset["story"])

        toggle_col1, toggle_col2, _ = st.columns([1, 1, 2.4])
        with toggle_col1:
            show_linear = st.checkbox("직선 모델 보기", value=True, key="d3_show_linear")
        with toggle_col2:
            show_quad = st.checkbox("곡선 모델 보기", value=True, key="d3_show_quad")

        compare_fig, linear_loss, quad_loss, linear_coeffs, quad_coeffs = make_regression_compare_figure(
            dataset_name, show_linear, show_quad
        )
        col_c1, col_c2 = st.columns([1, 1.25])
        with col_c1:
            st.markdown(pretty_title("모델 함수식 읽기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            st.markdown("**직선 모델 식**")
            st.latex(poly_to_latex(linear_coeffs))
            st.markdown("**곡선 모델 식**")
            st.latex(poly_to_latex(quad_coeffs, show_zero_terms=True, force_degree=2))
        with col_c2:
            st.markdown(pretty_title("그래프 비교하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            st.pyplot(compare_fig)

        st.markdown(pretty_title("모델 비교 결과", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        render_value_cards(
            [
                {
                    "title": "직선 모델의 오차",
                    "value": f"{linear_loss:.2f}",
                    "detail": "값이 작을수록 직선이 데이터에 더 잘 맞습니다.",
                    "bg": "#e3f2fd",
                    "border": "#64b5f6",
                },
                {
                    "title": "곡선 모델의 오차",
                    "value": f"{quad_loss:.2f}",
                    "detail": "값이 작을수록 곡선이 데이터에 더 잘 맞습니다.",
                    "bg": "#ffebee",
                    "border": "#ef9a9a",
                },
                {
                    "title": "더 자연스러운 모델",
                    "value": dataset["best"],
                    "detail": "데이터의 전체 모양을 보고 어떤 모델이 더 어울리는지 판단합니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
            ],
            columns=3,
        )
        if dataset["best"] == "직선":
            st.success("이 데이터는 전체적으로 일정하게 증가하므로 직선 모델이 더 자연스럽습니다.")
        else:
            st.success("이 데이터는 증가하다가 다시 감소하므로 곡선 모델이 더 자연스럽습니다.")

        principle_box(
            "3",
            "데이터에 맞는 모델 고르기",
            "어떤 데이터에는 직선이, 어떤 데이터에는 곡선이 더 잘 맞는 이유는 무엇일까요?",
            "d3_answer_3",
            "데이터가 일정한 비율로 늘어나면 직선 모델이 잘 맞고, 늘어나다가 다시 줄어드는 변화가 있으면 곡선 모델이 더 잘 맞습니다. 즉, 데이터의 모양을 먼저 읽고 그에 맞는 모델을 선택해야 오차를 더 잘 줄일 수 있습니다.",
        )

    with tabs[3]:
        stage_intro(
            "결과의 의미와 한계 고민하기",
            "내가 직접 맞춘 식과 컴퓨터가 자동으로 찾은 식을 비교하며, 어떤 예측이 더 잘 맞는지 해석하는 과정입니다.",
            "내가 맞춘 식과 머신러닝이 찾은 식은 무엇이 다를까?",
            "#f3e5f5",
            "#e1bee7",
        )

        st.markdown(pretty_title("기본 탐구: 직접 맞추기 vs 머신러닝 비교", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write(
            "아래에는 `2000년~2023년 평균기온` 자료가 들어 있습니다. 이 자료에 맞는 모델을 직접 조절해 보고, "
            "머신러닝이 자동으로 찾은 모델과 비교해 보세요. 직선은 숫자 2개, 곡선은 숫자 3개를 조절합니다."
        )
        st.caption("계산을 쉽게 하기 위해 x값은 `2000년 이후 몇 년이 지났는지`로 바꾸어 생각합니다. 예를 들어 2005년은 x=5, 2023년은 x=23입니다.")

        if st.session_state.get("d3_manual_model_type") not in ["직선", "곡선"]:
            st.session_state["d3_manual_model_type"] = "곡선"
        if st.session_state.get("d3_ml_degree") not in [1, 2]:
            st.session_state["d3_ml_degree"] = 2
        if "d3_future_guess" not in st.session_state:
            st.session_state["d3_future_guess"] = 13.5
        if "d3_show_prediction" not in st.session_state:
            st.session_state["d3_show_prediction"] = False

        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown(pretty_title("내가 직접 식 맞추기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.info("슬라이더를 움직이며 내가 생각한 식이 데이터에 얼마나 잘 맞는지 먼저 실험해 보세요.")
            manual_model_type = st.radio(
                "내가 직접 맞출 모델",
                ["직선", "곡선"],
                horizontal=True,
                key="d3_manual_model_type",
            )
            if manual_model_type == "직선":
                if not (-0.10 <= float(st.session_state.get("d3_manual_a", 0.03)) <= 0.15):
                    st.session_state["d3_manual_a"] = 0.03
                if not (11.0 <= float(st.session_state.get("d3_manual_b", 12.2)) <= 13.8):
                    st.session_state["d3_manual_b"] = 12.2
                manual_a = st.slider("기울기 m", -0.10, 0.15, 0.03, 0.01, key="d3_manual_a")
                manual_b = st.slider("절편 n", 11.0, 13.8, 12.2, 0.1, key="d3_manual_b")
                manual_c = 0.0
                st.latex(fr"y = {manual_a:.2f}x + {manual_b:.2f}")
            else:
                if not (-0.02 <= float(st.session_state.get("d3_manual_a", 0.0)) <= 0.02):
                    st.session_state["d3_manual_a"] = 0.0
                if not (-0.10 <= float(st.session_state.get("d3_manual_b", 0.03)) <= 0.20):
                    st.session_state["d3_manual_b"] = 0.03
                if not (11.0 <= float(st.session_state.get("d3_manual_c", 12.2)) <= 13.8):
                    st.session_state["d3_manual_c"] = 12.2
                manual_a = st.slider("a", -0.02, 0.02, 0.00, 0.001, key="d3_manual_a")
                manual_b = st.slider("b", -0.10, 0.20, 0.03, 0.01, key="d3_manual_b")
                manual_c = st.slider("c", 11.0, 13.8, 12.2, 0.1, key="d3_manual_c")
                st.latex(poly_to_latex(np.array([manual_a, manual_b, manual_c])))
        with col_p2:
            st.markdown(pretty_title("머신러닝에게 맡기기", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
            st.info("머신러닝은 선택한 차수 안에서 오차가 가장 작아지는 식을 자동으로 찾습니다.")
            degree = st.radio(
                "머신러닝 모델 선택",
                [1, 2],
                horizontal=True,
                key="d3_ml_degree",
            )
            st.write("1은 직선, 2는 곡선을 뜻합니다. 차수를 바꾸며 머신러닝이 어떤 모양의 식을 고르는지 비교해 보세요.")

        manual_fig, manual_loss, ml_loss, _, ml_coeffs, ml_future_pred = make_manual_vs_ml_figure(
            manual_model_type,
            manual_a,
            manual_b,
            manual_c,
            degree,
            show_prediction=False,
        )

        better_model = "내가 맞춘 식" if manual_loss <= ml_loss else "머신러닝이 찾은 식"
        st.markdown(pretty_title("기본 탐구 결과 해석", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([1.25, 1])
        with col_f1:
            st.markdown(pretty_title("그래프로 비교하기", "#f3e5f5", "#e1bee7"), unsafe_allow_html=True)
            st.pyplot(manual_fig)
        with col_f2:
            st.markdown(pretty_title("머신러닝이 찾은 식", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            if degree == 2:
                st.latex(poly_to_latex(ml_coeffs, show_zero_terms=True, force_degree=2))
            else:
                st.latex(poly_to_latex(ml_coeffs))
            render_value_cards(
                [
                    {
                        "title": "이번 비교에서 더 가까운 쪽",
                        "value": better_model,
                        "detail": "오차가 더 작은 모델이 현재 데이터에 더 잘 맞습니다.",
                        "bg": "#fff8e1",
                        "border": "#ffcc80",
                    },
                ]
            )
            feedback_type, feedback_text = manual_feedback(manual_loss)
            getattr(st, feedback_type)(feedback_text)

        st.markdown(pretty_title("핵심 결과 한눈에 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        render_value_cards(
            [
                {
                    "title": "내가 맞춘 식의 오차",
                    "value": f"{manual_loss:.2f}",
                    "detail": "값이 작을수록 실제 데이터에 더 가깝습니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "머신러닝 식의 오차",
                    "value": f"{ml_loss:.2f}",
                    "detail": f"현재 선택한 {degree}차 모델이 만든 오차입니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "예측할 연도",
                    "value": f"{FUTURE_YEAR:.0f}",
                    "detail": "이 연도 값은 아래 심화 도전에서 직접 예측합니다.",
                    "bg": "#ede7f6",
                    "border": "#b39ddb",
                },
            ],
            columns=3,
        )

        st.markdown(pretty_title("심화 도전: 미래 연도 예측 비교하기", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
        st.info("기본 탐구에서 만든 식을 바탕으로 미래 연도 값을 직접 예측하고, AI가 찾은 값과 실제값을 비교해 봅시다.")
        challenge_cols = st.columns([1, 1.2])
        with challenge_cols[0]:
            st.markdown(pretty_title("예측 입력하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            st.write(f"{FUTURE_YEAR:.0f}년의 평균기온을 먼저 예측해 보세요.")
            my_guess = st.number_input(
                f"나의 예측값 y ({FUTURE_YEAR:.0f}년 평균기온)",
                value=float(st.session_state.get("d3_future_guess", 13.5)),
                step=0.1,
                key="d3_future_guess",
            )
            if st.button("예측 확인", key="d3_predict_btn"):
                st.session_state["d3_show_prediction"] = True
            if st.session_state.get("d3_show_prediction", False):
                st.info(
                    f"그래프의 {FUTURE_YEAR:.0f}년 위치에서 검은 마름모는 실제값({FUTURE_TRUE_Y:.1f}℃)입니다. "
                    "파란 점과 빨간 별에서 실제값까지 이어진 세로선 길이가 각각 오차 거리입니다."
                )
        with challenge_cols[1]:
            st.markdown(pretty_title("예측 그래프 보기", "#f3e5f5", "#e1bee7"), unsafe_allow_html=True)
            if st.session_state.get("d3_show_prediction", False):
                prediction_fig, _, _, _, _, ml_future_pred = make_manual_vs_ml_figure(
                    manual_model_type,
                    manual_a,
                    manual_b,
                    manual_c,
                    degree,
                    show_prediction=True,
                    user_guess=float(st.session_state.get("d3_future_guess", 13.5)),
                )
                st.pyplot(prediction_fig)
            else:
                st.info("왼쪽에서 예측값을 적고 `예측 확인`을 누르면 미래 연도 비교 그래프가 나타납니다.")

        if st.session_state.get("d3_show_prediction", False):
            my_error = abs(float(st.session_state.get("d3_future_guess", 13.5)) - FUTURE_TRUE_Y)
            ai_error = abs(ml_future_pred - FUTURE_TRUE_Y)
            st.markdown(pretty_title("예측 오차 거리 비교", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
            render_value_cards(
                [
                    {
                        "title": "내 오차 거리",
                        "value": f"{my_error:.2f}",
                        "detail": f"{FUTURE_YEAR:.0f}년 실제값과 내 예측값 사이의 거리입니다.",
                        "bg": "#e3f2fd",
                        "border": "#64b5f6",
                    },
                    {
                        "title": "AI 오차 거리",
                        "value": f"{ai_error:.2f}",
                        "detail": f"{FUTURE_YEAR:.0f}년 실제값과 AI 예측값 사이의 거리입니다.",
                        "bg": "#ffebee",
                        "border": "#ef9a9a",
                    },
                ],
                columns=2,
            )
            st.caption(
                f"오차 거리 비교: 내 예측 {my_error:.2f}, AI 예측 {ai_error:.2f}. "
                "그래프의 세로선 길이가 바로 이 거리입니다."
            )

        st.markdown(pretty_title("나의 말로 결과 정리하기", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
        principle_box(
            "4",
            "직접 맞추기와 머신러닝 비교",
            "직접 맞춘 모델과 머신러닝이 찾은 모델을 비교했을 때, 두 방법의 장단점은 무엇이라고 생각하나요?",
            "d3_answer_4",
            "직접 맞추기는 내가 식의 모양을 스스로 생각해 보게 해 주지만, 정확하게 맞추려면 시간이 걸리고 오차가 남기 쉽습니다. 반면 머신러닝은 정해진 범위 안에서 오차가 작아지도록 식을 빠르게 조절합니다. 하지만 어떤 식을 선택할지, 결과를 믿어도 되는지는 여전히 사람이 판단해야 합니다.",
        )

    with tabs[4]:
        stage_intro(
            "우리의 삶과 사회로 연결하기",
            "오차가 작다고 해서 언제나 결과를 그대로 믿어도 되는 것은 아니므로, "
            "AI 예측이 실제 사회에 미치는 영향까지 함께 생각하는 과정입니다.",
            "오차가 작다고 해서 그 예측을 바로 믿어도 될까?",
            "#fff3e0",
            "#ffe0b2",
        )

        st.markdown(pretty_title("학생 정보 입력 및 탐구 포트폴리오 저장", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        group_name = st.text_input("모둠 이름 (예: 1모둠)", key="d3_group")
        student_col1, student_col2 = st.columns(2)
        with student_col1:
            st.markdown(pretty_title("학생 1", "#f4f9ff", "#dbeafe"), unsafe_allow_html=True)
            stu_id_1 = st.text_input("학생 1 학번 (예: 10101)", max_chars=5, key="d3_id_1")
            stu_name_1 = st.text_input("학생 1 이름 (예: 홍길동)", key="d3_name_1")
        with student_col2:
            st.markdown(pretty_title("학생 2", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            stu_id_2 = st.text_input("학생 2 학번 (예: 10102)", max_chars=5, key="d3_id_2")
            stu_name_2 = st.text_input("학생 2 이름 (예: 김미래)", key="d3_name_2")

        class_num = class_key_from_ids(stu_id_1, stu_id_2)

        st.markdown(pretty_title("탐구 포트폴리오 저장", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
        if group_name and stu_id_1 and stu_name_1 and stu_id_2 and stu_name_2:
            valid_classes = ["1", "2", "5", "6"]
            if class_num in valid_classes:
                    render_value_cards(
                        [
                            {
                                "title": "입력한 모둠",
                                "value": group_name,
                                "detail": "이 정보가 PDF와 공유 글 제목에 함께 반영됩니다.",
                                "bg": "#f4f9ff",
                                "border": "#90caf9",
                            },
                            {
                                "title": "학생 1",
                                "value": f"{stu_id_1} {stu_name_1}",
                                "detail": "첫 번째 학생 정보입니다.",
                                "bg": "#f1f8e9",
                                "border": "#aed581",
                            },
                            {
                                "title": "학생 2",
                                "value": f"{stu_id_2} {stu_name_2}",
                                "detail": "두 번째 학생 정보입니다.",
                                "bg": "#fff8e1",
                                "border": "#ffcc80",
                            },
                        ]
                    )
                    st.success("학생 2명의 정보를 입력했으므로 바로 포트폴리오를 저장할 수 있습니다. 아래 답변을 작성한 뒤 다시 저장하면 최신 내용이 함께 반영됩니다.")
                    student_info = {
                        "group": group_name,
                        "id_1": stu_id_1,
                        "name_1": stu_name_1,
                        "id_2": stu_id_2,
                        "name_2": stu_name_2,
                    }
                    social_meaning_for_pdf = st.session_state.get("d3_social_meaning", "")
                    action_plan_for_pdf = st.session_state.get("d3_action_plan", "")
                    q_deep_for_pdf = st.session_state.get("d3_q_deep", "")
                    principle_data = [
                        (
                            "문제 1. 오차란 무엇인가",
                            clean_text(st.session_state.get("d3_answer_1", ""))
                            + "\n\n[학생 질문]\n"
                            + clean_text(st.session_state.get("d3_answer_1_student_q", "")),
                        ),
                        (
                            "문제 2. 오차를 모아 본 그래프와 최솟값",
                            clean_text(st.session_state.get("d3_answer_2", ""))
                            + "\n\n[학생 질문]\n"
                            + clean_text(st.session_state.get("d3_answer_2_student_q", "")),
                        ),
                        (
                            "문제 3. 데이터에 맞는 모델 고르기",
                            clean_text(st.session_state.get("d3_answer_3", ""))
                            + "\n\n[학생 질문]\n"
                            + clean_text(st.session_state.get("d3_answer_3_student_q", "")),
                        ),
                        (
                            "문제 4. 직접 맞추기와 머신러닝 비교",
                            clean_text(st.session_state.get("d3_answer_4", ""))
                            + "\n\n[학생 질문]\n"
                            + clean_text(st.session_state.get("d3_answer_4_student_q", "")),
                        ),
                    ]
                    mission_rows = build_mission_rows()
                    figure_items = build_figure_items()
                    pdf_bytes = create_portfolio_pdf(
                        student_info,
                        principle_data,
                        mission_rows,
                        social_meaning_for_pdf,
                        action_plan_for_pdf,
                        q_deep_for_pdf,
                        figure_items,
                    )

                    portfolio_urls = {
                        "1": "https://padlet.com/ps0andd/p_1",
                        "2": "https://padlet.com/ps0andd/p_2",
                        "5": "https://padlet.com/ps0andd/p_5",
                        "6": "https://padlet.com/ps0andd/p_6",
                    }
                    padlet_portfolio_url = portfolio_urls.get(class_num, "https://padlet.com/")

                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button(
                            label="탐구 결과 PDF 다운로드",
                            data=pdf_bytes,
                            file_name=f"{group_name}_{stu_name_1}_{stu_name_2}_4차시_탐구포트폴리오.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    with col_btn2:
                        st.markdown(
                            f"""<a href="{padlet_portfolio_url}" target="_blank"
                               style="display: block; padding: 10px; background: linear-gradient(90deg, #43a047 0%, #66bb6a 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                               {class_num}반 포트폴리오 패들렛 열기
                            </a>""",
                            unsafe_allow_html=True,
                        )
            else:
                st.error("두 학생 중 한 명의 학번 세 번째 숫자가 해당 학급(1, 2, 5, 6반)인지 다시 확인해 주세요.")
        else:
            st.info("모둠 이름과 학생 2명의 학번, 이름을 입력하면 바로 탐구 포트폴리오를 저장할 수 있습니다.")

        st.markdown("---")
        st.markdown(pretty_title("교사 논쟁적 질문을 읽고 모둠의 생각 정리하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        st.info(
            "AI가 오차가 작은 식을 잘 찾더라도, 그 결과를 현실에 그대로 적용해도 되는지는 "
            "따로 판단해야 합니다. 예측 결과가 사람과 사회에 미칠 영향을 함께 생각해 봅시다."
            "예를 들어, AI가 어떤 학생의 성적 향상 가능성을 낮게 예측했다면 학교는 그 결과를 그대로 믿고 기회를 줄여도 될까요? "
            "아니면 한 번의 예측 오류 가능성을 고려해 사람이 다시 판단해야 할까요?"
        )
        social_cols = st.columns(2)
        with social_cols[0]:
            st.markdown(pretty_title("모둠의 생각 1", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
            social_meaning = st.text_area(
                "모둠의 생각 1. 사회에 주는 시사점",
                height=110,
                key="d3_social_meaning",
                placeholder="예: 예측이 빠르고 편리해도 한 사람의 기회와 삶에 영향을 줄 수 있으므로 결과를 신중하게 봐야 한다.",
            )
        with social_cols[1]:
            st.markdown(pretty_title("모둠의 생각 2", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
            action_plan = st.text_area(
                "모둠의 생각 2. 우리의 실천 제안",
                height=110,
                key="d3_action_plan",
                placeholder="예: 중요한 예측 결과는 사람이 다시 확인하고, 데이터 기준도 공개해야 한다.",
            )

        st.markdown("---")
        st.markdown(pretty_title("학생 질문 만들기 돕는 질문 씨앗", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        render_value_cards(
            [
                {
                    "title": "신뢰",
                    "value": "오차가 작아도 바로 믿어도 될까?",
                    "detail": "예측이 잘 맞아 보여도 사람이 다시 확인해야 하는 이유는 무엇일까요?",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "책임",
                    "value": "예측이 틀렸을 때 누가 책임질까?",
                    "detail": "AI가 잘못된 판단을 내렸을 때, 학교나 사회는 어떤 절차를 가져야 할까요?",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
                {
                    "title": "데이터 부족",
                    "value": "데이터가 부족하면 얼마나 위험할까?",
                    "detail": "자료가 적거나 치우쳐 있으면 예측 결과는 어떻게 달라질 수 있을까요?",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "사회적 영향",
                    "value": "AI 예측은 누구에게 영향을 줄까?",
                    "detail": "기후, 건강, 경제 예측이 사람들의 선택과 기회에 어떤 영향을 줄지 생각해 보세요.",
                    "bg": "#ede7f6",
                    "border": "#b39ddb",
                },
            ],
            columns=2,
        )

        st.markdown("---")
        st.markdown(pretty_title("모둠 심화 질문 만들기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write("모둠원과 함께 오늘 활동을 돌아보며 심화 질문을 하나 만들고, 앞서 작성한 답변과 함께 패들렛에 공유해 봅시다.")
        st.info(
            "🔥 [우리의 딥(Deep) 퀘스천] (윤리와 철학)\n"
            "배운 지식이나 기술이 실제 사회에 적용될 때 생길 수 있는 부작용이나 윤리적 딜레마를 다루며, "
            "정답 없이 서로의 가치관을 깊이 있게 나눌 수 있는 토론형 질문입니다.\n"
            "👉 예: 만약 인공지능이 과거 성적 데이터만으로 학생의 진로를 추천한다면, "
            "우리는 그 예측이 학생의 가능성까지 공정하게 반영한다고 믿을 수 있을까?"
        )
        q_deep = st.text_area(
            "🔥 [우리의 딥(Deep) 퀘스천]",
            height=110,
            key="d3_q_deep",
            placeholder="예: 오차가 작더라도 AI가 소수의 사례를 계속 불리하게 예측한다면, 우리는 그 결과를 공정하다고 말할 수 있을까?",
        )

        if group_name and stu_id_1 and stu_name_1 and stu_id_2 and stu_name_2 and social_meaning and action_plan and q_deep and class_num:
            if class_num in ["1", "2", "5", "6"]:
                report_text = f"""[F.U.T.U.R.E. 프로젝트 4DAY 성찰 일지]
모둠명: {group_name}
모둠원: {stu_id_1} {stu_name_1}, {stu_id_2} {stu_name_2}

🔥 [우리가 만든 딥(Deep) 퀘스천]
{q_deep}

💡 [교사의 심화 질문에 대한 우리의 생각]
(사회에 주는 시사점) {social_meaning}
(우리의 실천 제안) {action_plan}
"""
                st.success("✅ 성찰 질문 작성이 완료되었습니다! 텍스트를 복사하여 패들렛에 업로드하세요.")
                st.code(report_text, language="markdown")

                qa_urls = {
                    "1": "https://padlet.com/ps0andd/q_1",
                    "2": "https://padlet.com/ps0andd/q_2",
                    "5": "https://padlet.com/ps0andd/q_5",
                    "6": "https://padlet.com/ps0andd/q_6",
                }
                padlet_qa_url = qa_urls.get(class_num, "https://padlet.com/")

                st.markdown(
                    f"""<a href="{padlet_qa_url}" target="_blank"
                        style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 5px;">
                        {class_num}반 질문(Q&A) 패들렛으로 이동하기
                    </a>""",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("사회에 주는 시사점, 우리의 실천 제안, 딥(Deep) 퀘스천을 모두 작성해야 공유 양식이 나타납니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
