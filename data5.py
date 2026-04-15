import datetime
import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


font_path = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")

try:
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc("font", family=font_name)
    mpl.rc("axes", unicode_minus=False)
except Exception:
    pass


DATASETS = {
    "경제: 광고와 매출": {
        "table": pd.DataFrame(
            {
                "광고 게시 횟수(회)": [1, 2, 3, 4, 5, 6, 7],
                "매장 방문자 수(명)": [120, 148, 181, 219, 258, 304, 345],
                "주간 매출(만원)": [18, 24, 31, 39, 46, 55, 63],
            }
        ),
        "default_x": "광고 게시 횟수(회)",
        "default_y": "주간 매출(만원)",
        "story": "광고, 방문자 수, 매출이 함께 기록된 경제 데이터입니다. 모둠의 분석 의도에 맞게 원인과 결과가 될 만한 변수를 골라 보세요.",
        "prompt": "예를 들어 광고를 많이 할수록 매출이 늘어나는지, 방문자 수가 늘 때 매출이 얼마나 달라지는지 등을 탐구할 수 있습니다.",
    },
    "의학: 재활과 회복": {
        "table": pd.DataFrame(
            {
                "재활 훈련 주차(주)": [1, 2, 3, 4, 5, 6, 7],
                "하루 걷기 시간(분)": [10, 18, 24, 30, 34, 36, 38],
                "회복 점수(점)": [22, 37, 49, 58, 64, 68, 71],
            }
        ),
        "default_x": "재활 훈련 주차(주)",
        "default_y": "회복 점수(점)",
        "story": "재활 주차, 걷기 시간, 회복 점수가 함께 있는 의학 데이터입니다. 어떤 변수를 설명 기준으로 둘지 직접 정해 볼 수 있습니다.",
        "prompt": "재활 주차가 길수록 회복 점수가 오르는지, 걷기 시간이 늘 때 회복 점수가 어떻게 변하는지 등을 비교해 보세요.",
    },
    "공학: 태양광 설비": {
        "table": pd.DataFrame(
            {
                "패널 온도(℃)": [10, 15, 20, 25, 30, 35, 40],
                "일사량(kWh)": [2.1, 2.5, 2.9, 3.2, 3.0, 2.7, 2.3],
                "발전 효율(%)": [84, 91, 96, 98, 95, 89, 80],
            }
        ),
        "default_x": "패널 온도(℃)",
        "default_y": "발전 효율(%)",
        "story": "패널 온도, 일사량, 발전 효율이 함께 기록된 공학 데이터입니다. 모둠이 궁금한 관계를 골라 직접 모델을 만들어 보세요.",
        "prompt": "온도와 효율, 일사량과 효율처럼 두 변수의 관계를 선택하면 같은 데이터도 다른 질문으로 바뀝니다.",
    },
    "환경: 기후 변화": {
        "table": pd.DataFrame(
            {
                "관측 순서(차)": [1, 2, 3, 4, 5, 6, 7],
                "강수량(mm)": [92, 88, 95, 101, 106, 112, 118],
                "평균 기온(℃)": [12.1, 12.2, 12.5, 12.7, 13.0, 13.3, 13.6],
            }
        ),
        "default_x": "관측 순서(차)",
        "default_y": "평균 기온(℃)",
        "story": "관측 순서, 강수량, 평균 기온이 함께 있는 환경 데이터입니다. 모둠의 관심에 따라 어떤 변화를 예측할지 정할 수 있습니다.",
        "prompt": "시간이 지날수록 기온이 오르는지, 강수량이 늘 때 기온도 함께 달라지는지 등을 스스로 정해 볼 수 있습니다.",
    },
}


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
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 5차시 AI 데이터 예측 포트폴리오", ln=1, align="C")
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
        for idx, (key, value) in enumerate(kv_pairs):
            x = x0 + (idx % 2) * col_w
            if idx % 2 == 0 and idx > 0:
                self.ln(cell_h)
            self.set_x(x)
            self.set_text_color(120, 120, 120)
            self.cell(col_w * 0.35, cell_h, str(key), border=1)
            self.set_text_color(33, 33, 33)
            self.cell(col_w * 0.65, cell_h, str(value), border=1)
        if len(kv_pairs) % 2 == 1:
            self.set_x(x0 + col_w)
            self.cell(col_w * 0.35, cell_h, "", border=1)
            self.cell(col_w * 0.65, cell_h, "", border=1)
        self.ln(cell_h + 3)


def apply_local_style():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
        div[data-baseweb="tab-list"] {gap: 0.35rem;}
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


def clean_text(value, default="작성한 내용이 없습니다."):
    text = str(value).strip() if value is not None else ""
    return text if text else default


def normalize_pdf_output(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("latin1")
    return bytes(value)


def add_text_box_to_pdf(pdf, title, text, fill_color=(245, 245, 245)):
    pdf.set_font(pdf._font_family, "", 11)
    pdf.set_text_color(21, 101, 192)
    pdf.cell(0, 8, title, ln=1)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font(pdf._font_family, "", 10)
    pdf.set_fill_color(*fill_color)
    pdf.multi_cell(0, 6, clean_text(text), border=1, fill=True)
    pdf.ln(3)


def add_figure_to_pdf(pdf, title, fig):
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


def show_pretty_table(df, height=210):
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
            <div style="font-size:0.9rem; font-weight:700; color:#5e35b1; margin-bottom:8px;">F.U.T.U.R.E. 프로젝트 5DAY</div>
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


def format_value(value):
    num = float(value)
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.2f}"


def dataframe_to_matrix_latex(df):
    rows = []
    for row in df.to_numpy(dtype=float):
        rows.append(" & ".join(format_value(value) for value in row))
    return r"A = \begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def selected_ml_name(degree):
    return "직선 회귀" if int(degree) == 1 else "2차 회귀"


def selected_ml_latex(model_results, degree):
    return model_results["line_latex"] if int(degree) == 1 else model_results["quad_latex"]


def build_selected_comparison_df(model_results, degree):
    ml_name = selected_ml_name(degree)
    metric_df = model_results["metrics_df"].copy()
    filtered = metric_df[metric_df["모델"].isin([ml_name, "딥러닝"])].reset_index(drop=True)
    return filtered


def make_selected_model_compare_figure(dataset, split, model_results, degree):
    ml_name = selected_ml_name(degree)
    fig = Figure(figsize=(7.2, 4.4))
    ax = fig.subplots()
    x_line = np.linspace(float(np.min(split["x_obs"])), float(split["x_hidden"]), 220)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#1976d2", edgecolors="white", linewidths=1.8, label="입력 데이터")
    ml_color = "#ff9800" if ml_name == "직선 회귀" else "#fb8c00"
    ax.plot(x_line, pred_map[ml_name], color=ml_color, linewidth=2.6, linestyle="--", label=ml_name)
    ax.plot(x_line, pred_map["딥러닝"], color="#43a047", linewidth=2.6, label="딥러닝")
    ax.axvline(split["x_hidden"], color="#90a4ae", linestyle="--", linewidth=1.2)
    ax.set_title(f"{dataset['x_label']}과 {dataset['y_label']}의 관계와 AI 예측")
    ax.set_xlabel(dataset["x_label"])
    ax.set_ylabel(dataset["y_label"])
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_selected_prediction_figure(dataset, split, model_results, degree, student_guess=None, reveal=False):
    ml_name = selected_ml_name(degree)
    fig = Figure(figsize=(7.2, 4.4))
    ax = fig.subplots()
    x_line = np.linspace(float(np.min(split["x_obs"])), float(split["x_hidden"]), 220)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#1976d2", edgecolors="white", linewidths=1.8, label="입력 데이터")
    ml_color = "#ff9800" if ml_name == "직선 회귀" else "#fb8c00"
    ax.plot(x_line, pred_map[ml_name], color=ml_color, linewidth=2.4, linestyle="--", label=ml_name)
    ax.plot(x_line, pred_map["딥러닝"], color="#43a047", linewidth=2.4, label="딥러닝")
    x_hidden = split["x_hidden"]
    model_preds = predict_models(model_results, np.array([x_hidden], dtype=float))
    ml_pred = float(model_preds[ml_name][0])
    dl_pred = float(model_preds["딥러닝"][0])
    ax.scatter([x_hidden], [ml_pred], color="#d32f2f", edgecolors="black", s=110, marker="o", zorder=5, label=f"{ml_name} 예측")
    ax.scatter([x_hidden], [dl_pred], color="#f06292", edgecolors="black", s=120, marker="X", zorder=5, label="딥러닝 예측")
    if student_guess is not None:
        ax.scatter([x_hidden], [student_guess], color="#1565c0", edgecolors="black", s=100, marker="D", zorder=5, label="내 예측")
    if reveal:
        actual = split["y_hidden"]
        ax.scatter([x_hidden], [actual], color="#111111", s=90, marker="*", zorder=6, label="실제값")
        ax.vlines(x_hidden - 0.05, min(ml_pred, actual), max(ml_pred, actual), color="#d32f2f", linestyle="--", linewidth=1.8)
        ax.vlines(x_hidden + 0.05, min(dl_pred, actual), max(dl_pred, actual), color="#f06292", linestyle="--", linewidth=1.8)
        if student_guess is not None:
            ax.vlines(x_hidden + 0.15, min(student_guess, actual), max(student_guess, actual), color="#1565c0", linestyle="--", linewidth=1.8)
    ax.set_title("예측값과 실제값 비교")
    ax.set_xlabel(dataset["x_label"])
    ax.set_ylabel(dataset["y_label"])
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
def split_label_unit(label):
    text = str(label)
    if text.endswith(")") and "(" in text:
        title, unit = text.rsplit("(", 1)
        return title.strip(), unit[:-1].strip()
    return text, ""


def current_dataset(name, x_column, y_column):
    info = DATASETS[name]
    table = info["table"].copy()
    x_label, x_unit = split_label_unit(x_column)
    y_label, y_unit = split_label_unit(y_column)
    return {
        "name": name,
        "table": table,
        "selected_table": table[[x_column, y_column]].copy(),
        "x_column": x_column,
        "y_column": y_column,
        "x": table[x_column].to_numpy(dtype=float),
        "y": table[y_column].to_numpy(dtype=float),
        "x_label": x_label,
        "x_unit": x_unit,
        "y_label": y_label,
        "y_unit": y_unit,
        "story": info["story"],
        "prompt": info["prompt"],
    }


def dataset_split(dataset):
    x_all = dataset["x"]
    y_all = dataset["y"]
    return {
        "x_obs": x_all[:-1],
        "y_obs": y_all[:-1],
        "x_hidden": float(x_all[-1]),
        "y_hidden": float(y_all[-1]),
        "x_all": x_all,
        "y_all": y_all,
    }


def preprocess_values(x_values, y_values, use_scale):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    meta = {
        "use_scale": bool(use_scale),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }
    if not use_scale:
        return x.copy(), y.copy(), meta
    x_span = meta["x_max"] - meta["x_min"] or 1.0
    y_span = meta["y_max"] - meta["y_min"] or 1.0
    x_scaled = (x - meta["x_min"]) / x_span
    y_scaled = (y - meta["y_min"]) / y_span
    return x_scaled, y_scaled, meta


def transform_x(x_values, meta):
    x = np.asarray(x_values, dtype=float)
    if not meta["use_scale"]:
        return x.copy()
    x_span = meta["x_max"] - meta["x_min"] or 1.0
    return (x - meta["x_min"]) / x_span


def inverse_y(y_values, meta):
    y = np.asarray(y_values, dtype=float)
    if not meta["use_scale"]:
        return y.copy()
    y_span = meta["y_max"] - meta["y_min"] or 1.0
    return y * y_span + meta["y_min"]


def sse(y_true, y_pred):
    return float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


@st.cache_data(show_spinner=False)
def run_poly_regression(x_values, y_values, degree):
    x = np.asarray(x_values, dtype=float).reshape(-1, 1)
    y = np.asarray(y_values, dtype=float)
    poly = PolynomialFeatures(degree=int(degree), include_bias=False)
    x_train = poly.fit_transform(x)
    model = LinearRegression().fit(x_train, y)
    y_pred = model.predict(x_train)
    coeffs = polynomial_coeff_array(model, poly)
    latex = get_polynomial_equation_latex(model, poly)
    return {
        "coeffs": coeffs,
        "y_pred": np.asarray(y_pred, dtype=float),
        "latex": latex,
        "degree": int(degree),
    }


def poly_predict(coeffs, x_values):
    return np.polyval(np.asarray(coeffs, dtype=float), np.asarray(x_values, dtype=float))


def polynomial_coeff_array(model, poly):
    terms = poly.get_feature_names_out(["x"])
    coeff_map = {0: float(model.intercept_)}
    for term, coef in zip(terms, model.coef_):
        if "^" in term:
            degree = int(term.split("^")[1])
        else:
            degree = 1
        coeff_map[degree] = float(coef)
    max_degree = max(coeff_map.keys())
    return np.array([coeff_map.get(degree, 0.0) for degree in range(max_degree, -1, -1)], dtype=float)


def get_polynomial_equation_latex(model, poly):
    terms = poly.get_feature_names_out(["x"])
    coefs = model.coef_
    intercept = float(model.intercept_)
    parsed_terms = []
    for term, coef in zip(terms, coefs):
        if abs(coef) > 1e-9:
            degree = int(term.split("^")[1]) if "^" in term else 1
            parsed_terms.append((degree, float(coef)))
    parsed_terms.sort(reverse=True, key=lambda item: item[0])

    latex_terms = []
    for degree, coef in parsed_terms:
        if abs(coef) == 1.0:
            sign = "-" if coef < 0 else ""
            body = f"{sign}x^{{{degree}}}" if degree > 1 else f"{sign}x"
        else:
            body = f"{coef:.2f}x^{{{degree}}}" if degree > 1 else f"{coef:.2f}x"
        latex_terms.append(body)
    if abs(intercept) > 1e-9:
        sign = "-" if intercept < 0 else "+"
        latex_terms.append(f"{sign}{abs(intercept):.2f}")
    expr = " + ".join(latex_terms).replace("+ -", "- ")
    expr = expr[2:] if expr.startswith("+ ") else expr
    return f"y = {expr}" if expr else "y = 0"


@st.cache_resource(show_spinner=False)
def run_deep_learning(x_values, y_values, use_scale, hidden1=8, hidden2=4, epochs=120):
    x = np.asarray(x_values, dtype=float).reshape(-1, 1)
    y = np.asarray(y_values, dtype=float).reshape(-1, 1)

    scaler_x = None
    scaler_y = None
    x_train = x
    y_train = y
    if use_scale:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x_train = scaler_x.fit_transform(x)
        y_train = scaler_y.fit_transform(y)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = Sequential(
        [
            Input(shape=(1,)),
            Dense(hidden1, activation="relu"),
            Dense(hidden2, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(0.01), loss="mse")
    history = model.fit(
        x_train,
        y_train,
        epochs=int(epochs),
        batch_size=min(len(x_train), 8),
        verbose=0,
    )
    y_pred_train = model.predict(x_train, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_train).reshape(-1) if use_scale else y_pred_train.reshape(-1)
    return {
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "losses": np.asarray(history.history["loss"], dtype=float),
        "train_pred": np.asarray(y_pred, dtype=float),
        "architecture": f"1-{hidden1}-{hidden2}-1",
    }


def nn_predict(bundle, x_values):
    x = np.asarray(x_values, dtype=float).reshape(-1, 1)
    x_input = bundle["scaler_x"].transform(x) if bundle["scaler_x"] is not None else x
    y_hat = bundle["model"].predict(x_input, verbose=0)
    if bundle["scaler_y"] is not None:
        return bundle["scaler_y"].inverse_transform(y_hat).reshape(-1)
    return y_hat.reshape(-1)


def get_model_results(x_obs, y_obs, use_scale, hidden1=8, hidden2=4, epochs=120):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    line_result = run_poly_regression(tuple(x_obs), tuple(y_obs), 1)
    quad_result = run_poly_regression(tuple(x_obs), tuple(y_obs), 2)
    dl_result = run_deep_learning(tuple(x_obs), tuple(y_obs), bool(use_scale), int(hidden1), int(hidden2), int(epochs))
    y_line = line_result["y_pred"]
    y_quad = quad_result["y_pred"]
    y_nn = dl_result["train_pred"]

    rows = []
    for name, preds in [("직선 회귀", y_line), ("2차 회귀", y_quad), ("딥러닝", y_nn)]:
        rows.append(
            {
                "모델": name,
                "SSE": round(sse(y_obs, preds), 3),
                "평균 오차": round(mae(y_obs, preds), 3),
                "설명력(R²)": round(float(r2_score(y_obs, preds)), 3),
            }
        )

    return {
        "line_coeffs": line_result["coeffs"],
        "quad_coeffs": quad_result["coeffs"],
        "line_latex": line_result["latex"],
        "quad_latex": quad_result["latex"],
        "nn_model": dl_result,
        "metrics_df": pd.DataFrame(rows),
        "train_preds": {
            "직선 회귀": y_line,
            "2차 회귀": y_quad,
            "딥러닝": y_nn,
        },
    }


def predict_models(model_results, x_values):
    x_values = np.asarray(x_values, dtype=float)
    return {
        "직선 회귀": poly_predict(model_results["line_coeffs"], x_values),
        "2차 회귀": poly_predict(model_results["quad_coeffs"], x_values),
        "딥러닝": nn_predict(model_results["nn_model"], x_values),
    }


def make_observation_figure(dataset, split):
    fig = Figure(figsize=(7.0, 4.0))
    ax = fig.subplots()
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#1976d2", label="관찰한 데이터")
    ax.axvline(split["x_hidden"], color="#90a4ae", linestyle="--", linewidth=1.2)
    ax.scatter([split["x_hidden"]], [np.mean(split["y_obs"])], s=110, marker="X", color="#ef6c00")
    ax.text(
        split["x_hidden"],
        np.mean(split["y_obs"]) + (np.max(split["y_obs"]) - np.min(split["y_obs"])) * 0.08,
        "마지막 값은 숨겨 두고 예측합니다",
        ha="center",
        color="#ef6c00",
        fontsize=10,
    )
    ax.set_title("관찰한 데이터와 숨겨 둔 마지막 값")
    ax.set_xlabel(f"{dataset['x_label']} ({dataset['x_unit']})")
    ax.set_ylabel(f"{dataset['y_label']} ({dataset['y_unit']})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_preprocess_figure(dataset, split, use_scale):
    fig = Figure(figsize=(8.2, 3.7))
    axes = fig.subplots(1, 2)
    axes[0].scatter(split["x_obs"], split["y_obs"], s=60, color="#1976d2")
    axes[0].set_title("원래 데이터")
    axes[0].set_xlabel(dataset["x_label"])
    axes[0].set_ylabel(dataset["y_label"])
    axes[0].grid(alpha=0.25)

    x_scaled, y_scaled, _ = preprocess_values(split["x_obs"], split["y_obs"], use_scale)
    axes[1].scatter(x_scaled, y_scaled, s=60, color="#43a047")
    axes[1].set_title("전처리 후 데이터")
    axes[1].set_xlabel("변환된 X")
    axes[1].set_ylabel("변환된 Y")
    axes[1].set_xlim(-0.05, 1.05 if use_scale else max(split["x_obs"]) * 1.05)
    axes[1].set_ylim(-0.05, 1.05 if use_scale else max(split["y_obs"]) * 1.05)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def make_model_compare_figure(dataset, split, model_results):
    fig = Figure(figsize=(7.2, 4.2))
    ax = fig.subplots()
    x_line = np.linspace(float(np.min(split["x_obs"])), float(split["x_hidden"]), 220)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#263238", label="관찰 데이터")
    colors = {
        "직선 회귀": "#1e88e5",
        "2차 회귀": "#fb8c00",
        "딥러닝": "#d81b60",
    }
    for name, preds in pred_map.items():
        ax.plot(x_line, preds, color=colors[name], linewidth=2.5, label=name)
    ax.axvline(split["x_hidden"], color="#90a4ae", linestyle="--", linewidth=1.2)
    ax.set_title("세 모델의 예측 곡선 비교")
    ax.set_xlabel(f"{dataset['x_label']} ({dataset['x_unit']})")
    ax.set_ylabel(f"{dataset['y_label']} ({dataset['y_unit']})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_training_loss_figure(model_results):
    fig = Figure(figsize=(6.0, 3.0))
    ax = fig.subplots()
    losses = model_results["nn_model"]["losses"]
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#d81b60", linewidth=2.2)
    ax.set_title("딥러닝 학습 손실 변화")
    ax.set_xlabel("epoch")
    ax.set_ylabel("손실(MSE)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def make_network_figure():
    fig = Figure(figsize=(6.6, 3.0))
    ax = fig.subplots()
    ax.axis("off")
    layers = [
        {"x": 0.12, "ys": [0.35, 0.65], "label": "입력층", "color": "#bbdefb", "edge": "#1565c0"},
        {"x": 0.42, "ys": [0.2, 0.4, 0.6, 0.8], "label": "은닉층", "color": "#ffe082", "edge": "#ef6c00"},
        {"x": 0.78, "ys": [0.5], "label": "출력층", "color": "#c8e6c9", "edge": "#2e7d32"},
    ]
    for left, right in zip(layers, layers[1:]):
        for y1 in left["ys"]:
            for y2 in right["ys"]:
                ax.plot([left["x"], right["x"]], [y1, y2], color="#cfd8dc", linewidth=0.8, zorder=1)
    for layer in layers:
        for y in layer["ys"]:
            ax.scatter(layer["x"], y, s=420, color=layer["color"], edgecolors=layer["edge"], linewidths=1.2, zorder=3)
        ax.text(layer["x"], 0.04, layer["label"], ha="center", fontsize=10, fontweight="bold")
    ax.text(0.12, 0.92, "데이터 값", ha="center", color="#1565c0", fontsize=10)
    ax.text(0.42, 0.92, "특징 조합", ha="center", color="#ef6c00", fontsize=10)
    ax.text(0.78, 0.92, "예측 결과", ha="center", color="#2e7d32", fontsize=10)
    fig.tight_layout()
    return fig


def make_dynamic_network_figure(hidden1, hidden2):
    fig = Figure(figsize=(7.2, 4.0))
    ax = fig.subplots()
    ax.axis("off")

    def layer_positions(count):
        if count <= 1:
            return [0.5]
        return np.linspace(0.14, 0.86, int(count)).tolist()

    def node_size(count):
        return max(180, min(420, 1500 / max(int(count), 1)))

    layers = [
        {"x": 0.10, "ys": [0.5], "label": "입력층\n1개", "color": "#bbdefb", "edge": "#1565c0"},
        {"x": 0.36, "ys": layer_positions(hidden1), "label": f"은닉층 1\n{hidden1}개", "color": "#ffe082", "edge": "#ef6c00"},
        {"x": 0.62, "ys": layer_positions(hidden2), "label": f"은닉층 2\n{hidden2}개", "color": "#ffccbc", "edge": "#d84315"},
        {"x": 0.88, "ys": [0.5], "label": "출력층\n1개", "color": "#c8e6c9", "edge": "#2e7d32"},
    ]

    for left, right in zip(layers, layers[1:]):
        for y1 in left["ys"]:
            for y2 in right["ys"]:
                ax.plot([left["x"], right["x"]], [y1, y2], color="#cfd8dc", linewidth=0.8, zorder=1)

    for layer in layers:
        size = node_size(len(layer["ys"]))
        for y in layer["ys"]:
            ax.scatter(layer["x"], y, s=size, color=layer["color"], edgecolors=layer["edge"], linewidths=1.2, zorder=3)
        ax.text(layer["x"], 0.03, layer["label"], ha="center", fontsize=10, fontweight="bold")

    ax.text(0.10, 0.95, "입력값", ha="center", color="#1565c0", fontsize=10)
    ax.text(0.36, 0.95, "특징 찾기", ha="center", color="#ef6c00", fontsize=10)
    ax.text(0.62, 0.95, "패턴 조합", ha="center", color="#d84315", fontsize=10)
    ax.text(0.88, 0.95, "예측값", ha="center", color="#2e7d32", fontsize=10)
    ax.set_title("선택한 뉴런 수로 만든 딥러닝 구조", fontsize=12, pad=8)
    fig.tight_layout()
    return fig


def make_prediction_figure(dataset, split, model_results, student_guess=None, reveal=False):
    fig = Figure(figsize=(7.2, 4.2))
    ax = fig.subplots()
    x_line = np.linspace(float(np.min(split["x_obs"])), float(split["x_hidden"]), 220)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#263238", label="관찰 데이터")
    ax.plot(x_line, pred_map["직선 회귀"], color="#1e88e5", linewidth=2.0, alpha=0.9, label="직선 회귀")
    ax.plot(x_line, pred_map["2차 회귀"], color="#fb8c00", linewidth=2.0, alpha=0.9, label="2차 회귀")
    ax.plot(x_line, pred_map["딥러닝"], color="#d81b60", linewidth=2.3, alpha=0.9, label="딥러닝")

    x_hidden = split["x_hidden"]
    model_preds = predict_models(model_results, np.array([x_hidden], dtype=float))
    ai_pred = float(model_preds["딥러닝"][0])
    ax.scatter([x_hidden], [ai_pred], s=120, color="#d81b60", marker="*", label="AI 예측값")
    if student_guess is not None:
        ax.scatter([x_hidden], [student_guess], s=90, color="#1565c0", marker="o", label="내 예측값")
    if reveal:
        actual = split["y_hidden"]
        ax.scatter([x_hidden], [actual], s=90, color="#111111", marker="D", label="실제값")
        ax.vlines(x_hidden - 0.05, min(ai_pred, actual), max(ai_pred, actual), color="#d81b60", linestyle="--", linewidth=2.0)
        if student_guess is not None:
            ax.vlines(x_hidden + 0.05, min(student_guess, actual), max(student_guess, actual), color="#1565c0", linestyle="--", linewidth=2.0)
    else:
        ax.axvline(x_hidden, color="#90a4ae", linestyle="--", linewidth=1.2)

    ax.set_title("예측값과 실제값의 거리 비교")
    ax.set_xlabel(f"{dataset['x_label']} ({dataset['x_unit']})")
    ax.set_ylabel(f"{dataset['y_label']} ({dataset['y_unit']})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def build_mission_rows(dataset, split, use_scale, model_results, student_guess, reveal):
    metric_df = model_results["metrics_df"].copy()
    best_row = metric_df.loc[metric_df["SSE"].idxmin()]
    hidden_preds = predict_models(model_results, np.array([split["x_hidden"]], dtype=float))
    line_pred = float(hidden_preds["직선 회귀"][0])
    quad_pred = float(hidden_preds["2차 회귀"][0])
    dl_pred = float(hidden_preds["딥러닝"][0])
    final_text = "실제값을 아직 확인하지 않았습니다."
    if reveal:
        final_text = (
            f"실제값은 {split['y_hidden']:.2f}{dataset['y_unit']}이고, "
            f"내 예측 오차는 {abs(float(student_guess) - split['y_hidden']):.2f}{dataset['y_unit']}, "
            f"딥러닝 오차는 {abs(dl_pred - split['y_hidden']):.2f}{dataset['y_unit']}입니다."
        )
    return [
        (
            "문제 1. 데이터의 변화를 읽기",
            f"{dataset['name']} 데이터를 선택하고, 독립 변수는 {dataset['x_column']}, 종속 변수는 {dataset['y_column']}로 정했다. "
            f"관찰한 데이터는 {len(split['x_obs'])}개이며, 마지막 {dataset['y_label']} 값은 숨겨 두고 예측 문제로 사용했다.",
        ),
        (
            "문제 2. 전처리의 의미 이해하기",
            "0~1 정규화를 적용했다." if use_scale else "원래 단위를 유지하며 데이터를 해석했다.",
        ),
        (
            "문제 3. 모델 비교하기",
            f"직선 회귀 예측값은 {line_pred:.2f}, 2차 회귀 예측값은 {quad_pred:.2f}, 딥러닝 예측값은 {dl_pred:.2f}이다. "
            f"SSE 기준 가장 잘 맞는 모델은 {best_row['모델']}이다.",
        ),
        (
            "문제 4. 예측 결과 해석하기",
            f"내 예측값은 {float(student_guess):.2f}{dataset['y_unit']}이다. {final_text}",
        ),
    ]


def create_portfolio_pdf(student_info, dataset, split, model_results, ml_degree, use_scale, student_guess, reveal, analysis_text, interpretation_text, figure_items):
    pdf = ThemedPDF()
    pdf.add_font("Nanum", "", font_path, uni=True)
    pdf.set_font("Nanum", "", 12)
    pdf._font_family = "Nanum"
    pdf.footer_left = f"{student_info.get('group', '')} - {student_info.get('name', '')}"
    pdf.add_page()

    kvs = [
        ("모둠명", student_info.get("group", "")),
        ("학번", student_info.get("id", "")),
        ("이름", student_info.get("name", "")),
        ("작성일", datetime.datetime.now().strftime("%Y-%m-%d")),
    ]
    pdf.kv_card("학생 정보", kvs)

    active_ml_name = selected_ml_name(ml_degree)
    hidden_preds = predict_models(model_results, np.array([split["x_hidden"]], dtype=float))
    ml_pred = float(hidden_preds[active_ml_name][0])
    dl_pred = float(hidden_preds["딥러닝"][0])
    best_model = model_results["metrics_df"].loc[model_results["metrics_df"]["SSE"].idxmin(), "모델"]
    matrix_text = dataset["selected_table"].round(3).to_string(index=False)

    analysis_kvs = [
        ("활동 데이터", dataset["name"]),
        ("독립 변수", dataset["x_column"]),
        ("종속 변수", dataset["y_column"]),
        ("머신러닝", active_ml_name),
        ("딥러닝 구조", model_results["nn_model"]["architecture"]),
        ("정규화", "적용" if use_scale else "미적용"),
    ]
    pdf.kv_card("분석 설정", analysis_kvs)

    pdf.h2("선택한 자료 행렬")
    add_text_box_to_pdf(pdf, "행렬 형태로 본 자료", matrix_text, fill_color=(250, 250, 250))

    pdf.h2("모델 요약")
    pdf.p(f"머신러닝 식: {selected_ml_latex(model_results, ml_degree)}", size=10)
    pdf.p(f"머신러닝 예측값: {ml_pred:.3f}", size=10)
    pdf.p(f"딥러닝 예측값: {dl_pred:.3f}", size=10)
    pdf.p(f"SSE 기준 가장 잘 맞은 모델: {best_model}", size=10)
    if reveal:
        actual = split["y_hidden"]
        pdf.p(f"실제값: {actual:.3f}", size=10)
        pdf.p(f"내 오차: {abs(float(student_guess) - actual):.3f}", size=10)
        pdf.p(f"{active_ml_name} 오차: {abs(ml_pred - actual):.3f}", size=10)
        pdf.p(f"딥러닝 오차: {abs(dl_pred - actual):.3f}", size=10)

    pdf.h2("데이터 분석 및 예측 결과")
    pdf.p(clean_text(analysis_text))
    pdf.h2("연구 결과 및 해석")
    pdf.p(clean_text(interpretation_text))

    for fig_title, fig in figure_items:
        pdf.add_page()
        add_figure_to_pdf(pdf, fig_title, fig)

    return normalize_pdf_output(pdf.output(dest="S"))


def run():
    apply_local_style()
    page_banner(
        "AI를 이용한 데이터 예측",
        "실생활 데이터를 선택한 뒤 독립 변수와 종속 변수를 정하고, AI 예측 모델로 미래 값을 예측하며 "
        "그 결과를 수학적 해석과 보고서 작성으로 연결해 봅시다.",
    )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)

    dataset_names = list(DATASETS.keys())
    st.session_state.setdefault("d5_group", "")
    st.session_state.setdefault("d5_use_scale", True)
    st.session_state.setdefault("d5_dataset", dataset_names[0])
    st.session_state.setdefault("d5_ml_degree", 1)
    st.session_state.setdefault("d5_hidden1", 8)
    st.session_state.setdefault("d5_hidden2", 4)
    st.session_state.setdefault("d5_epochs", 120)

    tabs = st.tabs(
        [
            "1단계 [F.U] 데이터 선택",
            "2단계 [T] 자료를 행렬로 보기",
            "3단계 [U] 머신러닝 vs 딥러닝",
            "4단계 [R] 예측 및 시각화",
            "5단계 [E] 보고서 작성 및 저장",
        ]
    )

    with tabs[0]:
        stage_intro(
            "모둠과 분석 데이터 선택",
            "학생의 흥미에 맞는 분야 데이터를 고르고, 독립 변수와 종속 변수를 직접 정해 분석 방향을 여는 단계입니다.",
            "같은 데이터라도 어떤 변수를 원인(X)과 결과(Y)로 정하느냐에 따라 어떤 예측 질문이 만들어질까?",
            "#e3f2fd",
            "#bbdefb",
        )
        st.markdown(pretty_title("모둠과 데이터 분석 방향 정하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        input_col, select_col = st.columns([0.9, 1.1])
        with input_col:
            st.text_input("모둠명", key="d5_group", placeholder="예: 1모둠")
        with select_col:
            st.selectbox("활동 데이터 선택", dataset_names, key="d5_dataset")

        dataset_info = DATASETS[st.session_state["d5_dataset"]]
        all_columns = list(dataset_info["table"].columns)
        if st.session_state.get("d5_x_col") not in all_columns:
            st.session_state["d5_x_col"] = dataset_info["default_x"]

        x_col, y_col = st.columns(2)
        with x_col:
            st.selectbox("독립 변수(X) 선택", all_columns, key="d5_x_col")

        valid_y_options = [col for col in all_columns if col != st.session_state["d5_x_col"]]
        preferred_y = dataset_info["default_y"] if dataset_info["default_y"] in valid_y_options else valid_y_options[0]
        if st.session_state.get("d5_y_col") not in valid_y_options:
            st.session_state["d5_y_col"] = preferred_y
        with y_col:
            st.selectbox("종속 변수(Y) 선택", valid_y_options, key="d5_y_col")

        render_value_cards(
            [
                {
                    "title": "모둠명",
                    "value": st.session_state.get("d5_group", "") or "미입력",
                    "detail": "이 정보가 마지막 보고서와 PDF 저장 단계까지 이어집니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "활동 데이터",
                    "value": st.session_state["d5_dataset"],
                    "detail": "경제, 의학, 공학, 환경 중 관심 있는 분야를 고릅니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "독립 변수 X",
                    "value": st.session_state["d5_x_col"],
                    "detail": "원인이나 설명 기준으로 볼 변수를 선택합니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
                {
                    "title": "종속 변수 Y",
                    "value": st.session_state["d5_y_col"],
                    "detail": "예측하거나 설명받을 결과 변수를 선택합니다.",
                    "bg": "#fce4ec",
                    "border": "#f48fb1",
                },
            ],
            columns=2,
        )

        selected_preview = DATASETS[st.session_state["d5_dataset"]]["table"].copy()
        selected_preview.insert(0, "행 번호", np.arange(1, len(selected_preview) + 1))
        st.markdown(pretty_title("선택한 자료 미리 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        left, right = st.columns([1.15, 1.0])
        with left:
            show_pretty_table(selected_preview, height=280)
        with right:
            st.info(DATASETS[st.session_state["d5_dataset"]]["story"])
            st.markdown(f"**탐구 힌트**: {DATASETS[st.session_state['d5_dataset']]['prompt']}")
            st.write("학생이 관심 있는 분야를 먼저 고른 뒤, 어떤 변수를 원인(X)과 결과(Y)로 볼지 정하는 단계입니다.")
            st.write(
                f"현재 선택: X = **{st.session_state['d5_x_col']}** / "
                f"Y = **{st.session_state['d5_y_col']}**"
            )

        dataset = current_dataset(
            st.session_state["d5_dataset"],
            st.session_state["d5_x_col"],
            st.session_state["d5_y_col"],
        )
        split = dataset_split(dataset)
        selection_signature = (
            st.session_state["d5_dataset"],
            st.session_state["d5_x_col"],
            st.session_state["d5_y_col"],
        )
        if st.session_state.get("d5_last_selection") != selection_signature:
            st.session_state["d5_last_selection"] = selection_signature
            st.session_state["d5_prediction_revealed"] = False
            st.session_state["d5_student_guess"] = float(split["y_obs"][-1])
        st.session_state.setdefault("d5_student_guess", float(split["y_obs"][-1]))

        observation_fig = make_observation_figure(dataset, split)

    dataset = current_dataset(
        st.session_state["d5_dataset"],
        st.session_state["d5_x_col"],
        st.session_state["d5_y_col"],
    )
    split = dataset_split(dataset)
    st.session_state.setdefault("d5_student_guess", float(split["y_obs"][-1]))
    selected_matrix = dataset["selected_table"].copy()
    matrix_df = pd.DataFrame(
        selected_matrix.to_numpy(dtype=float),
        index=[f"행{i}" for i in range(1, len(selected_matrix) + 1)],
        columns=[f"열1: {dataset['x_label']}", f"열2: {dataset['y_label']}"],
    )
    matrix_latex = dataframe_to_matrix_latex(selected_matrix)
    summary_df = pd.DataFrame(
        {
            "항목": [
                "X 평균",
                "Y 평균",
                "X 최솟값",
                "Y 최솟값",
                "X 최댓값",
                "Y 최댓값",
                "상관계수",
            ],
            "값": [
                round(float(np.mean(dataset["x"])), 3),
                round(float(np.mean(dataset["y"])), 3),
                round(float(np.min(dataset["x"])), 3),
                round(float(np.min(dataset["y"])), 3),
                round(float(np.max(dataset["x"])), 3),
                round(float(np.max(dataset["y"])), 3),
                round(float(np.corrcoef(dataset["x"], dataset["y"])[0, 1]), 3),
            ],
        }
    )
    preprocess_fig = make_preprocess_figure(dataset, split, st.session_state["d5_use_scale"])

    with tabs[1]:
        stage_intro(
            "선택한 자료를 행렬로 이해하기",
            "선택한 두 변수를 행과 열로 정리해 보고, 공통수학1의 행렬 관점에서 데이터를 읽는 단계입니다.",
            "실생활 데이터도 숫자 배열인 행렬로 보면 어떤 관계를 더 분명하게 읽을 수 있을까?",
            "#fff8e1",
            "#ffecb3",
        )
        st.markdown(pretty_title("공통수학1 연결: 자료를 행렬로 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        left, right = st.columns([1.05, 1.15])
        with left:
            st.latex(matrix_latex)
            st.caption("행은 각 자료, 열1은 독립 변수 X, 열2는 종속 변수 Y를 뜻합니다.")
            st.pyplot(observation_fig, use_container_width=True)
        with right:
            matrix_display = matrix_df.reset_index().rename(columns={"index": "행"})
            show_pretty_table(matrix_display, height=300)
        render_value_cards(
            [
                {
                    "title": "자료 개수",
                    "value": str(len(selected_matrix)),
                    "detail": "행 하나가 하나의 관측 자료를 뜻합니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "행렬 크기",
                    "value": f"{selected_matrix.shape[0]}×{selected_matrix.shape[1]}",
                    "detail": "열1은 X, 열2는 Y로 구성된 데이터 행렬입니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
                {
                    "title": "정규화 상태",
                    "value": "적용" if st.session_state["d5_use_scale"] else "미적용",
                    "detail": "0~1 정규화는 AI가 자료를 읽기 쉽게 만드는 전처리입니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
            ],
            columns=3,
        )
        st.markdown("##### 자료 요약 정보")
        sum_col1, sum_col2 = st.columns([1.0, 1.15])
        with sum_col1:
            show_pretty_table(summary_df, height=250)
        with sum_col2:
            st.checkbox("0~1 정규화 적용하기", key="d5_use_scale")
            st.caption("정규화는 자료의 상대적 위치를 유지한 채 0~1 범위로 바꾸어 AI가 읽기 쉽게 만드는 과정입니다.")
            st.pyplot(preprocess_fig, use_container_width=True)

    with tabs[2]:
        stage_intro(
            "머신러닝 vs 딥러닝",
            "선택한 데이터에 대해 1차/2차 회귀와 딥러닝을 함께 실험하며 어떤 모델이 더 잘 맞는지 비교하는 단계입니다.",
            "같은 데이터를 보더라도 머신러닝과 딥러닝은 어떤 방식으로 다른 예측을 만들까?",
            "#e8f5e9",
            "#c8e6c9",
        )
        st.markdown(pretty_title("AI 예측 모델 실험하기", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
        ml_col, dl_col = st.columns(2)
        with ml_col:
            st.markdown(pretty_title("머신러닝 회귀 분석", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.info("머신러닝은 자료를 보고 스스로 함수식을 찾아 예측합니다. 공통수학1과 연결해 1차와 2차 함수까지만 분석합니다.")
            st.selectbox("회귀 차수 선택", options=[1, 2], key="d5_ml_degree")
        with dl_col:
            st.markdown(pretty_title("딥러닝 모델", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.info("딥러닝은 여러 층을 거치며 패턴을 읽습니다. data7.py처럼 실제 Keras 모델로 가볍게 학습합니다.")
            inner1, inner2, inner3 = st.columns(3)
            with inner1:
                st.slider("1층 뉴런 수", 4, 12, key="d5_hidden1")
            with inner2:
                st.slider("2층 뉴런 수", 2, 8, key="d5_hidden2")
            with inner3:
                st.slider("학습 횟수", 60, 150, step=10, key="d5_epochs")

        model_results = get_model_results(
            split["x_obs"],
            split["y_obs"],
            st.session_state["d5_use_scale"],
            st.session_state["d5_hidden1"],
            st.session_state["d5_hidden2"],
            st.session_state["d5_epochs"],
        )
        with dl_col:
            st.caption("뉴런 수를 바꾸면 아래 구조 그림의 동그라미 수도 함께 바뀝니다.")
            dl_viz1, dl_viz2 = st.columns(2)
            with dl_viz1:
                st.pyplot(
                    make_dynamic_network_figure(
                        st.session_state["d5_hidden1"],
                        st.session_state["d5_hidden2"],
                    ),
                    use_container_width=True,
                )
            with dl_viz2:
                st.pyplot(make_training_loss_figure(model_results), use_container_width=True)
        active_ml_name = selected_ml_name(st.session_state["d5_ml_degree"])
        comparison_df = build_selected_comparison_df(model_results, st.session_state["d5_ml_degree"])
        compare_fig = make_selected_model_compare_figure(dataset, split, model_results, st.session_state["d5_ml_degree"])
        model_cards = []
        for _, row in comparison_df.iterrows():
            model_cards.append(
                {
                    "title": str(row["모델"]),
                    "value": f"SSE {float(row['SSE']):.3f}",
                    "detail": f"평균 오차 {float(row['평균 오차']):.3f} / 설명력 {float(row['설명력(R²)']):.3f}",
                    "bg": "#f4f9ff" if str(row["모델"]) == active_ml_name else "#f1f8e9",
                    "border": "#90caf9" if str(row["모델"]) == active_ml_name else "#aed581",
                }
            )

        st.markdown(pretty_title("모델 비교 결과 보기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        st.pyplot(compare_fig, use_container_width=True)
        render_value_cards(model_cards, columns=len(model_cards))
        formula_col, structure_col = st.columns([1.15, 0.85])
        with formula_col:
            st.markdown("##### 머신러닝 식")
            st.latex(selected_ml_latex(model_results, st.session_state["d5_ml_degree"]))
        with structure_col:
            st.markdown("##### 딥러닝 구조")
            st.write(f"현재 구조: `{model_results['nn_model']['architecture']}`")
        st.markdown(pretty_title("실제값과 예측값 오차 비교", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        error_df = pd.DataFrame(
            {
                dataset["x_label"]: np.round(split["x_obs"], 3),
                "실제값": np.round(split["y_obs"], 3),
                f"{active_ml_name} 예측값": np.round(model_results["train_preds"][active_ml_name], 3),
                "딥러닝 예측값": np.round(model_results["train_preds"]["딥러닝"], 3),
            }
        )
        error_df[f"{active_ml_name} 오차"] = np.round(np.abs(split["y_obs"] - model_results["train_preds"][active_ml_name]), 3)
        error_df["딥러닝 오차"] = np.round(np.abs(split["y_obs"] - model_results["train_preds"]["딥러닝"]), 3)
        show_pretty_table(error_df, height=250)

    with tabs[3]:
        stage_intro(
            "예측 및 시각화",
            "숨겨 둔 값을 직접 예측해 보고, 머신러닝과 딥러닝의 예측을 실제값과 비교하며 오차를 시각적으로 확인하는 단계입니다.",
            "AI가 만든 예측값과 실제값의 차이를 보면 어떤 모델이 더 믿을 만한지 어떻게 판단할 수 있을까?",
            "#f3e5f5",
            "#e1bee7",
        )
        model_results = get_model_results(
            split["x_obs"],
            split["y_obs"],
            st.session_state["d5_use_scale"],
            st.session_state["d5_hidden1"],
            st.session_state["d5_hidden2"],
            st.session_state["d5_epochs"],
        )
        active_ml_name = selected_ml_name(st.session_state["d5_ml_degree"])
        prediction_fig = make_selected_prediction_figure(
            dataset,
            split,
            model_results,
            st.session_state["d5_ml_degree"],
            student_guess=float(st.session_state["d5_student_guess"]),
            reveal=bool(st.session_state.get("d5_prediction_revealed", False)),
        )
        hidden_preds = predict_models(model_results, np.array([split["x_hidden"]], dtype=float))
        ml_pred = float(hidden_preds[active_ml_name][0])
        dl_pred = float(hidden_preds["딥러닝"][0])

        st.markdown(pretty_title("예측 그래프 확인", "#f3e5f5", "#e1bee7"), unsafe_allow_html=True)
        st.pyplot(prediction_fig, use_container_width=True)

        action_col, result_col = st.columns(2)
        with action_col:
            st.markdown(pretty_title("숨겨 둔 값 예측하기", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
            st.info(
                f"현재 숨겨 둔 자료는 X = {split['x_hidden']:.2f}일 때의 "
                f"{dataset['y_label']} 값입니다."
            )
            st.number_input(
                f"내가 예측한 {dataset['y_label']}",
                key="d5_student_guess",
                format="%.2f",
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("예측 확인하기", use_container_width=True):
                    st.session_state["d5_prediction_revealed"] = True
                    st.rerun()
            with btn_col2:
                if st.button("다시 숨기기", use_container_width=True):
                    st.session_state["d5_prediction_revealed"] = False
                    st.rerun()

        with result_col:
            st.markdown(pretty_title("모델 예측값 보기", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
            render_value_cards(
                [
                    {
                        "title": active_ml_name,
                        "value": f"{ml_pred:.3f}",
                        "detail": "선택한 회귀 모델이 숨겨 둔 값을 예측한 결과입니다.",
                        "bg": "#f4f9ff",
                        "border": "#90caf9",
                    },
                    {
                        "title": "딥러닝",
                        "value": f"{dl_pred:.3f}",
                        "detail": "딥러닝 모델이 숨겨 둔 값을 예측한 결과입니다.",
                        "bg": "#f1f8e9",
                        "border": "#aed581",
                    },
                ],
                columns=2,
            )
            if st.session_state.get("d5_prediction_revealed", False):
                actual = split["y_hidden"]
                my_guess = float(st.session_state["d5_student_guess"])
                render_value_cards(
                    [
                        {
                            "title": "실제값",
                            "value": f"{actual:.3f}",
                            "detail": "숨겨 두었던 실제 데이터를 확인한 값입니다.",
                            "bg": "#fff8e1",
                            "border": "#ffcc80",
                        },
                        {
                            "title": "내 오차",
                            "value": f"{abs(my_guess - actual):.3f}",
                            "detail": "내가 직접 예측한 값과 실제값 사이의 거리입니다.",
                            "bg": "#fce4ec",
                            "border": "#f48fb1",
                        },
                        {
                            "title": f"{active_ml_name} 오차",
                            "value": f"{abs(ml_pred - actual):.3f}",
                            "detail": "머신러닝 예측값과 실제값 사이의 거리입니다.",
                            "bg": "#f4f9ff",
                            "border": "#90caf9",
                        },
                        {
                            "title": "딥러닝 오차",
                            "value": f"{abs(dl_pred - actual):.3f}",
                            "detail": "딥러닝 예측값과 실제값 사이의 거리입니다.",
                            "bg": "#f1f8e9",
                            "border": "#aed581",
                        },
                    ],
                    columns=2,
                )
                st.info(
                    f"실제값은 **{actual:.2f}{dataset['y_unit']}** 입니다. "
                    f"내 오차는 **{abs(my_guess - actual):.2f}{dataset['y_unit']}**, "
                    f"{active_ml_name} 오차는 **{abs(ml_pred - actual):.2f}{dataset['y_unit']}**, "
                    f"딥러닝 오차는 **{abs(dl_pred - actual):.2f}{dataset['y_unit']}** 입니다."
                )
            else:
                st.info("먼저 직접 예측한 뒤, 버튼을 눌러 실제값과의 거리를 확인해 보세요.")

        st.markdown(pretty_title("딥러닝은 층에서 이렇게 사고합니다", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        render_value_cards(
            [
                {
                    "title": "입력층",
                    "value": "데이터 받기",
                    "detail": "독립 변수 값을 그대로 받아들이는 출발점입니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "은닉층",
                    "value": "관계 찾기",
                    "detail": "값들 사이의 패턴을 조합하며 중요한 특징을 찾습니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "출력층",
                    "value": "최종 예측",
                    "detail": "앞에서 찾은 특징을 바탕으로 최종 예측값을 만듭니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
            ],
            columns=3,
        )

    with tabs[4]:
        stage_intro(
            "결과 해석과 포트폴리오",
            "분석 결과를 요약하고, 학생 정보와 보고서를 바탕으로 5차시 AI 데이터 예측 포트폴리오를 완성하는 단계입니다.",
            "선택한 데이터와 AI 예측 결과를 어떤 근거로 해석하고 정리하면 좋은 보고서가 될까?",
            "#fff3e0",
            "#ffe0b2",
        )
        model_results = get_model_results(
            split["x_obs"],
            split["y_obs"],
            st.session_state["d5_use_scale"],
            st.session_state["d5_hidden1"],
            st.session_state["d5_hidden2"],
            st.session_state["d5_epochs"],
        )
        active_ml_name = selected_ml_name(st.session_state["d5_ml_degree"])
        hidden_preds = predict_models(model_results, np.array([split["x_hidden"]], dtype=float))
        ml_pred = float(hidden_preds[active_ml_name][0])
        dl_pred = float(hidden_preds["딥러닝"][0])
        summary_metric_df = build_selected_comparison_df(model_results, st.session_state["d5_ml_degree"])
        best_model = summary_metric_df.loc[summary_metric_df["SSE"].idxmin(), "모델"]

        st.markdown(pretty_title("분석 결과 요약", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        render_value_cards(
            [
                {
                    "title": "활동 데이터",
                    "value": dataset["name"],
                    "detail": "학생이 선택한 탐구 분야 데이터입니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "독립 변수 / 종속 변수",
                    "value": f"{dataset['x_column']} / {dataset['y_column']}",
                    "detail": "이번 활동에서 원인과 결과로 정한 변수 조합입니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "가장 잘 맞은 모델",
                    "value": best_model,
                    "detail": "현재 SSE 기준으로 가장 작은 오차를 보인 모델입니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
                {
                    "title": "머신러닝 예측값",
                    "value": f"{ml_pred:.3f}",
                    "detail": f"{selected_ml_name(st.session_state['d5_ml_degree'])} 모델의 예측 결과입니다.",
                    "bg": "#e8f5e9",
                    "border": "#81c784",
                },
                {
                    "title": "딥러닝 예측값",
                    "value": f"{dl_pred:.3f}",
                    "detail": "딥러닝 모델의 예측 결과입니다.",
                    "bg": "#ede7f6",
                    "border": "#b39ddb",
                },
                {
                    "title": "정규화 상태",
                    "value": "적용" if st.session_state["d5_use_scale"] else "미적용",
                    "detail": "데이터 전처리에서 0~1 정규화를 썼는지 보여 줍니다.",
                    "bg": "#fce4ec",
                    "border": "#f48fb1",
                },
            ],
            columns=3,
        )

        st.markdown(pretty_title("핵심 식과 구조", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.markdown("**머신러닝 식**")
        st.latex(selected_ml_latex(model_results, st.session_state["d5_ml_degree"]))
        st.markdown("**딥러닝 구조**")
        st.write(f"`{model_results['nn_model']['architecture']}`")

        st.info(
            "머신러닝과 딥러닝의 예측 결과를 비교한 뒤, 아래 보고서 작성 칸에 분석 결과와 해석을 정리해 보세요."
        )

        st.markdown(pretty_title("학생 정보 입력", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        group_name = st.session_state.get("d5_group", "")
        info_col, guide_col = st.columns([1.15, 1.0])
        with info_col:
            st.markdown(f"**모둠명**: {group_name if group_name else '데이터 선택 탭에서 입력해 주세요.'}")
            info_sub1, info_sub2 = st.columns(2)
            with info_sub1:
                stu_id = st.text_input("학번", key="d5_id")
            with info_sub2:
                stu_name = st.text_input("이름", key="d5_name")
        with guide_col:
            st.info("모둠명, 학번, 이름을 확인한 뒤 아래 보고서를 작성하고 PDF를 저장하세요.")

        st.markdown(pretty_title("보고서 작성", "#fff3e0", "#ffe0b2"), unsafe_allow_html=True)
        st.text_area(
            "데이터 분석 및 예측 결과 작성",
            key="d5_analysis_report",
            height=180,
            placeholder="선택한 데이터, 독립 변수와 종속 변수, 그래프의 흐름, 머신러닝과 딥러닝의 예측 결과를 정리해 보세요.",
        )
        st.text_area(
            "연구 결과 및 해석 작성",
            key="d5_interpretation_report",
            height=180,
            placeholder="어떤 모델이 더 적절했는지, 예측과 실제값의 차이는 어땠는지, 활동을 통해 무엇을 이해했는지 정리해 보세요.",
        )

        st.markdown(pretty_title("PDF 저장", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        if group_name and stu_id and stu_name:
            student_info = {"group": group_name, "id": stu_id, "name": stu_name}
            figure_items = [
                ("관찰 데이터 그래프", observation_fig),
                ("전처리 그래프", preprocess_fig),
                ("모델 비교 그래프", compare_fig),
                ("예측 결과 그래프", prediction_fig),
            ]
            pdf_bytes = create_portfolio_pdf(
                student_info,
                dataset,
                split,
                model_results,
                st.session_state["d5_ml_degree"],
                st.session_state["d5_use_scale"],
                float(st.session_state["d5_student_guess"]),
                bool(st.session_state.get("d5_prediction_revealed", False)),
                st.session_state.get("d5_analysis_report", ""),
                st.session_state.get("d5_interpretation_report", ""),
                figure_items,
            )
            st.download_button(
                "보고서 PDF 저장하기",
                data=pdf_bytes,
                file_name=f"{stu_id}_{stu_name}_5차시_AI데이터예측보고서.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("모둠명, 학번, 이름을 입력하면 보고서 PDF를 저장할 수 있습니다.")


if __name__ == "__main__":
    run()
