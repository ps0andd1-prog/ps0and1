import csv
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

from future_extra_datasets import (
    EXTRA_DATASETS,
    FIELD_DATASETS,
    FIELD_ORDER,
    KOREA_CLIMATE_DATASET,
    field_for_dataset,
    normalize_dataset_name,
)


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


CLASS_OPTIONS = ["1", "2", "5", "6"]
PORT_URLS = {
    "1": "https://padlet.com/ps0andd/p_1",
    "2": "https://padlet.com/ps0andd/p_2",
    "5": "https://padlet.com/ps0andd/p_5",
    "6": "https://padlet.com/ps0andd/p_6",
}

DIRECT_FIELD = "직접 자료 수집"
DIRECT_DATASET_NAME = "직접 자료 수집 데이터"
DIRECT_X_COLUMN = "직접 입력 x"
DIRECT_Y_COLUMN = "직접 입력 y"
DIRECT_FIELD_OPTIONS = FIELD_ORDER + [DIRECT_FIELD]
DIRECT_DEFAULT_X_VALUES = "1, 2, 3, 4, 5"
DIRECT_DEFAULT_Y_VALUES = "2, 4, 6, 8, 10"


DATASETS = {
    "경제: 광고와 판매량": {
        "table": pd.DataFrame(
            {
                "TV 광고비": [230.1, 57.5, 97.5, 218.4, 240.1, 95.7, 177.0, 227.2, 262.7, 261.3, 237.4, 16.9, 75.3, 109.8, 197.6, 280.2, 255.4, 139.2, 123.1, 0.7, 273.7, 96.2, 280.7, 149.8, 117.2, 164.5, 276.7, 205.0, 17.2, 232.1],
                "라디오 광고비": [37.8, 32.8, 7.6, 27.7, 16.7, 1.4, 33.4, 15.8, 28.8, 42.7, 27.5, 43.7, 20.3, 47.8, 3.5, 10.1, 26.9, 14.3, 34.6, 39.6, 28.9, 14.8, 13.9, 1.3, 14.7, 20.9, 2.3, 45.1, 4.1, 8.6],
                "신문 광고비": [69.2, 23.5, 7.2, 53.4, 22.9, 7.4, 38.7, 49.9, 15.9, 54.7, 11.0, 89.4, 32.5, 51.4, 5.9, 21.4, 5.5, 25.6, 12.4, 8.7, 59.7, 38.9, 37.0, 24.3, 5.4, 47.4, 23.7, 19.6, 31.6, 8.7],
                "판매량": [22.1, 11.8, 9.7, 18.0, 15.9, 9.5, 17.1, 14.8, 20.2, 24.2, 18.9, 8.7, 11.3, 16.7, 11.7, 14.8, 19.8, 12.2, 15.2, 1.6, 20.8, 11.4, 16.1, 10.1, 11.9, 14.5, 11.8, 22.6, 5.9, 13.4],
            }
        ),
        "default_x": "TV 광고비",
        "default_y": "판매량",
        "story": "Kaggle의 Advertising 공개 데이터셋 표본 30개를 바탕으로 만든 실제 경제 데이터입니다. TV·라디오·신문 광고비와 판매량의 관계를 여러 각도에서 탐구할 수 있습니다.",
        "prompt": "예를 들어 TV 광고비가 커질수록 판매량이 늘어나는지, 신문 광고비와 판매량의 관계는 어떤지 비교해 볼 수 있습니다.",
        "source": "https://www.kaggle.com/datasets/purbar/advertising-data",
    },
    "의학: 건강과 의료비": {
        "table": pd.DataFrame(
            {
                "나이": [19, 18, 59, 54, 44, 42, 19, 34, 44, 19, 42, 21, 52, 52, 48, 47, 26, 50, 39, 23, 38, 21, 38, 24, 49, 43, 40, 18, 38, 61],
                "BMI": [27.9, 38.66, 29.83, 31.9, 30.69, 36.2, 20.62, 30.8, 32.02, 35.15, 30.0, 23.75, 31.73, 37.52, 30.78, 36.2, 23.7, 27.6, 21.85, 28.12, 31.0, 25.74, 27.6, 32.01, 29.92, 32.56, 29.36, 33.33, 19.95, 29.07],
                "자녀 수": [0, 2, 3, 3, 2, 1, 2, 0, 2, 0, 0, 2, 2, 2, 3, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 3, 1, 0, 2, 0],
                "의료비": [16884.92, 3393.36, 30184.94, 27322.73, 7731.43, 7443.64, 2803.7, 35491.64, 8116.27, 2134.9, 22144.03, 3077.1, 11187.66, 33471.97, 10141.14, 8068.18, 3484.33, 24520.26, 6117.49, 2690.11, 5488.26, 3279.87, 5383.54, 1981.58, 8988.16, 40941.29, 6393.6, 1135.94, 7133.9, 29141.36],
            }
        ),
        "default_x": "BMI",
        "default_y": "의료비",
        "story": "Kaggle의 Medical Cost Personal Datasets 공개 데이터셋 표본 30개를 바탕으로 만든 실제 건강 데이터입니다. 나이, BMI, 자녀 수, 의료비의 관계를 살펴볼 수 있습니다.",
        "prompt": "예를 들어 BMI가 높을수록 의료비가 어떻게 달라지는지, 나이나 자녀 수와 의료비의 관계는 어떤지 탐구할 수 있습니다.",
        "source": "https://www.kaggle.com/datasets/mirichoi0218/insurance",
    },
    "공학: 자동차 성능": {
        "table": pd.DataFrame(
            {
                "차량 무게": [3504, 3086, 4376, 4096, 1613, 4502, 2288, 4951, 2279, 2660, 4141, 2391, 3785, 2694, 4215, 1990, 2155, 4325, 1985, 3380, 2745, 3840, 3900, 2019, 2085, 2725, 2210, 2395, 1965, 2720],
                "마력": [130.0, 225.0, 200.0, 150.0, 69.0, 155.0, 92.0, 225.0, 88.0, 110.0, 140.0, 93.0, 95.0, 95.0, 152.0, 70.0, 80.0, 190.0, 48.0, 105.0, 105.0, 130.0, 125.0, 65.0, 48.0, 110.0, 75.0, 88.0, 67.0, 82.0],
                "가속 성능": [12.0, 10.0, 15.0, 13.0, 18.0, 13.5, 17.0, 11.0, 19.0, 14.0, 14.0, 15.5, 19.0, 15.0, 12.8, 17.0, 14.8, 12.2, 21.5, 15.8, 16.7, 15.4, 17.4, 16.4, 21.7, 12.6, 14.4, 18.0, 15.0, 19.4],
                "연비": [18.0, 14.0, 10.0, 14.0, 35.0, 13.0, 28.0, 12.0, 20.0, 24.0, 16.0, 26.0, 18.0, 23.0, 14.5, 32.0, 30.0, 15.5, 43.1, 20.6, 23.2, 17.0, 23.0, 37.2, 44.3, 23.5, 33.7, 34.0, 38.0, 31.0],
            }
        ),
        "default_x": "차량 무게",
        "default_y": "연비",
        "story": "Kaggle의 Auto MPG 공개 데이터셋 표본 30개를 바탕으로 만든 실제 공학 데이터입니다. 차량 무게, 마력, 가속 성능, 연비의 관계를 직접 고를 수 있습니다.",
        "prompt": "예를 들어 차량 무게가 무거워질수록 연비가 낮아지는지, 마력이나 가속 성능과 연비의 관계는 어떤지 비교해 볼 수 있습니다.",
        "source": "https://www.kaggle.com/datasets/uciml/autompg-dataset",
    },
    "환경: 델리 기후 변화": {
        "table": pd.DataFrame(
            {
                "평균 기온": [10.0, 17.71, 30.0, 32.0, 28.2, 31.0, 22.86, 16.12, 13.62, 24.12, 32.38, 36.0, 29.62, 25.0, 17.25, 13.75, 20.25, 35.43, 31.12, 28.62, 28.88, 18.5, 15.56, 27.31, 31.31, 36.0, 30.69, 30.04, 23.92, 10.0],
                "습도": [84.5, 74.71, 24.2, 54.0, 88.0, 57.5, 67.43, 89.12, 85.38, 46.75, 29.5, 37.12, 67.75, 47.67, 48.38, 88.12, 69.62, 13.43, 54.25, 82.62, 59.38, 63.0, 80.44, 44.25, 22.12, 43.31, 72.12, 67.63, 54.04, 100.0],
                "평균 풍속": [0.0, 5.81, 7.78, 13.44, 2.24, 42.22, 2.39, 3.25, 8.11, 13.9, 11.58, 13.65, 6.72, 25.01, 4.86, 0.92, 12.26, 15.34, 11.56, 1.85, 1.39, 1.62, 4.3, 6.14, 10.19, 12.85, 6.38, 8.11, 2.97, 0.0],
                "평균 기압": [1015.67, 1017.0, 1006.4, 998.75, 996.4, 1007.0, 1012.86, 1012.88, 1011.38, 1009.88, 1006.62, 996.5, 1003.75, 1013.62, 1014.75, 1019.88, 1013.5, 1003.57, 995.25, 1002.88, 1009.25, 1016.0, 1016.12, 1012.69, 1003.06, 998.19, 310.44, 1008.78, 1014.08, 1016.0],
            }
        ),
        "default_x": "평균 기온",
        "default_y": "습도",
        "story": "Kaggle의 Daily Climate Time Series Data 공개 데이터셋 표본 30개를 바탕으로 만든 실제 환경 데이터입니다. 평균 기온, 습도, 풍속, 기압의 변화를 연결해 볼 수 있습니다.",
        "prompt": "예를 들어 평균 기온이 높을 때 습도가 어떻게 달라지는지, 풍속이나 기압이 변할 때 다른 기후 요소가 함께 어떻게 움직이는지 탐구할 수 있습니다.",
        "source": "https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data",
    },
}

DATASETS.pop("환경: 델리 기후 변화", None)
DATASETS["환경: 대한민국 기후 변화"] = KOREA_CLIMATE_DATASET
DATASETS.update(EXTRA_DATASETS)


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


def show_styled_table(styler, height=210, column_config=None):
    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=column_config,
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


def render_link_button(url, label, gradient):
    st.markdown(
        f"""<a href="{url}" target="_blank"
           style="display:block;padding:11px;background:{gradient};color:white;text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-top:8px;">
           {label}
        </a>""",
        unsafe_allow_html=True,
    )


def render_portfolio_link(class_key):
    class_key = str(class_key)
    port_url = PORT_URLS.get(class_key)
    if port_url:
        render_link_button(
            port_url,
            f"{class_key}반 포트폴리오 패들렛 이동하기",
            "linear-gradient(90deg, #43a047 0%, #66bb6a 100%)",
        )
    else:
        st.info("반을 선택하면 포트폴리오 패들렛 버튼이 나타납니다.")


def _render_value_card(item):
    title = item.get("title", "")
    value = item.get("value", "")
    detail = item.get("detail", "")
    bg = item.get("bg", "#ffffff")
    border = item.get("border", "#dbe7f3")
    min_height = item.get("min_height", "auto")
    inner_html = item.get("inner_html")
    value_html = inner_html or (
        f'<div style="font-size:1.25rem; color:#263238; font-weight:700; margin-bottom:4px;">{value}</div>'
    )
    st.markdown(
        f"""
        <div style="
            height:100%;
            min-height:{min_height};
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            padding:14px 16px;
            border-radius:16px;
            background:{bg};
            border:1px solid {border};
            box-shadow:0 2px 8px rgba(33, 150, 243, 0.08);
            margin-bottom:8px;
        ">
            <div style="font-size:0.92rem; color:#546e7a; margin-bottom:6px; font-weight:600;">{title}</div>
            {value_html}
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


def render_summary_table(rows):
    summary_df = pd.DataFrame(
        [
            {
                "구분": row.get("label", ""),
                "결과": row.get("value", ""),
                "설명": row.get("note", ""),
            }
            for row in rows
        ]
    )
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "구분": st.column_config.TextColumn("구분", width="medium"),
            "결과": st.column_config.TextColumn("결과", width="medium"),
            "설명": st.column_config.TextColumn("설명", width="large"),
        },
    )


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


def selected_ml_display_name(degree):
    return f"{int(degree)}차 머신러닝"


def selected_ml_view_label(degree):
    return f"{selected_ml_display_name(degree)} 보기"


def display_model_name(model_name):
    if model_name == "직선 회귀":
        return "1차 머신러닝"
    if model_name == "2차 회귀":
        return "2차 머신러닝"
    return model_name


def selected_ml_latex(model_results, degree):
    return model_results["line_latex"] if int(degree) == 1 else model_results["quad_latex"]


def build_selected_comparison_df(model_results, degree):
    ml_name = selected_ml_name(degree)
    metric_df = model_results["metrics_df"].copy()
    filtered = metric_df[metric_df["모델"].isin([ml_name, "딥러닝"])].reset_index(drop=True)
    filtered["모델"] = filtered["모델"].map(display_model_name)
    return filtered


def prediction_input_bounds(x_values):
    x = np.asarray(x_values, dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_span = x_max - x_min
    padding = x_span * 0.2 if abs(x_span) > 1e-12 else 1.0
    return x_min - padding, x_max + padding


def prediction_line_x(split, points=220, extra_x=None):
    x_min = float(np.min(split["x_all"]))
    x_max = float(np.max(split["x_all"]))
    if extra_x is not None:
        x_min = min(x_min, float(extra_x))
        x_max = max(x_max, float(extra_x))
    if abs(x_max - x_min) < 1e-12:
        return np.array([x_min], dtype=float)
    return np.linspace(x_min, x_max, int(points))


def make_selected_model_compare_figure(dataset, split, model_results, degree):
    ml_name = selected_ml_name(degree)
    fig = Figure(figsize=(7.2, 4.4))
    ax = fig.subplots()
    x_line = prediction_line_x(split, 220)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#1976d2", edgecolors="white", linewidths=1.8, label="입력 데이터")
    ml_color = "#ff9800" if ml_name == "직선 회귀" else "#fb8c00"
    ax.plot(x_line, pred_map[ml_name], color=ml_color, linewidth=2.6, linestyle="--", label=display_model_name(ml_name))
    ax.plot(x_line, pred_map["딥러닝"], color="#43a047", linewidth=2.6, label="딥러닝")
    ax.axvline(split["x_hidden"], color="#90a4ae", linestyle="--", linewidth=1.2)
    ax.set_title(f"{dataset['x_label']}과 {dataset['y_label']}의 관계와 AI 예측")
    ax.set_xlabel(dataset["x_label"])
    ax.set_ylabel(dataset["y_label"])
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_selected_prediction_figure(
    dataset,
    split,
    model_results,
    degree,
    student_guess=None,
    reveal=False,
    show_ml=True,
    show_dl=True,
    prediction_x=None,
    actual_y=None,
):
    ml_name = selected_ml_name(degree)
    fig = Figure(figsize=(7.2, 4.4))
    ax = fig.subplots()
    x_line = prediction_line_x(split, 220, extra_x=prediction_x)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#1976d2", edgecolors="white", linewidths=1.8, label="입력 데이터")
    ml_color = "#ff9800" if ml_name == "직선 회귀" else "#fb8c00"
    x_hidden = float(split["x_hidden"] if prediction_x is None else prediction_x)
    model_preds = predict_models(model_results, np.array([x_hidden], dtype=float))
    ml_pred = float(model_preds[ml_name][0])
    dl_pred = float(model_preds["딥러닝"][0])
    if show_ml:
        ml_display_name = display_model_name(ml_name)
        ax.plot(x_line, pred_map[ml_name], color=ml_color, linewidth=2.4, linestyle="--", label=ml_display_name)
        ax.scatter([x_hidden], [ml_pred], color="#d32f2f", edgecolors="black", s=110, marker="o", zorder=5, label=f"{ml_display_name} 예측")
    if show_dl:
        ax.plot(x_line, pred_map["딥러닝"], color="#43a047", linewidth=2.4, label="딥러닝")
        ax.scatter([x_hidden], [dl_pred], color="#f06292", edgecolors="black", s=120, marker="X", zorder=5, label="딥러닝 예측")
    if student_guess is not None:
        ax.scatter([x_hidden], [student_guess], color="#1565c0", edgecolors="black", s=100, marker="D", zorder=5, label="내 예측")
    if reveal:
        actual = split["y_hidden"]
        ax.scatter([x_hidden], [actual], color="#111111", s=90, marker="*", zorder=6, label="실제값")
        if show_ml:
            ax.vlines(x_hidden - 0.05, min(ml_pred, actual), max(ml_pred, actual), color="#d32f2f", linestyle="--", linewidth=1.8)
        if show_dl:
            ax.vlines(x_hidden + 0.05, min(dl_pred, actual), max(dl_pred, actual), color="#f06292", linestyle="--", linewidth=1.8)
        if student_guess is not None:
            ax.vlines(x_hidden + 0.15, min(student_guess, actual), max(student_guess, actual), color="#1565c0", linestyle="--", linewidth=1.8)
    ax.set_title("입력값에 따른 예측값 비교")
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


def parse_direct_values(raw_text, label):
    values = []
    for item in str(raw_text).replace("\n", ",").split(","):
        token = item.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"입력 오류: {label} 자료에는 숫자만 입력해 주세요. 확인할 값: {token}") from exc
    return values


def format_direct_values(values):
    return ", ".join(f"{value:g}" for value in values)


def parse_direct_number_cell(raw_value, row_number, column_label):
    token = str(raw_value).strip().strip('"').replace("\u00a0", "")
    if not token:
        raise ValueError(f"{row_number}행 {column_label}열이 비어 있습니다.")
    try:
        return float(token.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"{row_number}행 {column_label}열에는 숫자만 입력해 주세요. 확인할 값: {token}") from exc


def can_parse_direct_number(raw_value):
    token = str(raw_value).strip().strip('"').replace("\u00a0", "")
    if not token:
        return False
    try:
        float(token.replace(",", ""))
        return True
    except ValueError:
        return False


def split_direct_table_row(line):
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t")]
    return next(csv.reader([line], skipinitialspace=True))


def parse_direct_table(raw_text):
    x_values = []
    y_values = []
    skipped_header = False
    for row_number, line in enumerate(str(raw_text).splitlines(), start=1):
        if not line.strip():
            continue
        cells = split_direct_table_row(line)
        if len(cells) < 2:
            raise ValueError(f"{row_number}행에는 x와 y 두 열이 필요합니다.")
        try:
            x_value = parse_direct_number_cell(cells[0], row_number, "x")
            y_value = parse_direct_number_cell(cells[1], row_number, "y")
        except ValueError:
            if (
                not x_values
                and not y_values
                and not skipped_header
                and not can_parse_direct_number(cells[0])
                and not can_parse_direct_number(cells[1])
            ):
                skipped_header = True
                continue
            raise
        x_values.append(x_value)
        y_values.append(y_value)
    if len(x_values) < 4:
        raise ValueError("분석하려면 x와 y 자료를 각각 4개 이상 붙여넣어 주세요.")
    return x_values, y_values


def apply_direct_excel_paste():
    raw_text = st.session_state.get("d5_direct_excel_paste", "")
    if not str(raw_text).strip():
        st.session_state["d5_direct_paste_status"] = ""
        st.session_state["d5_direct_paste_error"] = ""
        return
    try:
        x_values, y_values = parse_direct_table(raw_text)
    except ValueError as exc:
        st.session_state["d5_direct_paste_status"] = ""
        st.session_state["d5_direct_paste_error"] = f"엑셀 붙여넣기 오류: {exc}"
        return
    st.session_state["d5_direct_x_values"] = format_direct_values(x_values)
    st.session_state["d5_direct_y_values"] = format_direct_values(y_values)
    st.session_state["d5_direct_paste_status"] = f"엑셀 표에서 {len(x_values)}쌍의 x, y 자료를 자동 입력했습니다."
    st.session_state["d5_direct_paste_error"] = ""


def make_dataset(name, info, x_column, y_column):
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
        "source": info.get("source", ""),
    }


def current_dataset(name, x_column, y_column):
    return make_dataset(name, DATASETS[name], x_column, y_column)


def direct_dataset_from_session():
    x_values = parse_direct_values(st.session_state.get("d5_direct_x_values", ""), "x")
    y_values = parse_direct_values(st.session_state.get("d5_direct_y_values", ""), "y")
    if len(x_values) != len(y_values):
        raise ValueError("입력 오류: x의 자료 개수와 y의 자료 개수가 다릅니다. 쉼표(,)로 구분한 값의 개수를 맞춰 주세요.")
    if len(x_values) < 4:
        raise ValueError("입력 오류: 분석하려면 x와 y 자료를 각각 4개 이상 입력해 주세요.")
    table = pd.DataFrame({DIRECT_X_COLUMN: x_values, DIRECT_Y_COLUMN: y_values})
    info = {
        "table": table,
        "story": "학생이 직접 수집한 x, y 자료입니다. 쉼표(,)로 입력한 값을 바탕으로 행렬, 그래프, AI 예측 모델을 만듭니다.",
        "prompt": "예: 직접 조사한 공부 시간(x)과 점수(y), 운동 시간(x)과 심박수(y)처럼 두 변수가 어떤 관계를 보이는지 탐구할 수 있습니다.",
        "source": "",
    }
    return make_dataset(DIRECT_DATASET_NAME, info, DIRECT_X_COLUMN, DIRECT_Y_COLUMN)


def active_dataset_from_session():
    if st.session_state.get("d5_field") == DIRECT_FIELD:
        return direct_dataset_from_session()
    return current_dataset(
        st.session_state["d5_dataset"],
        st.session_state["d5_x_col"],
        st.session_state["d5_y_col"],
    )


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
def run_deep_learning(x_values, y_values, use_scale, hidden1=8, hidden2=4, epochs=30):
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


def get_model_results(x_obs, y_obs, use_scale, hidden1=8, hidden2=4, epochs=30):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    line_result = run_poly_regression(tuple(x_obs), tuple(y_obs), 1)
    quad_result = run_poly_regression(tuple(x_obs), tuple(y_obs), 2)
    dl_mask = iqr_inlier_mask(x_obs, y_obs)
    if int(np.sum(dl_mask)) < 4:
        dl_mask = np.ones_like(x_obs, dtype=bool)
    x_dl = x_obs[dl_mask]
    y_dl = y_obs[dl_mask]
    dl_result_raw = run_deep_learning(tuple(x_dl), tuple(y_dl), bool(use_scale), int(hidden1), int(hidden2), int(epochs))
    dl_result = {
        **dl_result_raw,
        "fit_count": int(len(x_dl)),
        "removed_count": int(len(x_obs) - len(x_dl)),
    }
    y_line = line_result["y_pred"]
    y_quad = quad_result["y_pred"]
    y_nn = nn_predict(dl_result, x_obs)

    rows = []
    for name, preds in [("직선 회귀", y_line), ("2차 회귀", y_quad), ("딥러닝", y_nn)]:
        rows.append(
            {
                "모델": name,
                "손실": round(sse(y_obs, preds), 3),
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


def iqr_inlier_mask(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    def _bounds(values):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        margin = iqr * 1.5
        return q1 - margin, q3 + margin

    x_low, x_high = _bounds(x)
    y_low, y_high = _bounds(y)
    return (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)


def make_preprocess_figure(dataset, split, use_scale):
    fig = Figure(figsize=(8.2, 3.7))
    axes = fig.subplots(1, 2)
    mask = iqr_inlier_mask(split["x_obs"], split["y_obs"])
    removed_count = int(np.size(mask) - np.sum(mask))

    axes[0].scatter(split["x_obs"][mask], split["y_obs"][mask], s=60, color="#1976d2", label="일반 데이터")
    if removed_count:
        axes[0].scatter(split["x_obs"][~mask], split["y_obs"][~mask], s=70, color="#e53935", label="이상치 후보")
    axes[0].set_title("원래 데이터")
    axes[0].set_xlabel(dataset["x_label"])
    axes[0].set_ylabel(dataset["y_label"])
    axes[0].grid(alpha=0.25)
    if removed_count:
        axes[0].legend(loc="best")

    axes[1].scatter(split["x_obs"][mask], split["y_obs"][mask], s=60, color="#43a047")
    axes[1].set_title("이상치 제거 후")
    axes[1].set_xlabel(dataset["x_label"])
    axes[1].set_ylabel(dataset["y_label"])
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig, removed_count


def make_model_compare_figure(dataset, split, model_results):
    fig = Figure(figsize=(7.2, 4.2))
    ax = fig.subplots()
    x_line = prediction_line_x(split, 220)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#263238", label="관찰 데이터")
    colors = {
        "직선 회귀": "#1e88e5",
        "2차 회귀": "#fb8c00",
        "딥러닝": "#d81b60",
    }
    for name, preds in pred_map.items():
        ax.plot(x_line, preds, color=colors[name], linewidth=2.5, label=display_model_name(name))
    ax.axvline(split["x_hidden"], color="#90a4ae", linestyle="--", linewidth=1.2)
    ax.set_title("세 모델의 예측 곡선 비교")
    ax.set_xlabel(f"{dataset['x_label']} ({dataset['x_unit']})")
    ax.set_ylabel(f"{dataset['y_label']} ({dataset['y_unit']})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_training_loss_figure(model_results):
    fig = Figure(figsize=(7.0, 3.6))
    ax = fig.subplots()
    losses = model_results["nn_model"]["losses"]
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#d81b60", linewidth=2.2)
    ax.set_title("딥러닝 학습 손실 변화")
    ax.set_xlabel("학습 횟수")
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


def build_error_styler(df, ml_error_col, dl_error_col):
    def highlight_lower_error(row):
        ml_error = float(row[ml_error_col])
        dl_error = float(row[dl_error_col])
        styles = [""] * len(row)
        ml_idx = row.index.get_loc(ml_error_col)
        dl_idx = row.index.get_loc(dl_error_col)
        if abs(ml_error - dl_error) < 1e-12:
            tie_style = "font-weight:700; border:2px solid #9e9e9e;"
            styles[ml_idx] = tie_style
            styles[dl_idx] = tie_style
        elif ml_error < dl_error:
            styles[ml_idx] = "font-weight:700; border:2px solid #1565c0;"
            styles[dl_idx] = "opacity:0.85;"
        else:
            styles[ml_idx] = "opacity:0.85;"
            styles[dl_idx] = "font-weight:700; border:2px solid #c2185b;"
        return styles

    return (
        df.style.format(precision=3)
        .background_gradient(subset=[ml_error_col], cmap="Blues")
        .background_gradient(subset=[dl_error_col], cmap="RdPu")
        .apply(highlight_lower_error, axis=1)
    )


def make_prediction_figure(dataset, split, model_results, student_guess=None, reveal=False):
    fig = Figure(figsize=(7.2, 4.2))
    ax = fig.subplots()
    x_line = prediction_line_x(split, 220)
    pred_map = predict_models(model_results, x_line)
    ax.scatter(split["x_obs"], split["y_obs"], s=70, color="#263238", label="관찰 데이터")
    ax.plot(x_line, pred_map["직선 회귀"], color="#1e88e5", linewidth=2.0, alpha=0.9, label="1차 머신러닝")
    ax.plot(x_line, pred_map["2차 회귀"], color="#fb8c00", linewidth=2.0, alpha=0.9, label="2차 머신러닝")
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
    best_row = metric_df.loc[metric_df["손실"].idxmin()]
    hidden_preds = predict_models(model_results, np.array([split["x_hidden"]], dtype=float))
    line_pred = float(hidden_preds["직선 회귀"][0])
    quad_pred = float(hidden_preds["2차 회귀"][0])
    dl_pred = float(hidden_preds["딥러닝"][0])
    outlier_removed = int(len(split["x_obs"]) - np.sum(iqr_inlier_mask(split["x_obs"], split["y_obs"])))
    final_text = "실제값을 아직 확인하지 않았습니다."
    if reveal:
        final_text = (
            f"실제값은 {split['y_hidden']:.2f}{dataset['y_unit']}이고, "
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
            f"IQR 기준으로 이상치 후보 {outlier_removed}개를 확인했고, 이상치를 제거했을 때 그래프의 흐름이 어떻게 달라지는지 비교했다."
            if outlier_removed
            else "IQR 기준으로 뚜렷한 이상치 후보가 거의 없어서, 원래 자료의 흐름이 비교적 안정적이라는 점을 확인했다.",
        ),
        (
            "문제 3. 모델 비교하기",
            f"1차 머신러닝 예측값은 {line_pred:.2f}, 2차 머신러닝 예측값은 {quad_pred:.2f}, 딥러닝 예측값은 {dl_pred:.2f}이다. "
            f"손실 기준으로 가장 잘 맞는 모델은 {display_model_name(best_row['모델'])}이다.",
        ),
        (
            "문제 4. 예측 결과 해석하기",
            f"딥러닝 예측값은 {dl_pred:.2f}{dataset['y_unit']}이다. {final_text}",
        ),
    ]


def build_auto_report_texts(dataset, model_results, active_ml_name, prediction_x, ml_pred, dl_pred, best_model, deep_question, research_motivation):
    question_text = clean_text(deep_question, "선택한 데이터에서 독립 변수가 종속 변수에 어떤 영향을 주는지 확인하고 싶었다.")
    motivation_text = clean_text(research_motivation, "실생활 데이터를 이용해 AI 예측 결과를 해석해 보고 싶었다.")
    architecture = model_results["nn_model"]["architecture"]
    analysis_text = (
        f"{dataset['name']} 데이터를 사용해 {dataset['x_label']} 값을 입력하면 {dataset['y_label']} 값을 예측하도록 모델을 만들었다. "
        f"이번 예측 입력값은 {prediction_x:.3f}이며, {active_ml_name} 예측값은 {ml_pred:.3f}, "
        f"딥러닝 예측값은 {dl_pred:.3f}이다. 손실 기준으로 가장 잘 맞은 모델은 {best_model}이다."
    )
    interpretation_text = (
        f"깊은 질문은 '{question_text}'이고, 탐구 동기는 '{motivation_text}'이다. "
        f"두 모델의 예측값을 비교하면 같은 입력값에서도 모델 구조에 따라 결과가 달라질 수 있음을 확인할 수 있다. "
        f"딥러닝은 {architecture} 구조로 학습했으며, 예측 결과는 데이터 범위와 표본 수의 영향을 함께 고려해 해석해야 한다."
    )
    return analysis_text, interpretation_text


def create_portfolio_pdf(
    student_info,
    dataset,
    split,
    model_results,
    ml_degree,
    use_scale,
    prediction_x,
    student_guess,
    reveal,
    deep_question,
    research_motivation,
    analysis_text,
    interpretation_text,
    figure_items,
):
    pdf = ThemedPDF()
    pdf.add_font("Nanum", "", font_path, uni=True)
    pdf.set_font("Nanum", "", 12)
    pdf._font_family = "Nanum"
    pdf.footer_left = f"{student_info.get('group', '')}"
    pdf.add_page()

    kvs = [
        ("모둠명", student_info.get("group", "")),
        ("작성일", datetime.datetime.now().strftime("%Y-%m-%d")),
    ]
    pdf.kv_card("모둠 정보", kvs)

    active_ml_name = selected_ml_name(ml_degree)
    prediction_x = float(prediction_x)
    hidden_preds = predict_models(model_results, np.array([prediction_x], dtype=float))
    ml_pred = float(hidden_preds[active_ml_name][0])
    dl_pred = float(hidden_preds["딥러닝"][0])
    active_ml_display_name = selected_ml_display_name(ml_degree)
    best_model = display_model_name(model_results["metrics_df"].loc[model_results["metrics_df"]["손실"].idxmin(), "모델"])
    auto_analysis_text, _ = build_auto_report_texts(
        dataset,
        model_results,
        active_ml_display_name,
        prediction_x,
        ml_pred,
        dl_pred,
        best_model,
        deep_question,
        research_motivation,
    )
    analysis_text = clean_text(analysis_text, auto_analysis_text)
    interpretation_text = clean_text(interpretation_text, "학생이 직접 작성하지 않았습니다.")

    summary_kvs = [
        ("활동 데이터", dataset["name"]),
        ("변수 관계", f"{dataset['x_label']} -> {dataset['y_label']}"),
        ("입력값", f"{prediction_x:.3f}"),
        ("머신러닝", active_ml_display_name),
        ("머신러닝 예측", f"{ml_pred:.3f}"),
        ("딥러닝 예측", f"{dl_pred:.3f}"),
        ("가장 잘 맞은 모델", best_model),
        ("딥러닝 구조", f"{model_results['nn_model']['architecture']} / {'정규화 적용' if use_scale else '정규화 미적용'}"),
    ]
    pdf.kv_card("핵심 요약", summary_kvs)

    pdf.h2("탐구 질문과 동기")
    add_text_box_to_pdf(pdf, "깊은 질문", clean_text(deep_question, "아직 작성하지 않았습니다."))
    add_text_box_to_pdf(pdf, "탐구 동기", clean_text(research_motivation, "아직 작성하지 않았습니다."))

    pdf.h2("보고서 요약")
    add_text_box_to_pdf(pdf, "데이터 분석 및 예측 결과", clean_text(analysis_text))
    add_text_box_to_pdf(pdf, "연구 결과 및 해석", clean_text(interpretation_text))

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
    st.session_state.setdefault("d5_class", CLASS_OPTIONS[0])
    if st.session_state.get("d5_class") not in CLASS_OPTIONS:
        st.session_state["d5_class"] = CLASS_OPTIONS[0]
    st.session_state.setdefault("d5_use_scale", True)
    st.session_state.setdefault("d5_dataset", dataset_names[0])
    st.session_state["d5_dataset"] = normalize_dataset_name(st.session_state.get("d5_dataset", dataset_names[0]))
    st.session_state.setdefault("d5_field", field_for_dataset(st.session_state["d5_dataset"]))
    if st.session_state.get("d5_field") not in DIRECT_FIELD_OPTIONS:
        st.session_state["d5_field"] = field_for_dataset(st.session_state["d5_dataset"])
    if st.session_state["d5_field"] == DIRECT_FIELD:
        st.session_state["d5_dataset"] = DIRECT_DATASET_NAME
    elif st.session_state["d5_dataset"] not in DATASETS:
        st.session_state["d5_dataset"] = FIELD_DATASETS[st.session_state["d5_field"]][0]
    if st.session_state["d5_field"] != DIRECT_FIELD and st.session_state["d5_dataset"] not in FIELD_DATASETS[st.session_state["d5_field"]]:
        st.session_state["d5_dataset"] = FIELD_DATASETS[st.session_state["d5_field"]][0]
    st.session_state.setdefault("d5_direct_x_values", DIRECT_DEFAULT_X_VALUES)
    st.session_state.setdefault("d5_direct_y_values", DIRECT_DEFAULT_Y_VALUES)
    st.session_state.setdefault("d5_direct_excel_paste", "")
    st.session_state.setdefault("d5_direct_paste_status", "")
    st.session_state.setdefault("d5_direct_paste_error", "")
    st.session_state.setdefault("d5_deep_question", "")
    st.session_state.setdefault("d5_research_motivation", "")
    st.session_state.setdefault("d5_ml_degree", 1)
    st.session_state.setdefault("d5_show_prediction_ml", True)
    st.session_state.setdefault("d5_show_prediction_dl", True)
    st.session_state.setdefault("d5_hidden1", 8)
    st.session_state.setdefault("d5_hidden2", 4)
    if st.session_state.get("d5_hidden1", 8) < 4 or st.session_state.get("d5_hidden1", 8) > 12:
        st.session_state["d5_hidden1"] = 8
    if st.session_state.get("d5_hidden2", 4) < 2 or st.session_state.get("d5_hidden2", 4) > 8:
        st.session_state["d5_hidden2"] = 4
    st.session_state.setdefault("d5_epochs", 15)
    if st.session_state.get("d5_epochs", 30) > 30:
        st.session_state["d5_epochs"] = 30

    tabs = st.tabs(
        [
            "1️⃣ [F.U] 데이터 선택",
            "2️⃣ [T] 자료를 행렬로 보기",
            "3️⃣ [U] 머신러닝 vs 딥러닝",
            "4️⃣ [R] 예측 및 시각화",
            "5️⃣ [E] 보고서 작성 및 저장",
        ]
    )

    with tabs[0]:
        stage_intro(
            "모둠과 분석 데이터 선택",
            "학생의 흥미에 맞는 분야 데이터를 고르고, 독립 변수와 종속 변수를 직접 정해 분석 방향을 여는 단계입니다.",
            "어떤 데이터를 고르고, 무엇을 원인과 결과로 정하면 좋은 예측 질문을 만들 수 있을까?",
            "#e3f2fd",
            "#bbdefb",
        )
        st.markdown(pretty_title("모둠과 데이터 분석 방향 정하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        class_col, input_col = st.columns([0.45, 1.0])
        with class_col:
            st.selectbox("반", CLASS_OPTIONS, key="d5_class")
        with input_col:
            st.text_input("모둠명", key="d5_group", placeholder="예: 1모둠")

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#1565c0; "
                "margin-bottom:6px;'>1️⃣ 데이터 분석 방향 선택</div>",
                unsafe_allow_html=True,
            )
            field_col, dataset_col = st.columns([0.8, 1.2])
            with field_col:
                st.selectbox("활동 분야 선택", DIRECT_FIELD_OPTIONS, key="d5_field")

            if st.session_state["d5_field"] == DIRECT_FIELD:
                st.session_state["d5_dataset"] = DIRECT_DATASET_NAME
                with dataset_col:
                    st.info("엑셀에서 x, y 두 열을 복사해 붙여넣거나, 쉼표(,)로 구분한 자료를 직접 입력해 분석합니다.")
                st.markdown(
                    "<div style='background:#f1f8e9; border-left:6px solid #43a047; "
                    "border-radius:8px; padding:8px 12px; font-weight:800; color:#1b5e20; "
                    "margin-bottom:6px;'>엑셀 표 붙여넣기</div>",
                    unsafe_allow_html=True,
                )
                st.text_area(
                    "엑셀 표 붙여넣기",
                    key="d5_direct_excel_paste",
                    height=92,
                    label_visibility="collapsed",
                    placeholder="엑셀에서 x, y 두 열을 복사한 뒤 여기에 붙여넣으세요.\n예:\n1\t2\n2\t4\n3\t6\n4\t8",
                    on_change=apply_direct_excel_paste,
                )
                if st.session_state.get("d5_direct_paste_status"):
                    st.success(st.session_state["d5_direct_paste_status"])
                if st.session_state.get("d5_direct_paste_error"):
                    st.error(st.session_state["d5_direct_paste_error"])
                x_col, y_col = st.columns(2)
                with x_col:
                    st.markdown(
                        "<div style='background:#e3f2fd; border-left:6px solid #1976d2; "
                        "border-radius:8px; padding:8px 12px; font-weight:800; color:#0d47a1; "
                        "margin-bottom:6px;'>독립 변수 자료 입력(x)</div>",
                        unsafe_allow_html=True,
                    )
                    st.text_area(
                        "독립 변수 자료 입력(x)",
                        key="d5_direct_x_values",
                        height=96,
                        label_visibility="collapsed",
                        placeholder="예: 1, 2, 3, 4, 5",
                    )
                with y_col:
                    st.markdown(
                        "<div style='background:#fce4ec; border-left:6px solid #d81b60; "
                        "border-radius:8px; padding:8px 12px; font-weight:800; color:#880e4f; "
                        "margin-bottom:6px;'>종속 변수 자료 입력(y)</div>",
                        unsafe_allow_html=True,
                    )
                    st.text_area(
                        "종속 변수 자료 입력(y)",
                        key="d5_direct_y_values",
                        height=96,
                        label_visibility="collapsed",
                        placeholder="예: 2, 4, 6, 8, 10",
                    )
                st.caption("엑셀 붙여넣기는 첫 번째 열을 x, 두 번째 열을 y로 자동 입력합니다. 제목 행이 있으면 자동으로 건너뜁니다. x와 y의 자료 개수가 같아야 분석할 수 있습니다.")
            else:
                field_dataset_names = FIELD_DATASETS[st.session_state["d5_field"]]
                if st.session_state.get("d5_dataset") not in field_dataset_names:
                    st.session_state["d5_dataset"] = field_dataset_names[0]
                with dataset_col:
                    st.selectbox("활동 데이터셋 선택", field_dataset_names, key="d5_dataset")

                dataset_info = DATASETS[st.session_state["d5_dataset"]]
                all_columns = list(dataset_info["table"].columns)
                if st.session_state.get("d5_x_col") not in all_columns:
                    st.session_state["d5_x_col"] = dataset_info["default_x"]

                x_col, y_col = st.columns(2)
                with x_col:
                    st.markdown(
                        "<div style='background:#e3f2fd; border-left:6px solid #1976d2; "
                        "border-radius:8px; padding:8px 12px; font-weight:800; color:#0d47a1; "
                        "margin-bottom:6px;'>독립 변수 선택(x)</div>",
                        unsafe_allow_html=True,
                    )
                    st.selectbox("독립 변수 선택(x)", all_columns, key="d5_x_col", label_visibility="collapsed")

                valid_y_options = [col for col in all_columns if col != st.session_state["d5_x_col"]]
                preferred_y = dataset_info["default_y"] if dataset_info["default_y"] in valid_y_options else valid_y_options[0]
                if st.session_state.get("d5_y_col") not in valid_y_options:
                    st.session_state["d5_y_col"] = preferred_y
                with y_col:
                    st.markdown(
                        "<div style='background:#fce4ec; border-left:6px solid #d81b60; "
                        "border-radius:8px; padding:8px 12px; font-weight:800; color:#880e4f; "
                        "margin-bottom:6px;'>종속 변수 선택(y)</div>",
                        unsafe_allow_html=True,
                    )
                    st.selectbox("종속 변수 선택(y)", valid_y_options, key="d5_y_col", label_visibility="collapsed")
                st.caption("각 분야마다 실제 데이터셋 2개 중 하나를 고르고, 학생이 이해하기 쉬운 4~5개 안팎의 변수 중에서 독립 변수와 종속 변수를 직접 선택할 수 있습니다. 시간 변수가 있는 자료는 시간 흐름 자체도 함께 탐구할 수 있습니다.")
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#4a148c; "
                "margin-bottom:6px;'>2️⃣보고서의 출발점 정하기</div>",
                unsafe_allow_html=True,
            )
            st.caption("깊은 질문은 보고서의 주제, 탐구 동기는 이 분석을 하려는 이유입니다.")
            question_col, motivation_col = st.columns(2)
            with question_col:
                st.text_area(
                    "깊은 질문",
                    key="d5_deep_question",
                    height=115,
                    placeholder="예: TV광고비가 늘어나면 판매량은 얼마나 증가할까?",
                )
            with motivation_col:
                st.text_area(
                    "탐구 동기",
                    key="d5_research_motivation",
                    height=115,
                    placeholder="예: 광고가 실제 판매량에 어떤 영향을 주는지 데이터로 확인해 보고 싶었다.",
                )

        try:
            tab_dataset = active_dataset_from_session()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        selected_preview = tab_dataset["table"].copy()
        selected_preview.insert(0, "행 번호", np.arange(1, len(selected_preview) + 1))
        st.markdown(pretty_title("선택한 자료 미리 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        left, right = st.columns([1.15, 1.0])
        with left:
            show_pretty_table(selected_preview, height=280)
        with right:
            st.info(tab_dataset["story"])
            st.markdown(f"**탐구 힌트**: {tab_dataset['prompt']}")
            if tab_dataset["source"]:
                st.markdown(f"**데이터 출처**: [공개 데이터셋 바로가기]({tab_dataset['source']})")
            else:
                st.markdown("**데이터 출처**: 직접 자료 수집")
        dataset = tab_dataset
        split = dataset_split(dataset)
        default_prediction_x = float(np.median(dataset["x"]))
        if st.session_state["d5_field"] == DIRECT_FIELD:
            selection_signature = (
                DIRECT_DATASET_NAME,
                st.session_state.get("d5_direct_x_values", ""),
                st.session_state.get("d5_direct_y_values", ""),
            )
        else:
            selection_signature = (
                st.session_state["d5_dataset"],
                st.session_state["d5_x_col"],
                st.session_state["d5_y_col"],
            )
        if st.session_state.get("d5_last_selection") != selection_signature:
            st.session_state["d5_last_selection"] = selection_signature
            st.session_state["d5_prediction_revealed"] = False
            st.session_state["d5_student_guess"] = float(split["y_obs"][-1])
            st.session_state["d5_prediction_x"] = default_prediction_x
        st.session_state.setdefault("d5_student_guess", float(split["y_obs"][-1]))
        st.session_state.setdefault("d5_prediction_x", default_prediction_x)

        observation_fig = make_observation_figure(dataset, split)

    dataset = active_dataset_from_session()
    split = dataset_split(dataset)
    default_prediction_x = float(np.median(dataset["x"]))
    st.session_state.setdefault("d5_student_guess", float(split["y_obs"][-1]))
    st.session_state.setdefault("d5_prediction_x", default_prediction_x)
    selected_matrix = dataset["selected_table"].copy()
    matrix_df = pd.DataFrame(
        selected_matrix.to_numpy(dtype=float),
        index=[f"행{i}" for i in range(1, len(selected_matrix) + 1)],
        columns=[f"열1: {dataset['x_label']}", f"열2: {dataset['y_label']}"],
    )
    first_mean = round(float(np.mean(dataset["x"])), 3)
    second_mean = round(float(np.mean(dataset["y"])), 3)
    corr_value = round(float(np.corrcoef(dataset["x"], dataset["y"])[0, 1]), 3)
    first_min = round(float(np.min(dataset["x"])), 3)
    second_min = round(float(np.min(dataset["y"])), 3)
    first_max = round(float(np.max(dataset["x"])), 3)
    second_max = round(float(np.max(dataset["y"])), 3)
    corr_abs = abs(corr_value)
    if corr_abs >= 0.7:
        corr_strength = "강한 관계"
    elif corr_abs >= 0.4:
        corr_strength = "중간 정도 관계"
    elif corr_abs >= 0.2:
        corr_strength = "약한 관계"
    else:
        corr_strength = "거의 약한 관계"
    corr_direction = "같은 방향" if corr_value > 0 else "반대 방향" if corr_value < 0 else "방향 없음"
    summary_df = pd.DataFrame(
        {
            "항목": [
                f"{dataset['x_label']} 평균",
                f"{dataset['y_label']} 평균",
                "두 변수의 상관계수",
                f"{dataset['x_label']} 최솟값",
                f"{dataset['y_label']} 최솟값",
                f"{dataset['x_label']} 최댓값",
                f"{dataset['y_label']} 최댓값"
            ],
            "값": [
                first_mean,
                second_mean,
                corr_value,
                first_min,
                second_min,
                first_max,
                second_max,
            ],
        }
    )

    def highlight_summary_row(row):
        if row["항목"] == "두 변수의 상관계수":
            return ["background-color:#fff3cd; color:#5d4037; font-weight:800; border:2px solid #ffb300;"] * len(row)
        return ["background-color:#f8fbff;" if row.name % 2 == 0 else "background-color:#ffffff;"] * len(row)

    summary_styler = summary_df.style.format({"값": "{:.3f}"}).apply(highlight_summary_row, axis=1)
    preprocess_fig, outlier_count = make_preprocess_figure(dataset, split, st.session_state["d5_use_scale"])

    with tabs[1]:
        stage_intro(
            "선택한 자료를 행렬로 이해하기",
            "선택한 두 변수를 행과 열로 정리해 보고, 행렬 관점에서 데이터를 읽는 단계입니다.",
            "선택한 두 변수의 값을 표로 정리하면 어떤 관계를 발견할 수 있을까?",
            "#fff8e1",
            "#ffecb3",
        )
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#1565c0; "
                "margin-bottom:6px;'>3️⃣ 자료를 행렬로 보기</div>",
                unsafe_allow_html=True,
            )
            left, right = st.columns([1.3, 0.7])
            with left:
                matrix_display = matrix_df.reset_index().rename(columns={"index": "행"})
                show_pretty_table(matrix_display, height=360)
                st.caption("독립 변수나 종속 변수를 다시 선택하면 표의 값도 함께 바뀝니다.")
            with right:
                render_value_cards(
                    [
                        {
                            "title": "상관계수",
                            "value": f"{corr_value:.3f}",
                            "detail": f"{corr_direction}으로 함께 변하는 {corr_strength}입니다.",
                            "bg": "#fff8e1",
                            "border": "#ffb300",
                            "min_height": "132px",
                        },
                        {
                            "title": dataset["x_label"],
                            "value": f"평균 {first_mean:.3f}",
                            "detail": f"자료 값 범위: {first_min:.3f} ~ {first_max:.3f}",
                            "bg": "#e3f2fd",
                            "border": "#64b5f6",
                            "min_height": "132px",
                        },
                        {
                            "title": dataset["y_label"],
                            "value": f"평균 {second_mean:.3f}",
                            "detail": f"자료 값 범위: {second_min:.3f} ~ {second_max:.3f}",
                            "bg": "#fce4ec",
                            "border": "#f48fb1",
                            "min_height": "132px",
                        },
                    ],
                    columns=1,
                )
        st.info("AI는 이런 숫자 배열(행렬)에서 관계를 읽고, 그 안의 규칙을 바탕으로 예측 모델을 만듭니다.")

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#2e7d32; "
                "margin-bottom:6px;'>4️⃣ 자료 정리하기(전처리로 이상치 제거하기)</div>",
                unsafe_allow_html=True,
            )
            preprocess_col, outlier_col = st.columns(2)
            with preprocess_col:
                st.info(
                    "**전처리란?**\n\n"
                    "AI가 데이터를 잘 읽도록 분석 전에 자료를 정리하는 과정입니다."
                )
            with outlier_col:
                st.warning(
                    "**이상치 처리란?**\n\n"
                    "자료 흐름에서 너무 멀리 떨어진 값을 찾아 확인하는 과정입니다."
                )
            st.caption("여기서는 IQR 기준으로 이상치 후보를 찾고, 제거하기 전과 후의 그래프 흐름을 비교합니다.")
            st.pyplot(preprocess_fig, use_container_width=True)

    with tabs[2]:
        stage_intro(
            "머신러닝 vs 딥러닝",
            "선택한 데이터에 대해 1차/2차 머신러닝과 딥러닝을 함께 실험하며 어떤 모델이 더 잘 맞는지 비교하는 단계입니다.",
            "머신러닝과 딥러닝은 같은 데이터를 보고 어떤 예측값을 만들며, 두 결과는 어떻게 다를까?",
            "#e8f5e9",
            "#c8e6c9",
        )
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#2e7d32; "
                "margin-bottom:6px;'>5️⃣ AI 모델 구성 및 학습</div>",
                unsafe_allow_html=True,
            )
            ml_col, dl_col = st.columns(2)
            with ml_col:
                st.markdown(pretty_title("머신러닝 분석", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
                st.info("머신러닝은 자료를 보고 스스로 함수식을 찾아 예측합니다. 1차와 2차 함수까지만 분석합니다.")
                st.selectbox(
                    "머신러닝 모델 선택",
                    options=[1, 2],
                    format_func=lambda degree: "직선(1차)" if int(degree) == 1 else "곡선(2차)",
                    key="d5_ml_degree",
                )
            with dl_col:
                st.markdown(pretty_title("딥러닝 모델", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
                st.info("딥러닝은 여러 층을 거치며 패턴을 읽습니다. 뉴런 수는 AI가 생각할 때 쓰는 작은 계산 칸의 개수입니다. 학습 횟수는 같은 자료를 몇 번 반복해서 보며 연습할지 정하는 값입니다.")
                inner1, inner2, inner3 = st.columns(3)
                with inner1:
                    st.slider("1층 뉴런 수", 4, 12, key="d5_hidden1")
                with inner2:
                    st.slider("2층 뉴런 수", 2, 8, key="d5_hidden2")
                with inner3:
                    st.slider("학습 횟수", 10, 30, step=5, key="d5_epochs")

        model_results = get_model_results(
            split["x_obs"],
            split["y_obs"],
            st.session_state["d5_use_scale"],
            st.session_state["d5_hidden1"],
            st.session_state["d5_hidden2"],
            st.session_state["d5_epochs"],
        )
        st.markdown(pretty_title("딥러닝 구조와 학습 변화 보기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.caption(
            f"뉴런 수를 바꾸면 구조 그림의 동그라미 수가 바로 바뀌고, 학습 횟수에 따라 손실 변화 그래프도 함께 달라집니다. "
            f"현재는 관찰 데이터 {len(split['x_obs'])}개 중 이상치 후보 {model_results['nn_model']['removed_count']}개를 제외한 "
            f"{model_results['nn_model']['fit_count']}개로 딥러닝을 학습합니다."
        )
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
        active_ml_display_name = selected_ml_display_name(st.session_state["d5_ml_degree"])
        metrics_df = model_results["metrics_df"]
        ml_loss = float(metrics_df.loc[metrics_df["모델"] == active_ml_name, "손실"].iloc[0])
        dl_loss = float(metrics_df.loc[metrics_df["모델"] == "딥러닝", "손실"].iloc[0])
        ml_formula_text = selected_ml_latex(model_results, st.session_state["d5_ml_degree"]).replace("{", "").replace("}", "")
        dl_architecture = model_results["nn_model"]["architecture"]

        st.markdown(pretty_title("모델 비교 결과 보기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        formula_col, structure_col = st.columns(2)
        with formula_col:
            st.markdown(
                f"""
                <div style='
                    min-height:238px;
                    display:flex;
                    flex-direction:column;
                    justify-content:space-between;
                    background:#f4f9ff;
                    border:1px solid #90caf9;
                    border-left:6px solid #1976d2;
                    border-radius:12px;
                    padding:16px 18px;
                '>
                    <div>
                        <div style='font-size:0.92rem; color:#1565c0; font-weight:800; margin-bottom:8px;'>머신러닝</div>
                        <div style='font-size:1.25rem; color:#263238; font-weight:900; line-height:1.35;'>{ml_formula_text}</div>
                        <div style='font-size:0.84rem; color:#546e7a; margin-top:8px;'>현재 선택한 모델: {active_ml_display_name}</div>
                    </div>
                    <div style='background:#ffffff; border:1px solid #bbdefb; border-radius:10px; padding:10px 12px; margin-top:14px;'>
                        <div style='font-size:0.86rem; color:#1565c0; font-weight:800; margin-bottom:4px;'>손실</div>
                        <div style='font-size:1.45rem; color:#263238; font-weight:900;'>{ml_loss:.3f}</div>
                        <div style='font-size:0.82rem; color:#546e7a; margin-top:4px;'>작을수록 실제 자료에 더 가깝습니다.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with structure_col:
            st.markdown(
                f"""
                <div style='
                    min-height:238px;
                    display:flex;
                    flex-direction:column;
                    justify-content:space-between;
                    background:#fff8e1;
                    border:1px solid #ffcc80;
                    border-left:6px solid #fb8c00;
                    border-radius:12px;
                    padding:16px 18px;
                '>
                    <div>
                        <div style='font-size:0.92rem; color:#ef6c00; font-weight:800; margin-bottom:8px;'>딥러닝</div>
                        <div style='font-size:1.85rem; color:#263238; font-weight:900; line-height:1.2;'>{dl_architecture}</div>
                        <div style='font-size:0.84rem; color:#546e7a; margin-top:8px;'>입력-1층-2층-출력 순서입니다.</div>
                    </div>
                    <div style='background:#ffffff; border:1px solid #ffcc80; border-radius:10px; padding:10px 12px; margin-top:14px;'>
                        <div style='font-size:0.86rem; color:#ef6c00; font-weight:800; margin-bottom:4px;'>손실</div>
                        <div style='font-size:1.45rem; color:#263238; font-weight:900;'>{dl_loss:.3f}</div>
                        <div style='font-size:0.82rem; color:#546e7a; margin-top:4px;'>작을수록 실제 자료에 더 가깝습니다.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(pretty_title("실제값과 예측값 오차 비교", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        actual_label = f"{dataset['y_label']} 실제값"
        ml_error_col = "머신러닝 오차"
        dl_error_col = "딥러닝 오차"
        error_df = pd.DataFrame(
            {
                dataset["x_label"]: np.round(split["x_obs"], 3),
                actual_label: np.round(split["y_obs"], 3),
                "머신러닝 예측값": np.round(model_results["train_preds"][active_ml_name], 3),
                "딥러닝 예측값": np.round(model_results["train_preds"]["딥러닝"], 3),
            }
        )
        error_df[ml_error_col] = np.round(np.abs(split["y_obs"] - model_results["train_preds"][active_ml_name]), 3)
        error_df[dl_error_col] = np.round(np.abs(split["y_obs"] - model_results["train_preds"]["딥러닝"]), 3)
        st.caption("파란색은 1차/2차 머신러닝 모델의 오차, 분홍색은 딥러닝 오차입니다. 색이 진할수록 오차가 더 큽니다.")
        equal_width_columns = {
            column: st.column_config.NumberColumn(width="small")
            for column in error_df.columns
        }
        show_styled_table(
            build_error_styler(error_df, ml_error_col, dl_error_col),
            height=250,
            column_config=equal_width_columns,
        )

    with tabs[3]:
        stage_intro(
            "예측 및 시각화",
            "입력값을 직접 넣고, 머신러닝과 딥러닝 모델이 계산한 예측값을 확인하는 단계입니다.",
            "같은 입력값에 대해 머신러닝과 딥러닝은 어떤 예측값을 만들까?",
            "#f3e5f5",
            "#e1bee7",
        )        
        st.markdown(pretty_title("예측 그래프 확인", "#f3e5f5", "#e1bee7"), unsafe_allow_html=True)

        model_results = get_model_results(
            split["x_obs"],
            split["y_obs"],
            st.session_state["d5_use_scale"],
            st.session_state["d5_hidden1"],
            st.session_state["d5_hidden2"],
            st.session_state["d5_epochs"],
        )
        active_ml_name = selected_ml_name(st.session_state["d5_ml_degree"])
        active_ml_display_name = selected_ml_display_name(st.session_state["d5_ml_degree"])
        visibility_col1, visibility_col2 = st.columns(2)
        with visibility_col1:
            st.checkbox(selected_ml_view_label(st.session_state["d5_ml_degree"]), key="d5_show_prediction_ml")
        with visibility_col2:
            st.checkbox("딥러닝 보기", key="d5_show_prediction_dl")
        prediction_min_x, prediction_max_x = prediction_input_bounds(dataset["x"])
        prediction_x = float(st.session_state.get("d5_prediction_x", default_prediction_x))
        prediction_x = min(max(prediction_x, prediction_min_x), prediction_max_x)
        st.session_state["d5_prediction_x"] = prediction_x
        prediction_fig = make_selected_prediction_figure(
            dataset,
            split,
            model_results,
            st.session_state["d5_ml_degree"],
            student_guess=None,
            reveal=False,
            show_ml=bool(st.session_state.get("d5_show_prediction_ml", True)),
            show_dl=bool(st.session_state.get("d5_show_prediction_dl", True)),
            prediction_x=prediction_x,
        )
        hidden_preds = predict_models(model_results, np.array([prediction_x], dtype=float))
        ml_pred = float(hidden_preds[active_ml_name][0])
        dl_pred = float(hidden_preds["딥러닝"][0])

        st.pyplot(prediction_fig, use_container_width=True)

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#6a1b9a; "
                "margin-bottom:6px;'>6️⃣ AI 예측</div>",
                unsafe_allow_html=True,
            )
            action_col, result_col = st.columns(2)
            with action_col:
                st.markdown(pretty_title("예측 입력값", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
                st.markdown(
                    """
                    <style>
                    div[data-testid="stNumberInput"] {
                        border: 3px solid #d81b60;
                        border-radius: 14px;
                        background: #fff5f8;
                        padding: 12px 14px 14px 14px;
                        box-shadow: 0 8px 18px rgba(216, 27, 96, 0.14);
                    }
                    div[data-testid="stNumberInput"] label p {
                        color: #ad1457;
                        font-size: 1.05rem;
                        font-weight: 800;
                    }
                    div[data-testid="stNumberInput"] input {
                        background: #ffffff;
                        color: #263238;
                        font-size: 1.18rem;
                        font-weight: 800;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.info(
                    f"{dataset['x_label']} 값을 직접 입력하면, 모델이 {dataset['y_label']} 값을 예측합니다."
                )
                prediction_x = st.number_input(
                    f"예측할 {dataset['x_label']} 입력",
                    key="d5_prediction_x",
                    min_value=prediction_min_x,
                    max_value=prediction_max_x,
                    format="%.3f",
                )
                st.caption(
                    f"기본값은 선택한 데이터의 중간값(중앙값)이며, 입력 범위는 "
                    f"{prediction_min_x:.3f} ~ {prediction_max_x:.3f}입니다."
                )
            with result_col:
                st.markdown(pretty_title("머신러닝·딥러닝 예측값", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
                render_value_cards(
                    [
                        {
                            "title": "머신러닝",
                            "value": f"{ml_pred:.3f}",
                            "detail": f"{active_ml_display_name} 모델이 입력값으로 계산한 예측 결과입니다.",
                            "bg": "#f4f9ff",
                            "border": "#90caf9",
                            "min_height": "128px",
                        },
                        {
                            "title": "딥러닝",
                            "value": f"{dl_pred:.3f}",
                            "detail": "딥러닝 모델이 입력값으로 계산한 예측 결과입니다.",
                            "bg": "#f1f8e9",
                            "border": "#aed581",
                            "min_height": "128px",
                        },
                    ],
                    columns=2,
                )
                st.info("이 영역은 입력값에 따른 모델의 예측값만 보여 줍니다.")

    with tabs[4]:
        stage_intro(
            "보고서 작성 및 저장",
            "분석 결과를 요약하고, 보고서를 작성해 5차시 AI 데이터 예측 포트폴리오를 완성하는 단계입니다.",
            "자료 분석과 예측 결과를 어떤 근거로 정리하면 탐구 결론을 잘 설명할 수 있을까?",
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
        active_ml_display_name = selected_ml_display_name(st.session_state["d5_ml_degree"])
        report_prediction_x = float(st.session_state.get("d5_prediction_x", default_prediction_x))
        hidden_preds = predict_models(model_results, np.array([report_prediction_x], dtype=float))
        ml_pred = float(hidden_preds[active_ml_name][0])
        dl_pred = float(hidden_preds["딥러닝"][0])
        summary_metric_df = build_selected_comparison_df(model_results, st.session_state["d5_ml_degree"])
        best_model = summary_metric_df.loc[summary_metric_df["손실"].idxmin(), "모델"]
        best_loss = float(summary_metric_df.loc[summary_metric_df["손실"].idxmin(), "손실"])
        scale_text = "정규화 적용" if st.session_state["d5_use_scale"] else "정규화 미적용"
        deep_question = clean_text(st.session_state.get("d5_deep_question", ""), "아직 작성하지 않았습니다.")
        research_motivation = clean_text(st.session_state.get("d5_research_motivation", ""), "아직 작성하지 않았습니다.")

        st.markdown(pretty_title("모둠 정보 확인", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        group_name = st.session_state.get("d5_group", "")
        info_col, guide_col = st.columns([1.15, 1.0])
        with info_col:
            group_display = group_name if group_name else "데이터 선택 탭에서 입력해 주세요."
            st.markdown(
                f"""
                <div style='
                    background:#f4f9ff;
                    border:1px solid #90caf9;
                    border-left:6px solid #1976d2;
                    border-radius:12px;
                    padding:14px 16px;
                    min-height:88px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                '>
                    <div style='font-size:0.9rem; color:#1565c0; font-weight:800; margin-bottom:6px;'>모둠명</div>
                    <div style='font-size:1.35rem; color:#263238; font-weight:900;'>{group_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with guide_col:
            st.info("이 차시는 모둠당 하나의 앱으로 활동합니다. 모둠명을 확인한 뒤 아래 보고서를 작성하고 PDF를 저장하세요.")

        st.markdown(pretty_title("분석 결과 요약", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        render_summary_table(
            [
                {
                    "label": "탐구 설정",
                    "value": dataset["name"],
                    "note": f"{dataset['x_label']} → {dataset['y_label']} 관계를 분석합니다.",
                },
                {
                    "label": "깊은 질문",
                    "value": deep_question,
                    "note": "이번 분석으로 답해 보고 싶은 핵심 질문입니다.",
                },
                {
                    "label": "탐구 동기",
                    "value": research_motivation,
                    "note": "이 데이터를 분석하려는 이유를 정리한 내용입니다.",
                },
                {
                    "label": "입력값",
                    "value": f"{dataset['x_label']} = {report_prediction_x:.3f}",
                    "note": "예측 및 시각화 탭에서 직접 입력한 값입니다.",
                },
                {
                    "label": "머신러닝 예측값",
                    "value": f"{ml_pred:.3f}",
                    "note": f"{active_ml_display_name} 모델이 입력값 {report_prediction_x:.3f}으로 계산한 예측 결과입니다.",
                },
                {
                    "label": "딥러닝 예측값",
                    "value": f"{dl_pred:.3f}",
                    "note": f"딥러닝 모델이 입력값 {report_prediction_x:.3f}으로 계산한 예측 결과입니다. 구조: {model_results['nn_model']['architecture']} / {scale_text}",
                },
                {
                    "label": "모델 선택 기준",
                    "value": best_model,
                    "note": f"학습 데이터 손실이 가장 작았습니다. 손실: {best_loss:.3f}",
                },
            ]
        )

        auto_analysis, _ = build_auto_report_texts(
            dataset,
            model_results,
            active_ml_display_name,
            report_prediction_x,
            ml_pred,
            dl_pred,
            best_model,
            st.session_state.get("d5_deep_question", ""),
            st.session_state.get("d5_research_motivation", ""),
        )
        previous_auto_analysis = st.session_state.get("d5_auto_analysis_report", "")
        if not st.session_state.get("d5_analysis_report") or st.session_state.get("d5_analysis_report") == previous_auto_analysis:
            st.session_state["d5_analysis_report"] = auto_analysis
        st.session_state["d5_auto_analysis_report"] = auto_analysis

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#ef6c00; "
                "margin-bottom:6px;'>7️⃣ 연구 결과 및 분석</div>",
                unsafe_allow_html=True,
            )
            st.caption("데이터 분석 및 예측 결과는 자동으로 들어갑니다. 연구 결과 및 해석은 모둠이 직접 작성하세요.")
            report_col1, report_col2 = st.columns(2)
            with report_col1:
                st.text_area(
                    "데이터 분석 및 예측 결과",
                    key="d5_analysis_report",
                    height=260,
                    placeholder=auto_analysis,
                )
            with report_col2:
                st.text_area(
                    "연구 결과 및 해석",
                    key="d5_interpretation_report",
                    height=260,
                    placeholder="예측 결과를 보고 알게 된 점, 어떤 모델이 더 적절하다고 생각하는지, 이 결과를 어떻게 해석할 수 있는지 모둠의 말로 작성하세요.",
                )

        st.markdown(pretty_title("PDF 저장", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        if group_name:
            student_info = {"group": group_name}
            pdf_prediction_fig = make_selected_prediction_figure(
                dataset,
                split,
                model_results,
                st.session_state["d5_ml_degree"],
                student_guess=None,
                reveal=False,
                show_ml=True,
                show_dl=True,
                prediction_x=float(st.session_state.get("d5_prediction_x", default_prediction_x)),
            )
            figure_items = [
                ("예측 결과 그래프", pdf_prediction_fig),
            ]
            pdf_bytes = create_portfolio_pdf(
                student_info,
                dataset,
                split,
                model_results,
                st.session_state["d5_ml_degree"],
                st.session_state["d5_use_scale"],
                float(st.session_state.get("d5_prediction_x", default_prediction_x)),
                float(st.session_state["d5_student_guess"]),
                False,
                st.session_state.get("d5_deep_question", ""),
                st.session_state.get("d5_research_motivation", ""),
                st.session_state.get("d5_analysis_report", ""),
                st.session_state.get("d5_interpretation_report", ""),
                figure_items,
            )
            st.download_button(
                "보고서 PDF 저장하기",
                data=pdf_bytes,
                file_name=f"{group_name}_5차시_AI데이터예측보고서.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.warning("⚠️ 모둠원들이 동시에 PDF 다운로드 버튼을 누르면 오류가 날 수 있습니다. 한 명씩 차례대로 눌러 주세요.")
            render_portfolio_link(st.session_state.get("d5_class", CLASS_OPTIONS[0]))
        else:
            st.info("데이터 선택 탭에서 모둠명을 입력하면 보고서 PDF를 저장할 수 있습니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
