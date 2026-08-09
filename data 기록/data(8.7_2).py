# 실행 명령: streamlit run data7.py

import html
import io
from functools import lru_cache
import os
import time
import tempfile
import zipfile

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# matplotlib 한글 표시 설정: 프로젝트의 NanumGothic 글꼴을 우선 사용합니다.
BASE_DIR = os.path.dirname(__file__)
FONT_PATH = os.path.join(BASE_DIR, "font", "NanumGothic.ttf")


@lru_cache(maxsize=1)
def configure_matplotlib_font():
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        font_name = fm.FontProperties(fname=FONT_PATH).get_name()
        mpl.rc("font", family=font_name)
    else:
        mpl.rc("font", family="Malgun Gothic")
    mpl.rc("axes", unicode_minus=False)


try:
    configure_matplotlib_font()
except Exception:
    plt.rcParams["axes.unicode_minus"] = False

def make_yearly_table(anchor_years, values_by_column):
    years = list(range(min(anchor_years), max(anchor_years) + 1))
    table = {"연도": years}
    for column, values in values_by_column.items():
        table[column] = [round(float(value), 3) for value in np.interp(years, anchor_years, values)]
    return pd.DataFrame(table)


DATASETS = {
    "인간: 보건과 삶의 질": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023],
            {
                "기대수명(년)": [65.109, 66.201, 67.650, 69.111, 70.683, 71.966, 72.183, 73.330],
                "5세 미만 사망률(%) (예시)": [9.350, 8.750, 7.670, 6.230, 5.060, 4.290, 3.920, 3.830],
                "극빈곤 인구 비율(%)": [38.0, 36.8, 36.205, 28.330, 20.984, 13.422, 11.412, 10.618],
            },
        ),
        "default_y": "5세 미만 사망률(%) (예시)",
        "source": "World Bank, UN IGME, Our World in Data",
    },
    "번영: 디지털 접근과 도시 변화": {
        "table": make_yearly_table(
            [1994, 2000, 2005, 2010, 2015, 2020, 2023],
            {
                "인터넷 이용률(%)": [0.400, 6.720, 15.600, 28.400, 39.900, 60.100, 69.200],
                "전기 접근률(%)": [74.200, 78.225, 80.705, 83.450, 86.927, 90.400, 91.603],
                "도시화율(%)": [44.100, 46.840, 49.359, 51.760, 54.426, 56.444, 57.312],
            },
        ),
        "default_y": "인터넷 이용률(%)",
        "source": "World Bank, Our World in Data",
    },
    "환경: 산림과 생물다양성 보호": {
        "table": make_yearly_table(
            [1991, 1995, 2000, 2005, 2010, 2015, 2020],
            {
                "산림 면적 비율(%)": [32.650, 32.540, 32.376, 32.139, 31.948, 31.369, 31.172],
                "육상 KBA 보호 비율(%)": [17.000, 21.500, 26.153, 32.577, 38.230, 41.673, 44.003],
                "해양 KBA 보호 비율(%)": [16.500, 20.800, 25.698, 31.127, 37.372, 42.568, 45.347],
            },
        ),
        "default_y": "육상 KBA 보호 비율(%)",
        "source": "UN SDG Global Database 2026.Q2.G.01 (AG_LND_FRST, ER_PTD_TERR, ER_MRN_MPA)",
    },
    "평화: 난민과 강제이주": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021, 2022, 2023],
            {
                "강제이주민 수(백만 명)": [38.0, 36.0, 37.3, 37.5, 43.7, 65.1, 79.5, 82.4, 89.3, 108.4, 117.3],
                "난민 수(백만 명)": [17.2, 14.9, 15.9, 13.5, 15.4, 21.3, 26.0, 26.4, 27.1, 35.3, 37.6],
            },
        ),
        "default_y": "강제이주민 수(백만 명)",
        "source": "UNHCR Global Trends",
    },
}

TEACHER_DEMO_DATASET = "인간: 보건과 삶의 질"

PUBLIC_DATASET_OPTIONS = [
    TEACHER_DEMO_DATASET,
    "번영: 디지털 접근과 도시 변화",
    "환경: 산림과 생물다양성 보호",
    "평화: 난민과 강제이주",
]

U_MAX_ATTEMPTS = 4
CLASS_OPTIONS = ["1", "2", "5", "6"]
GALLERY_URLS = {
    "1": "https://padlet.com/ps0andd/g_1",
    "2": "https://padlet.com/ps0andd/g_2",
    "5": "https://padlet.com/ps0andd/g_5",
    "6": "https://padlet.com/ps0andd/g_6",
}
PORTFOLIO_URLS = GALLERY_URLS.copy()
GPT_URL = "https://chatgpt.com/"

FACTFULNESS_LENS_GUIDES = {
    "직선 본능 점검": {
        "guide": "최근 흐름이 앞으로도 같은 속도로 계속된다고 단정하지 않고, 변화 속도와 꺾이는 구간을 확인합니다.",
        "placeholder": "직선 본능 점검 관점으로 직접 작성해 보세요.",
    },
    "일반화 본능 점검": {
        "guide": "전체 흐름이나 평균이 모든 구간의 상황을 똑같이 대표한다고 단정하지 않습니다.",
        "placeholder": "일반화 본능 점검 관점으로 직접 작성해 보세요.",
    },
    "격차 본능 점검": {
        "guide": "잘 맞음과 맞지 않음을 둘로만 나누지 않고, 중간 정도의 차이와 구간별 차이를 함께 봅니다.",
        "placeholder": "격차 본능 점검 관점으로 직접 작성해 보세요.",
    },
}

def clean_text(value, default="아직 작성하지 않았습니다."):
    text = str(value).strip() if value is not None else ""
    return text if text else default


def pretty_title(text, color1, color2):
    numbered_icons = {
        "1. ": "1️⃣ ",
        "2. ": "2️⃣ ",
        "3. ": "3️⃣ ",
        "4. ": "4️⃣ ",
    }
    for prefix, icon in numbered_icons.items():
        if text.startswith(prefix):
            text = icon + text[len(prefix):]
            break
    return f"""
    <div style='background:#ffffff;border:3px solid {color2};border-left:9px solid {color1};
        border-radius:10px;padding:6px 13px 5px 13px;margin:9px 0 8px 0;'>
        <div style='font-size:1.08rem;font-weight:900;line-height:1.38;color:#263238;'>{text}</div>
    </div>
    """


def apply_local_style():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
        div[data-baseweb="tab-list"] {gap: 0.35rem;}
        div[data-baseweb="tab"] {
            background:#f4f8fc;border-radius:0.8rem;padding:0.45rem 0.9rem;border:1px solid #dbe7f3;
        }
        div[data-baseweb="tab"][aria-selected="true"] {background:#e8f3ff;border-color:#90caf9;}
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(1) {
            background:#ffebee;
            border-color:#ffcdd2;
        }
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
            background:#ffcdd2;
            border-color:#ef9a9a;
        }
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(2) {
            background:#e8f5e9;
            border-color:#c8e6c9;
        }
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
            background:#c8e6c9;
            border-color:#a5d6a7;
        }
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(3) {
            background:#f3e5f5;
            border-color:#e1bee7;
        }
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
            background:#e1bee7;
            border-color:#ce93d8;
        }
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(4) {
            background:#e3f2fd;
            border-color:#bbdefb;
        }
        div[data-baseweb="tab-list"] > div[data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {
            background:#bbdefb;
            border-color:#90caf9;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {
            border-radius: 0.85rem;
            border-color: #cfe0f2;
            background: #ffffff;
        }
        [data-testid="stNumberInput"] input {
            border: 2px solid #1976d2;
            border-radius: 0.9rem;
            background: #f8fbff;
            color: #0d47a1;
            font-size: 1.25rem;
            font-weight: 900;
            text-align: center;
            box-shadow: 0 0 0 4px rgba(25, 118, 210, 0.10);
        }
        .prediction-input-card {
            background: linear-gradient(135deg, #fbfaff 0%, #f3e5f5 100%);
            border: 2px solid #ab47bc;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 8px 18px rgba(106, 27, 154, 0.14);
            margin: 8px 0 10px 0;
        }
        .prediction-input-kicker {
            color: #6a1b9a;
            font-size: 0.82rem;
            font-weight: 900;
            margin-bottom: 3px;
        }
        .prediction-input-title {
            color: #6a1b9a;
            font-size: 1.18rem;
            font-weight: 900;
            margin-bottom: 5px;
        }
        .prediction-input-help {
            color: #37474f;
            font-size: 0.94rem;
            line-height: 1.55;
        }
        .formula-panel .katex-display {
            margin: 0.35rem 0 0.85rem 0;
            overflow-x: auto;
        }
        .formula-panel .katex,
        [data-testid="stLatex"] .katex,
        .katex {
            font-size: clamp(0.86rem, 1.45vw, 1.18rem);
            white-space: nowrap;
        }
        [data-testid="stMarkdownContainer"] blockquote {
            background: #f3e5f5;
            border: 1px solid #ce93d8;
            border-left: 1px solid #ce93d8;
            border-radius: 8px;
            color: #000000;
            margin: 2px 0 12px 0;
            overflow-x: auto;
            padding: 10px 14px;
            white-space: nowrap;
        }
        [data-testid="stMarkdownContainer"] blockquote p {
            color: #000000;
            margin: 0;
        }
        [data-testid="stMarkdownContainer"] blockquote .katex {
            color: #000000;
        }
        .fit-eval-box ~ div [role="radiogroup"] label:first-child,
        .fit-eval-box ~ div [data-testid="stWidgetLabel"] p {
            font-weight: 900;
            color: #263238;
        }
        .fit-eval-box ~ div [role="radiogroup"] {
            flex-wrap: nowrap;
            gap: 0.8rem;
        }
        .fit-eval-box ~ div [role="radiogroup"] label,
        .fit-eval-box ~ div [role="radiogroup"] label p {
            white-space: nowrap;
        }
        details:has(.fit-eval-box) summary {
            background: #fff3e0;
            border: 1px solid #ffcc80;
            border-radius: 8px;
            padding: 0.55rem 0.75rem;
        }
        details:has(.fit-eval-box) summary p {
            color: #e65100;
            font-weight: 900;
        }
        .radical-formula-box {
            background: #f8fbff;
            border: 1px solid #dbe7f3;
            border-radius: 8px;
            color: #263238;
            font-weight: 800;
            padding: 10px 12px;
            text-align: center;
            white-space: nowrap;
        }
        .radical-formula-title {
            color: #455a64;
            font-family: "NanumGothic", sans-serif;
            font-size: 0.88rem;
            font-weight: 900;
            margin-bottom: 5px;
        }
        .radical-formula-expression {
            font-family: "Times New Roman", "NanumGothic", serif;
            font-size: clamp(1.9rem, 3.6vw, 2.75rem);
            font-weight: 800;
            line-height: 1.5;
        }
        .u-radical-formula .radical-formula-expression {
            font-size: clamp(1.4rem, 2.05vw, 1.8rem);
            max-width: 100%;
            overflow: hidden;
            text-overflow: clip;
        }
        .radical-sign-token {
            color: #1565c0;
            font-weight: 950;
            margin: 0 0.06em;
        }
        .radical-a-token {
            color: #ef6c00;
            font-weight: 950;
            margin: 0 0.04em;
        }
        .radical-root {
            border-top: 2px solid currentColor;
            padding: 0 0.08em 0 0.1em;
        }
        .radical-control-label {
            align-items: center;
            background: #ffffff;
            border: 1px solid #dbe7f3;
            border-radius: 8px;
            display: flex;
            gap: 8px;
            margin: 0 0 6px 0;
            padding: 5px 7px;
        }
        .radical-control-label .token {
            align-items: center;
            border-radius: 6px;
            color: #ffffff;
            display: inline-flex;
            font-size: 0.92rem;
            font-weight: 950;
            justify-content: center;
            min-width: 28px;
            padding: 2px 7px;
        }
        .radical-control-label .text {
            color: #263238;
            font-size: 0.88rem;
            font-weight: 800;
            line-height: 1.35;
        }
        .radical-control-label.sign .token {
            background: #1565c0;
        }
        .radical-control-label.a .token {
            background: #ef6c00;
        }
        .radical-control-label.p .token {
            background: #7b1fa2;
        }
        .radical-control-label.q .token {
            background: #2e7d32;
        }
        .st-key-d8_practice_translation_observe_start button {
            background: linear-gradient(135deg, #1565c0 0%, #26a69a 100%);
            border: 0;
            border-radius: 10px;
            box-shadow: 0 8px 18px rgba(21, 101, 192, 0.24);
            color: #ffffff;
            font-weight: 900;
            min-height: 44px;
        }
        .st-key-d8_practice_translation_observe_start button:hover {
            background: linear-gradient(135deg, #0d47a1 0%, #00897b 100%);
            border: 0;
            color: #ffffff;
            transform: translateY(-1px);
        }
        .stage-card {
            background: #ffffff;
            border: 1px solid #dbe7f3;
            border-radius: 14px;
            padding: 15px 17px;
            box-shadow: 0 8px 18px rgba(21, 101, 192, 0.08);
            margin: 10px 0 14px 0;
        }
        .stage-card-blue {
            background: linear-gradient(135deg, #f8fbff 0%, #eef7ff 100%);
            border-color: #bbdefb;
        }
        .stage-card-red {
            background: linear-gradient(135deg, #fff8f8 0%, #ffebee 100%);
            border-color: #ffcdd2;
        }
        .stage-card-yellow {
            background: linear-gradient(135deg, #fffdf5 0%, #fff8e1 100%);
            border-color: #ffe0b2;
        }
        .stage-card-green {
            background: linear-gradient(135deg, #fbfffb 0%, #f1f8e9 100%);
            border-color: #c8e6c9;
        }
        .stage-card-purple {
            background: linear-gradient(135deg, #fbfaff 0%, #ede7f6 100%);
            border-color: #d1c4e9;
        }
        .stage-kicker {
            font-size: 0.82rem;
            font-weight: 900;
            color: #1565c0;
            margin-bottom: 4px;
        }
        .stage-card-red .stage-kicker {color:#c62828;}
        .stage-card-green .stage-kicker {color:#2e7d32;}
        .stage-card-purple .stage-kicker {color:#6a1b9a;}
        .stage-card-blue .stage-kicker {color:#1565c0;}
        .stage-card-title {
            font-size: 1.14rem;
            font-weight: 900;
            color: #1f2937;
            margin-bottom: 5px;
        }
        .stage-card-help {
            color: #37474f;
            font-size: 0.94rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stage_card(title, help_text, variant="blue", kicker="활동 안내"):
    st.markdown(
        f"""
        <div class="stage-card stage-card-{html.escape(variant)}">
            <div class="stage-kicker">{html.escape(kicker)}</div>
            <div class="stage-card-title">{html.escape(title)}</div>
            <div class="stage-card-help">{html.escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_banner(title, description, question):
    description_html = (
        f"""<div style="font-size:1rem;line-height:1.7;color:#37474f;">{description}</div>"""
        if description
        else ""
    )
    question_html = (
        f"""
            <div style="margin-top:12px;background:rgba(255,255,255,0.72);border-radius:12px;
                padding:10px 12px;color:#1f2937;border:1px solid rgba(255,255,255,0.9);">
                <b>핵심 탐구 질문</b><br>{question}
            </div>
        """
        if question
        else ""
    )
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#e3f2fd 0%,#d1c4e9 100%);
            border-radius:16px;padding:16px 20px;border:1px solid #dbe7f3;margin-bottom:10px;">
            <div style="font-size:0.82rem;font-weight:700;color:#5e35b1;margin-bottom:5px;">F.U.T.U.R.E. 프로젝트</div>
            <div style="font-size:1.48rem;font-weight:800;color:#1f2937;margin-bottom:4px;">{title}</div>
            {description_html}
            {question_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_activity_flow():
    steps = [
        ("1", "F.U", "문제 발견", "데이터를 고르고 삶의 문제를 찾기", "#ffebee", "#ffcdd2", "#c62828"),
        ("2", "T", "수학의 언어", "도형의 평행이동 다시 익히기", "#e8f5e9", "#c8e6c9", "#2e7d32"),
        ("3", "U", "AI 이해", "추세선을 조절해 예측값 구하기", "#f3e5f5", "#e1bee7", "#7b1fa2"),
        ("4", "R.E", "세상과 연결", "예측 결과를 삶의 의미로 정리하기", "#e3f2fd", "#bbdefb", "#1565c0"),
    ]
    step_html = "".join(
        f"""
        <div style="
            flex:1 1 150px;
            min-width:140px;
            border:1px solid {border_color};
            background:linear-gradient(135deg,#ffffff 0%,{bg_color} 100%);
            border-radius:10px;
            padding:10px 12px;
        ">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;
                    border-radius:50%;background:{text_color};color:white;font-weight:800;font-size:0.82rem;">{num}</span>
                <span style="display:inline-flex;align-items:center;justify-content:center;
                    border-radius:999px;background:{chip_color};border:1px solid {border_color};color:{text_color};
                    font-weight:900;font-size:0.78rem;padding:2px 8px;">{stage}</span>
            </div>
            <div style="font-size:0.98rem;font-weight:900;color:#1f2937;margin-bottom:4px;">{title}</div>
            <div style="font-size:0.84rem;line-height:1.5;color:#455a64;font-weight:800;">{body}</div>
        </div>
        """
        for num, stage, title, body, bg_color, border_color, text_color in steps
        for chip_color in [bg_color]
    )
    st.markdown(
        f"""
        <div style="
            background:#f8fbff;
            border:1px solid #dbe7f3;
            border-radius:14px;
            padding:12px 14px;
            margin:-4px 0 14px 0;
        ">
            <div style="font-size:0.92rem;font-weight:900;color:#263238;margin-bottom:10px;">데이터 탐구의 흐름</div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">{step_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_intro(title, description, question, color1="#e8f5e9", color2="#c8e6c9", question_label="핵심 탐구 질문"):
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{color1} 0%,{color2} 100%);
            border-radius:18px;padding:18px 20px;border:1px solid rgba(0,0,0,0.06);margin-bottom:12px;">
            <div style="font-size:1.05rem;font-weight:800;color:#1f2937;margin-bottom:8px;">{title}</div>
            <div style="font-size:0.97rem;line-height:1.7;color:#37474f;margin-bottom:12px;">{description}</div>
            <div style="background:rgba(255,255,255,0.72);border-radius:12px;padding:10px 12px;
                border:1px solid rgba(255,255,255,0.85);color:#37474f;line-height:1.6;">
                <b>{question_label}</b><br>{question}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def numeric_columns(dataset_info):
    table = dataset_info["table"]
    return [col for col in table.columns if pd.api.types.is_numeric_dtype(table[col])]


def variable_meaning(column_name):
    meanings = {
        "연도": "자료가 관찰되거나 기록된 해입니다. 시간에 따른 변화를 볼 때 입력변수 x로 자주 사용합니다.",
        "기대수명(년)": "해당 해에 태어난 사람이 현재 사망률이 유지된다고 가정할 때 평균적으로 살 것으로 예상되는 햇수입니다.",
        "5세 미만 사망률(%) (예시)": "태어난 아이 100명 중 5세가 되기 전에 사망하는 아이의 수를 나타낸 비율 지표입니다.",
        "극빈곤 인구 비율(%)": "국제 빈곤선보다 낮은 소득으로 생활하는 인구가 전체 인구에서 차지하는 비율입니다.",
        "인터넷 이용률(%)": "최근 일정 기간 동안 인터넷을 사용한 사람이 전체 인구에서 차지하는 비율입니다.",
        "전기 접근률(%)": "가정이나 생활 공간에서 전기를 사용할 수 있는 인구가 전체 인구에서 차지하는 비율입니다.",
        "도시화율(%)": "전체 인구 중 도시 지역에 거주하는 인구의 비율입니다.",
        "강제이주민 수(백만 명)": "분쟁, 박해, 폭력, 인권 침해 등으로 살던 곳을 떠나야 했던 사람의 수입니다.",
        "난민 수(백만 명)": "국경을 넘어 다른 나라에서 보호를 필요로 하는 난민의 수입니다.",
        "산림 면적 비율(%)": "전체 육지 면적 중 산림이 차지하는 비율입니다.",
        "육상 KBA 보호 비율(%)": "육상 핵심생물다양성지역(KBA) 중 보호구역으로 지정되어 관리되는 면적의 비율입니다.",
        "해양 KBA 보호 비율(%)": "해양 핵심생물다양성지역(KBA) 중 보호구역으로 지정되어 관리되는 면적의 비율입니다.",
    }
    return meanings.get(column_name, f"{column_name}의 값을 나타내는 수치형 변수입니다.")


def life_change_placeholders(y_label):
    examples = {
        "기대수명(년)": (
            "예: 사람들이 더 오래 살고 노년의 삶을 준비할 시간이 늘어난다.",
            "예: 질병이나 사고로 삶의 시간이 짧아질 위험이 커진다.",
        ),
        "5세 미만 사망률(%) (예시)": (
            "예: 어린 나이에 생명을 잃는 아이들이 늘어난다.",
            "예: 더 많은 아이들이 건강하게 성장할 수 있다.",
        ),
        "극빈곤 인구 비율(%)": (
            "예: 기본적인 식사, 주거, 의료를 감당하기 어려운 사람이 늘어난다.",
            "예: 더 많은 사람이 안정적인 생활을 할 수 있다.",
        ),
        "인터넷 이용률(%)": (
            "예: 더 많은 사람이 온라인 정보와 교육 기회를 얻는다.",
            "예: 정보 접근이 어려운 사람이 늘어날 수 있다.",
        ),
        "전기 접근률(%)": (
            "예: 조명, 냉장 보관, 통신을 더 안정적으로 이용할 수 있다.",
            "예: 전기를 쓰기 어려워 일상생활의 불편이 커질 수 있다.",
        ),
        "도시화율(%)": (
            "예: 도시의 일자리와 교육·의료 서비스를 이용하는 사람이 늘어난다.",
            "예: 지역에 따라 일자리와 공공서비스 기회가 부족할 수 있다.",
        ),
        "강제이주민 수(백만 명)": (
            "예: 집을 떠나야 하는 사람이 늘어난다.",
            "예: 삶의 터전을 잃는 사람이 줄어든다.",
        ),
        "난민 수(백만 명)": (
            "예: 보호가 필요한 사람이 늘어나 국제적 지원이 더 중요해진다.",
            "예: 위험을 피해 떠나야 하는 사람이 줄어든다.",
        ),
        "육상 KBA 보호 비율(%)": (
            "예: 중요한 생태 지역이 더 많이 보호된다.",
            "예: 동식물의 서식지가 위협받을 수 있다.",
        ),
        "해양 KBA 보호 비율(%)": (
            "예: 중요한 해양 생태계가 더 많이 보호된다.",
            "예: 바다 생물과 해안 지역의 삶이 위협받을 수 있다.",
        ),
        "산림 면적 비율(%)": (
            "예: 숲이 늘어나 공기 질과 생물 서식지에 도움이 된다.",
            "예: 숲이 줄어들어 생물 서식지와 기후에 문제가 생길 수 있다.",
        ),
    }
    return examples.get(
        y_label,
        (
            f"예: {y_label} 값이 커질 때 삶에서 나타날 모습을 써 보세요.",
            f"예: {y_label} 값이 작아질 때 삶에서 나타날 모습을 써 보세요.",
        ),
    )


def strip_example_prefix(text):
    return str(text).removeprefix("예: ").strip()


def selected_life_change(y_label):
    direction = st.session_state.get("d8_life_direction")
    if direction == "커질 때":
        direction = "증가한다"
    elif direction == "작아질 때":
        direction = "감소한다"
    if direction not in ["증가한다", "감소한다"]:
        if clean_text(st.session_state.get("d8_y_decrease_life", "")):
            direction = "감소한다"
        else:
            direction = "증가한다"
    life_phrase = "커질 때" if direction == "증가한다" else "작아질 때"
    key = "d8_y_increase_life" if direction == "증가한다" else "d8_y_decrease_life"
    return direction, f"{y_label} 값이 {life_phrase}", clean_text(st.session_state.get(key, ""))


def trend_based_life_example(y_label, trend_text):
    increase_example, decrease_example = life_change_placeholders(y_label)
    if "감소" in trend_text:
        return strip_example_prefix(decrease_example)
    if "증가" in trend_text:
        return strip_example_prefix(increase_example)
    return f"{y_label}의 변화가 사람들의 삶에 어떤 의미인지 생각해 볼 수 있다."


def variable_help_text(columns):
    return "선택 가능한 변수 설명\n" + "\n".join(f"- {column}: {variable_meaning(column)}" for column in columns)


def ensure_xy_columns(dataset_name):
    info = DATASETS[dataset_name]
    columns = numeric_columns(info)
    fixed_x = "연도" if "연도" in columns else columns[0]
    default_y = info.get("default_y", columns[1] if len(columns) > 1 else columns[0])
    if fixed_x not in columns:
        fixed_x = columns[0]
    if default_y not in columns or default_y == fixed_x:
        default_y = next((col for col in columns if col != fixed_x), fixed_x)
    if st.session_state.get("d8_xy_dataset") != dataset_name:
        st.session_state["d8_x_col"] = fixed_x
        st.session_state["d8_y_col"] = default_y
        st.session_state["d8_xy_dataset"] = dataset_name
    st.session_state["d8_x_col"] = fixed_x
    if st.session_state.get("d8_y_col") not in columns:
        st.session_state["d8_y_col"] = default_y
    if st.session_state["d8_x_col"] == st.session_state["d8_y_col"] and len(columns) > 1:
        st.session_state["d8_y_col"] = next(col for col in columns if col != st.session_state["d8_x_col"])
    return columns


def selected_xy_data(dataset_name):
    info = DATASETS[dataset_name]
    ensure_xy_columns(dataset_name)
    x_label = st.session_state["d8_x_col"]
    y_label = st.session_state["d8_y_col"]
    clean_table = info["table"][[x_label, y_label]].dropna()
    return clean_table[x_label].to_numpy(float), clean_table[y_label].to_numpy(float), x_label, y_label


def calculate_function(x_values, params):
    x_arr = np.asarray(x_values, dtype=float)
    model = params.get("model", "직선")
    coefficient = float(params.get("coefficient", params.get("a", 1.0)))
    p = float(params.get("p", 0.0))
    q = float(params.get("q", 0.0))
    if model == "곡선":
        y_arr = coefficient * (x_arr - p) ** 2 + q
    else:
        y_arr = coefficient * (x_arr - p) + q
    mask = np.isfinite(y_arr)
    return y_arr, mask


def fit_default_params(x_data, y_data, model_type=None):
    selected_model = model_type or st.session_state.get("d8_u_function_type", "직선")
    return _fit_default_params_cached(tuple(map(float, x_data)), tuple(map(float, y_data)), selected_model).copy()


@st.cache_data(show_spinner=False, max_entries=32)
def _fit_default_params_cached(x_values, y_values, model_type):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if model_type == "곡선" and len(x) >= 3:
        a2, b2, c2 = np.polyfit(x, y, 2)
        if abs(a2) >= 1e-12:
            p = -b2 / (2 * a2)
            q = c2 - (a2 * p ** 2)
            return {"model": "곡선", "coefficient": float(a2), "p": float(p), "q": float(q)}
    m, b = np.polyfit(x, y, 1)
    p = float(np.min(x))
    q = float(m * p + b)
    return {"model": "직선", "coefficient": float(m), "p": p, "q": q}


def get_parameters(x_data, y_data):
    st.session_state.setdefault("d8_u_function_type", "직선")
    type_col, formula_preview_col = st.columns([1.1, 1.0], gap="small")
    with type_col:
        function_type = st.radio(
            "추세선 함수 선택",
            ["직선", "곡선"],
            key="d8_u_function_type",
            horizontal=True,
        )
    with formula_preview_col:
        st.markdown("**일반화된 함수**")
        st.latex(rf"\Large {general_translated_formula(function_type)}")
    defaults = fit_default_params(x_data, y_data, function_type)
    slider_context = (
        function_type,
        round(float(min(x_data)), 6),
        round(float(max(x_data)), 6),
        round(float(min(y_data)), 6),
        round(float(max(y_data)), 6),
    )
    if st.session_state.get("d8_slider_context") != slider_context:
        slider_keys = ["d8_u_coefficient", "d8_u_shift_p", "d8_u_shift_q"]
        for key in slider_keys:
            st.session_state.pop(key, None)
        st.session_state["d8_slider_context"] = slider_context
    params = {"model": function_type}

    coefficient_col, p_col, q_col = st.columns(3, gap="small")
    coefficient_default = float(defaults.get("coefficient", 1.0))
    coefficient_span = max(abs(coefficient_default) * 0.35, 0.05)
    coefficient_min = float(coefficient_default - coefficient_span)
    coefficient_max = float(coefficient_default + coefficient_span)
    if coefficient_min == coefficient_max:
        coefficient_min, coefficient_max = -1.0, 1.0
    with coefficient_col:
        coefficient_token = "m" if function_type == "직선" else "a"
        coefficient_name = "기울기" if function_type == "직선" else "최고차항의 계수"
        render_radical_control_label("a", coefficient_token, coefficient_name)
        params["coefficient"] = st.slider(
            f"{coefficient_token}: {coefficient_name}",
            coefficient_min,
            coefficient_max,
            coefficient_default,
            float(max((coefficient_max - coefficient_min) / 120, 1e-5)),
            key="d8_u_coefficient",
            label_visibility="collapsed",
        )
    x_span = max(float(np.max(x_data) - np.min(x_data)), 1.0)
    y_span = max(float(np.max(y_data) - np.min(y_data)), 1.0)
    p_default = float(defaults.get("p", np.mean(x_data)))
    q_default = float(defaults.get("q", np.mean(y_data)))
    p_window = max(x_span * 0.10, 1.0)
    q_window = max(y_span * 0.20, 0.5)
    with p_col:
        render_radical_control_label("p", "p", "x축 평행이동")
        params["p"] = st.slider(
            "p: x축 평행이동",
            float(p_default - p_window),
            float(p_default + p_window),
            p_default,
            float(max(p_window / 80, 0.05)),
            key="d8_u_shift_p",
            label_visibility="collapsed",
        )
    with q_col:
        render_radical_control_label("q", "q", "y축 평행이동")
        params["q"] = st.slider(
            "q: y축 평행이동",
            float(q_default - q_window),
            float(q_default + q_window),
            q_default,
            float(max(q_window / 80, 0.01)),
            key="d8_u_shift_q",
            label_visibility="collapsed",
        )
    return params


def calculate_loss(actual_y, predicted_y, valid_mask):
    actual = np.asarray(actual_y, dtype=float)
    pred = np.asarray(predicted_y, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred)
    if not np.any(mask):
        return None, 0
    return float(np.mean((actual[mask] - pred[mask]) ** 2)), int(np.sum(mask))


def build_function_text(params):
    return f"f(x) = {function_latex(params).replace('f(x)=', '')}"


def render_estimated_function_strip(params):
    model = params.get("model", "직선")
    st.markdown(
        rf"""
> **추세선({model})** &nbsp;&nbsp; $\large {function_latex(params)}$
"""
    )


def signed_latex_number(value):
    return f"+ {abs(value):.2f}" if value >= 0 else f"- {abs(value):.2f}"


def function_latex(params):
    coefficient = float(params.get("coefficient", params.get("a", 1.0)))
    p = float(params["p"])
    q = float(params["q"])
    model = params.get("model", "직선")
    if model == "곡선":
        return rf"f(x)={coefficient:.2f}(x {signed_latex_number(-p)})^2 {signed_latex_number(q)}"
    return rf"f(x)={coefficient:.2f}(x {signed_latex_number(-p)}) {signed_latex_number(q)}"


def emphasize_xy_axes(ax):
    ax.spines["left"].set_color("#d0d0d0")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("#d0d0d0")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["top"].set_color("#d0d0d0")
    ax.spines["right"].set_color("#d0d0d0")
    ax.tick_params(axis="both", colors="#111111", width=1.0)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if x_min <= 0 <= x_max:
        ax.axvline(0, color="#111111", linewidth=2.2, zorder=1)
        ax.annotate(
            "",
            xy=(0, y_max),
            xytext=(0, y_max - (y_max - y_min) * 0.08),
            arrowprops=dict(arrowstyle="-|>", color="#111111", linewidth=2.2, mutation_scale=14),
            annotation_clip=False,
            zorder=2,
        )
    if y_min <= 0 <= y_max:
        ax.axhline(0, color="#111111", linewidth=2.2, zorder=1)
        ax.annotate(
            "",
            xy=(x_max, 0),
            xytext=(x_max - (x_max - x_min) * 0.08, 0),
            arrowprops=dict(arrowstyle="-|>", color="#111111", linewidth=2.2, mutation_scale=14),
            annotation_clip=False,
            zorder=2,
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def draw_radical_practice_graph(sign_symbol, a_value, point_x=0.0, step_size=2.0, show_point=True):
    sign = 1 if sign_symbol == "+" else -1
    x_values = np.linspace(0.0, 10.0, 260)
    y_values = sign * np.sqrt(a_value * x_values)
    point_y = sign * np.sqrt(a_value * max(point_x, 0.0))

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.plot(x_values, y_values, color="#1976d2", linewidth=2.5, label=f"y={sign_symbol}√({int(a_value)}x)")
    ax.scatter([0], [0], color="#d32f2f", s=82, zorder=4, label="시작점 (0, 0)")
    if show_point:
        for current_x in np.arange(step_size, point_x + 0.001, step_size):
            previous_x = current_x - step_size
            previous_y = sign * np.sqrt(a_value * max(previous_x, 0.0))
            current_y = sign * np.sqrt(a_value * max(current_x, 0.0))
            delta_y = current_y - previous_y
            label_y = (current_y + previous_y) / 2
            ax.plot(
                [current_x, current_x],
                [previous_y, current_y],
                color="#f57c00",
                linestyle="--",
                linewidth=2.0,
                alpha=0.72,
            )
            ax.annotate(
                f"Δy={abs(delta_y):.2f}",
                xy=(current_x, label_y),
                xytext=(8, 0),
                textcoords="offset points",
                color="#e65100",
                fontsize=8.5,
                fontweight="bold",
                va="center",
            )
        ax.scatter([point_x], [point_y], color="#f57c00", edgecolor="#ffffff", linewidth=1.8, s=120, zorder=5)
        ax.annotate(
            "P",
            xy=(point_x, point_y),
            xytext=(8, 8),
            textcoords="offset points",
            color="#e65100",
            fontsize=12,
            fontweight="bold",
        )
    ax.annotate(
        "시작점",
        xy=(0, 0),
        xytext=(8, -18 if sign_symbol == "+" else 12),
        textcoords="offset points",
        color="#d32f2f",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlim(-1.0, 10.8)
    if sign_symbol == "+":
        ax.set_ylim(-1.0, 10.0)
    else:
        ax.set_ylim(-10.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    emphasize_xy_axes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def translated_practice_formula(function_type, sign_symbol, coefficient, p_value, q_value):
    coefficient_text = f"{coefficient:g}"
    p_latex = signed_latex_number(-float(p_value))
    q_latex = signed_latex_number(float(q_value))
    if function_type == "직선":
        return rf"y={sign_symbol}{coefficient_text}(x {p_latex}) {q_latex}"
    return rf"y={sign_symbol}{coefficient_text}(x {p_latex})^2 {q_latex}"


def base_practice_formula(function_type, sign_symbol, coefficient):
    coefficient_text = f"{coefficient:g}"
    if function_type == "직선":
        return rf"y={sign_symbol}{coefficient_text}x"
    return rf"y={sign_symbol}{coefficient_text}x^2"


def general_base_formula(function_type):
    if function_type == "직선":
        return r"y=mx"
    return r"y=ax^2"


def general_translated_formula(function_type):
    if function_type == "직선":
        return r"y=m(x-p)+q"
    return r"y=a(x-p)^2+q"


def translated_practice_values(x_values, function_type, sign_symbol, coefficient, p_value=0.0, q_value=0.0):
    x_arr = np.asarray(x_values, dtype=float)
    signed_coefficient = float(coefficient) if sign_symbol == "+" else -float(coefficient)
    if function_type == "직선":
        return signed_coefficient * (x_arr - float(p_value)) + float(q_value)
    return signed_coefficient * (x_arr - float(p_value)) ** 2 + float(q_value)


def translation_feature_text(function_type, sign_symbol, coefficient, p_value, q_value):
    signed_coefficient = float(coefficient) if sign_symbol == "+" else -float(coefficient)
    if function_type == "직선":
        direction = "증가" if signed_coefficient > 0 else "감소"
        return f"기울기 {signed_coefficient:g}의 부호가 직선의 {direction}를 결정하고, (p, q)=({p_value:g}, {q_value:g})는 이 직선이 지나는 한 점입니다."
    opening = "위로 열린다" if signed_coefficient > 0 else "아래로 열린다"
    return f"최고차항의 계수 {signed_coefficient:g}의 부호가 포물선이 {opening}는 방향을 결정하고, (p, q)=({p_value:g}, {q_value:g})는 꼭짓점입니다."


def translation_observation_html(function_type, sign_symbol, coefficient, p_value, q_value):
    signed_coefficient = float(coefficient) if sign_symbol == "+" else -float(coefficient)
    if function_type == "직선":
        direction = "증가" if signed_coefficient > 0 else "감소"
        feature_text = f"직선에서는 (p, q)=({p_value:g}, {q_value:g})가 이동한 직선이 지나는 한 점입니다."
        sign_text = f"m={signed_coefficient:g}이므로 기울기의 부호가 {'+' if signed_coefficient > 0 else '-'}입니다. 따라서 그래프는 {direction}합니다."
    else:
        opening = "위로 열린" if signed_coefficient > 0 else "아래로 열린"
        feature_text = f"이차함수에서는 (p, q)=({p_value:g}, {q_value:g})가 이동한 포물선의 꼭짓점 좌표입니다."
        sign_text = f"a={signed_coefficient:g}이므로 최고차항의 계수 부호가 {'+' if signed_coefficient > 0 else '-'}입니다. 따라서 포물선은 {opening} 모양입니다."
    return f"""
    <div style="background:#f8fbff;border:1px solid #bbdefb;border-left:6px solid #1565c0;
        border-radius:10px;padding:10px 12px;color:#263238;font-weight:800;line-height:1.55;margin:8px 0 0 0;">
        <div style="font-size:0.92rem;font-weight:950;color:#1565c0;margin-bottom:5px;">관찰 결과 개념 정리</div>
        <div>1. {html.escape(sign_text)}</div>
        <div>2. 평행이동해도 도형의 형태는 변하지 않고 위치만 변합니다.</div>
        <div>3. {html.escape(feature_text)}</div>
    </div>
    """


def animation_steps(start, end, step=0.5):
    start_float = float(start)
    end_float = float(end)
    if abs(end_float - start_float) < 1e-9:
        return [end_float]
    direction = 1.0 if end_float > start_float else -1.0
    values = [start_float]
    current = start_float
    while (end_float - current) * direction > step:
        current += direction * step
        values.append(round(current, 10))
    if abs(values[-1] - end_float) > 1e-9:
        values.append(end_float)
    return values


def draw_translated_practice_graph(function_type, sign_symbol, coefficient, p_value, q_value, show_observation=False):
    p_float = float(p_value)
    q_float = float(q_value)
    focus_half_width = 2.6 if function_type == "직선" else 3.2
    x_min = min(-focus_half_width, p_float - focus_half_width)
    x_max = max(focus_half_width, p_float + focus_half_width)
    x_values = np.linspace(x_min, x_max, 520)
    before_y = translated_practice_values(x_values, function_type, sign_symbol, coefficient)
    x_shift_y = translated_practice_values(x_values, function_type, sign_symbol, coefficient, p_value, 0.0)
    after_y = translated_practice_values(x_values, function_type, sign_symbol, coefficient, p_value, q_value)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.set_facecolor("#fbfdff")
    ax.axhspan(-0.05, 0.05, color="#263238", alpha=0.05, zorder=0)
    ax.axvspan(-0.05, 0.05, color="#263238", alpha=0.05, zorder=0)
    ax.plot(x_values, before_y, color="#78909c", linewidth=2.6, linestyle=(0, (5, 4)), label="1. 평행이동 전")
    if show_observation:
        ax.plot(x_values, x_shift_y, color="#f57c00", linewidth=2.8, linestyle="-.", label="2. x축 방향 이동")
    ax.plot(x_values, after_y, color="#1565c0", linewidth=3.4, label="3. 평행이동 후")

    base_point = (0.0, 0.0)
    x_shift_point = (float(p_value), 0.0)
    moved_point = (p_float, q_float)
    ax.scatter([base_point[0]], [base_point[1]], color="#607d8b", s=78, zorder=4)
    if show_observation:
        ax.scatter([x_shift_point[0]], [x_shift_point[1]], color="#f57c00", s=86, zorder=5)
    ax.scatter([moved_point[0]], [moved_point[1]], color="#d32f2f", s=96, zorder=6)
    ax.annotate(
        "(0, 0)",
        xy=base_point,
        xytext=(10, -18),
        textcoords="offset points",
        color="#455a64",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#cfd8dc", alpha=0.92),
    )
    feature_label = "지나는 점" if function_type == "직선" else "꼭짓점"
    ax.annotate(
        f"{feature_label} ({p_value:g}, {q_value:g})",
        xy=moved_point,
        xytext=(12, 12),
        textcoords="offset points",
        color="#b71c1c",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#ffebee", edgecolor="#ef9a9a", alpha=0.94),
    )

    if show_observation:
        ax.annotate(
            "",
            xy=x_shift_point,
            xytext=base_point,
            arrowprops=dict(arrowstyle="->", color="#f57c00", linewidth=2.4),
            zorder=6,
        )
        ax.annotate(
            "",
            xy=moved_point,
            xytext=x_shift_point,
            arrowprops=dict(arrowstyle="->", color="#43a047", linewidth=2.4),
            zorder=6,
        )
        ax.text(
            moved_point[0] / 2,
            base_point[1] - 0.7,
            f"x축 방향 {p_value:+g}",
            color="#e65100",
            fontsize=13,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff8e1", edgecolor="#f9a825", alpha=0.94),
        )
        ax.text(
            moved_point[0] + 0.25,
            moved_point[1] / 2,
            f"y축 방향 {q_value:+g}",
            color="#1b5e20",
            fontsize=13,
            fontweight="bold",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f5e9", edgecolor="#66bb6a", alpha=0.94),
        )
        ax.set_title("그래프 전체가 x축 방향 p만큼 이동한 뒤, y축 방향 q만큼 이동합니다.", fontsize=15, fontweight="bold")

    focus_mask = (x_values >= min(-focus_half_width, p_float - focus_half_width)) & (
        x_values <= max(focus_half_width, p_float + focus_half_width)
    )
    if function_type == "직선":
        signed_coefficient = float(coefficient) if sign_symbol == "+" else -float(coefficient)
        local_span = abs(signed_coefficient) * focus_half_width + 1.0
        y_min = min(0.0, q_float) - local_span
        y_max = max(0.0, q_float) + local_span
        y_margin = 0.6
    else:
        y_candidates = [
            before_y[focus_mask],
            x_shift_y[focus_mask],
            after_y[focus_mask],
            np.asarray([0.0, q_float], dtype=float),
        ]
        y_min = float(np.nanmin([np.nanmin(values) for values in y_candidates]))
        y_max = float(np.nanmax([np.nanmax(values) for values in y_candidates]))
        y_margin = max((y_max - y_min) * 0.12, 1.2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, which="major", color="#cfd8dc", linewidth=0.9, alpha=0.62)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#e7eef4", linewidth=0.55, alpha=0.75)
    emphasize_xy_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper left",
            fontsize=12,
            framealpha=0.96,
            facecolor="#ffffff",
            edgecolor="#dbe7f3",
            borderpad=0.75,
            labelspacing=0.6,
        )
    fig.tight_layout()
    return fig


def render_radical_formula_html(sign_text="±", a_text="a", suffix="", title=None):
    title_html = (
        f"""<div class="radical-formula-title">{html.escape(str(title))}</div>"""
        if title
        else ""
    )
    st.markdown(
        f"""
        <div class="radical-formula-box">
            {title_html}
            <div class="radical-formula-expression">
                y=<span class="radical-sign-token">{html.escape(str(sign_text))}</span>√<span class="radical-root">(<span class="radical-a-token">{html.escape(str(a_text))}</span>x)</span>{suffix}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_radical_shifted_formula_html(sign_text, a_text, p_text, q_text, title=None, extra_class=""):
    title_html = (
        f"""<div class="radical-formula-title">{html.escape(str(title))}</div>"""
        if title
        else ""
    )
    st.markdown(
        f"""
        <div class="radical-formula-box {html.escape(str(extra_class))}">
            {title_html}
            <div class="radical-formula-expression">
                y=<span class="radical-sign-token">{html.escape(str(sign_text))}</span>√<span class="radical-root">(<span class="radical-a-token">{html.escape(str(a_text))}</span>(x {html.escape(str(p_text))}))</span> {html.escape(str(q_text))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_radical_control_label(kind, token, text):
    st.markdown(
        f"""
        <div class="radical-control-label {html.escape(kind)}">
            <span class="token">{html.escape(token)}</span>
            <span class="text">{html.escape(text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_function_graph_practice():
    render_stage_card(
        "도형의 평행이동 개념을 다시 확인합니다",
        "직선 또는 이차함수 그래프를 선택하고, 계수와 x축·y축 평행이동 값을 조절하며 이동 전 그래프와 이동 후 그래프를 비교합니다.",
        "green",
        "개념 탐구",
    )
    function_col, selected_formula_col = st.columns([1.15, 1.0], gap="small")
    with function_col:
        st.markdown("**기본 그래프 선택**")
        function_type = st.radio(
            "직선 또는 곡선",
            ["직선", "곡선"],
            key="d8_practice_function_type",
            horizontal=True,
            label_visibility="collapsed",
        )
    with selected_formula_col:
        st.markdown("**평행이동 일반형**")
        st.latex(rf"\Large {general_translated_formula(function_type)}")
    translation_check_options = [
        "y=f(x-p)+q",
        "y=f(x+p)+q",
        "y=f(x)+p+q",
    ]
    if st.session_state.get("d8_translation_understanding") not in translation_check_options:
        st.session_state["d8_translation_understanding"] = None
    coefficient_name = "기울기" if function_type == "직선" else "최고차항의 계수"
    coefficient_token = "m" if function_type == "직선" else "a"
    coefficient_col, p_col, q_col = st.columns(3, gap="small")
    with coefficient_col:
        render_radical_control_label("a", coefficient_token, coefficient_name)
        coefficient_signed = st.slider(
            f"{coefficient_token}: {coefficient_name}",
            -2.0,
            2.0,
            1.0,
            0.1,
            key="d8_practice_coefficient_signed",
            help="0보다 크면 양의 방향, 0보다 작으면 음의 방향입니다.",
            label_visibility="collapsed",
        )
        if abs(float(coefficient_signed)) < 1e-9:
            coefficient_signed = 0.1
            st.caption("0은 그래프의 형태 확인이 어려워 0.1로 계산합니다.")
        sign_symbol = "+" if float(coefficient_signed) >= 0 else "-"
        coefficient = abs(float(coefficient_signed))
    with p_col:
        render_radical_control_label("p", "p", "x축 평행이동")
        if not -5.0 <= float(st.session_state.get("d8_practice_shift_p", 1.0)) <= 5.0:
            st.session_state["d8_practice_shift_p"] = 1.0
        p_value = st.slider(
            "p: x축 평행이동",
            -5.0,
            5.0,
            1.0,
            1.0,
            key="d8_practice_shift_p",
            help="p가 양수이면 오른쪽, 음수이면 왼쪽으로 이동합니다.",
            label_visibility="collapsed",
        )
        st.markdown(
            "<div style='display:flex;justify-content:center;margin-top:-16px;color:#7b1fa2;font-size:0.78rem;font-weight:950;line-height:1;'>0</div>",
            unsafe_allow_html=True,
        )
    with q_col:
        render_radical_control_label("q", "q", "y축 평행이동")
        if not -5.0 <= float(st.session_state.get("d8_practice_shift_q", 1.0)) <= 5.0:
            st.session_state["d8_practice_shift_q"] = 1.0
        q_value = st.slider(
            "q: y축 평행이동",
            -5.0,
            5.0,
            1.0,
            1.0,
            key="d8_practice_shift_q",
            help="q가 양수이면 위쪽, 음수이면 아래쪽으로 이동합니다.",
            label_visibility="collapsed",
        )
        st.markdown(
            "<div style='display:flex;justify-content:center;margin-top:-16px;color:#2e7d32;font-size:0.78rem;font-weight:950;line-height:1;'>0</div>",
            unsafe_allow_html=True,
        )

    show_observation = st.session_state.get("d8_practice_translation_observe", False)
    before_formula = base_practice_formula(function_type, sign_symbol, coefficient)
    after_formula = translated_practice_formula(function_type, sign_symbol, coefficient, p_value, q_value)
    before_general = general_base_formula(function_type)
    after_general = general_translated_formula(function_type)
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#f8fbff 0%,#eef7ff 100%);
            border:1px solid #dbe7f3;border-radius:12px;padding:10px 12px;margin:0 0 8px 0;">
            <div style="font-size:0.88rem;font-weight:950;color:#1565c0;margin-bottom:4px;">
                평행이동 전후의 함수식 비교
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    header_cols = st.columns([1, 0.9, 1], gap="small")
    with header_cols[0]:
        st.markdown(
            """
            <div style="background:#ffffff;border:2px dashed #b0bec5;border-radius:10px;
                padding:8px 10px;text-align:center;margin-bottom:4px;">
                <div style="font-size:0.86rem;font-weight:950;color:#546e7a;">평행이동 전</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        st.markdown(
            """
            <div style="background:#fff8e1;border:2px solid #f9a825;border-radius:10px;
                padding:8px 10px;text-align:center;margin-bottom:4px;">
                <div style="font-size:1.25rem;font-weight:950;color:#1565c0;line-height:1;">→</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_cols[2]:
        st.markdown(
            """
            <div style="background:#e3f2fd;border:2px solid #90caf9;border-radius:10px;
                padding:8px 10px;text-align:center;margin-bottom:4px;">
                <div style="font-size:0.86rem;font-weight:950;color:#0d47a1;">평행이동 후</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    general_cols = st.columns([1, 0.9, 1], gap="small")
    with general_cols[0]:
        st.latex(rf"\large {before_general}")
    with general_cols[1]:
        st.markdown(
            """
            <div style="background:#ffffff;border:1px solid #ffe082;border-radius:10px;
                min-height:48px;display:flex;align-items:center;justify-content:center;
                text-align:center;color:#e65100;font-weight:950;">
                x축 p, y축 q
            </div>
            """,
            unsafe_allow_html=True,
        )
    with general_cols[2]:
        st.latex(rf"\large {after_general}")

    actual_cols = st.columns([1, 0.9, 1], gap="small")
    with actual_cols[0]:
        st.latex(rf"\large {before_formula}")
    with actual_cols[1]:
        st.markdown(
            f"""
            <div style="background:#fffdf5;border:1px solid #ffe082;border-radius:10px;
                min-height:48px;display:flex;align-items:center;justify-content:center;
                text-align:center;color:#e65100;font-weight:950;">
                <div style="display:flex;justify-content:center;gap:8px;align-items:center;flex-wrap:wrap;">
                    <div style="background:#ffffff;border:1px solid #ffe082;border-radius:8px;padding:5px 9px;">
                        x {p_value:+g}
                    </div>
                    <div style="background:#ffffff;border:1px solid #ffe082;border-radius:8px;padding:5px 9px;">
                        y {q_value:+g}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with actual_cols[2]:
        st.latex(rf"\large {after_formula}")

    graph_col, current_formula_col = st.columns([4.6, 1.4], gap="medium")
    with graph_col:
        graph_area = st.container()
        with graph_area:
            graph_placeholder = st.empty()
            if show_observation:
                x_frames = [(float(frame_p), 0.0) for frame_p in animation_steps(0.0, float(p_value), 0.5)]
                y_frames = [(float(p_value), float(frame_q)) for frame_q in animation_steps(0.0, float(q_value), 0.5)]
                for frame_p, frame_q in x_frames + y_frames:
                    fig = draw_translated_practice_graph(
                        function_type,
                        sign_symbol,
                        coefficient,
                        frame_p,
                        frame_q,
                        show_observation=True,
                    )
                    graph_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    time.sleep(0.035)
                st.caption("그래프 전체가 먼저 x축 방향 p만큼 이동하고, 이어서 y축 방향 q만큼 이동합니다.")
                st.session_state["d8_practice_translation_observe"] = False
            else:
                fig = draw_translated_practice_graph(
                    function_type,
                    sign_symbol,
                    coefficient,
                    p_value,
                    q_value,
                    show_observation=False,
                )
                graph_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
    with current_formula_col:
        st.markdown(
            """
            <div style="background:#ffffff;border:1px solid #dbe7f3;border-radius:10px;
                padding:10px 12px;margin:0 0 8px 0;">
                <div style="font-size:0.9rem;font-weight:950;color:#1565c0;">그래프 관찰</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "▶ 평행이동 관찰하기",
            key="d8_practice_translation_observe_start",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"d8_practice_translation_observe": True}),
        )
        if st.button("관찰 결과 확인", key="d8_practice_translation_result_show", use_container_width=True):
            st.session_state["d8_practice_show_observation_result"] = True
    if st.session_state.get("d8_practice_show_observation_result"):
        st.markdown(
            translation_observation_html(function_type, sign_symbol, coefficient, p_value, q_value),
            unsafe_allow_html=True,
        )

    with st.expander("평행이동 후 도형의 형태 확인", expanded=False):
        st.markdown("**평행이동 전과 후를 비교해 봅시다.**")
        shape_answer = st.radio(
            "그래프를 x축으로 p만큼, y축으로 q만큼 평행이동하면 도형의 형태는 어떻게 될까요?",
            [
                "형태는 변하지 않고 위치만 변한다.",
                "p, q 값이 커질수록 그래프가 더 가파르게 변한다.",
                "평행이동하면 직선은 곡선으로, 곡선은 직선으로 변한다.",
            ],
            key="d8_shape_preservation_check",
            index=None,
        )
        if shape_answer == "형태는 변하지 않고 위치만 변한다.":
            st.success("맞습니다. 평행이동은 모든 점을 같은 방향, 같은 거리만큼 옮기므로 도형의 형태는 변하지 않습니다.")
        elif shape_answer:
            st.error("다시 확인해 봅시다. p와 q는 그래프의 위치를 바꾸는 값이고, 형태를 바꾸는 값은 아닙니다.")

    with st.expander("[예시] 오개념 확인", expanded=False):
        st.markdown("**오른쪽으로 p만큼 이동할 때 x 안에는 무엇을 넣을까?**")
        translation_answer = st.radio(
            "그래프 y=f(x)를 x축의 양의 방향으로 p만큼 평행이동하고, y축 방향으로 q만큼 평행이동한 식은?",
            translation_check_options,
            key="d8_translation_understanding",
            index=None,
            label_visibility="collapsed",
        )
        if translation_answer == translation_check_options[0]:
            st.success("정답입니다. 오른쪽으로 p만큼 이동하면 x 대신 x-p를 넣습니다. y축 방향 이동은 식 전체에 +q로 나타납니다.")
        elif translation_answer:
            st.error("다시 생각해 봅시다. 오른쪽으로 p만큼 이동한 그래프의 식은 x+p가 아니라 x-p를 사용합니다.")


def radical_understanding_result(answer):
    if answer == "y=f(x-p)+q":
        return "정답"
    if answer in [
        "y=f(x+p)+q",
        "y=f(x)+p+q",
    ]:
        return "오답"
    return "아직 확인하지 않았습니다."


def predict_value(x_value, params):
    y, mask = calculate_function([x_value], params)
    return None if not mask[0] or not np.isfinite(y[0]) else float(y[0])


def add_trend_ellipse(ax, x_data, y_data, label="경향성 영역"):
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return
    cov = np.cov(x_arr, y_arr)
    if not np.all(np.isfinite(cov)):
        return
    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    values = np.maximum(values, 0)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 4 * np.sqrt(values)
    ellipse = Ellipse(
        (float(np.mean(x_arr)), float(np.mean(y_arr))),
        width=float(width),
        height=float(height),
        angle=float(angle),
        facecolor="#64b5f6",
        edgecolor="#1565c0",
        linewidth=2,
        linestyle="--",
        alpha=0.16,
        label=label,
        zorder=1,
    )
    ax.add_patch(ellipse)


def prediction_trend_sentence(params):
    coefficient = float(params.get("coefficient", params.get("a", 1.0)))
    if params.get("model", "직선") == "곡선":
        if coefficient >= 0:
            return "꼭짓점을 기준으로 내려갔다가 다시 증가하는 곡선 경향을 보일 수 있습니다."
        return "꼭짓점을 기준으로 올라갔다가 다시 감소하는 곡선 경향을 보일 수 있습니다."
    if coefficient >= 0:
        return "앞으로도 증가하는 경향을 보일 것으로 예측됩니다."
    return "앞으로도 감소하는 경향을 보일 것으로 예측됩니다."


def simple_trend_label(params, x_start=None, x_end=None):
    if params.get("model", "직선") == "직선":
        return "증가하는 경향" if float(params.get("coefficient", 1.0)) >= 0 else "감소하는 경향"
    if x_start is not None and x_end is not None:
        y_values, _ = calculate_function([x_start, x_end], params)
        if np.isfinite(y_values[0]) and np.isfinite(y_values[1]):
            return "증가하는 경향" if y_values[1] >= y_values[0] else "감소하는 경향"
    return "증가하는 경향" if float(params.get("coefficient", 1.0)) >= 0 else "감소하는 경향"


def make_plot(
    x_data,
    y_data,
    params,
    new_x=None,
    predicted_y=None,
    x_label="x",
    y_label="y",
    figsize=(9.2, 5.1),
    show_data=True,
    show_function=True,
    show_prediction=True,
    loss=None,
    formula_label=None,
):
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    x_span = max(float(np.max(x_arr) - np.min(x_arr)), 1.0)
    plot_min = min(float(np.min(x_arr)), float(new_x) if new_x is not None else float(np.min(x_arr))) - x_span * 0.15
    plot_max = max(float(np.max(x_arr)), float(new_x) if new_x is not None else float(np.max(x_arr))) + x_span * 0.15
    x_line = np.linspace(plot_min, plot_max, 320)
    y_line, valid_line = calculate_function(x_line, params)

    fig, ax = plt.subplots(figsize=figsize)
    if show_data:
        ax.scatter(x_arr, y_arr, color="#1f77b4", s=80, label="실제 데이터", zorder=3)
        add_trend_ellipse(ax, x_arr, y_arr)
    if show_function:
        ax.plot(x_line[valid_line], y_line[valid_line], color="#000000", linewidth=3.6, label="함수 그래프")
    if show_prediction and predicted_y is not None:
        ax.scatter([new_x], [predicted_y], color="#2ca02c", marker="*", s=420, label="예측점", zorder=4)
        ax.annotate(
            "예측점",
            xy=(new_x, predicted_y),
            xytext=(12, 12),
            textcoords="offset points",
            color="#1b5e20",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#2ca02c", alpha=0.92),
        )
    if loss is not None:
        ax.text(
            0.94,
            0.90,
            f"손실값(MSE): {format_optional_number(loss)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            fontweight="bold",
            color="#ef6c00",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e1", edgecolor="#f9a825", alpha=0.96),
            zorder=6,
        )
    if formula_label:
        ax.text(
            0.04,
            0.94,
            formula_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor="#1565c0", alpha=0.95),
            zorder=7,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    emphasize_xy_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=13, markerscale=1.35, framealpha=0.95, borderpad=0.8, labelspacing=0.65)
    fig.tight_layout()
    return fig


def format_optional_number(value):
    return "계산 불가" if value is None else f"{value:.3f}"


def copy_numeric_params(params):
    copied = {}
    for key, value in params.items():
        if isinstance(value, str):
            copied[key] = value
        else:
            copied[key] = float(value)
    return copied


def render_u_attempt_tracker(current_params, current_loss):
    attempts = st.session_state.setdefault("d8_u_attempts", [])
    attempt_no = min(len(attempts) + 1, U_MAX_ATTEMPTS)
    status_text = f"남은 기회 {attempt_no}/{U_MAX_ATTEMPTS}" if len(attempts) < U_MAX_ATTEMPTS else f"시도 완료 {U_MAX_ATTEMPTS}/{U_MAX_ATTEMPTS}"

    tracker_col, table_col = st.columns([0.75, 1.25], gap="small")
    with tracker_col:
        st.markdown(
            f"""
            <div style="background:#fff8e1;border:1px solid #f9a825;border-radius:8px;
                padding:10px 12px;margin:2px 0 8px 0;text-align:center;">
                <div style="font-size:0.84rem;font-weight:900;color:#ef6c00;">시도 기록</div>
                <div style="font-size:1.18rem;font-weight:950;color:#263238;">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "현재 시도 기록",
            key="d8_u_record_attempt",
            disabled=len(attempts) >= U_MAX_ATTEMPTS or current_loss is None,
            use_container_width=True,
        ):
            attempts.append(
                {
                    "params": copy_numeric_params(current_params),
                    "loss": None if current_loss is None else float(current_loss),
                }
            )
            st.session_state["d8_u_attempts"] = attempts
            st.rerun()

    with table_col:
        if attempts:
            finite_losses = [
                (idx, attempt["loss"])
                for idx, attempt in enumerate(attempts)
                if attempt["loss"] is not None
            ]
            best_index = min(finite_losses, key=lambda item: item[1])[0] if finite_losses else None
            rows = []
            for idx, attempt in enumerate(attempts):
                is_best = idx == best_index
                rows.append(
                    {
                        "시도": idx + 1,
                        "함수": attempt["params"].get("model", "직선"),
                        "계수": format_optional_number(attempt["params"].get("coefficient")),
                        "p": format_optional_number(attempt["params"].get("p")),
                        "q": format_optional_number(attempt["params"].get("q")),
                        "손실": f"{format_optional_number(attempt['loss'])}{' ⭐' if is_best else ''}",
                        "_best": is_best,
                    }
                )
            display_df = pd.DataFrame(rows)

            def highlight_best(row):
                if row["_best"]:
                    return ["background-color: #fff8e1; color: #ef6c00; font-weight: 900"] * 6
                return [""] * 6

            styled_df = display_df[["시도", "함수", "계수", "p", "q", "손실"]].style.apply(
                lambda row: highlight_best(display_df.loc[row.name]),
                axis=1,
            )
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=144)
        else:
            st.caption("현재 조절한 값으로 시도를 기록하면 손실 비교표가 만들어집니다.")

    if len(attempts) >= U_MAX_ATTEMPTS:
        best_index, best_attempt = min(
            enumerate(attempts),
            key=lambda item: float("inf") if item[1]["loss"] is None else item[1]["loss"],
        )
        st.success(f"가장 작은 손실: {best_index + 1}번째 시도")
        return copy_numeric_params(best_attempt["params"])
    return current_params


def save_stage_snapshot(stage_no, title, fields):
    st.session_state[f"d8_saved_stage_{stage_no}"] = {"title": title, "fields": fields}
    st.session_state[f"d8_saved_stage_{stage_no}_time"] = pd.Timestamp.now().strftime("%H:%M:%S")
    st.success(f"{stage_no}단계 저장 완료")


def saved_stage_caption(stage_no):
    saved_time = st.session_state.get(f"d8_saved_stage_{stage_no}_time")
    st.caption(f"마지막 저장: {saved_time}" if saved_time else "아직 이 단계 결과를 저장하지 않았습니다.")


class PortfolioPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=15)
        self._font_family = "Nanum"
        self.footer_left = ""

    def header(self):
        self.set_fill_color(25, 118, 210)
        self.rect(0, 0, self.w, 22, "F")
        self.set_xy(10, 6)
        self.set_text_color(255, 255, 255)
        self.set_font(self._font_family, "", 16)
        self.cell(0, 10, "F.U.T.U.R.E. 함수 추세선 탐구 포트폴리오", ln=1, align="C")
        self.set_text_color(33, 33, 33)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.set_y(-12)
        self.set_font(self._font_family, "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, self.footer_left, 0, 0, "L")
        self.cell(0, 8, f"{self.page_no()} / {{nb}}", 0, 0, "R")

    def h2(self, text):
        self.set_fill_color(227, 242, 253)
        self.set_text_color(21, 101, 192)
        self.set_font(self._font_family, "", 12)
        self.cell(0, 8, text, ln=1, fill=True)
        self.ln(2)
        self.set_text_color(33, 33, 33)


def add_text_box_to_pdf(pdf, title, text):
    pdf.set_font(pdf._font_family, "", 11)
    pdf.set_text_color(21, 101, 192)
    pdf.cell(0, 8, title, ln=1)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font(pdf._font_family, "", 10)
    pdf.set_fill_color(245, 245, 245)
    pdf.multi_cell(0, 6, clean_text(text), border=1, fill=True)
    pdf.ln(3)


def add_figure_to_pdf(pdf, title, fig):
    tmp_path = None
    try:
        pdf.h2(title)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = tmp.name
        fig.savefig(tmp_path, format="png", dpi=180, bbox_inches="tight")
        pdf.image(tmp_path, x=12, w=pdf.w - 24)
        pdf.ln(4)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def create_portfolio_pdf(student_info, stage_rows, fig):
    pdf = PortfolioPDF()
    if os.path.exists(FONT_PATH):
        pdf.add_font("Nanum", "", FONT_PATH, uni=True)
        pdf._font_family = "Nanum"
    else:
        pdf._font_family = "Arial"
    pdf.set_font(pdf._font_family, "", 12)
    pdf.footer_left = f"{student_info.get('class', '')}반 {student_info.get('group', '')}"
    pdf.add_page()
    pdf.h2("모둠 정보")
    add_text_box_to_pdf(pdf, "반/모둠", f"{student_info.get('class', '')}반 / {student_info.get('group', '')}")
    add_text_box_to_pdf(pdf, "자료 묶음", student_info.get("dataset", ""))
    for row in stage_rows:
        pdf.h2(row["title"])
        for label, value in row["fields"]:
            add_text_box_to_pdf(pdf, label, value)
    pdf.add_page()
    add_figure_to_pdf(pdf, "최종 함수 추세선 그래프", fig)
    output = pdf.output(dest="S")
    return bytes(output) if isinstance(output, (bytes, bytearray)) else output.encode("latin1")


def build_cardnews_prompt(topic, life_view, future_text, future_question):
    if "증가" in future_text:
        trend_visual = "위 방향 화살표 1개"
    elif "감소" in future_text:
        trend_visual = "아래 방향 화살표 1개"
    else:
        trend_visual = "부드러운 방향 화살표 1개"
    return f"""GPT 이미지 생성으로 카드뉴스 2장을 만들어 주세요.

- 1080×1080px 정사각형 이미지 2장
- 각 장은 별도 이미지 파일
- 1~2분 안에 생성될 정도로 아주 단순하게 구성
- 단색 파스텔 배경, 큰 한글 글씨, 단순 일러스트 1개
- 복잡한 그래프, 많은 숫자, 긴 설명, 세밀한 배경 금지
- 아래 문구 외의 설명 문구는 추가하지 않기

### 1/2
제목: 숫자가 들려준 삶의 이야기
문구: {topic} 데이터 - {life_view}
그림: 주제와 삶의 모습을 함께 나타내는 단순 일러스트 1개

### 2/2
제목: 더 나은 미래 질문
문구: {future_question}
작은 문구: {future_text}
그림: {trend_visual}와 희망적인 상징 1개
"""


CARDNEWS_THEMES = {
    "블루": {
        "bg": "#f7fafc",
        "panel": "#ffffff",
        "primary": "#1d4ed8",
        "secondary": "#0f766e",
        "accent": "#eff6ff",
        "text": "#111827",
        "muted": "#4b5563",
    },
    "그린": {
        "bg": "#f7fdf9",
        "panel": "#ffffff",
        "primary": "#047857",
        "secondary": "#2563eb",
        "accent": "#dcfce7",
        "text": "#10231c",
        "muted": "#3f5f52",
    },
    "오렌지": {
        "bg": "#fffaf2",
        "panel": "#ffffff",
        "primary": "#c2410c",
        "secondary": "#7c3aed",
        "accent": "#ffedd5",
        "text": "#2f2a24",
        "muted": "#6b5f55",
    },
    "핑크": {
        "bg": "#fff7fb",
        "panel": "#ffffff",
        "primary": "#be185d",
        "secondary": "#7e22ce",
        "accent": "#fce7f3",
        "text": "#2b1621",
        "muted": "#6b4b5b",
    },
}


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=16)
def get_card_font(size, bold=False):
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size=size)
    try:
        return ImageFont.truetype("arial.ttf", size=size) if os.name == "nt" else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def draw_rounded(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text_to_width(draw, text, font, max_width, max_lines=None):
    words = []
    for part in str(text).replace("\n", " \n ").split(" "):
        if part:
            words.append(part)
    lines = []
    current = ""
    for word in words:
        if word == "\n":
            lines.append(current.strip())
            current = ""
            continue
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        while draw.textbbox((0, 0), lines[-1] + "...", font=font)[2] > max_width and len(lines[-1]) > 1:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "..."
    return lines


def draw_wrapped_text(draw, xy, text, font, fill, max_width, line_gap=8, max_lines=None):
    x, y = xy
    lines = wrap_text_to_width(draw, text, font, max_width, max_lines=max_lines)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def draw_centered_lines(draw, center_x, start_y, text, font, fill, max_width, line_gap=10, max_lines=None):
    lines = wrap_text_to_width(draw, text, font, max_width, max_lines=max_lines)
    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((center_x - (bbox[2] - bbox[0]) / 2, y), line, font=font, fill=fill)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def fig_to_pil(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def latex_to_pil(latex_text, color="#1d4ed8", fontsize=42):
    fig = plt.figure(figsize=(7.2, 1.35))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, f"${latex_text}$", ha="center", va="center", fontsize=fontsize, color=color)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=240, bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def make_cardnews_graph_image(x_data, y_data, params, new_x, predicted_y, x_label, y_label):
    fig = make_plot(
        x_data,
        y_data,
        params,
        new_x,
        predicted_y,
        x_label,
        y_label,
        figsize=(7.6, 4.5),
        show_data=True,
        show_function=True,
        show_prediction=True,
        loss=None,
        formula_label=function_latex(params),
    )
    try:
        return fig_to_pil(fig)
    finally:
        plt.close(fig)


def paste_contained(base, image, box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    image = image.copy()
    image.thumbnail((width, height), Image.LANCZOS)
    px = x1 + (width - image.width) // 2
    py = y1 + (height - image.height) // 2
    base.paste(image, (px, py))


def make_theme_background(theme_name, theme):
    image = Image.new("RGB", (1080, 1080), hex_to_rgb(theme["bg"]))
    draw = ImageDraw.Draw(image, "RGBA")
    grid = (*hex_to_rgb(theme["primary"]), 18)
    line = (*hex_to_rgb(theme["secondary"]), 42)
    for x in range(90, 1030, 90):
        draw.line((x, 90, x, 990), fill=grid, width=2)
    for y in range(90, 1030, 90):
        draw.line((90, y, 990, y), fill=grid, width=2)
    draw.line((120, 815, 290, 760, 455, 790, 640, 690, 875, 625), fill=line, width=14)
    return image.filter(ImageFilter.GaussianBlur(8))


def blend_background(card, background, alpha=0.38):
    return Image.blend(card, background, alpha)


def create_cardnews_images(theme_name, context, graph_image):
    theme = CARDNEWS_THEMES[theme_name]
    bg = hex_to_rgb(theme["bg"])
    panel = hex_to_rgb(theme["panel"])
    primary = hex_to_rgb(theme["primary"])
    secondary = hex_to_rgb(theme["secondary"])
    accent = hex_to_rgb(theme["accent"])
    text = hex_to_rgb(theme["text"])
    muted = hex_to_rgb(theme["muted"])

    title_font = get_card_font(58)
    subtitle_font = get_card_font(32)
    hero_font = get_card_font(42)
    big_hero_font = get_card_font(48)
    formula_title_font = get_card_font(28)
    trend_font = get_card_font(34)
    body_font = get_card_font(30)
    small_font = get_card_font(23)
    chip_font = get_card_font(24)

    cards = []
    group = context["group"]
    dataset_name = context["dataset_name"]
    y_label = context["y_label"]
    life_view = clean_text(context["life_view"])
    future_question = clean_text(context["future_question"])
    trend_text = context["trend_text"]
    function_text = context["function_text"]
    function_kind = context.get("function_kind", "함수")
    fit_judgement = clean_text(context.get("fit_judgement", ""))
    fit_reason = clean_text(context.get("fit_reason", ""))

    theme_bg = make_theme_background(theme_name, theme)

    card1 = blend_background(Image.new("RGB", (1080, 1080), bg), theme_bg)
    d1 = ImageDraw.Draw(card1, "RGBA")
    draw_rounded(d1, (54, 50, 1026, 1030), 26, (*panel, 232), outline=primary, width=3)
    d1.rounded_rectangle((54, 50, 78, 1030), radius=12, fill=(*primary, 210))
    d1.text((84, 110), "Connect to the World", font=subtitle_font, fill=muted)
    d1.text((84, 155), "CARD NEWS", font=title_font, fill=primary)
    draw_rounded(d1, (84, 250, 996, 378), 18, (*accent, 245), outline=primary, width=2)
    d1.text((112, 272), "데이터가 보여주는 삶의 모습", font=chip_font, fill=primary)
    draw_wrapped_text(d1, (112, 308), life_view, body_font, text, 840, line_gap=8, max_lines=2)
    draw_rounded(d1, (84, 405, 720, 840), 18, (255, 255, 255, 238), outline=hex_to_rgb("#dbe7f3"), width=2)
    paste_contained(card1, graph_image, (108, 434, 696, 812))
    draw_rounded(d1, (722, 405, 996, 840), 18, (*hex_to_rgb("#ffffff"), 246), outline=secondary, width=2)
    d1.text((748, 445), "그래프 해석", font=formula_title_font, fill=secondary)
    draw_wrapped_text(d1, (748, 492), f"{function_kind}\n{trend_text}", small_font, text, 220, line_gap=10, max_lines=2)
    d1.line((748, 600, 970, 600), fill=(*hex_to_rgb("#dbe7f3"), 255), width=3)
    d1.text((748, 626), "분석의 한계", font=chip_font, fill=secondary)
    draw_wrapped_text(d1, (748, 664), fit_reason, small_font, text, 220, line_gap=7, max_lines=6)
    draw_rounded(d1, (84, 850, 996, 965), 18, (*accent, 245), outline=secondary, width=2)
    d1.text((112, 872), "깊은 질문", font=chip_font, fill=secondary)
    draw_wrapped_text(d1, (112, 908), future_question, body_font, text, 840, line_gap=8, max_lines=2)
    d1.text((84, 988), f"{group} · {dataset_name}", font=small_font, fill=muted)
    cards.append(card1)

    return cards


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_cardnews_zip(card_images, base_name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, image in enumerate(card_images, start=1):
            zf.writestr(f"{base_name}_card{idx}.png", image_to_png_bytes(image))
    return buffer.getvalue()


def make_cardnews_pdf(card_images):
    rgb_images = [image.convert("RGB") for image in card_images]
    buffer = io.BytesIO()
    rgb_images[0].save(buffer, format="PDF", save_all=True, append_images=rgb_images[1:], resolution=150)
    return buffer.getvalue()


def render_link_button(url, label, gradient):
    st.markdown(
        f"""<a href="{url}" target="_blank" style="display:flex;align-items:center;justify-content:center;
        min-height:38px;padding:0 12px;background:{gradient};color:white;
        text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;">{label}</a>""",
        unsafe_allow_html=True,
    )


def render_gpt_gallery_links(class_key):
    gpt_col, gallery_col = st.columns(2)
    with gpt_col:
        render_link_button(GPT_URL, "GPT 바로가기", "linear-gradient(90deg,#10a37f,#1976d2)")
    gallery_url = GALLERY_URLS.get(str(class_key))
    with gallery_col:
        if gallery_url:
            render_link_button(gallery_url, f"{class_key}반 갤러리 패들렛", "linear-gradient(90deg,#7e57c2,#42a5f5)")
        else:
            st.info("반을 선택하면 갤러리 패들렛 버튼이 나타납니다.")


def run():
    apply_local_style()
    page_banner(
        "데이터에서 삶을 읽고 함수 추세선으로 미래 예측하기",
        "",
        "",
    )
    render_activity_flow()
    st.session_state.setdefault("d8_group", "우리 모둠")
    st.session_state.setdefault("d8_class", CLASS_OPTIONS[0])
    st.session_state.setdefault("d8_dataset", TEACHER_DEMO_DATASET)
    ensure_xy_columns(st.session_state["d8_dataset"])
    x_data, y_data, x_label, y_label = selected_xy_data(st.session_state["d8_dataset"])
    st.session_state.setdefault("d8_new_x", float(max(x_data) + (x_data[1] - x_data[0] if len(x_data) > 1 else 1)))

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
            "F.U 숫자 속 삶의 모습 발견하기",
            "UN이 제시한 모두가 더 나은 삶을 살아갈 수 있는 지속가능한 미래를 만들기 위한 공동목표(SDG) 중 관심있는 지표를 선택하고, 지표의 숫자가 커지거나 작아질 때 사람들의 삶에서 어떤 모습이 나타나는지 생각합니다.",
            "이 데이터셋의 숫자 뒤에는 어떤 삶의 모습이 담겨 있을까?",
            "#ffebee",
            "#ffcdd2",
        )
        with st.container(border=True):
            with st.expander(":orange[생각 열기]", expanded=False):
                st.markdown(
                    """
- 인공지능은 **예측값**과 **실제값**의 차이인 **오차**를 줄이는 방향으로 학습합니다.  
- 오차가 작아지도록 예측값을 조정하는 것이 인공지능 학습의 기본 원리입니다.
"""
                )
                st.latex(r"\Large \text{(오차)}=\text{(실제값)}-\text{(예측값)}")
                st.markdown(
                    """
Quick, Draw! AI가 그림을 어떻게 예측하는지 봅시다.
"""
                )
                st.link_button("Quick, Draw! 열기", "https://quickdraw.withgoogle.com/", use_container_width=True)

            class_col, group_col = st.columns([0.45, 0.55])
            with class_col:
                st.selectbox("반", CLASS_OPTIONS, key="d8_class")
            with group_col:
                st.text_input("모둠명", key="d8_group", placeholder="예: 1모둠")

            if st.session_state.get("d8_dataset") not in PUBLIC_DATASET_OPTIONS:
                st.session_state["d8_dataset"] = TEACHER_DEMO_DATASET
            dataset_name = st.session_state["d8_dataset"]
            chosen_info = DATASETS[dataset_name]
            numeric_options = ensure_xy_columns(dataset_name)
            x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)

            st.markdown(
                f"""
                <div style="
                    background:#fff8f8;
                    border:1px solid #ffcdd2;
                    border-radius:10px;
                    padding:12px 14px;
                    margin:10px 0 14px 0;
                ">
                    <div style="display:flex;gap:10px;flex-wrap:wrap;">
                        <div style="flex:1 1 180px;">
                            <div style="font-size:0.76rem;font-weight:900;color:#c62828;margin-bottom:4px;">선택한 데이터셋</div>
                            <div style="font-size:0.95rem;font-weight:800;color:#263238;">{html.escape(dataset_name)}</div>
                        </div>
                        <div style="flex:1 1 130px;">
                            <div style="font-size:0.76rem;font-weight:900;color:#c62828;margin-bottom:4px;">입력 변수 x</div>
                            <div style="font-size:0.95rem;color:#263238;">{html.escape(x_label)}</div>
                        </div>
                        <div style="flex:1 1 130px;">
                            <div style="font-size:0.76rem;font-weight:900;color:#c62828;margin-bottom:4px;">출력 변수 y</div>
                            <div style="font-size:0.95rem;color:#263238;">{html.escape(y_label)}</div>
                        </div>
                    </div>
                    <div style="margin-top:10px;font-size:0.88rem;line-height:1.5;color:#455a64;">
                        {html.escape(chosen_info.get("source", "출처 정보 없음"))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            variable_col, question_col = st.columns([1, 1])
            with variable_col:
                st.markdown(pretty_title("1. 데이터셋 및 변수 설정", "#ffebee", "#ffcdd2"), unsafe_allow_html=True)
                render_stage_card(
                    "데이터와 변수를 고릅니다",
                    "",
                    "red",
                    "문제 발견",
                )
                dataset_name = st.selectbox(
                    "탐구 데이터셋 선택",
                    PUBLIC_DATASET_OPTIONS,
                    key="d8_dataset",
                )
                chosen_info = DATASETS[dataset_name]
                numeric_options = ensure_xy_columns(dataset_name)
                variable_help = variable_help_text(numeric_options)
                st.markdown(
                    f"""
                    <div style="
                        background:#fff8f8;
                        border:1px solid #ffcdd2;
                        border-radius:8px;
                        padding:9px 12px;
                        margin:8px 0 10px 0;
                        color:#263238;
                        line-height:1.5;
                        font-size:0.92rem;
                    ">
                        입력변수 x는 <b>{html.escape(st.session_state["d8_x_col"])}</b>로 고정합니다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                y_options = [col for col in numeric_options if col != st.session_state["d8_x_col"]] or numeric_options
                if st.session_state.get("d8_y_col") not in y_options:
                    st.session_state["d8_y_col"] = y_options[0]
                st.selectbox("출력변수 y 선택", y_options, key="d8_y_col", help=variable_help)
                x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)

            with question_col:
                st.markdown(pretty_title("2. 지표 속 삶의 모습 발견하기", "#ffebee", "#ffcdd2"), unsafe_allow_html=True)
                render_stage_card(
                    "변화 경향과 삶의 모습 예상하기",
                    "지표의 값이 직접 나타내는 삶의 모습을 중심으로 작성하세요.",
                    "red",
                    "문제제기",
                )
                life_direction_options = ["증가한다", "감소한다"]
                if st.session_state.get("d8_life_direction") == "커질 때":
                    st.session_state["d8_life_direction"] = "증가한다"
                elif st.session_state.get("d8_life_direction") == "작아질 때":
                    st.session_state["d8_life_direction"] = "감소한다"
                if st.session_state.get("d8_life_direction") not in life_direction_options:
                    st.session_state["d8_life_direction"] = "증가한다"
                selected_life_direction = st.radio(
                    f"시간에 따라 출력 변수 y({y_label})가 어떻게 변할 것으로 예상하나요?",
                    life_direction_options,
                    key="d8_life_direction",
                    horizontal=True,
                )
                increase_placeholder, decrease_placeholder = life_change_placeholders("5세 미만 사망률(%) (예시)")
                selected_life_phrase = "커질 때" if selected_life_direction == "증가한다" else "작아질 때"
                selected_life_key = "d8_y_increase_life" if selected_life_direction == "증가한다" else "d8_y_decrease_life"
                selected_placeholder = increase_placeholder if selected_life_direction == "증가한다" else decrease_placeholder
                st.text_area(
                    f"예상대로 {y_label} 값이 {selected_life_phrase} 사람들의 삶에서는 어떤 모습이 나타날까요?",
                    key=selected_life_key,
                    height=120,
                    placeholder=selected_placeholder,
                )
            default_new_x = float(max(x_data) + (x_data[1] - x_data[0] if len(x_data) > 1 else 1))
            if st.session_state.get("d8_new_x_dataset") != dataset_name:
                st.session_state["d8_new_x"] = default_new_x
                st.session_state["d8_new_x_dataset"] = dataset_name
            if st.button("F.U 단계 저장", use_container_width=True):
                selected_direction, _, selected_life_text = selected_life_change(y_label)
                save_stage_snapshot(
                    1,
                    "F.U 단계: 실생활 문제와 주요 변수 발견하기",
                    [
                        ("자료 묶음", dataset_name),
                        ("선택한 두 변수", f"{x_label}, {y_label}"),
                        ("예상한 y 변화", f"시간에 따라 {y_label} 값이 {selected_direction}"),
                        ("작성한 삶의 모습", selected_life_text or "아직 작성하지 않았습니다."),
                    ],
                )
            saved_stage_caption(1)

    with tabs[1]:
        dataset_name = st.session_state["d8_dataset"]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        stage_intro(
            "T 도형의 평행이동 다시 익히기",
            "1. 도형의 방정식 - 3. 도형의 이동 - 1. 평행이동 개념을 직선과 이차함수 그래프에서 다시 확인합니다.",
            "직선과 곡선을 평행이동하면 그래프의 식과 위치는 어떻게 달라질까?",
            "#e8f5e9",
            "#c8e6c9",
        )
        with st.container(border=True):
            render_function_graph_practice()
            translation_answer = st.session_state.get("d8_translation_understanding")
            shape_answer = st.session_state.get("d8_shape_preservation_check")
            function_type = st.session_state.get("d8_practice_function_type", "직선")
            coefficient_signed = float(st.session_state.get("d8_practice_coefficient_signed", 1.0))
            if abs(coefficient_signed) < 1e-9:
                coefficient_signed = 0.1
            sign_symbol = "+" if coefficient_signed >= 0 else "-"
            coefficient = abs(coefficient_signed)
            p_value = st.session_state.get("d8_practice_shift_p", 2)
            q_value = st.session_state.get("d8_practice_shift_q", 1)
            if st.button("T 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    2,
                    "T 단계: 도형의 평행이동 다시 익히기",
                    [
                        ("핵심 내용", "1. 도형의 방정식 - 3. 도형의 이동 - 1. 평행이동"),
                        ("선택한 기본 그래프", function_type),
                        ("계수 값", f"{coefficient_signed:g}"),
                        ("x축 평행이동 p", str(p_value)),
                        ("y축 평행이동 q", str(q_value)),
                        ("현재 그래프 식", translated_practice_formula(function_type, sign_symbol, coefficient, p_value, q_value)),
                        ("형태 보존 확인", shape_answer or "아직 체크하지 않았습니다."),
                        ("오개념 확인 문제", "y=f(x)를 x축의 양의 방향으로 p만큼 평행이동하면 식은 어떻게 바뀔까요?"),
                        ("체크한 답", translation_answer or "아직 체크하지 않았습니다."),
                        ("정답 여부", radical_understanding_result(translation_answer)),
                    ],
                )
            saved_stage_caption(2)

    with tabs[2]:
        dataset_name = st.session_state["d8_dataset"]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        stage_intro(
            "U 함수 추세선으로 미래를 예측하기",
            "직선 y=m(x-p)+q 또는 곡선 y=a(x-p)^2+q를 선택하고 계수, p, q를 조절하며 손실값을 줄이고, 선택한 x값에서 y값을 예측합니다.",
            "데이터의 경향을 가장 잘 설명하는 함수 추세선은 무엇이며, 그 한계는 무엇일까?",
            "#f3e5f5",
            "#e1bee7",
        )
        with st.container(border=True):
            render_stage_card(
                "추세선을 조절해 예측합니다",
                "직선 또는 곡선을 선택하고 계수, p, q 값을 조절해 그래프가 데이터에 가까워지도록 만들고, 손실값을 비교한 뒤 선택한 x값의 y값을 예측합니다. 마지막에는 데이터 분석의 한계를 생각하며 함수 추세선이 데이터를 얼마나 잘 나타내는지 판단하고, 팩트풀니스 본능 관점으로 그 이유를 정리합니다.",
                "purple",
                "AI 이해",
            )
            attempt_context = (dataset_name, x_label, y_label, st.session_state.get("d8_u_function_type", "직선"))
            if st.session_state.get("d8_u_attempt_context") != attempt_context:
                st.session_state["d8_u_attempts"] = []
                st.session_state["d8_u_attempt_context"] = attempt_context
            params = get_parameters(x_data, y_data)
            predicted_data_y, valid_data_mask = calculate_function(x_data, params)
            loss, _ = calculate_loss(y_data, predicted_data_y, valid_data_mask)
            params = render_u_attempt_tracker(params, loss)
            predicted_data_y, valid_data_mask = calculate_function(x_data, params)
            loss, _ = calculate_loss(y_data, predicted_data_y, valid_data_mask)
            st.session_state["d8_params"] = params
            default_new_x = float(max(x_data) + (x_data[1] - x_data[0] if len(x_data) > 1 else 1))
            if "d8_new_x" not in st.session_state:
                st.session_state["d8_new_x"] = default_new_x
            st.session_state["d8_new_x"] = float(round(st.session_state["d8_new_x"]))
            new_x = float(st.session_state.get("d8_new_x", default_new_x))
            predicted_y = predict_value(float(new_x), params)
            render_estimated_function_strip(params)
            toggle_cols = st.columns(3, gap="small")
            with toggle_cols[0]:
                show_data = st.checkbox("데이터", value=True, key="d8_show_graph_data")
            with toggle_cols[1]:
                show_function = st.checkbox("함수 그래프", value=True, key="d8_show_graph_function")
            with toggle_cols[2]:
                show_prediction = st.checkbox("예측점", value=True, key="d8_show_graph_prediction")
            trend_fig = make_plot(
                x_data,
                y_data,
                params,
                float(new_x),
                predicted_y,
                x_label,
                y_label,
                figsize=(9.2, 5.1),
                show_data=show_data,
                show_function=show_function,
                show_prediction=show_prediction,
                loss=loss,
            )
            st.pyplot(trend_fig, use_container_width=True)
            plt.close(trend_fig)
            input_col, value_col = st.columns([1, 1])
            with input_col:
                st.markdown(
                    f"""
                    <div class="prediction-input-card">
                        <div class="prediction-input-title">예측할 {html.escape(str(x_label))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                new_x = st.number_input(
                    f"예측할 x값({x_label})",
                    key="d8_new_x",
                    step=1.0,
                    format="%.0f",
                    help=f"그래프에서 예측하고 싶은 독립 변수 {x_label} 값을 입력합니다.",
            )
            predicted_y = predict_value(float(new_x), params)
            with value_col:
                if predicted_y is not None:
                    trend_sentence = prediction_trend_sentence(params)
                    st.markdown(
                        f"""
                        <div class="stage-card stage-card-purple">
                            <div class="stage-kicker">선택한 함수에서의 예측값</div>
                            <div class="prediction-input-title">예측값 y({html.escape(str(y_label))})</div>
                            <div class="stage-card-help">
                                <div style="margin-top:8px;background:#ffffff;border:1px solid #e1bee7;
                                    border-radius:10px;padding:10px 12px;color:#6a1b9a;font-size:1.05rem;
                                    font-weight:900;text-align:center;">
                                    <div style="font-size:1.35rem;color:#4a148c;margin-bottom:4px;">
                                        {float(new_x):g}{'년' if str(x_label) == '연도' else ''} → 약 {predicted_y:.2f}
                                    </div>
                                    <div style="font-size:0.95rem;color:#455a64;font-weight:800;line-height:1.45;">
                                        {html.escape(trend_sentence)}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("선택한 함수로 예측값을 계산할 수 없습니다.")

            with st.expander("데이터 분석의 한계", expanded=False):
                st.markdown(
                    """
                    <div class="fit-eval-box" style="background:#f8fbff;border:1px solid #dbe7f3;border-radius:8px;
                        padding:10px 12px;margin:0 0 10px 0;color:#37474f;font-weight:800;">
                        손실값과 그래프 모양을 함께 보고 선택한 함수 추세선이 데이터를 얼마나 잘 나타내는지 판단합니다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fit_judgement = st.radio(
                    "**1. 우리가 만든 함수는?**",
                    [
                        "비교적 잘 나타낸다.",
                        "일부 구간에서 차이가 난다.",
                        "선택한 함수로 나타내기 어렵다.",
                    ],
                    key="d8_fit_judgement",
                    index=None,
                    horizontal=True,
                )
                fit_reason = ""
                fit_lens = ""
                if fit_judgement:
                    lens_col, reason_col = st.columns(2)
                    with lens_col:
                        fit_lens = st.selectbox(
                            "**2. 이유를 쓸 때 참고할 팩트풀니스 본능 관점을 선택하세요.**",
                            list(FACTFULNESS_LENS_GUIDES.keys()),
                            key="d8_fit_factfulness_lens",
                            index=list(FACTFULNESS_LENS_GUIDES.keys()).index("직선 본능 점검"),
                        )
                        lens_info = FACTFULNESS_LENS_GUIDES[fit_lens]
                        st.info(f"{fit_lens}: {lens_info['guide']}")
                    with reason_col:
                        fit_reason = st.text_area(
                            f"**3. {fit_lens} 관점으로 그렇게 판단한 이유를 작성하세요.**",
                            key="d8_fit_reason_text",
                            height=145,
                            placeholder=lens_info["placeholder"],
                        )
    
            function_text = build_function_text(params)
    
            if st.button("U 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    3,
                    "U 단계: 함수 추세선으로 예측값 찾기",
                    [
                        ("사용한 함수", params.get("model", "직선")),
                        ("함수식", function_text),
                        ("손실값", format_optional_number(loss)),
                        (f"예측할 x값({x_label}) / 예측값 y({y_label})", f"x={float(new_x):g}, y={predicted_y:.2f}" if predicted_y is not None else "계산 불가"),
                        ("적합성 평가", st.session_state.get("d8_fit_judgement", "")),
                        ("참고한 팩트풀니스 관점", st.session_state.get("d8_fit_factfulness_lens", "")),
                        ("평가 이유", fit_reason.strip() if fit_reason.strip() else "아직 작성하지 않았습니다."),
                    ],
                )
            saved_stage_caption(3)

    with tabs[3]:
        dataset_name = st.session_state["d8_dataset"]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        params = st.session_state.get("d8_params") or fit_default_params(x_data, y_data)
        predicted_data_y, valid_data_mask = calculate_function(x_data, params)
        loss, _ = calculate_loss(y_data, predicted_data_y, valid_data_mask)
        predicted_y = predict_value(float(st.session_state.get("d8_new_x", max(x_data))), params)

        stage_intro(
            "R.E 데이터 속 삶과 미래 고민하기",
            "함수 추세선으로 예측한 데이터의 변화 경향을 바탕으로 숫자가 들려주는 삶의 모습을 이해하고, 더 나은 미래를 함께 고민하는 카드뉴스를 만들어 봅시다.",
            "예측 결과를 보고 미래의 삶의 모습에 대해 어떤 질문을 할 수 있을까?",
            "#e3f2fd",
            "#bbdefb",
        )
        with st.container(border=True):
            selected_direction, _, selected_life_text = selected_life_change(y_label)
            reference_rows = [
                ("탐구 데이터", dataset_name),
                ("살펴본 변화", f"{x_label}이/가 달라질 때 {y_label}의 변화"),
                ("F.U에서 예상한 y 변화", f"시간에 따라 {y_label} 값이 {selected_direction}"),
                ("F.U에서 작성한 삶의 모습", selected_life_text or "아직 작성하지 않았습니다."),
                ("예측한 변화 경향", prediction_trend_sentence(params)),
                ("앞에서 판단한 흐름", clean_text(st.session_state.get("d8_fit_judgement", ""))),
            ]
            st.markdown(pretty_title("앞 단계 활동 자료 요약", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(reference_rows, columns=["활동 자료", "내용"]),
                use_container_width=True,
                hide_index=True,
                height=250,
            )

            life_col, future_col = st.columns(2)
            with life_col:
                st.markdown(pretty_title("1. 데이터가 보여 주는 삶의 모습", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
                render_stage_card(
                    "삶의 의미와 질문을 정리합니다",
                    "예측한 변화 경향을 바탕으로 데이터가 사람들의 삶에 대해 무엇을 말해 주는지 생각해 봅시다.",
                    "blue",
                    "삶의 모습 이해하기",
                )
                life_example = trend_based_life_example(y_label, prediction_trend_sentence(params))
                life_view = st.text_area(
                    "이 데이터가 보여 주는 삶의 모습은 무엇인가요?",
                    key="d8_life_view",
                    height=120,
                    placeholder=f"예: {life_example}",
                ).strip()

            with future_col:
                st.markdown(pretty_title("2. 깊은 질문 만들기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
                render_stage_card(
                    "깊은 질문 만들기",
                    "데이터가 보여 주는 삶의 모습을 바탕으로 더 나은 미래를 함께 고민하는 질문을 만들어 봅시다.",
                    "blue",
                    "더 나은 미래 고민하기",
                )
                future_question = st.text_area(
                    "더 나은 미래를 위해 함께 생각해 볼 깊은 질문을 적어 봅시다.",
                    key="d8_future_question",
                    height=120,
                    placeholder="예: 모든 아이들이 건강하게 성장하는 세상을 만들 수 있을까?",
                )

            future_text = prediction_trend_sentence(params)
            cardnews_topic = "5세 미만 아동 사망률" if y_label == "5세 미만 사망률(%) (예시)" else y_label
            default_life_view = trend_based_life_example(y_label, future_text)

            st.markdown(pretty_title("3. 카드뉴스 직접 제작", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            render_stage_card(
                "발표용 카드뉴스 1장이 완성됩니다",
                "앞 단계에서 작성한 삶의 모습, 깊은 질문, U 단계 그래프와 데이터 분석의 한계를 한 장에 배치합니다.",
                "blue",
                "카드뉴스 제작",
            )
            theme_names = list(CARDNEWS_THEMES.keys())
            selected_theme = st.session_state.get("d8_cardnews_theme", theme_names[0])
            theme_cols = st.columns(len(theme_names), gap="small")
            for idx, theme_name in enumerate(theme_names):
                with theme_cols[idx]:
                    if st.button(
                        theme_name,
                        key=f"d8_cardnews_theme_{idx}",
                        type="primary" if selected_theme == theme_name else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["d8_cardnews_theme"] = theme_name
                        selected_theme = theme_name

            new_x_for_card = float(st.session_state.get("d8_new_x", max(x_data)))
            predicted_y_for_card = predict_value(new_x_for_card, params)
            card_context = {
                "group": st.session_state.get("d8_group", "우리 모둠"),
                "dataset_name": dataset_name,
                "y_label": y_label,
                "life_view": life_view or default_life_view,
                "future_question": future_question.strip() or "이 변화가 계속된다면 우리는 어떤 선택을 해야 할까?",
                "trend_text": simple_trend_label(params, float(np.min(x_data)), new_x_for_card),
                "function_text": build_function_text(params),
                "function_latex": function_latex(params).replace("f(x)=", "y="),
                "function_kind": "직선" if params.get("model", "직선") == "직선" else "곡선(이차함수)",
                "fit_judgement": st.session_state.get("d8_fit_judgement", "아직 판단하지 않았습니다."),
                "fit_reason": st.session_state.get("d8_fit_reason_text", "데이터의 일부 구간에서는 실제값과 추세선 사이에 차이가 있을 수 있습니다."),
            }
            card_context_key = (
                selected_theme,
                tuple(card_context.items()),
                tuple(float(value) for value in params.values() if not isinstance(value, str)),
                params.get("model", ""),
                new_x_for_card,
            )
            if st.button("카드뉴스 만들기", key="d8_make_cardnews", use_container_width=True):
                graph_image = make_cardnews_graph_image(
                    x_data,
                    y_data,
                    params,
                    new_x_for_card,
                    predicted_y_for_card,
                    x_label,
                    y_label,
                )
                st.session_state["d8_cardnews_images"] = create_cardnews_images(selected_theme, card_context, graph_image)
                st.session_state["d8_cardnews_context"] = card_context_key
            elif st.session_state.get("d8_cardnews_context") != card_context_key:
                st.session_state.pop("d8_cardnews_images", None)

            card_images = st.session_state.get("d8_cardnews_images")
            if card_images:
                st.image(card_images[0], caption="발표용 카드뉴스", use_container_width=True)
                card_pdf_col, gallery_share_col = st.columns(2, gap="medium")
                with card_pdf_col:
                    st.download_button(
                        "카드뉴스 PDF 저장",
                        data=make_cardnews_pdf(card_images),
                        file_name=f"{st.session_state.get('d8_group', '우리모둠')}_카드뉴스.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                with gallery_share_col:
                    class_key = str(st.session_state.get("d8_class", CLASS_OPTIONS[0]))
                    gallery_url = GALLERY_URLS.get(class_key)
                    if gallery_url:
                        render_link_button(gallery_url, f"{class_key}반 갤러리 패들렛 공유", "linear-gradient(90deg,#1565c0,#26a69a)")
                    else:
                        st.caption("반을 선택하면 갤러리 패들렛 공유 버튼이 나타납니다.")
            else:
                st.caption("카드뉴스 만들기를 누르면 미리보기와 다운로드 버튼이 나타납니다.")

            if st.button("R.E 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    4,
                    "R.E 단계: 데이터 속 삶의 모습과 더 나은 미래 고민하기",
                    [
                        ("삶의 모습", life_view),
                        ("고민", future_question.strip()),
                    ],
                )
            saved_stage_caption(4)
    
            stage_rows = []
            for idx in range(1, 5):
                row = st.session_state.get(f"d8_saved_stage_{idx}")
                if row:
                    stage_rows.append(row)
            pdf_col, portfolio_col = st.columns(2)
            with pdf_col:
                if stage_rows:
                    pdf_context = (
                        dataset_name,
                        x_label,
                        y_label,
                        float(st.session_state.get("d8_new_x", max(x_data))),
                        tuple((row["title"], tuple(row["fields"])) for row in stage_rows),
                    )
                    if st.session_state.get("d8_pdf_context") != pdf_context:
                        st.session_state.pop("d8_pdf_bytes", None)
                        st.session_state["d8_pdf_context"] = pdf_context
                    if st.button("PDF 만들기", key="d8_create_pdf", use_container_width=True):
                        final_fig = make_plot(
                            x_data,
                            y_data,
                            params,
                            float(st.session_state.get("d8_new_x", max(x_data))),
                            predicted_y,
                            x_label,
                            y_label,
                        )
                        try:
                            st.session_state["d8_pdf_bytes"] = create_portfolio_pdf(
                                {
                                    "class": st.session_state.get("d8_class", ""),
                                    "group": st.session_state.get("d8_group", ""),
                                    "student_id": "",
                                    "dataset": dataset_name,
                                },
                                stage_rows,
                                final_fig,
                            )
                        finally:
                            plt.close(final_fig)
                    pdf_bytes = st.session_state.get("d8_pdf_bytes")
                    if pdf_bytes:
                        st.download_button(
                            "PDF 저장",
                            data=pdf_bytes,
                            file_name=f"{st.session_state.get('d8_group', '우리모둠')}_함수추세선탐구.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.caption("각 단계 저장 후 PDF를 만들 수 있습니다.")
            with portfolio_col:
                class_key = str(st.session_state.get("d8_class", CLASS_OPTIONS[0]))
                portfolio_url = PORTFOLIO_URLS.get(class_key)
                if portfolio_url:
                    render_link_button(portfolio_url, f"{class_key}반 포트폴리오 패들렛", "linear-gradient(90deg,#1565c0,#26a69a)")
                else:
                    st.caption("반을 선택하면 포트폴리오 패들렛 버튼이 나타납니다.")


if __name__ == "__main__":
    st.set_page_config(page_title="함수 추세선 탐구", layout="centered")
    run()
