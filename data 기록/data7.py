# 실행 명령: streamlit run data7.py

import html
import os
import tempfile
import time

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF


# matplotlib 한글 표시 설정: 프로젝트의 NanumGothic 글꼴을 우선 사용합니다.
try:
    font_path = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rc("font", family=font_name)
    else:
        mpl.rc("font", family="Malgun Gothic")
    mpl.rc("axes", unicode_minus=False)
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
        .st-key-d8_practice_radical_sign_plus button,
        .st-key-d8_practice_radical_sign_minus button,
        .st-key-d8_u_radical_sign_plus button,
        .st-key-d8_u_radical_sign_minus button {
            font-weight: 950;
            min-height: 52px;
            padding: 0.2rem 0.4rem;
        }
        .st-key-d8_practice_radical_sign_plus button p,
        .st-key-d8_practice_radical_sign_minus button p,
        .st-key-d8_u_radical_sign_plus button p,
        .st-key-d8_u_radical_sign_minus button p {
            font-size: 2rem;
            font-weight: 950;
            line-height: 1;
            margin: 0;
        }
        .st-key-d8_practice_radical_point_start button {
            background: linear-gradient(135deg, #1565c0 0%, #26a69a 100%);
            border: 0;
            border-radius: 10px;
            box-shadow: 0 8px 18px rgba(21, 101, 192, 0.24);
            color: #ffffff;
            font-weight: 900;
            min-height: 44px;
        }
        .st-key-d8_practice_radical_point_start button:hover {
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
        ("2", "T", "수학의 언어", "무리함수 그래프의 모양 관찰하기", "#e8f5e9", "#c8e6c9", "#2e7d32"),
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
    radicand = float(params["a"]) * (x_arr - params["p"])
    mask = radicand >= 0
    y_arr = np.full_like(x_arr, np.nan, dtype=float)
    sign = 1.0 if float(params.get("sign", 1.0)) >= 0 else -1.0
    y_arr[mask] = sign * np.sqrt(radicand[mask]) + params["q"]
    return y_arr, mask


def fit_default_params(x_data, y_data):
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    x_span = max(float(np.max(x) - np.min(x)), 1.0)
    candidates = np.linspace(float(np.min(x) - x_span), float(np.min(x) - 1e-6), 40)
    best_params, best_loss = None, float("inf")
    for p in candidates:
        basis = np.sqrt(np.maximum(x - p, 0))
        a, q = np.linalg.lstsq(np.column_stack([basis, np.ones_like(x)]), y, rcond=None)[0]
        pred = a * basis + q
        loss = float(np.mean((y - pred) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_params = {"sign": 1.0 if a >= 0 else -1.0, "a": float(max(a * a, 1e-6)), "p": float(p), "q": float(q)}
    return best_params or {"sign": 1.0, "a": 1.0, "p": float(min(x) - 1), "q": float(min(y))}


def get_parameters(x_data, y_data):
    defaults = fit_default_params(x_data, y_data)
    slider_context = (
        round(float(min(x_data)), 6),
        round(float(max(x_data)), 6),
        round(float(min(y_data)), 6),
        round(float(max(y_data)), 6),
    )
    if st.session_state.get("d8_slider_context") != slider_context:
        slider_keys = ["d8_u_radical_sign", "d8_무리함수_a"]
        for key in slider_keys:
            st.session_state.pop(key, None)
        st.session_state["d8_slider_context"] = slider_context
    params = {}
    default_sign = "+" if float(defaults.get("sign", 1.0)) >= 0 else "-"
    st.session_state.setdefault("d8_u_radical_sign", default_sign)
    if st.session_state["d8_u_radical_sign"] not in ["+", "-"]:
        st.session_state["d8_u_radical_sign"] = default_sign

    formula_col, sign_col, a_col = st.columns([1.2, 0.8, 1.5], gap="small")
    with formula_col:
        render_radical_shifted_formula_html(
            sign_text="±",
            a_text="a",
            p_text="- p",
            q_text="+ q",
            title="일반화된 함수",
            extra_class="u-radical-formula",
        )
    with sign_col:
        render_radical_control_label("sign", "±", "근호 앞의 부호")
        sign_button_cols = st.columns(2, gap="small")
        with sign_button_cols[0]:
            if st.button(
                r"\+",
                key="d8_u_radical_sign_plus",
                type="primary" if st.session_state["d8_u_radical_sign"] == "+" else "secondary",
                use_container_width=True,
            ):
                st.session_state["d8_u_radical_sign"] = "+"
        with sign_button_cols[1]:
            if st.button(
                r"\-",
                key="d8_u_radical_sign_minus",
                type="primary" if st.session_state["d8_u_radical_sign"] == "-" else "secondary",
                use_container_width=True,
            ):
                st.session_state["d8_u_radical_sign"] = "-"
    with a_col:
        a_default = float(max(defaults["a"], 1e-6))
        a_half = max(a_default * 0.45, 0.01)
        a_min = float(max(a_default - a_half, 1e-6))
        a_max = float(a_default + a_half)
        initial_a = float(min(max(a_default * 0.85, a_min), a_max))
        current_a = float(st.session_state.get("d8_무리함수_a", initial_a))
        if not (a_min <= current_a <= a_max):
            st.session_state.pop("d8_무리함수_a", None)
        render_radical_control_label("a", "a", "근호 안 x의 계수")
        params["a"] = st.slider(
            "a: x의 계수 (a>0)",
            a_min,
            a_max,
            initial_a,
            float(max(a_half / 50, 1e-4)),
            key="d8_무리함수_a",
            label_visibility="collapsed",
        )
    params["sign"] = 1.0 if st.session_state["d8_u_radical_sign"] == "+" else -1.0
    params["p"] = float(defaults["p"])
    params["q"] = float(defaults["q"])
    return params


def calculate_loss(actual_y, predicted_y, valid_mask):
    actual = np.asarray(actual_y, dtype=float)
    pred = np.asarray(predicted_y, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred)
    if not np.any(mask):
        return None, 0
    return float(np.mean((actual[mask] - pred[mask]) ** 2)), int(np.sum(mask))


def build_function_text(params):
    sign_text = "+" if float(params.get("sign", 1.0)) >= 0 else "-"
    return f"f(x) = {sign_text}sqrt({params['a']:.2f}(x - {params['p']:.2f})) + {params['q']:.2f}"


def render_estimated_function_strip(params):
    st.markdown(
        rf"""
> **추세선(무리함수)** &nbsp;&nbsp; $\large {function_latex(params)}$
"""
    )


def signed_latex_number(value):
    return f"+ {abs(value):.2f}" if value >= 0 else f"- {abs(value):.2f}"


def function_latex(params):
    a = float(params["a"])
    p = float(params["p"])
    q = float(params["q"])
    sign_text = "+" if float(params.get("sign", 1.0)) >= 0 else "-"
    return rf"f(x)={sign_text}\sqrt{{{a:.2f}(x {signed_latex_number(-p)})}} {signed_latex_number(q)}"


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
    x_values = np.linspace(0.0, 10.0, 400)
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
        "무리함수 모양을 살펴봅니다",
        "y=±√(ax)에서 근호 앞의 부호와 a를 조절하며 그래프의 증가·감소와 변화 정도를 살펴봅니다.",
        "green",
        "개념 탐구",
    )
    function_col, sign_col, slider_col = st.columns([1.2, 0.8, 1.5], gap="small")
    with function_col:
        render_radical_formula_html(title="일반화된 함수")
    radical_check_options = [
        "일정하다.",
        "일정하지 않다.",
    ]
    if st.session_state.get("d8_radical_understanding") not in radical_check_options:
        st.session_state["d8_radical_understanding"] = None
    with sign_col:
        render_radical_control_label("sign", "±", "근호 앞의 부호")
        legacy_sign = st.session_state.get("d8_practice_radical_sign")
        if legacy_sign in ["+ 증가형", "+: 증가형"]:
            st.session_state["d8_practice_radical_sign"] = "+"
        elif legacy_sign in ["- 감소형", "-: 감소형"]:
            st.session_state["d8_practice_radical_sign"] = "-"
        st.session_state.setdefault("d8_practice_radical_sign", "+")
        sign_button_cols = st.columns(2, gap="small")
        with sign_button_cols[0]:
            if st.button(
                r"\+",
                key="d8_practice_radical_sign_plus",
                type="primary" if st.session_state["d8_practice_radical_sign"] == "+" else "secondary",
                use_container_width=True,
            ):
                st.session_state["d8_practice_radical_sign"] = "+"
        with sign_button_cols[1]:
            if st.button(
                r"\-",
                key="d8_practice_radical_sign_minus",
                type="primary" if st.session_state["d8_practice_radical_sign"] == "-" else "secondary",
                use_container_width=True,
            ):
                st.session_state["d8_practice_radical_sign"] = "-"
        sign_label = st.session_state["d8_practice_radical_sign"]
    with slider_col:
        render_radical_control_label("a", "a", "x의 계수")
        a_value = st.slider(
            "a: 근호안의 x의 계수",
            5,
            10,
            5,
            1,
            key="d8_practice_radical_a_positive",
            help="a는 x에 곱해지는 양수입니다. a가 커질수록 시작점 근처에서 더 빠르게 변합니다.",
            label_visibility="collapsed",
        )
    sign_symbol = "+" if str(sign_label).startswith("+") else "-"

    graph_col, current_formula_col = st.columns([4.6, 1.4], gap="medium")
    animate_point = st.session_state.get("d8_practice_radical_point_animate", False)
    with graph_col:
        graph_area = st.container()
        with graph_area:
            if animate_point:
                graph_placeholder = st.empty()
                for point_x in range(0, 11, 2):
                    fig = draw_radical_practice_graph(
                        sign_symbol,
                        a_value,
                        point_x=float(point_x),
                        step_size=2.0,
                    )
                    graph_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    time.sleep(0.5)
                st.session_state["d8_practice_radical_point_animate"] = False
            else:
                fig = draw_radical_practice_graph(sign_symbol, a_value, point_x=0.0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    with current_formula_col:
        st.markdown("**현재 그래프**")
        st.latex(rf"\Large y={sign_symbol}\sqrt{{{int(a_value)}x}}")
        st.button(
            "▶ 점 P의 높이 변화 관찰하기",
            key="d8_practice_radical_point_start",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"d8_practice_radical_point_animate": True}),
        )

    with st.expander("[예시] 오개념 확인", expanded=False):
        st.markdown("**x가 같은 간격으로 증가할때 y의 변화량?**")
        radical_answer = st.radio(
            "x가 같은 만큼 증가하면 함숫값도 항상 같은 만큼 변할까요?",
            radical_check_options,
            key="d8_radical_understanding",
            index=None,
            label_visibility="collapsed",
        )
        if radical_answer == "일정하지 않다.":
            st.success("정답입니다. 무리함수는 곡선이므로 x가 같은 간격으로 증가해도 y의 변화량은 일정하지 않습니다.")
        elif radical_answer == "일정하다.":
            st.error("다시 생각해 봅시다. 그래프가 직선이 아니기 때문에 같은 x 간격에서도 y의 변화량은 달라집니다.")


def radical_understanding_result(answer):
    if answer == "일정하지 않다.":
        return "정답"
    if answer == "일정하다.":
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
    if float(params.get("sign", 1.0)) >= 0:
        return "앞으로도 증가하는 경향을 보일 것으로 예측됩니다."
    return "앞으로도 감소하는 경향을 보일 것으로 예측됩니다."


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
):
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    x_span = max(float(np.max(x_arr) - np.min(x_arr)), 1.0)
    plot_min = min(float(np.min(x_arr)), float(new_x) if new_x is not None else float(np.min(x_arr))) - x_span * 0.15
    plot_max = max(float(np.max(x_arr)), float(new_x) if new_x is not None else float(np.max(x_arr))) + x_span * 0.15
    x_line = np.linspace(plot_min, plot_max, 500)
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
    return {key: float(value) for key, value in params.items()}


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
                        "a 값": format_optional_number(attempt["params"].get("a")),
                        "손실": f"{format_optional_number(attempt['loss'])}{' ⭐' if is_best else ''}",
                        "_best": is_best,
                    }
                )
            display_df = pd.DataFrame(rows)

            def highlight_best(row):
                if row["_best"]:
                    return ["background-color: #fff8e1; color: #ef6c00; font-weight: 900"] * 3
                return [""] * 3

            styled_df = display_df[["시도", "a 값", "손실"]].style.apply(
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
    if os.path.exists(font_path):
        pdf.add_font("Nanum", "", font_path, uni=True)
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
        "데이터에서 삶을 읽고 무리함수로 미래 예측하기",
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
            "T 무리함수 그래프 특징 탐구하기",
            "근호 앞의 부호와 a를 조절하며 그래프의 증가·감소와 변화 정도를 살펴봅니다.",
            "무리함수의 그래프를 관찰하면 어떤 특징을 발견할 수 있을까?",
            "#e8f5e9",
            "#c8e6c9",
        )
        with st.container(border=True):
            render_function_graph_practice()
            radical_answer = st.session_state.get("d8_radical_understanding")
            if st.button("T 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    2,
                    "T 단계: 무리함수 그래프 특징 탐구하기",
                    [
                        ("탐구한 함수", r"y=±√(ax)"),
                        ("근호 앞의 부호", st.session_state.get("d8_practice_radical_sign", "")),
                        ("a 값", str(st.session_state.get("d8_practice_radical_a_positive", ""))),
                        ("오개념 확인 문제", "x가 같은 만큼 증가하면 함숫값도 항상 같은 만큼 변할까요?"),
                        ("체크한 답", radical_answer or "아직 체크하지 않았습니다."),
                        ("정답 여부", radical_understanding_result(radical_answer)),
                    ],
                )
            saved_stage_caption(2)

    with tabs[2]:
        dataset_name = st.session_state["d8_dataset"]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        stage_intro(
            "U 무리함수로 미래를 예측하기",
            "무리함수 y=±√(a(x-p))+q에서 근호 앞의 부호와 양수 a를 조절하며 손실값을 줄이고, 선택한 x값에서 y값을 예측합니다.",
            "데이터의 경향을 가장 잘 설명하는 무리함수는 무엇이며, 그 한계는 무엇일까?",
            "#f3e5f5",
            "#e1bee7",
        )
        with st.container(border=True):
            render_stage_card(
                "추세선을 조절해 예측합니다",
                "근호 앞의 부호와 양수 a 값을 조절해 그래프가 데이터에 가까워지도록 만들고, 손실값을 비교한 뒤 선택한 x값의 y값을 예측합니다. 마지막에는 데이터 분석의 한계를 생각하며 무리함수가 데이터를 얼마나 잘 나타내는지 판단하고, 팩트풀니스 본능 관점으로 그 이유를 정리합니다.",
                "purple",
                "AI 이해",
            )
            attempt_context = (dataset_name, x_label, y_label)
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
            st.pyplot(
                make_plot(
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
                ),
                use_container_width=True,
            )
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
                    st.warning("선택한 함수의 정의역 때문에 예측값을 계산할 수 없습니다.")

            with st.expander("데이터 분석의 한계", expanded=False):
                st.markdown(
                    """
                    <div class="fit-eval-box" style="background:#f8fbff;border:1px solid #dbe7f3;border-radius:8px;
                        padding:10px 12px;margin:0 0 10px 0;color:#37474f;font-weight:800;">
                        손실값과 그래프 모양을 함께 보고 무리함수가 데이터를 얼마나 잘 나타내는지 판단합니다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fit_judgement = st.radio(
                    "**1. 우리가 만든 함수는?**",
                    [
                        "비교적 잘 나타낸다.",
                        "일부 구간에서 차이가 난다.",
                        "무리함수로 나타내기 어렵다.",
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
                    "U 단계: 무리함수 그래프로 예측값 찾기",
                    [
                        ("사용한 함수", "무리함수"),
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
            "무리함수로 예측한 데이터의 변화 경향을 바탕으로 숫자가 들려주는 삶의 모습을 이해하고, 더 나은 미래를 함께 고민하는 카드뉴스를 만들어 봅시다.",
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
            canva_prompt = build_cardnews_prompt(
                cardnews_topic,
                life_view or "아직 작성하지 않았습니다.",
                future_text,
                future_question.strip() or "아직 작성하지 않았습니다.",
            )

            st.markdown(pretty_title("3. GPT 프롬프트", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            render_stage_card(
                "앞에서 작성한 내용을 자동으로 조합합니다",
                "학생이 추가로 입력하지 않아도 GPT에 넣을 카드뉴스 제작 프롬프트가 완성됩니다.",
                "blue",
                "자동 생성",
            )
            if st.button("프롬프트 생성", key="d8_generate_cardnews_prompt", use_container_width=True):
                st.session_state["d8_show_cardnews_prompt"] = True
            if st.session_state.get("d8_show_cardnews_prompt", False):
                st.code(canva_prompt, language="markdown")

            st.markdown(pretty_title("4. 우리 모둠의 카드뉴스 공유", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            render_gpt_gallery_links(st.session_state.get("d8_class", CLASS_OPTIONS[0]))

            if st.button("R.E 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    4,
                    "R.E 단계: 데이터 속 삶의 모습과 더 나은 미래 고민하기",
                    [
                        ("삶의 모습", life_view),
                        ("고민", future_question.strip()),
                        ("GPT 프롬프트", canva_prompt),
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
                    final_fig = make_plot(x_data, y_data, params, float(st.session_state.get("d8_new_x", max(x_data))), predicted_y, x_label, y_label)
                    pdf_bytes = create_portfolio_pdf(
                        {
                            "class": st.session_state.get("d8_class", ""),
                            "group": st.session_state.get("d8_group", ""),
                            "student_id": "",
                            "dataset": dataset_name,
                        },
                        stage_rows,
                        final_fig,
                    )
                    st.download_button("PDF 저장", data=pdf_bytes, file_name=f"{st.session_state.get('d8_group', '우리모둠')}_함수추세선탐구.pdf", mime="application/pdf", use_container_width=True)
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
