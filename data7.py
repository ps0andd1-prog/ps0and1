# 실행 명령: streamlit run data7.py

import html
import os
import tempfile

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
                "기대수명(년)": [64.0, 65.2, 67.650, 69.111, 70.683, 71.966, 72.183, 73.330],
                "5세 미만 사망률": [9.3, 8.5, 7.668, 6.233, 5.059, 4.288, 3.916, 3.833],
                "극빈곤 인구 비율(%)": [38.0, 36.8, 36.205, 28.330, 20.984, 13.422, 11.412, 10.618],
            },
        ),
        "default_x": "연도",
        "default_y": "기대수명(년)",
        "source": "World Bank, UN IGME, Our World in Data",
        "source_url": "https://ourworldindata.org/grapher/under-5-mortality-rate-sdgs",
        "deep_question": "보건 지표가 좋아지는 흐름 속에서도 아직 남아 있는 취약성은 어떻게 말할 수 있을까?",
        "factfulness_lens": "부정 본능 점검",
    },
    "인간: 교육과 기초 역량": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023, 2024],
            {
                "성인 문해율(%)": [75.0, 78.0, 81.120, 82.570, 84.370, 85.960, 86.980, 87.580, 87.740],
                "DTP3 접종률(%)": [75.0, 70.0, 72.0, 77.0, 83.0, 85.0, 83.0, 84.0, 85.0],
            },
        ),
        "default_x": "연도",
        "default_y": "성인 문해율(%)",
        "source": "UNESCO, WHO/UNICEF, Our World in Data",
        "source_url": "https://ourworldindata.org/grapher/cross-country-literacy-rates",
        "deep_question": "평균 문해율이 높아져도 교육 기회의 격차는 어디에 남아 있을까?",
        "factfulness_lens": "일반화 본능 점검",
    },
    "번영: 디지털 접근과 도시 변화": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023],
            {
                "인터넷 이용률(%)": [0.05, 0.8, 6.720, 15.600, 28.500, 39.900, 59.300, 67.400],
                "전기 접근률(%)": [71.0, 74.5, 78.221, 80.701, 83.447, 86.924, 90.396, 91.599],
                "도시화율(%)": [43.0, 45.0, 46.836, 49.368, 51.778, 54.470, 56.503, 57.354],
            },
        ),
        "default_x": "연도",
        "default_y": "인터넷 이용률(%)",
        "source": "World Bank, Our World in Data",
        "source_url": "https://fred.stlouisfed.org/series/ITNETUSERP2WLD",
        "deep_question": "디지털 접근이 늘어날수록 정보 격차는 정말 줄어들까?",
        "factfulness_lens": "일반화 본능 점검",
    },
    "번영: 한국 인구와 노동": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023, 2024],
            {
                "합계출산율(명)": [1.57, 1.65, 1.480, 1.085, 1.226, 1.239, 0.837, 0.721, 0.748],
                "65세 이상 인구 비율(%)": [5.1, 5.9, 7.091, 9.000, 10.992, 12.955, 15.822, 18.335, 19.274],
                "청년실업률(%)": [7.5, 8.5, 10.002, 9.321, 8.540, 9.899, 10.143, 5.405, 6.434],
            },
        ),
        "default_x": "연도",
        "default_y": "65세 이상 인구 비율(%)",
        "source": "World Bank WDI / FRED",
        "source_url": "https://fred.stlouisfed.org/series/SPPOP65UPTOZSKOR",
        "deep_question": "인구 구조의 변화는 미래의 돌봄과 일자리 문제를 어떻게 바꿀까?",
        "factfulness_lens": "일반화 본능 점검",
    },
    "환경: 에너지 전환과 탄소": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2020],
            {
                "재생에너지 비중(%)": [7.0, 7.1, 7.426, 7.192, 8.370, 10.000, 12.781],
                "1인당 CO2 배출량": [5.8, 7.9, 9.408, 10.422, 12.187, 12.439, 11.524],
            },
        ),
        "default_x": "연도",
        "default_y": "재생에너지 비중(%)",
        "source": "Our World in Data",
        "source_url": "https://ourworldindata.org/grapher/renewable-share-energy",
        "deep_question": "재생에너지 비중이 늘어나는 속도는 기후 위기 대응에 충분할까?",
        "factfulness_lens": "직선 본능 점검",
    },
    "환경: 대기질과 도시 생활": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2020],
            {
                "PM2.5 평균 노출": [30.0, 27.5, 25.438, 24.474, 21.791, 26.290, 25.944],
                "1인당 CO2 배출량": [5.8, 7.9, 9.408, 10.422, 12.187, 12.439, 11.524],
            },
        ),
        "default_x": "연도",
        "default_y": "PM2.5 평균 노출",
        "source": "Our World in Data / World Bank",
        "source_url": "https://ourworldindata.org/grapher/average-exposure-pm25-pollution",
        "deep_question": "대기질 지표의 변화는 시민의 건강과 생활 격차에 어떤 영향을 줄까?",
        "factfulness_lens": "직선 본능 점검",
    },
    "평화: 안전과 폭력 감소": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021, 2022, 2023],
            {
                "고의살인율": [7.4, 7.1, 6.9, 6.4, 6.1, 5.7, 5.6, 5.8, 5.8, 5.6, 5.5],
                "분쟁 관련 사망률": [3.3, 2.8, 2.1, 1.4, 1.2, 2.3, 1.0, 0.9, 1.5, 2.1, 2.4],
            },
        ),
        "default_x": "연도",
        "default_y": "고의살인율",
        "source": "UNODC, UN SDG Global Database, Our World in Data",
        "source_url": "https://unstats.un.org/sdgs/dataportal",
        "deep_question": "안전 지표가 좋아져도 특정 시기와 지역의 불안은 어떻게 드러날까?",
        "factfulness_lens": "직선 본능 점검",
    },
    "평화: 난민과 강제이주": {
        "table": make_yearly_table(
            [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021, 2022, 2023],
            {
                "강제이주민 수(백만 명)": [38.0, 36.0, 37.3, 37.5, 43.7, 65.1, 79.5, 82.4, 89.3, 108.4, 117.3],
                "난민 수(백만 명)": [17.2, 14.9, 15.9, 13.5, 15.4, 21.3, 26.0, 26.4, 27.1, 35.3, 37.6],
            },
        ),
        "default_x": "연도",
        "default_y": "강제이주민 수(백만 명)",
        "source": "UNHCR Global Trends",
        "source_url": "https://www.unhcr.org/global-trends",
        "deep_question": "강제이주민 수의 증가는 국제 사회의 책임을 어떻게 묻고 있을까?",
        "factfulness_lens": "부정 본능 점검",
    },
}

DATASET_GROUPS = {
    "인간": ["인간: 보건과 삶의 질", "인간: 교육과 기초 역량"],
    "번영": ["번영: 디지털 접근과 도시 변화", "번영: 한국 인구와 노동"],
    "환경": ["환경: 에너지 전환과 탄소", "환경: 대기질과 도시 생활"],
    "평화": ["평화: 안전과 폭력 감소", "평화: 난민과 강제이주"],
}

FUNCTION_OPTIONS = ["일차함수", "이차함수", "유리함수", "무리함수"]
CLASS_OPTIONS = ["1", "2", "5", "6"]
GALLERY_URLS = {
    "1": "https://padlet.com/ps0andd/g_1",
    "2": "https://padlet.com/ps0andd/g_2",
    "5": "https://padlet.com/ps0andd/g_5",
    "6": "https://padlet.com/ps0andd/g_6",
}
PORTFOLIO_URLS = GALLERY_URLS.copy()
CANVA_AI_URL = "https://www.canva.com/ai"
APP_OUTPUT_TYPE = "정보 카드"
TARGET_USERS = ["고등학생", "중학생", "학부모", "지역사회", "정책 결정자", "일반 대중"]
APP_FEATURES = [
    "핵심 예측값 카드",
    "산점도와 추세선 설명",
    "오차와 손실값 안내",
    "보간·외삽 주의 문구",
    "삶과 연결하는 질문 제시",
    "실천 메시지 강조",
    "사용자 선택 버튼",
    "퀴즈 또는 미션",
]
FACTFULNESS_LENSES = {
    "부정 본능 점검": "나쁜 변화만 강조하지 않고 좋아진 점과 아직 남은 문제를 함께 봅니다.",
    "직선 본능 점검": "최근 흐름이 앞으로도 같은 속도로 계속된다고 단정하지 않고, 변화 속도와 구간을 확인합니다.",
    "일반화 본능 점검": "평균이나 전체 흐름이 모든 지역과 사람에게 똑같이 적용된다고 말하지 않습니다.",
    "격차 본능 점검": "세상을 둘로만 나누지 않고 중간 단계와 다양한 차이를 함께 봅니다.",
}


def clean_text(value, default="아직 작성하지 않았습니다."):
    text = str(value).strip() if value is not None else ""
    return text if text else default


def pretty_title(text, color1, color2):
    return f"""
    <div style='background:linear-gradient(90deg,{color1} 0%,{color2} 100%);
        border-radius:12px;padding:7px 16px 2px 16px;margin:12px 0 10px 0;'>
        <h4 style='margin-top:0;'><b>{text}</b></h4>
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
            background: linear-gradient(135deg, #e3f2fd 0%, #fff8e1 100%);
            border: 2px solid #1976d2;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 8px 18px rgba(21, 101, 192, 0.16);
            margin: 8px 0 10px 0;
        }
        .prediction-input-kicker {
            color: #ef6c00;
            font-size: 0.82rem;
            font-weight: 900;
            margin-bottom: 3px;
        }
        .prediction-input-title {
            color: #0d47a1;
            font-size: 1.18rem;
            font-weight: 900;
            margin-bottom: 5px;
        }
        .prediction-input-help {
            color: #37474f;
            font-size: 0.94rem;
            line-height: 1.55;
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
        .summary-chip-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin: 8px 0 12px 0;
        }
        .summary-chip {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(25,118,210,0.16);
            border-radius: 12px;
            padding: 10px 12px;
        }
        .summary-label {
            color: #607d8b;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .summary-value {
            color: #1f2937;
            font-size: 0.98rem;
            font-weight: 900;
        }
        .param-meaning-card {
            min-height: 96px;
            display: flex;
            flex-direction: column;
            justify-content: center;
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


def render_summary_chips(items, variant="blue"):
    chip_html = "".join(
        f"""
        <div class="summary-chip">
            <div class="summary-label">{html.escape(str(label))}</div>
            <div class="summary-value">{html.escape(str(value))}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(
        f"""
        <div class="stage-card stage-card-{html.escape(variant)}">
            <div class="summary-chip-grid">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_banner(title, description, question):
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
            border-radius:20px;padding:22px 24px;border:1px solid #dbe7f3;margin-bottom:14px;">
            <div style="font-size:0.9rem;font-weight:700;color:#5e35b1;margin-bottom:8px;">F.U.T.U.R.E. 프로젝트</div>
            <div style="font-size:1.75rem;font-weight:800;color:#1f2937;margin-bottom:8px;">{title}</div>
            <div style="font-size:1rem;line-height:1.7;color:#37474f;">{description}</div>
            {question_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_intro(title, description, question, color1="#e8f5e9", color2="#c8e6c9"):
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{color1} 0%,{color2} 100%);
            border-radius:18px;padding:18px 20px;border:1px solid rgba(0,0,0,0.06);margin-bottom:12px;">
            <div style="font-size:1.05rem;font-weight:800;color:#1f2937;margin-bottom:8px;">{title}</div>
            <div style="font-size:0.97rem;line-height:1.7;color:#37474f;margin-bottom:12px;">{description}</div>
            <div style="background:rgba(255,255,255,0.72);border-radius:12px;padding:10px 12px;
                border:1px solid rgba(255,255,255,0.85);color:#37474f;line-height:1.6;">
                <b>탐구 질문</b><br>{question}
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
        "연도": "자료가 관찰되거나 기록된 해입니다. 시간에 따른 변화를 볼 때 x로 자주 사용합니다.",
        "인터넷 이용률(%)": "전체 인구 중 인터넷을 사용하는 사람의 비율입니다.",
        "전기 접근성(%)": "전체 인구 중 전기를 사용할 수 있는 사람의 비율입니다.",
        "도시화율(%)": "전체 인구 중 도시에 사는 사람의 비율입니다.",
        "성인 문해율(%)": "성인 중 글을 읽고 쓸 수 있는 사람의 비율입니다.",
        "기대수명(년)": "태어난 사람이 평균적으로 몇 년 정도 살 것으로 예상되는지를 나타내는 값입니다.",
        "5세 미만 사망률(출생 100명당)": "태어난 아이 100명 중 5세가 되기 전에 사망하는 아이의 수입니다.",
        "DTP3 접종률(%)": "디프테리아, 파상풍, 백일해 예방접종을 3회까지 마친 아동의 비율입니다.",
        "극빈곤 인구 비율(%)": "매우 낮은 소득으로 생활하는 인구의 비율입니다.",
        "세계 재생에너지 비중(%)": "세계 에너지 소비 중 재생에너지가 차지하는 비율입니다.",
        "한국 1인당 CO₂ 배출량(톤)": "한국에서 한 사람이 평균적으로 배출한 이산화탄소의 양입니다.",
        "한국 PM2.5 평균 노출(μg/m³)": "한국 사람들이 평균적으로 노출되는 초미세먼지 농도입니다.",
        "한국 합계출산율(명)": "여성 한 명이 평생 낳을 것으로 예상되는 평균 자녀 수입니다.",
        "한국 65세 이상 인구 비율(%)": "한국 전체 인구 중 65세 이상 인구가 차지하는 비율입니다.",
        "한국 청년실업률(%)": "일할 의사와 능력이 있지만 일자리를 구하지 못한 청년의 비율입니다.",
    }
    return meanings.get(column_name, f"{column_name} 변수입니다. 값이 커지거나 작아질 때 다른 변수와 어떤 관계가 있는지 살펴봅니다.")


def variable_help_text(columns):
    return "\n".join(f"- {column}: {variable_meaning(column)}" for column in columns)


def dataset_group_names():
    return list(DATASET_GROUPS.keys())


def datasets_for_group(group_name):
    return [name for name in DATASET_GROUPS.get(group_name, []) if name in DATASETS]


def group_for_dataset(dataset_name):
    for group_name, names in DATASET_GROUPS.items():
        if dataset_name in names:
            return group_name
    return dataset_group_names()[0]


def ensure_xy_columns(dataset_name):
    info = DATASETS[dataset_name]
    columns = numeric_columns(info)
    default_x = info.get("default_x", columns[0])
    default_y = info.get("default_y", columns[1] if len(columns) > 1 else columns[0])
    if default_x not in columns:
        default_x = columns[0]
    if default_y not in columns or default_y == default_x:
        default_y = next((col for col in columns if col != default_x), default_x)
    if st.session_state.get("d8_xy_dataset") != dataset_name:
        st.session_state["d8_x_col"] = default_x
        st.session_state["d8_y_col"] = default_y
        st.session_state["d8_xy_dataset"] = dataset_name
    if st.session_state.get("d8_x_col") not in columns:
        st.session_state["d8_x_col"] = default_x
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


def calculate_function(function_type, x_values, params):
    x_arr = np.asarray(x_values, dtype=float)
    if function_type == "일차함수":
        return params["a"] * x_arr + params["b"], np.ones_like(x_arr, dtype=bool)
    if function_type == "이차함수":
        return params["a"] * (x_arr - params["p"]) ** 2 + params["q"], np.ones_like(x_arr, dtype=bool)
    if function_type == "유리함수":
        denom = x_arr - params["p"]
        mask = np.abs(denom) > 1e-8
        y_arr = np.full_like(x_arr, np.nan, dtype=float)
        y_arr[mask] = params["a"] / denom[mask] + params["q"]
        return y_arr, mask
    radicand = x_arr - params["p"]
    mask = radicand >= 0
    y_arr = np.full_like(x_arr, np.nan, dtype=float)
    y_arr[mask] = params["a"] * np.sqrt(radicand[mask]) + params["q"]
    return y_arr, mask


def fit_default_params(function_type, x_data, y_data):
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    if function_type == "일차함수":
        a, b = np.polyfit(x, y, 1)
        return {"a": float(a), "b": float(b)}
    if function_type == "이차함수":
        coef = np.polyfit(x, y, 2)
        a = float(coef[0]) if abs(coef[0]) > 1e-8 else 0.01
        p = float(-coef[1] / (2 * a))
        q = float(np.polyval(coef, p))
        return {"a": a, "p": p, "q": q}
    if function_type == "유리함수":
        x_span = max(float(np.max(x) - np.min(x)), 1.0)
        candidates = np.concatenate(
            [
                np.linspace(float(np.min(x) - x_span), float(np.min(x) - x_span * 0.05), 20),
                np.linspace(float(np.max(x) + x_span * 0.05), float(np.max(x) + x_span), 20),
            ]
        )
        best_params, best_loss = None, float("inf")
        for p in candidates:
            basis = 1 / (x - p)
            if not np.all(np.isfinite(basis)):
                continue
            a, q = np.linalg.lstsq(np.column_stack([basis, np.ones_like(x)]), y, rcond=None)[0]
            pred = a * basis + q
            loss = float(np.mean((y - pred) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_params = {"a": float(a), "p": float(p), "q": float(q)}
        return best_params or {"a": float((max(y) - min(y)) * max(max(x) - min(x), 1)), "p": float(min(x) - 1), "q": float(np.mean(y))}
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
            best_params = {"a": float(a), "p": float(p), "q": float(q)}
    return best_params or {"a": float((max(y) - min(y)) / max(np.sqrt(max(x) - min(x)), 1)), "p": float(min(x) - 1), "q": float(min(y))}


def narrow_slider(label, center, half_width, key):
    half_width = max(float(half_width), 1e-6)
    step = max(half_width / 50, 1e-4)
    return st.slider(
        label,
        float(center - half_width),
        float(center + half_width),
        float(center),
        float(step),
        key=key,
    )


def shifted_initial(center, half_width, direction=1.0):
    return float(center + direction * half_width * 0.45)


def render_parameter_slider(function_type, name, label, min_value, max_value, value, step, key):
    selected_value = st.slider(label, min_value, max_value, value, step, key=key)
    st.markdown(
        f"""
        <div class="param-meaning-card" style="background:#f8fbff;border:1px solid #dbeafe;border-radius:10px;
            padding:10px 12px;margin:10px 0 4px 0;">
            <div style="font-size:0.84rem;font-weight:900;color:#1565c0;margin-bottom:3px;">변수의 의미</div>
            <div style="font-size:0.94rem;line-height:1.5;color:#37474f;">
                <b>{html.escape(name)}</b>: {html.escape(param_role_text(function_type, name))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected_value


def get_parameters(function_type, x_data, y_data):
    defaults = fit_default_params(function_type, x_data, y_data)
    st.session_state["d8_optimal_params"] = defaults
    x_span = max(float(max(x_data) - min(x_data)), 1.0)
    y_span = max(float(max(y_data) - min(y_data)), 1.0)
    slider_context = (
        function_type,
        round(float(min(x_data)), 6),
        round(float(max(x_data)), 6),
        round(float(min(y_data)), 6),
        round(float(max(y_data)), 6),
    )
    if st.session_state.get("d8_slider_context") != slider_context:
        slider_keys = ["d8_linear_a", "d8_linear_b"]
        for option in FUNCTION_OPTIONS:
            slider_keys.extend([f"d8_{option}_a", f"d8_{option}_p", f"d8_{option}_q"])
        for key in slider_keys:
            st.session_state.pop(key, None)
        st.session_state["d8_slider_context"] = slider_context
    params = {}
    if function_type == "일차함수":
        a_half = max(abs(defaults["a"]) * 0.25, y_span / x_span * 0.12)
        b_half = max(abs(defaults["b"]) * 0.05, y_span * 0.12)
        slider_specs = [
            (
                "a",
                "a: 기울기",
                float(defaults["a"] - a_half),
                float(defaults["a"] + a_half),
                shifted_initial(defaults["a"], a_half, 1),
                float(max(a_half / 50, 1e-4)),
                "d8_linear_a",
            ),
            (
                "b",
                "b: y절편",
                float(defaults["b"] - b_half),
                float(defaults["b"] + b_half),
                shifted_initial(defaults["b"], b_half, -1),
                float(max(b_half / 50, 1e-4)),
                "d8_linear_b",
            ),
        ]
    else:
        if function_type == "이차함수":
            a_half = max(abs(defaults["a"]) * 0.35, y_span / (x_span**2) * 0.08)
        elif function_type == "유리함수":
            a_half = max(abs(defaults["a"]) * 0.30, y_span * x_span * 0.08)
        else:
            a_half = max(abs(defaults["a"]) * 0.30, y_span / max(np.sqrt(x_span), 1.0) * 0.10)
        p_half = x_span * 0.12
        q_half = y_span * 0.12
        slider_specs = [
            (
                "a",
                "a: 변화 방향과 세기",
                float(defaults["a"] - a_half),
                float(defaults["a"] + a_half),
                shifted_initial(defaults["a"], a_half, 1),
                float(max(a_half / 50, 1e-4)),
                f"d8_{function_type}_a",
            ),
            (
                "p",
                "p: 좌우 이동",
                float(defaults["p"] - p_half),
                float(defaults["p"] + p_half),
                shifted_initial(defaults["p"], p_half, -1),
                float(max(p_half / 50, 1e-4)),
                f"d8_{function_type}_p",
            ),
            (
                "q",
                "q: 위아래 이동",
                float(defaults["q"] - q_half),
                float(defaults["q"] + q_half),
                shifted_initial(defaults["q"], q_half, 1),
                float(max(q_half / 50, 1e-4)),
                f"d8_{function_type}_q",
            ),
        ]

    slider_cols = st.columns(len(slider_specs))
    for col, (name, label, min_value, max_value, value, step, key) in zip(slider_cols, slider_specs):
        with col:
            params[name] = render_parameter_slider(
                function_type,
                name,
                label,
                min_value,
                max_value,
                value,
                step,
                key,
            )
    return params


def calculate_loss(actual_y, predicted_y, valid_mask):
    actual = np.asarray(actual_y, dtype=float)
    pred = np.asarray(predicted_y, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred)
    if not np.any(mask):
        return None, 0
    return float(np.mean((actual[mask] - pred[mask]) ** 2)), int(np.sum(mask))


def calculate_error_metrics(actual_y, predicted_y, valid_mask):
    actual = np.asarray(actual_y, dtype=float)
    pred = np.asarray(predicted_y, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred)
    residuals = np.full_like(actual, np.nan, dtype=float)
    if not np.any(mask):
        return {"mse": None, "mae": None, "rmse": None, "r2": None, "valid_count": 0, "mask": mask, "residuals": residuals}
    residuals[mask] = actual[mask] - pred[mask]
    mse = float(np.mean(residuals[mask] ** 2))
    mae = float(np.mean(np.abs(residuals[mask])))
    rmse = float(np.sqrt(mse))
    sst = float(np.sum((actual[mask] - np.mean(actual[mask])) ** 2))
    r2 = None if sst == 0 else float(1 - np.sum(residuals[mask] ** 2) / sst)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "valid_count": int(np.sum(mask)), "mask": mask, "residuals": residuals}


def build_function_text(function_type, params):
    if function_type == "일차함수":
        return f"f(x) = {params['a']:.2f}x + {params['b']:.2f}"
    if function_type == "이차함수":
        return f"f(x) = {params['a']:.2f}(x - {params['p']:.2f})^2 + {params['q']:.2f}"
    if function_type == "유리함수":
        return f"f(x) = {params['a']:.2f} / (x - {params['p']:.2f}) + {params['q']:.2f}"
    return f"f(x) = {params['a']:.2f}sqrt(x - {params['p']:.2f}) + {params['q']:.2f}"


def function_general_latex(function_type):
    if function_type == "일차함수":
        return r"f(x)=ax+b"
    if function_type == "이차함수":
        return r"f(x)=a(x-p)^2+q"
    if function_type == "유리함수":
        return r"f(x)=\frac{a}{x-p}+q"
    return r"f(x)=a\sqrt{x-p}+q"


def signed_latex_number(value):
    return f"+ {abs(value):.2f}" if value >= 0 else f"- {abs(value):.2f}"


def function_latex(function_type, params):
    a = float(params["a"])
    if function_type == "일차함수":
        b = float(params["b"])
        return rf"f(x)={a:.2f}x {signed_latex_number(b)}"
    p = float(params["p"])
    q = float(params["q"])
    if function_type == "이차함수":
        return rf"f(x)={a:.2f}(x {signed_latex_number(-p)})^2 {signed_latex_number(q)}"
    if function_type == "유리함수":
        return rf"f(x)=\frac{{{a:.2f}}}{{x {signed_latex_number(-p)}}} {signed_latex_number(q)}"
    return rf"f(x)={a:.2f}\sqrt{{x {signed_latex_number(-p)}}} {signed_latex_number(q)}"


def render_function_formula_panel(function_type, params):
    st.markdown(
        """
        <div style="background:#f8fbff;border:1px solid #bbdefb;border-radius:16px;
            padding:14px 16px;margin:10px 0;">
            <div style="font-weight:900;color:#1565c0;">함수식 확인</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**일반화된 함수**")
    st.latex(function_general_latex(function_type))
    st.markdown("**학생이 추정한 함수**")
    st.latex(function_latex(function_type, params))


def param_role_text(function_type, name):
    roles = {
        "일차함수": {
            "a": "기울기입니다. 값이 커지면 그래프가 더 가파르게 올라가고, 작아지면 더 완만해집니다.",
            "b": "y절편입니다. 값이 커지면 그래프 전체가 위로, 작아지면 아래로 이동합니다.",
        },
        "이차함수": {
            "a": "굽은 방향과 폭입니다. 절댓값이 커지면 그래프가 좁아지고, 작아지면 넓어집니다.",
            "p": "꼭짓점의 좌우 위치입니다. 값이 커지면 꼭짓점이 오른쪽으로 이동합니다.",
            "q": "꼭짓점의 위아래 위치입니다. 값이 커지면 그래프 전체가 위로 이동합니다.",
        },
        "유리함수": {
            "a": "가지가 벌어지는 정도와 방향입니다. 절댓값이 커지면 점근선에서 더 급하게 변합니다.",
            "p": "세로 점근선의 좌우 위치입니다. 값이 커지면 점근선이 오른쪽으로 이동합니다.",
            "q": "가로 점근선의 위아래 위치입니다. 값이 커지면 그래프 전체가 위로 이동합니다.",
        },
        "무리함수": {
            "a": "증가 또는 감소 방향과 변화의 세기입니다. 절댓값이 커지면 더 빠르게 변합니다.",
            "p": "그래프가 시작되는 지점의 좌우 위치입니다. 값이 커지면 시작점이 오른쪽으로 이동합니다.",
            "q": "그래프가 시작되는 지점의 위아래 위치입니다. 값이 커지면 그래프 전체가 위로 이동합니다.",
        },
    }
    return roles.get(function_type, {}).get(name, "그래프의 모양을 조정하는 값입니다.")


def render_loss_card(current_loss):
    current_loss_text = format_optional_number(current_loss)
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#fff8e1 0%,#ffecb3 100%);
            border:2px solid #f9a825;border-radius:12px;padding:14px 16px;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <div style="font-size:0.9rem;font-weight:800;color:#ef6c00;">현재 손실값(MSE)</div>
                <span title="손실값은 현재 함수 그래프와 실제 데이터의 차이를 수치로 나타낸 값입니다."
                    style="display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;
                    border-radius:50%;background:#ffffff;color:#ef6c00;border:1px solid #f9a825;
                    font-size:0.82rem;font-weight:900;cursor:help;">?</span>
            </div>
            <div style="font-size:2rem;font-weight:900;color:#1f2937;line-height:1;">{current_loss_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_value(function_type, x_value, params):
    y, mask = calculate_function(function_type, [x_value], params)
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


def prediction_range_label(new_x, x_data):
    x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
    if x_min <= float(new_x) <= x_max:
        return "관찰 범위 안", f"x={float(new_x):g}는 실제 데이터에서 본 x 범위 [{x_min:g}, {x_max:g}] 안에 있습니다."
    return "관찰 범위 밖", f"x={float(new_x):g}는 실제 데이터에서 본 x 범위 [{x_min:g}, {x_max:g}] 밖에 있습니다. 그래프를 밖으로 이어서 예상하는 값이므로 더 조심해서 해석해야 합니다."


def make_plot(x_data, y_data, function_type, params, new_x=None, predicted_y=None, x_label="x", y_label="y", figsize=(8, 4.8)):
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    x_span = max(float(np.max(x_arr) - np.min(x_arr)), 1.0)
    plot_min = min(float(np.min(x_arr)), float(new_x) if new_x is not None else float(np.min(x_arr))) - x_span * 0.15
    plot_max = max(float(np.max(x_arr)), float(new_x) if new_x is not None else float(np.max(x_arr))) + x_span * 0.15
    x_line = np.linspace(plot_min, plot_max, 500)
    y_line, valid_line = calculate_function(function_type, x_line, params)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_arr, y_arr, color="#1f77b4", s=80, label="실제 데이터", zorder=3)
    add_trend_ellipse(ax, x_arr, y_arr)
    ax.plot(x_line[valid_line], y_line[valid_line], color="#d62728", linewidth=2.4, label="함수 그래프")
    if predicted_y is not None:
        ax.scatter([new_x], [predicted_y], color="#2ca02c", marker="*", s=220, label="예측점", zorder=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_residual_table(x_data, actual_y, predicted_y, valid_mask):
    rows = []
    for x_value, actual, pred, ok in zip(x_data, actual_y, predicted_y, valid_mask):
        if ok and np.isfinite(pred):
            residual = actual - pred
            rows.append({"x": x_value, "실제 y": actual, "함수값 f(x)": pred, "잔차": residual, "잔차제곱": residual**2})
        else:
            rows.append({"x": x_value, "실제 y": actual, "함수값 f(x)": np.nan, "잔차": np.nan, "잔차제곱": np.nan})
    return pd.DataFrame(rows)


def make_residual_plot(x_data, residuals, valid_mask, x_label):
    x_arr = np.asarray(x_data, dtype=float)
    res = np.asarray(residuals, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(res)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhline(0, color="#555", linestyle="--", linewidth=1)
    ax.scatter(x_arr[mask], res[mask], color="#7b1fa2", s=70)
    ax.set_xlabel(x_label)
    ax.set_ylabel("잔차")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def format_optional_number(value):
    return "계산 불가" if value is None else f"{value:.3f}"


def selected_factfulness_text():
    lens = st.session_state.get("d8_factfulness_lens", "직선 본능 점검")
    return f"{lens}: {FACTFULNESS_LENSES.get(lens, '')}"


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


def render_data_preview(dataset_name, x_data, y_data, x_label, y_label):
    col1, col2 = st.columns([1.45, 0.75])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6.2))
        ax.scatter(x_data, y_data, color="#1f77b4", s=80, label="실제 데이터", zorder=3)
        add_trend_ellipse(ax, x_data, y_data)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        st.pyplot(fig, use_container_width=True)
    with col2:
        st.dataframe(DATASETS[dataset_name]["table"], use_container_width=True, hide_index=True, height=240)
        source_text = DATASETS[dataset_name].get("source", "출처 정보 없음")
        source_url = DATASETS[dataset_name].get("source_url")
        st.markdown(
            f"""
            <div class="stage-card stage-card-blue">
                <div class="stage-kicker">자료 출처</div>
                <div class="stage-card-help">{html.escape(source_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if source_url:
            st.markdown(f"[원자료 또는 참고 자료 열기]({source_url})")


def build_canva_prompt(group_name, dataset_name, prediction_question, function_type, function_text, loss_text, prediction_text, deep_question, limitation, action_message, ds_reflection, design_tone):
    return f"""Canva AI에서 만들 정보 카드 기획안

모둠명: {group_name}
데이터 주제: {dataset_name}
예측 질문: {prediction_question}
깊은 질문: {deep_question}
함수 종류: {function_type}
함수식: {function_text}
손실값(MSE): {loss_text}
예측 결과: {prediction_text}
디자인 분위기: {design_tone}

주의해서 보여 줄 점:
{limitation}

우리 모둠의 생각:
{ds_reflection}

실천 메시지:
{action_message}

요청:
예측값을 단정적으로 보여 주지 말고, 데이터 범위와 한계를 함께 설명하는 교육용 정보 카드로 만들어 주세요.
"""


def render_link_button(url, label, gradient):
    st.markdown(
        f"""<a href="{url}" target="_blank" style="display:flex;align-items:center;justify-content:center;
        min-height:38px;padding:0 12px;background:{gradient};color:white;
        text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;">{label}</a>""",
        unsafe_allow_html=True,
    )


def render_canva_gallery_links(class_key):
    canva_col, gallery_col = st.columns(2)
    with canva_col:
        render_link_button(CANVA_AI_URL, "Canva AI 바로가기", "linear-gradient(90deg,#00c4cc,#7d2ae8)")
    gallery_url = GALLERY_URLS.get(str(class_key))
    with gallery_col:
        if gallery_url:
            render_link_button(gallery_url, f"{class_key}반 갤러리 패들렛", "linear-gradient(90deg,#7e57c2,#42a5f5)")
        else:
            st.info("반을 선택하면 갤러리 패들렛 버튼이 나타납니다.")


def run():
    st.set_page_config(page_title="함수 추세선 탐구", layout="centered")
    apply_local_style()
    page_banner(
        "질문으로 깨우고 함수로 예측하는 데이터 탐구",
        "실생활 데이터를 함수 그래프로 표현하고, 오차와 예측 한계를 근거로 해석한 뒤 정보 콘텐츠로 공유합니다.",
        "",
    )
    dataset_names = list(DATASETS.keys())
    group_names = dataset_group_names()
    st.session_state.setdefault("d8_group", "우리 모둠")
    st.session_state.setdefault("d8_class", CLASS_OPTIONS[0])
    st.session_state.setdefault("d8_dataset", dataset_names[0])
    st.session_state.setdefault("d8_dataset_group", group_for_dataset(st.session_state["d8_dataset"]))
    st.session_state.setdefault("d8_function", FUNCTION_OPTIONS[0])
    st.session_state.setdefault("d8_prediction_question", "선택한 데이터에서 x값이 달라지면 y값은 어떻게 변할까?")
    st.session_state.setdefault("d8_deep_question", "")

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
            "문제제기: 실생활 데이터는 어떤 문제를 보여줄까?",
            "실생활 데이터가 어떤 사람과 사회의 문제를 담고 있는지 발견하고, 분석할 x와 y를 정해 예측 질문을 만듭니다.",
            "실생활 데이터는 어떤 문제를 보여줄까?",
            "#e3f2fd",
            "#bbdefb",
        )
        with st.container(border=True):
            with st.expander("생각 열기", expanded=False):
                st.markdown(
                    """
인공지능은 **예측값**과 **실제값**의 차이인 **오차**를 줄이는 방향으로 학습합니다.  
오차가 작아지도록 함수값이나 매개변수를 조정하는 것이 인공지능 학습의 기본 원리입니다.
"""
                )
                st.latex(r"\text{오차}=\text{실제값}-\text{예측값}")
                st.markdown(
                    """
Quick, Draw! 교사 시연을 보며 AI가 그림을 어떻게 예측하는지 떠올려 봅시다.  
AI가 틀렸다면 실제 제시어와 AI의 예측 사이에 오차가 생긴 것입니다.
"""
                )
                st.link_button("Quick, Draw! 열기", "https://quickdraw.withgoogle.com/", use_container_width=True)

            st.markdown(pretty_title("1. 데이터 분석 방향 정하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            render_stage_card(
                "모둠과 분석할 자료를 정합니다",
                "UN SDG는 전 세계가 함께 해결하려는 지속가능발전목표입니다. 반, 모둠명, SDG 주제, 자료 묶음, x/y 변수를 차례대로 선택하면 이후 단계의 그래프와 예측이 자동으로 이어집니다.",
                "blue",
                "F.U 준비",
            )
            class_col, group_col = st.columns([0.45, 0.55])
            with class_col:
                st.selectbox("반", CLASS_OPTIONS, key="d8_class")
            with group_col:
                st.text_input("모둠명", key="d8_group", placeholder="예: 1모둠")
            topic_col, dataset_col = st.columns([0.8, 1.2])
            with topic_col:
                selected_group = st.selectbox("UN SDG 주제 선택", group_names, key="d8_dataset_group")
            field_dataset_names = datasets_for_group(selected_group)
            if st.session_state.get("d8_dataset") not in field_dataset_names:
                st.session_state["d8_dataset"] = field_dataset_names[0]
            with dataset_col:
                dataset_name = st.selectbox("자료 묶음 선택", field_dataset_names, key="d8_dataset")
            chosen_info = DATASETS[dataset_name]
            numeric_options = ensure_xy_columns(dataset_name)
            variable_help = variable_help_text(numeric_options)
            x_col, y_col = st.columns(2)
            with x_col:
                st.selectbox("입력변수 x 선택", numeric_options, key="d8_x_col", help=variable_help)
            y_options = [col for col in numeric_options if col != st.session_state["d8_x_col"]] or numeric_options
            if st.session_state.get("d8_y_col") not in y_options:
                st.session_state["d8_y_col"] = y_options[0]
            with y_col:
                st.selectbox("종속변수 y 선택", y_options, key="d8_y_col", help=variable_help)

            st.markdown(pretty_title("2. 예측 질문과 삶과 연결하는 질문 만들기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            render_stage_card(
                "수치 예측 질문과 삶과 연결하는 질문을 함께 만듭니다",
                "삶과 연결하는 질문은 데이터의 숫자가 사람들의 삶, 어려움, 변화와 어떻게 이어지는지 묻는 질문입니다.",
                "yellow",
                "질문 만들기",
            )
            st.text_area(
                "우리 모둠의 예측 질문",
                key="d8_prediction_question",
                height=90,
                placeholder="예: 시간이 지날수록 재생에너지 비중은 얼마나 증가할까?",
            )
            st.text_area(
                "삶과 연결하는 질문",
                key="d8_deep_question",
                height=96,
                placeholder=chosen_info.get("deep_question", "데이터 속에서 보이는 인간 또는 사회적 문제를 질문으로 바꾸어 적어 보세요."),
            )
            st.caption("삶과 연결하는 질문은 데이터의 수치가 아니라, 그 숫자가 누구의 삶과 어떤 문제에 연결되는지 생각하게 하는 질문입니다.")
            x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
            st.markdown(pretty_title("3. 선택한 자료와 변수 확인하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="stage-card stage-card-purple">
                    <div class="stage-card-help" style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;">
                        <span><b>선택한 자료</b>: {html.escape(str(dataset_name))}</span>
                        <span><b>입력변수 x</b>: {html.escape(str(x_label))}</span>
                        <span><b>종속변수 y</b>: {html.escape(str(y_label))}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"자료 출처: {chosen_info.get('source', '출처 정보 없음')}")
            if chosen_info.get("source_url"):
                st.markdown(f"[원자료 또는 참고 자료 열기]({chosen_info['source_url']})")
            if st.button("F.U 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    1,
                    "F.U 단계: 데이터 분석 방향 정하기",
                    [
                        ("자료 묶음", dataset_name),
                        ("입력변수 x / 종속변수 y", f"x={x_label}, y={y_label}"),
                        ("우리 모둠의 예측 질문", st.session_state.get("d8_prediction_question", "")),
                        ("삶과 연결하는 질문", st.session_state.get("d8_deep_question", "")),
                    ],
                )
            saved_stage_caption(1)

    with tabs[1]:
        dataset_name = st.session_state["d8_dataset"]
        chosen_info = DATASETS[dataset_name]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        stage_intro(
            "수학의 언어: 데이터를 어떻게 수학적으로 표현할까?",
            "선택한 데이터를 표와 산점도로 구조화하고, 점들의 방향, 굽음, 범위, 이상치를 말로 설명합니다.",
            "데이터의 관계를 어떻게 수학적으로 표현할까?",
            "#fff8e1",
            "#ffecb3",
        )
        with st.container(border=True):
            st.markdown(pretty_title("1. 선택한 자료와 산점도 먼저 확인하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            render_stage_card(
                "표와 산점도로 데이터의 모양을 먼저 봅니다",
                "함수 종류를 고르기 전에 점들이 직선처럼 모이는지, 굽은 흐름인지, 멀리 떨어진 점이 있는지 확인합니다.",
                "purple",
                "자료 확인",
            )
            render_data_preview(dataset_name, x_data, y_data, x_label, y_label)

            st.markdown(pretty_title("2. 데이터 점으로 흐름 관찰하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            corr = float(np.corrcoef(x_data, y_data)[0, 1]) if len(x_data) > 1 else np.nan
            render_summary_chips(
                [
                    ("자료 개수", f"{len(x_data)}개"),
                    ("상관계수 r", "계산 불가" if np.isnan(corr) else f"{corr:.3f}"),
                    ("x 범위", f"{np.min(x_data):g} ~ {np.max(x_data):g}"),
                ],
                "blue",
            )
            render_stage_card(
                "점들의 흐름을 말로 표시합니다",
                "체크한 내용은 추세선 함수 종류를 고를 때 근거가 됩니다. 여러 개를 선택해도 됩니다.",
                "blue",
                "흐름 관찰",
            )
            flow_options = ["대체로 증가한다.", "대체로 감소한다.", "처음에는 빠르게 변하다가 점점 완만해진다.", "직선보다 곡선에 가깝다.", "일부 점이 전체 흐름에서 벗어나 보인다."]
            selected_flows = st.multiselect(
                "데이터 흐름 선택",
                flow_options,
                key="d8_selected_flows",
                placeholder="해당하는 흐름을 선택하세요.",
            )
            flow_sentences = {
                "대체로 증가한다.": "x가 커질수록 y도 대체로 증가하는 흐름이 보입니다.",
                "대체로 감소한다.": "x가 커질수록 y는 대체로 감소하는 흐름이 보입니다.",
                "처음에는 빠르게 변하다가 점점 완만해진다.": "처음에는 변화가 크지만 뒤로 갈수록 변화가 점점 완만해집니다.",
                "직선보다 곡선에 가깝다.": "점들의 흐름은 직선보다는 곡선 모양에 더 가깝게 보입니다.",
                "일부 점이 전체 흐름에서 벗어나 보인다.": "일부 점은 전체 흐름에서 벗어나 있어 따로 살펴볼 필요가 있습니다.",
            }
            flow_signature = tuple(selected_flows)
            if st.session_state.get("d8_flow_signature") != flow_signature:
                st.session_state["observation_text"] = " ".join(flow_sentences[flow] for flow in selected_flows)
                st.session_state["d8_flow_signature"] = flow_signature
            st.text_area(
                "우리 모둠이 관찰한 데이터의 흐름",
                key="observation_text",
                height=110,
                placeholder="예: x가 커질수록 y도 대체로 증가하지만, 뒤쪽에서는 증가 속도가 조금 느려지는 것 같다.",
            )

            st.markdown(pretty_title("3. 데이터 해석 렌즈로 관찰 점검하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            render_stage_card(
                "그래프를 볼 때 빠지기 쉬운 해석 습관을 점검합니다",
                "상관관계, 평균, 최근 흐름만 보고 성급하게 결론 내리지 않도록 하나의 렌즈를 골라 다시 봅니다.",
                "yellow",
                "해석 점검",
            )
            if st.session_state.get("d8_lens_dataset") != dataset_name:
                st.session_state["d8_factfulness_lens"] = chosen_info.get("factfulness_lens", "직선 본능 점검")
                st.session_state["d8_lens_dataset"] = dataset_name
            st.selectbox("데이터를 보고 분석할 때 특히 조심할 관점", list(FACTFULNESS_LENSES.keys()), key="d8_factfulness_lens")
            selected_lens = st.session_state["d8_factfulness_lens"]
            st.markdown(
                f"""
                <div style="
                    border-left: 5px solid #f9a825;
                    background: #fffdf3;
                    padding: 12px 14px;
                    margin: 8px 0 12px 0;
                    border-radius: 6px;
                ">
                    <div style="font-size:0.78rem;font-weight:800;color:#8d6e00;margin-bottom:4px;">
                        선택한 해석 렌즈
                    </div>
                    <div style="font-size:1.04rem;font-weight:900;color:#3e2723;margin-bottom:5px;">
                        {html.escape(selected_lens)}
                    </div>
                    <div style="font-size:0.92rem;line-height:1.55;color:#5d4037;">
                        {html.escape(FACTFULNESS_LENSES[selected_lens])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "점검한 내용",
                key="d8_factfulness_question",
                height=82,
                placeholder="예: 평균만 보고 모든 나라나 지역이 같은 변화를 겪는다고 말하지 않도록 조심해야 한다.",
            )
            if st.button("T 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    2,
                    "T 단계: 데이터 점으로 흐름 관찰하기",
                    [
                        ("데이터 흐름 선택", ", ".join(selected_flows) or "선택 없음"),
                        ("우리 모둠이 관찰한 데이터의 흐름", st.session_state.get("observation_text", "")),
                        ("데이터를 보고 분석할 때 특히 조심할 관점", selected_factfulness_text()),
                        ("점검한 내용", st.session_state.get("d8_factfulness_question", "")),
                    ],
                )
            saved_stage_caption(2)

    with tabs[2]:
        dataset_name = st.session_state["d8_dataset"]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        stage_intro(
            "AI 이해·활용: AI는 데이터를 바탕으로 어떤 예측을 할까?",
            "데이터의 흐름을 설명할 수 있는 함수 종류를 고르고, 매개변수를 조절해 예측 모델을 만듭니다.",
            "AI는 데이터를 바탕으로 어떤 예측을 할까?",
            "#fff8e1",
            "#ffecb3",
        )
        with st.container(border=True):
            st.markdown(pretty_title("1. 함수 종류와 그래프 조작하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            function_type = st.selectbox("함수 종류 선택", FUNCTION_OPTIONS, key="d8_function")
            st.markdown("**변수의 의미를 확인하며 조정하기**")
            params = get_parameters(function_type, x_data, y_data)
            predicted_data_y, valid_data_mask = calculate_function(function_type, x_data, params)
            loss, _ = calculate_loss(y_data, predicted_data_y, valid_data_mask)
            metrics = calculate_error_metrics(y_data, predicted_data_y, valid_data_mask)
            optimal_params = st.session_state.get("d8_optimal_params") or fit_default_params(function_type, x_data, y_data)
            optimal_predicted_y, optimal_valid_mask = calculate_function(function_type, x_data, optimal_params)
            optimal_loss, _ = calculate_loss(y_data, optimal_predicted_y, optimal_valid_mask)
            st.session_state["d8_params"] = params
            st.session_state["d8_function_type"] = function_type
            default_new_x = float(max(x_data) + (x_data[1] - x_data[0] if len(x_data) > 1 else 1))
            if "d8_new_x" not in st.session_state:
                st.session_state["d8_new_x"] = default_new_x
            new_x = float(st.session_state.get("d8_new_x", default_new_x))
            predicted_y = predict_value(function_type, float(new_x), params)
            range_type, range_message = prediction_range_label(float(new_x), x_data)
            graph_col, formula_col = st.columns([7, 3])
            with graph_col:
                st.pyplot(
                    make_plot(
                        x_data,
                        y_data,
                        function_type,
                        params,
                        float(new_x),
                        predicted_y,
                        x_label,
                        y_label,
                        figsize=(8, 6.4),
                    ),
                    use_container_width=True,
                )
            with formula_col:
                render_function_formula_panel(function_type, params)
                render_loss_card(loss)
            input_col, value_col = st.columns([1, 1])
            with input_col:
                x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
                current_x = float(st.session_state.get("d8_new_x", default_new_x))
                inside_range = x_min <= current_x <= x_max
                range_label = "관찰 범위 안" if inside_range else "관찰 범위 밖"
                range_color = "#2e7d32" if inside_range else "#ef6c00"
                st.markdown(
                    f"""
                    <div class="prediction-input-card">
                        <div class="prediction-input-title">예측할 x값({html.escape(str(x_label))})을 입력하세요</div>
                        <div class="prediction-input-help">
                            학생이 선택한 독립 변수는 <b>{html.escape(str(x_label))}</b>입니다.
                            관찰된 x 범위는 <b>{x_min:g} ~ {x_max:g}</b>이며,
                            현재 입력값은 <span style="font-weight:900;color:{range_color};">{range_label}</span>입니다.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                new_x = st.number_input(
                    f"예측할 x값({x_label})",
                    key="d8_new_x",
                    step=0.5,
                    format="%.2f",
                    help=f"그래프에서 예측하고 싶은 독립 변수 {x_label} 값을 입력합니다.",
                )
            predicted_y = predict_value(function_type, float(new_x), params)
            range_type, range_message = prediction_range_label(float(new_x), x_data)
            with value_col:
                if predicted_y is not None:
                    st.markdown(
                        f"""
                        <div class="stage-card stage-card-green">
                            <div class="stage-kicker">선택한 함수에서의 예측값</div>
                            <div class="prediction-input-title">예측값 y({html.escape(str(y_label))})</div>
                            <div class="stage-card-title">{html.escape(str(y_label))} = {predicted_y:.2f}</div>
                            <div class="stage-card-help">
                                입력한 <b>{html.escape(str(x_label))}</b> 값을 현재 선택한 함수에 넣어 계산한 <b>{html.escape(str(y_label))}</b> 값입니다.<br>
                                <div style="margin-top:8px;background:#ffffff;border:1px solid #c8e6c9;
                                    border-radius:10px;padding:8px 10px;color:#1b5e20;font-size:1.05rem;
                                    font-weight:900;text-align:center;">
                                    f({float(new_x):g}) = {predicted_y:.2f}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("선택한 함수의 정의역 때문에 예측값을 계산할 수 없습니다.")
    
            st.markdown(pretty_title("2. 그래프 특징으로 데이터 흐름 판단하기", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
            render_stage_card(
                "그래프를 보고 핵심만 정리합니다",
                "선택한 함수가 데이터 흐름을 어떻게 설명하는지와 예측할 때 조심할 점만 짧게 적습니다.",
                "green",
                "해석 정리",
            )
            st.text_area(
                "이 함수가 데이터 흐름을 설명하는 점",
                key="choice_reason",
                height=110,
                placeholder="예: x가 커질수록 y가 완만하게 증가하는 흐름이 그래프 모양과 비슷하다.",
            )
            st.text_area(
                "예측할 때 조심할 점",
                key="limitation",
                height=100,
                placeholder="예: 입력한 x값이 관찰된 범위 밖이면 실제로도 같은 흐름이 계속된다고 단정하기 어렵다.",
            )
    
            function_text = build_function_text(function_type, params)
            st.caption(
                "U 단계에서는 함수 그래프를 단순히 그리는 데서 끝내지 않고, 손실값과 예측값을 함께 보며 AI가 데이터를 바탕으로 예측하는 방식을 이해합니다."
            )
            if st.button("U 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    3,
                    "U 단계: 함수 그래프와 예측할 때 조심할 점 정리하기",
                    [
                        ("함수 종류 선택", function_type),
                        ("함수식", function_text),
                        ("손실값", format_optional_number(loss)),
                        (f"예측할 x값({x_label}) / 예측값 y({y_label})", f"x={float(new_x):g}, y={predicted_y:.2f}" if predicted_y is not None else "계산 불가"),
                        ("이 함수가 데이터 흐름을 설명하는 점", st.session_state.get("choice_reason", "")),
                        ("예측할 때 조심할 점", st.session_state.get("limitation", "")),
                    ],
                )
            saved_stage_caption(3)

    with tabs[3]:
        dataset_name = st.session_state["d8_dataset"]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        function_type = st.session_state.get("d8_function", FUNCTION_OPTIONS[0])
        params = st.session_state.get("d8_params") or fit_default_params(function_type, x_data, y_data)
        predicted_data_y, valid_data_mask = calculate_function(function_type, x_data, params)
        loss, _ = calculate_loss(y_data, predicted_data_y, valid_data_mask)
        predicted_y = predict_value(function_type, float(st.session_state.get("d8_new_x", max(x_data))), params)
        function_text = build_function_text(function_type, params)
        prediction_text = f"x={float(st.session_state.get('d8_new_x', max(x_data))):g}일 때 y={predicted_y:.2f}" if predicted_y is not None else "계산 불가"

        stage_intro(
            "세상과 연결: 정보 콘텐츠로 공유하기",
            "함수 추세선 탐구 결과를 그대로 믿게 하기보다, 근거와 한계를 함께 보여 주는 정보 콘텐츠로 확장합니다.",
            "우리의 분석 결과를 사람들이 책임 있게 이해하도록 어떤 콘텐츠로 전달할까?",
            "#fce4ec",
            "#f8bbd0",
        )
        with st.container(border=True):
            st.markdown(pretty_title("1. 예측 결과의 의미와 한계 정리하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            render_stage_card(
                "예측 결과를 사람들에게 어떻게 설명할지 정합니다",
                "깊은 질문을 바탕으로 어떤 실천 메시지를 전할지 정하고, 예측의 한계도 함께 담아야 정보 콘텐츠가 과장되지 않습니다.",
                "blue",
                "콘텐츠 메시지",
            )
            new_x_value = float(st.session_state.get("d8_new_x", max(x_data)))
            trend_reason = st.session_state.get("choice_reason", "").strip()
            trend_sentence = f" 학생들이 정리한 추세선의 특징은 '{trend_reason}'입니다." if trend_reason else ""
            if predicted_y is not None:
                result_sentence = (
                    f"우리 모둠은 '{dataset_name}' 데이터를 선택하고, {function_type} 추세선을 사용했습니다. "
                    f"이 추세선의 손실값(MSE)은 {format_optional_number(loss)}입니다. "
                    f"{trend_sentence} "
                    f"{x_label}이(가) {new_x_value:g}일 때 {y_label}을(를) 예측했으며, "
                    f"예측값은 {predicted_y:.2f}입니다."
                )
            else:
                result_sentence = (
                    f"우리 모둠은 '{dataset_name}' 데이터를 선택하고, {function_type} 추세선을 사용했습니다. "
                    f"이 추세선의 손실값(MSE)은 {format_optional_number(loss)}입니다. "
                    f"{trend_sentence} "
                    f"{x_label}이(가) {new_x_value:g}일 때 {y_label}을(를) 예측하려 했지만, "
                    "선택한 함수의 정의역 때문에 예측값을 계산할 수 없습니다."
                )
            st.markdown(
                f"""
                <div style="
                    background:#f5f9ff;
                    border:1px solid #cfe3ff;
                    border-left:5px solid #1976d2;
                    border-radius:8px;
                    padding:12px 14px;
                    margin:8px 0 14px 0;
                    color:#263238;
                    line-height:1.6;
                    font-size:0.95rem;
                ">
                    <div style="font-weight:900;color:#0d47a1;margin-bottom:4px;">앞 단계 결과 요약</div>
                    {html.escape(result_sentence)}
                </div>
                """,
                unsafe_allow_html=True,
            )
            deep_question = st.text_area(
                "깊은 질문",
                key="d8_final_deep_question",
                value=st.session_state.get("d8_deep_question", ""),
                height=95,
                placeholder="예: 수치가 좋아져도 여전히 혜택을 받지 못하는 사람들은 누구일까?",
            )
            action_message = st.text_area(
                "실천 메시지",
                key="action_message",
                height=95,
                placeholder="예: 그 변화 속에서 여전히 도움이 필요한 사람들을 함께 살펴보자.",
            )
    
            st.markdown(pretty_title("2. 정보 콘텐츠 방향 정하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            render_stage_card(
                "콘텐츠의 표현 방향을 정합니다",
                "디자인 분위기와 특히 강조해야 할 점을 정하면, 최종 프롬프트가 더 명확해집니다.",
                "yellow",
                "제작 방향",
            )
            design_tone = st.text_input("디자인 분위기", key="d8_design_tone", value="밝고 명확한 교육용 정보 카드")
            ds_reflection = st.text_area(
                "콘텐츠에 담고 싶은 우리 모둠의 생각",
                key="d8_ds_reflection_re",
                height=95,
                placeholder="예: 이 데이터가 단순한 숫자가 아니라 사람들의 삶과 연결되어 있다는 점을 보여 주고 싶다.",
            )
            choice_reason = st.session_state.get("choice_reason", "").strip()
            limitation = st.session_state.get("limitation", "").strip()
            if st.button("R.E 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    4,
                    "R.E 단계: 정보 콘텐츠로 공유하기",
                    [
                        ("깊은 질문", deep_question),
                        ("이 함수가 데이터 흐름을 설명하는 점", choice_reason),
                        ("예측할 때 조심할 점", limitation),
                        ("실천 메시지", action_message),
                        ("디자인 분위기", design_tone),
                        ("콘텐츠에 담고 싶은 우리 모둠의 생각", ds_reflection),
                    ],
                )
            saved_stage_caption(4)

            canva_limitation_text = "\n".join(
                part
                for part in [
                    f"이 함수가 데이터 흐름을 설명하는 점: {choice_reason}" if choice_reason else "",
                    f"예측할 때 조심할 점: {limitation}" if limitation else "",
                ]
                if part
            )
            canva_prompt = build_canva_prompt(
                st.session_state.get("d8_group", "우리 모둠"),
                dataset_name,
                st.session_state.get("d8_prediction_question", ""),
                function_type,
                function_text,
                format_optional_number(loss),
                prediction_text,
                deep_question,
                canva_limitation_text,
                action_message,
                ds_reflection,
                design_tone,
            )
    
            st.markdown(pretty_title("3. 제작 도구와 최종 프롬프트 연결하기", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
            render_stage_card(
                "Canva 제작과 갤러리 공유로 연결합니다",
                "아래 프롬프트를 확인한 뒤 Canva AI에서 콘텐츠를 만들고, 반별 갤러리에 결과물을 공유합니다.",
                "green",
                "제작 연결",
            )
            if st.button("프롬프트 생성", use_container_width=True):
                st.session_state["d8_show_canva_prompt"] = True
            if st.session_state.get("d8_show_canva_prompt", False):
                st.markdown(pretty_title("최종 제작 프롬프트", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
                st.code(canva_prompt, language="markdown")
                render_canva_gallery_links(st.session_state.get("d8_class", CLASS_OPTIONS[0]))
    
            stage_rows = []
            for idx in range(1, 5):
                row = st.session_state.get(f"d8_saved_stage_{idx}")
                if row:
                    stage_rows.append(row)
            pdf_col, portfolio_col = st.columns(2)
            with pdf_col:
                if stage_rows:
                    final_fig = make_plot(x_data, y_data, function_type, params, float(st.session_state.get("d8_new_x", max(x_data))), predicted_y, x_label, y_label)
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
    run()
