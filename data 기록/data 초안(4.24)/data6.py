import datetime
import os 
import pandas as pd
import streamlit as st
from fpdf import FPDF

from future_extra_datasets import (
    EXTRA_DATASETS,
    FIELD_DATASETS,
    FIELD_ORDER,
    KOREA_CLIMATE_DATASET,
    field_for_dataset,
    normalize_dataset_name,
)


font_path = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")


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
        "app_examples": "예: 광고비 종류를 입력하면 예상 판매량을 알려 주고, 어떤 광고 전략이 더 효과적인지 보여 주는 홍보 계획 앱",
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
        "app_examples": "예: 나이와 BMI를 바탕으로 예상 의료비를 보여 주고, 건강 관리의 중요성을 안내하는 건강 정보 앱",
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
        "app_examples": "예: 차량 무게나 마력을 바탕으로 예상 연비를 보여 주고, 효율적인 자동차 선택을 돕는 공학 정보 앱",
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
        "app_examples": "예: 기온과 습도 변화를 바탕으로 환경 변화를 이해하고 생활 속 대응을 제안하는 기후 정보 앱",
    },
}

DATASETS.pop("환경: 델리 기후 변화", None)
DATASETS["환경: 대한민국 기후 변화"] = KOREA_CLIMATE_DATASET
DATASETS.update(EXTRA_DATASETS)

TARGET_USERS = [
    "중학생/고등학생",
    "일반 시민",
    "소상공인",
    "의료·보건 사용자",
    "공학/설비 관리자",
]

OUTPUT_TYPES = ["정보형", "게임형", "카드뉴스형", "포스터형"]

TYPE_FEATURE_OPTIONS = {
    "정보형": [
        "핵심 데이터 카드",
        "그래프 또는 시각화 자료",
        "짧은 설명 문장",
        "주의 문구/한계 안내",
        "간단한 사용 안내",
    ],
    "게임형": [
        "시작 화면 안내",
        "선택 버튼 또는 단계 구성",
        "선택에 따른 피드백 문구",
        "점수/성공 조건",
        "마무리 배운 점 정리",
    ],
    "카드뉴스형": [
        "표지 카드",
        "문제 상황 소개 카드",
        "데이터 설명 카드",
        "의미 해석 카드",
        "실천 제안 카드",
    ],
    "포스터형": [
        "강한 제목",
        "핵심 수치 강조",
        "그래프 또는 아이콘 시각화",
        "한눈에 보이는 핵심 메시지",
        "실천 제안 문구",
        "주의 문구",
    ],
}

STYLE_OPTIONS = [
    "밝고 친근한 교육용",
    "깔끔한 정보 전달형",
    "강조가 뚜렷한 캠페인형",
    "몰입감 있는 게임형",
]


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
        self.set_font(self._font_family, "", 18)
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 6차시 앱 기획 프롬프트", ln=1, align="C")
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
            <div style="font-size:0.9rem; font-weight:700; color:#5e35b1; margin-bottom:8px;">F.U.T.U.R.E. 프로젝트 STEP 3-2</div>
            <div style="font-size:1.9rem; font-weight:800; color:#1f2937; margin-bottom:8px;">{title}</div>
            <div style="font-size:1rem; line-height:1.7; color:#37474f;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_intro(title, description, question, color1="#fff3e0", color2="#ffe0b2"):
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


def apply_output_type_selection(selected):
    st.session_state["d6_output_type"] = selected
    st.session_state["d6_feature_options"] = TYPE_FEATURE_OPTIONS[selected][:3]
    st.session_state["d6_last_output_type"] = selected


def apply_recommended_features(features):
    st.session_state["d6_feature_options"] = list(features)


def render_app_type_selector_cards(selected_type):
    items = app_type_overview_cards()
    for start in range(0, len(items), 2):
        row_items = items[start:start + 2]
        row_cols = st.columns(2)
        for col, item in zip(row_cols, row_items):
            with col:
                is_selected = item["title"] == selected_type
                border = "#fb8c00" if is_selected else item["border"]
                shadow = "0 8px 18px rgba(251, 140, 0, 0.20)" if is_selected else "0 4px 12px rgba(33, 150, 243, 0.08)"
                badge_bg = "#fff3e0" if is_selected else "rgba(255,255,255,0.72)"
                badge_text = "현재 선택됨" if is_selected else "아래 버튼으로 선택"
                st.markdown(
                    f"""
                    <div style="
                        height:100%;
                        min-height:180px;
                        padding:16px 18px;
                        border-radius:18px;
                        background:{item['bg']};
                        border:2px solid {border};
                        box-shadow:{shadow};
                        margin-bottom:10px;
                        transition:all 0.2s ease;
                    ">
                        <div style="font-size:1.05rem; color:#37474f; margin-bottom:8px; font-weight:800;">{item['title']}</div>
                        <div style="font-size:1.2rem; color:#263238; font-weight:800; margin-bottom:8px;">{item['value']}</div>
                        <div style="font-size:0.9rem; color:#546e7a; line-height:1.65; margin-bottom:12px;">{item['detail']}</div>
                        <div style="
                            display:inline-block;
                            padding:6px 10px;
                            border-radius:999px;
                            background:{badge_bg};
                            border:1px solid rgba(0,0,0,0.08);
                            color:#6d4c41;
                            font-size:0.82rem;
                            font-weight:700;
                        ">{badge_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if is_selected:
                    st.button("현재 선택됨", key=f"d6_select_type_{item['title']}", use_container_width=True, disabled=True)
                else:
                    if st.button(f"{item['title']} 선택", key=f"d6_select_type_{item['title']}", use_container_width=True):
                        apply_output_type_selection(item["title"])
        for col in row_cols[len(row_items):]:
            with col:
                st.empty()


def ensure_state():
    dataset_names = list(DATASETS.keys())
    d5_dataset = normalize_dataset_name(st.session_state.get("d5_dataset", dataset_names[0]))
    if d5_dataset not in DATASETS:
        d5_dataset = FIELD_DATASETS[FIELD_ORDER[0]][0]
    d5_group = st.session_state.get("d5_group", "")
    d5_degree = int(st.session_state.get("d5_ml_degree", 1))
    d5_field = st.session_state.get("d5_field", field_for_dataset(d5_dataset))
    st.session_state.setdefault("d6_group", d5_group)
    st.session_state.setdefault("d6_field", d5_field if d5_field in FIELD_ORDER else field_for_dataset(d5_dataset))
    st.session_state.setdefault("d6_dataset", d5_dataset if d5_dataset in DATASETS else dataset_names[0])
    st.session_state["d6_dataset"] = normalize_dataset_name(st.session_state.get("d6_dataset", d5_dataset))
    if st.session_state["d6_dataset"] not in DATASETS:
        st.session_state["d6_dataset"] = FIELD_DATASETS[FIELD_ORDER[0]][0]
    if st.session_state.get("d6_field") not in FIELD_ORDER:
        st.session_state["d6_field"] = field_for_dataset(st.session_state["d6_dataset"])
    if st.session_state["d6_dataset"] not in FIELD_DATASETS[st.session_state["d6_field"]]:
        st.session_state["d6_dataset"] = FIELD_DATASETS[st.session_state["d6_field"]][0]

    dataset = DATASETS[st.session_state["d6_dataset"]]
    columns = list(dataset["table"].columns)
    default_x = st.session_state.get("d5_x_col", dataset["default_x"])
    default_y = st.session_state.get("d5_y_col", dataset["default_y"])
    if st.session_state.get("d6_x_col") not in columns:
        st.session_state["d6_x_col"] = default_x if default_x in columns else dataset["default_x"]
    valid_y = [col for col in columns if col != st.session_state["d6_x_col"]]
    if st.session_state.get("d6_y_col") not in valid_y:
        st.session_state["d6_y_col"] = default_y if default_y in valid_y else valid_y[0]

    model_default = "직선 회귀 기반 앱" if d5_degree == 1 else "2차 회귀 기반 앱"
    st.session_state.setdefault("d6_target_user", TARGET_USERS[0])
    st.session_state.setdefault("d6_target_reason", "")
    st.session_state.setdefault("d6_model_type", model_default)
    st.session_state.setdefault("d6_app_name", "")
    st.session_state.setdefault("d6_real_problem", "")
    st.session_state.setdefault("d6_app_goal", "")
    st.session_state.setdefault("d6_main_output", "")
    st.session_state.setdefault("d6_ui_style", STYLE_OPTIONS[0])
    st.session_state.setdefault("d6_feature_options", TYPE_FEATURE_OPTIONS[OUTPUT_TYPES[0]][:3])
    st.session_state.setdefault("d6_material_mode", "FUTURE 프로젝트 자료 반영하기")
    st.session_state.setdefault("d6_social_value", "")
    st.session_state.setdefault("d6_limit_note", "")
    st.session_state.setdefault("d6_human_check", "")
    st.session_state.setdefault("d6_prompt_reflection", "")
    inherited_analysis = st.session_state.get("d5_analysis_report", "")
    inherited_interpretation = st.session_state.get("d5_interpretation_report", "")
    st.session_state.setdefault("d6_output_type", OUTPUT_TYPES[0])
    st.session_state.setdefault("d6_last_output_type", st.session_state["d6_output_type"])
    st.session_state.setdefault("d6_project_title", "")
    st.session_state.setdefault("d6_problem_situation", inherited_analysis)
    st.session_state.setdefault("d6_project_goal", "")
    st.session_state.setdefault("d6_visual_style", STYLE_OPTIONS[0])
    st.session_state.setdefault("d6_data_meaning", inherited_interpretation)
    st.session_state.setdefault("d6_prompt_extra", "")
    st.session_state.setdefault("d6_game_meaning", inherited_interpretation)
    st.session_state.setdefault("d6_game_story", "")
    st.session_state.setdefault("d6_game_goal", "")
    st.session_state.setdefault("d6_game_rule", "")
    st.session_state.setdefault("d6_game_feedback", "")
    st.session_state.setdefault("d6_custom_data_name", "")
    st.session_state.setdefault("d6_custom_topic", "")
    st.session_state.setdefault("d6_custom_x", "")
    st.session_state.setdefault("d6_custom_y", "")
    st.session_state.setdefault("d6_custom_rows", "")
    st.session_state.setdefault("d6_custom_analysis", "")
    st.session_state.setdefault("d6_custom_interpretation", "")
    if not st.session_state.get("d6_project_title"):
        dataset_label = st.session_state["d6_dataset"].split(":")[0]
        st.session_state["d6_project_title"] = f"{dataset_label} 데이터 기반 앱 기획"


def current_dataset():
    info = DATASETS[st.session_state["d6_dataset"]]
    table = info["table"].copy()
    return {
        "name": st.session_state["d6_dataset"],
        "table": table,
        "selected_table": table[[st.session_state["d6_x_col"], st.session_state["d6_y_col"]]].copy(),
        "x_column": st.session_state["d6_x_col"],
        "y_column": st.session_state["d6_y_col"],
        "story": info["story"],
        "app_examples": info["app_examples"],
    }


def rows_to_text(df):
    display_df = df.copy()
    display_df.insert(0, "행 번호", range(1, len(display_df) + 1))
    return display_df.round(3).to_string(index=False)


def get_inherited_reports():
    return {
        "analysis": clean_text(
            st.session_state.get("d5_analysis_report", ""),
            "FUTURE 프로젝트 자료에서 아직 데이터 분석 요약을 작성하지 않았습니다.",
        ),
        "interpretation": clean_text(
            st.session_state.get("d5_interpretation_report", ""),
            "FUTURE 프로젝트 자료에서 아직 결과 해석을 작성하지 않았습니다.",
        ),
    }


def using_future_materials():
    return st.session_state.get("d6_material_mode", "FUTURE 프로젝트 자료 반영하기") == "FUTURE 프로젝트 자료 반영하기"


def get_project_materials():
    if using_future_materials():
        dataset = current_dataset()
        reports = get_inherited_reports()
        return {
            "name": dataset["name"],
            "x_column": dataset["x_column"],
            "y_column": dataset["y_column"],
            "selected_table": dataset["selected_table"],
            "sample_rows": rows_to_text(dataset["selected_table"]),
            "analysis": reports["analysis"],
            "interpretation": reports["interpretation"],
            "topic": dataset["story"],
            "source_label": "FUTURE 프로젝트 자료",
            "uses_future": True,
        }

    data_name = clean_text(st.session_state.get("d6_custom_data_name", ""), "학생이 직접 설계한 프로젝트 자료")
    sample_rows = clean_text(
        st.session_state.get("d6_custom_rows", ""),
        "아직 예시 자료를 입력하지 않았습니다. 학생이 생각한 자료 예시나 숫자, 표 아이디어를 적어 주세요.",
    )
    analysis = clean_text(
        st.session_state.get("d6_custom_analysis", ""),
        "아직 자료의 특징 요약을 작성하지 않았습니다. 직접 설계한 자료에서 보일 경향이나 특징을 적어 주세요.",
    )
    interpretation = clean_text(
        st.session_state.get("d6_custom_interpretation", ""),
        "아직 자료 해석을 작성하지 않았습니다. 이 자료를 통해 전달하고 싶은 의미나 시사점을 적어 주세요.",
    )
    topic = clean_text(
        st.session_state.get("d6_custom_topic", ""),
        "아직 프로젝트 자료의 주제와 문제 상황을 작성하지 않았습니다.",
    )
    return {
        "name": data_name,
        "x_column": "",
        "y_column": "",
        "selected_table": None,
        "sample_rows": sample_rows,
        "analysis": analysis,
        "interpretation": interpretation,
        "topic": topic,
        "source_label": "학생이 직접 설계한 프로젝트 자료",
        "uses_future": False,
    }


def output_type_guide(output_type):
    guides = {
        "정보형": "정보형은 데이터를 차분하게 설명하는 앱입니다. 핵심 수치, 그래프, 짧은 설명, 주의 문구가 또렷하게 보이면 좋습니다.",
        "카드뉴스형": "카드뉴스형은 여러 장면으로 나누어 순서 있게 전달하는 앱입니다. 시작-근거-해석-제안 흐름이 잘 보이면 좋습니다.",
        "포스터형": "포스터형은 한 장에서 강한 메시지를 전하는 앱입니다. 제목, 핵심 수치, 시각 자료, 행동 제안이 한눈에 보여야 합니다.",
        "게임형": "게임형은 데이터의 의미를 선택과 경험으로 느끼게 하는 앱입니다. 규칙은 단순하고, 선택 뒤 피드백은 바로 보이는 것이 좋습니다.",
    }
    return guides.get(output_type, "")


def prompt_input_guide(output_type):
    guides = {
        "정보형": "입력할 때는 '무엇을 보여 줄까'를 정하면 됩니다. 핵심 수치, 설명 문장, 주의 안내를 중심으로 적으면 충분합니다.",
        "카드뉴스형": "입력할 때는 '어떤 순서로 보여 줄까'를 정하면 됩니다. 카드별 제목과 핵심 문장을 순서대로 적으면 좋습니다.",
        "포스터형": "입력할 때는 '무엇을 가장 크게 보여 줄까'를 정하면 됩니다. 한 줄 메시지와 핵심 수치 중심으로 적으면 좋습니다.",
        "게임형": "입력할 때는 '무엇을 하게 할까'를 정하면 됩니다. 배경 상황, 목표, 선택, 피드백 흐름만 정해도 충분합니다.",
    }
    return guides.get(output_type, "")


def output_type_focus_points(output_type):
    guides = {
        "정보형": [
            "핵심 데이터 1~2개를 고르고 짧게 설명하기",
            "그래프와 설명을 같은 화면에 두기",
            "마지막에 주의 문구를 꼭 넣기",
        ],
        "카드뉴스형": [
            "카드 흐름을 5장 안팎으로 단순하게 잡기",
            "각 카드마다 제목과 한 문장만 분명히 쓰기",
            "마지막 카드는 실천 제안으로 마무리하기",
        ],
        "포스터형": [
            "한 장에서 가장 중요한 수치 하나를 크게 보이기",
            "짧은 제목과 한 줄 메시지로 압축하기",
            "실천 제안은 짧고 강하게 넣기",
        ],
        "게임형": [
            "선택은 2~3개 수준으로 단순하게 만들기",
            "선택 뒤 결과와 피드백이 바로 나오게 하기",
            "마지막에 데이터 의미를 다시 정리해 주기",
        ],
    }
    return guides.get(output_type, [])


def app_type_overview_cards():
    return [
        {
            "title": "정보형",
            "value": "설명 중심",
            "detail": "그래프, 표, 설명 문장을 읽으며 프로젝트 자료의 결과를 이해하도록 돕는 형태입니다.",
            "bg": "#f4f9ff",
            "border": "#90caf9",
        },
        {
            "title": "카드뉴스형",
            "value": "순서 중심",
            "detail": "여러 장면으로 나누어 문제 상황, 데이터 근거, 해석, 제안을 순서 있게 전달하는 형태입니다.",
            "bg": "#fff8e1",
            "border": "#ffcc80",
        },
        {
            "title": "포스터형",
            "value": "강조 중심",
            "detail": "한 장 안에서 제목, 핵심 수치, 시각 자료, 실천 제안을 짧고 강하게 보여 주는 형태입니다.",
            "bg": "#fce4ec",
            "border": "#f48fb1",
        },
        {
            "title": "게임형",
            "value": "체험 중심",
            "detail": "데이터가 주는 의미를 규칙, 선택, 피드백을 통해 직접 느끼게 하는 형태입니다.",
            "bg": "#ede7f6",
            "border": "#b39ddb",
        },
    ]


def recommended_feature_pack(output_type):
    packs = {
        "정보형": {
            "features": ["핵심 데이터 카드", "그래프 또는 시각화 자료", "짧은 설명 문장", "주의 문구/한계 안내"],
            "reason": "정보형은 사용자가 숫자와 그래프를 보고 바로 이해할 수 있어야 하므로, 정보 구조를 단순하게 만드는 것이 중요합니다.",
        },
        "게임형": {
            "features": ["시작 화면 안내", "선택 버튼 또는 단계 구성", "선택에 따른 피드백 문구", "마무리 배운 점 정리"],
            "reason": "게임형은 복잡한 기능보다 목표-선택-피드백 흐름이 분명해야 학생이 쉽게 참여할 수 있습니다.",
        },
        "카드뉴스형": {
            "features": ["표지 카드", "문제 상황 소개 카드", "데이터 설명 카드", "실천 제안 카드"],
            "reason": "카드뉴스형은 카드 순서만 잘 잡아도 프롬프트가 훨씬 쉬워집니다. 시작-근거-해석-제안 흐름이면 충분합니다.",
        },
        "포스터형": {
            "features": ["강한 제목", "핵심 수치 강조", "그래프 또는 아이콘 시각화", "실천 제안 문구"],
            "reason": "포스터형은 한 장에서 메시지를 강하게 보여 줘야 하므로, 요소 수를 줄이고 핵심만 남기는 것이 중요합니다.",
        },
    }
    return packs.get(output_type, {"features": [], "reason": ""})


def render_future_flow_box():
    st.markdown(pretty_title("FUTURE 흐름 한눈에 보기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
            border-radius: 18px;
            border: 1px solid #dbe7f3;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.06);
            line-height: 1.8;
            color: #37474f;
        ">
            <b>F.U.</b> 누구의 어떤 문제를 해결할까?<br>
            <b>T</b> 데이터가 말해 주는 핵심 의미는 무엇일까?<br>
            <b>U</b> 어떤 앱 유형으로 구현하면 가장 잘 전달될까?<br>
            <b>R</b> 결과를 볼 때 어떤 한계와 책임을 함께 적어야 할까?<br>
            <b>E</b> 이 결과물은 어떤 사회적 실천으로 이어질 수 있을까?
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_presentation_summary_text():
    materials = get_project_materials()
    output_type = st.session_state.get("d6_output_type", OUTPUT_TYPES[0])
    problem_text = (
        clean_text(st.session_state.get("d6_game_story", ""), materials["topic"])
        if output_type == "게임형"
        else clean_text(st.session_state.get("d6_problem_situation", ""), materials["topic"])
    )
    meaning_text = (
        clean_text(st.session_state.get("d6_game_meaning", ""), materials["interpretation"])
        if output_type == "게임형"
        else clean_text(st.session_state.get("d6_data_meaning", ""), materials["interpretation"])
    )
    action_text = clean_text(st.session_state.get("d6_social_value", ""), "이 결과물을 통해 제안하고 싶은 실천을 더 적어 주세요.")

    return (
        f"1. 우리 모둠은 {problem_text}\n"
        f"2. 이 결과물은 {meaning_text}\n"
        f"3. 그래서 우리는 {action_text}"
    )


def _legacy_build_prompt_text_unused(dataset):
    group_name = clean_text(st.session_state.get("d6_group", ""), "모둠명 미입력")
    app_name = clean_text(st.session_state.get("d6_app_name", ""), "앱 이름 미정")
    target_user = clean_text(st.session_state.get("d6_target_user", ""))
    real_problem = clean_text(st.session_state.get("d6_real_problem", "실제 문제 상황을 추가로 정리해 주세요."))
    app_goal = clean_text(st.session_state.get("d6_app_goal", "앱의 목표를 더 구체적으로 정리해 주세요."))
    main_output = clean_text(st.session_state.get("d6_main_output", "예측 결과를 어떻게 보여 줄지 정해 주세요."))
    social_value = clean_text(st.session_state.get("d6_social_value", "이 앱이 사회에 주는 가치를 적어 주세요."))
    limit_note = clean_text(st.session_state.get("d6_limit_note", "예측 결과를 그대로 믿으면 안 되는 이유를 적어 주세요."))
    human_check = clean_text(st.session_state.get("d6_human_check", "사람이 다시 확인해야 하는 상황을 적어 주세요."))
    prompt_reflection = clean_text(st.session_state.get("d6_prompt_reflection", "학생이 이해하기 쉬운 설명 방식이 들어가면 좋겠습니다."))
    feature_lines = "\n".join(f"- {item}" for item in st.session_state.get("d6_feature_options", [])) or "- 예측값 결과 카드"
    sample_rows = rows_to_text(dataset["selected_table"])

    return f"""다음 조건으로 고등학교 1학년 학생도 사용할 수 있는 Streamlit 앱을 만들어줘.

[앱 기본 정보]
- 앱 이름: {app_name}
- 사용 모둠: {group_name}
- 대상 사용자: {target_user}
- 앱의 예측 방식: {st.session_state.get('d6_model_type', '')}
- 화면 스타일: {st.session_state.get('d6_ui_style', '')}

[앱이 해결하려는 실제 문제]
- 문제 상황: {real_problem}
- 앱 목표: {app_goal}
- 사회적 가치: {social_value}

[데이터 설정]
- 분야 데이터: {dataset['name']}
- 독립 변수(X): {dataset['x_column']}
- 종속 변수(Y): {dataset['y_column']}
- 앱에 포함할 표본 데이터:
{sample_rows}

[앱에 꼭 들어갈 기능]
{feature_lines}

[예측 결과 표현 방식]
- 사용자가 직접 X값을 입력할 수 있어야 한다.
- 예측값은 카드와 그래프로 함께 보여 준다.
- 예측 결과를 학생의 말로 쉽게 설명하는 문장이 함께 나온다.
- 입력값과 결과값이 어떤 관계인지 한눈에 알 수 있게 해 준다.
- {main_output}

[신뢰와 책임]
- 한계 안내: {limit_note}
- 사람이 다시 확인해야 하는 상황: {human_check}
- 예측 결과를 절대 정답처럼 표현하지 말고, 참고용이라는 점을 분명히 적어 줘.

[개발 요청]
- 전체 코드는 Streamlit 한 파일로 작성해 줘.
- 모든 UI 문구는 한국어로 작성해 줘.
- 학생 활동용이라 복잡한 관리자 기능은 넣지 말아 줘.
- 그래프와 입력창, 예측 결과 카드, 설명 상자가 균형 있게 배치되게 해 줘.
- 결과를 PDF나 텍스트로 저장할 수 있는 버튼도 포함해 줘.

[수업 맥락]
- 이 앱은 FUTURE 프로젝트 자료에서 수행한 AI 데이터 예측 활동을 실제 문제 해결 앱으로 확장하는 수업이다.
- 학생이 데이터와 예측을 사회와 연결해 생각하도록 도와줘.
- {prompt_reflection}
"""


def _legacy_create_prompt_pdf_unused(group_name, dataset, prompt_text):
    pdf = ThemedPDF()
    pdf.add_font("Nanum", "", font_path, uni=True)
    pdf.set_font("Nanum", "", 12)
    pdf._font_family = "Nanum"
    pdf.footer_left = group_name
    pdf.add_page()

    pdf.kv_card(
        "모둠 정보",
        [
            ("모둠명", group_name),
            ("활동 데이터", dataset["name"]),
            ("독립 변수", dataset["x_column"]),
            ("종속 변수", dataset["y_column"]),
            ("작성일", datetime.datetime.now().strftime("%Y-%m-%d")),
        ],
    )

    add_text_box_to_pdf(pdf, "앱 이름", st.session_state.get("d6_app_name", ""))
    add_text_box_to_pdf(pdf, "실제 문제 상황", st.session_state.get("d6_real_problem", ""))
    add_text_box_to_pdf(pdf, "앱 목표", st.session_state.get("d6_app_goal", ""))
    add_text_box_to_pdf(pdf, "예측 결과 표현 방식", st.session_state.get("d6_main_output", ""))
    add_text_box_to_pdf(pdf, "사회적 가치", st.session_state.get("d6_social_value", ""))
    add_text_box_to_pdf(pdf, "예측의 한계", st.session_state.get("d6_limit_note", ""))
    add_text_box_to_pdf(pdf, "사람의 재확인 기준", st.session_state.get("d6_human_check", ""))
    add_text_box_to_pdf(pdf, "최종 앱 제작 프롬프트", prompt_text, fill_color=(250, 250, 250))
    return normalize_pdf_output(pdf.output(dest="S"))


def _legacy_run_unused():
    apply_local_style()
    ensure_state()
    page_banner(
        "AI 데이터 예측을 앱 제작 프롬프트로 연결하기",
        "data5에서 분석한 실생활 데이터 예측 활동을 바탕으로, 실제 사회 문제를 해결할 수 있는 앱을 기획하고 "
        "AI에게 앱 제작을 요청할 프롬프트를 완성해 봅시다.",
    )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)


def build_prompt_text(dataset):
    materials = get_project_materials()
    group_name = clean_text(st.session_state.get("d6_group", ""), "모둠명 미입력")
    output_type = st.session_state.get("d6_output_type", OUTPUT_TYPES[0])
    title = clean_text(st.session_state.get("d6_project_title", ""), "제목 미정")
    target_user = clean_text(st.session_state.get("d6_target_user", ""))
    target_reason = clean_text(st.session_state.get("d6_target_reason", ""), "이 사용자를 선택한 이유를 더 적어 주세요.")
    style = clean_text(st.session_state.get("d6_visual_style", ""))
    feature_lines = "\n".join(f"- {item}" for item in st.session_state.get("d6_feature_options", []))
    if not feature_lines:
        feature_lines = "- 핵심 데이터와 메시지를 분명하게 보여 주는 기본 구성"
    source_condition = (
        "- 이번 결과물은 FUTURE 프로젝트 자료에서 이미 분석한 데이터와 해석을 그대로 이어받아 활용합니다."
        if materials["uses_future"]
        else "- 이번 결과물은 학생이 직접 설계한 프로젝트 자료를 바탕으로 구성합니다."
    )
    source_request = (
        "- 보고서를 다시 업로드하라고 요구하지 말고, 아래에 제공한 데이터와 분석 내용을 바로 활용해 주세요."
        if materials["uses_future"]
        else "- 아래에 정리한 프로젝트 자료의 주제, 예시 자료, 특징, 해석을 바탕으로 결과물을 구체화해 주세요."
    )
    axis_lines = (
        f"- 독립 변수(X): {materials['x_column']}\n- 종속 변수(Y): {materials['y_column']}\n"
        if materials["uses_future"]
        else ""
    )
    focus_lines = "\n".join(f"- {item}" for item in output_type_focus_points(output_type))
    if not focus_lines:
        focus_lines = "- 핵심 메시지가 분명하게 보이도록 구성해 주세요."

    common_header = f"""당신은 Canva AI의 바이브 코딩으로 바로 구현할 수 있는 학생용 결과물을 기획하는 도우미입니다.

아래 조건을 바탕으로 한국어 결과물을 만들어 주세요.

[기본 정보]
- 결과물 유형: {output_type}
- 프로젝트 제목: {title}
- 제작 주체: {group_name}
- 주요 대상: {target_user}
- 이 사용자를 선택한 이유: {target_reason}
- 시각 스타일: {style}

[중요 조건]
{source_condition}
{source_request}
- 고등학교 1학년 학생이 이해할 수 있는 쉬운 문장으로 작성해 주세요.
- 결과물 안에는 데이터의 의미, 해석, 주의점이 함께 드러나야 합니다.
- 복잡한 로그인, 관리자 기능, 긴 메뉴 구조는 넣지 말고 학생이 바로 사용할 수 있게 단순하게 구성해 주세요.

[프로젝트 자료]
- 자료 이름: {materials['name']}
- 자료 출처 방식: {materials['source_label']}
- 자료가 다루는 주제/문제 상황: {materials['topic']}
{axis_lines}
- 선택한 데이터 일부:
{materials['sample_rows']}

[프로젝트 자료 분석 내용]
{materials['analysis']}

[프로젝트 자료 해석 내용]
{materials['interpretation']}

[이번 결과물에서 강조할 구성 요소]
{feature_lines}

[이 유형에서 특히 중요한 점]
{focus_lines}

[결과물 방향]
{output_type_guide(output_type)}
"""

    if output_type == "게임형":
        game_meaning = clean_text(st.session_state.get("d6_game_meaning", ""), "이전 데이터에서 얻은 핵심 의미를 더 분명하게 적어 주세요.")
        game_story = clean_text(st.session_state.get("d6_game_story", ""), "게임의 배경 이야기와 상황을 적어 주세요.")
        game_goal = clean_text(st.session_state.get("d6_game_goal", ""), "플레이어가 게임에서 이루어야 할 목표를 적어 주세요.")
        game_rule = clean_text(st.session_state.get("d6_game_rule", ""), "플레이 규칙과 진행 방식을 적어 주세요.")
        game_feedback = clean_text(st.session_state.get("d6_game_feedback", ""), "성공·실패·선택 결과에 따라 어떤 피드백을 줄지 적어 주세요.")
        social_value = clean_text(st.session_state.get("d6_social_value", ""), "이 게임이 학생에게 주는 학습적 가치와 사회적 의미를 적어 주세요.")
        limit_note = clean_text(st.session_state.get("d6_limit_note", ""), "게임이 단순화한 부분과 실제 상황과의 차이를 적어 주세요.")
        human_check = clean_text(st.session_state.get("d6_human_check", ""), "교사나 사용자가 다시 확인해야 할 부분을 적어 주세요.")
        prompt_extra = clean_text(st.session_state.get("d6_prompt_extra", ""), "추가 요청이 있으면 적어 주세요.")
        return f"""{common_header}
[게임으로 바꾸려는 핵심 의미]
- 이 게임은 데이터의 숫자를 그대로 계산하기보다, 이전 분석에서 얻은 의미를 경험하게 하는 것이 핵심입니다.
- 데이터에서 얻은 핵심 의미: {game_meaning}

[게임 기획]
- 배경 상황: {game_story}
- 게임 목표: {game_goal}
- 규칙과 진행 방식: {game_rule}
- 피드백 방식: {game_feedback}

[교육적 가치와 책임]
- 이 게임이 주는 가치: {social_value}
- 게임이 단순화한 부분: {limit_note}
- 사람이 다시 확인할 점: {human_check}

[최종 요청]
- Canva AI의 바이브 코딩으로 구현할 수 있는 간단한 교육용 게임 앱 기획안을 만들어 주세요.
- 화면 수는 4개 안팎으로 단순하게 구성해 주세요: 시작 화면, 선택 화면, 피드백 화면, 마무리 화면.
- 선택지는 한 단계마다 2~3개 정도로 단순하게 제안해 주세요.
- 학생이 데이터의 의미를 자연스럽게 이해하도록 질문, 선택, 피드백을 포함해 주세요.
- 결과물은 바로 제작에 사용할 수 있도록 화면별 문구와 기능을 짧고 구체적으로 제안해 주세요.
- 추가 요청: {prompt_extra}
"""

    problem_situation = clean_text(st.session_state.get("d6_problem_situation", ""), "이 결과물이 다루려는 실제 문제 상황을 적어 주세요.")
    project_goal = clean_text(st.session_state.get("d6_project_goal", ""), "이 결과물을 통해 사용자가 무엇을 이해하거나 행동하게 할지 적어 주세요.")
    data_meaning = clean_text(st.session_state.get("d6_data_meaning", ""), "이전 데이터 분석에서 드러난 핵심 의미를 적어 주세요.")
    social_value = clean_text(st.session_state.get("d6_social_value", ""), "이 결과물이 사회에 주는 시사점이나 실천 가치를 적어 주세요.")
    limit_note = clean_text(st.session_state.get("d6_limit_note", ""), "결과물을 볼 때 주의해야 할 점을 적어 주세요.")
    human_check = clean_text(st.session_state.get("d6_human_check", ""), "사람이 다시 확인해야 할 점을 적어 주세요.")
    prompt_extra = clean_text(st.session_state.get("d6_prompt_extra", ""), "추가 요청이 있으면 적어 주세요.")

    structure_request = {
        "정보형": "- 정보형 앱 또는 정보 페이지 형식으로 4개 안팎의 섹션을 제안해 주세요: 제목, 핵심 데이터, 그래프 설명, 해석/주의 안내.",
        "카드뉴스형": "- 카드뉴스는 5장 안팎으로 제안해 주세요. 각 장마다 제목, 한 줄 핵심 문장, 근거 데이터를 넣어 주세요.",
        "포스터형": "- 포스터형은 한 화면 안에 제목, 핵심 수치, 시각 요소, 행동 제안이 분명하게 보이도록 영역별로 제안해 주세요.",
    }.get(output_type, "- 결과물 유형에 맞는 구성안을 제시해 주세요.")

    return f"""{common_header}
[실제 문제와 목표]
- 문제 상황: {problem_situation}
- 결과물 목표: {project_goal}
- 데이터에서 드러난 핵심 의미: {data_meaning}

[사회적 의미와 책임]
- 사회에 주는 시사점/실천 가치: {social_value}
- 결과물의 한계와 주의점: {limit_note}
- 사람이 다시 확인할 점: {human_check}

[최종 요청]
- Canva AI에서 바로 제작할 수 있도록 결과물의 구성안과 문구를 짧고 구체적으로 작성해 주세요.
{structure_request}
- 데이터와 해석 내용을 시각 자료와 함께 자연스럽게 연결해 주세요.
- 학생이 한눈에 이해할 수 있도록 핵심 메시지는 한두 문장으로 정리해 주세요.
- 마지막에는 학생이 기억해야 할 핵심 메시지와 실천 제안을 포함해 주세요.
- 추가 요청: {prompt_extra}
"""


def create_prompt_pdf(group_name, dataset, prompt_text):
    materials = get_project_materials()
    output_type = st.session_state.get("d6_output_type", OUTPUT_TYPES[0])
    presentation_summary = build_presentation_summary_text()

    pdf = ThemedPDF()
    pdf.add_font("Nanum", "", font_path, uni=True)
    pdf.set_font("Nanum", "", 12)
    pdf._font_family = "Nanum"
    pdf.footer_left = group_name
    pdf.add_page()

    pdf.kv_card(
        "모둠 기획 정보",
        (
            [
            ("모둠명", group_name),
            ("결과물 유형", output_type),
            ("자료 출처 방식", materials["source_label"]),
            ("활동 데이터", materials["name"]),
            ("자료 주제", materials["topic"]),
            ("작성일", datetime.datetime.now().strftime("%Y-%m-%d")),
        ]
        + (
            [("독립 변수", materials["x_column"]), ("종속 변수", materials["y_column"])]
            if materials["uses_future"]
            else []
        )),
    )

    add_text_box_to_pdf(pdf, "프로젝트 제목", st.session_state.get("d6_project_title", ""))
    add_text_box_to_pdf(pdf, "주요 대상 선택 이유", st.session_state.get("d6_target_reason", ""))
    add_text_box_to_pdf(pdf, "프로젝트 자료 분석 내용", materials["analysis"])
    add_text_box_to_pdf(pdf, "프로젝트 자료 해석 내용", materials["interpretation"])
    add_text_box_to_pdf(pdf, "프로젝트 자료 예시", materials["sample_rows"])

    if output_type == "게임형":
        add_text_box_to_pdf(pdf, "게임으로 바꾼 핵심 의미", st.session_state.get("d6_game_meaning", ""))
        add_text_box_to_pdf(pdf, "게임 배경 상황", st.session_state.get("d6_game_story", ""))
        add_text_box_to_pdf(pdf, "게임 목표", st.session_state.get("d6_game_goal", ""))
        add_text_box_to_pdf(pdf, "게임 진행 방식", st.session_state.get("d6_game_rule", ""))
        add_text_box_to_pdf(pdf, "피드백 방식", st.session_state.get("d6_game_feedback", ""))
    else:
        add_text_box_to_pdf(pdf, "보여 주고 싶은 문제", st.session_state.get("d6_problem_situation", ""))
        add_text_box_to_pdf(pdf, "사용자에게 기대하는 변화", st.session_state.get("d6_project_goal", ""))
        add_text_box_to_pdf(pdf, "데이터 핵심 메시지", st.session_state.get("d6_data_meaning", ""))

    add_text_box_to_pdf(pdf, "이 앱이 주는 가치", st.session_state.get("d6_social_value", ""))
    add_text_box_to_pdf(pdf, "조심해서 봐야 할 점", st.session_state.get("d6_limit_note", ""))
    add_text_box_to_pdf(pdf, "사람이 다시 판단할 부분", st.session_state.get("d6_human_check", ""))
    add_text_box_to_pdf(pdf, "추가 요청", st.session_state.get("d6_prompt_extra", ""))
    add_text_box_to_pdf(pdf, "모둠 발표용 3문장 요약", presentation_summary)
    add_text_box_to_pdf(pdf, "최종 Canva 구현 프롬프트", prompt_text, fill_color=(250, 250, 250))
    return normalize_pdf_output(pdf.output(dest="S"))

    tabs = st.tabs(["5️⃣ [E] 세상과 연결"])
    with tabs[0]:
        stage_intro(
            "세상과 연결: 앱 제작 프롬프트 만들기",
            "데이터 예측 결과를 실제 사회 문제 해결과 연결하고, 앱 제작을 위한 구체적인 프롬프트로 바꾸는 단계입니다.",
            "data5에서 만든 예측을 실제로 도움이 되는 앱으로 바꾸려면 어떤 목적과 기능을 담아야 할까?",
        )

        dataset = current_dataset()
        st.markdown(pretty_title("data5에서 이어 온 탐구 정보", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        render_value_cards(
            [
                {
                    "title": "모둠명",
                    "value": st.session_state.get("d6_group", "") or "미입력",
                    "detail": "data5에서 입력한 모둠명을 그대로 이어받습니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "활동 데이터",
                    "value": dataset["name"],
                    "detail": "선택한 분야 안에서 실제 데이터셋 2개 중 하나를 기준으로 앱을 설계합니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "활동 분야",
                    "value": st.session_state.get("d6_field", ""),
                    "detail": "경제, 의학, 공학, 환경, 스포츠, 사회 중 하나를 고릅니다.",
                    "bg": "#ede7f6",
                    "border": "#b39ddb",
                },
                {
                    "title": "X / Y 변수",
                    "value": f"{dataset['x_column']} / {dataset['y_column']}",
                    "detail": "앱에서 입력값과 예측 결과로 이어질 핵심 변수입니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
                {
                    "title": "추천 예측 방식",
                    "value": st.session_state.get("d6_model_type", ""),
                    "detail": "data5에서 다룬 머신러닝·딥러닝 흐름을 앱에 어떻게 반영할지 정합니다.",
                    "bg": "#fce4ec",
                    "border": "#f48fb1",
                },
            ],
            columns=3,
        )

        st.markdown(pretty_title("앱이 다룰 데이터와 사용자 정하기", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
        top_left, top_mid, top_right = st.columns([0.8, 1.0, 1.0])
        with top_left:
            st.text_input("모둠명", key="d6_group", placeholder="예: 1모둠")
            st.selectbox("활동 분야", FIELD_ORDER, key="d6_field")
        field_dataset_names = FIELD_DATASETS[st.session_state["d6_field"]]
        if st.session_state.get("d6_dataset") not in field_dataset_names:
            st.session_state["d6_dataset"] = field_dataset_names[0]
        with top_mid:
            st.selectbox("활동 데이터셋", field_dataset_names, key="d6_dataset")
        with top_right:
            info = DATASETS[st.session_state["d6_dataset"]]
            all_columns = list(info["table"].columns)
            if st.session_state.get("d6_x_col") not in all_columns:
                st.session_state["d6_x_col"] = info["default_x"]
            st.selectbox("독립 변수(X)", all_columns, key="d6_x_col")
            valid_y = [col for col in all_columns if col != st.session_state["d6_x_col"]]
            if st.session_state.get("d6_y_col") not in valid_y:
                st.session_state["d6_y_col"] = info["default_y"] if info["default_y"] in valid_y else valid_y[0]
            st.selectbox("종속 변수(Y)", valid_y, key="d6_y_col")

        dataset = current_dataset()
        preview_col, story_col = st.columns([1.1, 0.9])
        with preview_col:
            preview_df = dataset["selected_table"].copy()
            preview_df.insert(0, "행 번호", range(1, len(preview_df) + 1))
            st.dataframe(preview_df, use_container_width=True, hide_index=True, height=260)
        with story_col:
            st.info(dataset["story"])
            st.markdown(f"**앱 아이디어 예시**: {DATASETS[st.session_state['d6_dataset']]['app_examples']}")
            st.selectbox("주요 사용자", TARGET_USERS, key="d6_target_user")
            st.selectbox("앱에 넣을 예측 방식", MODEL_OPTIONS, key="d6_model_type")

        st.markdown(pretty_title("실제 문제와 앱 목표 정하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        issue_col, goal_col = st.columns(2)
        with issue_col:
            st.text_input("앱 이름", key="d6_app_name", placeholder="예: 우리 동네 매출 예측 도우미")
            st.text_area(
                "이 앱이 해결하려는 실제 문제",
                key="d6_real_problem",
                height=140,
                placeholder="예: 소상공인이 광고를 얼마나 해야 매출이 늘어날지 감으로만 판단하기 어렵다.",
            )
        with goal_col:
            st.text_area(
                "앱의 목표",
                key="d6_app_goal",
                height=140,
                placeholder="예: 사용자가 광고 횟수를 입력하면 예상 매출과 그래프를 보여 주어 계획을 세우도록 돕는다.",
            )
            st.text_area(
                "예측 결과를 어떻게 보여 주고 싶은가",
                key="d6_main_output",
                height=110,
                placeholder="예: 예측값 카드, 그래프, 한 줄 해석, 주의 문구를 함께 보여 주고 싶다.",
            )

        st.markdown(pretty_title("앱 기능과 화면 요소 정하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        feature_col, style_col = st.columns([1.15, 0.85])
        with feature_col:
            st.multiselect(
                "앱에 꼭 넣고 싶은 기능",
                FEATURE_OPTIONS,
                key="d6_feature_options",
            )
        with style_col:
            st.selectbox("원하는 화면 분위기", STYLE_OPTIONS, key="d6_ui_style")
            st.info("복잡한 기능보다 입력-예측-해석-주의 안내 흐름이 분명한 교육용 앱이 되도록 정해 보세요.")

        st.markdown(pretty_title("신뢰와 책임까지 생각하기", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
        trust_col1, trust_col2, trust_col3 = st.columns(3)
        with trust_col1:
            st.text_area(
                "이 앱이 사회에 주는 가치",
                key="d6_social_value",
                height=120,
                placeholder="예: 데이터를 바탕으로 합리적인 판단을 돕고, 학생이 AI 예측의 쓰임을 이해하게 한다.",
            )
        with trust_col2:
            st.text_area(
                "예측의 한계와 주의점",
                key="d6_limit_note",
                height=120,
                placeholder="예: 데이터가 적으면 예측이 빗나갈 수 있고, 실제 상황을 모두 반영하지 못할 수 있다.",
            )
        with trust_col3:
            st.text_area(
                "사람이 다시 확인해야 하는 순간",
                key="d6_human_check",
                height=120,
                placeholder="예: 중요한 결정에 영향을 줄 때는 사람이 결과를 다시 확인해야 한다.",
            )

        st.markdown(pretty_title("AI에게 요청할 앱 제작 프롬프트", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.text_area(
            "프롬프트에 꼭 반영하고 싶은 추가 요청",
            key="d6_prompt_reflection",
            height=100,
            placeholder="예: 학생이 이해하기 쉬운 설명 문장과 예측 결과 해석 안내를 꼭 넣어 달라고 요청하고 싶다.",
        )
        prompt_text = build_prompt_text(dataset)
        st.code(prompt_text, language="markdown")

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "프롬프트 TXT 저장하기",
                data=prompt_text.encode("utf-8"),
                file_name=f"{clean_text(st.session_state.get('d6_group', ''), '모둠')}_앱제작프롬프트.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with download_col2:
            if st.session_state.get("d6_group", "").strip():
                pdf_bytes = create_prompt_pdf(st.session_state["d6_group"], dataset, prompt_text)
                st.download_button(
                    "앱 기획 PDF 저장하기",
                    data=pdf_bytes,
                    file_name=f"{st.session_state['d6_group']}_앱기획프롬프트.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("모둠명을 입력하면 앱 기획 PDF를 저장할 수 있습니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)
def run():
    apply_local_style()
    ensure_state()
    page_banner(
        "STEP 3-2. 결과를 앱으로 확장하기",
        "앞 차시에서 분석한 데이터 예측 결과를 실제 사람들에게 도움이 되는 결과물로 바꾸는 단계입니다. "
        "6DAY에서는 FUTURE 프로젝트 자료의 해석을 그대로 이어받아, Canva AI에서 만들 결과물의 종류를 정하고 그에 맞는 최종 프롬프트를 완성합니다.",
    )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)

    stage_intro(
        "세상과 연결: 어떤 결과물로 확장할까?",
        "이 단계에서는 FUTURE 프로젝트 자료의 데이터 분석 결과를 바탕으로, 실제로 사람들에게 보여 주거나 체험하게 할 결과물을 기획합니다. "
        "정보형, 카드뉴스형, 포스터형은 FUTURE 프로젝트 자료의 데이터를 그대로 활용하고, 게임형은 데이터에서 읽은 의미를 경험으로 바꾸는 데 집중합니다.",
        "우리 모둠은 FUTURE 프로젝트 자료의 분석 결과를 어떤 형태의 결과물로 바꾸면 가장 잘 전달할 수 있을까?",
    )
    render_future_flow_box()

    dataset = current_dataset()
    reports = get_inherited_reports()
    if st.session_state["d6_output_type"] != st.session_state.get("d6_last_output_type"):
        st.session_state["d6_feature_options"] = TYPE_FEATURE_OPTIONS[st.session_state["d6_output_type"]][:3]
        st.session_state["d6_last_output_type"] = st.session_state["d6_output_type"]

    subtabs = st.tabs(
        [
            "1️⃣[E] 앱 유형 선택",
            "2️⃣[E] 프로젝트 자료 정리",
            "3️⃣[E] 결과물 설계",
            "4️⃣[E] 프롬프트 확인",
        ]
    )

    with subtabs[0]:
        st.markdown(pretty_title("1. 어떤 앱 유형으로 만들까?", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
        material_label = "FUTURE 프로젝트 자료" if using_future_materials() else "직접 설계한 프로젝트 자료"

        st.info(
            "앱 유형 선택은 프롬프트를 쓰는 출발점입니다. 정보형은 설명 구조, 카드뉴스형은 전달 순서, "
            "포스터형은 한 장의 핵심 메시지, 게임형은 선택과 피드백 흐름이 중요합니다."
        )
        render_app_type_selector_cards(st.session_state["d6_output_type"])
        st.caption(
            f"즉, 같은 {material_label}라도 어떤 앱 유형을 고르느냐에 따라 Canva AI에 적어야 하는 프롬프트 방식이 달라집니다."
        )
        st.radio(
            "프로젝트 자료 반영 방식 선택",
            ["FUTURE 프로젝트 자료 반영하기", "학생이 직접 프로젝트 자료 설계하기"],
            key="d6_material_mode",
            horizontal=True,
        )
        if using_future_materials():
            st.info("현재는 FUTURE 프로젝트 자료를 그대로 이어받아 결과물을 기획합니다. 필요하면 아래에서 자료를 펼쳐 보고 앱 유형을 판단할 수 있습니다.")
            with st.expander("FUTURE 프로젝트 자료를 보며 앱 유형 정하기", expanded=False):
                preview_col, note_col = st.columns([1.05, 0.95])
                with preview_col:
                    preview_df = dataset["selected_table"].copy()
                    preview_df.insert(0, "행", range(1, len(preview_df) + 1))
                    st.dataframe(preview_df, use_container_width=True, hide_index=True, height=220)
                    st.caption("FUTURE 프로젝트 자료에서 선택한 X, Y 데이터를 그대로 확인하며 어떤 앱 유형이 잘 맞을지 생각해 볼 수 있습니다.")
                with note_col:
                    st.markdown("**FUTURE 프로젝트 자료 분석 내용**")
                    st.write(reports["analysis"])
                    st.markdown("**FUTURE 프로젝트 자료 결과 해석 내용**")
                    st.write(reports["interpretation"])
        else:
            st.info("현재는 FUTURE 프로젝트 자료를 사용하지 않고, 학생이 직접 프로젝트 자료를 설계하는 방식입니다. 다음 탭에서 자료 이름, 주제, 예시 자료, 해석을 직접 정리하게 됩니다.")

        st.markdown(pretty_title("가장 먼저, 앱 유형을 선택해 주세요", "#fff3e0", "#ffccbc"), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #fff8e1 0%, #ffe0b2 100%);
                border: 2px solid #ffb74d;
                border-radius: 18px;
                padding: 16px 18px;
                margin-bottom: 12px;
                box-shadow: 0 4px 12px rgba(255, 167, 38, 0.12);
            ">
                <div style="font-size:0.95rem; font-weight:700; color:#e65100; margin-bottom:6px;">앱 유형 선택이 가장 중요합니다</div>
                <div style="font-size:1.15rem; font-weight:800; color:#263238; margin-bottom:6px;">현재 선택: {st.session_state.get("d6_output_type", OUTPUT_TYPES[0])}</div>
                <div style="font-size:0.92rem; line-height:1.7; color:#5d4037;">
                    이 선택에 따라 뒤에서 작성할 프롬프트의 문장 구조와 필요한 입력 항목이 달라집니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        select_col, title_col = st.columns([1, 1])
        with select_col:
            st.text_input("모둠명", key="d6_group", placeholder="예: 1모둠")
            st.selectbox("주요 대상", TARGET_USERS, key="d6_target_user")
            st.text_input(
                "이 사용자를 선택한 이유",
                key="d6_target_reason",
                placeholder="예: 기후 데이터를 처음 접하는 학생들이 쉽게 이해하고 바로 행동으로 옮길 수 있게 돕고 싶다.",
            )
        with title_col:
            st.text_input("프로젝트 제목", key="d6_project_title", placeholder="예: 환경 데이터를 쉽게 알려 주는 카드뉴스")
            st.selectbox("원하는 시각 스타일", STYLE_OPTIONS, key="d6_visual_style")
            st.info(output_type_guide(st.session_state["d6_output_type"]))
            st.caption(prompt_input_guide(st.session_state["d6_output_type"]))
            focus_points = output_type_focus_points(st.session_state["d6_output_type"])
            if focus_points:
                st.markdown("**이 유형에서는 이 정도만 정하면 충분해요**")
                for item in focus_points:
                    st.markdown(f"- {item}")

    with subtabs[1]:
            if using_future_materials():
                st.markdown(pretty_title("2. FUTURE 프로젝트 자료의 데이터와 해석을 그대로 이어받기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
                data_col, report_col = st.columns([1.05, 0.95])
                with data_col:
                    preview_df = dataset["selected_table"].copy()
                    preview_df.insert(0, "행", range(1, len(preview_df) + 1))
                    st.dataframe(preview_df, use_container_width=True, hide_index=True, height=260)
                    st.caption("FUTURE 프로젝트 자료에서 선택한 X, Y 데이터를 그대로 이어받아 프롬프트의 근거 자료로 사용합니다.")
                with report_col:
                    with st.expander("FUTURE 프로젝트 자료 분석 내용 보기", expanded=True):
                        st.write(reports["analysis"])
                    with st.expander("FUTURE 프로젝트 자료 결과 해석 내용 보기", expanded=True):
                        st.write(reports["interpretation"])
            else:
                st.markdown(pretty_title("2. 학생이 직접 프로젝트 자료 설계하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
                left, right = st.columns(2)
                with left:
                    st.text_input("프로젝트 자료 이름", key="d6_custom_data_name", placeholder="예: 우리 학교 급식 만족도와 잔반량 자료")
                    st.info("직접 설계 모드에서는 독립변수와 종속변수를 꼭 나누지 않아도 됩니다. 자료의 주제와 의미를 중심으로 자유롭게 설계해 보세요.")
                    st.text_area(
                        "프로젝트 자료의 주제와 문제 상황",
                        key="d6_custom_topic",
                        height=120,
                        placeholder="예: 학생들의 급식 만족도와 실제 잔반량 사이의 관계를 바탕으로 더 나은 급식 선택 문화를 돕고 싶다.",
                    )
                with right:
                    st.text_area(
                        "예시 자료 또는 표 아이디어",
                        key="d6_custom_rows",
                        height=120,
                        placeholder="예: 만족도 2점-잔반량 많음, 만족도 4점-잔반량 보통, 만족도 5점-잔반량 적음",
                    )
                    st.text_area(
                        "자료의 특징 요약",
                        key="d6_custom_analysis",
                        height=120,
                        placeholder="예: 만족도가 높을수록 잔반량이 줄어드는 경향이 보일 것이라고 예상한다.",
                    )
                    st.text_area(
                        "자료 해석과 시사점",
                        key="d6_custom_interpretation",
                        height=120,
                        placeholder="예: 학생의 선택과 만족도를 반영하면 음식물 쓰레기를 줄이는 데 도움이 될 수 있다.",
                    )
                st.caption("직접 설계한 프로젝트 자료는 이후 결과물 설계와 최종 Canva 프롬프트에 그대로 반영됩니다.")

    with subtabs[2]:
            st.markdown(pretty_title("3. 결과물 방향을 간단히 정하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            output_type = st.session_state["d6_output_type"]
            recommended_pack = recommended_feature_pack(output_type)
            render_value_cards(
                [
                    {
                        "title": "추천 기능",
                        "value": ", ".join(recommended_pack["features"][:2]) if recommended_pack["features"] else "기본 구성",
                        "detail": "선택한 앱 유형에 맞춰 특히 잘 어울리는 기능을 추천합니다.",
                        "bg": "#f4f9ff",
                        "border": "#90caf9",
                    },
                    {
                        "title": "추천 이유",
                        "value": output_type,
                        "detail": recommended_pack["reason"],
                        "bg": "#fff8e1",
                        "border": "#ffcc80",
                    },
                ],
                columns=2,
            )

            feature_left, feature_right = st.columns([1.2, 0.8])
            with feature_left:
                st.multiselect(
                    "결과물에 꼭 넣고 싶은 기능",
                    TYPE_FEATURE_OPTIONS[output_type],
                    key="d6_feature_options",
                )
            with feature_right:
                st.button(
                    "추천 기능 적용",
                    key=f"d6_apply_features_{output_type}",
                    use_container_width=True,
                    on_click=apply_recommended_features,
                    args=(recommended_pack["features"],),
                )
                st.caption("추천 기능을 바탕으로 시작한 뒤, 모둠 아이디어에 맞게 더 추가하거나 빼도 됩니다.")

            if output_type == "게임형":
                st.info("게임형은 복잡한 기능보다 핵심 의미, 배경 상황, 목표, 진행 방식만 분명하면 충분합니다.")
                left, right = st.columns(2)
                with left:
                    st.text_area(
                        "게임으로 바꾸고 싶은 데이터 의미",
                        key="d6_game_meaning",
                        height=120,
                        placeholder="예: 광고 횟수가 늘어날수록 방문자 수와 매출이 함께 증가한다는 의미를 게임 속 선택 결과로 느끼게 하고 싶다.",
                    )
                    st.text_area(
                        "게임 배경 상황",
                        key="d6_game_story",
                        height=140,
                        placeholder="예: 플레이어는 작은 가게의 운영자가 되어 제한된 선택 속에서 더 나은 결과를 만들어야 한다.",
                    )
                    st.text_area(
                        "게임 목표",
                        key="d6_game_goal",
                        height=110,
                        placeholder="예: 주어진 선택을 통해 가장 좋은 결과를 만들며 데이터의 의미를 이해한다.",
                    )
                with right:
                    st.text_area(
                        "게임 진행 방식",
                        key="d6_game_rule",
                        height=160,
                        placeholder="예: 한 단계마다 선택지를 고르고, 선택 결과에 따라 점수와 메시지가 바뀌도록 한다.",
                    )
                    st.text_area(
                        "선택 뒤 피드백 방식",
                        key="d6_game_feedback",
                        height=160,
                        placeholder="예: 좋은 선택을 하면 데이터 의미를 설명하는 칭찬 메시지, 아쉬운 선택을 하면 다시 생각하게 하는 힌트를 준다.",
                    )
            else:
                st.info("정보형, 카드뉴스형, 포스터형은 세 가지만 정하면 충분합니다. 보여 주고 싶은 문제, 사용자가 느낄 변화, 데이터의 핵심 메시지를 정해 보세요.")
                left, right = st.columns(2)
                with left:
                    st.text_area(
                        "누구에게 어떤 문제를 보여 줄까?",
                        key="d6_problem_situation",
                        height=140,
                        placeholder="예: 기후 변화 데이터를 학생들이 숫자만 보고 지나치지 않고 생활 문제와 연결해 이해하도록 돕고 싶다.",
                    )
                    st.text_area(
                        "사용자가 보고 무엇을 느끼거나 하게 할까?",
                        key="d6_project_goal",
                        height=140,
                        placeholder="예: 사용자가 데이터를 보고 어떤 경향이 있는지 이해하고, 생활 속 실천까지 생각하게 한다.",
                    )
                with right:
                    st.text_area(
                        "데이터로 꼭 전하고 싶은 핵심 한 문장",
                        key="d6_data_meaning",
                        height=140,
                        placeholder="예: 데이터의 증가와 감소가 단순한 숫자 변화가 아니라 실제 생활과 연결된다는 점을 강조하고 싶다.",
                    )
                    st.info(f"{output_type}은 {output_type_guide(output_type)}")

            st.markdown(pretty_title("4. 마지막으로 꼭 넣을 안내", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
            st.markdown(pretty_title("사회적 가치와 실천을 돕는 질문 씨앗", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            render_value_cards(
                [
                    {
                        "title": "문제 예방",
                        "value": "이 결과물이 어떤 문제를 예방하는 데 도움이 될까?",
                        "detail": "단순 정보 제공이 아니라 실제 위험이나 혼란을 줄이는 방향을 떠올려 보세요.",
                        "bg": "#f4f9ff",
                        "border": "#90caf9",
                    },
                    {
                        "title": "더 나은 판단",
                        "value": "사람들이 어떤 판단을 더 잘 내리게 될까?",
                        "detail": "이 결과물을 본 뒤 사용자가 무엇을 더 똑똑하게 결정할 수 있을지 생각해 보세요.",
                        "bg": "#fff8e1",
                        "border": "#ffcc80",
                    },
                    {
                        "title": "필요한 대상",
                        "value": "이 결과물이 누구에게 가장 필요할까?",
                        "detail": "선택한 사용자 중에서도 특히 더 필요한 상황이나 사람을 떠올려 보세요.",
                        "bg": "#f1f8e9",
                        "border": "#aed581",
                    },
                    {
                        "title": "기대 행동",
                        "value": "결과를 보고 사용자가 무엇을 하게 되길 바랄까?",
                        "detail": "읽고 끝나는 것이 아니라 어떤 행동으로 이어지면 좋을지 적어 보세요.",
                        "bg": "#ede7f6",
                        "border": "#b39ddb",
                    },
                ],
                columns=2,
            )
            trust_col1, trust_col2, trust_col3 = st.columns(3)
            with trust_col1:
                st.text_area(
                    "이 앱이 주는 가치",
                    key="d6_social_value",
                    height=120,
                    placeholder="예: 데이터를 근거로 생각하고 판단하는 힘을 기를 수 있다.",
                )
            with trust_col2:
                st.text_area(
                    "조심해서 봐야 할 점",
                    key="d6_limit_note",
                    height=120,
                    placeholder="예: 적은 데이터만으로 만든 결과이므로 모든 상황을 대표한다고 볼 수는 없다.",
                )
            with trust_col3:
                st.text_area(
                    "사람이 다시 판단할 부분",
                    key="d6_human_check",
                    height=120,
                    placeholder="예: 실제 중요한 판단은 사람의 검토와 다른 자료 확인이 함께 필요하다.",
                )

    with subtabs[3]:
            st.markdown(pretty_title("4. 최종 프롬프트 확인", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.info("앞에서 고른 앱 유형, 프로젝트 자료, 핵심 의미, 주의점이 아래 프롬프트에 자동으로 연결됩니다. 마지막으로 꼭 넣고 싶은 요청만 짧게 덧붙이면 됩니다.")
            st.text_area(
                "추가 요청 또는 꼭 반영하고 싶은 점",
                key="d6_prompt_extra",
                height=110,
                placeholder="예: 학생 눈높이에 맞는 문장으로, 색감은 밝고 친근하게, 그래프 설명은 짧고 분명하게 넣어 달라.",
            )

            prompt_text = build_prompt_text(dataset)
            presentation_summary = build_presentation_summary_text()
            st.code(prompt_text, language="markdown")
            st.markdown(pretty_title("모둠 발표용 3문장 요약", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            st.code(presentation_summary, language="markdown")

            download_col1, download_col2 = st.columns(2)
            with download_col1:
                st.download_button(
                    "Canva 프롬프트 TXT 저장",
                    data=prompt_text.encode("utf-8"),
                    file_name=f"{clean_text(st.session_state.get('d6_group', ''), '모둠')}_앱제작프롬프트.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with download_col2:
                if st.session_state.get("d6_group", "").strip():
                    pdf_bytes = create_prompt_pdf(st.session_state["d6_group"], dataset, prompt_text)
                    st.download_button(
                        "Canva 기획 PDF 저장",
                        data=pdf_bytes,
                        file_name=f"{st.session_state['d6_group']}_6차시_앱기획프롬프트.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.info("모둠명을 입력하면 앱 기획 PDF를 저장할 수 있습니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
