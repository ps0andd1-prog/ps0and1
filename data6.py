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


CLASS_OPTIONS = ["1", "2", "5", "6"]
GALLERY_URLS = {
    "1": "https://padlet.com/ps0andd/g_1",
    "2": "https://padlet.com/ps0andd/g_2",
    "5": "https://padlet.com/ps0andd/g_5",
    "6": "https://padlet.com/ps0andd/g_6",
}
CANVA_AI_URL = "https://www.canva.com/ai"


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

FEATURE_DESCRIPTIONS = {
    "핵심 데이터 카드": "가장 중요한 수치나 비교 결과를 카드처럼 짧게 보여 줍니다.",
    "그래프 또는 시각화 자료": "데이터의 변화나 관계를 막대, 선, 아이콘 등으로 쉽게 보이게 합니다.",
    "짧은 설명 문장": "그래프나 수치가 뜻하는 바를 한두 문장으로 풀어 줍니다.",
    "주의 문구/한계 안내": "데이터를 해석할 때 조심해야 할 점이나 한계를 알려 줍니다.",
    "간단한 사용 안내": "사용자가 화면을 어떻게 보면 되는지 짧게 안내합니다.",
    "시작 화면 안내": "게임의 배경, 목표, 시작 방법을 처음 화면에서 알려 줍니다.",
    "선택 버튼 또는 단계 구성": "사용자가 선택하며 진행할 수 있도록 버튼이나 단계를 만듭니다.",
    "선택에 따른 피드백 문구": "선택 결과가 어떤 의미인지 바로 설명해 줍니다.",
    "점수/성공 조건": "사용자가 목표를 달성했는지 알 수 있게 점수나 성공 기준을 둡니다.",
    "마무리 배운 점 정리": "게임을 끝낸 뒤 데이터에서 배운 핵심 내용을 정리합니다.",
    "표지 카드": "카드뉴스의 주제와 핵심 메시지를 첫 장에서 보여 줍니다.",
    "문제 상황 소개 카드": "이 데이터가 왜 필요한지 실제 문제 상황을 설명합니다.",
    "데이터 설명 카드": "사용한 자료와 주요 변수를 쉽게 소개합니다.",
    "의미 해석 카드": "데이터를 보고 알 수 있는 의미를 카드 한 장으로 정리합니다.",
    "실천 제안 카드": "사용자가 할 수 있는 행동이나 해결 방향을 제안합니다.",
    "강한 제목": "한눈에 주제가 보이도록 짧고 힘 있는 제목을 넣습니다.",
    "핵심 수치 강조": "가장 중요한 숫자나 결과를 크게 보여 줍니다.",
    "그래프 또는 아이콘 시각화": "그래프나 아이콘으로 포스터의 메시지를 빠르게 전달합니다.",
    "한눈에 보이는 핵심 메시지": "포스터를 스쳐 봐도 남는 한 문장 메시지를 넣습니다.",
    "실천 제안 문구": "보는 사람이 바로 떠올릴 수 있는 행동 제안을 넣습니다.",
    "주의 문구": "자료를 오해하지 않도록 필요한 주의점을 짧게 넣습니다.",
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


def page_banner(title, description, question=None):
    question_html = ""
    if question:
        question_html = (
            '<div style="background:rgba(255,255,255,0.72);border-radius:12px;'
            'padding:10px 12px;border:1px solid rgba(255,255,255,0.85);'
            'color:#37474f;line-height:1.6;margin-top:12px;">'
            f"<b>핵심 탐구 질문</b><br>{question}</div>"
        )
    st.markdown(
        (
            '<div style="background:linear-gradient(135deg,#e3f2fd 0%,#d1c4e9 100%);'
            'border-radius:22px;padding:22px 24px;box-shadow:0 8px 20px rgba(33,150,243,0.10);'
            'border:1px solid #dbe7f3;margin-bottom:14px;">'
            '<div style="font-size:0.9rem;font-weight:700;color:#5e35b1;margin-bottom:8px;">F.U.T.U.R.E. 프로젝트</div>'
            f'<div style="font-size:1.9rem;font-weight:800;color:#1f2937;margin-bottom:8px;">{title}</div>'
            f'<div style="font-size:1rem;line-height:1.7;color:#37474f;">{description}</div>'
            f"{question_html}</div>"
        ),
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


def render_canva_gallery_links(class_key):
    class_key = str(class_key)
    gallery_url = GALLERY_URLS.get(class_key)
    link_cols = st.columns(2)
    with link_cols[0]:
        render_link_button(
            CANVA_AI_URL,
            "Canva AI 바로가기",
            "linear-gradient(90deg, #00c4cc 0%, #7d2ae8 100%)",
        )
    with link_cols[1]:
        if gallery_url:
            render_link_button(
                gallery_url,
                f"{class_key}반 갤러리 패들렛 이동하기",
                "linear-gradient(90deg, #7e57c2 0%, #42a5f5 100%)",
            )
        else:
            st.info("반을 선택하면 갤러리 패들렛 버튼이 나타납니다.")


def apply_output_type_selection(selected):
    st.session_state["d6_output_type"] = selected
    st.session_state["d6_feature_options"] = TYPE_FEATURE_OPTIONS[selected][:3]
    st.session_state["d6_last_output_type"] = selected


def apply_recommended_features(features):
    st.session_state["d6_feature_options"] = list(features)


def render_app_type_selector_cards(selected_type):
    items = app_type_overview_cards()
    columns_per_row = min(4, len(items))
    for start in range(0, len(items), columns_per_row):
        row_items = items[start:start + columns_per_row]
        row_cols = st.columns(columns_per_row)
        for col, item in zip(row_cols, row_items):
            with col:
                is_selected = item["title"] == selected_type
                border = "#fb8c00" if is_selected else item["border"]
                shadow = "0 5px 12px rgba(251, 140, 0, 0.16)" if is_selected else "0 3px 8px rgba(33, 150, 243, 0.07)"
                badge_bg = "#fff3e0" if is_selected else "rgba(255,255,255,0.72)"
                badge_text = "선택됨" if is_selected else "선택 가능"
                st.markdown(
                    f"""
                    <div style="
                        height:100%;
                        min-height:118px;
                        padding:10px 12px;
                        border-radius:12px;
                        background:{item['bg']};
                        border:2px solid {border};
                        box-shadow:{shadow};
                        margin-bottom:6px;
                        transition:all 0.2s ease;
                    ">
                        <div style="font-size:0.9rem; color:#37474f; margin-bottom:4px; font-weight:800;">{item['title']}</div>
                        <div style="font-size:1rem; color:#263238; font-weight:800; margin-bottom:5px;">{item['value']}</div>
                        <div style="font-size:0.8rem; color:#546e7a; line-height:1.45; margin-bottom:8px;">{item['detail']}</div>
                        <div style="
                            display:inline-block;
                            padding:3px 8px;
                            border-radius:999px;
                            background:{badge_bg};
                            border:1px solid rgba(0,0,0,0.08);
                            color:#6d4c41;
                            font-size:0.74rem;
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
    d5_field = st.session_state.get("d5_field", field_for_dataset(d5_dataset))
    st.session_state.setdefault("d6_group", "")
    st.session_state.setdefault("d6_class", CLASS_OPTIONS[0])
    if st.session_state.get("d6_class") not in CLASS_OPTIONS:
        st.session_state["d6_class"] = CLASS_OPTIONS[0]
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

    st.session_state.setdefault("d6_target_user", TARGET_USERS[0])
    st.session_state.setdefault("d6_target_reason", "")
    st.session_state.setdefault("d6_feature_options", TYPE_FEATURE_OPTIONS[OUTPUT_TYPES[0]][:3])
    st.session_state.setdefault("d6_material_mode", "FUTURE 프로젝트 자료 반영하기")
    st.session_state.setdefault("d6_ai_ethics", "")
    st.session_state.setdefault("d6_output_type", OUTPUT_TYPES[0])
    st.session_state.setdefault("d6_last_output_type", st.session_state["d6_output_type"])
    st.session_state.setdefault("d6_project_title", "")
    st.session_state.setdefault("d6_visual_style", STYLE_OPTIONS[0])
    st.session_state.setdefault("d6_prompt_extra", "")
    st.session_state.setdefault("d6_generated_prompt", "")
    st.session_state.setdefault("d6_game_story", "")
    st.session_state.setdefault("d6_game_goal", "")
    st.session_state.setdefault("d6_game_rule", "")
    st.session_state.setdefault("d6_game_feedback", "")
    st.session_state.setdefault("d6_info_question", "")
    st.session_state.setdefault("d6_info_key_data", "")
    st.session_state.setdefault("d6_info_action", "")
    st.session_state.setdefault("d6_card_cover", "")
    st.session_state.setdefault("d6_card_problem", "")
    st.session_state.setdefault("d6_card_interpretation", "")
    st.session_state.setdefault("d6_card_action", "")
    st.session_state.setdefault("d6_card_tone", "")
    st.session_state.setdefault("d6_poster_title", "")
    st.session_state.setdefault("d6_poster_number", "")
    st.session_state.setdefault("d6_poster_visual", "")
    st.session_state.setdefault("d6_poster_message", "")
    dataset_label = st.session_state["d6_dataset"].split(":")[0]
    if st.session_state.get("d6_project_title") == f"{dataset_label} 데이터 기반 앱 기획":
        st.session_state["d6_project_title"] = ""


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

    title = clean_text(st.session_state.get("d6_project_title", ""), "학생이 직접 설계한 앱")
    return {
        "name": title,
        "x_column": "",
        "y_column": "",
        "selected_table": None,
        "sample_rows": "별도 예시 자료를 입력하지 않고, 앱 설계 탭에서 작성한 방향을 바탕으로 구성합니다.",
        "analysis": "학생이 직접 정한 앱 유형별 작성 내용을 중심으로 구성합니다.",
        "interpretation": "앱 설계 탭의 화면 구성, 메시지, 행동 제안, 피드백 내용을 최종 프롬프트에 반영합니다.",
        "topic": title,
        "source_label": "학생 직접 앱 설계",
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
            "카드 흐름을 4장으로 압축해서 잡기",
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


def render_type_specific_prompt_inputs(output_type):
    st.markdown(pretty_title(f"{output_type} 앱에 맞게 프롬프트 내용 쓰기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
    if output_type == "정보형":
        st.info("정보형은 5DAY에서 이어 온 자료를 바탕으로, 사용자가 더 쉽게 이해하도록 어떤 내용과 기능을 추가할지 정합니다.")
        st.text_area(
            "첫 화면에서 던질 질문 또는 문제",
            key="d6_info_question",
            height=90,
            placeholder="예: 우리 지역의 기후 변화는 생활에 어떤 영향을 줄까?",
        )
        left, right = st.columns(2)
        with left:
            st.text_area(
                "앱에 추가로 보여 주고 싶은 내용",
                key="d6_info_key_data",
                height=105,
                placeholder="예: 그래프 아래에 사용자가 쉽게 이해할 수 있는 핵심 해석 카드와 예시 상황을 넣고 싶다.",
            )
        with right:
            st.text_area(
                "앱에 넣고 싶은 상호작용 또는 기능",
                key="d6_info_action",
                height=105,
                placeholder="예: 사용자가 궁금한 값을 선택하면 관련 설명이 바뀌거나, 핵심 내용을 퀴즈처럼 확인하는 기능을 넣고 싶다.",
            )
    elif output_type == "카드뉴스형":
        st.info("카드뉴스형은 4장으로 압축해 만듭니다. 표지-문제/데이터-의미 해석-실천 제안 흐름으로 정리합니다.")
        left, right = st.columns(2)
        with left:
            st.text_area(
                "1장 표지 핵심 문장",
                key="d6_card_cover",
                height=95,
                placeholder="예: 숫자로 보는 우리 생활 속 변화",
            )
            st.text_area(
                "2장 문제 상황과 데이터 소개",
                key="d6_card_problem",
                height=105,
                placeholder="예: 어떤 문제가 있고, 5DAY 자료에서 어떤 변수와 그래프를 보면 되는지 함께 소개한다.",
            )
        with right:
            st.text_area(
                "3장 의미 해석",
                key="d6_card_interpretation",
                height=95,
                placeholder="예: 데이터의 경향이 실제 생활에서 어떤 의미인지 한 문장으로 정리한다.",
            )
            st.text_area(
                "4장 실천 제안",
                key="d6_card_action",
                height=105,
                placeholder="예: 사용자가 오늘부터 할 수 있는 작은 행동을 제안한다.",
            )
        st.text_area(
            "카드뉴스 분위기",
            key="d6_card_tone",
            height=80,
            placeholder="예: 짧고 선명한 문장, 밝은 색감, 학생 눈높이 표현",
        )
    elif output_type == "포스터형":
        st.info("포스터형은 한 화면 안에서 제목, 핵심 수치, 시각 자료, 행동 제안이 바로 보여야 합니다.")
        left, right = st.columns(2)
        with left:
            st.text_area(
                "1. 포스터에 크게 넣을 제목",
                key="d6_poster_title",
                height=95,
                placeholder="예: 데이터가 알려 주는 작은 변화의 신호",
            )
            st.text_area(
                "2. 가장 강조할 숫자 또는 결과",
                key="d6_poster_number",
                height=105,
                placeholder="예: 그래프에서 가장 눈에 띄는 증가, 감소, 차이를 크게 보여 준다.",
            )
        with right:
            st.text_area(
                "3. 넣고 싶은 시각 요소",
                key="d6_poster_visual",
                height=95,
                placeholder="예: 그래프 1개, 아이콘 3개, 핵심 수치를 담은 큰 숫자",
            )
            st.text_area(
                "4. 핵심 메시지와 행동 제안",
                key="d6_poster_message",
                height=105,
                placeholder="예: 데이터는 생활 속 선택을 더 나은 방향으로 바꾸는 힌트가 된다. 결과를 보고 자신의 생활 습관이나 선택을 점검해 보자.",
            )
    else:
        st.info("게임형은 복잡한 기능보다 핵심 의미, 배경 상황, 목표, 진행 방식만 분명하면 충분합니다.")
        left, right = st.columns(2)
        with left:
            st.text_area(
                "게임 배경 상황",
                key="d6_game_story",
                height=110,
                placeholder="예: 플레이어는 작은 가게의 운영자가 되어 제한된 선택 속에서 더 나은 결과를 만들어야 한다.",
            )
            st.text_area(
                "게임 목표",
                key="d6_game_goal",
                height=100,
                placeholder="예: 주어진 선택을 통해 가장 좋은 결과를 만들며 데이터의 의미를 이해한다.",
            )
        with right:
            st.text_area(
                "게임 진행 방식",
                key="d6_game_rule",
                height=110,
                placeholder="예: 한 단계마다 선택지를 고르고, 선택 결과에 따라 점수와 메시지가 바뀌도록 한다.",
            )
            st.text_area(
                "선택 뒤 피드백 방식",
                key="d6_game_feedback",
                height=100,
                placeholder="예: 좋은 선택을 하면 데이터 의미를 설명하는 칭찬 메시지, 아쉬운 선택을 하면 다시 생각하게 하는 힌트를 준다.",
            )


def app_type_prompt_plan(output_type):
    if output_type == "정보형":
        return f"""[정보형 앱 작성 내용]
- 첫 화면 질문/문제: {clean_text(st.session_state.get("d6_info_question", ""), "첫 화면에서 던질 질문이나 문제를 적어 주세요.")}
- 추가로 보여 주고 싶은 내용: {clean_text(st.session_state.get("d6_info_key_data", ""), "5DAY 자료를 바탕으로 앱에 추가하고 싶은 내용이나 화면 요소를 적어 주세요.")}
- 넣고 싶은 상호작용/기능: {clean_text(st.session_state.get("d6_info_action", ""), "정보형 앱에 넣고 싶은 상호작용이나 기능을 자유롭게 적어 주세요.")}"""
    if output_type == "카드뉴스형":
        return f"""[카드뉴스형 앱 작성 내용]
- 1장 표지 핵심 문장: {clean_text(st.session_state.get("d6_card_cover", ""), "표지에 넣을 핵심 문장을 적어 주세요.")}
- 2장 문제 상황과 데이터 소개: {clean_text(st.session_state.get("d6_card_problem", ""), "문제 상황과 5DAY 자료를 어떻게 소개할지 적어 주세요.")}
- 3장 의미 해석: {clean_text(st.session_state.get("d6_card_interpretation", ""), "데이터의 의미 해석을 적어 주세요.")}
- 4장 실천 제안: {clean_text(st.session_state.get("d6_card_action", ""), "마지막 실천 제안을 적어 주세요.")}
- 카드뉴스 분위기: {clean_text(st.session_state.get("d6_card_tone", ""), "카드뉴스의 문장과 시각 분위기를 적어 주세요.")}"""
    if output_type == "포스터형":
        return f"""[포스터형 앱 작성 내용]
- 1. 크게 넣을 제목: {clean_text(st.session_state.get("d6_poster_title", ""), "포스터 제목을 적어 주세요.")}
- 2. 강조할 숫자/결과: {clean_text(st.session_state.get("d6_poster_number", ""), "강조할 숫자나 결과를 적어 주세요.")}
- 3. 시각 요소: {clean_text(st.session_state.get("d6_poster_visual", ""), "넣고 싶은 그래프, 아이콘, 배치 요소를 적어 주세요.")}
- 4. 핵심 메시지와 행동 제안: {clean_text(st.session_state.get("d6_poster_message", ""), "한눈에 남길 핵심 메시지와 행동 제안을 적어 주세요.")}"""
    return f"""[게임형 앱 작성 내용]
- 배경 상황: {clean_text(st.session_state.get("d6_game_story", ""), "게임의 배경 이야기와 상황을 적어 주세요.")}
- 게임 목표: {clean_text(st.session_state.get("d6_game_goal", ""), "플레이어가 게임에서 이루어야 할 목표를 적어 주세요.")}
- 규칙과 진행 방식: {clean_text(st.session_state.get("d6_game_rule", ""), "플레이 규칙과 진행 방식을 적어 주세요.")}
- 피드백 방식: {clean_text(st.session_state.get("d6_game_feedback", ""), "성공·실패·선택 결과에 따라 어떤 피드백을 줄지 적어 주세요.")}"""


def build_prompt_text():
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
        "- 이 앱은 FUTURE 프로젝트 자료에서 이미 분석한 데이터와 해석을 그대로 이어받아 활용합니다."
        if materials["uses_future"]
        else "- 이 앱은 별도 자료 정리보다 학생이 직접 정한 앱 설계 방향을 중심으로 구성합니다."
    )
    source_request = (
        "- 보고서를 다시 업로드하라고 요구하지 말고, 아래에 제공한 데이터와 분석 내용을 바로 활용해 주세요."
        if materials["uses_future"]
        else "- 아래 앱 유형별 작성 내용을 바탕으로 화면 구성, 문구, 기능 흐름을 구체화해 주세요."
    )
    axis_lines = (
        f"- 독립 변수(X): {materials['x_column']}\n- 종속 변수(Y): {materials['y_column']}\n"
        if materials["uses_future"]
        else ""
    )
    focus_lines = "\n".join(f"- {item}" for item in output_type_focus_points(output_type))
    if not focus_lines:
        focus_lines = "- 핵심 메시지가 분명하게 보이도록 구성해 주세요."
    type_plan = app_type_prompt_plan(output_type)

    if materials["uses_future"]:
        material_section = f"""[프로젝트 자료]
- 자료 이름: {materials['name']}
- 자료 출처 방식: {materials['source_label']}
- 자료가 다루는 주제/문제 상황: {materials['topic']}
{axis_lines}
- 선택한 데이터 일부:
{materials['sample_rows']}

[프로젝트 자료 분석 내용]
{materials['analysis']}

[프로젝트 자료 해석 내용]
{materials['interpretation']}"""
    else:
        material_section = f"""[직접 앱 설계 방식]
- 자료 출처 방식: {materials['source_label']}
- 별도 자료 입력 없이, 학생이 앱 설계 탭에서 정한 방향을 중심으로 앱을 만듭니다.
- 프로젝트 제목: {title}"""

    common_header = f"""Canva AI의 바이브 코딩으로 바로 구현할 학생용 앱을 제작해 주세요.

[기본 정보]
- 앱 유형: {output_type}
- 프로젝트 제목: {title}
- 제작 주체: {group_name}
- 주요 대상: {target_user}
- 이 사용자를 선택한 이유: {target_reason}
- 시각 스타일: {style}

[중요 조건]
{source_condition}
{source_request}
- 고등학교 1학년 학생이 바로 이해하고 사용할 수 있게 만듭니다.
- 데이터의 의미, 해석, 주의점이 앱 안에서 자연스럽게 보이게 합니다.
- 로그인, 관리자 기능, 긴 메뉴는 넣지 않고 단순한 화면 흐름으로 구성합니다.

{material_section}

[앱 유형에 맞는 상호작용 기능]
{feature_lines}

[이 앱 유형에서 중요한 점]
{focus_lines}

[앱 방향]
{output_type_guide(output_type)}

{type_plan}
"""

    ai_ethics = clean_text(
        st.session_state.get("d6_ai_ethics", ""),
        "AI가 만든 앱을 사용할 때 데이터의 한계, 과장 금지, 사람의 확인이 필요하다는 점을 적어 주세요.",
    )
    prompt_extra = clean_text(st.session_state.get("d6_prompt_extra", ""), "추가 요청이 있으면 적어 주세요.")

    if output_type == "게임형":
        return f"""{common_header}
[AI 윤리와 사용 주의]
- {ai_ethics}

[최종 요청]
- 간단한 교육용 게임 앱으로 구현해 주세요.
- 화면은 4개 안팎으로 구성합니다: 시작, 선택, 피드백, 마무리.
- 선택지는 단계마다 2~3개로 제한합니다.
- 질문, 선택, 피드백을 통해 데이터의 의미를 이해하게 합니다.
- 각 화면에 들어갈 문구와 기능을 짧고 구체적으로 써 주세요.
- 추가 요청: {prompt_extra}
"""

    structure_request = {
        "정보형": "- 정보형 앱은 4개 안팎의 섹션으로 구성합니다: 제목, 핵심 데이터, 그래프 설명, 해석/주의 안내.",
        "카드뉴스형": "- 카드뉴스형 앱은 4장으로 압축해 구성합니다: 표지, 문제/데이터, 의미 해석, 실천 제안.",
        "포스터형": "- 포스터형 앱은 한 화면에 제목, 핵심 수치, 시각 요소, 행동 제안이 보이게 구성합니다.",
    }.get(output_type, "- 선택한 앱 유형에 맞는 구성안을 제시해 주세요.")

    return f"""{common_header}
[AI 윤리와 사용 주의]
- {ai_ethics}

[최종 요청]
- Canva AI에서 바로 구현할 수 있는 앱 구성안과 화면 문구를 짧고 구체적으로 작성해 주세요.
{structure_request}
- 데이터와 해석을 시각 요소와 연결해 주세요.
- 핵심 메시지는 한두 문장으로 정리해 주세요.
- 마지막 화면에는 핵심 메시지와 실천 제안을 넣어 주세요.
- 추가 요청: {prompt_extra}
"""


def create_prompt_pdf(group_name, prompt_text):
    materials = get_project_materials()
    output_type = st.session_state.get("d6_output_type", OUTPUT_TYPES[0])

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
            ("앱 유형", output_type),
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
    if materials["uses_future"]:
        add_text_box_to_pdf(pdf, "프로젝트 자료 분석 내용", materials["analysis"])
        add_text_box_to_pdf(pdf, "프로젝트 자료 해석 내용", materials["interpretation"])
        add_text_box_to_pdf(pdf, "프로젝트 자료 예시", materials["sample_rows"])
    else:
        add_text_box_to_pdf(pdf, "직접 앱 설계 방식", "별도 자료 입력 없이 앱 설계 탭에서 작성한 유형별 방향을 중심으로 구성했습니다.")

    add_text_box_to_pdf(pdf, f"{output_type} 작성 내용", app_type_prompt_plan(output_type))
    add_text_box_to_pdf(pdf, "AI 윤리와 사용 주의", st.session_state.get("d6_ai_ethics", ""))
    add_text_box_to_pdf(pdf, "추가 요청", st.session_state.get("d6_prompt_extra", ""))
    add_text_box_to_pdf(pdf, "최종 Canva 구현 프롬프트", prompt_text, fill_color=(250, 250, 250))
    return normalize_pdf_output(pdf.output(dest="S"))

def run():
    apply_local_style()
    ensure_state()
    page_banner(
        "AI 바이브 코딩으로 앱 제작",
        "앞 차시에서 분석한 데이터 예측 결과를 실제 사람들에게 도움이 되는 앱으로 바꾸는 단계입니다. "
        "6DAY에서는 FUTURE 프로젝트 자료의 해석을 그대로 이어받아, Canva AI에서 만들 앱의 종류를 정하고 그에 맞는 최종 프롬프트를 완성합니다.",
        "AI 예측 결과를 어떻게 사회적 실천으로 확장할까?",
    )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)

    dataset = current_dataset()
    reports = get_inherited_reports()
    if st.session_state["d6_output_type"] != st.session_state.get("d6_last_output_type"):
        st.session_state["d6_feature_options"] = TYPE_FEATURE_OPTIONS[st.session_state["d6_output_type"]][:3]
        st.session_state["d6_last_output_type"] = st.session_state["d6_output_type"]

    subtabs = st.tabs(
        [
            "1️⃣[E] 앱 유형 선택",
            "2️⃣[E] 프로젝트 설정",
            "3️⃣[E] 앱 설계",
            "4️⃣[E] 프롬프트 확인",
        ]
    )

    with subtabs[0]:
        st.markdown(pretty_title("1. 앱 유형 선택", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
        class_col, group_col = st.columns([0.35, 0.65])
        with class_col:
            st.selectbox("반", CLASS_OPTIONS, key="d6_class")
        with group_col:
            st.text_input("모둠명", key="d6_group", placeholder="예: 1모둠")

        material_label = "FUTURE 프로젝트 자료" if using_future_materials() else "직접 설계한 앱 방향"

        st.info(
            "앱 유형 선택은 프롬프트를 쓰는 출발점입니다. 정보형은 설명 구조, 카드뉴스형은 전달 순서, "
            "포스터형은 한 장의 핵심 메시지, 게임형은 선택과 피드백 흐름이 중요합니다."
        )
        render_app_type_selector_cards(st.session_state["d6_output_type"])
        st.caption(
            f"즉, 같은 {material_label}라도 어떤 앱 유형을 고르느냐에 따라 Canva AI에 적어야 하는 프롬프트 방식이 달라집니다."
        )

    with subtabs[1]:
        st.markdown(pretty_title("2. 프로젝트 설정", "#fff3e0", "#ffccbc"), unsafe_allow_html=True)

        material_col, project_col = st.columns([0.9, 1.1])
        with material_col:
            st.markdown("**프로젝트 자료 반영하기**")
            st.caption("기본값은 FUTURE 프로젝트 자료 반영입니다.")
            future_selected = using_future_materials()
            future_btn, custom_btn = st.columns(2)
            with future_btn:
                if st.button(
                    "자료 반영",
                    key="d6_use_future_materials",
                    type="primary" if future_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["d6_material_mode"] = "FUTURE 프로젝트 자료 반영하기"
                    st.rerun()
            with custom_btn:
                if st.button(
                    "직접 설계",
                    key="d6_use_custom_materials",
                    type="secondary" if future_selected else "primary",
                    use_container_width=True,
                ):
                    st.session_state["d6_material_mode"] = "학생이 직접 앱 설계하기"
                    st.rerun()
            if using_future_materials():
                st.success("FUTURE 프로젝트 자료를 앱 기획에 반영합니다.")
                with st.expander("반영할 FUTURE 프로젝트 자료 확인", expanded=False):
                    preview_df = dataset["selected_table"].copy()
                    preview_df.insert(0, "행", range(1, len(preview_df) + 1))
                    st.dataframe(preview_df, use_container_width=True, hide_index=True, height=180)
                    st.markdown("**분석 내용**")
                    st.write(reports["analysis"])
                    st.markdown("**결과 해석 내용**")
                    st.write(reports["interpretation"])
            else:
                st.info("직접 설계 모드입니다. 별도 자료 입력 없이 다음 탭에서 학생이 원하는 앱 방향을 유형별로 설계합니다.")

        with project_col:
            st.text_input("프로젝트 제목", key="d6_project_title", placeholder="예: 환경 데이터를 쉽게 알려 주는 앱")
            target_col, style_col = st.columns([1, 1])
            with target_col:
                st.selectbox("주요 대상", TARGET_USERS, key="d6_target_user")
            with style_col:
                st.selectbox("원하는 시각 스타일", STYLE_OPTIONS, key="d6_visual_style")
            st.text_area(
                f"{st.session_state.get('d6_target_user', '이 사용자')}를 선택한 이유",
                key="d6_target_reason",
                height=86,
                placeholder=f"{st.session_state.get('d6_target_user', '주요 대상')}에게 왜 이 앱이 필요한지, 어떤 도움을 주고 싶은지 적어 보세요.",
            )
            st.info(output_type_guide(st.session_state["d6_output_type"]))
            st.caption(prompt_input_guide(st.session_state["d6_output_type"]))

    with subtabs[2]:
            st.markdown(pretty_title("3. 앱 방향을 간단히 정하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
            output_type = st.session_state["d6_output_type"]
            recommended_pack = recommended_feature_pack(output_type)

            feature_left, feature_right = st.columns([1.2, 0.8])
            with feature_left:
                st.multiselect(
                    "앱 유형에 맞는 상호작용 기능",
                    TYPE_FEATURE_OPTIONS[output_type],
                    key="d6_feature_options",
                )
            with feature_right:
                st.button(
                    "추천 상호작용 적용",
                    key=f"d6_apply_features_{output_type}",
                    use_container_width=True,
                    on_click=apply_recommended_features,
                    args=(recommended_pack["features"],),
                )
                st.caption("추천 상호작용을 바탕으로 시작한 뒤, 모둠 아이디어에 맞게 더 추가하거나 빼도 됩니다.")

            with st.expander("앱 유형에 맞는 상호작용 설명 보기", expanded=False):
                recommended_features = set(recommended_pack["features"])
                for feature in TYPE_FEATURE_OPTIONS[output_type]:
                    badge = " 추천" if feature in recommended_features else ""
                    description = FEATURE_DESCRIPTIONS.get(feature, "이 앱 유형에 맞게 추가할 수 있는 구성 요소입니다.")
                    st.markdown(f"- **{feature}**{badge}: {description}")
                st.markdown("**추천 기준**")
                st.write(recommended_pack["reason"])

            render_type_specific_prompt_inputs(output_type)

            st.markdown(pretty_title("4. AI 윤리와 사용 주의", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
            st.text_area(
                "AI가 만든 앱을 사용할 때 꼭 안내해야 할 점",
                key="d6_ai_ethics",
                height=130,
                placeholder="예: 이 앱의 결과는 참고용이며, 데이터가 적거나 한쪽으로 치우치면 결과가 달라질 수 있다. 중요한 판단은 사람이 다른 자료와 함께 다시 확인해야 한다.",
            )

    with subtabs[3]:
            st.markdown(pretty_title("5. 최종 프롬프트 확인", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
            st.info("앞에서 고른 앱 유형, 프로젝트 설정, 앱 설계 내용, AI 윤리 안내가 아래 프롬프트에 자동으로 연결됩니다. 마지막으로 꼭 넣고 싶은 요청만 짧게 덧붙이면 됩니다.")
            st.text_area(
                "추가 요청 또는 꼭 반영하고 싶은 점",
                key="d6_prompt_extra",
                height=110,
                placeholder="예: 학생 눈높이에 맞는 문장으로, 색감은 밝고 친근하게, 그래프 설명은 짧고 분명하게 넣어 달라.",
            )

            if st.button("프롬프트 생성", key="d6_generate_prompt_btn", use_container_width=True):
                st.session_state["d6_generated_prompt"] = build_prompt_text()

            prompt_text = st.session_state.get("d6_generated_prompt", "")
            if prompt_text:
                st.success("프롬프트가 생성되었습니다. 입력 내용을 바꾸었다면 버튼을 다시 눌러 새 프롬프트를 만들어 주세요.")
                st.code(prompt_text, language="markdown")
                render_canva_gallery_links(st.session_state.get("d6_class", CLASS_OPTIONS[0]))

                if st.session_state.get("d6_group", "").strip():
                    pdf_bytes = create_prompt_pdf(st.session_state["d6_group"], prompt_text)
                    st.download_button(
                        "Canva 기획 PDF 저장",
                        data=pdf_bytes,
                        file_name=f"{st.session_state['d6_group']}_6차시_앱기획프롬프트.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.warning("⚠️ 모둠원들이 동시에 PDF 다운로드 버튼을 누르면 오류가 날 수 있습니다. 한 명씩 차례대로 눌러 주세요.")
                else:
                    st.info("모둠명을 입력하면 앱 기획 PDF를 저장할 수 있습니다.")
            else:
                st.info("프롬프트 생성 버튼을 누르면 완성된 프롬프트를 확인할 수 있습니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
