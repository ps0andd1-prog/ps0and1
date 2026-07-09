# 실행 명령: streamlit run data8.py

import html
import os

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import streamlit as st


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


DATASETS = {
    "세계 인터넷 이용률": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [6.72, 15.60, 28.50, 39.90, 59.30, 67.40],
        "x_label": "2000년 이후 경과년",
        "y_label": "인터넷 이용 인구 비율(%)",
        "source": "World Bank WDI / FRED: Individuals using the Internet (% of population)",
        "source_url": "https://fred.stlouisfed.org/series/ITNETUSERP2WLD",
        "deep_question": "인터넷 이용률이 높아질수록 정보 접근의 불평등은 정말 줄어들까?",
    },
    "한국 합계출산율": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [1.48, 1.085, 1.226, 1.239, 0.837, 0.721],
        "x_label": "2000년 이후 경과년",
        "y_label": "합계출산율(명)",
        "source": "World Bank WDI / FRED: Fertility Rate, Total for the Republic of Korea",
        "source_url": "https://fred.stlouisfed.org/series/SPDYNTFRTINKOR",
        "deep_question": "출산율의 감소를 개인의 선택 문제로만 설명해도 될까?",
    },
    "한국 고령인구 비율": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [7.091, 9.000, 10.992, 12.955, 15.822, 18.335],
        "x_label": "2000년 이후 경과년",
        "y_label": "65세 이상 인구 비율(%)",
        "source": "World Bank WDI / FRED: Population ages 65 and above for the Republic of Korea",
        "source_url": "https://fred.stlouisfed.org/series/SPPOP65UPTOZSKOR",
        "deep_question": "고령인구가 늘어나는 사회에서 돌봄과 노동의 책임은 어떻게 나누어야 할까?",
    },
    "한국 청년실업률": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [10.002, 9.321, 8.540, 9.899, 10.143, 5.405],
        "x_label": "2000년 이후 경과년",
        "y_label": "청년실업률(%)",
        "source": "World Bank WDI / FRED: Youth Unemployment Rate for the Republic of Korea",
        "source_url": "https://fred.stlouisfed.org/series/SLUEM1524ZSKOR",
        "deep_question": "청년실업률이 낮아졌다는 수치만으로 청년의 삶이 나아졌다고 말할 수 있을까?",
    },
    "세계 기대수명": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [67.650, 69.111, 70.683, 71.966, 72.183, 73.330],
        "x_label": "2000년 이후 경과년",
        "y_label": "기대수명(년)",
        "source": "World Bank WDI / FRED: Life Expectancy at Birth, Total for World",
        "source_url": "https://fred.stlouisfed.org/series/SPDYNLE00INWLD",
        "deep_question": "기대수명이 늘어나는 것이 모든 사람의 건강한 삶을 보장한다는 뜻일까?",
    },
    "세계 전기 접근성": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [78.221, 80.701, 83.447, 86.924, 90.396, 91.599],
        "x_label": "2000년 이후 경과년",
        "y_label": "전기 접근 가능 인구 비율(%)",
        "source": "Our World in Data / World Bank: Share of the population with access to electricity",
        "source_url": "https://ourworldindata.org/grapher/share-of-the-population-with-access-to-electricity",
        "deep_question": "전기를 사용할 수 있다는 것은 교육·의료·안전의 기회와 어떻게 연결될까?",
    },
    "세계 도시화율": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [46.836, 49.368, 51.778, 54.470, 56.503, 57.354],
        "x_label": "2000년 이후 경과년",
        "y_label": "도시 거주 인구 비율(%)",
        "source": "Our World in Data / World Bank: Urban population (% of total population)",
        "source_url": "https://ourworldindata.org/grapher/share-of-population-urban",
        "deep_question": "도시 인구가 늘어나는 흐름은 주거, 교통, 환경 문제를 어떻게 바꿀까?",
    },
    "한국 초미세먼지 노출": {
        "years": [2000, 2005, 2010, 2015, 2020],
        "x": [0, 5, 10, 15, 20],
        "y": [25.438, 24.474, 21.791, 26.290, 25.944],
        "x_label": "2000년 이후 경과년",
        "y_label": "PM2.5 평균 노출(μg/m³)",
        "source": "Our World in Data / World Bank: PM2.5 air pollution, mean annual exposure",
        "source_url": "https://ourworldindata.org/grapher/average-exposure-pm25-pollution",
        "deep_question": "대기오염 수치가 오르내릴 때, 누구의 건강이 더 크게 영향을 받을까?",
    },
    "세계 재생에너지 비중": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [7.426, 7.192, 8.370, 10.000, 12.781, 13.943],
        "x_label": "2000년 이후 경과년",
        "y_label": "재생에너지 소비 비중(%)",
        "source": "Our World in Data: Renewable share of energy",
        "source_url": "https://ourworldindata.org/grapher/renewable-share-energy",
        "deep_question": "재생에너지 비중이 증가해도 기후 위기 대응이 충분하다고 말할 수 있을까?",
    },
    "한국 1인당 CO₂ 배출량": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023],
        "x": [0, 5, 10, 15, 20, 23],
        "y": [9.408, 10.422, 12.187, 12.439, 11.524, 11.385],
        "x_label": "2000년 이후 경과년",
        "y_label": "1인당 CO₂ 배출량(톤)",
        "source": "Our World in Data: CO₂ emissions per capita",
        "source_url": "https://ourworldindata.org/grapher/co-emissions-per-capita",
        "deep_question": "1인당 탄소배출량의 변화는 개인의 책임과 산업 구조의 책임을 어떻게 함께 보여 줄까?",
    },
    "세계 극빈곤 인구 비율": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023, 2024],
        "x": [0, 5, 10, 15, 20, 23, 24],
        "y": [36.205, 28.330, 20.984, 13.422, 11.412, 10.618, 10.400],
        "x_label": "2000년 이후 경과년",
        "y_label": "하루 $3 미만 생활 인구 비율(%)",
        "source": "Our World in Data / World Bank Poverty and Inequality Platform: Share of population in extreme poverty",
        "source_url": "https://ourworldindata.org/grapher/share-of-population-in-extreme-poverty",
        "deep_question": "세계 빈곤이 줄어든 흐름을 보면서도, 아직 남은 빈곤 문제를 어떻게 함께 말할 수 있을까?",
        "factfulness_lens": "부정 본능 점검",
    },
    "세계 5세 미만 아동 사망률": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023, 2024],
        "x": [0, 5, 10, 15, 20, 23, 24],
        "y": [7.668, 6.233, 5.059, 4.288, 3.916, 3.833, 3.737],
        "x_label": "2000년 이후 경과년",
        "y_label": "5세 미만 사망률(출생 100명당)",
        "source": "Our World in Data / UN IGME: Child mortality rate",
        "source_url": "https://ourworldindata.org/grapher/under-5-mortality-rate-sdgs",
        "deep_question": "아동 사망률이 낮아졌다는 사실은 보건·영양·위생의 어떤 변화를 보여 줄까?",
        "factfulness_lens": "부정 본능 점검",
    },
    "세계 성인 문해율": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023, 2024],
        "x": [0, 5, 10, 15, 20, 23, 24],
        "y": [81.12, 82.57, 84.37, 85.96, 86.98, 87.58, 87.74],
        "x_label": "2000년 이후 경과년",
        "y_label": "15세 이상 문해율(%)",
        "source": "Our World in Data / UNESCO: Literacy rate",
        "source_url": "https://ourworldindata.org/grapher/cross-country-literacy-rates",
        "deep_question": "세계 평균 문해율이 높아져도, 교육 기회의 격차는 어디에 남아 있을까?",
        "factfulness_lens": "일반화 본능 점검",
    },
    "세계 DTP3 예방접종률": {
        "years": [2000, 2005, 2010, 2015, 2020, 2023, 2024],
        "x": [0, 5, 10, 15, 20, 23, 24],
        "y": [72.0, 77.0, 83.0, 85.0, 83.0, 84.0, 85.0],
        "x_label": "2000년 이후 경과년",
        "y_label": "DTP3 접종 완료 아동 비율(%)",
        "source": "Our World in Data / WHO and UNICEF: Share of children immunized with DTP3",
        "source_url": "https://ourworldindata.org/grapher/share-of-children-immunized-dtp3",
        "deep_question": "예방접종률이 높아졌다는 평균 뒤에 어떤 지역과 집단의 취약성이 가려질 수 있을까?",
        "factfulness_lens": "격차 본능 점검",
    },
}


DATASETS = {
    "디지털 접근과 도시 변화": {
        "table": pd.DataFrame({
            "연도": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "인터넷 이용률(%)": [6.720, 8.023, 10.478, 12.161, 14.037, 15.600, 17.300, 20.300, 22.800, 25.400, 28.500, 31.000, 33.500, 35.400, 37.500, 39.900, 42.900, 45.400, 48.600, 53.200, 59.300, 62.200, 64.400, 67.400],
            "전기 접근성(%)": [78.221, 78.715, 79.101, 79.965, 79.952, 80.701, 81.353, 81.971, 82.668, 82.866, 83.447, 84.506, 84.936, 85.711, 86.196, 86.924, 88.104, 88.932, 89.797, 90.108, 90.396, 91.334, 91.288, 91.599],
            "도시화율(%)": [46.836, 47.301, 47.827, 48.370, 48.873, 49.368, 49.880, 50.425, 50.889, 51.385, 51.778, 52.489, 52.951, 53.465, 53.947, 54.470, 54.944, 55.375, 55.814, 56.204, 56.503, 56.877, 57.130, 57.354],
            "성인 문해율(%)": [81.120, 81.370, 81.740, 82.220, 82.560, 82.570, 82.530, 83.120, 83.480, 83.800, 84.370, 84.710, 84.910, 85.210, 85.560, 85.960, 86.280, 86.470, 86.760, 86.790, 86.980, 87.170, 87.390, 87.580],
        }),
        "default_x": "연도",
        "default_y": "인터넷 이용률(%)",
        "source": "FRED/World Bank, Our World in Data, UNESCO",
        "source_url": "https://fred.stlouisfed.org/series/ITNETUSERP2WLD",
        "deep_question": "디지털 접근이 늘어날수록 정보 격차와 도시 생활의 기회는 어떻게 달라질까?",
        "factfulness_lens": "일반화 본능 점검",
    },
    "세계 보건과 삶의 질": {
        "table": pd.DataFrame({
            "연도": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "기대수명(년)": [67.650, 67.945, 68.231, 68.516, 68.771, 69.111, 69.469, 69.805, 70.004, 70.380, 70.683, 70.969, 71.265, 71.533, 71.776, 71.966, 72.186, 72.366, 72.644, 72.870, 72.183, 71.216, 72.969, 73.330, 73.482],
            "5세 미만 사망률(출생 100명당)": [7.668, 7.433, 7.131, 6.833, 6.584, 6.233, 5.956, 5.695, 5.469, 5.292, 5.059, 4.951, 4.671, 4.506, 4.392, 4.288, 4.172, 4.089, 3.996, 3.957, 3.916, 3.961, 3.998, 3.833, 3.737],
            "DTP3 접종률(%)": [72.0, 72.0, 72.0, 74.0, 76.0, 77.0, 78.0, 78.0, 81.0, 83.0, 83.0, 84.0, 84.0, 83.0, 85.0, 85.0, 86.0, 86.0, 86.0, 86.0, 83.0, 82.0, 85.0, 84.0, 85.0],
            "극빈곤 인구 비율(%)": [36.205, 35.254, 33.740, 32.155, 30.262, 28.330, 26.992, 25.285, 24.138, 23.060, 20.984, 19.102, 17.825, 15.504, 14.508, 13.422, 12.574, 11.793, 11.119, 10.763, 11.412, 11.327, 10.889, 10.618, 10.400],
        }),
        "default_x": "연도",
        "default_y": "기대수명(년)",
        "source": "FRED/World Bank, Our World in Data, UN IGME, WHO/UNICEF",
        "source_url": "https://ourworldindata.org/grapher/under-5-mortality-rate-sdgs",
        "deep_question": "세계 보건 지표가 좋아졌다는 사실과 아직 남은 취약성은 어떻게 함께 설명할 수 있을까?",
        "factfulness_lens": "부정 본능 점검",
    },
    "환경 에너지와 대기질": {
        "table": pd.DataFrame({
            "연도": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            "세계 재생에너지 비중(%)": [7.426, 7.183, 7.186, 6.975, 7.153, 7.192, 7.339, 7.383, 7.855, 8.161, 8.370, 8.549, 8.963, 9.367, 9.732, 10.000, 10.431, 10.793, 11.202, 11.651, 12.781],
            "한국 1인당 CO₂ 배출량(톤)": [9.408, 9.685, 10.065, 10.212, 10.302, 10.422, 10.488, 10.829, 11.015, 11.078, 12.187, 12.690, 12.660, 12.674, 12.453, 12.439, 12.435, 12.713, 12.977, 12.481, 11.524],
            "한국 PM2.5 평균 노출(μg/m³)": [25.438, 25.345, 25.197, 24.999, 24.756, 24.474, 23.984, 23.255, 22.504, 21.945, 21.791, 24.883, 25.577, 27.493, 28.376, 26.290, 26.670, 26.004, 25.934, 26.015, 25.944],
        }),
        "default_x": "연도",
        "default_y": "세계 재생에너지 비중(%)",
        "source": "Our World in Data: Renewable share of energy, CO₂ emissions per capita, PM2.5 exposure",
        "source_url": "https://ourworldindata.org/grapher/renewable-share-energy",
        "deep_question": "재생에너지 확대, 탄소배출, 대기질 지표는 같은 방향으로 움직인다고 말할 수 있을까?",
        "factfulness_lens": "직선 본능 점검",
    },
    "한국 인구와 노동": {
        "table": pd.DataFrame({
            "연도": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "한국 합계출산율(명)": [1.480, 1.309, 1.178, 1.191, 1.164, 1.085, 1.132, 1.259, 1.192, 1.149, 1.226, 1.244, 1.297, 1.187, 1.205, 1.239, 1.172, 1.052, 0.977, 0.918, 0.837, 0.808, 0.778, 0.721, 0.748],
            "한국 65세 이상 인구 비율(%)": [7.091, 7.445, 7.816, 8.201, 8.601, 9.000, 9.421, 9.857, 10.248, 10.613, 10.992, 11.350, 11.703, 12.097, 12.525, 12.955, 13.363, 13.874, 14.454, 15.063, 15.822, 16.660, 17.485, 18.335, 19.274],
            "한국 청년실업률(%)": [10.002, 9.359, 7.902, 9.424, 9.798, 9.321, 8.916, 7.853, 8.650, 9.221, 8.540, 8.298, 7.738, 8.943, 8.594, 9.899, 10.145, 9.781, 10.120, 9.827, 10.143, 8.055, 6.638, 5.405, 6.434],
        }),
        "default_x": "연도",
        "default_y": "한국 65세 이상 인구 비율(%)",
        "source": "World Bank WDI / FRED: Korea fertility, aging population, youth unemployment",
        "source_url": "https://fred.stlouisfed.org/series/SPPOP65UPTOZSKOR",
        "deep_question": "인구 구조 변화와 청년 노동 지표를 함께 보면 어떤 사회정책 질문이 생길까?",
        "factfulness_lens": "일반화 본능 점검",
    },
}


DATASET_GROUPS = {
    "정보통신·데이터 분야": ["디지털 접근과 도시 변화"],
    "보건의료·국제개발 분야": ["세계 보건과 삶의 질"],
    "환경·에너지 분야": ["환경 에너지와 대기질"],
    "사회정책·인구 분야": ["한국 인구와 노동"],
}


FUNCTION_OPTIONS = ["일차함수", "이차함수", "유리함수", "무리함수"]

CLASS_OPTIONS = ["1", "2", "5", "6"]
GALLERY_URLS = {
    "1": "https://padlet.com/ps0andd/g_1",
    "2": "https://padlet.com/ps0andd/g_2",
    "5": "https://padlet.com/ps0andd/g_5",
    "6": "https://padlet.com/ps0andd/g_6",
}
CANVA_AI_URL = "https://www.canva.com/ai"

APP_OUTPUT_TYPE = "정보형 앱"
TARGET_USERS = ["고등학생", "중학생", "학부모", "지역사회 시민", "정책 결정자", "일반 대중"]
APP_FEATURES = [
    "핵심 예측값 카드",
    "산점도와 추세선 설명",
    "잔차와 손실값 안내",
    "보간·외삽 주의 문구",
    "깊은 질문 제시",
    "실천 메시지 강조",
    "사용자 선택 버튼",
    "퀴즈 또는 미션",
]

FACTFULNESS_LENSES = {
    "부정 본능 점검": "나쁜 점만 보거나 좋은 변화만 과장하지 않고, 좋아진 점과 남은 문제를 함께 봅니다.",
    "직선 본능 점검": "최근 흐름이 앞으로도 같은 속도로 계속된다고 단정하지 않고, 증가 속도와 꺾이는 지점을 살핍니다.",
    "일반화 본능 점검": "평균이나 전체 흐름이 모든 사람과 지역의 상황을 대표한다고 단정하지 않습니다.",
    "격차 본능 점검": "세상을 둘로만 나누지 않고, 중간 단계와 다양한 차이를 함께 봅니다.",
}


def calculate_function(function_type, x_values, params):
    """선택한 함수 종류와 매개변수에 따라 함수값을 계산합니다."""
    x_arr = np.asarray(x_values, dtype=float)

    if function_type == "일차함수":
        y_arr = params["a"] * x_arr + params["b"]
        valid_mask = np.ones_like(x_arr, dtype=bool)
        return y_arr, valid_mask

    if function_type == "이차함수":
        y_arr = params["a"] * (x_arr - params["p"]) ** 2 + params["q"]
        valid_mask = np.ones_like(x_arr, dtype=bool)
        return y_arr, valid_mask

    if function_type == "유리함수":
        denominator = x_arr - params["p"]
        valid_mask = np.abs(denominator) > 1e-8
        y_arr = np.full_like(x_arr, np.nan, dtype=float)
        y_arr[valid_mask] = params["a"] / denominator[valid_mask] + params["q"]
        return y_arr, valid_mask

    if function_type == "무리함수":
        radicand = x_arr - params["p"]
        valid_mask = radicand >= 0
        y_arr = np.full_like(x_arr, np.nan, dtype=float)
        y_arr[valid_mask] = params["a"] * np.sqrt(radicand[valid_mask]) + params["q"]
        return y_arr, valid_mask

    raise ValueError("지원하지 않는 함수 종류입니다.")


def calculate_loss(actual_y, predicted_y, valid_mask):
    """정의되는 데이터만 사용하여 평균제곱오차 손실값을 계산합니다."""
    actual_arr = np.asarray(actual_y, dtype=float)
    pred_arr = np.asarray(predicted_y, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred_arr)

    if np.sum(mask) == 0:
        return None, 0

    loss = np.mean((actual_arr[mask] - pred_arr[mask]) ** 2)
    return float(loss), int(np.sum(mask))


def calculate_error_metrics(actual_y, predicted_y, valid_mask):
    """잔차, MSE, MAE, RMSE, R²를 계산합니다."""
    actual_arr = np.asarray(actual_y, dtype=float)
    pred_arr = np.asarray(predicted_y, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred_arr)

    if np.sum(mask) == 0:
        return {
            "mse": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "valid_count": 0,
            "mask": mask,
            "residuals": np.full_like(actual_arr, np.nan, dtype=float),
        }

    residuals = np.full_like(actual_arr, np.nan, dtype=float)
    residuals[mask] = actual_arr[mask] - pred_arr[mask]
    sse = float(np.sum(residuals[mask] ** 2))
    mse = float(np.mean(residuals[mask] ** 2))
    mae = float(np.mean(np.abs(residuals[mask])))
    rmse = float(np.sqrt(mse))
    sst = float(np.sum((actual_arr[mask] - np.mean(actual_arr[mask])) ** 2))
    r2 = None if sst == 0 else float(1 - sse / sst)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "valid_count": int(np.sum(mask)),
        "mask": mask,
        "residuals": residuals,
    }


def make_residual_table(x_data, actual_y, predicted_y, valid_mask):
    """실제값, 예측값, 잔차를 표로 정리합니다."""
    x_arr = np.asarray(x_data, dtype=float)
    actual_arr = np.asarray(actual_y, dtype=float)
    pred_arr = np.asarray(predicted_y, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred_arr)

    rows = []
    for x_value, actual_value, pred_value, is_valid in zip(x_arr, actual_arr, pred_arr, mask):
        if is_valid:
            residual = actual_value - pred_value
            rows.append(
                {
                    "x": x_value,
                    "실제 y": actual_value,
                    "함수값 f(x)": pred_value,
                    "잔차 y-f(x)": residual,
                    "잔차제곱": residual**2,
                }
            )
        else:
            rows.append(
                {
                    "x": x_value,
                    "실제 y": actual_value,
                    "함수값 f(x)": np.nan,
                    "잔차 y-f(x)": np.nan,
                    "잔차제곱": np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_residual_plot(x_data, residuals, valid_mask, x_label):
    """잔차가 무작위로 흩어지는지 확인하는 잔차 그래프를 그립니다."""
    x_arr = np.asarray(x_data, dtype=float)
    residual_arr = np.asarray(residuals, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(residual_arr)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhline(0, color="#555555", linewidth=1.2, linestyle="--")
    ax.scatter(x_arr[mask], residual_arr[mask], color="#7b1fa2", s=80)
    ax.set_title("잔차 그래프")
    ax.set_xlabel(x_label)
    ax.set_ylabel("잔차 = 실제 y - 함수값")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def prediction_range_label(new_x, x_data):
    """새 입력값이 보간인지 외삽인지 구분합니다."""
    x_arr = np.asarray(x_data, dtype=float)
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    if x_min <= float(new_x) <= x_max:
        return "보간", f"x={float(new_x):g}은 관찰된 x 범위 [{x_min:g}, {x_max:g}] 안에 있습니다."
    return "외삽", f"x={float(new_x):g}은 관찰된 x 범위 [{x_min:g}, {x_max:g}] 밖에 있습니다."


def model_complexity_text(function_type):
    """함수 종류별 매개변수 수와 해석 초점을 안내합니다."""
    details = {
        "일차함수": ("매개변수 2개", "가장 단순한 모델입니다. 일정한 증가·감소를 설명하기 쉽지만 굽은 흐름은 놓칠 수 있습니다."),
        "이차함수": ("매개변수 3개", "꼭짓점과 굽은 방향을 설명할 수 있습니다. 증가하다 감소하는 흐름을 표현하기 좋습니다."),
        "유리함수": ("매개변수 3개", "점근선이 있어 급격한 변화와 포화되는 흐름을 설명할 수 있지만 점근선 근처 예측은 조심해야 합니다."),
        "무리함수": ("매개변수 3개", "정의역의 시작점과 점점 완만해지는 흐름을 설명할 수 있습니다. 정의되지 않는 x가 생길 수 있습니다."),
    }
    return details[function_type]


def build_function_text(function_type, params):
    """포스터 카드와 결과 설명에 사용할 함수식을 만듭니다."""
    if function_type == "일차함수":
        return f"f(x) = {params['a']:.2f}x + {params['b']:.2f}"
    if function_type == "이차함수":
        return f"f(x) = {params['a']:.2f}(x - {params['p']:.2f})² + {params['q']:.2f}"
    if function_type == "유리함수":
        return f"f(x) = {params['a']:.2f} / (x - {params['p']:.2f}) + {params['q']:.2f}"
    return f"f(x) = {params['a']:.2f}√(x - {params['p']:.2f}) + {params['q']:.2f}"


def predict_value(function_type, x_value, params):
    """새 입력값 x에 대한 예측값을 계산합니다."""
    y_arr, valid_mask = calculate_function(function_type, [x_value], params)
    if not valid_mask[0] or not np.isfinite(y_arr[0]):
        return None
    return float(y_arr[0])


def make_plot(
    x_data,
    y_data,
    function_type,
    params,
    new_x,
    predicted_y,
    x_label,
    y_label,
):
    """산점도, 함수 그래프, 예측점을 한 그래프에 표시합니다."""
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)

    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    x_span = max(x_max - x_min, 1.0)
    plot_min = min(x_min, float(new_x)) - x_span * 0.15
    plot_max = max(x_max, float(new_x)) + x_span * 0.15
    x_line = np.linspace(plot_min, plot_max, 600)

    y_line, valid_line = calculate_function(function_type, x_line, params)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x_arr, y_arr, color="#1f77b4", s=90, label="실제 데이터", zorder=3)
    add_trend_ellipse(ax, x_arr, y_arr)

    if function_type == "유리함수":
        valid_indices = np.where(valid_line & np.isfinite(y_line))[0]
        if len(valid_indices) > 0:
            split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
            segments = np.split(valid_indices, split_points)
            for segment_index, segment in enumerate(segments):
                ax.plot(
                    x_line[segment],
                    y_line[segment],
                    color="#d62728",
                    linewidth=2.5,
                    label="선택한 함수 그래프" if segment_index == 0 else None,
                )
        ax.axvline(params["p"], color="#777777", linestyle="--", linewidth=1.2, label="세로 점근선")
        ax.axhline(params["q"], color="#999999", linestyle=":", linewidth=1.2, label="가로 점근선")
    else:
        ax.plot(
            x_line[valid_line],
            y_line[valid_line],
            color="#d62728",
            linewidth=2.5,
            label="선택한 함수 그래프",
        )

    if predicted_y is not None:
        ax.scatter(
            [new_x],
            [predicted_y],
            color="#2ca02c",
            marker="*",
            s=220,
            label="새 입력값 예측점",
            zorder=4,
        )
        ax.annotate(
            f"예측점\n({new_x:g}, {predicted_y:.2f})",
            xy=(new_x, predicted_y),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=10,
            color="#2ca02c",
        )

    ax.set_title("데이터 산점도와 선택한 함수 그래프")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def show_function_description(function_type):
    """함수 그래프 특징을 학생 활동 문장으로 안내합니다."""
    descriptions = {
        "일차함수": [
            "a는 기울기를 나타낸다.",
            "a가 양수이면 증가하고, 음수이면 감소한다.",
            "b는 y절편이며 그래프를 위아래로 이동시킨다.",
            "점들이 거의 일정한 비율로 증가하거나 감소할 때 일차함수를 추세선으로 생각할 수 있다.",
        ],
        "이차함수": [
            "a가 양수이면 아래로 볼록, 음수이면 위로 볼록이다.",
            "|a|가 커질수록 그래프의 폭이 좁아진다.",
            "꼭짓점은 (p,q)이다.",
            "데이터가 증가하다가 감소하거나, 감소하다가 증가할 때 이차함수를 추세선으로 생각할 수 있다.",
        ],
        "유리함수": [
            "x=p는 세로 점근선이다.",
            "y=q는 가로 점근선이다.",
            "x가 p에 가까워질수록 그래프가 급격히 변한다.",
            "처음에는 빠르게 변하다가 점점 일정한 값에 가까워지는 데이터에 적절할 수 있다.",
        ],
        "무리함수": [
            "x-p가 0 이상일 때 그래프가 그려진다.",
            "시작점은 (p,q)이다.",
            "처음에는 빠르게 변하고 이후 점점 완만해질 수 있다.",
            "시작점이 있고 점점 완만하게 증가하거나 감소하는 데이터에 적절할 수 있다.",
        ],
    }
    for item in descriptions[function_type]:
        st.markdown(f"- {item}")


def show_function_hints(function_type):
    """학생이 직접 그래프 특징을 찾을 수 있도록 단계형 힌트를 제공합니다."""
    hints = {
        "일차함수": [
            "a를 0보다 크게, 0보다 작게 바꾸어 보세요. 그래프가 어느 방향으로 기울어지나요?",
            "a의 절댓값을 크게 만들면 같은 x 변화에서 y가 얼마나 더 빠르게 변하나요?",
            "b만 바꾸면 그래프의 기울기는 그대로인가요, 아니면 위치만 바뀌나요?",
            "산점도의 점들이 거의 일정한 간격으로 오르거나 내려가면 이 함수가 어울릴 수 있습니다.",
        ],
        "이차함수": [
            "a의 부호를 바꾸어 보세요. 그래프가 위로 볼록인지 아래로 볼록인지 어떻게 달라지나요?",
            "p를 움직이면 그래프의 가장 높거나 낮은 점이 좌우로 이동하나요?",
            "q를 움직이면 꼭짓점의 높이가 어떻게 달라지나요?",
            "데이터가 증가하다가 감소하거나, 감소하다가 증가하면 꼭짓점이 있는 함수가 어울릴 수 있습니다.",
        ],
        "유리함수": [
            "p 근처에서 그래프가 갑자기 크게 변하는지 관찰해 보세요.",
            "q를 움직이면 그래프가 멀리서 가까워지는 높이가 바뀌나요?",
            "x=p 근처에는 그래프가 지나갈 수 없는 세로 기준선이 생깁니다.",
            "처음에는 빠르게 변하다가 점점 일정한 값에 가까워지는 흐름인지 확인해 보세요.",
        ],
        "무리함수": [
            "p보다 작은 x에서는 그래프가 그려지는지 확인해 보세요.",
            "p와 q를 움직이면 그래프가 시작하는 위치가 어떻게 달라지나요?",
            "a의 부호를 바꾸면 증가하는 모양과 감소하는 모양이 어떻게 달라지나요?",
            "시작점 이후 처음에는 빠르게 변하고 점점 완만해지는 흐름인지 확인해 보세요.",
        ],
    }
    for idx, hint in enumerate(hints[function_type], start=1):
        st.markdown(f"**힌트 {idx}.** {hint}")


def get_parameters(function_type, x_data, y_data):
    """사이드바에서 선택한 함수에 필요한 매개변수 슬라이더를 제공합니다."""
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    params = {}

    st.subheader("T 단계 조작 슬라이더")
    if function_type == "일차함수":
        params["a"] = st.slider("a: 기울기", -10.0, 10.0, 1.0, 0.1)
        params["b"] = st.slider("b: y절편", -100.0, 100.0, 0.0, 0.5)
    elif function_type == "이차함수":
        params["a"] = st.slider("a: 볼록한 방향과 폭", -10.0, 10.0, 1.0, 0.1)
        params["p"] = st.slider(
            "p: 꼭짓점의 x좌표",
            float(np.min(x_arr)),
            float(np.max(x_arr)),
            float(np.mean(x_arr)),
            0.1,
        )
        params["q"] = st.slider(
            "q: 꼭짓점의 y좌표",
            float(np.min(y_arr) - 50),
            float(np.max(y_arr) + 50),
            float(np.mean(y_arr)),
            0.5,
        )
    elif function_type == "유리함수":
        params["a"] = st.slider("a: 그래프의 휘어짐과 방향", -100.0, 100.0, 10.0, 0.5)
        params["p"] = st.slider(
            "p: 세로 점근선 x=p",
            float(np.min(x_arr) - 5),
            float(np.max(x_arr) + 5),
            float(np.min(x_arr) - 1),
            0.1,
        )
        params["q"] = st.slider(
            "q: 가로 점근선 y=q",
            float(np.min(y_arr) - 50),
            float(np.max(y_arr) + 50),
            float(np.mean(y_arr)),
            0.5,
        )
    else:
        params["a"] = st.slider("a: 증가·감소 방향과 변화 정도", -20.0, 20.0, 5.0, 0.1)
        params["p"] = st.slider(
            "p: 정의역의 시작과 관련된 값",
            float(np.min(x_arr) - 5),
            float(np.max(x_arr)),
            float(np.min(x_arr)),
            0.1,
        )
        params["q"] = st.slider(
            "q: 시작점의 y좌표",
            float(np.min(y_arr) - 50),
            float(np.max(y_arr) + 50),
            float(np.min(y_arr)),
            0.5,
        )

    return params


def format_optional_number(value):
    if value is None:
        return "정의되지 않음"
    return f"{value:.2f}"


def render_prediction_input_card(x_data, x_label, key="d8_new_x"):
    x_arr = np.asarray(x_data, dtype=float)
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    current_value = float(st.session_state.get(key, x_max))
    inside_range = x_min <= current_value <= x_max
    range_label = "관찰 범위 안" if inside_range else "관찰 범위 밖"
    range_color = "#2e7d32" if inside_range else "#ef6c00"
    st.markdown(
        f"""
        <div class="prediction-input-card">
            <div class="prediction-input-title">예측할 새 x값을 정하세요</div>
            <div class="prediction-input-help">
                선택한 독립변수는 <b>{html.escape(str(x_label))}</b>입니다.
                관찰된 x 범위는 <b>{x_min:g} ~ {x_max:g}</b>이며,
                현재 입력값은 <span style="font-weight:900;color:{range_color};">{range_label}</span>입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.number_input(
        "예측할 새 x값",
        key=key,
        step=0.5,
        format="%.2f",
        help="그래프에서 예측하고 싶은 독립변수 x값을 입력합니다.",
    )


def selected_factfulness_text():
    lens = st.session_state.get("d8_factfulness_lens", "직선 본능 점검")
    note = FACTFULNESS_LENSES.get(lens, "")
    return f"{lens}: {note}" if note else lens


def factfulness_app_message():
    return st.session_state.get("d8_factfulness_app_message", "").strip()


def dataset_group_names():
    return list(DATASET_GROUPS.keys())


def datasets_for_group(group_name):
    names = DATASET_GROUPS.get(group_name, [])
    return [name for name in names if name in DATASETS]


def group_for_dataset(dataset_name):
    for group_name, names in DATASET_GROUPS.items():
        if dataset_name in names:
            return group_name
    return dataset_group_names()[0]


def numeric_columns(dataset_info):
    table = dataset_info["table"]
    return [col for col in table.columns if pd.api.types.is_numeric_dtype(table[col])]


def ensure_xy_columns(dataset_name):
    dataset_info = DATASETS[dataset_name]
    columns = numeric_columns(dataset_info)
    if not columns:
        raise ValueError(f"{dataset_name}에는 선택할 수 있는 숫자 열이 없습니다.")

    default_x = dataset_info.get("default_x", columns[0])
    if default_x not in columns:
        default_x = columns[0]
    default_y = dataset_info.get("default_y", columns[1] if len(columns) > 1 else columns[0])
    if default_y not in columns:
        default_y = columns[1] if len(columns) > 1 else columns[0]
    if default_y == default_x and len(columns) > 1:
        default_y = next(col for col in columns if col != default_x)

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
    dataset_info = DATASETS[dataset_name]
    ensure_xy_columns(dataset_name)
    x_label = st.session_state["d8_x_col"]
    y_label = st.session_state["d8_y_col"]
    clean_table = dataset_info["table"][[x_label, y_label]].dropna()
    x_data = clean_table[x_label].to_numpy(dtype=float)
    y_data = clean_table[y_label].to_numpy(dtype=float)
    return x_data, y_data, x_label, y_label


def make_download_text(
    group_name,
    dataset_name,
    function_type,
    function_text,
    loss,
    new_x,
    predicted_y,
    observation_checks,
    observation_text,
    choice_reason,
    deep_question,
    limitation,
    action_message,
):
    """활동 결과를 txt 파일로 저장할 수 있도록 문자열로 정리합니다."""
    checks = ", ".join(observation_checks) if observation_checks else "선택 없음"
    return f"""공통수학Ⅱ 함수 단원 활동 결과

모둠명: {group_name}
데이터 주제: {dataset_name}

[FU 단계: 데이터 흐름 관찰하기]
선택한 흐름: {checks}
관찰 내용:
{observation_text}

[T/U 단계: 추세선 선택과 예측]
선택한 함수 종류: {function_type}
함수식: {function_text}
손실값: {format_optional_number(loss)}
새 입력값 x: {new_x:g}
예측값: {format_optional_number(predicted_y)}
선택 이유:
{choice_reason}

[RE 단계: 깊은 질문과 정리]
데이터 해석 렌즈:
{selected_factfulness_text()}
앱 점검 문구:
{factfulness_app_message() or "작성하지 않음"}

깊은 질문:
{deep_question}

예측 결과의 한계:
{limitation}
/per
실천 메시지:
{action_message}
"""


def build_canva_prompt(
    group_name,
    dataset_name,
    prediction_question,
    function_type,
    function_text,
    loss_text,
    prediction_text,
    output_type,
    target_user,
    selected_features,
    deep_question,
    limitation,
    action_message,
    ds_reflection,
    design_tone,
):
    """Canva AI에서 교육용 앱 시안을 만들기 위한 프롬프트를 구성합니다."""
    features_text = ", ".join(selected_features) if selected_features else "핵심 예측값 카드, 산점도와 추세선 설명, 실천 메시지 강조"
    return f"""다음 내용을 바탕으로 Canva AI에서 만들 교육용 {output_type} 시안을 제작해 주세요.

[수업 맥락]
- 과목: 고등학교 공통수학Ⅱ 함수 단원
- 수업 주제: 데이터의 흐름을 가장 잘 설명하는 함수는 무엇일까?
- 수업 방향: AI 도구 사용 자체가 아니라, 함수 그래프·잔차·손실값·예측 한계를 근거로 데이터의 관계를 해석하고 사회적 실천 메시지로 확장하는 활동
- 해석 점검: 데이터를 볼 때 본능적 단정과 과장을 줄이고, 근거·비교·범위·한계를 함께 확인합니다.

[프로젝트 정보]
- 모둠명: {group_name}
- 데이터 주제: {dataset_name}
- 예측 질문: {prediction_question}
- 대상 사용자: {target_user}
- 산출물 유형: {output_type}
- 원하는 분위기: {design_tone}

[수학적 분석 결과]
- 선택한 함수 종류: {function_type}
- 함수식: {function_text}
- 손실값(MSE): {loss_text}
- 예측 결과: {prediction_text}
- 깊은 질문: {deep_question}
- 예측의 한계: {limitation}
- AI·데이터 과학 관점의 해석: {ds_reflection}
- 데이터 해석 렌즈: {selected_factfulness_text()}
- 앱에 넣을 점검 문구: {factfulness_app_message() or "예측값을 단정하지 말고 데이터 범위와 비교 기준을 함께 확인하도록 안내해 주세요."}

[앱에 반드시 포함할 기능]
- {features_text}

[화면 구성 요청]
1. 첫 화면에는 데이터 주제와 예측 질문을 크게 보여 주세요.
2. 산점도와 선택한 함수 추세선을 설명하는 영역을 넣어 주세요.
3. 손실값, 잔차, 예측값을 학생 눈높이에 맞게 짧은 문장으로 설명해 주세요.
4. 보간과 외삽, 데이터 범위, 표본 수의 한계를 경고 문구로 넣어 주세요.
5. 깊은 질문과 실천 메시지를 마지막 화면 또는 강조 카드에 넣어 주세요.
6. 사용자가 결과를 그대로 믿지 않고 근거와 한계를 함께 보도록 안내해 주세요.

[실천 메시지]
{action_message}

디자인은 수업 발표용으로 깔끔하고 읽기 쉽게 만들고, 수학 용어는 정확하되 고등학생이 이해할 수 있는 문장으로 작성해 주세요.
"""


def render_math_summary():
    st.markdown(
        """
1. 산점도는 두 변수 사이의 관계를 점으로 나타낸 것이다.
2. 추세선은 데이터의 전체적인 흐름을 설명하는 함수의 그래프이다.
3. 일차함수는 일정하게 증가하거나 감소하는 흐름을 설명할 수 있다.
4. 이차함수는 증가하다가 감소하거나 감소하다가 증가하는 흐름을 설명할 수 있다.
5. 유리함수는 점근선을 가지며, 처음에는 급격히 변하다가 점점 일정한 값에 가까워지는 흐름을 설명할 수 있다.
6. 무리함수는 정의역의 시작점이 있으며, 이후 점점 완만하게 변하는 흐름을 설명할 수 있다.
7. 잔차는 실제값과 함수값의 차이이며, 잔차가 만드는 패턴은 모델이 놓친 구조를 보여 줄 수 있다.
8. 손실값은 실제값과 함수값의 차이를 수치화한 것이다.
9. MSE는 큰 오차를 더 크게 벌점으로 주고, RMSE는 y값과 같은 단위로 해석하기 쉽다.
10. 예측값은 함수의 그래프를 바탕으로 구한 값이므로, 데이터의 범위와 모델의 한계를 함께 고려해야 한다.
11. AI나 데이터 과학에서 중요한 것은 결과를 그대로 믿는 것이 아니라, 데이터의 근거와 오차의 이유를 해석하는 것이다.
12. 수학적 분석은 수치와 그래프에서 끝나지 않고, 사람과 사회에 도움이 되는 책임 있는 실천으로 확장될 수 있다.
13. 데이터를 해석할 때는 과장된 직감보다 근거, 비교 기준, 예측 범위, 빠진 변수를 함께 확인해야 한다.
"""
    )


def apply_local_style():
    """data1~data6과 비슷한 탭형 수업 앱 스타일을 적용합니다."""
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: #f7fafd;
        }
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
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {
            border-radius: 0.8rem;
            border: 1px solid #cfe0f2;
            background: #ffffff;
        }
        [data-testid="stNumberInput"] input {
            font-size: 1.2rem;
            font-weight: 800;
            color: #1565c0;
            text-align: center;
            border: 2px solid #1976d2;
            box-shadow: 0 0 0 4px rgba(25, 118, 210, 0.10);
            background: #f8fbff;
        }
        .soft-card {
            background: #ffffff;
            border: 1px solid #dbe7f3;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 4px 14px rgba(33, 150, 243, 0.07);
            margin: 8px 0 14px 0;
        }
        .step-badge {
            display: inline-block;
            background: #e3f2fd;
            color: #1565c0;
            border: 1px solid #90caf9;
            border-radius: 999px;
            padding: 4px 11px;
            font-weight: 800;
            font-size: 0.86rem;
            margin-bottom: 8px;
        }
        .prediction-panel {
            background: linear-gradient(135deg, #ffffff 0%, #eef7ff 100%);
            border: 2px solid #90caf9;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 18px rgba(21, 101, 192, 0.12);
            margin-bottom: 14px;
        }
        .prediction-title {
            color: #0d47a1;
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 4px;
        }
        .prediction-help {
            color: #455a64;
            font-size: 0.92rem;
            line-height: 1.55;
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
        .poster-card {
            border: 2px solid #1F4E79;
            border-radius: 14px;
            background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 100%);
            padding: 1.4rem;
            margin-top: 1rem;
            box-shadow: 0 4px 16px rgba(31, 78, 121, 0.12);
        }
        .poster-title {
            color: #1F4E79;
            font-size: 1.55rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .poster-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .poster-item {
            background: #FFFFFF;
            border: 1px solid #D9E8F5;
            border-radius: 8px;
            padding: 0.75rem;
        }
        .poster-label {
            color: #456;
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .poster-value {
            color: #111;
            font-size: 1.02rem;
            line-height: 1.45;
            white-space: pre-wrap;
        }
        @media (max-width: 800px) {
            .poster-grid {
                grid-template-columns: 1fr;
            }
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


def page_banner(title, description, question):
    question_html = (
        f"""
            <div style="
                margin-top:12px;
                background:rgba(255,255,255,0.70);
                border-radius:12px;
                padding:10px 12px;
                color:#1f2937;
                border:1px solid rgba(255,255,255,0.9);
            "><b>핵심 질문</b><br>{question}</div>
        """
        if question
        else ""
    )
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
            <div style="font-size:0.9rem; font-weight:700; color:#5e35b1; margin-bottom:8px;">F.U.T.U.R.E. 프로젝트 공개수업 1차시</div>
            <div style="font-size:1.9rem; font-weight:800; color:#1f2937; margin-bottom:8px;">{title}</div>
            <div style="font-size:1rem; line-height:1.7; color:#37474f;">{description}</div>
            {question_html}
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
                <b>탐구 질문</b><br>{question}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_preview(dataset_name, x_data, y_data, x_label, y_label):
    dataset_info = DATASETS.get(dataset_name, {})
    df = dataset_info.get("table", pd.DataFrame({x_label: x_data, y_label: y_data}))
    data_col, scatter_col = st.columns([1, 1.4])
    with data_col:
        st.markdown(pretty_title("선택한 데이터 표", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"현재 분석 변수: x = {x_label}, y = {y_label}")
        if dataset_info.get("source"):
            st.caption(f"자료 출처: {dataset_info['source']}")
        if dataset_info.get("source_url"):
            st.markdown(f"[원자료 또는 참고 자료 열기]({dataset_info['source_url']})")
    with scatter_col:
        st.markdown(pretty_title("데이터 산점도", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        fig_scatter, ax_scatter = plt.subplots(figsize=(6.5, 4))
        ax_scatter.scatter(x_data, y_data, color="#1f77b4", s=80, label="실제 데이터", zorder=3)
        add_trend_ellipse(ax_scatter, x_data, y_data)
        ax_scatter.set_title(dataset_name)
        ax_scatter.set_xlabel(x_label)
        ax_scatter.set_ylabel(y_label)
        ax_scatter.grid(True, alpha=0.25)
        ax_scatter.legend(loc="best")
        fig_scatter.tight_layout()
        st.pyplot(fig_scatter)


def render_poster_card(
    group_name,
    dataset_name,
    prediction_question,
    function_type,
    function_text,
    loss_text,
    prediction_text,
    deep_question,
    limitation,
    action_message,
):
    safe_group_name = html.escape(group_name)
    safe_dataset_name = html.escape(dataset_name)
    safe_prediction_question = html.escape(prediction_question if prediction_question else "아직 작성하지 않았습니다.")
    safe_function_type = html.escape(function_type)
    safe_function_text = html.escape(function_text)
    safe_loss_text = html.escape(loss_text)
    safe_prediction_text = html.escape(prediction_text)
    safe_deep_question = html.escape(deep_question if deep_question else "아직 작성하지 않았습니다.")
    safe_limitation = html.escape(limitation if limitation else "아직 작성하지 않았습니다.")
    safe_action_message = html.escape(action_message if action_message else "아직 작성하지 않았습니다.")

    st.markdown(
        f"""
        <div class="poster-card">
            <div class="poster-title">{safe_group_name}의 추세선 탐구 카드</div>
            <div><b>데이터의 흐름을 함수의 그래프로 설명하고, 예측의 의미와 한계를 정리합니다.</b></div>
            <div class="poster-grid">
                <div class="poster-item">
                    <div class="poster-label">모둠명</div>
                    <div class="poster-value">{safe_group_name}</div>
                </div>
                <div class="poster-item">
                <div class="poster-label">데이터 주제</div>
                <div class="poster-value">{safe_dataset_name}</div>
            </div>
            <div class="poster-item" style="grid-column: 1 / -1;">
                <div class="poster-label">우리 모둠의 예측 질문</div>
                <div class="poster-value">{safe_prediction_question}</div>
            </div>
            <div class="poster-item">
                <div class="poster-label">선택한 함수 종류</div>
                <div class="poster-value">{safe_function_type}</div>
                </div>
                <div class="poster-item">
                    <div class="poster-label">함수식</div>
                    <div class="poster-value">{safe_function_text}</div>
                </div>
                <div class="poster-item">
                    <div class="poster-label">손실값</div>
                    <div class="poster-value">{safe_loss_text}</div>
                </div>
                <div class="poster-item">
                    <div class="poster-label">예측값</div>
                    <div class="poster-value">{safe_prediction_text}</div>
                </div>
                <div class="poster-item">
                    <div class="poster-label">깊은 질문</div>
                    <div class="poster-value">{safe_deep_question}</div>
                </div>
                <div class="poster-item">
                    <div class="poster-label">예측의 한계</div>
                    <div class="poster-value">{safe_limitation}</div>
                </div>
                <div class="poster-item" style="grid-column: 1 / -1;">
                    <div class="poster-label">실천 메시지</div>
                    <div class="poster-value">{safe_action_message}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_link_button(url, label, gradient):
    st.markdown(
        f"""
        <a href="{url}" target="_blank"
           style="display:block;padding:11px;background:{gradient};color:white;text-decoration:none;
           border-radius:8px;font-weight:bold;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,0.1);
           margin-top:8px;">
           {label}
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_canva_gallery_links(class_key):
    gallery_url = GALLERY_URLS.get(str(class_key))
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


def run():
    st.set_page_config(
        page_title="함수의 그래프로 만드는 추세선과 예측",
        layout="wide",
    )
    apply_local_style()

    page_banner(
        "질문으로 깨우고 함수로 예측하는 데이터 탐구",
        "공통수학Ⅱ 함수 그래프를 이용해 실생활 데이터의 흐름을 구조화하고, 오차와 예측의 한계를 근거로 해석한 뒤 사람과 사회에 도움이 되는 실천 메시지로 확장합니다.",
        "",
    )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)

    dataset_names = list(DATASETS.keys())
    group_names = dataset_group_names()
    st.session_state.setdefault("d8_group", "우리 모둠")
    st.session_state.setdefault("d8_class", CLASS_OPTIONS[0])
    if st.session_state["d8_class"] not in CLASS_OPTIONS:
        st.session_state["d8_class"] = CLASS_OPTIONS[0]
    st.session_state.setdefault("d8_student_id", "")
    st.session_state.setdefault("d8_dataset", dataset_names[0])
    if st.session_state["d8_dataset"] not in DATASETS:
        st.session_state["d8_dataset"] = dataset_names[0]
    st.session_state.setdefault("d8_dataset_group", group_for_dataset(st.session_state["d8_dataset"]))
    if st.session_state["d8_dataset_group"] not in group_names:
        st.session_state["d8_dataset_group"] = group_for_dataset(st.session_state["d8_dataset"])
    current_group_datasets = datasets_for_group(st.session_state["d8_dataset_group"])
    if not current_group_datasets:
        st.session_state["d8_dataset_group"] = group_names[0]
        current_group_datasets = datasets_for_group(st.session_state["d8_dataset_group"])
    if st.session_state["d8_dataset"] not in current_group_datasets:
        st.session_state["d8_dataset"] = current_group_datasets[0]
    st.session_state.setdefault("d8_function", FUNCTION_OPTIONS[0])
    if st.session_state["d8_function"] not in FUNCTION_OPTIONS:
        st.session_state["d8_function"] = FUNCTION_OPTIONS[0]
    ensure_xy_columns(st.session_state["d8_dataset"])
    x_data, y_data, x_label, y_label = selected_xy_data(st.session_state["d8_dataset"])
    default_new_x = float(
        max(x_data) + (x_data[1] - x_data[0] if len(x_data) > 1 else 1)
    )
    st.session_state.setdefault("d8_new_x", default_new_x)
    st.session_state.setdefault(
        "d8_prediction_question",
        "선택한 데이터에서 x값이 달라질 때 y값은 어떻게 변할까?",
    )

    group_name = st.session_state["d8_group"]
    dataset_name = st.session_state["d8_dataset"]
    function_type = st.session_state["d8_function"]
    new_x = float(st.session_state["d8_new_x"])
    x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)

    flow_options = [
        "대체로 증가한다.",
        "대체로 감소한다.",
        "처음에는 증가하다가 나중에 감소한다.",
        "처음에는 빠르게 변하다가 점점 완만해진다.",
        "직선보다 곡선에 가깝다.",
        "일부 점은 전체 흐름에서 벗어나 보인다.",
    ]

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
            "실생활 데이터가 어떤 사람과 사회의 문제를 담고 있는지 발견하고, 독립변수와 종속변수를 정해 예측 질문을 만듭니다.",
            "실생활 데이터는 어떤 문제를 보여줄까?",
            "#e3f2fd",
            "#bbdefb",
        )

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#1565c0; "
                "margin-bottom:6px;'>1️⃣ Quick, Draw!로 예측과 손실 다시 보기</div>",
                unsafe_allow_html=True,
            )
            quick_left, quick_right = st.columns([1.0, 1.0])
            with quick_left:
                st.markdown("Quick, Draw!를 직접 실행해 보고, AI가 그림을 보고 어떤 예측을 하는지 확인합니다.")
                st.link_button("Quick, Draw! 열기", "https://quickdraw.withgoogle.com/", use_container_width=True)
                st.info("제시어 1~2개를 빠르게 그린 뒤, AI가 맞힌 경우나 헷갈린 경우 중 하나를 골라 기록합니다.")
                st.text_input("실제 제시어", key="d8_review_actual", placeholder="예: 자전거")
                st.text_input("AI가 예측한 말", key="d8_review_predicted", placeholder="예: 안경, 자동차, 시계 등")
                st.text_area(
                    "AI는 어떤 특징을 보고 그렇게 예측했을까요?",
                    key="d8_fu_video_q1",
                    height=82,
                    placeholder="예: 동그란 바퀴 모양, 긴 선, 반복되는 무늬, 전체 윤곽 등",
                )
            with quick_right:
                st.markdown("**실제값, 예측값, 편차 확인**")
                st.markdown(
                    """
- **실제값**: 실제로 나타난 결과 또는 정답입니다.
- **예측값**: AI나 함수가 미리 예상한 값입니다.
- **편차**: 실제값과 예측값 사이의 차이입니다.
"""
                )
                st.latex(r"\text{편차}=\text{실제값}-\text{예측값}")
                st.latex(r"\text{손실}=(\text{편차})^2")
                answer_cols = st.columns(2)
                with answer_cols[0]:
                    st.info("실제값 = ______")
                    if st.button("실제값 답 확인", key="d8_actual_answer_btn", use_container_width=True):
                        st.session_state["d8_show_actual_answer"] = True
                    if st.session_state.get("d8_show_actual_answer", False):
                        st.success("실제값 = 제시어")
                with answer_cols[1]:
                    st.info("예측값 = ______")
                    if st.button("예측값 답 확인", key="d8_pred_answer_btn", use_container_width=True):
                        st.session_state["d8_show_pred_answer"] = True
                    if st.session_state.get("d8_show_pred_answer", False):
                        st.success("예측값 = AI가 추측한 말")
                st.success("AI 학습은 손실이 줄어드는 방향으로 예측 기준을 고쳐 가는 과정입니다.")

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#2e7d32; "
                "margin-bottom:6px;'>2️⃣ 데이터 분석 방향 정하기</div>",
                unsafe_allow_html=True,
            )
            group_name = st.text_input("모둠명", key="d8_group", placeholder="예: 1모둠")
            topic_col, dataset_col = st.columns([0.8, 1.2])
            with topic_col:
                selected_group = st.selectbox("공통 주제 선택", group_names, key="d8_dataset_group")
            field_dataset_names = datasets_for_group(selected_group)
            if st.session_state.get("d8_dataset") not in field_dataset_names:
                st.session_state["d8_dataset"] = field_dataset_names[0]
            with dataset_col:
                dataset_name = st.selectbox("데이터 주제 선택", field_dataset_names, key="d8_dataset")
            chosen_info = DATASETS[dataset_name]
            numeric_options = ensure_xy_columns(dataset_name)
            xy_left, xy_right = st.columns(2)
            with xy_left:
                st.selectbox("독립변수 x 선택", numeric_options, key="d8_x_col")
            y_options = numeric_options if len(numeric_options) == 1 else [
                col for col in numeric_options if col != st.session_state.get("d8_x_col")
            ]
            if st.session_state.get("d8_y_col") not in y_options:
                st.session_state["d8_y_col"] = y_options[0]
            with xy_right:
                st.selectbox("종속변수 y 선택", y_options, key="d8_y_col")
            if st.session_state["d8_x_col"] == st.session_state["d8_y_col"]:
                st.warning("x와 y가 같으면 관계를 탐구하기 어렵습니다. 가능하면 서로 다른 열을 선택하세요.")
            x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
            xy_context = f"{dataset_name}|{x_label}|{y_label}"
            if st.session_state.get("d8_new_x_context") != xy_context:
                st.session_state["d8_new_x"] = float(
                    max(x_data) + (x_data[1] - x_data[0] if len(x_data) > 1 else 1)
                )
                st.session_state["d8_new_x_context"] = xy_context
            st.text_area(
                "우리 모둠의 예측 질문",
                key="d8_prediction_question",
                height=90,
                placeholder="예: 이 데이터의 흐름이 계속된다면 사람과 사회에 어떤 문제가 생길까?",
            )
            if st.session_state.get("d8_lens_dataset") != dataset_name:
                st.session_state["d8_factfulness_lens"] = chosen_info.get("factfulness_lens", "직선 본능 점검")
                st.session_state["d8_lens_dataset"] = dataset_name
            st.success(chosen_info.get("deep_question", "데이터 흐름이 사람과 사회에 주는 의미를 질문으로 바꾸어 봅시다."))
            st.caption(f"자료 출처: {chosen_info.get('source', '출처 정보 없음')}")
            if chosen_info.get("source_url"):
                st.markdown(f"[출처 확인하기]({chosen_info['source_url']})")
            st.markdown("**데이터 해석 렌즈로 생각 되돌아보기**")
            st.selectbox(
                "우리 모둠이 가장 조심해야 할 생각 습관",
                list(FACTFULNESS_LENSES.keys()),
                key="d8_factfulness_lens",
                help="데이터를 볼 때 생기기 쉬운 단정과 과장을 줄이기 위한 관점입니다.",
            )
            st.caption(FACTFULNESS_LENSES[st.session_state["d8_factfulness_lens"]])
            st.text_area(
                "이 관점으로 보면 처음 생각을 어떻게 고쳐야 할까요?",
                key="d8_factfulness_question",
                height=82,
                placeholder="예: 계속 증가한다고 단정하기보다 어느 구간에서 증가 속도가 달라지는지 확인해야 한다.",
            )

        st.caption("다음 [T] 단계에서는 선택한 데이터를 표와 산점도로 구조화하고 흐름을 관찰합니다.")

    with tabs[1]:
        dataset_name = st.session_state["d8_dataset"]
        selected_data = DATASETS[dataset_name]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        stage_intro(
            "수학의 언어: 데이터를 어떻게 수학적으로 표현할까?",
            "선택한 데이터를 표와 산점도로 구조화하고, 함수 그래프와 매개변수로 데이터의 관계를 표현합니다.",
            "데이터의 관계를 어떻게 수학적으로 표현할까?",
            "#fff8e1",
            "#ffecb3",
        )

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:800; color:#6a1b9a; "
                "margin-bottom:6px;'>1️⃣ 데이터 구조와 흐름 관찰하기</div>",
                unsafe_allow_html=True,
            )
            corr_value = float(np.corrcoef(x_data, y_data)[0, 1]) if len(x_data) > 1 else np.nan
            structure_cols = st.columns(2)
            structure_cols[0].metric("자료 개수", f"{len(x_data)}개")
            structure_cols[1].metric("상관계수 r", "계산 불가" if np.isnan(corr_value) else f"{corr_value:.3f}")
            range_cols = st.columns(2)
            range_cols[0].metric("x 범위", f"{np.min(x_data):g} ~ {np.max(x_data):g}")
            range_cols[1].metric("y 범위", f"{np.min(y_data):g} ~ {np.max(y_data):g}")
            st.caption(
                "상관계수 r은 두 변수가 직선에 가깝게 함께 변하는 정도를 나타냅니다. "
                "단, r이 작아도 이차함수처럼 곡선 관계가 있을 수 있으므로 산점도 모양을 함께 보아야 합니다."
            )
            for idx, option in enumerate(flow_options):
                st.checkbox(option, key=f"flow_{idx}")
            st.text_area(
                "우리 모둠이 관찰한 데이터의 흐름을 설명해 봅시다.",
                height=110,
                placeholder="예: x가 커질수록 y가 대체로 증가하지만, 증가하는 정도가 점점 작아지는 것 같다.",
                key="observation_text",
            )
            render_data_preview(dataset_name, x_data, y_data, x_label, y_label)
        with st.expander("T 단계 수학 보충: 산점도와 상관을 어떻게 읽을까?", expanded=False):
            st.markdown(
                """
- **독립변수 x**는 예측에 사용하는 입력값이고, **종속변수 y**는 x에 따라 달라진다고 보고 해석하는 값입니다.
- **산점도**는 두 변수의 관계를 점의 위치로 나타낸 그래프입니다. 점들이 직선 모양인지, 굽은 모양인지, 한쪽으로 몰리는지 먼저 봅니다.
- **양의 상관**은 x가 커질수록 y도 대체로 커지는 관계이고, **음의 상관**은 x가 커질수록 y가 대체로 작아지는 관계입니다.
- **상관계수 r**은 직선 관계의 강도를 나타냅니다. r이 1에 가까우면 강한 양의 직선 관계, -1에 가까우면 강한 음의 직선 관계입니다.
- 하지만 r은 주로 **직선 관계**를 보는 수치입니다. 오르내림이 섞인 자료는 r만 보면 중요한 변화가 가려질 수 있습니다.
- 산점도에서 한두 점이 전체 흐름에서 멀리 떨어져 있으면 **이상치**일 수 있습니다. 이상치는 손실값과 추세선 선택에 큰 영향을 줄 수 있습니다.
"""
            )

        st.markdown(pretty_title("2. 함수 선택과 그래프 조작", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        t_left, t_right = st.columns([0.85, 1.15])
        with t_left:
            st.markdown("**왼쪽 활동: 함수 종류와 매개변수 조작**")
            function_type = st.selectbox("함수 종류 선택", FUNCTION_OPTIONS, key="d8_function")
            params = get_parameters(function_type, x_data, y_data)
            st.session_state["d8_params"] = params
            st.session_state["d8_function_type"] = function_type

        new_x = float(st.session_state["d8_new_x"])
        predicted_y_preview = predict_value(function_type, new_x, params)
        with t_right:
            st.markdown("**오른쪽 확인: 조작 결과 그래프**")
            st.pyplot(
                make_plot(
                    x_data,
                    y_data,
                    function_type,
                    params,
                    new_x,
                    predicted_y_preview,
                    x_label,
                    y_label,
                ),
                use_container_width=True,
            )
            st.caption("왼쪽 슬라이더를 움직이며 빨간 함수 그래프가 파란 산점도의 전체 흐름을 따라가는지 판단합니다.")

        st.markdown(pretty_title("3. 그래프 특징으로 판단하기", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        feature_left, feature_right = st.columns([0.85, 1.15])
        with feature_left:
            st.markdown("**왼쪽 활동: 그래프 특징 직접 찾아보기**")
            st.markdown(
                """
아래 질문에 답하려면 왼쪽 위 슬라이더를 다시 움직여 보세요.  
처음에는 힌트를 열지 말고 그래프 변화만 보고 예상해 봅니다.
"""
            )
            st.text_area(
                "1. 매개변수를 움직였을 때 그래프에서 가장 크게 달라진 점은 무엇인가요?",
                height=78,
                key="d8_t_discovery_1",
                placeholder="예: 그래프가 위아래로 이동했다 / 더 가파르게 변했다 / 시작점이 바뀌었다.",
            )
            st.text_area(
                "2. 이 함수 그래프가 현재 산점도의 흐름과 잘 맞는 부분은 무엇인가요?",
                height=78,
                key="d8_t_discovery_2",
                placeholder="예: 증가하다가 감소하는 흐름이 비슷하다.",
            )
            st.text_area(
                "3. 이 함수 그래프가 현재 데이터와 잘 맞지 않는 부분은 무엇인가요?",
                height=78,
                key="d8_t_discovery_3",
                placeholder="예: 뒤쪽 점들은 잘 맞지만 앞쪽 점들은 많이 벗어난다.",
            )
            if function_type == "유리함수":
                st.warning("점근선 근처에서는 함수값이 매우 커질 수 있습니다.")
            if function_type == "무리함수":
                st.warning("x-p가 음수이면 함수값이 정의되지 않으므로, 그래프에는 정의되는 구간만 표시됩니다.")
            with st.expander("필요할 때 열어 보는 단계형 힌트", expanded=False):
                show_function_hints(function_type)
        with feature_right:
            st.markdown("**오른쪽 확인: 현재 데이터에 맞는지 판단**")
            complexity_label, _ = model_complexity_text(function_type)
            st.info(f"현재 선택한 함수의 매개변수 수: {complexity_label}")
            st.markdown(
                """
**이 단계의 수학적 초점**
- 함수의 매개변수는 그래프의 위치, 방향, 굽은 정도를 바꿉니다.
- 추세선은 점을 모두 지나가는 선이 아니라, 전체 흐름을 설명하는 함수 그래프입니다.
- 복잡한 함수가 항상 좋은 것은 아니며, 데이터의 모양을 설명할 만큼만 복잡해야 합니다.
"""
            )
            st.text_area(
                "힌트를 보고 난 뒤, 우리 모둠이 정리한 그래프 특징",
                height=90,
                key="d8_t_graph_judgement",
                placeholder="예: a를 바꾸면 굽은 방향이 달라지고, p와 q를 움직이면 꼭짓점 위치가 바뀐다.",
            )
        with st.expander("T 단계 수학 보충: 함수 그래프를 모델로 본다는 뜻", expanded=False):
            st.markdown(
                """
- 데이터 과학에서 **모델**은 현실의 관계를 단순화한 수학적 표현입니다. 이 앱에서는 일차함수, 이차함수, 유리함수, 무리함수가 모델입니다.
- **매개변수**는 모델의 모양을 정하는 값입니다. 일차함수의 a, b나 이차함수의 a, p, q가 모두 매개변수입니다.
- **일차함수 모델**은 변화율이 거의 일정할 때 적절합니다. 잔차가 곡선 모양으로 남으면 일차함수가 너무 단순할 수 있습니다.
- **이차함수 모델**은 증가하다 감소하거나 감소하다 증가하는 흐름을 설명할 수 있습니다. 꼭짓점의 위치가 데이터 흐름의 전환점과 맞는지 봅니다.
- **유리함수 모델**은 점근선을 가지므로 포화, 급격한 변화, 한계값에 가까워지는 흐름을 설명할 수 있습니다. 단, 점근선 근처 예측은 불안정합니다.
- **무리함수 모델**은 시작점과 점점 완만해지는 변화를 설명할 수 있습니다. 정의역이 제한되므로 x-p가 음수인 입력은 예측할 수 없습니다.
- **과소적합**은 모델이 너무 단순해서 데이터의 구조를 놓치는 경우이고, **과적합**은 관찰된 점에는 과하게 맞지만 새로운 값 예측에는 약한 경우입니다.
"""
            )
        st.caption("다음 [U] 단계에서는 같은 그래프를 잔차와 손실값으로 더 엄밀하게 해석합니다.")

    with tabs[2]:
        dataset_name = st.session_state["d8_dataset"]
        selected_data = DATASETS[dataset_name]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        function_type = st.session_state["d8_function"]
        stage_intro(
            "AI 이해·활용: AI는 데이터를 바탕으로 어떤 예측을 할까?",
            "함수 그래프를 예측 모델로 보고, 잔차와 손실값을 근거로 예측 결과와 모델의 한계를 해석합니다.",
            "AI는 데이터를 바탕으로 어떤 예측을 할까?",
            "#ede7f6",
            "#d1c4e9",
        )
        default_new_x = float(max(x_data) + (x_data[1] - x_data[0] if len(x_data) > 1 else 1))
        if "d8_new_x" not in st.session_state:
            st.session_state["d8_new_x"] = default_new_x
        st.markdown(
            """
            <div class="prediction-panel">
                <div class="prediction-title">예측할 새 입력값 정하기</div>
                <div class="prediction-help">
                    관찰된 데이터 범위 안의 값을 넣으면 보간, 범위 밖의 값을 넣으면 외삽입니다.
                    입력한 x값은 오른쪽 그래프의 초록 예측점으로 표시됩니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        new_x = render_prediction_input_card(x_data, x_label)

        params = st.session_state.get("d8_params")
        if params is None or st.session_state.get("d8_function_type") != function_type:
            st.info("먼저 [T] 단계에서 함수 슬라이더를 한 번 확인해 주세요. 현재는 기본값으로 계산합니다.")
            with st.expander("기본 슬라이더 값으로 계산하기", expanded=True):
                params = get_parameters(function_type, x_data, y_data)
                st.session_state["d8_params"] = params
                st.session_state["d8_function_type"] = function_type

        predicted_data_y, valid_data_mask = calculate_function(function_type, x_data, params)
        loss, valid_count = calculate_loss(y_data, predicted_data_y, valid_data_mask)
        metrics = calculate_error_metrics(y_data, predicted_data_y, valid_data_mask)
        predicted_y = predict_value(function_type, float(new_x), params)
        function_text = build_function_text(function_type, params)

        if function_type in ["유리함수", "무리함수"] and valid_count < len(x_data):
            st.warning(
                f"선택한 함수가 정의되지 않는 데이터가 있어, 정의되는 {valid_count}개의 데이터만으로 손실값을 계산했습니다."
            )
        if function_type == "유리함수" and np.any(np.abs(x_data - params["p"]) <= 1e-8):
            st.warning("점근선 근처에서는 함수값이 매우 커질 수 있습니다.")

        st.markdown(pretty_title("1. 예측 기준 정하기", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        pred_left, pred_right = st.columns([0.85, 1.15])
        with pred_left:
            st.markdown("**왼쪽 활동: 새 입력값과 함수식 확인**")
            if loss is None:
                st.error("현재 추세선의 손실값: 계산할 수 없음")
            else:
                st.metric("현재 추세선의 손실값", f"{loss:.3f}")
                st.caption("현재 앱의 기본 손실값은 MSE입니다.")

            st.markdown("**현재 함수식**")
            st.code(function_text, language="text")

            metric_cards = st.columns(2)
            metric_cards[0].metric("RMSE", "계산 불가" if metrics["rmse"] is None else f"{metrics['rmse']:.3f}")
            metric_cards[1].metric("MAE", "계산 불가" if metrics["mae"] is None else f"{metrics['mae']:.3f}")
            metric_cards_2 = st.columns(2)
            metric_cards_2[0].metric("R²", "계산 불가" if metrics["r2"] is None else f"{metrics['r2']:.3f}")
            metric_cards_2[1].metric("사용한 점", f"{metrics['valid_count']}개")

            st.caption(
                "MSE는 큰 오차를 더 강하게 벌점으로 주고, MAE는 평균적인 오차 크기를 직관적으로 보여 줍니다. "
                "RMSE는 y값과 같은 단위로 해석하기 쉽습니다. R²는 데이터 변동을 모델이 어느 정도 설명하는지 보는 참고 지표입니다."
            )

            range_type, range_message = prediction_range_label(float(new_x), x_data)
            if range_type == "보간":
                st.info(f"예측 위치: 보간. {range_message}")
            else:
                st.warning(f"예측 위치: 외삽. {range_message} 관찰 범위 밖 예측은 그래프 모양을 믿고 연장하는 것이므로 더 조심해야 합니다.")

            if predicted_y is None:
                st.warning("새 입력값이 선택한 함수의 정의역에 맞지 않아 예측값을 계산할 수 없습니다.")
            else:
                st.success(f"선택한 함수에서 f({float(new_x):g})={predicted_y:.2f}입니다.")
                st.markdown(f"따라서 x={float(new_x):g}일 때 y값은 약 **{predicted_y:.2f}**로 예측됩니다.")

        with pred_right:
            st.markdown("**오른쪽 확인: 예측점이 포함된 추세선 그래프**")
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
                ),
                use_container_width=True,
            )
            st.caption(
                "손실값이 작을수록 실제 데이터와 함수 그래프의 차이가 작다는 뜻입니다. 그러나 최선의 추세선은 손실값뿐 아니라 그래프의 모양이 데이터의 전체 흐름을 잘 설명하는지도 함께 고려해야 합니다."
            )

        st.markdown(pretty_title("2. 오차로 추세선 확인하기", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        error_cols = st.columns(4)
        error_cols[0].metric("MSE", "계산 불가" if metrics["mse"] is None else f"{metrics['mse']:.3f}")
        error_cols[1].metric("RMSE", "계산 불가" if metrics["rmse"] is None else f"{metrics['rmse']:.3f}")
        error_cols[2].metric("MAE", "계산 불가" if metrics["mae"] is None else f"{metrics['mae']:.3f}")
        error_cols[3].metric("R²", "계산 불가" if metrics["r2"] is None else f"{metrics['r2']:.3f}")
        st.caption(
            "잔차는 실제값과 함수값의 차이입니다. 여기서는 표와 잔차 그래프 대신 핵심 지표만 확인합니다. "
            "MSE와 RMSE가 작을수록 오차가 작고, R²는 데이터 변동을 모델이 어느 정도 설명하는지 보여 주는 참고값입니다."
        )

        st.markdown(pretty_title("3. 최종 추세선 판단하기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        judge_left, judge_right = st.columns([0.9, 1.1])
        with judge_left:
            st.markdown("**왼쪽 활동: 우리 모둠의 선택 근거 작성**")
            st.text_area(
                "우리 모둠이 이 함수를 최선의 추세선으로 선택한 이유를 쓰세요.",
                height=150,
                placeholder="그래프의 모양, 증가·감소, 굽은 정도, 손실값, 잔차 패턴을 근거로 설명해 보세요.",
                key="choice_reason",
            )
            st.text_area(
                "AI·데이터 과학 관점에서 우리 모둠의 추세선을 다시 평가해 보세요.",
                height=130,
                placeholder="예: 손실값은 작지만 잔차가 한쪽으로 남아 있어 다른 함수도 비교해야 한다. 새 x가 관찰 범위 밖이라 예측은 조심해야 한다.",
                key="ds_reflection",
            )
        with judge_right:
            st.markdown("**오른쪽 확인: 판단 질문과 모델 복잡도**")
            complexity_label, complexity_detail = model_complexity_text(function_type)
            st.markdown(
                f"""
- **모델**: 지금 선택한 `{function_type}` 그래프
- **복잡도**: {complexity_label}
- **현재 해석**: {complexity_detail}
- **최소제곱 관점**: 잔차제곱의 합을 작게 만드는 매개변수를 찾는 것이 회귀의 핵심 아이디어입니다.
"""
            )
            st.markdown(
                f"""
**판단 질문**
1. 손실값이 작은 이유가 전체 흐름을 잘 설명해서인가, 특정 점 몇 개에 끌려가서인가?
2. 잔차가 x축 위아래에 고르게 흩어져 있는가, 아니면 패턴이 남아 있는가?
3. {range_message}
4. 이 함수는 현실의 원인 관계를 설명하는가, 아니면 관찰된 점들의 모양만 설명하는가?
"""
            )
            st.markdown("**데이터 해석 렌즈로 다시 점검**")
            st.info(selected_factfulness_text())
            st.text_area(
                "이 관점으로 보면 우리 추세선 해석에서 조심해야 할 점은 무엇인가요?",
                height=96,
                key="d8_factfulness_model_check",
                placeholder="예: 최근 두 점만 보고 앞으로도 같은 속도로 변한다고 말하면 직선 본능에 빠질 수 있다.",
            )
            with st.expander("회귀 분석 개념 더 보기", expanded=False):
                st.markdown(
                    """
- **SSE**는 잔차제곱합입니다. 모든 점에서 `(실제값 - 예측값)²`을 더한 값입니다.
- **MSE**는 SSE를 데이터 개수로 나눈 평균제곱오차입니다. 이 앱의 기본 손실값입니다.
- **RMSE**는 MSE의 제곱근입니다. y와 같은 단위라서 “평균적으로 어느 정도 틀리는가”를 말하기 쉽습니다.
- **MAE**는 절댓값 오차의 평균입니다. 큰 이상치의 영향을 MSE보다 덜 받습니다.
- **R²**는 전체 y 변동 중 모델이 설명한 비율을 나타냅니다. 다만 함수 종류와 데이터 수가 다르면 R²만으로 비교하면 위험합니다.
- **최소제곱 추세선**은 SSE가 작아지도록 매개변수를 정하는 관점입니다. 학생 활동에서는 슬라이더로 이 과정을 직접 탐색합니다.
- **외삽**은 관찰한 x 범위 밖으로 그래프를 연장하는 예측입니다. 유리함수와 무리함수처럼 정의역이나 점근선이 있는 함수에서는 특히 조심해야 합니다.
- **상관관계는 인과관계가 아닙니다.** 추세선은 두 변수의 관계를 설명하는 도구이지, 반드시 원인을 증명하는 도구는 아닙니다.
"""
                )

        st.session_state["d8_loss"] = loss
        st.session_state["d8_predicted_y"] = predicted_y
        st.session_state["d8_function_text"] = function_text
        st.session_state["d8_metrics"] = metrics

    with tabs[3]:
        group_name = st.session_state.get("d8_group", "우리 모둠")
        dataset_name = st.session_state["d8_dataset"]
        selected_data = DATASETS[dataset_name]
        x_data, y_data, x_label, y_label = selected_xy_data(dataset_name)
        function_type = st.session_state["d8_function"]
        new_x = float(st.session_state["d8_new_x"])
        prediction_question = st.session_state.get("d8_prediction_question", "")
        stage_intro(
            "세상과 연결: AI 예측 결과를 사회문제 해결에 어떻게 활용할까?",
            "예측 결과를 그대로 믿기보다, 데이터의 근거와 한계를 성찰하고 사람과 사회에 도움이 되는 앱 제작 프롬프트로 확장합니다.",
            "AI 예측 결과를 바탕으로 사람과 사회에 도움이 되는 앱으로 어떻게 확장할까?",
            "#fce4ec",
            "#f8bbd0",
        )

        params = st.session_state.get("d8_params")
        if params is None:
            params = get_parameters(function_type, x_data, y_data)
        predicted_data_y, valid_data_mask = calculate_function(function_type, x_data, params)
        loss, _ = calculate_loss(y_data, predicted_data_y, valid_data_mask)
        predicted_y = predict_value(function_type, float(new_x), params)
        function_text = build_function_text(function_type, params)
        loss_text = format_optional_number(loss)
        prediction_text = (
            f"x={float(new_x):g}일 때 y≈{predicted_y:.2f}" if predicted_y is not None else "정의역 문제로 예측 불가"
        )

        st.markdown(pretty_title("1. 예측 결과의 의미와 한계 정리", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.info(f"우리 모둠의 예측 질문: {prediction_question if prediction_question else '아직 작성하지 않았습니다.'}")
        re_col1, re_col2, re_col3 = st.columns(3)
        with re_col1:
            deep_question = st.text_area(
                "깊은 질문",
                height=120,
                key="deep_question",
                placeholder=selected_data.get("deep_question", "데이터가 보여 주는 인간 또는 사회 문제를 질문으로 바꾸어 보세요."),
            )
            st.caption("깊은 질문은 예측값 자체보다 데이터 속 인간 또는 사회적 문제의 의미를 묻는 질문입니다.")
        with re_col2:
            limitation = st.text_area(
                "예측 결과의 한계",
                height=120,
                key="limitation",
                placeholder="예: 자료가 적고, 다른 원인을 포함하지 않았으며, 관찰 범위 밖 예측은 조심해야 한다.",
            )
        with re_col3:
            action_message = st.text_area(
                "실천 메시지",
                height=120,
                key="action_message",
                placeholder="예: 예측 결과를 단정하지 말고, 데이터의 한계까지 함께 확인하자.",
            )
        with st.expander("데이터 해석 렌즈를 앱 메시지에 반영하기", expanded=False):
            st.markdown(
                f"""
- 선택한 관점: **{selected_factfulness_text()}**
- 앱 화면에는 예측값을 보여 주되, 사용자가 성급하게 결론 내리지 않도록 비교 기준과 한계를 함께 넣습니다.
- 좋은 정보형 앱은 “무엇이 변했는가”뿐 아니라 “어디까지 믿을 수 있는가”를 같이 안내합니다.
"""
            )
            st.text_area(
                "앱에 넣고 싶은 한 문장 반성 문구",
                key="d8_factfulness_app_message",
                height=78,
                placeholder="예: 이 예측은 관찰된 기간의 흐름을 바탕으로 한 것이며, 모든 지역과 사람에게 똑같이 적용되지 않습니다.",
            )

        st.markdown(pretty_title("2. Canva AI로 만들 앱 방향 정하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
        app_col1, app_col2 = st.columns([0.9, 1.1])
        with app_col1:
            output_type = APP_OUTPUT_TYPE
            st.info("이번 활동의 Canva AI 산출물은 **정보형 앱** 하나로 고정합니다.")
            target_user = st.selectbox("대상 사용자", TARGET_USERS, key="d8_target_user")
            design_tone = st.text_input(
                "원하는 디자인 분위기",
                key="d8_design_tone",
                value="밝고 명확한 교육용 디자인, 핵심 수치가 잘 보이는 카드형 구성",
            )
        with app_col2:
            selected_features = st.multiselect(
                "앱에 넣을 핵심 기능",
                APP_FEATURES,
                default=["핵심 예측값 카드", "산점도와 추세선 설명", "보간·외삽 주의 문구", "실천 메시지 강조"],
                key="d8_app_features",
            )
            st.text_area(
                "Canva AI가 특히 강조해야 할 점",
                height=110,
                key="d8_ds_reflection_re",
                value=st.session_state.get("ds_reflection", ""),
                placeholder="예: 결과만 보여주지 말고 데이터 범위와 예측 한계를 함께 안내해 주세요.",
            )

        ds_reflection = st.session_state.get("d8_ds_reflection_re", "") or st.session_state.get("ds_reflection", "")
        canva_prompt = build_canva_prompt(
            group_name,
            dataset_name,
            prediction_question,
            function_type,
            function_text,
            loss_text,
            prediction_text,
            output_type,
            target_user,
            selected_features,
            deep_question,
            limitation,
            action_message,
            ds_reflection,
            design_tone,
        )

        st.markdown(pretty_title("3. Canva AI와 갤러리 패들렛 연결", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        link_left, link_right = st.columns([0.75, 1.25])
        with link_left:
            st.selectbox("반", CLASS_OPTIONS, key="d8_class")
            st.text_input("학번", key="d8_student_id", placeholder="예: 10123")
            st.caption("자신의 반을 선택한 뒤 오른쪽 버튼으로 Canva AI와 반별 갤러리 패들렛에 접속합니다.")
        with link_right:
            render_canva_gallery_links(st.session_state.get("d8_class", CLASS_OPTIONS[0]))
            st.caption("Canva AI에는 아래 최종 프롬프트를 붙여 넣고, 완성한 정보형 앱은 자기 반 갤러리 패들렛에 공유합니다.")

        st.markdown(pretty_title("4. Canva AI 입력용 최종 프롬프트", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.code(canva_prompt, language="markdown")
        st.download_button(
            "Canva AI 프롬프트 txt 다운로드",
            data=canva_prompt,
            file_name=f"{group_name}_CanvaAI_앱제작_프롬프트.txt",
            mime="text/plain",
            use_container_width=True,
        )

        with st.expander("포스터형 요약 카드 미리보기", expanded=False):
            render_poster_card(
                group_name,
                dataset_name,
                prediction_question,
                function_type,
                function_text,
                loss_text,
                prediction_text,
                deep_question,
                limitation,
                action_message,
            )

        observation_checks = [
            option for idx, option in enumerate(flow_options) if st.session_state.get(f"flow_{idx}", False)
        ]
        download_text = make_download_text(
            group_name,
            dataset_name,
            function_type,
            function_text,
            loss,
            float(new_x),
            predicted_y,
            observation_checks,
            st.session_state.get("observation_text", ""),
            st.session_state.get("choice_reason", ""),
            deep_question,
            limitation,
            action_message,
        )
        st.download_button(
            "활동 결과 txt 파일 다운로드",
            data=download_text,
            file_name=f"{group_name}_함수_추세선_활동결과.txt",
            mime="text/plain",
            use_container_width=True,
        )

        with st.expander("오늘의 수학 정리", expanded=False):
            render_math_summary()

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
# 실행: streamlit run data7.py

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


try:
    font_path = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        mpl.rc("font", family=fm.FontProperties(fname=font_path).get_name())
    else:
        mpl.rc("font", family="Malgun Gothic")
    mpl.rc("axes", unicode_minus=False)
except Exception:
    font_path = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")
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
CANVA_AI_URL = "https://www.canva.com/ai"
APP_OUTPUT_TYPE = "정보 카드"
TARGET_USERS = ["고등학생", "중학생", "학부모", "지역사회", "정책 결정자", "일반 대중"]
APP_FEATURES = [
    "핵심 예측값 카드",
    "산점도와 추세선 설명",
    "오차와 손실값 안내",
    "보간·외삽 주의 문구",
    "깊은 질문 제시",
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
    add_text_box_to_pdf(pdf, "선택한 데이터", student_info.get("dataset", ""))
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


def build_canva_prompt(group_name, dataset_name, prediction_question, function_type, function_text, loss_text, prediction_text, target_user, selected_features, deep_question, limitation, action_message, ds_reflection, design_tone):
    feature_text = ", ".join(selected_features) if selected_features else "핵심 예측값, 추세선 설명, 한계 안내"
    return f"""Canva AI에서 만들 정보 카드 기획안

모둠명: {group_name}
데이터 주제: {dataset_name}
예측 질문: {prediction_question}
깊은 질문: {deep_question}
함수 종류: {function_type}
함수식: {function_text}
손실값(MSE): {loss_text}
예측 결과: {prediction_text}
대상 사용자: {target_user}
디자인 분위기: {design_tone}
포함 기능: {feature_text}

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
        f"""<a href="{url}" target="_blank" style="display:block;padding:11px;background:{gradient};color:white;
        text-decoration:none;border-radius:8px;font-weight:bold;text-align:center;margin-top:8px;">{label}</a>""",
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
    st.session_state.setdefault("d8_student_id", "")
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

            st.markdown(pretty_title("2. 예측 질문과 깊은 질문 만들기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            render_stage_card(
                "수치 예측 질문과 사회적 질문을 함께 만듭니다",
                "예측 질문은 x와 y의 관계를 묻고, 깊은 질문은 그 관계가 사람과 사회에 주는 의미를 묻습니다.",
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
                "깊은 질문",
                key="d8_deep_question",
                height=96,
                placeholder=chosen_info.get("deep_question", "데이터 속에서 보이는 인간 또는 사회적 문제를 질문으로 바꾸어 적어 보세요."),
            )
            st.caption("깊은 질문은 예측값 자체보다 데이터 속 인간 또는 사회적 문제의 의미를 묻는 질문입니다.")
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
                        ("데이터", dataset_name),
                        ("x/y 변수", f"x={x_label}, y={y_label}"),
                        ("예측 질문", st.session_state.get("d8_prediction_question", "")),
                        ("깊은 질문", st.session_state.get("d8_deep_question", "")),
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
                        ("관찰한 흐름", ", ".join(selected_flows) or "선택 없음"),
                        ("관찰 설명", st.session_state.get("observation_text", "")),
                        ("해석 렌즈", selected_factfulness_text()),
                        ("조심할 점", st.session_state.get("d8_factfulness_question", "")),
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
                        <div class="prediction-input-title">새로운 x값을 입력하세요</div>
                        <div class="prediction-input-help">
                            독립변수는 <b>{html.escape(str(x_label))}</b>입니다.
                            관찰된 x 범위는 <b>{x_min:g} ~ {x_max:g}</b>이며,
                            현재 입력값은 <span style="font-weight:900;color:{range_color};">{range_label}</span>입니다.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                new_x = st.number_input(
                    "예측할 x값",
                    key="d8_new_x",
                    step=0.5,
                    format="%.2f",
                    help="그래프에서 예측하고 싶은 독립변수 x값을 입력합니다.",
                )
            predicted_y = predict_value(function_type, float(new_x), params)
            range_type, range_message = prediction_range_label(float(new_x), x_data)
            with value_col:
                if predicted_y is not None:
                    st.markdown(
                        f"""
                        <div class="stage-card stage-card-green">
                            <div class="stage-kicker">선택한 함수에서의 함숫값</div>
                            <div class="prediction-input-title">함숫값(예측값)</div>
                            <div class="stage-card-title">예측값 = {predicted_y:.2f}</div>
                            <div class="stage-card-help">
                                입력한 x값을 현재 선택한 함수에 넣어 계산한 y값입니다.<br>
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
                    "U 단계: 함수 그래프와 예측 한계 정리하기",
                    [
                        ("함수 종류", function_type),
                        ("함수식", function_text),
                        ("손실값", format_optional_number(loss)),
                        ("예측 결과", f"x={float(new_x):g}, y={predicted_y:.2f}" if predicted_y is not None else "계산 불가"),
                        ("데이터 흐름 설명", st.session_state.get("choice_reason", "")),
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
                "수치 자체보다 깊은 질문과 예측의 한계를 함께 담아야 정보 콘텐츠가 과장되지 않습니다.",
                "blue",
                "콘텐츠 메시지",
            )
            deep_question = st.text_area(
                "깊은 질문(D.E.E.P Question)",
                key="d8_final_deep_question",
                value=st.session_state.get("d8_deep_question", ""),
                height=95,
                placeholder="예: 수치가 좋아져도 여전히 혜택을 받지 못하는 사람들은 누구일까?",
            )
            action_message = st.text_area(
                "실천 메시지",
                key="action_message",
                height=95,
                placeholder="예: 예측 결과를 볼 때는 숫자만 믿지 말고, 데이터 범위와 예측 한계를 함께 확인하자.",
            )
    
            st.markdown(pretty_title("2. 정보 콘텐츠 방향 정하기", "#fff8e1", "#ffecb3"), unsafe_allow_html=True)
            render_stage_card(
                "콘텐츠의 표현 방향을 정합니다",
                "디자인 분위기와 특히 강조해야 할 점을 정하면, 최종 프롬프트가 더 명확해집니다.",
                "yellow",
                "제작 방향",
            )
            target_user = "학급 친구들"
            selected_features = []
            design_tone = st.text_input("디자인 분위기", key="d8_design_tone", value="밝고 명확한 교육용 정보 카드")
            ds_reflection = st.text_area(
                "콘텐츠에 담고 싶은 우리 모둠의 생각",
                key="d8_ds_reflection_re",
                height=95,
                placeholder="예: 이 데이터가 단순한 숫자가 아니라 사람들의 삶과 연결되어 있다는 점을 보여 주고 싶다.",
            )
            choice_reason = st.session_state.get("choice_reason", "").strip()
            limitation = st.session_state.get("limitation", "").strip()
            if st.button("E 단계 저장", use_container_width=True):
                save_stage_snapshot(
                    4,
                    "E 단계: 정보 콘텐츠로 공유하기",
                    [
                        ("깊은 질문", deep_question),
                        ("최종 추세선 선택 이유", choice_reason),
                        ("예측 한계", limitation),
                        ("실천 메시지", action_message),
                        ("디자인 분위기", design_tone),
                        ("모둠의 생각", ds_reflection),
                    ],
                )
            saved_stage_caption(4)

            canva_limitation_text = "\n".join(
                part
                for part in [
                    f"최종 추세선 선택 이유: {choice_reason}" if choice_reason else "",
                    f"예측 한계: {limitation}" if limitation else "",
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
                target_user,
                selected_features,
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


if __name__ == "__main__":
    run()
