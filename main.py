import streamlit as st
import importlib

# 페이지 기본 설정
st.set_page_config(page_title="F.U.T.U.R.E Studio", page_icon="💡", layout="centered")

# 페이지 제목
st.title(":rainbow[F.U.T.U.R.E Studio]")
st.markdown(
    """
    <style>
    .top-qna-link {
        display: inline-block;
        background: linear-gradient(90deg, #1976d2 5%, #42a5f5 90%);
        color: #fff !important;
        font-size: 17px;
        font-weight: 800;
        padding: 5px 10px 5px 12px;
        border-radius: 2em;
        box-shadow: 0 4px 18px rgba(25,118,210,0.13);
        margin: 0px 0 0px 0;
        letter-spacing: 1.2px;
        text-decoration: none !important;
        transition: background 0.16s, box-shadow 0.18s, transform 0.13s;
        position: relative;
    }
    .top-qna-link:hover {
        background: linear-gradient(90deg,#42a5f5 5%,#1976d2 90%);
        color: #fff !important;
        transform: translateY(-2px) scale(1.045);
        box-shadow: 0 7px 24px #1976d222;
        text-decoration: none !important;
    }
    </style>
    <div style='text-align: right;'>
        <a class="top-qna-link" href="https://padlet.com/ps0andd/hub" target="_blank">
            <span class="qna-emoji">📢</span> 패들렛(Padlet) 바로 가기
    </div>
    """,
    unsafe_allow_html=True
)


# 1~7차시 드롭다운 메뉴 구성
days = [
    "1DAY - 📦 수학의 언어를 파이썬으로 (변수와 함수)",
    "2DAY - 🔀 데이터 최적화 알고리즘 (if문)",
    "3DAY - 🔢 경우의 수 폭발과 AI 탐색",
    "4DAY - 🖼️ 세상의 데이터는 행렬이다 (이미지 처리)",
    "5DAY - 📉 수학적 오차와 AI 예측 (딥러닝 원리)",
    "6DAY - 🔮 Data Matrix 예측 스튜디오 (실생활 분석)",
    "7DAY - 📢 바이브 코딩 & 감성 수학 확산"
]

modules = {
    days[0]: "data1",
    days[1]: "data2",
    days[2]: "data3",
    days[3]: "data4",
    days[4]: "data5",
    days[5]: "data6",
    days[6]: "data7",
}

# 💡 수정된 부분: 복잡한 콜백 함수를 지우고 key 하나로 상태를 동기화합니다.
if 'current_day' not in st.session_state:
    st.session_state.current_day = days[0]

# 사이드바가 아닌 메인 화면 중앙에 드롭다운 배치
selected_day = st.selectbox(
    "👇 도전을 시작합시다! 수업을 선택하세요",
    options=days,
    key="current_day"  # key를 지정하면 자동으로 session_state에 저장 및 동기화됩니다.
)

st.divider()

# 선택된 모듈 동적 실행
current_module_name = modules[st.session_state.current_day]

try:
    module = importlib.import_module(current_module_name)
    importlib.reload(module)
    if hasattr(module, 'run'):
        module.run()
except ModuleNotFoundError:
    st.error(f"개발 중입니다: {current_module_name}.py 파일을 생성해주세요.")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")