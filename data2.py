import streamlit as st
from streamlit_ace import st_ace
import pandas as pd
import io
import sys
import datetime
import os
import builtins
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.figure import Figure # 다중 접속 시 그래프 충돌을 막기 위해 Figure 사용
from fpdf import FPDF

# ==========================================
# 0. Matplotlib 한글 폰트 설정
# ==========================================
try:
    font_path = os.path.join(os.path.dirname(__file__), "font", "NanumGothic.ttf")
    fm.fontManager.addfont(font_path)  
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc('font', family=font_name)   
    mpl.rc('axes', unicode_minus=False) 
except Exception as e:
    pass

# ==========================================
# 1. 고품질 PDF 생성 클래스 (ThemedPDF)
# ==========================================
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
        self.rect(0, 0, self.w, 22, 'F')
        self.set_xy(10, 6)
        self.set_text_color(255, 255, 255)
        self.set_font(self._font_family, '', 20)
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 2차시 학습 포트폴리오", ln=1, align='C')
        self.set_text_color(33, 33, 33)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.set_y(-12)
        self.set_font(self._font_family, '', 9)
        self.set_text_color(*self.c_text_muted)
        if self.footer_left:
            self.cell(0, 8, self.footer_left, 0, 0, 'L')
        self.cell(0, 8, f"{self.page_no()} / {{nb}}", 0, 0, 'R')

    def h2(self, text):
        self.set_fill_color(*self.c_primary_lt)
        self.set_text_color(21, 101, 192)
        self.set_font(self._font_family, '', 12)
        self.cell(0, 9, text, ln=1, fill=True)
        self.ln(2)
        self.set_text_color(33, 33, 33)

    def p(self, text, size=11, lh=6):
        self.set_font(self._font_family, '', size)
        self.multi_cell(0, lh, text)
        self.ln(3)

    def kv_card(self, title, kv_pairs):
        self.h2(title)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.3)
        self.set_font(self._font_family, '', 11)
        self.set_fill_color(255, 255, 255)
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
        self.ln(cell_h + 2)

def create_portfolio_pdf(student_info, code_data):
    pdf = ThemedPDF()
    pdf.add_font('Nanum', '', font_path, uni=True)
    pdf.set_font('Nanum', '', 12)
    pdf._font_family = "Nanum"   
    pdf.footer_left = f"{student_info.get('group','')} • {student_info.get('name','')}"
    pdf.add_page()
    
    kvs = [
        ("모둠명", student_info.get('group', '')),
        ("학번", student_info.get('id', '')),
        ("이름", student_info.get('name', '')),
        ("작성일", datetime.datetime.now().strftime("%Y-%m-%d")),
    ]
    pdf.ln(5)  
    pdf.kv_card("👤 학생 정보", kvs)
    
    # 교사의 딥 퀘스천 부분 삭제됨
    
    pdf.h2("💻 나의 파이썬 코딩 실습 결과")
    for title, code_text, result_text in code_data:
        pdf.set_font(pdf._font_family, '', 11)
        pdf.set_text_color(21, 101, 192)
        pdf.cell(0, 8, f"▶ {title}", ln=1)
        
        pdf.set_font(pdf._font_family, '', 10)
        pdf.set_text_color(50, 50, 50)
        pdf.set_fill_color(245, 245, 245)
        pdf.multi_cell(0, 6, code_text if code_text else "작성된 코드가 없습니다.", border=1, fill=True)
        pdf.ln(2)

        pdf.set_font(pdf._font_family, '', 11)
        pdf.set_text_color(46, 125, 50)
        pdf.cell(0, 8, f"🖥️ 실행 결과", ln=1)
        
        pdf.set_font(pdf._font_family, '', 10)
        pdf.set_text_color(50, 50, 50)
        pdf.set_fill_color(232, 245, 233)
        pdf.multi_cell(0, 6, result_text if result_text else "실행 결과가 없습니다.", border=1, fill=True)
        pdf.ln(6)
    
    return bytes(pdf.output(dest='S'))


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
        [data-testid="stDataFrame"] {
            border: 1px solid #e5eef7;
            border-radius: 0.75rem;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            border-radius: 0.8rem;
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
            <div style="font-size:0.9rem; font-weight:700; color:#5e35b1; margin-bottom:8px;">F.U.T.U.R.E. 프로젝트 2DAY</div>
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
            border:1px solid {border};
            background:{bg};
            box-shadow: 0 4px 10px rgba(0,0,0,0.04);
            margin-bottom:10px;
        ">
            <div style="font-size:0.88rem; color:#607d8b; font-weight:700; margin-bottom:6px;">{title}</div>
            <div style="font-size:1.18rem; color:#1f2937; font-weight:800; margin-bottom:6px;">{value}</div>
            <div style="font-size:0.92rem; color:#455a64; line-height:1.6;">{detail}</div>
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

# ==========================================
# 2. 파이썬 코드 실행 엔진 (다중 접속 동시성 + 시각화 지원 완벽 방어)
# ==========================================
def code_runner(code_input):
    output_buffer = io.StringIO()
    result, status = "", "success"
    fig = None 
    
    def custom_print(*args, sep=' ', end='\n', file=None, flush=False):
        if file is None:
            output_buffer.write(sep.join(map(str, args)) + end)
        else:
            builtins.print(*args, sep=sep, end=end, file=file, flush=flush)

    # 💡 동시 접속 방어를 위해 plt 상태 머신 대신 객체(Figure)를 직접 생성합니다.
    def draw_graph(a, b, c):
        nonlocal fig
        fig = Figure(figsize=(6, 4))
        ax = fig.subplots()
        x = np.linspace(-10, 10, 400)
        y = a*x**2 + b*x + c
        ax.plot(x, y, label=f'y = {a}x^2 + {b}x + {c}', color='#1976d2', linewidth=2)
        ax.axhline(0, color='#d32f2f', linewidth=2, label='x-axis (y=0)') 
        ax.axvline(0, color='black', linewidth=1) 
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_ylim(-15, 20)

    safe_builtins = builtins.__dict__.copy()
    safe_builtins['print'] = custom_print
    
    exec_globals = {
        '__builtins__': safe_builtins,
        'draw_graph': draw_graph 
    }
    
    try:
        exec(code_input, exec_globals)
        result = output_buffer.getvalue() or "출력된 내용이 없습니다."
    except Exception as e:
        result = f"{e.__class__.__name__}: {e}"
        status = "error"
        
    return result, status, fig

def display_output(result, status, fig):
    if status == "success":
        st.markdown(f"```bash\n{result}\n```")
        if fig is not None:
            st.pyplot(fig) 
    else:
        st.markdown("##### ❌ 실행 중 오류 발생")
        st.markdown(f"<pre style='color: red; background-color: #ffe6e6; padding: 10px; border-radius: 5px;'>{result}</pre>", unsafe_allow_html=True)

def code_block(problem_number, title, starter_code, prefix="", height=280):
    key_prefix = f"{prefix}{problem_number}"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(pretty_title(f"📥 {title} (코드 입력)", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        code_input = st_ace(value=starter_code, language='python', theme='github', height=height, key=f"{key_prefix}_editor")
        if st.button("▶️ 실행", key=f"{key_prefix}_run", use_container_width=True):
            st.session_state[f"{key_prefix}_result"] = code_runner(code_input)
    with c2:
        st.markdown(pretty_title("🖥️ 실행 결과", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        if f"{key_prefix}_result" in st.session_state:
            res, stat, fig = st.session_state[f"{key_prefix}_result"]
            display_output(res, stat, fig)
        else:
            st.info("실행 버튼을 누르면 결과가 표시됩니다.")

# ==========================================
# 3. 메인 앱 화면 (UI)
# ==========================================
def run():
    # 데스크탑 오류 방지를 위해 메인페이지 설정은 주석처리
    # st.set_page_config(page_title="F.U.T.U.R.E. 2차시", page_icon="🚀", layout="centered")
    apply_local_style()
    page_banner(
        "이차함수와 알고리즘의 만남",
        "파이썬의 조건문(if/elif/else)을 이해하고, 이를 통해 이차함수와 x축의 위치 관계를 코드로 판별하고 "
        "시각화하는 능력을 기르는 수업입니다.",
    )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True) 
    
    tabs = st.tabs([
        "1️⃣ [F.U] 문제 발견", 
        "2️⃣ [T] 수학의 언어", 
        "3️⃣ [U] AI 활용", 
        "4️⃣ [R] 결과 해석",
        "5️⃣ [E] 세상과 연결"
    ])
    
    # ------------------------------------------
    # 탭 1: 현실 탐색 [문맥화]
    # ------------------------------------------
    with tabs[0]:
        stage_intro(
            "현실 상황을 수학과 알고리즘으로 보기",
            "드론의 비행 궤적처럼 현실의 움직임 속에서 숨겨진 수학 구조를 발견하고, 컴퓨터가 어떤 규칙으로 상황을 판단하는지 이해하는 단계입니다.",
            "컴퓨터는 드론의 궤적이 지면과 만나는지 어떻게 판단할까?",
            "#e3f2fd",
            "#bbdefb",
        )
        
        st.markdown(pretty_title("❔ 문제 제기: 드론 비행 시뮬레이터", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write("당신은 자율주행 드론 비행 시스템의 개발자입니다. 드론의 고도 비행 궤적은 이차함수 $y = ax^2 + bx + c$ 의 포물선을 그리며 움직입니다. 여기서 **지면은 x축 ($y = 0$)**을 의미합니다.")
        st.info("""💡 **지면(x축)과 드론 궤적의 3가지 시나리오:**
1. 드론이 지면에 두 번 부딪히며 추락한다. (서로 다른 두 점에서 만남)
2. 드론이 지면에 아슬아슬하게 스치고 다시 날아오른다. (접함)
3. 드론이 지면에 전혀 닿지 않고 안전하게 비행한다. (만나지 않음)""")
        render_value_cards(
            [
                {
                    "title": "추락",
                    "value": "교점 2개",
                    "detail": "드론 궤적이 지면과 두 번 만나 위험한 상황입니다.",
                    "bg": "#ffebee",
                    "border": "#ef9a9a",
                },
                {
                    "title": "스침",
                    "value": "교점 1개",
                    "detail": "지면에 한 번 닿는 아슬아슬한 상황입니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
                {
                    "title": "안전",
                    "value": "교점 0개",
                    "detail": "지면과 만나지 않아 안전하게 비행합니다.",
                    "bg": "#e8f5e9",
                    "border": "#a5d6a7",
                },
            ],
            columns=3,
        )
        
        st.write("우리는 그래프를 눈으로 보고 판단하지만, 눈이 없는 **컴퓨터(AI)**는 드론이 추락할지 안전할지를 어떻게 수식으로 계산해서 판단할 수 있을까요?")
        
        hypothesis = st.text_input("💡 나의 가설 세워보기 (컴퓨터에게 지면 충돌 여부를 어떻게 수식으로 명령할 수 있을까?)", placeholder="나의 생각을 자유롭게 적어보세요.")
        if st.button("🔍 가설 확인"):
            if hypothesis.strip() == "":
                st.warning("틀려도 괜찮아요! 먼저 자신의 생각을 적어주세요.")
            else:
                st.success(f"**✅ 좋은 접근입니다!** '{hypothesis}'라고 생각했군요.")
                st.divider()
                st.markdown("""
                #### 🔓 알고리즘의 핵심: 위치 관계와 조건문
                * **수학적 해법:** 이차함수 그래프와 x축의 교점 개수는 이차방정식 $ax^2 + bx + c = 0$ 의 판별식 $D = b^2 - 4ac$ 의 부호에 따라 결정됩니다.
                * **컴퓨팅 해법:** 컴퓨터에게 조건문(`if`, `elif`, `else`)을 사용하여, **"만약 D > 0 이면 추락 경고를 출력하라!"** 와 같이 논리적으로 판단하게 만들 수 있습니다.
                """)

    # ------------------------------------------
    # 탭 2: 수학적 구조화 [수평적 수학화]
    # ------------------------------------------
    with tabs[1]:
        stage_intro(
            "판별식과 조건문으로 판단 규칙 만들기",
            "드론의 비행 문제를 판별식과 조건문으로 번역하여, 컴퓨터가 이해할 수 있는 명확한 판단 규칙으로 바꾸는 단계입니다.",
            "판별식 D와 조건문을 연결하면 어떤 판단 규칙을 만들 수 있을까?",
            "#fff8e1",
            "#ffecb3",
        )
        st.markdown(pretty_title("▶️ 조건문 (if / elif / else)", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.write("조건문은 주어진 조건의 참·거짓에 따라 서로 다른 명령을 실행하도록 컴퓨터의 논리적 흐름을 제어하는 구문입니다.")
        st.code("""
if 첫번째_조건:
    조건이 참일 때 실행 (※ 들여쓰기 필수!)
elif 두번째_조건:
    첫 번째는 거짓이지만, 두 번째 조건이 참일 때 실행
else:
    위의 모든 조건이 거짓일 때 마지막으로 실행
        """)
        
        # 💡 이미지 파일이 없어도 앱이 멈추지 않도록 예외 처리
        img_path = "image/data2_img1.png"
        if os.path.exists(img_path):
            st.image(img_path)
    
        st.markdown("""###### 💻 [예제 1] 온도에 따른 날씨 판별기""")
        st.write("온도(`temp`)에 따라 3가지 날씨 상태를 판별하는 기본 예제입니다. (아래 코드를 실행창에 넣어 테스트해보세요.)")
        st.code("""temp = 25
if temp >= 30:
    print('더운 날씨입니다.')
elif temp >= 15:
    print('따뜻한 날씨입니다.')
else:
    print('추운 날씨입니다.')""", language="python")
        
        starter_ex1 = """temp = 25
if temp >= 30:
    print('더운 날씨입니다.')
elif temp >= 15:
    print('따뜻한 날씨입니다.')
else:
    print('추운 날씨입니다.')
"""
        code_block("ex1", "예제 1 연습장", starter_ex1, prefix="d2_", height=210)
        
        st.divider()

        st.markdown(pretty_title("💻 문제 1. 양수, 0, 음수 판별기", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        st.write("위 예제 1을 참고하여 어떤 수 `num`이 양수인지, 0인지, 음수인지 3가지 경우로 나누어 판단하는 코드를 완성해 봅시다.")
        with st.expander("💡 힌트 보기"):
            st.markdown("`num > 0` 이면 양수, `num == 0` 이면 0, 그 외에는 `else:` 로 처리합니다.")
        with st.expander("💡 정답 보기"):
            st.code("""num = -5\nif num > 0:\n    print('양수입니다.')\nelif num == 0:\n    print('0입니다.')\nelse:\n    print('음수입니다.')""", language="python")

        starter_q1 = """num = -5
if num > 0:
    print('양수입니다.')
# 👇 if-elif-else 문을 완성하세요.
elif
    
"""
        code_block("q1", "조건문 기초 (문제 1)", starter_q1, prefix="d2_", height=230)
        

    # ------------------------------------------
    # 탭 3: 컴퓨팅 도구 활용 [수직적 수학화]
    # ------------------------------------------
    with tabs[2]:
        stage_intro(
            "그래프와 코드로 직접 실험하기",
            "그래프와 코드 도구를 활용해 계수를 바꾸고 결과를 시각화하면서, 컴퓨터의 판단 과정을 직접 실험하는 단계입니다.",
            "도구를 이용하면 어떤 계수에서 어떤 비행 결과가 나오는지 어떻게 빠르게 찾을 수 있을까?",
            "#e8f5e9",
            "#c8e6c9",
        )
        st.markdown(pretty_title("📈 시각적으로 이해하는 위치 관계", "#e8f5e9", "#c8e6c9"), unsafe_allow_html=True)
        st.write("왼쪽의 슬라이더를 움직여 이차함수의 계수를 조절해보고, 오른쪽 그래프와 설명을 통해 x축과의 위치 관계를 이해해 보세요.")

        col_viz1, col_viz2 = st.columns([1, 1.2])
        with col_viz1:
            st.markdown("**1. 계수 조절 (a, b, c)**")
            a_val = st.slider("이차항 계수 (a)", -5, 5, 1, 1, key="slider_a")
            b_val = st.slider("일차항 계수 (b)", -10, 10, -4, 1, key="slider_b")
            c_val = st.slider("상수항 (c)", -10, 10, 3, 1, key="slider_c")

            if a_val == 0:
                st.warning("a=0 이면 이차함수가 아닙니다. a값을 조절해주세요!")
                a_val = 1
            
            b_op = "+" if b_val >= 0 else "-"
            c_op = "+" if c_val >= 0 else "-"
            
            st.markdown("**현재 함수식:**")
            st.latex(f"y = {a_val}x^2 {b_op} {abs(b_val)}x {c_op} {abs(c_val)}")
            
        with col_viz2:
            st.markdown("**2. 수학적 설명 및 그래프**")
            # 💡 동시 접속 방어를 위해 Figure로 분리
            fig_viz = Figure(figsize=(5, 3))
            ax_viz = fig_viz.subplots()
            x_viz = np.linspace(-10, 10, 400)
            y_viz = a_val * x_viz**2 + b_val * x_viz + c_val
            ax_viz.plot(x_viz, y_viz, color='#1976d2', linewidth=2)
            ax_viz.axhline(0, color='#d32f2f', linewidth=2, label='x축')
            ax_viz.grid(True, linestyle='--', alpha=0.6)
            ax_viz.set_ylim(-15, 20)
            ax_viz.set_xlim(-8, 8)
            st.pyplot(fig_viz)
            
            D_val = b_val**2 - 4*a_val*c_val
            st.markdown(f"**3. 판별식 계산:** $D = ({b_val})^2 - 4({a_val})({c_val}) = {D_val}$")

            if D_val > 0:
                st.error("🔹 **서로 다른 두 점에서 만남 ($D > 0$)** 이차함수 그래프가 x축과 두 번 교차합니다. (실근 2개)")
            elif D_val == 0:
                st.warning("🔹 **한 점에서 만남 ($D = 0$)** 이차함수 그래프가 x축에 아슬아슬하게 접합니다. (중근 1개)")
            else:
                st.success("🔹 **만나지 않음 ($D < 0$)** 이차함수 그래프가 x축과 전혀 닿지 않습니다. (실근 0개)")

        st.divider()

        st.markdown(pretty_title("❔ 근과 x축의 위치 관계", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write("이차방정식 $ax^2 + bx + c = 0$ 의 실근은, 시각적으로 볼 때 **이차함수 $y = ax^2 + bx + c$ 의 그래프가 x축($y=0$)과 만나는 교점의 x좌표**를 의미합니다.")
        st.info("""💡 **수식과 그래프의 연결**
* **$D > 0$ (근 2개)** $\\rightarrow$ 그래프가 x축을 관통하며 **두 점**에서 만납니다.
* **$D = 0$ (근 1개)** $\\rightarrow$ 그래프가 x축에 살짝 **접합니다**.
* **$D < 0$ (근 0개)** $\\rightarrow$ 그래프가 x축과 **만나지 않고 붕 떠 있거나 가라앉아** 있습니다.

컴퓨터는 판별식의 부호 하나만으로 시뮬레이션 환경(예: 드론과 지면)에서 물리적 충돌 여부를 정확히 예측할 수 있습니다.""")
        
        st.info("💡 **특별 기능 안내:** 코드 창 마지막에 `draw_graph(a, b, c)` 함수를 적고 실행하면, 입력한 값에 맞춰 컴퓨터가 알아서 이차함수 그래프를 그려줍니다!")

        st.markdown("""###### 💻 [예제 2] 수학 문제: 이차함수와 직선의 위치 관계""")
        st.write("이차함수 $y = x^2 - 6x + 5$ 와 x축의 위치 관계를 판별식을 통해 판단하고 그래프로 그리는 코드입니다. (아래 코드를 실행해보세요)")
        
        st.code("""a = 1
b = -6
c = 5
D = b**2 - 4*a*c
print("판별식 D의 값:", D)

if D > 0:
    print("결과: 서로 다른 두 점에서 만납니다.")
elif D == 0:
    print("결과: 한 점에서 만납니다(접한다).")
else:
    print("결과: 만나지 않습니다.")

# 그래프 출력 함수
draw_graph(a, b, c)""", language="python")

        starter_ex2 = """a = 1
b = -6
c = 5
D = b**2 - 4*a*c
print("판별식 D의 값:", D)

if D > 0:
    print("결과: 서로 다른 두 점에서 만납니다.")
elif D == 0:
    print("결과: 한 점에서 만납니다(접한다).")
else:
    print("결과: 만나지 않습니다.")

draw_graph(a, b, c)
"""
        code_block("ex2", "수학적 위치 관계 (예제 2)", starter_ex2, prefix="d2_", height=350)

        st.divider()

        st.markdown(pretty_title("💻 문제 2. 드론 비행 시뮬레이터 구축", "#fce4ec", "#f8bbd0"), unsafe_allow_html=True)
        st.write("당신의 드론은 협곡으로 하강했다가 다시 올라오는 $y = x^2 - 4x + 5$ 의 궤적을 그립니다. 이 드론은 지면(x축)에 충돌할까요?")
        
        with st.expander("💡 힌트 보기"):
            st.markdown("`a=1, b=-4, c=5` 로 설정하고 판별식 D를 `b**2 - 4*a*c` 로 계산하세요. 마지막에 `draw_graph(a, b, c)`를 적어주세요.")
        with st.expander("💡 정답 보기"):
            st.code("""a = 1\nb = -4\nc = 5\nD = b**2 - 4*a*c\n\nif D > 0:\n    print("🚨 추락 경고! 지면에 충돌합니다!")\nelif D == 0:\n    print("⚠️ 주의! 지면에 스칩니다!")\nelse:\n    print("✅ 안전! 지면에 닿지 않습니다.")\n\ndraw_graph(a, b, c)""", language="python")

        starter_q3 = """# 1. 궤적 계수 입력
a = 
b = 
c = 

# 2. 판별식 D 계산
D = 

# 3. 상황 판별 (조건문)
if : 
    print("🚨 추락 경고! 지면에 충돌합니다!")
elif :
    print("⚠️ 주의! 지면에 스칩니다!")
else:
    print("✅ 안전! 지면에 닿지 않습니다.")

# 4. 그래프 출력
draw_graph(a, b, c)
"""
        code_block("q3", "드론 충돌 시뮬레이터 (문제 2)", starter_q3, prefix="d2_", height=380)

    # ------------------------------------------
    # 탭 4: 적용 및 비판적 성찰 [응용적 수학화 1]
    # ------------------------------------------
    with tabs[3]:
        stage_intro(
            "계산 결과를 비판적으로 해석하기",
            "계산 결과가 실제 상황에서도 타당한지 검토하고, 알고리즘을 확장해 더 정확한 해석으로 나아가는 단계입니다.",
            "같은 수식 결과라도 현실에서는 무엇을 더 살펴봐야 할까?",
            "#f3e5f5",
            "#e1bee7",
        )
        
        st.markdown(pretty_title("💻 문제 3. 수준별 종합 도전", "#f3e5f5", "#e1bee7"), unsafe_allow_html=True)
        st.write("배운 조건문(`if-elif-else`)을 활용하여 자신만의 시뮬레이터를 확장해 봅시다.")
        
        level = st.radio("자신의 실력에 맞는 난이도를 선택하세요:", 
                         ("🌱 하 (판별식 기초)", "🌿 중 (시뮬레이터 함수 만들기)", "🌳 상 (근의 공식 결합)"), 
                         horizontal=True)
        
        if "하" in level:
            st.info("**[기초]** 포물선 운동을 하는 물체의 궤적이 $y = -x^2 + 4x - 4$ 일 때, 이 물체가 지면에 닿는지 판별하고 그래프를 그리세요.")
            st_code = """a = -1
b = 4
c = -4

D =  # 👈 판별식 연산을 작성하세요

if D > 0:
    print("지면과 2번 만납니다.")
# 👈 여기에 elif 와 else 문을 마저 완성하세요


# 그래프 출력
draw_graph(a, b, c)
"""
        elif "중" in level:
            st.info("**[응용]** 계수 `a, b, c`를 넣으면 안전 여부를 `return`하는 함수 `def check_drone(a, b, c):` 를 만들어보세요.")
            st_code = """def check_drone(a, b, c):
    D = b**2 - 4*a*c
    if D > 0:
        return "추락 (교점 2개)"
    elif D == 0:
        return "스침 (접함)"
    else:
        return "안전 비행"

# 함수 호출하여 결과 출력하기
result = check_drone(1, -2, 3)
print("비행 상태:", result)

# 그래프로 증명하기
draw_graph(1, -2, 3)
"""
        else:
            st.info("**[심화 종합]** 충돌한다면(D>0) 근의 공식을 이용해 실제 추락하는 x축 좌표(근)까지 계산해서 출력하고 시각화하는 완벽한 시스템을 구축하세요. \n*(힌트: 거듭제곱 연산자 `**0.5` 를 사용하면 루트(√)를 계산할 수 있습니다.)*")
            st_code = """a = -1
b = 6
c = -5

D = b**2 - 4*a*c

if D > 0:
    print("추락 경고! 충돌 좌표 계산 중...")
    root1 = (-b + D**0.5) / (2*a)
    root2 = (-b - D**0.5) / (2*a)
    print("충돌 예상 x좌표는:", root1, "그리고", root2)
elif D == 0:
    print("스침! 접점 계산 중...")
    # 👈 접점을 구하는 식을 작성해보세요
else:
    print("안전 비행 (실근 없음)")

draw_graph(a, b, c)
"""
        if st.session_state.get("d2_q4_level") != level:
            st.session_state["d2_q4_level"] = level
            if "d2_q4_editor" in st.session_state:
                del st.session_state["d2_q4_editor"] 
            if "d2_q4_result" in st.session_state:
                del st.session_state["d2_q4_result"]

        if "하" in level:
            level_key = "ha"
        elif "중" in level:
            level_key = "jung"
        else:
            level_key = "sang"
            
        code_block("q4", f"도전 코드 ({level[:2]})", st_code, prefix=f"d2_{level_key}_", height=320)

    # ------------------------------------------
    # 탭 5: 세상 연결 [응용적 수학화 2]
    # ------------------------------------------
    # ------------------------------------------
    # 탭 5: 세상과 연결 및 실천 [응용적 수학화 2]
    # ------------------------------------------
    with tabs[4]:
        stage_intro(
            "배움을 사회와 연결해 보기",
            "오늘 만든 판단 규칙과 시뮬레이션 결과를 실제 사회 문제와 연결해 보고, 우리의 생각을 함께 나누는 단계입니다.",
            "수학과 알고리즘으로 만든 기준을 우리는 언제 믿고 언제 의심해야 할까?",
            "#fff3e0",
            "#ffe0b2",
        )
        
        # ------------------------------------------
        # 1. 학생 정보 입력 및 포트폴리오 PDF 생성
        # ------------------------------------------
        st.markdown(pretty_title("💾 1. 학생 정보 입력 및 포트폴리오 저장", "#f1f8e9", "#dcedc8"), unsafe_allow_html=True)
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            group_name = st.text_input("모둠 이름 (예: 1모둠)")
        with col_info2:
            stu_id = st.text_input("학번 (예: 10101)", max_chars=5)
        with col_info3:
            stu_name = st.text_input("이름 (예: 홍길동)")
        render_value_cards(
            [
                {
                    "title": "모둠 이름",
                    "value": group_name if group_name else "입력 대기",
                    "detail": "포트폴리오와 패들렛 공유에 함께 표시됩니다.",
                    "bg": "#f4f9ff",
                    "border": "#90caf9",
                },
                {
                    "title": "학번",
                    "value": stu_id if stu_id else "입력 대기",
                    "detail": "반 패들렛 연결과 PDF 파일명에 반영됩니다.",
                    "bg": "#f1f8e9",
                    "border": "#aed581",
                },
                {
                    "title": "이름",
                    "value": stu_name if stu_name else "입력 대기",
                    "detail": "나의 실습 결과를 정리한 포트폴리오에 표시됩니다.",
                    "bg": "#fff8e1",
                    "border": "#ffcc80",
                },
            ],
            columns=3,
        )

        if group_name and stu_id and stu_name:
            if len(stu_id) >= 3:
                class_num = stu_id[2]
                valid_classes = ["1", "2", "5", "6"]
                
                if class_num in valid_classes:
                    st.success("✅ 학습 포트폴리오가 완성되었습니다! 코드를 PDF로 저장하고 아래 패들렛에 업로드해주세요.")
                    
                    if "하" in level:
                        level_key = "ha"
                    elif "중" in level:
                        level_key = "jung"
                    else:
                        level_key = "sang"
                        
                    code_sections = [
                        ("예제 1. 기온/날씨 판별기", "d2_ex1"),
                        ("문제 1. 양수/0/음수 판별기", "d2_q1"),
                        ("예제 2. 수학적 위치 관계", "d2_ex2"),
                        ("문제 2. 드론 충돌 시뮬레이터", "d2_q3"),
                        ("문제 3. 수준별 종합 도전", f"d2_{level_key}_q4")
                    ]
                    
                    code_data = []
                    for title, prefix in code_sections:
                        code_text = st.session_state.get(f"{prefix}_editor", "")
                        res_tuple = st.session_state.get(f"{prefix}_result", ("", "", None))
                        result_text = res_tuple[0] if res_tuple else ""
                        code_data.append((title, code_text, result_text))
                    
                    student_info = {"group": group_name, "id": stu_id, "name": stu_name}
                    
                    # 주의: PDF 생성 함수가 순수 코드만 받도록 수정된 버전이어야 합니다.
                    pdf_bytes = create_portfolio_pdf(student_info, code_data)
                    
                    portfolio_urls = {
                        "1": "https://padlet.com/ps0andd/p_1",
                        "2": "https://padlet.com/ps0andd/p_2",
                        "5": "https://padlet.com/ps0andd/p_5",
                        "6": "https://padlet.com/ps0andd/p_6",
                    }
                    padlet_portfolio_url = portfolio_urls.get(class_num, "https://padlet.com/")

                    # 버튼 2개를 나란히 배치 (다운로드 & 패들렛 링크)
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button(
                            label="📥 내 코딩 실습 결과 PDF 다운로드",
                            data=pdf_bytes,
                            file_name=f"{stu_id}_{stu_name}_2차시_코드포트폴리오.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.warning("⚠️ 모둠원들이 동시에 PDF 다운로드 버튼을 누르면 오류가 날 수 있습니다. 한 명씩 차례대로 눌러 주세요.")
                    with col_btn2:
                        st.markdown(
                            f"""<a href="{padlet_portfolio_url}" target="_blank" 
                               style="display: block; padding: 10px; background: linear-gradient(90deg, #43a047 0%, #66bb6a 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                               📂 {class_num}반 포트폴리오 패들렛 열기
                            </a>""", unsafe_allow_html=True
                        )
                else:
                    st.error("❌ **오류:** 담당 학급(1, 2, 5, 6반)의 학번이 아닙니다. 학번을 다시 확인해 주세요.")
            else:
                st.warning("⚠️ 올바른 5자리 학번(예: 10101)을 입력해 주세요.")
        else:
            st.warning("⚠️ 학생 정보(모둠, 학번, 이름)를 입력해야 코드를 저장할 수 있습니다.")

        st.markdown("---")
        
        # ------------------------------------------
        # 2. 교사의 심화 질문 답변
        # ------------------------------------------
        st.markdown(pretty_title("💡 2. 교사의 심화 질문 답변", "#e3f2fd", "#bbdefb"), unsafe_allow_html=True)
        st.info("🔥 **교사의 심화 질문(Deep Question):**\n\n컴퓨터(AI)가 그래프를 그려주고 교점도 다 찾아주는 시대입니다. 그렇다면 우리는 왜 굳이 이차함수와 x축의 위치 관계(판별식)를 수학적으로 이해하고, 알고리즘(if-elif)으로 설계하는 방법을 배워야 할까요?")
        teacher_ans = st.text_area("위 질문에 대한 우리의 답을 논리적으로 작성해 보세요.", height=100, key="d2_teacher_ans")

        st.markdown("---")

        # ------------------------------------------
        # 3. 모둠 질문 만들기 및 패들렛 공유
        # ------------------------------------------
        st.markdown(pretty_title("💬 3. 모둠 심화 질문 만들기", "#ede7f6", "#d1c4e9"), unsafe_allow_html=True)
        st.write("모둠원과 함께 오늘 활동을 돌아보며 심화 질문을 하나 만들고, 앞서 작성한 답변과 함께 패들렛에 공유해 봅시다.")
        
        q_deep = st.text_area("🔥 **[우리의 딥(Deep) 퀘스천]** *(윤리와 철학)* 배운 지식이나 기술이 실제 사회에 적용될 때 발생할 수 있는 부작용이나 윤리적 딜레마를 다루며, 정답 없이 서로의 가치관을 깊이 있게 나눌 수 있는 토론형 질문입니다. 👉 :blue[예)만약 인공지능이 복잡한 조건문만으로 회사 면접의 합격자를 결정한다면, 우리는 그 알고리즘의 기준이 인간보다 공정하다고 믿을 수 있을까?]", height=100)

        if group_name and stu_id and teacher_ans and q_deep:
            if len(stu_id) >= 3 and stu_id[2] in ["1", "2", "5", "6"]:
                class_num = stu_id[2]
                st.success("✅ 성찰 질문 작성이 완료되었습니다! 텍스트를 복사하여 패들렛에 업로드하세요.")
                
                report_text = f"""[F.U.T.U.R.E. 프로젝트 2DAY 성찰 일지] (👉 게시물 제목)
모둠명: {group_name}

🔥 [우리가 만든 딥(Deep) 퀘스천] 
{q_deep}

💡 [교사의 심화 질문에 대한 우리의 생각]
{teacher_ans}
"""
                st.code(report_text, language="markdown")

                qa_urls = {
                    "1": "https://padlet.com/ps0andd/q_1", # 1반 주소
                    "2": "https://padlet.com/ps0andd/q_2", # 2반 주소
                    "5": "https://padlet.com/ps0andd/q_5", # 5반 주소
                    "6": "https://padlet.com/ps0andd/q_6", # 6반 주소
                }
                padlet_qa_url = qa_urls.get(class_num, "https://padlet.com/")

                st.info(f"💾 **[미션 2]** 복사한 성찰 일지를 아래 '{class_num}반 질문(Q&A) 패들렛'에 업로드하고, 친구 글에 댓글을 달아주세요!")
                st.markdown(
                    f"""<a href="{padlet_qa_url}" target="_blank" 
                        style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 5px;">
                        🚀 {class_num}반 질문(Q&A) 패들렛으로 이동하기
                    </a>""", unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ 교사의 심화 질문에 대한 답변과 모둠 심화 질문을 모두 작성해야 패들렛 공유 양식이 나타납니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True) 

if __name__ == "__main__":
    run()
