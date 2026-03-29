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
from matplotlib.figure import Figure
from fpdf import FPDF
import math
import itertools
import time

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
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 3차시 학습 포트폴리오", ln=1, align='C')
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

def create_portfolio_pdf(student_info, teacher_ans, code_data):
    pdf = ThemedPDF()
    try:
        pdf.add_font('Nanum', '', font_path, uni=True)
    except:
        pass 
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
    
    pdf.h2("🤔 교사의 딥(Deep) 퀘스천")
    pdf.set_font(pdf._font_family, '', 11)
    pdf.set_text_color(211, 47, 47) 
    pdf.multi_cell(0, 6, "Q. 비밀번호를 무작위로 대입해서 푸는 방식(브루트 포스)은 경우의 수가 많아지면 인간과 컴퓨터 모두에게 물리적으로 불가능에 가까워집니다. 그렇다면 알파고와 같은 AI는 우주 원자 수보다 많은 바둑의 경우의 수를 어떻게 전부 계산하지 않고도 최적의 수를 찾아낼 수 있었을까요?")
    pdf.ln(3)
    
    pdf.set_text_color(21, 101, 192) 
    pdf.cell(0, 8, "▶ 나의 답변", ln=1)
    pdf.set_text_color(50, 50, 50) 
    pdf.p(teacher_ans if teacher_ans else "작성된 내용이 없습니다.")
    
    pdf.add_page()
    
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

# ==========================================
# 2. 파이썬 코드 실행 엔진 (과부하 방지 4중 안전장치 적용)
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

    # 📈 방어막 1: 그래프 크기 제한 (브라우저 마비 방지)
    def draw_growth_graph(max_n):
        nonlocal fig
        if max_n > 50:
            custom_print(f"⚠️ [경고] n={max_n}은 너무 큽니다! 서버 보호를 위해 그래프는 n=50까지만 그려집니다.")
            max_n = 50
            
        fig = Figure(figsize=(6, 4))
        ax = fig.subplots()
        x = np.arange(1, max_n + 1)
        y = [math.factorial(int(i)) for i in x]
        
        ax.plot(x, y, marker='o', color='#d32f2f', linewidth=2, label='n! (팩토리얼)')
        ax.set_title("경우의 수 폭발 (Combinatorial Explosion)")
        ax.set_xlabel("데이터 개수 (n)")
        ax.set_ylabel("경우의 수 (Log scale)")
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # 🛡️ 방어막 2: math 라이브러리 제한 (초거대 숫자 연산 방지)
    class SafeMath:
        pass
    for k, v in math.__dict__.items():
        setattr(SafeMath, k, v)
        
    def safe_factorial(n):
        if n > 2000:
            raise ValueError(f"💥 서버 과부하 방지: {n}!은 너무 큰 숫자입니다. (최대 2000까지만 입력 가능)")
        return math.factorial(n)
    SafeMath.factorial = safe_factorial

    # 🛡️ 방어막 3: time 라이브러리 제한 (장난으로 인한 앱 멈춤 방지)
    class SafeTime:
        time = time.time
        @staticmethod
        def sleep(secs):
            if secs > 3:
                raise ValueError(f"💥 서버 대기 방지: time.sleep({secs})은 허용되지 않습니다. (최대 3초까지만 가능)")
            time.sleep(secs)

    # 🛡️ 방어막 4: itertools 제한 (RAM 메모리 폭발 방지)
    class SafeItertools:
        pass
    for k, v in itertools.__dict__.items():
        setattr(SafeItertools, k, v)
        
    def safe_permutations(iterable, r=None):
        lst = list(iterable)
        n = len(lst)
        r_val = n if r is None else r
        if math.perm(n, r_val) > 1000000:
            raise ValueError("💥 메모리 보호: 생성되는 경우의 수가 100만 개를 초과하여 실행을 차단합니다.")
        return itertools.permutations(lst, r)
        
    def safe_combinations(iterable, r):
        lst = list(iterable)
        if math.comb(len(lst), r) > 1000000:
            raise ValueError("💥 메모리 보호: 생성되는 조합의 수가 100만 개를 초과하여 실행을 차단합니다.")
        return itertools.combinations(lst, r)
        
    SafeItertools.permutations = safe_permutations
    SafeItertools.combinations = safe_combinations

    # ⏳ 최후의 보루: 전체 실행 시간 제한 (while 무한 루프 강제 종료)
    start_time_exec = time.time()
    def trace_calls(frame, event, arg):
        if time.time() - start_time_exec > 2.0: # 2초 초과 시 에러 발생
            raise TimeoutError("💥 실행 시간 초과! (무한 루프나 너무 긴 연산을 방지하기 위해 2초 만에 강제 종료되었습니다.)")
        return trace_calls

    safe_builtins = builtins.__dict__.copy()
    safe_builtins['print'] = custom_print
    
    exec_globals = {
        '__builtins__': safe_builtins,
        'math': SafeMath,
        'itertools': SafeItertools,
        'time': SafeTime,
        'draw_growth_graph': draw_growth_graph 
    }
    
    sys.settrace(trace_calls) # 🔍 코드 한 줄 한 줄 감시 시작
    try:
        exec(code_input, exec_globals)
        result = output_buffer.getvalue() or "출력된 내용이 없습니다."
    except Exception as e:
        result = f"{e.__class__.__name__}: {e}"
        status = "error"
    finally:
        sys.settrace(None) # 감시 종료
        
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
        st.markdown(f"##### 📥 {title} (코드 입력)")
        code_input = st_ace(value=starter_code, language='python', theme='github', height=height, key=f"{key_prefix}_editor")
        if st.button("▶️ 실행", key=f"{key_prefix}_run"):
            st.session_state[f"{key_prefix}_result"] = code_runner(code_input)
    with c2:
        st.markdown("##### 🖥️ 실행 결과")
        if f"{key_prefix}_result" in st.session_state:
            res, stat, fig = st.session_state[f"{key_prefix}_result"]
            display_output(res, stat, fig)
        else:
            st.info("실행 버튼을 누르면 결과가 표시됩니다.")

# ==========================================
# 3. 메인 앱 화면 (UI)
# ==========================================
def run():
    st.header("3DAY - 🔢 경우의 수 폭발과 AI 탐색")
    st.markdown("**🎯 학습 목표:** 경우의 수(순열과 조합)를 파이썬으로 계산해보고, 데이터가 늘어날 때 계산량이 기하급수적으로 폭발하는 현상을 통해 AI의 효율적인 탐색(Search) 원리를 이해합니다.")
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
        st.success("**[문제 인식 및 숨겨진 데이터 찾기]** 우리 주변의 무수한 선택지와 경우의 수를 발견하고, 모든 것을 다 계산하는 방식의 한계를 탐색합니다.")
        st.markdown("---")
        
        st.markdown("#### 📌 [문제 제기] 🔓 스마트폰 비밀번호와 우주 배달 로봇")
        st.write("스마트폰의 4자리 비밀번호를 잊어버렸을 때, 0000부터 9999까지 하나씩 다 눌러보는 방식을 **브루트 포스(Brute-Force, 무차별 대입)**라고 합니다.")
        
        st.info("""💡 **생각해 볼 문제:**
1. **비밀번호:** 4자리 숫자를 모두 눌러보려면 최대 몇 번을 시도해야 할까요? 만약 영문자와 특수문자까지 섞어 10자리라면 어떨까요?
2. **배달 로봇 (외판원 순회 문제):** 10개의 집을 방문해 택배를 배달해야 하는 로봇이 있습니다. 어떤 순서로 방문해야 가장 짧은 거리를 이동할 수 있을까요?""")
        
        st.write("방문해야 할 집이 10곳이라면, 로봇이 고려해야 할 경로의 순서(경우의 수)는 자그마치 **362만 8,800가지($10!$)**입니다. 집이 20곳으로 늘어나면 컴퓨터조차 우주의 나이보다 긴 시간이 필요해집니다.")
        
        hypothesis = st.text_input("💡 나의 가설 세워보기 (알파고 같은 AI는 이 엄청난 경우의 수를 어떻게 다 계산하고 정답을 찾을까?)", placeholder="나의 생각을 자유롭게 적어보세요.", key="d3_hypothesis")
        if st.button("🔍 가설 확인", key="d3_btn_hypo"):
            if hypothesis.strip() == "":
                st.warning("틀려도 괜찮아요! 먼저 자신의 생각을 적어주세요.")
            else:
                st.success(f"**✅ 흥미로운 접근입니다!** '{hypothesis}'라고 생각했군요.")
                st.divider()
                st.markdown("""
                #### 🔓 알고리즘의 핵심: '다 해보기'의 한계와 가지치기(Pruning)
                * 인간은 무수한 경우의 수 앞에서 직관을 사용해 **'안 될 것 같은 길'**은 애초에 생각하지 않습니다.
                * AI 역시 모든 것을 다 계산하지 않습니다. **가지치기(Pruning)**나 **휴리스틱(Heuristic)** 기법을 사용하여, 정답이 아닐 확률이 높은 경우의 수는 아예 탐색 과정에서 잘라내 버립니다.
                """)

    # ------------------------------------------
    # 탭 2: 수학적 구조화 [수평적 수학화]
    # ------------------------------------------
    with tabs[1]:
        st.success("**[현상을 수학의 언어로 바꾸기]** 파이썬의 `math`와 `itertools` 라이브러리를 활용하여 순열, 조합, 팩토리얼을 코드로 계산해 봅니다.")
        st.markdown("---")

        st.markdown("#### 📌 [코딩] 경우의 수를 구하는 파이썬 도구들")
        st.write("파이썬은 수학 계산에 특화된 유용한 라이브러리를 기본 제공합니다.")
        st.code("""
import math
import itertools

# 1. 팩토리얼 (n!): math.factorial(n)
# 2. 순열 (nPr): itertools.permutations(리스트, r)
# 3. 조합 (nCr): itertools.combinations(리스트, r)
        """)
        
        st.markdown("""###### 💻 [예제 1] 팩토리얼과 순열 계산기""")
        st.write("A, B, C 세 명의 학생을 한 줄로 세우는 경우의 수($3!$)와, 5명 중 2명을 뽑아 반장/부반장으로 앉히는 경우의 수($_{5}P_{2}$)를 구해봅시다.")
        
        starter_ex1 = """import math
import itertools

# 1. 3명을 한 줄로 세우는 경우의 수 (3!)
fact_3 = math.factorial(3)
print("3! 의 값:", fact_3)

# 2. 5명 중 2명을 뽑아 순서대로 나열 (순열)
students = ['A', 'B', 'C', 'D', 'E']
perm = list(itertools.permutations(students, 2))

print(f"5명 중 2명을 뽑는 경우의 수: {len(perm)}가지")
print("모든 경우:", perm)
"""
        code_block("ex1", "순열과 팩토리얼 연습장", starter_ex1, prefix="d3_", height=300)
        
        st.divider()

        st.markdown("""###### 💻 [문제 1] 아이스크림 조합(Combination) 구하기""")
        st.write("베스킨라빈스에 31가지 맛이 있습니다. 이 중 **순서에 상관없이 3가지 맛**을 고르는(파인트) 경우의 수는 총 몇 가지일까요? (조합 $_{31}C_{3}$)")
        with st.expander("💡 힌트 보기"):
            st.markdown("1부터 31까지의 숫자를 리스트로 만들거나, 단순히 식을 계산할 수도 있습니다. `itertools.combinations(range(1, 32), 3)`를 사용한 후 `len()`으로 길이를 구해보세요.")
        with st.expander("💡 정답 보기"):
            st.code("""import itertools\n\n# 1~31의 숫자로 이루어진 범위에서 3개 고르기\nice_cream = range(1, 32)\ncomb = list(itertools.combinations(ice_cream, 3))\n\nprint("31가지 중 3가지를 고르는 경우의 수:", len(comb))""", language="python")

        starter_q1 = """import itertools

ice_cream = range(1, 32)
# 아래에 itertools.combinations를 사용하여 조합을 구하세요.
comb = 

print("31가지 중 3가지를 고르는 경우의 수:", len(comb))
"""
        code_block("q1", "조합 구하기 (문제 1)", starter_q1, prefix="d3_", height=210)
        

    # ------------------------------------------
    # 탭 3: 컴퓨팅 도구 활용 [수직적 수학화]
    # ------------------------------------------
    with tabs[2]:
        st.success("**[AI 도구로 시뮬레이션하기]** 데이터가 늘어날 때 컴퓨터가 연산하는 데 걸리는 시간(Time)을 측정하고, 경우의 수 폭발을 시각적으로 확인합니다.")
        st.markdown("---")

        st.markdown("#### 📌 [도구적 질문] 컴퓨터는 무조건 빠를까?")
        st.write("파이썬의 `time` 모듈을 사용하면 코드 실행에 걸린 시간을 측정할 수 있습니다. 10개의 도시를 방문하는 최단 경로를 브루트 포스(모두 탐색)로 찾으려면 시간이 얼마나 걸릴까요?")

        st.info("💡 **특별 기능 안내:** 코드 창 마지막에 `draw_growth_graph(n)` 함수를 적으면, 1부터 n까지의 팩토리얼 증가량을 보여주는 지수 그래프를 그려줍니다!")

        st.markdown("""###### 💻 [예제 2] 연산 시간 측정하기 및 그래프 출력""")
        st.write("도시의 개수(n)가 10개일 때, 모든 경로의 수($10!$)를 계산하는 데 걸리는 시간과 경우의 수 증가 그래프를 확인해 봅시다.")
        
        starter_ex2 = """import time
import math

n = 10 # 도시의 개수

# 1. 시작 시간 측정
start_time = time.time()

# 2. 10! 계산 (모든 경로 탐색을 가정)
total_cases = math.factorial(n)

# 3. 종료 시간 측정
end_time = time.time()

print(f"{n}개 도시 방문 경로 수: {total_cases}가지")
print(f"계산에 걸린 시간: {end_time - start_time:.6f} 초")

# 경우의 수 폭발 그래프 그리기 (n=10)
draw_growth_graph(n)
"""
        code_block("ex2", "연산 시간과 폭발 그래프 (예제 2)", starter_ex2, prefix="d3_", height=350)

        st.divider()

        st.markdown("""###### 💻 [문제 2] 브루트 포스 한계 체험하기""")
        st.write("위 예제에서 도시의 개수 `n`을 **10에서 15, 그리고 20으로** 늘려보세요. 20개일 때 경우의 수는 몇 자리 숫자가 되나요? 컴퓨터가 순식간에 계산할 수 있는 범위를 넘어서는지 확인해 봅시다.")
        
        starter_q3 = """import math

# 도시의 개수 n을 15, 20으로 변경하며 실행해보세요.
n = 20

total_cases = math.factorial(n)
print(f"{n}개 도시 방문 경로 수: {total_cases}가지")

# 숫자가 너무 길어서 길이를 재봅시다.
digit_length = len(str(total_cases))
print(f"이 숫자는 무려 {digit_length}자리 숫자입니다!")

# 그래프도 n에 맞게 그려보세요.
draw_growth_graph(n)
"""
        code_block("q3", "한계 돌파 시뮬레이션 (문제 2)", starter_q3, prefix="d3_", height=280)

    # ------------------------------------------
    # 탭 4: 적용 및 비판적 성찰 [응용적 수학화 1]
    # ------------------------------------------
    with tabs[3]:
        st.success("**[결과의 의미와 한계 고민하기]** 모든 것을 다 탐색하는 방식의 한계를 깨달았다면, 어떻게 '똑똑하게' 정답을 찾을 수 있을지 나만의 알고리즘을 구상해 보는 단계입니다.")
        st.markdown("---")
        
        st.markdown("#### 📌 [문제 3] 수준별 종합 도전")
        st.write("배운 내용을 바탕으로 탐색 알고리즘을 만들어 봅시다.")
        
        level = st.radio("자신의 실력에 맞는 난이도를 선택하세요:", 
                         ("🌱 하 (비밀번호 맞추기)", "🌿 중 (알파벳 암호 해독)", "🌳 상 (탐욕(Greedy) 알고리즘 기초)"), 
                         horizontal=True, key="d3_radio")
        
        if "하" in level:
            st.info("**[기초]** 0000부터 9999까지 4자리 비밀번호 중, 정답 비밀번호 '7777'을 찾을 때까지 몇 번 시도해야 하는지 반복문(`for`)으로 찾아보세요.")
            st.write("*(힌트: `range(10000)`을 사용하세요.)*")
            st_code = """target_pw = 7777
attempts = 0

for i in range(10000):
    attempts += 1
    if i == target_pw:
        print(f"찾았습니다! 비밀번호는 {i}입니다.")
        print(f"총 {attempts}번 시도했습니다.")
        break  # 정답을 찾으면 반복문을 탈출!
"""
        elif "중" in level:
            st.info("**[응용]** 영문 소문자 a, b, c 세 개로 이루어진 3자리 암호(예: abc, cba 등)를 `itertools.product` (중복 순열)를 이용해 모두 생성하고, 그 중 'cab'를 찾는 코드를 작성하세요.")
            st_code = """import itertools

chars = ['a', 'b', 'c']
# chars에서 3개를 뽑아 중복 순열 생성
passwords = list(itertools.product(chars, repeat=3))
print("생성된 총 암호 개수:", len(passwords))

target = ('c', 'a', 'b')

for count, pw in enumerate(passwords, 1):
    if pw == target:
        print(f"{count}번째 시도만에 암호 {pw}를 찾았습니다!")
        break
"""
        else:
            st.info("**[심화 종합]** 10개의 도시를 모두 탐색($10!$)하는 대신, '현재 위치에서 가장 가까운 도시를 다음 목적지로 선택'하는 휴리스틱 기법인 **탐욕(Greedy) 알고리즘**의 개념을 코드로 간단히 구현해 보세요.")
            st_code = """# A에서 출발하여 모든 도시를 방문해야 합니다.
# (모든 경로를 다 구하지 않고, 가장 거리가 짧은 곳만 쫓아가는 방식)
cities = ['A', 'B', 'C', 'D']
# 현재 도시에서 다른 도시까지의 거리 (임의 설정)
distances = {
    'A': {'B': 10, 'C': 15, 'D': 20},
    'B': {'A': 10, 'C': 35, 'D': 25},
    'C': {'A': 15, 'B': 35, 'D': 30},
    'D': {'A': 20, 'B': 25, 'C': 30}
}

current = 'A'
visited = [current]

# 아직 3개의 도시가 남았습니다.
for _ in range(3):
    next_city = None
    min_dist = 999
    
    # 현재 도시와 연결된 길 중에서 안 가본 곳 중 가장 짧은 길 선택
    for neighbor, dist in distances[current].items():
        if neighbor not in visited and dist < min_dist:
            min_dist = dist
            next_city = neighbor
            
    visited.append(next_city)
    current = next_city

print("AI가 찾은 탐욕적 경로:", visited)
print("이 방식은 계산은 매우 빠르지만, 무조건 '최적의 해'를 보장하지는 않는 한계가 있습니다.")
"""
        if st.session_state.get("d3_q4_level") != level:
            st.session_state["d3_q4_level"] = level
            if "d3_q4_editor" in st.session_state:
                del st.session_state["d3_q4_editor"] 
            if "d3_q4_result" in st.session_state:
                del st.session_state["d3_q4_result"]

        if "하" in level:
            level_key = "ha"
        elif "중" in level:
            level_key = "jung"
        else:
            level_key = "sang"
            
        code_block("q4", f"도전 코드 ({level[:2]})", st_code, prefix=f"d3_{level_key}_", height=320)

    # ------------------------------------------
    # 탭 5: 세상 연결 [응용적 수학화 2]
    # ------------------------------------------
    with tabs[4]:
        st.success("**[우리의 삶과 사회로 연결하기]** 방대한 경우의 수와 AI의 탐색 한계를 고려할 때, 우리는 AI 기술을 사회에 어떻게 적용하고 활용해야 할지 고민해 봅니다.")
        st.markdown("---")
        
        st.markdown("#### 📌 1. 학생 정보 입력")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            group_name = st.text_input("모둠 이름 (예: 1모둠)", key="d3_group")
        with col_info2:
            stu_id = st.text_input("학번 (예: 10101)", max_chars=5, key="d3_id")
        with col_info3:
            stu_name = st.text_input("이름 (예: 홍길동)", key="d3_name")

        st.markdown("---")

        st.markdown("#### 📌 2. 나의 생각 쓰기 및 코드 포트폴리오 저장")
        st.info("🔥 **교사의 심화 질문(Deep Question):**\n\n비밀번호를 무작위로 대입해서 푸는 방식(브루트 포스)은 경우의 수가 많아지면 물리적으로 불가능해집니다. 그렇다면 알파고와 같은 AI는 우주 원자 수보다 많은 바둑의 경우의 수를 어떻게 전부 계산하지 않고도 이길 수 있었을까요? 오늘 배운 '경우의 수 폭발'과 연관 지어 생각해보세요.")
        
        teacher_ans = st.text_area("위 질문에 대한 나만의 답을 논리적으로 작성해 보세요.", height=100, key="d3_teacher_ans")

        if group_name and stu_id and stu_name and teacher_ans:
            if len(stu_id) >= 3:
                class_num = stu_id[2]
                valid_classes = ["1", "2", "5", "6"]
                
                if class_num in valid_classes:
                    st.success("✅ 학습 포트폴리오가 완성되었습니다! 아래 버튼을 눌러 그동안 작성한 코드와 함께 PDF로 저장하세요.")
                    
                    if "하" in level:
                        level_key = "ha"
                    elif "중" in level:
                        level_key = "jung"
                    else:
                        level_key = "sang"
                        
                    code_sections = [
                        ("예제 1. 팩토리얼과 순열", "d3_ex1"),
                        ("문제 1. 아이스크림 조합 구하기", "d3_q1"),
                        ("예제 2. 연산 시간과 폭발 그래프", "d3_ex2"),
                        ("문제 2. 브루트 포스 한계 체험", "d3_q3"),
                        ("문제 3. 수준별 종합 도전", f"d3_{level_key}_q4")
                    ]
                    
                    code_data = []
                    for title, prefix in code_sections:
                        code_text = st.session_state.get(f"{prefix}_editor", "")
                        res_tuple = st.session_state.get(f"{prefix}_result", ("", "", None))
                        result_text = res_tuple[0] if res_tuple else ""
                        code_data.append((title, code_text, result_text))
                    
                    student_info = {"group": group_name, "id": stu_id, "name": stu_name}
                    pdf_bytes = create_portfolio_pdf(student_info, teacher_ans, code_data)
                    
                    st.download_button(
                        label="⬇️ 📄 나의 파이썬 코드 포트폴리오 저장하기 (PDF)",
                        data=pdf_bytes,
                        file_name=f"{stu_id}_{stu_name}_3차시_코드포트폴리오.pdf",
                        mime="application/pdf",
                        key="d3_pdf_btn"
                    )

                    portfolio_urls = {
                        "1": "https://padlet.com/ps0andd/p_1",
                        "2": "https://padlet.com/ps0andd/p_2",
                        "5": "https://padlet.com/ps0andd/p_5",
                        "6": "https://padlet.com/ps0andd/p_6",
                    }
                    padlet_portfolio_url = portfolio_urls.get(class_num, "https://padlet.com/")

                    st.info(f"📌 **[미션 1]** 방금 다운로드한 **PDF 파일**을 아래 '{class_num}반 포트폴리오 갤러리'에 업로드해 주세요!")
                    st.markdown(
                        f"""<a href="{padlet_portfolio_url}" target="_blank" 
                            style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #43a047 0%, #66bb6a 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 5px;">
                            ☁️ {class_num}반 포트폴리오 패들렛으로 이동하기
                        </a>""", unsafe_allow_html=True
                    )
                else:
                    st.error("❌ **오류:** 담당 학급(1, 2, 5, 6반)의 학번이 아닙니다. 학번을 다시 확인해 주세요.")
            else:
                st.warning("⚠️ 올바른 5자리 학번(예: 10101)을 입력해 주세요.")
        else:
            st.warning("⚠️ 학생 정보(모둠, 학번, 이름)와 교사의 딥 퀘스천 답변을 모두 작성해야 진행할 수 있습니다.")

        st.markdown("---")
        
        st.markdown("#### 📌 3. 모둠 성찰 질문 만들기 (질문 패들렛 공유용)")
        st.write("모둠원과 함께 오늘 활동을 돌아보며 3가지 질문을 완성하고, 결과를 패들렛에 공유해 봅시다.")
        
        q1 = st.text_area("📖 [사실적 질문] 데이터(n)가 조금만 늘어나도 팩토리얼(n!)과 경우의 수가 기하급수적으로 증가하는 현상을 무엇이라고 부르나요?", height=100, key="d3_q1_ans")
        q2 = st.text_area("🧩 [개념적 질문] 모든 경우를 탐색하는 무차별 대입(Brute-Force) 방식과 비교할 때, AI의 가지치기(Pruning)나 탐욕(Greedy) 알고리즘은 어떤 장단점이 있을까요?", height=100, key="d3_q2_ans")
        q3 = st.text_area("🔥 [논쟁적/딥 퀘스천] 만약 완벽한 정답을 보장하지 않지만 매우 빠른 AI(휴리스틱)와, 100년이 걸리지만 100% 정답을 찾는 시스템 중 의료/국방 분야에서는 어떤 알고리즘을 채택해야 할까요?", height=100, key="d3_q3_ans")

        if group_name and stu_id and q1 and q2 and q3:
            if len(stu_id) >= 3 and stu_id[2] in ["1", "2", "5", "6"]:
                class_num = stu_id[2]
                st.success("✨ 성찰 질문 작성이 완료되었습니다! 텍스트를 복사하여 패들렛에 업로드하세요.")
                report_text = f"""[F.U.T.U.R.E. 프로젝트 3DAY 성찰 일지] (👉 게시물 제목)
모둠명: {group_name}
1. 📖 [사실적 질문]
{q1}
2. 🧩 [개념적 질문]
{q2}
3. 🔥 [논쟁적/딥 퀘스천] 
{q3}
"""
                st.code(report_text, language="markdown")

                qa_urls = {
                    "1": "https://padlet.com/ps0andd/q_1",
                    "2": "https://padlet.com/ps0andd/q_2",
                    "5": "https://padlet.com/ps0andd/q_5",
                    "6": "https://padlet.com/ps0andd/q_6",
                }
                padlet_qa_url = qa_urls.get(class_num, "https://padlet.com/")

                st.info(f"📤 **[미션 2]** 복사한 성찰 일지를 아래 '{class_num}반 질문(Q&A) 패들렛'에 업로드하고, 친구 글에 댓글을 달아주세요!")
                st.markdown(
                    f"""<a href="{padlet_qa_url}" target="_blank" 
                        style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 5px;">
                        🚀 {class_num}반 질문(Q&A) 패들렛으로 이동하기
                    </a>""", unsafe_allow_html=True
                )
            else:
                pass 
        else:
            st.warning("⚠️ 학생 정보(모둠, 학번 등)와 3가지 모둠 성찰 질문을 모두 작성해야 합니다.") 

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True) 

if __name__ == "__main__":
    run()