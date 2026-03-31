import streamlit as st
from streamlit_ace import st_ace
import pandas as pd
import io
import sys
import datetime
import os
import builtins
from fpdf import FPDF

# ==========================================
# 1. 고품질 PDF 생성 클래스 (ThemedPDF)
# ==========================================
font_path = os.path.join(os.path.dirname(__file__), "font/NanumGothic.ttf")

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
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 1차시 학습 포트폴리오", ln=1, align='C')
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
# ==========================================
# 2. 파이썬 코드 실행 엔진 (동시성 및 보안 완벽 방어 버전)
# ==========================================
def code_runner(code_input):
    output_buffer = io.StringIO()
    
    # 1. builtins 전역 공간을 오염시키지 않는 독립적인 커스텀 print 함수
    def custom_print(*args, sep=' ', end='\n', file=None, flush=False):
        if file is None:
            output_buffer.write(sep.join(map(str, args)) + end)
        else:
            builtins.print(*args, sep=sep, end=end, file=file, flush=flush)

    # 💡 [추가/수정됨] 안전한 임포트(import) 함수 정의
    original_import = builtins.__import__
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        # 학생이 import 할 수 있도록 허용할 모듈 목록 (해킹 방지)
        allowed_modules = ['sympy', 'math', 'random']
        base_name = name.split('.')[0] # 서브 모듈(sympy.abc 등) 허용 처리
        
        if base_name in allowed_modules:
            return original_import(name, globals, locals, fromlist, level)
        raise ImportError(f"🚨 보안 경고: '{name}' 모듈은 이 실습에서 임포트할 수 없습니다.")

    # 2. 보안: 학생이 사용할 수 있는 안전한 기본 함수들만 화이트리스트로 허용
    safe_builtins = {
        'print': custom_print,
        '__import__': safe_import,  # 👈 핵심! import 구문 실행을 위해 필수 추가
        'range': range, 'len': len, 'int': int, 'float': float, 'str': str,
        'bool': bool, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
        'enumerate': enumerate, 'zip': zip, 'type': type, # 자주 쓰는 기본 함수 추가
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        '__build_class__': __build_class__, # 클래스나 함수(def) 정의를 위해 필수
    }
    
    try:
        import sympy as sp
    except ImportError:
        sp = None

    # exec 실행 환경 격리
    exec_globals = {
        '__builtins__': safe_builtins,
        'sp': sp,
        'sympy': sp
    }
    
    # 3. 보안: 위험한 코드 문자열 1차 필터링
    forbidden_keywords = ["import os", "import sys", "import subprocess", "open(", "eval(", "exec("]
    if any(keyword in code_input for keyword in forbidden_keywords):
        return "❌ 시스템 보안 경고: 허용되지 않은 키워드나 명령어 호출이 포함되어 있습니다.", "error"

    try:
        exec(code_input, exec_globals)
        result = output_buffer.getvalue() or "출력된 내용이 없습니다."
        status = "success"
    except Exception as e:
        result = f"{e.__class__.__name__}: {e}"
        status = "error"
        
    return result, status

def display_output(result, status):
    if status == "success":
        st.markdown(f"```bash\n{result}\n```")
    else:
        st.markdown("##### ❌ 실행 중 오류 발생")
        st.markdown(f"<pre style='color: red; background-color: #ffe6e6; padding: 10px; border-radius: 5px;'>{result}</pre>", unsafe_allow_html=True)

def code_block(problem_number, title, starter_code, prefix="", height=280):
    # 키 값이 꼬이지 않도록 prefix와 숫자를 결합
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
            res, stat = st.session_state[f"{key_prefix}_result"]
            display_output(res, stat)
        else:
            st.info("실행 버튼을 누르면 결과가 표시됩니다.")


# ==========================================
# 3. 메인 앱 화면 (UI)
# ==========================================
def run():
    #st.set_page_config(page_title="F.U.T.U.R.E. 1차시", page_icon="🚀", layout="centered")
    
    st.header("1DAY - 📦 수학의 언어를 파이썬으로")
    st.markdown("**🎯 학습 목표:** 공통수학1의 '다항식의 연산', '나머지 정리', '인수정리'가 파이썬 알고리즘과 어떻게 연결되는지 탐구합니다.")
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True) 
    tabs = st.tabs([
        "1️⃣ [F.U] 문제 발견", 
        "2️⃣ [T] 수학의 언어", 
        "3️⃣ [U] AI 활용", 
        "4️⃣ [R] 결과 해석",
        "5️⃣ [E] 세상과 연결"
    ])
    
    # ------------------------------------------
    # 탭 1: 발견 [문맥화]
    # ------------------------------------------
    with tabs[0]:
        st.success("**[문제 인식 및 숨겨진 데이터 찾기]** 우리가 당연하게 여기던 수학적 고정관념을 깨고, 컴퓨터가 데이터를 이해하는 새로운 규칙을 발견하는 과정입니다.\n\n* **✨ 오늘의 레벨업:** 익숙한 수학 기호(=)의 다양한 의미 찾아내기\n* **💬 핵심 탐구 질문:** *\"컴퓨터는 이 수식을 도대체 어떻게 이해하고 있을까?\"*")
        st.markdown("---")
        st.markdown("#### ❔[문제 제기] 수학과 컴퓨터의 충돌, 컴퓨터는 다항식을 어떻게 이해할까?")
        st.latex(r"x = x + 1")
        st.info("방정식으로 풀면 $0 = 1$이 되므로 수학에서는 절대 성립할 수 없는 식입니다. \n\n하지만 파이썬에서는 가장 핵심적인 문법입니다. 왜 그럴까요?")
        hypothesis = st.text_input("💡 나의 가설 세워보기 (컴퓨터에서 '=' 기호와 'x'는 어떤 의미일까?)", placeholder="나의 생각을 자유롭게 적어보세요.")
        if st.button("🔍 가설 확인"):
            if hypothesis.strip() == "":
                st.warning("틀려도 괜찮아요! 먼저 자신의 생각을 적어주세요.")
            else:
                st.success(f"**✅좋은 접근입니다!** '{hypothesis}'라고 생각했군요.")
                st.divider()
                st.markdown("""
                #### 🔓 데이터 구조의 이해
                * **수학에서의 $x$:** 찾아야 할 미지의 고정된 값(**미지수**)
                * **파이썬에서의 `x`:** 데이터를 담아두는 **상자(메모리 공간)**
                * **파이썬에서의 `= `:** 같다는 뜻이 아니라 **'오른쪽의 값을 왼쪽 상자에 넣어라(대입)'**라는 뜻!
                """)

    # ------------------------------------------
    # 탭 2: 번역 [수평적 수학화]
    # ------------------------------------------
    with tabs[1]:
        st.success("**[현상을 수학의 언어로 바꾸기]** 현실의 문제를 수학적 기호와 알고리즘으로 모델링(번역)하는 과정입니다.\n\n* **✨ 오늘의 레벨업:** 🧮 현실을 수학의 언어로 번역하는 힘\n* **💬 핵심 탐구 질문:** *\"어떤 함수와 연산자를 활용하여 이 관계를 모델링할 수 있을까?\"*")
        st.markdown("---")

        st.markdown("#### ▶️ [코딩] 자료형과 출력")
        st.write("""
        수학에서 수와 식을 다루듯, 파이썬에서는 다양한 형태의 자료형을 다룹니다.
        * **문자열(String):** 텍스트 데이터. 따옴표('')로 감싸서 입력합니다. (예: `'Hello World'`, `'다항식'`)
        * **숫자열(Number):** 정수 및 실수 데이터. 사칙연산이 가능합니다. (예: `52`, `3.14`)
        * **불(Boolean):** 참/거짓을 나타내는 논리 데이터. (예: `True`, `False`)
        """)
        
        st.info("💡 **출력 명령어 `print()`:** 괄호 안의 데이터를 화면에 출력하는 함수입니다. 쉼표(`,`)로 구분하여 여러 데이터를 동시에 출력할 수 있습니다.")
        
        st.markdown("""###### 💻 [문제 1] 다양한 자료형 출력하기""")
        code_block(1, "print() 함수 활용", "print('hello', 320)\nprint()", prefix="d1_q", height=180)
        
        st.divider() 
        
        st.markdown("#### ❔ [수학적 질문] 다항식 연산, 컴퓨터의 언어로 어떻게 번역할 수 있을까?")
        st.write("다항식의 사칙연산은 파이썬의 기본 산술 연산자와 직관적으로 대응됩니다. 수학적 기호가 컴퓨팅 명령어로 어떻게 변환되는지 아래의 대응표를 통해 확인해 보십시오.")
        
        data = {
            "연산 개념": ["덧셈", "뺄셈", "곱셈", "실수 나눗셈", "정수 나눗셈(몫)", "나머지", "거듭제곱(차수)"],
            "수학적 표기": ["+", "-", "×", "÷", "Q(x)", "R(x)", "x²"],
            "파이썬 연산자": ["+", "-", "*", "/", "//", "%", "**"],
            "예시 코드": ["3 + 2", "5 - 2", "4 * 2", "10 / 4", "10 // 4", "10 % 4", "2 ** 3"],
            "결과": [5, 3, 8, 2.5, 2, 2, 8]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("""###### 💻 [문제 2] 숫자 연산 출력하기""")
        code_block(2, "산술 연산자 활용", "print('5+7=', 5+7)\nprint('', )\n", prefix="d1_q", height=180)       
        

    # ------------------------------------------
    # 탭 3: AI 도구로 시뮬레이션하기 [수직적 수학화]
    # ------------------------------------------
    with tabs[2]:
        st.success("**[AI 도구로 시뮬레이션하기]** 파이썬과 AI를 활용하여 연산을 심화하고 데이터를 빠르고 정확하게 예측하는 과정입니다.\n\n* **✨ 오늘의 레벨업:** 💻 컴퓨터(AI)를 나의 도구로 부리는 힘\n* **💬 핵심 탐구 질문:** *\"도구를 이용하여 어떻게 더 효율적이고 정확하게 해답을 도출할 수 있을까?\"*")
        st.markdown("---")

        st.markdown("#### ▶️ [코딩] 함수(def)의 이해")
        st.write("""
        수학의 함수 $f(x)$가 입력값에 따라 정해진 출력을 반환하듯, 파이썬에서도 `def` 를 사용하여 특정한 연산을 수행하는 '함수'를 직접 설계할 수 있습니다.
        * **`def` (Define):** 새로운 함수를 정의하겠다는 선언입니다. (예: `def f(x):`)
        * **`return`:** 함수 내부에서 연산된 최종 결괏값을 외부로 반환합니다.
        """)
        
        st.info("💡 **함수 호출**: 정의된 함수는 `print(f(3))`과 같이 특정 입력값을 넣어 실행해야만 결과를 확인할 수 있습니다.")
        
        st.markdown("""###### 💻 [예제 1] $f(x) = x + 10$ 의 함수를 만들고 $f(5)$를 호출해보세요.""")
        st.code("""
def f(x):
    return x + 10
print("f(5)의 결과는:", f(5)) """)
        st.markdown("""###### 💻 [문제 3] 일차함수 $f(x)$를 만들고 $f(3)$를 호출해보세요.""")
        with st.expander("💡 힌트 보기"):
            st.markdown("`def f(x):`를 활용해보세요")
        with st.expander("💡 정답 보기"):
            st.markdown("""```python\ndef f(x):\n    return 2*x+1\nprint("f(3)의 결과는:",f(3))""")
        starter_basic_func = """def """
        code_block(3, "함수 기초", starter_basic_func, prefix="d1_q", height=280)
        
        st.divider()
        
        st.markdown("#### ❔ [수학적 질문] 직접 나누지 않고 나머지를 구하는 원리, 컴퓨터의 함수로 어떻게 구현할까?")
        st.write("수학의 나머지 정리에 따르면, 다항식 $f(x)$를 일차식 $(x-a)$로 나눈 나머지는 함숫값 $f(a)$와 동일합니다. 즉, 복잡한 나눗셈을 직접 할 필요 없이 앞서 배운 파이썬의 함수(`def`)에 값만 대입하여 함숫값을 구하는 것이 곧 나머지를 구하는 가장 빠르고 정확한 알고리즘이 됩니다.")        
        st.info("💡 나머지 정리: ($f(x)$를 $(x-a)$로 나눈 나머지)= $f(a)$")
        st.markdown("""###### 💻 [예제 2] 다항식 $f(x) = 2x^2 - 3x + 7$ 을 $(x-4)$ 로 나눈 나머지를 구하시오.""")
        st.code("""def f(x):
    return 2*(x**2) - 3*x + 7
remainder = f(4)
print("나머지:", remainder)""")

        st.markdown("""###### 💻 [문제 4] 이차다항식 $f(x)$를 만들고 $(x-3)$ 로 나눈 나머지를 구하시오.""")
        with st.expander("💡 힌트 보기"):
            st.markdown("`def f(x):`를 활용해보세요")
        with st.expander("💡 정답 보기"):
            st.markdown("""```python\ndef f(x):\n    return x**2+1\nremainder = f(3)\nprint("나머지:", remainder)""")
        starter_func = """def f(x):
    return 
remainder = 
print("나머지:", remainder)
"""
        code_block(4, "함수를 활용한 나머지", starter_func, prefix="d1_q", height=240)
        
        st.divider()
        st.markdown("#### ❔ [수학적 질문] $x$로 이루어진 복잡한 다항식의 연산, 컴퓨터가 직접 연산할 수 있을까?")
        st.write("💡 SymPy(심파이) 라이브러리란? 숫자 대신 문자를 사용하여 다항식을 수학 시간에 손으로 푸는 것과 똑같이 계산해 주는 파이썬의 강력한 수학 도구입니다.")
        st.info("""
* `import sympy as sp`: SymPy 도구를 불러와서 `sp`라는 짧은 별명으로 부르겠다는 뜻입니다.
* `sp.Symbol('x')`: 컴퓨터에게 $x$가 단순한 글자나 데이터 상자가 아니라, '수학식의 미지수'임을 알려주는 가장 중요한 명령어입니다.
(Symbol에서 S는 대문자로 입력)""")
        st.markdown("""###### 💻 [문제 5] 다항식 계산기(덧셈,뺄셈,곱셈) 만들기""")
        with st.expander("💡 정답 보기"):
            st.markdown("""```python\nimport sympy as sp\nx = sp.Symbol('x')
p = x + 2
q = x**2
print("🧮 다항식 사칙연산 계산기")
print("1. P + Q =", p+q)
print("2. P - Q =", p-q) 
print("3. P * Q =", p*q) 
""")

        starter_sympy = """import sympy as sp
x = sp.Symbol('x')
# 👇식을 직접 입력해보세요!
p = 
q =  

print("🧮 다항식 사칙연산 계산기")
# 👇 연산을 입력해보세요!
print("1. P + Q =", p+q)
print("2. P - Q =",) 
print("3. P * Q =",) 

"""
        code_block(5, "기호 기반 다항식 연산", starter_sympy, prefix="d1_q", height=360)

    # ------------------------------------------
    # 탭 4: 결과의 의미와 한계 고민하기 [응용적 수학화 1]
    # ------------------------------------------
    with tabs[3]:
        st.success("**[결과의 의미와 한계 고민하기]** 산출된 데이터와 결과의 타당성을 검토하고, 한계를 비판적으로 분석하여 적용하는 과정입니다.\n\n* **✨ 오늘의 레벨업:** 🤔 AI의 정답을 의심하고 검증하는 비판적 눈\n* **💬 핵심 탐구 질문:** *\"이 예측 결과는 항상 옳을까? 우리가 직접 적용하며 발견한 오류나 한계는 없을까?\"*")
        st.markdown("---")
        
        st.markdown("#### 💻 [문제 6] 수준별 종합 도전")
        st.write("앞서 배운 개념(자료형, 연산자, 함수, SymPy)을 바탕으로 스스로 문제를 코딩해 보며, 컴퓨터의 연산 논리를 완벽히 내 것으로 만들어 봅시다.")
        
        level = st.radio("자신의 실력에 맞는 난이도를 선택하세요:", 
                         ("🌱 하 (항등식 연산)", "🌿 중 (나머지 정리 함수)", "🌳 상 (SymPy와 인수정리 종합)"), 
                         horizontal=True)
        
        if "하" in level:
            st.info("**[기초]** $A = 20, B = 6$ 일 때, $A$를 $B$로 나눈 몫(Q)과 나머지(R)를 구하고, 항등식 `A == B * Q + R`이 성립하는지 검증하세요.")
            st_code = """A = 20
B = 6

Q =  # 👈 몫을 구하는 연산자를 쓰세요
R =  # 👈 나머지를 구하는 연산자를 쓰세요

print("몫:", Q, "/ 나머지:", R)
print("항등식 확인:", A == (B * Q + R))"""
        elif "중" in level:
            st.info("**[응용]** 다항식 $f(x) = x^3 + 2x^2 - 5x + 3$ 을 파이썬 함수 `def f(x):` 로 정의하고, $x+2$ 로 나눈 나머지를 함숫값을 이용해 출력하세요.")
            st_code = """# 1. 함수 f(x)를 정의하세요
def f(x):
    return 

# 2. x+2로 나눈 나머지는 f(-2)와 같습니다.
result = 
print("x+2로 나눈 나머지는:", result)"""
        else:
            st.info("**[심화 종합]** 고차식 $x^3 - 7x + 6$ 을 SymPy를 이용해 인수분해(factor)해보고, 찾아낸 해를 함수 `f(x)`에 직접 넣었을 때 진짜 나머지가 0이 나오는지 교차 검증해 보세요.")
            st_code = """import sympy as sp
x = sp.Symbol('x')

# 1. SymPy로 인수분해 기계적 계산하기
poly_expr = x**3 - 7*x + 6
print("1. 인수분해 결과:", sp.factor(poly_expr))

# 2. 위에서 찾은 인수(숫자) 중 하나를 골라 진짜 0이 되는지 확인하기
def f(x):
    return (x**3) - 7*x + 6

a =  # 👈 인수분해 결과에서 찾은 숫자를 넣어보세요 (예: 1, 2, -3 등)
print(f"2. 교차 검증: f({a})의 결과는?", f(a))"""
        if st.session_state.get("d1_q6_level") != level:
            st.session_state["d1_q6_level"] = level
            
            # 에디터 키가 존재하면 새 코드로 덮어씌우기
            if "d1_q6_editor" in st.session_state:
                st.session_state["d1_q6_editor"] = st_code
                
            # 실행 결과 초기화
            if "d1_q6_result" in st.session_state:
                st.session_state["d1_q6_result"] = ("", "")

        if "하" in level:
            level_key = "ha"
        elif "중" in level:
            level_key = "jung"
        else:
            level_key = "sang"
        code_block(6, f"도전 코드 ({level[:2]})", st_code, prefix=f"d1_q_{level_key}_", height=300)

    # ------------------------------------------
    # 탭 5: 세상과 연결 및 실천 [응용적 수학화 2]
    # ------------------------------------------
    # ------------------------------------------
    # 탭 5: 세상과 연결 및 실천 [응용적 수학화 2]
    # ------------------------------------------
    with tabs[4]:
        st.success("**[우리의 삶과 사회로 연결하기]** 해석한 결과를 바탕으로 세상을 바꿀 실천적 대안을 기획하고 공유하는 과정입니다.\n\n* **✨ 오늘의 레벨업:** 🤝 배운 것을 세상과 나누며 실천하는 힘\n* **💬 핵심 탐구 질문:** *\"이 결과를 바탕으로 우리는 사회를 위해 무엇을 실천해야 할까?\"*")
        st.markdown("---")
        
        # ------------------------------------------
        # 1. 학생 정보 입력 및 포트폴리오 PDF 생성
        # ------------------------------------------
        st.markdown("#### 💾 1. 학생 정보 입력 및 포트폴리오 저장")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            group_name = st.text_input("모둠 이름 (예: 1모둠)")
        with col_info2:
            stu_id = st.text_input("학번 (예: 10101)", max_chars=5)
        with col_info3:
            stu_name = st.text_input("이름 (예: 홍길동)")

        if group_name and stu_id and stu_name:
            if len(stu_id) >= 3:
                class_num = stu_id[2]
                valid_classes = ["1", "2", "5", "6"]
                
                if class_num in valid_classes:
                    st.success("✅ 학습 포트폴리오가 완성되었습니다! 코드를 PDF로 저장하고 아래 패들렛에 업로드해주세요.")
                    
                    # PDF 데이터 수집 로직
                    if "하" in level:
                        level_key = "ha"
                    elif "중" in level:
                        level_key = "jung"
                    else:
                        level_key = "sang"
                        
                    code_sections = [
                        ("문제 1. 다양한 자료형 출력하기", "d1_q1"),
                        ("문제 2. 숫자 연산 출력하기", "d1_q2"),
                        ("문제 3. 파이썬 함수(def) 기초", "d1_q3"),
                        ("문제 4. 함수를 활용한 나머지 산출", "d1_q4"),
                        ("문제 5. 기호 기반 다항식 연산", "d1_q5"),
                        ("문제 6. 수준별 종합 도전", f"d1_q_{level_key}_6")
                    ]
                    code_data = []
                    for title, prefix in code_sections:
                        code_text = st.session_state.get(f"{prefix}_editor", "")
                        res_tuple = st.session_state.get(f"{prefix}_result", ("", ""))
                        result_text = res_tuple[0] if res_tuple else ""
                        code_data.append((title, code_text, result_text))
                    
                    student_info = {"group": group_name, "id": stu_id, "name": stu_name}
                    pdf_bytes = create_portfolio_pdf(student_info, code_data)
                    
                    # 패들렛 주소 매핑
                    portfolio_urls = {
                        "1": "https://padlet.com/ps0andd/p_1",
                        "2": "https://padlet.com/ps0andd/p_2",
                        "5": "https://padlet.com/ps0andd/p_5",
                        "6": "https://padlet.com/ps0andd/p_6",
                    }
                    padlet_portfolio_url = portfolio_urls.get(class_num, "https://padlet.com/")

                    # 버튼 2개를 나란히 배치
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button(
                            label="📥 내 코딩 실습 결과 PDF 다운로드",
                            data=pdf_bytes,
                            file_name=f"{stu_id}_{stu_name}_1차시_코드포트폴리오.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
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
        st.markdown("#### 💡 2. 교사의 딥(Deep) 퀘스천에 대한 우리의 답변 작성하기")
        st.info("🔥 **교사의 심화 질문(Deep Question):**\n\n컴퓨터(AI)가 알고리즘을 통해 복잡한 연산을 0.1초 만에 수행하는 시대입니다. 그렇다면 우리는 왜 다항식의 연산과 나머지 정리 같은 수학적 원리를 학습해야 할까요?")
        teacher_ans = st.text_area("위 질문에 대한 우리의 답을 논리적으로 작성해 보세요.", height=100)

        st.markdown("---")
        
        # ------------------------------------------
        # 3. 모둠 질문 만들기 및 패들렛 공유
        # ------------------------------------------
        st.markdown("#### 💬 3. 모둠 질문 만들기 (질문 패들렛 공유용)")
        st.write("모둠원과 함께 오늘 활동을 돌아보며 심화 질문을 하나 만들고, 앞서 작성한 답변과 함께 패들렛에 공유해 봅시다.")
        
        q_deep = st.text_area("🔥 **[우리의 딥(Deep) 퀘스천]** *(윤리와 철학)* 배운 지식이나 기술이 실제 사회에 적용될 때 발생할 수 있는 부작용이나 윤리적 딜레마를 다루며, 정답 없이 서로의 가치관을 깊이 있게 나눌 수 있는 토론형 질문입니다. 👉 :blue[예)AI가 0.1초 만에 계산을 다 해주는 시대, 수학을 못해도 AI만 잘 다루면 괜찮을까요?]", height=100)

        if group_name and stu_id and teacher_ans and q_deep:
            if len(stu_id) >= 3 and stu_id[2] in ["1", "2", "5", "6"]:
                class_num = stu_id[2]
                st.success("✅ 답변 작성이 완료되었습니다! 아래 양식을 복사하여 패들렛에 업로드하세요.")
                
                report_text = f"""[F.U.T.U.R.E. 프로젝트 1DAY 성찰 일지]
모둠명: {group_name}

🔥 [우리가 만든 딥(Deep) 퀘스천] 
{q_deep}

💡 [교사의 딥(Deep) 퀘스천에 대한 우리의 생각]
{teacher_ans}
"""
                st.code(report_text, language="markdown")

                # 선생님의 1차시 반별 Q&A 패들렛 주소 매핑
                qa_urls = {
                    "1": "https://padlet.com/ps0andd/q_1",
                    "2": "https://padlet.com/ps0andd/q_2",
                    "5": "https://padlet.com/ps0andd/q_5",
                    "6": "https://padlet.com/ps0andd/q_6",
                }
                padlet_qa_url = qa_urls.get(class_num, "https://padlet.com/")

                st.info(f"📝 **[미션 2]** 복사한 성찰 일지를 아래 '{class_num}반 질문(Q&A) 패들렛'에 모둠 게시판에 업로드하고, 친구 글에 댓글을 달아주세요!")
                st.markdown(
                    f"""<a href="{padlet_qa_url}" target="_blank" 
                       style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 5px;">
                       🚀 {class_num}반 질문(Q&A) 패들렛으로 이동하기
                    </a>""", unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ 교사의 심화 질문에 대한 답변과 모둠 질문을 모두 작성해야 패들렛 공유 양식이 나타납니다.")

    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True) 

if __name__ == "__main__":
    run()