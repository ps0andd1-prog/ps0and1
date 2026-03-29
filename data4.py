import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.figure import Figure
import os
import datetime
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
        self.cell(0, 10, "F.U.T.U.R.E. 프로젝트 4차시 실습 포트폴리오", ln=1, align='C')
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

def create_portfolio_pdf(student_info, teacher_ans, sim_data):
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
    pdf.multi_cell(0, 6, "Q. 자율주행 자동차나 의료 인공지능은 카메라로 들어온 '숫자 행렬(이미지)'을 분석하여 판단을 내립니다. 만약 카메라 렌즈에 이물질이 묻거나 조명이 어두워져 행렬 데이터에 오류(노이즈)가 생긴다면 어떤 문제가 발생할 수 있으며, 이를 예방하거나 해결하기 위해 알고리즘적으로 어떤 행렬 연산이 필요할까요?")
    pdf.ln(3)
    
    pdf.set_text_color(21, 101, 192) 
    pdf.cell(0, 8, "▶ 나의 답변", ln=1)
    pdf.set_text_color(50, 50, 50) 
    pdf.p(teacher_ans if teacher_ans else "작성된 내용이 없습니다.")
    
    pdf.add_page()
    
    pdf.h2("📊 나의 행렬 시뮬레이터 실습 결과")
    
    pdf.set_font(pdf._font_family, '', 11)
    pdf.set_text_color(21, 101, 192)
    pdf.cell(0, 8, "▶ 실습 1. 행렬 덧셈/뺄셈 (밝기 조절과 색상 반전)", ln=1)
    pdf.set_font(pdf._font_family, '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.p(f"- 내가 적용한 밝기(덧셈) 값: {sim_data.get('brightness', 0)}")
    pdf.p(f"- 색상 반전(뺄셈) 적용 여부: {'적용함' if sim_data.get('invert', False) else '적용 안 함'}")
    pdf.ln(5)

    pdf.set_font(pdf._font_family, '', 11)
    pdf.set_text_color(21, 101, 192)
    pdf.cell(0, 8, "▶ 실습 2. 행렬 곱셈 (AI 이미지 필터/합성곱)", ln=1)
    pdf.set_font(pdf._font_family, '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.set_fill_color(245, 245, 245)
    pdf.multi_cell(0, 6, sim_data.get('filter_reflection', '작성된 내용이 없습니다.'), border=1, fill=True)

    return bytes(pdf.output(dest='S'))

# ==========================================
# 2. 이미지 시각화 도우미 함수
# ==========================================
def draw_matrix_image(matrix, title):
    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()
    ax.imshow(matrix, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # 칸 안에 숫자 표시 (너무 크지 않을 때만)
    if matrix.shape[0] <= 10:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = 'white' if val < 128 else 'black'
                ax.text(j, i, str(int(val)), ha="center", va="center", color=text_color, fontsize=10)
    return fig

# ==========================================
# 3. 메인 앱 화면 (UI)
# ==========================================
def run():
    st.header("4DAY - 🖼️ 세상의 데이터는 행렬이다")
    st.markdown("**🎯 학습 목표:** 코딩 대신 시뮬레이터를 조작하며 컴퓨터가 이미지를 인식하는 원리를 이해하고, 행렬의 **덧셈, 뺄셈, 곱셈**이 실제 이미지 처리에 어떻게 적용되는지 시각적으로 체험합니다.")
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True) 
    
    tabs = st.tabs([
        "1️⃣ [F.U] 문제 발견", 
        "2️⃣ [T] 수학 시뮬레이터", 
        "3️⃣ [U] AI 비전 실습", 
        "4️⃣ [R] 결과 해석",
        "5️⃣ [E] 세상과 연결"
    ])
    
    # ------------------------------------------
    # 탭 1: 현실 탐색 [문맥화]
    # ------------------------------------------
    with tabs[0]:
        st.success("**[문제 인식 및 숨겨진 데이터 찾기]** 우리 눈에 보이는 선명한 사진이 컴퓨터의 눈에는 어떻게 보이는지 그 숨겨진 구조를 탐색합니다.")
        st.markdown("---")
        
        st.markdown("#### 📌 [문제 제기] 🤖 눈이 없는 인공지능은 어떻게 사진을 볼까?")
        st.write("스마트폰으로 찍은 사진을 끝없이 확대해 본 적이 있나요? 사진은 결국 수많은 네모난 점(**픽셀, Pixel**)들의 모음입니다.")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.info("""💡 **디지털 이미지의 비밀**
* 컴퓨터는 색상을 숫자로 기억합니다.
* 흑백 사진에서 **0은 가장 어두운 검은색(⬛)**을, **255는 가장 밝은 흰색(⬜)**을 의미합니다.
* 0부터 255 사이의 숫자들이 가로, 세로 표(행렬) 형태로 빽빽하게 저장되어 있는 것이 바로 '이미지 데이터'입니다.""")
        with col_img2:
            st.markdown("""
| 0 (검은색) | 128 (회색) | 255 (흰색) |
| :---: | :---: | :---: |
| ⬛ | 🔲 | ⬜ |
""")
            st.write("즉, 인공지능에게 사진을 보여준다는 것은 거대한 **'숫자 행렬(Matrix)'**을 입력하는 것과 같습니다.")

    # ------------------------------------------
    # 탭 2: 수학적 시뮬레이터 [행렬 생성 및 덧/뺄셈]
    # ------------------------------------------
    with tabs[1]:
        st.success("**[현상을 수학의 언어로 바꾸기]** 행렬의 **덧셈**과 **뺄셈**이 스마트폰의 '사진 필터(밝기/반전)'로 어떻게 변하는지 체험합니다.")
        st.markdown("---")

        st.markdown("#### 🎮 시뮬레이터 1: 행렬 덧셈과 뺄셈 (밝기와 색상 반전)")
        st.write("왼쪽 표의 숫자를 직접 수정해 보세요! 숫자가 바뀌면 오른쪽 이미지가 즉시 변합니다. (0~255 사이 입력)")
        
        col_matrix, col_image = st.columns([1.2, 1])
        
        with col_matrix:
            st.markdown("##### ✏️ 1. 원본 행렬 그리기 (5x5)")
            # 초기 행렬 세팅 (십자가 모양)
            if 'd4_matrix' not in st.session_state:
                init_matrix = pd.DataFrame(
                    [[0, 0, 255, 0, 0],
                     [0, 0, 255, 0, 0],
                     [255, 255, 255, 255, 255],
                     [0, 0, 255, 0, 0],
                     [0, 0, 255, 0, 0]]
                )
                st.session_state.d4_matrix = init_matrix

            edited_df = st.data_editor(st.session_state.d4_matrix, use_container_width=True, hide_index=True)
            
            # 입력값 0~255 강제 보정
            base_matrix = edited_df.to_numpy()
            base_matrix = np.clip(base_matrix, 0, 255)

            st.markdown("##### 🎛️ 2. 수학 연산 적용하기")
            brightness = st.slider("➕ 행렬 덧셈 (밝기 조절)", -255, 255, 0, key="d4_brightness")
            invert = st.toggle("➖ 행렬 뺄셈 (색상 반전: 255 - 행렬)", key="d4_invert")
            
        with col_image:
            st.markdown("##### 🖥️ 결과 이미지")
            
            # 연산 적용
            result_matrix = base_matrix.copy()
            if invert:
                result_matrix = 255 - result_matrix
            
            result_matrix = result_matrix + brightness
            result_matrix = np.clip(result_matrix, 0, 255) # 0~255 벗어나지 않게 고정
            
            fig = draw_matrix_image(result_matrix, "행렬 연산이 적용된 이미지")
            st.pyplot(fig)
            
            st.info(f"**현재 적용된 수학 공식:**\n\n `결과 행렬 = {'(255 - 원본 행렬)' if invert else '원본 행렬'} + {brightness}`")

    # ------------------------------------------
    # 탭 3: AI 비전 실습 [행렬 곱셈/합성곱]
    # ------------------------------------------
    with tabs[2]:
        st.success("**[AI 필터 원리 체험]** 인공지능이 이미지의 특징(테두리 등)을 찾아낼 때 사용하는 **행렬의 곱셈(Convolution, 합성곱)** 원리를 실습합니다.")
        st.markdown("---")

        st.markdown("#### 🎮 시뮬레이터 2: 행렬 곱셈 (AI 이미지 필터)")
        st.write("AI는 이미지 행렬에 **작은 필터(3x3 가중치 행렬)**를 겹쳐서 곱하고 더하며 이미지를 변환합니다. 다양한 필터를 적용해 원본 이미지가 어떻게 변하는지 관찰하세요.")

        # 샘플 10x10 이미지 (가운데 네모)
        sample_img = np.zeros((10, 10))
        sample_img[2:8, 2:8] = 200
        sample_img[4:6, 4:6] = 50 # 안쪽 구멍

        col_filter, col_result = st.columns([1, 1.5])
        
        with col_filter:
            filter_type = st.radio("🔍 곱해줄 필터(행렬) 선택:", 
                                   ("선택 안 함 (원본)", 
                                    "윤곽선 찾기 (Edge Detection)", 
                                    "흐리게 만들기 (Blur)", 
                                    "날카롭게 (Sharpen)"), key="d4_filter_type")
            
            # 필터 행렬 정의
            kernel = np.zeros((3, 3))
            if filter_type == "윤곽선 찾기 (Edge Detection)":
                kernel = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]])
            elif filter_type == "흐리게 만들기 (Blur)":
                kernel = np.ones((3, 3)) / 9
            elif filter_type == "날카롭게 (Sharpen)":
                kernel = np.array([[ 0, -1,  0],
                                   [-1,  5, -1],
                                   [ 0, -1,  0]])
            elif filter_type == "선택 안 함 (원본)":
                kernel = np.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])

            st.write("선택된 필터의 수학적 행렬:")
            st.dataframe(pd.DataFrame(kernel).round(2), hide_index=True)

        with col_result:
            from scipy.signal import convolve2d
            
            # 행렬 곱셈(합성곱) 연산 수행
            filtered_img = convolve2d(sample_img, kernel, mode='same', boundary='fill', fillvalue=0)
            filtered_img = np.clip(filtered_img, 0, 255)

            fig2 = Figure(figsize=(8, 4))
            ax1, ax2 = fig2.subplots(1, 2)
            
            ax1.imshow(sample_img, cmap='gray', vmin=0, vmax=255)
            ax1.set_title("원본 이미지 (10x10)")
            ax1.axis('off')
            
            ax2.imshow(filtered_img, cmap='gray', vmin=0, vmax=255)
            ax2.set_title("필터 곱셈이 완료된 결과")
            ax2.axis('off')
            
            st.pyplot(fig2)

    # ------------------------------------------
    # 탭 4: 결과 해석
    # ------------------------------------------
    with tabs[3]:
        st.success("**[실습 내용 정리]** 시뮬레이터를 통해 관찰한 '행렬 연산'과 '이미지 변화'의 관계를 정리해 봅니다.")
        st.markdown("---")
        
        st.markdown("#### 📌 실습 관찰 보고서")
        st.write("탭 3에서 **행렬 곱셈(필터)**을 적용했을 때, 이미지가 어떻게 변했는지 수학적 이유와 함께 적어주세요.")
        
        filter_reflection = st.text_area(
            "예: '윤곽선 찾기' 필터를 곱했더니 이미지의 테두리만 하얗게 남았다. 그 이유는...", 
            height=150, 
            key="d4_filter_reflection"
        )

    # ------------------------------------------
    # 탭 5: 세상 연결
    # ------------------------------------------
    with tabs[4]:
        st.success("**[우리의 삶과 사회로 연결하기]** 컴퓨터 비전과 이미지 처리 기술이 사회의 어떤 문제를 해결할 수 있을지 고민하고 성찰합니다.")
        st.markdown("---")
        
        st.markdown("#### 📌 1. 학생 정보 입력")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            group_name = st.text_input("모둠 이름 (예: 1모둠)", key="d4_group")
        with col_info2:
            stu_id = st.text_input("학번 (예: 10101)", max_chars=5, key="d4_id")
        with col_info3:
            stu_name = st.text_input("이름 (예: 홍길동)", key="d4_name")

        st.markdown("---")

        st.markdown("#### 📌 2. 나의 생각 쓰기 및 실습 포트폴리오 저장")
        st.info("🔥 **교사의 심화 질문(Deep Question):**\n\n자율주행 자동차나 의료 인공지능은 카메라로 들어온 '숫자 행렬(이미지)'을 분석하여 판단을 내립니다. 만약 카메라 렌즈에 이물질이 묻거나 조명이 어두워져 행렬 데이터에 오류(노이즈)가 생긴다면 어떤 문제가 발생할 수 있으며, 이를 예방하거나 해결하기 위해 알고리즘적으로 어떤 행렬 연산이 필요할까요?")
        
        teacher_ans = st.text_area("위 질문에 대한 나만의 답을 논리적으로 작성해 보세요.", height=120, key="d4_teacher_ans")

        if group_name and stu_id and stu_name and teacher_ans and filter_reflection:
            if len(stu_id) >= 3:
                class_num = stu_id[2]
                valid_classes = ["1", "2", "5", "6"]
                
                if class_num in valid_classes:
                    st.success("✅ 학습 포트폴리오가 완성되었습니다! 아래 버튼을 눌러 나의 시뮬레이터 실습 결과와 함께 PDF로 저장하세요.")
                    
                    sim_data = {
                        "brightness": st.session_state.get("d4_brightness", 0),
                        "invert": st.session_state.get("d4_invert", False),
                        "filter_reflection": filter_reflection
                    }
                    
                    student_info = {"group": group_name, "id": stu_id, "name": stu_name}
                    pdf_bytes = create_portfolio_pdf(student_info, teacher_ans, sim_data)
                    
                    st.download_button(
                        label="⬇️ 📄 나의 행렬 실습 포트폴리오 저장하기 (PDF)",
                        data=pdf_bytes,
                        file_name=f"{stu_id}_{stu_name}_4차시_실습포트폴리오.pdf",
                        mime="application/pdf",
                        key="d4_pdf_btn"
                    )

                    portfolio_urls = {
                        "1": "https://padlet.com/",
                        "2": "https://padlet.com/",
                        "5": "https://padlet.com/",
                        "6": "https://padlet.com/",
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
            st.warning("⚠️ 학생 정보, 교사 질문, 실습 관찰 보고서를 모두 작성해야 PDF를 다운로드할 수 있습니다.")

        st.markdown("---")
        
        st.markdown("#### 📌 3. 모둠 성찰 질문 만들기")
        
        q1 = st.text_area("📖 [사실적 질문] 컴퓨터가 인식하는 이미지에서 숫자 0과 255는 각각 어떤 색을 의미하나요?", height=80, key="d4_q1_ans")
        q2 = st.text_area("🧩 [개념적 질문] 스마트폰에서 어두운 사진을 밝게 보정하거나 색상을 반전시키는 기능은 행렬의 어떤 사칙연산을 이용한 것인가요?", height=80, key="d4_q2_ans")
        q3 = st.text_area("🔥 [논쟁적/딥 퀘스천] 범죄 수사에서 흐릿하게 찍힌 CCTV(노이즈가 많은 행렬)를 AI 필터 곱셈으로 선명하게 복원하는 기술이 법적 증거로 100% 채택되어도 괜찮을까요?", height=80, key="d4_q3_ans")

        if group_name and stu_id and q1 and q2 and q3:
            if len(stu_id) >= 3 and stu_id[2] in ["1", "2", "5", "6"]:
                class_num = stu_id[2]
                st.success("✨ 성찰 질문 작성이 완료되었습니다! 텍스트를 복사하여 패들렛에 업로드하세요.")
                report_text = f"""[F.U.T.U.R.E. 프로젝트 4DAY 성찰 일지] (👉 게시물 제목)
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
                    "1": "https://padlet.com/",
                    "2": "https://padlet.com/",
                    "5": "https://padlet.com/",
                    "6": "https://padlet.com/",
                }
                padlet_qa_url = qa_urls.get(class_num, "https://padlet.com/")

                st.info(f"📤 **[미션 2]** 복사한 성찰 일지를 아래 '{class_num}반 질문(Q&A) 패들렛'에 업로드하고, 친구 글에 댓글을 달아주세요!")
                st.markdown(
                    f"""<a href="{padlet_qa_url}" target="_blank" 
                        style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 5px;">
                        🚀 {class_num}반 질문(Q&A) 패들렛으로 이동하기
                    </a>""", unsafe_allow_html=True
                )
    st.markdown("<hr style='border: 2px solid #2196F3;'>", unsafe_allow_html=True) 

if __name__ == "__main__":
    run()