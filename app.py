import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- CẤU HÌNH TRANG - NHÌN LÀ THẤY CƯNG ---
st.set_page_config(page_title="Moto Price AI | Slay & Ride", page_icon="🛵", layout="wide")

# CSS Custom: Style này là sự kết hợp giữa sự chuyên nghiệp và chút "dễ thương"
st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;500;700&display=swap');



    html, body, [class*="st-"] {

        font-family: 'Comfortaa', cursive;

    }



    .main { background-color: #f0f2f6; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #FF4B4B; font-weight: bold; }
    .st-emotion-cache-1r6slb0, .st-emotion-cache-12w0qpk {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px;
    }

   

    /* Card style cho các block */

    .st-emotion-cache-1r6slb0, .st-emotion-cache-12w0qpk {

        background-color: rgba(255, 255, 255, 0.9);

        padding: 25px;

        border-radius: 20px;

        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);

        backdrop-filter: blur(4px);

        border: 1px solid rgba(255, 255, 255, 0.18);

        margin-bottom: 20px;

    }



    /* Nút bấm kiểu Gen Z: Bo tròn + Gradient */

    div.stButton > button:first-child {

        background: linear-gradient(45deg, #FF9A9E 0%, #FAD0C4 99%, #FAD0C4 100%);

        color: #4A4A4A;

        border-radius: 30px;

        border: none;

        height: 55px;

        width: 100%;

        font-size: 20px;

        font-weight: bold;

        transition: all 0.4s ease;

        box-shadow: 0 4px 15px rgba(255, 154, 158, 0.4);

    }

   

    div.stButton > button:hover {

        transform: translateY(-3px);

        box-shadow: 0 6px 20px rgba(255, 154, 158, 0.6);

        color: white;

    }



    /* Metric xịn xò */

    [data-testid="stMetricValue"] {

        color: #FF6B6B;

        font-family: 'Comfortaa';

    }



    /* Header */

    h1 { color: #FF6B6B; text-align: center; font-weight: 700; }

    h2, h3 { color: #5D5D5D; }

    </style>

    """, unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess():
    file_path = "motorbike_data.csv"
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    current_year = datetime.now().year
    
    def clean_price(val):
        if pd.isna(val): return np.nan
        try:
            s = str(val).strip()
            if s.endswith('.00') or s.endswith(',00'): s = s[:-3]
            s = s.replace('.', '').replace(',', '')
            return float(s)
        except: return np.nan

    df['Giá bán'] = df['Giá bán'].apply(clean_price)
    for col in ['Hãng xe', 'Dòng xe', 'Khu vực bán']:
        df[col] = df[col].astype(str).str.strip()
    df['Tình trạng xe'] = df['Tình trạng xe'].astype(str).str.replace(',', '.').astype(float)
    df['đã phụ tùng chưa thay'] = df['đã phụ tùng chưa thay'].astype(str).str.strip()
    df['Phụ tùng'] = df['đã phụ tùng chưa thay'].apply(lambda x: 1 if 'Đã thay' in x.lower() else 0)
    df['Tuổi xe'] = df['Năm sản xuất'].apply(lambda x: max(0, current_year - x))
    df = df.dropna(subset=['Giá bán', 'Hãng xe', 'Dòng xe'])
    Q1, Q3 = df['Giá bán'].quantile(0.25), df['Giá bán'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Giá bán'] >= Q1 - 1.5 * IQR) & (df['Giá bán'] <= Q3 + 1.5 * IQR)]
    return df

@st.cache_resource
def train_model(df):
    features = ["Hãng xe", "Dòng xe", "Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng", "Khu vực bán"]
    X, y = df[features], df["Giá bán"]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng"]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ["Hãng xe", "Dòng xe", "Khu vực bán"])
    ])
    pipeline = Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(n_estimators=200, random_state=42))])
    pipeline.fit(X, y)
    return pipeline, r2_score(y, pipeline.predict(X)), mean_absolute_error(y, pipeline.predict(X))

df = load_and_preprocess()
current_year = datetime.now().year

if df is not None:
    model_pl, r2, mae = train_model(df)
    
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>✨ MOTO VIBES ✨</h2>", unsafe_allow_html=True)
        st.image("https://i.ibb.co/dwkNgdxm/Thumb-Luoc-su.jpg", use_container_width=True)
        st.write(f"🌈 **Tình trạng:** Đang cực 'mượt'")
        st.write(f"📊 **Data:** 200 'em' xe")
        st.write(f"🎯 **Độ chuẩn:** {r2*100:.1f}% (Khá là slay)")
        st.success("Tư vấn bằng cái tâm, định giá bằng cái tầm! ✨")

    st.markdown("<h1>🏍️ TECHBEES - MUA XE LÀ PHẢI CHÁY</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Check giá xe cũ nhanh như cách crush 'seen' tin nhắn của bạn</p>", unsafe_allow_html=True)

    # Chỉnh sửa Tab nhìn cho nó mướt
    tab1, tab2, tab3 = st.tabs(["✨ Định giá nhanh", "📈 Chuyên gia soi", "📂 Kho dữ liệu"])

    with tab1:
        left_col, right_col = st.columns([1, 1], gap="large")
        
        with left_col:
            st.markdown("### 📝 'Profile' em xe")
            list_hang = sorted(df["Hãng xe"].unique())
            brand = st.selectbox("Chọn brand (Hãng xe)", list_hang)
            
            list_dong = sorted(df[df["Hãng xe"] == brand]["Dòng xe"].unique())
            model_bike = st.selectbox("Dòng xe nào đây?", list_dong)
            
            c1, c2 = st.columns(2)
            year = c1.number_input("Năm ra lò", 1990, 2030, current_year-2)
            km = c2.number_input("Số km đã lượn", 0, 500000, 10000)
            
            cond = st.slider("Độ 'mới' của ngoại hình (1-10)", 1.0, 10.0, 8.5)
            part = st.radio("Máy móc sao rồi?", ["Zin (Chưa thay)", "Đã qua 'phẫu thuật'"], horizontal=True)
            area = st.selectbox("Hộ khẩu ở đâu? (Khu vực)", sorted(df["Khu vực bán"].unique()))
            
            predict_btn = st.button("🚀 CHỐT GIÁ NGAY!")

        with right_col:
            st.markdown("### 💰 'Ví' bạn cần bao nhiêu?")
            if predict_btn:
                with st.spinner("Đợi tí, AI đang 'cook' dữ liệu..."):
                    time.sleep(1)
                    input_data = pd.DataFrame([{
                        "Hãng xe": brand, "Dòng xe": model_bike, "Tuổi xe": max(0, current_year - year),
                        "Số km đã chạy": km, "Tình trạng xe": cond, 
                        "Phụ tùng": 1 if "phẫu thuật" in part.lower() else 0, "Khu vực bán": area
                    }])
                    res = model_pl.predict(input_data)[0]
                    
                    st.balloons()
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1E1E1E 0%, #434343 100%); padding: 40px; border-radius: 25px; text-align: center; border: 2px solid #FF9A9E;">
                            <h2 style="color: #FF9A9E; margin: 0; font-size: 20px;">GIÁ NÀY LÀ HỢP LÝ</h2>
                            <h1 style="color: white; font-size: 50px; margin: 15px 0;">{res:,.0f} <span style='font-size:20px;'>VNĐ</span></h1>
                            <p style="color: #ccc; font-size: 14px;">(Giá này để bạn tham khảo, mặc cả là việc của bạn nhé 😉)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    avg_price = df[df['Dòng xe'] == model_bike]['Giá bán'].mean()
                    diff = res - avg_price
                    delta_color = "normal" if diff > 0 else "inverse"
                    st.write("")
                    st.metric(f"So với các 'đồng môn' {model_bike}", f"{res:,.0f}đ", f"{diff:,.0f}đ", delta_color=delta_color)
            else:
                st.info("Nhập thông tin rồi bấm nút phía dưới để xem kết quả nha bà con! 👇")

    with tab2:
        st.markdown("### 📊 Bản tin thị trường (Real-time-ish)")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.box(df, x="Hãng xe", y="Giá bán", color="Hãng xe", 
                         title="Xe hãng nào 'chát' nhất?", template="plotly_white",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.scatter(df, x="Tuổi xe", y="Giá bán", color="Hãng xe", size="Số km đã chạy", 
                             hover_data=['Dòng xe'], title="Càng già càng rẻ hay sao?",
                             color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 📂 Toàn bộ dữ liệu 'mắt thấy tai nghe'")
        st.dataframe(df.style.background_gradient(subset=['Giá bán'], cmap='BuPu'), use_container_width=True)

else:
    st.error("🆘 Toang rồi! Không tìm thấy file 'motorbike_data.csv'. Coi lại folder đi ông cháu ơi!")