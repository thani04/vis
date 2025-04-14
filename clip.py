import streamlit as st

# 🔗 วิดีโอใน GitHub: https://github.com/nutteerabn/InfoVisual/tree/main/Clips%20(small%20size)
video_files = {
    "APPAL_2a": "APPAL_2a_c.mp4",
    "SIMPS_9a": "SIMPS_9a_c.mp4",
    "SIMPS_19a": "SIMPS_19a_c.mp4",
    "FOODI_2a": "FOODI_2a_c.mp4",
    "MARCH_12a": "MARCH_12a_c.mp4",
    "Cloud_17a": "Cloud_17a_c.mp4",
    "SHREK_3a": "SHREK_3a_c.mp4",
    "DEEPB_3a": "DEEPB_3a_c.mp4",
    "NANN_3a": "NANN_3a_c.mp4"
}

# base URL สำหรับ GitHub raw
base_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/Clips%20(small%20size)/"

st.title("🎬 Play Video from GitHub")

# dropdown เลือกวิดีโอ
selected_video = st.selectbox("เลือกวิดีโอ", list(video_files.keys()))

# ลิงก์วิดีโอแบบ raw
video_url = base_url + video_files[selected_video]

# แสดงวิดีโอ
st.video(video_url)
