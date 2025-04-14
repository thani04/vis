import os
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
from scipy.spatial import ConvexHull
import alphashape

# ----------------------------
# CONFIG
# ----------------------------
video_files = {
    "APPAL_2a": "APPAL_2a_hull_area.mp4",
    "FOODI_2a": "FOODI_2a_hull_area.mp4",
    "MARCH_12a": "MARCH_12a_hull_area.mp4",
    "NANN_3a": "NANN_3a_hull_area.mp4",
    "SHREK_3a": "SHREK_3a_hull_area.mp4",
    "SIMPS_19a": "SIMPS_19a_hull_area.mp4"
}
base_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/processed%20hull%20area%20overlay/"

# ----------------------------
# APP START
# ----------------------------
st.title("🎬 Gaze Hull Visualization")

selected_video = st.selectbox("🎥 เลือกวิดีโอ", list(video_files.keys()))
video_url = base_url + video_files[selected_video]
st.video(video_url)

# ----------------------------
# Helper functions
# ----------------------------
@st.cache_data
def load_gaze_data(mat_files):
    gaze_data_per_viewer = []
    for mat_file in mat_files:
        mat = scipy.io.loadmat(mat_file)
        eyetrack = mat['eyetrackRecord']
        gaze_x = eyetrack['x'][0, 0].flatten()
        gaze_y = eyetrack['y'][0, 0].flatten()
        timestamps = eyetrack['t'][0, 0].flatten()
        valid = (gaze_x != -32768) & (gaze_y != -32768)
        gaze_x = gaze_x[valid]
        gaze_y = gaze_y[valid]
        timestamps = timestamps[valid] - timestamps[0]
        gaze_x_norm = gaze_x / np.max(gaze_x)
        gaze_y_norm = gaze_y / np.max(gaze_y)
        gaze_data_per_viewer.append((gaze_x_norm, gaze_y_norm, timestamps))
    return gaze_data_per_viewer

@st.cache_resource
def process_video_analysis(gaze_data_per_viewer, video_path, alpha=0.007, window_size=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_numbers = []
    convex_areas = []
    concave_areas = []
    video_frames = []

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gaze_points = []
        for gaze_x_norm, gaze_y_norm, timestamps in gaze_data_per_viewer:
            frame_indices = (timestamps / 1000 * fps).astype(int)
            if frame_num in frame_indices:
                idx = np.where(frame_indices == frame_num)[0]
                for i in idx:
                    gx = int(np.clip(gaze_x_norm[i], 0, 1) * (w - 1))
                    gy = int(np.clip(gaze_y_norm[i], 0, 1) * (h - 1))
                    gaze_points.append((gx, gy))

        if len(gaze_points) >= 3:
            points = np.array(gaze_points)
            try:
                convex_area = ConvexHull(points).volume
            except:
                convex_area = 0

            try:
                concave = alphashape.alphashape(points, alpha)
                concave_area = concave.area if concave.geom_type == 'Polygon' else 0
            except:
                concave_area = 0

            frame_numbers.append(frame_num)
            convex_areas.append(convex_area)
            concave_areas.append(concave_area)
            video_frames.append(frame)

        frame_num += 1

    cap.release()

    df = pd.DataFrame({
        'Frame': frame_numbers,
        'Convex Area': convex_areas,
        'Concave Area': concave_areas
    })
    df.set_index('Frame', inplace=True)
    df['Convex Area (Rolling Avg)'] = df['Convex Area'].rolling(window=window_size, min_periods=1).mean()
    df['Concave Area (Rolling Avg)'] = df['Concave Area'].rolling(window=window_size, min_periods=1).mean()
    df['F-C score'] = 1 - (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)']
    df['F-C score'] = df['F-C score'].fillna(0)

    return df, video_frames

# ----------------------------
# Process and UI rendering
# ----------------------------
# mock loading data result (use pre-processed CSV & video_frames for now)
df = st.session_state.get("df", None)
video_frames = st.session_state.get("video_frames", [])

if df is not None and not df.empty:
    min_frame = int(df.index.min())
    max_frame = int(df.index.max())
    st.subheader("\ud83d\udcca Convex vs Concave Hull Area Over Time")

    current_frame = st.slider("Select Frame", min_value=min_frame, max_value=max_frame, value=min_frame)

    df_melt = df.reset_index().melt(id_vars='Frame', value_vars=[
        'Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'
    ], var_name='Metric', value_name='Area')

    chart = alt.Chart(df_melt).mark_line().encode(
        x='Frame',
        y='Area',
        color=alt.Color(
            'Metric:N',
            scale=alt.Scale(
                domain=['Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'],
                range=['rgb(0, 210, 0)', 'rgb(0, 200, 255)']
            ),
            legend=alt.Legend(orient='bottom', title='Hull Type')
        )
    ).properties(width=500, height=300)

    rule = alt.Chart(pd.DataFrame({'Frame': [current_frame]})).mark_rule(color='red').encode(x='Frame')

    col_chart, col_right = st.columns([2, 1])

    with col_chart:
        st.altair_chart(chart + rule, use_container_width=True)

    with col_right:
        if current_frame < len(video_frames):
            frame_rgb = cv2.cvtColor(video_frames[current_frame], cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Frame {current_frame}", use_container_width=True)
            st.metric("Focus-Concentration Score", f"{df.loc[current_frame, 'F-C score']:.3f}")

            # --- compactness box ---
            convex = df.loc[current_frame, 'Convex Area']
            concave = df.loc[current_frame, 'Concave Area']
            compactness = concave / convex if convex > 0 else 0

            if compactness >= 0.8:
                color, label = 'green', 'Highly Focused'
            elif compactness >= 0.5:
                color, label = 'orange', 'Moderate Focus'
            else:
                color, label = 'red', 'Scattered'

            st.markdown(
                f"<div style='border:2px solid {color}; border-radius:10px; padding:10px; text-align:center;'>"
                f"<h4 style='color:{color}; margin:0;'>Compactness: {compactness:.2f}</h4>"
                f"<small style='color:gray'>{label}</small>"
                f"</div>",
                unsafe_allow_html=True
            )
else:
    st.warning("\u26a0\ufe0f ไม่พบข้อมูลสำหรับแสดงผล อาจเกิดจากการดาวน์โหลดผิดพลาด หรือ .mat ไม่มีข้อมูล valid")
