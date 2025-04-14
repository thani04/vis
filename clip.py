import os
import math
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import MultiPoint
import requests


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


def get_github_folder_contents(owner, repo, folder_path):
    api_url = f"https://github.com/thani04/InfoVisual/tree/main/clips_folder"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"❌ Failed to fetch folder contents from GitHub. Status code: {response.status_code}")
        st.stop()

def download_files_from_github(file_list, temp_dir):
    mat_paths = []
    video_path = None

    for file_info in file_list:
        download_url = file_info['download_url']
        file_name = file_info['name']
        file_path = os.path.join(temp_dir, file_name)

        # ดาวน์โหลดไฟล์
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            if file_name.endswith('.mat'):
                mat_paths.append(file_path)
            elif file_name.endswith('.mp4'):
                video_path = file_path
        else:
            st.warning(f"⚠️ Failed to download {file_name} from GitHub.")

    return mat_paths, video_path



# Helper function to load gaze data
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
    df['F-C score'] = 1- (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)'] / df['Convex Area (Rolling Avg)'])
    df['F-C score'] = df['F-C score'].fillna(0)

    return df, video_frames

# Streamlit UI
st.title("🎯 Gaze & Hull Analysis Tool")

if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0


# Display analysis
if st.session_state.data_processed:
    csv_path = st.session_state.get('csv_path')
    if csv_path and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path, index_col='Frame')
    else:
        st.error("❌ Could not load the data. Please upload files and run the analysis again.")
        st.stop()

    video_frames = st.session_state.video_frames
    current_frame = st.session_state.current_frame
    min_frame, max_frame = int(df.index.min()), int(df.index.max())
    frame_increment = 10

    st.subheader("📊 Convex vs Concave Hull Area Over Time")

    # Frame slider
    new_frame = st.slider("Select Frame", min_frame, max_frame, current_frame)
    st.session_state.current_frame = new_frame

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Previous <10"):
            st.session_state.current_frame = max(min_frame, st.session_state.current_frame - frame_increment)
    with col3:
        if st.button("Next >10"):
            st.session_state.current_frame = min(max_frame, st.session_state.current_frame + frame_increment)

    current_frame = st.session_state.current_frame

    # Prepare data for Altair chart
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
    ).properties(
        width=500,
        height=300
    )

    rule = alt.Chart(pd.DataFrame({'Frame': [current_frame]})).mark_rule(color='red').encode(x='Frame')

    col_chart, col_right = st.columns([2, 1])

    with col_chart:
        st.altair_chart(chart + rule, use_container_width=True)

    with col_right:
        frame_rgb = cv2.cvtColor(video_frames[current_frame], cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {current_frame}", use_container_width=True)
        st.metric("Focus-Concentration Score", f"{df.loc[current_frame, 'F-C score']:.3f}")
