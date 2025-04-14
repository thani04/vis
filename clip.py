import os
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
import requests
from scipy.spatial import ConvexHull
import alphashape

# ========== CONFIG ==========
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

video_base_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/Clips%20(small%20size)/"
mat_folder_base_api = "https://api.github.com/repos/nutteerabn/InfoVisual/contents/clips_folder"

# ========== FUNCTIONS ==========

def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error(f"❌ Failed to download: {url}")
        st.stop()

def list_mat_files_from_github(video_name):
    api_url = f"{mat_folder_base_api}/{video_name}"
    response = requests.get(api_url)
    if response.status_code != 200:
        st.error(f"❌ Cannot fetch .mat list from GitHub. Status code: {response.status_code}")
        st.stop()
    file_list = response.json()
    return [f["download_url"] for f in file_list if f["name"].endswith(".mat")]

def download_multiple_mats(mat_urls, temp_dir):
    mat_paths = []
    for url in mat_urls:
        file_name = os.path.basename(url)
        save_path = os.path.join(temp_dir, file_name)
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            mat_paths.append(save_path)
        else:
            st.warning(f"⚠️ Failed to download {file_name}")
    return mat_paths

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
    df['F-C score'] = 1 - (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)'] / df['Convex Area (Rolling Avg)'])
    df['F-C score'] = df['F-C score'].fillna(0)
    return df, video_frames

# ========== UI ==========
st.title("🎯 Gaze & Hull Analysis Tool")

video_names = list(video_files.keys())
selected_video = st.selectbox("เลือกวิดีโอ", video_names)

mp4_filename = video_files[selected_video]
video_url = video_base_url + mp4_filename

temp_dir = "temp_data"
os.makedirs(temp_dir, exist_ok=True)

if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

# Download and process data
with st.spinner("📥 กำลังโหลดวิดีโอและ .mat files จาก GitHub..."):
    video_path = os.path.join(temp_dir, mp4_filename)
    download_file(video_url, video_path)

    mat_urls = list_mat_files_from_github(selected_video)
    mat_paths = download_multiple_mats(mat_urls, temp_dir)

    gaze_data = load_gaze_data(mat_paths)
    df, video_frames = process_video_analysis(gaze_data, video_path)

    if df is not None:
        st.session_state.df = df
        st.session_state.video_frames = video_frames
        st.session_state.csv_path = os.path.join(temp_dir, "analysis.csv")
        df.to_csv(st.session_state.csv_path)
        st.session_state.data_processed = True
        st.session_state.current_frame = int(df.index.min())
        st.success("✅ โหลดและประมวลผลเสร็จแล้ว")

# Display
if st.session_state.data_processed:
    df = pd.read_csv(st.session_state.csv_path, index_col='Frame')
    video_frames = st.session_state.video_frames
    current_frame = st.session_state.current_frame
    min_frame, max_frame = int(df.index.min()), int(df.index.max())
    frame_increment = 10

    st.subheader("📊 Convex vs Concave Hull Area Over Time")

    new_frame = st.slider("Select Frame", min_frame, max_frame, current_frame)
    st.session_state.current_frame = new_frame

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Previous <10"):
            st.session_state.current_frame = max(min_frame, st.session_state.current_frame - frame_increment)
    with col3:
        if st.button("Next >10"):
            st.session_state.current_frame = min(max_frame, st.session_state.current_frame + frame_increment)

    current_frame = st.session_state.current_frame

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
        frame_rgb = cv2.cvtColor(video_frames[current_frame], cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {current_frame}", use_container_width=True)
        st.metric("Focus-Concentration Score", f"{df.loc[current_frame, 'F-C score']:.3f}")
