import os
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
import scipy.io
from io import BytesIO
from scipy.spatial import ConvexHull
import alphashape

st.set_page_config(page_title="Gaze Hull Visualizer", layout="wide")

# ----------------------------
# INTRO SECTION
# ----------------------------
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

with st.expander("\ud83d\udccc Goal of This Visualization", expanded=True):
    st.markdown("""
    _Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?_

    The goal of this visualization is to understand how viewers engage with a video by examining **where and how they focus their attention**. By comparing the areas where viewers look (represented by **convex and concave hulls**), the visualization highlights whether their attention stays focused on a specific part of the video or shifts around.

    Ultimately, this helps us uncover **patterns of focus and exploration**, providing insights into how viewers interact with different elements of the video.
    """)

with st.expander("\ud83d\udcd0 Explain Convex and Concave Concept"):
    st.markdown("""
    To analyze visual attention, we enclose gaze points with geometric boundaries:

    - **Convex Hull** wraps around all gaze points to show the overall extent of where viewers looked.
    - **Concave Hull** creates a tighter boundary that closely follows the actual shape of the gaze pattern, adapting to gaps and contours in the data.

    \ud83d\udd20 **The difference in area between them reveals how dispersed or concentrated the viewers’ gaze is.**
    """)

with st.expander("\ud83d\udcca Focus-Concentration (F-C) Score"):
    st.markdown("""
    The **Focus Concentration Score (FCS)** quantifies how focused or scattered a viewer’s attention is during the video:

    - A score **close to 1** → gaze is tightly grouped → **high concentration**.
    - A score **much lower than 1** → gaze is more spread out → **lower concentration / more exploration**.

    It helps to measure whether attention is **locked onto a specific spot** or **wandering across the frame**.
    """)

with st.expander("\ud83c\udfa5 Example: High vs Low F-C Score"):
    st.markdown("**High F-C Score**: The viewer’s gaze remains focused in one tight area, suggesting strong interest or attention.")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif")

    st.markdown("**Low F-C Score**: The gaze is scattered, moving across many regions of the screen, indicating exploration or distraction.")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif")

    st.markdown("You can observe this difference visually in the graph and video overlays as you explore different frames.")

# ----------------------------
# CONFIG
# ----------------------------
video_files = {
    "APPAL_2a": "APPAL_2a_hull_area.mp4",
    "FOODI_2a": "FOODI_2a_hull_area.mp4",
    "MARCH_12a": "MARCH_12a_hull_area.mp4",
    "NANN_3a": "NANN_3a_hull_area.mp4",
    "SHREK_3a": "SHREK_3a_hull_area.mp4",
    "SIMPS_19a": "SIMPS_19a_hull_area.mp4",
    "SIMPS_9a": "SIMPS_9a_hull_area.mp4",
    "SUND_36a_POR": "SUND_36a_POR_hull_area.mp4",
}

base_video_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/processed%20hull%20area%20overlay/"
user = "nutteerabn"
repo = "InfoVisual"
clips_folder = "clips_folder"

# ----------------------------
# HELPERS
# ----------------------------
@st.cache_data(show_spinner=False)
def list_mat_files(user, repo, folder):
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    files = r.json()
    return [f["name"] for f in files if f["name"].endswith(".mat")]

@st.cache_data(show_spinner=True)
def load_gaze_data(user, repo, folder):
    mat_files = list_mat_files(user, repo, folder)
    gaze_data = []
    for file in mat_files:
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{folder}/{file}"
        res = requests.get(raw_url)
        if res.status_code == 200:
            mat = scipy.io.loadmat(BytesIO(res.content))
            record = mat['eyetrackRecord']
            x = record['x'][0, 0].flatten()
            y = record['y'][0, 0].flatten()
            t = record['t'][0, 0].flatten()
            valid = (x != -32768) & (y != -32768)
            if valid.sum() > 0:
                gaze_data.append({
                    'x': x[valid] / np.max(x[valid]),
                    'y': y[valid] / np.max(y[valid]),
                    't': t[valid] - t[valid][0]
                })
    return [(d['x'], d['y'], d['t']) for d in gaze_data]

@st.cache_data(show_spinner=True)
def download_video(video_url, save_path):
    r = requests.get(video_url)
    with open(save_path, "wb") as f:
        f.write(r.content)

@st.cache_data(show_spinner=True)
def analyze_gaze(gaze_data, video_path, alpha=0.007, window=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames, convex, concave, images = [], [], [], []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        points = []
        for x, y, t in gaze_data:
            idx = (t / 1000 * fps).astype(int)
            if i in idx:
                pts = np.where(idx == i)[0]
                for p in pts:
                    px = int(np.clip(x[p], 0, 1) * (w - 1))
                    py = int(np.clip(y[p], 0, 1) * (h - 1))
                    points.append((px, py))

        if len(points) >= 3:
            arr = np.array(points)
            try:
                convex_area = ConvexHull(arr).volume
            except:
                convex_area = 0
            try:
                shape = alphashape.alphashape(arr, alpha)
                concave_area = shape.area if shape.geom_type == 'Polygon' else 0
            except:
                concave_area = 0
        else:
            convex_area = concave_area = 0

        frames.append(i)
        convex.append(convex_area)
        concave.append(concave_area)
        images.append(frame)
        i += 1

    cap.release()

    df = pd.DataFrame({
        'Frame': frames,
        'Convex Area': convex,
        'Concave Area': concave
    }).set_index('Frame')
    df['Convex Area (Rolling)'] = df['Convex Area'].rolling(window, min_periods=1).mean()
    df['Concave Area (Rolling)'] = df['Concave Area'].rolling(window, min_periods=1).mean()
    df['F-C score'] = 1 - (df['Convex Area (Rolling)'] - df['Concave Area (Rolling)']) / df['Convex Area (Rolling)']
    df['F-C score'] = df['F-C score'].fillna(0)

    return df, images

# ----------------------------
# UI: VIDEO SELECTION + PROCESSING
# ----------------------------
st.title("\ud83c\udf1f Stay Focused or Float Away? : Focus-Concentration Analysis")

selected_video = st.selectbox("\ud83c\udfa5 Select a video", list(video_files.keys()))

if selected_video:
    st.video(base_video_url + video_files[selected_video])

    folder = f"{clips_folder}/{selected_video}"
    with st.spinner("Running analysis..."):
        gaze = load_gaze_data(user, repo, folder)

        video_filename = f"{selected_video}.mp4"
        if not os.path.exists(video_filename):
            download_video(base_video_url + video_files[selected_video], video_filename)

        df, frames = analyze_gaze(gaze, video_filename)
        if not df.empty:
            st.session_state.df = df
            st.session_state.frames = frames
            st.session_state.frame = int(df.index.min())
        else:
            st.warning("\u26a0\ufe0f No valid data found in the .mat files for this video.")

# ----------------------------
# DISPLAY
# ----------------------------
if "df" in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df
    frames = st.session_state.frames
    frame = st.slider("\ud83c\udf9e\ufe0f Select Frame", int(df.index.min()), int(df.index.max()), st.session_state.frame)

    col1, col2 = st.columns([2, 1])
    with col1:
        data = df.reset_index().melt(id_vars="Frame", value_vars=[
            "Convex Area (Rolling)", "Concave Area (Rolling)"
        ], var_name="Metric", value_name="Area")
        chart = alt.Chart(data).mark_line().encode(
            x="Frame:Q", y="Area:Q", color="Metric:N"
        ).properties(width=600, height=300)
        rule = alt.Chart(pd.DataFrame({'Frame': [frame]})).mark_rule(color='red').encode(x='Frame')
        st.altair_chart(chart + rule, use_container_width=True)

    with col2:
        rgb = cv2.cvtColor(frames[frame], cv2.COLOR_BGR2RGB)
        st.image(rgb, caption=f"Frame {frame}", use_container_width=True)
        st.metric("F-C Score", f"{df.loc[frame, 'F-C score']:.3f}")
