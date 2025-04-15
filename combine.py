import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import requests
import altair as alt
import scipy.io
from io import BytesIO
from scipy.spatial import ConvexHull
import alphashape

# Page config must go first
st.set_page_config(page_title="Gaze Hull Visualizer", layout="wide")

# Top header image
st.image("conclip/Image.jpeg", use_container_width=True)

# Tabs
intro_tab, concept_tab, score_tab, example_tab = st.tabs([
    "1. Visualization Goal",
    "2. Convex & Concave Hull",
    "3. F-C Score",
    "4. Example: High vs Low F-C Score"
])

with intro_tab:
    st.subheader("🎯 Goal of This Visualization")
    st.markdown("""
    <blockquote style="font-size: 1.3em; text-align: center; font-weight: bold; font-style: italic; border-left: 4px solid #999; padding-left: 1em; margin: 1.5em 0;">
        “Is the viewer’s attention firmly focused on key moments,<br>
        or does it float, drifting between different scenes in search of something new?”
    </blockquote>
    """, unsafe_allow_html=True)
    st.write("""
    This visualization aims to reveal how viewers focus their gaze while watching video content. It analyzes where and how users visually engage, comparing convex and concave hulls over time to detect focused or scattered attention.
    """)

with concept_tab:
    st.subheader("📐 Convex & Concave Hull Concept")
    st.write("To analyze visual attention, we enclose gaze points with geometric boundaries:")
    st.markdown("""
    <div style='padding: 0.5em; background-color: #e6f0ff; border-left: 4px solid #1e88e5; margin: 1em 0'>
        <ul style='margin: 0; padding-left: 1.2em;'>
            <li><strong>Convex Hull</strong>: Shows the outer boundary of all gaze points, like stretching a rubber band around them.</li>
            <li><strong>Concave Hull</strong>: Fits tightly around actual gaze clusters, capturing internal gaps and curves.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("The difference between their areas reflects how focused or spread the gaze is.")

with score_tab:
    st.subheader("📊 Focus Concentration (F-C) Score")
    st.write("FCS quantifies attention dispersion using the ratio of Concave to Convex hull areas.")
    st.markdown("""
    <div style='padding: 0.7em; background-color: #f0f8ff; border-left: 5px solid #1976d2; margin: 1em 0; font-size: 1.05em'>
        <ul style='margin: 0; padding-left: 1.2em;'>
            <li><strong>Score close to 1.0</strong> → Gaze is concentrated → <span style="color:#2e7d32;"><strong>High focus</strong></span></li>
            <li><strong>Score much lower</strong> → Gaze is dispersed → <span style="color:#d32f2f;"><strong>Exploration</strong></span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with example_tab:
    st.subheader("🎬 Example: High vs Low F-C Score")
    st.markdown("""
    <div style='padding: 0.7em; background-color: #f9f9f9; border-left: 5px solid #90caf9; margin: 1em 0; font-size: 1.05em'>
        <ul style='margin: 0; padding-left: 1.2em;'>
            <li><strong>High F-C Score</strong>: Viewer fixates on a small region → <span style='color:#2e7d32'><strong>Strong interest</strong></span></li>
            <li><strong>Low F-C Score</strong>: Gaze shifts across the screen → <span style='color:#c62828'><strong>Exploratory behavior</strong></span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif", caption="High F-C Score", use_column_width=True)
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif", caption="Low F-C Score", use_column_width=True)

# Optional sections after the tabs
st.markdown("---")
st.subheader("5. Graph: Focus-Score over Time")
st.write("📈 This section will include line charts comparing Convex vs Concave Hull areas over time, and their derived F-C Score.")

# st.subheader("6. Summary Insight")
# st.write("Summarize insights from graph or patterns observed across different video segments.")


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

base_video_url = "https://github.com/nutteerabn/InfoVisual/tree/main/processed%20hull%20area%20overlay"
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
# UI
# ----------------------------
st.title("🎯 Stay Focused or Float Away? : Focus-Concentration Analysis")

selected_video = st.selectbox("🎬 Select a video", list(video_files.keys()))

if selected_video:
    st.video(base_video_url + video_files[selected_video])

    folder = f"{clips_folder}/{selected_video}"
    with st.spinner("Running analysis..."):
        gaze = load_gaze_data(user, repo, folder)

        video_filename = f"{selected_video}.mp4"
        if not os.path.exists(video_filename):
            download_video(base_video_url + video_files[selected_video], video_filename)

        df, frames = analyze_gaze(gaze, video_filename)
        st.session_state.df = df
        st.session_state.frames = frames
        st.session_state.frame = int(df.index.min())

# ----------------------------
# Results
# ----------------------------
if "df" in st.session_state:
    df = st.session_state.df
    frames = st.session_state.frames
    frame = st.slider("🎞️ Select Frame", int(df.index.min()), int(df.index.max()), st.session_state.frame)

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
