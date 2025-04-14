import os
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
import requests
import matplotlib.pyplot as plt
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
mat_api_base = "https://api.github.com/repos/nutteerabn/InfoVisual/contents/clips_folder"

# ========== DOWNLOAD ==========
def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error(f"❌ Failed to download: {url}")
        st.stop()

def list_mat_files(video_name):
    api_url = f"{mat_api_base}/{video_name}"
    res = requests.get(api_url)
    if res.status_code != 200:
        st.error("❌ Cannot fetch .mat list from GitHub.")
        st.stop()
    return [f["download_url"] for f in res.json() if f["name"].endswith(".mat")]

def download_mat_files(mat_urls, temp_dir, limit=5):
    mat_paths = []
    for url in mat_urls[:limit]:
        name = os.path.basename(url)
        path = os.path.join(temp_dir, name)
        r = requests.get(url)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)
            mat_paths.append(path)
        else:
            st.warning(f"⚠️ Failed to download {name}")
    return mat_paths

# ========== GAZE & HULL ==========
@st.cache_data
def load_gaze_data(mat_files):
    data = []
    for file in mat_files:
        mat = scipy.io.loadmat(file)
        et = mat['eyetrackRecord']
        x = et['x'][0, 0].flatten()
        y = et['y'][0, 0].flatten()
        t = et['t'][0, 0].flatten()
        valid = (x != -32768) & (y != -32768)
        x = x[valid]
        y = y[valid]
        t = t[valid] - t[valid][0]
        x_norm = x / np.max(x)
        y_norm = y / np.max(y)
        data.append((x_norm, y_norm, t))
    return data

@st.cache_resource
def process_video_analysis(gaze_data, video_path, alpha=0.007, window=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Video error")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))

    frame_nums, convex_areas, concave_areas = [], [], []
    frame_num = 0

    while True:
        ret, _ = cap.read()
        if not ret:
            break

        gaze_pts = []
        for x_norm, y_norm, t in gaze_data:
            idx = (t / 1000 * fps).astype(int)
            if frame_num in idx:
                i = np.where(idx == frame_num)[0]
                for j in i:
                    gx = int(np.clip(x_norm[j], 0, 1) * (w - 1))
                    gy = int(np.clip(y_norm[j], 0, 1) * (h - 1))
                    gaze_pts.append((gx, gy))

        if len(gaze_pts) >= 3:
            pts = np.array(gaze_pts)
            try:
                convex = ConvexHull(pts).volume
            except:
                convex = 0
            try:
                concave = alphashape.alphashape(pts, alpha)
                concave_area = concave.area if concave.geom_type == 'Polygon' else 0
            except:
                concave_area = 0

            frame_nums.append(frame_num)
            convex_areas.append(convex)
            concave_areas.append(concave_area)

        frame_num += 1

    cap.release()
    df = pd.DataFrame({
        'Frame': frame_nums,
        'Convex Area': convex_areas,
        'Concave Area': concave_areas
    })
    df.set_index('Frame', inplace=True)
    df['Convex Area (Rolling Avg)'] = df['Convex Area'].rolling(window=window, min_periods=1).mean().fillna(0)
    df['Concave Area (Rolling Avg)'] = df['Concave Area'].rolling(window=window, min_periods=1).mean().fillna(0)
    df['F-C score'] = 1 - ((df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)'])
    df['F-C score'] = df['F-C score'].fillna(0)
    return df, fps

def get_frame_at(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def plot_hull_shapes_chart(gaze_points, alpha=0.007):
    if len(gaze_points) < 3:
        st.info("Not enough points to plot hull.")
        return

    points = np.array(gaze_points)
    fig, ax = plt.subplots()

    try:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'g-', lw=2, label="Convex" if 'Convex' not in ax.get_legend_handles_labels()[1] else "")
    except:
        pass

    try:
        concave = alphashape.alphashape(points, alpha)
        if concave.geom_type == 'Polygon':
            x, y = concave.exterior.xy
            ax.plot(x, y, 'b--', lw=2, label="Concave")
    except:
        pass

    ax.set_title("Convex vs Concave Hull Shape", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=8)
    st.pyplot(fig)

# ========== APP ==========
st.title("🎯 Gaze & Hull Analysis Tool")

video_names = list(video_files.keys())
selected_video = st.selectbox("เลือกวิดีโอ", video_names)

mp4_name = video_files[selected_video]
video_url = video_base_url + mp4_name

temp_dir = "temp_data"
os.makedirs(temp_dir, exist_ok=True)

if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

with st.spinner("📥 กำลังโหลดข้อมูล..."):
    video_path = os.path.join(temp_dir, mp4_name)
    download_file(video_url, video_path)

    mat_urls = list_mat_files(selected_video)
    mat_paths = download_mat_files(mat_urls, temp_dir)
    gaze_data = load_gaze_data(mat_paths)
    df, fps = process_video_analysis(gaze_data, video_path)

    if df is not None:
        st.session_state.df = df
        st.session_state.video_path = video_path
        st.session_state.fps = fps
        st.session_state.data_processed = True
        st.session_state.current_frame = int(df.index.min())
        st.success("✅ ประมวลผลสำเร็จ!")

# ========== UI ==========
if st.session_state.data_processed:
    df = st.session_state.df
    video_path = st.session_state.video_path
    fps = st.session_state.fps
    current_frame = st.session_state.current_frame
    min_frame, max_frame = int(df.index.min()), int(df.index.max())
    frame_increment = 10

    st.subheader("📊 Convex vs Concave Hull Area Over Time")
    new_frame = st.slider("Select Frame", min_frame, max_frame, current_frame)
    st.session_state.current_frame = new_frame
    current_frame = st.session_state.current_frame

    frame = get_frame_at(video_path, current_frame)
    gaze_points_this_frame = []
    if frame is not None:
        for x_norm, y_norm, t in gaze_data:
            idx = (t / 1000 * fps).astype(int)
            indices = np.where(idx == current_frame)[0]
            for i in indices:
                gx = int(np.clip(x_norm[i], 0, 1) * (frame.shape[1] - 1))
                gy = int(np.clip(y_norm[i], 0, 1) * (frame.shape[0] - 1))
                gaze_points_this_frame.append((gx, gy))

        # 🌀 Mini chart Hull Shape
        st.markdown("### 🌀 Gaze Hull Shape (Per Frame)")
        plot_hull_shapes_chart(gaze_points_this_frame)

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Previous <10"):
            st.session_state.current_frame = max(min_frame, current_frame - frame_increment)
    with col3:
        if st.button("Next >10"):
            st.session_state.current_frame = min(max_frame, current_frame + frame_increment)

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
                range=['rgb(0, 210, 0)', 'rgb(0, 200, 255)']),
            legend=alt.Legend(title='Hull Type', orient='bottom')
        )
    ).properties(width=500, height=300)

    rule = alt.Chart(pd.DataFrame({'Frame': [current_frame]})).mark_rule(color='red').encode(x='Frame')

    col_chart, col_right = st.columns([2, 1])
    with col_chart:
        st.altair_chart(chart + rule, use_container_width=True)

    with col_right:
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {current_frame}", use_container_width=True)
        st.metric("Focus-Concentration Score", f"{df.loc[current_frame, 'F-C score']:.3f}")
