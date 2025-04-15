
import os
import cv2
import altair as alt
import streamlit as st
from utils import load_gaze_data, download_video, analyze_gaze

st.set_page_config(layout="wide")
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

# กำหนดโทนสีแบบจับคู่
COLOR_GROUP1 = "#DCDCDC"   
COLOR_GROUP2 = "#F3ECCF"   

# SECTION 1: Hook
st.markdown(f"""
<div style="background-color: {COLOR_GROUP1}; padding: 20px; border-radius: 10px;">
    <h3>📌 What Captures Attention?</h3>
    <p>
    Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?
    </p>
    <p>
    This visualization explores how viewers engage with a video by examining <strong>where and how they focus their attention</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# SECTION 2: Hull Concepts
st.markdown(f"""
<div style="background-color: {COLOR_GROUP1}; padding: 20px; border-radius: 10px;">
    <h3>📐 How Do We Measure Focus?</h3>
    <p>
    We use geometric shapes to visualize how tightly the viewer’s gaze is grouped:
    </p>
    <ul>
        <li><strong>Convex Hull</strong>: Encloses all gaze points loosely.</li>
        <li><strong>Concave Hull</strong>: Follows the actual shape of gaze, revealing true focus.</li>
    </ul>
    <p>👉 The <strong>difference in area</strong> between the two tells us how spread out or concentrated the gaze is.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.image(
        "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_image.jpg",
        caption="📊 Diagram: Convex vs Concave Hulls"
    )
with col2:
    st.image(
        "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif",
        caption="🎥 Real Example: Gaze Boundaries Over Time"
    )

# SECTION 3: F-C Score
st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px;">
    <h3>📊 Focus-Concentration (F-C) Score</h3>
</div>
""", unsafe_allow_html=True)

st.image(
    "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg",
    caption="🧮 Area calculation using a rolling average across the last 20 frames", width=900
)

st.markdown(f"""
<div style="background-color: #FFFAFA; padding: 15px; border-left: 5px solid #F3ECCF; margin-top: 10px;">
    <ul>
        <li><strong>Close to 1</strong> → tight gaze cluster → <span style="color:#2e7d32;"><strong>high concentration</strong></span></li>
        <li><strong>Much lower than 1</strong> → scattered gaze → <span style="color:#d32f2f;"><strong>low concentration</strong></span></li>
    </ul>
    <p>This metric reveals whether attention is <strong>locked in</strong> or <strong>wandering</strong>.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# SECTION 4: Visual Examples
st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px;">
    <h3>🎥 Visual Examples of Focus</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### High F-C Score")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif")
    st.caption("Gaze remains tightly grouped in one region.")
with col2:
    st.markdown("### Low F-C Score")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif")
    st.caption("Gaze jumps around, showing exploration or distraction.")

st.markdown("""
<p style="margin-top: 1em;">
You’ll see this visualized dynamically in the graph and overlays as you explore different segments of the video.
</p>
""", unsafe_allow_html=True)


# st.set_page_config(page_title="Gaze Hull Visualizer", layout="wide")

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
