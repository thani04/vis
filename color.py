import streamlit as st

# Set page layout
st.set_page_config(layout="wide")

# Sidebar Navigation
st.sidebar.title("📚 Navigation")
st.sidebar.markdown("""
### 🧭 Introduction
- [1. What Captures Attention?](#what-captures-attention)
- [2. How Do We Measure Focus?](#how-do-we-measure-focus)
- [3. Focus-Concentration (F-C) Score](#focus-concentration-f-c-score)
- [4. Visual Examples of Focus](#visual-examples-of-focus)

### 📈 Visualization
- [5. Focus-Concentration Visualization](#focus-concentration-visualization)
""", unsafe_allow_html=True)

# Main Title
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

# Color themes
COLOR_GROUP1 = "#ECF0F1"
COLOR_GROUP2 = "#F8F3EF"

# SECTION 1: What Captures Attention
st.markdown("<h3 id='what-captures-attention'>📌 What Captures Attention?</h3>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background-color: {COLOR_GROUP1}; padding: 20px; border-radius: 10px;">
    <p>
    Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?
    </p>
    <p>
    This visualization explores how viewers engage with a video by examining <strong>where and how they focus their attention</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# SECTION 2: How Do We Measure Focus
st.markdown("<h3 id='how-do-we-measure-focus'>📐 How Do We Measure Focus?</h3>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background-color: {COLOR_GROUP1}; padding: 20px; border-radius: 10px;">
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
        caption="📊 Diagram: Convex vs Concave Hulls", width=320
    )
with col2:
    st.image(
        "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif",
        caption="🎥 Real Example: Gaze Boundaries Over Time"
    )

# SECTION 3: F-C Score
st.markdown("<h3 id='focus-concentration-f-c-score'>📊 Focus-Concentration (F-C) Score</h3>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px;">
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg" 
             alt="F-C Score Formula"
             style="height: 100px; border-radius: 10px;"/>
        <p><em>🧮 Area calculation using a rolling average across the last 20 frames</em></p>
    </div>
</div>
""", unsafe_allow_html=True)

# SECTION 4: Visual Examples
st.markdown("<h3 id='visual-examples-of-focus'>🎥 Visual Examples of Focus</h3>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px;">
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

st.markdown(f"""
</div>
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px; margin-top: 1em;">
    <p>You’ll see this visualized dynamically in the graph and overlays as you explore different segments of the video.</p>
</div>
""", unsafe_allow_html=True)

# SECTION 5 is to be added here later: Focus-Concentration Visualization
