import streamlit as st

st.set_page_config(layout="wide")

# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown("""
### 📚 Introduction
- [1. What Captures Attention?](#what-captures-attention)
- [2. How Do We Measure Focus?](#how-do-we-measure-focus)
- [3. Focus-Concentration (F-C) Score](#focus-concentration-f-c-score)
- [4. Visual Examples of Focus](#visual-examples-of-focus)

### 📊 Visualization
- [5. Focus-Concentration Visualization](#focus-concentration-visualization)
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

# ---------------- Section 1 ----------------
st.markdown("<h3 id='what-captures-attention'>📌 What Captures Attention?</h3>", unsafe_allow_html=True)
st.markdown("""
<blockquote style="font-size:1.2em; font-style: italic;">
“Is the viewer’s attention firmly focused on key moments, or does it float?”
</blockquote>
This visualization explores how viewers engage with a video by examining **where and how they focus their attention**.
""", unsafe_allow_html=True)

# ---------------- Section 2 ----------------
st.markdown("<h3 id='how-do-we-measure-focus'>📐 How Do We Measure Focus?</h3>", unsafe_allow_html=True)
st.markdown("""
We use geometric shapes to visualize how tightly the viewer’s gaze is grouped:

- **Convex Hull**: Encloses all gaze points loosely.
- **Concave Hull**: Follows the actual shape of gaze, revealing true focus.

👉 The **difference in area** tells us how spread out or concentrated the gaze is.
""")

col1, col2 = st.columns(2)
with col1:
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_image.jpg", caption="📊 Convex vs Concave Hulls")
with col2:
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif", caption="🎥 Gaze Boundaries Over Time")

# ---------------- Section 3 ----------------
st.markdown("<h3 id='focus-concentration-f-c-score'>📊 Focus-Concentration (F-C) Score</h3>", unsafe_allow_html=True)
st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg", caption="🧮 Rolling average area over 20 frames", use_column_width=True)
st.markdown("""
The **F-C Score** helps quantify gaze behavior:
- **Close to 1** → tightly grouped gaze → **high concentration**
- **Much lower than 1** → scattered gaze → **low concentration / exploration**
""")

# ---------------- Section 4 ----------------
st.markdown("<h3 id='visual-examples-of-focus'>🎥 Visual Examples of Focus</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**High F-C Score**")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif")
with col2:
    st.markdown("**Low F-C Score**")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif")

# ---------------- Section 5: Visualization ----------------
st.markdown("<h3 id='focus-concentration-visualization'>📊 Focus-Concentration Visualization</h3>", unsafe_allow_html=True)
st.markdown("_This is where your graph/video overlay will go_")
