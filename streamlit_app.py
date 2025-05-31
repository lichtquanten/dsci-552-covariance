import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")
st.title("3D Covariance Explorer")

# Initialize session state for presets
if "preset_clicked" not in st.session_state:
    st.session_state.preset_clicked = False

# Shape presets
PRESETS = {
    "Sphere": {
        "var_x": 1.0,
        "var_y": 1.0,
        "var_z": 1.0,
        "cov_xy": 0.0,
        "cov_xz": 0.0,
        "cov_yz": 0.0,
        "desc": "All variances equal, no correlation. Produces a spherical cloud.",
    },
    "Cigar": {
        "var_x": 3.0,
        "var_y": 0.2,
        "var_z": 0.2,
        "cov_xy": 0.0,
        "cov_xz": 0.0,
        "cov_yz": 0.0,
        "desc": "One axis dominates. Long and thin like a cigar.",
    },
    "Pancake": {
        "var_x": 1.0,
        "var_y": 1.0,
        "var_z": 0.01,
        "cov_xy": 0.0,
        "cov_xz": 0.0,
        "cov_yz": 0.0,
        "desc": "One axis almost flat. Wide and thin like a pancake.",
    },
    "Egg": {
        "var_x": 1.0,
        "var_y": 0.8,
        "var_z": 0.5,
        "cov_xy": 0.4,
        "cov_xz": 0.2,
        "cov_yz": 0.1,
        "desc": "Moderate correlation and uneven variance. Skewed ellipsoid.",
    },
}

# Button row
st.subheader("Presets")
preset_cols = st.columns(len(PRESETS))
selected_preset = None
for i, (name, values) in enumerate(PRESETS.items()):
    if preset_cols[i].button(name):
        selected_preset = name
        st.session_state.preset_clicked = True
        for key, val in values.items():
            if key != "desc":
                st.session_state[key] = val

# Main layout
col1, col2 = st.columns([1, 3])


# Helper: Safe get from session state or fallback
def sget(name, default):
    return st.session_state.get(name, default)


# Helper: Safe bounds for covariance
def cov_bounds(v1, v2):
    return -np.sqrt(v1 * v2), np.sqrt(v1 * v2)


# Helper: Conditional covariance slider or message
def conditional_cov_slider(label, key, var1, var2, default_val):
    disabled = var1 == 0 or var2 == 0
    if disabled:
        st.slider(label, 0.0, 0.0, 0.0, disabled=True)
        st.session_state[key] = 0.0
        return 0.0
    else:
        min_val, max_val = cov_bounds(var1, var2)
        return st.slider(
            label,
            float(min_val),
            float(max_val),
            float(sget(key, default_val)),
            step=0.05,
            key=key,
        )


with col1:
    st.header("Parameters")

    var_x = st.slider("Variance X", 0.0, 5.0, sget("var_x", 1.0), step=0.1, key="var_x")
    var_y = st.slider("Variance Y", 0.0, 5.0, sget("var_y", 1.0), step=0.1, key="var_y")
    var_z = st.slider("Variance Z", 0.0, 5.0, sget("var_z", 1.0), step=0.1, key="var_z")

    cov_xy = conditional_cov_slider("Covariance XY", "cov_xy", var_x, var_y, 0.0)
    cov_xz = conditional_cov_slider("Covariance XZ", "cov_xz", var_x, var_z, 0.0)
    cov_yz = conditional_cov_slider("Covariance YZ", "cov_yz", var_y, var_z, 0.0)

    n_points = st.slider(
        "Number of Points", 100, 2000, sget("n_points", 1000), step=100, key="n_points"
    )

    if selected_preset:
        st.markdown(
            f"**Shape:** {selected_preset}  \n{PRESETS[selected_preset]['desc']}"
        )

# Covariance matrix and mean
cov_matrix = np.array(
    [[var_x, cov_xy, cov_xz], [cov_xy, var_y, cov_yz], [cov_xz, cov_yz, var_z]]
)
mean = np.zeros(3)


# Singularity explanation
def explain_singularity(cov):
    reasons = []
    variances = np.diag(cov)
    for i, v in enumerate(variances):
        if np.isclose(v, 0):
            reasons.append(f"- Variance of axis {chr(88 + i)} is zero")

    def is_perfect_corr(i, j):
        if np.isclose(cov[i, i], 0) or np.isclose(cov[j, j], 0):
            return False
        corr = cov[i, j] / (np.sqrt(cov[i, i]) * np.sqrt(cov[j, j]))
        return np.isclose(abs(corr), 1.0)

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if is_perfect_corr(i, j):
            reasons.append(
                f"- Axes {chr(88 + i)} and {chr(88 + j)} are perfectly correlated"
            )

    return reasons


# Check for positive definiteness
try:
    np.linalg.cholesky(cov_matrix)
    singular = False
except np.linalg.LinAlgError:
    singular = True
    reasons = explain_singularity(cov_matrix)
    with col2:
        st.warning(
            "⚠️ The covariance matrix is **singular** or not positive semi-definite.\n\n"
            "**Why?**\n"
            + (
                "\n".join(reasons)
                if reasons
                else "- Linear dependence or rounding errors"
            )
            + "\n\nPoints will still be shown, but the ellipsoid cannot be rendered."
        )

# Sample from the distribution
points = np.random.multivariate_normal(
    mean, cov_matrix, size=n_points, check_valid="warn", tol=1e-8
)
x, y, z = points.T


# Ellipsoid generator (exaggerated scale)
def generate_ellipsoid(mean, cov, scale=3.5, n_points=40):
    vals, vecs = np.linalg.eigh(cov)
    radii = np.sqrt(np.maximum(vals, 0)) * scale
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack((x, y, z), axis=-1)
    ellipsoid = sphere @ np.diag(radii) @ vecs.T + mean
    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]


# Plot
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=3, color="blue", opacity=0.6),
        name="Sampled Points",
    )
)

if not singular:
    ex, ey, ez = generate_ellipsoid(mean, cov_matrix)
    fig.add_trace(
        go.Surface(
            x=ex,
            y=ey,
            z=ez,
            opacity=0.25,
            colorscale="Reds",
            showscale=False,
            name="Covariance Ellipsoid",
        )
    )

fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data",  # important to preserve true shape
    ),
    width=1000,
    height=800,
    margin=dict(l=0, r=0, b=0, t=0),
)

with col2:
    st.plotly_chart(fig, use_container_width=True)
