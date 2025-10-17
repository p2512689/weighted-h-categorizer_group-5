import streamlit as st
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import random

# Set page configuration
st.set_page_config(page_title="Convex Hull Visualizer", layout="wide")

def weighted_h_categorizer(A, R, w, eps=sys.float_info.epsilon, max_iteration=2000):
    """
    Weighted h-categorizer with intrinsic weights on arguments.
    """
    fh = [{a: w.get(a, 1.0) for a in A}]
    i = 1
    while True:
        fh.append({})
        for a in A:
            attackers = [b for (b, t) in R if t == a]
            total_attack = sum(fh[i-1][b] for b in attackers)
            fh[i][a] = w.get(a, 1.0) / (1 + total_attack)
        
        diff = sum(abs(fh[i][a] - fh[i-1][a]) for a in A) / len(A)
        if diff < eps:
            break
        if i >= max_iteration:
            break
        i += 1
    return fh[-1]

def generate_random_weights(A):
    return {a: random.random() for a in A}

def compute_acceptability_vectors(A, R, n_samples=100000):
    X = np.zeros((n_samples, len(A)))
    for i in range(n_samples):
        w = generate_random_weights(A)
        deg = weighted_h_categorizer(A, R, w)
        X[i] = np.array([deg[a] for a in A])
    return X

def compute_convex_hull(X):
    n_samples, dim = X.shape
    hull = None
    if dim > 1:
        hull = ConvexHull(X)
    return hull

def visualize_convex_hull_1d(X, A):
    xmin, xmax = np.min(X), np.max(X)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(X, np.zeros_like(X), color="blue", label="Points", alpha=0.6, s=20)
    ax.plot([xmin, xmax], [0, 0], 'o', color="red", markersize=10, label="Convex Hull Bounds")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Acceptability of {list(A)[0]}")
    ax.set_title("1D Convex Hull Visualization")
    ax.legend()
    return fig

def visualize_convex_hull_2d(X, A, hull=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], color="blue", alpha=0.4, label="Points", s=10)
    
    if hull is not None:
        for simplex in hull.simplices:
            ax.plot(X[simplex, 0], X[simplex, 1], "r-", linewidth=2)
        ax.fill(X[hull.vertices, 0], X[hull.vertices, 1], "red", alpha=0.2, label="Convex Hull")
    else:
        ax.set_title("Not enough points for a convex hull")
    
    args_list = list(A)
    ax.set_xlabel(f"Acceptability of {args_list[0]}")
    ax.set_ylabel(f"Acceptability of {args_list[1]}")
    ax.set_title("2D Convex Hull Visualization")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig

def visualize_convex_hull_3d(X, A, hull=None, axis_indexes=[0, 1, 2]):
    args_list = list(A)
    
    # Extract only the 3 dimensions we want to visualize
    X_3d = X[:, axis_indexes]
    
    xs = X_3d[:, 0]
    ys = X_3d[:, 1]
    zs = X_3d[:, 2]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='green', alpha=0.3, s=10, label="Points")
    
    if hull is not None:
        # Compute convex hull for the 3D projection
        try:
            hull_3d = ConvexHull(X_3d)
            hull_collection = Poly3DCollection(
                X_3d[hull_3d.simplices],
                alpha=0.25,
                facecolor='cyan',
                linewidths=1,
                edgecolors='k'
            )
            ax.add_collection3d(hull_collection)
            
            hull_vertices_3d = X_3d[hull_3d.vertices]
            ax.plot(hull_vertices_3d[:, 0], hull_vertices_3d[:, 1], hull_vertices_3d[:, 2],
                    'r.', markersize=8, label="Hull Vertices")
        except Exception as e:
            st.warning(f"Could not compute 3D hull for selected axes: {e}")
    
    ax.set_title("3D Convex Hull Visualization")
    ax.set_xlabel(f'Acceptability of {args_list[axis_indexes[0]]}')
    ax.set_ylabel(f'Acceptability of {args_list[axis_indexes[1]]}')
    ax.set_zlabel(f'Acceptability of {args_list[axis_indexes[2]]}')
    ax.legend()
    
    return fig

# Streamlit App
st.title("🔷 Weighted H-Categorizer Convex Hull Visualizer")
st.markdown("Visualize the convex hull of acceptability degrees for argumentation frameworks")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Arguments input
st.sidebar.subheader("Arguments")
args_input = st.sidebar.text_input(
    "Enter arguments (comma-separated)", 
    value="A, B, C",
    help="Example: A, B, C, D"
)
A = set([arg.strip() for arg in args_input.split(",") if arg.strip()])

# Attack relations input
st.sidebar.subheader("Attack Relations")
st.sidebar.markdown("Enter attack relations in format: `attacker -> target`")
relations_input = st.sidebar.text_area(
    "Attack Relations (one per line)",
    value="B -> A\nC -> A\nC -> B\nA -> C",
    height=150,
    help="Example: B -> A means B attacks A"
)

# Parse relations
R = set()
if relations_input:
    for line in relations_input.strip().split("\n"):
        if "->" in line:
            parts = line.split("->")
            if len(parts) == 2:
                attacker = parts[0].strip()
                target = parts[1].strip()
                if attacker in A and target in A:
                    R.add((attacker, target))

# Number of samples
n_samples = st.sidebar.slider("Number of samples", 100, 10000, 1000, 100)

# 3D axis selection (only shown if 3+ arguments)
axis_indexes = None
if len(A) >= 3:
    st.sidebar.subheader("3D Visualization Settings")
    args_list = sorted(list(A))
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        x_axis = st.selectbox("X-axis", args_list, index=0)
    with col2:
        y_axis = st.selectbox("Y-axis", args_list, index=min(1, len(args_list)-1))
    with col3:
        z_axis = st.selectbox("Z-axis", args_list, index=min(2, len(args_list)-1))
    
    axis_indexes = [args_list.index(x_axis), args_list.index(y_axis), args_list.index(z_axis)]

# Display current configuration
st.header("Current Configuration")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Arguments")
    st.write(sorted(list(A)))

with col2:
    st.subheader("Attack Relations")
    if R:
        for attacker, target in sorted(R):
            st.write(f"• {attacker} → {target}")
    else:
        st.write("No attack relations defined")

# Compute and visualize button
if st.button("Generate Convex Hull", type="primary"):
    if len(A) == 0:
        st.error("Please define at least one argument")
    else:
        with st.spinner("Computing acceptability vectors..."):
            # Compute acceptability vectors
            X = compute_acceptability_vectors(A, R, n_samples)
            
            # Compute convex hull
            hull = compute_convex_hull(X) if len(A) > 1 else None
            
            st.success(f"Computed {n_samples} acceptability vectors for {len(A)} arguments")
            
            # Visualize based on dimension
            st.header("Convex Hull Visualization")
            
            if len(A) == 1:
                fig = visualize_convex_hull_1d(X, A)
                st.pyplot(fig)
                
            elif len(A) == 2:
                fig = visualize_convex_hull_2d(X, A, hull)
                st.pyplot(fig)
                
            elif len(A) >= 3:
                fig = visualize_convex_hull_3d(X, A, hull, axis_indexes)
                st.pyplot(fig)
            
            # Display statistics
            st.header("Statistics")
            col1, col2, col3 = st.columns(3)
            
            args_list = sorted(list(A))
            for idx, arg in enumerate(args_list):
                with [col1, col2, col3][idx % 3]:
                    st.metric(
                        f"Arg {arg} - Mean Acceptability",
                        f"{X[:, idx].mean():.4f}",
                        help=f"Standard deviation: {X[:, idx].std():.4f}"
                    )

# Information section
with st.expander("ℹ️ About this tool"):
    st.markdown("""
    This tool implements the **Weighted H-Categorizer** algorithm for abstract argumentation frameworks.
    
    **How it works:**
    1. Define a set of arguments
    2. Specify attack relations between arguments
    3. The algorithm computes acceptability degrees by randomly sampling intrinsic weights
    4. The convex hull of all possible acceptability vectors is visualized
    
    **Formula:**
    ```
    HC_{k+1}(a) = w(a) / (1 + sum(HC_k(b) for b in Att(a)))
    ```
    
    **Reference:** [Amgoud & Ben-Naim, AIJ 2018](https://www.irit.fr/~Leila.Amgoud/AIJ-V0.pdf)
    """)