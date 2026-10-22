"""
Parallel: High-Performance GPU & CPU Image Processing Engine and Benchmarking Dashboard
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image
import tempfile

from components.runner import get_system_info, process_image, run_benchmark
from components.charts import build_operation_time_chart, build_scaling_chart

# Page configuration
st.set_page_config(
    page_title="Parallel - GPU Computing & Benchmarking",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Charcoal + Cyan/Blue Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Metrics and cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00e5ff;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #58a6ff;
        margin-top: 4px;
    }
    
    /* Tech badges */
    .badge-cuda {
        background-color: rgba(0, 196, 159, 0.15);
        border: 1px solid #00c49f;
        color: #00c49f;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-cpu {
        background-color: rgba(0, 168, 232, 0.15);
        border: 1px solid #00a8e8;
        color: #00a8e8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Fetch verified hardware info
sys_info = get_system_info()

# Sidebar Navigation
st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0;">
    <h1 style="color: #00e5ff; margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em;">⚡ PARALLEL</h1>
    <p style="color: #8b949e; font-size: 0.85rem; margin-top: 4px;">Harness parallel computing. Measure real performance.</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Image Lab", "Performance Lab", "Results Explorer", "Documentation"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Environment Status")
st.sidebar.markdown(f"**CPU:** `{sys_info.get('cpu_model', 'Unknown')}`")
st.sidebar.markdown(f"**GPU:** `{sys_info.get('gpu_model', 'N/A')}`")
cuda_active = sys_info.get("cuda_available", False)
if cuda_active:
    st.sidebar.markdown('<span class="badge-cuda">CUDA ENGINE ACTIVE</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="badge-cpu">CPU ENGINE ACTIVE (CPU-ONLY)</span>', unsafe_allow_html=True)

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "Overview":
    st.title("System Overview & Hardware Profile")
    st.markdown("Parallel demonstrates the practical and mathematical differences between CPU and GPU execution on high-density image-processing kernels.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CPU Toolchain</div>
            <div class="metric-value">C++17</div>
            <div class="metric-sub">{sys_info.get('cpu_model', 'Standard Host CPU')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">GPU Backend</div>
            <div class="metric-value">{'CUDA 12/13' if cuda_active else 'Disabled'}</div>
            <div class="metric-sub">{sys_info.get('gpu_model', 'CPU-Only Mode')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        csv_file = "results/benchmark.csv"
        runs = 0
        if os.path.exists(csv_file):
            try:
                runs = len(pd.read_csv(csv_file))
            except Exception:
                runs = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Benchmark Records</div>
            <div class="metric-value">{runs}</div>
            <div class="metric-sub">Recorded in CSV</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Algorithms</div>
            <div class="metric-value">4</div>
            <div class="metric-sub">Gray, Blur, Sharpen, Sobel</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Execution Pipeline Architecture")
    st.markdown("""
    The application architecture guarantees that the graphical interface communicates directly with the compiled native C++ engine (`bin/parallel.exe`):
    - **No Mock Python Algorithms:** Algorithms are written in C++17 (`src/cpu/`) and NVIDIA CUDA C++ (`src/cuda/`).
    - **Independent Timing:** Wall-clock, GPU kernel execution, and memory transfer (H2D / D2H) times are isolated and verified.
    - **Strict Correctness:** Image outputs are compared pixel-by-pixel with Mean Absolute Error tolerances.
    """)

# ==========================================
# PAGE 2: IMAGE LAB
# ==========================================
elif page == "Image Lab":
    st.title("Image Processing Lab")
    st.markdown("Execute individual image transformations using the native compiled engine.")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Input Configuration")
        source_mode = st.radio("Image Source", ["Preloaded Sample", "Upload Image"])

        input_path = ""
        if source_mode == "Preloaded Sample":
            samples = {
                "Sample Landscape (1080p)": "samples/sample.png",
                "Geometric Checkerboard (512x512)": "samples/checkerboard.png"
            }
            choice = st.selectbox("Select Sample", list(samples.keys()))
            input_path = samples[choice]
        else:
            uploaded = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if uploaded:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file.write(uploaded.read())
                temp_file.close()
                input_path = temp_file.name

        st.markdown("#### Operation & Parameters")
        op = st.selectbox("Operation", ["grayscale", "blur", "sharpen", "sobel"])

        k_size = 3
        sigma = 1.0
        if op == "blur":
            k_size = st.slider("Gaussian Kernel Size", 3, 11, 3, step=2)
            sigma = st.slider("Gaussian Sigma", 0.5, 5.0, 1.0, step=0.1)

        backend_options = ["cpu"]
        if cuda_active:
            backend_options.append("cuda")
        backend = st.selectbox("Execution Backend", backend_options)

        run_btn = st.button("🚀 Run Image Processing", use_container_width=True, type="primary")

    with col_right:
        if input_path and os.path.exists(input_path):
            in_img = Image.open(input_path)
            st.markdown(f"**Source Dimensions:** {in_img.width}x{in_img.height} ({len(in_img.getbands())} channels)")

            if run_btn:
                output_path = "results/dashboard_output.png"
                with st.spinner(f"Processing on {backend.upper()} backend..."):
                    try:
                        res = process_image(input_path, output_path, op, backend, k_size, sigma)
                        time_ms = res.get("execution_time_ms", 0.0)

                        st.success(f"Processing completed successfully in {time_ms:.3f} ms!")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(in_img, caption="Original Input", use_container_width=True)
                        with col_b:
                            out_img = Image.open(output_path)
                            st.image(out_img, caption=f"Processed ({op.capitalize()} - {backend.upper()})", use_container_width=True)

                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="💾 Download Processed Image",
                                data=f.read(),
                                file_name=f"parallel_{op}_{backend}.png",
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
            else:
                st.image(in_img, caption="Original Image Preview", use_container_width=True)

# ==========================================
# PAGE 3: PERFORMANCE LAB
# ==========================================
elif page == "Performance Lab":
    st.title("Performance & Benchmarking Lab")
    st.markdown("Benchmark execution latency and speedup using high-precision timers.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Benchmark Configuration")
        bench_op = st.selectbox("Operation Suite", ["all", "grayscale", "blur", "sharpen", "sobel"])
        bench_backend = "cpu" if not cuda_active else st.selectbox("Backend Mode", ["cpu", "cuda", "both"])
        iterations = st.number_input("Repetitions per test", min_value=5, max_value=100, value=15, step=5)
        warmup = st.number_input("Warmup repetitions", min_value=1, max_value=10, value=2, step=1)

        run_bench_btn = st.button("⚡ Run Benchmarking Suite", type="primary", use_container_width=True)

    with col2:
        if run_bench_btn:
            with st.spinner("Executing benchmark iterations across resolutions..."):
                try:
                    success = run_benchmark(None, bench_op, bench_backend, iterations, warmup)
                    if success:
                        st.success("Benchmarks executed successfully! New measurements saved to CSV.")
                    else:
                        st.error("Benchmark runner reported non-zero exit code.")
                except Exception as e:
                    st.error(f"Failed to run benchmark: {e}")

        csv_file = "results/benchmark.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if not df.empty:
                st.markdown("### Measured Performance Metrics")
                st.plotly_chart(build_operation_time_chart(df), use_container_width=True)
                if len(df['pixel_count'].unique()) > 1:
                    st.plotly_chart(build_scaling_chart(df), use_container_width=True)
        else:
            st.info("No benchmark results found. Click 'Run Benchmarking Suite' to generate measurements.")

# ==========================================
# PAGE 4: RESULTS EXPLORER
# ==========================================
elif page == "Results Explorer":
    st.title("Benchmark Dataset Explorer")
    st.markdown("Inspect raw timing measurements, filter by resolution or operation, and export datasets.")

    csv_file = "results/benchmark.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                selected_op = st.multiselect("Filter Operations", options=df['operation'].unique().tolist(), default=df['operation'].unique().tolist())
            with col2:
                selected_backend = st.multiselect("Filter Backend", options=df['backend'].unique().tolist(), default=df['backend'].unique().tolist())

            filtered_df = df[df['operation'].isin(selected_op) & df['backend'].isin(selected_backend)]
            st.dataframe(filtered_df, use_container_width=True)

            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Filtered CSV",
                data=csv_data,
                file_name="parallel_export.csv",
                mime="text/csv"
            )
        else:
            st.info("Benchmark CSV is empty.")
    else:
        st.info("No benchmark file found at `results/benchmark.csv`.")

# ==========================================
# PAGE 5: DOCUMENTATION
# ==========================================
elif page == "Documentation":
    st.title("Architecture & Engineering Documentation")
    st.markdown("""
    ### 1. High-Performance Parallel Computing Principles
    - **Compute vs. Memory Bottleneck:** Image processing operations have low arithmetic intensity (few arithmetic instructions per byte loaded from memory). Therefore, performance is bounded by memory bandwidth.
    - **Host-to-Device (H2D) Transfer Latency:** Copying an image buffer from CPU RAM across the PCIe bus to GPU VRAM takes finite time. For small images (< 512x512), transfer and kernel launch latency exceed CPU compute time. GPU speedup shines on large resolutions (1080p, 4K).
    - **Thread Block Geometry:** Parallel uses a 2D block decomposition of $16 \\times 16 = 256$ threads per block. This aligns with NVIDIA warp granularity (32 threads per warp) and maximizes Streaming Multiprocessor (SM) occupancy.

    ### 2. Algorithmic Formulations
    1. **Grayscale:**
       $$Y = 0.299 \\cdot R + 0.587 \\cdot G + 0.114 \\cdot B$$
    2. **Gaussian Blur:**
       Discrete 2D convolution with continuous normalized Gaussian distribution weights:
       $$G(x, y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2+y^2}{2\\sigma^2}}$$
    3. **Sharpen (Laplacian):**
       $$K_{\\text{sharp}} = \\begin{bmatrix} 0 & -1 & 0 \\\\ -1 & 5 & -1 \\\\ 0 & -1 & 0 \\end{bmatrix}$$
    4. **Sobel Edge Magnitude:**
       $$G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}$$
       $$G = \\min\\left(255, \\sqrt{G_x^2 + G_y^2}\\right)$$
    """)
