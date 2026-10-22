"""
Interactive Plotly visualizations for the Parallel dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

COLOR_PRIMARY = "#00e5ff"      # Cyan
COLOR_SECONDARY = "#00a8e8"    # Tech Blue
COLOR_ACCENT = "#7000ff"       # Deep Indigo
COLOR_GPU = "#00c49f"          # NVIDIA Greenish Cyan
COLOR_BG = "#161b22"
COLOR_CARD = "#0d1117"
COLOR_TEXT = "#e6edf3"

def build_operation_time_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    has_gpu = (df['gpu_kernel_median_ms'] > 0).any() if 'gpu_kernel_median_ms' in df.columns else False

    # Group by operation
    grouped = df.groupby('operation').agg({
        'cpu_median_ms': 'median',
        'gpu_kernel_median_ms': 'median' if has_gpu else lambda x: 0,
        'gpu_pipeline_median_ms': 'median' if has_gpu else lambda x: 0
    }).reset_index()

    fig.add_trace(go.Bar(
        x=[op.capitalize() for op in grouped['operation']],
        y=grouped['cpu_median_ms'],
        name='CPU (Median)',
        marker_color=COLOR_SECONDARY,
        hovertemplate='<b>%{x}</b><br>CPU Time: %{y:.3f} ms<extra></extra>'
    ))

    if has_gpu:
        fig.add_trace(go.Bar(
            x=[op.capitalize() for op in grouped['operation']],
            y=grouped['gpu_kernel_median_ms'],
            name='GPU Kernel Only',
            marker_color=COLOR_GPU,
            hovertemplate='<b>%{x}</b><br>GPU Kernel: %{y:.3f} ms<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            x=[op.capitalize() for op in grouped['operation']],
            y=grouped['gpu_pipeline_median_ms'],
            name='GPU End-to-End Pipeline',
            marker_color=COLOR_PRIMARY,
            hovertemplate='<b>%{x}</b><br>GPU Pipeline: %{y:.3f} ms<extra></extra>'
        ))

    fig.update_layout(
        title="Execution Latency by Operation",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_TEXT, family="Inter, system-ui, sans-serif"),
        xaxis=dict(gridcolor='#30363d', showgrid=True),
        yaxis=dict(gridcolor='#30363d', showgrid=True, title="Time (ms)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def build_scaling_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    has_gpu = (df['gpu_kernel_median_ms'] > 0).any() if 'gpu_kernel_median_ms' in df.columns else False

    for op in df['operation'].unique():
        sub = df[df['operation'] == op].sort_values(by='pixel_count')
        mp = sub['pixel_count'] / 1e6

        fig.add_trace(go.Scatter(
            x=mp,
            y=sub['cpu_median_ms'],
            mode='lines+markers',
            name=f"{op.capitalize()} (CPU)",
            line=dict(width=2),
            hovertemplate='<b>%{fullData.name}</b><br>Res: %{x:.2f} MP<br>Time: %{y:.3f} ms<extra></extra>'
        ))

        if has_gpu:
            fig.add_trace(go.Scatter(
                x=mp,
                y=sub['gpu_pipeline_median_ms'],
                mode='lines+markers',
                name=f"{op.capitalize()} (GPU)",
                line=dict(dash='dash', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Res: %{x:.2f} MP<br>Time: %{y:.3f} ms<extra></extra>'
            ))

    fig.update_layout(
        title="Execution Time vs. Image Resolution",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_TEXT, family="Inter, system-ui, sans-serif"),
        xaxis=dict(gridcolor='#30363d', showgrid=True, title="Resolution (Megapixels)"),
        yaxis=dict(gridcolor='#30363d', showgrid=True, title="Time (ms)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig
