from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

DATA = Path(__file__).parent / "data_export"

st.set_page_config(
    page_title="Dashboard Rekomendasi Controller SDN",
    page_icon="🌐",
    layout="wide",
)

# ============================================================
# Muat data
# ============================================================
@st.cache_data
def load(name):
    return pd.read_csv(DATA / name)

lookup     = load("lookup_recommendation.csv")
metrics    = load("metrics.csv")
metrics_sd = load("metrics_by_seed.csv")
segments   = load("segments.csv")
confusion  = load("confusion.csv")
feat_imp   = load("feature_importance.csv")
boot_sum   = load("bootstrap_compare_summary.csv")
boot_seed  = load("bootstrap_compare_by_seed.csv")
edges      = load("graph_edges.csv")
topo_sum   = load("graph_topology_summary.csv")
gt         = load("GROUND_TRUTH.csv")
preds      = load("predictions.csv")

# ============================================================
# Ambang warna
# ============================================================
TOP1_HIJAU, TOP1_KUNING = 0.857627, 0.7786
REG_HIJAU,  REG_KUNING  = 0.012352, 0.0264
FAIL_KUNING = 0.05

def ikon_top1(v):
    if pd.isna(v): return "⚪"
    if v >= TOP1_HIJAU: return "🟢"
    if v >= TOP1_KUNING: return "🟡"
    return "🔴"

def ikon_regret(v):
    if pd.isna(v): return "⚪"
    if v <= REG_HIJAU: return "🟢"
    if v <= REG_KUNING: return "🟡"
    return "🔴"

def ikon_failure(v):
    if pd.isna(v): return "⚪"
    if v == 0: return "🟢"
    if v < FAIL_KUNING: return "🟡"
    return "🔴"

# ============================================================
# Fungsi gambar topologi
# ============================================================
def buat_graf(topologi, n):
    G = nx.Graph()
    if n <= 0:
        return G
    if topologi == "star":
        for i in range(1, n): G.add_edge(0, i)
    elif topologi == "ring":
        if n == 1:
            G.add_node(0)
        else:
            for i in range(n): G.add_edge(i, (i + 1) % n)
    elif topologi == "linear":
        for i in range(n - 1): G.add_edge(i, i + 1)
    elif topologi == "mesh":
        for i in range(n):
            for j in range(i + 1, n): G.add_edge(i, j)
    elif topologi == "tree":
        for i in range(1, n): G.add_edge((i - 1) // 2, i)
    return G

def gambar_topologi(topologi, n):
    G = buat_graf(topologi, n)
    if len(G) == 0:
        return None

    if topologi == "star":
        pos = {0: (0, 0)}
        for i in range(1, n):
            import math
            sudut = 2 * math.pi * (i - 1) / max(n - 1, 1)
            pos[i] = (math.cos(sudut), math.sin(sudut))
    elif topologi == "ring":
        import math
        pos = {i: (math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n)) for i in G.nodes()}
    elif topologi == "linear":
        pos = {i: (i - (n-1)/2, 0) for i in G.nodes()}
    elif topologi == "tree":
        pos = {}
        level = {}
        for node in G.nodes():
            lv = 0
            cur = node
            while cur != 0:
                cur = (cur - 1) // 2
                lv += 1
            level.setdefault(lv, []).append(node)
        max_lv = max(level.keys()) if level else 0
        for lv, nodes_lv in level.items():
            for idx, nd in enumerate(nodes_lv):
                x = (idx - (len(nodes_lv) - 1) / 2) * 1.2
                y = -lv + max_lv / 2
                pos[nd] = (x, y)
    else:
        pos = nx.spring_layout(G, seed=42)

    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]

    et = go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=1.5, color="#555"),
        hoverinfo="none",
    )
    nx_ = [pos[i][0] for i in G.nodes()]
    ny_ = [pos[i][1] for i in G.nodes()]

    labels = []
    for i in G.nodes():
        if i == 0:
            labels.append("ctrl")
        else:
            labels.append(str(i))

    nt = go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        marker=dict(size=28, color="#4C9AFF",
                    line=dict(color="#FAFAFA", width=1)),
        text=labels,
        textposition="middle center",
        textfont=dict(color="#0E1117", size=10),
        hoverinfo="text",
        hovertext=[f"Node {i}" for i in G.nodes()],
    )
    fig = go.Figure(data=[et, nt])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=380,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
    )
    return fig

# ============================================================
# Sidebar filter
# ============================================================
st.sidebar.header("Filter Skenario")
st.sidebar.caption("Filter ini memengaruhi KPI segmen dan tabel analitik di bawah.")

topo_opt = ["Semua"] + sorted(lookup["topology_type"].dropna().unique().tolist())
topo_pilih = st.sidebar.selectbox("Topologi", topo_opt)

df = lookup.copy()
if topo_pilih != "Semua":
    df = df[df["topology_type"] == topo_pilih]

nodes_opt = ["Semua"] + sorted(df["nodes"].dropna().unique().tolist())
nodes_pilih = st.sidebar.selectbox("Jumlah nodes", nodes_opt)
if nodes_pilih != "Semua":
    df = df[df["nodes"] == nodes_pilih]

bw_opt = ["Semua"] + sorted(df["bandwidth_mhz"].dropna().unique().tolist())
bw_pilih = st.sidebar.selectbox("Bandwidth (MHz)", bw_opt)
if bw_pilih != "Semua":
    df = df[df["bandwidth_mhz"] == bw_pilih]

ifl_opt = ["Semua", 0, 1]
ifl_pilih = st.sidebar.selectbox("iFlipper", ifl_opt)
if ifl_pilih != "Semua":
    df = df[df["iflipper_applied"] == ifl_pilih]

st.sidebar.caption(f"Kombinasi lookup setelah filter: {len(df)} dari {len(lookup)}")

# --- Saring predictions.csv dengan filter yang sama ---
preds_f = preds.copy()
if topo_pilih != "Semua":
    preds_f = preds_f[preds_f["topology_type"] == topo_pilih]
if nodes_pilih != "Semua":
    preds_f = preds_f[preds_f["nodes"] == nodes_pilih]
if bw_pilih != "Semua":
    preds_f = preds_f[preds_f["bandwidth_mhz"] == bw_pilih]
if ifl_pilih != "Semua":
    preds_f = preds_f[preds_f["iflipper_applied"] == ifl_pilih]

# ============================================================
# Header
# ============================================================
st.title("Dashboard Rekomendasi Controller SDN")
st.caption(
    "Ini adalah hasil performa model yang dilatih menggunakan model Random Forest Regressor Berbasis simulasi Mininet. Angka di dashboard ini hanya untuk "
    "lingkungan lab/simulasi, bukan jaringan produksi. "
)

# ============================================================
# KPI utama (pooled 5 seed, tidak terpengaruh filter)
# ============================================================
rf = metrics[metrics["method"].astype(str).str.contains("RF", case=False, na=False)]
ryu = metrics[metrics["method"].astype(str).str.lower() == "always_ryu"]

top1_rf  = rf["top1_mean"].iloc[0]  if len(rf) else None
reg_rf   = rf["regret_mean"].iloc[0] if len(rf) else None
top1_ryu = ryu["top1_mean"].iloc[0] if len(ryu) else None

k1, k2, k3 = st.columns(3)
k1.metric(
    "Top1 model (RF)",
    f"{top1_rf:.4f}" if top1_rf else "-",
    help=(
        "seberapa sering model menebak controller terbaik dengan benar\n"
        "semakin tinggi semakin bagus"
    ),
)

k2.metric(
    "Regret model (RF)",
    f"{reg_rf:.6f}" if reg_rf else "-",
    help=(
        "selisih skor antara Ground Truth dan pilihan model\n"
        "semakin kecil semakin bagus"
    ),
)

k3.metric(
    "Top1 baseline selalu ryu",
    f"{top1_ryu:.4f}" if top1_ryu else "-",
    help="Pembanding: kebijakan satu controller tetap",
)

# ============================================================
# Segmen terpilih (responsif terhadap filter sidebar)
# ============================================================
if len(preds_f) > 0:
    top1_f = preds_f["top1"].mean()
    reg_f  = preds_f["regret"].mean()
    n_sken = preds_f["scenario_id"].nunique()
    n_pred = len(preds_f)

    label_filter = []
    if topo_pilih != "Semua": label_filter.append(f"topologi={topo_pilih}")
    if nodes_pilih != "Semua": label_filter.append(f"nodes={nodes_pilih}")
    if bw_pilih != "Semua": label_filter.append(f"bandwidth={bw_pilih} MHz")
    if ifl_pilih != "Semua": label_filter.append(f"iFlipper={ifl_pilih}")
    label_filter = ", ".join(label_filter) if label_filter else "seluruh data"

    st.markdown(f"**Segmen terpilih:** {label_filter}")
    s1, s2, s3 = st.columns(3)
    s1.metric("Top1 segmen", f"{ikon_top1(top1_f)} {top1_f:.4f}")
    s2.metric("Regret segmen", f"{ikon_regret(reg_f)} {reg_f:.6f}")
    s3.metric("Skenario / prediksi", f"{n_sken} / {n_pred}",
              help="pooled_5seed: tiap skenario diulang 5 kali")
else:
    st.warning("Tidak ada prediksi yang cocok dengan filter. Coba longgarkan filter.")

st.divider()

# ============================================================
# Widget rekomendasi (topologi, nodes, bandwidth - tanpa iFlipper)
# ============================================================
st.subheader("Cari rekomendasi controller")
st.caption(
    "Pilih topologi, jumlah nodes, dan bandwidth. Dashboard akan "
    "menampilkan rekomendasi untuk kondisi tanpa iFlipper dan dengan "
    "iFlipper sekaligus."
)

kol1, kol2, kol3 = st.columns(3)

topo_pil = kol1.selectbox(
    "Topologi",
    sorted(lookup["topology_type"].dropna().unique()),
    key="widget_topo",
)

nodes_untuk_topo = sorted(
    lookup.loc[lookup["topology_type"] == topo_pil, "nodes"].dropna().unique()
)
nodes_pil = kol2.selectbox("Jumlah nodes", nodes_untuk_topo, key="widget_nodes")

bw_untuk_topo = sorted(
    lookup.loc[
        (lookup["topology_type"] == topo_pil) & (lookup["nodes"] == nodes_pil),
        "bandwidth_mhz",
    ].dropna().unique()
)
bw_pil = kol3.selectbox("Bandwidth (MHz)", bw_untuk_topo, key="widget_bw")

baris = lookup[
    (lookup["topology_type"] == topo_pil)
    & (lookup["nodes"] == nodes_pil)
    & (lookup["bandwidth_mhz"] == bw_pil)
].copy()

if len(baris) == 0:
    st.warning("Kombinasi ini belum ada di lookup.")
    mirip = lookup[lookup["topology_type"] == topo_pil].head(5)
    if len(mirip):
        st.info("Saran kombinasi yang mirip:")
        mirip_tampil = mirip[
            ["topology_type", "nodes", "bandwidth_mhz",
             "best_controller", "second_controller"]
        ].copy()
        mirip_tampil.columns = [
            "Topologi", "Nodes", "Bandwidth (MHz)",
            "Controller terbaik", "Cadangan",
        ]
        st.dataframe(mirip_tampil, use_container_width=True, hide_index=True)
else:
    baris["label"] = baris["iflipper_applied"].map(
        {0: "🟦 Tanpa iFlipper", 1: "🟧 Dengan iFlipper"}
    )
    baris = baris.sort_values("iflipper_applied")

    for _, r in baris.iterrows():
        st.markdown(f"**{r['label']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Controller terbaik", r["best_controller"])
        c2.metric("Cadangan", r["second_controller"])
        c3.metric("Prediksi AHP terbaik", f"{r['best_pred_ahp']:.4f}")
        c4.metric("Selisih ke cadangan", f"{r['margin_vs_second']:.4f}")

        if r["seen_in_data"] == 0:
            st.error(
                "⚠️ Skenario ini **di luar data pengamatan**. "
                "Rekomendasi hanya ekstrapolasi, wajib divalidasi."
            )
        else:
            st.success("Skenario ini ada di data pengamatan model.")

        st.write("")

    st.markdown("**Bentuk topologi untuk skenario ini:**")
    fig_topo = gambar_topologi(topo_pil, int(nodes_pil))
    if fig_topo is not None:
        st.plotly_chart(fig_topo, use_container_width=True)
    else:
        st.info("Tidak bisa menggambar topologi untuk jumlah nodes ini.")

st.divider()

# ============================================================
# Insight 1
# ============================================================
st.header("Insight 1 — Model memilih controller lebih akurat daripada selalu memilih ryu")

met1 = metrics.copy()
met1 = met1[met1["method"].isin(
    ["RF", "always_ryu", "always_floodlight", "always_onos"]
)]
if len(met1):
    fig1 = px.bar(
        met1, x="method", y="top1_mean",
        color="method", text_auto=".3f",
        labels={"method": "Metode", "top1_mean": "Top1"},
    )
    fig1.add_hline(
        y=TOP1_HIJAU, line_dash="dot", line_color="green",
        annotation_text=f"Ambang hijau {TOP1_HIJAU}",
    )
    fig1.update_layout(
        showlegend=False, height=360,
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    )
    st.plotly_chart(fig1, use_container_width=True)

    if top1_rf is not None and top1_ryu is not None and len(ryu):
        st.caption(
            f"Selisih top1 model − selalu ryu: {top1_rf - top1_ryu:+.4f} poin. "
            f"Regret model hanya {reg_rf / ryu['regret_mean'].iloc[0]:.2f}x "
            f"regret baseline."
        )

st.markdown(
    "**Fitur yang paling sering dipakai model** "
    "(importance, bukan bukti sebab-akibat):"
)
fi = feat_imp.sort_values("importance", ascending=True)
fig_fi = px.bar(
    fi, x="importance", y="feature", orientation="h",
    color="feature_group", text_auto=".3f",
)
fig_fi.update_layout(
    height=420, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    legend_title="Kelompok",
)
st.plotly_chart(fig_fi, use_container_width=True)

st.divider()

# ============================================================
# Insight 2
# ============================================================
st.header("Insight 2 — Controller terbaik berbeda per topologi")

tbl = topo_sum.copy()
if topo_pilih != "Semua":
    tbl = tbl[tbl["topology_type"] == topo_pilih]

if len(tbl) == 0:
    st.warning("Tidak ada topologi yang cocok dengan filter.")
else:
    tbl["top1 pooled"] = tbl["ml_top1_pooled"].apply(
        lambda v: f"{ikon_top1(v)} {v:.4f}" if pd.notna(v) else "-"
    )
    tbl["top1 seed42"] = tbl["ml_top1_seed42"].apply(
        lambda v: f"{v:.4f}" if pd.notna(v) else "-"
    )
    tbl["regret pooled"] = tbl["ml_regret_pooled"].apply(
        lambda v: f"{v:.4f}" if pd.notna(v) else "-"
    )

    st.dataframe(
        tbl[[
            "topology_type", "n_scenarios", "dominant_controller",
            "second_controller", "dominant_avg_ahp", "margin",
            "dominant_win_rate", "top1 pooled", "top1 seed42",
            "regret pooled", "note",
        ]],
        use_container_width=True, hide_index=True,
    )

    fig2 = px.bar(
        tbl, x="topology_type", y="ml_top1_pooled",
        color="dominant_controller", text_auto=".3f",
        labels={
            "topology_type": "Topologi",
            "ml_top1_pooled": "Top1 model (pooled)",
        },
    )
    fig2.add_hline(
        y=TOP1_HIJAU, line_dash="dot", line_color="green",
        annotation_text="Ambang hijau",
    )
    fig2.add_hline(
        y=TOP1_KUNING, line_dash="dot", line_color="orange",
        annotation_text="Ambang kuning",
    )
    fig2.update_layout(
        height=380, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.info(
    "Catatan penting: ring di graf tegas ke onos, tapi model justru "
    "paling lemah di ring. Keduanya tidak bertentangan — graf pakai "
    "rata-rata AHP per topologi, model harus memilih per skenario. "
    "Mesh dan tree masih indikasi awal (n kecil), belum keputusan."
)

st.divider()

# ============================================================
# Insight 3
# ============================================================
st.header("Insight 3 — Risiko terpusat di floodlight, iFlipper, dan data tipis")

gt_f = gt.copy()
if topo_pilih != "Semua":
    gt_f = gt_f[gt_f["topology_type"] == topo_pilih]
if nodes_pilih != "Semua":
    gt_f = gt_f[gt_f["nodes"] == nodes_pilih]
if ifl_pilih != "Semua":
    gt_f = gt_f[gt_f["iflipper_applied"] == ifl_pilih]

gt_f["fail"] = (gt_f["throughput_mbps"] == 0).astype(int)

if len(gt_f) == 0:
    st.warning("Tidak ada baris GROUND_TRUTH yang cocok dengan filter.")
else:
    st.caption(f"Baris run setelah filter: {len(gt_f)} (dari {len(gt)}).")

    per_ctrl = gt_f.groupby("controller_name")["fail"].agg(["sum", "count"])
    per_ctrl["rate"] = per_ctrl["sum"] / per_ctrl["count"]
    per_ctrl = per_ctrl.reset_index()

    per_topo = gt_f.groupby("topology_type")["fail"].agg(["sum", "count"])
    per_topo["rate"] = per_topo["sum"] / per_topo["count"]
    per_topo = per_topo.reset_index()

    silang = gt_f.groupby(["controller_name", "topology_type"])["fail"].agg(
        ["sum", "count"]
    ).reset_index()
    silang["rate"] = silang["sum"] / silang["count"]
    silang = silang[silang["count"] > 0]

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Failure per controller**")
        per_ctrl["ikon"] = per_ctrl["rate"].apply(ikon_failure)
        per_ctrl["rate"] = per_ctrl.apply(
            lambda r: f"{r['ikon']} {r['rate']:.4f}", axis=1
        )
        st.dataframe(
            per_ctrl[["controller_name", "sum", "count", "rate"]],
            use_container_width=True, hide_index=True,
        )

    with colB:
        st.markdown("**Failure per topologi**")
        per_topo["ikon"] = per_topo["rate"].apply(ikon_failure)
        per_topo["rate"] = per_topo.apply(
            lambda r: f"{r['ikon']} {r['rate']:.4f}", axis=1
        )
        st.dataframe(
            per_topo[["topology_type", "sum", "count", "rate"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("**Kombinasi controller × topologi dengan failure:**")
    silang_tampil = silang[silang["sum"] > 0].sort_values("rate", ascending=False)
    if len(silang_tampil) == 0:
        st.info("Tidak ada failure pada segmen yang dipilih.")
    else:
        silang_tampil["ikon"] = silang_tampil["rate"].apply(ikon_failure)
        silang_tampil["rate"] = silang_tampil.apply(
            lambda r: f"{r['ikon']} {r['rate']:.4f}", axis=1
        )
        st.dataframe(
            silang_tampil[[
                "controller_name", "topology_type", "sum", "count", "rate"
            ]],
            use_container_width=True, hide_index=True,
        )

    st.warning(
        "Floodlight di tree dan linear menyumbang hampir semua failure. "
        "Penyebabnya tidak bisa disimpulkan dari data ini — perlu diselidiki "
        "manual sebelum rekomendasi floodlight dipakai di sana."
    )

    ctr = lookup.groupby("seen_in_data").size().reset_index(name="jumlah")
    ctr["porsi"] = ctr["jumlah"] / ctr["jumlah"].sum()
    pct_seen = ctr.loc[ctr["seen_in_data"] == 1, "porsi"].values
    pct_seen = pct_seen[0] if len(pct_seen) else 0
    st.markdown(
        f"**Cakupan data:** hanya **{pct_seen:.2%}** kombinasi lookup yang "
        f"pernah teramati. Sisanya ekstrapolasi — di widget rekomendasi "
        f"otomatis diberi label merah."
    )

st.divider()

# ============================================================
# Insight 4
# ============================================================
st.header("Insight 4 — Bootstrap tidak memperbaiki rekomendasi")

label = {
    "A_original": "A — data asli",
    "B_bootstrap": "B — bootstrap saja",
    "C_original+bootstrap": "C — asli + bootstrap",
}
bs = boot_sum.copy()
bs["label"] = bs["condition"].map(label).fillna(bs["condition"])

c1, c2 = st.columns(2)
with c1:
    f1 = px.bar(
        bs, x="label", y="top1_mean", text_auto=".4f",
        labels={"label": "", "top1_mean": "Top1"},
    )
    f1.add_hline(y=TOP1_HIJAU, line_dash="dot", line_color="green")
    f1.update_layout(
        height=340, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        title="Top1 per kondisi",
    )
    st.plotly_chart(f1, use_container_width=True)

with c2:
    f2 = px.bar(
        bs, x="label", y="regret_mean", text_auto=".5f",
        labels={"label": "", "regret_mean": "Regret"},
    )
    f2.add_hline(y=REG_HIJAU, line_dash="dot", line_color="green")
    f2.update_layout(
        height=340, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        title="Regret per kondisi",
    )
    st.plotly_chart(f2, use_container_width=True)

st.caption(
    "Bootstrap menambah baris latih dari ratusan jadi ribuan, tapi top1 "
    "turun dan regret naik di 3/3 seed. Tambah data bootstrap belum "
    "terbukti menolong. Angka ini global, tidak dipengaruhi filter."
)

st.divider()
st.caption(
    "Semua angka berasal dari simulasi Mininet. Skor AHP menggabungkan "
    "keandalan, latency, CPU/memori, dan jitter; kaitannya ke kualitas "
    "layanan pelanggan nyata belum diuji."
)