import streamlit as st
import pandas as pd
import numpy as np
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TED Talks Recommender",
    page_icon="🎤",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #1c1c1c;
        color: #f0f0f0;
    }
    .stApp { background-color: #1c1c1c; }

    /* Force semua teks input jadi putih */
    input, textarea, [data-baseweb="input"] input,
    [data-baseweb="select"] input,
    .stSelectbox input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"] li,
    [data-testid="stSelectbox"] * {
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }
    /* Dropdown list items */
    [role="option"], [role="listbox"] * {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
    }
    [role="option"]:hover {
        background-color: #3a3a3a !important;
    }

    /* Header */
    .ted-header {
        background: linear-gradient(135deg, #1c1c1c 0%, #2a2a2a 100%);
        border-left: 6px solid #E62B1E;
        padding: 24px 32px;
        margin-bottom: 24px;
        border-radius: 0 12px 12px 0;
    }
    .ted-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 900;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .ted-header .subtitle {
        color: #aaaaaa;
        font-size: 0.88rem;
        margin: 0 0 16px 0;
        line-height: 1.5;
    }
    .ted-header .desc {
        color: #cccccc;
        font-size: 0.88rem;
        line-height: 1.7;
        border-top: 1px solid #3a3a3a;
        padding-top: 14px;
        max-width: 820px;
    }
    .ted-header .desc b { color: #E62B1E; }
    .ted-red { color: #E62B1E; }

    /* Tabs — full width */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #aaaaaa;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 0;
        flex: 1;
        text-align: center;
        justify-content: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E62B1E !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px !important;
    }

    /* Label putih */
    label, .stSelectbox label, .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #2a2a2a !important;
        border-color: #3a3a3a !important;
        color: #ffffff !important;
    }

    /* Metric cards */
    .metric-card {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-card h2 {
        color: #E62B1E;
        font-size: 1.8rem;
        font-weight: 900;
        margin: 0;
    }
    .metric-card p {
        color: #aaaaaa;
        font-size: 0.8rem;
        margin: 4px 0 0 0;
    }

    /* Section title */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        border-bottom: 2px solid #E62B1E;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Input talk card */
    .input-card {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .input-card h4 {
        color: #ffffff;
        font-weight: 700;
        margin: 0 0 4px 0;
        font-size: 0.95rem;
    }
    .input-card .speaker {
        color: #E62B1E;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Rec card */
    .rec-card {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-left: 4px solid #E62B1E;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        cursor: pointer;
    }
    .rec-card:hover { background: #2f2f2f; }
    .rec-card h4 {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .speaker, .rec-card .speaker {
        color: #E62B1E !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
    }
    .rec-card .desc, .full-desc {
        color: #ffffff !important;
        font-size: 0.88rem !important;
        line-height: 1.7 !important;
        margin-bottom: 14px !important;
    }
    .rec-card .meta {
        display: flex;
        gap: 16px;
        font-size: 0.78rem;
        color: #888888;
        align-items: center;
    }
    .score-badge {
        background: #E62B1E;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .watch-link {
        color: #E62B1E;
        font-weight: 600;
        font-size: 0.78rem;
        text-decoration: none;
    }
    .watch-link:hover { text-decoration: underline; }

    /* Expanded detail */
    .detail-box {
        background: #232323;
        border: 1px solid #E62B1E33;
        border-radius: 0 0 12px 12px;
        padding: 16px 22px;
        margin-top: -14px;
        margin-bottom: 14px;
    }
    .rec-card .desc, .full-desc {
        color: #ffffff !important;
        font-size: 0.88rem !important;
        line-height: 1.7 !important;
        margin-bottom: 14px !important;
    }
    .detail-box .tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 14px;
    }
    .tag-pill {
        background: #3a3a3a;
        color: #dddddd;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
    }
    .detail-meta {
        display: flex;
        gap: 20px;
        font-size: 0.8rem;
        color: #aaaaaa;
        margin-bottom: 14px;
    }
    .cta-btn {
        display: inline-block;
        background: #E62B1E;
        color: #ffffff !important;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.83rem;
        text-decoration: none;
    }
    .cta-btn:hover { background: #c42318; }
            
    /* Expander styling */
    details {
        background-color: #2a2a2a !important;
        border: 1px solid #3a3a3a !important;
        border-left: 4px solid #E62B1E !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }
    details summary {
        padding: 18px 22px !important;
        cursor: pointer !important;
    }
    details summary:hover p {
        color: #E62B1E !important;
    }
    details[open] summary {
        border-bottom: 1px solid #3a3a3a !important;
    }
    details[open] {
        border-color: #E62B1E !important;
    }
    details > div {
        padding: 20px 22px 28px 22px !important;
    }
    /* Judul expander */
    [data-testid="stExpander"] details summary p {
        font-size: 1.2rem !important;
        font-weight: 900 !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] details[open] summary p {
        color: #ffffff !important;
    }
    [data-testid="stExpander"] details summary:hover p {
        color: #E62B1E !important;
    }

/* Saat hover */
[data-testid="stExpander"] summary:hover {
    color: #E62B1E !important;
}
    details summary:hover {
        color: #E62B1E !important;
    }
    details[open] summary {
        color: #E62B1E !important;
        border-bottom: 1px solid #3a3a3a !important;
        padding-bottom: 18px !important;
    }
    details[open] {
        border-color: #E62B1E !important;
    }
    /* Konten dalam expander */
    details > div {
        padding: 20px 22px 28px 22px !important;
    }
            
    .score-wrapper {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 12px;
}

.score-badge {
    background: #E62B1E;
    color: white;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(230,43,30,0.25);
}
            
    /* Fix background header expander jangan sampai putih */
    [data-testid="stExpander"] details summary {
        background-color: #2a2a2a !important;
    }
    [data-testid="stExpander"] details summary:hover {
        background-color: #333333 !important;
    }
    [data-testid="stExpander"] details[open] summary {
        background-color: #2a2a2a !important;
    }
    [data-testid="stExpander"] details[open] summary:hover {
        background-color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/df_clean.csv")
    cosine_sim = np.load("data/cosine_sim.npy")
    return df, cosine_sim

df, cosine_sim = load_data()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="ted-header">
    <h1>🎤 <span class="ted-red">TED</span> Talks Recommender</h1>
    <p class="subtitle">Powered by Content-Based Filtering · Audio & Text Features</p>
    <div class="desc">
        Platform ini membantu kamu menemukan <b>TED Talks yang relevan</b> berdasarkan talk yang sudah kamu tonton atau sukai.
        Cukup pilih satu talk, dan sistem akan merekomendasikan talk lain yang serupa — berdasarkan
        <b>reaksi penonton</b> (seperti <i>Inspiring</i>, <i>Funny</i>, <i>Jaw-dropping</i>),
        <b>topik & tags</b>, serta <b>konten pembicaraan</b>.
        Tersedia juga <b>Dashboard</b> untuk mengeksplorasi tren dan statistik dari {len(df):,} talk dalam dataset ini.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["  Dashboard", "  Rekomendasi"])

# ══════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════
with tab1:

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h2>{len(df):,}</h2><p>Total Talks</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{df["main_speaker"].nunique():,}</h2><p>Unique Speakers</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{int(df["views"].sum() / 1_000_000):,}M</h2><p>Total Views</p></div>', unsafe_allow_html=True)
    with col4:
        avg_dur = int(df['duration'].mean() // 60)
        st.markdown(f'<div class="metric-card"><h2>{avg_dur} min</h2><p>Avg. Duration</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">🏆 Top 10 Most Viewed Talks</div>', unsafe_allow_html=True)
        top10 = df.nlargest(10, 'views')[['title', 'main_speaker', 'views']].copy()
        top10['views_M'] = (top10['views'] / 1_000_000).round(2)
        top10['label'] = top10['title'].str[:45] + '...'
        fig_top10 = px.bar(
            top10.sort_values('views_M'), x='views_M', y='label',
            orientation='h', text='views_M', color_discrete_sequence=['#E62B1E']
        )
        fig_top10.update_traces(texttemplate='%{text}M', textposition='outside')
        fig_top10.update_layout(
            paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0',
            xaxis_title="Views (Millions)", yaxis_title="",
            margin=dict(l=0, r=40, t=10, b=10), height=360
        )
        fig_top10.update_xaxes(gridcolor='#3a3a3a')
        fig_top10.update_yaxes(gridcolor='#3a3a3a')
        st.plotly_chart(fig_top10, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">❤️ Reaction Distribution</div>', unsafe_allow_html=True)

        def parse_ratings(r):
            try: return ast.literal_eval(r)
            except: return []

        all_reactions = {}
        for r in df['ratings']:
            for item in parse_ratings(r):
                name = item.get('name', '')
                count = item.get('count', 0)
                all_reactions[name] = all_reactions.get(name, 0) + count

        reactions_df = pd.DataFrame(
            list(all_reactions.items()), columns=['Reaction', 'Count']
        ).sort_values('Count', ascending=False).head(8)

        fig_react = px.pie(
            reactions_df, values='Count', names='Reaction',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig_react.update_layout(
            paper_bgcolor='#2a2a2a', font_color='#f0f0f0',
            margin=dict(l=0, r=0, t=10, b=10), height=360,
            legend=dict(font=dict(size=11))
        )
        st.plotly_chart(fig_react, use_container_width=True)

    col_left2, col_right2 = st.columns([2, 3])

    with col_left2:
        st.markdown('<div class="section-title">⏱️ Talk Duration Distribution</div>', unsafe_allow_html=True)
        df['duration_min'] = df['duration'] / 60
        fig_dur = px.histogram(df, x='duration_min', nbins=30, color_discrete_sequence=['#E62B1E'])
        fig_dur.update_layout(
            paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0',
            xaxis_title="Duration (minutes)", yaxis_title="Number of Talks",
            margin=dict(l=0, r=0, t=10, b=10), height=300
        )
        fig_dur.update_xaxes(gridcolor='#3a3a3a')
        fig_dur.update_yaxes(gridcolor='#3a3a3a')
        st.plotly_chart(fig_dur, use_container_width=True)

    with col_right2:
        st.markdown('<div class="section-title">🏷️ Popular Tags</div>', unsafe_allow_html=True)
        all_tags = []
        for t in df['tags_clean'].dropna():
            all_tags.extend(t.split())
        wc = WordCloud(
            width=700, height=280, background_color='#2a2a2a',
            colormap='Reds', max_words=80, prefer_horizontal=0.8
        ).generate(' '.join(all_tags))
        fig_wc, ax = plt.subplots(figsize=(7, 2.8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        fig_wc.patch.set_facecolor('#2a2a2a')
        st.pyplot(fig_wc)

# ══════════════════════════════════════════════
# TAB 2 — REKOMENDASI
# ══════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-title"> Temukan Talk yang Relevan Untukmu</div>', unsafe_allow_html=True)

    col_input, col_slider = st.columns([3, 1])
    with col_input:
        selected_title = st.selectbox(
            "Pilih TED Talk yang kamu suka:",
            options=df['title'].tolist(),
            index=0
        )
    with col_slider:
        top_n = st.slider("Jumlah rekomendasi:", min_value=3, max_value=10, value=5)

    # Info talk yang dipilih
    selected_talk = df[df['title'] == selected_title].iloc[0]
    url_selected = selected_talk.get('url', '#')
    st.markdown(f"""
    <div class="input-card">
        <h4>🎤 {selected_talk['title']}</h4>
        <div class="speaker">by {selected_talk['main_speaker']} · {int(selected_talk['views']):,} views
        · <a href="{url_selected}" target="_blank" class="watch-link">▶ Watch on TED</a></div>
    </div>
    """, unsafe_allow_html=True)

    # Fungsi rekomendasi
    def recommend(title, df, cosine_sim, top_n):
        idx = df[df['title'] == title].index[0]
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        results = []
        for i, score in sim_scores:
            tags_raw = df.loc[i, 'tags_clean'] if pd.notna(df.loc[i, 'tags_clean']) else ''
            tags_list = tags_raw.split()[:6]
            dur_min = int(df.loc[i, 'duration'] // 60)
            results.append({
                'title': df.loc[i, 'title'],
                'speaker': df.loc[i, 'main_speaker'],
                'occupation': df.loc[i, 'speaker_occupation'] if pd.notna(df.loc[i, 'speaker_occupation']) else '',
                'views': int(df.loc[i, 'views']),
                'description': df.loc[i, 'description'],
                'url': df.loc[i, 'url'] if 'url' in df.columns else '#',
                'score': round(score, 4),
                'tags': tags_list,
                'duration': dur_min,
                'comments': int(df.loc[i, 'comments'])
            })
        return results

    results = recommend(selected_title, df, cosine_sim, top_n)

    st.markdown(f"<p style='color:#aaa; margin-bottom:16px;'>Menampilkan <b style='color:#fff'>{top_n} rekomendasi</b> berdasarkan talk yang kamu pilih:</p>", unsafe_allow_html=True)

# Reset expanded state saat hasil berubah
    if 'last_query' not in st.session_state or st.session_state.last_query != selected_title:
        st.session_state.last_query = selected_title

    for i, r in enumerate(results, 1):
        col_card, col_score = st.columns([11, 2])

        with col_card:
            with st.expander(f"#{i}  {r['title']}"):
                tags_html = ''.join([f'<span class="tag-pill">{t}</span>' for t in r['tags']])
                st.markdown(f"""
                <div class="speaker" style="margin-bottom:10px;">
                    🎤 {r['speaker']} {f"· <i style='color:#888;font-weight:400'>{r['occupation']}</i>" if r['occupation'] else ''}
                </div>
                <div class="full-desc">{r['description']}</div>
                <div class="tags" style="margin-bottom:12px;">{tags_html}</div>
                <div class="detail-meta" style="margin-bottom:16px;">
                    <span>👁️ {r['views']:,} views</span>
                    <span>💬 {r['comments']:,} comments</span>
                    <span>⏱️ {r['duration']} min</span>
                </div>
                <a href="{r['url']}" target="_blank" class="cta-btn">▶ Watch on TED</a>
                """, unsafe_allow_html=True)

        with col_score:
            st.markdown(f"""
            <div style="display:flex; align-items:center; height:58px; justify-content:flex-end;">
                <span style="
                    background:#E62B1E;
                    color:#ffffff;
                    padding:6px 14px;
                    border-radius:999px;
                    font-weight:700;
                    font-size:0.85rem;
                    white-space:nowrap;
                    box-shadow:0 2px 8px rgba(230,43,30,0.3);
                ">Score: {r['score']}</span>
            </div>
            """, unsafe_allow_html=True)