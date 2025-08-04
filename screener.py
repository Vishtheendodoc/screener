import os
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading

# Configure page
st.set_page_config(
    page_title="Stock Delta Screener", 
    layout="wide",
    page_icon="🎯"
)

# Auto-refresh controls
refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [30, 60, 120, 300], index=1)
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="screener_refresh", limit=None)

# --- Configuration ---
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"
STOCK_LIST_FILE = "stock_list.csv"
LOCAL_CACHE_DIR = "screener_cache"
ALERT_CACHE_DIR = "alert_cache"

# Create cache directories
for cache_dir in [LOCAL_CACHE_DIR, ALERT_CACHE_DIR]:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

# --- Load stock mapping ---
@st.cache_data(ttl=3600)
def load_stock_mapping():
    try:
        stock_df = pd.read_csv(STOCK_LIST_FILE)
        mapping = {str(k): v for k, v in zip(stock_df['security_id'], stock_df['symbol'])}
        return mapping, stock_df
    except Exception as e:
        st.error(f"⚠️ Failed to load stock list: {e}")
        return {}, pd.DataFrame()

stock_mapping, stock_df = load_stock_mapping()

# --- Data fetching functions ---
def fetch_stock_data_quick(security_id, timeout=10):
    """Fetch from GitHub, local cache, and live API, then merge"""
    def load_from_local_cache(security_id):
        path = os.path.join(LOCAL_CACHE_DIR, f"cache_{security_id}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp')
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    def save_to_local_cache(df, security_id):
        if not df.empty:
            path = os.path.join(LOCAL_CACHE_DIR, f"cache_{security_id}.csv")
            df.to_csv(path, index=False)

    def fetch_from_github(security_id):
        try:
            headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
            url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            files = r.json()

            dfs = []
            for f in files:
                if f["name"].endswith(".csv"):
                    df = pd.read_csv(f["download_url"])
                    df = df[df['security_id'] == security_id]
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        dfs.append(df)
            if dfs:
                return pd.concat(dfs).sort_values("timestamp")
        except Exception as e:
            st.warning(f"⚠️ GitHub fetch error: {e}")
        return pd.DataFrame()

    def fetch_from_api(security_id):
        try:
            url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values("timestamp")
        except:
            return pd.DataFrame()

    # Load from all 3 sources
    df_github = fetch_from_github(security_id)
    df_cache = load_from_local_cache(security_id)
    df_live = fetch_from_api(security_id)

    # Merge all and cache
    full_df = pd.concat([df_github, df_cache, df_live]).drop_duplicates("timestamp").sort_values("timestamp")
    save_to_local_cache(full_df, security_id)

    # Filter to today
    today = datetime.now().date()
    start = datetime.combine(today, time(9, 0))
    end = datetime.combine(today, time(23, 59, 59))
    return full_df[(full_df['timestamp'] >= start) & (full_df['timestamp'] <= end)]


def aggregate_stock_data(df, interval_minutes=5):
    """Aggregate stock data into intervals"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)
    
    df_agg = df_copy.resample(f"{interval_minutes}min").agg({
        'buy_initiated': 'sum',
        'sell_initiated': 'sum',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'buy_volume': 'sum',
        'sell_volume': 'sum'
    }).dropna().reset_index()

    df_agg['tick_delta'] = df_agg['buy_initiated'] - df_agg['sell_initiated']
    df_agg['cumulative_tick_delta'] = df_agg['tick_delta'].cumsum()
    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    df_agg['cumulative_delta'] = df_agg['delta'].cumsum()
    
    return df_agg

def analyze_stock_pattern(df):
    """Analyze cumulative tick delta pattern"""
    if df.empty or len(df) < 2:
        return {
            'status': 'No Data',
            'current_cum_delta': 0,
            'trend': 'Unknown',
            'zero_crosses': 0,
            'latest_time': None,
            'price': 0,
            'total_periods': 0,
            'positive_periods': 0,
            'negative_periods': 0
        }
    
    latest = df.iloc[-1]
    current_cum_delta = int(latest['cumulative_tick_delta'])
    
    # Count zero crosses
    zero_crosses = 0
    prev_sign = None
    for delta in df['cumulative_tick_delta']:
        current_sign = 'pos' if delta > 0 else 'neg' if delta < 0 else 'zero'
        if prev_sign and prev_sign != current_sign and current_sign != 'zero':
            zero_crosses += 1
        prev_sign = current_sign
    
    # Calculate consistency
    total_periods = len(df)
    positive_periods = len(df[df['cumulative_tick_delta'] > 0])
    negative_periods = len(df[df['cumulative_tick_delta'] < 0])
    
    # Determine trend
    if current_cum_delta > 0:
        if positive_periods / total_periods > 0.8:
            trend = 'Consistently Positive'
        else:
            trend = 'Currently Positive'
    elif current_cum_delta < 0:
        if negative_periods / total_periods > 0.8:
            trend = 'Consistently Negative'
        else:
            trend = 'Currently Negative'
    else:
        trend = 'Neutral'
    
    # Check for recent zero cross
    status = trend
    if zero_crosses > 0:
        # Check if there was a recent cross (last 3 periods)
        recent_df = df.tail(3)
        if len(recent_df) >= 2:
            signs = [1 if x > 0 else -1 if x < 0 else 0 for x in recent_df['cumulative_tick_delta']]
            if len(set(signs)) > 1 and 0 not in signs:  # Sign change without zero
                if signs[0] != signs[-1]:
                    status = f'Zero Cross: {trend}'
    
    return {
        'status': status,
        'current_cum_delta': current_cum_delta,
        'trend': trend,
        'zero_crosses': zero_crosses,
        'latest_time': latest['timestamp'],
        'price': float(latest['close']),
        'total_periods': total_periods,
        'positive_periods': positive_periods,
        'negative_periods': negative_periods,
        'buy_initiated': int(latest['buy_initiated']),
        'sell_initiated': int(latest['sell_initiated']),
        'tick_delta': int(latest['tick_delta'])
    }

def process_stock_batch(security_ids, interval_minutes=5):
    """Process a batch of stocks concurrently"""
    results = []
    
    def worker(security_id):
        try:
            df = fetch_stock_data_quick(security_id)
            if not df.empty:
                agg_df = aggregate_stock_data(df, interval_minutes)
                analysis = analyze_stock_pattern(agg_df)
                analysis['security_id'] = security_id
                analysis['symbol'] = stock_mapping.get(str(security_id), f'Stock {security_id}')
                return analysis
        except Exception:
            pass
        return None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(worker, sid): sid for sid in security_ids}
        
        for future in as_completed(future_to_id):
            result = future.result()
            if result:
                results.append(result)
    
    return results

# --- Sidebar Controls ---
st.sidebar.title("🎯 Delta Screener")
st.sidebar.markdown("---")

# Filter controls
filter_mode = st.sidebar.selectbox(
    "📊 Filter Mode:",
    ["All Stocks", "Consistently Positive", "Consistently Negative", "Zero Crosses", "Currently Positive", "Currently Negative"]
)

interval_minutes = st.sidebar.selectbox("⏱️ Aggregation Interval (minutes):", [1, 3, 5, 15, 30], index=2)

min_periods = st.sidebar.slider("📈 Minimum Data Points:", 3, 50, 10, help="Minimum number of data points required")

# Advanced filters
st.sidebar.markdown("#### 🔧 Advanced Filters")
min_cum_delta = st.sidebar.number_input("Min |Cumulative Delta|:", value=0, step=10)
min_price = st.sidebar.number_input("Min Price:", value=0.0, step=1.0)
max_price = st.sidebar.number_input("Max Price (0=no limit):", value=0.0, step=1.0)

# Display options
show_mini_charts = st.sidebar.toggle("📊 Show Mini Charts", value=False)
auto_sort = st.sidebar.selectbox("🔀 Sort By:", ["Cumulative Delta", "Zero Crosses", "Price", "Symbol"], index=0)

st.sidebar.markdown("---")

# Main content
st.title("🎯 Stock Delta Screener")
st.caption(f"Real-time screening of cumulative tick delta patterns • {interval_minutes}min intervals")

# Create placeholders for dynamic content
status_placeholder = st.empty()
results_placeholder = st.empty()

# Load all stocks
if stock_df.empty:
    st.error("❌ No stock data available. Please check stock_list.csv")
    st.stop()

with status_placeholder.container():
    st.info(f"🔄 Scanning {len(stock_df)} stocks...")

# Process all stocks
all_security_ids = stock_df['security_id'].unique()
batch_size = 50  # Process in batches to avoid overwhelming

all_results = []
for i in range(0, len(all_security_ids), batch_size):
    batch = all_security_ids[i:i+batch_size]
    batch_results = process_stock_batch(batch, interval_minutes)
    all_results.extend(batch_results)
    
    # Update progress
    progress = min(100, (i + batch_size) / len(all_security_ids) * 100)
    status_placeholder.info(f"🔄 Processing... {progress:.0f}% complete ({len(all_results)} stocks analyzed)")

# Filter results based on selected criteria
filtered_results = []
for result in all_results:
    # Apply minimum periods filter
    if result['total_periods'] < min_periods:
        continue
    
    # Apply cumulative delta filter
    if abs(result['current_cum_delta']) < min_cum_delta:
        continue
    
    # Apply price filters
    if min_price > 0 and result['price'] < min_price:
        continue
    if max_price > 0 and result['price'] > max_price:
        continue
    
    # Apply mode filter
    if filter_mode == "Consistently Positive" and "Consistently Positive" not in result['status']:
        continue
    elif filter_mode == "Consistently Negative" and "Consistently Negative" not in result['status']:
        continue
    elif filter_mode == "Zero Crosses" and "Zero Cross" not in result['status']:
        continue
    elif filter_mode == "Currently Positive" and result['current_cum_delta'] <= 0:
        continue
    elif filter_mode == "Currently Negative" and result['current_cum_delta'] >= 0:
        continue
    
    filtered_results.append(result)

# Sort results
if auto_sort == "Cumulative Delta":
    filtered_results.sort(key=lambda x: abs(x['current_cum_delta']), reverse=True)
elif auto_sort == "Zero Crosses":
    filtered_results.sort(key=lambda x: x['zero_crosses'], reverse=True)
elif auto_sort == "Price":
    filtered_results.sort(key=lambda x: x['price'], reverse=True)
else:  # Symbol
    filtered_results.sort(key=lambda x: x['symbol'])

# Update status
status_placeholder.success(f"✅ Found {len(filtered_results)} stocks matching criteria (from {len(all_results)} analyzed)")

# Display results
if filtered_results:
    with results_placeholder.container():
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        positive_count = len([r for r in filtered_results if r['current_cum_delta'] > 0])
        negative_count = len([r for r in filtered_results if r['current_cum_delta'] < 0])
        zero_cross_count = len([r for r in filtered_results if 'Zero Cross' in r['status']])
        avg_cum_delta = sum([r['current_cum_delta'] for r in filtered_results]) / len(filtered_results)
        
        with col1:
            st.metric("🟢 Positive Delta", positive_count)
        with col2:
            st.metric("🔴 Negative Delta", negative_count)
        with col3:
            st.metric("🔄 Zero Crosses", zero_cross_count)
        with col4:
            st.metric("📊 Avg Cum Delta", f"{avg_cum_delta:.0f}")
        
        st.markdown("---")
        
        # Create results table
        table_data = []
        for result in filtered_results:
            # Status with emoji
            status_emoji = {
                'Consistently Positive': '🟢',
                'Consistently Negative': '🔴',
                'Currently Positive': '🟡',
                'Currently Negative': '🟠',
                'Zero Cross': '🔄',
                'Neutral': '⚪'
            }
            
            main_status = result['status'].split(':')[0] if ':' in result['status'] else result['status']
            emoji = status_emoji.get(main_status, '❓')
            
            table_data.append({
                'Symbol': result['symbol'],
                'Status': f"{emoji} {result['status']}",
                'Cum Delta': result['current_cum_delta'],
                'Current Tick Δ': result['tick_delta'],
                'Price': f"₹{result['price']:.1f}",
                'Zero Crosses': result['zero_crosses'],
                'Buy/Sell': f"{result['buy_initiated']}/{result['sell_initiated']}",
                'Data Points': result['total_periods'],
                'Latest Time': result['latest_time'].strftime('%H:%M') if result['latest_time'] else 'N/A'
            })
        
        results_df = pd.DataFrame(table_data)
        
        # Style the dataframe
        def style_cum_delta(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'background-color: rgba(76, 175, 80, 0.3); color: green; font-weight: bold'
                elif val < 0:
                    return 'background-color: rgba(244, 67, 54, 0.3); color: red; font-weight: bold'
            return ''
        
        def style_tick_delta(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
            return ''
        
        styled_df = results_df.style.applymap(style_cum_delta, subset=['Cum Delta']) \
                                   .applymap(style_tick_delta, subset=['Current Tick Δ'])
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Show mini charts if enabled
        if show_mini_charts and len(filtered_results) <= 20:  # Limit to prevent performance issues
            st.markdown("---")
            st.subheader("📊 Mini Charts (Top 20)")
            
            # Create mini charts in a grid
            cols_per_row = 4
            for i in range(0, min(20, len(filtered_results)), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(filtered_results):
                        result = filtered_results[i + j]
                        
                        with col:
                            # Fetch detailed data for mini chart
                            df = fetch_stock_data_quick(result['security_id'])
                            if not df.empty:
                                agg_df = aggregate_stock_data(df, interval_minutes)
                                
                                if not agg_df.empty:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=agg_df['timestamp'],
                                        y=agg_df['cumulative_tick_delta'],
                                        mode='lines',
                                        line=dict(
                                            color='green' if result['current_cum_delta'] > 0 else 'red',
                                            width=2
                                        ),
                                        showlegend=False
                                    ))
                                    
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                                    
                                    fig.update_layout(
                                        height=200,
                                        margin=dict(l=20, r=20, t=30, b=20),
                                        title=f"{result['symbol']}<br>Δ{result['current_cum_delta']:+d}",
                                        title_font_size=10,
                                        showlegend=False,
                                        xaxis=dict(showticklabels=False),
                                        yaxis=dict(title="Cum Δ", title_font_size=8)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Export functionality
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                csv_data,
                f"delta_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create summary report
            summary_report = f"""
# Delta Screener Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Filter Settings
- Mode: {filter_mode}
- Interval: {interval_minutes} minutes
- Min Periods: {min_periods}
- Min |Cum Delta|: {min_cum_delta}

## Results Summary
- Total Stocks Scanned: {len(all_results)}
- Matching Criteria: {len(filtered_results)}
- Positive Delta: {positive_count}
- Negative Delta: {negative_count}
- Zero Crosses: {zero_cross_count}
- Average Cumulative Delta: {avg_cum_delta:.0f}

## Top 10 by |Cumulative Delta|
"""
            top_10 = sorted(filtered_results, key=lambda x: abs(x['current_cum_delta']), reverse=True)[:10]
            for i, stock in enumerate(top_10, 1):
                summary_report += f"{i}. {stock['symbol']}: {stock['current_cum_delta']:+d} ({stock['status']})\n"
            
            st.download_button(
                "📄 Download Summary Report",
                summary_report.encode('utf-8'),
                f"delta_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                "text/markdown",
                use_container_width=True
            )

else:
    with results_placeholder.container():
        st.warning(f"🔍 No stocks found matching the criteria '{filter_mode}'")
        st.info("💡 Try adjusting the filters or reducing the minimum requirements")

# Footer
st.markdown("---")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh: {'ON' if refresh_enabled else 'OFF'}")
