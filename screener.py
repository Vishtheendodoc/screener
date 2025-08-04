# Create cache directories
for cache_dir in [LOCAL_CACHE_DIR, ALERT_CACHE_DIR]:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

# --- Load stock mapping ---
@st.cache_data(ttl=3600)
def load_stock_mapping():
    try:
        if os.path.exists(STOCK_LIST_FILE):
            stock_df = pd.read_csv(STOCK_LIST_FILE)
            mapping = {str(k): v for k, v in zip(stock_df['security_id'], stock_df['symbol'])}
            return mapping, stock_df
        else:
            st.error(f"⚠️ Stock list file '{STOCK_LIST_FILE}' not found!")
            return {}, pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Failed to load stock list: {e}")
        return {}, pd.DataFrame()

stock_mapping, stock_df = load_stock_mapping()
import os
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading

# Handle optional imports
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    st.warning("⚠️ streamlit-autorefresh not installed. Auto-refresh disabled.")

try:
    import pytz
    IST = pytz.timezone('Asia/Kolkata')
    TIMEZONE_AVAILABLE = True
except ImportError:
    TIMEZONE_AVAILABLE = False
    st.warning("⚠️ pytz not installed. Using system timezone.")

# Configure page
st.set_page_config(
    page_title="Stock Delta Screener", 
    layout="wide",
    page_icon="🎯"
)

# Auto-refresh controls (only if available)
if AUTOREFRESH_AVAILABLE:
    refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
    refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [30, 60, 120, 300], index=1)
    if refresh_enabled:
        st_autorefresh(interval=refresh_interval * 1000, key="screener_refresh", limit=None)
else:
    refresh_enabled = False
    refresh_interval = 60

# --- Configuration ---
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"
STOCK_LIST_FILE = "stock_list.csv"
LOCAL_CACHE_DIR = "screener_cache"
ALERT_CACHE_DIR = "alert_cache"

# --- Enhanced market hours calculation ---
def get_market_hours_today():
    """Get market start and end times for today"""
    if TIMEZONE_AVAILABLE:
        now_ist = datetime.now(IST)
        today_date = now_ist.date()
        
        # Indian market hours: 9:15 AM to 3:30 PM IST
        market_start = IST.localize(datetime.combine(today_date, time(9, 15)))
        market_end = IST.localize(datetime.combine(today_date, time(15, 30)))
        
        # For screening purposes, extend end time to capture after-market data
        extended_end = IST.localize(datetime.combine(today_date, time(23, 59, 59)))
    else:
        # Fallback to system timezone
        now_local = datetime.now()
        today_date = now_local.date()
        
        # Assume Indian market hours in local time
        market_start = datetime.combine(today_date, time(9, 15))
        market_end = datetime.combine(today_date, time(15, 30))
        extended_end = datetime.combine(today_date, time(23, 59, 59))
    
    return market_start, extended_end

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
            # Check if GitHub token is available
            if 'GITHUB_TOKEN' not in st.secrets:
                return pd.DataFrame()
                
            headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
            url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            files = r.json()

            dfs = []
            for f in files:
                if f["name"].endswith(".csv"):
                    try:
                        df = pd.read_csv(f["download_url"])
                        df = df[df['security_id'] == security_id]
                        if not df.empty:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            dfs.append(df)
                    except Exception:
                        continue  # Skip problematic files
            if dfs:
                return pd.concat(dfs).sort_values("timestamp")
        except Exception as e:
            # Only show warning if it's not a common error
            if "403" not in str(e) and "404" not in str(e):
                st.warning(f"⚠️ GitHub fetch error for {security_id}: {e}")
        return pd.DataFrame()

    def fetch_from_api(security_id):
        try:
            url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values("timestamp")
        except Exception as e:
            # Silently handle API errors for cleaner logs
            pass
        return pd.DataFrame()

    # Load from all 3 sources
    df_github = fetch_from_github(security_id)
    df_cache = load_from_local_cache(security_id)
    df_live = fetch_from_api(security_id)

    # Merge all sources and remove duplicates
    all_dfs = [df for df in [df_github, df_cache, df_live] if not df.empty]
    
    if not all_dfs:
        return pd.DataFrame()
    
    full_df = pd.concat(all_dfs).drop_duplicates("timestamp").sort_values("timestamp")
    save_to_local_cache(full_df, security_id)

    # Convert timestamps to consistent timezone
    if full_df.empty:
        return full_df
    
    # Handle timezone conversion based on availability
    if TIMEZONE_AVAILABLE:
        if full_df['timestamp'].dt.tz is None:
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(IST)
        elif full_df['timestamp'].dt.tz != IST:
            full_df['timestamp'] = full_df['timestamp'].dt.tz_convert(IST)
    else:
        # Ensure timestamps are datetime objects
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    
    # Filter for today's market session
    today_data = full_df[
        (full_df['timestamp'] >= market_start) & 
        (full_df['timestamp'] <= market_end)
    ].copy()
    
    return today_data

def aggregate_stock_data(df, interval_minutes=5):
    """Aggregate stock data into intervals with enhanced calculations"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)
    
    # Aggregate data
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

    if df_agg.empty:
        return df_agg

    # Calculate deltas
    df_agg['tick_delta'] = df_agg['buy_initiated'] - df_agg['sell_initiated']
    df_agg['cumulative_tick_delta'] = df_agg['tick_delta'].cumsum()
    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    df_agg['cumulative_delta'] = df_agg['delta'].cumsum()
    
    # Calculate percentage of session completed
    market_start, market_end = get_market_hours_today()
    session_duration = (market_end - market_start).total_seconds() / 60  # in minutes
    
    if not df_agg.empty:
        df_agg['session_progress'] = (
            (df_agg['timestamp'] - market_start).dt.total_seconds() / 60 / session_duration * 100
        ).clip(0, 100)
    
    return df_agg

def analyze_stock_pattern(df):
    """Enhanced analysis of cumulative tick delta pattern"""
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
            'negative_periods': 0,
            'max_positive': 0,
            'max_negative': 0,
            'volatility_score': 0,
            'strength_score': 0,
            'session_progress': 0
        }
    
    latest = df.iloc[-1]
    current_cum_delta = int(latest['cumulative_tick_delta'])
    
    # Calculate additional metrics
    max_positive = int(df['cumulative_tick_delta'].max())
    max_negative = int(df['cumulative_tick_delta'].min())
    
    # Count zero crosses
    zero_crosses = 0
    prev_sign = None
    for delta in df['cumulative_tick_delta']:
        current_sign = 'pos' if delta > 0 else 'neg' if delta < 0 else 'zero'
        if prev_sign and prev_sign != current_sign and current_sign != 'zero':
            zero_crosses += 1
        prev_sign = current_sign
    
    # Calculate consistency and strength
    total_periods = len(df)
    positive_periods = len(df[df['cumulative_tick_delta'] > 0])
    negative_periods = len(df[df['cumulative_tick_delta'] < 0])
    
    # Volatility score (how much it oscillates)
    volatility_score = zero_crosses / max(1, total_periods) * 100
    
    # Strength score (how strong the current trend is)
    if current_cum_delta > 0:
        strength_score = (positive_periods / total_periods) * 100
    else:
        strength_score = (negative_periods / total_periods) * 100
    
    # Determine trend with enhanced logic
    consistency_threshold = 0.75  # 75% of periods in same direction
    
    if current_cum_delta > 0:
        if positive_periods / total_periods >= consistency_threshold:
            trend = 'Strongly Bullish'
        elif positive_periods / total_periods > 0.6:
            trend = 'Moderately Bullish'
        else:
            trend = 'Currently Bullish'
    elif current_cum_delta < 0:
        if negative_periods / total_periods >= consistency_threshold:
            trend = 'Strongly Bearish'
        elif negative_periods / total_periods > 0.6:
            trend = 'Moderately Bearish'
        else:
            trend = 'Currently Bearish'
    else:
        trend = 'Neutral'
    
    # Enhanced status determination
    status = trend
    
    # Check for recent significant moves
    if len(df) >= 5:
        recent_change = df.iloc[-1]['cumulative_tick_delta'] - df.iloc[-5]['cumulative_tick_delta']
        if abs(recent_change) > abs(current_cum_delta) * 0.3:  # 30% change recently
            status = f"Recent Move: {trend}"
    
    # Check for zero crosses
    if zero_crosses > 0:
        recent_df = df.tail(5)
        if len(recent_df) >= 3:
            signs = [1 if x > 0 else -1 if x < 0 else 0 for x in recent_df['cumulative_tick_delta']]
            if len(set([s for s in signs if s != 0])) > 1:  # Multiple non-zero signs
                status = f"Zero Cross: {trend}"
    
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
        'buy_initiated': int(latest['buy_initiated']) if 'buy_initiated' in latest else 0,
        'sell_initiated': int(latest['sell_initiated']) if 'sell_initiated' in latest else 0,
        'tick_delta': int(latest['tick_delta']) if 'tick_delta' in latest else 0,
        'max_positive': max_positive,
        'max_negative': max_negative,
        'volatility_score': round(volatility_score, 1),
        'strength_score': round(strength_score, 1),
        'session_progress': round(latest.get('session_progress', 0), 1)
    }

def process_stock_batch(security_ids, interval_minutes=5, max_workers=15):
    """Process a batch of stocks concurrently with improved error handling"""
    results = []
    
    def worker(security_id):
        try:
            df = fetch_stock_data_quick(security_id, timeout=8)
            if not df.empty:
                agg_df = aggregate_stock_data(df, interval_minutes)
                if not agg_df.empty:
                    analysis = analyze_stock_pattern(agg_df)
                    analysis['security_id'] = security_id
                    analysis['symbol'] = stock_mapping.get(str(security_id), f'Stock {security_id}')
                    return analysis
        except Exception as e:
            # Log error for debugging
            logging.warning(f"Error processing {security_id}: {e}")
        return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(worker, sid): sid for sid in security_ids}
        
        for future in as_completed(future_to_id, timeout=300):  # 5 minute timeout
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logging.warning(f"Future result error: {e}")
    
    return results

# --- Sidebar Controls ---
st.sidebar.title("🎯 Delta Screener")

# Market status
market_start, market_end = get_market_hours_today()

if TIMEZONE_AVAILABLE:
    now_ist = datetime.now(IST)
else:
    now_ist = datetime.now()

is_market_open = market_start <= now_ist <= market_end

if is_market_open:
    st.sidebar.success("🟢 Market Open")
    time_to_close = market_end - now_ist
    hours, remainder = divmod(time_to_close.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    st.sidebar.caption(f"Closes in {hours}h {minutes}m")
else:
    if now_ist < market_start:
        st.sidebar.info("🔵 Pre-Market")
        time_to_open = market_start - now_ist
        hours, remainder = divmod(time_to_open.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        st.sidebar.caption(f"Opens in {hours}h {minutes}m")
    else:
        st.sidebar.warning("🟡 After Hours")

st.sidebar.markdown("---")

# Enhanced filter controls
filter_mode = st.sidebar.selectbox(
    "📊 Filter Mode:",
    ["All Stocks", "Strongly Bullish", "Moderately Bullish", "Currently Bullish", 
     "Strongly Bearish", "Moderately Bearish", "Currently Bearish", 
     "Zero Crosses", "Recent Moves", "High Volatility"]
)

interval_minutes = st.sidebar.selectbox("⏱️ Aggregation Interval (minutes):", [1, 3, 5, 15, 30], index=2)

min_periods = st.sidebar.slider("📈 Minimum Data Points:", 3, 50, 10, help="Minimum number of data points required")

# Advanced filters
st.sidebar.markdown("#### 🔧 Advanced Filters")
min_cum_delta = st.sidebar.number_input("Min |Cumulative Delta|:", value=0, step=10)
min_strength = st.sidebar.slider("Min Strength Score:", 0, 100, 0, help="Minimum trend strength percentage")
max_volatility = st.sidebar.slider("Max Volatility Score:", 0, 100, 100, help="Maximum volatility allowed")
min_price = st.sidebar.number_input("Min Price:", value=0.0, step=1.0)
max_price = st.sidebar.number_input("Max Price (0=no limit):", value=0.0, step=1.0)

# Display options
show_mini_charts = st.sidebar.toggle("📊 Show Mini Charts", value=False)
auto_sort = st.sidebar.selectbox("🔀 Sort By:", 
    ["Cumulative Delta", "Strength Score", "Volatility Score", "Zero Crosses", "Price", "Symbol"], 
    index=0)

st.sidebar.markdown("---")

# Main content
st.title("🎯 Enhanced Stock Delta Screener")
st.caption(f"Real-time screening from market open ({market_start.strftime('%H:%M')}) • {interval_minutes}min intervals")

# Create placeholders for dynamic content
status_placeholder = st.empty()
metrics_placeholder = st.empty()
results_placeholder = st.empty()

# Load all stocks
if stock_df.empty:
    st.error("❌ No stock data available. Please check stock_list.csv")
    st.stop()

# Processing indicator
with status_placeholder.container():
    st.info(f"🔄 Scanning {len(stock_df)} stocks from market open...")

# Process all stocks
all_security_ids = stock_df['security_id'].unique()
batch_size = 50  # Process in batches

all_results = []
progress_bar = st.progress(0)

for i in range(0, len(all_security_ids), batch_size):
    batch = all_security_ids[i:i+batch_size]
    batch_results = process_stock_batch(batch, interval_minutes)
    all_results.extend(batch_results)
    
    # Update progress
    progress = min(1.0, (i + batch_size) / len(all_security_ids))
    progress_bar.progress(progress)
    status_placeholder.info(f"🔄 Processing... {progress*100:.0f}% complete ({len(all_results)} stocks analyzed)")

progress_bar.empty()

# Filter results based on selected criteria
filtered_results = []
for result in all_results:
    # Apply minimum periods filter
    if result['total_periods'] < min_periods:
        continue
    
    # Apply cumulative delta filter
    if abs(result['current_cum_delta']) < min_cum_delta:
        continue
    
    # Apply strength filter
    if result['strength_score'] < min_strength:
        continue
    
    # Apply volatility filter
    if result['volatility_score'] > max_volatility:
        continue
    
    # Apply price filters
    if min_price > 0 and result['price'] < min_price:
        continue
    if max_price > 0 and result['price'] > max_price:
        continue
    
    # Apply mode filter
    if filter_mode == "Strongly Bullish" and "Strongly Bullish" not in result['trend']:
        continue
    elif filter_mode == "Moderately Bullish" and "Moderately Bullish" not in result['trend']:
        continue
    elif filter_mode == "Currently Bullish" and result['current_cum_delta'] <= 0:
        continue
    elif filter_mode == "Strongly Bearish" and "Strongly Bearish" not in result['trend']:
        continue
    elif filter_mode == "Moderately Bearish" and "Moderately Bearish" not in result['trend']:
        continue
    elif filter_mode == "Currently Bearish" and result['current_cum_delta'] >= 0:
        continue
    elif filter_mode == "Zero Crosses" and "Zero Cross" not in result['status']:
        continue
    elif filter_mode == "Recent Moves" and "Recent Move" not in result['status']:
        continue
    elif filter_mode == "High Volatility" and result['volatility_score'] < 20:
        continue
    
    filtered_results.append(result)

# Sort results
if auto_sort == "Cumulative Delta":
    filtered_results.sort(key=lambda x: abs(x['current_cum_delta']), reverse=True)
elif auto_sort == "Strength Score":
    filtered_results.sort(key=lambda x: x['strength_score'], reverse=True)
elif auto_sort == "Volatility Score":
    filtered_results.sort(key=lambda x: x['volatility_score'], reverse=True)
elif auto_sort == "Zero Crosses":
    filtered_results.sort(key=lambda x: x['zero_crosses'], reverse=True)
elif auto_sort == "Price":
    filtered_results.sort(key=lambda x: x['price'], reverse=True)
else:  # Symbol
    filtered_results.sort(key=lambda x: x['symbol'])

# Update status
status_placeholder.success(f"✅ Found {len(filtered_results)} stocks matching criteria (from {len(all_results)} analyzed)")

# Display enhanced metrics
if all_results:
    with metrics_placeholder.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        
        bullish_count = len([r for r in filtered_results if r['current_cum_delta'] > 0])
        bearish_count = len([r for r in filtered_results if r['current_cum_delta'] < 0])
        zero_cross_count = len([r for r in filtered_results if 'Zero Cross' in r['status']])
        high_strength_count = len([r for r in filtered_results if r['strength_score'] > 70])
        avg_cum_delta = sum([r['current_cum_delta'] for r in filtered_results]) / len(filtered_results) if filtered_results else 0
        
        with col1:
            st.metric("🟢 Bullish", bullish_count)
        with col2:
            st.metric("🔴 Bearish", bearish_count)
        with col3:
            st.metric("🔄 Zero Crosses", zero_cross_count)
        with col4:
            st.metric("💪 High Strength", high_strength_count)
        with col5:
            st.metric("📊 Avg Cum Δ", f"{avg_cum_delta:.0f}")

# Display results
if filtered_results:
    with results_placeholder.container():
        st.markdown("---")
        
        # Create enhanced results table
        table_data = []
        for result in filtered_results:
            # Status with emoji
            status_emojis = {
                'Strongly Bullish': '🟢',
                'Moderately Bullish': '🟡',
                'Currently Bullish': '🔵',
                'Strongly Bearish': '🔴',
                'Moderately Bearish': '🟠',
                'Currently Bearish': '🟤',
                'Zero Cross': '🔄',
                'Recent Move': '⚡',
                'Neutral': '⚪'
            }
            
            # Get main status
            main_status = result['status'].split(':')[0] if ':' in result['status'] else result['status']
            for key in status_emojis.keys():
                if key in result['status']:
                    emoji = status_emojis[key]
                    break
            else:
                emoji = '❓'
            
            table_data.append({
                'Symbol': result['symbol'],
                'Status': f"{emoji} {result['status']}",
                'Cum Δ': result['current_cum_delta'],
                'Strength': f"{result['strength_score']:.0f}%",
                'Volatility': f"{result['volatility_score']:.1f}",
                'Current Tick Δ': result['tick_delta'],
                'Price': f"₹{result['price']:.1f}",
                'Zero Crosses': result['zero_crosses'],
                'Max+/Max-': f"{result['max_positive']}/{result['max_negative']}",
                'Session': f"{result['session_progress']:.0f}%",
                'Data Points': result['total_periods'],
                'Latest Time': result['latest_time'].strftime('%H:%M') if result['latest_time'] else 'N/A'
            })
        
        results_df = pd.DataFrame(table_data)
        
        # Enhanced styling
        def style_cum_delta(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'background-color: rgba(76, 175, 80, 0.3); color: green; font-weight: bold'
                elif val < 0:
                    return 'background-color: rgba(244, 67, 54, 0.3); color: red; font-weight: bold'
            return ''
        
        def style_strength(val):
            if isinstance(val, str) and '%' in val:
                num_val = float(val.replace('%', ''))
                if num_val >= 80:
                    return 'background-color: rgba(76, 175, 80, 0.2); color: green; font-weight: bold'
                elif num_val >= 60:
                    return 'background-color: rgba(255, 193, 7, 0.2); color: orange; font-weight: bold'
            return ''
        
        def style_volatility(val):
            if isinstance(val, (int, float, str)):
                try:
                    num_val = float(str(val))
                    if num_val > 30:
                        return 'background-color: rgba(244, 67, 54, 0.2); color: red'
                    elif num_val > 15:
                        return 'background-color: rgba(255, 193, 7, 0.2); color: orange'
                except:
                    pass
            return ''
        
        styled_df = results_df.style.applymap(style_cum_delta, subset=['Cum Δ']) \
                                   .applymap(style_strength, subset=['Strength']) \
                                   .applymap(style_volatility, subset=['Volatility'])
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Enhanced mini charts
        if show_mini_charts and len(filtered_results) <= 20:
            st.markdown("---")
            st.subheader("📊 Mini Charts (Top 20)")
            
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
                                    
                                    # Main line
                                    color = 'green' if result['current_cum_delta'] > 0 else 'red'
                                    fig.add_trace(go.Scatter(
                                        x=agg_df['timestamp'],
                                        y=agg_df['cumulative_tick_delta'],
                                        mode='lines+markers',
                                        line=dict(color=color, width=2),
                                        marker=dict(size=3),
                                        showlegend=False
                                    ))
                                    
                                    # Zero line
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                                    
                                    # Highlight current value
                                    if not agg_df.empty:
                                        latest_val = agg_df.iloc[-1]['cumulative_tick_delta']
                                        fig.add_trace(go.Scatter(
                                            x=[agg_df.iloc[-1]['timestamp']],
                                            y=[latest_val],
                                            mode='markers',
                                            marker=dict(size=8, color=color, symbol='diamond'),
                                            showlegend=False
                                        ))
                                    
                                    fig.update_layout(
                                        height=200,
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        title=f"{result['symbol']}<br>Δ{result['current_cum_delta']:+d} | {result['strength_score']:.0f}%",
                                        title_font_size=10,
                                        showlegend=False,
                                        xaxis=dict(showticklabels=False),
                                        yaxis=dict(title="Cum Δ", title_font_size=8)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Enhanced export functionality
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
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
            # Create detailed summary report
            summary_report = f"""
# Enhanced Delta Screener Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Market Session: {market_start.strftime('%H:%M')} - {market_end.strftime('%H:%M')}
Market Status: {'🟢 Open' if is_market_open else '🔴 Closed'}

## Filter Settings
- Mode: {filter_mode}
- Interval: {interval_minutes} minutes
- Min Periods: {min_periods}
- Min |Cum Delta|: {min_cum_delta}
- Min Strength Score: {min_strength}%
- Max Volatility Score: {max_volatility}

## Results Summary
- Total Stocks Scanned: {len(all_results)}
- Matching Criteria: {len(filtered_results)}
- Bullish Stocks: {bullish_count}
- Bearish Stocks: {bearish_count}
- Zero Crosses: {zero_cross_count}
- High Strength (>70%): {high_strength_count}
- Average Cumulative Delta: {avg_cum_delta:.0f}

## Top 10 by |Cumulative Delta|
"""
            top_10 = sorted(filtered_results, key=lambda x: abs(x['current_cum_delta']), reverse=True)[:10]
            for i, stock in enumerate(top_10, 1):
                summary_report += f"{i}. {stock['symbol']}: {stock['current_cum_delta']:+d} ({stock['trend']}) - Strength: {stock['strength_score']:.0f}%\n"
            
            summary_report += f"""
## Top 10 by Strength Score
"""
            top_strength = sorted(filtered_results, key=lambda x: x['strength_score'], reverse=True)[:10]
            for i, stock in enumerate(top_strength, 1):
                summary_report += f"{i}. {stock['symbol']}: {stock['strength_score']:.0f}% - Delta: {stock['current_cum_delta']:+d}\n"
            
            st.download_button(
                "📄 Download Summary Report",
                summary_report.encode('utf-8'),
                f"delta_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                "text/markdown",
                use_container_width=True
            )
        
        with col3:
            # Create alerts for significant moves
            alerts = []
            for result in filtered_results:
                if abs(result['current_cum_delta']) > 100 and result['strength_score'] > 80:
                    direction = "BULLISH" if result['current_cum_delta'] > 0 else "BEARISH"
                    alerts.append(f"🚨 {result['symbol']}: Strong {direction} signal - Delta: {result['current_cum_delta']:+d}, Strength: {result['strength_score']:.0f}%")
                elif "Zero Cross" in result['status'] and abs(result['current_cum_delta']) > 50:
                    alerts.append(f"🔄 {result['symbol']}: Zero Cross Alert - Delta: {result['current_cum_delta']:+d}")
                elif "Recent Move" in result['status'] and result['strength_score'] > 70:
                    alerts.append(f"⚡ {result['symbol']}: Recent Strong Move - Delta: {result['current_cum_delta']:+d}")
            
            if alerts:
                alert_text = f"""
# Delta Screener Alerts
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Active Alerts ({len(alerts)})
"""
                for alert in alerts[:20]:  # Limit to top 20 alerts
                    alert_text += f"\n{alert}"
                
                st.download_button(
                    "🚨 Download Alerts",
                    alert_text.encode('utf-8'),
                    f"delta_alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain",
                    use_container_width=True
                )
            else:
                st.button("🚨 No Alerts", disabled=True, use_container_width=True)

else:
    with results_placeholder.container():
        st.warning(f"🔍 No stocks found matching the criteria '{filter_mode}'")
        st.info("💡 Try adjusting the filters or reducing the minimum requirements")
        
        # Show some sample data from all results
        if all_results:
            st.markdown("### Sample of Available Data")
            sample_data = []
            for result in all_results[:5]:
                sample_data.append({
                    'Symbol': result['symbol'],
                    'Cumulative Delta': result['current_cum_delta'],
                    'Trend': result['trend'],
                    'Strength': f"{result['strength_score']:.0f}%",
                    'Data Points': result['total_periods']
                })
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

# Market insights section
if all_results:
    st.markdown("---")
    st.subheader("📈 Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall market sentiment
        total_bullish = len([r for r in all_results if r['current_cum_delta'] > 0])
        total_bearish = len([r for r in all_results if r['current_cum_delta'] < 0])
        
        sentiment_score = (total_bullish - total_bearish) / len(all_results) * 100
        
        if sentiment_score > 20:
            sentiment = "🟢 Strongly Bullish"
        elif sentiment_score > 5:
            sentiment = "🟡 Moderately Bullish"
        elif sentiment_score > -5:
            sentiment = "⚪ Neutral"
        elif sentiment_score > -20:
            sentiment = "🟠 Moderately Bearish"
        else:
            sentiment = "🔴 Strongly Bearish"
        
        st.metric("Overall Market Sentiment", sentiment, f"{sentiment_score:+.1f}%")
        
        # Active vs inactive stocks
        active_stocks = len([r for r in all_results if r['total_periods'] >= min_periods])
        st.metric("Active Stocks", f"{active_stocks}/{len(all_results)}", f"{active_stocks/len(all_results)*100:.0f}%")
    
    with col2:
        # Volatility insights
        avg_volatility = sum([r['volatility_score'] for r in all_results]) / len(all_results)
        high_vol_count = len([r for r in all_results if r['volatility_score'] > 25])
        
        st.metric("Average Volatility", f"{avg_volatility:.1f}", f"{high_vol_count} high-vol stocks")
        
        # Session progress
        session_complete = (now_ist - market_start).total_seconds() / (market_end - market_start).total_seconds() * 100
        session_complete = max(0, min(100, session_complete))
        
        st.metric("Session Progress", f"{session_complete:.0f}%", 
                 f"{'Market Open' if is_market_open else 'Market Closed'}")

# Footer with enhanced information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")

with col2:
    st.caption(f"🔄 Auto-refresh: {'ON' if refresh_enabled else 'OFF'} ({refresh_interval}s)")

with col3:
    st.caption(f"📊 Interval: {interval_minutes}min | Stocks: {len(all_results) if all_results else 0}")

# Add legend for status meanings
with st.expander("📖 Status Legend"):
    st.markdown("""
    **Trend Classifications:**
    - 🟢 **Strongly Bullish**: >75% of periods positive, consistent upward movement
    - 🟡 **Moderately Bullish**: 60-75% of periods positive, generally upward
    - 🔵 **Currently Bullish**: Positive delta but mixed history
    - 🔴 **Strongly Bearish**: >75% of periods negative, consistent downward movement
    - 🟠 **Moderately Bearish**: 60-75% of periods negative, generally downward
    - 🟤 **Currently Bearish**: Negative delta but mixed history
    
    **Special Indicators:**
    - 🔄 **Zero Cross**: Recently crossed zero line (trend reversal)
    - ⚡ **Recent Move**: Significant change in last few periods
    - ⚪ **Neutral**: Around zero with no clear direction
    
    **Metrics:**
    - **Strength Score**: Consistency of trend direction (0-100%)
    - **Volatility Score**: Frequency of zero crosses (higher = more volatile)
    - **Session Progress**: Percentage of trading day completed
    """)

# Performance monitoring
if st.sidebar.button("🔧 Clear Cache"):
    st.cache_data.clear()
    # Clear local cache files
    import shutil
    if os.path.exists(LOCAL_CACHE_DIR):
        shutil.rmtree(LOCAL_CACHE_DIR)
        os.makedirs(LOCAL_CACHE_DIR)
    st.sidebar.success("Cache cleared!")
    st.rerun()
