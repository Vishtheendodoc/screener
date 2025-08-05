import os
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import json
from datetime import datetime, timedelta, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
import pytz

# Configure page for mobile
st.set_page_config(
    page_title="Stock Delta Screener", 
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="collapsed"
)

# Auto-refresh with simplified controls
refresh_enabled = True
refresh_interval = 600  # 10 minutes default
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="screener_refresh", limit=None)

# --- Configuration (Default Settings) ---
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"
STOCK_LIST_FILE = "stock_list.csv"
LOCAL_CACHE_DIR = "screener_cache"
ALERT_CACHE_DIR = "alert_cache"

# Default filter settings (no sidebar customization)
FILTER_MODE = "All Stocks"
INTERVAL_MINUTES = 5
MIN_PERIODS = 10
MIN_CUM_DELTA = 0
MIN_STRENGTH = 0
MAX_VOLATILITY = 100
MIN_PRICE = 0.0
MAX_PRICE = 0.0
AUTO_SORT = "Cumulative Delta"

# Indian timezone for market hours
IST = pytz.timezone('Asia/Kolkata')

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

# --- Enhanced market hours calculation ---
def get_market_hours_today():
    """Get market start and end times for today in IST"""
    now_ist = datetime.now(IST)
    today_date = now_ist.date()
    
    # Indian market hours: 9:15 AM to 3:30 PM IST
    market_start = IST.localize(datetime.combine(today_date, time(9, 15)))
    market_end = IST.localize(datetime.combine(today_date, time(15, 30)))
    
    # For screening purposes, extend end time to capture after-market data
    extended_end = IST.localize(datetime.combine(today_date, time(23, 59, 59)))
    
    return market_start, extended_end

def fetch_current_price(security_id, timeout=5):
    """Fetch current live price from delta_data API (same as main dashboard)"""
    try:
        # Use the same endpoint as the main dashboard
        url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        
        if data:
            # Get the latest record and extract close price
            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                latest_row = df.sort_values('timestamp').iloc[-1]
                return float(latest_row['close'])
    except Exception as e:
        # Silently handle API errors for cleaner logs
        logging.debug(f"Price fetch error for {security_id}: {e}")
    return None

def get_live_or_latest_price(df, security_id):
    """Get live price if available, otherwise use latest close from data"""
    # Try to get live price first using the same API as main dashboard
    live_price = fetch_current_price(security_id)
    if live_price is not None:
        return live_price
    
    # Fallback to latest close price from existing data
    if not df.empty:
        return float(df.iloc[-1]['close'])
    
    return 0.0

# --- Data fetching functions ---
def fetch_stock_data_quick(security_id, timeout=10):
    """Updated to use the same API structure as main dashboard"""
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
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            files = r.json()

            dfs = []
            for f in files:
                if f["name"].endswith(".csv"):
                    df = pd.read_csv(f["download_url"])
                    # Ensure security_id is treated as string for comparison
                    df = df[df['security_id'].astype(str) == str(security_id)]
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        # Convert numeric columns properly
                        numeric_cols = [
                            'buy_initiated', 'buy_volume', 'close', 'delta', 'high', 'low', 'open',
                            'sell_initiated', 'sell_volume', 'tick_delta'
                        ]
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        dfs.append(df)
            if dfs:
                return pd.concat(dfs).sort_values("timestamp")
        except Exception as e:
            logging.debug(f"GitHub fetch error for {security_id}: {e}")
        return pd.DataFrame()

    def fetch_from_api(security_id):
        try:
            # Use the same API endpoint as main dashboard
            url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values("timestamp")
        except Exception as e:
            logging.debug(f"API fetch error for {security_id}: {e}")
        return pd.DataFrame()

    # Load from all 3 sources (same as main dashboard)
    df_github = fetch_from_github(security_id)
    df_cache = load_from_local_cache(security_id)
    df_live = fetch_from_api(security_id)

    # Merge all sources and remove duplicates
    all_dfs = [df for df in [df_github, df_cache, df_live] if not df.empty]
    
    if not all_dfs:
        return pd.DataFrame()
    
    full_df = pd.concat(all_dfs).drop_duplicates("timestamp").sort_values("timestamp")
    
    # Save updated cache
    save_to_local_cache(full_df, security_id)

    # Filter to today's market session (same logic as main dashboard)
    market_start, market_end = get_market_hours_today()
    
    # Convert timestamps to IST if they aren't already timezone-aware
    if full_df.empty:
        return full_df
    
    if full_df['timestamp'].dt.tz is None:
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(IST)
    elif full_df['timestamp'].dt.tz != IST:
        full_df['timestamp'] = full_df['timestamp'].dt.tz_convert(IST)
    
    # Filter for today's market session
    today_data = full_df[
        (full_df['timestamp'] >= market_start) & 
        (full_df['timestamp'] <= market_end)
    ].copy()
    
    return today_data

def aggregate_stock_data(df, interval_minutes=5):
    """Aggregate stock data into intervals with enhanced calculations and zero-handling"""
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

    # Calculate tick deltas with zero handling
    df_agg['tick_delta'] = df_agg['buy_initiated'] - df_agg['sell_initiated']
    
    # Handle cumulative tick delta with API failure protection
    cumulative_values = []
    running_total = 0
    
    for i, tick_delta in enumerate(df_agg['tick_delta']):
        # If tick_delta is zero and both buy/sell initiated are zero (API failure)
        if tick_delta == 0 and df_agg.iloc[i]['buy_initiated'] == 0 and df_agg.iloc[i]['sell_initiated'] == 0:
            # Keep previous cumulative value (don't change running_total)
            pass
        else:
            # Normal case: add tick_delta to running total
            running_total += tick_delta
        
        cumulative_values.append(running_total)
    
    df_agg['cumulative_tick_delta'] = cumulative_values
    
    # Volume delta calculations (similar protection)
    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    
    # Handle cumulative volume delta
    cumulative_vol_values = []
    running_vol_total = 0
    
    for i, vol_delta in enumerate(df_agg['delta']):
        # If volume delta is zero and both volumes are zero (API failure)
        if vol_delta == 0 and df_agg.iloc[i]['buy_volume'] == 0 and df_agg.iloc[i]['sell_volume'] == 0:
            # Keep previous cumulative value
            pass
        else:
            running_vol_total += vol_delta
        
        cumulative_vol_values.append(running_vol_total)
    
    df_agg['cumulative_delta'] = cumulative_vol_values
    
    # Calculate percentage of session completed
    market_start, market_end = get_market_hours_today()
    session_duration = (market_end - market_start).total_seconds() / 60  # in minutes
    
    if not df_agg.empty:
        df_agg['session_progress'] = (
            (df_agg['timestamp'] - market_start).dt.total_seconds() / 60 / session_duration * 100
        ).clip(0, 100)
    
    return df_agg

def analyze_stock_pattern(df, security_id):
    """Enhanced analysis with improved live price fetching"""
    if df.empty or len(df) < 2:
        # Try to get live price even if no data using the working API
        live_price = fetch_current_price(security_id)
        return {
            'status': 'No Data',
            'current_cum_delta': 0,
            'trend': 'Unknown',
            'zero_crosses': 0,
            'latest_time': None,
            'price': live_price if live_price else 0,
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
    
    # Get live price using the working method
    current_price = get_live_or_latest_price(df, security_id)
    
    # Calculate additional metrics
    max_positive = int(df['cumulative_tick_delta'].max())
    max_negative = int(df['cumulative_tick_delta'].min())
    
    # Count zero crosses (only count non-API-failure periods)
    zero_crosses = 0
    prev_sign = None
    for i, delta in enumerate(df['cumulative_tick_delta']):
        # Skip periods that might be API failures
        current_tick = df.iloc[i]['tick_delta'] if 'tick_delta' in df.columns else 0
        buy_init = df.iloc[i]['buy_initiated'] if 'buy_initiated' in df.columns else 0
        sell_init = df.iloc[i]['sell_initiated'] if 'sell_initiated' in df.columns else 0
        
        # Skip if this looks like an API failure period
        if current_tick == 0 and buy_init == 0 and sell_init == 0:
            continue
            
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
    
    # Check for recent significant moves (exclude API failure periods)
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
        'price': current_price,
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
                    # Pass security_id to analysis function for live price fetching
                    analysis = analyze_stock_pattern(agg_df, security_id)
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

# --- Main Application ---

# Market status header (mobile-friendly)
market_start, market_end = get_market_hours_today()
now_ist = datetime.now(IST)
is_market_open = market_start <= now_ist <= market_end

# Mobile-friendly header
st.title("🎯 Stock Delta Screener")

# Compact market status
col1, col2, col3 = st.columns(3)
with col1:
    if is_market_open:
        st.success("🟢 Market Open")
    else:
        if now_ist < market_start:
            st.info("🔵 Pre-Market")
        else:
            st.warning("🟡 After Hours")

with col2:
    st.caption(f"📊 5min intervals")

with col3:
    if is_market_open:
        time_to_close = market_end - now_ist
        hours, remainder = divmod(time_to_close.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        st.caption(f"⏰ {hours}h {minutes}m left")
    else:
        st.caption(f"🔄 Auto-refresh: 10min")

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
    st.info(f"🔄 Scanning {len(stock_df)} stocks...")

# Process all stocks with default settings
all_security_ids = stock_df['security_id'].unique()
batch_size = 50  # Process in batches

all_results = []
progress_bar = st.progress(0)

for i in range(0, len(all_security_ids), batch_size):
    batch = all_security_ids[i:i+batch_size]
    batch_results = process_stock_batch(batch, INTERVAL_MINUTES)
    all_results.extend(batch_results)
    
    # Update progress
    progress = min(1.0, (i + batch_size) / len(all_security_ids))
    progress_bar.progress(progress)
    status_placeholder.info(f"🔄 Processing... {progress*100:.0f}% complete ({len(all_results)} stocks analyzed)")

progress_bar.empty()

# Filter results based on default criteria
filtered_results = []
for result in all_results:
    # Apply minimum periods filter
    if result['total_periods'] < MIN_PERIODS:
        continue
    
    # Apply cumulative delta filter
    if abs(result['current_cum_delta']) < MIN_CUM_DELTA:
        continue
    
    # Apply strength filter
    if result['strength_score'] < MIN_STRENGTH:
        continue
    
    # Apply volatility filter
    if result['volatility_score'] > MAX_VOLATILITY:
        continue
    
    # Apply price filters
    if MIN_PRICE > 0 and result['price'] < MIN_PRICE:
        continue
    if MAX_PRICE > 0 and result['price'] > MAX_PRICE:
        continue
    
    filtered_results.append(result)

# Sort results by cumulative delta (default)
filtered_results.sort(key=lambda x: abs(x['current_cum_delta']), reverse=True)

# Update status
status_placeholder.success(f"✅ Found {len(filtered_results)} active stocks (from {len(all_results)} analyzed)")

# Display mobile-friendly metrics
if all_results:
    with metrics_placeholder.container():
        # Compact metrics for mobile
        col1, col2, col3, col4 = st.columns(4)
        
        bullish_count = len([r for r in filtered_results if r['current_cum_delta'] > 0])
        bearish_count = len([r for r in filtered_results if r['current_cum_delta'] < 0])
        zero_cross_count = len([r for r in filtered_results if 'Zero Cross' in r['status']])
        avg_cum_delta = sum([r['current_cum_delta'] for r in filtered_results]) / len(filtered_results) if filtered_results else 0
        
        with col1:
            st.metric("🟢 Bull", bullish_count)
        with col2:
            st.metric("🔴 Bear", bearish_count)
        with col3:
            st.metric("🔄 Cross", zero_cross_count)
        with col4:
            st.metric("📊 Avg Δ", f"{avg_cum_delta:.0f}")

# Display mobile-friendly results
if filtered_results:
    with results_placeholder.container():
        st.markdown("---")
        
        # Quick filters for mobile
        filter_tabs = st.tabs(["🔥 All", "🟢 Bullish", "🔴 Bearish", "🔄 Crosses", "⚡ Recent"])
        
        with filter_tabs[0]:  # All stocks
            display_results = filtered_results
        with filter_tabs[1]:  # Bullish
            display_results = [r for r in filtered_results if r['current_cum_delta'] > 0]
        with filter_tabs[2]:  # Bearish
            display_results = [r for r in filtered_results if r['current_cum_delta'] < 0]
        with filter_tabs[3]:  # Zero crosses
            display_results = [r for r in filtered_results if 'Zero Cross' in r['status']]
        with filter_tabs[4]:  # Recent moves
            display_results = [r for r in filtered_results if 'Recent Move' in r['status']]
        
        if display_results:
            # Mobile-optimized table with essential columns only
            mobile_data = []
            for result in display_results:  # Show all results
                # Compact status with emoji
                if result['current_cum_delta'] > 0:
                    if 'Strongly' in result['trend']:
                        status_emoji = "🟢"
                    elif 'Moderately' in result['trend']:
                        status_emoji = "🟡"
                    else:
                        status_emoji = "🔵"
                elif result['current_cum_delta'] < 0:
                    if 'Strongly' in result['trend']:
                        status_emoji = "🔴"
                    elif 'Moderately' in result['trend']:
                        status_emoji = "🟠"
                    else:
                        status_emoji = "🟤"
                else:
                    status_emoji = "⚪"
                
                # Add special indicators
                if 'Zero Cross' in result['status']:
                    status_emoji = "🔄"
                elif 'Recent Move' in result['status']:
                    status_emoji = "⚡"
                
                mobile_data.append({
                    'Stock': result['symbol'],
                    'Status': status_emoji,
                    'Cum Δ': result['current_cum_delta'],
                    'Strength': f"{result['strength_score']:.0f}%",
                    'Price': f"₹{result['price']:.1f}",
                    'Time': result['latest_time'].strftime('%H:%M') if result['latest_time'] else 'N/A'
                })
            
            mobile_df = pd.DataFrame(mobile_data)
            
            # Mobile-friendly styling
            def style_mobile_delta(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'background-color: rgba(76, 175, 80, 0.3); color: green; font-weight: bold; font-size: 14px;'
                    elif val < 0:
                        return 'background-color: rgba(244, 67, 54, 0.3); color: red; font-weight: bold; font-size: 14px;'
                return 'font-size: 14px;'
            
            def style_mobile_strength(val):
                if isinstance(val, str) and '%' in val:
                    num_val = float(val.replace('%', ''))
                    if num_val >= 80:
                        return 'background-color: rgba(76, 175, 80, 0.2); color: green; font-weight: bold; font-size: 12px;'
                    elif num_val >= 60:
                        return 'background-color: rgba(255, 193, 7, 0.2); color: orange; font-weight: bold; font-size: 12px;'
                return 'font-size: 12px;'

            def style_mobile_general(val):
                return 'font-size: 14px; padding: 4px;'
            
            styled_mobile_df = mobile_df.style.applymap(style_mobile_delta, subset=['Cum Δ']) \
                                             .applymap(style_mobile_strength, subset=['Strength']) \
                                             .applymap(style_mobile_general, subset=['Stock', 'Status', 'Price', 'Time'])
            
            # Display with mobile-friendly height
            st.dataframe(styled_mobile_df, use_container_width=True, height=400)
            
            # Mobile-friendly summary cards for top performers
            if len(display_results) > 0:
                st.markdown("### 🏆 Top Performers")
                
                # Top 3 bullish and bearish
                top_bullish = [r for r in display_results if r['current_cum_delta'] > 0][:3]
                top_bearish = [r for r in display_results if r['current_cum_delta'] < 0][:3]
                
                if top_bullish:
                    st.markdown("**🟢 Top Bullish:**")
                    for i, stock in enumerate(top_bullish, 1):
                        st.write(f"{i}. **{stock['symbol']}** - Delta: {stock['current_cum_delta']:+d} | Strength: {stock['strength_score']:.0f}% | ₹{stock['price']:.1f}")
                
                if top_bearish:
                    st.markdown("**🔴 Top Bearish:**")
                    for i, stock in enumerate(top_bearish, 1):
                        st.write(f"{i}. **{stock['symbol']}** - Delta: {stock['current_cum_delta']:+d} | Strength: {stock['strength_score']:.0f}% | ₹{stock['price']:.1f}")
            
            # Mobile download option
            st.markdown("---")
            csv_data = mobile_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results",
                csv_data,
                f"delta_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )

else:
    with results_placeholder.container():
        st.warning("🔍 No active stocks found")
        st.info("💡 Stocks need at least 10 data points to appear in results")

# Mobile-friendly market insights
if all_results:
    st.markdown("---")
    st.subheader("📈 Market Summary")
    
    # Compact insights for mobile
    total_bullish = len([r for r in all_results if r['current_cum_delta'] > 0])
    total_bearish = len([r for r in all_results if r['current_cum_delta'] < 0])
    
    sentiment_score = (total_bullish - total_bearish) / len(all_results) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        if sentiment_score > 20:
            sentiment = "🟢 Strong Bull"
        elif sentiment_score > 5:
            sentiment = "🟡 Mild Bull"
        elif sentiment_score > -5:
            sentiment = "⚪ Neutral"
        elif sentiment_score > -20:
            sentiment = "🟠 Mild Bear"
        else:
            sentiment = "🔴 Strong Bear"
        
        st.metric("Market Mood", sentiment, f"{sentiment_score:+.0f}%")
    
    with col2:
        active_stocks = len([r for r in all_results if r['total_periods'] >= MIN_PERIODS])
        st.metric("Active Stocks", f"{active_stocks}/{len(all_results)}", f"{active_stocks/len(all_results)*100:.0f}%")

# Mobile-friendly footer
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.caption(f"🕒 Updated: {datetime.now(IST).strftime('%H:%M IST')}")

with col2:
    st.caption(f"📊 {len(all_results) if all_results else 0} stocks • 5min intervals")

# Mobile legend (collapsible)
with st.expander("📖 Quick Guide"):
    st.markdown("""
    **Status Indicators:**
    - 🟢 **Strong Bullish** (>75% positive periods)
    - 🟡 **Moderate Bullish** (60-75% positive)
    - 🔵 **Currently Bullish** (positive now)
    - 🔴 **Strong Bearish** (>75% negative periods)
    - 🟠 **Moderate Bearish** (60-75% negative)
    - 🟤 **Currently Bearish** (negative now)
    - 🔄 **Zero Cross** (trend reversal)
    - ⚡ **Recent Move** (significant change)
    
    **Metrics:**
    - **Cum Δ**: Cumulative tick delta (buy - sell pressure)
    - **Strength**: Trend consistency percentage
    - **Price**: Current or latest available price
    """)

# Performance note for mobile
st.info("💡 **Mobile Tip**: Swipe left/right on tables to see more columns. Tap tabs above to filter results quickly.")
