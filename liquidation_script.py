import os
import requests
import zipfile
import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta

# ================= 核心配置区 =================
SYMBOL = "BTCUSDT"
TIMEFRAMES = [1, 7, 15]

PRICE_BIN_SIZE = 50.0
MIN_TRADE_VALUE = 0
ZOOM_PERCENTAGE = 0.12

VOLUME_TO_OI_RATIO = 0.15
DECAY_BASE = 0.88

# ================= Coinglass 拟合参数 =================
# bar_scale：左轴，单个价格区间爆仓量
# cum_scale：右轴，累计爆仓量
#
# 你的情况：
# 1天：柱子接近，累计 7亿 -> Coinglass 2.7亿，所以 2.7 / 7 ≈ 0.386
# 7天：基本接近，不动
# 15天：累计接近，柱子偏低，所以只放大柱子
TIMEFRAME_FIT_CONFIG = {
    1: {
        "bar_scale": 1.00,
        "cum_scale": 0.386
    },
    7: {
        "bar_scale": 1.00,
        "cum_scale": 1.00
    },
    15: {
        "bar_scale": 1.65,
        "cum_scale": 1.00
    }
}
# =====================================================

PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

DATA_DIR = "binance_data"

LEVERAGE_CONFIG = {
    "100x": (0.994, 1.006, 0.35),
    "50x": (0.984, 1.016, 0.45),
    "25x": (0.964, 1.036, 0.15),
    "10x": (0.904, 1.096, 0.05)
}

COLOR_MAP = {
    "10x": "#5b99ff",
    "25x": "#4052ff",
    "50x": "#ffb800",
    "100x": "#ff7700"
}
# ==============================================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def get_current_price(symbol):
    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"

    try:
        response = requests.get(url, proxies=PROXIES, timeout=10)

        if response.status_code == 200:
            price = float(response.json()['price'])
            print(f"✅ 获取最新价格: {price}")
            return price

        return 70000.0

    except Exception as e:
        print(f"⚠️ 获取价格失败: {e}")
        return 70000.0


def get_open_interest_notional(symbol, current_price):
    url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"

    try:
        response = requests.get(url, proxies=PROXIES, timeout=10)

        if response.status_code == 200:
            data = response.json()
            open_interest = float(data["openInterest"])
            oi_notional = open_interest * current_price

            print(f"✅ 当前 Open Interest: {open_interest:.4f} BTC")
            print(f"✅ 当前 OI 名义价值: {oi_notional / 1e8:.2f} 亿 USDT")

            return oi_notional

        print("⚠️ 获取 Open Interest 失败，跳过 OI 约束")
        return None

    except Exception as e:
        print(f"⚠️ 获取 Open Interest 报错: {e}")
        return None


def download_and_extract_binance_data(symbol, date_str):
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_filepath = os.path.join(
        DATA_DIR,
        f"{symbol}-aggTrades-{date_str}.csv"
    )

    if os.path.exists(csv_filepath):
        return csv_filepath

    url = (
        f"https://data.binance.vision/data/futures/um/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{date_str}.zip"
    )

    try:
        response = requests.get(url, proxies=PROXIES, timeout=30)

        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(DATA_DIR)

            return csv_filepath

    except Exception as e:
        print(f"⚠️ 下载失败 {date_str}: {e}")

    return None


def load_and_filter_data(days_to_fetch, oi_notional=None):
    usecols = ["price", "quantity", "is_buyer_maker"]
    df_list = []

    print(f"\n>>> 正在处理 {days_to_fetch} 天的数据...")

    for i in range(1, days_to_fetch + 1):
        target_date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        csv_filepath = download_and_extract_binance_data(SYMBOL, target_date)

        if csv_filepath:
            try:
                df_day = pd.read_csv(csv_filepath, usecols=usecols)

                decay_factor = DECAY_BASE ** (i - 1)

                df_day['amount'] = (
                    df_day['price']
                    * df_day['quantity']
                    * VOLUME_TO_OI_RATIO
                    * decay_factor
                )

                if MIN_TRADE_VALUE > 0:
                    notional_value = df_day['price'] * df_day['quantity']
                    df_day = df_day[notional_value >= MIN_TRADE_VALUE]

                df_list.append(df_day)

            except Exception as e:
                print(f"处理报错: {e}")

    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)

    # ================= OI 上限约束：只压缩，不放大 =================
    if oi_notional is not None and oi_notional > 0:
        estimated_total = df['amount'].sum()

        if estimated_total > 0:
            print(f"✅ {days_to_fetch} 天原始估算持仓: {estimated_total / 1e8:.2f} 亿 USDT")

            if estimated_total > oi_notional:
                scale_factor = oi_notional / estimated_total
                df['amount'] = df['amount'] * scale_factor

                print(f"⚠️ 超过当前 OI，进行压缩")
                print(f"✅ OI 压缩系数: {scale_factor:.4f}")
                print(f"✅ 压缩后估算持仓: {df['amount'].sum() / 1e8:.2f} 亿 USDT")
            else:
                print(f"✅ 未超过当前 OI，不放大、不处理")
    # ===============================================================

    return df


def human_format_chinese(x, pos):
    if x >= 1e8:
        return f'{x / 1e8:.2f}'.rstrip('0').rstrip('.') + '亿'
    elif x >= 1e4:
        return f'{x / 1e4:.0f}万'
    return str(int(x))


def build_liquidation_map(days, current_price, oi_notional=None):
    df = load_and_filter_data(days, oi_notional)

    if df is None or df.empty:
        return

    fit_config = TIMEFRAME_FIT_CONFIG.get(
        days,
        {
            "bar_scale": 1.00,
            "cum_scale": 1.00
        }
    )

    bar_scale = fit_config["bar_scale"]
    cum_scale = fit_config["cum_scale"]

    print(f"✅ 当前周期拟合参数: bar_scale={bar_scale}, cum_scale={cum_scale}")

    df_buy = df[df['is_buyer_maker'] == False].copy()
    df_sell = df[df['is_buyer_maker'] == True].copy()

    long_frames, short_frames = [], []

    for lev_name, (long_ratio, short_ratio, weight) in LEVERAGE_CONFIG.items():
        temp_long = df_buy.copy()
        temp_long['liq_price'] = temp_long['price'] * long_ratio
        temp_long['amount'] = temp_long['amount'] * weight
        temp_long = temp_long[temp_long['liq_price'] <= current_price]
        temp_long['leverage'] = lev_name
        long_frames.append(temp_long[['liq_price', 'amount', 'leverage']])

        temp_short = df_sell.copy()
        temp_short['liq_price'] = temp_short['price'] * short_ratio
        temp_short['amount'] = temp_short['amount'] * weight
        temp_short = temp_short[temp_short['liq_price'] >= current_price]
        temp_short['leverage'] = lev_name
        short_frames.append(temp_short[['liq_price', 'amount', 'leverage']])

    all_longs = pd.concat(long_frames) if long_frames else pd.DataFrame()
    all_shorts = pd.concat(short_frames) if short_frames else pd.DataFrame()

    if all_longs.empty and all_shorts.empty:
        print("⚠️ 没有可绘制的清算数据")
        return

    all_longs['bin'] = (all_longs['liq_price'] // PRICE_BIN_SIZE) * PRICE_BIN_SIZE
    all_shorts['bin'] = (all_shorts['liq_price'] // PRICE_BIN_SIZE) * PRICE_BIN_SIZE

    long_grouped_raw = all_longs.groupby(['bin', 'leverage'])['amount'].sum().unstack(fill_value=0)
    short_grouped_raw = all_shorts.groupby(['bin', 'leverage'])['amount'].sum().unstack(fill_value=0)

    for col in COLOR_MAP.keys():
        if col not in long_grouped_raw.columns:
            long_grouped_raw[col] = 0
        if col not in short_grouped_raw.columns:
            short_grouped_raw[col] = 0

    long_grouped_raw = long_grouped_raw[list(COLOR_MAP.keys())]
    short_grouped_raw = short_grouped_raw[list(COLOR_MAP.keys())]

    # ================= 显示层拟合 =================
    # 左轴柱子使用 bar_scale
    long_grouped_bar = long_grouped_raw * bar_scale
    short_grouped_bar = short_grouped_raw * bar_scale

    # 右轴累计线使用 cum_scale
    long_grouped_cum = long_grouped_raw * cum_scale
    short_grouped_cum = short_grouped_raw * cum_scale
    # =================================================

    long_totals = long_grouped_cum.sum(axis=1).sort_index(ascending=False)
    long_cumsum = long_totals.cumsum()

    short_totals = short_grouped_cum.sum(axis=1).sort_index(ascending=True)
    short_cumsum = short_totals.cumsum()

    long_bar_max = long_grouped_bar.sum(axis=1).max() if not long_grouped_bar.empty else 0
    short_bar_max = short_grouped_bar.sum(axis=1).max() if not short_grouped_bar.empty else 0
    max_bar = max(long_bar_max, short_bar_max)

    max_cum = 0
    if not long_cumsum.empty:
        max_cum = max(max_cum, long_cumsum.max())
    if not short_cumsum.empty:
        max_cum = max(max_cum, short_cumsum.max())

    print(f"✅ 左轴单区间最高: {max_bar / 1e8:.2f} 亿 USDT")
    print(f"✅ 右轴累计最高: {max_cum / 1e8:.2f} 亿 USDT")

    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(16, 8))

    fig.patch.set_facecolor('#131722')
    ax1.set_facecolor('#131722')

    ax2 = ax1.twinx()

    bottom = pd.Series(0.0, index=long_grouped_bar.index)

    for lev in COLOR_MAP.keys():
        ax1.bar(
            long_grouped_bar.index,
            long_grouped_bar[lev],
            width=PRICE_BIN_SIZE * 0.8,
            bottom=bottom,
            color=COLOR_MAP[lev],
            alpha=0.9,
            label=f"{lev} 杠杆"
        )
        bottom += long_grouped_bar[lev]

    bottom = pd.Series(0.0, index=short_grouped_bar.index)

    for lev in COLOR_MAP.keys():
        ax1.bar(
            short_grouped_bar.index,
            short_grouped_bar[lev],
            width=PRICE_BIN_SIZE * 0.8,
            bottom=bottom,
            color=COLOR_MAP[lev],
            alpha=0.9
        )
        bottom += short_grouped_bar[lev]

    if not long_cumsum.empty:
        ax2.plot(
            long_cumsum.index,
            long_cumsum.values,
            color='#ff4d4d',
            linewidth=2.5,
            label="累计多单清算强度"
        )

    if not short_cumsum.empty:
        ax2.plot(
            short_cumsum.index,
            short_cumsum.values,
            color='#00e676',
            linewidth=2.5,
            label="累计空单清算强度"
        )

    ax1.axvline(
        x=current_price,
        color='#ff3333',
        linestyle='--',
        linewidth=1.5
    )

    ax1.text(
        current_price,
        ax1.get_ylim()[1] * 0.96,
        f'当前价格:{current_price}',
        color='white',
        ha='center',
        va='bottom',
        backgroundcolor='#131722',
        fontsize=11
    )

    x_min = current_price * (1 - ZOOM_PERCENTAGE)
    x_max = current_price * (1 + ZOOM_PERCENTAGE)
    ax1.set_xlim(x_min, x_max)

    ax1.yaxis.set_major_formatter(FuncFormatter(human_format_chinese))
    ax2.yaxis.set_major_formatter(FuncFormatter(human_format_chinese))

    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    ax1.grid(
        True,
        axis='y',
        linestyle='--',
        alpha=0.15,
        color='#787c86'
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_2 + lines_1,
        labels_2 + labels_1,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        frameon=False,
        fontsize=10
    )

    plt.title(
        f"Binance BTC/USDT 清算地图 ({days} 天)",
        pad=35,
        fontsize=18,
        fontweight='bold',
        color='white',
        loc='left'
    )

    plt.tight_layout()

    filename = f"coinglass_liquidation_map_{days}d.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight',
        facecolor='#131722'
    )

    print(f"✅ 生成完毕！已保存: {filename}\n")

    plt.close()


def main():
    current_price = get_current_price(SYMBOL)

    oi_notional = get_open_interest_notional(SYMBOL, current_price)

    for days in TIMEFRAMES:
        build_liquidation_map(days, current_price, oi_notional)

    print("🎉 全部地图处理完成！")


if __name__ == "__main__":
    main()