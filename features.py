# features.py
"""
Módulo 2: Ingeniería de Características (Feature Engineering) & Detección de Ballenas (Whales)
Cálculo de indicadores técnicos, detección de volumen atípico (ballenas) y métricas de tiempo transcurrido.
"""

import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import config

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los indicadores técnicos estándar solicitados y los añade al DataFrame.
    """
    logger.info("Calculando indicadores técnicos estándar...")
    df = df.copy()
    
    # 1. RSI (14)
    df['rsi'] = RSIIndicator(close=df['close'], window=config.RSI_PERIOD).rsi()
    
    # 2. MACD (12, 26, 9)
    macd_ind = MACD(close=df['close'], window_fast=config.MACD_FAST, window_slow=config.MACD_SLOW, window_sign=config.MACD_SIGNAL)
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_diff'] = macd_ind.macd_diff()
    
    # 3. Promedios Móviles Exponenciales (EMA 9, EMA 21, EMA 200)
    df['ema_9'] = EMAIndicator(close=df['close'], window=config.EMA_SHORT).ema_indicator()
    df['ema_21'] = EMAIndicator(close=df['close'], window=config.EMA_MEDIUM).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=config.EMA_LONG).ema_indicator()
    
    # 4. Bandas de Bollinger (SMA 20, Desviación Estándar 2)
    bb_ind = BollingerBands(close=df['close'], window=config.BB_PERIOD, window_dev=config.BB_STD)
    df['bb_high'] = bb_ind.bollinger_hband()
    df['bb_low'] = bb_ind.bollinger_lband()
    df['bb_mid'] = bb_ind.bollinger_mavg()
    
    # 5. ATR (14) para Volatilidad
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=config.ATR_PERIOD).average_true_range()
    
    return df

def detect_whale_activity(df: pd.DataFrame, 
                          zscore_threshold: float = config.WHALE_ZSCORE_THRESHOLD, 
                          window: int = config.WHALE_MA_WINDOW) -> pd.DataFrame:
    """
    Detecta transacciones de ballenas (volumen atípico elevado) y calcula el tiempo/barras
    transcurridas desde la última intervención de una ballena.
    """
    logger.info("Detectando actividad de Ballenas (Whale Volume Spikes)...")
    df = df.copy()
    
    # Media y desviación estándar móvil del volumen
    # La referencia usa exclusivamente velas anteriores: calculo causal y un
    # z-score que no queda rebajado por el propio pico de volumen actual.
    vol_mean = df['volume'].rolling(window=window, min_periods=window).mean().shift(1)
    vol_std = df['volume'].rolling(window=window, min_periods=window).std().shift(1)
    
    # Z-Score del Volumen
    df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
    df['whale_volume_ratio'] = df['volume'] / (vol_mean + 1e-8)
    
    # Máscara de detección de ballena
    is_whale = df['volume_zscore'] >= zscore_threshold
    
    # Tipo de ballena: 1 = Ballena Compradora (Alcista), -1 = Ballena Vendedora (Bajista), 0 = Normal
    df['whale_type'] = 0
    df.loc[is_whale & (df['close'] >= df['open']), 'whale_type'] = 1   # Buy Whale
    df.loc[is_whale & (df['close'] < df['open']), 'whale_type'] = -1   # Sell Whale
    
    # Marcador booleano
    df['is_whale'] = is_whale.astype(int)
    
    # Calcular barras/velas transcurridas desde la última ballena
    # Usamos cumsum para agrupar los bloques sin ballena y resetear el contador
    whale_indices = df[df['is_whale'] == 1].index
    
    bars_since_whale = []
    last_idx = None
    
    for idx in range(len(df)):
        if df.iloc[idx]['is_whale'] == 1:
            last_idx = idx
            bars_since_whale.append(0)
        else:
            if last_idx is None:
                bars_since_whale.append(999) # Si aún no ha aparecido la primera ballena
            else:
                bars_since_whale.append(idx - last_idx)
                
    df['bars_since_last_whale'] = bars_since_whale
    
    # Estimar tiempo transcurrido en minutos entre velas
    if len(df) > 1:
        time_diff_min = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds() / 60.0
        df['minutes_since_last_whale'] = df['bars_since_last_whale'] * time_diff_min
    else:
        df['minutes_since_last_whale'] = df['bars_since_last_whale'] * 5.0
        
    return df

def generate_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Crea características estacionarias y normalizadas adecuadas para el entrenamiento
    del modelo de Machine Learning, agregando las métricas de Ballenas.
    """
    logger.info("Generando características estacionarias para el modelo de ML (incluyendo Ballenas)...")
    df = df.copy()
    
    # Primero detectar ballenas si aún no han sido calculadas
    if 'volume_zscore' not in df.columns:
        df = detect_whale_activity(df)
        
    features_list = []
    
    # Osciladores y Métricas de Ballenas
    features_list.append('rsi')
    features_list.extend(['volume_zscore', 'whale_volume_ratio', 'bars_since_last_whale', 'whale_type'])
    
    # Diferencia normalizada de MACD con la señal
    df['macd_diff_norm'] = df['macd_diff'] / df['close']
    features_list.append('macd_diff_norm')
    
    # Retornos pasados (varios lags)
    for lag in [1, 2, 3, 5]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(periods=lag)
        features_list.append(f'return_lag_{lag}')

    # Forma de vela y volatilidad realizada, conocidas al cierre de la vela.
    safe_open = df['open'].replace(0, np.nan)
    df['candle_body_pct'] = (df['close'] - df['open']) / safe_open
    df['candle_range_pct'] = (df['high'] - df['low']) / safe_open
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / safe_open
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / safe_open
    features_list.extend(['candle_body_pct', 'candle_range_pct', 'upper_wick_pct', 'lower_wick_pct'])

    one_bar_return = df['close'].pct_change()
    df['realized_vol_12'] = one_bar_return.rolling(12).std()
    df['realized_vol_48'] = one_bar_return.rolling(48).std()
    df['volume_log_change'] = np.log1p(df['volume']).diff()
    features_list.extend(['realized_vol_12', 'realized_vol_48', 'volume_log_change'])
        
    # Volúmenes relativos
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    features_list.append('volume_ratio')
    
    # Distancia porcentual del precio a las medias móviles
    df['dist_ema_9'] = (df['close'] - df['ema_9']) / df['ema_9']
    df['dist_ema_21'] = (df['close'] - df['ema_21']) / df['ema_21']
    df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['ema_200']
    features_list.extend(['dist_ema_9', 'dist_ema_21', 'dist_ema_200'])
    
    # Posición del precio respecto a las Bandas de Bollinger
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-8)
    features_list.append('bb_position')
    
    # Volatilidad relativa (ATR / close)
    df['atr_pct'] = df['atr'] / df['close']
    features_list.append('atr_pct')
    
    # Relación entre medias móviles
    df['ema_9_21_ratio'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
    features_list.append('ema_9_21_ratio')

    # Estacionalidad intradia y semanal sin saltos artificiales.
    minutes = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    df['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
    df['time_cos'] = np.cos(2 * np.pi * minutes / 1440)
    weekday = df['timestamp'].dt.dayofweek
    df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    features_list.extend(['time_sin', 'time_cos', 'weekday_sin', 'weekday_cos'])
    
    return df, features_list

def add_target_and_clean(
    df: pd.DataFrame,
    horizon_bars: int = config.TARGET_HORIZON_BARS,
    atr_multiplier: float = config.ATR_SL_MULTIPLIER,
    risk_reward_ratio: float = config.RISK_REWARD_RATIO,
    fee: float = config.TRANSACTION_FEE,
    slippage: float = config.SLIPPAGE,
) -> pd.DataFrame:
    """
    Define la variable objetivo (Target) y limpia valores nulos (NaN).
    """
    logger.info("Definiendo variable objetivo (Target) y limpiando NaNs...")
    df = df.copy()
    
    if horizon_bars < 1:
        raise ValueError("horizon_bars debe ser mayor o igual a 1")

    # Clases: -1=short, 0=flat/no operar, 1=long. Se simulan ambos lados desde
    # la apertura siguiente. Si SL y TP aparecen en una misma vela se asume el
    # SL primero, evitando un resultado optimista imposible de verificar con OHLC.
    opens = df['open'].to_numpy(dtype=float)
    highs = df['high'].to_numpy(dtype=float)
    lows = df['low'].to_numpy(dtype=float)
    atrs = df['atr'].to_numpy(dtype=float)
    targets = np.full(len(df), np.nan)
    round_trip_cost = 2.0 * (fee + slippage)

    for i in range(len(df) - horizon_bars):
        entry = opens[i + 1]
        sl_distance = atrs[i] * atr_multiplier
        if not np.isfinite(entry) or not np.isfinite(sl_distance) or sl_distance <= 0:
            continue

        tp_distance = sl_distance * risk_reward_ratio + entry * round_trip_cost
        long_stop, long_tp = entry - sl_distance, entry + tp_distance
        short_stop, short_tp = entry + sl_distance, entry - tp_distance
        long_result = short_result = None
        long_finish = short_finish = None

        end = min(len(df), i + 1 + horizon_bars)
        for j in range(i + 1, end):
            if long_result is None:
                if lows[j] <= long_stop:
                    long_result, long_finish = False, j
                elif highs[j] >= long_tp:
                    long_result, long_finish = True, j
            if short_result is None:
                if highs[j] >= short_stop:
                    short_result, short_finish = False, j
                elif lows[j] <= short_tp:
                    short_result, short_finish = True, j
            if long_result is not None and short_result is not None:
                break

        long_wins = long_result is True
        short_wins = short_result is True
        if long_wins and (not short_wins or long_finish < short_finish):
            targets[i] = 1
        elif short_wins and (not long_wins or short_finish < long_finish):
            targets[i] = -1
        else:
            targets[i] = 0

    df['target'] = targets
    df_clean = df.dropna().copy()
    df_clean['target'] = df_clean['target'].astype(int)
    
    logger.info(f"Limpieza de datos finalizada. Filas útiles: {len(df_clean)}")
    return df_clean

if __name__ == "__main__":
    dates = pd.date_range(start="2024-01-01", periods=250, freq="5min")
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.normal(10, 100, size=250))
    high_prices = close_prices + np.random.uniform(10, 50, size=250)
    low_prices = close_prices - np.random.uniform(10, 50, size=250)
    open_prices = close_prices - np.random.normal(0, 10, size=250)
    volume = np.random.uniform(10, 100, size=250)
    # Simular una ballena en el índice 100
    volume[100] = 1000.0
    
    test_df = pd.DataFrame({
        'timestamp': dates, 'open': open_prices, 'high': high_prices,
        'low': low_prices, 'close': close_prices, 'volume': volume
    })
    
    df_ind = add_technical_indicators(test_df)
    df_whale = detect_whale_activity(df_ind)
    df_feat, feat_list = generate_ml_features(df_whale)
    df_final = add_target_and_clean(df_feat)
    
    print("Características generadas:")
    print(feat_list)
    print("\nEjemplo de detección de ballenas:")
    print(df_final[df_final['is_whale'] == 1][['timestamp', 'close', 'volume', 'volume_zscore', 'bars_since_last_whale']])
