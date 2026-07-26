# config.py
"""
Configuración global para el pipeline de Machine Learning, Backtesting y Base de Datos SQLite.
"""

import os

# Configuración de Datos
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"  # Opciones: '5m', '15m', '1h', '4h', '1d'
START_DATE = "2024-01-01T00:00:00Z"  # Fecha de inicio para descargar datos históricos
END_DATE = None  # None para descargar hasta la fecha actual

# Configuración de Base de Datos SQLite para Caché Local
DB_FILE = "trading_data.db"

# Configuración de Detección de Ballenas (Whale Detection)
WHALE_ZSCORE_THRESHOLD = 3.0  # Número de desviaciones estándar en volumen para detectar una ballena
WHALE_MA_WINDOW = 50           # Ventana móvil para calcular volumen promedio y desviación

# Configuración de Ingeniería de Características (Features)
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 200
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14

# Configuración del Modelo de Machine Learning
TEST_SIZE = 0.2  # 20% para el conjunto de prueba final
N_SPLITS = 5     # Número de splits para TimeSeriesSplit en validación cruzada
MODEL_RANDOM_STATE = 42
TARGET_HORIZON_BARS = 12
SIGNAL_THRESHOLD_MIN = 0.35
SIGNAL_THRESHOLD_MAX = 0.75
SIGNAL_THRESHOLD_STEPS = 41

# Configuración del Backtesting y Gestión de Riesgos
INITIAL_CAPITAL = 10000.0  # Capital inicial en USDT
RISK_PER_TRADE = 0.01       # 1% de riesgo del capital acumulado por operación
ATR_SL_MULTIPLIER = 2.0     # Multiplicador ATR para el Stop Loss dinámico
RISK_REWARD_RATIO = 1.5     # Relación Riesgo:Beneficio (1:1.5)
TRADE_DIRECTION = "both"    # 'long', 'short' o 'both'
TRANSACTION_FEE = 0.001     # Comisión por operación (0.1% estándar de Binance)
SLIPPAGE = 0.0002            # Deslizamiento estimado por lado (0.02%)
MAX_HOLDING_BARS = TARGET_HORIZON_BARS
