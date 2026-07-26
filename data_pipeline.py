# data_pipeline.py
"""
Módulo 1: Extracción y Persistencia de Datos (Data Pipeline)
Descarga incremental de datos históricos de Binance usando CCXT y caché local en SQLite.
 Evita descargas repetidas acelerando la ejecución en temporalidades cortas (ej. 5m).
"""

import pandas as pd
import ccxt
import sqlite3
import time
from datetime import datetime
import logging
import os
import config

logger = logging.getLogger(__name__)

def get_db_connection(db_file: str = config.DB_FILE):
    """Establece conexión con la base de datos SQLite."""
    conn = sqlite3.connect(db_file)
    return conn

def init_sqlite_db(db_file: str = config.DB_FILE):
    """Crea la tabla klines en SQLite si no existe."""
    conn = get_db_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS klines (
            symbol TEXT,
            timeframe TEXT,
            timestamp INTEGER,
            datetime_str TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    """)
    conn.commit()
    conn.close()

def get_latest_timestamp_from_sqlite(symbol: str, timeframe: str, db_file: str = config.DB_FILE) -> int:
    """Obtiene el último timestamp guardado en SQLite en milisegundos."""
    init_sqlite_db(db_file)
    conn = get_db_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(timestamp) FROM klines 
        WHERE symbol = ? AND timeframe = ?
    """, (symbol, timeframe))
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0] is not None:
        return int(row[0])
    return None

def save_klines_to_sqlite(df: pd.DataFrame, symbol: str, timeframe: str, db_file: str = config.DB_FILE):
    """Guarda nuevas velas en la base de datos SQLite evitando duplicados."""
    if df.empty:
        return
        
    init_sqlite_db(db_file)
    conn = get_db_connection(db_file)
    cursor = conn.cursor()
    
    records = []
    for _, row in df.iterrows():
        # Convertir a timestamp en ms si es Datetime
        ts = int(row['timestamp'].timestamp() * 1000) if isinstance(row['timestamp'], pd.Timestamp) else int(row['timestamp'])
        dt_str = str(row['timestamp'])
        records.append((
            symbol,
            timeframe,
            ts,
            dt_str,
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume'])
        ))
        
    cursor.executemany("""
        INSERT INTO klines
        (symbol, timeframe, timestamp, datetime_str, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
            datetime_str=excluded.datetime_str,
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume
    """, records)
    
    conn.commit()
    conn.close()
    logger.info(f"Guardados {len(records)} registros en SQLite ({symbol} - {timeframe}).")

def load_klines_from_sqlite(symbol: str, timeframe: str, start_timestamp_ms: int = None, db_file: str = config.DB_FILE) -> pd.DataFrame:
    """Carga las velas almacenadas en SQLite."""
    init_sqlite_db(db_file)
    conn = get_db_connection(db_file)
    
    query = "SELECT timestamp, open, high, low, close, volume FROM klines WHERE symbol = ? AND timeframe = ?"
    params = [symbol, timeframe]
    
    if start_timestamp_ms:
        query += " AND timestamp >= ?"
        params.append(start_timestamp_ms)
        
    query += " ORDER BY timestamp ASC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
    return df

def fetch_historical_data(symbol: str = config.SYMBOL, 
                          timeframe: str = config.TIMEFRAME, 
                          start_date_iso: str = config.START_DATE, 
                          end_date_iso: str = config.END_DATE,
                          db_file: str = config.DB_FILE) -> pd.DataFrame:
    """
    Descarga e incrementa datos de Binance con persistencia SQLite.
    Si ya existen datos guardados en SQLite, solo descarga las velas recientes faltantes.
    """
    init_sqlite_db(db_file)
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    start_ms = exchange.parse8601(start_date_iso)
    end_ms = exchange.parse8601(end_date_iso) if end_date_iso else exchange.milliseconds()
    
    # Consultar si ya tenemos datos en SQLite
    latest_db_ms = get_latest_timestamp_from_sqlite(symbol, timeframe, db_file)
    
    # Determinar desde qué milisegundo descargar de Binance
    if latest_db_ms and latest_db_ms >= start_ms:
        # Reconsultar la ultima vela corrige un registro que hubiera sido
        # guardado mientras aun estaba formandose.
        fetch_since = latest_db_ms
        logger.info(f"💾 SQLite: Datos locales encontrados hasta {datetime.fromtimestamp(latest_db_ms/1000)}. Descargando solo actualización incremental...")
    else:
        fetch_since = start_ms
        logger.info(f"🌐 Binance API: Descargando historial completo para {symbol} ({timeframe}) desde {start_date_iso}...")
        
    # Descargar diferencial de Binance si hace falta
    if fetch_since < end_ms:
        all_kline_data = []
        limit = 1000
        current_since = fetch_since
        
        while current_since < end_ms:
            klines = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not klines:
                break
                
            timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
            closed_klines = [k for k in klines if k[0] + timeframe_ms <= end_ms]
            all_kline_data.extend(closed_klines)
            last_ts = klines[-1][0]
            
            if last_ts <= current_since:
                break
                
            current_since = last_ts + 1
            logger.info(f"Descargadas {len(klines)} velas de Binance. Fecha: {datetime.fromtimestamp(last_ts/1000)}")
            time.sleep(exchange.rateLimit / 1000)
            
        if all_kline_data:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_new = pd.DataFrame(all_kline_data, columns=columns)
            save_klines_to_sqlite(df_new, symbol, timeframe, db_file)
            
    # Cargar conjunto completo desde SQLite
    df_full = load_klines_from_sqlite(symbol, timeframe, start_timestamp_ms=start_ms, db_file=db_file)
    
    if df_full.empty:
        raise ValueError(f"No se pudieron cargar datos para {symbol} ({timeframe}).")
        
    logger.info(f"✅ Carga finalizada desde SQLite. Total de registros: {len(df_full)}")
    return df_full

if __name__ == "__main__":
    df_test = fetch_historical_data("BTC/USDT", "5m", "2024-01-01T00:00:00Z")
    print(df_test.head())
    print(df_test.tail())
