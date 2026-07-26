# main.py
"""
Script principal para ejecutar el pipeline de Machine Learning y Backtesting por consola.
Coordina la descarga, ingeniería de características, entrenamiento y simulación de trading.
"""

import logging
import pandas as pd
import numpy as np

# Configurar logs para consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# Importar configuraciones y módulos
import config
from data_pipeline import fetch_historical_data
from features import add_technical_indicators, generate_ml_features, add_target_and_clean
from model import split_time_series_data, train_xgboost_model, evaluate_model, get_feature_importances
from backtest import run_backtest

def run_pipeline():
    logger.info("======================================================================")
    logger.info("   INICIANDO PIPELINE DE MACHINE LEARNING PARA ALGORITHMIC TRADING    ")
    logger.info("======================================================================")
    
    # 1. Extracción de Datos (Módulo 1)
    df_raw = fetch_historical_data(
        symbol=config.SYMBOL,
        timeframe=config.TIMEFRAME,
        start_date_iso=config.START_DATE,
        end_date_iso=config.END_DATE
    )
    
    # 2. Ingeniería de Características (Módulo 2)
    df_with_indicators = add_technical_indicators(df_raw)
    
    df_with_features, feature_cols = generate_ml_features(df_with_indicators)
    
    df_final = add_target_and_clean(
        df_with_features,
        horizon_bars=config.TARGET_HORIZON_BARS,
        atr_multiplier=config.ATR_SL_MULTIPLIER,
        risk_reward_ratio=config.RISK_REWARD_RATIO,
        fee=config.TRANSACTION_FEE,
        slippage=config.SLIPPAGE,
    )
    
    # 3. Entrenamiento del Modelo (Módulo 3)
    X_train, X_test, y_train, y_test, df_train, df_test = split_time_series_data(
        df=df_final,
        feature_cols=feature_cols,
        test_size=config.TEST_SIZE
    )
    
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        n_splits=config.N_SPLITS
    )
    
    # Evaluar modelo de clasificación
    eval_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test
    )
    
    # Reportar importancia de características
    importances = get_feature_importances(model, feature_cols)
    logger.info("\n--- IMPORTANCIA DE CARACTERÍSTICAS (Top 5) ---")
    for idx, row in importances.head(5).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
    # 4. Simulación de Backtesting (Módulo 4)
    # Ejecutamos el backtesting sobre el conjunto de prueba (Out-of-sample)
    df_equity, trades, backtest_metrics = run_backtest(
        df_test=df_test,
        predictions=eval_metrics['predictions'],
        initial_capital=config.INITIAL_CAPITAL,
        risk_per_trade=config.RISK_PER_TRADE,
        atr_multiplier=config.ATR_SL_MULTIPLIER,
        risk_reward_ratio=config.RISK_REWARD_RATIO,
        trade_direction=config.TRADE_DIRECTION,
        fee=config.TRANSACTION_FEE,
        slippage=config.SLIPPAGE,
        max_holding_bars=config.MAX_HOLDING_BARS,
    )
    
    # 5. Reporte de Resultados del Backtesting
    logger.info("======================================================================")
    logger.info("                 REPORTE DE RENDIMIENTO (BACKTESTING)                 ")
    logger.info("======================================================================")
    logger.info(f"Par de Divisas:                  {config.SYMBOL}")
    logger.info(f"Temporalidad:                    {config.TIMEFRAME}")
    logger.info(f"Capital Inicial:                 {config.INITIAL_CAPITAL:.2f} USDT")
    logger.info(f"Capital Final:                   {backtest_metrics['final_capital']:.2f} USDT")
    logger.info(f"Retorno de la Estrategia (ML):   {backtest_metrics['total_return_pct']:.2f} %")
    logger.info(f"Retorno Buy & Hold (Mercado):    {backtest_metrics['hold_return_pct']:.2f} %")
    logger.info(f"Win Rate:                        {backtest_metrics['win_rate_pct']:.2f} %")
    logger.info(f"Máximo Drawdown (Riesgo):        {backtest_metrics['max_drawdown_pct']:.2f} %")
    logger.info(f"Factor de Ganancia (Profit F.):  {backtest_metrics['profit_factor']:.2f}")
    logger.info(f"Costes Totales:                  {backtest_metrics['total_costs']:.2f} USDT")
    logger.info(f"Exposición al Mercado:             {backtest_metrics['exposure_pct']:.2f} %")
    logger.info(f"Total Operaciones Cerradas:      {backtest_metrics['total_trades']}")
    logger.info(f"  - Operaciones Ganadas (Win):   {backtest_metrics['winning_trades']}")
    logger.info(f"  - Operaciones Perdidas (Loss): {backtest_metrics['losing_trades']}")
    logger.info("======================================================================")
    
    if len(trades) > 0:
        df_trades = pd.DataFrame(trades)
        logger.info("\nPrimeras 5 operaciones:")
        print(df_trades[['type', 'result', 'entry_price', 'exit_price', 'net_pnl', 'capital_after']].head())
        logger.info("\nÚltimas 5 operaciones:")
        print(df_trades[['type', 'result', 'entry_price', 'exit_price', 'net_pnl', 'capital_after']].tail())
    else:
        logger.warning("No se ejecutaron operaciones en el backtest. Revisa la precisión del modelo y el spread.")

if __name__ == "__main__":
    run_pipeline()
