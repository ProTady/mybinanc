# app.py
"""
Dashboard Interactivo Premium en Streamlit para el Pipeline de ML y Algorithmic Trading.
Permite configurar parámetros en tiempo real, visualizar la equidad y analizar señales.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime

# Importar módulos del proyecto
from data_pipeline import fetch_historical_data
from features import add_technical_indicators, generate_ml_features, add_target_and_clean
from model import (
    split_time_series_data, train_xgboost_model, evaluate_model,
    get_feature_importances, probabilities_to_signals,
)
import config
from backtest import run_backtest
from whale_trades import (
    classify_large_trades,
    calculate_flow_adjusted_levels,
    calculate_flow_pressure,
    format_duration,
    fetch_taker_volume_summaries,
    load_aggregate_trades,
    predict_next_large_buy,
    seconds_since_last_event,
    sync_aggregate_trades,
)

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Crypto ML Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=60, show_spinner=False)
def cached_taker_volume_summaries(selected_symbol: str) -> dict:
    return fetch_taker_volume_summaries(selected_symbol, periods_hours=(24, 72))

# Estilo CSS personalizado para apariencia premium
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e222b;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 14px;
        color: #8a919e;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #e9ecef;
    }
    .metric-value-green {
        color: #00e676;
    }
    .metric-value-red {
        color: #ff1744;
    }
</style>
""", unsafe_allow_html=True)

# Título Principal
st.title("📈 Crypto Machine Learning Trading System")
st.markdown("Predicción de la dirección de precios de Cripto y Backtesting con Gestión de Riesgo.")

# Barra Lateral de Configuración
st.sidebar.header("⚙️ Configuración del Pipeline")

# Parámetros de Datos
symbol = st.sidebar.text_input("Par de Criptomoneda", value="BTC/USDT")
timeframe = st.sidebar.selectbox(
    "Temporalidad (Timeframe)",
    options=["5m", "15m", "1h", "4h", "1d"],
    index=0
)
start_date = st.sidebar.date_input("Fecha de Inicio", value=datetime(2024, 1, 1))
start_date_iso = f"{start_date}T00:00:00Z"

# Parámetros del Backtest
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Gestión de Riesgo y Operación")
initial_capital = st.sidebar.number_input("Capital Inicial (USDT)", min_value=100.0, value=10000.0, step=1000.0)
risk_per_trade = st.sidebar.slider("Riesgo por Operación (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100.0
atr_multiplier = st.sidebar.slider("Multiplicador Stop Loss (ATR)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("Relación Riesgo:Beneficio (TP)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
trade_direction = st.sidebar.selectbox("Dirección Operativa", options=["both", "long", "short"], index=0)
transaction_fee = st.sidebar.slider("Comisión por Operación (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.01) / 100.0
slippage = st.sidebar.slider("Slippage estimado por lado (%)", min_value=0.0, max_value=0.2, value=0.02, step=0.01) / 100.0
target_horizon = st.sidebar.slider("Horizonte máximo de la operación (velas)", min_value=3, max_value=48, value=12, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🐋 Flujo de Operaciones Grandes")
large_trade_min_usdt = st.sidebar.number_input(
    "Valor mínimo de operación (USDT)",
    min_value=1_000.0, max_value=10_000_000.0, value=50_000.0, step=10_000.0,
)
large_trade_percentile = st.sidebar.slider(
    "Percentil dinámico", min_value=90.0, max_value=99.9, value=99.5, step=0.1
)
large_trade_lookback = st.sidebar.selectbox(
    "Ventana del flujo", options=[1, 6, 24, 72], index=2,
    format_func=lambda hours: f"Últimas {hours} horas",
)
auto_refresh_whales = st.sidebar.toggle("Actualizar flujo cada 5 segundos", value=True)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Ejecutar Simulación", use_container_width=True)

# Mostrar datos en vivo del mercado (Binance 24h)
st.markdown("### 📊 Datos del Mercado en Vivo (Binance 24h)")
live_col1, live_col2, live_col3 = st.columns(3)
try:
    import ccxt
    exchange_temp = ccxt.binance({'enableRateLimit': True})
    ticker_temp = exchange_temp.fetch_ticker(symbol)
    live_price = ticker_temp['last']
    high_24h = ticker_temp['high']
    low_24h = ticker_temp['low']
    change_pct = ticker_temp['percentage']
    
    change_color = "green" if change_pct >= 0 else "red"
    
    with live_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precio Actual ({symbol})</div>
            <div class="metric-value" style="color: #00d2ff;">${live_price:,.2f}</div>
            <div class="metric-label" style="color: {change_color};">{"+" if change_pct >= 0 else ""}{change_pct:.2f}% (24h)</div>
        </div>
        """, unsafe_allow_html=True)
    with live_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Máximo Diario (24h High)</div>
            <div class="metric-value metric-value-green">${high_24h:,.2f}</div>
            <div class="metric-label">Punto más alto del día</div>
        </div>
        """, unsafe_allow_html=True)
    with live_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mínimo Diario (24h Low)</div>
            <div class="metric-value metric-value-red">${low_24h:,.2f}</div>
            <div class="metric-label">Punto más bajo del día</div>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"No se pudieron cargar los datos en vivo para '{symbol}'. Asegúrate de usar un símbolo válido de Binance (ej. BTC/USDT) y tener conexión a Internet.")
st.markdown("<br>", unsafe_allow_html=True)


@st.fragment(run_every=5)
def render_large_trades_monitor():
    """Monitor aislado: su refresco no vuelve a entrenar el modelo."""
    st.markdown("### 🐋 Compras y ventas grandes en Binance Spot")
    refresh_now = st.button("🔄 Actualizar ahora", key="refresh_large_trades")

    try:
        if auto_refresh_whales or refresh_now:
            sync_result = sync_aggregate_trades(
                symbol,
                initial_trades=5_000,
                max_incremental_pages=5,
                resync_recent_trades=5_000,
            )
            if sync_result["inserted"]:
                st.caption(f"Se incorporaron {sync_result['inserted']:,} operaciones nuevas.")
            if sync_result.get("resynced") and sync_result.get("skipped", 0) > 0:
                st.warning(
                    f"Se detectó un atraso de {sync_result['gap_before']:,} aggTrades. "
                    f"Se omitieron {sync_result['skipped']:,} eventos intermedios y se saltó "
                    "a la ventana más reciente para recuperar el tiempo real. "
                    "Los totales de 24/72 horas no se ven afectados."
                )

        flow = load_aggregate_trades(symbol, lookback_hours=large_trade_lookback)
        if flow.empty:
            st.info("Aún no hay operaciones almacenadas. Pulsa 'Actualizar ahora' para iniciar la captura.")
            return

        large, effective_threshold = classify_large_trades(
            flow,
            min_quote_value=large_trade_min_usdt,
            percentile=large_trade_percentile,
        )
        if large.empty:
            st.warning(
                f"No hay operaciones que superen el umbral efectivo de "
                f"${effective_threshold:,.0f} en la muestra disponible."
            )

        now = pd.Timestamp.now(tz="UTC")
        newest_flow_time = flow["timestamp"].max()
        flow_age_seconds = max(0.0, float((now - newest_flow_time).total_seconds()))
        since_buy = seconds_since_last_event(large, "buy", now)
        since_sell = seconds_since_last_event(large, "sell", now)
        recent_cutoff = now - pd.Timedelta(minutes=30)
        last_30m = large[large["timestamp"] >= recent_cutoff]
        buy_usdt = float(last_30m.loc[last_30m["side"] == "buy", "quote_value"].sum())
        sell_usdt = float(last_30m.loc[last_30m["side"] == "sell", "quote_value"].sum())
        imbalance = (buy_usdt - sell_usdt) / max(1.0, buy_usdt + sell_usdt) * 100.0

        c0, c1, c2, c3, c4 = st.columns(5)
        c0.metric(
            "Estado del flujo",
            "AL DÍA" if flow_age_seconds <= 30 else "ATRASADO",
            format_duration(flow_age_seconds),
        )
        c1.metric("Última compra grande", format_duration(since_buy))
        c2.metric("Última venta grande", format_duration(since_sell))
        c3.metric("Umbral efectivo", f"${effective_threshold:,.0f}")
        c4.metric("Desequilibrio 30 min", f"{imbalance:+.1f}%", help="Positivo: dominan compras agresoras; negativo: ventas.")

        st.markdown("#### Volumen comprado y vendido en 24/72 horas")
        volume_summaries = cached_taker_volume_summaries(symbol)
        base_asset = symbol.split("/")[0].upper()
        volume_rows = []
        for hours in (24, 72):
            summary = volume_summaries[hours]
            volume_rows.append({
                "Periodo": f"Últimas {hours} h",
                f"{base_asset} comprado (taker)": summary["buy_base"],
                f"{base_asset} vendido (taker)": summary["sell_base"],
                f"Balance {base_asset}": summary["net_base"],
                "Dominio comprador": summary["buy_pct"],
            })
        st.dataframe(
            pd.DataFrame(volume_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                f"{base_asset} comprado (taker)": st.column_config.NumberColumn(format="%.4f"),
                f"{base_asset} vendido (taker)": st.column_config.NumberColumn(format="%.4f"),
                f"Balance {base_asset}": st.column_config.NumberColumn(format="%+.4f"),
                "Dominio comprador": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.caption(
            "Estos totales clasifican el volumen por el lado que tomó liquidez. "
            "Cada trade siempre tiene comprador y vendedor; no representan entrada o salida neta de BTC del exchange."
        )

        flow_pressure = calculate_flow_pressure(volume_summaries, buy_usdt, sell_usdt)
        st.session_state.flow_pressure = flow_pressure
        pressure_color = "🟢" if flow_pressure["score"] > 0.10 else "🔴" if flow_pressure["score"] < -0.10 else "🟡"
        st.metric(
            "Presión combinada del flujo",
            f"{flow_pressure['score']:+.3f}",
            help="-1 indica presión vendedora extrema; +1 presión compradora extrema.",
        )
        st.caption(
            f"{pressure_color} Lectura: {flow_pressure['label']}. "
            f"Desequilibrio total 24h {flow_pressure['imbalance_24h'] * 100:+.2f}%, "
            f"72h {flow_pressure['imbalance_72h'] * 100:+.2f}% y grandes 30m "
            f"{flow_pressure['large_imbalance'] * 100:+.2f}%."
        )

        prediction = predict_next_large_buy(large, now=now)
        st.markdown("#### Predictor temporal de la próxima compra grande")
        if prediction["status"] != "ok":
            st.info(
                f"Datos insuficientes: {prediction['events']} compras grandes observadas; "
                f"se necesitan al menos {prediction['minimum_events']}. La estimación aparecerá al acumular historial."
            )
        else:
            p1, p2, p3, p4 = st.columns(4)
            probabilities = prediction["probabilities"]
            p1.metric("Próximos 5 min", f"{probabilities[5] * 100:.1f}%")
            p2.metric("Próximos 15 min", f"{probabilities[15] * 100:.1f}%")
            p3.metric("Próximos 30 min", f"{probabilities[30] * 100:.1f}%")
            eta_low, eta_high = prediction["eta_range_minutes"]
            p4.metric("Intervalo estimado", f"{eta_low:.0f}–{eta_high:.0f} min")
            st.caption(
                f"Confianza estadística: {prediction['confidence']}. Estimación basada en "
                f"{prediction['events']} compras grandes y sus intervalos; no identifica ni anticipa a una persona."
            )

        st.markdown("#### Registro de operaciones grandes")
        order = st.radio(
            "Ordenar tabla por", ["Más recientes", "Mayor valor"],
            horizontal=True, key="large_trade_order",
        )
        display = large.copy()
        display["Tipo"] = display["side"].map({"buy": "🟢 Compra", "sell": "🔴 Venta"})
        display["Hora (Lima)"] = display["timestamp"].dt.tz_convert("America/Lima").dt.strftime("%Y-%m-%d %H:%M:%S")
        display["Valor USDT"] = display["quote_value"]
        display["Precio"] = display["price"]
        display["Cantidad"] = display["quantity"]
        display["Intensidad"] = display["intensity_percentile"]
        display["Desde anterior"] = display["seconds_since_previous"].map(
            lambda value: format_duration(value) if pd.notna(value) else "Primera"
        )
        sort_column = "timestamp" if order == "Más recientes" else "quote_value"
        display = display.sort_values(sort_column, ascending=False).head(100)
        st.dataframe(
            display[["Hora (Lima)", "Tipo", "Precio", "Cantidad", "Valor USDT", "Intensidad", "Desde anterior"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precio": st.column_config.NumberColumn(format="$%.2f"),
                "Cantidad": st.column_config.NumberColumn(format="%.6f"),
                "Valor USDT": st.column_config.NumberColumn(format="$%.2f"),
                "Intensidad": st.column_config.NumberColumn("Percentil", format="%.2f%%"),
            },
        )
        st.caption(
            f"Muestra almacenada: {len(flow):,} aggTrades desde "
            f"{flow['timestamp'].min().tz_convert('America/Lima'):%Y-%m-%d %H:%M:%S}. "
            f"Último dato: {newest_flow_time.tz_convert('America/Lima'):%Y-%m-%d %H:%M:%S} "
            f"(hace {format_duration(flow_age_seconds)}). "
            "El lado indica al agresor (taker), no la identidad del operador."
        )
    except Exception as exc:
        st.warning(f"No se pudo actualizar el flujo de operaciones grandes: {exc}")


render_large_trades_monitor()
st.markdown("<br>", unsafe_allow_html=True)

# Inicializar sesión para guardar datos y evitar recargar innecesariamente
if 'run_done' not in st.session_state:
    st.session_state.run_done = False

if run_btn:
    with st.spinner("Descargando datos históricos y ejecutando pipeline..."):
        try:
            # 1. Descargar datos
            df_raw = fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date_iso=start_date_iso
            )
            
            # 2. Agregar indicadores técnicos y características de ML
            df_with_indicators = add_technical_indicators(df_raw)
            df_with_features, feature_cols = generate_ml_features(df_with_indicators)
            # Inferencia conserva las velas recientes sin futuro conocido; el
            # entrenamiento solo usa filas que ya pueden etiquetarse.
            df_inference = df_with_features.dropna(subset=feature_cols).copy()
            df_final = add_target_and_clean(
                df_with_features,
                horizon_bars=target_horizon,
                atr_multiplier=atr_multiplier,
                risk_reward_ratio=risk_reward_ratio,
                fee=transaction_fee,
                slippage=slippage,
            )
            
            # 3. Dividir datos y entrenar modelo
            X_train, X_test, y_train, y_test, df_train, df_test = split_time_series_data(
                df=df_final,
                feature_cols=feature_cols,
                test_size=0.2
            )
            
            model = train_xgboost_model(X_train, y_train, n_splits=5, gap=target_horizon)
            
            # Evaluar modelo
            eval_metrics = evaluate_model(model, X_test, y_test)
            importances = get_feature_importances(model, feature_cols)
            
            # 4. Backtesting
            df_equity, trades, metrics = run_backtest(
                df_test=df_test,
                predictions=eval_metrics['predictions'],
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                atr_multiplier=atr_multiplier,
                risk_reward_ratio=risk_reward_ratio,
                trade_direction=trade_direction,
                fee=transaction_fee,
                slippage=slippage,
                max_holding_bars=target_horizon,
            )
            
            # Guardar en session_state
            st.session_state.df_test = df_test
            st.session_state.df_equity = df_equity
            st.session_state.trades = trades
            st.session_state.metrics = metrics
            st.session_state.eval_metrics = eval_metrics
            st.session_state.importances = importances
            st.session_state.model = model
            st.session_state.feature_cols = feature_cols
            st.session_state.df_final = df_final
            st.session_state.df_inference = df_inference
            st.session_state.run_done = True
            
            st.success("¡Simulación completada con éxito!")
            
        except Exception as e:
            st.error(f"Error durante la ejecución del pipeline: {e}")
            logging.exception(e)

# Renderizar contenido si la simulación ya se ejecutó
if st.session_state.run_done:
    df_test = st.session_state.df_test
    df_equity = st.session_state.df_equity
    trades = st.session_state.trades
    metrics = st.session_state.metrics
    eval_metrics = st.session_state.eval_metrics
    importances = st.session_state.importances
    model = st.session_state.model
    feature_cols = st.session_state.feature_cols
    df_final = st.session_state.df_final
    df_inference = st.session_state.get('df_inference', df_final)
    
    # --- SECCIÓN DE PREDICCIÓN DE COMPRA EN TIEMPO REAL ---
    st.markdown("### 🎯 Predicción y Señal de Compra Sugerida (Modelo + Ballenas)")
    
    # Obtener el último registro disponible para predecir la siguiente vela
    latest_features = df_inference[feature_cols].iloc[[-1]]
    latest_candle = df_inference.iloc[-1]
    
    # Probabilidad del modelo XGBoost para la siguiente vela
    latest_probabilities = model.predict_proba(latest_features)[0]
    prob_short, prob_flat, prob_long = (latest_probabilities * 100.0).tolist()
    signal_threshold = float(getattr(model, 'signal_threshold_', 0.5))
    latest_signal = int(probabilities_to_signals(
        latest_probabilities.reshape(1, -1), signal_threshold
    )[0])
    
    current_close = float(latest_candle['close'])
    current_atr = float(latest_candle['atr'])
    
    # Calcular niveles clave de operación (Setup)
    flow_context = st.session_state.get('flow_pressure', {'score': 0.0, 'label': 'neutral'})
    adjusted_levels = calculate_flow_adjusted_levels(
        current_close,
        current_atr,
        flow_context['score'],
        atr_multiplier,
        risk_reward_ratio,
        transaction_fee,
        slippage,
    )
    selected_setup = adjusted_levels['short'] if latest_signal == -1 else adjusted_levels['long']
    recommended_entry = selected_setup['entry']
    recommended_sl = selected_setup['stop']
    recommended_tp = selected_setup['target']
    sl_distance = abs(recommended_entry - recommended_sl)
    risk_amount_usdt = initial_capital * risk_per_trade
    suggested_btc_qty = risk_amount_usdt / sl_distance if sl_distance > 0 else 0.0
    
    # Verificar estado de Ballenas
    whales_in_df = df_final[df_final['is_whale'] == 1]
    if not whales_in_df.empty:
        last_w = whales_in_df.iloc[-1]
        last_w_type = last_w['whale_type']
        last_w_bars = int(latest_candle['bars_since_last_whale'])
        last_w_mins = float(latest_candle['minutes_since_last_whale'])
        
        if last_w_type == 1 and last_w_bars <= 12:
            whale_alignment = "🟢 Confluencia Alcista Fuerte (Ballena Compradora Reciente)"
            whale_bonus = 5.0
        elif last_w_type == -1 and last_w_bars <= 12:
            whale_alignment = "🔴 Presión Bajista (Ballena Vendedora Reciente)"
            whale_bonus = -5.0
        else:
            whale_alignment = "⚪ Sin actividad de ballenas en las últimas velas"
            whale_bonus = 0.0
    else:
        whale_alignment = "⚪ Sin registros de ballenas"
        whale_bonus = 0.0
        
    # Dictamen final de compra
    trading_enabled = bool(getattr(model, 'threshold_metrics_', {}).get('enabled', True))
    if not trading_enabled:
        signal_action = "🟡 MODELO EN PAUSA (ESPERAR)"
        signal_box_color = "#ffd600"
        signal_desc = "La validación temporal no encontró utilidad neta positiva con los costes configurados."
    elif latest_signal == 1:
        signal_action = "🟢 OPORTUNIDAD DE COMPRA (LONG)"
        signal_box_color = "#00e676"
        signal_desc = f"Señal LONG con probabilidad {prob_long:.1f}% y umbral temporal {signal_threshold * 100:.1f}%."
    elif latest_signal == -1:
        signal_action = "🔴 SEÑAL BAJISTA (SHORT)"
        signal_box_color = "#ff1744"
        signal_desc = f"Señal SHORT con probabilidad {prob_short:.1f}% y umbral temporal {signal_threshold * 100:.1f}%."
    else:
        signal_action = "🟡 MERCADO EN RANGO / NEUTRAL (ESPERAR)"
        signal_box_color = "#ffd600"
        signal_desc = f"El modelo se abstiene: short {prob_short:.1f}%, neutral {prob_flat:.1f}%, long {prob_long:.1f}%."

    pred_col1, pred_col2, pred_col3 = st.columns([1.2, 1.0, 1.0])
    
    with pred_col1:
        st.markdown(f"""
        <div style="background-color: #1e222b; border-left: 6px solid {signal_box_color}; border-radius: 8px; padding: 18px;">
            <div style="font-size: 13px; color: #8a919e; font-weight: bold;">SEÑAL DEL MODELO + BALLENAS</div>
            <div style="font-size: 20px; font-weight: bold; color: {signal_box_color}; margin-top: 5px;">{signal_action}</div>
            <div style="font-size: 13px; color: #e9ecef; margin-top: 8px;">{signal_desc}</div>
            <div style="font-size: 12px; color: #8a919e; margin-top: 5px;">Alineación de Ballenas: {whale_alignment}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with pred_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Entrada ajustada por flujo (orientativa)</div>
            <div class="metric-value" style="color: #00d2ff;">${recommended_entry:,.2f} USDT</div>
            <div class="metric-label">Short {prob_short:.1f}% | Flat {prob_flat:.1f}% | Long {prob_long:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with pred_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Niveles Sugeridos de Riesgo (1:1.5)</div>
            <div style="font-size: 14px; color: #00e676; font-weight: bold;">🎯 Take Profit: ${recommended_tp:,.2f}</div>
            <div style="font-size: 14px; color: #ff1744; font-weight: bold;">🛑 Stop Loss: ${recommended_sl:,.2f}</div>
            <div class="metric-label" style="font-size: 11px; margin-top: 3px;">Tamaño Sugerido: {suggested_btc_qty:.4f} BTC (Riesgo ${risk_amount_usdt:.2f})</div>
        </div>
        """, unsafe_allow_html=True)

    setup_table = pd.DataFrame([
        {
            "Escenario": "LONG (comprar y luego vender)",
            "Entrada ajustada": adjusted_levels['long']['entry'],
            "Stop": adjusted_levels['long']['stop'],
            "Salida objetivo": adjusted_levels['long']['target'],
            "Espera ATR": adjusted_levels['long']['wait_atr'],
        },
        {
            "Escenario": "SHORT (vender y luego recomprar)",
            "Entrada ajustada": adjusted_levels['short']['entry'],
            "Stop": adjusted_levels['short']['stop'],
            "Salida objetivo": adjusted_levels['short']['target'],
            "Espera ATR": adjusted_levels['short']['wait_atr'],
        },
    ])
    st.dataframe(
        setup_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entrada ajustada": st.column_config.NumberColumn(format="$%.2f"),
            "Stop": st.column_config.NumberColumn(format="$%.2f"),
            "Salida objetivo": st.column_config.NumberColumn(format="$%.2f"),
            "Espera ATR": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        f"Ajuste por flujo: {flow_context['score']:+.3f} ({flow_context['label']}). "
        "Los niveles son zonas limit orientativas; sólo son accionables cuando el modelo habilita una señal. "
        "Este ajuste en tiempo real todavía no forma parte del backtest histórico."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- FILA 1: TARJETAS DE MÉTRICAS PRINCIPALES ---
    st.subheader("📊 Métricas de Rendimiento")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    ret_color = "green" if metrics['total_return_pct'] >= 0 else "red"
    hold_color = "green" if metrics['hold_return_pct'] >= 0 else "red"
    
    # Datos de la última ballena detectada en el test set
    whales_in_test = df_test[df_test['is_whale'] == 1]
    if not whales_in_test.empty:
        last_whale = whales_in_test.iloc[-1]
        last_whale_type = "🟢 Compradora (Buy)" if last_whale['whale_type'] == 1 else "🔴 Vendedora (Sell)"
        last_whale_bars = int(df_test.iloc[-1]['bars_since_last_whale'])
        last_whale_mins = float(df_test.iloc[-1]['minutes_since_last_whale'])
        whale_time_str = f"Hace {int(last_whale_mins)} min ({last_whale_bars} velas)" if last_whale_mins < 1440 else f"Hace {last_whale_bars} velas"
    else:
        last_whale_type = "Sin Detección"
        whale_time_str = "N/A"
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Retorno Estrategia (ML)</div>
            <div class="metric-value metric-value-{ret_color}">{metrics['total_return_pct']:.2f}%</div>
            <div class="metric-label">Capital Final: {metrics['final_capital']:.2f} USDT</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Retorno Buy & Hold</div>
            <div class="metric-value metric-value-{hold_color}">{metrics['hold_return_pct']:.2f}%</div>
            <div class="metric-label">Comportamiento del Activo</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value" style="color: #00d2ff;">{metrics['win_rate_pct']:.2f}%</div>
            <div class="metric-label">Factor Ganancia: {metrics['profit_factor']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Máximo Drawdown</div>
            <div class="metric-value metric-value-red">{metrics['max_drawdown_pct']:.2f}%</div>
            <div class="metric-label">Total Trades: {metrics['total_trades']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🐳 Última Ballena (Whale)</div>
            <div class="metric-value" style="color: #e040fb; font-size: 18px;">{last_whale_type}</div>
            <div class="metric-label">{whale_time_str}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- FILA 2: PESTAÑAS DETALLADAS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Evolución del Capital",
        "🔮 Señales de Entrada & Gráfico de Precios",
        "🧠 Evaluación del Modelo ML",
        "📋 Historial de Operaciones"
    ])
    
    # Pestaña 1: Gráfico de Equidad
    with tab1:
        st.subheader("Curva de Crecimiento de Capital")
        
        # Comparar Estrategia de ML vs Hold normalizado
        df_equity['hold_capital'] = initial_capital * (df_equity['price'] / df_equity['price'].iloc[0])
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df_equity['timestamp'],
            y=df_equity['capital'],
            mode='lines',
            name='Estrategia ML (Riesgo Controlado)',
            line=dict(color='#00e676', width=2.5)
        ))
        fig_equity.add_trace(go.Scatter(
            x=df_equity['timestamp'],
            y=df_equity['hold_capital'],
            mode='lines',
            name='Buy & Hold (Mercado)',
            line=dict(color='#ff9100', width=1.5, dash='dash')
        ))
        
        fig_equity.update_layout(
            template='plotly_dark',
            height=600,
            title='Comparación de Equidad: Estrategia de ML vs. Mercado',
            xaxis_title='Fecha',
            yaxis_title='Capital (USDT)',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            xaxis=dict(tickfont=dict(size=13)),
            yaxis=dict(tickfont=dict(size=13)),
            xaxis_title_font=dict(size=15),
            yaxis_title_font=dict(size=15)
        )
        st.plotly_chart(fig_equity, use_container_width=True, config={'scrollZoom': True})
        
    # Pestaña 2: Gráfico de Precios con Señales
    with tab2:
        st.subheader(f"Gráfico de Precios de {symbol} con Indicadores y Operaciones")
        
        # Opciones para optimizar el rendimiento y la visibilidad del gráfico
        chart_range = st.radio(
            "🔍 Seleccionar Rango del Gráfico:",
            options=["Último Mes (30 días)", "Últimos 3 Meses (90 días)", "Todo el Test Set"],
            index=0,
            horizontal=True
        )
        
        # Filtrar datos de test según el rango seleccionado
        max_ts = df_test['timestamp'].max()
        if chart_range == "Último Mes (30 días)":
            min_ts = max_ts - pd.Timedelta(days=30)
            df_test_plot = df_test[df_test['timestamp'] >= min_ts].copy()
        elif chart_range == "Últimos 3 Meses (90 días)":
            min_ts = max_ts - pd.Timedelta(days=90)
            df_test_plot = df_test[df_test['timestamp'] >= min_ts].copy()
        else:
            df_test_plot = df_test.copy()
            
        # Filtrar trades para el rango seleccionado
        df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        if not df_trades.empty:
            df_trades_plot = df_trades[df_trades['entry_time'] >= df_test_plot['timestamp'].min()].copy()
        else:
            df_trades_plot = pd.DataFrame()
        
        fig_prices = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Precio
        fig_prices.add_trace(go.Scatter(
            x=df_test_plot['timestamp'], y=df_test_plot['close'],
            mode='lines', name='Precio Cierre', line=dict(color='#e9ecef', width=1.5)
        ), row=1, col=1)
        
        # EMAs
        fig_prices.add_trace(go.Scatter(
            x=df_test_plot['timestamp'], y=df_test_plot['ema_9'],
            mode='lines', name='EMA 9', line=dict(color='#2979ff', width=1, dash='dot')
        ), row=1, col=1)
        fig_prices.add_trace(go.Scatter(
            x=df_test_plot['timestamp'], y=df_test_plot['ema_21'],
            mode='lines', name='EMA 21', line=dict(color='#ff1744', width=1, dash='dot')
        ), row=1, col=1)
        fig_prices.add_trace(go.Scatter(
            x=df_test_plot['timestamp'], y=df_test_plot['ema_200'],
            mode='lines', name='EMA 200', line=dict(color='#ffd600', width=1.5)
        ), row=1, col=1)
        
        # Graficar puntos de entrada del backtest
        if not df_trades_plot.empty:
            long_entries = df_trades_plot[df_trades_plot['type'] == 'long']
            short_entries = df_trades_plot[df_trades_plot['type'] == 'short']
            
            if not long_entries.empty:
                fig_prices.add_trace(go.Scatter(
                    x=long_entries['entry_time'], y=long_entries['entry_price'],
                    mode='markers', name='Entrada LONG',
                    marker=dict(symbol='triangle-up', size=11, color='#00e676', line=dict(width=1, color='black'))
                ), row=1, col=1)
                
            if not short_entries.empty:
                fig_prices.add_trace(go.Scatter(
                    x=short_entries['entry_time'], y=short_entries['entry_price'],
                    mode='markers', name='Entrada SHORT',
                    marker=dict(symbol='triangle-down', size=11, color='#ff1744', line=dict(width=1, color='black'))
                ), row=1, col=1)
                
        # Graficar Detecciones de Ballenas (Whales)
        whales_buy = df_test_plot[(df_test_plot['is_whale'] == 1) & (df_test_plot['whale_type'] == 1)]
        whales_sell = df_test_plot[(df_test_plot['is_whale'] == 1) & (df_test_plot['whale_type'] == -1)]
        
        if not whales_buy.empty:
            fig_prices.add_trace(go.Scatter(
                x=whales_buy['timestamp'], y=whales_buy['high'] * 1.002,
                mode='markers', name='🐳 Ballena Compradora',
                marker=dict(symbol='star', size=12, color='#e040fb', line=dict(width=1, color='white'))
            ), row=1, col=1)
            
        if not whales_sell.empty:
            fig_prices.add_trace(go.Scatter(
                x=whales_sell['timestamp'], y=whales_sell['low'] * 0.998,
                mode='markers', name='🐳 Ballena Vendedora',
                marker=dict(symbol='star', size=12, color='#ff6e40', line=dict(width=1, color='white'))
            ), row=1, col=1)
            
        # Banda Bollinger
        fig_prices.add_trace(go.Scatter(
            x=df_test['timestamp'], y=df_test['bb_high'],
            mode='lines', name='BB Upper', line=dict(color='rgba(173, 181, 189, 0.3)', width=1)
        ), row=1, col=1)
        fig_prices.add_trace(go.Scatter(
            x=df_test['timestamp'], y=df_test['bb_low'],
            mode='lines', name='BB Lower', line=dict(color='rgba(173, 181, 189, 0.3)', width=1),
            fill='tonexty', fillcolor='rgba(173, 181, 189, 0.05)'
        ), row=1, col=1)
        
        # Volumen (Fila 2) con resalte para ballenas
        vol_colors = ['#e040fb' if w == 1 else 'rgba(0, 229, 118, 0.3)' for w in df_test['is_whale']]
        fig_prices.add_trace(go.Bar(
            x=df_test['timestamp'], y=df_test['volume'],
            name='Volumen', marker_color=vol_colors
        ), row=2, col=1)
        
        fig_prices.update_layout(
            template='plotly_dark',
            height=850,
            hovermode='x unified',
            title=f'Historial de Precios y Señales ({symbol})'
        )
        fig_prices.update_xaxes(tickfont=dict(size=13), row=1, col=1)
        fig_prices.update_xaxes(tickfont=dict(size=13), row=2, col=1, title_text="Fecha", title_font=dict(size=15))
        fig_prices.update_yaxes(tickfont=dict(size=13), row=1, col=1, title_text="Precio (USDT)", title_font=dict(size=15))
        fig_prices.update_yaxes(tickfont=dict(size=11), row=2, col=1, title_text="Volumen", title_font=dict(size=13))
        st.plotly_chart(fig_prices, use_container_width=True, config={'scrollZoom': True})

    # Pestaña 3: Métricas del Modelo ML
    with tab3:
        st.subheader("Evaluación de la Inteligencia del Modelo (XGBoost)")
        
        c_eval1, c_eval2 = st.columns(2)
        
        with c_eval1:
            st.markdown("#### Métricas de Clasificación")
            st.markdown(f"- **Accuracy (Tasa de acierto general)**: {eval_metrics['accuracy']:.4f}")
            st.markdown(f"- **Balanced Accuracy**: {eval_metrics['balanced_accuracy']:.4f}")
            st.markdown(f"- **Precision (Acierto de señales de Compra/Venta)**: {eval_metrics['precision']:.4f}")
            st.markdown(f"- **Recall (Tasa de captura de movimientos)**: {eval_metrics['recall']:.4f}")
            st.markdown(f"- **F1-Score (Equilibrio de precisión y recall)**: {eval_metrics['f1_score']:.4f}")
            st.markdown(f"- **Precisión direccional**: {eval_metrics['directional_precision']:.4f}")
            st.markdown(f"- **Cobertura de señales**: {eval_metrics['signal_coverage'] * 100:.2f}%")
            st.markdown(f"- **Umbral aprendido (solo train)**: {eval_metrics['signal_threshold']:.3f}")
            
            st.markdown("#### Matriz de Confusión (Confusion Matrix)")
            cm = eval_metrics['confusion_matrix']
            # Construir DataFrame para visualizar bonito
            cm_df = pd.DataFrame(
                cm, 
                index=['Real Short', 'Real Flat', 'Real Long'],
                columns=['Modelo Short', 'Modelo Flat', 'Modelo Long']
            )
            st.dataframe(cm_df, use_container_width=True)
            
            st.info("💡 La rentabilidad depende del valor esperado neto de costes, no solo de superar 50% de accuracy.")
            
        with c_eval2:
            st.markdown("#### Importancia de las Características (Feature Importance)")
            # Graficar feature importance
            fig_importance = go.Figure(go.Bar(
                x=importances['importance'],
                y=importances['feature'],
                orientation='h',
                marker_color='#2979ff'
            ))
            fig_importance.update_layout(
                template='plotly_dark',
                xaxis_title='Importancia relativa',
                yaxis_title='Característica',
                yaxis=dict(autorange="reversed"),
                height=400
            )
            st.plotly_chart(fig_importance, use_container_width=True)

    # Pestaña 4: Tabla de Operaciones
    with tab4:
        st.subheader("Registro Detallado de Operaciones (Trades Log)")
        if trades:
            df_trades = pd.DataFrame(trades)
            # Dar formato a las columnas
            df_trades_styled = df_trades.copy()
            df_trades_styled['entry_price'] = df_trades_styled['entry_price'].map(lambda x: f"${x:,.2f}")
            df_trades_styled['exit_price'] = df_trades_styled['exit_price'].map(lambda x: f"${x:,.2f}")
            df_trades_styled['pnl'] = df_trades_styled['pnl'].map(lambda x: f"${x:,.2f}")
            df_trades_styled['fees'] = df_trades_styled['fees'].map(lambda x: f"${x:,.2f}")
            df_trades_styled['slippage'] = df_trades_styled['slippage'].map(lambda x: f"${x:,.2f}")
            df_trades_styled['net_pnl'] = df_trades_styled['net_pnl'].map(lambda x: f"${x:,.2f}")
            df_trades_styled['capital_after'] = df_trades_styled['capital_after'].map(lambda x: f"${x:,.2f}")
            df_trades_styled['risk'] = df_trades_styled['risk'].map(lambda x: f"${x:,.2f}")
            
            st.dataframe(
                df_trades_styled[['type', 'result', 'exit_reason', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'fees', 'slippage', 'net_pnl', 'risk', 'capital_after']],
                use_container_width=True
            )
        else:
            st.warning("No hay operaciones ejecutadas.")
            
else:
    st.info("👈 Selecciona los parámetros en la barra lateral izquierda y haz clic en 'Ejecutar Simulación' para ver los resultados.")
