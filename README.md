# Binance ML Trading Dashboard

Dashboard experimental de trading para Binance Spot con:

- Descarga y caché local de velas OHLCV.
- Ingeniería de características técnicas.
- Modelo XGBoost `short / flat / long`.
- Validación temporal walk-forward y abstención cuando no hay utilidad estimada.
- Backtest con comisiones, slippage y gestión de riesgo.
- Monitor de compras y ventas agresoras grandes.
- Volumen taker comprador/vendedor de 24 y 72 horas.
- Presión de flujo y zonas orientativas de entrada, stop y objetivo.
- Pronóstico del precio de cierre a 15 min, 30 min, 1 h y 1 día con
  intervalos cuantiles calibrados y cobertura histórica fuera de muestra.

## Requisitos

- Python 3.10 o superior.

## Instalación

```powershell
git clone https://github.com/ProTady/mybinanc.git
cd mybinanc
python -m pip install -r requirements.txt
```

## Ejecutar el dashboard

```powershell
python -m streamlit run app.py
```

Después abre [http://localhost:8501](http://localhost:8501).

## Ejecutar las pruebas

```powershell
python -m unittest discover -s tests -v
```

La base `trading_data.db` se crea localmente y no se incluye en Git.

> Este proyecto es experimental y educativo. Las señales y precios mostrados no
> garantizan resultados ni constituyen asesoramiento financiero.
