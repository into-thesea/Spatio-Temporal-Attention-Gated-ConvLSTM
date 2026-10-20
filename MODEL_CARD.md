# Model Card: STAGC

STAGC here stands for Spatio-Temporal Attention Gated ConvLSTM. It is the name
used in this project, not an established public traffic-forecasting model.

## Status

Personal learning and portfolio project. Code and documentation were partly
drafted with AI assistance; the reported results were produced by running the
repository.

## Intended use

Regional short-term traffic speed forecasting for navigation ETA, fleet
dispatch, and traffic monitoring. The model returns a 16x16 speed map for the
next hour at 5-minute intervals.

## Not intended for

- Lane-level ETA or turn-by-turn routing
- Incident detection or emergency response
- Direct traffic signal control
- Long-term planning beyond one hour
- Safety-critical decisions without human review

## Training data

- METR-LA: 207 loop detectors, Los Angeles County
- PeMS-BAY: 325 sensors, San Francisco Bay Area
- 5-minute resolution; 12 input steps and 12 output steps
- Sensors are mapped to a 16x16 grid; missing readings are excluded from the
  loss and metrics.

## Metrics

METR-LA test set, observed sensor cells:

| Configuration | mean MAE | 60-min MAE | 60-min RMSE | 60-min MAPE |
|---|---:|---:|---:|---:|
| linear encoding | 2.711 | 3.099 | 5.302 | 6.84% |
| cyclic encoding | 2.678 | 3.028 | 5.209 | 6.70% |
| cyclic + residual | 2.685 | 3.104 | 5.333 | 6.97% |

PeMS-BAY test set:

| Configuration | mean MAE | 60-min MAE | 60-min RMSE | 60-min MAPE |
|---|---:|---:|---:|---:|
| linear encoding | 1.432 | 1.731 | 3.340 | 3.28% |
| cyclic encoding | 1.438 | 1.717 | 3.265 | 3.25% |
| cyclic + residual | 1.386 | 1.723 | 3.258 | 3.26% |

## Limitations

- 16x16 resolution loses lane-level and some local road detail.
- One-hour horizon only.
- No weather, incident, flow, or queue data.
- Performance may degrade during non-recurring congestion.
- The current service is a reference implementation without authentication,
  rate limiting, or monitoring.

## Operational notes

- Use the reported metrics only on the same data split and masking protocol.
- Treat predictions as decision support, not as ground truth.
- Re-train or fine-tune before applying the model to a new city.
