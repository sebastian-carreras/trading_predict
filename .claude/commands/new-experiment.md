# New Experiment

Scaffold a new model variant within an existing strategy.

## Arguments
$ARGUMENTS: <strategy> <variant_name> [model_type]
Example: e1 attention_gru AttentionGRU

If model_type is omitted, default to the strategy's current model type.

## Instructions

1. Look at the existing structure under `src/<strategy>/` for patterns
2. Read the current champion model class (e.g., `src/e1/gru.py` for E1) as a template
3. Read the training pipeline (e.g., `src/e1/train_pipeline.py`) for the training pattern
4. Create the following files based on existing patterns:
   - `src/<strategy>/<variant_name>.py` - New model class (extend from existing model or create new)
   - `src/<strategy>/train_<variant_name>_pipeline.py` - Training pipeline using the new model
5. Add a new strategy variant section to `src/config/base.yaml` under `strategies:` if needed
6. Print a checklist of:
   - Files created
   - Config sections to review
   - How to run the first training: `python -m src.<strategy>.train_<variant_name>_pipeline --tickers AAPL`
   - How to register as candidate: the pipeline will auto-register via lifecycle integration
   - How to compare: `/compare-models <strategy> AAPL`
   - How to promote: `/promote-model <strategy> AAPL`

Keep the new code minimal - follow existing patterns exactly, only change the model class.
