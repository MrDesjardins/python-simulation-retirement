# python-simulation-retirement

# Data

Download:
Download:

1. https://shillerdata.com/

# Environment

```sh
```sh
uv init
uv init
``` 

# Run Code

```sh 
uv run 01_success.py
uv run 02_simulation_lines.py
uv run 03_historical_value.py
```



# Simulation with Hyperparameter Optimization

```sh
uv run 04_tuning.py
# or
uv run 04_tuning_improved.py

uv run optuna-dashboard sqlite:///db.sqlite3
uv run optuna-dashboard sqlite:///db_05.sqlite3uv run optuna-dashboard sqlite:///db_05.sqlite3
```


# Unit Test

```sh
uv run pytest -v -s ./*_test.py 
```