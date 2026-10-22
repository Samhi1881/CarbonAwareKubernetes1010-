.PHONY: install test forecast simulate spike
install:
	pip install -e ".[dev]"
test:
	pytest -q
forecast:
	carbon-autoscaler forecast-eval
simulate:
	carbon-autoscaler simulate --plot
spike:
	carbon-autoscaler simulate --load spike --initial-replicas 2 --outdir results_spike --plot
