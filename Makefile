fast-test:
	pytest -m "not slow and not gpu" tests/*

test:
	pytest tests/*