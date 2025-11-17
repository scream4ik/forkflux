linters:
	@pre-commit run --all-files -c .pre-commit-config.yaml

ui:
	@python -m streamlit run app/main.py

test-ai:
	@deepeval test run tests/ai/ -c

test-app:
	@python -m pytest -s tests/app
