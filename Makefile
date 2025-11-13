linters:
	@pre-commit run --all-files -c .pre-commit-config.yaml

ui:
	@python -m streamlit run app/main.py
