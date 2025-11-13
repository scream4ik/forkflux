run:
	@python -m app.main

linters:
	@pre-commit run --all-files -c .pre-commit-config.yaml
