deps-pre:
	pip install pip pip-tools --upgrade

deps-compile: deps-pre
	pip-compile requirements.in -o requirements.txt

deps-install: deps-pre
	pip-sync requirements.txt

notebook:
	jupyter notebook

venv:
	python3 -m venv env