compile:
	pip-compile setup.py
compile-dev:
	pip-compile requirements-dev.in
compile-plugins:
	pip-compile requirements-plugins.in
compile-all:
	pip-compile setup.py
	pip-compile requirements-dev.in
	pip-compile requirements-plugins.in
sync:
	pip-sync requirements.txt requirements-dev.txt requirements-plugins.txt
sync-fix-m1:
	arch -arm64 pip-sync requirements.txt requirements-dev.txt requirements-plugins.txt --pip-args "--no-cache-dir --force-reinstall"
latest-nbdev:
	pip install git+https://github.com/fastai/nbdev.git --upgrade
exec:
	for f in `find nbs -name '*.ipynb' -not -name '*checkpoint*'`; do \
		echo $$f && exec_nb $$f --dest $$f; \
	done
exec-modules:
	for f in `find nbs -name '*.ipynb' -not -name '*checkpoint*' -not -path 'nbs/guide*'`; do \
		echo $$f && exec_nb $$f --dest $$f; \
	done
