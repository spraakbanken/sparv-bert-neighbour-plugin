

default: run-all-tests

PLATFORM := `uname -o`
PROJECT := "sparv_kb_ner"

INVENV := if env_var_or_default('VIRTUAL_ENV', "") == "" { "rye run" } else { "" }

info:
	@echo "Platform: {{PLATFORM}}"
	@echo "INVENV: '{{INVENV}}'"

alias dev := install-dev

# setup development environment
install-dev:
	rye sync 

alias test := run-all-tests

# run all tests
run-all-tests:
	{{INVENV}} pytest -vv tests

# run all doc tests
run-doc-tests:
	{{INVENV}} python -m doctest -v karp_lex/value_objects/unique_id.py

# run all tests with coverage collection
run-all-tests-w-coverage:
	{{INVENV}} pytest -vv --cov={{PROJECT}}  --cov-report=xml tests

# check types
type-check:
	{{INVENV}} mypy {{PROJECT}}

# lint the code
lint:
	{{INVENV}} ruff {{PROJECT}} tests

lint-refactorings:
	{{INVENV}} pylint --disable=C,W,E --enable=R {{PROJECT}} tests

bumpversion-major:
	{{INVENV}} bump2version major

bumpversion-minor:
	{{INVENV}} bump2version minor

bumpversion:
	{{INVENV}} bump2version patch

# run formatter(s)
fmt:
	{{INVENV}} black {{PROJECT}} tests

# check formatting
check-fmt:
	{{INVENV}} black --check {{PROJECT}} tests
