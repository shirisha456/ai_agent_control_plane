# On Windows, run these commands directly if `make` is unavailable --
# each target is a single copy-pasteable command.
.PHONY: up down logs migrate test test-unit test-db fmt lint psql testdb

up:            ## start postgres + api
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f api

migrate:
	docker compose run --rm migrate

testdb:        ## create the isolated test database
	docker compose exec -T postgres psql -U acp -d postgres -c "CREATE DATABASE acp_test" || true

test: testdb
	pytest -q

test-unit:     ## pure domain tests, no database
	pytest -q tests/unit

test-db: testdb
	pytest -q tests/integration tests/concurrency

test-race: testdb  ## run the CAS race test 20x to prove it is not flaky
	pytest -q tests/concurrency --count=20

psql:
	docker compose exec postgres psql -U acp -d acp

fmt:
	ruff format src tests

lint:
	ruff check src tests
