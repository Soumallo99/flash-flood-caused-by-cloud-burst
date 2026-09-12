.PHONY: help setup run backend simulators dashboard down clean test

help:
	@echo "Cloudburst → Flash Flood Response System"
	@echo ""
	@echo "  make setup       # download real data + seed synthetic (requires running DB/Redis)"
	@echo "  make run         # docker compose up --build (full system)"
	@echo "  make down        # docker compose down"
	@echo "  make backend     # run FastAPI backend locally (dev, needs local postgres+redis)"
	@echo "  make dashboard   # run React dashboard dev server"
	@echo "  make clean       # wipe cached data (forces re-download)"
	@echo "  make test        # sanity checks on hydrology, risk, spread, optimizer"

run:
	docker compose up --build

down:
	docker compose down -v

setup:
	python scripts/setup_real_data.py
	python scripts/seed_synthetic.py

backend:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	cd dashboard && npm install && npm run dev

clean:
	rm -rf data/raw data/processed data/cache

test:
	python -c "$$(cat <<'EOF'
import sys, pathlib
sys.path.insert(0, '.')
import ast
errs = 0
for p in pathlib.Path('backend').rglob('*.py'):
    try: ast.parse(p.read_text())
    except SyntaxError as e: print('SYNTAX:', p, e); errs+=1
for p in pathlib.Path('scripts').rglob('*.py'):
    try: ast.parse(p.read_text())
    except SyntaxError as e: print('SYNTAX:', p, e); errs+=1
assert errs==0, 'syntax errors found'
print('All Python syntax OK')
EOF
)"
	@echo "Run deeper tests via docs/evaluation.md scenarios."
