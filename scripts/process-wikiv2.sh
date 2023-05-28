#!/bin/bash
PYTHONPATH=src python src/scripts/build_db.py data/wiki-pages data/fever/fever.db
PYTHONPATH=src python src/scripts/alt/matrix_data.py data/fever/fever.db data/matrix/
PYTHONPATH=src python src/scripts/alt/build_tfidf.py data/fever/fever.db data/matrix/ data/index/
