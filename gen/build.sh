#!/bin/bash

cmake -D CMAKE_BUILD_TYPE=Debug -B build . &&
  make -C build VERBOSE=1

./build/gen_attrs >attrs.py

if command -v ruff; then
  ruff format attrs.py
  ruff check --extend-select=I --fix attrs.py
fi

mv attrs.py ../src/latform/_attrs.py
git add ../src/latform/_attrs.py
