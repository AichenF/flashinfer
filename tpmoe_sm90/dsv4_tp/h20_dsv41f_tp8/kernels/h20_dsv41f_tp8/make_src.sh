#!/bin/bash
# make_src.sh : refresh suite_adapter/src from the repo's committed HEAD (tp8v41.cu + headers + host + weight prep), record the commit
cd "$(dirname "$0")/.."
rm -rf suite_adapter/src && mkdir -p suite_adapter/src
for f in tp8v41.cu tp8v41.py v41_prep.py $(git ls-files | grep -E "^[^/]+\.cuh$"); do git show HEAD:$f > suite_adapter/src/$f; done
git rev-parse --short HEAD > suite_adapter/src/COMMIT
( cd suite_adapter/src && sha256sum * > SHA256SUMS )
echo "src from $(cat suite_adapter/src/COMMIT): $(ls suite_adapter/src | wc -l) files"
