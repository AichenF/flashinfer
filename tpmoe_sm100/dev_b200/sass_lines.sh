#!/bin/bash
# usage: sass_lines.sh [extra nvcc flags] -> /tmp/tpb2_lines.txt: "line count" (SASS instructions attributed to each source line of tpmoe_b2.cu)
set -e
nvcc -O3 -gencode=arch=compute_100a,code=sm_100a -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -DB2_NO_TORCH -lineinfo -cubin "$@" -o /tmp/tpb2.cubin tpmoe_b2.cu
nvdisasm -g -c /tmp/tpb2.cubin > /tmp/tpb2.sass
python3 - <<'PY'
import re, collections
cnt = collections.Counter(); cur = None; tot = 0
for l in open('/tmp/tpb2.sass'):
    m = re.search(r'line (\d+)', l)
    if m and ('##' in l or l.strip().startswith('//')):
        if 'tpmoe_b2.cu' in l or 'File' not in l: cur = int(m.group(1))
        else: cur = -1
        continue
    if re.match(r'\s+/\*[0-9a-f]{4,}\*/', l):
        tot += 1; cnt[cur] += 1
open('/tmp/tpb2_lines.txt', 'w').write('\n'.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda x: (x[0] is None, x[0] or 0))))
print('total', tot, 'attributed', sum(v for k, v in cnt.items() if k and k > 0))
PY
