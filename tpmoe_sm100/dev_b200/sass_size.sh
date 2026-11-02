#!/bin/bash
# usage: sass_size.sh SO  -> per-function SASS instruction counts / bytes
cuobjdump -sass "$1" > /tmp/sass_$$.txt
awk '/Function :/ {if (fn) print n, fn; fn=$3; n=0} /^        \/\*[0-9a-f]+\*\// {n++} END {print n, fn}' /tmp/sass_$$.txt | sort -rn | head -12
rm -f /tmp/sass_$$.txt
