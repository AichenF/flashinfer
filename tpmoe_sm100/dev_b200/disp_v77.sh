# v77 = v76 + all-reduce rework (ARV flags; same arithmetic, bitwise): one-shot builds (M <= 32) ARV=$AOS (8 in-flight source polls +
# 32 exit without fence / returning atomic), the M33-48 build switched to RS + AG (ONESHOT_MAXM=32) and every RS + AG build ARV=$ARS
# (2 T2 + T3 in one in-flight poll loop + 32)
. runs/disp_v76.sh
AOS=${AOS:-40}; ARS=${ARS:-34}
IFS='+' read -ra BL76 <<< "$X"
X77=""
for i in "${!BL76[@]}"; do
  if [ $i -le 14 ]; then f="${BL76[$i]},-DTPB2_ARV=$AOS"
  elif [ $i -eq 15 ]; then f="${BL76[$i]},-DTPB2_ARV=$ARS,-DTPB2_ONESHOT_MAXM=32"
  else f="${BL76[$i]},-DTPB2_ARV=$ARS"; fi
  X77="${X77:+$X77+}$f"
done
X=$X77
