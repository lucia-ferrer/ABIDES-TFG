while true; do
  clear
  total=$(cat /proc/meminfo | grep -oE "MemTotal:\s+[0-9]+" | grep -oE "[0-9]+")
  available=$(cat /proc/meminfo | grep -oE "MemAvailable:\s+[0-9]+" | grep -oE "[0-9]+")
  used=$(($total-$available))
  bars=75
  used_bars=$(($used*$bars/$total))

  printf '█%.0s' $(eval "echo {1.."$(($used_bars))"}")
  printf '░%.0s' $(eval "echo {1.."$(($bars-$used_bars))"}")
  printf '  %s GB / %s GB\n' $(echo "scale=2 ; $used / (1024*1024)" | bc) $(echo "scale=2 ; $total / (1024*1024)" | bc)

  for file in $(ls results/out); do
    echo "$(cat results/out/$file | grep /50 | sed s/].*/]/ | tail -n 1)" $file
  done
  sleep 2
done