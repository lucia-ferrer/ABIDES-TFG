kill $(ps aux | grep tests | awk '{print $2}')
