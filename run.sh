set -x
for dataset in Stocks; do
  for instance in `/bin/ls -d storage/experiments/$dataset/*/*`; do
      echo $instance
      make run command=${instance}/command
  done
done
