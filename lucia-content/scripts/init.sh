Help()
{
   # Display Help
   echo "Creates the required folders the project needs"
   echo
   echo "Syntax: init.sh [-c|h]"
   echo "options:"
   echo "h     Print this Help."
   echo "c     Makes the additional setups needed for the cluster"
   echo
}

# adds python system shebang to files
add_shebang(){
  for file in "$@"; do
    if [ $(head -1 $file | cut -c1-2) != "#!" ]; then
      echo "#!/usr/bin/env python3" | cat - "$file" > temp && mv temp "$file"
    fi
  done
}

cluster=0
# Get the options
while getopts "hc" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      c) # display Help
         cluster=1
         ;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

# get environment list from environment config
envs=$(cat ./config/environments.txt)
# create data and results folders
for env in $envs;do
  mkdir -p ./data/$env
  mkdir -p ./results/$env
done

# if in cluster add shebang to python files and recreate output folders
if [ $cluster = 1 ];then
  add_shebang *.py
  if [ -d output ]; then
    rm -r output
  fi
  mkdir -p output/err
  mkdir -p output/log
fi