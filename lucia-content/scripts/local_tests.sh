Help()
{
   # Display Help
   echo "Runs the experiments defined in the condor file"
   echo "in local in batches of BATCH_NUMBER"
   echo
   echo "Syntax: detector_tests.sh [-h] [-f CONDOR_FILE] [-b BATCH_NUMBER]"
   echo
}

batch=1
# Get the options
while getopts "hb:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      b) batch="$OPTARG";;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac

done


i=0
for agent in PT1; do
  for norm in z-score; do
    for file in recovery_tests; do
      for attack in UniformAttack Empty STAttack VFAttack; do
        for detector in DBSCAN KernelDensity; do
          echo Executing $agent $norm $file $detector $attack
          outfile=./results/out/"$file"_"$norm"_"$agent"_"$detector"_"$attack".out
          python3 "$file".py --agent $agent --norm $norm --detector $detector --attack $attack > $outfile 2>&1 &
          i=$((i+1))
          if [ $((i%$batch)) = 0 ]; then
            echo waiting for finish
            wait
          fi
	      done
      done
    done
  done
done
if [ $((i%$batch)) -ne 0 ]; then
  echo Waiting for finish
  wait
fi
echo All done!
