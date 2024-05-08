for logdir in log/202* 
do
	largestN=0
	for eventfi in ${logdir}/events* 
	do 
		n=`wc -l ${eventfi} | awk '{print $1}' `
		if [ ${n} -gt $largestN ]; then
			largestN=$n
		fi
	done
	if [ $largestN -lt 10 ]; then
		echo "removing" $logdir
		rm -r $logdir
	fi
	#
	#	if [ ${n} -lt 10 ]; then 
	#	rm -r $logdir ; 
	#fi ; 
done
