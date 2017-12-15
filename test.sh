if (( $# != 2 )); then  # grayscale
	for i in $(seq 0 4)
	do
	    python main.py --is_train False --sample_num $i
	done
else  # RGB
	for i in $(seq 0 4)
	do
	    python main.py --is_train False --sample_num $i --is_grayscale False
	done
fi

cd sample/

if (( $# != 2 )); then  # grayscale
	python compute_psnr.py
else  # RGB
	python compute_psnr.py $1 $2
fi
