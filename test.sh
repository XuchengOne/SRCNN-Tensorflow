for i in $(seq 0 4)
do
    python main.py --is_train False --sample_num $i
done

cd sample/
python compute_psnr.py
