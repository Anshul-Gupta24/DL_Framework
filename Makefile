@ batch_size_ablation:
	echo "Batch size ablation: "
	echo ""
	python test.py --batch_size 4 --num_runs 10
	python test.py --batch_size 8 --num_runs 10
	python test.py --batch_size 16 --num_runs 10
	python test.py --batch_size 32 --num_runs 10
	python test.py --batch_size 64 --num_runs 10
	python test.py --batch_size 128 --num_runs 10
	python test.py --batch_size 256 --num_runs 10

