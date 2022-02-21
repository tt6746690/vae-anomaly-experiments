
note_proc:
	jupyter nbconvert --to script note_proc.ipynb
	/data/vision/polina/shared_software/miniconda3/envs/inr/bin/python note_proc.py

note_train:
	jupyter nbconvert --to script note_train.ipynb
	/data/vision/polina/shared_software/miniconda3/envs/inr/bin/python note_train.py


create_env:
	conda env create -f inr-mr.yml
	pip3 install -r requirements.txt

	# https://github.com/MIC-DKFZ/vae-anomaly-experiments/issues/1
	# https://github.com/MIC-DKFZ/batchgenerators/tree/david-crap    0cc8116
	pip install git+https://github.com/MIC-DKFZ/batchgenerators.git@david-crap

update_env:
	conda env update -f inr-mr.yml --prune