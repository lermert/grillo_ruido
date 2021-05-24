
# activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grillo

# pull git repo
cd codes/ants_2/
git stash
git pull
git stash apply
cd


echo "update git repo"
date

# copy data from s3
# -config files
aws s3 cp s3://ruido/config_preprocess_rsvm.json ccs_ants/input/config_preprocess.json
# - raw data
aws s3 cp --recursive --exclude "*" --include "*.001" s3://ruido/RSVM/AOVM/ ccs_ants/data/raw/
# - meta data
aws s3 cp --recursive s3://ruido/stationxml ccs_ants/meta/stationxml/
python copy_stationxml.py

echo "copied data"
date


# process data
cd ccs_ants
ls -ltr meta/stationxml
mpirun -np 1 ants preprocess

echo "preprocessing done, reuploading results"
date

# upload results to s3
tar -cvf processed_data_rsvm.tar data/processed/
aws s3 cp processed_data_rsvm.tar s3://ruido/

echo "done"
date
