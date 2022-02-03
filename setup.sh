#! /bin/bash
# Copy necessary files to build the full book
# If an argument is provided, files will be sent to that folder.
# Otherwise default folder is ./book/data



if [ $# -eq 0 ]
    then
        DATA_DIR="./book/data"
else
    DATA_DIR=$1
fi
    
echo "Copying files to $DATA_DIR..."

# Create if non-existent
[ ! -d $DATA_DIR ] && mkdir $DATA_DIR

download () {
    # $1 = fileid, $2 = filename
    wget --load-cookies $DATA_DIR/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies $DATA_DIR/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf $DATA_DIR/cookies.txt
}

if [ -d /sdf/group/neutrino/ldomine ] # Get from SDF
    then
        # Copy weight files
        [ ! -f $DATA_DIR/weights_full_mpvmpr_082021.ckpt ] && scp /sdf/group/neutrino/ldomine/weights_full_mpvmpr_082021.ckpt $DATA_DIR
        echo "- weights_full_mpvmpr_082021.ckpt [1/2 done]"
        # copy small dataset file
        [ ! -f $DATA_DIR/mpvmpr_082021_test_small.root ] && scp /sdf/group/neutrino/ldomine/mpvmpr_082021_test_small.root $DATA_DIR
        echo "- mpvmpr_082021_test_small.root [2/2 done]"
else # Get from Google Drive
        # TODO update
        # Copy weight files
        [ ! -f $DATA_DIR/weights_full5_snapshot-999.cpkt ] && download "1-ptcD6dHyVtxdgfo6dQLdUSrSZPlnvlz" "weights_full5_snapshot-999.cpkt"
        echo "- weights_full5_snapshot-999.cpkt [1/3 done]"
        [ ! -f $DATA_DIR/weights_ppn3_snapshot-1999.ckpt ] && download "155yaJ6YMEZmZBGkT8DYQijakP919dOuJ" "weights_ppn3_snapshot-1999.ckpt"
        echo "- weights_ppn3_snapshot-1999.ckpt [2/3 done]"
        # copy small dataset file
        [ ! -f $DATA_DIR/wire_mpvmpr_2020_04_test_small.root ] && download "1UNPtKemYkUYuLc2kGZmjKftFHKu5uXbG" "wire_mpvmpr_2020_04_test_small.root"
        echo "- wire_mpvmpr_2020_04_test_small.root [3/3 done]"
fi

# Copy inference configuration file
[ ! -f $DATA_DIR/inference.cfg ] && wget -O $DATA_DIR/inference.cfg https://raw.githubusercontent.com/DeepLearnPhysics/lartpc_mlreco3d_tutorials/master/book/data/inference.cfg 

# Save data directory for tutorials - absolute path
export DATA_DIR="$(realpath $DATA_DIR)"
echo "Set DATA_DIR = $DATA_DIR"
echo "... done."
