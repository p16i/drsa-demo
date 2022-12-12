#!/usr/bin/bash

rsync -r --progress \
    mpg-server:~/artifact-repos/concept-xai/artifacts/2022-12-paper-submission/raw-figure1/ \
    ./data/raw

# Candidates: [ 74 152 336 447]
# Remark: be careful with the index. We have to +2 when looking at training-samples.csv
IMAGES="
train/n02802426/n02802426_11794.JPEG
train/n02802426/n02802426_2959.JPEG
train/n02802426/n02802426_20198.JPEG
train/n02802426/n02802426_7332.JPEG
"
# convert the string to array
IMAGES=($IMAGES)

for i in "${!IMAGES[@]}"
do 
    FILENAME="${IMAGES[$i]}"
    echo "Loading $i: $FILENAME"
    rsync -r --progress mpg-server:~/datasets/imagenet/$FILENAME ./data/images/img-$i.jpg
done
