# NetDissect Results 

[[Download]](downloadlink)

The folder contains statistics computed by NetDissect (via NetDissect-Lite and ...). These statistics are from three VGG16 models at two layers `conv4_3` and `conv5_3`


| Dataset | Provider | Remark |
|:---:|:---:|:---:|
| ImageNet | TorchVision | .... |
| ImageNet | NetDissect | .... |
| Places365 | NetDissect | ... |


## How to Reproduce the Results?

Please consult https://github.com/heytitle/NetDissect-Lite/wiki.



## How to Prepare the Result Directory?

The downloaded file contains statistics as well as images and HTML files for visualization proposes. In the scope of this project, we are only interested in the statistics. 

Below are two command lines that help prepare `./results`.


```

# Suppose the download file is at ~/downloads/netdissect/result.tar.gz
# and we are in ~/downloads/netdissect.

# extract the tarball
tar -xvf result.tar.gz

# prepare the structure of the result directory
ls   */*/*.csv | xargs -I{} dirname {} | xargs -I{} mkdir -p tmp/{}


# copy all the CSV files to the directory
ls   */*/*.csv | xargs -I{} cp  {} ./tmp/{}


cp -r ./tmp/result DESTINATION

```


[downloadlink]: https://tubcloud.tu-berlin.de/apps/files/?dir=/projects/2022-concept-xai/netdissect&fileid=3186553718#