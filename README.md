# SARAS-Net: Scale And Relation Aware Siamese Network for Change Detection

**Internal use instructions**

**Target:** Change detection aims to find the difference between two images at different times and output a change map.  

Adapation of the network to operate in video SCD.

For more information, please see our paper at [arxiv](https://arxiv.org/abs/2212.01287).


## Frame extraction and ground-truth maps generation.

For trainig the network in video SCD, first step is to extract video frames and ground truth maps.

Video Frames

```ruby
python read_yuv_GBR_420.py
```
to extract GBR frames from the videos.

```ruby
python read_yuv_GRAY_420.py
```
to extract grayscale frames from the videos.

Ground Truth


Use "GT_map_from_csv_file.py" to read the .csv file with the ground truth and generate the correct set of ground truth map.

## Data structure

For model training seperate the dataset as the following structure.

### Train Data Path
```ruby
train_dataset  
  |- train_dataset 
      |- image1, image2, gt  
  |- val_dataset  
      |- image1, image2, gt  
  |- train.txt
  |- val.txt
```  

### Test Data Path
```ruby
test_dataset  
  |- A 
      |- image1 
  |- B  
      |- image2 
  |- label
      |- gt 
```
The format of `train.txt` and `val.txt` please refer to `SARAS-Net/train_dataset/train.txt` and `SARAS-Net/train_dataset/val.txt`. To generate the `train.txt` and `val.txt` run:

```ruby
python train_test_txt_data_writer.py
```


## Train
You can find `SARAS-Net/cfgs/config.py` to set the training parameter.
```ruby
python train.py
```

## Test  
After training, you can put weight in `SARAS-Net/`.  
Then, run a cal_acc.py to get started as follows: 
```ruby
python cal_acc.py
```
You can set `show_result = True` in `cal_acc.py` to show the result for each pairs.

To directly extract change maps for every frames of the video, run

```ruby
python change_map_from_video.py
```
and run

```ruby
python SCD_from_map.py
```

which extracts .txt files with the list of frames and the detection parameter from each change map.
