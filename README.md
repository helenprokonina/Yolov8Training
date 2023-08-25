# Yolov8Training


First **make_frames_from video.py** has some auxiliary functions: 

* making frames from videos "people.mp4", "people_from_above.mp4" ans "people_walking_on_street.mp4";
* collect images and annotations for these frames in datasets/images and datasets/labels
* add some images and annotations from OID (using downloader.py, image_lists - list of images ids, that need to be downloaded)
```
python downloader.py image_lists/image_lists.txt --download_folder=datasets/images --num_processes=5 
```

In **train_yolo8.py** the fine_tuning of pre-trained of YOLOv8 model goes for 20 epochs.
All settings are in the **data_config.yaml**

```
python train_yolo8.py 
```

Then in **predict_yolo8.py** prediction on the test image is performed.
```
python predict_yolo8.py 
```

In **track_yolo8.py** tracking with SAHI (sliding window prediction) using YOLOv8 takes place. 
**video_path** is path to the mp4 file with sequence
```
python track_yolo8.py --video_path MOT20-07.mp4
```

The result video with tracking will be saved as outs/MOT20-07.mp4 file.

Resulting videos are available via link: https://drive.google.com/drive/folders/1-tio25jy6FH8ZH5Qeeks7YPiX2tqjI-E?usp=sharing
