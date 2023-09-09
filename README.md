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

**tracker.py** - auxiliary module for tracking using DeepSORT.

The result video with tracking will be saved as outs/MOT20-07.mp4 file.

Resulting videos are available via link: https://drive.google.com/drive/folders/1Wr14mHCdV5i1XiJPNiSr1dGIFXbWscxC?usp=sharing

## SAHI folders are:

all results with sahi were for slices: [256, 512] and overlap [0.2, 0.5, 0.8]

### with SAHI
**yolo_coco_pretrained**: results from applying sahi with pretrained YOLOv8m model on COCO val dataset (coco-2017/validation/data) with slice (256-512) and overlap (0.2, 0.5, 0.8)
**yolo_coco_finetuned**: results from applying sahi with finetuned on people dataset YOLOv8 model to COCO val dataset (model - best_small.pt)

**yolo_people_pretrained**: results from applying sahi with pretrained YOLOv8m model on PEOPLE dataset (people_test_images)
**yolo_people_finetuned**: results from applying sahi with finetuned on people dataset YOLOv8 model to PEOPLE dataset (model - best_small.pt)


### without SAHI
**yolo_no_slice_coco_pretrained**: results from applying usual YOLOv8m on COCO dataset
**yolo_no_slice_coco_finetuned**: results from applying pretrained on people Yolov8 model (best_small.pt) on COCO

**yolo_no_slice_people_pretrained**: results from applying usual YOLOv8m on PEOPLE dataset
**yolo_no_slice_people_finetuned**: results from applying pretrained on people Yolov8 model (best_small.pt) on PEOPLE


## Checking --postprocess_match_threshold and --postprocess_match_metric

**postprocess_match_threshold**: 0.1, 0.3, 0.5
**postprocess_match_metric**: IOU, IOS

**sahi_people**: applying sahi with pretrained Yolov8m model on PEOPLE with three match thresholds and two match metrics
**sahi_coco**: applying sahi with pretrained Yolov8m model on COCO with three match thresholds and two match metrics
