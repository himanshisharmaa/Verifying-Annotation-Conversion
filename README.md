### annotation_conversion.py
This file is used to convert the annotations from one format to another. It is task specific, works according to the task mentioned like object detection or instance segmentation. 

### visualize_annotation.py
This file is used for visualizing and verifying the annotations. After conversion the annotation_conversion.py file will use this for visualizing the annotations. 

Output to this code:
 
![comparison_bart_pic_0000](https://github.com/user-attachments/assets/bfd58562-f25a-4490-a008-2b1b042129a8)


To run "annotation_conversion.py" file run the following command :
    
    python annotation_conversion.py --image_dir [path/to/image directory] --input_path [path/to/input directory or file] --target_format [target_format choices:[COCO,YOLO,VOC]] --task [choices:[object_detection,instance_segmentation]] --output_dir [path/to/output directory]
