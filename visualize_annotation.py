import cv2
import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_coco(image_dir, annotation_path, task, output_dir):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    ann_images={}

    for image_info in coco_data['images']:
        image_path = os.path.join(image_dir, image_info['file_name'].strip())
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read image {image_info['file_name']}")
            continue

        for ann in coco_data['annotations']:
            if ann['image_id'] == image_info['id']:
                if task == 'object_detection':
                    # Draw bounding box
                    bbox = ann['bbox']
                    xmin, ymin, width, height = bbox
                    xmax = xmin + width
                    ymax = ymin + height
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

                elif task == 'instance_segmentation':
                    # Draw polygon
                    if 'segmentation' in ann:
                        polygon = np.array(ann['segmentation'][0]).reshape((-1, 2))
                        cv2.polylines(image, [polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        ann_images[image_info['file_name'].strip()]={"coco":image}
        # print(image_info['file_name'])
        # output_image_path = os.path.join(output_dir, "coco_"+image_info['file_name'])
        # cv2.imwrite(output_image_path, image)
        # print(f"Saved visualization for {image_info['file_name']} to {output_image_path}")
    return ann_images


def visualize_yolo(image_dir, annotation_dir, task, output_dir):
    ann_images={}
    class_file = os.path.join(annotation_dir, 'classes.txt')
    if not os.path.exists(class_file):
        print(f"Class file not found: {class_file}")
        return

    with open(class_file, 'r') as f:
        classes = f.read().splitlines()

    for filename in os.listdir(annotation_dir):
        # print(filename)
        if filename.endswith('.txt') and filename != 'classes.txt':
            image_name = filename.replace('.txt', '.jpg').strip()
           
            image_path = os.path.join(image_dir, image_name)
            
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image {image_name}")
                continue

            annotation_path = os.path.join(annotation_dir, filename)
           
            with open(annotation_path, 'r') as f:
                annotations = f.readlines()

            for annotation in annotations:
                elements = annotation.strip().split()
                class_id = int(elements[0])
                if task == 'object_detection':
                    # Draw bounding box
                    x_center, y_center, width, height = map(float, elements[1:])
                    img_h, img_w = image.shape[:2]
                    xmin = int((x_center - width / 2) * img_w)
                    ymin = int((y_center - height / 2) * img_h)
                    xmax = int((x_center + width / 2) * img_w)
                    ymax = int((y_center + height / 2) * img_h)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                elif task == 'instance_segmentation':
                    # Draw polygon
                    polygon = np.array(list(map(float, elements[1:]))).reshape(-1, 2)
                    polygon[:, 0] *= image.shape[1]  # Scale x
                    polygon[:, 1] *= image.shape[0]  # Scale y
                    cv2.polylines(image, [polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            ann_images[image_name]={"yolo":image}
            # print(image_name)
            # output_image_path = os.path.join(output_dir, "yolo_"+image_name)
            # cv2.imwrite(output_image_path, image)
            # print(f"Saved visualization for {image_name} to {output_image_path}")
    return ann_images

def visualize_voc(image_dir, annotation_dir, task, output_dir):
    ann_images={}
    for filename in os.listdir(annotation_dir):
        # print(filename)
        if filename.endswith('.xml'):
            tree = ET.parse(os.path.join(annotation_dir, filename.strip()))
            root = tree.getroot()

            image_name = root.find('filename').text.strip()
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image {image_name}")
                continue

            for obj in root.findall('object'):
                if task == 'object_detection':
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                elif task == 'instance_segmentation' and obj.find('segmentation') is not None:
                    # Draw polygon for segmentation
                    polygon_points = obj.find('segmentation').text.split(' ')
                    polygon = np.array([list(map(float, point.split(','))) for point in polygon_points]).reshape(-1, 2)
                    cv2.polylines(image, [polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            ann_images[image_name]={"voc":image}
            # print(image_name)
            # output_image_path = os.path.join(output_dir, "voc_"+image_name)
            # cv2.imwrite(output_image_path, image)
            # print(f"Saved visualization for {image_name} to {output_image_path}")
    return ann_images

def write_outputs(ann_img,converted_ann_img,output_dir):
    print(ann_img)
    print(converted_ann_img)
    print(output_dir)
    for filename,img in ann_img.items():
        print(filename)
        if filename not in converted_ann_img:
            print(f"Skipping {filename}, not found in converted annotations.")
            continue
        
        img_before=img[next(iter(img))]
        img_before_format=next(iter(img))
        
        img_after=converted_ann_img[filename][next(iter(converted_ann_img[filename]))]
        img_after_format=next(iter(converted_ann_img[filename]))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_before)
        axes[0].set_title(f"Before (Format: {img_before_format})")
        axes[0].axis('off')  # Hide axis for cleaner look

        # Display the after image in the right subplot
        axes[1].imshow(img_after)
        axes[1].set_title(f"After (Format: {img_after_format})")
        axes[1].axis('off')
        fig.suptitle(f"Before and After Comparison: {filename}", fontsize=16)

        # Save the comparison image to the output directory
        output_image_path = os.path.join(output_dir, f"comparison_{filename}")
        plt.savefig(output_image_path)
        plt.close(fig)  # Close the figure after saving to free memory
        
        print(f"Saved comparison image to {output_image_path}")

def main_fn(image_dir, annotation_path, converted_annotation_path, task, output_dir):
    print("Main")
    print(image_dir)
    print(annotation_path)
    print(converted_annotation_path)
    print(task)
    print(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if annotation_path.endswith('.json'):
        print("Visualizing COCO format...")
        ann_img=visualize_coco(image_dir, annotation_path, task, output_dir)
    elif os.path.isdir(annotation_path) and os.listdir(annotation_path)[0].endswith('.xml'):
        print("Visualizing VOC format...")
        ann_img=visualize_voc(image_dir, annotation_path, task, output_dir)
    elif os.path.isdir(annotation_path):
        print("Visualizing YOLO format...")
        ann_img=visualize_yolo(image_dir, annotation_path, task, output_dir)
    else:
        print("Unsupported annotation format")

    if converted_annotation_path.endswith('.json'):
        print("Visualizing COCO format...")
        converted_ann_img=visualize_coco(image_dir, converted_annotation_path, task, output_dir)
    elif os.path.isdir(converted_annotation_path) and os.listdir(converted_annotation_path)[0].endswith('.xml'):
        print("Visualizing VOC format...")
        converted_ann_img=visualize_voc(image_dir, converted_annotation_path, task, output_dir)
    elif os.path.isdir(converted_annotation_path):
        print("Visualizing YOLO format...")
        converted_ann_img=visualize_yolo(image_dir, converted_annotation_path, task, output_dir)
    else:
        print("Unsupported annotation format")

    write_outputs(ann_img,converted_ann_img,output_dir)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--image_dir",type=str,required=True,help="Path to the image Directory")
    ap.add_argument("-a","--annotation",type=str,required=True,help="Path to the annotation directory or file")
    ap.add_argument("-c","--converted_annotation",type=str,required=True,help="Path to the converted annotation path or file")
    ap.add_argument("-t","--task",type=str,required=True, choices=["object_detection","instance_segmentation"],help="Name of the task for which annotation has taken place")
    ap.add_argument("-o","--output_dir",type=str,default="Comparison/",help="Path to the the output directory")

    args=vars(ap.parse_args())
    image_dir = args["image_dir"]
    annotation_path = args["annotation"]
    converted_annotation_path = args["converted_annotation"]
    task = args["task"]
    # image_dir = "E:\DeepLearning\Annotations\drone_dataset\\valid\\images"
    # annotation_path = "E:\DeepLearning\Annotations\drone_dataset\\valid\labels"
    # converted_annotation_path = "E:\DeepLearning\Annotations\Outputs\converted_coco_dataset.json"
    # task = "object_detection"
    output_dir = args["output_dir"]

    main_fn(image_dir, annotation_path, converted_annotation_path, task, output_dir)
