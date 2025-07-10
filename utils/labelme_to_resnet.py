import json
import os
import shutil
import sys
import argparse
import random

source_folder_path = "raw_image"
target_folder_path = "line_follow_dataset"

train_radio = 8 # 80% train, 20% test, (random.random() * 10 < 8)

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder_path", type=str, help="Path to the source folder containing images and JSONs.")
parser.add_argument("--target_folder_path", type=str, help="Path to the target folder for the processed dataset.")
args = parser.parse_args()

if args.source_folder_path:
    source_folder_path = args.source_folder_path
if args.target_folder_path:
    target_folder_path = args.target_folder_path

train_folder = os.path.join(target_folder_path, 'train')
train_image = os.path.join(train_folder, 'image')
train_label = os.path.join(train_folder, 'label')

test_folder = os.path.join(target_folder_path, 'test')
test_image = os.path.join(test_folder, 'image')
test_label = os.path.join(test_folder, 'label')

if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(train_image):
    os.makedirs(train_image)
if not os.path.exists(train_label):
    os.makedirs(train_label)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)
if not os.path.exists(test_image):
    os.makedirs(test_image)
if not os.path.exists(test_label):
    os.makedirs(test_label)

print(f"Processing data from: {source_folder_path}")
print(f"Outputting to: {target_folder_path}")
print(f"Train/Test split ratio (train_radio): {train_radio}/10 (e.g., 8 means 80% train)")

for filename in os.listdir(source_folder_path):
    if filename.endswith(".jpg"):
        jpg_path = os.path.join(source_folder_path, filename)
        base_name = os.path.splitext(filename)[0] 
        json_path = os.path.join(source_folder_path, f"{base_name}.json")

        rand = random.random() * 10
        destination_image_path_dir = train_image if (rand < train_radio) else test_image
        destination_label_path_dir = train_label if (rand < train_radio) else test_label

        shutil.copy(jpg_path, destination_image_path_dir)
        print(f"Copied {jpg_path} --> {destination_image_path_dir}")

        txt_file_path = os.path.join(destination_label_path_dir, f"{base_name}.txt")

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as json_file:
                    data = json.load(json_file)

                shapes = data.get("shapes", [])
                num_points_found = 0
                x, y = "NaN", "NaN" 

                for shape in shapes:
                    shape_type = shape.get("shape_type")
                    if shape_type == "point":
                        num_points_found += 1
                        if num_points_found == 2:
                            print(f"\nWARNING: Two points appear in {json_path}. Script expects only one point. Using the first one found.")
                            break

                        if num_points_found == 1: 
                            points = shape.get("points", [])
                            if points and len(points[0]) == 2: 
                                x, y = points[0] # point 是 [[x,y]] 形式，所以取 points[0]
                            else:
                                print(f"WARNING: Invalid point data in {json_path} for shape {shape}. Defaulting to NaN NaN.")
                                x, y = "NaN", "NaN"
                            break 

                if num_points_found == 0:
                    # JSON存在，但没有找到任何"point"类型的shape
                    print(f"WARNING: JSON {json_path} exists but contains no 'point' shape. Labeling as NaN NaN 0.")
                    with open(txt_file_path, 'w') as txt_file:
                        txt_file.write("NaN NaN 0")
                else:
                    # 找到点，写入 x y 1
                    with open(txt_file_path, 'w') as txt_file:
                        txt_file.write(f"{int(x)} {int(y)} 1")
                    print(f"Created label for {base_name}.txt with point ({int(x)}, {int(y)}) and flag 1.")

            except json.JSONDecodeError:
                print(f"ERROR: Could not decode JSON from {json_path}. Labeling as NaN NaN 0.")
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write("NaN NaN 0")
            except Exception as e:
                print(f"An unexpected error occurred while processing {json_path}: {e}. Labeling as NaN NaN 0.")
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write("NaN NaN 0")
        else:
            # 如果没有对应的JSON文件
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write("NaN NaN 0")
            print(f"JSON not found for {base_name}.jpg. Created label as NaN NaN 0.")

print("\nDataset preparation complete.")
