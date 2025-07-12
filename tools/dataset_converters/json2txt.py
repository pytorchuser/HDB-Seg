import cv2
import os
import json
import glob
import numpy as np


def convert_json_label_to_yolov_seg_label():
    json_path = r"D:\\dataset\\needle\\2\\24.9.12 00.11.57"  # 本地json路径
    json_files = glob.glob(json_path + "/*.json")
    print(json_files)

    # 指定输出文件夹
    output_folder = "D:/dataset/needle/2/24.9.12 00.11.57/txt"  # txt存放路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for json_file in json_files:
        print(json_file)
        with open(json_file, 'r') as f:
            json_info = json.load(f)

        img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height, width, _ = img.shape
        np_w_h = np.array([[width, height]], np.int32)

        txt_file = os.path.join(output_folder, os.path.basename(json_file).replace(".json", ".txt"))

        with open(txt_file, "w") as f:
            for point_json in json_info["shapes"]:
                txt_content = ""
                np_points = np.array(point_json["points"], np.int32)
                norm_points = np_points / np_w_h
                norm_points_list = norm_points.tolist()
                txt_content += "0 " + " ".join(
                    [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                f.write(txt_content)


convert_json_label_to_yolov_seg_label()