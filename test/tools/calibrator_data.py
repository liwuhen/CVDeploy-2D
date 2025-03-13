import os
import random
import shutil
import argparse
from tqdm import tqdm

def random_copy_images(source_folder, destination_folder, num_images=1000):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    else:
        os.rmdir(destination_folder)

    # 获取源文件夹中的所有图片文件
    image_files = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 随机选择1000张图片
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # 复制选中的图片到目标文件夹
    for image_file in tqdm(selected_images):
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copy(source_path, destination_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, help="image data path")
    parser.add_argument('--dest_path',type=str, help="label data path")
    parser.add_argument('--num_images', type=int, help="save result path")
    args = parser.parse_args()

    random_copy_images(args.src_path, args.dest_path, args.num_images)
