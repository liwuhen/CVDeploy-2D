import os
import random
import shutil

def random_copy_images(source_folder, destination_folder, num_images=1000):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中的所有图片文件
    image_files = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 随机选择1000张图片
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # 复制选中的图片到目标文件夹
    for image_file in selected_images:
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copy(source_path, destination_path)

if __name__ == "__main__":

    source_folder = '/home/selflearning/dataset/VOC/images/train'            # 带有图片的文件夹路径
    destination_folder = '/home/selflearning/dataset/VOC/images/calib_data_2000'  # 目标文件夹路径
    num_images = 2000                                                        # 需要随机获取的图片数量
    random_copy_images(source_folder, destination_folder, num_images)
