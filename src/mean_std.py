import numpy as np
from PIL import Image
import os
from categories import categories, class_dirs


class_labels = categories
class_dirs = class_dirs

class_means = []
class_stds = []
for i, class_dir in enumerate(class_dirs):
    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = 0
    for filename in os.listdir(class_dir):
        img = Image.open(os.path.join(class_dir, filename)).convert('RGB')
        # pixels = np.asarray(img)
        # print("{} - {}".format(pixels.min(), pixels.max()))
        img_np = np.array(img)
        img_np = img_np / 255.0
        mean += np.mean(img_np, axis=(0,1))
        std += np.std(img_np, axis=(0,1))
        num_images += 1
    mean /= num_images
    std /= num_images
    class_means.append(mean)
    class_stds.append(std)
    print(f"Class {i+1} - Mean: {mean}, Std: {std}")

print(np.mean(class_means, axis=0))
print(np.mean(class_stds, axis=0))

