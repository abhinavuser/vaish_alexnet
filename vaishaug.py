import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

random.seed(42)

class data_augmentor:
    def __init__(self, base_path='tomato_leaves'):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / 'images'
        self.augmented_path = self.base_path / 'augmented_images'

    def apply_aug(self, img, aug_type):
        if aug_type == 'r90':
            return img.rotate(90, expand=True)
        if aug_type == 'r180':
            return img.rotate(180, expand=True)
        if aug_type == 'r270':
            return img.rotate(270, expand=True)
        if aug_type == 'flip_h':
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        if aug_type == 'flip_v':
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        if aug_type == 'bright_up':
            return ImageEnhance.Brightness(img).enhance(1.2)
        if aug_type == 'bright_down':
            return ImageEnhance.Brightness(img).enhance(0.8)
        if aug_type == 'contrast_up':
            return ImageEnhance.Contrast(img).enhance(1.2)
        if aug_type == 'contrast_down':
            return ImageEnhance.Contrast(img).enhance(0.8)
        if aug_type == 'blur':
            return img.filter(ImageFilter.BLUR)
        if aug_type == 'sharp':
            return img.filter(ImageFilter.SHARPEN)
        return img

    def balance_data(self, target=1000):
        aug_list = [
            'r90', 'r180', 'r270',
            'flip_h', 'flip_v',
            'bright_up', 'bright_down',
            'contrast_up', 'contrast_down',
            'blur', 'sharp'
        ]

        train_src = self.images_path / 'train'
        train_dst = self.augmented_path / 'train'
        val_src = self.images_path / 'val'
        val_dst = self.augmented_path / 'val'

        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        healthy = list(train_src.glob('H*.jpg'))
        diseased = list(train_src.glob('D*.jpg'))

        for img_path in healthy + diseased:
            shutil.copy(img_path, train_dst / img_path.name)

        def augment_class(images):
            needed = target - len(images)
            count = 0
            while count < needed:
                for img_path in images:
                    if count >= needed:
                        break
                    img = Image.open(img_path).convert('RGB')
                    aug_type = random.choice(aug_list)
                    new_img = self.apply_aug(img, aug_type)
                    name = f"{img_path.stem}_aug{count}.jpg"
                    new_img.save(train_dst / name, quality=95)
                    count += 1

        augment_class(healthy)
        augment_class(diseased)

        for img_path in val_src.glob('*.jpg'):
            shutil.copy(img_path, val_dst / img_path.name)

        h_count = len(list(train_dst.glob('H*.jpg')))
        d_count = len(list(train_dst.glob('D*.jpg')))
        v_count = len(list(val_dst.glob('*.jpg')))

        print("done")
        print("healthy:", h_count)
        print("diseased:", d_count)
        print("validation:", v_count)
        print("total:", h_count + d_count + v_count)

def vaishnavi():
    obj = data_augmentor()
    obj.balance_data(1000)

if __name__ == "__main__":
    vaishnavi()