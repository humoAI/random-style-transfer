
from glob import glob
import os, sys, numpy as np, cv2
from PIL import Image
this_folder = os.getcwd()
sys.path.insert(0, this_folder)
from src.models.WCT import WCT
class Generator:
    def __init__(self, alpha=0.5) -> None:
        self.photos_folder = "src/data/datasets/Painter_Myself/photo_jpg"
        self.monet_folder = "src/data/datasets/Painter_Myself/monet_jpg"
        self.wct = WCT("src/data/universal_style_transfer_weights/pytorch", alpha=alpha).cuda()
        self.images = {"photos":[], "monet":[]}
        self.load_images()
    def load_images(self):
        photo_paths = glob(self.photos_folder+"/*.jpg")
        for photo_path in photo_paths:
            self.images["photos"].append((photo_path, Image.open(photo_path).convert('RGB')))
        monet_paths = glob(self.monet_folder+"/*.jpg")
        for monet_path in monet_paths:
            self.images["monet"].append((monet_path, Image.open(monet_path).convert('RGB')))
    def generate(self):
        """
        it will generate images  using randomly selected style images
        """
        for photo_path, content_img in self.images["photos"]:
            random_style_index = np.random.randint(0, len(self.images["monet"]))
            style_img = Image.open(self.images["monet"][random_style_index][0]).convert('RGB')
            content_img, style_img= self.wct.preprocess(content_img, style_img, fineSize=256)
            result_img = self.wct(content_img.cuda(), style_img.cuda())
            result_img = result_img.detach().data.cpu().float()
            result_img = result_img.numpy()*255#, 0, -1)
            result_img= result_img[0].transpose(1,2,0)
            result_img[result_img < 0] = 0
            result_img[result_img > 255] = 255
            result_img = np.uint8(result_img)
            
            new_path =photo_path.replace("photo_jpg", "images")

            dirname = os.path.dirname(new_path)
            if(not os.path.exists(dirname)):
                os.makedirs(dirname)
            cv2.imwrite(new_path, result_img)

if __name__ == "__main__":
    generator = Generator()
    generator.generate()
