
from glob import glob
import os, sys, numpy as np, cv2, torch
from PIL import Image
this_folder = os.getcwd()
sys.path.insert(0, this_folder)
from src.models.WCT import WCT
class Generator:
    def __init__(self, alpha=0.1) -> None:
        self.photos_folder = "src/data/datasets/Painter_Myself/photo_jpg"
        self.monet_folder = "src/data/datasets/Painter_Myself/monet_jpg"
        self.wct = WCT("src/data/universal_style_transfer_weights/pytorch", alpha=alpha).cuda()
        self.images = {"photos":[], "monet":[]}
        self.load_images()
    def load_images(self):
        print("starting to load images")
        import threading
        def helper1():
            photo_paths = glob(self.photos_folder+"/*.jpg")
            for photo_path in photo_paths:
                self.images["photos"].append((photo_path, Image.open(photo_path).convert('RGB')))
        def helper2():
            monet_paths = glob(self.monet_folder+"/*.jpg")
            for monet_path in monet_paths:
                self.images["monet"].append((monet_path, Image.open(monet_path).convert('RGB')))
        t1 = threading.Thread(target=helper1)
        t2 = threading.Thread(target=helper2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print("finished loading images")
    def generate(self):
        """
        it will generate images  using randomly selected style images
        """
        with torch.no_grad():
            for photo_path, content_img in self.images["photos"]:
                random_style_indexes = np.random.randint(0, len(self.images["monet"]), size=30)
                style_imgs = []
                for random_style_index in random_style_indexes:
                    style_img = Image.open(self.images["monet"][random_style_index][0]).convert('RGB')
                    style_img= self.wct.preprocess(style_img, fineSize=512)
                    style_imgs.append(style_img)
                style_imgs = torch.cat(style_imgs, 0)
                content_img= self.wct.preprocess(content_img, fineSize=512)
                result_img = self.wct.predict_with_styles(content_img.cuda(), style_imgs.cuda())
                result_img = result_img.detach().data.cpu().float()
                result_img = result_img.numpy()*255
                result_img= result_img[0].transpose(1,2,0)
                result_img[result_img < 0] = 0
                result_img[result_img > 255] = 255
                result_img = np.uint8(result_img)
                
                new_path =photo_path.replace("photo_jpg", "images")

                dirname = os.path.dirname(new_path)
                if(not os.path.exists(dirname)):
                    os.makedirs(dirname)
                cv2.imwrite(new_path, cv2.resize(result_img, (256,256)))

if __name__ == "__main__":
    generator = Generator()
    generator.generate()
