import torch.nn as nn
import torch, os, numpy as np
from .backbone.universal_style_transfer import VGG1, VGG2, VGG3, VGG4, VGG5
from .backbone.universal_style_transfer import VGG1Inv, VGG2Inv, VGG3Inv, VGG4Inv, VGG5Inv, load_weights
import torchvision.transforms as transforms

import torchvision.utils as vutils
class WCT(nn.Module):
    def __init__(self, load_folder=None, alpha=0.5):
        super().__init__()
        self.load_folder = load_folder
        self.alpha = alpha
        self.vgg1 = VGG1()                    
        self.vgg2 = VGG2()
        self.vgg3 = VGG3()
        self.vgg4 = VGG4()
        self.vgg5 = VGG5()
        self.vgg1Inv = VGG1Inv()
        self.vgg2Inv = VGG2Inv()
        self.vgg3Inv = VGG3Inv()
        self.vgg4Inv = VGG4Inv()
        self.vgg5Inv = VGG5Inv()
        if(self.load_folder is not None):
            self.load_weights()
    def load_weights_lua(self):
        from torch.utils.serialization import load_lua
        folder = "src/data/universal_style_transfer_weights/lua/models/"
        folderInv = "src/data/universal_style_transfer_weights/lua/decoders_noCudnn/"
        load_weights(load_lua(folder+'vgg_normalised_conv1_1.t7'), self.vgg1)
        load_weights(load_lua(folderInv + 'feature_invertor_conv1_1.t7'), self.vgg1Inv)
        
        load_weights(load_lua(folder+'vgg_normalised_conv2_1.t7'), self.vgg2)
        load_weights(load_lua(folderInv + 'feature_invertor_conv2_1.t7'), self.vgg2Inv)
        
        load_weights(load_lua(folder+'vgg_normalised_conv3_1.t7'), self.vgg3)
        load_weights(load_lua(folderInv + 'feature_invertor_conv3_1.t7'), self.vgg3Inv)

        load_weights(load_lua(folder+'vgg_normalised_conv4_1.t7'), self.vgg4)
        load_weights(load_lua(folderInv + 'feature_invertor_conv4_1.t7'), self.vgg4Inv)
        
        load_weights(load_lua(folder+'vgg_normalised_conv5_1.t7'), self.vgg5)
        load_weights(load_lua(folderInv + 'feature_invertor_conv5_1.t7'), self.vgg5Inv)
        print("finished loading lua weights")
    def load_weights(self):
        """
        loads pretreained weights 
        """
        self.vgg1.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg1.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg2.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg2.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg3.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg3.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg4.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg4.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg5.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg5.pth"),  map_location=torch.device('cpu'))
            )
        
        self.vgg1Inv.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg1Inv.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg2Inv.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg2Inv.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg3Inv.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg3Inv.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg4Inv.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg4Inv.pth"),  map_location=torch.device('cpu'))
            )
        self.vgg5Inv.load_state_dict(
                torch.load(os.path.join(self.load_folder, "vgg5Inv.pth"),  map_location=torch.device('cpu'))
            )
        
    def forward(self, cImg: torch.Tensor, sImg: torch.Tensor):
        """
        Args:
        x: is the content image
        xs is the style image
        """
        cX1 = self.vgg5(cImg)
        sX1 = self.vgg5(sImg)
        cX1 = cX1.squeeze(0)
        sX1 = sX1.squeeze(0)

        c_x1 = self.transform(cX1, sX1,  self.alpha)
        resImg1 = self.vgg5Inv(c_x1)

        cX2 = self.vgg4(resImg1)
        sX2 = self.vgg4(sImg)
        cX2 = cX2.squeeze(0)
        sX2 = sX2.squeeze(0)

        c_x2 = self.transform(cX2, sX2,  self.alpha)
        resImg2 = self.vgg4Inv(c_x2)

        cX3 = self.vgg3(resImg2)
        sX3 = self.vgg3(sImg)
        cX3 = cX3.squeeze(0)
        sX3 = sX3.squeeze(0)
        c_x3 = self.transform(cX3, sX3,  self.alpha)
        resImg3 = self.vgg3Inv(c_x3)
        
        cX4 = self.vgg2(resImg3)
        sX4 = self.vgg2(sImg)
        cX4 = cX4.squeeze(0)
        sX4 = sX4.squeeze(0)
        c_x4 = self.transform(cX4, sX4,  self.alpha)
        resImg4 = self.vgg2Inv(c_x4)

        cX5 = self.vgg1(resImg4)
        sX5 = self.vgg1(sImg)
        cX5 = cX5.squeeze(0)
        sX5 = sX5.squeeze(0)
        c_x5 = self.transform(cX5, sX5,  self.alpha)
        resImg5 = self.vgg1Inv(c_x5)
        return resImg5
    
    def whiteting_coloring(self, cX, sX, thresh=1e-5):
        """
        whitening transform
        it should delete the style from content image cX
        and convert its mean and variance to that of sX

        The seminal work by Gatys et al. [8, 9] show that the correlation between features, 
        i.e., Gram matrix or covariance matrix (shown to be as effective as Gram matrix in 
        [20]), extracted by a trained deep neural network has remarkable ability of capturing
        visual styles. 
        """
        cX = cX - torch.mean(cX, 1).unsqueeze(1).expand_as(cX)
        sXmean = sX.mean(1)
        sX = sX -sXmean.unsqueeze(1).expand_as(sX)
        covCX = cX@cX.t()/(cX.shape[0]-1)# + torch.eye(cX.size(0)).cuda()
        covSX = sX@sX.t()/(sX.shape[0]-1)
        uCx, eCx, vCx = torch.svd(covCX, some=False)
        uSx, eSx, vSx = torch.svd(covSX, some=False)
        kCx = torch.where(eCx < thresh)[0][:1].sum()
        kSx = torch.where(eSx < thresh)[0][:1].sum()
        # removing unsignificant data
        if(kCx > 0):
            vCx = vCx[:, :kCx]
            eCx = eCx[:kCx]
        if(kSx > 0):
            vSx = vSx[:, :kSx]
            eSx = eSx[:kSx]
        whitenedCx = vCx @ torch.diag(eCx.pow(-0.5)) @ vCx.t() @ cX
        coloredCx  = vSx @ torch.diag(eSx.pow(0.5)) @ vSx.t() @ whitenedCx
        return coloredCx +\
            sXmean.unsqueeze(1).expand_as(coloredCx)
    def transform(self, cX, sX, alpha):
        """
        some comment here
        """
        n_batch = cX.shape[0]
        cX_ = cX.view(n_batch, -1)
        sX_ = sX.view(n_batch, -1)

        c_sX = self.whiteting_coloring(cX_, sX_)
        c_sX = c_sX.view_as(cX)
        result = alpha* c_sX + (1-alpha) * cX

        result = result.float().unsqueeze(0)
        return result

    def preprocess(self, contentImg, styleImg, fineSize= 512):
        contentImg = contentImg.resize((fineSize,fineSize))
        styleImg = styleImg.resize((fineSize,fineSize))
        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.unsqueeze(0), styleImg.unsqueeze(0)

if __name__ == '__main__':
    import time, cv2     
    from PIL import Image
    import matplotlib.pyplot as plt
    wct = WCT("src/data/universal_style_transfer_weights/pytorch").cuda()
    with torch.no_grad():
        start = time.time()
        content_img = Image.open('src/data/imgs/grumpy.jpg').convert('RGB')
        style_img = Image.open('src/data/imgs/picasso.jpg').convert('RGB')
        content_img, style_img= wct.preprocess(content_img, style_img)
        result_img = wct(content_img.cuda(), style_img.cuda())
        result_img = result_img.detach().data.cpu().float()
        end = time.time()
        print(end- start)

    result_img = result_img.numpy()*255#, 0, -1)
    result_img= result_img[0].transpose(1,2,0)
    result_img[result_img < 0] = 0
    result_img[result_img > 255] = 255
    result_img = np.uint8(result_img)[:,:, ::-1]
    # cv2.imshow("img",  result_img)
    # cv2.waitKey(0)
    cv2.imwrite("result.jpg")