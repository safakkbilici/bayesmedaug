import cv2
import imgaug.augmenters as iaa
import numpy as np

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
}

def to_tuple(param, low=None, bias=None):
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")
    if param is None:
        return param
    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")
    if bias is not None:
        return tuple(bias + x for x in param)
    return tuple(param)

class Gamma():
    """
    Gamma:
      - Applies gamma correction to image.
      - hyperparameters:
        * gamma: float
    """
    def __init__(self, gamma):
        self.gamma = gamma
        self.to_mask = False
        self.to_img = True
    
    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            if img.dtype == np.uint8:
                table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** self.gamma) * 255
                img = cv2.LUT(img, table.astype(np.uint8))
            else:
                img = np.power(img, self.gamma)
        if self.to_mask:
            raise NotImplementedError()

        return img, mask

class GaussianBlur():
    """
    GaussianBlur:
      - Applies gaussian blurring to image.
      - hyperparameters:
        * sigma_min: float, sigma_max: float, kernel_size: int
    """
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size
        self.to_mask = False
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            img = cv2.GaussianBlur(np.squeeze(img,axis=0), (self.kernel_size, self.kernel_size), sigma)
            img = img[None, :, :]
        if self.to_mask:
            raise NotImplementedError()
        return img, mask


class OpticalDistortion():
    """
    OpticalDistortion:
      - Applies optical distortion to image.
      - hyperparameters:
        * k: int, dx: int, dy: int
    """
    def __init__(self, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
        self.k = k
        self.dx = dx
        self.dy = dy
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.to_mask = True
        self.to_img = True
    
    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            height, width = img.shape[-2], img.shape[-1]
            fx = width
            fy = height

            cx = width * 0.5 + self.dx
            cy = height * 0.5 + self.dy

            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            distortion = np.array([self.k, self.k, 0, 0, 0], dtype=np.float32)
            map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)

            img = cv2.remap(img, map1, map2, interpolation=self.interpolation, borderMode=self.border_mode, borderValue=self.value)
        
        if self.to_mask:
            height, width = mask.shape[-2], mask.shape[-1]
            fx = width
            fy = height

            cx = width * 0.5 + self.dx
            cy = height * 0.5 + self.dy

            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            distortion = np.array([self.k, self.k, 0, 0, 0], dtype=np.float32)
            map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)

            mask = cv2.remap(mask, map1, map2, interpolation=self.interpolation, borderMode=self.border_mode, borderValue=self.value)
        
        return img, mask


class GaussianNoise():
    """
    GaussianNoise:
      - Elementwise addition with 2D gaussian noise to image.
      - Chooses random variance from var_limit range.
      - hyperparameters:
        * mu: float, var_limit: tuple
    """
    def __init__(self, mu = 0, var_limit=(10.0, 50.0)):
        self.mu = mu
        self.var_limit = var_limit
        self.to_mask = False
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            height, width = img.shape[-2], img.shape[-1]
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            sigma = var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            noise = random_state.normal(self.mu, sigma, (height, width))
            img = (np.squeeze(img, axis = 0) + noise)[None, :, :]

        if self.to_mask:
            raise NotImplementedError()
        
        return img, mask


class GaussianNoiseDeterministic():
    """
    GaussianNoiseDeterministic:
      - Mean zero 2D gaussian noise.
      - hyperparameters:
        * var: float
    """
    def __init__(self, var = 1):
        self.mu = 0
        self.var = var
        self.to_mask = False
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            height, width = img.shape[-2], img.shape[-1]
            sigma = self.var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            noise = random_state.normal(self.mu, sigma, (height, width))
            img = (np.squeeze(img, axis = 0) + noise)[None, :, :]

        if self.to_mask:
            raise NotImplementedError()
        
        return img, mask


class BrightnessContrast():
    """
    BrightnessContrast:
      - Change brightness and contrast of the input image.
      - hyperparameters:
        * brightness_limit (float): factor range for changing brightness.
        * contrast_limit (float): factor range for changing contrast.
        * p (float): probability of applying the transform.
    """
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5):
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max
        self.to_mask = False
        self.to_img = True

    def __call__(self, pairs, alpha=None, beta=None):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            if alpha == None:
                alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            if beta == None:
                beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
            img = np.squeeze(img, axis = 0)

            if img.dtype == np.uint8:
                dtype = np.dtype("uint8")
                max_value = MAX_VALUES_BY_DTYPE[dtype]
                lut = np.arange(0, max_value + 1).astype("float32")
                if alpha != 1:
                    lut *= alpha
                if beta != 0:
                    if self.brightness_by_max:
                        lut += beta * max_value
                    else:
                        lut += beta * np.mean(img)

                lut = np.clip(lut, 0, max_value).astype(dtype)
                img = cv2.LUT(img, lut)
                img = img[None, :, :]

            else:
                dtype = img.dtype
                img = img.astype("float32")
                if alpha != 1:
                    img *= alpha
                if beta != 0:
                    if self.brightness_by_max:
                        max_value = MAX_VALUES_BY_DTYPE[dtype]
                        img += beta * max_value
                    else:
                        img += beta * np.mean(img)
                img = img[None, :, :]

        if self.to_mask:
            raise NotImplementedError()

        return img, mask


class Rotate():
    """
    Rotate:
      - Rotates the image.
      - hyperparameters:
        * angle: int
    """
    def __init__(self, angle):
        self.angle = angle
        self.to_mask = True
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            seq = iaa.Sequential([iaa.Rotate((self.angle))])
            img = seq(images=img)
        if self.to_mask:
            seq = iaa.Sequential([iaa.Rotate((self.angle))])
            mask = seq(images=mask)
        return img, mask

class ShearY():
    """
    ShearY:
      - Moves one side of the image along the Y axis.
      - hyperparameters:
        * shear_amount: int
    """
    def __init__(self, shear_amount):
        self.shear_amount = shear_amount
        self.to_mask = True
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            seq = iaa.Sequential([
                iaa.ShearY((self.shear_amount)),
            ])

            img = seq(images=img)

        if self.to_mask:
            seq = iaa.Sequential([
                iaa.ShearY((self.shear_amount)),
            ])
        
            mask = seq(images=mask)
        return img, mask


class ShearX():
    """
    ShearX:
      - Moves one side of the image along the X axis.
      - hyperparameters:
        * shear_amount: int
    """
    def __init__(self, shear_amount):
        self.shear_amount = shear_amount
        self.to_mask = True
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            seq = iaa.Sequential([
                iaa.ShearX((self.shear_amount)),
            ])

            img = seq(images=img)

        if self.to_mask:
            seq = iaa.Sequential([
                iaa.ShearX((self.shear_amount)),
            ])
        
            mask = seq(images=mask)
        return img, mask

class ShiftY():
    """
    ShiftY:
      - Shifts the image along the Y axis.
      - hyperparameters:
        * shift_amount: int
    """
    def __init__(self, shift_amount):
        self.shift_amount = shift_amount
        self.to_mask = True
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            seq = iaa.Sequential([
                iaa.TranslateY(px=(self.shift_amount)),
            ])

            img = seq(images=img)

        if self.to_mask:
            seq = iaa.Sequential([
                iaa.TranslateY(px=(self.shift_amount)),
            ])
        
            mask = seq(images=mask)
        return img, mask

class ShiftX():
    """
    ShiftX:
      - Shifts the image along the X axis.
      - hyperparameters:
        * shift_amount: int
    """
    def __init__(self, shift_amount):
        self.shift_amount = shift_amount
        self.to_mask = True
        self.to_img = True

    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        if self.to_img:
            seq = iaa.Sequential([
                iaa.TranslateX(px=(self.shift_amount)),
            ])

            img = seq(images=img)

        if self.to_mask:
            seq = iaa.Sequential([
                iaa.TranslateX(px=(self.shift_amount)),
            ])
        
            mask = seq(images=mask)
        return img, mask
    
    
class ZoomOut():
    """
    ZoomOut:
      - (non)zooms the image.
      - hyperparameters:
        * zoom_amount: float
    """
    def __init__(self, zoom_amount):
        self.zoom_amount = zoom_amount
        self.to_mask = True
        self.to_img = True
        
    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        
        if self.to_img:
            seq = iaa.Sequential([
                iaa.Affine(scale={"x": (self.zoom_amount), "y": (self.zoom_amount)})
            ])
            img = seq(images=img)
            
        if self.to_mask:
            seq = iaa.Sequential([
                iaa.Affine(scale={"x": (self.zoom_amount), "y": (self.zoom_amount)})
            ])
            mask = seq(images=mask)
        return img, mask

class RandomCrop():
    """
    RandomCrop:
      - Crops random area of the image
      - hyperparameters:
        * crop_height: int, crop_width: int
    """
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.to_mask = True
        self.to_img = True
        
    def __call__(self, pairs):
        img = pairs[0]
        mask = pairs[1]
        
        h, w = img.shape[-2], img.shape[-1]
        h_start = random.random()
        w_start = random.random()
        
        if self.to_img:
            try:
                img = np.squeeze(img, axis = 0)
            except:
                pass
            if h < self.crop_height or w < self.crop_width:
                raise ValueError()

            y1 = int((h - self.crop_height) * h_start)
            y2 = y1 + self.crop_height
            x1 = int((w - self.crop_width) * w_start)
            x2 = x1 + self.crop_width
        
            img = img[y1:y2, x1:x2]
            img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
            img = img[None, :, :]
            
        if self.to_mask:
            try:
                mask = np.squeeze(mask, axis = 0)
            except:
                pass
            if h < self.crop_height or w < self.crop_width:
                raise ValueError()

            y1 = int((h - self.crop_height) * h_start)
            y2 = y1 + self.crop_height
            x1 = int((w - self.crop_width) * w_start)
            x2 = x1 + self.crop_width
        
            mask = mask[y1:y2, x1:x2]
            mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
            mask = mask[None, :, :]
            
        return img, mask
