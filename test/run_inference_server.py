import io
import os
import torch

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request

import logging
import numpy as np

from model import UNet
from data_loading import BasicDataset
from plot_utils import plot_img_and_mask

out_filename = "output.jpg"
net = UNet("gelu")
weights = "simple_unet.pth"
# Scale factor for the input images, 96*96 
scale = 1 
# Minimum probability value to consider a mask pixel white
mask_threshold = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Loading model weights {weights}")
logging.info(f"Using device {device}")

net.to(device=device)
net.load_state_dict(torch.load(weights, map_location=device))

logging.info("Model loaded!")

app = Flask(__name__)


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        probs = F.softmax(output, dim=1)[0]

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((full_img.size[1], full_img.size[0])),
                transforms.ToTensor(),
            ]
        )

        full_mask = tf(probs.cpu()).squeeze()

    return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f"{split[0]}_OUT{split[1]}"

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray(
            (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
        )


def get_prediction(image_bytes):

    mask = predict_img(
        net=net,
        full_img=image_bytes,
        scale_factor=scale,
        out_threshold=mask_threshold,
        device=device,
    )

    # save img
    result = mask_to_image(mask)
    print(f"the facial landmarks coordinates predicted on the image is: \n{result}")
    result.save(out_filename)
    logging.info(f"Mask saved to {out_filename}")

    # visualize
    logging.info(f"Visualizing results for image , close to continue...")
    plot_img_and_mask(image_bytes, mask)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # we will get the file from the request
        file = request.files["file"]
        # convert that to bytes
        img_bytes = file.read()
        get_prediction(image_bytes=img_bytes)


if __name__ == "__main__":
    app.run()
