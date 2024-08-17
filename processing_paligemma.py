import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Iterable
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompts(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Quoting from the huggingface blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
        The input text is tokenized normally.
        A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
        This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
        The tokenized text is also prefixed with a fixed number of <image> tokens.
        NOTE: In the paper it should be tokenized separately but huggingface process it directly (put the github link)
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(image: Image.Image, size: Tuple[int, int], resample: Image.Resampling = None, reducing_gap: Optional[int] = None) -> Image.Image:
    height, width = size
    return image.resize((width, height), resample=resample, reducing_gap=reducing_gap)

def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image: np.ndarray, mean: Union[float, List[float]], std: Union[float, List[float]]) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    return (image - mean) / std

def process_images(
        images: List[Image.Image],
        size: Tuple[int, int] = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    images = [np.array(image) for image in images]
    # Rescale pixel values to range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the image
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Reorder the dimensions of the image to [channels, height, width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcesssor:
    """
    Processor for the PaliGemma model. This processor is used to preprocess the image (resize, normalize, etc.) 
    and the text by creating the tokens for it and the placeholders for the image tokens.
    """

    #Placeholder for the image token
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_len = num_image_tokens # Number of token to generate for the image
        self.image_size = image_size

        # Adding special tokens to the tokenizer because the tokenizer of the Gemma model wasn't created to handle specific tokens related to the image.
        # Here: https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma/README.md
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_token(tokens_to_add)
        # For object detection
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        # For segmentation
        EXTRA_TOKENS += [
            f"<seg{i:04d}" for i in range(128)
        ]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # Adding the BOS and EOS tokens if needed
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        """ Preparing the inputs for PaliGemma """
        # For now, we will only consider one image and one text
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."
        
        # Preprocess images
        pixel_values = process_images(images, size=(self.image_size, self.image_size), 
                                      resample=Image.Resampling.BICUBIC, rescal_factor=1 / 255.0, 
                                      image_mean=IMAGENET_STANDARD_MEAN, image_std=IMAGENET_STANDARD_STD)
        
        # Convert the pixel_values shape to [batch_size, channels, height, width]
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        # Prepare the complete tokens (image + text) by including the image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompts(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_len,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        # Return the input_ids (list of int that represent the token position) and attention_mask as tensor
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data