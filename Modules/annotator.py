from typing import List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import supervision
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

import cv2

    
def is_color_bright(color):
    
    """
    Determine whether a given RGB color is considered bright based on its relative luminance.

    Args:
        color (tuple): A tuple representing the RGB color values in the range [0, 255].

    Returns:
        bool: True if the color is considered bright, False otherwise.

    Notes:
        The function calculates the relative luminance of the color using the following formula:
        L = 0.2126 * (R / 255) + 0.7152 * (G / 255) + 0.0722 * (B / 255)
        If the relative luminance is greater than 0.5, the color is considered bright.

    Examples:
        >>> is_color_bright((255, 255, 255))
        True  # White is considered bright

        >>> is_color_bright((0, 0, 0))
        False  # Black is considered dark
    """
    
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.5  # You can adjust the threshold as needed  


class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        text_color: Color = Color.white() ,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        font_path='font/CascadiaMono-Bold.otf',
        font_size=20,
        text_padding: int = 5,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.font_path = font_path
        self.font_size = font_size

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
    ) -> np.ndarray:

        image = Image.fromarray(scene)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.font_path, self.font_size)
        font2= ImageFont.truetype('font/Depot Regular 400.ttf', size=20)
        font5 = ImageFont.truetype('font/Exo-Medium.ttf', size=20)

  
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
       

            id_detected=True
            if detections.tracker_id is None:
                id_detected=False
                return np.array(image)
               
            ##################################################################
            if not id_detected:
                continue    
            ####################################################################
            
            
            class_id = ( detections.class_id[i] if detections.class_id is not None else None )
            idx = class_id if class_id is not None else i
            
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            
            # Drawing box for the around the detection
            draw.rectangle(
                [x1, y1, x2, y2],
                # outline=color.as_rgb(),
                outline=color.as_rgb(),
                width=self.thickness
            )


            if isinstance(image, Image.Image):  # Check if it's a PIL Image
                image = np.array(image)


            
            roi = image[y1:y2, x1:x2]

            # Convert ROI to HSV for better color thresholding
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define white color range in HSV
            lower_white = np.array([0, 0, 200])  # Adjust the range as needed
            upper_white = np.array([180, 55, 255])

            # Create a mask for white colors
            mask = cv2.inRange(hsv_roi, lower_white, upper_white)

            # Create a light green color overlay (RGBA)
            overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Light green in BGR
            alpha = 0.5  # Transparency factor

            # Convert mask to 3-channel for blending
            mask_3c = cv2.merge([mask, mask, mask])

            # Apply the transparent overlay to the white regions
            roi_overlay = np.where(mask_3c > 0, (alpha * overlay_color + (1 - alpha) * roi).astype(np.uint8), roi)

            # Replace the masked ROI back into the image
            image[y1:y2, x1:x2] = roi_overlay















                            
            
            if not skip_label:
                text = (
                    f"{class_id}"
                    if (labels is None or len(detections) != len(labels))
                    else labels[i]
                )
                
                # Create a temporary image to get the size of the text
                temp_image = Image.new('RGB', (1, 1))  # Create a 1x1 pixel image
                drawx = ImageDraw.Draw(temp_image)

                # Get the bounding box of the text
                bbox = drawx.textbbox((0, 0), text, font )

                # Calculate the width and height of the text
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                
                text_x = x1
                text_y = y1 #- (0.4 * text_height) # Adjusted to be inside the bounding box

                text_background_x1 = x1
                text_background_y1 = y1

                text_background_x2 = x1 + text_width
                text_background_y2 = y1 + text_height + self.text_padding
                
                # # Drawing box for the label_text
                # draw.rectangle(
                #     [text_background_x1, text_background_y1, text_background_x2, text_background_y2],
                #     fill=color.as_rgb()
                # )
                
              
                
                draw.text(
                    (text_x, text_y),
                    text,
                    fill= (0,0,0) if is_color_bright(color.as_rgb()) else (255,255,255), # Selecting better text colors for better visuals 
                    # fill=self.text_color.as_rgb(),
                    # font=font2
                    font=font5
                )
        
        
          
        return image


