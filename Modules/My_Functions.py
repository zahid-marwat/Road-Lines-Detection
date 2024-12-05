import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
import csv

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont



import cv2
import numpy as np

def blur_rectangular_area(frame, x1, y1, x2, y2, blur_amount=25):
    """
    Blur a rectangular area in a frame.

    Parameters:
    - frame (numpy.ndarray): The input frame in NumPy array format.
    - x1 (int): The x-coordinate of the first corner of the rectangular area.
    - y1 (int): The y-coordinate of the first corner of the rectangular area.
    - x2 (int): The x-coordinate of the opposite corner of the rectangular area.
    - y2 (int): The y-coordinate of the opposite corner of the rectangular area.
    - blur_amount (int): The amount of blur to be applied. Default is 25.

    Returns:
    - numpy.ndarray: The frame with the specified rectangular area blurred.
    """
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    
    # Ensure x1 is less than x2 and y1 is less than y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Extract the specified rectangular area
    region_of_interest = frame[y1:y2, x1:x2]

    # Apply blur to the region_of_interest
    blurred_roi = cv2.GaussianBlur(region_of_interest, (blur_amount, blur_amount), 0)

    # Replace the original region with the blurred one
    frame_copy = frame.copy()
    frame_copy[y1:y2, x1:x2] = blurred_roi

    return frame_copy


def select_polygon_points(video_path,laneNumber,total_lanes,twowayy):
    """
    Selects 4 polygon points from a specified frame of a video.

    Parameters:
        video_path (str): The path to the video file.

    Returns:
        list: A list containing 4 user-selected points.
    """
    HalfLanes=total_lanes//2
    if laneNumber in range(0,HalfLanes):
        text1='Incoming'
        text2=f'Lane {laneNumber+1}'
            
    elif laneNumber in range(HalfLanes,total_lanes):
        text1='OutGoing'
        text2=f'Lane {laneNumber-HalfLanes+1}'
    else:
        print("Error in lanes selection")
            
    if twowayy:
        instructionTXT=f'Select 4 Points for {text1} {text2} !'
    
    else:
        instructionTXT=f'Select 4 Points for Lane {laneNumber+1} !'
    
    points = get_user_points(video_path,  50,instructionTXT)
    return points


def get_user_points(video_path, frame_number, instruction_text):
    """
    Handles user interaction to select points from a specific frame of a video.

    Parameters:
        video_path (str): The path to the video file.
        instruction_text (str): The instruction text displayed to the user.

    Returns:
        list: A list containing user-selected points.
    """

    if video_path=='live':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return []

    # Set the video to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Unable to read frame {frame_number}.")
        cap.release()
        return []
    
    # Draw instruction text on the 
    text_rectangle=get_centered_rectangle(frame,650,150)
    frame = blur_rectangular_area(frame, 
                                  *(text_rectangle[0],text_rectangle[1]), 
                                  *(text_rectangle[2],text_rectangle[3]))
    
    
    # Draw a white rounded rectangle
    overlay = frame.copy()
    cv2.rectangle(frame, (text_rectangle[0],text_rectangle[1]),
                        (text_rectangle[2],text_rectangle[3]),
                        (0, 0, 0), -1)
    alpha = 0.5  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (text_rectangle[0],text_rectangle[1]),
                        (text_rectangle[2],text_rectangle[3]),
                        (255, 255, 255), 1)


    
    # Display the frame for user interaction
    window_name = "Select Points"
    cv2.imshow(window_name, frame)
    
    
    # Calculate text size and position
    (text_width, text_height), baseline = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = text_rectangle[0] + (text_rectangle[2] - text_rectangle[0] - text_width) // 2
    text_y = text_rectangle[1] + (text_rectangle[3] - text_rectangle[1] + text_height) // 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, instruction_text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)
    
    # Initialize list to store user-selected points
    points = []

    # Function to handle mouse clicks
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, frame)

    # Set the mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)

    # Wait for the user to click points
    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4 and frame_number == 50:
            break  # Exit after selecting 4 points in phase 1
        elif len(points) == 2 and frame_number == 51:
            break  # Exit after selecting 2 points in phase 2
        elif key == 27:  # Press 'Esc' to exit at any time
            break

    # Release video capture
    cap.release()
    cv2.destroyAllWindows()

    return points

def get_centered_rectangle(frame, rect_width, rect_height):
    """
    Calculate the coordinates of a rectangle centered within a given frame.

    Parameters:
    frame (np.array): A numpy array representing the image frame with shape (height, width, channels).
    rect_width (float): The width of the rectangle to be centered.
    rect_height (float): The height of the rectangle to be centered.

    Returns:
    (list): A list containing two tuples representing the coordinates of the 
            upper-left and lower-right corners of the centered rectangle.
    """
    frame_height, frame_width, _ = frame.shape
    
    # Calculate the center of the frame
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    # Calculate the upper-left corner of the centered rectangle
    rect_x1 = int(frame_center_x - rect_width / 2)
    rect_y1 = int(frame_center_y - rect_height / 2)
    
    # Calculate the lower-right corner of the centered rectangle
    rect_x2 = int(frame_center_x + rect_width /2)
    rect_y2 = int(frame_center_y + rect_height / 2)
    
    return [rect_x1, rect_y1, rect_x2, rect_y2]



def drawing_on_frame_after_detection(frame,
                                     names,
                                     VehLeaving,
                                     VehIncoming,
                                     counter1,
                                     twowaytraffic,
                                     each_lane_counters
                                     ) -> np.ndarray:
    """
    Draw overlays and information on a frame after detection.

    Parameters:
    - frame (numpy.ndarray): The input frame to be processed.
    - cracks_areas (dict): Dictionary with keys as crack types and values as their areas.
    - cracks_counter (dict): Dictionary with keys as crack types and values as their counts.
    - name2var (dict): Dictionary with keys representing variable names.

    Returns:
    - numpy.ndarray: The frame with overlays and information.
    """

    
    def get_limited_counts( namesList,countersList):
        typ_of_veh_detected=len(countersList)
        names_list=list(namesList.values())
        vehicles_detected={}
        
        for key,val in countersList.items():
            vehicles_detected[names_list[key]] = val
        
        return vehicles_detected , typ_of_veh_detected
    
    
    
    
    
    allveh , typesDetected=get_limited_counts(names,counter1)
    # Define polygon points for the Traffic In/Out box on top (Left side box)   
    overlay  =  frame.copy()
    im0 = frame
    alpha = 0.4
    thickness = 2
    isClosed = True
    
    if twowaytraffic:
        
        
        frame = blur_rectangular_area(frame, *(675,40), *(900, 70))         
        points_x = np.array([[655,40],[900,40],[900,70],[675,70]]).reshape((-1, 1, 2))    
        cv2.polylines(frame, [points_x], isClosed, (0,0,0), 1)
        cv2.fillPoly(overlay, [points_x], (0, 0, 0, 60))
        
        im0 = cv2.addWeighted(overlay, alpha, frame, 1 - alpha ,0)
        frame = im0
        overlay  =  frame.copy()
        
        frame = blur_rectangular_area(frame, *(1010,  40), *(1247, 70)) 
        points_z = np.array([[1267,40],[1010,40],[1010,70],[1247,70]]).reshape((-1, 1, 2))
        cv2.polylines(frame, [points_z], isClosed, (0,0,0), 1)           
        cv2.fillPoly(overlay, [points_z], (0, 0, 0)) 
                     
    else:
        
        
        # Define polygon points for the Traffic In/Out box on top
        frame = blur_rectangular_area(frame, *(675,40), *(1247,70)) 
        points_x = np.array([[655,40],[1267,40],[1247,70],[675,70]]).reshape((-1, 1, 2))    
        cv2.polylines(frame, [points_x], isClosed, (0,0,0), 1)
        cv2.fillPoly(overlay, [points_x], (0, 0, 0, 60))
        
    
  
    #################################################################################################
    
    #################################################################################################
    
    
    
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha ,0)
    
    # Adding the boundry lines across the frame   
    color = (0, 0, 255)
    # color = (219, 255, 51)
    # color = (255,255,255)
    # color = (0,0,0)
    
    isClosed = False
    thickness = 1
    
    points = np.array([[10,10],[900,10],[925,30],[995,30],[1020,10],[1910,10],[1910,1070],[1020,1070],[995,1050],[925,1050],[900,1070],[10,1070],[10,10]]).reshape((-1, 1, 2))
    points_1 = np.array([[30,60],[60,30],[800,30]]).reshape((-1, 1, 2))
    points_2 = np.array( [[1120,30],[1860,30],[1890,60]]).reshape((-1, 1, 2))
    points_3 = np.array([[500,50],[650,50],[670,80],[900,80]]).reshape((-1, 1, 2))
    points_4 = np.array([[1420,50],[1270,50],[1250,80],[1020,80]]).reshape((-1, 1, 2))
    lower_linepoints= np.array([[500,50],[650,50],[670,80],[1250,80],[1270,50],[1420,50]]).reshape((-1, 1, 2))
    
    
    cv2.polylines(frame, [points], isClosed, color, thickness)
    cv2.polylines(frame, [points_1], isClosed, color, thickness)
    cv2.polylines(frame, [points_2], isClosed, color, thickness)
    cv2.polylines(frame, [points_3], isClosed, color, thickness)
    cv2.polylines(frame, [points_4], isClosed, color, thickness)
    
    if twowaytraffic:
        
        cv2.polylines(frame, [points_3], isClosed, color, thickness)
        cv2.polylines(frame, [points_4], isClosed, color, thickness)
        
    else:        
        cv2.polylines(frame, [lower_linepoints], isClosed, color, thickness)
        



    font2 = ImageFont.truetype('font/Roboto-Bold.ttf', size=25)
    font3 = ImageFont.truetype('font/Roboto-Light.ttf', size=18)
    font4 = ImageFont.truetype('font/Depot Regular 400.ttf', size=18)
    font5 = ImageFont.truetype('font/Exo-Medium.ttf', size=16)
    
    
    frame = blur_rectangular_area(frame, *(1580, 980), *(1900, 1060)) # to blur the background of the LOGO
    frame = blur_rectangular_area(frame, *(25, 65), *(205,65+((typesDetected+1)*32))) # to blur the background of the info table            
    
    # to blur the background of the info boxes on right side
    frame = blur_rectangular_area(frame, *(1730,  70), *(1890, 160))             
    frame = blur_rectangular_area(frame, *(1730, 170), *(1890, 260))           
    frame = blur_rectangular_area(frame, *(1730, 270), *(1890, 360))          
    frame = blur_rectangular_area(frame, *(1730, 370), *(1890, 460))           

    
    
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame,"RGBA")
    (x, y) = (500, 500)
    color = 'rgb(0, 0, 0)' # black color

    overlay  =  frame.copy()
    alpha = 0.5
    
    logo = cv2.imread("FrameByFrame/logo.png",cv2.IMREAD_UNCHANGED)
    
    #Drawing right side boxes
    draw.rounded_rectangle((1730, 70 , 1890, 160), fill=(0, 0, 0, 60),width=1, radius=15,corners = (False, True, False, True),outline=(255, 255, 255)) 
    draw.rounded_rectangle((1730, 170, 1890, 260), fill=(0, 0, 0, 60),width=1, radius=15,corners = (False, True, False, True),outline=(255, 255, 255)) 
    draw.rounded_rectangle((1730, 270, 1890, 360), fill=(0, 0, 0, 60),width=1, radius=15,corners = (False, True, False, True),outline=(255, 255, 255)) 
    draw.rounded_rectangle((1730, 370, 1890, 460), fill=(0, 0, 0, 60),width=1, radius=15,corners = (False, True, False, True),outline=(255, 255, 255)) 

    #Background box for LOGO
    draw.rounded_rectangle((1580, 980, 1900, 1060),fill=(255, 255, 255, 60),width=1, radius=1,corners = (True, True, True, True),outline=(255, 255, 255)) 
    
    ht, wd = im0.shape[:2]
    ht2, wd2 = logo.shape[:2]

    # extract alpha channel as mask and base bgr images
    bgr = logo[:,:,0:3]
    mask = logo[:,:,3]

    # insert bgr into img at desired location and insert mask into black image
    x = 1590
    y = 1000

    bgr_new = im0.copy()
    bgr_new[y:y+ht2, x:x+wd2] = bgr

    mask_new = np.zeros((ht,wd), dtype=np.uint8)
    mask_new[y:y+ht2, x:x+wd2] = mask
    mask_new = cv2.cvtColor(mask_new, cv2.COLOR_GRAY2BGR)

    ret, mask_new = cv2.threshold(mask_new, 0, 255, cv2.THRESH_BINARY)
    frame = np.where(mask_new==255, bgr_new, frame)

    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame,"RGBA")
    overlay  =  frame.copy()
       
    
    
    def get_coords_for_centered_text(rect_left, 
                                    rect_top, 
                                    rect_right,  
                                    rect_bottom,
                                    the_text,
                                    fonttype
                                    ):
        
        # Get the size of the text
        text_bbox = draw.textbbox((0, 0), the_text, font=fonttype)
        
        # Calculate the width and height of the text
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate the position to draw the text to center it within the rectangle
        text_x = rect_left + (rect_right - rect_left - text_width) // 2
        text_y = rect_top + (rect_bottom - rect_top - text_height) // 2
        return text_x,text_y

    # Drawing text into the right side boxes
    draw.text((1790,80), "DATE", fill=(255,255,255), font=font3)
    date_text_x, date_text_y = get_coords_for_centered_text(1730, 70, 1890, 160,"23-05-2024",font2)    
    draw.text((date_text_x, date_text_y), format("23-05-2024"), fill=(255,255,255), font=font2)

    draw.text((1790,180), "TIME", fill=(255,255,255), font=font3)
    Time_text_x, Time_text_y = get_coords_for_centered_text(1730, 170, 1890, 260,"08:30:41",font2)  
    draw.text((Time_text_x,Time_text_y),  format("08:30:41"), fill=(255,255,255), font=font2)
    
    draw.text((1788,280), "ROAD", fill=(255,255,255), font=font3)
    Road_text_x, Road_text_y = get_coords_for_centered_text(1730, 270, 1890, 360,"Pak M-1" ,font2)    
    draw.text((Road_text_x, Road_text_y), format("M-1"), fill=(255,255,255), font=font2)
    
    
    draw.text((1773,380), "WEATHER", fill=(255,255,255), font=font3)
    Weather_text_x, Weather_text_y = get_coords_for_centered_text(1730, 370, 1890, 460,"Cloudy",font2)  
    draw.text((Weather_text_x,Weather_text_y),  format("Cloudy"), fill=(255,255,255), font=font2)

    
    
    
    
    # Drawing text into the upper two 
    if twowaytraffic:
    
        draw.text((680,45), "TRAFFIC LEAVING: ", fill=(255,255,255), font=font5)
        draw.text((820,45), format(VehLeaving), fill=(255,255,255), font=font5)
        
        draw.text((1020,45), "TRAFFIC INCOMING: ", fill=(255,255,255), font=font5)
        draw.text((1175,45), format(VehIncoming), fill=(255,255,255), font=font5)
        
    else:
            
        
        draw.text((775,45), "T O T A L   V E H I C L E S   D E T E C T E D : ", fill=(255,255,255), font=font5)
        draw.text((1110,45), format(VehLeaving+VehIncoming), fill=(255,255,255), font=font5)
    
    
    

    frame = np.array(frame)
    
    font4 = ImageFont.truetype('font/Depot Regular 400.ttf', size=20)
    font5 = ImageFont.truetype('font/Exo-Medium.ttf', size=17)
    font6 = ImageFont.truetype('font/Exo-Medium.ttf', size=14) 
            
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame,"RGBA")
    
    overlay  =  frame.copy()
    alpha = 0.5
    
    # Draw Vehicles type information
    draw.rectangle((25, 96,135,96+(32*typesDetected)), fill=(0, 0, 0, 80),width=1,outline=(256,256,256))   # 'Vehicles Type' text background
    draw.rectangle((135, 96,205,96+(32*typesDetected)), fill=(0, 0, 0, 80),width=1,outline=(256,256,256)) # 'Vehicles Count' text background
    draw.rectangle((25, 65,205,90), fill=(0, 0, 0, 130),width=1,outline=(256,256,256))    # 'Header' text background
    
    draw.text((45,67), "TYPE", fill=(255,255,255), font=font4)
    draw.text((135,67), "COUNT", fill=(255,255,255), font=font4)

    
    
    
    # Draw vehicles' names
    x_coord1 = 40
    [draw.text((x_coord1, 105 + i * 30), format(variable), fill=(255, 255, 255), font=font5) for i, variable in enumerate(allveh.keys())]

    # Draw vehicles' counts
    x_coord2 = 140
    [draw.text((x_coord2, 105 + i * 30), format(variable), fill=(255, 255, 255), font=font5) for i, variable in enumerate(allveh.values())]
    
    
    frame = np.array(frame)
    return frame




def get_areas_and_counts(crack_areas_dict,
                         crack_counts_dict, 
                         total_counts_and_areas, 
                         annotation_variables_dict, 
                         crack_names_list):
    """
    Update count and area values for specific cracks in the main dictionaries.

    Parameters:
    - crack_counts_dict (dict): Dictionary for counting crack occurrences.
    - crack_areas_dict (dict): Dictionary for measuring crack areas.
    - total_counts_and_areas (tuple of two dicts): Total counts and total areas for all cracks.
    - annotation_variables_dict (dict): Dictionary with keys as annotation labels and values as variable names.
    - crack_names_list (list): List of crack names.

    Returns:
    - tuple: Updated crack areas dictionary and crack counts dictionary.
    """

    # Iterate through the keys of the total counts dictionary
    for crack_key in total_counts_and_areas[0].keys():
        # Get the annotation label for the current crack
        annotation_label = crack_names_list[crack_key]

        # Get the variable name associated with the annotation label
        crack_variable = annotation_variables_dict[annotation_label]

        # Check if the crack variable is not 'CARRIAGEWAY'
        if crack_variable != 'CARRIAGEWAY':
            # Update count and area values in the main dictionaries
            crack_counts_dict[crack_variable + '_count'] = total_counts_and_areas[0][crack_key]
            crack_areas_dict[crack_variable + '_area'] = total_counts_and_areas[1][crack_key]

    # Return the updated crack areas and counts dictionaries
    return crack_areas_dict, crack_counts_dict








def compare_frames2video(input_folder_before, input_folder_results, output_video_path, fps=30):
    """
    Combine two sets of image frames (before and results) into a single video.

    Args:
        input_folder_before (str): Path to the folder containing "before" frames.
        input_folder_results (str): Path to the folder containing "results" frames.
        output_video_path (str): Path to the output video file.
        fps (int, optional): Frames per second for the output video. Default is 30.

    Returns:
        None

    Example:
        input_folder_path_before = 'frameBYframe/before/'
        input_folder_path_results = 'frameBYframe/after/'
        output_video_path = 'output_video2.mp4'
        frames_to_video(input_folder_path_before, input_folder_path_results, output_video_path)
    """
    
    # Get the list of image files in each input folder
    image_files_before = [f for f in os.listdir(input_folder_before) if f.endswith('.jpg')]
    image_files_results = [f for f in os.listdir(input_folder_results) if f.endswith('.jpg')]

    # Sort the files in ascending order
    image_files_before.sort()
    image_files_results.sort()

    # Determine the width and height from the first image
    img_before = cv2.imread(os.path.join(input_folder_before, image_files_before[0]))
    img_results = cv2.imread(os.path.join(input_folder_results, image_files_results[0]))
    height, width, _ = img_before.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec based on your preference
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (2 * width, height))

    # Iterate through both sets of image files and write each frame to the video
    for img_file_before, img_file_results in zip(image_files_before, image_files_results):
        img_path_before = os.path.join(input_folder_before, img_file_before)
        img_path_results = os.path.join(input_folder_results, img_file_results)

        img_before = cv2.imread(img_path_before)
        img_results = cv2.imread(img_path_results)

        # Create a blank frame
        img_combined = 255 * np.ones((height, 2 * width, 3), dtype=np.uint8)

        # Place "before" frames on the left and "results" frames on the right
        img_combined[:, :width] = img_before
        img_combined[:, width:] = img_results

        video_writer.write(img_combined)

    # Release the video writer
    video_writer.release()
    
    
    
    
def frames2video(input_folder, output_video_path, fps=30):
    """
    Convert a sequence of images in a folder to a video.

    Args:
        input_folder (str): Path to the folder containing image frames.
        output_video_path (str): Path to the output video file.
        fps (int, optional): Frames per second for the output video. Default is 30.

    Returns:
        None

    Example:
        input_folder_path = 'path/to/your/image/folder'
        output_video_path = 'output_video.mp4'
        images_to_video(input_folder_path, output_video_path)
    """
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    image_files.sort()  # Sort the files in ascending order

    # Determine the width and height from the first image
    img = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = img.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec based on your preference
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate through the image files and write each frame to the video
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        video_writer.write(img)

    # Release the video writer
    video_writer.release()    
    
    
    
    
    
    
    
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


def get_color_set(no_of_lanes):
    color_set=[]
    selected_colors=[(80, 127, 255),
                    (128, 128, 240),
                    (0, 69, 255),
                    (32, 165, 218),
                    (122, 150, 233),
                    (128, 128, 240),
                    (170, 178, 32),
                    (128, 128, 0),
                    (160, 158, 95),
                    (255, 144, 30),
                    (225, 105, 65),
                    (160, 158, 95)]
    
    for lane_no in range(no_of_lanes):
        if lane_no  >=  no_of_lanes//2:
            color_set.append(selected_colors[(-1)*no_of_lanes+lane_no])
        else:
            color_set.append(selected_colors[lane_no])
    return color_set
 
 
def draw_polygone(scene: np.ndarray,
                  points_array: list,
                  Lanes_veh_count: list[int],
                  frame_laneCounts: list[int],
                  lane_no: int,
                  thickness: int) -> np.ndarray:
    """
    Draws a polygon with transparent overlay on the given frame using the provided points.

    Args:
        scene (np.ndarray): The image on which the polygon will be drawn.
        points_array (list): A list of points defining the polygon.
        Lanes_veh_count (list[int]): List of vehicle counts per lane.
        frame_laneCounts (list[int]): List of frame lane counts.
        lane_no (int): The lane number to draw.
        thickness (int): Thickness of the line.

    Returns:
        np.ndarray: The image with the polygon drawn on it.
    """
    # Convert the scene to an RGBA image
    image = Image.fromarray(scene).convert('RGBA')
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    all_colors = generate_gradient_colors_heat_map(frame_laneCounts)
    color = all_colors[lane_no]
    fill_color = (int(color[0]), int(color[1]), int(color[2]), 90)  # Semi-transparent fill color

    # Fill the polygon with the transparent color
    draw.polygon(points_array, fill=fill_color)

    # Draw the edges of the polygon
    for pnt in range(len(points_array)):
        x1, y1 = points_array[pnt]
        if pnt + 1 == len(points_array):
            x2, y2 = points_array[0]
        else:
            x2, y2 = points_array[pnt + 1]
        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)
    
    # Composite the overlay with the original image
    image = Image.alpha_composite(image, overlay).convert('RGB')

    # Find the lower two points and midpoint for text placement
    points = sorted(points_array, key=lambda x: x[1], reverse=True)[:2]
    point1, point2 = sorted(points, key=lambda x: x[0])
    midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2) 

    font5 = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(str(Lanes_veh_count[lane_no]), font5, 1, thickness=2)
    text_position = (int(midpoint[0] - text_size[0] / 2), int(midpoint[1] + text_size[1] / 2))

    image = np.array(image)
    cv2.putText(image, str(Lanes_veh_count[lane_no]), text_position, font5, 1, color, 2)

    return image


def draw_circle_on_frame(frame, center, radius=5, color=(0, 255, 0), thickness=-1):
    # Draw a filled circle at the specified center point
    frame_with_circle = frame.copy()
    cv2.circle(frame_with_circle, center, radius, color, thickness)

    return frame_with_circle



def is_point_inside_polygon(point, polygon_points):
    """
    Check if a point is inside a polygon defined by its vertices.

    Args:
        point (tuple): Coordinates of the point (x, y).
        polygon_points (list): List of polygon vertices [(x1, y1), (x2, y2), ...].

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    polygon = Polygon(polygon_points)
    point = Point(point)
    
    return polygon.contains(point)


def is_point_under_line(centerpoint, line_points):
    x, y = centerpoint
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]

    # Calculate the slope and y-intercept of the line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Calculate the expected y-coordinate on the line for the given x-coordinate
    expected_y = m * x + b

    # Check if the actual y-coordinate is below the expected y-coordinate
    return y > expected_y


def filter_unique_boxes(boxes_list):
    """
    Filter a list of dictionaries containing boxes based on unique 'box_id' values.
    If there are duplicates with the same 'box_id' and 'box_class', keep the one with higher 'box_conf'.

    Args:
        boxes_list (list): A list of dictionaries, each representing a box with 'box_id', 'box_area', 'box_class', and 'box_conf'.

    Returns:
        list: A filtered list containing dictionaries with unique 'box_id' values.
    """
    # Create a dictionary to store unique "box_id" values along with the corresponding box with the highest 'box_conf'
    unique_boxes = {}

    # Iterate through the list and filter out duplicates
    for box in boxes_list:
        box_id = box["box_id"]
        box_class = box["box_class"]
        box_conf = box["box_conf"]

        # Check if box_id is already in the unique_boxes dictionary
        if box_id not in unique_boxes:
            unique_boxes[box_id] = {"box_class": box_class, "box_conf": box_conf, "box": box}
        else:
            # Check if the current box has a higher 'box_conf' than the one in the dictionary
            if box_conf > unique_boxes[box_id]["box_conf"]:
                unique_boxes[box_id] = {"box_class": box_class, "box_conf": box_conf, "box": box}

    # Extract the final filtered boxes from the dictionary
    filtered_boxes_list = [entry["box"] for entry in unique_boxes.values()]

    return filtered_boxes_list

def filter_unique_boxes_dict(boxes_dict):
    """
    Filter a dictionary of lists containing boxes based on unique 'box_id' values.
    If there are duplicates with the same 'box_id' and 'box_class', keep the one with higher 'box_conf'.

    Args:
        boxes_dict (dict): A dictionary where keys are identifiers and values are lists of dictionaries,
                           each representing a box with 'box_id', 'box_class', and 'box_conf'.

    Returns:
        dict: A filtered dictionary containing lists of dictionaries with unique 'box_id' values.
    """
    # Create a dictionary to store unique "box_id" values along with the corresponding box with the highest 'box_conf'
    filtered_new_dict = {}

    # Iterate through the dictionary
    for key, lane_data in boxes_dict.items():
        if lane_data:
            unique_boxes = {}
            # Iterate through the list and filter out duplicates
            for det_box in lane_data:
                box_id = det_box["box_id"]
                box_class = det_box["box_class"]
                box_conf = det_box["box_conf"]

                # Check if box_id is already in the unique_boxes dictionary
                if box_id not in unique_boxes:
                    unique_boxes[box_id] = {"box_class": box_class, "box_conf": box_conf, "box": det_box}
                else:
                    # Check if the current box has a higher 'box_conf' than the one in the dictionary
                    if box_conf > unique_boxes[box_id]["box_conf"]:
                        unique_boxes[box_id] = {"box_class": box_class, "box_conf": box_conf, "box": det_box}

            # Extract the final filtered boxes from the dictionary
            filtered_boxes_list = [entry["box"] for entry in unique_boxes.values()]
            filtered_new_dict[key] = filtered_boxes_list
        else:
            filtered_new_dict[key] = []  # If lane_data is empty, keep it as is

    return filtered_new_dict


def append_dict_values(main_lane_dict, frame_lane_dict):
    updated_dict={}

    if len(main_lane_dict)==0:
        print("Main dictionary for each lane is empty: possible errors(1. First Frame 2. Some other error!)")
        updated_dict=frame_lane_dict
    else:    
        for key,value in frame_lane_dict.items():

            appendedList=main_lane_dict[key] + value
            updated_dict[key]=appendedList

    return updated_dict

def filter_objects_movement(dictionary):
    filtered_dict = {}
    prev_boxes = None

    for key, boxes in sorted(dictionary.items()):
        # If the list of boxes is empty, keep it as is
        if not boxes:
            filtered_dict[key] = boxes
            continue

        if prev_boxes is not None:
            # Find box_ids that appear in both consecutive elements
            moved_box_ids = set(prev_box["box_id"] for prev_box in prev_boxes).intersection(
                curr_box["box_id"] for curr_box in boxes
            )
            # Filter out moved boxes from previous element
            filtered_prev_boxes = [prev_box for prev_box in prev_boxes if prev_box["box_id"] not in moved_box_ids]
            if filtered_prev_boxes:
                filtered_dict[key - 1] = filtered_prev_boxes  # Update previous element
            # Add moved boxes to current element
            filtered_curr_boxes = [curr_box for curr_box in boxes if curr_box["box_id"] in moved_box_ids]
            if filtered_curr_boxes:
                filtered_dict[key] = filtered_curr_boxes
        else:
            filtered_dict[key] = boxes  # For the first element, keep as is
        prev_boxes = boxes

    if prev_boxes:  # If there are boxes in the last element
        filtered_dict[max(dictionary.keys())] = prev_boxes

    return filtered_dict

def count_similar_class_ids(box_list):
    """
    Count the occurrences of each class ID in a list of dictionaries.

    Args:
        box_list (list): A list of dictionaries, each representing a box with 'box_id', 'box_class', and 'box_conf'.

    Returns:
        dict: A dictionary where keys are class IDs and values are the counts.
    """
    class_counts = {}

    for box in box_list:
        box_class = box["box_class"]

        # Increment the count for the current class ID
        class_counts[box_class] = class_counts.get(box_class, 0) + 1

    return class_counts

def count_similar_class_ids_Dict(my_dictionary):
    """
    Count the occurrences of each class ID in a dictionary of lists of dictionaries.

    Args:
        boxes_dict (dict): A dictionary where keys are identifiers and values are lists of dictionaries,
                           each representing a box with 'box_id', 'box_class', and 'box_conf'.

    Returns:
        dict: A dictionary where keys are the identifiers from the input dictionary and values are dictionaries
              containing class ID counts for each identifier.
    """
    class_counts_dict = {}

    for key, box_list in my_dictionary.items():
        class_counts_dict[key] = count_similar_class_ids(box_list)

    return class_counts_dict


def save_extracted_data(
    video_file_path,
    objects_names,
    objects_counts,
    each_lane_counts,
    lane_total_veh,
    total_sum,
    
):
    """Save extracted data into csv file."""
    
    
    
    video_name = video_file_path.split(os.sep)[-1].split('.')[0]
    file_dir_name = os.path.dirname(video_file_path) + os.sep 
    lanes_numbering = ['Lane-' + str(i+1) for i in range(len(each_lane_counts))]

    # adjusting each_laneCounts
    for lane_no,objects in each_lane_counts.items():
        for obj_class,count in objects_counts.items():
            if not objects.get(obj_class):
                each_lane_counts[lane_no][obj_class]=0
        
    
    with open(file_dir_name + os.sep + video_name+ "_data.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header row with keys
        csv_writer.writerow(['Name']+lanes_numbering+['Total'])
        
        # Write the data rows with key-value pairs
        for key,value in objects_counts.items():
            values_to_add= [objects_names[key]]+ [ each_lane_counts[i][key] for i in range(len(each_lane_counts))] + [value]
                            
            
            csv_writer.writerow(values_to_add)
        csv_writer.writerow(['Total']+lane_total_veh+[total_sum])


def update_trailing_objects(detected_objects, new_detections):
    # Determine the number of frames
    num_frames = len(detected_objects)
    
    # Remove the oldest frame's detections
    del detected_objects[f'n{num_frames}']
    
    # Shift all previous frames' detections
    for i in range(num_frames - 1, 0, -1):
        detected_objects[f'n{i+1}'] = detected_objects[f'n{i}']
    
    # Add new detections to the first position
    detected_objects['n1'] = new_detections       


def draw_trail(trail_points_dict,frame,radius_val):
    # Generate colors for 15 frames
    colors = generate_gradient_colors(len(trail_points_dict))

    # Draw points on the frame
    alpha = 0.4  # Transparency factor
    for i, key in enumerate(trail_points_dict.keys()):
        overlay = frame.copy()  
        for point in trail_points_dict[key]:
            # Convert the point to integer values
            center = (int(point[0]), int(point[1]))
            # Draw a circle with the corresponding color
            cv2.circle(frame, center, radius=radius_val, color=colors[i], thickness=-1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame


# Function to generate a list of colors from dark to light
def generate_gradient_colors(n):
    colors = []
    for i in range(n):
        # Interpolating from light red (255, 182, 193) to light green (144, 238, 144)
        r = int(255 + (144 - 255) * (i / (n - 1)))
        g = int(182 + (238 - 182) * (i / (n - 1)))
        b = int(193 + (144 - 193) * (i / (n - 1)))
        colors.append((b, g, r))  # OpenCV uses BGR format
    return colors

    
def generate_gradient_colors_heat_map(numbers):
    # Get the number of shades
    n = len(numbers)
    
    # Generate gradient colors
    gradient_colors = generate_gradient_colors(n)
    
    # Sort numbers and get their original indices
    sorted_indices = sorted(range(n), key=lambda k: numbers[k],reverse=True)
    
    # Create a color mapping based on sorted indices
    color_map = [None] * n
    for i, idx in enumerate(sorted_indices):
        color_map[idx] = gradient_colors[i]
    
    return color_map