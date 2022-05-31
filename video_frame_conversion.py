import cv2
import numpy as np
import torch
import config as cfg

def convert_video_to_frames(video_capture, trained_model, sec = 0, frame_rate = 1/60):
    video_capture.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = video_capture.read()

    image_list = []    

    while hasFrames:
        
        with torch.no_grad():
            image = cfg.video_input_transform(image=image)["image"]
            image = torch.reshape(image, (1,3,256,256)).to(cfg.device)
            image = trained_model(image)
            image = image * 0.5 + 0.5
        
        image_list.append(image)
        sec = sec + frame_rate
        video_capture.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = video_capture.read()
    
    return image_list

def convert_frames_to_video(frame_list, generated_video_path, fps = 60, size = (1920, 1080)):
    width, height, channels = torch.squeeze(frame_list[0]).cpu().numpy().transpose(1, 2, 0).shape
    out = cv2.VideoWriter(generated_video_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    
    for i in range(len(frame_list)):
        a = torch.squeeze(frame_list[i]).cpu().numpy().transpose(1, 2, 0)
        a = cv2.resize(a, size)
        a = np.uint8(255 * a)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        out.write(a)
        
    out.release()