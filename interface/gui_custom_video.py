import gradio as gr
import matplotlib.pyplot as plt
import cv2

def display_video(video_name):
    cap = cv2.VideoCapture(video_name)
   
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            cv2.imshow('Frame', frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
    
    # When everything done, release 
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()



def display_segmented_videos(video_path):
    
    display_video(video_path)



if __name__ == '__main__':

    demo = gr.Interface(fn=display_segmented_videos, inputs="text", outputs="video")
    demo.launch(share=True)