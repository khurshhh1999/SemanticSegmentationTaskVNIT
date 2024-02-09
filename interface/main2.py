# importing the module
import cv2
 
# reading the video
source = cv2.VideoCapture('E:\Acads\FYP\Videos/turtle_orig_gif_1_150.mp4')
 
# running the loop
while True:
 
    # extracting the frames
    ret, img = source.read()
     
    # converting to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    # displaying the video
    cv2.imshow("Live", gray)
 
    if writer is None:
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(output.shape[1], output.shape[0]), True)

    # exiting the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
     
# closing the window
cv2.destroyAllWindows()
source.release()