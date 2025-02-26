from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaFromNumpy, cudaToNumpy, cudaDrawLine, cudaAllocMapped, cudaResize
import Jetson.GPIO as GPIO


import cv2
import threading
import numpy as np

class Web_Camera:
	def __init__(self):
		self.video_capture = None
		self.frame = None
		self.grabbed = False
		self.read_thread = None
		self.read_lock = threading.Lock()
		self.running = False
	def open(self, gstreamer_pipeline_string):
		try:
			self.video_capture = cv2.VideoCapture(
				gstreamer_pipeline_string, cv2.CAP_V4L2
			)
			self.grabbed, self.frame = self.video_capture.read()

		except RuntimeError:
			self.video_capture = None
			print("Unable to open camera")

	def start(self):
		if self.running:
			print('Video capturing is already running')
			return None
		if self.video_capture != None:
			self.running = True
			self.read_thread = threading.Thread(target=self.updateCamera)
			self.read_thread.start()
		return self

	def stop(self):
		self.running = False
		self.read_thread.join()
		self.read_thread = None

	def updateCamera(self):
		while self.running:
			try:
				grabbed, frame = self.video_capture.read()
				with self.read_lock:
					self.grabbed = grabbed
					self.frame = frame
			except RuntimeError:
				print("Could not read image from camera")

	def read(self):
		with self.read_lock:
			frame = self.frame.copy()
			grabbed = self.grabbed
		return grabbed, frame

	def release(self):
		if self.video_capture != None:
			self.video_capture.release()
			self.video_capture = None
		if self.read_thread != None:
			self.read_thread.join()

def gstreamer_pipeline(
	sensor_id=0
):
	return (
		"/dev/video%d"
		% (
			sensor_id
		)
	)



def resize(img, resize_factor):
	resized_img = cudaAllocMapped(width=resize_factor[0],height=resize_factor[1],format=img.format)
	cudaResize(img, resized_img)
	return resized_img

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)
GPIO.output(11, GPIO.LOW)
GPIO.output(7, GPIO.LOW)


def run_cameras():
	left_camera = Web_Camera()
	left_camera.open(
		gstreamer_pipeline(
			sensor_id=0
		)
	)
	left_camera.start()

	right_camera = Web_Camera()
	right_camera.open(
		gstreamer_pipeline(
			sensor_id=1
		)
	)
	right_camera.start()
	
	net=detectNet(argv=["--model=12cm/ssd-mobilenet.onnx", "--labels=12cm/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)
	display=videoOutput()
	cnt=0
	detect_r=0
	detect_l=0
	if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():
		try:


			while display.IsStreaming():
				cnt = cnt+1
				_, left_image = left_camera.read()
				_, right_image = right_camera.read()


				cuda_left = cv2.cvtColor(left_image,  cv2.COLOR_BGR2RGB)
				l_img = cudaFromNumpy(cuda_left)

				cuda_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
				r_img = cudaFromNumpy(cuda_right)

				resize_l_img = resize(l_img,(640, 480))
				resize_r_img = resize(r_img,(640, 480))
				detection_l = net.Detect(resize_l_img)
				detection_r = net.Detect(resize_r_img)

				cudaDrawLine(resize_l_img,(240,0),(160,480),(255,1,1,255),5)
				cudaDrawLine(resize_l_img,(400,0),(480,480),(255,1,1,255),5)

				cudaDrawLine(resize_r_img,(240,0),(160,480),(255,1,1,255),5)
				cudaDrawLine(resize_r_img,(400,0),(480,480),(255,1,1,255),5)


				if detection_l!=None:
					for i in detection_l:
						if i.ClassID==1:
							if  (6*i.Center[0] + i.Center[1]) >= 1440 and (6*i.Center[0] - i.Center[1])<= 2400 :
								detect_l= detect_l+1
								print("left_site_Car is Detected")
				if detection_r!=None:
					for i in detection_r:
						if i.ClassID==1:
							if  (6*i.Center[0] + i.Center[1]) >= 1440 and (6*i.Center[0] - i.Center[1])<= 2400 :
								detect_r = detect_r+1								
								print("right_site_Car is Detected")

				if cnt==30:
					if detect_r>10:
						GPIO.output(11, GPIO.HIGH)
						print("high")
					else:
						GPIO.output(11, GPIO.LOW)
						print("low")

					if detect_l>10:
						GPIO.output(7, GPIO.HIGH)
						print("high")
					else:
						GPIO.output(7, GPIO.LOW)
						print("low")
					cnt=0
					detect_r=0
					detect_l=0

				l_numpy = cudaToNumpy(resize_l_img)
				r_numpy = cudaToNumpy(resize_r_img)
				img = np.hstack((l_numpy, r_numpy))


				final_img = cudaFromNumpy(img)

				display.Render(final_img)
				display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

				if not display.IsStreaming():
					break
		finally:
			left_camera.stop()
			left_camera.release()
			right_camera.stop()
			right_camera.release()
			GPIO.cleanup()
		cv2.destroyAllWindows()
	else:
		print("Error: Unable to open both cameras")




if __name__ == "__main__":
    run_cameras()
