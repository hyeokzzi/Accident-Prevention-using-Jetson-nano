from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaFromNumpy, cudaToNumpy

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
	
	net=detectNet("ssd-mobilenet-v2", threshold=0.5)
	display=videoOutput()

	if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():
		try:


			while display.IsStreaming():
				_, left_image = left_camera.read()
				_, right_image = right_camera.read()


				cuda_left = cv2.cvtColor(left_image,  cv2.COLOR_BGR2RGB)
				l_img = cudaFromNumpy(cuda_left)

				cuda_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
				r_img = cudaFromNumpy(cuda_right)

				detection_l = net.Detect(l_img)
				detection_r = net.Detect(r_img)

				
				l_numpy = cudaToNumpy(l_img)
				r_numpy = cudaToNumpy(r_img)
				img = np.hstack((l_numpy, r_numpy))


				final_img = cudaFromNumpy(img)

				#print("detected {:d} objects in image".format(len(detections)))
				display.Render(final_img)
				display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))





				if not display.IsStreaming():
					break
		finally:
			left_camera.stop()
			left_camera.release()
			right_camera.stop()
			right_camera.release()
		cv2.destroyAllWindows()
	else:
		print("Error: Unable to open both cameras")




if __name__ == "__main__":
    run_cameras()
