from src.Vision.camera import Camera
from timeit import default_timer as timer
from src.Vision.video import Video

class imageCapture:
    def __init__(self, camObj, videoObj ,bgVideoObj):
        self.cam = camObj
        self.video = videoObj
        self.bgVideo = bgVideoObj
        pass
    def getImage(self):
        if self.cam is not None:
            frame, nFrame, t = self.cam.getImage()
            return frame, nFrame, t
        if self.video is not None:
            frame, nFrame, t = self.video.getImageFromVideo()
            return frame, nFrame, t
    def getBgImage(self):
            frame, nFrame, t = self.bgVideo.getImageFromVideo()
            return frame, nFrame, t

    def getCamFps(self, nFrame):
        if self.cam is not None:
            return self.cam.getCamFps
        if self.video is not None:
            return str(self.video.fps)

    def destroy(self):
        #adapt
        pass




