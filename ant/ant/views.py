from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from ant import camera
def index(request):
    return render(request,'ant\home.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        yield frame
        yield b'\r\n\r\n'
def video_feed(request):
    return StreamingHttpResponse(gen(camera.VideoCamera()),
                                    content_type='multipart/x-mixed-replace; boundary=frame')
