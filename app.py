# coding: utf-8
from flask import Flask
from flask import send_file
from flask import request
from flask import redirect

import io
import cv2
import numpy as np
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html lang="jp">
  <body>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" />
      <input type="submit" value="送信する">
    </form>
  </body>
</html>
'''

@app.route("/", methods=['POST'])
def create():
    if request.files.get('file', None) == None:
        return redirect('')
    body = request.files['file'].read()
    #if body == b'':
    #    return redirect('')

    img_buf = np.fromstring(body, dtype='uint8')
    img = cv2.imdecode(img_buf, 1)
    img = np.asarray(img)
    lines = img.copy()
    canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.GaussianBlur(canny, (25,25), 0)
    canny = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY_INV)[1]
    canny = cv2.Canny(canny, 50, 100)

    cnts = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts.sort(key=cv2.contourArea, reverse=True)

    warp = []
    for i, c in enumerate(cnts):
        arclen = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1*arclen, True)

        level = 1 - float(i)/len(cnts)
        if len(approx) == 4:
            cv2.drawContours(lines, [approx], -1, (0, 0, 255*level), 2)
            if warp == []:
                warp = approx.copy()
        else:
            cv2.drawContours(lines, [approx], -1, (0, 255*level, 0), 2)
        for pos in approx:
            cv2.circle(lines, tuple(pos[0]), 4, (255*level, 0, 0))


    points = warp[:,0,:]
    points = sorted(points, key=lambda x:x[1])
    top = sorted(points[:2], key=lambda x:x[0])
    bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)
    points = np.array(top + bottom, dtype='float32')

    width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
    height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))
    dst = np.array([np.array([0, 0]), np.array([width-1, 0]), np.array([width-1, height-1]), np.array([0, height-1])], np.float32)

    trans = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(img, trans, (int(width), int(height)))

    retval, buffer = cv2.imencode('.png', warped)
    return send_file(io.BytesIO(buffer), mimetype='image/png')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
