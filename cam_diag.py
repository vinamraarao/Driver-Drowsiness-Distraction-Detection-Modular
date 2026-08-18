import cv2
import platform
import time

backends = [('Default', None)]
if platform.system() == 'Windows':
    backends = [('DirectShow', cv2.CAP_DSHOW), ('Media Foundation', cv2.CAP_MSMF), ('Default', None)]

for bname, b in backends:
    print('== Backend:', bname, '==')
    for idx in range(5):
        try:
            if b is None:
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx, b)
            opened = cap.isOpened()
            print(f'index={idx} opened={opened}')
            if opened:
                # give hardware a moment
                time.sleep(0.2)
                ret, frame = cap.read()
                print('  read_ret=', ret, 'frame_none=', frame is None, 'size=', 0 if frame is None else getattr(frame, 'size', 0))
                cap.release()
        except Exception as e:
            print('  error:', e)
print('Diagnostic complete')
