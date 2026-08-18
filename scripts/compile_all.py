import py_compile
import glob
import traceback

files = glob.glob('d:/Phy_cam_proj/Driver-Drowsiness-Distraction-Detection-Modular_2/driver_monitor_v2/**/*.py', recursive=True)
failed = []
for f in sorted(files):
    try:
        py_compile.compile(f, doraise=True)
    except Exception:
        failed.append((f, traceback.format_exc()))

print('Files checked:', len(files))
print('Failures:', len(failed))
for f, tb in failed:
    print('---')
    print(f)
    print(tb)
