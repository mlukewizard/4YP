from distutils.core import setup
import py2exe
import matplotlib
import FileDialog
import matplotlib.backends.backend_tkagg

setup(
    options = {
                "py2exe":{
                "dll_excludes": ["MSVCP90.dll", "HID.DLL", "w9xpopen.exe"],
                "includes" : ["matplotlib.backends.backend_tkagg"],
                'packages': ['FileDialog']
            }
        },
    data_files=matplotlib.get_py2exe_datafiles(),
    console=['viewSegmentations.py'])