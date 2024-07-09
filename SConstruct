#!/usr/bin/env python
import os
import sys
import shutil

env = SConscript("godot-cpp/SConstruct")

# For reference:
# - CCFLAGS are compilation flags shared between C and C++
# - CFLAGS are for C-specific compilation flags
# - CXXFLAGS are for C++-specific compilation flags
# - CPPFLAGS are for pre-processor flags
# - CPPDEFINES are for pre-processor defines
# - LINKFLAGS are for linking flags

# set this path to your library files. This is the location of dll, dylib and so files.
opencv_library_path = [
    'opencv/build/x64/vc16/lib',
    #'YourCustomOpencvFolder/build/x64/vc16/lib'
]

# tweak this if you want to use different folders, or more folders, to store your source code in.

opencv_header_files = [
    "src/",
    "opencv/build/include",
    #"YourCustomOpencvFolder/build/include"
]

opencv_library_files = {
    'windows': [
        'opencv_world490.lib',
        'opencv_world490d.lib'
    ],
    'macos': [
        'libopencv_core.dylib',
        'libopencv_imgcodecs.dylib',
        'libopencv_imgproc.dylib',
        'libopencv_videoio.dylib',
        'libopencv_objdetect.dylib',
        'libopencv_video.dylib',
        'libopencv_tracking.dylib'
    ],
    'linux': [
        'libopencv_core.so',
        'libopencv_imgcodecs.so',
        'libopencv_imgproc.so',
        'libopencv_videoio.so',
        'libopencv_objdetect.so',
        'libopencv_video.so',
        'libopencv_tracking.so'
    ]
}

env.Append(CPPPATH=opencv_header_files)
env.Append(LIBPATH=opencv_library_path)
env.Append(LIBS=opencv_library_files[env["platform"]])

sources = Glob("src/*.cpp")


# Create SharedLibrary

if env["platform"] == "macos":
    library = env.SharedLibrary(
        "demo/bin/godotopencvextension.{}.{}.framework/godotopencvextension.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "demo/bin/godotopencvextension{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

Default(library)


# Copy gdextension file

def copy_extension(target, source, env):
    print(f"Copying {source[0]} to {target[0]}")
    shutil.copy(str(source[0]), str(target[0]))


source_path = 'opencvextension.gdextension'
target_path = 'demo/bin/opencvextension.gdextension'

env.Command(target=target_path, source=source_path, action=copy_extension)

Default(target_path)
