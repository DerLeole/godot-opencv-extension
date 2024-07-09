# godot-opencv-extension
Attempt at an unholy remake of [lukacu/bouncy](https://github.com/lukacu/bouncy) for Godot 4 using GDExtension.
I created this as I had a class that was based on [lukacu/bouncy](https://github.com/lukacu/bouncy) and required heavy modification of the GDNative C++ code, which is quite cumbersone due the large amount of boilerplate in GDNative and also couldn't be run in Godot 4.

This project does not aim for feature parity, nor for an extensive mapping of the opencv API to GDScript ([this project](https://github.com/matt-s-clark/godot-gdextension-opencv) attempts that).
It is more of an example or primer on how to implement custom opencv functionality into a Godot project via GDEextension.

Inspired by:
- https://github.com/lukacu/bouncy
- https://github.com/matt-s-clark/godot-gdextension-opencv

# Building
+ Install Scons
+ Download opencv-4.9.0 for your OS and place the header and library files inside the opencv folder
  - Make sure to change the paths in the SConstruct file to match your file paths
+ Initialize the `godot-cpp` submodule
```
git submodule update --init
```
+ Build with Scons
```
scons platform=windows disable_exceptions=false
```

# Demo
To use the demo you first have to build the library files. 

- The building process will automatically copy the .gdextension file and any built libraries to the bin/ folder inside the demo project folder
- You will still have to manually copy the required opencv library files into the demo/bin/ folder.
  - The required library files for your system are noted in the SConstruct file, the path on where to find them depends on where you installed opencv
