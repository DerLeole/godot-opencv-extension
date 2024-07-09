# godot-opencv-extension
Attempt at an unholy remake of [lukacu/bouncy](https://github.com/lukacu/bouncy) in Godot 4 using GDExtension.
I created this as I had a class that was based on [lukacu/bouncy](https://github.com/lukacu/bouncy) and required heavy modification of the GDNative C++ code, which is quite cumbersone due the large amount of boilerplate and also couldn't be run in Godot 4.

This project does not aim for feature parity, nor for an extensive mapping of the opencv API to GDScript ([this project](https://github.com/matt-s-clark/godot-gdextension-opencv) attempts that).
It is more of a example or primer on how to implement custom opencv functionality into a Godot project via GDEextension.

Inspired by:
- https://github.com/lukacu/bouncy
- https://github.com/matt-s-clark/godot-gdextension-opencv

# Building
