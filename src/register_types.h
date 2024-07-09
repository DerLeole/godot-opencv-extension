#ifndef GDOPENCVEXTENSION_REGISTER_TYPES_H
#define GDOPENCVEXTENSION_REGISTER_TYPES_H

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void initialize_opencv_extension(ModuleInitializationLevel p_level);
void uninitialize_opencv_extension(ModuleInitializationLevel p_level);

#endif // GDOPENCVEXTENSION_REGISTER_TYPES_H