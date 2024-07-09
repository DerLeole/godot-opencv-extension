#ifndef CV_CAMERA_H
#define CV_CAMERA_H

#include <stdio.h>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/image.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace godot {

class CVCamera : public RefCounted {
	GDCLASS(CVCamera, RefCounted)

private:
	cv::VideoCapture capture;
    cv::Mat frame;
    cv::Mat frame_tresh;
    bool flip_lr, flip_ud;

protected:
    static void _bind_methods();
	String _to_string() const;

public:
	CVCamera();
	~CVCamera();

    void open(int device);
    void close();
    Ref<Image> get_image();
    int get_width();
    int get_height();
    void flip(bool flip_lr, bool flip_ud);
};

} //namespace godot

#endif