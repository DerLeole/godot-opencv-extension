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
    uint64_t last_update_frame;
    cv::Mat frame_raw;
    cv::Mat frame_rgb;
    cv::Mat frame_gray;
    cv::Mat frame_tresh;
    cv::Mat frame_overlay;
    bool flip_lr, flip_ud;
    double threshold;

    void update_frame();
    Ref<Image> mat_to_image(cv::Mat mat);

protected:
    static void _bind_methods();
	String _to_string() const;

public:
	CVCamera();
	~CVCamera();

    void open(int device);
    void close();
    Ref<Image> get_image();
    Ref<Image> get_gray_image();
    Ref<Image> get_overlay_image();
    int get_width();
    int get_height();
    void flip(bool flip_lr, bool flip_ud);
    void set_threshold(double threshold);
    Ref<Image> get_threshold_image();
};

} //namespace godot

#endif