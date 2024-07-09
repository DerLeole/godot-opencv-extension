#include "CVCamera.h"

#include <opencv2/imgproc.hpp>

using namespace godot;

void CVCamera::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_to_string"), &CVCamera::_to_string);
    ClassDB::bind_method(D_METHOD("open"), &CVCamera::open);
    ClassDB::bind_method(D_METHOD("close"), &CVCamera::close);
    ClassDB::bind_method(D_METHOD("get_image"), &CVCamera::get_image);
    ClassDB::bind_method(D_METHOD("get_width"), &CVCamera::get_width);
    ClassDB::bind_method(D_METHOD("get_height"), &CVCamera::get_height);
    ClassDB::bind_method(D_METHOD("flip"), &CVCamera::flip);

}

CVCamera::CVCamera() {
}

CVCamera::~CVCamera() {
    close();
}

void CVCamera::open(int device) {
    capture.open(device);
    if (!capture.isOpened()) {
        capture.release();
        printf("Error: Could not open camera\n");
    }
}

void CVCamera::close() {
    capture.release();
}

Ref<Image> CVCamera::get_image() {
    capture.read(frame);

    if (frame.empty()) {
        printf("Error: Could not read frame\n");
    }

    if (flip_lr || flip_ud) {
        int code = flip_lr ? (flip_ud ? -1 : 1) : 0;
        cv::flip(frame, frame, code);
    }

    cv::Mat frame_rgb;
    cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
    frame_rgb.convertTo(frame_rgb, CV_8U);

    int sizear = frame_rgb.cols * frame_rgb.rows * frame_rgb.channels();

    // TODO: Conversion to image should depend on type

    PackedByteArray bytes;
    bytes.resize(sizear);
    memcpy(bytes.ptrw(), frame_rgb.data, sizear);

    Ref<Image> image = Image::create_from_data(frame_rgb.cols, frame_rgb.rows, false, Image::Format::FORMAT_RGB8, bytes);

    return image;
}

int CVCamera::get_width() {
    return frame.cols;
}

int CVCamera::get_height() {
    return frame.rows;
}

void CVCamera::flip(bool flip_lr, bool flip_ud) {
    this->flip_lr = flip_lr;
    this->flip_ud = flip_ud;
}

String CVCamera::_to_string() const {
	return "[ CVCamera instance ]";
} 