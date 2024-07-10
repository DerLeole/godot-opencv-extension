#include "CVCamera.h"

#include <opencv2/imgproc.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/engine.hpp>

using namespace godot;

void CVCamera::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("_to_string"), &CVCamera::_to_string);
    ClassDB::bind_method(D_METHOD("open"), &CVCamera::open);
    ClassDB::bind_method(D_METHOD("close"), &CVCamera::close);
    ClassDB::bind_method(D_METHOD("get_image"), &CVCamera::get_image);
    ClassDB::bind_method(D_METHOD("get_width"), &CVCamera::get_width);
    ClassDB::bind_method(D_METHOD("get_height"), &CVCamera::get_height);
    ClassDB::bind_method(D_METHOD("flip"), &CVCamera::flip);
}

CVCamera::CVCamera()
{
    last_update_frame = -1;
}

CVCamera::~CVCamera()
{
    close();
}

void CVCamera::open(int device)
{
    capture.open(device);
    if (!capture.isOpened())
    {
        capture.release();
        printf("Error: Could not open camera\n");
    }
}

void CVCamera::close()
{
    capture.release();
}

void CVCamera::update_frame()
{
    // Only update the frame once per godot process frame
    uint64_t current_frame = Engine::get_singleton()->get_process_frames();
    if (current_frame == last_update_frame)
    {
        return;
    }
    last_update_frame = current_frame;

    // Read the frame from the camera
    capture.read(frame_raw);

    if (frame_raw.empty())
    {
        printf("Error: Could not read frame\n");
    }

    if (flip_lr || flip_ud)
    {
        int code = flip_lr ? (flip_ud ? -1 : 1) : 0;
        cv::flip(frame_raw, frame_raw, code);
    }

    cv::cvtColor(frame_raw, frame_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(frame_rgb, frame_gray, cv::COLOR_RGB2GRAY);
}

Ref<Image> CVCamera::mat_to_image(cv::Mat mat)
{
    cv::Mat image_mat;
    mat.convertTo(image_mat, CV_8U);

    int sizear = image_mat.cols * image_mat.rows * image_mat.channels();

    PackedByteArray bytes;
    bytes.resize(sizear);
    memcpy(bytes.ptrw(), image_mat.data, sizear);

    Ref<Image> image = Image::create_from_data(image_mat.cols, image_mat.rows, false, Image::Format::FORMAT_RGB8, bytes);
    return image;
}

Ref<Image> CVCamera::get_image()
{
    update_frame();

    return mat_to_image(frame_rgb);
}

int CVCamera::get_width()
{
    return frame_raw.cols;
}

int CVCamera::get_height()
{
    return frame_raw.rows;
}

void CVCamera::flip(bool flip_lr, bool flip_ud)
{
    this->flip_lr = flip_lr;
    this->flip_ud = flip_ud;
}

String CVCamera::_to_string() const
{
    return "[ CVCamera instance ]";
}

void CVCamera::set_threshold(double threshold)
{
    this->threshold = threshold;
}