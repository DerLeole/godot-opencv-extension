#include "CVCamera.h"

#include <opencv2/imgproc.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/engine.hpp>

using namespace godot;

typedef cv::Vec<uint8_t, 4> Pixel;

void CVCamera::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("_to_string"), &CVCamera::_to_string);
    ClassDB::bind_method(D_METHOD("open"), &CVCamera::open);
    ClassDB::bind_method(D_METHOD("close"), &CVCamera::close);
    ClassDB::bind_method(D_METHOD("get_image"), &CVCamera::get_image);
    ClassDB::bind_method(D_METHOD("get_gray_image"), &CVCamera::get_gray_image);
    ClassDB::bind_method(D_METHOD("get_overlay_image"), &CVCamera::get_overlay_image);
    ClassDB::bind_method(D_METHOD("get_width"), &CVCamera::get_width);
    ClassDB::bind_method(D_METHOD("get_height"), &CVCamera::get_height);
    ClassDB::bind_method(D_METHOD("flip"), &CVCamera::flip);
    ClassDB::bind_method(D_METHOD("set_threshold"), &CVCamera::set_threshold);
    ClassDB::bind_method(D_METHOD("get_threshold_image"), &CVCamera::get_threshold_image);

    ClassDB::bind_method(D_METHOD("find_rectangles"), &CVCamera::find_rectangles);
}

CVCamera::CVCamera()
{
    last_update_frame = -1;
    threshold = 0.0;
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
    frame_overlay = cv::Mat::zeros(frame_raw.size(), CV_8UC4);
}

Ref<Image> CVCamera::mat_to_image(cv::Mat mat)
{
    cv::Mat image_mat;
    if (mat.channels() == 1)
    {
        cv::cvtColor(mat, image_mat, cv::COLOR_GRAY2RGB);
    }
    else if (mat.channels() == 4)
    {
        // Turn Pixels alpha value opaque, where there is anything but black
        image_mat = mat;
        image_mat.forEach<Pixel>([](Pixel &p, const int *position) -> void
                                 {
            if (p[0] > 0 || p[1] > 0 || p[2] > 0)
            {
                p[3] = 255;
            } });
    }
    else
    {
        image_mat = mat;
    }

    int sizear = image_mat.cols * image_mat.rows * image_mat.channels();

    PackedByteArray bytes;
    bytes.resize(sizear);
    memcpy(bytes.ptrw(), image_mat.data, sizear);

    Ref<Image> image;
    if (image_mat.channels() == 4)
    {
        image = Image::create_from_data(image_mat.cols, image_mat.rows, false, Image::Format::FORMAT_RGBA8, bytes);
    }
    else
    {
        image = Image::create_from_data(image_mat.cols, image_mat.rows, false, Image::Format::FORMAT_RGB8, bytes);
    }
    return image;
}

Ref<Image> CVCamera::get_image()
{
    update_frame();

    return mat_to_image(frame_rgb);
}

Ref<Image> CVCamera::get_gray_image()
{
    update_frame();

    return mat_to_image(frame_gray);
}

Ref<Image> CVCamera::get_overlay_image()
{
    update_frame();

    return mat_to_image(frame_overlay);
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

Ref<Image> CVCamera::get_threshold_image()
{
    update_frame();

    if (threshold <= 0.0)
    {
        cv::adaptiveThreshold(frame_gray, frame_tresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    }
    else
    {
        cv::threshold(frame_gray, frame_tresh, threshold, 255, cv::THRESH_BINARY);
    }

    return mat_to_image(frame_tresh);
}

int CVCamera::find_rectangles(bool draw_on_overlay = false)
{
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(frame_tresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> rectangles;

    for (int i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> approximation;

        // Approximate contour with lines
        cv::approxPolyDP(contours[i], approximation, cv::arcLength(contours[i], true) * 0.02, true);

        // Check for four corners
        if (approximation.size() == 4)
        {
            // Check for size
            cv::Rect bounding_box = cv::boundingRect(approximation);

            if (bounding_box.height > 10 && bounding_box.width > 10)
            {
                // Check for convexn
                if (cv::isContourConvex(approximation))
                {
                    rectangles.insert(rectangles.end(), approximation);

                    if (draw_on_overlay)
                    {
                        cv::polylines(frame_overlay, approximation, true, cv::Scalar(255, 0, 0));
                    }
                }
            }
        }
    }

    // Subdivide each line into 7 parts
    std::vector<std::vector<std::vector<cv::Point>>> subpoints(rectangles.size(), std::vector<std::vector<cv::Point>>(4, std::vector<cv::Point>(7, cv::Point2i(0, 0))));
    for (int i = 0; i < rectangles.size(); i++)
    {
        for (int j = 0; j < rectangles[i].size(); j++)
        {
            cv::Point line = rectangles[i][(j + 1) % 4] - rectangles[i][j];

            for (int k = 0; k < 7; k++)
            {
                subpoints[i][j][k] = rectangles[i][j] + (line * (k / 7.0));

                if (draw_on_overlay)
                {
                    cv::circle(frame_overlay, subpoints[i][j][k], 3, cv::Scalar(0, 255, 0));

                    // Draw Chess lines
                    if (j >= 2 && k >= 1) // 2/4 sides of the rectangle done & dont do it for the 0th element
                    {
                        cv::line(frame_overlay, subpoints[i][j][k], subpoints[i][j - 2][7 - k], cv::Scalar(0, 0, 255));
                    }
                }

            }
        }


    }

    return rectangles.size();
}