#include "CVCamera.h"

#include <opencv2/imgproc.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <bitset>

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
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1080);
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
        cv::adaptiveThreshold(frame_gray, frame_tresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 33, 5);
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
    cv::findContours(frame_tresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

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

            if (bounding_box.height > 20 && bounding_box.width > 20 && bounding_box.height < frame_raw.cols - 10 && bounding_box.width < frame_raw.rows - 10)
            {
                // Check for convexion
                if (cv::isContourConvex(approximation))
                {
                    // Save to array
                    rectangles.insert(rectangles.end(), approximation);

                    // Draw Outline
                    if (draw_on_overlay)
                    {
                        cv::polylines(frame_overlay, approximation, true, cv::Scalar(255, 0, 0));
                    }

                    // This array saves the edge lines create from subpixel accurate edge points
                    float subpix_edge_line_params[16];
                    cv::Mat subpix_edge_line_params_mat(cv::Size(4, 4), CV_32F, subpix_edge_line_params);

                    // Subdivide each of the 4 edges
                    for (int j = 0; j < approximation.size(); j++)
                    {
                        cv::Point line = approximation[(j + 1) % 4] - approximation[j];

                        StripDimensions strip_dimensions;

                        cv::Mat image_pixel_strip = calculateStripDimensions(line.x / 7, line.y / 7, strip_dimensions, draw_on_overlay);

                        cv::Point2f subpix_edge_points[6];

                        // Subdivision into 7 parts
                        for (int k = 0; k < 7; k++)
                        {
                            // Calculate Subdivision point
                            cv::Point subdivisionPoint = approximation[j] + (line * (k / 7.0));

                            // Draw Corner and Subdivision circles
                            if (draw_on_overlay)
                            {
                                cv::circle(frame_overlay, subdivisionPoint, 1, cv::Scalar(0, 255, 0));
                            }

                            // Skip the corner
                            if (k == 0)
                            {
                                continue;
                            }

                            // Compute Strip
                            cv::Point point_approx_edge;
                            point_approx_edge.x = (int)subdivisionPoint.x;
                            point_approx_edge.y = (int)subdivisionPoint.y;
                            computeStrip(&point_approx_edge, &strip_dimensions, &image_pixel_strip, draw_on_overlay);

                            // Simple sobel over the y direction
                            cv::Mat sobel_gradient_y;
                            cv::Sobel(image_pixel_strip, sobel_gradient_y, CV_8UC1, 0, 1);

                            double max_intensity = -1;
                            int max_intensity_index = 0;

                            // Finding the max value
                            for (int n = 0; n < strip_dimensions.stripLength - 2; ++n)
                            {
                                if (sobel_gradient_y.at<uchar>(n, 1) > max_intensity)
                                {
                                    max_intensity = sobel_gradient_y.at<uchar>(n, 1);
                                    max_intensity_index = n;
                                }
                            }

                            // Added in Sheet 3 - Ex7 (b) End *****************************************************************

                            // Added in Sheet 3 - Ex7 (d) Start *****************************************************************

                            // f(x) slide 7 -> y0 .. y1 .. y2
                            double y0, y1, y2;

                            // Point before and after
                            unsigned int max1 = max_intensity_index - 1, max2 = max_intensity_index + 1;

                            // If the index is at the border we are out of the stripe, then we will take 0
                            y0 = (max_intensity_index <= 0) ? 0 : sobel_gradient_y.at<uchar>(max1, 1);
                            y1 = sobel_gradient_y.at<uchar>(max_intensity_index, 1);
                            // If we are going out of the array of the sobel values
                            y2 = (max_intensity_index >= strip_dimensions.stripLength - 3) ? 0 : sobel_gradient_y.at<uchar>(max2, 1);

                            // Formula for calculating the x-coordinate of the vertex of a parabola, given 3 points with equal distances
                            // (xv means the x value of the vertex, d the distance between the points):
                            // xv = x1 + (d / 2) * (y2 - y0)/(2*y1 - y0 - y2)

                            // d = 1 because of the normalization and x1 will be added later
                            double pos = (y2 - y0) / (4 * y1 - 2 * y0 - 2 * y2);

                            // What happens when there is no solution -> /0 or Number == other Number
                            // If the found pos is not a number -> there is no solution
                            if (isnan(pos))
                            {
                                continue;
                            }

                            // Exact point with subpixel accuracy
                            cv::Point2d edge_center_subpix;

                            // Where is the edge (max gradient) in the picture?
                            int max_index_shift = max_intensity_index - (strip_dimensions.stripLength >> 1);

                            // Find the original edgepoint -> Is the pixel point at the top or bottom?
                            edge_center_subpix.x = (double)point_approx_edge.x + (((double)max_index_shift + pos) * strip_dimensions.stripeVecY.x);
                            edge_center_subpix.y = (double)point_approx_edge.y + (((double)max_index_shift + pos) * strip_dimensions.stripeVecY.y);

                            // Highlight the subpixel with blue color
                            if (draw_on_overlay)
                            {
                                cv::circle(frame_overlay, edge_center_subpix, 2, CV_RGB(255, 0, 0), -1);
                            }

                            // Save point (has to be k-1 as we only have an array of size 6 but loop through 7 points)
                            subpix_edge_points[k - 1] = edge_center_subpix;

                            // Added in Sheet 3 - Ex7 (d) End *****************************************************************
                        }

                        // Added in sheet 4 Ex9(a) - Start * ****************************************************************
                        // Fit line through all subpixel points
                        cv::Mat subpix_point_mat(cv::Size(1, 6), CV_32FC2, subpix_edge_points);
                        cv::fitLine(subpix_point_mat, subpix_edge_line_params_mat.col(j), cv::DIST_L2, 0, 0.01, 0.01);

                        // We need two points to draw the line
                        cv::Point p1;
                        // We have to jump through the 4x4 matrix, meaning the next value for the wanted line is in the next row -> +4
                        // d = -50 is the scalar -> Length of the line, g: Point + d*Vector
                        // p1<----Middle---->p2
                        //   <-----100----->
                        p1.x = (int)subpix_edge_line_params[8 + j] - (int)(50.0 * subpix_edge_line_params[j]);
                        p1.y = (int)subpix_edge_line_params[12 + j] - (int)(50.0 * subpix_edge_line_params[4 + j]);

                        cv::Point p2;
                        p2.x = (int)subpix_edge_line_params[8 + j] + (int)(50.0 * subpix_edge_line_params[j]);
                        p2.y = (int)subpix_edge_line_params[12 + j] + (int)(50.0 * subpix_edge_line_params[4 + j]);

                        // Draw line
                        if (draw_on_overlay)
                        {
                            cv::line(frame_overlay, p1, p2, CV_RGB(0, 255, 255), 1, 8, 0);
                        }
                    }

                    // Added in sheet 4 Ex9 (b)- Start *****************************************************************

                    // So far we stored the exact line parameters and show the lines in the image now we have to calculate the exact corners
                    std::array<cv::Point2f, 4> subpix_corners = calculateSubpixCorners(subpix_edge_line_params, draw_on_overlay);

                    // Ex9 (c)
                    getMarkerId(frame_tresh, subpix_corners, draw_on_overlay);
                }
            }
        }
    }

    return rectangles.size();
}

cv::Mat CVCamera::calculateStripDimensions(double dx, double dy, StripDimensions &st, bool drawOnOverlay = false)
{
    // Norm (euclidean distance) from the direction vector is the length (derived from the Pythagoras Theorem)
    double diffLength = sqrt(dx * dx + dy * dy);

    // Length proportional to the marker size
    st.stripLength = (int)(0.8 * diffLength);

    if (st.stripLength < 5)
        st.stripLength = 5;

    // Make stripeLength odd (because of the shift in nStop), Example 6: both sides of the strip must have the same length XXXOXXX
    // st.stripeLength |= 1;
    if (st.stripLength % 2 == 0)
        st.stripLength++;

    // E.g. stripeLength = 5 --> from -2 to 2: Shift -> half top, the other half bottom
    // st.nStop = st.stripeLength >> 1;
    st.nStop = st.stripLength / 2;
    st.nStart = -st.nStop;

    cv::Size stripeSize;

    // Sample a strip of width 3 pixels
    stripeSize.width = 3;
    stripeSize.height = st.stripLength;

    // Normalized direction vector
    st.stripeVecX.x = dx / diffLength;
    st.stripeVecX.y = dy / diffLength;

    // Normalized perpendicular direction vector (rotated 90  clockwise, rotation matrix)
    st.stripeVecY.x = st.stripeVecX.y;
    st.stripeVecY.y = -st.stripeVecX.x;

    // 8 bit unsigned char with 1 channel, gray
    return cv::Mat(stripeSize, CV_8UC1);
}

void CVCamera::computeStrip(cv::Point *centerPoint, StripDimensions *strip, cv::Mat *outImagePixelStrip, bool drawOnOverlay = false)
{
    // Iterate over width (3 pixels)
    for (int m = -1; m <= 1; m++)
    {
        for (int n = strip->nStart; n <= strip->nStop; n++)
        {
            cv::Point2f subPixel;

            // m -> going over the 3 pixel thickness of the stripe, n -> over the length of the stripe, direction comes from the orthogonal vector in st
            // Going from bottom to top and defining the pixel coordinate for each pixel belonging to the stripe
            subPixel.x = (double)centerPoint->x + ((double)m * strip->stripeVecX.x) + ((double)n * strip->stripeVecY.x);
            subPixel.y = (double)centerPoint->y + ((double)m * strip->stripeVecX.y) + ((double)n * strip->stripeVecY.y);

            if (drawOnOverlay)
            {
                // Just for markings in the image!
                cv::Point p2;
                p2.x = (int)subPixel.x;
                p2.y = (int)subPixel.y;

                cv::circle(frame_overlay, p2, 1, CV_RGB(255, 0, 255), -1);
            }

            // Combined Intensity of the subpixel
            int pixelIntensity = subpixSampleSafe(frame_gray, subPixel);
            // int pixelIntensity = (((m+1)+n) % 2) * 255; // TEST

            // Converte from index to pixel coordinate
            // m (Column, real) -> -1,0,1 but we need to map to 0,1,2 -> add 1 to 0..2
            int w = m + 1;

            // n (Row, real) -> add stripelenght >> 1 to shift to 0..stripeLength
            // n=0 -> -length/2, n=length/2 -> 0 ........ + length/2
            int h = n + (strip->stripLength >> 1);

            // Set pointer to correct position and safe subpixel intensity
            outImagePixelStrip->at<uchar>(h, w) = (uchar)pixelIntensity;

            // Added in Sheet 3 - Ex7 (a) End *****************************************************************
        }
    }
}

int CVCamera::subpixSampleSafe(const cv::Mat &pSrc, const cv::Point2f &p)
{
    int x = int(floorf(p.x));
    int y = int(floorf(p.y));
    if (x < 0 || x >= pSrc.cols - 1 || y < 0 || y >= pSrc.rows - 1)
        return 127;
    int dx = int(256 * (p.x - floorf(p.x)));
    int dy = int(256 * (p.y - floorf(p.y)));
    unsigned char *i = (unsigned char *)((pSrc.data + y * pSrc.step) + x);
    int a = i[0] + ((dx * (i[1] - i[0])) >> 8);
    i += pSrc.step;
    int b = i[0] + ((dx * (i[1] - i[0])) >> 8);
    return a + ((dy * (b - a)) >> 8);
}

std::array<cv::Point2f, 4> CVCamera::calculateSubpixCorners(float subpix_line_params[16], bool draw_on_overlay)
{
    std::array<cv::Point2f, 4> subpix_corners;

    // Calculate the intersection points of both lines
    for (int i = 0; i < 4; ++i)
    {
        // Go through the corners of the rectangle, 3 -> 0
        int j = (i + 1) % 4;

        double x0, x1, y0, y1, u0, u1, v0, v1;

        // We have to jump through the 4x4 matrix, meaning the next value for the wanted line is in the next row -> +4
        // g: Point + d*Vector
        // g1 = (x0,y0) + scalar0*(u0,v0) == g2 = (x1,y1) + scalar1*(u1,v1)
        x0 = subpix_line_params[i + 8];
        y0 = subpix_line_params[i + 12];
        x1 = subpix_line_params[j + 8];
        y1 = subpix_line_params[j + 12];

        // Direction vector
        u0 = subpix_line_params[i];
        v0 = subpix_line_params[i + 4];
        u1 = subpix_line_params[j];
        v1 = subpix_line_params[j + 4];

        // (x|y) = p + s * vec --> Vector Equation

        // (x|y) = p + (Ds / D) * vec

        // p0.x = x0; p0.y = y0; vec0.x= u0; vec0.y=v0;
        // p0 + s0 * vec0 = p1 + s1 * vec1
        // p0-p1 = vec(-vec0 vec1) * vec(s0 s1)

        // s0 = Ds0 / D (see cramer's rule)
        // s1 = Ds1 / D (see cramer's rule)
        // Ds0 = -(x0-x1)v1 + (y0-y1)u1 --> You need to just calculate one, here Ds0

        // (x|y) = (p * D / D) + (Ds * vec / D)
        // (x|y) = (p * D + Ds * vec) / D

        // x0 * D + Ds0 * u0 / D    or   x1 * D + Ds1 * u1 / D     --> a / D
        // y0 * D + Ds0 * v0 / D    or   y1 * D + Ds1 * v1 / D     --> b / D

        // (x|y) = a / c;

        // Cramer's rule
        // 2 unknown a,b -> Equation system
        double a = x1 * u0 * v1 - y1 * u0 * u1 - x0 * u1 * v0 + y0 * u0 * u1;
        double b = -x0 * v0 * v1 + y0 * u0 * v1 + x1 * v0 * v1 - y1 * v0 * u1;

        // Calculate the cross product to check if both direction vectors are parallel -> = 0
        // c -> Determinant = 0 -> linear dependent -> the direction vectors are parallel -> No division with 0
        double c = v1 * u0 - v0 * u1;
        if (fabs(c) < 0.001)
        {
            std::cout << "lines parallel" << std::endl;
            continue;
        }

        // We have checked for parallelism of the direction vectors
        // -> Cramer's rule, now divide through the main determinant
        a /= c;
        b /= c;

        // Exact corner
        subpix_corners[i].x = a;
        subpix_corners[i].y = b;

        // Added in sheet 4 Ex9.b)- End *******************************************************************

        if (draw_on_overlay)
        {
            cv::Point point_draw;
            point_draw.x = (int)subpix_corners[i].x;
            point_draw.y = (int)subpix_corners[i].y;

            circle(frame_overlay, point_draw, 5, CV_RGB(255, 255, 0), -1); // Added in sheet 4 Ex9.c)
        }
    } // End of the loop to extract the exact corners

    return subpix_corners;
}

int CVCamera::getMarkerId(cv::Mat frame_src, std::array<cv::Point2f, 4> subpix_corners, bool draw_marker_id = false)
{
    // Create transformation matrix
    std::array<cv::Point2f, 4> stencil_corners;
    stencil_corners[0] = cv::Point2f(-0.5, -0.5);
    stencil_corners[1] = cv::Point2f(5.5, -0.5);
    stencil_corners[2] = cv::Point2f(5.5, 5.5);
    stencil_corners[3] = cv::Point2f(-0.5, 5.5);

    cv::Mat perspective_transform_mat(cv::Size(3, 3), CV_32FC1);
    perspective_transform_mat = cv::getPerspectiveTransform(subpix_corners, stencil_corners);

    // Create stencil to check
    cv::Mat stencil_mat = cv::Mat(cv::Size(6, 6), CV_8UC1);

    cv::warpPerspective(frame_src, stencil_mat, perspective_transform_mat, cv::Size(6, 6));

    // Retresh for corner pieces
    cv::threshold(stencil_mat, stencil_mat, 200, 256, cv::THRESH_BINARY);


    // Discard everything that doesnt have a black border
    for (int i = 0; i < 6; i++)
    {
        if (stencil_mat.at<uchar>(0, i) > 0 || stencil_mat.at<uchar>(i, 0) > 0 || stencil_mat.at<uchar>(5, i) > 0 || stencil_mat.at<uchar>(i, 5) > 0)
        {
            return -1;
        }
    }

    std::array<std::bitset<16>, 4> codes;
    // Check all possible rotations of the marker at once
    for (int i = 1; i < 5; i++)
    {
        for (int j = 1; j < 5; j++)
        {
            // idx of position in 6x6 matrix transfered to 4x4 matrix
            int position = (j - 1) + ((i - 1) * 4);

            // Save everything in bitset
            codes[0][position] = (stencil_mat.at<uchar>(i, j) == 0);
            codes[1][position] = (stencil_mat.at<uchar>(j, 5 - i) == 0);
            codes[2][position] = (stencil_mat.at<uchar>(5 - i, 5 - j) == 0);
            codes[3][position] = (stencil_mat.at<uchar>(5 - j, i) == 0);
        }
    }

    // Check if there is any symmetric sides
    if (codes[0] == codes[2] || codes[1] == codes[3])
    {
        return -1;
    }

    // Find smallest
    int code = codes[0].to_ulong();
    for (int i = 0; i < 4; i++)
    {
        unsigned long current_code = codes[i].to_ulong();
        if (current_code < code)
        {
            code = current_code;
        }
    }

    // Draw text
    if (draw_marker_id)
    {
        std::ostringstream ss;
        ss << std::hex << code;
        std::string result = ss.str();
        cv::putText(frame_overlay, result, subpix_corners[0], cv::FONT_HERSHEY_SIMPLEX, 5, CV_RGB(255, 255, 0));
    }

    return code;
}