#include <opencv4/opencv2/opencv.hpp>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>

#include "core/options.hpp"
#include "preview.hpp"

#define WINDOW_NAME "preview"

using namespace cv;
using namespace std::chrono_literals;

class OpencvPreview : public Preview
{
    public:
        OpencvPreview(Options const *options) : Preview(options), img()
        {
            namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            window_width = options->preview_width;
            window_height = options->preview_height;

            img = Mat(window_height, window_width, CV_8UC3);

            thread_ = std::thread(&OpencvPreview::threadFunc, this, options);
        }

        virtual ~OpencvPreview() override
        {
            th_run = false;
            thread_.join();
        }

        virtual void Reset() override {}

        virtual bool Quit() override
        {
            if (getWindowProperty(WINDOW_NAME, WND_PROP_VISIBLE) < 1)
                return false;

            return true;
        }

        virtual void MaxImageSize(unsigned int &w, unsigned int &h) const override { w = h = 0; }

        virtual void Show(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info) override
        {
            // Cache the x sampling locations for speed. This is a quick nearest neighbour resize.
            if (last_image_width_ != info.width)
            {
                last_image_width_ = info.width;
                x_locations_.resize(window_width);
                for (unsigned int i = 0; i < window_width; i++)
                    x_locations_[i] = (i * (info.width - 1) + (window_width - 1) / 2) / (window_width - 1);
            }

            uint8_t *Y_start = span.data();
            uint8_t *U_start = Y_start + info.stride * info.height;
            int uv_size = (info.stride / 2) * (info.height / 2);
            uint8_t *dest = img.data;

            // Choose the right matrix to convert YUV back to RGB.
            static const float YUV2RGB[3][9] = {
                { 1.0,   0.0, 1.402, 1.0,   -0.344, -0.714, 1.0,   1.772, 0.0 }, // JPEG
                { 1.164, 0.0, 1.596, 1.164, -0.392, -0.813, 1.164, 2.017, 0.0 }, // SMPTE170M
                { 1.164, 0.0, 1.793, 1.164, -0.213, -0.533, 1.164, 2.112, 0.0 }, // Rec709
            };
            const float *M = YUV2RGB[0];
            if (info.colour_space == libcamera::ColorSpace::Sycc)
                M = YUV2RGB[0];
            else if (info.colour_space == libcamera::ColorSpace::Smpte170m)
                M = YUV2RGB[1];
            else if (info.colour_space == libcamera::ColorSpace::Rec709)
                M = YUV2RGB[2];
            else
                LOG(1, "QtPreview: unexpected colour space " << libcamera::ColorSpace::toString(info.colour_space));

            // Possibly this should be locked in case a repaint is happening? In practice the risk
            // is only that there might be some tearing, so I don't think we worry. We could speed
            // it up by getting the ISP to supply RGB, but I'm not sure I want to handle that extra
            // possibility in our main application code, so we'll put up with the slow conversion.
            for (unsigned int y = 0; y < window_height; y++)
            {
                int row = (y * (info.height - 1) + (window_height - 1) / 2) / (window_height - 1);
                uint8_t *Y_row = Y_start + row * info.stride;
                uint8_t *U_row = U_start + (row / 2) * (info.stride / 2);
                uint8_t *V_row = U_row + uv_size;
                for (unsigned int x = 0; x < window_width;)
                {
                    int y_off0 = x_locations_[x++];
                    int y_off1 = x_locations_[x++];
                    int uv_off0 = y_off0 >> 1;
                    int uv_off1 = y_off0 >> 1;
                    int Y0 = Y_row[y_off0];
                    int Y1 = Y_row[y_off1];
                    int U0 = U_row[uv_off0];
                    int V0 = V_row[uv_off0];
                    int U1 = U_row[uv_off1];
                    int V1 = V_row[uv_off1];
                    U0 -= 128;
                    V0 -= 128;
                    U1 -= 128;
                    V1 -= 128;
                    int R0 = M[0] * Y0 + M[2] * V0;
                    int G0 = M[3] * Y0 + M[4] * U0 + M[5] * V0;
                    int B0 = M[6] * Y0 + M[7] * U0;
                    int R1 = M[0] * Y1 + M[2] * V1;
                    int G1 = M[3] * Y1 + M[4] * U1 + M[5] * V1;
                    int B1 = M[6] * Y1 + M[7] * U1;
                    *(dest++) = std::clamp(B0, 0, 255);
                    *(dest++) = std::clamp(G0, 0, 255);
                    *(dest++) = std::clamp(R0, 0, 255);
                    *(dest++) = std::clamp(B1, 0, 255);
                    *(dest++) = std::clamp(G1, 0, 255);
                    *(dest++) = std::clamp(R1, 0, 255);
                }
            }

            data_ready.notify_one();
            done_callback_(fd);
        }

        virtual void SetInfoText(const std::string &text) override
        {
            std::unique_lock<std::mutex> lck(mutex);
            title = text;
        }

    private:
        unsigned int window_width, window_height;
        unsigned int last_image_width_ = 0;
        std::atomic_bool th_run = true;
        std::condition_variable data_ready;
        std::vector<uint16_t> x_locations_;
        std::mutex mutex;
        std::thread thread_;
        std::string title;
        Mat img;

        void threadFunc(Options const *options)
        {
            std::mutex m;
            unsigned int radius = 100, thickness = 1, x = window_width / 2, y = window_height / 2;
            Point center(x, y);
            Point p1(x - radius, y);
            Point p2(x + radius, y);
            Point p3(x, y - radius);
            Point p4(x, y + radius);
            Scalar color(0, 0, 255);

            resizeWindow(WINDOW_NAME, window_width, window_height);
            setWindowTitle(WINDOW_NAME, "Ardcam 64 MP preview (" + std::to_string(options->viewfinder_width) + 
            " x " + std::to_string(options->viewfinder_height) + ")");

            while(th_run)
            {
                std::unique_lock lk(m);
                if (data_ready.wait_for(lk, 200ms) != std::cv_status::timeout) {
                    circle(img, center, radius, color, thickness);
                    line(img, p1, p2, color, thickness);
                    line(img, p3, p4, color, thickness);
                    imshow(WINDOW_NAME, img);
                    waitKey(1);
                }
            }
        }
};

Preview *make_opencv_preview(Options const *options)
{
    return new OpencvPreview(options);
}