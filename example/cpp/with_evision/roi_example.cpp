// Alternative main for a Windows project that ALREADY has Open eVision set up.
// Build this INSTEAD of main.cpp. Do not add a main to an existing MFC project;
// use roi_example.h and the startup/inspection calls below in that application.
#include "roi_example.h"
#include <iostream>

int main() {
    try {
        // 1. Application startup: keep this object alive between inspections.
        // Change these paths to your exported JSON and a BW8 image file.
        EvisionRoiExample inspector(L"C:/models/model.json");

        // 2. A camera image already exists in a real application.
        // Loading a file here makes this example usable without a camera.
        Euresys::Open_eVision::EImageBW8 image;
        image.Load("C:/images/sample.bmp");

        // 3. Create an ROI for this standalone example only.
        // If your application already has an EROIBW8, pass that object directly.
        const int x = 10, y = 20, width = 100, height = 100;
        if (image.GetWidth() < x + width || image.GetHeight() < y + height)
            throw std::invalid_argument("Image is too small for the example ROI");
        Euresys::Open_eVision::EROIBW8 roi;
        roi.Attach(&image);
        roi.SetPlacement(x, y, width, height);

        // 4. Inspection button / worker: reuse inspector with your current ROI.
        const auto result = inspector.Inspect(roi);
        std::cout << "class_id=" << result.class_id
                  << " class_name=" << result.class_name
                  << " confidence=" << result.confidence
                  << " elapsed_ms=" << result.inference_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "Open eVision or application error\n";
        return 1;
    }
}
