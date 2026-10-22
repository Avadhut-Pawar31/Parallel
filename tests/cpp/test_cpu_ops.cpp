#include "parallel/image_io.hpp"
#include "parallel/image_ops.hpp"
#include "parallel/backend.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

using namespace parallel;

static int g_testsPassed = 0;
static int g_testsFailed = 0;

#define TEST_ASSERT(cond, msg)                                                \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":"        \
                      << __LINE__ << ")\n";                                   \
            g_testsFailed++;                                                  \
            return;                                                           \
        }                                                                     \
    } while (0)

#define RUN_TEST(fn)                                                          \
    do {                                                                      \
        std::cout << "[RUNNING] " << #fn << " ... " << std::flush;            \
        int before = g_testsFailed;                                           \
        fn();                                                                 \
        if (g_testsFailed == before) {                                        \
            std::cout << "PASSED\n";                                          \
            g_testsPassed++;                                                  \
        }                                                                     \
    } while (0)

// Test 1: Grayscale conversion with exact ITU-R BT.601 math
void testGrayscaleCorrectness() {
    Image img(2, 2, 3);
    // Pixel (0,0): Pure Red -> 0.299 * 255 = 76.245 -> 76
    img.at(0, 0, 0) = 255; img.at(0, 0, 1) = 0;   img.at(0, 0, 2) = 0;
    // Pixel (1,0): Pure Green -> 0.587 * 255 = 149.685 -> 150
    img.at(1, 0, 0) = 0;   img.at(1, 0, 1) = 255; img.at(1, 0, 2) = 0;
    // Pixel (0,1): Pure Blue -> 0.114 * 255 = 29.07 -> 29
    img.at(0, 1, 0) = 0;   img.at(0, 1, 1) = 0;   img.at(0, 1, 2) = 255;
    // Pixel (1,1): White -> 255
    img.at(1, 1, 0) = 255; img.at(1, 1, 1) = 255; img.at(1, 1, 2) = 255;

    Image gray = cpuGrayscale(img);
    TEST_ASSERT(gray.width == 2 && gray.height == 2 && gray.channels == 1, "Grayscale dimensions mismatch");
    TEST_ASSERT(gray.at(0, 0) == 76, "Red pixel grayscale mismatch");
    TEST_ASSERT(gray.at(1, 0) == 150, "Green pixel grayscale mismatch");
    TEST_ASSERT(gray.at(0, 1) == 29, "Blue pixel grayscale mismatch");
    TEST_ASSERT(gray.at(1, 1) == 255, "White pixel grayscale mismatch");
}

// Test 2: Gaussian Blur on uniform image preserves constant value
void testGaussianBlurUniform() {
    Image uniform(16, 16, 3);
    for (size_t i = 0; i < uniform.size(); ++i) {
        uniform.data[i] = 120;
    }

    Image blurred = cpuGaussianBlur(uniform, 3, 1.0f);
    TEST_ASSERT(blurred.width == 16 && blurred.height == 16, "Blurred dimensions mismatch");
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            for (int c = 0; c < 3; ++c) {
                TEST_ASSERT(blurred.at(x, y, c) == 120, "Uniform blur did not preserve constant pixel value");
            }
        }
    }
}

// Test 3: Sharpen on uniform image preserves constant value
void testSharpenUniform() {
    Image uniform(16, 16, 3);
    for (size_t i = 0; i < uniform.size(); ++i) {
        uniform.data[i] = 100;
    }

    Image sharp = cpuSharpen(uniform);
    TEST_ASSERT(sharp.width == 16 && sharp.height == 16, "Sharpen dimensions mismatch");
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            for (int c = 0; c < 3; ++c) {
                // Laplacian kernel: 5*100 - 4*100 = 100
                TEST_ASSERT(sharp.at(x, y, c) == 100, "Uniform sharpen did not preserve constant pixel value");
            }
        }
    }
}

// Test 4: Sobel edge detection on uniform image yields zero gradient everywhere
void testSobelUniform() {
    Image uniform(16, 16, 1);
    for (size_t i = 0; i < uniform.size(); ++i) {
        uniform.data[i] = 180;
    }

    Image edges = cpuSobel(uniform);
    TEST_ASSERT(edges.width == 16 && edges.height == 16 && edges.channels == 1, "Sobel output dimensions mismatch");
    for (size_t i = 0; i < edges.size(); ++i) {
        TEST_ASSERT(edges.data[i] == 0, "Uniform image produced non-zero edge magnitude in Sobel");
    }
}

// Test 5: Sobel detects a clear vertical edge
void testSobelVerticalEdge() {
    // Left half black (0), right half white (255)
    Image stepImg(10, 10, 1);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            stepImg.at(x, y) = (x < 5) ? 0 : 255;
        }
    }

    Image edges = cpuSobel(stepImg);
    // Along boundary x = 4 and x = 5, gradient should be very high
    TEST_ASSERT(edges.at(4, 5) > 100, "Sobel failed to detect high vertical gradient at edge");
    // At interior x = 1, gradient should be 0
    TEST_ASSERT(edges.at(1, 5) == 0, "Sobel reported false edge in uniform interior");
}

// Test 6: Odd and non-square dimensions
void testOddAndNonSquareDimensions() {
    Image oddImg(17, 31, 3);
    for (size_t i = 0; i < oddImg.size(); ++i) {
        oddImg.data[i] = static_cast<uint8_t>(i % 256);
    }

    Image gray = cpuGrayscale(oddImg);
    TEST_ASSERT(gray.width == 17 && gray.height == 31 && gray.channels == 1, "Odd dimension grayscale failed");

    Image blur = cpuGaussianBlur(oddImg, 5, 1.2f);
    TEST_ASSERT(blur.width == 17 && blur.height == 31 && blur.channels == 3, "Odd dimension blur failed");

    Image sharp = cpuSharpen(oddImg);
    TEST_ASSERT(sharp.width == 17 && sharp.height == 31 && sharp.channels == 3, "Odd dimension sharpen failed");

    Image sobel = cpuSobel(oddImg);
    TEST_ASSERT(sobel.width == 17 && sobel.height == 31 && sobel.channels == 1, "Odd dimension sobel failed");
}

// Test 7: Clamp boundary handling on 1x1 image
void testSinglePixelImage() {
    Image single(1, 1, 3);
    single.at(0, 0, 0) = 50;
    single.at(0, 0, 1) = 100;
    single.at(0, 0, 2) = 150;

    Image blur = cpuGaussianBlur(single, 3, 1.0f);
    TEST_ASSERT(blur.width == 1 && blur.height == 1, "1x1 blur failed");
    TEST_ASSERT(blur.at(0, 0, 0) == 50, "1x1 blur channel 0 mismatch");
    TEST_ASSERT(blur.at(0, 0, 1) == 100, "1x1 blur channel 1 mismatch");
    TEST_ASSERT(blur.at(0, 0, 2) == 150, "1x1 blur channel 2 mismatch");

    Image sharp = cpuSharpen(single);
    TEST_ASSERT(sharp.at(0, 0, 0) == 50, "1x1 sharpen channel 0 mismatch");
}

// Test 8: Error handling on invalid/empty images
void testInvalidInputHandling() {
    Image emptyImg;
    bool caughtGrayscale = false;
    try {
        cpuGrayscale(emptyImg);
    } catch (const std::exception&) {
        caughtGrayscale = true;
    }
    TEST_ASSERT(caughtGrayscale, "Grayscale did not reject empty image");

    bool caughtBlur = false;
    try {
        cpuGaussianBlur(emptyImg);
    } catch (const std::exception&) {
        caughtBlur = true;
    }
    TEST_ASSERT(caughtBlur, "Blur did not reject empty image");
}

int main() {
    std::cout << "========================================================\n"
              << "  Parallel: Automated C++ Correctness Test Suite        \n"
              << "========================================================\n";

    RUN_TEST(testGrayscaleCorrectness);
    RUN_TEST(testGaussianBlurUniform);
    RUN_TEST(testSharpenUniform);
    RUN_TEST(testSobelUniform);
    RUN_TEST(testSobelVerticalEdge);
    RUN_TEST(testOddAndNonSquareDimensions);
    RUN_TEST(testSinglePixelImage);
    RUN_TEST(testInvalidInputHandling);

    std::cout << "========================================================\n"
              << "  Test Summary: " << g_testsPassed << " passed, "
              << g_testsFailed << " failed\n"
              << "========================================================\n";

    return (g_testsFailed == 0) ? 0 : 1;
}
