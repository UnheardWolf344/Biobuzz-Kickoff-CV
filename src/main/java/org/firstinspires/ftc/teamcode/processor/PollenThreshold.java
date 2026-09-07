/*
 * Copyright (c) 2023 Sebastian Erives
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package org.firstinspires.ftc.teamcode.processor;

import android.graphics.Canvas;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Applies a configurable color-threshold mask to the incoming camera frame. The pipeline
 * converts the frame to a selected color space, keeps pixels inside the tuned scalar bounds,
 * and draws the masked result back onto the input frame.
 */
public class PollenThreshold implements VisionProcessor {

    /*
     * These values are exposed to the live variable tuner so the threshold can be adjusted
     * without recompiling the pipeline. OpenCV Scalar values are ordered according to the
     * selected color space, so these represent the lower and upper bounds for the active
     * channel values.
     */
    public Scalar lower = new Scalar(0, 130.0, 0);
    public Scalar upper = new Scalar(255, 255, 102.0);

    /**
     * Selects the color space used during thresholding so the tuning interface can swap
     * conversion modes without changing code.
     */
    public ColorSpace colorSpace = ColorSpace.YCrCb;

    /*
     * Reusing these Mat buffers avoids repeatedly allocating new image objects every frame.
     * That reduces garbage collection pressure and helps prevent memory spikes during live
     * vision processing.
     */
    private final Mat ycrcbMat = new Mat();
    private final Mat binaryMat = new Mat();
    private final Mat maskedInputMat = new Mat();

    private final Telemetry telemetry;

    /**
     * Enum used to select the color-space conversion needed before thresholding the frame.
     */
    enum ColorSpace {
       RGB(Imgproc.COLOR_RGBA2RGB),
       HSV(Imgproc.COLOR_RGB2HSV),
       YCrCb(Imgproc.COLOR_RGB2YCrCb),
       Lab(Imgproc.COLOR_RGB2Lab);

       public int cvtCode = 0;

       ColorSpace(int cvtCode) {
           this.cvtCode = cvtCode;
       }
    }

    /**
     * Creates the processor with telemetry for tuning feedback while the camera is live.
     */
    public PollenThreshold(Telemetry telemetry) {
       this.telemetry = telemetry;
    }

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
       // The pipeline does not need any setup work beyond the reusable Mats created above.
    }

    @Override
    public Object processFrame(Mat frame, long captureTimeNanos) {
       /*
        * Convert the incoming RGB frame to the active color space before applying the
        * threshold. This matches the tuning values stored in the Scalar bounds.
        */
       Imgproc.cvtColor(frame, ycrcbMat, colorSpace.cvtCode);

       /*
        * InRange keeps only pixels whose channel values fall between the configured lower
        * and upper boundaries. The result is a binary mask where valid pixels are set to 255.
        */
       Core.inRange(ycrcbMat, lower, upper, binaryMat);

       /*
        * Clear the previous masked buffer before we combine the threshold mask with the
        * source image; otherwise stale pixel data could remain in the reused Mat.
        */
       maskedInputMat.release();

       /*
        * Bitwise AND keeps only the original image pixels that passed the threshold mask,
        * while all rejected pixels are forced to black in the output.
        */
       Core.bitwise_and(frame, frame, maskedInputMat, binaryMat);

       /*
        * Display the current tuning values so they can be adjusted live while testing the
        * camera feed in the EOCV simulator or on-device dashboard.
        */
       telemetry.addData("[>]", "Change these values in tuner menu");
       telemetry.addData("[Color Space]", colorSpace.name());
       telemetry.addData("[Lower Scalar]", lower);
       telemetry.addData("[Upper Scalar]", upper);
       telemetry.update();

       /*
        * Unlike the OpenCVPipeline API, VisionProcessor cannot return a Mat from
        * processFrame. Instead, the processed pixels are copied directly back to the input
        * frame so the viewport reflects the filtered image.
        */
       maskedInputMat.copyTo(frame);
       return null;
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx,
           float scaleCanvasDensity, Object userContext) {
       // No custom drawing logic is required for this simple threshold processor.
    }
}
