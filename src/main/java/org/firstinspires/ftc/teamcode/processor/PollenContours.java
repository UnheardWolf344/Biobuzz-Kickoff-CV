/*
 * Copyright (c) 2020 OpenFTC Team
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.firstinspires.ftc.teamcode.processor;

import java.util.ArrayList;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

/**
 * Detects and annotates contours in the Cb channel of the image. The pipeline
 * isolates the
 * pollen color, removes small noise, and draws the contour and center point for
 * debugging.
 */
public class PollenContours extends OpenCvPipeline {
    /*
     * Working image buffers reused across frames to reduce allocation churn during
     * live image
     * processing.
     */
    Mat cbMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedThreshold = new Mat();
    Mat contoursOnPlainImageMat = new Mat();
    Mat outputImg = new Mat();

    private final Telemetry telemetry;

    /*
     * Threshold settings for the Cb channel, which is used to isolate the target
     * color.
     */
    public static int CB_MAX_THRESHOLD = 255;
    public static int CB_MIN_THRESHOLD = 102;
    static final double DENSITY_UPRIGHT_THRESHOLD = 0.03;

    /*
     * Morphological kernels used to clean up the binary mask before contour
     * detection.
     */
    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(8, 8));
    Mat erodeToCircle = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(16, 16));
    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(12, 12));

    /*
     * Colors used for annotation overlays in the debug viewport.
     */
    static final Scalar TEAL = new Scalar(3, 148, 252);
    static final Scalar PURPLE = new Scalar(158, 52, 235);
    static final Scalar RED = new Scalar(255, 0, 0);
    static final Scalar GREEN = new Scalar(0, 255, 0);
    static final Scalar BLUE = new Scalar(0, 0, 255);

    static final int CONTOUR_LINE_THICKNESS = 2;
    static final int CB_CHAN_IDX = 2;

    /**
     * Creates the pipeline and stores the telemetry used to report the active debug
     * stage.
     */
    public PollenContours(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    /*
     * These stages let the user inspect the intermediate image buffers during
     * debugging.
     */
    enum Stage {
        FINAL,
        Cb,
        MASK,
        MASK_NR,
        CONTOURS;
    }

    Stage[] stages = Stage.values();

    /**
     * Tracks which debug buffer is currently selected in the viewport.
     */
    int stageNum = 0;

    @Override
    public void onViewportTapped() {
        /*
         * This callback runs on the UI thread, so the work needs to stay minimal and
         * quick.
         */
        int nextStageNum = stageNum + 1;

        if (nextStageNum >= stages.length) {
            nextStageNum = 0;
        }

        stageNum = nextStageNum;
    }

    @Override
    public Mat processFrame(Mat input) {
        input.copyTo(outputImg);

        /*
         * Detect contours in the filtered image and draw their centers on the output
         * buffer.
         */
        for (MatOfPoint contour : findContours(input)) {
            analyzeContour(contour, input);
        }

        /*
         * Return the requested debug buffer for the selected stage so the viewport can
         * be used
         * to inspect the threshold, filtered mask, or contour overlay.
         */
        switch (stages[stageNum]) {
            case Cb:
                return cbMat;

            case FINAL:
                return outputImg;

            case MASK:
                return thresholdMat;

            case MASK_NR:
                return morphedThreshold;

            case CONTOURS:
                return contoursOnPlainImageMat;

        }

        telemetry.addData("Current Stage", stages[stageNum].name());
        telemetry.update();

        return input;
    }

    /**
     * Converts the image to YCrCb, extracts the Cb channel, thresholds it to form a
     * mask, and
     * searches for external contours in the cleaned binary image.
     */
    ArrayList<MatOfPoint> findContours(Mat input) {
        ArrayList<MatOfPoint> contoursList = new ArrayList<>();

        // Convert the input image to YCrCb color space, then extract the Cb channel.
        Imgproc.cvtColor(input, cbMat, Imgproc.COLOR_RGB2YCrCb);
        Core.extractChannel(cbMat, cbMat, CB_CHAN_IDX);

        // Threshold the Cb channel to form a mask, then run some noise reduction.
        Imgproc.threshold(cbMat, thresholdMat, CB_MIN_THRESHOLD, CB_MAX_THRESHOLD, Imgproc.THRESH_BINARY_INV);
        morphMask(thresholdMat, morphedThreshold);

        // Search for external contours only, which keeps the detection focused on real
        // target
        // regions instead of nested internal edges.
        Imgproc.findContours(morphedThreshold, contoursList, new Mat(), Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_NONE);

        // Copy the original frame to a dedicated overlay buffer so contour outlines can
        // be drawn
        // without modifying the source image used for the next pass.
        input.copyTo(contoursOnPlainImageMat);
        Imgproc.drawContours(contoursOnPlainImageMat, contoursList, -1, BLUE, CONTOUR_LINE_THICKNESS, 8);

        return contoursList;
    }

    /**
     * Erodes and dilates the mask to remove noise while preserving the main target
     * shape.
     */
    void morphMask(Mat input, Mat output) {
        Imgproc.erode(input, output, erodeElement);
        Imgproc.erode(output, output, erodeElement);
        Imgproc.erode(output, output, erodeElement);

        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);

        Imgproc.erode(output, output, erodeToCircle);
    }

    /**
     * Determines the minimum enclosing circle for a contour and draws it on the
     * output image.
     */
    void analyzeContour(MatOfPoint contour, Mat input) {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

        float[] radius = { 0 };
        Point center = new Point();

        Imgproc.minEnclosingCircle(contour2f, center, radius);
        drawCircle(center, radius[0], outputImg);
    }

    /**
     * Draws a circle and label to help visualize the contour center during
     * debugging.
     */
    static void drawCircle(Point center, float radius, Mat drawOn) {
        Imgproc.circle(drawOn, center, (int) radius, BLUE, 2);
        Imgproc.circle(drawOn, center, 8, RED, -1);
        Imgproc.putText(drawOn, String.format("Ctr: %.1f, %.1f", center.x, center.y),
                new Point(center.x + 15, center.y - 15), Imgproc.FONT_HERSHEY_PLAIN, (2.0 / 640.0) * drawOn.cols(),
                new Scalar(0, 0, 0, 0), (int) ((2.0 / 640.0) * drawOn.cols()));
    }
}
