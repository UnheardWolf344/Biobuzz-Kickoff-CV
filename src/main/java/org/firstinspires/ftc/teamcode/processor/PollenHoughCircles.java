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

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

/**
 * Uses the Cb channel of a YCrCb image and Hough circle detection to find circular pollen
 * targets. The viewport can toggle between the processed output and the extracted channel.
 */
public class PollenHoughCircles extends OpenCvPipeline {
    /*
     * These working buffers keep the source image and the extracted color channel available
     * across frames without repeated allocation.
     */
    Mat cbMat = new Mat();
    Mat outputImg = new Mat();

    private final Telemetry telemetry;

    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_OPEN, new Size(6, 6));

    /*
     * Colors used to annotate detected regions in the output image.
     */
    static final Scalar TEAL = new Scalar(3, 148, 252);
    static final Scalar PURPLE = new Scalar(158, 52, 235);
    static final Scalar RED = new Scalar(255, 0, 0);
    static final Scalar GREEN = new Scalar(0, 255, 0);
    static final Scalar BLUE = new Scalar(0, 0, 255);

    static final int CONTOUR_LINE_THICKNESS = 2;
    static final int CB_CHAN_IDX = 2;

    /**
     * Tunable Hough parameters: distance, edge threshold, accumulator threshold, and ratio.
     */
    public Scalar params = new Scalar(4.0, 25, 60, 0.8);

    enum Stage {
        FINAL,
        Cb,
    }

    Stage[] stages = Stage.values();

    /**
     * Tracks which image buffer is being shown in the viewport when the user taps the frame.
     */
    int stageNum = 0;

    /**
     * Creates the pipeline and stores the telemetry used for live stage information.
     */
    public PollenHoughCircles(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    /**
     * Cycles between the final processed output and the extracted Cb channel when the user
     * taps the viewport.
     */
    @Override
    public void onViewportTapped() {
        /*
         * This callback runs on the UI thread, so it must stay short and not perform heavy
         * image work.
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

        findCircles(outputImg);

        /*
         * Update the telemetry before returning the chosen buffer so the active stage is clear
         * while debugging the threshold and circle detection.
         */
        telemetry.addData("Current Stage", stages[stageNum].name());
        telemetry.update();

        switch (stages[stageNum]) {
            case FINAL:
                return outputImg;
            case Cb:
                return cbMat;
        }

        return input;
    }

    /**
     * Converts the incoming frame to YCrCb, extracts the Cb channel, and runs Hough circle
     * detection on the filtered image.
     */
    void findCircles(Mat input) {
        // Convert the input image to YCrCb color space, then extract the Cb channel.
        Imgproc.cvtColor(input, cbMat, Imgproc.COLOR_RGB2YCrCb);
        Core.extractChannel(cbMat, cbMat, CB_CHAN_IDX);

        Imgproc.medianBlur(cbMat, cbMat, 5);
        Mat circles = new Mat();

        Imgproc.HoughCircles(cbMat, circles, Imgproc.HOUGH_GRADIENT_ALT, params.val[0], params.val[1], params.val[2],
                params.val[3], 20);

        for (int x = 0; x < circles.cols(); x++) {
            double[] c = circles.get(0, x);
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            // Draw the circle center marker.
            Imgproc.circle(outputImg, center, 1, new Scalar(0, 100, 100), (int) ((2 / 640.0) * outputImg.cols()), 8, 0);
            // Draw the detected circle outline.
            int radius = (int) Math.round(c[2]);
            Imgproc.circle(outputImg, center, radius, new Scalar(255, 0, 255), 2, 8, 0);
        }

        circles.release();
    }
}
