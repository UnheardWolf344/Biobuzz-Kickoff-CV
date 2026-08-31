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

import org.openftc.easyopencv.OpenCvPipeline;
import org.firstinspires.ftc.robotcore.external.Telemetry;


import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

public class PollenContours extends OpenCvPipeline {
    /*
     * Our working image buffers
     */
    Mat cbMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedThreshold = new Mat();
    Mat contoursOnPlainImageMat = new Mat();
    Mat outputImg = new Mat();

    private Telemetry telemetry = null;

    /*
     * Threshold values
     */
    public static int CB_MAX_THRESHOLD = 255;
    public static int CB_MIN_THRESHOLD = 102;
    static final double DENSITY_UPRIGHT_THRESHOLD = 0.03;

    /*
     * The elements we use for noise reduction
     */
    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(8, 8));
    Mat erodeToCircle = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(16,16));
    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(12,12));


    /*
     * Colors
     */
    static final Scalar TEAL = new Scalar(3, 148, 252);
    static final Scalar PURPLE = new Scalar(158, 52, 235);
    static final Scalar RED = new Scalar(255, 0, 0);
    static final Scalar GREEN = new Scalar(0, 255, 0);
    static final Scalar BLUE = new Scalar(0, 0, 255);

    static final int CONTOUR_LINE_THICKNESS = 2;
    static final int CB_CHAN_IDX = 2;

    public PollenContours(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    /*
     * Some stuff to handle returning our various buffers
     */
    enum Stage
    {
        FINAL,
        Cb,
        MASK,
        MASK_NR,
        CONTOURS;
    }

    Stage[] stages = Stage.values();

    // Keep track of what stage the viewport is showing
    int stageNum = 0;

    @Override
    public void onViewportTapped()
    {
        /*
         * Note that this method is invoked from the UI thread
         * so whatever we do here, we must do quickly.
         */

        int nextStageNum = stageNum + 1;

        if(nextStageNum >= stages.length)
        {
            nextStageNum = 0;
        }

        stageNum = nextStageNum;
    }

    @Override
    public Mat processFrame(Mat input)
    {
        input.copyTo(outputImg);
        /*
         * Run the image processing
         */
        for(MatOfPoint contour : findContours(input))
        {
            analyzeContour(contour, input);
        }

        /*
         * Decide which buffer to send to the viewport
         */
        switch (stages[stageNum])
        {
            case Cb:
            {
                return cbMat;
            }

            case FINAL:
            {
                return outputImg;
            }

            case MASK:
            {
                return thresholdMat;
            }

            case MASK_NR:
            {
                return morphedThreshold;
            }

            case CONTOURS:
            {
                return contoursOnPlainImageMat;
            }
        }

        telemetry.addData("Current Stage", stages[stageNum].name());
        telemetry.update();

        return input;

    }

    ArrayList<MatOfPoint> findContours(Mat input)
    {
        // A list we'll be using to store the contours we find
        ArrayList<MatOfPoint> contoursList = new ArrayList<>();

        // Convert the input image to YCrCb color space, then extract the Cb channel
        Imgproc.cvtColor(input, cbMat, Imgproc.COLOR_RGB2YCrCb);
        Core.extractChannel(cbMat, cbMat, CB_CHAN_IDX);

        // Threshold the Cb channel to form a mask, then run some noise reduction
        Imgproc.threshold(cbMat, thresholdMat, CB_MIN_THRESHOLD, CB_MAX_THRESHOLD, Imgproc.THRESH_BINARY_INV);
        morphMask(thresholdMat, morphedThreshold);

        // Ok, now actually look for the contours! We only look for external contours.
        Imgproc.findContours(morphedThreshold, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        // We do draw the contours we find, but not to the main input buffer.
        input.copyTo(contoursOnPlainImageMat);
        Imgproc.drawContours(contoursOnPlainImageMat, contoursList, -1, BLUE, CONTOUR_LINE_THICKNESS, 8);

        return contoursList;
    }

    void morphMask(Mat input, Mat output)
    {
        /*
         * Apply some erosion and dilation for noise reduction
         */

        Imgproc.erode(input, output, erodeElement);
        
        Imgproc.erode(output, output, erodeElement);
        Imgproc.erode(output, output, erodeElement);

        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);

        Imgproc.erode(output, output, erodeToCircle);


    }

    void analyzeContour(MatOfPoint contour, Mat input)
    {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

        float[] radius = {0};
        Point center = new Point();

        Imgproc.minEnclosingCircle(contour2f, center, radius);
        drawCircle(center, radius[0], outputImg);
    }

    static void drawCircle(Point center, float radius, Mat drawOn)
    {
        Imgproc.circle(drawOn, center, (int) radius, BLUE, 2);
        Imgproc.circle(drawOn, center, 8, RED, -1);
        Imgproc.putText(drawOn, String.format("Ctr: %.1f, %.1f", center.x, center.y), new Point(center.x + 15, center.y -15), Imgproc.FONT_HERSHEY_PLAIN, ( 2.0 / 640.0 ) * drawOn.cols(), new Scalar(0,0,0,0), (int) ((2.0 / 640.0) * drawOn.cols()));
    }
}
