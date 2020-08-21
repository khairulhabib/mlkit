/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.facedetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.VisionProcessorBase;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Face Detector Demo.
 */
public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {

    private static final String TAG = "FaceDetectorProcessor";

    private final FaceDetector detector;
    private FaceNetModel model;
    private float[] habibFace = new float[]{-0.21660966f, -0.38680297f, -0.4796357f, -0.27849814f, 0.4796357f, 0.20113754f, -0.015472119f, 0.092832714f, 0.030944237f, 0.07736059f, 0.3403866f, -0.20113754f, 0.061888475f, -0.6807732f, -0.046416357f, 0.0f, 0.18566543f, -0.44869146f, 0.44869146f, -0.6962454f, -0.13924907f, -0.092832714f, -0.27849814f, 0.030944237f, 0.030944237f, -0.23208179f, 0.5879405f, 0.1701933f, -0.61888474f, -0.061888475f, -0.46416357f, 0.12377695f, -0.061888475f, -0.030944237f, 0.51057994f, 0.015472119f, 0.6807732f, -0.20113754f, 0.38680297f, -0.030944237f, 0.4332193f, 0.4177472f, -0.046416357f, -0.046416357f, -0.46416357f, -0.4796357f, 0.15472119f, -0.061888475f, 0.5724684f, 0.15472119f, -0.6807732f, 0.37133086f, 0.4796357f, -0.2475539f, -0.20113754f, 0.44869146f, 0.3403866f, -0.046416357f, -0.3249145f, -0.7581338f, -0.44869146f, -0.061888475f, 0.015472119f, 0.7271896f, 0.015472119f, 0.649829f, 0.1701933f, 0.21660966f, 0.15472119f, -0.4332193f, 0.4796357f, -0.38680297f, -0.092832714f, -0.51057994f, 0.30944237f, -0.13924907f, -0.092832714f, 0.8045502f, -0.3403866f, -0.2475539f, -0.4796357f, -0.38680297f, -0.1701933f, -0.07736059f, 0.4022751f, 0.26302603f, 0.1701933f, 0.07736059f, -0.5415242f, 0.7426617f, 0.37133086f, 0.12377695f, -0.12377695f, 0.20113754f, 0.4022751f, 0.63435686f, -0.15472119f, -0.6807732f, -0.78907806f, 0.5415242f, -0.12377695f, 0.52605206f, 0.4796357f, -0.6807732f, -0.030944237f, 0.13924907f, -0.15472119f, -0.4177472f, -0.35585874f, -0.46416357f, 0.092832714f, 0.21660966f, 0.2475539f, 0.44869146f, -0.12377695f, 0.23208179f, -0.38680297f, 0.4951078f, 0.51057994f, 0.44869146f, -0.5724684f, -0.63435686f, 0.52605206f, 0.1701933f, 0.37133086f, -0.4022751f, 0.77360594f, 0.29397026f};

    public FaceDetectorProcessor(Context context) {
        this(
                context,
                new FaceDetectorOptions.Builder()
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                        .enableTracking()
                        .build());
        model = new FaceNetModel(context);
    }

    public FaceDetectorProcessor(Context context, FaceDetectorOptions options) {
        super(context);
        Log.v(MANUAL_TESTING_LOG, "Face detector options: " + options);
        detector = FaceDetection.getClient(options);
        model = new FaceNetModel(context);
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<Face>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected void onSuccess(@NonNull List<Face> faces, @NonNull GraphicOverlay graphicOverlay, @NonNull Bitmap bitmap) {
        for (Face face : faces) {
            graphicOverlay.add(new FaceGraphic(graphicOverlay, face));
            logExtrasForTesting(face);
            if(bitmap!=null){
                Log.d(TAG,"Try to get face embedding!");
                float[] faceEncoding = model.getFaceEmbedding(bitmap,face.getBoundingBox(),false);
                Log.d(TAG,"Face Encoding = " + Arrays.toString(faceEncoding));
                float similarity = model.cosineSimilarity(faceEncoding,habibFace);
                Log.d(TAG,"Similarity = " + similarity);
            }else{
                Log.e(TAG,"Bitmap is null");
            }

        }
    }

    private static void logExtrasForTesting(Face face) {
        if (face != null) {
            Log.v(MANUAL_TESTING_LOG, "face bounding box: " + face.getBoundingBox().flattenToString());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle X: " + face.getHeadEulerAngleX());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle Y: " + face.getHeadEulerAngleY());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle Z: " + face.getHeadEulerAngleZ());

            // All landmarks
            int[] landMarkTypes =
                    new int[]{
                            FaceLandmark.MOUTH_BOTTOM,
                            FaceLandmark.MOUTH_RIGHT,
                            FaceLandmark.MOUTH_LEFT,
                            FaceLandmark.RIGHT_EYE,
                            FaceLandmark.LEFT_EYE,
                            FaceLandmark.RIGHT_EAR,
                            FaceLandmark.LEFT_EAR,
                            FaceLandmark.RIGHT_CHEEK,
                            FaceLandmark.LEFT_CHEEK,
                            FaceLandmark.NOSE_BASE
                    };
            String[] landMarkTypesStrings =
                    new String[]{
                            "MOUTH_BOTTOM",
                            "MOUTH_RIGHT",
                            "MOUTH_LEFT",
                            "RIGHT_EYE",
                            "LEFT_EYE",
                            "RIGHT_EAR",
                            "LEFT_EAR",
                            "RIGHT_CHEEK",
                            "LEFT_CHEEK",
                            "NOSE_BASE"
                    };
            for (int i = 0; i < landMarkTypes.length; i++) {
                FaceLandmark landmark = face.getLandmark(landMarkTypes[i]);
                if (landmark == null) {
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "No landmark of type: " + landMarkTypesStrings[i] + " has been detected");
                } else {
                    PointF landmarkPosition = landmark.getPosition();
                    String landmarkPositionStr =
                            String.format(Locale.US, "x: %f , y: %f", landmarkPosition.x, landmarkPosition.y);
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "Position for face landmark: "
                                    + landMarkTypesStrings[i]
                                    + " is :"
                                    + landmarkPositionStr);
                }
            }
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face left eye open probability: " + face.getLeftEyeOpenProbability());
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face right eye open probability: " + face.getRightEyeOpenProbability());
            Log.v(MANUAL_TESTING_LOG, "face smiling probability: " + face.getSmilingProbability());
            Log.v(MANUAL_TESTING_LOG, "face tracking id: " + face.getTrackingId());
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }


}
