// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Surface
import android.widget.FrameLayout
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Runs detect and classify simultaneously on a single camera stream.
 * Two Predictor instances each receive raw landscape Bitmaps on separate executors so both
 * run concurrently without serialisation. The Flutter app locks orientation to portrait;
 * we hard-code ImageAnalysis targetRotation = ROTATION_0 so rotationDegrees=90 always, and
 * pass the bitmap to predictors with rotateForCamera=false — the raw sensor bitmap is already
 * landscape (wide), which is the orientation the models expect.
 */
class YOLOMultiTaskAndroidView(context: Context) : FrameLayout(context) {

    companion object {
        private const val TAG = "YOLOMultiTaskAndroidView"
        private val CLASS_NAMES = listOf("Móp/bẹp", "Vỡ/nứt", "Thủng/rách", "Trầy/xước")
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val previewView = PreviewView(context)

    // Each predictor gets its own single-thread executor so both run in parallel.
    private val detectExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val classifyExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var detectPredictor: Predictor? = null
    private var classifyPredictor: Predictor? = null

    // One-frame-deep back-pressure: skip if previous frame is still being processed.
    private val detectBusy = AtomicBoolean(false)
    private val classifyBusy = AtomicBoolean(false)

    private var lifecycleOwner: LifecycleOwner? = null
    private var imageCaptureUseCase: ImageCapture? = null
    private var cameraProvider: ProcessCameraProvider? = null

    @Volatile private var isStopped = false

    /** Fired on the main thread for every inference result from either task. */
    var onMultiTaskStream: ((Map<String, Any>) -> Unit)? = null

    // Per-task FPS (calculated from wall-clock interval between results)
    private var detectLastMs: Long = 0
    private var classifyLastMs: Long = 0
    private var detectFps: Double = 0.0
    private var classifyFps: Double = 0.0

    // Camera FPS
    private var camFrameCount = 0
    private var camFpsWindowStart = System.currentTimeMillis()
    private var camFps = 0.0

    init {
        // COMPATIBLE forces a TextureView instead of a SurfaceView. Inside a Flutter
        // AndroidView (virtual display) a SurfaceView punches through the Flutter UI and
        // covers every widget drawn on top of the platform view; a TextureView composites
        // normally so Stack overlays (buttons, panels) stay visible.
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        addView(previewView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
    }

    // region Public API

    fun onLifecycleOwnerAvailable(owner: LifecycleOwner) {
        lifecycleOwner = owner
    }

    fun loadModels(
        detectPath: String,
        classifyPath: String,
        useGpu: Boolean,
        confidenceThreshold: Double,
        iouThreshold: Double,
        lensFacing: Int,
        completion: () -> Unit
    ) {
        val loadedCount = AtomicInteger(0)

        fun tryDone() {
            if (loadedCount.incrementAndGet() == 2) {
                mainHandler.post {
                    startCamera(lensFacing)
                    completion()
                }
            }
        }

        detectExecutor.execute {
            try {
                val p = ObjectDetector(context, detectPath, emptyList(), useGpu)
                p.setConfidenceThreshold(confidenceThreshold)
                p.setIouThreshold(iouThreshold)
                detectPredictor = p
                Log.d(TAG, "✅ detect predictor loaded")
            } catch (e: Exception) {
                Log.e(TAG, "⚠️ detect load failed: ${e.message}")
            }
            tryDone()
        }

        classifyExecutor.execute {
            try {
                val p = Classifier(context, classifyPath, emptyList(), useGpu)
                p.setConfidenceThreshold(confidenceThreshold)
                classifyPredictor = p
                Log.d(TAG, "✅ classify predictor loaded")
            } catch (e: Exception) {
                Log.e(TAG, "⚠️ classify load failed: ${e.message}")
            }
            tryDone()
        }
    }

    fun stopCamera() {
        isStopped = true
        // Actually release the camera — unbinding must happen on the main thread.
        mainHandler.post {
            cameraProvider?.unbindAll()
            cameraProvider = null
            imageCaptureUseCase = null
        }
    }

    fun capturePhoto(callback: (ByteArray?) -> Unit) {
        val ic = imageCaptureUseCase ?: run { callback(null); return }
        ic.takePicture(ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    // ImageCapture targetRotation = ROTATION_90 → rotationDegrees=0 for standard
                    // back camera (sensorOrientation=90). normalizeJpeg is a no-op; JPEG is landscape.
                    val rotationDegrees = image.imageInfo.rotationDegrees
                    val plane = image.planes[0]
                    val buf = plane.buffer
                    val raw = ByteArray(buf.remaining()).also { buf.get(it) }
                    val jpeg = if (image.format == android.graphics.ImageFormat.JPEG) raw else {
                        val bmp = BitmapFactory.decodeByteArray(raw, 0, raw.size)
                        ByteArrayOutputStream().also { out ->
                            bmp?.compress(Bitmap.CompressFormat.JPEG, 92, out)
                            bmp?.recycle()
                        }.toByteArray()
                    }
                    callback(normalizeJpeg(jpeg, rotationDegrees))
                } catch (e: Exception) {
                    Log.e(TAG, "capturePhoto processing failed: ${e.message}")
                    callback(null)
                } finally {
                    image.close()
                }
            }

            override fun onError(e: ImageCaptureException) {
                Log.e(TAG, "capturePhoto error: ${e.message}")
                callback(null)
            }
        })
    }

    // endregion

    // region Camera setup

    private fun startCamera(lensFacing: Int) {
        val owner = lifecycleOwner ?: return
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (isStopped) return@addListener
            val provider = runCatching { future.get() }.getOrNull() ?: return@addListener
            cameraProvider = provider

            val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Preview: standard CameraX PreviewView — handles display rotation automatically
            // so the viewfinder looks correct to the user regardless of how they hold the device.
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()

            // ImageAnalysis: hard-coded ROTATION_0 (portrait target) so rotationDegrees=90 always.
            // We do NOT rotate the bitmap before inference (rotateForCamera=false) — the raw sensor
            // frame is landscape, which is the orientation we want the models to receive.
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(Surface.ROTATION_0)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { proxy -> onFrame(proxy) } }

            // ImageCapture: ROTATION_90 target → rotationDegrees=0 for standard back camera
            // (sensorOrientation=90). Captured JPEG is landscape without needing EXIF rotation.
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(Surface.ROTATION_90)
                .build()

            provider.unbindAll()
            try {
                provider.bindToLifecycle(owner, selector, preview, analysis, capture)
                imageCaptureUseCase = capture
            } catch (e: Exception) {
                Log.w(TAG, "3-use-case bind failed, retrying without ImageCapture: ${e.message}")
                imageCaptureUseCase = null
                try {
                    provider.bindToLifecycle(owner, selector, preview, analysis)
                } catch (e2: Exception) {
                    Log.e(TAG, "Camera bind failed entirely: ${e2.message}")
                }
            }
            preview.setSurfaceProvider(previewView.surfaceProvider)
        }, ContextCompat.getMainExecutor(context))
    }

    // endregion

    // region Frame processing

    private fun onFrame(imageProxy: ImageProxy) {
        if (isStopped) { imageProxy.close(); return }

        // Track camera FPS
        val now = System.currentTimeMillis()
        camFrameCount++
        val elapsed = now - camFpsWindowStart
        if (elapsed >= 500) {
            camFps = camFrameCount * 1000.0 / elapsed
            camFrameCount = 0
            camFpsWindowStart = now
        }

        // toBitmap copies pixels into an independent Bitmap — safe to close imageProxy immediately.
        val bitmap = ImageUtils.toBitmap(imageProxy) ?: run { imageProxy.close(); return }
        imageProxy.close()

        if (isStopped) { bitmap.recycle(); return }

        val w = bitmap.width      // landscape width  (e.g. 1280)
        val h = bitmap.height     // landscape height (e.g.  720)
        val camFpsNow = camFps

        dispatchDetect(bitmap, w, h, camFpsNow)
        dispatchClassify(bitmap, w, h, camFpsNow)

        bitmap.recycle()
    }

    private fun dispatchDetect(src: Bitmap, w: Int, h: Int, camFpsNow: Double) {
        val p = detectPredictor ?: return
        if (!detectBusy.compareAndSet(false, true)) return
        val copy = src.copy(Bitmap.Config.ARGB_8888, false)
        detectExecutor.execute {
            try {
                val result = p.predict(copy, w, h, rotateForCamera = false, isLandscape = true)
                val data = buildDetectData(result, camFpsNow)
                mainHandler.post { onMultiTaskStream?.invoke(data) }
            } catch (e: Exception) {
                Log.e(TAG, "detect predict error: ${e.message}")
            } finally {
                copy.recycle()
                detectBusy.set(false)
            }
        }
    }

    private fun dispatchClassify(src: Bitmap, w: Int, h: Int, camFpsNow: Double) {
        val p = classifyPredictor ?: return
        if (!classifyBusy.compareAndSet(false, true)) return
        val copy = src.copy(Bitmap.Config.ARGB_8888, false)
        classifyExecutor.execute {
            try {
                val result = p.predict(copy, w, h, rotateForCamera = false, isLandscape = true)
                val data = buildClassifyData(result, camFpsNow)
                mainHandler.post { onMultiTaskStream?.invoke(data) }
            } catch (e: Exception) {
                Log.e(TAG, "classify predict error: ${e.message}")
            } finally {
                copy.recycle()
                classifyBusy.set(false)
            }
        }
    }

    // endregion

    // region Stream data builders

    private fun buildDetectData(result: YOLOResult, camFpsNow: Double): Map<String, Any> {
        val now = System.currentTimeMillis()
        if (detectLastMs > 0) {
            val dt = now - detectLastMs
            if (dt > 0) detectFps = 1000.0 / dt
        }
        detectLastMs = now

        val detections: List<Map<String, Any>> = result.boxes.take(50).map { box ->
            val name = if (box.index < CLASS_NAMES.size) CLASS_NAMES[box.index] else box.cls
            mapOf(
                "className" to name,
                "confidence" to box.conf.toDouble(),
                "normalizedBox" to mapOf(
                    "left" to box.xywhn.left.toDouble(),
                    "top" to box.xywhn.top.toDouble(),
                    "right" to box.xywhn.right.toDouble(),
                    "bottom" to box.xywhn.bottom.toDouble()
                )
            )
        }

        return mapOf(
            "type" to "detect",
            "fps" to detectFps,
            "cameraFps" to camFpsNow,
            // YOLOResult.speed is already in milliseconds on Android (FrameTiming.speedMs).
            "processingTimeMs" to result.speed,
            "detections" to detections
        )
    }

    private fun buildClassifyData(result: YOLOResult, camFpsNow: Double): Map<String, Any> {
        val now = System.currentTimeMillis()
        if (classifyLastMs > 0) {
            val dt = now - classifyLastMs
            if (dt > 0) classifyFps = 1000.0 / dt
        }
        classifyLastMs = now

        val classMap = mutableMapOf<String, Any>(
            "top1" to (result.probs?.top1Label ?: ""),
            "top1Confidence" to (result.probs?.top1Conf?.toDouble() ?: 0.0)
        )
        result.probs?.let { probs ->
            classMap["top5"] = (0 until minOf(probs.top5Labels.size, probs.top5Confs.size)).map { i ->
                mapOf("name" to probs.top5Labels[i], "confidence" to probs.top5Confs[i].toDouble())
            }
        }

        return mapOf(
            "type" to "classify",
            "fps" to classifyFps,
            "cameraFps" to camFpsNow,
            // YOLOResult.speed is already in milliseconds on Android (FrameTiming.speedMs).
            "processingTimeMs" to result.speed,
            "classification" to classMap
        )
    }

    // endregion

    // region Photo normalization

    /** Rotate JPEG bytes so pixel data is upright. Flutter's Image.memory() ignores EXIF. */
    private fun normalizeJpeg(bytes: ByteArray, rotationDegrees: Int): ByteArray {
        if (rotationDegrees == 0) return bytes
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return bytes
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        bmp.recycle()
        return ByteArrayOutputStream().also { out ->
            rotated.compress(Bitmap.CompressFormat.JPEG, 92, out)
            rotated.recycle()
        }.toByteArray()
    }

    // endregion

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        stopCamera()
        detectExecutor.shutdownNow()
        classifyExecutor.shutdownNow()
        cameraExecutor.shutdownNow()
        (detectPredictor as? BasePredictor)?.close()
        (classifyPredictor as? BasePredictor)?.close()
        detectPredictor = null
        classifyPredictor = null
    }
}
