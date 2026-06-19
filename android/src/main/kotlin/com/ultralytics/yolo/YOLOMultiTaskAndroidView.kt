// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.Surface
import android.widget.FrameLayout
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Runs up to three YOLO models simultaneously on a single camera stream.
 * Each Predictor instance receives raw landscape Bitmaps on a separate executor so all
 * models run concurrently without serialisation.
 */
class YOLOMultiTaskAndroidView(context: Context) : FrameLayout(context) {

    companion object {
        private const val TAG = "YOLOMultiTaskAndroidView"
        private val CLASS_NAMES = listOf("Móp/bẹp", "Vỡ/nứt", "Thủng/rách", "Trầy/xước")
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val previewView = PreviewView(context)

    // Each predictor gets its own single-thread executor so all run in parallel.
    private val detectExecutor:   ExecutorService = Executors.newSingleThreadExecutor()
    private val classifyExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val thirdExecutor:    ExecutorService = Executors.newSingleThreadExecutor()
    private val cameraExecutor:   ExecutorService = Executors.newSingleThreadExecutor()

    private var detectPredictor:   Predictor? = null
    private var classifyPredictor: Predictor? = null
    private var thirdPredictor:    Predictor? = null
    private var thirdTaskType:     String = "detect"
    /** Stable id for the third predictor's results so consumers can tell two detect models apart. */
    private var thirdModelId:      String = "detect2"

    /** Predictor slot — bookkeeping (busy flags, FPS) keys on this, never on the task name,
     *  so two detect models never clobber each other's state. */
    private enum class Slot { DETECT, CLASSIFY, THIRD }

    // One-frame-deep back-pressure: skip if previous frame is still being processed.
    private val detectBusy   = AtomicBoolean(false)
    private val classifyBusy = AtomicBoolean(false)
    private val thirdBusy    = AtomicBoolean(false)

    private var lifecycleOwner: LifecycleOwner? = null
    private var imageCaptureUseCase: ImageCapture? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null

    @Volatile private var isStopped = false
    private var pendingLensFacing: Int? = null

    /** Fired on the main thread for every inference result from any task. */
    var onMultiTaskStream: ((Map<String, Any>) -> Unit)? = null

    // Per-task FPS (calculated from wall-clock interval between results)
    private var detectLastMs:   Long = 0
    private var classifyLastMs: Long = 0
    private var thirdLastMs:    Long = 0
    private var detectFps:   Double = 0.0
    private var classifyFps: Double = 0.0
    private var thirdFps:    Double = 0.0

    // Camera FPS
    private var camFrameCount = 0
    private var camFpsWindowStart = System.currentTimeMillis()
    private var camFps = 0.0

    init {
        // COMPATIBLE forces a TextureView so Stack overlays stay visible inside a Flutter AndroidView.
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
        thirdModelPath: String? = null,
        thirdModelTask: String = "detect",
        thirdModelId: String = "detect2",
        useGpu: Boolean,
        detectConfidenceThreshold: Double,
        detectIouThreshold: Double,
        classifyConfidenceThreshold: Double,
        thirdConfidenceThreshold: Double,
        thirdIouThreshold: Double,
        lensFacing: Int,
        completion: () -> Unit
    ) {
        this.thirdTaskType = thirdModelTask
        this.thirdModelId = thirdModelId
        val totalModels = if (thirdModelPath != null) 3 else 2
        val loadedCount = AtomicInteger(0)

        fun tryDone() {
            if (loadedCount.incrementAndGet() == totalModels) {
                mainHandler.post {
                    initCamera(lensFacing)
                    completion()
                }
            }
        }

        detectExecutor.execute {
            try {
                val p = ObjectDetector(context, detectPath, emptyList(), useGpu)
                p.setConfidenceThreshold(detectConfidenceThreshold)
                p.setIouThreshold(detectIouThreshold)
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
                p.setConfidenceThreshold(classifyConfidenceThreshold)
                classifyPredictor = p
                Log.d(TAG, "✅ classify predictor loaded")
            } catch (e: Exception) {
                Log.e(TAG, "⚠️ classify load failed: ${e.message}")
            }
            tryDone()
        }

        if (thirdModelPath != null) {
            thirdExecutor.execute {
                try {
                    val p = createPredictor(thirdModelTask, thirdModelPath, useGpu)
                    if (p != null) {
                        p.setConfidenceThreshold(thirdConfidenceThreshold)
                        p.setIouThreshold(thirdIouThreshold)
                        thirdPredictor = p
                        Log.d(TAG, "✅ third predictor ($thirdModelTask) loaded")
                    } else {
                        Log.e(TAG, "⚠️ third predictor ($thirdModelTask) is null after load")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "⚠️ third load ($thirdModelTask) failed: ${e.message}")
                }
                tryDone()
            }
        }
    }

    private fun createPredictor(task: String, path: String, useGpu: Boolean): BasePredictor? {
        return when (task.lowercase()) {
            "classify" -> Classifier(context, path, emptyList(), useGpu)
            "segment"  -> Segmenter(context, path, emptyList(), useGpu)
            "pose"     -> PoseEstimator(context, path, emptyList(), useGpu)
            "obb"      -> ObbDetector(context, path, emptyList(), useGpu)
            else       -> ObjectDetector(context, path, emptyList(), useGpu)
        }
    }

    fun stopCamera() {
        isStopped = true
        if (Looper.myLooper() == Looper.getMainLooper()) {
            stopCameraInternal()
        } else {
            mainHandler.post { stopCameraInternal() }
        }
    }

    private fun stopCameraInternal() {
        camera?.cameraControl?.enableTorch(false)
        cameraProvider?.unbindAll()
        cameraProvider = null
        imageCaptureUseCase = null
        camera = null
    }

    /** Release all resources: camera, executors, and predictors. Safe to call on any thread. */
    fun release() {
        isStopped = true
        onMultiTaskStream = null
        if (Looper.myLooper() == Looper.getMainLooper()) {
            stopCameraInternal()
        } else {
            mainHandler.post { stopCameraInternal() }
        }
        detectExecutor.shutdownNow()
        classifyExecutor.shutdownNow()
        thirdExecutor.shutdownNow()
        cameraExecutor.shutdownNow()
        (detectPredictor as? BasePredictor)?.close()
        (classifyPredictor as? BasePredictor)?.close()
        (thirdPredictor as? BasePredictor)?.close()
        detectPredictor = null
        classifyPredictor = null
        thirdPredictor = null
    }

    fun initCamera(lensFacing: Int) {
        if (allPermissionsGranted()) {
            startCamera(lensFacing)
        } else {
            pendingLensFacing = lensFacing
            val activity = context as? Activity ?: run {
                Log.e(TAG, "Context is not an Activity; cannot request camera permission")
                return
            }
            ActivityCompat.requestPermissions(activity, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            val facing = pendingLensFacing ?: return
            if (allPermissionsGranted()) {
                pendingLensFacing = null
                startCamera(facing)
            } else {
                Log.w(TAG, "Camera permission denied")
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    /** Toggle the flash torch. Returns the requested state, or false if the device has no flash. */
    fun setTorchMode(enable: Boolean): Boolean {
        val cam = camera ?: return false
        if (!cam.cameraInfo.hasFlashUnit()) return false
        cam.cameraControl.enableTorch(enable)
        return enable
    }

    fun capturePhoto(callback: (ByteArray?) -> Unit) {
        val ic = imageCaptureUseCase ?: run { callback(null); return }
        ic.takePicture(ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
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

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()

            // Stream to the models at HD (1280x720) for efficient inference, while still
            // capturing FullHD (1920x1080) stills. CameraX lets each use-case request its
            // own resolution from the camera ISP. Sizes are expressed in the sensor's
            // natural (landscape) orientation.
            val hdSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()
            val fullHdSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1920, 1080), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(hdSelector)
                .setTargetRotation(Surface.ROTATION_0)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { proxy -> onFrame(proxy) } }

            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(fullHdSelector)
                .setTargetRotation(Surface.ROTATION_90)
                .build()

            provider.unbindAll()
            try {
                camera = provider.bindToLifecycle(owner, selector, preview, analysis, capture)
                imageCaptureUseCase = capture
            } catch (e: Exception) {
                Log.w(TAG, "3-use-case bind failed, retrying without ImageCapture: ${e.message}")
                imageCaptureUseCase = null
                try {
                    camera = provider.bindToLifecycle(owner, selector, preview, analysis)
                } catch (e2: Exception) {
                    Log.e(TAG, "Camera bind failed entirely: ${e2.message}")
                    camera = null
                }
            }
            preview.setSurfaceProvider(previewView.surfaceProvider)
        }, ContextCompat.getMainExecutor(context))
    }

    // endregion

    // region Frame processing

    private fun onFrame(imageProxy: ImageProxy) {
        if (isStopped) { imageProxy.close(); return }

        val now = System.currentTimeMillis()
        camFrameCount++
        val elapsed = now - camFpsWindowStart
        if (elapsed >= 500) {
            camFps = camFrameCount * 1000.0 / elapsed
            camFrameCount = 0
            camFpsWindowStart = now
        }

        val bitmap = ImageUtils.toBitmap(imageProxy) ?: run { imageProxy.close(); return }
        imageProxy.close()

        if (isStopped) { bitmap.recycle(); return }

        val w = bitmap.width
        val h = bitmap.height
        val camFpsNow = camFps

        dispatchDetect(bitmap, w, h, camFpsNow)
        dispatchClassify(bitmap, w, h, camFpsNow)
        dispatchThird(bitmap, w, h, camFpsNow)

        bitmap.recycle()
    }

    private fun dispatchDetect(src: Bitmap, w: Int, h: Int, camFpsNow: Double) {
        val p = detectPredictor ?: return
        if (!detectBusy.compareAndSet(false, true)) return
        val copy = src.copy(Bitmap.Config.ARGB_8888, false)
        detectExecutor.execute {
            try {
                val result = p.predict(copy, w, h, rotateForCamera = false, isLandscape = true)
                val data = buildTaskData(result, Slot.DETECT, camFpsNow)
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
                val data = buildTaskData(result, Slot.CLASSIFY, camFpsNow)
                mainHandler.post { onMultiTaskStream?.invoke(data) }
            } catch (e: Exception) {
                Log.e(TAG, "classify predict error: ${e.message}")
            } finally {
                copy.recycle()
                classifyBusy.set(false)
            }
        }
    }

    private fun dispatchThird(src: Bitmap, w: Int, h: Int, camFpsNow: Double) {
        val p = thirdPredictor ?: return
        if (!thirdBusy.compareAndSet(false, true)) return
        val copy = src.copy(Bitmap.Config.ARGB_8888, false)
        thirdExecutor.execute {
            try {
                val result = p.predict(copy, w, h, rotateForCamera = false, isLandscape = true)
                val data = buildTaskData(result, Slot.THIRD, camFpsNow)
                mainHandler.post { onMultiTaskStream?.invoke(data) }
            } catch (e: Exception) {
                Log.e(TAG, "third predict ($thirdTaskType) error: ${e.message}")
            } finally {
                copy.recycle()
                thirdBusy.set(false)
            }
        }
    }

    // endregion

    // region Stream data builder

    private fun buildTaskData(result: YOLOResult, slot: Slot, camFpsNow: Double): Map<String, Any> {
        val now = System.currentTimeMillis()
        val fps: Double
        val task: String
        val modelId: String
        when (slot) {
            Slot.DETECT -> {
                if (detectLastMs > 0) { val dt = now - detectLastMs; if (dt > 0) detectFps = 1000.0 / dt }
                detectLastMs = now; fps = detectFps
                task = "detect"; modelId = "detect"
            }
            Slot.CLASSIFY -> {
                if (classifyLastMs > 0) { val dt = now - classifyLastMs; if (dt > 0) classifyFps = 1000.0 / dt }
                classifyLastMs = now; fps = classifyFps
                task = "classify"; modelId = "classify"
            }
            Slot.THIRD -> {
                if (thirdLastMs > 0) { val dt = now - thirdLastMs; if (dt > 0) thirdFps = 1000.0 / dt }
                thirdLastMs = now; fps = thirdFps
                task = thirdTaskType; modelId = thirdModelId
            }
        }

        val base = mutableMapOf<String, Any>(
            "type" to task,
            "modelId" to modelId,
            "fps" to fps,
            "cameraFps" to camFpsNow,
            "processingTimeMs" to result.speed
        )

        when (task) {
            "classify" -> {
                val classMap = mutableMapOf<String, Any>(
                    "top1" to (result.probs?.top1Label ?: ""),
                    "top1Confidence" to (result.probs?.top1Conf?.toDouble() ?: 0.0)
                )
                result.probs?.let { probs ->
                    classMap["top5"] = (0 until minOf(probs.top5Labels.size, probs.top5Confs.size)).map { i ->
                        mapOf("name" to probs.top5Labels[i], "confidence" to probs.top5Confs[i].toDouble())
                    }
                }
                base["classification"] = classMap
            }
            else -> {
                // detect, segment, pose, obb — all have boxes.
                // The custom Vietnamese damage labels only apply to the primary detect model; any
                // other detect model reports its own class names via box.cls.
                val useCustomNames = modelId == "detect"
                val detections: List<Map<String, Any>> = result.boxes.take(50).map { box ->
                    val name = if (useCustomNames && box.index < CLASS_NAMES.size) CLASS_NAMES[box.index] else box.cls
                    mapOf(
                        "className" to name,
                        "confidence" to box.conf.toDouble(),
                        "normalizedBox" to mapOf(
                            "left"   to box.xywhn.left.toDouble(),
                            "top"    to box.xywhn.top.toDouble(),
                            "right"  to box.xywhn.right.toDouble(),
                            "bottom" to box.xywhn.bottom.toDouble()
                        )
                    )
                }
                base["detections"] = detections
            }
        }

        return base
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
        release()
    }
}
