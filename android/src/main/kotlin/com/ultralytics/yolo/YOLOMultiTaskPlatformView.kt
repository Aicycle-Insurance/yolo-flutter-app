// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.app.Activity
import android.content.Context
import android.util.Log
import android.view.View
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory

/** Flutter platform-view wrapper for YOLOMultiTaskAndroidView. */
class YOLOMultiTaskPlatformView(
    private val context: Context,
    private val viewId: Int,
    args: Any?,
    messenger: BinaryMessenger,
    private val onDispose: ((Int) -> Unit)? = null
) : PlatformView {

    private val TAG = "YOLOMultiTaskPlatformView"
    private val multiTaskView = YOLOMultiTaskAndroidView(context)
    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())

    @Volatile private var eventSink: EventChannel.EventSink? = null
    private val eventChannel: EventChannel
    private val methodChannel: MethodChannel

    init {
        val params = args as? Map<*, *>
        val idStr = params?.get("viewId") as? String ?: viewId.toString()

        // Channels must match the names in lib/multi_task_yolo_view.dart.
        eventChannel = EventChannel(messenger, "com.ultralytics.yolo/multiTaskResults_$idStr")
        eventChannel.setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    eventSink = events
                }
                override fun onCancel(arguments: Any?) {
                    eventSink = null
                }
            })

        methodChannel = MethodChannel(messenger, "com.ultralytics.yolo/multiTaskControl_$idStr")
        methodChannel.setMethodCallHandler { call, result ->
                when (call.method) {
                    "stop" -> {
                        multiTaskView.stopCamera()
                        result.success(null)
                    }
                    "capturePhoto" -> {
                        multiTaskView.capturePhoto { bytes ->
                            mainHandler.post {
                                if (bytes != null) result.success(bytes)
                                else result.error("capture_failed", "Failed to capture photo", null)
                            }
                        }
                    }
                    "setTorch" -> {
                        val enable = (call.arguments as? Map<*, *>)?.get("enable") as? Boolean
                            ?: run { result.error("bad_args", "enable (bool) is required", null); return@setMethodCallHandler }
                        result.success(multiTaskView.setTorchMode(enable))
                    }
                    else -> result.notImplemented()
                }
            }

        // Forward inference results to Flutter via the EventChannel sink.
        multiTaskView.onMultiTaskStream = { data ->
            mainHandler.post {
                try { eventSink?.success(data) } catch (e: Exception) {
                    Log.e(TAG, "EventSink error: ${e.message}")
                }
            }
        }

        // Provide lifecycle so CameraX can bind use-cases.
        if (context is LifecycleOwner) {
            multiTaskView.onLifecycleOwnerAvailable(context)
        }

        val detectPath = params?.get("detectModel") as? String
        val classifyPath = params?.get("classifyModel") as? String
        if (detectPath != null && classifyPath != null) {
            val useGpu = params["useGpu"] as? Boolean ?: true
            val lensFacing = if ((params["lensFacing"] as? String) == "front")
                androidx.camera.core.CameraSelector.LENS_FACING_FRONT
            else
                androidx.camera.core.CameraSelector.LENS_FACING_BACK
            val confidence = params["confidenceThreshold"] as? Double ?: 0.25
            val iou = params["iouThreshold"] as? Double ?: 0.7
            val segmentModel = params["segmentModel"] as? String

            multiTaskView.loadModels(
                detectPath = detectPath,
                classifyPath = classifyPath,
                thirdModelPath = segmentModel,
                thirdModelTask = "segment",
                useGpu = useGpu,
                confidenceThreshold = confidence,
                iouThreshold = iou,
                lensFacing = lensFacing,
                completion = { Log.d(TAG, "Models loaded, camera started") }
            )
        } else {
            Log.e(TAG, "Missing detectModel or classifyModel in creation params")
        }
    }

    override fun getView(): View = multiTaskView

    fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        multiTaskView.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    override fun dispose() {
        onDispose?.invoke(viewId)
        multiTaskView.release()
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        eventSink = null
    }
}

// MARK: - Factory

class YOLOMultiTaskPlatformViewFactory(
    private val messenger: BinaryMessenger
) : PlatformViewFactory(StandardMessageCodec.INSTANCE) {

    private var activity: Activity? = null
    val activeViews = mutableMapOf<Int, YOLOMultiTaskPlatformView>()

    fun setActivity(activity: Activity?) {
        this.activity = activity
    }

    override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
        val view = YOLOMultiTaskPlatformView(activity ?: context, viewId, args, messenger) { id ->
            activeViews.remove(id)
        }
        activeViews[viewId] = view
        return view
    }
}
