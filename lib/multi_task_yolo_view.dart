// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

import 'core/yolo_model_resolver.dart';

/// Callback fired on the main thread for each inference result from one of the
/// active YOLO tasks.
///
/// [data] contains:
/// - `"type"`: `"detect"` | `"classify"` | `"segment"`
/// - `"fps"`: per-task FPS
/// - `"cameraFps"`: raw camera feed FPS
/// - `"processingTimeMs"`: inference time in ms
/// - `"detections"`: list of detection maps (detect / segment tasks)
/// - `"classification"`: map with `top1`, `top1Confidence`, `top5` (classify task)
typedef MultiTaskStreamCallback = void Function(Map<String, dynamic> data);

/// Controller for [MultiTaskYOLOView]. Pass to the widget and call [capturePhoto]
/// to take a still JPEG from the live camera stream.
class MultiTaskYOLOController {
  MethodChannel? _channel;
  bool _torchEnabled = false;

  void _attach(MethodChannel channel) => _channel = channel;
  void _detach() => _channel = null;

  bool get isTorchEnabled => _torchEnabled;

  /// Capture a JPEG still from the current camera frame.
  /// Returns the JPEG bytes, or throws if the camera is not ready.
  Future<Uint8List> capturePhoto() async {
    final ch = _channel;
    if (ch == null) throw StateError('MultiTaskYOLOView is not attached');
    final result = await ch.invokeMethod<Uint8List>('capturePhoto');
    if (result == null) throw StateError('capturePhoto returned null');
    return result;
  }

  /// Turn the camera torch on ([enable] = true) or off ([enable] = false).
  /// Returns the actual torch state after the call — `false` when the device
  /// has no torch or the call is made before the view is attached.
  Future<bool> setTorch(bool enable) async {
    final ch = _channel;
    if (ch == null) return false;
    final result = await ch.invokeMethod<bool>('setTorch', {'enable': enable});
    _torchEnabled = result ?? false;
    return _torchEnabled;
  }

  Future<bool> toggleTorch() => setTorch(!_torchEnabled);

  /// Stops the camera and releases all CoreML model predictors from memory.
  /// Call this before removing [MultiTaskYOLOView] from the tree so GPU/ANE
  /// memory is freed immediately rather than waiting for a potentially-delayed deinit.
  Future<void> stop() async {
    final ch = _channel;
    if (ch == null) return;
    await ch.invokeMethod<void>('stop');
  }
}

/// A Flutter widget that runs detect, classify, and optionally segment simultaneously
/// on a single camera stream — no JPEG round-trip, no MethodChannel per frame.
///
/// Pass [segmentModelPath] to enable a third segmentation model running in parallel.
/// The [onStreamingData] callback fires for every result; check `data["type"]`
/// (`"detect"`, `"classify"`, or `"segment"`) to route results.
class MultiTaskYOLOView extends StatefulWidget {
  const MultiTaskYOLOView({
    super.key,
    required this.detectModelPath,
    required this.classifyModelPath,
    this.segmentModelPath,
    this.controller,
    this.onStreamingData,
    this.lensFacing = 'back',
    this.useGpu = true,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.7,
    this.detectConfidenceThreshold,
    this.detectIouThreshold,
    this.classifyConfidenceThreshold,
    this.segmentConfidenceThreshold,
    this.segmentIouThreshold,
  });

  final String detectModelPath;
  final String classifyModelPath;

  /// Optional segmentation model. When non-null, runs in parallel with detect + classify.
  /// Results arrive via [onStreamingData] with `data["type"] == "segment"`.
  final String? segmentModelPath;

  final MultiTaskYOLOController? controller;
  final MultiTaskStreamCallback? onStreamingData;
  final String lensFacing;
  final bool useGpu;

  /// Default confidence threshold, applied to any task without a per-model override.
  final double confidenceThreshold;

  /// Default IoU (NMS) threshold, applied to any task without a per-model override.
  final double iouThreshold;

  /// Per-model overrides. When null, the corresponding model falls back to the
  /// global [confidenceThreshold] / [iouThreshold]. Classification only uses a
  /// confidence threshold (IoU does not apply to it).
  final double? detectConfidenceThreshold;
  final double? detectIouThreshold;
  final double? classifyConfidenceThreshold;
  final double? segmentConfidenceThreshold;
  final double? segmentIouThreshold;

  @override
  State<MultiTaskYOLOView> createState() => _MultiTaskYOLOViewState();
}

class _MultiTaskYOLOViewState extends State<MultiTaskYOLOView> {
  static int _nextId = 0;
  late final String _viewId;
  EventChannel? _eventChannel;
  MethodChannel? _methodChannel;
  StreamSubscription<dynamic>? _eventSubscription;

  // Resolved absolute paths (null = still resolving)
  String? _detectResolved;
  String? _classifyResolved;
  String? _segmentResolved;
  String? _resolutionError;

  @override
  void initState() {
    super.initState();
    _viewId = 'multi_task_${_nextId++}';
    _resolveModels();
  }

  Future<void> _resolveModels() async {
    try {
      final futures = [
        YOLOModelResolver.preparePath(widget.detectModelPath),
        YOLOModelResolver.preparePath(widget.classifyModelPath),
        if (widget.segmentModelPath != null)
          YOLOModelResolver.preparePath(widget.segmentModelPath!),
      ];
      final results = await Future.wait(futures);
      if (!mounted) return;
      setState(() {
        _detectResolved = results[0];
        _classifyResolved = results[1];
        if (widget.segmentModelPath != null) _segmentResolved = results[2];
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _resolutionError = e.toString());
    }
  }

  void _onPlatformViewCreated(int id) {
    _eventChannel = EventChannel(
      'com.ultralytics.yolo/multiTaskResults_$_viewId',
    );
    _methodChannel = MethodChannel(
      'com.ultralytics.yolo/multiTaskControl_$_viewId',
    );

    widget.controller?._attach(_methodChannel!);

    _eventSubscription?.cancel();
    _eventSubscription = _eventChannel!.receiveBroadcastStream().listen(
      (event) {
        if (event is Map && widget.onStreamingData != null) {
          widget.onStreamingData!(Map<String, dynamic>.from(event));
        }
      },
      onError: (Object error) {
        // ignore stream errors to prevent unhandled exceptions
      },
      cancelOnError: false,
    );
  }

  Map<String, dynamic> get _creationParams {
    final params = <String, dynamic>{
      'viewId': _viewId,
      'detectModel': _detectResolved!,
      'classifyModel': _classifyResolved!,
      'lensFacing': widget.lensFacing,
      'useGpu': widget.useGpu,
      'confidenceThreshold': widget.confidenceThreshold,
      'iouThreshold': widget.iouThreshold,
      'detectConfidenceThreshold':
          widget.detectConfidenceThreshold ?? widget.confidenceThreshold,
      'detectIouThreshold': widget.detectIouThreshold ?? widget.iouThreshold,
      'classifyConfidenceThreshold':
          widget.classifyConfidenceThreshold ?? widget.confidenceThreshold,
    };
    if (_segmentResolved != null) {
      params['segmentModel'] = _segmentResolved!;
      params['segmentConfidenceThreshold'] =
          widget.segmentConfidenceThreshold ?? widget.confidenceThreshold;
      params['segmentIouThreshold'] =
          widget.segmentIouThreshold ?? widget.iouThreshold;
    }
    return params;
  }

  @override
  Widget build(BuildContext context) {
    if (!Platform.isIOS && !Platform.isAndroid) {
      return const ColoredBox(color: Color(0xFF000000));
    }

    if (_resolutionError != null) {
      return ColoredBox(
        color: const Color(0xFF000000),
        child: Center(
          child: Text(
            'Model error: $_resolutionError',
            style: const TextStyle(color: Color(0xFFFF4444), fontSize: 12),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    final segmentPending = widget.segmentModelPath != null && _segmentResolved == null;
    if (_detectResolved == null || _classifyResolved == null || segmentPending) {
      // Still resolving model paths
      return const ColoredBox(color: Color(0xFF000000));
    }

    if (Platform.isIOS) {
      return UiKitView(
        viewType: 'com.ultralytics.yolo/YOLOMultiTaskPlatformView',
        onPlatformViewCreated: _onPlatformViewCreated,
        creationParams: _creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        gestureRecognizers: const <Factory<OneSequenceGestureRecognizer>>{},
      );
    }

    // Android
    return AndroidView(
      viewType: 'com.ultralytics.yolo/YOLOMultiTaskPlatformView',
      onPlatformViewCreated: _onPlatformViewCreated,
      creationParams: _creationParams,
      creationParamsCodec: const StandardMessageCodec(),
      gestureRecognizers: const <Factory<OneSequenceGestureRecognizer>>{},
    );
  }

  @override
  void dispose() {
    _eventSubscription?.cancel();
    _eventSubscription = null;
    widget.controller?._detach();
    _methodChannel?.invokeMethod('stop');
    super.dispose();
  }
}
