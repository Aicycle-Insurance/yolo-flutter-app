// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

import 'core/yolo_model_resolver.dart';

/// Callback fired on the main thread for each inference result from one of the
/// two simultaneous YOLO tasks.
///
/// [data] contains:
/// - `"type"`: `"detect"` | `"classify"`
/// - `"fps"`: per-task FPS
/// - `"cameraFps"`: raw camera feed FPS
/// - `"processingTimeMs"`: CoreML inference time in ms
/// - `"detections"`: list of detection maps (detect task)
/// - `"classification"`: map with `top1`, `top1Confidence`, `top5` (classify task)
typedef MultiTaskStreamCallback = void Function(Map<String, dynamic> data);

/// Controller for [MultiTaskYOLOView]. Pass to the widget and call [capturePhoto]
/// to take a still JPEG from the live camera stream.
class MultiTaskYOLOController {
  MethodChannel? _channel;

  void _attach(MethodChannel channel) => _channel = channel;
  void _detach() => _channel = null;

  /// Capture a JPEG still from the current camera frame.
  /// Returns the JPEG bytes, or throws if the camera is not ready.
  Future<Uint8List> capturePhoto() async {
    final ch = _channel;
    if (ch == null) throw StateError('MultiTaskYOLOView is not attached');
    final result = await ch.invokeMethod<Uint8List>('capturePhoto');
    if (result == null) throw StateError('capturePhoto returned null');
    return result;
  }
}

/// A Flutter widget that hosts a native iOS view running detection and classification
/// simultaneously on a single camera stream — no JPEG round-trip,
/// no MethodChannel serialisation per frame.
///
/// Android is not yet implemented; on Android the widget renders an empty black box.
class MultiTaskYOLOView extends StatefulWidget {
  const MultiTaskYOLOView({
    super.key,
    required this.detectModelPath,
    required this.classifyModelPath,
    this.controller,
    this.onStreamingData,
    this.lensFacing = 'back',
    this.useGpu = true,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.7,
  });

  final String detectModelPath;
  final String classifyModelPath;
  final MultiTaskYOLOController? controller;
  final MultiTaskStreamCallback? onStreamingData;
  final String lensFacing;
  final bool useGpu;
  final double confidenceThreshold;
  final double iouThreshold;

  @override
  State<MultiTaskYOLOView> createState() => _MultiTaskYOLOViewState();
}

class _MultiTaskYOLOViewState extends State<MultiTaskYOLOView> {
  static int _nextId = 0;
  late final String _viewId;
  EventChannel? _eventChannel;
  MethodChannel? _methodChannel;

  // Resolved absolute paths (null = still resolving)
  String? _detectResolved;
  String? _classifyResolved;
  String? _resolutionError;

  @override
  void initState() {
    super.initState();
    _viewId = 'multi_task_${_nextId++}';
    _resolveModels();
  }

  Future<void> _resolveModels() async {
    try {
      final results = await Future.wait([
        YOLOModelResolver.preparePath(widget.detectModelPath),
        YOLOModelResolver.preparePath(widget.classifyModelPath),
      ]);
      if (!mounted) return;
      setState(() {
        _detectResolved = results[0];
        _classifyResolved = results[1];
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

    _eventChannel!.receiveBroadcastStream().listen((event) {
      if (event is Map && widget.onStreamingData != null) {
        widget.onStreamingData!(Map<String, dynamic>.from(event));
      }
    });
  }

  Map<String, dynamic> get _creationParams => {
    'viewId': _viewId,
    'detectModel': _detectResolved!,
    'classifyModel': _classifyResolved!,
    'lensFacing': widget.lensFacing,
    'useGpu': widget.useGpu,
    'confidenceThreshold': widget.confidenceThreshold,
    'iouThreshold': widget.iouThreshold,
  };

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

    if (_detectResolved == null || _classifyResolved == null) {
      // Still resolving (extracting zip from assets → Documents/files dir)
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
    widget.controller?._detach();
    _methodChannel?.invokeMethod('stop');
    super.dispose();
  }
}
