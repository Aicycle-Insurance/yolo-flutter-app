// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:aicycle_yolo/ultralytics_yolo.dart';

/// Multi-task inference screen — two detection models and a classification model run
/// simultaneously on a single camera stream using one native [MultiTaskYOLOView].
/// All three models share the same AVCaptureSession; raw camera frames are dispatched
/// to each predictor on its own background queue with no JPEG overhead. Results are
/// routed by `data["modelId"]` (`"detect"`, `"detect2"`, `"classify"`).
class MultiTaskScreen extends StatefulWidget {
  const MultiTaskScreen({super.key});

  @override
  State<MultiTaskScreen> createState() => _MultiTaskScreenState();
}

class _MultiTaskScreenState extends State<MultiTaskScreen> {
  final _controller = MultiTaskYOLOController();

  // Per-model latest results
  List<Map<String, dynamic>> _detections = [];
  List<Map<String, dynamic>> _detections2 = [];
  Map<String, dynamic>? _classification;

  // Per-model performance
  double _detectMs = 0, _detectFps = 0;
  double _detect2Ms = 0, _detect2Fps = 0;
  double _classifyMs = 0, _classifyFps = 0;
  double _cameraFps = 0;

  bool _capturing = false;

  Future<void> _capturePhoto() async {
    if (_capturing) return;
    setState(() => _capturing = true);
    try {
      final bytes = await _controller.capturePhoto();
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (context) => _PhotoPreviewDialog(bytes: bytes),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Capture failed: $e')));
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  void _onStreamingData(Map<String, dynamic> data) {
    if (!mounted) return;
    // Route on modelId, not type — both detection models report type == 'detect'.
    final modelId = data['modelId'] as String?;
    final fps = (data['fps'] as num?)?.toDouble() ?? 0;
    final ms = (data['processingTimeMs'] as num?)?.toDouble() ?? 0;
    final camFps = (data['cameraFps'] as num?)?.toDouble() ?? 0;

    List<Map<String, dynamic>> parseDetections() {
      final dList = data['detections'];
      return dList is List
          ? dList.whereType<Map>().map(Map<String, dynamic>.from).toList()
          : [];
    }

    setState(() {
      if (camFps > 0) _cameraFps = camFps;
      switch (modelId) {
        case 'detect':
          _detectFps = fps;
          _detectMs = ms;
          _detections = parseDetections();
        case 'detect2':
          _detect2Fps = fps;
          _detect2Ms = ms;
          _detections2 = parseDetections();
        case 'classify':
          _classifyFps = fps;
          _classifyMs = ms;
          final raw = data['classification'];
          _classification = raw is Map ? Map<String, dynamic>.from(raw) : null;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Native multi-task camera view (fills screen).
          // iOS  → .mlpackage.zip (extracted by YOLOModelResolver)
          // Android → .tflite
          // The native layer delivers landscape pixel buffers to the ML models
          // (videoOrientation = .landscapeLeft) while the preview layer stays
          // portrait so the viewfinder looks correct to the user.
          MultiTaskYOLOView(
            detectModelPath: Platform.isAndroid
                ? 'assets/models/car_damage_detection_mobile_26m.tflite'
                : 'assets/models/car_damage_detection_mobile_26m.mlpackage.zip',
            classifyModelPath: Platform.isAndroid
                ? 'assets/models/car_corner_classification_yolo26n-cls_int8.tflite'
                : 'assets/models/car_corner_classification.mlpackage.zip',
            // Second detection model running in parallel. Reuses the damage model here
            // purely to demonstrate two detectors at once — swap in any detect model.
            secondDetectModelPath: Platform.isAndroid
                ? 'assets/models/car_damage_detection_mobile_26m.tflite'
                : 'assets/models/car_damage_detection_mobile_26m.mlpackage.zip',
            controller: _controller,
            onStreamingData: _onStreamingData,
          ),

          // Back button
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: IconButton(
                icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white),
                onPressed: () => Navigator.pop(context),
              ),
            ),
          ),

          // Performance overlay (top-right)
          SafeArea(
            child: Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: _PerformanceOverlay(
                  cameraFps: _cameraFps,
                  detectMs: _detectMs,
                  detectFps: _detectFps,
                  detect2Ms: _detect2Ms,
                  detect2Fps: _detect2Fps,
                  classifyMs: _classifyMs,
                  classifyFps: _classifyFps,
                ),
              ),
            ),
          ),

          // Results panel (bottom)
          Align(
            alignment: Alignment.bottomCenter,
            child: _ResultsPanel(
              detections: _detections,
              detections2: _detections2,
              classification: _classification,
            ),
          ),

          // Shutter button (above the results panel)
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 146),
              child: GestureDetector(
                onTap: _capturePhoto,
                child: Container(
                  width: 64,
                  height: 64,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white.withValues(alpha: 0.25),
                    border: Border.all(color: Colors.white, width: 3),
                  ),
                  child: _capturing
                      ? const Padding(
                          padding: EdgeInsets.all(18),
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2.5,
                          ),
                        )
                      : const Icon(
                          Icons.camera_alt,
                          color: Colors.white,
                          size: 28,
                        ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// MARK: - Photo preview

/// Shows the captured JPEG with [Image.memory] — which ignores EXIF rotation,
/// so the image only displays upright because the native layer normalizes
/// the pixel data before returning it. The photo is landscape (wide) by design.
class _PhotoPreviewDialog extends StatelessWidget {
  const _PhotoPreviewDialog({required this.bytes});

  final Uint8List bytes;

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.black,
      insetPadding: const EdgeInsets.all(16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.memory(bytes, fit: BoxFit.contain),
          ),
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '${(bytes.length / 1024).toStringAsFixed(0)} KB',
                  style: const TextStyle(color: Colors.white54, fontSize: 12),
                ),
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Đóng'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// MARK: - Performance overlay

class _PerformanceOverlay extends StatelessWidget {
  const _PerformanceOverlay({
    required this.cameraFps,
    required this.detectMs,
    required this.detectFps,
    required this.detect2Ms,
    required this.detect2Fps,
    required this.classifyMs,
    required this.classifyFps,
  });

  final double cameraFps;
  final double detectMs, detectFps;
  final double detect2Ms, detect2Fps;
  final double classifyMs, classifyFps;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          // Camera FPS chip
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(
              color: const Color(0xFF374151),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              'CAM  ${cameraFps.toStringAsFixed(1)} fps',
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 11,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
          const SizedBox(height: 6),
          _LatencyBar(
            label: 'DET',
            color: const Color(0xFF3B82F6),
            ms: detectMs,
            fps: detectFps,
          ),
          const SizedBox(height: 4),
          _LatencyBar(
            label: 'DET2',
            color: const Color(0xFF22D3EE),
            ms: detect2Ms,
            fps: detect2Fps,
          ),
          const SizedBox(height: 4),
          _LatencyBar(
            label: 'CLS',
            color: const Color(0xFFF59E0B),
            ms: classifyMs,
            fps: classifyFps,
          ),
        ],
      ),
    );
  }
}

class _LatencyBar extends StatelessWidget {
  const _LatencyBar({
    required this.label,
    required this.color,
    required this.ms,
    required this.fps,
  });

  final String label;
  final Color color;
  final double ms;
  final double fps;

  @override
  Widget build(BuildContext context) {
    final barWidth = (ms / 200).clamp(0.0, 1.0);
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: 28,
          child: Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 10,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        Container(
          width: 80,
          height: 6,
          decoration: BoxDecoration(
            color: Colors.white12,
            borderRadius: BorderRadius.circular(3),
          ),
          child: FractionallySizedBox(
            widthFactor: barWidth,
            alignment: Alignment.centerLeft,
            child: Container(
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),
        ),
        const SizedBox(width: 6),
        SizedBox(
          width: 90,
          child: Text(
            '${ms.toStringAsFixed(0)}ms  ${fps.toStringAsFixed(1)}fps',
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 10,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
        ),
      ],
    );
  }
}

// MARK: - Results panel

class _ResultsPanel extends StatelessWidget {
  const _ResultsPanel({
    required this.detections,
    required this.detections2,
    required this.classification,
  });

  final List<Map<String, dynamic>> detections;
  final List<Map<String, dynamic>> detections2;
  final Map<String, dynamic>? classification;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 130,
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.72),
        border: const Border(top: BorderSide(color: Colors.white12)),
      ),
      child: Row(
        children: [
          _TaskColumn(
            title: 'DETECT',
            color: const Color(0xFF3B82F6),
            items: _detectionItems(detections),
          ),
          _TaskColumn(
            title: 'DETECT 2',
            color: const Color(0xFF22D3EE),
            items: _detectionItems(detections2),
          ),
          _TaskColumn(
            title: 'CLASSIFY',
            color: const Color(0xFFF59E0B),
            items: _classifyItems(),
          ),
        ],
      ),
    );
  }

  List<String> _detectionItems(List<Map<String, dynamic>> dets) {
    return dets.take(4).map((d) {
      final name = d['className'] as String? ?? '?';
      final conf = ((d['confidence'] as num?)?.toDouble() ?? 0) * 100;
      return '$name ${conf.toStringAsFixed(0)}%';
    }).toList();
  }

  List<String> _classifyItems() {
    if (classification == null) return [];
    final rawTop5 = classification!['top5'];
    if (rawTop5 is! List) return [];
    return rawTop5.take(4).map((e) {
      final m = e is Map ? e : <Object?, Object?>{};
      final name = m['name']?.toString() ?? '?';
      final conf = ((m['confidence'] as num?)?.toDouble() ?? 0) * 100;
      return '$name ${conf.toStringAsFixed(0)}%';
    }).toList();
  }
}

class _TaskColumn extends StatelessWidget {
  const _TaskColumn({
    required this.title,
    required this.color,
    required this.items,
  });

  final String title;
  final Color color;
  final List<String> items;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: TextStyle(
                color: color,
                fontSize: 10,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.8,
              ),
            ),
            const SizedBox(height: 4),
            ...items.map(
              (item) => Padding(
                padding: const EdgeInsets.only(bottom: 2),
                child: Text(
                  item,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: Colors.white70, fontSize: 11),
                ),
              ),
            ),
            if (items.isEmpty)
              const Text(
                '—',
                style: TextStyle(color: Colors.white30, fontSize: 11),
              ),
          ],
        ),
      ),
    );
  }
}
