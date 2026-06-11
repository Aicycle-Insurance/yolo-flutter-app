// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';
import 'package:ultralytics_yolo_example/presentation/screens/multi_task_screen.dart'
    show MultiTaskScreen;
import 'package:ultralytics_yolo_example/presentation/screens/single_image_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Lock UI to portrait regardless of device rotation lock setting.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const YOLOExampleApp());
}

class YOLOExampleApp extends StatelessWidget {
  const YOLOExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ultralytics YOLO',
      themeMode: ThemeMode.dark,
      theme: ThemeData.dark(useMaterial3: true),
      initialRoute: '/multi-task',
      routes: {
        '/': (_) => const CameraInferenceScreen(),
        '/single': (_) => const SingleImageScreen(),
        '/multi-task': (_) => const MultiTaskScreen(),
      },
    );
  }
}
