// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  SwiftYOLOMultiTaskPlatformView — Flutter platform-view wrapper for YOLOMultiTaskView.
//  Registers an EventChannel that streams per-task inference results to Dart and a
//  MethodChannel for lifecycle control (stop / start).

import AVFoundation
@preconcurrency import Flutter
import UIKit
import UltralyticsYOLO

@MainActor
public final class SwiftYOLOMultiTaskPlatformView: NSObject,
  @preconcurrency FlutterPlatformView,
  @preconcurrency FlutterStreamHandler
{
  private let viewId: Int64
  private let eventChannel: FlutterEventChannel
  private let methodChannel: FlutterMethodChannel
  private var eventSink: FlutterEventSink?
  private var multiTaskView: YOLOMultiTaskView?

  init(frame: CGRect, viewId: Int64, args: Any?, messenger: FlutterBinaryMessenger) {
    self.viewId = viewId

    let idStr: String
    if let dict = args as? [String: Any], let s = dict["viewId"] as? String {
      idStr = s
    } else {
      idStr = "\(viewId)"
    }

    eventChannel = FlutterEventChannel(
      name: "com.ultralytics.yolo/multiTaskResults_\(idStr)",
      binaryMessenger: messenger)
    methodChannel = FlutterMethodChannel(
      name: "com.ultralytics.yolo/multiTaskControl_\(idStr)",
      binaryMessenger: messenger)

    super.init()

    eventChannel.setStreamHandler(self)
    setupMethodChannel()

    guard
      let dict = args as? [String: Any],
      dict["detectModel"] != nil || dict["classifyModel"] != nil || dict["secondDetectModel"] != nil
    else {
      return
    }

    let detectPath = dict["detectModel"] as? String
    let classifyPath = dict["classifyModel"] as? String

    let useGpu = dict["useGpu"] as? Bool ?? true
    let lensFacing = dict["lensFacing"] as? String ?? "back"
    let cameraPosition: AVCaptureDevice.Position = lensFacing == "front" ? .front : .back
    let confidenceThreshold = dict["confidenceThreshold"] as? Double ?? 0.25
    let iouThreshold = dict["iouThreshold"] as? Double ?? 0.7
    let detectConfidenceThreshold = dict["detectConfidenceThreshold"] as? Double ?? confidenceThreshold
    let detectIouThreshold = dict["detectIouThreshold"] as? Double ?? iouThreshold
    let classifyConfidenceThreshold = dict["classifyConfidenceThreshold"] as? Double ?? confidenceThreshold
    let secondDetectModelPath = dict["secondDetectModel"] as? String
    let secondDetectConfidenceThreshold = dict["secondDetectConfidenceThreshold"] as? Double ?? confidenceThreshold
    let secondDetectIouThreshold = dict["secondDetectIouThreshold"] as? Double ?? iouThreshold

    let view = YOLOMultiTaskView(frame: frame)
    multiTaskView = view
    view.thirdModelId = "detect2"

    view.onMultiTaskStream = { [weak self] data in
      guard let self, let sink = self.eventSink else { return }
      sink(data)
    }

    view.loadModels(
      detectPath: detectPath,
      classifyPath: classifyPath,
      thirdModelPath: secondDetectModelPath,
      thirdModelTask: "detect",
      useGpu: useGpu,
      detectConfidenceThreshold: detectConfidenceThreshold,
      detectIouThreshold: detectIouThreshold,
      classifyConfidenceThreshold: classifyConfidenceThreshold,
      thirdConfidenceThreshold: secondDetectConfidenceThreshold,
      thirdIouThreshold: secondDetectIouThreshold,
      cameraPosition: cameraPosition
    ) {
      // Models loaded — camera starts automatically inside loadModels.
    }
  }

  private func setupMethodChannel() {
    methodChannel.setMethodCallHandler { [weak self] call, result in
      guard let self else {
        result(FlutterError(code: "disposed", message: "View was disposed", details: nil))
        return
      }
      switch call.method {
      case "stop":
        self.multiTaskView?.stopCamera()
        self.multiTaskView?.releaseResources()
        self.multiTaskView = nil
        result(nil)
      case "capturePhoto":
        guard let view = self.multiTaskView else {
          result(FlutterError(code: "unavailable", message: "Camera not ready", details: nil))
          return
        }
        view.capturePhoto { data in
          DispatchQueue.main.async {
            if let data {
              result(FlutterStandardTypedData(bytes: data))
            } else {
              result(FlutterError(code: "capture_failed", message: "Failed to capture photo", details: nil))
            }
          }
        }
      case "setTorch":
        guard let args = call.arguments as? [String: Any],
          let enable = args["enable"] as? Bool
        else {
          result(FlutterError(code: "bad_args", message: "enable (bool) is required", details: nil))
          return
        }
        let active = self.multiTaskView?.setTorchMode(enable) ?? false
        result(active)
      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }

  public func view() -> UIView { multiTaskView ?? UIView() }

  // MARK: FlutterStreamHandler

  public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink)
    -> FlutterError?
  {
    eventSink = events
    return nil
  }

  public func onCancel(withArguments arguments: Any?) -> FlutterError? {
    eventSink = nil
    return nil
  }

  deinit {
    MainActor.assumeIsolated {
      eventSink = nil
      eventChannel.setStreamHandler(nil)
      methodChannel.setMethodCallHandler(nil)
      multiTaskView?.releaseResources()
      multiTaskView = nil
    }
  }
}

// MARK: - Factory

@MainActor
public final class SwiftYOLOMultiTaskPlatformViewFactory: NSObject,
  @preconcurrency FlutterPlatformViewFactory
{
  private let messenger: FlutterBinaryMessenger

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  public func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol {
    return FlutterStandardMessageCodec.sharedInstance()
  }

  public func create(withFrame frame: CGRect, viewIdentifier viewId: Int64, arguments args: Any?)
    -> FlutterPlatformView
  {
    return SwiftYOLOMultiTaskPlatformView(
      frame: frame, viewId: viewId, args: args, messenger: messenger)
  }
}
