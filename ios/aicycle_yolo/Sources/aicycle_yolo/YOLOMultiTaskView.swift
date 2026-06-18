// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  YOLOMultiTaskView — runs up to three YOLO models on a single camera stream simultaneously.
//  Each BasePredictor receives raw CVPixelBuffers on its own dispatch queue so all models
//  run concurrently via Apple's CoreML async scheduling.

import AVFoundation
import CoreVideo
import UIKit
import UltralyticsYOLO

// MARK: - Per-predictor result adapter

/// Bridges ResultsListener / InferenceTimeListener callbacks back to a labelled slot
/// in YOLOMultiTaskView. All callbacks are re-dispatched onto `cameraQueue` so the
/// busy-flag bookkeeping stays on one serial queue (no locks needed).
final class MultiTaskPredictorAdapter: ResultsListener, InferenceTimeListener,
  @unchecked Sendable
{
  let taskName: String
  private let cameraQueue: DispatchQueue

  // Set by owner before use
  var onResult: ((YOLOResult) -> Void)?
  var onTime: ((Double, Double) -> Void)?

  init(taskName: String, cameraQueue: DispatchQueue) {
    self.taskName = taskName
    self.cameraQueue = cameraQueue
  }

  func on(result: YOLOResult) {
    let r = result
    cameraQueue.async { [weak self] in self?.onResult?(r) }
  }

  func on(inferenceTime: Double, fpsRate: Double) {
    let ms = inferenceTime
    let fps = fpsRate
    cameraQueue.async { [weak self] in self?.onTime?(ms, fps) }
  }
}

// MARK: - YOLOMultiTaskView

/// A UIView that hosts a single AVCaptureSession and dispatches each incoming
/// camera frame to up to three independent YOLO predictors simultaneously.
@MainActor
public class YOLOMultiTaskView: UIView {

  // MARK: Camera

  private let captureSession = AVCaptureSession()
  private var previewLayer: AVCaptureVideoPreviewLayer?
  private let photoOutput = AVCapturePhotoOutput()
  private var photoCaptureCompletion: ((Data?) -> Void)?
  private var captureDevice: AVCaptureDevice?

  /// Serial queue for camera delegate callbacks and busy-flag mutations only.
  let cameraQueue = DispatchQueue(label: "yolo.multi-task.camera", qos: .userInteractive)

  /// Per-predictor inference queues — each predictor runs concurrently.
  private let detectQueue  = DispatchQueue(label: "yolo.infer.detect",   qos: .userInteractive)
  private let classifyQueue = DispatchQueue(label: "yolo.infer.classify", qos: .userInteractive)
  private let thirdQueue   = DispatchQueue(label: "yolo.infer.third",    qos: .userInteractive)

  // MARK: Predictors

  var detectPredictor:   BasePredictor?
  var classifyPredictor: BasePredictor?
  var thirdPredictor:    BasePredictor?
  var thirdTaskType:     String = "detect"
  /// Stable identifier for the third predictor's results, so consumers can tell two detect
  /// models apart (the primary detect model reports `modelId == "detect"`).
  var thirdModelId:      String = "detect2"

  /// One-frame-deep back-pressure per predictor. Accessed only on cameraQueue.
  var detectBusy   = false
  var classifyBusy = false
  var thirdBusy    = false

  private lazy var detectAdapter   = MultiTaskPredictorAdapter(taskName: "detect",   cameraQueue: cameraQueue)
  private lazy var classifyAdapter = MultiTaskPredictorAdapter(taskName: "classify", cameraQueue: cameraQueue)
  private lazy var thirdAdapter    = MultiTaskPredictorAdapter(taskName: "third",    cameraQueue: cameraQueue)

  // MARK: Per-task FPS tracking (cameraQueue)

  private var detectLastResultTime:   Double = 0
  private var classifyLastResultTime: Double = 0
  private var thirdLastResultTime:    Double = 0

  private var detectFps:   Double = 0
  private var classifyFps: Double = 0
  private var thirdFps:    Double = 0

  // Camera FPS (cameraQueue)
  private var camFrameCount = 0
  private var camFpsWindowStart: Double = 0
  private var camFps: Double = 0

  // MARK: Callback

  /// Called on the main thread with a stream-data dict. Keys: "type", "fps", "cameraFps",
  /// "processingTimeMs", plus task-specific keys ("detections", "classification", etc.).
  var onMultiTaskStream: (([String: Any]) -> Void)?

  // MARK: Loading indicator

  public let activityIndicator = UIActivityIndicatorView(style: .large)
  private var loadedCount = 0
  private var expectedCount = 0

  // MARK: Init

  public override init(frame: CGRect) {
    super.init(frame: frame)
    setupUI()
    wireAdapters()
  }

  required init?(coder: NSCoder) {
    super.init(coder: coder)
    setupUI()
    wireAdapters()
  }

  private func setupUI() {
    backgroundColor = .black
    activityIndicator.color = .white
    activityIndicator.startAnimating()
    addSubview(activityIndicator)
  }

  public override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer?.frame = bounds
    activityIndicator.center = CGPoint(x: bounds.midX, y: bounds.midY)
  }

  // MARK: - Adapter wiring

  private func wireAdapters() {
    detectAdapter.onResult   = { [weak self] r in self?.handleResult(r, slot: .detect) }
    classifyAdapter.onResult = { [weak self] r in self?.handleResult(r, slot: .classify) }
    thirdAdapter.onResult    = { [weak self] r in self?.handleResult(r, slot: .third) }

    detectAdapter.onTime   = { [weak self] _, fps in self?.detectFps   = fps }
    classifyAdapter.onTime = { [weak self] _, fps in self?.classifyFps = fps }
    thirdAdapter.onTime    = { [weak self] _, fps in self?.thirdFps    = fps }
  }

  // MARK: - Result handling (cameraQueue)

  /// Identifies which predictor slot a result came from. Bookkeeping (busy flags, FPS) is
  /// keyed on the slot — not the task — so two detect models never clobber each other's state.
  private enum Slot { case detect, classify, third }

  private func handleResult(_ result: YOLOResult, slot: Slot) {
    let now = CACurrentMediaTime()
    let task: String
    let modelId: String
    var taskFps: Double = 0

    switch slot {
    case .detect:
      detectBusy = false
      detectPredictor?.isUpdating = false
      if detectLastResultTime > 0 {
        let dt = now - detectLastResultTime
        if dt > 0 { detectFps = 1.0 / dt }
      }
      detectLastResultTime = now
      taskFps = detectFps
      task = "detect"
      modelId = "detect"
    case .classify:
      classifyBusy = false
      classifyPredictor?.isUpdating = false
      if classifyLastResultTime > 0 {
        let dt = now - classifyLastResultTime
        if dt > 0 { classifyFps = 1.0 / dt }
      }
      classifyLastResultTime = now
      taskFps = classifyFps
      task = "classify"
      modelId = "classify"
    case .third:
      thirdBusy = false
      thirdPredictor?.isUpdating = false
      if thirdLastResultTime > 0 {
        let dt = now - thirdLastResultTime
        if dt > 0 { thirdFps = 1.0 / dt }
      }
      thirdLastResultTime = now
      taskFps = thirdFps
      task = thirdTaskType
      modelId = thirdModelId
    }

    let camFpsSnapshot = camFps
    let streamData = buildStreamData(
      result: result, task: task, modelId: modelId, fps: taskFps, cameraFps: camFpsSnapshot)

    DispatchQueue.main.async { [weak self] in
      self?.onMultiTaskStream?(streamData)
    }
  }

  // MARK: - Stream data builder

  private func buildStreamData(
    result: YOLOResult, task: String, modelId: String, fps: Double, cameraFps: Double
  ) -> [String: Any] {
    var map: [String: Any] = [
      "type": task,
      "modelId": modelId,
      "fps": fps,
      "cameraFps": cameraFps,
      "processingTimeMs": result.speed * 1000,
    ]

    switch task {
    case "classify":
      if let probs = result.probs {
        var top5: [[String: Any]] = []
        for i in 0..<min(probs.top5Labels.count, probs.top5Confs.count) {
          top5.append(["name": probs.top5Labels[i], "confidence": Double(probs.top5Confs[i])])
        }
        map["classification"] = [
          "top1": probs.top1Label,
          "top1Confidence": Double(probs.top1Conf),
          "top5": top5,
        ]
      }
    default:
      // detect, segment, pose, obb — all have boxes.
      // The custom Vietnamese damage labels only apply to the primary detect model; any
      // other detect model reports its own class names via box.cls.
      let useCustomNames = (modelId == "detect")
      let classNames = ["Móp/bẹp", "Vỡ/nứt", "Thủng/rách", "Trầy/xước"]
      var detections: [[String: Any]] = []
      for box in result.boxes.prefix(50) {
        let name = (useCustomNames && box.index < classNames.count) ? classNames[box.index] : box.cls
        detections.append([
          "className": name,
          "confidence": Double(box.conf),
          "normalizedBox": [
            "left":   Double(box.xywhn.minX),
            "top":    Double(box.xywhn.minY),
            "right":  Double(box.xywhn.maxX),
            "bottom": Double(box.xywhn.maxY),
          ],
        ])
      }
      map["detections"] = detections
    }

    return map
  }

  // MARK: - Model loading

  /// Load two or three models concurrently. `thirdModelPath` / `thirdModelTask` are optional.
  /// `completion` fires on the main thread once all requested models finish loading.
  /// Camera starts automatically after all models are ready.
  func loadModels(
    detectPath: String,
    classifyPath: String,
    thirdModelPath: String? = nil,
    thirdModelTask: String = "detect",
    useGpu: Bool = true,
    detectConfidenceThreshold: Double = 0.25,
    detectIouThreshold: Double = 0.7,
    classifyConfidenceThreshold: Double = 0.25,
    thirdConfidenceThreshold: Double = 0.25,
    thirdIouThreshold: Double = 0.7,
    cameraPosition: AVCaptureDevice.Position = .back,
    completion: @escaping () -> Void
  ) {
    self.thirdTaskType = thirdModelTask
    expectedCount = thirdModelPath != nil ? 3 : 2
    loadedCount = 0

    func tryDone() {
      loadedCount += 1
      if loadedCount == expectedCount {
        DispatchQueue.main.async { [weak self] in
          self?.activityIndicator.stopAnimating()
          self?.startCamera(position: cameraPosition)
          completion()
        }
      }
    }

    load(path: detectPath, task: .detect, useGpu: useGpu) { [weak self] p in
      if p == nil { NSLog("YOLOMultiTaskView: ⚠️ detect predictor is nil after load") }
      else { NSLog("YOLOMultiTaskView: ✅ detect predictor loaded") }
      p?.setConfidenceThreshold(confidence: detectConfidenceThreshold)
      p?.setIouThreshold(iou: detectIouThreshold)
      self?.detectPredictor = p
      tryDone()
    }

    load(path: classifyPath, task: .classify, useGpu: useGpu) { [weak self] p in
      if p == nil { NSLog("YOLOMultiTaskView: ⚠️ classify predictor is nil after load") }
      else { NSLog("YOLOMultiTaskView: ✅ classify predictor loaded") }
      p?.setConfidenceThreshold(confidence: classifyConfidenceThreshold)
      self?.classifyPredictor = p
      tryDone()
    }

    if let thirdPath = thirdModelPath {
      let yoloTask = yoloTaskFromString(thirdModelTask)
      load(path: thirdPath, task: yoloTask, useGpu: useGpu) { [weak self] p in
        if p == nil { NSLog("YOLOMultiTaskView: ⚠️ third predictor (\(thirdModelTask)) is nil after load") }
        else { NSLog("YOLOMultiTaskView: ✅ third predictor (\(thirdModelTask)) loaded") }
        p?.setConfidenceThreshold(confidence: thirdConfidenceThreshold)
        p?.setIouThreshold(iou: thirdIouThreshold)
        self?.thirdPredictor = p
        tryDone()
      }
    }
  }

  private func yoloTaskFromString(_ task: String) -> YOLOTask {
    switch task.lowercased() {
    case "classify": return .classify
    case "segment":  return .segment
    case "pose":     return .pose
    case "obb":      return .obb
    default:         return .detect
    }
  }

  private func load(
    path: String, task: YOLOTask, useGpu: Bool,
    completion: @escaping (BasePredictor?) -> Void
  ) {
    guard let url = resolveModelURL(path) else {
      NSLog("YOLOMultiTaskView: model not found: %@", path)
      completion(nil)
      return
    }
    BasePredictor.create(for: task, modelURL: url, isRealTime: true, useGpu: useGpu) { result in
      switch result {
      case .success(let p):
        let bp = p as? BasePredictor
        bp?.capturesOriginalImage = false
        completion(bp)
      case .failure(let err):
        NSLog("YOLOMultiTaskView: load failed for %@: %@", path, err.localizedDescription)
        completion(nil)
      }
    }
  }

  private func resolveModelURL(_ nameOrPath: String) -> URL? {
    let lc = nameOrPath.lowercased()
    let fm = FileManager.default

    // Direct path: .mlpackage (must be a directory), .mlmodelc, .mlmodel
    if lc.hasSuffix(".mlmodelc") || lc.hasSuffix(".mlpackage") || lc.hasSuffix(".mlmodel") {
      let u = URL(fileURLWithPath: nameOrPath)
      var isDir: ObjCBool = false
      if fm.fileExists(atPath: u.path, isDirectory: &isDir) {
        // Only return if it's a directory (proper bundle) — files are unextracted zips
        // handled by the Dart layer; if they reach here, skip them.
        if isDir.boolValue { return u }
        NSLog("YOLOMultiTaskView: ⚠️ path is a file (unextracted zip?): %@", nameOrPath)
        return nil
      }
    }

    // .mlpackage.zip: should have been extracted by Dart resolver, but check derived path
    if lc.hasSuffix(".mlpackage.zip") {
      let derived = URL(fileURLWithPath: String(nameOrPath.dropLast(4))) // drop ".zip"
      var isDir: ObjCBool = false
      if fm.fileExists(atPath: derived.path, isDirectory: &isDir), isDir.boolValue {
        return derived
      }
      return nil
    }

    if let u = Bundle.main.url(forResource: nameOrPath, withExtension: "mlmodelc") { return u }
    if let u = Bundle.main.url(forResource: nameOrPath, withExtension: "mlpackage") { return u }
    return nil
  }

  // MARK: - Camera

  private func startCamera(position: AVCaptureDevice.Position) {
    let pos = position
    cameraQueue.async { [weak self] in self?.setupCamera(position: pos) }
  }

  private func setupCamera(position: AVCaptureDevice.Position) {
    captureSession.beginConfiguration()
    captureSession.sessionPreset = .hd1280x720

    guard let device = bestCaptureDevice(position: position),
      let input = try? AVCaptureDeviceInput(device: device),
      captureSession.canAddInput(input)
    else {
      captureSession.commitConfiguration()
      return
    }
    captureDevice = device
    captureSession.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]
    output.alwaysDiscardsLateVideoFrames = true
    output.setSampleBufferDelegate(self, queue: cameraQueue)
    if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
    if captureSession.canAddOutput(photoOutput) { captureSession.addOutput(photoOutput) }

    // Rotate the pixel buffer to landscape so the model always receives wide frames
    // regardless of how the user holds the device. The preview connection is left at
    // its default (.portrait) so the viewfinder appears upright to the user.
    if let conn = output.connection(with: .video) {
      conn.videoOrientation = .landscapeRight
    }
    if let conn = photoOutput.connection(with: .video) {
      conn.videoOrientation = .landscapeRight
    }

    captureSession.commitConfiguration()

    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      let preview = AVCaptureVideoPreviewLayer(session: self.captureSession)
      preview.videoGravity = .resizeAspectFill
      preview.frame = self.bounds
      self.layer.insertSublayer(preview, at: 0)
      self.previewLayer = preview
    }

    captureSession.startRunning()
    camFpsWindowStart = CACurrentMediaTime()
  }

  public func capturePhoto(completion: @escaping (Data?) -> Void) {
    photoCaptureCompletion = completion
    let settings = AVCapturePhotoSettings()
    settings.flashMode = .off
    cameraQueue.async { [weak self] in
      guard let self else { completion(nil); return }
      self.photoOutput.capturePhoto(with: settings, delegate: self)
    }
  }

  @discardableResult
  public func setTorchMode(_ enabled: Bool) -> Bool {
    guard let device = captureDevice, device.hasTorch else { return false }
    do {
      try device.lockForConfiguration()
      defer { device.unlockForConfiguration() }
      if enabled {
        try device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel)
      } else {
        device.torchMode = .off
      }
      return device.torchMode == .on
    } catch {
      NSLog("YOLOMultiTaskView: Failed to set torch mode: %@", error.localizedDescription)
      return false
    }
  }

  public func stopCamera() {
    cameraQueue.async { [weak self] in
      self?.captureSession.stopRunning()
    }
  }

  /// Full resource release: stops camera, removes preview layer, nils predictors and callback.
  /// Call from the platform view's dispose/deinit path so GPU/ANE memory is freed promptly
  /// even if deinit is delayed by a retain cycle in the Flutter EventChannel stream handler.
  public func releaseResources() {
    onMultiTaskStream = nil
    cameraQueue.async { [weak self] in
      self?.captureSession.stopRunning()
    }
    DispatchQueue.main.async { [weak self] in
      self?.previewLayer?.removeFromSuperlayer()
      self?.previewLayer = nil
    }
    detectPredictor = nil
    classifyPredictor = nil
    thirdPredictor = nil
  }

  deinit {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }
  }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension YOLOMultiTaskView: AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {

  public func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    // All mutable state accesses in this extension run on cameraQueue.

    // Track camera FPS
    let now = CACurrentMediaTime()
    camFrameCount += 1
    let elapsed = now - camFpsWindowStart
    if elapsed >= 0.5 {
      camFps = Double(camFrameCount) / elapsed
      camFrameCount = 0
      camFpsWindowStart = now
    }

    // Dispatch each predictor to its own queue so all run concurrently.
    if let p = detectPredictor, !detectBusy, !p.isUpdating {
      detectBusy = true
      p.isUpdating = true
      let buf = sampleBuffer
      let adapter = detectAdapter
      detectQueue.async { p.predict(sampleBuffer: buf, onResultsListener: adapter, onInferenceTime: adapter) }
    }
    if let p = classifyPredictor, !classifyBusy, !p.isUpdating {
      classifyBusy = true
      p.isUpdating = true
      let buf = sampleBuffer
      let adapter = classifyAdapter
      classifyQueue.async { p.predict(sampleBuffer: buf, onResultsListener: adapter, onInferenceTime: adapter) }
    }
    if let p = thirdPredictor, !thirdBusy, !p.isUpdating {
      thirdBusy = true
      p.isUpdating = true
      let buf = sampleBuffer
      let adapter = thirdAdapter
      thirdQueue.async { p.predict(sampleBuffer: buf, onResultsListener: adapter, onInferenceTime: adapter) }
    }
  }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension YOLOMultiTaskView: AVCapturePhotoCaptureDelegate {
  public func photoOutput(
    _ output: AVCapturePhotoOutput,
    didFinishProcessingPhoto photo: AVCapturePhoto,
    error: Error?
  ) {
    let completion = photoCaptureCompletion
    photoCaptureCompletion = nil
    guard error == nil, let data = photo.fileDataRepresentation() else {
      completion?(nil)
      return
    }
    guard let src = UIImage(data: data), src.imageOrientation != .up else {
      completion?(data)
      return
    }
    let renderer = UIGraphicsImageRenderer(size: src.size)
    let normalized = renderer.jpegData(withCompressionQuality: 0.92) { _ in
      src.draw(in: CGRect(origin: .zero, size: src.size))
    }
    completion?(normalized)
  }
}
