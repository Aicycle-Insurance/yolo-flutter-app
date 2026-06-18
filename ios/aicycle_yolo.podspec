#
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint aicycle_yolo.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'aicycle_yolo'
  s.version          = '0.6.0'
  s.summary          = 'Flutter plugin for YOLO (You Only Look Once) models'
  s.description      = <<-DESC
Flutter plugin for YOLO (You Only Look Once) models, supporting object detection, segmentation, classification, pose estimation and oriented bounding boxes (OBB) on both Android and iOS.
                       DESC
  s.homepage         = 'https://github.com/ultralytics/yolo-flutter-app'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Ultralytics' => 'info@ultralytics.com' }
  s.source           = { :path => '.' }
  s.source_files = 'aicycle_yolo/Sources/aicycle_yolo/**/*.{swift,h,m}'
  s.dependency 'Flutter'
  s.dependency 'UltralyticsYOLO', '>= 8.9.5', '< 9.0'
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # If your plugin requires a privacy manifest, for example if it uses any
  # required reason APIs, update the PrivacyInfo.xcprivacy file to describe your
  # plugin's privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy-manifest-files
  s.resource_bundles = {'aicycle_yolo_privacy' => ['aicycle_yolo/Sources/aicycle_yolo/PrivacyInfo.xcprivacy']}
end
