// swift-tools-version: 5.9
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import PackageDescription

let package = Package(
  name: "aicycle_yolo",
  platforms: [
    .iOS("13.0")
  ],
  products: [
    .library(name: "aicycle-yolo", targets: ["aicycle_yolo"])
  ],
  dependencies: [
    .package(name: "FlutterFramework", path: "../FlutterFramework"),
    .package(url: "https://github.com/ultralytics/yolo-ios-app.git", .upToNextMajor(from: "8.9.5")),
  ],
  targets: [
    .target(
      name: "aicycle_yolo",
      dependencies: [
        .product(name: "FlutterFramework", package: "FlutterFramework"),
        .product(name: "UltralyticsYOLO", package: "yolo-ios-app"),
      ],
      resources: [
        .process("PrivacyInfo.xcprivacy")
      ]
    )
  ]
)
