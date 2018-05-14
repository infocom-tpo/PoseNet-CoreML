let NUM_KEYPOINTS = partNames.count

let partNames: [String] = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

let partIds = partNames.enumerated().reduce(into: [String: Int]()) {
    $0[$1.element] = $1.offset
}
