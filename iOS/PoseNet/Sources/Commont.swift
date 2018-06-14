let NUM_KEYPOINTS = partNames.count

let partNames: [String] = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

let partIds = partNames.enumerated().reduce(into: [String: Int]()) {
    $0[$1.element] = $1.offset
}

let connectedPartNames = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

let connectedPartIndices = connectedPartNames.map {
    (partIds[$0.0]!,  partIds[$0.1]!)
}

let kLocalMaximumRadius = 1

let poseChain = [
    ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
    ("rightEye", "rightEar"), ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle")
]

let parentChildrenTuples = poseChain.map {
    (partIds[$0.0]!,  partIds[$0.1]!)
}

let parentToChildEdges = parentChildrenTuples.map { $1 }
let childToParentEdges = parentChildrenTuples.map { $0.0 }



