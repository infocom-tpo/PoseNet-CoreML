import TensorSwift

extension PoseNet {
    
    func getOffsetPoint(
        y: Int, x: Int, keypoint: Int, offsets: Tensor) -> Vector2D {
        return Vector2D(
            x: offsets[keypoint + NUM_KEYPOINTS, y, x ],
            y: offsets[keypoint, y, x ]
        )
    }

    func squaredDistance(
        y1: Float, x1: Float, y2: Float, x2: Float) -> Float {
        let dy = y2 - y1
        let dx = x2 - x1
        return dy * dy + dx * dx
    }

    func addVectors(_ a: Vector2D,_ b: Vector2D) -> Vector2D {
        return Vector2D(x: a.x + b.x, y: a.y + b.y)
    }

    func getImageCoords(
        part: Part, outputStride: Int, offsets: Tensor) -> Vector2D {
        
        let vec = getOffsetPoint(y: part.heatmapY, x: part.heatmapX,
                                 keypoint: part.id,offsets: offsets)
        return Vector2D(
            x: Float(part.heatmapX * outputStride) + vec.x,
            y: Float(part.heatmapY * outputStride) + vec.y
        )
    }

    func scalePose(pose: Pose, scale: Int) -> Pose {
        
        let s = Float(scale)
        return Pose(
            keypoints: pose.keypoints.map {
                    Keypoint(score: $0.score,
                             position: Vector2D(
                                x: $0.position.x * s,
                                y: $0.position.y * s),
                             part: $0.part)
                },
                score: pose.score
            )
    }

    func scalePoses(poses: [Pose], scale: Int) -> [Pose] {
        if (scale == 1) {
            return poses
        }
        return poses.map{scalePose(pose: $0, scale: scale)}
    }

    func getValidResolution(imageScaleFactor: Float,
                            inputDimension: Int,
                            outputStride: Int) -> Int {
            let evenResolution = Int(Float(inputDimension) * imageScaleFactor) - 1
            return evenResolution - (evenResolution % outputStride) + 1
    }
}

extension Comparable
{
    func clamp<T: Comparable>(_ lower: T, _ upper: T) -> T {
        return min(max(self as! T, lower), upper)
    }
}

extension UIImage {
    func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        return resizedImage
    }
}

extension Array where Iterator.Element == Double {
    public var asArrayOfFloat: [Float] {
        return self.map { return Float($0) } // compiler error
    }
}
