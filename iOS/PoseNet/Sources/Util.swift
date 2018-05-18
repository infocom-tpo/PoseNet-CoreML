import TensorSwift
import CoreML

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

func measure <T> (_ f: @autoclosure () -> T) -> (result: T, duration: String) {
    let startTime = CFAbsoluteTimeGetCurrent()
    let result = f()
    let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
    return (result, "Elapsed time is \(timeElapsed) seconds.")
}


func getTensor(_ mlarray: MLMultiArray!) -> Tensor {
    let length = mlarray.count
    let doublePtr = mlarray.dataPointer.bindMemory(to: Double.self, capacity: length)
    let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: length)
    let element = Array(doubleBuffer).asArrayOfFloat
    let dim = mlarray.shape.map { Dimension(($0 as? Int)!)}
    return Tensor(shape: [dim[0], dim[1], dim[2]], elements: element)
}

func getTensor(_ name: String,_ shape: Shape) -> Tensor {
    let url = Bundle.main.url(forResource: name, withExtension: "bin")!
    let binaryData = try! Data(contentsOf: url, options: [])
    let values: [Float] = binaryData.withUnsafeBytes {
        [Float](UnsafeBufferPointer(start: $0, count: binaryData.count/MemoryLayout<Float>.stride))
    }
    return Tensor(shape: shape, elements: values)
}
