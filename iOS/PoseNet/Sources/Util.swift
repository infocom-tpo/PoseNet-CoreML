import TensorSwift

func getTensor(_ name: String,_ shape: Shape) -> Tensor {
    let url = Bundle.main.url(forResource: name, withExtension: "bin")!
    let binaryData = try! Data(contentsOf: url, options: [])
    let values: [Float32] = binaryData.withUnsafeBytes {
        [Float32](UnsafeBufferPointer(start: $0, count: binaryData.count/MemoryLayout<Float32>.stride))
    }
    return Tensor(shape: shape, elements: values)
}

func getOffsetPoint(
    y: Int, x: Int, keypoint: Int, offsets: Tensor) -> Vector2D {
    return Vector2D(
        x: offsets[keypoint + NUM_KEYPOINTS, y, x],
        y: offsets[keypoint, y, x]
    )
}

func squaredDistance(
    y1: Float32, x1: Float32, y2: Float32, x2: Float32) -> Float32 {
    let dy = y2 - y1
    let dx = x2 - x1
    return dy * dy + dx * dx
}

extension Comparable
{
    func clamp<T: Comparable>(_ lower: T, _ upper: T) -> T {
        return min(max(self as! T, lower), upper)
    }
}

func addVectors(_ a: Vector2D,_ b: Vector2D) -> Vector2D {
    return Vector2D(x: a.x + b.x, y: a.y + b.y)
}

func getImageCoords(
    part: Part, outputStride: Int, offsets: Tensor) -> Vector2D {
    
    let vec = getOffsetPoint(y: part.heatmapY, x: part.heatmapX,
                             keypoint: part.id,offsets: offsets)
    return Vector2D(
        x: Float32(part.heatmapX * outputStride) + vec.x,
        y: Float32(part.heatmapY * outputStride) + vec.y
    )
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
