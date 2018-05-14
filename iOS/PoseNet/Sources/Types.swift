import Foundation

struct Vector2D {
    var x:Float32, y:Float32
}

struct Vector2DInt {
    var x:Int, y:Int
}

struct PartWithScore {
    var score: Float32 = 0
    var part: Part
}

struct Part {
    var heatmapX: Int
    var heatmapY: Int
    var id: Int
}

struct Keypoint {
    var score: Float32
    var position: Vector2D
    var part: String
}

struct Pose {
    var keypoints: [Keypoint]
    var score: Float32
}

func half(_ k: Int) -> Int {
    return Int(floor(Double(k/2)))
}

public struct Queue<T> {
    fileprivate var array = [T]()
    
    public var isEmpty: Bool {
        return array.isEmpty
    }
    
    public var count: Int {
        return array.count
    }
    
    public mutating func enqueue(_ element: T) {
        array.append(element)
    }
    
    public mutating func dequeue() -> T? {
        if isEmpty {
            return nil
        } else {
            return array.removeFirst()
        }
    }
    
    public var front: T? {
        return array.first
    }
}
