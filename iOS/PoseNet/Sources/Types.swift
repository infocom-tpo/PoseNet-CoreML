import Foundation

struct Vector2D {
    var x:Float, y:Float
}

struct Vector2DInt {
    var x:Int, y:Int
}

struct PartWithScore {
    var score: Float = 0
    var part: Part
}

struct Part {
    var heatmapX: Int
    var heatmapY: Int
    var id: Int
}

struct Keypoint {
    var score: Float
    var position: Vector2D
    var part: String
}

struct Pose {
    var keypoints: [Keypoint]
    var score: Float
}

func half(_ k: Int) -> Int {
    return Int(floor(Double(k/2)))
}



