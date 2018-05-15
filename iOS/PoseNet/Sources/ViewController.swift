import Foundation
import UIKit
import Vision
import TensorSwift

class ViewController: UIViewController {
    
    let model = posenet()
    let ImageWidth = 513
    let ImageHeight = 513
    
    @IBOutlet weak var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        
//        let poses = runOffline()
//        drawResults(poses)
//        let fname = "lifting.png"
//        let fname = "soccer.png"
//        let fname = "frisbee.jpg"
        let fname = "tennis_in_crowd.jpg"
        if let image = UIImage(named: fname){
            imageView.image = image
            let result = measure(runCoreML(image))
            
            print(result.duration)
            drawResults(result.result)
        }
    }
    func drawResults(_ poses: [Pose]){
        let minPoseConfidence: Float32 = 0.5
        
        poses.forEach { pose in
            if (pose.score >= minPoseConfidence){
                drawKeypoints(keypoints: pose.keypoints,minConfidence: minPoseConfidence)
                drawSkeleton(keypoints: pose.keypoints,minConfidence: minPoseConfidence)
            }
        }
    }
    
    func drawKeypoints(keypoints: [Keypoint], minConfidence: Float32){
        keypoints.forEach { keypoint in
            if (keypoint.score < minConfidence) {
                return
            }
            
            let center = CGPoint(x: Int(keypoint.position.x), y: Int(keypoint.position.y))
            let line = CAShapeLayer()
            
            let trackPath = UIBezierPath(arcCenter: center,
                                         radius: 3, startAngle: 0,
                                         endAngle: 2.0 * .pi, clockwise: true)
            line.path = trackPath.cgPath
            line.strokeColor = UIColor.green.cgColor
            self.view.layer.addSublayer(line)
        }
    }

    func drawSegment(fromPoint start: CGPoint, toPoint end:CGPoint) {
        let line = CAShapeLayer()
        let linePath = UIBezierPath()
        linePath.move(to: start)
        linePath.addLine(to: end)
        line.path = linePath.cgPath
        line.strokeColor = UIColor.red.cgColor
        line.lineWidth = 3
        line.lineJoin = kCALineJoinRound
        self.view.layer.addSublayer(line)
    }
    func drawSkeleton(keypoints: [Keypoint], minConfidence: Float32){
        let adjacentKeyPoints = getAdjacentKeyPoints(
            keypoints: keypoints, minConfidence: minConfidence);
        
        adjacentKeyPoints.forEach { keypoint in
            drawSegment(
                fromPoint:
                    CGPoint(x: Int(keypoint[0].position.x),y: Int(keypoint[0].position.y)),
                toPoint:
                    CGPoint(x: Int(keypoint[1].position.x),y: Int(keypoint[1].position.y))
            )
        }
    }
    
    func eitherPointDoesntMeetConfidence(
        _ a: Float32,_ b: Float32,_ minConfidence: Float32) -> Bool {
        return (a < minConfidence || b < minConfidence)
    }
    
    func getAdjacentKeyPoints(
        keypoints: [Keypoint], minConfidence: Float32)-> [[Keypoint]] {
    
        return connectedPartIndeces.reduce([[Keypoint]](), { res , joint in
            var arr = res
            if (eitherPointDoesntMeetConfidence(
                keypoints[joint.0].score,
                keypoints[joint.1].score,
                minConfidence)){
                return res
            }
            arr.append([keypoints[joint.0],keypoints[joint.1]])
            return arr
        })
    }

//        Implementation
//        CoreML: 33,33,17 = y,x,z
//        Offline: 17,33,33 = z,y,x
//    func runOffline() -> [Pose] {
//
//        let scores = getTensor("heatmapScores",[33, 33, 17])
//        let offsets = getTensor("offsets",[33, 33, 34])
//        let displacementsFwd = getTensor("displacementsFwd",[33, 33, 32])
//        let displacementsBwd = getTensor("displacementsBwd",[33, 33, 32])
//        let outputStride = 16
//
//        let posenet = PoseNet()
//
//        let poses = posenet.decodeMultiplePoses(
//            scores: scores,
//            offsets: offsets,
//            displacementsFwd: displacementsFwd,
//            displacementsBwd: displacementsBwd,
//            outputStride: outputStride, maxPoseDetections: 15,
//            scoreThreshold: 0.5,nmsRadius: 20)
//
//        return poses
//    }
//
    func runCoreML(_ image: UIImage) -> [Pose]{
        let posnet = PoseNet()
        
        let img = image.pixelBuffer(width: ImageWidth, height: ImageWidth)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try? model.prediction(image__0: img!)
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("coreml Time elapsed for roop: \(timeElapsed) seconds")

        let names: [String] = ["heatmap__0","offset_2__0","displacement_fwd_2__0","displacement_bwd_2__0"]
        let tensors = names.reduce(into: [String: Tensor]()) {
            $0[$1] = getTensor(
                result?.featureValue(for: $1)?.multiArrayValue)
        }
//        let mlarray = result?.featureValue(for: "heatmap__0")?.multiArrayValue
//
//        let length = mlarray!.count
//        let doublePtr =  mlarray!.dataPointer.bindMemory(to: Double.self, capacity: length)
//        let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: length)
//        let buffer = Array(doubleBuffer)
//        let sum = buffer.reduce(0, +) / (17 * 33 * 33)
//        print(sum)
        let outputStride = 16

        let poses = posnet.decodeMultiplePoses(
                        scores: tensors["heatmap__0"]!,
                        offsets: tensors["offset_2__0"]!,
                        displacementsFwd: tensors["displacement_fwd_2__0"]!,
                        displacementsBwd: tensors["displacement_bwd_2__0"]!,
                        outputStride: outputStride, maxPoseDetections: 15,
                        scoreThreshold: 0.5,nmsRadius: 20)
        
        return poses
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func measure <T> (_ f: @autoclosure () -> T) -> (result: T, duration: String) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = f()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        return (result, "Elapsed time is \(timeElapsed) seconds.")
    }
    
    func getTensor(_ mlarray: MLMultiArray!) -> Tensor {
        let length = mlarray.count
        let doublePtr =  mlarray.dataPointer.bindMemory(to: Double.self, capacity: length)
        let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: length)
        let element = Array(doubleBuffer).asArrayOfFloat
        
        let dim = mlarray.shape.map { Dimension(($0 as? Int)!)}
        return Tensor(shape: [dim[0], dim[1], dim[2]], elements: element)
    }
    
    func getTensor(_ name: String,_ shape: Shape) -> Tensor {
        let url = Bundle.main.url(forResource: name, withExtension: "bin")!
        let binaryData = try! Data(contentsOf: url, options: [])
        let values: [Float32] = binaryData.withUnsafeBytes {
            [Float32](UnsafeBufferPointer(start: $0, count: binaryData.count/MemoryLayout<Float32>.stride))
        }
        return Tensor(shape: shape, elements: values)
    }
}

class PoseNet {
}

