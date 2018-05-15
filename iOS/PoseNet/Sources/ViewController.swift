import Foundation
import UIKit
import Vision
import TensorSwift
import AVFoundation

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
        
        imageView.frame = UIScreen.main.bounds
        imageView.contentMode = .scaleAspectFit
        
        let fname = "tennis_in_crowd.jpg"
        if let image = UIImage(named: fname)?.resize(to:
            CGSize(width: ImageWidth, height: ImageHeight)){
            imageView.image = image
            let result = measure(runCoreML(image))
            print(result.duration)
            drawResults(result.result)
        }
    }
    
    func drawResults(_ poses: [Pose]){
        let minPoseConfidence: Float = 0.5
        
        let screen = UIScreen.main.bounds
        let scale = screen.width / CGFloat(ImageWidth)
        
        let size = AVMakeRect(aspectRatio:
            CGSize(width: CGFloat(ImageWidth),height: CGFloat(ImageHeight)),
                               insideRect: imageView.frame)
        poses.forEach { pose in
            if (pose.score >= minPoseConfidence){
                drawKeypoints(keypoints: pose.keypoints,minConfidence: minPoseConfidence,
                              size: size.origin, scale: scale)
                drawSkeleton(keypoints: pose.keypoints,minConfidence: minPoseConfidence,
                             size: size.origin, scale: scale)
            }
        }
    }
    
    func drawKeypoints(keypoints: [Keypoint], minConfidence: Float,
                       size: CGPoint,scale: CGFloat = 1){
        
        keypoints.forEach { keypoint in
            if (keypoint.score < minConfidence) {
                return
            }
            let center = CGPoint(x: CGFloat(keypoint.position.x) * scale + size.x,
                                 y: CGFloat(keypoint.position.y) * scale + size.y)
            let line = CAShapeLayer()
            line.frame = imageView.frame
            let trackPath = UIBezierPath(arcCenter: center,
                                         radius: 3, startAngle: 0,
                                         endAngle: 2.0 * .pi, clockwise: true)
            line.path = trackPath.cgPath
            line.strokeColor = UIColor.green.cgColor
            self.view.layer.addSublayer(line)
            
        }
    }

    func drawSegment(fromPoint start: CGPoint, toPoint end:CGPoint,
                     size: CGPoint, scale: CGFloat = 1) {
        let line = CAShapeLayer()
        let linePath = UIBezierPath()
        linePath.move(to:
            CGPoint(x: start.x * scale + size.x, y: start.y * scale + size.y))
        linePath.addLine(to:
            CGPoint(x: end.x * scale + size.x, y: end.y * scale + size.y))
        line.path = linePath.cgPath
        line.strokeColor = UIColor.red.cgColor
        line.lineWidth = 3
        line.lineJoin = kCALineJoinRound
        self.view.layer.addSublayer(line)
    }
    func drawSkeleton(keypoints: [Keypoint], minConfidence: Float,
                      size: CGPoint, scale: CGFloat = 1){
        let adjacentKeyPoints = getAdjacentKeyPoints(
            keypoints: keypoints, minConfidence: minConfidence);
        
        adjacentKeyPoints.forEach { keypoint in
            drawSegment(
                fromPoint:
                    CGPoint(x: CGFloat(keypoint[0].position.x),y: CGFloat(keypoint[0].position.y)),
                toPoint:
                    CGPoint(x: CGFloat(keypoint[1].position.x),y: CGFloat(keypoint[1].position.y)),
                size: size,
                scale: scale
                
            )
        }
    }
    
    func eitherPointDoesntMeetConfidence(
        _ a: Float,_ b: Float,_ minConfidence: Float) -> Bool {
        return (a < minConfidence || b < minConfidence)
    }
    
    func getAdjacentKeyPoints(
        keypoints: [Keypoint], minConfidence: Float)-> [[Keypoint]] {
    
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
        let img = image.pixelBuffer(width: ImageWidth, height: ImageWidth)!
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try? model.prediction(image__0: img)
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
        let values: [Float] = binaryData.withUnsafeBytes {
            [Float](UnsafeBufferPointer(start: $0, count: binaryData.count/MemoryLayout<Float>.stride))
        }
        return Tensor(shape: shape, elements: values)
    }
}

class PoseNet {
}

