import Foundation
import UIKit
import Vision
import TensorSwift

class ViewController: UIViewController {
    
    let model = posenet()
    let ImageWidth = 513
    let ImageHeight = 513
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let fname = "tennis_in_crowd.jpg"
        if let image = UIImage(named: fname){
            print(measure(runCoreML(image)).duration)
        }
    }
    
    func runCoreML(_ image: UIImage) {
//        imageView.image = image
        
        let img = image.pixelBuffer(width: ImageWidth, height: ImageWidth)
        let result = try? model.prediction(image__0: img!)
        
        let names: [String] = ["heatmap__0","offset_2__0","displacement_fwd_2__0","displacement_bwd_2__0"]
        let tensors = names.reduce(into: [String: Tensor]()) {
            $0[$1] = getTensor(
                result?.featureValue(for: $1)?.multiArrayValue)
        }
        
        print(tensors["heatmap__0"])
        let poses = decodeMultiplePoses(
                        scores: tensors["heatmap__0"]!,
                        offsets: tensors["offset_2__0"]!,
                        displacementsFwd: tensors["displacement_fwd_2__0"]!,
                        displacementsBwd: tensors["displacement_bwd_2__0"]!,
                        outputStride: 16, maxPoseDetections: 5,
                        scoreThreshold: 0.5,nmsRadius: 20)
        
        print(poses)
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
}

extension Array where Iterator.Element == Double {
    public var asArrayOfFloat: [Float] {
        return self.map { return Float($0) } // compiler error
    }
}
