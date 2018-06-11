import Foundation
import UIKit
import Vision
import TensorSwift
import AVFoundation

let posnet = PoseNet()
var isXcode : Bool = true // true: localfile , false: device camera

// controlling the pace of the machine vision analysis
var lastAnalysis: TimeInterval = 0
var pace: TimeInterval = 0.08 // in seconds, classification will not repeat faster than this value
// performance tracking
let trackPerformance = false // use "true" for performance logging
var frameCount = 0
let framesPerSample = 10
var startDate = NSDate.timeIntervalSinceReferenceDate
let semaphore = DispatchSemaphore(value: 1)

class ViewController: UIViewController {
    
    @IBOutlet weak var previewView: UIImageView!
    @IBOutlet weak var lineView: UIImageView!
    
    let model = posenet337()
    let targetImageSize = CGSize(width: 337, height: 337)
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    let videoQueue = DispatchQueue(label: "videoQueue")
    let drawQueue = DispatchQueue(label: "drawQueue")
    var captureSession = AVCaptureSession()
    var captureDevice: AVCaptureDevice?
    let videoOutput = AVCaptureVideoDataOutput()
    var isWriting : Bool = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        previewView.frame = UIScreen.main.bounds
        previewView.contentMode = .scaleAspectFit
        
        if (isXcode){
            let fname = "tennis_in_crowd.jpg"
            if let image = UIImage(named: fname)?.resize(to: targetImageSize){
                previewView.image = image
                let result = measure(
                    runCoreML(
                        image.pixelBuffer(width: Int(targetImageSize.width), height: Int(image.size.height))!
                    )
                )
                print(result.duration)
                drawResults(result.result)
            }
        } else {
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewView.layer.addSublayer(previewLayer)
        }
        
    }
    
    override func viewDidAppear(_ animated: Bool) {
        if (!isXcode){
            setupCamera()
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        if (!isXcode){
            previewLayer.frame = previewView.bounds;
            lineView.frame = previewView.bounds;
        }
    }
    
    func setupCamera() {
        let deviceDiscovery = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back)

        if let device = deviceDiscovery.devices.last {
            captureDevice = device
            beginSession()
        }
    }
    
    func beginSession() {
        do {
            videoOutput.videoSettings = [((kCVPixelBufferPixelFormatTypeKey as NSString) as String) : (NSNumber(value: kCVPixelFormatType_32BGRA) as! UInt32)]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            
            if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiom.phone) {
//                captureSession.sessionPreset = .hd1920x1080
                captureSession.sessionPreset = .photo
            } else if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiom.pad) {
                captureSession.sessionPreset = .photo
            }
            
            captureSession.addOutput(videoOutput)
            
            let input = try AVCaptureDeviceInput(device: captureDevice!)
            captureSession.addInput(input)
            
            captureSession.startRunning()
        } catch {
            print("error connecting to capture device")
        }
    }
    
    func drawResults(_ poses: [Pose]){
        
        let minPoseConfidence: Float = 0.5
        
        let screen = UIScreen.main.bounds
        let scale = screen.width / self.targetImageSize.width
        let size = AVMakeRect(aspectRatio: self.targetImageSize,
                              insideRect: self.previewView.frame)
        
        var linePath = UIBezierPath()
        var arcPath = UIBezierPath()
        poses.forEach { pose in
            if (pose.score >= minPoseConfidence){
                self.drawKeypoints(arcPath: &arcPath, keypoints: pose.keypoints,minConfidence: minPoseConfidence,
                                   size: size.origin, scale: scale)
                self.drawSkeleton(linePath: &linePath, keypoints: pose.keypoints,
                                  minConfidence: minPoseConfidence,
                                  size: size.origin, scale: scale)
            }
        }
        
        // Draw
        let arcLine = CAShapeLayer()
        arcLine.path = arcPath.cgPath
        arcLine.strokeColor = UIColor.green.cgColor
        
        let line = CAShapeLayer()
        line.path = linePath.cgPath
        line.strokeColor = UIColor.red.cgColor
        line.lineWidth = 2
        line.lineJoin = kCALineJoinRound
        
        self.lineView.layer.sublayers = nil
        self.lineView.layer.addSublayer(arcLine)
        self.lineView.layer.addSublayer(line)
        linePath.removeAllPoints()
        arcPath.removeAllPoints()
        semaphore.wait()
        isWriting = false
        semaphore.signal()
        
    }
    
    func drawKeypoints(arcPath: inout UIBezierPath, keypoints: [Keypoint], minConfidence: Float,
                       size: CGPoint,scale: CGFloat = 1){
        
        keypoints.forEach { keypoint in
            if (keypoint.score < minConfidence) {
                return
            }
            let center = CGPoint(x: CGFloat(keypoint.position.x) * scale + size.x,
                                 y: CGFloat(keypoint.position.y) * scale + size.y)
            let trackPath = UIBezierPath(arcCenter: center,
                                         radius: 3, startAngle: 0,
                                         endAngle: 2.0 * .pi, clockwise: true)
            
            arcPath.append(trackPath)
        }
    }
    
    func drawSegment(linePath: inout UIBezierPath,fromPoint start: CGPoint, toPoint end:CGPoint,
                     size: CGPoint, scale: CGFloat = 1) {
        
        let newlinePath = UIBezierPath()
        newlinePath.move(to:
            CGPoint(x: start.x * scale + size.x, y: start.y * scale + size.y))
        newlinePath.addLine(to:
            CGPoint(x: end.x * scale + size.x, y: end.y * scale + size.y))
        linePath.append(newlinePath)
    }
    func drawSkeleton(linePath: inout UIBezierPath,keypoints: [Keypoint], minConfidence: Float,
                      size: CGPoint, scale: CGFloat = 1){
        let adjacentKeyPoints = getAdjacentKeyPoints(
            keypoints: keypoints, minConfidence: minConfidence);
        
        adjacentKeyPoints.forEach { keypoint in
            drawSegment(linePath: &linePath,
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
        
        return connectedPartIndeces.reduce(into: [[Keypoint]]()) {
            if (!eitherPointDoesntMeetConfidence(
                keypoints[$1.0].score,
                keypoints[$1.1].score,
                minConfidence)){
                $0.append([keypoints[$1.0],keypoints[$1.1]])
            }
        }
    }
    
    func runCoreML(_ img: CVPixelBuffer) -> [Pose]{
        
        let result = try? model.prediction(image__0: img)
        
        let tensors = result?.featureNames.reduce(into: [String: Tensor]()) {
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
        //        print(sum)!
        let poses = posnet.decodeMultiplePoses(
            scores: tensors!["heatmap__0"]!,
            offsets: tensors!["offset_2__0"]!,
            displacementsFwd: tensors!["displacement_fwd_2__0"]!,
            displacementsBwd: tensors!["displacement_bwd_2__0"]!,
            outputStride: 16, maxPoseDetections: 15,
            scoreThreshold: 0.5,nmsRadius: 20)
        
        return poses
    }
    
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // called for each frame of video
    func captureOutput(_ captureOutput: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        let currentDate = NSDate.timeIntervalSinceReferenceDate
        
        // control the pace of the machine vision to protect battery life
        if currentDate - lastAnalysis >= pace {
            lastAnalysis = currentDate
        } else {
            return // don't run the classifier more often than we need
        }
        
        // keep track of performance and log the frame rate
        if trackPerformance {
            frameCount = frameCount + 1
            if frameCount % framesPerSample == 0 {
                let diff = currentDate - startDate
                if (diff > 0) {
                    if pace > 0.0 {
                        print("WARNING: Frame rate of image classification is being limited by \"pace\" setting. Set to 0.0 for fastest possible rate.")
                    }
                    print("\(String.localizedStringWithFormat("%0.2f", (diff/Double(framesPerSample))))s per frame (average)")
                }
                startDate = currentDate
            }
        }
        
//        DispatchQueue.global(qos: .default).async {
        drawQueue.async {
            semaphore.wait()
            if (self.isWriting == false) {
                self.isWriting = true
                semaphore.signal()
                let startTime = CFAbsoluteTimeGetCurrent()
                guard let croppedBuffer = croppedSampleBuffer(sampleBuffer, targetSize: self.targetImageSize) else {
                    return
                }
                let poses = self.runCoreML(croppedBuffer)
                DispatchQueue.main.sync {
                    self.drawResults(poses)
                }
                let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print ("Elapsed time is \(timeElapsed) seconds.")
            } else {
                semaphore.signal()
            }
        }
    }
}

let context = CIContext()
var rotateTransform: CGAffineTransform?
var scaleTransform: CGAffineTransform?
var cropTransform: CGAffineTransform?
var resultBuffer: CVPixelBuffer?

func croppedSampleBuffer(_ sampleBuffer: CMSampleBuffer, targetSize: CGSize) -> CVPixelBuffer? {
    
    guard let imageBuffer: CVImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        fatalError("Can't convert to CVImageBuffer.")
    }
    
    // Only doing these calculations once for efficiency.
    // If the incoming images could change orientation or size during a session, this would need to be reset when that happens.
    if rotateTransform == nil {
        let imageSize = CVImageBufferGetEncodedSize(imageBuffer)
        let rotatedSize = CGSize(width: imageSize.height, height: imageSize.width)
        
        guard targetSize.width < rotatedSize.width, targetSize.height < rotatedSize.height else {
            fatalError("Captured image is smaller than image size for model.")
        }
        
        let shorterSize = (rotatedSize.width < rotatedSize.height) ? rotatedSize.width : rotatedSize.height
        rotateTransform = CGAffineTransform(translationX: imageSize.width / 2.0, y: imageSize.height / 2.0).rotated(by: -CGFloat.pi / 2.0).translatedBy(x: -imageSize.height / 2.0, y: -imageSize.width / 2.0)
        
        let scale = targetSize.width / shorterSize
        scaleTransform = CGAffineTransform(scaleX: scale, y: scale)
        
        // Crop input image to output size
        let xDiff = rotatedSize.width * scale - targetSize.width
        let yDiff = rotatedSize.height * scale - targetSize.height
        cropTransform = CGAffineTransform(translationX: xDiff/2.0, y: yDiff/2.0)
    }
    
    // Convert to CIImage because it is easier to manipulate
    let ciImage = CIImage(cvImageBuffer: imageBuffer)
    let rotated = ciImage.transformed(by: rotateTransform!)
    let scaled = rotated.transformed(by: scaleTransform!)
    let cropped = scaled.transformed(by: cropTransform!)
    
    // Note that the above pipeline could be easily appended with other image manipulations.
    // For example, to change the image contrast. It would be most efficient to handle all of
    // the image manipulation in a single Core Image pipeline because it can be hardware optimized.
    
    // Only need to create this buffer one time and then we can reuse it for every frame
    if resultBuffer == nil {
        let result = CVPixelBufferCreate(kCFAllocatorDefault, Int(targetSize.width), Int(targetSize.height), kCVPixelFormatType_32BGRA, nil, &resultBuffer)
        
        guard result == kCVReturnSuccess else {
            fatalError("Can't allocate pixel buffer.")
        }
    }
    
    // Render the Core Image pipeline to the buffer
    context.render(cropped, to: resultBuffer!)
    
    //  For debugging
    //  let image = imageBufferToUIImage(resultBuffer!)
    //  print(image.size) // set breakpoint to see image being provided to CoreML
    
    return resultBuffer
}

