import TensorSwift

extension Posing {
    
    func withinNmsRadiusOfCorrespondingPoint(
        poses: [Pose], squaredNmsRadius: Float32, vec: Vector2D, keypointId: Int) -> Bool {
        
        let x = vec.x, y = vec.y
        return poses.contains { pose in
            let correspondingKeypoint = pose.keypoints[keypointId].position
            // Binary operator '<=' cannot be applied to operands of type 'Float32' (aka 'Float') and 'Int'
            return squaredDistance(y1: y, x1: x, y2: correspondingKeypoint.y, x2: correspondingKeypoint.x) <= squaredNmsRadius
        }
    }

    func getInstanceScore(
        _ existingPoses: [Pose],_ squaredNmsRadius: Float32,
        _ instanceKeypoints: [Keypoint]) -> Float32 {

        var notOverlappedKeypointScores : Float = 0.0
        for (keypointId, p) in instanceKeypoints.enumerated() {
            if (!withinNmsRadiusOfCorrespondingPoint(
                poses: existingPoses,
                squaredNmsRadius: squaredNmsRadius,
                vec: p.position,
                keypointId: keypointId))
            {
                notOverlappedKeypointScores += Float(p.score)
            }
        }
        return Float(notOverlappedKeypointScores / Float(instanceKeypoints.count))
    }
    func decodeMultiplePoses(
        scores: Tensor, offsets: Tensor,
        displacementsFwd: Tensor, displacementsBwd: Tensor,
        outputStride: Int, maxPoseDetections: Int, scoreThreshold: Float32 = 0.5,
        nmsRadius : Int = 20) -> [Pose] {
        
        var poses: [Pose] = []
        let squaredNmsRadius = Float32(nmsRadius * nmsRadius)
        
        var queue = buildPartWithScoreQueue(scoreThreshold: scoreThreshold,
                                            localMaximumRadius: kLocalMaximumRadius, scores: scores)
        
        while (poses.count < maxPoseDetections && !queue.isEmpty) {
            let root = queue.dequeue()
            //        if (root!.part.heatmapY == 7 && root!.part.heatmapX == 15){
            //            print(root!.part)
            //            print("OK")
            //        }
            //
            let rootImageCoords =
                getImageCoords(part: root!.part,outputStride: outputStride,offsets: offsets)
            
            if (withinNmsRadiusOfCorrespondingPoint(poses: poses,squaredNmsRadius: squaredNmsRadius,
                                                    vec: rootImageCoords,keypointId: root!.part.id))
            {
                continue
            }
            //        print(root!.part)
            //        print(rootImageCoords)
            
            // Start a new detection instance at the position of the root.
            let keypoints = decodePose(
                root!, scores, offsets, outputStride, displacementsFwd,
                displacementsBwd)
            
            let score = getInstanceScore(poses, squaredNmsRadius, keypoints)
            
            poses.append(Pose(keypoints: keypoints, score: score))
            //        print(root!.score)
        }
        //    print(poses.count)
        //    print(poses)
        return poses
    }
}


