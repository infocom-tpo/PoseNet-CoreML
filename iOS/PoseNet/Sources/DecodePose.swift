import TensorSwift

extension PoseNet {

    func getDisplacement(_ i: Int,_ point: Vector2DInt,_ displacements: Tensor) -> Vector2D {
        let numEdges = Int(displacements.shape.dimensions[2].value / 2)
        return Vector2D(
            x: displacements[point.y, point.x, numEdges + i],
            y: displacements[point.y, point.x, i]
        )
    }

    func decode(
        _ point: Vector2D,_  outputStride: Int,_ height: Int,
        _ width: Int) -> Vector2DInt {
        
        return Vector2DInt(
            x: Int((Float(point.x) / Float(outputStride)).rounded()).clamp(0,width-1),
            y: Int((Float(point.y) / Float(outputStride)).rounded()).clamp(0,height-1)
        )
    }

    /**
     * We get a new keypoint along the `edgeId` for the pose instance, assuming
     * that the position of the `idSource` part is already known. For this, we
     * follow the displacement vector from the source to target part (stored in
     * the `i`-t channel of the displacement tensor).
     */
    func traverseToTargetKeypoint(
        _ edgeId: Int, _ sourceKeypoint: Keypoint,_ targetKeypointId: Int,
        _ scoresBuffer: Tensor,_ offsets: Tensor,_ outputStride: Int,
        _ displacements: Tensor) -> Keypoint {
        let height = scoresBuffer.shape.dimensions[0].value
        let width = scoresBuffer.shape.dimensions[1].value
        
        // Nearest neighbor interpolation for the source->target displacements.
        let sourceKeypointIndeces =
            decode(sourceKeypoint.position, outputStride, height, width)
        
        let displacement =
            getDisplacement(edgeId, sourceKeypointIndeces, displacements)
        
        let displacedPoint = addVectors(sourceKeypoint.position, displacement)
        
        let displacedPointIndeces =
            decode(displacedPoint, outputStride, height, width)
        
        let offsetPoint = getOffsetPoint(
            y: displacedPointIndeces.y,x: displacedPointIndeces.x,
            keypoint: targetKeypointId, offsets: offsets)
        
        let targetKeypoint =
            addVectors(displacedPoint, Vector2D(x: offsetPoint.x, y: offsetPoint.y))
        
        let targetKeypointIndeces =
            decode(targetKeypoint, outputStride, height, width)
        
        let score = scoresBuffer[targetKeypointIndeces.y, targetKeypointIndeces.x, targetKeypointId]
        
        return Keypoint(score: score, position: targetKeypoint, part: partNames[targetKeypointId])
    }

    /**
     * Follows the displacement fields to decode the full pose of the object
     * instance given the position of a part that acts as root.
     *
     * @return An array of decoded keypoints and their scores for a single pose
     */
    func decodePose(
        _ root: PartWithScore,_ scores: Tensor,_ offsets: Tensor,
        _ outputStride: Int,_ displacementsFwd: Tensor,
        _ displacementsBwd: Tensor) -> [Keypoint] {
        
        let numParts = scores.shape.dimensions[2].value
        let numEdges = parentToChildEdges.count
        
        var instanceKeypoints = [Keypoint](
            repeating: Keypoint(score: 0,position: Vector2D(x:0,y:0),part: ""),count: numParts)
        
        // Start a new detection instance at the position of the root.
        let rootPart = root.part, rootScore = root.score
        let rootPoint = getImageCoords(part: rootPart,outputStride: outputStride,offsets: offsets)
        
        instanceKeypoints[rootPart.id] = Keypoint(
            score: rootScore,
            position: rootPoint,
            part: partNames[rootPart.id]
        )
        
    //    print(instanceKeypoints.count)
       
        // Decode the part positions upwards in the tree, following the backward
        // displacements.
        for edge in (0..<numEdges).reversed() {
            let sourceKeypointId = parentToChildEdges[edge]
            let targetKeypointId = childToParentEdges[edge]
            if (instanceKeypoints[sourceKeypointId].score > 0.0 &&
                instanceKeypoints[targetKeypointId].score == 0.0) {
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
                    edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                    offsets, outputStride, displacementsBwd)
            }
        }

        // Decode the part positions downwards in the tree, following the forward
        // displacements.
        for edge in 0..<numEdges {
            let sourceKeypointId = childToParentEdges[edge]
            let targetKeypointId = parentToChildEdges[edge]
            if (instanceKeypoints[sourceKeypointId].score > 0.0 &&
                instanceKeypoints[targetKeypointId].score == 0.0) {
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
                    edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                    offsets, outputStride, displacementsFwd)
            }
        }

        return instanceKeypoints
    }
}
