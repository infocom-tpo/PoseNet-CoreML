import TensorSwift

extension PoseNet {

    func getDisplacement(_ edgeId: Int,_ point: Vector2DInt,_ displacements: Tensor) -> Vector2D {
        let numEdges = Int(displacements.shape.dimensions[0].value / 2)
        return Vector2D(
            x: displacements[numEdges + edgeId, point.y, point.x],
            y: displacements[edgeId, point.y, point.x]
        )
    }

    func getStridedIndexNearPoint(
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

        let height = scoresBuffer.shape.dimensions[2].value
        let width = scoresBuffer.shape.dimensions[1].value

        // Nearest neighbor interpolation for the source->target displacements.
        let sourceKeypointIndices = getStridedIndexNearPoint(
            sourceKeypoint.position, outputStride, height, width)
        
        let displacement =
            getDisplacement(edgeId, sourceKeypointIndices, displacements)
        
        let displacedPoint = addVectors(sourceKeypoint.position, displacement)
        
        let displacedPointIndices =
            getStridedIndexNearPoint(displacedPoint, outputStride, height, width)
        
        let offsetPoint = getOffsetPoint(
            y: displacedPointIndices.y, x: displacedPointIndices.x,
            keypoint: targetKeypointId, offsets: offsets)
        
        let score = scoresBuffer[targetKeypointId, displacedPointIndices.y, displacedPointIndices.x]
        
        let targetKeypoint =
                addVectors(
                    Vector2D(x: Float(displacedPointIndices.x * outputStride),
                             y: Float(displacedPointIndices.y * outputStride)),
                    Vector2D(x: offsetPoint.x, y: offsetPoint.y))
        
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
        
        let numParts = scores.shape.dimensions[0].value
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
