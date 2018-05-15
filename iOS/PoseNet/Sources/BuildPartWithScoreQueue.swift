import TensorSwift


private func > (m1: PartWithScore, m2: PartWithScore) -> Bool {
    return m1.score > m2.score
}

extension PoseNet {
    
    func buildPartWithScoreQueue(
        scoreThreshold: Float, localMaximumRadius: Int,
        scores: Tensor) -> PriorityQueue<PartWithScore> {
        
        var queue = PriorityQueue<PartWithScore>(sort: >)
        
        let height = scores.shape.dimensions[1].value
        let width = scores.shape.dimensions[2].value
        let numKeypoints = scores.shape.dimensions[0].value
        
        for heatmapY in 0..<height {
            for heatmapX in 0..<width {
                for keypointId in 0..<numKeypoints {
                    
                    let score = scores[keypointId, heatmapY ,heatmapX]
                    if (score < scoreThreshold) {
                        continue
                    }
                    
                    if (scoreIsMaximumInLocalWindow(
                        keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
                        
                        queue.enqueue(
                            PartWithScore(score: score,
                                          part: Part(heatmapX: heatmapX, heatmapY: heatmapY, id: keypointId))
                        )
                    }
                }
            }
        }
        return queue
    }

    func scoreIsMaximumInLocalWindow(
        _ keypointId: Int,_ score: Float,_ heatmapY: Int,_ heatmapX: Int,
        _ localMaximumRadius: Int,_ scores: Tensor) -> Bool {
        let height = scores.shape.dimensions[1].value
        let width = scores.shape.dimensions[2].value
        
        var localMaximum = true
        let yStart = max(heatmapY - localMaximumRadius, 0)
        let yEnd = min(heatmapY + localMaximumRadius + 1, height)
        for yCurrent in yStart..<yEnd {
            let xStart = max(heatmapX - localMaximumRadius, 0)
            let xEnd = min(heatmapX + localMaximumRadius + 1, width)
            for xCurrent in xStart..<xEnd {
                if (scores[keypointId, yCurrent, xCurrent] > score) {
                    localMaximum = false
                    break
                }
            }
            if (!localMaximum) {
                break
            }
        }
        return localMaximum
    }
}
