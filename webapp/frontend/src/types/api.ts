export interface ImageSet{
    original: string
    heatmap_overlay: string
    disc_mask: string
    cup_mask: string
    segmentation_overlay: string
}

export interface PredictionResponse{
    prediction: 'glaucoma' | 'healthy'
    probability: number
    confidence: 'high' | 'medium' | 'low'
    cdr: number
    cdr_risk: 'elevated' | 'borderline' | 'normal'
    processing_time_ms: number
    images: ImageSet
    clinical_note: string
}

export interface ModelMetric{
    name: string
    type: 'classical_ml' | 'deep_learning' | 'hybrid'
    auc: number
    sensitivity: number
    specificity: number
    f1: number
}

export interface AblationEntry{
    variant: string
    auc: number
}

export interface SegmentationEntry {
    structure: string
    dice: number
    iou: number
}

export interface MetricsResponse{
    dataset_info: {
        total_images: number
        test_set_size: number
        datasets: string[]
    }
    models: ModelMetric[]
    ablation: AblationEntry[]
    segmentation: SegmentationEntry[]
}