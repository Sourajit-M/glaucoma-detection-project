// @ts-nocheck
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '../lib/api'
import type { PredictionResponse } from '../types/api'

async function predict(file: File): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await apiClient.post<PredictionResponse>('/predict', form)
  return data
}

export function usePrediction() {
  return useMutation({
    mutationFn: predict,
  })
}