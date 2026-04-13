// @ts-nocheck
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'
import { apiClient } from '../lib/api'
import type { PredictionResponse } from '../types/api'

const RETRY_DELAY_MS = 4000
const MAX_ATTEMPTS = 3

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitForBackendWakeup() {
  try {
    await apiClient.get('/health', { timeout: 20000 })
  } catch {
    // Ignore warmup failures here. The actual prediction request below
    // still gets retried because Render cold starts often fail the first probe.
  }
}

async function predict(file: File): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file)

  await waitForBackendWakeup()

  let lastError: unknown

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    try {
      const { data } = await apiClient.post<PredictionResponse>('/predict', form, {
        timeout: 90000,
      })
      return data
    } catch (error) {
      lastError = error

      const isAxiosError = axios.isAxiosError(error)
      const status = error?.response?.status
      const isRetryable =
        !isAxiosError ||
        !error.response ||
        status === 502 ||
        status === 503 ||
        status === 504

      if (!isRetryable || attempt === MAX_ATTEMPTS) {
        throw error
      }

      await sleep(RETRY_DELAY_MS)
    }
  }

  throw lastError
}

export function usePrediction() {
  return useMutation({
    mutationFn: predict,
  })
}
