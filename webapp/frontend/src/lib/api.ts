import axios from 'axios'

const apiBaseUrl = import.meta.env.VITE_API_URL?.trim() || '/api'

export const apiClient = axios.create({
    baseURL: apiBaseUrl,
    timeout: 90000,
})

