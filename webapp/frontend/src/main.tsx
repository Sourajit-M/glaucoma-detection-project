import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'

import Predict   from './pages/Predict'
import Dashboard from './pages/Dashboard'
import About     from './pages/About'
import Navbar    from './components/Navbar'

const queryClient = new QueryClient()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/"          element={<Predict />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/about"     element={<About />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>
)