'use client'

import { useEffect, useRef } from 'react'
import { GeographicData } from '../types'

interface Props {
  data: GeographicData | null
}

export default function GeographicHeatmap({ data }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!data || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw world map grid (simplified)
    ctx.strokeStyle = '#444'
    ctx.lineWidth = 1

    // Draw latitude lines
    for (let lat = -90; lat <= 90; lat += 30) {
      const y = ((90 - lat) / 180) * canvas.height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }

    // Draw longitude lines
    for (let lon = -180; lon <= 180; lon += 60) {
      const x = ((lon + 180) / 360) * canvas.width
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
    }

    // Draw heatmap points
    data.gridSquareData.forEach(point => {
      const x = ((point.longitude + 180) / 360) * canvas.width
      const y = ((90 - point.latitude) / 180) * canvas.height

      // Calculate color based on intensity
      const intensity = Math.min(point.intensity, 1)
      const r = Math.floor(255 * intensity)
      const g = Math.floor(100 * (1 - intensity))
      const b = 50

      // Draw gradient circle
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20)
      gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.8)`)
      gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)

      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(x, y, 20, 0, 2 * Math.PI)
      ctx.fill()
    })

    // Draw latitude band indicators
    const bands = [
      { name: 'Arctic', lat: 66.5, color: '#00D4FF' },
      { name: 'Temperate N', lat: 23.5, color: '#00C851' },
      { name: 'Tropical', lat: 0, color: '#FFD700' },
      { name: 'Temperate S', lat: -23.5, color: '#00C851' },
      { name: 'Antarctic', lat: -66.5, color: '#00D4FF' },
    ]

    bands.forEach(band => {
      const y = ((90 - band.lat) / 180) * canvas.height
      ctx.strokeStyle = band.color
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
      ctx.setLineDash([])

      // Label
      ctx.fillStyle = band.color
      ctx.font = '12px sans-serif'
      ctx.fillText(band.name, 10, y - 5)
    })

  }, [data])

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        className="w-full h-auto rounded-lg bg-gray-900"
      />
      {data && (
        <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
          <div className="text-gray-300">
            <span className="text-cascade-accent">●</span> High density regions
          </div>
          <div className="text-gray-300">
            <span className="text-yellow-500">●</span> Medium density regions
          </div>
          <div className="text-gray-300">
            <span className="text-red-500">●</span> Low density / gaps
          </div>
          <div className="text-gray-300">
            <span className="text-gray-500">---</span> Latitude bands
          </div>
        </div>
      )}
    </div>
  )
}