'use client'

import { useEffect, useRef, useState } from 'react'
import { QASample } from '../types'

interface Props {
  sample: QASample
}

export default function WaterfallDisplay({ sample }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [colormap, setColormap] = useState('viridis')
  const [waterfallData, setWaterfallData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [timePosition, setTimePosition] = useState(0)

  // Measurement cursors (FR-045, FR-046)
  const [cursor1, setCursor1] = useState<{ x: number; y: number } | null>(null)
  const [cursor2, setCursor2] = useState<{ x: number; y: number } | null>(null)
  const [isPlacingCursor, setIsPlacingCursor] = useState<1 | 2 | null>(null)
  const [dynamicRange, setDynamicRange] = useState(60) // dB range (FR-046 minimum)

  useEffect(() => {
    // Fetch waterfall data when sample changes
    fetchWaterfallData()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [sample])

  const fetchWaterfallData = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/waterfall', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recording_id: sample.sessionId,
          start_time: 0,
          duration: Math.min(sample.duration, 10), // Max 10 seconds
          fft_size: 1024,
          overlap: 0.5,
          colormap
        })
      })

      const data = await response.json()
      setWaterfallData(data.waterfall)
      drawWaterfall(data.waterfall)
    } catch (error) {
      console.error('Error fetching waterfall:', error)
    } finally {
      setLoading(false)
    }
  }

  const drawWaterfall = (data: any) => {
    if (!canvasRef.current || !data) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw waterfall data
    const imageData = ctx.createImageData(canvas.width, canvas.height)

    // Map waterfall data to canvas pixels
    const { data: waterfallArray, frequencies, timestamps } = data
    const numFreqBins = waterfallArray.length
    const numTimeBins = waterfallArray[0].length

    const freqScale = canvas.height / numFreqBins
    const timeScale = canvas.width / numTimeBins

    for (let f = 0; f < numFreqBins; f++) {
      for (let t = 0; t < numTimeBins; t++) {
        const value = waterfallArray[f][t]
        const color = getColor(value, colormap)

        const x = Math.floor(t * timeScale * zoom)
        const y = Math.floor((numFreqBins - f - 1) * freqScale)

        // Draw scaled pixel
        for (let dx = 0; dx < Math.ceil(timeScale * zoom); dx++) {
          for (let dy = 0; dy < Math.ceil(freqScale); dy++) {
            const px = x + dx
            const py = y + dy

            if (px < canvas.width && py < canvas.height) {
              const index = (py * canvas.width + px) * 4
              imageData.data[index] = color.r
              imageData.data[index + 1] = color.g
              imageData.data[index + 2] = color.b
              imageData.data[index + 3] = 255
            }
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0)

    // Draw axes and labels
    drawAxes(ctx, frequencies, timestamps)
  }

  const getColor = (value: number, colormap: string): { r: number; g: number; b: number } => {
    // Normalize value to 0-1
    const normalized = Math.max(0, Math.min(1, value))

    // Simple colormap implementations
    switch (colormap) {
      case 'viridis':
        return {
          r: Math.floor(68 + (253 - 68) * normalized),
          g: Math.floor(1 + (231 - 1) * normalized),
          b: Math.floor(84 + (37 - 84) * normalized)
        }
      case 'jet':
        if (normalized < 0.25) {
          return { r: 0, g: 0, b: Math.floor(128 + 127 * normalized * 4) }
        } else if (normalized < 0.5) {
          return { r: 0, g: Math.floor(255 * (normalized - 0.25) * 4), b: 255 }
        } else if (normalized < 0.75) {
          return { r: Math.floor(255 * (normalized - 0.5) * 4), g: 255, b: Math.floor(255 * (1 - (normalized - 0.5) * 4)) }
        } else {
          return { r: 255, g: Math.floor(255 * (1 - (normalized - 0.75) * 4)), b: 0 }
        }
      case 'hot':
        return {
          r: Math.floor(255 * Math.min(1, normalized * 3)),
          g: Math.floor(255 * Math.max(0, Math.min(1, (normalized - 0.33) * 3))),
          b: Math.floor(255 * Math.max(0, (normalized - 0.67) * 3))
        }
      case 'cool':
        return {
          r: Math.floor(normalized * 255),
          g: Math.floor(255 * (1 - normalized)),
          b: 255
        }
      default:
        return { r: normalized * 255, g: normalized * 255, b: normalized * 255 }
    }
  }

  const drawAxes = (ctx: CanvasRenderingContext2D, frequencies: number[], timestamps: number[]) => {
    ctx.strokeStyle = '#666'
    ctx.lineWidth = 1
    ctx.font = '10px monospace'
    ctx.fillStyle = '#999'

    // Frequency axis (vertical)
    const minFreq = Math.min(...frequencies) / 1000
    const maxFreq = Math.max(...frequencies) / 1000
    const freqRange = maxFreq - minFreq

    for (let i = 0; i <= 5; i++) {
      const freq = minFreq + (freqRange * i) / 5
      const y = ctx.canvas.height - (ctx.canvas.height * i) / 5

      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(10, y)
      ctx.stroke()

      ctx.fillText(`${freq.toFixed(1)} kHz`, 15, y + 3)
    }

    // Time axis (horizontal)
    const maxTime = Math.max(...timestamps)
    for (let i = 0; i <= 5; i++) {
      const time = (maxTime * i) / 5
      const x = (ctx.canvas.width * i) / 5

      ctx.beginPath()
      ctx.moveTo(x, ctx.canvas.height)
      ctx.lineTo(x, ctx.canvas.height - 10)
      ctx.stroke()

      ctx.fillText(`${time.toFixed(1)}s`, x - 10, ctx.canvas.height - 15)
    }
  }

  const handlePlayPause = () => {
    if (isPlaying) {
      setIsPlaying(false)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    } else {
      setIsPlaying(true)
      animateWaterfall()
    }
  }

  const animateWaterfall = () => {
    if (!waterfallData) return

    const animate = () => {
      setTimePosition((prev) => {
        const next = prev + 0.1
        if (next >= sample.duration) {
          setIsPlaying(false)
          return 0
        }
        return next
      })

      if (isPlaying) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animationRef.current = requestAnimationFrame(animate)
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4">
        <button
          onClick={handlePlayPause}
          className="bg-cascade-accent hover:bg-cascade-blue text-white px-4 py-2 rounded transition-colors"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
            className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded"
          >
            Zoom -
          </button>
          <span className="text-white">{(zoom * 100).toFixed(0)}%</span>
          <button
            onClick={() => setZoom(Math.min(2, zoom + 0.1))}
            className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded"
          >
            Zoom +
          </button>
        </div>

        {/* Measurement cursor controls (FR-045) */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlacingCursor(1)}
            className={`px-3 py-1 rounded ${
              isPlacingCursor === 1
                ? 'bg-cascade-accent text-white'
                : 'bg-gray-700 hover:bg-gray-600 text-white'
            }`}
          >
            Cursor 1
          </button>
          <button
            onClick={() => setIsPlacingCursor(2)}
            className={`px-3 py-1 rounded ${
              isPlacingCursor === 2
                ? 'bg-cascade-accent text-white'
                : 'bg-gray-700 hover:bg-gray-600 text-white'
            }`}
          >
            Cursor 2
          </button>
          {cursor1 && cursor2 && (
            <span className="text-cascade-accent font-mono text-sm">
              Δf: {Math.abs(cursor1.y - cursor2.y).toFixed(1)} kHz,
              Δt: {Math.abs(cursor1.x - cursor2.x).toFixed(1)} s
            </span>
          )}
        </div>

        {/* Dynamic range control (FR-046) */}
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-sm">Dynamic Range:</span>
          <select
            value={dynamicRange}
            onChange={(e) => setDynamicRange(Number(e.target.value))}
            className="bg-gray-700 text-white px-2 py-1 rounded text-sm"
          >
            <option value="30">30 dB</option>
            <option value="60">60 dB</option>
            <option value="90">90 dB</option>
          </select>
        </div>

        <select
          value={colormap}
          onChange={(e) => {
            setColormap(e.target.value)
            if (waterfallData) drawWaterfall(waterfallData)
          }}
          className="bg-gray-700 text-white px-3 py-1 rounded"
        >
          <option value="viridis">Viridis</option>
          <option value="jet">Jet</option>
          <option value="hot">Hot</option>
          <option value="cool">Cool</option>
        </select>

        {isPlaying && (
          <div className="flex items-center gap-2">
            <span className="text-gray-400">Time:</span>
            <span className="text-white font-mono">
              {timePosition.toFixed(1)}s / {sample.duration}s
            </span>
          </div>
        )}
      </div>

      {/* Waterfall Canvas */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div className="text-white animate-pulse">Loading waterfall...</div>
          </div>
        )}
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          className="w-full h-auto cursor-crosshair"
          onClick={(e) => {
            if (isPlacingCursor && canvasRef.current) {
              const rect = canvasRef.current.getBoundingClientRect()
              const x = ((e.clientX - rect.left) / rect.width) * sample.duration
              const y = ((rect.bottom - e.clientY) / rect.height) * 12 // 12 kHz bandwidth

              if (isPlacingCursor === 1) {
                setCursor1({ x, y })
              } else {
                setCursor2({ x, y })
              }
              setIsPlacingCursor(null)
            }
          }}
        />

        {/* Draw measurement cursors */}
        {cursor1 && (
          <div
            className="absolute border-l-2 border-yellow-400 pointer-events-none"
            style={{
              left: `${(cursor1.x / sample.duration) * 100}%`,
              top: 0,
              bottom: 0,
            }}
          >
            <div className="text-yellow-400 text-xs ml-1">
              C1: {cursor1.x.toFixed(2)}s, {(sample.frequencyKhz - 6 + cursor1.y).toFixed(1)}kHz
            </div>
          </div>
        )}
        {cursor2 && (
          <div
            className="absolute border-l-2 border-cyan-400 pointer-events-none"
            style={{
              left: `${(cursor2.x / sample.duration) * 100}%`,
              top: 0,
              bottom: 0,
            }}
          >
            <div className="text-cyan-400 text-xs ml-1">
              C2: {cursor2.x.toFixed(2)}s, {(sample.frequencyKhz - 6 + cursor2.y).toFixed(1)}kHz
            </div>
          </div>
        )}

        {/* Frequency scale on left */}
        <div className="absolute left-0 top-0 h-full w-16 bg-gradient-to-r from-gray-900 to-transparent pointer-events-none" />

        {/* Time scale on bottom */}
        <div className="absolute bottom-0 left-0 w-full h-8 bg-gradient-to-t from-gray-900 to-transparent pointer-events-none" />
      </div>

      {/* Info Bar */}
      <div className="flex justify-between text-sm text-gray-400">
        <span>
          Center: <span className="text-white">{sample.frequencyKhz}</span> kHz
        </span>
        <span>
          Bandwidth: <span className="text-white">12</span> kHz
        </span>
        <span>
          Sample Rate: <span className="text-white">{sample.sampleRate}</span> Hz
        </span>
        <span>
          FFT Size: <span className="text-white">1024</span>
        </span>
      </div>
    </div>
  )
}