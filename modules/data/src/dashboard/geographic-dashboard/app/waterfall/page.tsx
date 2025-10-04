'use client'

import { useState, useEffect, useRef } from 'react'
import WaterfallDisplay from '../components/WaterfallDisplay'
import QASampleSearch from '../components/QASampleSearch'
import { QASample } from '../types'

export default function WaterfallViewer() {
  const [selectedSample, setSelectedSample] = useState<QASample | null>(null)
  const [samples, setSamples] = useState<QASample[]>([])
  const [loading, setLoading] = useState(false)
  const [searchFilters, setSearchFilters] = useState({
    band: '',
    callsignHash: '',
    startDate: '',
    endDate: '',
    propagationMode: '',
    minSNR: 0
  })

  // Fetch QA samples
  const fetchSamples = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      Object.entries(searchFilters).forEach(([key, value]) => {
        if (value) params.append(key, String(value))
      })

      const response = await fetch(`/api/qa/search?${params}`)
      const data = await response.json()
      setSamples(data.samples || [])
    } catch (error) {
      console.error('Error fetching samples:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSamples()
  }, [])

  return (
    <div className="min-h-screen bg-cascade-dark p-6">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          QA Waterfall Viewer
        </h1>
        <p className="text-gray-300">
          Visualize and analyze 1% QA samples from CASCADE collection (T051)
        </p>
      </header>

      {/* Search and Filter Section */}
      <div className="mb-8">
        <QASampleSearch
          filters={searchFilters}
          onFiltersChange={setSearchFilters}
          onSearch={fetchSamples}
          loading={loading}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Waterfall Display (2/3 width) */}
        <div className="xl:col-span-2">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-semibold text-white mb-4">
              Waterfall Display
            </h2>
            {selectedSample ? (
              <WaterfallDisplay sample={selectedSample} />
            ) : (
              <div className="h-96 bg-gray-900 rounded flex items-center justify-center text-gray-500">
                Select a QA sample to view waterfall
              </div>
            )}
          </div>
        </div>

        {/* Sample List (1/3 width) */}
        <div className="xl:col-span-1">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-semibold text-white mb-4">
              QA Samples ({samples.length})
            </h2>
            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {loading ? (
                <div className="text-gray-400 animate-pulse">Loading samples...</div>
              ) : samples.length > 0 ? (
                samples.map((sample) => (
                  <SampleCard
                    key={sample.id}
                    sample={sample}
                    isSelected={selectedSample?.id === sample.id}
                    onClick={() => setSelectedSample(sample)}
                  />
                ))
              ) : (
                <div className="text-gray-400">No samples found</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Sample Details */}
      {selectedSample && (
        <div className="mt-8 bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-white mb-4">
            Sample Details
          </h2>
          <SampleDetails sample={selectedSample} />
        </div>
      )}
    </div>
  )
}

function SampleCard({
  sample,
  isSelected,
  onClick
}: {
  sample: QASample
  isSelected: boolean
  onClick: () => void
}) {
  return (
    <div
      className={`p-3 rounded-lg cursor-pointer transition-all ${
        isSelected
          ? 'bg-cascade-blue text-white'
          : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
      }`}
      onClick={onClick}
    >
      <div className="flex justify-between items-start">
        <div>
          <p className="font-medium">{sample.band} - {sample.mode}</p>
          <p className="text-sm opacity-75">
            {new Date(sample.timestamp).toLocaleString()}
          </p>
          <p className="text-xs mt-1">
            SNR: {sample.snr?.toFixed(1)} dB | {sample.duration}s
          </p>
        </div>
        {sample.hasWaterfall && (
          <span className="text-xs bg-green-500 bg-opacity-20 text-green-300 px-2 py-1 rounded">
            Waterfall
          </span>
        )}
      </div>
    </div>
  )
}

function SampleDetails({ sample }: { sample: QASample }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <DetailItem label="Session ID" value={sample.sessionId} />
      <DetailItem label="Frequency" value={`${sample.frequencyKhz} kHz`} />
      <DetailItem label="Band" value={sample.band} />
      <DetailItem label="Mode" value={sample.mode || 'Unknown'} />
      <DetailItem label="Sample Rate" value={`${sample.sampleRate} Hz`} />
      <DetailItem label="Duration" value={`${sample.duration} seconds`} />
      <DetailItem label="Grid Square" value={sample.gridSquare || 'N/A'} />
      <DetailItem label="Correlation ID" value={sample.correlationId || 'N/A'} />
      <DetailItem label="SNR" value={sample.snr ? `${sample.snr.toFixed(1)} dB` : 'N/A'} />
      <DetailItem label="Propagation Mode" value={sample.propagationMode || 'N/A'} />
      <DetailItem label="Space Weather" value={sample.spaceWeatherCondition || 'Normal'} />
      <DetailItem label="File Size" value={`${(sample.fileSizeBytes / 1024 / 1024).toFixed(1)} MB`} />
    </div>
  )
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-gray-400 text-sm">{label}</p>
      <p className="text-white font-medium">{value}</p>
    </div>
  )
}