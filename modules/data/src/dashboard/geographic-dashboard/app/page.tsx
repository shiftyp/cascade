'use client'

import { useState, useEffect } from 'react'
import GeographicHeatmap from './components/GeographicHeatmap'
import LatitudeBandProgress from './components/LatitudeBandProgress'
import BiasWarnings from './components/BiasWarnings'
import RebalancingRecommendations from './components/RebalancingRecommendations'
import DiversityMetrics from './components/DiversityMetrics'
import { GeographicData, DiversityMetrics as DiversityMetricsType } from './types'

export default function GeographicDashboard() {
  const [geographicData, setGeographicData] = useState<GeographicData | null>(null)
  const [diversityMetrics, setDiversityMetrics] = useState<DiversityMetricsType | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])
  const [recommendations, setRecommendations] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  // Fetch data every 30 seconds
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch geographic distribution data
        const geoResponse = await fetch('/api/diversity/geographic-data')
        const geoData = await geoResponse.json()
        setGeographicData(geoData)

        // Fetch diversity metrics
        const metricsResponse = await fetch('/api/diversity/metrics')
        const metrics = await metricsResponse.json()
        setDiversityMetrics(metrics)

        // Fetch warnings
        const warningsResponse = await fetch('/api/diversity/warnings')
        const warningsData = await warningsResponse.json()
        setWarnings(warningsData.warnings || [])

        // Fetch recommendations
        const recsResponse = await fetch('/api/diversity/recommendations')
        const recsData = await recsResponse.json()
        setRecommendations(recsData.recommendations || [])

        setLoading(false)
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000) // Update every 30 seconds

    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-cascade-dark flex items-center justify-center">
        <div className="text-white text-2xl animate-pulse">
          Loading Geographic Diversity Dashboard...
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-cascade-dark p-6">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          CASCADE Geographic Diversity Monitor
        </h1>
        <p className="text-gray-300">
          Real-time monitoring of global data collection coverage
        </p>
      </header>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <DiversityMetrics metrics={diversityMetrics} />
      </div>

      {/* Warnings Section */}
      {warnings.length > 0 && (
        <div className="mb-8">
          <BiasWarnings warnings={warnings} />
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-8">
        {/* Geographic Heatmap */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-white mb-4">
            Geographic Distribution Heatmap (T087a)
          </h2>
          <GeographicHeatmap data={geographicData} />
        </div>

        {/* Latitude Band Progress */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-white mb-4">
            Latitude Band Collection Progress (T087b)
          </h2>
          <LatitudeBandProgress data={geographicData} />
        </div>
      </div>

      {/* Recommendations Section */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-semibold text-white mb-4">
          Automatic Rebalancing Recommendations (T087d)
        </h2>
        <RebalancingRecommendations recommendations={recommendations} />
      </div>

      {/* Status Bar */}
      <footer className="mt-8 pt-6 border-t border-gray-700">
        <div className="flex justify-between text-gray-400 text-sm">
          <span>Last updated: {new Date().toLocaleTimeString()}</span>
          <span>Auto-refresh: 30 seconds</span>
          <span className="text-cascade-accent">
            {geographicData?.totalHours.toFixed(0)} total hours collected
          </span>
        </div>
      </footer>
    </div>
  )
}