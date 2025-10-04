'use client'

import { DiversityMetrics as DiversityMetricsType } from '../types'

interface Props {
  metrics: DiversityMetricsType | null
}

export default function DiversityMetrics({ metrics }: Props) {
  if (!metrics) {
    return (
      <div className="col-span-3 bg-gray-800 rounded-lg p-6">
        <p className="text-gray-400">Loading metrics...</p>
      </div>
    )
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-success-green'
    if (score >= 0.6) return 'text-warning-yellow'
    return 'text-warning-red'
  }

  const continents = [
    { key: 'northAmerica', label: 'N. America' },
    { key: 'southAmerica', label: 'S. America' },
    { key: 'europe', label: 'Europe' },
    { key: 'africa', label: 'Africa' },
    { key: 'asia', label: 'Asia' },
    { key: 'oceania', label: 'Oceania' },
    { key: 'antarctica', label: 'Antarctica' },
  ]

  const coveredCount = Object.values(metrics.continentalCoverage).filter(Boolean).length

  return (
    <>
      {/* Overall Diversity Score */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Overall Diversity Score</h3>
        <div className={`text-4xl font-bold ${getScoreColor(metrics.overallDiversityScore)}`}>
          {(metrics.overallDiversityScore * 100).toFixed(0)}%
        </div>
        <div className="mt-4">
          <div className="w-full bg-gray-700 rounded-full h-3">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                metrics.overallDiversityScore >= 0.8
                  ? 'bg-success-green'
                  : metrics.overallDiversityScore >= 0.6
                  ? 'bg-warning-yellow'
                  : 'bg-warning-red'
              }`}
              style={{ width: `${metrics.overallDiversityScore * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Simpson's Diversity Index */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Simpson's Diversity Index</h3>
        <div className={`text-4xl font-bold ${getScoreColor(metrics.simpsonDiversityIndex)}`}>
          {metrics.simpsonDiversityIndex.toFixed(3)}
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Measures geographic distribution uniformity
        </p>
        <div className="mt-4">
          <div className="w-full bg-gray-700 rounded-full h-3">
            <div
              className="h-full bg-cascade-accent rounded-full transition-all duration-500"
              style={{ width: `${metrics.simpsonDiversityIndex * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Hemisphere Balance */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Hemisphere Balance</h3>
        <div className={`text-4xl font-bold ${getScoreColor(metrics.hemisphereBalanceScore)}`}>
          {(metrics.hemisphereBalanceScore * 100).toFixed(0)}%
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Target: 0.8-1.2 ratio between hemispheres
        </p>
        <div className="mt-4">
          <div className="w-full bg-gray-700 rounded-full h-3">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-500"
              style={{ width: `${metrics.hemisphereBalanceScore * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Continental Coverage */}
      <div className="col-span-3 bg-gray-800 rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-white text-lg font-semibold">Continental Coverage</h3>
          <div className="text-2xl font-bold text-cascade-accent">
            {coveredCount}/7
          </div>
        </div>
        <div className="grid grid-cols-7 gap-2">
          {continents.map(continent => {
            const isCovered = metrics.continentalCoverage[
              continent.key as keyof typeof metrics.continentalCoverage
            ]
            return (
              <div
                key={continent.key}
                className={`text-center p-3 rounded-lg transition-all ${
                  isCovered
                    ? 'bg-success-green bg-opacity-20 border border-success-green'
                    : 'bg-gray-700 border border-gray-600'
                }`}
              >
                <div className="text-2xl mb-1">{isCovered ? '✓' : '✗'}</div>
                <p className={`text-xs ${isCovered ? 'text-success-green' : 'text-gray-400'}`}>
                  {continent.label}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}