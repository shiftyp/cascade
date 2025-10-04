'use client'

import { RebalancingRecommendation } from '../types'

interface Props {
  recommendations: RebalancingRecommendation[]
}

export default function RebalancingRecommendations({ recommendations }: Props) {
  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="text-gray-400 text-center py-8">
        <p className="text-lg mb-2">✅ Collection is well balanced</p>
        <p className="text-sm">No rebalancing actions needed at this time</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {recommendations.map((rec, index) => (
        <div
          key={index}
          className="bg-gray-700 rounded-lg p-4 hover:bg-gray-650 transition-colors"
        >
          <div className="flex justify-between items-start mb-3">
            <div>
              <h3 className="text-white font-semibold text-lg">{rec.region}</h3>
              <p className="text-gray-400 text-sm">{rec.action}</p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-cascade-accent">
                {rec.priorityMultiplier.toFixed(1)}x
              </div>
              <p className="text-xs text-gray-400">Priority</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Current</p>
              <p className="text-white font-semibold">
                {rec.currentPercentage.toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-gray-400">Target</p>
              <p className="text-success-green font-semibold">
                {rec.targetPercentage.toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-gray-400">Deficit</p>
              <p className="text-warning-red font-semibold">
                -{rec.deficit.toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Visual progress bar */}
          <div className="mt-3">
            <div className="w-full bg-gray-800 rounded-full h-2 relative">
              <div
                className="absolute top-0 h-full bg-warning-red rounded-full"
                style={{ width: `${(rec.currentPercentage / rec.targetPercentage) * 100}%` }}
              />
              <div
                className="absolute top-0 h-full w-0.5 bg-success-green"
                style={{ left: '100%' }}
              />
            </div>
          </div>
        </div>
      ))}

      <div className="mt-6 p-4 bg-blue-900 bg-opacity-30 rounded-lg border border-blue-700">
        <p className="text-blue-200 text-sm">
          <strong>Auto-rebalancing:</strong> The system automatically adjusts SDR selection
          priorities based on these recommendations. Higher multipliers mean stronger
          preference for stations in those regions.
        </p>
      </div>
    </div>
  )
}