'use client'

import { GeographicData } from '../types'

interface Props {
  data: GeographicData | null
}

export default function LatitudeBandProgress({ data }: Props) {
  if (!data) return <div className="text-gray-400">Loading...</div>

  const bands = [
    { key: 'arctic', label: 'Arctic (>66.5°N)', color: 'bg-blue-500' },
    { key: 'temperate', label: 'Temperate (23.5-66.5°)', color: 'bg-green-500' },
    { key: 'tropical', label: 'Tropical (±23.5°)', color: 'bg-yellow-500' },
    { key: 'antarctic', label: 'Antarctic (<-66.5°S)', color: 'bg-purple-500' },
  ]

  return (
    <div className="space-y-6">
      {bands.map(band => {
        const bandData = data.latitudeBands[band.key as keyof typeof data.latitudeBands]
        const isUnderTarget = bandData.percentage < bandData.target
        const progressWidth = Math.min(100, (bandData.percentage / bandData.target) * 100)

        return (
          <div key={band.key}>
            <div className="flex justify-between mb-2">
              <span className="text-white font-medium">{band.label}</span>
              <div className="text-right">
                <span className={`${isUnderTarget ? 'text-warning-red' : 'text-success-green'}`}>
                  {bandData.percentage.toFixed(1)}%
                </span>
                <span className="text-gray-400 ml-2">
                  (target: {bandData.target}%)
                </span>
              </div>
            </div>
            <div className="relative">
              <div className="w-full bg-gray-700 rounded-full h-6 overflow-hidden">
                <div
                  className={`h-full ${band.color} transition-all duration-500 flex items-center justify-end pr-2`}
                  style={{ width: `${progressWidth}%` }}
                >
                  <span className="text-xs text-white font-semibold">
                    {bandData.hours.toFixed(0)}h
                  </span>
                </div>
              </div>
              {/* Target indicator */}
              <div
                className="absolute top-0 h-full w-0.5 bg-white opacity-50"
                style={{ left: `${(bandData.target / 100) * 100}%` }}
              />
            </div>
            {isUnderTarget && (
              <p className="text-warning-yellow text-xs mt-1">
                ⚠ {(bandData.target - bandData.percentage).toFixed(1)}% below target
              </p>
            )}
          </div>
        )
      })}

      {/* Hemisphere Balance */}
      <div className="mt-8 pt-6 border-t border-gray-700">
        <h3 className="text-white font-semibold mb-4">Hemispheric Balance</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {data.hemispheres.north.percentage.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">Northern</div>
            <div className="text-xs text-gray-500">Target: {data.hemispheres.north.target}%</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {data.hemispheres.equatorial.percentage.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">Equatorial</div>
            <div className="text-xs text-gray-500">Target: {data.hemispheres.equatorial.target}%</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {data.hemispheres.south.percentage.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">Southern</div>
            <div className="text-xs text-gray-500">Target: {data.hemispheres.south.target}%</div>
          </div>
        </div>
      </div>

      {/* Ocean Path Coverage */}
      <div className="mt-6 pt-6 border-t border-gray-700">
        <div className="flex justify-between mb-2">
          <span className="text-white font-medium">Ocean Path Coverage</span>
          <span className={data.oceanPathPercentage < 30 ? 'text-warning-red' : 'text-success-green'}>
            {data.oceanPathPercentage.toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-4">
          <div
            className={`h-full ${data.oceanPathPercentage < 30 ? 'bg-warning-red' : 'bg-cascade-accent'} rounded-full transition-all duration-500`}
            style={{ width: `${Math.min(100, data.oceanPathPercentage)}%` }}
          />
        </div>
        <p className="text-xs text-gray-400 mt-1">Minimum target: 30%</p>
      </div>
    </div>
  )
}